from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import tempfile

import numpy as np
import pytest

from corespec_mapper.spectral_v5 import load_v5_catalog
from corespec_mapper.v5_library import (
    V5ReferenceSelectionError,
    build_v5_library_ensemble,
)


SWIR_WAVELENGTHS_NM = np.linspace(1000.0, 2500.0, 301)


def test_bundled_runtime_selection_is_pure_sample_unique_diverse_and_serializable():
    library = build_v5_library_ensemble(SWIR_WAVELENGTHS_NM, ["calcite", "illite"])

    assert library.spectra.shape == (6, SWIR_WAVELENGTHS_NM.size)
    assert library.reference_counts() == {"calcite": 3, "illite": 3}
    assert len(library.database_sha256) == 64
    assert len(library.catalog_sha256) == 64
    assert library.database_release == "5.1.0"
    assert library.catalog_version == "5.1.0"
    for mineral_id in ("calcite", "illite"):
        chosen = [candidate for candidate in library.selected_candidates if candidate.mineral_id == mineral_id]
        assert len({candidate.sample_id for candidate in chosen}) == len(chosen)
        assert len({candidate.source_family for candidate in chosen}) >= 2
        assert chosen[0].anchor_kind == "robust_medoid"
        assert chosen[0].selection_reason == "robust_medoid_minimum_median_shape_distance"
    assert all(candidate.purity_status == "declared_pure" for candidate in library.candidates)
    assert all(candidate.qc_status == "eligible" for candidate in library.candidates)
    assert all(candidate.source_role == "mineral_reference" for candidate in library.candidates)

    payload = library.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert encoded
    assert len(payload["wavelengths_nm"]) == SWIR_WAVELENGTHS_NM.size
    assert all(len(reference["values"]) == SWIR_WAVELENGTHS_NM.size for reference in payload["references"])
    assert payload["database"]["sha256"] == library.database_sha256
    assert payload["catalog"]["sha256"] == library.catalog_sha256


def test_repeated_measurements_of_one_physical_sample_are_collapsed_before_selection():
    library = build_v5_library_ensemble(SWIR_WAVELENGTHS_NM, ["calcite"])
    sample = "igcp264:calcite:co2004"
    measurements = [candidate for candidate in library.candidates if candidate.sample_id == sample]

    assert len(measurements) == 5
    assert sum(candidate.rejection_reason == "duplicate_measurement_of_same_sample" for candidate in measurements) == 4
    retained = [candidate for candidate in measurements if candidate.rejection_reason != "duplicate_measurement_of_same_sample"]
    assert len(retained) == 1
    assert library.availability_for("calcite").independent_sample_count == 9
    assert len({candidate.sample_id for candidate in library.selected_candidates}) == len(library.selected_candidates)


def test_catalog_anchor_has_priority_over_medoid_and_is_traceable():
    catalog = deepcopy(load_v5_catalog())
    catalog["minerals"]["calcite"]["canonical_anchors"] = [{"measurement_id": 5}]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "catalog_with_anchor.json"
        path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
        library = build_v5_library_ensemble(
            SWIR_WAVELENGTHS_NM,
            ["calcite"],
            catalog_path=path,
        )

    first = library.selected_candidates[0]
    assert first.measurement_id == 5
    assert first.anchor_kind == "catalog_anchor"
    assert first.selection_reason == "catalog_anchor_priority"
    assert library.availability_for("calcite").selection_mode == "catalog_anchor_plus_diversity"
    assert library.catalog_sha256 != build_v5_library_ensemble(
        SWIR_WAVELENGTHS_NM, ["calcite"]
    ).catalog_sha256


def test_unsupported_and_sensor_incompatible_targets_return_explicit_reasons():
    library = build_v5_library_ensemble(
        SWIR_WAVELENGTHS_NM,
        ["calcite", "hematite", "quartz"],
    )

    assert library.availability_for("calcite").available
    hematite = library.availability_for("hematite")
    quartz = library.availability_for("quartz")
    assert not hematite.available
    assert not quartz.available
    assert {reason.code for reason in hematite.reasons} >= {
        "expert_not_implemented",
        "recognition_disabled",
        "sensor_required_windows_missing",
    }
    assert {reason.code for reason in quartz.reasons} >= {
        "incompatible_data_physics",
        "expert_not_implemented",
        "recognition_disabled",
        "sensor_required_windows_missing",
    }
    assert library.spectra.shape[0] == 3


def test_missing_diagnostic_window_can_be_returned_or_made_strict():
    incomplete = np.linspace(1000.0, 2200.0, 241)
    library = build_v5_library_ensemble(incomplete, ["calcite"])
    availability = library.availability_for("calcite")

    assert not availability.available
    assert [reason.code for reason in availability.reasons] == ["sensor_required_windows_missing"]
    assert availability.required_window_coverage[0]["coverage_fraction"] < 0.97
    assert library.spectra.shape == (0, incomplete.size)
    with pytest.raises(V5ReferenceSelectionError, match="calcite: sensor_required_windows_missing"):
        build_v5_library_ensemble(incomplete, ["calcite"], require_all_available=True)


def test_fwhm_convolution_and_selection_are_deterministic():
    fwhm = np.full(SWIR_WAVELENGTHS_NM.shape, 10.0)
    first = build_v5_library_ensemble(
        SWIR_WAVELENGTHS_NM,
        ["calcite", "kaolinite"],
        target_fwhm_nm=fwhm,
    )
    second = build_v5_library_ensemble(
        SWIR_WAVELENGTHS_NM,
        ["calcite", "kaolinite"],
        target_fwhm_nm=fwhm,
    )

    assert first.resampling_method == "gaussian_fwhm_convolution"
    assert not first.warnings
    assert [candidate.measurement_id for candidate in first.selected_candidates] == [
        candidate.measurement_id for candidate in second.selected_candidates
    ]
    np.testing.assert_allclose(first.spectra, second.spectra, equal_nan=True)


def test_exact_v4_container_adapter_preserves_selected_matrices_and_v5_groups():
    library = build_v5_library_ensemble(
        SWIR_WAVELENGTHS_NM,
        ["calcite"],
        maximum_representatives=2,
        minimum_references=1,
    )
    adapted = library.to_v4_ensemble()

    assert adapted.spectra is library.spectra
    assert adapted.wavelengths_nm is library.wavelengths_nm
    assert adapted.mineral_labels == library.mineral_labels
    assert adapted.group_labels == library.group_labels
    assert adapted.reference_counts() == {"calcite": 2}
    assert all(candidate.group_id == "carbonate_2300" for candidate in adapted.selected_candidates)
    assert adapted.libraries[0]["sha256"] == library.database_sha256


@pytest.mark.parametrize(
    "wavelengths,fwhm,error",
    [
        ([1000.0], None, "at least two bands"),
        ([1000.0, 999.0], None, "strictly increasing"),
        ([1000.0, 1010.0], [5.0], "one finite positive value"),
        ([1000.0, 1010.0], [5.0, 0.0], "one finite positive value"),
    ],
)
def test_invalid_sensor_axes_are_rejected(wavelengths, fwhm, error):
    with pytest.raises(ValueError, match=error):
        build_v5_library_ensemble(wavelengths, ["calcite"], target_fwhm_nm=fwhm)


def test_unknown_mineral_is_a_configuration_error():
    with pytest.raises(KeyError, match="unknownite"):
        build_v5_library_ensemble(SWIR_WAVELENGTHS_NM, ["unknownite"])
