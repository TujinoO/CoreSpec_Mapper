from __future__ import annotations

"""Sensor-aware runtime selection of the bundled V5 mineral references.

This module deliberately sits above :mod:`spectral_db`.  The database owns
provenance and curation; this layer only reads its ``eligible`` whitelist,
adapts those measurements to a sensor, collapses repeated measurements of the
same physical sample, and chooses a small, diverse ensemble for each mineral.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import numpy as np

from .library_ensemble import resample_spectrum
from .spectral_db import V5SpectralDatabase, default_v5_database_path
from .spectral_v5 import default_v5_catalog_path, describe_source, load_v5_catalog


SELECTION_SCHEMA_VERSION = 1


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _array_record(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values).ravel()]


def _normalised_identifier(value: str) -> str:
    return str(value).replace("\\", "/").strip().casefold()


@dataclass(frozen=True)
class V5UnavailabilityReason:
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


@dataclass
class V5TargetAvailability:
    mineral_id: str
    display_name_en: str
    display_name_zh: str
    taxonomy_group_id: str
    spectral_family_id: str
    support_level: str
    recognition_enabled: bool
    available: bool = False
    reasons: list[V5UnavailabilityReason] = field(default_factory=list)
    required_window_coverage: list[dict[str, Any]] = field(default_factory=list)
    eligible_measurement_count: int = 0
    physics_compatible_measurement_count: int = 0
    independent_sample_count: int = 0
    selected_reference_count: int = 0
    selection_mode: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "mineral_id": self.mineral_id,
            "display_name_en": self.display_name_en,
            "display_name_zh": self.display_name_zh,
            "taxonomy_group_id": self.taxonomy_group_id,
            "spectral_family_id": self.spectral_family_id,
            "support_level": self.support_level,
            "recognition_enabled": self.recognition_enabled,
            "available": self.available,
            "reasons": [reason.to_record() for reason in self.reasons],
            "required_window_coverage": list(self.required_window_coverage),
            "eligible_measurement_count": self.eligible_measurement_count,
            "physics_compatible_measurement_count": self.physics_compatible_measurement_count,
            "independent_sample_count": self.independent_sample_count,
            "selected_reference_count": self.selected_reference_count,
            "selection_mode": self.selection_mode,
        }


@dataclass(eq=False)
class V5ReferenceCandidate:
    mineral_id: str
    taxonomy_group_id: str
    spectral_family_id: str
    measurement_id: int
    sample_id: str
    source_id: str
    source_index: int
    source_name: str
    source_family: str
    source_role: str
    purity_status: str
    qc_status: str
    data_physics: str
    measurement_geometry: str
    resampling_method: str
    resampled: np.ndarray = field(repr=False)
    embedding: np.ndarray | None = field(default=None, repr=False)
    required_coverage: float = float("nan")
    absorption_depth: float = float("nan")
    roughness: float = float("nan")
    quality_score: float = float("nan")
    catalog_anchor_order: int | None = None
    anchor_kind: str | None = None
    medoid_distance_rad: float = float("nan")
    selected: bool = False
    selected_rank: int | None = None
    selection_score: float = float("nan")
    selection_reason: str | None = None
    rejection_reason: str | None = None
    duplicate_of: str | None = None

    @property
    def identifier(self) -> str:
        return f"{self.source_id}#{self.source_index}"

    def to_record(self, *, include_curve: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "identifier": self.identifier,
            "mineral_id": self.mineral_id,
            "taxonomy_group_id": self.taxonomy_group_id,
            "spectral_family_id": self.spectral_family_id,
            "measurement_id": self.measurement_id,
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "source_index": self.source_index,
            "source_name": self.source_name,
            "source_family": self.source_family,
            "source_role": self.source_role,
            "purity_status": self.purity_status,
            "qc_status": self.qc_status,
            "data_physics": self.data_physics,
            "measurement_geometry": self.measurement_geometry,
            "resampling_method": self.resampling_method,
            "required_coverage": _finite_or_none(self.required_coverage),
            "absorption_depth": _finite_or_none(self.absorption_depth),
            "roughness": _finite_or_none(self.roughness),
            "quality_score": _finite_or_none(self.quality_score),
            "catalog_anchor_order": self.catalog_anchor_order,
            "anchor_kind": self.anchor_kind,
            "medoid_distance_rad": _finite_or_none(self.medoid_distance_rad),
            "selected": self.selected,
            "selected_rank": self.selected_rank,
            "selection_score": _finite_or_none(self.selection_score),
            "selection_reason": self.selection_reason,
            "rejection_reason": self.rejection_reason,
            "duplicate_of": self.duplicate_of,
        }
        if include_curve:
            record["values"] = _array_record(self.resampled)
        return record


@dataclass(frozen=True)
class V5ReferenceLibrary:
    wavelengths_nm: np.ndarray
    spectra: np.ndarray
    mineral_labels: tuple[str, ...]
    group_labels: tuple[str, ...]
    spectrum_names: tuple[str, ...]
    selected_candidates: tuple[V5ReferenceCandidate, ...]
    candidates: tuple[V5ReferenceCandidate, ...]
    availability: tuple[V5TargetAvailability, ...]
    data_physics: str
    database_path: Path
    database_sha256: str
    database_release: str
    database_schema_version: int
    catalog_path: Path
    catalog_sha256: str
    catalog_version: str
    resampling_method: str
    warnings: tuple[str, ...]

    def group_indices(self, group_id: str) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.group_labels, dtype=object) == group_id)

    def reference_counts(self) -> dict[str, int]:
        return {
            availability.mineral_id: availability.selected_reference_count
            for availability in self.availability
        }

    def availability_for(self, mineral_id: str) -> V5TargetAvailability:
        key = str(mineral_id).casefold()
        for availability in self.availability:
            if availability.mineral_id == key:
                return availability
        raise KeyError(f"Mineral was not requested: {mineral_id}")

    def to_v4_ensemble(self):
        """Adapt the result to the exact V4 container type when needed.

        Group labels deliberately retain V5 spectral-family identifiers; this
        adapter is for matrix/container interoperability, not for silently
        replacing V5 taxonomy with the narrower V4 taxonomy.
        """

        from .v4_library import V4EnsembleLibrary, V4LibraryCandidate

        converted: list[V4LibraryCandidate] = []
        by_measurement: dict[int, V4LibraryCandidate] = {}
        for source in self.candidates:
            candidate = V4LibraryCandidate(
                mineral_id=source.mineral_id,
                group_id=source.spectral_family_id,
                source_library=source.source_id,
                source_index=source.source_index,
                source_name=source.source_name,
                resampling_method=source.resampling_method,
                rejection_reason=source.rejection_reason,
                coverage=source.required_coverage,
                absorption_depth=source.absorption_depth,
                roughness=source.roughness,
                intrinsic_quality=source.quality_score,
                diagnostic_quality=source.quality_score,
                selection_score=source.selection_score,
                anchor=source.anchor_kind is not None,
                selected=source.selected,
                selected_rank=source.selected_rank,
                duplicate_of=source.duplicate_of,
                resampled=source.resampled,
                embedding=source.embedding,
            )
            converted.append(candidate)
            by_measurement[source.measurement_id] = candidate
        selected = tuple(by_measurement[item.measurement_id] for item in self.selected_candidates)
        return V4EnsembleLibrary(
            wavelengths_nm=self.wavelengths_nm,
            spectra=self.spectra,
            mineral_labels=self.mineral_labels,
            group_labels=self.group_labels,
            spectrum_names=self.spectrum_names,
            selected_candidates=selected,
            candidates=tuple(converted),
            libraries=(
                {
                    "path": self.database_path.name,
                    "sha256": self.database_sha256,
                    "release": self.database_release,
                    "catalog_sha256": self.catalog_sha256,
                },
            ),
            warnings=self.warnings,
        )

    def to_dict(self, *, include_candidates: bool = True, include_curves: bool = True) -> dict[str, Any]:
        """Return a strict-JSON-safe record for the desktop and run manifest."""

        references = []
        for candidate in self.selected_candidates:
            record = candidate.to_record(include_curve=include_curves)
            references.append(record)
        result: dict[str, Any] = {
            "selection_schema_version": SELECTION_SCHEMA_VERSION,
            "data_physics": self.data_physics,
            "wavelengths_nm": _array_record(self.wavelengths_nm) if include_curves else None,
            "reference_count": len(self.selected_candidates),
            "reference_counts": self.reference_counts(),
            "references": references,
            "availability": [item.to_record() for item in self.availability],
            "database": {
                "filename": self.database_path.name,
                "sha256": self.database_sha256,
                "release": self.database_release,
                "schema_version": self.database_schema_version,
            },
            "catalog": {
                "filename": self.catalog_path.name,
                "sha256": self.catalog_sha256,
                "version": self.catalog_version,
            },
            "resampling_method": self.resampling_method,
            "warnings": list(self.warnings),
        }
        if include_candidates:
            result["candidates"] = [candidate.to_record() for candidate in self.candidates]
        return result


class V5ReferenceSelectionError(ValueError):
    """Raised only when a caller explicitly requires every target to work."""

    def __init__(self, unavailable: Sequence[V5TargetAvailability]):
        self.unavailable = tuple(unavailable)
        summary = "; ".join(
            f"{item.mineral_id}: {', '.join(reason.code for reason in item.reasons) or 'unavailable'}"
            for item in self.unavailable
        )
        super().__init__(f"V5 reference selection unavailable for {summary}")


def _validated_axis(
    target_wavelengths_nm: Sequence[float],
    target_fwhm_nm: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    wavelengths = np.asarray(target_wavelengths_nm, dtype=np.float64)
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise ValueError("Target wavelengths must be a one-dimensional axis with at least two bands")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0) or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Target wavelengths must be finite, positive, and strictly increasing")
    if target_fwhm_nm is None:
        return wavelengths, None
    fwhm = np.asarray(target_fwhm_nm, dtype=np.float64)
    if fwhm.shape != wavelengths.shape or np.any(~np.isfinite(fwhm)) or np.any(fwhm <= 0):
        raise ValueError("FWHM must contain one finite positive value per target band")
    return wavelengths, fwhm


def _band_intervals(wavelengths: np.ndarray, fwhm: np.ndarray | None) -> list[tuple[float, float]]:
    if fwhm is not None:
        raw = [(float(center - width / 2.0), float(center + width / 2.0)) for center, width in zip(wavelengths, fwhm)]
    else:
        midpoints = (wavelengths[:-1] + wavelengths[1:]) / 2.0
        lower = np.concatenate(([wavelengths[0] - (midpoints[0] - wavelengths[0])], midpoints))
        upper = np.concatenate((midpoints, [wavelengths[-1] + (wavelengths[-1] - midpoints[-1])]))
        raw = [(float(low), float(high)) for low, high in zip(lower, upper)]
    merged: list[tuple[float, float]] = []
    for lower, upper in raw:
        if not merged or lower > merged[-1][1]:
            merged.append((lower, upper))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
    return merged


def _required_window_records(
    wavelengths: np.ndarray,
    fwhm: np.ndarray | None,
    windows: Sequence[Sequence[float]],
    minimum_coverage: float,
    minimum_bands: int,
) -> list[dict[str, Any]]:
    intervals = _band_intervals(wavelengths, fwhm)
    records: list[dict[str, Any]] = []
    for window in windows:
        lower, upper = float(window[0]), float(window[1])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError(f"Invalid required wavelength window: {window}")
        covered = sum(max(0.0, min(upper, right) - max(lower, left)) for left, right in intervals)
        coverage = float(np.clip(covered / (upper - lower), 0.0, 1.0))
        bands = int(np.count_nonzero((wavelengths >= lower) & (wavelengths <= upper)))
        records.append(
            {
                "lower_nm": lower,
                "upper_nm": upper,
                "coverage_fraction": coverage,
                "band_count": bands,
                "compatible": bool(coverage >= minimum_coverage and bands >= minimum_bands),
            }
        )
    return records


def _window_mask(wavelengths: np.ndarray, windows: Sequence[Sequence[float]]) -> np.ndarray:
    mask = np.zeros(wavelengths.shape, dtype=bool)
    for lower, upper in windows:
        mask |= (wavelengths >= float(lower)) & (wavelengths <= float(upper))
    return mask


def _prepare_source(source_wavelengths: np.ndarray, source_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wavelengths = np.asarray(source_wavelengths, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    valid = np.isfinite(wavelengths) & np.isfinite(values) & (values > 0.0) & (np.abs(values) < 1e6)
    wavelengths, values = wavelengths[valid], values[valid]
    if wavelengths.size < 2:
        return wavelengths, values
    order = np.argsort(wavelengths, kind="stable")
    wavelengths, values = wavelengths[order], values[order]
    unique, inverse = np.unique(wavelengths, return_inverse=True)
    if unique.size != wavelengths.size:
        sums = np.zeros(unique.size, dtype=np.float64)
        counts = np.zeros(unique.size, dtype=np.int64)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1)
        wavelengths, values = unique, sums / np.maximum(counts, 1)
    return wavelengths, values


def _resample_with_optional_fwhm(
    source_wavelengths: np.ndarray,
    source_values: np.ndarray,
    target_wavelengths: np.ndarray,
    target_fwhm: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if target_fwhm is None:
        return (
            resample_spectrum(source_wavelengths, source_values, target_wavelengths),
            "gap_constrained_interpolation",
        )
    wavelengths, values = _prepare_source(source_wavelengths, source_values)
    result = np.full(target_wavelengths.shape, np.nan, dtype=np.float64)
    if wavelengths.size < 2:
        return result, "gaussian_fwhm_convolution"
    positive_spacing = np.diff(wavelengths)
    typical_spacing = float(np.median(positive_spacing[positive_spacing > 0]))
    maximum_gap = max(45.0, 8.0 * typical_spacing)
    for index, (center, width) in enumerate(zip(target_wavelengths, target_fwhm)):
        sigma = float(width) / 2.354820045
        local = np.abs(wavelengths - center) <= 3.0 * sigma
        if np.count_nonzero(local) < 2:
            continue
        local_wavelengths = wavelengths[local]
        if np.max(np.diff(local_wavelengths)) > maximum_gap:
            continue
        weights = np.exp(-0.5 * ((local_wavelengths - center) / sigma) ** 2)
        result[index] = float(np.sum(values[local] * weights) / np.sum(weights))
    return result, "gaussian_fwhm_convolution"


def _shape_diagnostics(
    values: np.ndarray,
    wavelengths: np.ndarray,
    windows: Sequence[Sequence[float]],
    minimum_coverage: float,
) -> tuple[float, float, float, float, np.ndarray | None, str | None]:
    selected = _window_mask(wavelengths, windows)
    if not np.any(selected):
        return 0.0, float("nan"), float("nan"), 0.0, None, "no_target_bands_in_required_windows"
    relevant = np.asarray(values[selected], dtype=np.float64)
    window_coverages = [
        float(np.mean(np.isfinite(values[(wavelengths >= float(lower)) & (wavelengths <= float(upper))])))
        for lower, upper in windows
    ]
    coverage = min(window_coverages, default=0.0)
    if any(value < minimum_coverage for value in window_coverages):
        return coverage, float("nan"), float("nan"), 0.0, None, "insufficient_resampled_coverage"
    if np.any(~np.isfinite(relevant)) or np.any(relevant <= 0):
        return coverage, float("nan"), float("nan"), 0.0, None, "invalid_values_in_required_windows"

    embedding_parts: list[np.ndarray] = []
    for lower, upper in windows:
        indices = np.flatnonzero((wavelengths >= float(lower)) & (wavelengths <= float(upper)))
        curve = np.asarray(values[indices], dtype=np.float64)
        if curve.size < 2:
            continue
        continuum = np.linspace(curve[0], curve[-1], curve.size)
        scale = max(float(np.median(np.abs(curve))), 1e-12)
        continuum = np.where(np.abs(continuum) > scale * 1e-8, continuum, scale)
        embedding_parts.append(np.clip(1.0 - curve / continuum, 0.0, 1.5))
    if not embedding_parts:
        return coverage, float("nan"), float("nan"), 0.0, None, "insufficient_shape_samples"
    shape = np.concatenate(embedding_parts)
    norm = float(np.sqrt(np.sum(shape * shape)))
    if not np.isfinite(norm) or norm <= 1e-10:
        return coverage, 0.0, 0.0, 0.0, None, "featureless_reference_shape"
    depth = float(np.max(shape))
    second = np.diff(shape, n=2)
    roughness = float(np.median(np.abs(second)) / max(depth, 1e-12)) if second.size else 0.0
    smoothness = 1.0 - min(roughness / 0.25, 1.0)
    contrast = min(depth / 0.10, 1.0)
    quality = float(np.clip(0.55 * coverage + 0.25 * smoothness + 0.20 * contrast, 0.0, 1.0))
    return coverage, depth, roughness, quality, shape / norm, None


def _shape_angle(left: V5ReferenceCandidate, right: V5ReferenceCandidate) -> float:
    assert left.embedding is not None and right.embedding is not None
    cosine = float(np.clip(np.einsum("i,i->", left.embedding, right.embedding), -1.0, 1.0))
    return float(np.arccos(cosine))


def _anchor_specs(mineral: Mapping[str, Any]) -> tuple[Any, ...]:
    for key in ("canonical_anchors", "reference_anchors", "preferred_references"):
        values = mineral.get(key)
        if values:
            return tuple(values)
    return ()


def _anchor_order(candidate: V5ReferenceCandidate, specifications: Sequence[Any]) -> int | None:
    candidate_identifier = _normalised_identifier(candidate.identifier)
    for order, specification in enumerate(specifications):
        if isinstance(specification, int):
            if candidate.measurement_id == specification:
                return order
            continue
        if isinstance(specification, str):
            normalised = _normalised_identifier(specification)
            if normalised == candidate_identifier or normalised == str(candidate.measurement_id):
                return order
            continue
        if not isinstance(specification, Mapping):
            continue
        measurement_id = specification.get("measurement_id")
        if measurement_id is not None and int(measurement_id) == candidate.measurement_id:
            return order
        sample_id = specification.get("sample_id")
        if sample_id is not None and str(sample_id).casefold() == candidate.sample_id.casefold():
            return order
        source_id = specification.get("source_id")
        source_index = specification.get("source_index")
        if source_id is not None and source_index is not None:
            identifier = f"{source_id}#{int(source_index)}"
            if _normalised_identifier(identifier) == candidate_identifier:
                return order
    return None


def _collapse_measurements_by_sample(candidates: list[V5ReferenceCandidate]) -> list[V5ReferenceCandidate]:
    by_sample: dict[str, list[V5ReferenceCandidate]] = {}
    for candidate in candidates:
        if candidate.rejection_reason is None:
            by_sample.setdefault(candidate.sample_id, []).append(candidate)
    retained: list[V5ReferenceCandidate] = []
    for sample_id in sorted(by_sample, key=str.casefold):
        measurements = by_sample[sample_id]
        measurements.sort(
            key=lambda item: (
                item.catalog_anchor_order is None,
                item.catalog_anchor_order if item.catalog_anchor_order is not None else math.inf,
                -item.quality_score,
                item.measurement_id,
            )
        )
        best = measurements[0]
        retained.append(best)
        for duplicate in measurements[1:]:
            duplicate.rejection_reason = "duplicate_measurement_of_same_sample"
            duplicate.duplicate_of = best.identifier
    return retained


def _robust_medoid(candidates: Sequence[V5ReferenceCandidate]) -> V5ReferenceCandidate:
    if len(candidates) == 1:
        candidates[0].medoid_distance_rad = 0.0
        return candidates[0]
    distances = np.zeros((len(candidates), len(candidates)), dtype=np.float64)
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index):
            angle = _shape_angle(left, candidates[right_index])
            distances[left_index, right_index] = distances[right_index, left_index] = angle
    scores = np.median(distances, axis=1)
    for candidate, score in zip(candidates, scores):
        candidate.medoid_distance_rad = float(score)
    best_index = min(
        range(len(candidates)),
        key=lambda index: (scores[index], -candidates[index].quality_score, candidates[index].identifier.casefold()),
    )
    return candidates[best_index]


def _choose_references(
    candidates: list[V5ReferenceCandidate],
    maximum_references: int,
    minimum_references: int,
) -> tuple[list[V5ReferenceCandidate], str]:
    if not candidates:
        return [], "unavailable"
    desired = min(
        maximum_references,
        len(candidates),
        max(minimum_references, int(math.ceil(math.sqrt(len(candidates))))),
    )
    explicit = sorted(
        (candidate for candidate in candidates if candidate.catalog_anchor_order is not None),
        key=lambda item: (item.catalog_anchor_order, -item.quality_score, item.identifier.casefold()),
    )[: min(2, desired)]
    if explicit:
        selected = list(explicit)
        for candidate in selected:
            candidate.anchor_kind = "catalog_anchor"
            candidate.selection_reason = "catalog_anchor_priority"
            candidate.selection_score = 1.0
        mode = "catalog_anchor_plus_diversity"
    else:
        medoid = _robust_medoid(candidates)
        medoid.anchor_kind = "robust_medoid"
        medoid.selection_reason = "robust_medoid_minimum_median_shape_distance"
        medoid.selection_score = 1.0
        selected = [medoid]
        mode = "robust_medoid_plus_diversity"

    selected_families = {candidate.source_family for candidate in selected}
    selected_geometries = {candidate.measurement_geometry for candidate in selected}
    while len(selected) < desired:
        best: V5ReferenceCandidate | None = None
        best_score = -np.inf
        best_components: tuple[float, float, float, float] | None = None
        for candidate in candidates:
            if candidate in selected:
                continue
            diversity_angle = min(_shape_angle(candidate, chosen) for chosen in selected)
            shape_gain = min(diversity_angle / 0.20, 1.0)
            source_gain = 1.0 if candidate.source_family not in selected_families else 0.0
            geometry_gain = 1.0 if candidate.measurement_geometry not in selected_geometries else 0.0
            score = 0.50 * shape_gain + 0.25 * candidate.quality_score + 0.20 * source_gain + 0.05 * geometry_gain
            key = (score, candidate.quality_score, diversity_angle, -candidate.measurement_id)
            if best is None or key > (best_score, best_components[1], best_components[0], -best.measurement_id):
                best = candidate
                best_score = float(score)
                best_components = (diversity_angle, candidate.quality_score, source_gain, geometry_gain)
        assert best is not None and best_components is not None
        best.selection_score = best_score
        best.selection_reason = (
            "greedy_shape_source_sample_diversity"
            f"(shape_angle_rad={best_components[0]:.6f},source_gain={best_components[2]:.0f},"
            f"geometry_gain={best_components[3]:.0f},quality={best_components[1]:.6f})"
        )
        selected.append(best)
        selected_families.add(best.source_family)
        selected_geometries.add(best.measurement_geometry)

    for rank, candidate in enumerate(selected, start=1):
        candidate.selected = True
        candidate.selected_rank = rank
    selected_ids = {id(candidate) for candidate in selected}
    for candidate in candidates:
        if id(candidate) not in selected_ids and candidate.rejection_reason is None:
            candidate.rejection_reason = "not_selected_as_representative"
    return selected, mode


def build_v5_library_ensemble(
    target_wavelengths_nm: Sequence[float],
    mineral_ids: Sequence[str],
    *,
    target_fwhm_nm: Sequence[float] | None = None,
    data_physics: str = "reflectance",
    database_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    maximum_references: int = 6,
    maximum_representatives: int | None = None,
    minimum_references: int = 3,
    minimum_window_coverage: float = 0.97,
    minimum_bands_per_window: int = 3,
    require_all_available: bool = False,
) -> V5ReferenceLibrary:
    """Build a compact reference ensemble directly from the packaged V5 DB.

    Known but unsupported targets are represented by an availability record
    instead of causing a desktop workflow to crash.  Scientific batch callers
    can request strict behaviour with ``require_all_available=True``.
    """

    wavelengths, fwhm = _validated_axis(target_wavelengths_nm, target_fwhm_nm)
    if not 0.0 < float(minimum_window_coverage) <= 1.0:
        raise ValueError("minimum_window_coverage must be in (0, 1]")
    if int(minimum_bands_per_window) < 2:
        raise ValueError("minimum_bands_per_window must be at least 2")
    if maximum_representatives is not None:
        maximum_references = int(maximum_representatives)
    if int(maximum_references) < 1:
        raise ValueError("maximum_references must be at least 1")
    if not 1 <= int(minimum_references) <= int(maximum_references):
        raise ValueError("minimum_references must be between 1 and maximum_references")
    physics = str(data_physics).strip().casefold()
    if not physics:
        raise ValueError("data_physics cannot be empty")
    requested = tuple(dict.fromkeys(str(item).strip().casefold() for item in mineral_ids if str(item).strip()))
    if not requested:
        raise ValueError("At least one mineral is required")

    database_source = (default_v5_database_path() if database_path is None else Path(database_path)).resolve()
    catalog_source = (default_v5_catalog_path() if catalog_path is None else Path(catalog_path)).resolve()
    catalog = load_v5_catalog(catalog_source)
    definitions = catalog.get("minerals", {})
    unknown = [mineral_id for mineral_id in requested if mineral_id not in definitions]
    if unknown:
        raise KeyError(f"Unknown V5 mineral(s): {', '.join(unknown)}")

    database_hash = _hash_file(database_source)
    catalog_hash = _hash_file(catalog_source)
    candidates: list[V5ReferenceCandidate] = []
    selected: list[V5ReferenceCandidate] = []
    availability_records: list[V5TargetAvailability] = []
    warnings: list[str] = []
    if fwhm is None:
        warnings.append("FWHM unavailable: references use gap-constrained interpolation")

    with V5SpectralDatabase(database_source) as database:
        summary = database.summary()
        database_minerals = database.mineral_definitions()
        if str(summary.get("catalog_version")) != str(catalog.get("version")):
            raise ValueError(
                "V5 catalog/database version mismatch: "
                f"catalog={catalog.get('version')}, database={summary.get('catalog_version')}"
            )
        for mineral_id in requested:
            mineral = definitions[mineral_id]
            taxonomy_group = str(mineral["taxonomy_group"])
            spectral_family = str(mineral["spectral_family"])
            record = V5TargetAvailability(
                mineral_id=mineral_id,
                display_name_en=str(mineral["display_name_en"]),
                display_name_zh=str(mineral["display_name_zh"]),
                taxonomy_group_id=taxonomy_group,
                spectral_family_id=spectral_family,
                support_level=str(mineral.get("support_level", "unknown")),
                recognition_enabled=bool(mineral.get("recognition_enabled", False)),
            )
            availability_records.append(record)
            if mineral_id not in database_minerals:
                record.reasons.append(
                    V5UnavailabilityReason(
                        "mineral_not_in_database",
                        "The selected mineral has no definition in the bundled spectral database.",
                    )
                )

            family_definition = catalog["spectral_families"][spectral_family]
            expert_id = str(family_definition["expert"])
            expert = catalog["experts"][expert_id]
            allowed_physics = tuple(str(value).casefold() for value in expert.get("data_physics", ()))
            if physics not in allowed_physics:
                record.reasons.append(
                    V5UnavailabilityReason(
                        "incompatible_data_physics",
                        f"{record.display_name_en} requires {', '.join(allowed_physics) or 'unspecified physics'}, not {physics}.",
                        {"requested": physics, "allowed": list(allowed_physics), "expert_id": expert_id},
                    )
                )
            if not bool(expert.get("implemented", False)):
                record.reasons.append(
                    V5UnavailabilityReason(
                        "expert_not_implemented",
                        str(expert.get("limitation") or f"The {expert_id} expert is not implemented."),
                        {"expert_id": expert_id},
                    )
                )
            if not record.recognition_enabled:
                record.reasons.append(
                    V5UnavailabilityReason(
                        "recognition_disabled",
                        str(mineral.get("limitation") or "Mineral-level recognition is disabled in the V5 catalog."),
                        {"support_level": record.support_level},
                    )
                )

            required_windows = tuple(tuple(float(value) for value in window) for window in mineral["required_windows_nm"])
            record.required_window_coverage = _required_window_records(
                wavelengths,
                fwhm,
                required_windows,
                float(minimum_window_coverage),
                int(minimum_bands_per_window),
            )
            incompatible_windows = [item for item in record.required_window_coverage if not item["compatible"]]
            if incompatible_windows:
                record.reasons.append(
                    V5UnavailabilityReason(
                        "sensor_required_windows_missing",
                        "The sensor does not adequately sample every required diagnostic window.",
                        {
                            "minimum_coverage": float(minimum_window_coverage),
                            "minimum_bands": int(minimum_bands_per_window),
                            "incompatible_windows": incompatible_windows,
                        },
                    )
                )

            stored = list(database.iter_spectra(mineral_id)) if mineral_id in database_minerals else []
            # ``iter_spectra`` is already an eligible-only SQL query.  Keep
            # these checks as a defence against malformed future overlays.
            stored = [
                item
                for item in stored
                if item.primary_phase_id == mineral_id
                and item.purity_status == "declared_pure"
                and item.qc_status == "eligible"
                and item.source_role == "mineral_reference"
            ]
            record.eligible_measurement_count = len(stored)
            physics_compatible = [item for item in stored if item.data_physics.casefold() == physics]
            record.physics_compatible_measurement_count = len(physics_compatible)

            if record.reasons:
                continue
            if not stored:
                record.reasons.append(
                    V5UnavailabilityReason(
                        "no_declared_pure_reference",
                        "No conflict-free declared-pure mineral reference is available in the bundled database.",
                    )
                )
                continue
            if not physics_compatible:
                record.reasons.append(
                    V5UnavailabilityReason(
                        "no_physics_compatible_reference",
                        f"No declared-pure {physics} measurement is available for this target.",
                    )
                )
                continue

            mineral_candidates: list[V5ReferenceCandidate] = []
            anchor_specifications = _anchor_specs(mineral)
            for spectrum in physics_compatible:
                descriptor = describe_source(spectrum.source_id)
                resampled, method = _resample_with_optional_fwhm(
                    spectrum.wavelengths_nm,
                    spectrum.values,
                    wavelengths,
                    fwhm,
                )
                candidate = V5ReferenceCandidate(
                    mineral_id=mineral_id,
                    taxonomy_group_id=taxonomy_group,
                    spectral_family_id=spectral_family,
                    measurement_id=spectrum.measurement_id,
                    sample_id=spectrum.sample_id,
                    source_id=spectrum.source_id,
                    source_index=spectrum.source_index,
                    source_name=spectrum.raw_name,
                    source_family=descriptor.source_family,
                    source_role=spectrum.source_role,
                    purity_status=spectrum.purity_status,
                    qc_status=spectrum.qc_status,
                    data_physics=spectrum.data_physics,
                    measurement_geometry=spectrum.measurement_geometry,
                    resampling_method=method,
                    resampled=resampled,
                )
                candidate.catalog_anchor_order = _anchor_order(candidate, anchor_specifications)
                (
                    candidate.required_coverage,
                    candidate.absorption_depth,
                    candidate.roughness,
                    candidate.quality_score,
                    candidate.embedding,
                    candidate.rejection_reason,
                ) = _shape_diagnostics(
                    candidate.resampled,
                    wavelengths,
                    required_windows,
                    float(minimum_window_coverage),
                )
                candidates.append(candidate)
                mineral_candidates.append(candidate)

            independent_samples = _collapse_measurements_by_sample(mineral_candidates)
            record.independent_sample_count = len(independent_samples)
            if not independent_samples:
                record.reasons.append(
                    V5UnavailabilityReason(
                        "no_resampled_reference",
                        "Declared-pure references exist, but none adequately covers the sensor's required bands.",
                        {
                            "candidate_rejections": sorted(
                                {candidate.rejection_reason for candidate in mineral_candidates if candidate.rejection_reason}
                            )
                        },
                    )
                )
                continue
            if anchor_specifications and not any(item.catalog_anchor_order is not None for item in independent_samples):
                warnings.append(f"{mineral_id}: configured catalog anchors were incompatible; robust medoid used")
            chosen, selection_mode = _choose_references(
                independent_samples,
                int(maximum_references),
                int(minimum_references),
            )
            selected.extend(chosen)
            record.selected_reference_count = len(chosen)
            record.selection_mode = selection_mode
            record.available = bool(chosen)

    unavailable = [record for record in availability_records if not record.available]
    if require_all_available and unavailable:
        raise V5ReferenceSelectionError(unavailable)

    order = {mineral_id: index for index, mineral_id in enumerate(requested)}
    selected.sort(key=lambda item: (order[item.mineral_id], item.selected_rank or 0, item.identifier.casefold()))
    if selected:
        spectra = np.vstack([candidate.resampled for candidate in selected]).astype(np.float64, copy=False)
    else:
        spectra = np.empty((0, wavelengths.size), dtype=np.float64)
    names = tuple(
        f"{definitions[item.mineral_id]['display_name_en']} | {item.source_id} #{item.source_index} | {item.source_name}"
        for item in selected
    )
    return V5ReferenceLibrary(
        wavelengths_nm=wavelengths,
        spectra=spectra,
        mineral_labels=tuple(item.mineral_id for item in selected),
        group_labels=tuple(item.spectral_family_id for item in selected),
        spectrum_names=names,
        selected_candidates=tuple(selected),
        candidates=tuple(candidates),
        availability=tuple(availability_records),
        data_physics=physics,
        database_path=database_source,
        database_sha256=database_hash,
        database_release=str(summary["database_release"]),
        database_schema_version=int(summary["schema_version"]),
        catalog_path=catalog_source,
        catalog_sha256=catalog_hash,
        catalog_version=str(catalog["version"]),
        resampling_method="gaussian_fwhm_convolution" if fwhm is not None else "gap_constrained_interpolation",
        warnings=tuple(warnings),
    )


# Readable aliases for service/UI callers.  The V4 API remains untouched.
select_v5_references = build_v5_library_ensemble
build_v5_reference_library = build_v5_library_ensemble


__all__ = [
    "V5ReferenceCandidate",
    "V5ReferenceLibrary",
    "V5ReferenceSelectionError",
    "V5TargetAvailability",
    "V5UnavailabilityReason",
    "build_v5_library_ensemble",
    "build_v5_reference_library",
    "select_v5_references",
]
