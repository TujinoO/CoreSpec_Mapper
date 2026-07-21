from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

import numpy as np

from corespec_mapper.envi import parse_envi_header, write_envi
from corespec_mapper.v4_models import CancellationToken, RunCancelled
from corespec_mapper import v5_service


def _write_bsq_cube(
    path: Path,
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    *,
    spatial_metadata: bool = False,
) -> Path:
    lines, samples, bands = cube.shape
    np.moveaxis(np.asarray(cube, dtype="<f4"), -1, 0).tofile(path)
    wavelength_text = ", ".join(f"{item:.6f}" for item in wavelengths_nm)
    extra = ""
    if spatial_metadata:
        extra = (
            "map info = {UTM, 1, 1, 500000, 4100000, 2, 3, 48, North, WGS-84, units=Meters}\n"
            "coordinate system string = {PROJCS[\"Synthetic UTM\"]}\n"
            "x start = 10\n"
            "y start = 20\n"
        )
    path.with_suffix(".hdr").write_text(
        "ENVI\n"
        f"samples = {samples}\n"
        f"lines = {lines}\n"
        f"bands = {bands}\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 4\n"
        "interleave = bsq\n"
        "byte order = 0\n"
        "wavelength units = Nanometers\n"
        f"wavelength = {{{wavelength_text}}}\n"
        + extra,
        encoding="utf-8",
    )
    return path


def _write_mask(path: Path, mask: np.ndarray) -> Path:
    values = np.asarray(mask, dtype=np.uint8)
    values.tofile(path)
    path.with_suffix(".hdr").write_text(
        "ENVI\n"
        f"samples = {values.shape[1]}\n"
        f"lines = {values.shape[0]}\n"
        "bands = 1\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 1\n"
        "interleave = bsq\n"
        "byte order = 0\n",
        encoding="utf-8",
    )
    return path


def _synthetic_swir() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260720)
    wavelengths = np.linspace(700.0, 2500.0, 181, dtype=np.float64)
    material = np.zeros((48, 32), dtype=bool)
    material[4:44, 4:12] = True
    material[4:44, 20:28] = True
    cube = rng.normal(0.020, 0.0003, size=(48, 32, wavelengths.size)).astype(np.float32)
    continuum = 0.38 + 0.015 * np.sin(wavelengths / 170.0)
    absorption = 0.11 * np.exp(-0.5 * ((wavelengths - 2335.0) / 22.0) ** 2)
    spectrum = continuum - absorption
    cube[material] = spectrum + rng.normal(0.0, 0.001, size=(int(material.sum()), wavelengths.size))
    return cube, material, wavelengths


class _FakeReferenceLibrary:
    database_path = Path("corespec_spectral_v5.sqlite3")
    database_sha256 = "db-hash"
    database_release = "5.1.0"
    catalog_path = Path("mineral_catalog_v5.json")
    catalog_sha256 = "catalog-hash"
    catalog_version = "5.1.0"
    wavelengths_nm = np.array([1600.0])
    selected_candidates: tuple[object, ...] = ()

    def to_dict(self, **_kwargs):
        return {"schema_version": 1, "selected": []}


class _FakeMaskAudit:
    source = "existing_same_grid_mask"

    def to_dict(self):
        return {
            "status": "ready",
            "source": self.source,
            "final_mask_pixels": 12,
            "final_mask_fraction": 1.0,
            "flags": [],
        }


class V5ServiceTests(unittest.TestCase):
    def setUp(self):
        with v5_service._CACHE_LOCK:
            v5_service._SNAPSHOT_CACHE.clear()

    def tearDown(self):
        with v5_service._CACHE_LOCK:
            v5_service._SNAPSHOT_CACHE.clear()

    def test_audit_uses_bundled_library_with_external_and_automatic_masks(self):
        cube, material, wavelengths = _synthetic_swir()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = _write_bsq_cube(root / "swir.dat", cube, wavelengths)
            mask_path = _write_mask(root / "mask.dat", material)
            base = {
                "project": {"name": "Synthetic"},
                "inputs": {
                    "analysis_image": str(image_path),
                    "data_physics": "reflectance",
                    "rgb": str(image_path),
                },
                "minerals": {"requested": ["calcite"], "include_internal_confusers": True},
                "advanced": {
                    "preprocessing": {"sg_window": 7, "sg_polyorder": 2},
                    "sampling": {
                        "blocks": 4,
                        "block_rows": 16,
                        "minimum_valid_pixels_per_block": 64,
                        "maximum_sample_rows": 32,
                    },
                    "library_ensemble": {"maximum_representatives_per_mineral": 3},
                },
            }
            for mode in ("external", "automatic"):
                with self.subTest(mode=mode):
                    config = deepcopy(base)
                    config["mask"] = {
                        "mode": mode,
                        "path": str(mask_path) if mode == "external" else None,
                        "minimum_component_pixels": 16,
                    }
                    context = (
                        patch.object(
                            v5_service,
                            "run_geocore_m12",
                            return_value=SimpleNamespace(
                                mask=material,
                                metadata={
                                    "engine": "integrated_core_foreground_v2",
                                    "registration_review_required": False,
                                },
                            ),
                        )
                        if mode == "automatic"
                        else patch.object(v5_service, "run_geocore_m12", wraps=v5_service.run_geocore_m12)
                    )
                    with context:
                        audit = v5_service.audit_v5_project(config)
                    self.assertTrue(audit["ready"], audit)
                    expected_source = (
                        "existing_same_grid_mask"
                        if mode == "external"
                        else "automatic_integrated_rgb_foreground_registered"
                    )
                    self.assertEqual(audit["mask"]["source"], expected_source)
                    self.assertTrue(Path(audit["mask"]["preview_path"]).is_file())
                    self.assertEqual(audit["requested_minerals"], ["calcite"])
                    self.assertNotIn("spectral_library_root", audit)
                    self.assertTrue(any(item["mineral_id"] == "calcite" for item in audit["selected_references"]))
                    self.assertTrue(all(
                        item["support_level"] == item["engine_support_level"]
                        for item in audit["mineral_support"]
                    ))
                    self.assertEqual(audit["threshold_trial"]["status"], "resolved")
                    self.assertTrue(audit["thresholds"])
                    json.dumps(audit, allow_nan=False)

    def test_runtime_contract_rejects_ambiguous_mask_and_mineral_shapes(self):
        base = {"inputs": {"analysis_image": "cube.dat"}, "minerals": {"requested": ["calcite"]}}
        runtime = v5_service._runtime_config({**base, "mask": {"mode": "automatic"}})
        self.assertNotIn("spectral_library_root", runtime["v4"])
        self.assertNotIn("analysis_mask", runtime)

        external = deepcopy(base)
        external["mask"] = {"mode": "external", "path": "mask.dat"}
        self.assertEqual(v5_service._runtime_config(external)["analysis_mask"], "mask.dat")

        with self.assertRaisesRegex(ValueError, "requires mask.path"):
            v5_service._runtime_config({**base, "mask": {"mode": "external"}})
        with self.assertRaisesRegex(ValueError, "automatic.*external"):
            v5_service._runtime_config({**base, "mask": {"mode": "typo"}})
        with self.assertRaisesRegex(ValueError, "JSON array"):
            v5_service._runtime_config({"inputs": {"analysis_image": "cube.dat"}, "minerals": {"requested": "calcite"}})

    def test_cache_reuses_audit_across_desktop_navigation_and_tracks_top_level_input(self):
        cube, _, wavelengths = _synthetic_swir()
        with tempfile.TemporaryDirectory() as directory:
            image = _write_bsq_cube(Path(directory) / "cube.dat", cube[:4], wavelengths)
            first = {"analysis_image": str(image), "desktop": {"last_step": 1, "preview_key": "a"}}
            second = {"analysis_image": str(image), "desktop": {"last_step": 6, "preview_key": "b"}}
            self.assertEqual(v5_service._cache_key(first), v5_service._cache_key(second))
            before = v5_service._cache_key(first)
            with image.open("ab") as stream:
                stream.write(b"x")
            self.assertNotEqual(before, v5_service._cache_key(first))

    def test_json_output_is_strict_and_accepts_path_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "record.json"
            v5_service._save_json(
                {
                    "path": Path(directory),
                    "finite": 1.5,
                    "nan": float("nan"),
                    "array": np.array([1.0, np.nan]),
                    "flag": np.bool_(True),
                },
                output,
            )
            text = output.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            record = json.loads(text)
            self.assertEqual(record["path"], directory)
            self.assertIsNone(record["nan"])
            self.assertEqual(record["array"], [1.0, None])
            self.assertIs(record["flag"], True)

    def test_optional_orphan_header_is_reported_without_blocking_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "NIR.hdr"
            header.write_text(
                "ENVI\nsamples = 2\nlines = 2\nbands = 2\ndata type = 4\ninterleave = bil\nbyte order = 0\n",
                encoding="utf-8",
            )
            record = v5_service._input_record("nir", header)
            self.assertEqual(record["status"], "header_only_missing_data")
            self.assertFalse(record["blocking"])
            with self.assertRaises(FileNotFoundError):
                v5_service._input_record("analysis", header, required=True)

    def test_snapshot_reuse_threshold_manifest_spatial_metadata_cancellation_and_non_overwrite(self):
        cube = np.ones((4, 3, 1), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = _write_bsq_cube(
                root / "analysis.dat",
                cube,
                np.array([1600.0]),
                spatial_metadata=True,
            )
            output_root = root / "outputs"
            config = {
                "project": {"name": "Reuse Test", "output_root": str(output_root)},
                "inputs": {"analysis_image": str(image_path)},
                "mask": {"mode": "automatic"},
                "minerals": {"requested": ["calcite"]},
                "desktop": {"last_step": 4},
            }
            prepare_calls: list[str] = []
            prepared_marker = object()

            def fake_prepare(current, **_kwargs):
                key = v5_service._cache_key(current)
                prepare_calls.append(key)
                return v5_service.V5AuditSnapshot(
                    cache_key=key,
                    runtime_config={
                        "analysis_image": str(image_path),
                        "analysis_input_is_smoothed": False,
                        "chunk_rows": 2,
                        "v4": {"algorithm_version": "V5.1.0"},
                    },
                    catalog=SimpleNamespace(version="5.1.0"),
                    knowledge_catalog={},
                    prepared=prepared_marker,
                    material_mask=SimpleNamespace(mask=np.ones((4, 3), dtype=bool), audit=_FakeMaskAudit()),
                    reference_library=_FakeReferenceLibrary(),
                    support={},
                    audit_record={
                        "format": "CoreSpec Mapper V5 Audit",
                        "ready": True,
                        "inputs": {"analysis": {"data_path": str(image_path)}},
                        "audit_elapsed_seconds": 0.25,
                    },
                )

            threshold_search = {
                "status": "resolved",
                "safe_percentile_bounds": [0.02, 0.12],
                "safe_absolute_threshold_bounds_rad": [0.08, 0.15],
                "resolved_percentile_fraction": 0.06,
                "resolved_absolute_threshold_rad": 0.11,
                "candidates": [{"objective": 0.8}],
                "resolved_reason": "synthetic optimum",
            }

            def fake_run_v4(runtime, _run, *, start_line, stop_line, prepared, **_kwargs):
                self.assertIs(prepared, prepared_marker)
                self.assertNotIn("spectral_library_root", runtime.get("v4", {}))
                return {
                    "output_lines": [start_line, 4 if stop_line is None else stop_line],
                    "groups": {"carbonate": {"threshold_search": {"balanced": threshold_search}}},
                    "sensor_signature": "synthetic-sensor",
                    "requested_minerals": ["calcite"],
                    "internal_competitors": ["calcite"],
                    "elapsed_seconds_before_qa": 0.1,
                }

            events = []
            with (
                patch.object(v5_service, "_prepare_snapshot", side_effect=fake_prepare),
                patch.object(v5_service, "run_v4", side_effect=fake_run_v4),
                patch.object(v5_service, "make_v4_previews", return_value={}),
                patch.object(v5_service, "validate_v4_run", return_value={"status": "Completed", "quality_grade": "A"}),
            ):
                v5_service.audit_v5_project(config)
                run_config = deepcopy(config)
                run_config["desktop"] = {"last_step": 5, "preview_key": "comparison_balanced"}
                result = v5_service.run_v5_project(
                    run_config,
                    output_root,
                    run_id="fixed_run",
                    start_line=1,
                    stop_line=3,
                    progress=events.append,
                )
                self.assertEqual(len(prepare_calls), 1)
                self.assertTrue(result["manifest"]["timing"]["audit_snapshot_reused"])
                self.assertTrue(any(event.stage == "audit_reuse" for event in events))

                run = Path(result["run_directory"])
                manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
                inventory = [item["path"] for item in manifest["outputs"]]
                self.assertNotIn("run_manifest.json", inventory)
                self.assertEqual(manifest["thresholds"]["carbonate"]["balanced"]["final"]["absolute_sam_rad"], 0.11)
                self.assertEqual(
                    json.loads((run / "thresholds.json").read_text(encoding="utf-8"))["carbonate"]["balanced"]["final"]["column_percentile"],
                    0.06,
                )
                mask_header = parse_envi_header(run / "capability" / "material_mask.hdr")
                self.assertAlmostEqual(float(mask_header["map info"][4]), 4099997.0)
                self.assertEqual(mask_header["y start"], 21)
                self.assertEqual(mask_header["corespec source start line"], 1)

                marker = run / "user_marker.txt"
                marker.write_text("keep", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    v5_service.run_v5_project(run_config, output_root, run_id="fixed_run")
                self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

                token = CancellationToken()
                token.cancel()
                with self.assertRaises(RunCancelled):
                    v5_service.run_v5_project(
                        run_config,
                        output_root,
                        run_id="cancelled_run",
                        cancel_token=token,
                        progress=lambda _event: None,
                    )
                self.assertFalse((output_root / "Reuse_Test" / "cancelled_run").exists())

    def test_run_id_cannot_escape_output_root_or_use_windows_devices(self):
        config = {"inputs": {"analysis_image": "unused.dat"}}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            v5_service,
            "_prepare_snapshot",
            side_effect=AssertionError("invalid run id must fail before audit"),
        ):
            for run_id in ("../escape", "nested/run", "..", "CON", "has space"):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    v5_service.run_v5_project(config, directory, run_id=run_id, progress=lambda _event: None)

    def test_v5_subtype_preview_copies_envi_data_before_closing_memmap(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            image = np.linspace(0.2, 0.6, 24, dtype=np.float32).reshape(4, 6, 1)
            image_path = _write_bsq_cube(run / "image.dat", image, np.array([1600.0]))
            mask = np.ones((4, 6), dtype=bool)
            subtype_dir = run / "groups" / "white_mica_illite"
            write_envi(
                np.array(
                    [
                        [0, 1, 1, 2, 3, 4],
                        [0, 1, 2, 2, 3, 4],
                        [0, 0, 2, 3, 3, 4],
                        [0, 1, 2, 3, 3, 4],
                    ],
                    dtype=np.uint8,
                ),
                subtype_dir / "aloh_wavelength_subtype.dat",
                class_names=[
                    "Unclassified",
                    "Short-wave Al-OH",
                    "Medium-wave Al-OH",
                    "Long-wave Al-OH",
                    "Masked Pixels",
                ],
            )
            previews: dict[str, str] = {}

            v5_service._add_v5_previews(run, previews, image_path, mask, 0, 4)

            self.assertTrue(Path(previews["material_mask"]).is_file())
            self.assertTrue(Path(previews["aloh_wavelength_subtype"]).is_file())


if __name__ == "__main__":
    unittest.main()
