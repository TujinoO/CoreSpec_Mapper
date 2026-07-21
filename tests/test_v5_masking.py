from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from corespec_mapper.masking import MaskGateError, build_material_mask


def _write_bsq_cube(path: Path, cube: np.ndarray) -> Path:
    lines, samples, bands = cube.shape
    np.moveaxis(np.asarray(cube, dtype="<f4"), -1, 0).tofile(path)
    wavelengths = ", ".join(str(1000 + 150 * index) for index in range(bands))
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
        f"wavelength = {{{wavelengths}}}\n",
        encoding="utf-8",
    )
    return path


class V5MaterialMaskTests(unittest.TestCase):
    def _synthetic_cube(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(24072026)
        cube = rng.normal(0.045, 0.0015, size=(96, 64, 7)).astype(np.float32)
        truth = np.zeros((96, 64), dtype=bool)
        truth[8:88, 8:26] = True
        truth[8:88, 38:56] = True
        signature = np.array([0.28, 0.32, 0.35, 0.30, 0.24, 0.27, 0.22], dtype=np.float32)
        cube[truth] = signature + rng.normal(0.0, 0.002, size=(int(truth.sum()), 7))
        cube[2:4, 30:32] = signature
        return cube, truth

    def test_automatic_mask_recovers_disconnected_core_material_and_audits_cleanup(self):
        cube, truth = self._synthetic_cube()
        with tempfile.TemporaryDirectory() as directory:
            image_path = _write_bsq_cube(Path(directory) / "reflectance.dat", cube)
            result = build_material_mask(
                image_path,
                minimum_mask_pixels=100,
                minimum_component_pixels=20,
                maximum_mask_fraction=0.75,
            )
        intersection = np.count_nonzero(result.mask & truth)
        union = np.count_nonzero(result.mask | truth)
        self.assertTrue(result.ready, result.audit.to_dict())
        self.assertGreater(intersection / union, 0.95)
        self.assertEqual(result.audit.source, "automatic_reflectance_material_mask")
        self.assertGreater(result.audit.removed_component_pixels, 0)
        self.assertGreater(result.audit.interior_pixel_fraction, 0.85)
        self.assertEqual(result.audit.selected_band_indices, tuple(range(7)))
        json.dumps(result.to_dict())

    def test_existing_same_grid_mask_is_preserved_and_serialisable(self):
        cube, truth = self._synthetic_cube()
        with tempfile.TemporaryDirectory() as directory:
            image_path = _write_bsq_cube(Path(directory) / "reflectance.dat", cube)
            mask_path = _write_bsq_cube(Path(directory) / "mask.dat", truth[..., None].astype(np.float32))
            result = build_material_mask(
                image_path,
                mask_path,
                minimum_mask_pixels=100,
                maximum_mask_fraction=0.75,
            )
        self.assertTrue(result.ready)
        self.assertEqual(result.audit.source, "existing_same_grid_mask")
        np.testing.assert_array_equal(result.mask, truth)
        json.dumps(result.audit.to_dict())

    def test_existing_mask_dimension_mismatch_is_rejected(self):
        cube, _ = self._synthetic_cube()
        with tempfile.TemporaryDirectory() as directory:
            image_path = _write_bsq_cube(Path(directory) / "reflectance.dat", cube)
            wrong = np.ones((cube.shape[0] - 1, cube.shape[1]), dtype=bool)
            with self.assertRaisesRegex(ValueError, "dimensions do not match"):
                build_material_mask(image_path, wrong)

    def test_full_frame_and_tiny_masks_fail_hard_gates(self):
        cube, _ = self._synthetic_cube()
        with tempfile.TemporaryDirectory() as directory:
            image_path = _write_bsq_cube(Path(directory) / "reflectance.dat", cube)
            full = build_material_mask(image_path, np.ones(cube.shape[:2], dtype=bool), maximum_mask_fraction=0.80)
            tiny_mask = np.zeros(cube.shape[:2], dtype=bool)
            tiny_mask[20, 20] = True
            tiny = build_material_mask(image_path, tiny_mask, minimum_mask_pixels=32)
            with self.assertRaises(MaskGateError):
                build_material_mask(
                    image_path,
                    np.ones(cube.shape[:2], dtype=bool),
                    maximum_mask_fraction=0.80,
                    strict=True,
                )
        self.assertFalse(full.ready)
        self.assertFalse(tiny.ready)
        self.assertIn("MASK_TOO_LARGE", {flag.code for flag in full.audit.flags})
        self.assertIn("MASK_TOO_SMALL", {flag.code for flag in tiny.audit.flags})

    def test_edge_network_diagnostic_distinguishes_outlines_from_filled_material(self):
        cube, _ = self._synthetic_cube()
        edge_like = np.zeros(cube.shape[:2], dtype=bool)
        edge_like[8:88, 8] = True
        edge_like[8:88, 25] = True
        edge_like[8, 8:26] = True
        edge_like[87, 8:26] = True
        with tempfile.TemporaryDirectory() as directory:
            image_path = _write_bsq_cube(Path(directory) / "reflectance.dat", cube)
            result = build_material_mask(
                image_path,
                edge_like,
                source_override="automatic_test_adapter",
                minimum_mask_pixels=100,
            )
        self.assertFalse(result.ready)
        self.assertEqual(result.audit.source, "automatic_test_adapter")
        self.assertLess(result.audit.interior_pixel_fraction, 0.20)
        self.assertIn("MASK_EDGE_LIKE", {flag.code for flag in result.audit.flags})


if __name__ == "__main__":
    unittest.main()
