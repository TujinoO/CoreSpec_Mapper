from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from corespec_mapper.geocore_mask_adapter import inspect_geocore_m12, resize_mask_nearest


class GeocoreMaskAdapterTests(unittest.TestCase):
    def test_default_application_model_is_packaged_and_ready(self):
        status = inspect_geocore_m12()
        self.assertTrue(status.available, status.to_dict())
        self.assertTrue(status.code_complete)
        self.assertTrue(Path(str(status.weights)).is_file())

    def test_incomplete_delivery_is_reported_instead_of_silently_falling_back(self):
        with tempfile.TemporaryDirectory() as directory:
            status = inspect_geocore_m12(directory)
        self.assertFalse(status.available)
        self.assertFalse(status.code_complete)
        self.assertTrue(status.missing)

    def test_manifest_and_weights_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "geocore_mask/inference/predictor.py",
                "geocore_mask/inference/tiler.py",
                "geocore_mask/inference/merger.py",
                "geocore_mask/postprocess/refine_mask.py",
                "geocore_mask/models/registry.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            package = root / "models" / "core_mask_unet_v1"
            package.mkdir(parents=True)
            (package / "model_manifest.json").write_text(
                json.dumps({"weights": "weights.pth"}), encoding="utf-8"
            )
            missing_weights = inspect_geocore_m12(root)
            (package / "weights.pth").write_bytes(b"test")
            ready = inspect_geocore_m12(root)
        self.assertFalse(missing_weights.available)
        self.assertTrue(ready.available)

    def test_nearest_resize_preserves_binary_regions(self):
        source = np.array([[0, 1], [1, 0]], dtype=bool)
        resized = resize_mask_nearest(source, (4, 4))
        np.testing.assert_array_equal(
            resized,
            np.array(
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                ],
                dtype=bool,
            ),
        )


if __name__ == "__main__":
    unittest.main()
