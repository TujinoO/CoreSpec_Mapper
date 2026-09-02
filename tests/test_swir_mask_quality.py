import unittest

import numpy as np

from corespec_mapper.swir_mask_quality import (
    analyse_depth_coverage,
    refine_swir_mask_spatially,
)


class SwirMaskQualityTests(unittest.TestCase):
    def test_spatial_refinement_closes_cracks_fills_small_holes_and_removes_specks(self):
        mask = np.zeros((128, 64), dtype=bool)
        mask[16:112, 8:28] = True
        mask[16:112, 36:56] = True
        mask[50:55, 18:23] = False
        mask[63, 28:36] = True
        mask[2, 2] = True
        refined, audit = refine_swir_mask_spatially(
            mask, closing_size=3, maximum_hole_pixels=32, minimum_component_pixels=20
        )
        self.assertTrue(refined[50:55, 18:23].all())
        self.assertFalse(refined[2, 2])
        self.assertGreater(audit.filled_hole_pixels, 0)
        self.assertGreater(audit.removed_component_pixels, 0)

    def test_multiscale_depth_gate_flags_internal_collapse_but_not_blank_edges(self):
        mask = np.zeros((4096, 80), dtype=bool)
        mask[256:3840, 10:70] = True
        mask[1792:2304] = False
        blocks, regions = analyse_depth_coverage(mask, block_scales=(256, 512))
        critical = [item for item in regions if item.severity == "critical"]
        self.assertTrue(critical)
        self.assertTrue(any(item.start_row <= 1792 and item.stop_row >= 2304 for item in critical))
        self.assertFalse(any(item.anomaly and item.start_row == 0 for item in blocks))
        self.assertFalse(any(item.anomaly and item.stop_row == 4096 for item in blocks))

    def test_single_scale_alignment_effect_is_non_blocking_structural_gap(self):
        mask = np.ones((4096, 80), dtype=bool)
        mask[1088:1216] = False
        _, regions = analyse_depth_coverage(
            mask,
            block_scales=(256, 512),
            minimum_drop_fraction=0.20,
            maximum_drop_ratio=0.60,
        )
        self.assertTrue(regions)
        self.assertTrue(any(item.severity == "info" for item in regions))
        self.assertFalse(any(item.severity == "critical" for item in regions))


if __name__ == "__main__":
    unittest.main()
