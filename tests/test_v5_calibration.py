import json
import unittest

import numpy as np

from corespec_mapper.catalog import MineralCatalog
from corespec_mapper.v5_calibration import (
    CalibrationSearchSettings,
    apply_resolved_thresholds,
    calibrate_group_thresholds,
    catalog_safe_search_bounds,
)


class V5CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.group = MineralCatalog.load().group("clays")

    def test_catalog_bounds_are_the_conservative_to_sensitive_envelope(self):
        percentile, absolute = catalog_safe_search_bounds(self.group)
        self.assertEqual(percentile, (0.035, 0.10))
        self.assertEqual(absolute, (0.08, 0.13))

    def test_search_prefers_supported_patch_over_persistent_detector_column(self):
        scores = np.full((100, 40), 0.30, dtype=np.float64)
        valid = np.ones_like(scores, dtype=bool)
        scores[20:60, 10:25] = 0.05
        scores[:, 2] = 0.11
        result = calibrate_group_thresholds(
            self.group,
            "balanced",
            scores,
            valid,
            percentile_candidates=[0.05, 0.10],
            absolute_threshold_candidates_rad=[0.08, 0.13],
            settings=CalibrationSearchSettings(
                strata=4,
                minimum_candidate_pixels=20,
                maximum_coverage=0.25,
                hard_maximum_coverage=0.60,
            ),
        )
        self.assertTrue(result.ready, result.to_dict())
        self.assertEqual(result.resolved_absolute_threshold_rad, 0.08)
        low = next(item for item in result.candidates if item.percentile_fraction == 0.05 and item.absolute_threshold_rad == 0.08)
        wide = next(item for item in result.candidates if item.percentile_fraction == 0.05 and item.absolute_threshold_rad == 0.13)
        self.assertLess(low.fixed_column_penalty, wide.fixed_column_penalty)
        mapped = apply_resolved_thresholds(scores, valid, result)
        self.assertGreater(np.count_nonzero(mapped[20:60, 10:25]), 500)
        self.assertEqual(np.count_nonzero(mapped[:, 2]), 0)
        self.assertEqual(len(result.candidates), 4)
        self.assertIn("highest proxy objective", result.resolved_reason)
        json.dumps(result.to_dict())

    def test_every_grid_candidate_is_reported_when_coverage_gate_blocks_resolution(self):
        scores = np.full((40, 20), 0.05, dtype=np.float64)
        valid = np.ones_like(scores, dtype=bool)
        result = calibrate_group_thresholds(
            self.group,
            "balanced",
            scores,
            valid,
            percentile_candidates=[0.05, 0.10],
            absolute_threshold_candidates_rad=[0.08, 0.13],
            settings=CalibrationSearchSettings(
                minimum_candidate_pixels=8,
                maximum_coverage=0.20,
                hard_maximum_coverage=0.50,
            ),
        )
        self.assertFalse(result.ready)
        self.assertEqual(len(result.candidates), 4)
        self.assertTrue(all("hard_maximum_coverage_exceeded" in item.reasons for item in result.candidates))
        with self.assertRaisesRegex(ValueError, "calibration is blocked"):
            result.require_resolved()

    def test_identical_candidate_masks_do_not_reward_a_looser_absolute_threshold(self):
        scores = np.full((100, 20), 0.30, dtype=np.float64)
        valid = np.ones_like(scores, dtype=bool)
        scores[20:80, 5:15] = 0.05
        result = calibrate_group_thresholds(
            self.group,
            "balanced",
            scores,
            valid,
            percentile_candidates=[0.10],
            absolute_threshold_candidates_rad=[0.08, 0.13],
            settings=CalibrationSearchSettings(
                minimum_candidate_pixels=20,
                maximum_coverage=0.40,
                hard_maximum_coverage=0.60,
            ),
        )
        self.assertTrue(result.ready, result.to_dict())
        records = {item.absolute_threshold_rad: item for item in result.candidates}
        self.assertEqual(records[0.08].candidate_pixels, records[0.13].candidate_pixels)
        self.assertAlmostEqual(records[0.08].evidence_strength, records[0.13].evidence_strength)
        self.assertAlmostEqual(records[0.08].objective, records[0.13].objective)
        self.assertEqual(result.resolved_absolute_threshold_rad, 0.08)

    def test_explicit_candidates_cannot_escape_catalog_safe_bounds(self):
        scores = np.linspace(0.01, 0.20, 400).reshape(20, 20)
        valid = np.ones_like(scores, dtype=bool)
        with self.assertRaisesRegex(ValueError, "outside Catalog-safe bounds"):
            calibrate_group_thresholds(
                self.group,
                "balanced",
                scores,
                valid,
                percentile_candidates=[0.50],
                absolute_threshold_candidates_rad=[0.10],
            )


if __name__ == "__main__":
    unittest.main()
