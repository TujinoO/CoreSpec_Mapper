import unittest

import numpy as np

from corespec_mapper.v5_project_calibration import (
    calibrate_detection_settings,
    learn_feature_gate_override,
    learn_evidence_gate,
    learn_score_offsets,
)


class V5ProjectCalibrationTests(unittest.TestCase):
    def test_detection_threshold_uses_labels_but_honours_coverage_cap(self):
        scores = np.linspace(0.01, 0.20, 100).reshape(20, 5)
        material = np.ones_like(scores, dtype=bool)
        positive = np.zeros_like(material)
        positive[2:8, 1:4] = True
        training = np.ones_like(material)
        holdout = np.zeros_like(material)
        holdout[5:8] = True
        training[5:8] = False
        settings = {
            "minimum_pixels_per_mineral": 5,
            "detection_recall_targets": {"balanced": 0.85},
            "maximum_candidate_coverage": {"balanced": 0.24},
        }

        resolved, record = calibrate_detection_settings(
            scores,
            material,
            {"test": positive},
            training,
            holdout,
            "balanced",
            settings,
            {"absolute_sam_threshold_rad": 0.05},
        )

        self.assertEqual(record["status"], "resolved")
        self.assertLessEqual(record["candidate_coverage"], 0.25)
        self.assertEqual(np.asarray(resolved["column_thresholds_rad"]).shape, (5,))
        self.assertEqual(resolved["threshold_source"], "project_weak_label_train_with_heldout_depth_blocks")

    def test_score_offset_calibration_corrects_systematic_class_bias(self):
        scores = np.empty((2, 40, 1), dtype=np.float64)
        scores[0, :20, 0] = 0.30
        scores[1, :20, 0] = 0.20
        scores[0, 20:, 0] = 0.36
        scores[1, 20:, 0] = 0.18
        labels = {
            "left": np.arange(40)[:, None] < 20,
            "right": np.arange(40)[:, None] >= 20,
        }
        training = np.ones((40, 1), dtype=bool)
        holdout = np.zeros((40, 1), dtype=bool)

        offsets, record = learn_score_offsets(
            scores,
            ("left", "right"),
            labels,
            training,
            holdout,
            minimum_pixels=5,
        )

        self.assertEqual(record["status"], "resolved")
        self.assertGreater(offsets[0], offsets[1])
        calibrated = scores - offsets[:, None, None]
        self.assertGreater(np.mean(np.argmin(calibrated[:, :20], axis=0) == 0), 0.9)

    def test_score_offset_promotes_labeled_output_over_unlabeled_confuser(self):
        scores = np.empty((2, 40, 1), dtype=np.float64)
        scores[0, :, 0] = 0.34
        scores[1, :, 0] = 0.18
        labels = {"output": np.ones((40, 1), dtype=bool)}
        mask = np.ones((40, 1), dtype=bool)

        offsets, record = learn_score_offsets(
            scores,
            ("output", "confuser"),
            labels,
            mask,
            np.zeros_like(mask),
            minimum_pixels=5,
        )

        self.assertEqual(record["status"], "resolved")
        self.assertGreater(offsets[0], offsets[1])
        self.assertGreater(record["training_winner_recall"]["output"], 0.95)

    def test_feature_gate_is_learned_inside_catalog_feature_window(self):
        positive = np.ones((10, 1), dtype=bool)
        feature_maps = {
            "center": np.linspace(2180.0, 2240.0, 10)[:, None],
            "ratio": np.linspace(0.05, 0.50, 10)[:, None],
        }
        override, _ = learn_feature_gate_override(
            "test",
            positive,
            positive,
            feature_maps,
            {"center_feature": "center", "minimum_feature_values": {"ratio": 0.2}},
            "balanced",
            {"feature_lower_quantile": 0.1, "feature_upper_quantile": 0.9},
            {"center": (2190.0, 2235.0)},
        )

        self.assertGreaterEqual(override["center_window_nm"][0], 2190.0)
        self.assertLessEqual(override["center_window_nm"][1], 2235.0)
        self.assertLess(override["minimum_feature_values"]["ratio"], 0.2)

    def test_feature_gate_profiles_keep_later_evidence_strict_and_separated(self):
        positive = np.ones((100, 1), dtype=bool)
        feature_maps = {
            "center": np.linspace(2180.0, 2240.0, 100)[:, None],
            "ratio": np.linspace(0.0, 0.5, 100)[:, None],
        }
        settings = {
            "feature_lower_quantile": 0.02,
            "feature_upper_quantile": 0.98,
            "feature_quantiles_by_policy": {
                "conservative": [0.20, 0.80],
                "balanced": [0.05, 0.95],
                "sensitive": [0.01, 0.99],
            },
            "feature_catalog_floor_fraction": {
                "conservative": 0.75,
                "balanced": 0.45,
                "sensitive": 0.20,
            },
        }
        minima = {}
        widths = {}
        for policy in ("conservative", "balanced", "sensitive"):
            override, _ = learn_feature_gate_override(
                "test",
                positive,
                positive,
                feature_maps,
                {"center_feature": "center", "minimum_feature_values": {"ratio": 0.2}},
                policy,
                settings,
                {"center": (2180.0, 2240.0)},
            )
            minima[policy] = override["minimum_feature_values"]["ratio"]
            lower, upper = override["center_window_nm"]
            widths[policy] = upper - lower

        self.assertGreater(minima["conservative"], minima["balanced"])
        self.assertGreater(minima["balanced"], minima["sensitive"])
        self.assertLess(widths["conservative"], widths["balanced"])
        self.assertLess(widths["balanced"], widths["sensitive"])

    def test_evidence_gate_separates_reviewed_positive_cluster(self):
        rows = 200
        feature = np.linspace(-2.0, 2.0, rows)[:, None]
        positive = feature > 0.8
        eligible = np.ones((rows, 1), dtype=bool)
        training = np.ones_like(eligible)
        holdout = np.zeros_like(eligible)
        holdout[::5] = True
        training[::5] = False

        gate, record = learn_evidence_gate(
            [feature, feature**2],
            positive,
            eligible,
            training,
            holdout,
            minimum_pixels=10,
        )

        self.assertEqual(record["status"], "resolved")
        self.assertGreater(record["holdout"]["recall"], 0.8)
        self.assertLess(record["holdout"]["false_positive_rate"], 0.2)
        self.assertGreater(np.count_nonzero(gate & positive), 30)

    def test_evidence_gate_can_use_reviewed_within_group_competitors(self):
        rows = 240
        feature = np.linspace(-3.0, 3.0, rows)[:, None]
        positive = feature > 1.0
        competitor = feature < -1.0
        eligible = np.ones((rows, 1), dtype=bool)
        training = np.ones_like(eligible)
        holdout = np.zeros_like(eligible)
        holdout[::6] = True
        training[::6] = False

        gate, record = learn_evidence_gate(
            [feature, feature**2],
            positive,
            eligible,
            training,
            holdout,
            minimum_pixels=10,
            weak_negative=competitor,
        )

        self.assertEqual(record["training_negative_source"], "reviewed_within_group_competitors")
        self.assertGreater(np.count_nonzero(gate & positive), 50)

    def test_evidence_gate_resolves_nested_profiles_with_material_area_separation(self):
        rows = 600
        feature = np.linspace(-3.0, 3.0, rows)[:, None]
        # Reviewed positives overlap the scene distribution, as real
        # high-confidence weak labels do; a more sensitive profile therefore
        # has legitimate lower-ranked evidence to add.
        positive = (feature > -0.5) & ((np.arange(rows)[:, None] % 2) == 0)
        eligible = np.ones((rows, 1), dtype=bool)
        training = np.ones_like(eligible)
        holdout = np.zeros_like(eligible)
        holdout[::5] = True
        training[::5] = False

        profiles, record = learn_evidence_gate(
            [feature, feature**2],
            positive,
            eligible,
            training,
            holdout,
            minimum_pixels=20,
            return_profiles=True,
            profile_settings={
                "evidence_recall_targets": {
                    "conservative": 0.45,
                    "balanced": 0.80,
                    "sensitive": 0.98,
                },
                "evidence_area_multipliers": {
                    "conservative": 0.50,
                    "balanced": 1.00,
                    "sensitive": 2.00,
                },
                "evidence_maximum_domain_fraction": {
                    "conservative": 0.0,
                    "balanced": 0.0,
                    "sensitive": 0.0,
                },
            },
        )

        conservative = profiles["conservative"]
        balanced = profiles["balanced"]
        sensitive = profiles["sensitive"]
        self.assertFalse(np.any(conservative & ~balanced))
        self.assertFalse(np.any(balanced & ~sensitive))
        self.assertLess(np.count_nonzero(conservative), np.count_nonzero(balanced))
        self.assertLess(np.count_nonzero(balanced), np.count_nonzero(sensitive))
        self.assertIn("profiles", record)


if __name__ == "__main__":
    unittest.main()
