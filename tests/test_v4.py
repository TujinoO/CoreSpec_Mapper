from pathlib import Path
import unittest

import numpy as np

from corespec_mapper.artifacts import (
    edge_risk_score,
    filter_artifacts,
    low_exposure_high_density_column_mask,
    repeated_segment_column_stripe_mask,
)
from corespec_mapper.catalog import FeatureDefinition, GroupDefinition, MineralCatalog, MineralDefinition
from corespec_mapper.qa import _repeated_column_residual_metrics
from corespec_mapper.sampling import build_stratified_sample_plan, sample_plan_mask
from corespec_mapper.sensor import infer_spectral_domain, resolve_mineral_support
from corespec_mapper.v4_calibration import policy_candidate, resolve_group_policies
from corespec_mapper.v4_library import resample_with_fwhm
from corespec_mapper.v4_models import CancellationToken, RunCancelled, SampleBlock, SamplePlan, SensorCapabilityCard, SupportLevel
from corespec_mapper.v4_service import audit_v4_project
from corespec_mapper.swir_expert import mineral_feature_gate, reference_evidence
from corespec_mapper.v4_pipeline import (
    _apply_post_nesting_repeated_column_filter,
    _arbitrate_cross_group_profiles,
    _competition_margin_pass,
    _enforce_profile_nesting,
    _reference_consensus_threshold,
    _winning_mineral_sff_quality,
)


class V4CatalogAndCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.catalog = MineralCatalog.load()
        self.wavelengths = np.linspace(1000.0, 2500.0, 212)
        self.card = SensorCapabilityCard(
            sensor_signature="test",
            spectral_domain="nir+swir",
            data_physics="reflectance",
            wavelength_min_nm=1000.0,
            wavelength_max_nm=2500.0,
            band_count=212,
            valid_band_count=212,
            median_spacing_nm=float(np.median(np.diff(self.wavelengths))),
            fwhm_status="measured",
            fwhm_median_nm=10.0,
            estimated_snr=120.0,
            bad_band_indices=(),
            detector_axis="samples",
            preprocessing_state="raw",
            implemented_experts=("swir_reflectance",),
        )

    def test_catalog_has_three_profiles_for_every_group(self):
        for group in self.catalog.groups.values():
            self.assertEqual(set(group.policies), {"conservative", "balanced", "sensitive"})

    def test_actual_wavelengths_gate_vnir_and_tir_minerals(self):
        support = resolve_mineral_support(
            self.card,
            self.catalog,
            wavelengths_nm=self.wavelengths,
            reference_counts={mineral: 3 for mineral in self.catalog.minerals},
        )
        self.assertEqual(support["calcite"].level, SupportLevel.SUPPORTED)
        self.assertEqual(support["hematite"].level, SupportLevel.UNSUPPORTED)
        self.assertEqual(support["quartz"].level, SupportLevel.UNSUPPORTED)
        self.assertIn("not implemented", " ".join(support["quartz"].reasons))

    def test_spectral_domain_uses_wavelength_coverage(self):
        self.assertIn("swir", infer_spectral_domain(self.wavelengths))
        self.assertEqual(infer_spectral_domain(np.linspace(8000, 12000, 100)), "tir")

    def test_pre_cancelled_audit_stops_before_opening_input_files(self):
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(RunCancelled):
            audit_v4_project({"v4": {}}, cancel_token=token)


class V4SamplingAndCalibrationTests(unittest.TestCase):
    def test_sample_plan_covers_full_depth_without_overlap(self):
        mask = np.zeros((1000, 24), dtype=bool)
        mask[:, 3:21] = True
        plan = build_stratified_sample_plan(mask, desired_blocks=10, block_rows=40)
        self.assertGreaterEqual(len(plan.blocks), 8)
        self.assertLess(plan.blocks[0].start_line, 100)
        self.assertGreater(plan.blocks[-1].stop_line, 900)
        for left, right in zip(plan.blocks[:-1], plan.blocks[1:]):
            self.assertLessEqual(left.stop_line, right.start_line)
        selected = sample_plan_mask(plan, mask)
        self.assertEqual(int(selected.sum()), plan.total_valid_pixels)

    def test_three_policy_candidates_are_nested(self):
        catalog = MineralCatalog.load()
        group = catalog.group("carbonates")
        rows, columns = 200, 8
        scores = np.tile(np.linspace(0.02, 0.18, rows)[:, None], (1, columns))
        mask = np.ones_like(scores, dtype=bool)
        card = SensorCapabilityCard(
            "test", "swir", "reflectance", 1000, 2500, 212, 212, 7.0,
            "unknown", None, 100.0, (), "samples", "raw", ("swir_reflectance",), (),
        )
        policies = resolve_group_policies(group, card, scores, mask, relative_noise=0.0)
        candidates = {name: policy_candidate(scores, mask, settings) for name, settings in policies.items()}
        self.assertTrue(np.all(~candidates["conservative"] | candidates["balanced"]))
        self.assertTrue(np.all(~candidates["balanced"] | candidates["sensitive"]))


class V4ReferenceAndArtifactTests(unittest.TestCase):
    def test_missing_optional_feature_does_not_poison_reference_score(self):
        group = GroupDefinition(
            group_id="test",
            display_name="test",
            expert_id="swir_reflectance",
            detection_windows_nm=((2200.0, 2250.0),),
            classification_windows_nm=((2200.0, 2250.0),),
            features=(
                FeatureDefinition("observable", "depth_ratio", (2200.0, 2250.0)),
                FeatureDefinition("missing", "depth_ratio", (1900.0, 1950.0)),
            ),
            score_weights={"shape": 0.5, "derivative": 0.2, "features": 0.3, "fit": 0.0},
            policies={},
            domain_calibration_strength=0.0,
            detection_best_k=1,
            classification_best_k=1,
            spatial_cleanup={},
        )
        wavelengths = np.array([2200.0, 2225.0, 2250.0])
        absorption = np.array([[0.0, 0.2, 0.0]])

        evidence = reference_evidence(group, absorption, absorption.copy(), wavelengths)

        self.assertTrue(np.isfinite(evidence.total[0, 0]))
        self.assertAlmostEqual(float(evidence.total[0, 0]), 0.0, places=12)

    def test_single_mineral_group_bypasses_nonexistent_margin(self):
        margin = np.array([[np.nan, np.nan]])
        classified = np.array([[True, False]])

        passed = _competition_margin_pass(
            margin,
            0.02,
            mineral_count=1,
            classified=classified,
        )

        np.testing.assert_array_equal(passed, classified)

    def test_reference_consensus_floor_tracks_discrete_reference_count(self):
        self.assertAlmostEqual(_reference_consensus_threshold(0.50, "balanced", 3), 1.0 / 3.0)
        self.assertAlmostEqual(_reference_consensus_threshold(0.66, "conservative", 3), 0.66)
        self.assertAlmostEqual(_reference_consensus_threshold(0.54, "balanced", 4), 0.25)

    def test_sensitive_feature_gate_can_require_any_one_diagnostic_feature(self):
        mineral = MineralDefinition(
            mineral_id="test",
            display_name_en="Test",
            display_name_zh="Test",
            formula="",
            group_id="test",
            name_patterns=(),
            reject_patterns=(),
            confusers=(),
            canonical_anchors=(),
            experts={
                "swir_reflectance": {
                    "feature_gate": {
                        "minimum_feature_values": {"a": 0.2, "b": 0.3},
                    }
                }
            },
            support_level="supported",
        )
        passed = mineral_feature_gate(
            mineral,
            "swir_reflectance",
            "sensitive",
            np.ones((1, 3)),
            {
                "a": np.array([[0.25, 0.10, 0.10]]),
                "b": np.array([[0.10, 0.35, 0.10]]),
            },
            gate_override={"minimum_feature_match": "any"},
        )
        np.testing.assert_array_equal(passed, [[True, True, False]])

    def test_sff_quality_is_taken_only_from_the_winning_mineral_references(self):
        quality = np.array(
            [
                [2.0, 9.0, 4.0],
                [8.0, 1.0, 3.0],
            ]
        )
        resolved = _winning_mineral_sff_quality(
            quality,
            ("calcite", "dolomite", "dolomite"),
            ("calcite", "dolomite"),
            np.array([0, 1]),
        )
        np.testing.assert_allclose(resolved, [2.0, 3.0])

    def test_profile_nesting_preserves_stronger_cleaned_evidence(self):
        profiles = {
            "conservative": np.array([[1, 0, 0]], dtype=np.uint8),
            "balanced": np.array([[0, 2, 0]], dtype=np.uint8),
            "sensitive": np.array([[0, 0, 1]], dtype=np.uint8),
        }

        _enforce_profile_nesting(profiles)

        np.testing.assert_array_equal(profiles["balanced"], [[1, 2, 0]])
        np.testing.assert_array_equal(profiles["sensitive"], [[1, 2, 1]])

    def test_post_nesting_stripe_removal_cascades_to_stricter_profiles(self):
        valid = np.zeros((150, 50), dtype=bool)
        profiles = {
            policy: np.zeros(valid.shape, dtype=np.uint8)
            for policy in ("conservative", "balanced", "sensitive")
        }
        for start in (10, 55, 100):
            valid[start : start + 30] = True
            profiles["sensitive"][start + 3 : start + 25, 22] = 1
        profiles["balanced"][13:35, 22] = 1
        profiles["conservative"][13:35, 22] = 1

        applied, records = _apply_post_nesting_repeated_column_filter(
            profiles,
            valid,
            {},
        )

        self.assertEqual(records["sensitive"]["detected_pixels"], 66)
        self.assertEqual(records["balanced"]["detected_pixels"], 0)
        self.assertEqual(records["balanced"]["inherited_pixels"], 22)
        self.assertEqual(np.count_nonzero(applied["conservative"]), 22)
        self.assertFalse(any(np.any(profile) for profile in profiles.values()))

    def test_quality_metrics_detect_repeated_narrow_columns_by_class(self):
        valid = np.zeros((150, 50), dtype=bool)
        labels = np.zeros(valid.shape, dtype=np.uint8)
        for start in (10, 55, 100):
            valid[start : start + 30] = True
            labels[start + 3 : start + 25, 22] = 1

        metrics = _repeated_column_residual_metrics(
            labels,
            valid,
            ["Unclassified", "Illite", "Masked Pixels"],
        )

        self.assertEqual(metrics["maximum_segment_repetition"], 3)
        self.assertEqual(metrics["repeated_narrow_column_count"], 1)
        self.assertEqual(metrics["repeated_stripe_like_pixels"], 66)
        self.assertEqual(
            metrics["classes"]["Illite"]["repeated_columns"],
            [22],
        )

    def test_cross_group_competition_uses_one_shared_confidence_winner(self):
        profiles = {
            "illite": {
                "conservative": np.array([[1, 0]], dtype=np.uint8),
                "balanced": np.array([[1, 0]], dtype=np.uint8),
                "sensitive": np.array([[1, 1]], dtype=np.uint8),
            },
            "smectite": {
                "conservative": np.array([[0, 0]], dtype=np.uint8),
                "balanced": np.array([[1, 0]], dtype=np.uint8),
                "sensitive": np.array([[1, 0]], dtype=np.uint8),
            },
        }
        confidence = {
            "illite": np.array([[0.4, 0.7]], dtype=np.float32),
            "smectite": np.array([[0.8, 0.0]], dtype=np.float32),
        }

        record = _arbitrate_cross_group_profiles(
            profiles,
            confidence,
            ("illite", "smectite"),
        )

        self.assertEqual(record["profiles"]["balanced"]["overlap_pixels_before"], 1)
        self.assertEqual(record["profiles"]["balanced"]["overlap_pixels_after"], 0)
        np.testing.assert_array_equal(profiles["illite"]["balanced"], [[0, 0]])
        np.testing.assert_array_equal(profiles["smectite"]["balanced"], [[1, 0]])

    def test_edge_risk_marks_each_disconnected_core_segment(self):
        mask = np.zeros((1, 14), dtype=bool)
        mask[0, 1:6] = True
        mask[0, 8:13] = True
        risk = edge_risk_score(mask, width=1)
        np.testing.assert_array_equal(np.flatnonzero(risk[0]), [1, 5, 8, 12])

    def test_fwhm_resampling_preserves_absorption_location(self):
        source_wavelengths = np.linspace(2200.0, 2450.0, 501)
        source = 0.8 - 0.22 * np.exp(-0.5 * ((source_wavelengths - 2340.0) / 6.0) ** 2)
        target = np.linspace(2220.0, 2430.0, 80)
        resampled, method = resample_with_fwhm(source_wavelengths, source, target, np.full(target.size, 10.0))
        self.assertEqual(method, "gaussian_fwhm_convolution")
        self.assertLess(abs(float(target[np.nanargmin(resampled)]) - 2340.0), 5.0)

    def test_strong_spectral_evidence_protects_a_thin_vertical_feature(self):
        valid = np.ones((81, 31), dtype=bool)
        labels = np.zeros(valid.shape, dtype=np.uint8)
        labels[5:76, 5] = 1
        settings = {
            "vertical_window": 31,
            "min_vertical_density": 0.3,
            "column_ratio": 2.0,
            "column_excess": 0.1,
            "max_lateral_support": 2,
            "min_line_height": 20,
            "max_line_width": 2,
            "min_aspect": 5.0,
            "min_component": 2,
            "strong_evidence_protection": 0.72,
        }
        low, _, _ = filter_artifacts(labels, np.full(valid.shape, 0.2), valid, settings, policy="balanced")
        high, _, _ = filter_artifacts(labels, np.full(valid.shape, 0.9), valid, settings, policy="balanced")
        self.assertEqual(np.count_nonzero(low), 0)
        self.assertEqual(np.count_nonzero(high), 71)

    def test_project_calibration_can_disable_directional_filter_for_one_class(self):
        valid = np.ones((81, 31), dtype=bool)
        labels = np.zeros(valid.shape, dtype=np.uint8)
        labels[5:76, 5] = 1
        settings = {
            "vertical_window": 31,
            "min_vertical_density": 0.3,
            "column_ratio": 2.0,
            "column_excess": 0.1,
            "max_lateral_support": 2,
            "min_line_height": 200,
            "max_line_width": 2,
            "min_aspect": 5.0,
            "min_component": 2,
            "strong_evidence_protection": 0.72,
            "disable_directional_filter_by_class": [1],
            "minimum_component_by_class": {1: 1},
        }

        cleaned, _, counts = filter_artifacts(
            labels,
            np.full(valid.shape, 0.2),
            valid,
            settings,
            policy="balanced",
        )

        self.assertEqual(np.count_nonzero(cleaned), 71)
        self.assertEqual(counts["directional_removed"], 0)

    def test_repeated_weak_multicolumn_band_is_removed(self):
        valid = np.ones((120, 30), dtype=bool)
        labels = np.zeros(valid.shape, dtype=np.uint8)
        labels[10:110, 12:17] = 1
        settings = {
            "vertical_window": 31,
            "min_vertical_density": 0.3,
            "column_ratio": 2.0,
            "column_excess": 0.1,
            "max_lateral_support": 2,
            "min_line_height": 20,
            "max_line_width": 2,
            "min_aspect": 5.0,
            "min_component": 1,
            "strong_evidence_protection": 0.72,
        }
        risk = np.zeros(30)
        risk[12:17] = 0.5
        cleaned, _, counts = filter_artifacts(
            labels,
            np.full(valid.shape, 0.3),
            valid,
            settings,
            policy="balanced",
            column_risk=risk,
        )
        self.assertEqual(np.count_nonzero(cleaned), 0)
        self.assertGreater(counts["residual_column_band_removed"], 0)

    def test_repeated_confident_detector_band_is_removed_without_outside_support(self):
        valid = np.ones((160, 40), dtype=bool)
        labels = np.zeros(valid.shape, dtype=np.uint8)
        labels[5:155, 18:21] = 1
        settings = {
            "vertical_window": 31,
            "min_vertical_density": 0.3,
            "column_ratio": 2.0,
            "column_excess": 0.1,
            "max_lateral_support": 2,
            "min_line_height": 20,
            "max_line_width": 2,
            "min_aspect": 5.0,
            "min_component": 1,
            "strong_evidence_protection": 0.72,
        }
        risk = np.zeros(40)
        risk[18:21] = 0.9

        cleaned, _, counts = filter_artifacts(
            labels,
            np.full(valid.shape, 0.95),
            valid,
            settings,
            policy="sensitive",
            column_risk=risk,
        )

        self.assertEqual(np.count_nonzero(cleaned), 0)
        self.assertGreater(
            counts["directional_removed"] + counts["residual_column_band_removed"],
            0,
        )

    def test_repeated_low_risk_residual_column_peak_is_still_removed(self):
        valid = np.ones((160, 40), dtype=bool)
        labels = np.zeros(valid.shape, dtype=np.uint8)
        labels[5:155, 20] = 1
        settings = {
            "vertical_window": 31,
            "min_vertical_density": 0.3,
            "column_ratio": 2.0,
            "column_excess": 0.1,
            "max_lateral_support": 2,
            "min_line_height": 2000,
            "max_line_width": 2,
            "min_aspect": 5.0,
            "min_component": 1,
            "strong_evidence_protection": 0.72,
            "directional_column_risk_threshold": 1.0,
            "column_risk_threshold": 1.0,
        }
        risk = np.zeros(40)
        risk[20] = 0.15
        cleaned, _, counts = filter_artifacts(
            labels,
            np.full(valid.shape, 0.3),
            valid,
            settings,
            policy="balanced",
            column_risk=risk,
        )
        self.assertEqual(np.count_nonzero(cleaned), 0)
        self.assertGreater(counts["residual_column_band_removed"], 0)

    def test_low_exposure_isolated_column_is_removed(self):
        valid = np.ones((400, 40), dtype=bool)
        valid[:, 8] = False
        valid[120:160, 8] = True
        candidate = np.zeros_like(valid)
        candidate[123:157, 8] = True

        removed, record = low_exposure_high_density_column_mask(
            candidate, valid, {}, policy="balanced"
        )

        self.assertEqual(record["columns"], [8])
        self.assertEqual(record["candidate_removed_pixels"], 34)
        self.assertTrue(np.array_equal(removed, candidate))

    def test_low_exposure_column_preserves_lateral_crossing(self):
        valid = np.ones((400, 40), dtype=bool)
        valid[:, 8] = False
        valid[120:160, 8] = True
        candidate = np.zeros_like(valid)
        candidate[123:157, 8] = True
        candidate[140:146, 5:12] = True

        removed, _ = low_exposure_high_density_column_mask(
            candidate, valid, {}, policy="balanced"
        )
        cleaned = candidate & ~removed

        self.assertTrue(np.all(cleaned[140:146, 5:12]))
        self.assertFalse(cleaned[130, 8])

    def test_local_geologic_patch_crossing_risky_column_is_preserved(self):
        valid = np.ones((160, 40), dtype=bool)
        labels = np.zeros(valid.shape, dtype=np.uint8)
        labels[5:155, 20] = 1
        labels[72:88, 15:26] = 1
        settings = {
            "vertical_window": 31,
            "min_vertical_density": 0.3,
            "column_ratio": 2.0,
            "column_excess": 0.1,
            "max_lateral_support": 2,
            "min_line_height": 20,
            "max_line_width": 2,
            "min_aspect": 5.0,
            "min_component": 1,
            "strong_evidence_protection": 0.72,
        }
        risk = np.zeros(40)
        risk[20] = 0.9

        cleaned, _, _ = filter_artifacts(
            labels,
            np.full(valid.shape, 0.95),
            valid,
            settings,
            policy="balanced",
            column_risk=risk,
        )

        self.assertEqual(np.count_nonzero(cleaned[:60, 20]), 0)
        self.assertGreaterEqual(np.count_nonzero(cleaned[72:88, 15:26]), 170)

    def test_oblique_geologic_band_is_not_treated_as_fixed_column_noise(self):
        valid = np.ones((140, 50), dtype=bool)
        labels = np.zeros(valid.shape, dtype=np.uint8)
        for row in range(15, 125):
            column = 8 + (row - 15) // 6
            labels[row, column : column + 2] = 1
        settings = {
            "vertical_window": 31,
            "min_vertical_density": 0.3,
            "column_ratio": 2.0,
            "column_excess": 0.1,
            "max_lateral_support": 2,
            "min_line_height": 20,
            "max_line_width": 2,
            "min_aspect": 5.0,
            "min_component": 1,
            "strong_evidence_protection": 0.72,
        }
        risk = np.full(50, 0.6)

        cleaned, _, _ = filter_artifacts(
            labels,
            np.full(valid.shape, 0.9),
            valid,
            settings,
            policy="balanced",
            column_risk=risk,
        )

        self.assertGreaterEqual(np.count_nonzero(cleaned), 200)

    def test_same_narrow_column_repeated_across_core_sections_is_removed(self):
        valid = np.zeros((150, 50), dtype=bool)
        candidate = np.zeros_like(valid)
        for start in (10, 55, 100):
            valid[start : start + 30] = True
            candidate[start + 3 : start + 25, 22] = True

        removed, record = repeated_segment_column_stripe_mask(
            candidate,
            valid,
            {},
            policy="balanced",
        )

        self.assertTrue(np.array_equal(removed, candidate))
        self.assertEqual(record["depth_segment_count"], 3)
        self.assertEqual(record["repeated_column_count"], 1)
        self.assertEqual(record["maximum_segment_repetition"], 3)

    def test_single_core_section_vertical_feature_is_preserved(self):
        valid = np.ones((150, 50), dtype=bool)
        candidate = np.zeros_like(valid)
        candidate[15:135, 22] = True

        removed, record = repeated_segment_column_stripe_mask(
            candidate,
            valid,
            {},
            policy="balanced",
        )

        self.assertFalse(np.any(removed))
        self.assertEqual(record["depth_segment_count"], 1)

    def test_oblique_support_crossing_repeated_column_is_preserved_locally(self):
        valid = np.zeros((150, 50), dtype=bool)
        candidate = np.zeros_like(valid)
        crossing_pixels: list[tuple[int, int]] = []
        for start in (10, 55, 100):
            valid[start : start + 30] = True
            candidate[start + 3 : start + 25, 22] = True
            for offset, column in enumerate(range(18, 27)):
                row = start + 12 + offset // 2
                candidate[row, column] = True
                crossing_pixels.append((row, column))

        removed, record = repeated_segment_column_stripe_mask(
            candidate,
            valid,
            {},
            policy="balanced",
        )
        cleaned = candidate & ~removed

        self.assertGreater(record["candidate_removed_pixels"], 0)
        self.assertTrue(all(cleaned[row, column] for row, column in crossing_pixels))
        self.assertFalse(cleaned[13, 22])

    def test_lateral_patch_crossing_repeated_column_is_preserved(self):
        valid = np.zeros((150, 50), dtype=bool)
        candidate = np.zeros_like(valid)
        patches: list[tuple[slice, slice]] = []
        for start in (10, 55, 100):
            valid[start : start + 30] = True
            candidate[start + 3 : start + 25, 22] = True
            patch = (slice(start + 12, start + 18), slice(17, 28))
            candidate[patch] = True
            patches.append(patch)

        removed, record = repeated_segment_column_stripe_mask(
            candidate,
            valid,
            {},
            policy="balanced",
        )
        cleaned = candidate & ~removed

        self.assertGreater(record["candidate_removed_pixels"], 0)
        self.assertTrue(all(np.all(cleaned[patch]) for patch in patches))
        self.assertFalse(cleaned[13, 22])

    def test_dominant_repeated_corridor_overrides_one_sided_support(self):
        valid = np.zeros((340, 60), dtype=bool)
        candidate = np.zeros_like(valid)
        for start in (5, 55, 105, 155, 205, 255):
            valid[start : start + 30] = True
            candidate[start + 3 : start + 25, 25:28] = True
            for row in range(start + 3, start + 25):
                candidate[row, 21 + (row - start) % 4] = True

        removed, record = repeated_segment_column_stripe_mask(
            candidate,
            valid,
            {},
            policy="balanced",
        )
        cleaned = candidate & ~removed

        self.assertEqual(record["dominant_corridor_count"], 1)
        self.assertGreaterEqual(record["candidate_removed_pixels"], 396)
        self.assertFalse(np.any(cleaned[:, 25:28]))

    def test_dominant_corridor_preserves_bilateral_geologic_crossing(self):
        valid = np.zeros((340, 60), dtype=bool)
        candidate = np.zeros_like(valid)
        patches: list[tuple[slice, slice]] = []
        for start in (5, 55, 105, 155, 205, 255):
            valid[start : start + 30] = True
            candidate[start + 3 : start + 25, 25:28] = True
            patch = (slice(start + 12, start + 15), slice(18, 35))
            candidate[patch] = True
            patches.append(patch)

        removed, record = repeated_segment_column_stripe_mask(
            candidate,
            valid,
            {},
            policy="balanced",
        )
        cleaned = candidate & ~removed

        self.assertEqual(record["dominant_corridor_count"], 1)
        self.assertTrue(all(np.all(cleaned[patch]) for patch in patches))
        self.assertFalse(cleaned[8, 26])


if __name__ == "__main__":
    unittest.main()
