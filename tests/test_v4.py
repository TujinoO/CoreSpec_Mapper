from pathlib import Path
import unittest

import numpy as np

from corespec_mapper.artifacts import edge_risk_score, filter_artifacts
from corespec_mapper.catalog import MineralCatalog
from corespec_mapper.sampling import build_stratified_sample_plan, sample_plan_mask
from corespec_mapper.sensor import infer_spectral_domain, resolve_mineral_support
from corespec_mapper.v4_calibration import policy_candidate, resolve_group_policies
from corespec_mapper.v4_library import resample_with_fwhm
from corespec_mapper.v4_models import CancellationToken, RunCancelled, SampleBlock, SamplePlan, SensorCapabilityCard, SupportLevel
from corespec_mapper.v4_service import audit_v4_project


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


if __name__ == "__main__":
    unittest.main()
