from pathlib import Path
import unittest

import numpy as np

from corespec_mapper.envi import EnviDataset
from corespec_mapper.library_ensemble import build_library_ensemble, match_pure_target, resample_spectrum
from corespec_mapper.v3_pipeline import aggregate_reference_scores, balanced_tie_labels, column_percentile_thresholds


ROOT = Path(r"F:\NC-1-31_40")
LIBRARY_ROOT = Path(__file__).parents[1] / "spec_lib"


class V3AlgorithmTests(unittest.TestCase):
    def test_pure_target_matching_rejects_mixtures_and_rocks(self):
        self.assertEqual(match_pure_target("Calcite C-3A"), ("calcite", None))
        self.assertEqual(match_pure_target("Illite-bearing Shale")[1], "rock_or_mixture_name")
        self.assertEqual(match_pure_target("Montmorillonite+Illite CM37")[1], "mixed_target_minerals")
        self.assertEqual(match_pure_target("Halloysite+Kaolinite CM29")[1], "mixed_clay_name")

    def test_resampling_does_not_bridge_large_source_gap(self):
        result = resample_spectrum(
            [1000.0, 1010.0, 1300.0, 1310.0],
            [0.5, 0.6, 0.7, 0.8],
            [1005.0, 1150.0, 1305.0],
            max_gap_nm=50.0,
        )
        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isfinite(result[2]))

    def test_column_percentile_is_calibrated_per_detector_column(self):
        scores = np.column_stack([np.arange(100), np.arange(100) + 1000.0]).astype(np.float64)
        thresholds, counts = column_percentile_thresholds(scores, np.ones_like(scores, dtype=bool), 0.05)
        np.testing.assert_allclose(thresholds, [4.95, 1004.95])
        np.testing.assert_array_equal(counts, [100, 100])

    def test_reference_aggregation_uses_equal_best_k_per_mineral(self):
        scores = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        aggregated = aggregate_reference_scores(
            scores,
            ["a", "a", "a", "b", "b"],
            ["a", "b"],
            best_k=2,
        )
        np.testing.assert_allclose(aggregated, [[0.15, 0.45]])

    def test_bias_only_changes_a_near_tie(self):
        scores = np.array([[0.20, 0.21], [0.20, 0.40]])
        labels, ties, changed = balanced_tie_labels(
            scores,
            ["present", "missing"],
            tie_tolerance=0.03,
            tie_bias={"missing": 0.02},
        )
        np.testing.assert_array_equal(labels, [1, 0])
        np.testing.assert_array_equal(ties, [2, 1])
        np.testing.assert_array_equal(changed, [True, False])


@unittest.skipUnless(ROOT.exists() and LIBRARY_ROOT.exists(), "NC-1 data or ENVI libraries are unavailable")
class V3RealLibraryTests(unittest.TestCase):
    def test_automatic_ensemble_is_balanced_and_deterministic(self):
        image = EnviDataset(ROOT / r"FILL\SWIR_SG")
        first = build_library_ensemble(LIBRARY_ROOT, image.info.wavelengths_nm)
        second = build_library_ensemble(LIBRARY_ROOT, image.info.wavelengths_nm)
        self.assertEqual(first.group_reference_counts, {"carbonates": 4, "sulfates": 4, "clays": 4})
        self.assertEqual(first.spectrum_names, second.spectrum_names)
        self.assertEqual(len(first.spectrum_names), 28)
        for mineral in ("calcite", "dolomite"):
            selected = [candidate for candidate in first.selected_candidates if candidate.mineral == mineral]
            families = {candidate.source_library.split("\\", 1)[0] for candidate in selected}
            self.assertGreaterEqual(len(families), 2)
            self.assertTrue(all(candidate.absorption_depth >= 0.05 for candidate in selected))
            self.assertTrue(all(candidate.roughness <= 0.08 for candidate in selected))
        image.close()


if __name__ == "__main__":
    unittest.main()
