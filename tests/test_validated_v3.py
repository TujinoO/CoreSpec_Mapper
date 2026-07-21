from __future__ import annotations

import unittest

from corespec_mapper.validated_v3 import _rejection_waterfalls, _validation_settings


class ValidatedV3Tests(unittest.TestCase):
    def test_profile_contract_is_explicit(self):
        value = _validation_settings(
            {"advanced": {"validation": {"v3_profiles": {"balanced": "balanced.json"}}}}
        )
        self.assertEqual(value["v3_profiles"]["balanced"], "balanced.json")
        with self.assertRaisesRegex(ValueError, "v3_profiles"):
            _validation_settings({"advanced": {"validation": {}}})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _validation_settings(
                {"advanced": {"validation": {"v3_profiles": {"unknown": "x.json"}}}}
            )

    def test_rejection_waterfall_balances_stage_counts(self):
        result = _rejection_waterfalls(
            {
                "balanced": {
                    "mask_pixels": 100,
                    "groups": {
                        "carbonates": {
                            "counts": {
                                "column_calibrated_candidates": 40,
                                "classified_before_spatial": 20,
                                "elongated_component_removed": 3,
                                "small_component_removed": 2,
                                "classified_final": 15,
                            },
                            "mineral_counts_final": {"Calcite": 10, "Dolomite": 5},
                        }
                    },
                }
            }
        )["balanced"]["carbonates"]
        self.assertEqual(result["rejected_by_spectral_depth_or_class_evidence"], 20)
        self.assertEqual(result["classified_final"], 15)
        self.assertEqual(result["other_spatial_or_overlap_removed"], 0)


if __name__ == "__main__":
    unittest.main()
