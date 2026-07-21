from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from corespec_mapper.envi import write_envi
from corespec_mapper.v5_validation import (
    RegressionGate,
    binary_agreement,
    validate_v5_regression,
)


class V5ValidationTests(unittest.TestCase):
    def test_binary_agreement_and_release_gate(self):
        reference = np.zeros((4, 5), dtype=bool)
        reference[1:3, 1:4] = True
        candidate = reference.copy()
        candidate[1, 1] = False
        candidate[0, 0] = True
        score = binary_agreement(reference, candidate)
        self.assertEqual(score.reference_pixels, 6)
        self.assertEqual(score.candidate_pixels, 6)
        self.assertEqual(score.intersection_pixels, 5)
        self.assertAlmostEqual(score.iou, 5 / 7)
        self.assertTrue(RegressionGate().failures(score))

    def test_full_regression_report_is_strict_and_case_insensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((6, 5), dtype=np.uint8)
            mask[1:5, 1:4] = 1
            reference_classes = np.zeros(mask.shape, dtype=np.uint8)
            reference_classes[1:3, 1:4] = 1
            reference_classes[3:5, 1:4] = 2
            candidate_classes = reference_classes.copy()
            write_envi(mask, root / "reference_mask.dat")
            write_envi(mask, root / "candidate_mask.dat")
            write_envi(
                reference_classes,
                root / "reference_classes.dat",
                class_names=["Unclassified", "Calcite", "Dolomite"],
            )
            write_envi(
                candidate_classes,
                root / "candidate_classes.dat",
                class_names=["Unclassified", "calcite", "dolomite"],
            )
            report = validate_v5_regression(
                {
                    "mask": {"reference": "reference_mask.dat", "candidate": "candidate_mask.dat"},
                    "classifications": [
                        {
                            "name": "carbonates_balanced",
                            "reference": "reference_classes.dat",
                            "candidate": "candidate_classes.dat",
                            "minerals": ["Calcite", "Dolomite"],
                            "domain": "reference_mask",
                        }
                    ],
                },
                config_dir=root,
            )
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["mask"]["passed"])
        self.assertEqual(report["classifications"]["carbonates_balanced"]["evidence_level"], "historical_workflow_weak_label")


if __name__ == "__main__":
    unittest.main()
