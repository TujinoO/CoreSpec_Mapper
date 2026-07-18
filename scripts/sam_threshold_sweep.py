from pathlib import Path

import numpy as np

from corespec_mapper.algorithms import classify_sam
from corespec_mapper.envi import EnviDataset


ROOT = Path(r"E:\Code\CoreSpec_Mapper\output\pilot_stripe_balanced_0000_0800")
THRESHOLDS = {
    "carbonates": (0.0600, 0.0625, 0.0650, 0.0675, 0.0700),
    "sulfates": (0.0600, 0.0625, 0.0650, 0.0675, 0.0700),
    "clays": (0.0300, 0.0310, 0.0320, 0.0330, 0.0340, 0.0350),
}


for group, thresholds in THRESHOLDS.items():
    rules_dataset = EnviDataset(ROOT / group / "sam_rules.dat")
    rules = np.array(rules_dataset.read_rows(0, rules_dataset.info.lines), dtype=np.float64, copy=True)
    names = [str(name) for name in rules_dataset.info.metadata.get("band names", [])]
    rules_dataset.close()
    class_dataset = EnviDataset(ROOT / group / "sam_only_classes.dat")
    masked_id = int(class_dataset.info.metadata["classes"]) - 1
    existing = np.array(class_dataset.read_rows(0, class_dataset.info.lines)[..., 0], copy=True)
    class_dataset.close()
    valid = existing != masked_id
    values = rules[valid]

    print(f"\n{group}: valid pixels={int(valid.sum())}")
    for threshold in thresholds:
        labels = classify_sam(values, threshold)
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        details = ", ".join(
            f"{names[int(class_id) - 1] if int(class_id) - 1 < len(names) else class_id}={int(count)}"
            for class_id, count in zip(ids, counts)
        )
        print(f"  {threshold:.4f}: total={int(np.count_nonzero(labels))}; {details}")
