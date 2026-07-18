from pathlib import Path

import numpy as np

from corespec_mapper.envi import EnviDataset


ROOT = Path(r"E:\Code\CoreSpec_Mapper\output\pilot_corrected_0000_0800\sulfates")


def read(name: str) -> np.ndarray:
    dataset = EnviDataset(ROOT / name)
    values = np.asarray(dataset.read_rows(0, dataset.info.lines), dtype=np.float64)
    dataset.close()
    return values


sam = read("sam_rules.dat")
quality = read("sff_quality.dat")
scale = read("sff_scale.dat")
rms = read("sff_rms.dat")
classes = read("sam_only_classes.dat")[..., 0].astype(np.uint8)

valid = (classes > 0) & (classes < 4)
best_sam = np.argmin(np.where(np.isfinite(sam), sam, np.inf), axis=-1)
best_quality = np.argmax(np.where(np.isfinite(quality), quality, -np.inf), axis=-1)
rows, columns = np.where(valid)
sam_endpoint = best_sam[valid]
quality_endpoint = best_quality[valid]
same_endpoint = sam_endpoint == quality_endpoint
matched_quality = quality[rows, columns, sam_endpoint]
matched_scale = scale[rows, columns, sam_endpoint]
matched_rms = rms[rows, columns, sam_endpoint]

print(f"SAM candidates: {int(valid.sum())}")
print(f"SAM/SFF best-endpoint agreement: {int(same_endpoint.sum())} ({same_endpoint.mean():.6f})")
for endpoint, name in enumerate(("Anhydrite", "Gypsum HS333", "Gypsum SU2202")):
    selected = sam_endpoint == endpoint
    print(f"\n{name}: SAM-selected pixels = {int(selected.sum())}")
    if not np.any(selected):
        continue
    for label, values in (
        ("same-endpoint quality", matched_quality[selected]),
        ("same-endpoint scale", matched_scale[selected]),
        ("same-endpoint RMS", matched_rms[selected]),
    ):
        percentiles = np.percentile(values[np.isfinite(values)], [1, 5, 25, 50, 75, 95, 99])
        print(f"  {label}: " + ", ".join(f"p{p}={v:.6g}" for p, v in zip((1, 5, 25, 50, 75, 95, 99), percentiles)))
    passed = (matched_quality[selected] >= 0.55) & (matched_scale[selected] >= 0.1)
    print(f"  pass current SFF gates: {int(passed.sum())}")

ids, counts = np.unique(quality_endpoint, return_counts=True)
print("\nBest SFF endpoint among SAM candidates:")
for endpoint, count in zip(ids, counts):
    print(f"  {int(endpoint)} = {int(count)}")

for label, score_values, maximize in (
    ("minimum RMS", rms, False),
    ("maximum scale/RMS", scale / np.maximum(rms, 1e-12), True),
):
    safe = np.where(np.isfinite(score_values), score_values, -np.inf if maximize else np.inf)
    best = np.argmax(safe, axis=-1) if maximize else np.argmin(safe, axis=-1)
    best = best[valid]
    ids, counts = np.unique(best, return_counts=True)
    print(f"\n{label} endpoint among SAM candidates:")
    for endpoint, count in zip(ids, counts):
        print(f"  {int(endpoint)} = {int(count)}")
    print(f"  agreement with SAM endpoint: {int(np.count_nonzero(best == sam_endpoint))}")
