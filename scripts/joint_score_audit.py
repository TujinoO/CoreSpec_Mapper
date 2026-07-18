from pathlib import Path

import numpy as np

from corespec_mapper.envi import EnviDataset


ROOT = Path(r"E:\Code\CoreSpec_Mapper\output\pilot_corrected_0000_0800")


def read(path: Path) -> np.ndarray:
    dataset = EnviDataset(path)
    values = np.asarray(dataset.read_rows(0, dataset.info.lines), dtype=np.float64)
    dataset.close()
    return values


for group in ("carbonates", "sulfates", "clays"):
    group_dir = ROOT / group
    sam = read(group_dir / "sam_rules.dat")
    scale = read(group_dir / "sff_scale.dat")
    rms = read(group_dir / "sff_rms.dat")
    classes = read(group_dir / "sam_only_classes.dat")[..., 0].astype(np.uint8)
    endpoint_count = sam.shape[-1]
    valid = (classes > 0) & (classes <= endpoint_count)
    rows, columns = np.where(valid)
    sam_endpoint = np.argmin(np.where(np.isfinite(sam), sam, np.inf), axis=-1)[valid]
    classic = np.divide(scale, np.maximum(rms, 1e-12), out=np.zeros_like(scale), where=np.isfinite(scale) & np.isfinite(rms))
    fit_endpoint = np.argmax(classic, axis=-1)[valid]
    matched_fit = classic[rows, columns, sam_endpoint]
    best_fit = np.max(classic, axis=-1)[valid]
    agreement = sam_endpoint == fit_endpoint

    print(f"\n{group}")
    print(f"  SAM candidates: {int(valid.sum())}")
    print(f"  best-endpoint agreement: {int(agreement.sum())} ({agreement.mean():.6f})")
    for label, values in (("SAM-endpoint fit", matched_fit), ("best fit", best_fit)):
        finite = values[np.isfinite(values)]
        percentiles = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
        print(f"  {label}: " + ", ".join(f"p{p}={v:.6g}" for p, v in zip((1, 5, 25, 50, 75, 95, 99), percentiles)))
    for endpoint in range(endpoint_count):
        count = int(np.count_nonzero(sam_endpoint == endpoint))
        agreed = int(np.count_nonzero((sam_endpoint == endpoint) & agreement))
        if count:
            print(f"  endpoint {endpoint + 1}: candidates={count}, fit-agreed={agreed}")
    matrix = np.zeros((endpoint_count, endpoint_count), dtype=int)
    for sam_id, fit_id in zip(sam_endpoint, fit_endpoint):
        matrix[int(sam_id), int(fit_id)] += 1
    print("  SAM rows x SFF columns:")
    for endpoint, row in enumerate(matrix):
        if np.any(row):
            print(f"    {endpoint + 1}: " + ", ".join(str(int(value)) for value in row))
