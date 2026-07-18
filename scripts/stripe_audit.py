from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from corespec_mapper.envi import EnviDataset, derive_mask


PILOT = Path(r"E:\Code\CoreSpec_Mapper\output\pilot_corrected_v2_0000_0800")
MASK_PATH = Path(r"F:\NC-1-31_40\FILL\mask_SG")
IMAGE_PATH = Path(r"F:\NC-1-31_40\FILL\SWIR_SG")
GROUPS = ("carbonates", "sulfates", "clays")
START, STOP = 0, 800


def read_classes(group: str, name: str) -> np.ndarray:
    dataset = EnviDataset(PILOT / group / name)
    values = np.array(dataset.read_rows(0, dataset.info.lines)[..., 0], dtype=np.uint8, copy=True)
    dataset.close()
    return values


def components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    visited = np.zeros_like(mask, dtype=bool)
    result: list[tuple[int, int, int, int, int]] = []
    height, width = mask.shape
    for row, column in np.argwhere(mask):
        row, column = int(row), int(column)
        if visited[row, column]:
            continue
        queue = deque([(row, column)])
        visited[row, column] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            current_row, current_column = queue.popleft()
            pixels.append((current_row, current_column))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    nr, nc = current_row + dr, current_column + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        rows, columns = zip(*pixels)
        result.append((len(pixels), min(rows), max(rows), min(columns), max(columns)))
    return result


mask_dataset = EnviDataset(MASK_PATH)
core_mask = derive_mask(mask_dataset, start=START, stop=STOP)
mask_dataset.close()
core_column_counts = np.maximum(core_mask.sum(axis=0), 1)

profiles: dict[str, np.ndarray] = {}
for group in GROUPS:
    classes = read_classes(group, "sam_only_classes.dat")
    endpoint_count = int(classes[core_mask].max(initial=0))
    candidate = core_mask & (classes > 0) & (classes <= endpoint_count)
    profile = candidate.sum(axis=0) / core_column_counts
    profiles[group] = profile
    top = np.argsort(profile)[-15:][::-1]
    print(f"\n{group} SAM column density: total={int(candidate.sum())}")
    print("  " + ", ".join(f"x={int(index)}:{profile[index]:.3f}" for index in top))

    final_classes = read_classes(group, "preliminary_classes.dat")
    final = core_mask & (final_classes > 0) & (final_classes <= endpoint_count)
    items = components(final)
    elongated = [item for item in items if (item[2] - item[1] + 1) >= 20 and (item[4] - item[3] + 1) <= 4]
    print(f"  final components={len(items)}, elongated thin components={len(elongated)}")
    areas = np.asarray([item[0] for item in items], dtype=int)
    if areas.size:
        print("  component area percentiles: " + ", ".join(
            f"p{point}={value:.1f}"
            for point, value in zip((25, 50, 75, 90, 95, 99), np.percentile(areas, (25, 50, 75, 90, 95, 99)))
        ))
        print("  retained pixels by minimum area: " + ", ".join(
            f">={minimum}:{int(areas[areas >= minimum].sum())}"
            for minimum in (4, 8, 12, 20, 30)
        ))
    for area, min_row, max_row, min_col, max_col in sorted(elongated, reverse=True)[:12]:
        print(f"    area={area}, rows={min_row}:{max_row}, cols={min_col}:{max_col}")

print("\nPairwise column-profile correlations")
for first_index, first in enumerate(GROUPS):
    for second in GROUPS[first_index + 1 :]:
        correlation = np.corrcoef(profiles[first], profiles[second])[0, 1]
        print(f"  {first}/{second}: {correlation:.6f}")

image = EnviDataset(IMAGE_PATH)
wavelengths = image.info.wavelengths_nm
assert wavelengths is not None
target_wavelengths = (1600.0, 2200.0, 2350.0)
band_indices = [int(np.argmin(np.abs(wavelengths - target))) for target in target_wavelengths]
cube = np.array(image.read_rows(START, STOP, bands=band_indices), dtype=np.float64, copy=True)
image.close()

print("\nReflectance fixed-column residuals")
for band_position, band_index in enumerate(band_indices):
    values = cube[..., band_position]
    medians = np.array([
        np.median(values[:, column][core_mask[:, column]]) if np.any(core_mask[:, column]) else np.nan
        for column in range(values.shape[1])
    ])
    local = np.array([
        np.nanmedian(medians[max(0, column - 5) : min(medians.size, column + 6)])
        for column in range(medians.size)
    ])
    residual = medians - local
    finite = residual[np.isfinite(residual)]
    center = np.median(finite)
    mad = np.median(np.abs(finite - center))
    zscore = np.abs(residual - center) / max(1.4826 * mad, 1e-12)
    top = np.argsort(np.nan_to_num(zscore, nan=-1.0))[-15:][::-1]
    print(f"  {wavelengths[band_index]:.2f} nm: " + ", ".join(f"x={int(index)}:z={zscore[index]:.2f}" for index in top))
