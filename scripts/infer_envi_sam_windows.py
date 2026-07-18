from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from corespec_mapper.envi import EnviDataset, SpectralLibrary, derive_mask


ROOT = Path(r"F:\NC-1-31_40\FILL")
START = 0
STOP = 800
SAMPLE_SIZE = 400


@dataclass(frozen=True)
class SearchCase:
    name: str
    library: str
    rule: str
    ranges: tuple[range, ...]


CASES = (
    SearchCase(
        "sulfates",
        "Anhydrite_Gypsum.sli",
        "SAM_Mask_Anhydrite_Gypsum_005_rule.dat",
        (range(45, 57), range(105, 120), range(112, 127), range(176, 198)),
    ),
    SearchCase(
        "clays",
        "Illite_Montmorillonite_Kaolinite.sli",
        "SAM_Mask_Illite_Montmorillonite_Kaolinite_0025_rule.dat",
        (range(140, 169), range(166, 199)),
    ),
)


def prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.zeros((*values.shape[:-1], 1), dtype=np.float64), np.cumsum(values, axis=-1)],
        axis=-1,
    )


def interval(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    return values[..., stop] - values[..., start]


def search(case: SearchCase) -> None:
    image = EnviDataset(ROOT / "SWIR_SG")
    mask_cube = EnviDataset(ROOT / "mask_SG")
    mask = derive_mask(mask_cube, start=START, stop=STOP)
    pixels = np.asarray(image.read_rows(START, STOP)[mask], dtype=np.float64)
    library = SpectralLibrary.open(ROOT / case.library)
    reference_dataset = EnviDataset(ROOT / case.rule)
    reference = np.asarray(reference_dataset.read_rows(START, STOP)[mask], dtype=np.float64)

    sample_indices = np.linspace(0, pixels.shape[0] - 1, SAMPLE_SIZE, dtype=int)
    pixels = pixels[sample_indices]
    reference = reference[sample_indices]
    endpoints = library.spectra
    wavelengths = image.info.wavelengths_nm
    assert wavelengths is not None

    pixel_sq = prefix(pixels * pixels)
    endpoint_sq = prefix(endpoints * endpoints)
    products = prefix(pixels[:, None, :] * endpoints[None, :, :])
    finite_reference = np.isfinite(reference)

    def score(intervals: tuple[tuple[int, int], ...]) -> float:
        numerator = sum(interval(products, lo, hi) for lo, hi in intervals)
        pixel_energy = sum(interval(pixel_sq, lo, hi) for lo, hi in intervals)
        endpoint_energy = sum(interval(endpoint_sq, lo, hi) for lo, hi in intervals)
        denominator = np.sqrt(pixel_energy[:, None] * endpoint_energy[None, :])
        cosine = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))
        return float(np.mean(np.abs(angles[finite_reference] - reference[finite_reference])))

    candidates: list[tuple[float, tuple[tuple[int, int], ...]]] = []
    if len(case.ranges) == 2:
        for start in case.ranges[0]:
            for stop_inclusive in case.ranges[1]:
                stop = stop_inclusive + 1
                if stop - start < 3:
                    continue
                item = ((start, stop),)
                candidates.append((score(item), item))
    else:
        for first_start in case.ranges[0]:
            for first_stop_inclusive in case.ranges[1]:
                first_stop = first_stop_inclusive + 1
                if first_stop - first_start < 3:
                    continue
                for second_start in case.ranges[2]:
                    if second_start < first_stop:
                        continue
                    for second_stop_inclusive in case.ranges[3]:
                        second_stop = second_stop_inclusive + 1
                        if second_stop - second_start < 3:
                            continue
                        item = ((first_start, first_stop), (second_start, second_stop))
                        candidates.append((score(item), item))

    candidates.sort(key=lambda item: item[0])
    print(f"\n{case.name}: {len(candidates)} candidates")
    for error, intervals in candidates[:12]:
        labels = [
            f"bands {lo + 1}:{hi} ({wavelengths[lo]:.2f}-{wavelengths[hi - 1]:.2f} nm)"
            for lo, hi in intervals
        ]
        print(f"  MAE={error:.12g}  " + " + ".join(labels))

    reference_dataset.close()
    mask_cube.close()
    image.close()


for current_case in CASES:
    search(current_case)
