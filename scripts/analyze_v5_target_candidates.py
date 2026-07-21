from __future__ import annotations

"""Summarise bundled reference coverage and absorption peaks for V5 targets."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corespec_mapper.algorithms import continuum_remove  # noqa: E402
from corespec_mapper.spectral_db import V5SpectralDatabase  # noqa: E402


DEFAULT_TARGETS = (
    "jarosite",
    "nontronite",
    "talc",
    "tremolite",
    "actinolite",
    "biotite",
    "phlogopite",
    "siderite",
    "sepiolite",
    "vermiculite",
    "buddingtonite",
)


def _local_peaks(wavelengths: np.ndarray, values: np.ndarray) -> list[tuple[float, float]]:
    selected = (
        np.isfinite(wavelengths)
        & np.isfinite(values)
        & (wavelengths >= 1000.0)
        & (wavelengths <= 2500.0)
    )
    wavelengths = wavelengths[selected]
    values = values[selected]
    if wavelengths.size < 12 or np.any(np.diff(wavelengths) <= 0):
        return []
    absorption = np.clip(1.0 - continuum_remove(values[None, :], wavelengths)[0], 0.0, None)
    candidates = np.flatnonzero(
        (absorption[1:-1] > absorption[:-2])
        & (absorption[1:-1] >= absorption[2:])
        & (absorption[1:-1] >= 0.004)
    ) + 1
    ordered = sorted(candidates, key=lambda index: float(absorption[index]), reverse=True)
    kept: list[int] = []
    for index in ordered:
        if all(abs(float(wavelengths[index] - wavelengths[other])) >= 18.0 for other in kept):
            kept.append(int(index))
        if len(kept) == 8:
            break
    return [(float(wavelengths[index]), float(absorption[index])) for index in kept]


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", default=DEFAULT_TARGETS)
    args = parser.parse_args()
    with V5SpectralDatabase() as database:
        for mineral_id in args.targets:
            spectra = [
                item
                for item in database.iter_spectra(mineral_id, eligible_only=False)
                if item.source_role == "mineral_reference" and item.purity_status == "declared_pure"
            ]
            samples = {item.sample_id for item in spectra}
            sources = {item.source_id for item in spectra}
            coverage = (
                min(float(item.wavelengths_nm[0]) for item in spectra),
                max(float(item.wavelengths_nm[-1]) for item in spectra),
            ) if spectra else (float("nan"), float("nan"))
            bins: dict[int, list[float]] = defaultdict(list)
            for spectrum in spectra:
                for center, depth in _local_peaks(spectrum.wavelengths_nm, spectrum.values):
                    bins[int(round(center / 5.0) * 5)].append(depth)
            peaks = sorted(
                (
                    (center, len(depths), float(np.median(depths)))
                    for center, depths in bins.items()
                    if len(depths) >= max(2, len(spectra) // 5)
                ),
                key=lambda item: (item[1], item[2]),
                reverse=True,
            )[:12]
            print(
                mineral_id,
                f"measurements={len(spectra)}",
                f"samples={len(samples)}",
                f"sources={len(sources)}",
                f"coverage_nm={coverage[0]:.1f}-{coverage[1]:.1f}",
            )
            print("  peaks(center_nm,count,median_depth)=", peaks)
            for spectrum in spectra[:3]:
                print("  sample=", spectrum.raw_name, "source=", spectrum.source_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
