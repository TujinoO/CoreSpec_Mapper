from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corespec_mapper.envi import EnviDataset, SpectralLibrary, derive_mask, wavelength_indices, write_envi
from corespec_mapper.preview import _overlay, _stretched_gray, write_png


CONFIG_PATH = Path(r"E:\Code\CoreSpec_Mapper\configs\nc1.json")
PILOT = Path(r"E:\Code\CoreSpec_Mapper\output\pilot_corrected_v2_0000_0800")
OUTPUT = Path(r"E:\GRP Docs\2026.07 岩心影像矿物智能识别\destripe_experiment")
START, STOP = 0, 800
RADIUS = 2
STRENGTHS = (0.5, 1.0)
BAND_CHUNK = 16


def mineral_label(name: str) -> str:
    lowered = name.casefold()
    for mineral in ("calcite", "dolomite", "anhydrite", "gypsum", "illite", "montmorillonite", "kaolinite"):
        if mineral in lowered:
            return mineral
    return lowered


def robust_column_bias(cube: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    masked = np.where(mask[..., None], cube, np.nan)
    padded = np.pad(masked, ((0, 0), (radius, radius), (0, 0)), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1, axis=1)
    with np.errstate(all="ignore"):
        local = np.nanmedian(windows, axis=-1)
        residual = masked - local
        bias = np.nanmedian(residual, axis=0)
        center = np.nanmedian(bias, axis=0)
    bias = bias - center[None, :]
    return np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)


config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
image = EnviDataset(config["analysis_image"])
mask_dataset = EnviDataset(config["analysis_mask"])
mask = derive_mask(mask_dataset, start=START, stop=STOP)
wavelengths = image.info.wavelengths_nm
assert wavelengths is not None

background_band = int(np.argmin(np.abs(wavelengths - 1600.0)))
background_values = np.array(image.read_rows(START, STOP, bands=[background_band])[..., 0], copy=True)
gray = _stretched_gray(background_values, mask)
background = np.repeat(gray[..., None], 3, axis=2)
write_png(OUTPUT / "background.png", background)

panels = {strength: [background] for strength in STRENGTHS}
for group in config["groups"]:
    group_name = group["name"]
    library = SpectralLibrary.open(group["library"])
    indices = wavelength_indices(wavelengths, group["sam_windows"])
    endpoint_count = library.spectra.shape[0]
    pixel_count = int(mask.sum())
    accumulators = {
        strength: {
            "dot": np.zeros((pixel_count, endpoint_count), dtype=np.float64),
            "pixel_energy": np.zeros(pixel_count, dtype=np.float64),
            "endpoint_energy": np.zeros(endpoint_count, dtype=np.float64),
        }
        for strength in STRENGTHS
    }

    for offset in range(0, indices.size, BAND_CHUNK):
        chunk_indices = indices[offset : offset + BAND_CHUNK]
        cube = np.array(image.read_rows(START, STOP, bands=chunk_indices), dtype=np.float64, copy=True)
        bias = robust_column_bias(cube, mask, RADIUS)
        references = library.spectra[:, chunk_indices]
        for strength, accumulator in accumulators.items():
            corrected = cube - strength * bias[None, :, :]
            pixels = corrected[mask]
            accumulator["dot"] += pixels @ references.T
            accumulator["pixel_energy"] += np.sum(pixels * pixels, axis=1)
            accumulator["endpoint_energy"] += np.sum(references * references, axis=1)

    original_dataset = EnviDataset(PILOT / group_name / "sam_only_classes.dat")
    original = np.array(original_dataset.read_rows(0, STOP - START)[..., 0], dtype=np.uint8, copy=True)
    class_names = [str(name) for name in original_dataset.info.metadata["class names"]]
    original_dataset.close()
    original_valid = mask & (original > 0) & (original <= endpoint_count)
    endpoint_minerals = np.array([mineral_label(name) for name in library.names], dtype=object)

    print(f"\n{group_name}: original candidates={int(original_valid.sum())}")
    for strength, accumulator in accumulators.items():
        denominator = np.sqrt(accumulator["pixel_energy"][:, None] * accumulator["endpoint_energy"][None, :])
        cosine = np.divide(accumulator["dot"], denominator, out=np.full_like(accumulator["dot"], np.nan), where=denominator > 0)
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))
        best_endpoint = np.argmin(np.where(np.isfinite(angles), angles, np.inf), axis=1)
        best_angle = angles[np.arange(pixel_count), best_endpoint]
        corrected_labels = np.where(best_angle <= float(group["sam_threshold"]), best_endpoint + 1, 0).astype(np.uint8)

        original_labels = original[mask]
        confirmed = np.zeros(pixel_count, dtype=np.uint8)
        for endpoint, mineral in enumerate(endpoint_minerals):
            selected = original_labels == endpoint + 1
            if not np.any(selected):
                continue
            same_mineral = np.flatnonzero(endpoint_minerals == mineral)
            supported = np.min(angles[selected][:, same_mineral], axis=1) <= float(group["sam_threshold"])
            confirmed_indices = np.flatnonzero(selected)[supported]
            confirmed[confirmed_indices] = original_labels[confirmed_indices]

        output_classes = np.zeros_like(original)
        output_classes[mask] = confirmed
        output_classes[~mask] = endpoint_count + 1
        variant_dir = OUTPUT / f"strength_{strength:.1f}" / group_name
        write_envi(
            output_classes,
            variant_dir / "destripe_confirmed_classes.dat",
            class_names=class_names,
            description=f"{group_name} robust column-destriped SAM confirmation",
        )
        overlay = _overlay(background, output_classes, class_names)
        write_png(OUTPUT / f"strength_{strength:.1f}" / f"{group_name}.png", overlay)
        panels[strength].append(overlay)
        print(
            f"  strength={strength:.1f}: corrected candidates={int(np.count_nonzero(corrected_labels))}, "
            f"confirmed original={int(np.count_nonzero(confirmed))}"
        )

for strength, images in panels.items():
    write_png(OUTPUT / f"comparison_strength_{strength:.1f}.png", np.concatenate(images, axis=1))

mask_dataset.close()
image.close()
