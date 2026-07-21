from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np

from .envi import EnviDataset, classification_palette, derive_mask
from .preview import _overlay, _stretched_gray, write_png
from .v4_calibration import POLICY_ORDER


def make_v4_previews(
    config: dict[str, Any],
    run_dir: str | Path,
    group_ids: Sequence[str],
    start: int,
    stop: int,
    *,
    material_mask: np.ndarray | None = None,
) -> dict[str, str]:
    run = Path(run_dir)
    output = run / "previews"
    output.mkdir(parents=True, exist_ok=True)
    image = EnviDataset(config["analysis_image"])
    mask_dataset = EnviDataset(config["analysis_mask"]) if material_mask is None else None
    try:
        wavelengths = image.info.wavelengths_nm
        if wavelengths is None:
            raise ValueError("Image wavelength vector is required for previews")
        band = int(np.argmin(np.abs(wavelengths - 1600.0)))
        if material_mask is None:
            assert mask_dataset is not None
            default_bands = [
                min(20, mask_dataset.info.bands - 1),
                min(mask_dataset.info.bands // 2, mask_dataset.info.bands - 1),
                min(190, mask_dataset.info.bands - 1),
            ]
            mask_bands = sorted(set(int(item) for item in config.get("v4", {}).get("mask_bands", default_bands)))
            mask = derive_mask(mask_dataset, bands=mask_bands, start=start, stop=stop)
        else:
            full_mask = np.asarray(material_mask, dtype=bool)
            if full_mask.shape != (image.info.lines, image.info.samples):
                raise ValueError("Material mask and preview image dimensions do not match")
            mask = full_mask[start:stop]
        reflectance = image.read_rows(start, stop, bands=[band])[..., 0]
        gray = _stretched_gray(reflectance, mask)
        background = np.repeat(gray[..., None], 3, axis=2)
        paths: dict[str, str] = {}
        background_path = output / "swir_background_1600nm.png"
        write_png(background_path, background)
        paths["background"] = str(background_path)
        balanced_panels = [background]
        profile_panels: dict[str, list[np.ndarray]] = {policy: [background] for policy in POLICY_ORDER}
        stripe_panels = [background]
        legends: dict[str, Any] = {}
        for group_id in group_ids:
            group_dir = run / "groups" / group_id
            legends[group_id] = []
            for policy in POLICY_ORDER:
                dataset = EnviDataset(group_dir / f"final_{policy}.dat")
                classes = np.array(dataset.read_rows(0, stop - start)[..., 0], copy=True)
                names = [str(item) for item in dataset.info.metadata.get("class names", [])]
                dataset.close()
                overlay = _overlay(background, classes, names)
                path = output / f"group_final_{group_id}_{policy}.png"
                write_png(path, overlay)
                paths[f"{group_id}_{policy}"] = str(path)
                profile_panels[policy].append(overlay)
                if policy == "balanced":
                    balanced_panels.append(overlay)
                    colors = np.asarray(classification_palette(names), dtype=np.uint8).reshape(-1, 3)
                    legends[group_id] = [
                        {"class_id": index, "name": name, "rgb": colors[index].tolist()}
                        for index, name in enumerate(names)
                    ]
            stripe_dataset = EnviDataset(group_dir / "stripe_noise_mask.dat")
            stripes = np.array(stripe_dataset.read_rows(0, stop - start)[..., 0], dtype=bool, copy=True)
            stripe_dataset.close()
            stripe_overlay = background.astype(np.float64)
            stripe_overlay[stripes] = 0.15 * stripe_overlay[stripes] + 0.85 * np.array([255, 0, 255])
            stripe_overlay = np.clip(stripe_overlay, 0, 255).astype(np.uint8)
            stripe_panels.append(stripe_overlay)

        balanced_path = output / "comparison_balanced.png"
        write_png(balanced_path, np.concatenate(balanced_panels, axis=1))
        paths["comparison_balanced"] = str(balanced_path)
        three_profile_rows: list[np.ndarray] = []
        for policy in POLICY_ORDER:
            row = np.concatenate(profile_panels[policy], axis=1)
            three_profile_rows.append(row)
            path = output / f"comparison_{policy}.png"
            write_png(path, row)
            paths[f"comparison_{policy}"] = str(path)
        three_path = output / "comparison_three_profiles.png"
        write_png(three_path, np.concatenate(three_profile_rows, axis=0))
        paths["comparison_three_profiles"] = str(three_path)
        stripe_path = output / "stripe_diagnosis.png"
        write_png(stripe_path, np.concatenate(stripe_panels, axis=1))
        paths["stripe_diagnosis"] = str(stripe_path)
        (output / "legend.json").write_text(json.dumps(legends, ensure_ascii=False, indent=2), encoding="utf-8")
        return paths
    finally:
        image.close()
        if mask_dataset is not None:
            mask_dataset.close()
