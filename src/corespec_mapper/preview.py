from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json
import struct
import zlib

import numpy as np

from .envi import EnviDataset, classification_palette, derive_mask


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_png(path: str | Path, rgb: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG image must have shape (height, width, 3)")
    height, width, _ = image.shape
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)
    return path


def _stretched_gray(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sample = values[np.isfinite(values) & mask]
    if sample.size == 0:
        sample = values[np.isfinite(values)]
    lower, upper = np.percentile(sample, [1.0, 99.0])
    if upper <= lower:
        upper = lower + 1.0
    scaled = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    gray = np.nan_to_num(np.sqrt(scaled), nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(gray * 255.0).astype(np.uint8)


def _overlay(background: np.ndarray, classes: np.ndarray, class_names: Sequence[str], alpha: float = 0.78) -> np.ndarray:
    result = np.asarray(background, dtype=np.float64).copy()
    palette = np.asarray(classification_palette(class_names), dtype=np.float64).reshape(-1, 3)
    for class_id in range(1, len(class_names) - 1):
        selected = classes == class_id
        if np.any(selected):
            result[selected] = (1.0 - alpha) * result[selected] + alpha * palette[class_id]
    return np.clip(result, 0, 255).astype(np.uint8)


def make_pilot_previews(config: dict[str, Any], output_dir: str | Path, start: int, stop: int) -> dict[str, str]:
    output_dir = Path(output_dir)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    image = EnviDataset(config.get("baseline_image", config["image"]))
    mask_cube = EnviDataset(config.get("baseline_mask", config["mask"]))
    mask = derive_mask(mask_cube, chunk_rows=int(config.get("chunk_rows", 128)), start=start, stop=stop)
    wavelengths = image.info.wavelengths_nm
    if wavelengths is None:
        raise ValueError("Image wavelength vector is required for previews")
    band = int(np.argmin(np.abs(wavelengths - 1600.0)))
    reflectance = np.asarray(image.read_rows(start, stop, bands=[band])[..., 0], dtype=np.float64)
    gray = _stretched_gray(reflectance, mask)
    background = np.repeat(gray[..., None], 3, axis=2)
    write_png(preview_dir / "swir_background_1600nm.png", background)

    stages = {
        "sam_only": "sam_only_classes.dat",
        "destriped_sam_confirmed": "destriped_sam_confirmed_classes.dat",
        "joint_before_spatial": "candidate_classes_before_spatial.dat",
        "final": "preliminary_classes.dat",
    }
    combined: dict[str, list[np.ndarray]] = {stage: [background] for stage in stages}
    stripe_panels: list[np.ndarray] = [background]
    output_paths: dict[str, str] = {"background": str(preview_dir / "swir_background_1600nm.png")}
    legends: dict[str, Any] = {}
    for group in config["groups"]:
        group_name = group["name"]
        group_dir = output_dir / group_name
        legends[group_name] = {}
        for stage, filename in stages.items():
            dataset = EnviDataset(group_dir / filename)
            classes = np.array(dataset.read_rows(0, stop - start)[..., 0], copy=True)
            names = [str(name) for name in dataset.info.metadata.get("class names", [])]
            dataset.close()
            overlay = _overlay(background, classes, names)
            path = preview_dir / f"{group_name}_{stage}.png"
            write_png(path, overlay)
            output_paths[f"{group_name}_{stage}"] = str(path)
            combined[stage].append(overlay)
            if not legends[group_name]:
                colors = np.asarray(classification_palette(names), dtype=np.uint8).reshape(-1, 3)
                legends[group_name] = [
                    {"class_id": index, "name": name, "rgb": colors[index].tolist()}
                    for index, name in enumerate(names)
                ]
        stripe_dataset = EnviDataset(group_dir / "directional_stripe_noise_mask.dat")
        stripe_mask = np.array(stripe_dataset.read_rows(0, stop - start)[..., 0], dtype=bool, copy=True)
        stripe_dataset.close()
        stripe_overlay = background.astype(np.float64)
        stripe_overlay[stripe_mask] = 0.18 * stripe_overlay[stripe_mask] + 0.82 * np.array([255, 0, 255])
        stripe_overlay = np.clip(stripe_overlay, 0, 255).astype(np.uint8)
        stripe_path = preview_dir / f"{group_name}_detected_stripes.png"
        write_png(stripe_path, stripe_overlay)
        output_paths[f"{group_name}_detected_stripes"] = str(stripe_path)
        stripe_panels.append(stripe_overlay)

    for stage, panels in combined.items():
        path = preview_dir / f"comparison_{stage}.png"
        write_png(path, np.concatenate(panels, axis=1))
        output_paths[f"comparison_{stage}"] = str(path)
    stripe_comparison = preview_dir / "comparison_detected_stripes.png"
    write_png(stripe_comparison, np.concatenate(stripe_panels, axis=1))
    output_paths["comparison_detected_stripes"] = str(stripe_comparison)
    (preview_dir / "legend.json").write_text(json.dumps(legends, ensure_ascii=False, indent=2), encoding="utf-8")
    image.close()
    mask_cube.close()
    return output_paths


def make_v3_previews(config: dict[str, Any], output_dir: str | Path, start: int, stop: int) -> dict[str, str]:
    output_dir = Path(output_dir)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    image = EnviDataset(config["analysis_image"])
    mask_cube = EnviDataset(config["analysis_mask"])
    middle_band = min(mask_cube.info.bands // 2, mask_cube.info.bands - 1)
    mask = derive_mask(mask_cube, bands=[middle_band], chunk_rows=int(config.get("chunk_rows", 128)), start=start, stop=stop)
    wavelengths = image.info.wavelengths_nm
    if wavelengths is None:
        raise ValueError("Image wavelength vector is required for previews")
    band = int(np.argmin(np.abs(wavelengths - 1600.0)))
    reflectance = np.asarray(image.read_rows(start, stop, bands=[band])[..., 0], dtype=np.float64)
    gray = _stretched_gray(reflectance, mask)
    background = np.repeat(gray[..., None], 3, axis=2)
    background_path = preview_dir / "swir_background_1600nm.png"
    write_png(background_path, background)

    stages = {
        "before_spatial": "v3_classes_before_spatial.dat",
        "final": "v3_final_classes.dat",
    }
    combined: dict[str, list[np.ndarray]] = {stage: [background] for stage in stages}
    candidate_panels: list[np.ndarray] = [background]
    stripe_panels: list[np.ndarray] = [background]
    output_paths: dict[str, str] = {"background": str(background_path)}
    legends: dict[str, Any] = {}
    for group in config["v3"]["groups"]:
        group_name = group["name"]
        group_dir = output_dir / group_name
        legends[group_name] = []
        for stage, filename in stages.items():
            dataset = EnviDataset(group_dir / filename)
            classes = np.array(dataset.read_rows(0, stop - start)[..., 0], copy=True)
            names = [str(name) for name in dataset.info.metadata.get("class names", [])]
            dataset.close()
            overlay = _overlay(background, classes, names)
            path = preview_dir / f"{group_name}_{stage}.png"
            write_png(path, overlay)
            output_paths[f"{group_name}_{stage}"] = str(path)
            combined[stage].append(overlay)
            if not legends[group_name]:
                colors = np.asarray(classification_palette(names), dtype=np.uint8).reshape(-1, 3)
                legends[group_name] = [
                    {"class_id": index, "name": name, "rgb": colors[index].tolist()}
                    for index, name in enumerate(names)
                ]

        candidate_dataset = EnviDataset(group_dir / "column_calibrated_candidates.dat")
        candidates = np.array(candidate_dataset.read_rows(0, stop - start)[..., 0] == 1, copy=True)
        candidate_dataset.close()
        candidate_overlay = background.astype(np.float64)
        candidate_overlay[candidates] = 0.18 * candidate_overlay[candidates] + 0.82 * np.array([255, 0, 255])
        candidate_overlay = np.clip(candidate_overlay, 0, 255).astype(np.uint8)
        candidate_path = preview_dir / f"{group_name}_column_candidates.png"
        write_png(candidate_path, candidate_overlay)
        output_paths[f"{group_name}_column_candidates"] = str(candidate_path)
        candidate_panels.append(candidate_overlay)

        stripe_dataset = EnviDataset(group_dir / "stripe_noise_mask.dat")
        stripes = np.array(stripe_dataset.read_rows(0, stop - start)[..., 0], dtype=bool, copy=True)
        stripe_dataset.close()
        stripe_overlay = background.astype(np.float64)
        stripe_overlay[stripes] = 0.18 * stripe_overlay[stripes] + 0.82 * np.array([255, 0, 255])
        stripe_overlay = np.clip(stripe_overlay, 0, 255).astype(np.uint8)
        stripe_path = preview_dir / f"{group_name}_stripe_noise.png"
        write_png(stripe_path, stripe_overlay)
        output_paths[f"{group_name}_stripe_noise"] = str(stripe_path)
        stripe_panels.append(stripe_overlay)

    for stage, panels in combined.items():
        path = preview_dir / f"comparison_{stage}.png"
        write_png(path, np.concatenate(panels, axis=1))
        output_paths[f"comparison_{stage}"] = str(path)
    candidate_path = preview_dir / "comparison_column_candidates.png"
    write_png(candidate_path, np.concatenate(candidate_panels, axis=1))
    output_paths["comparison_column_candidates"] = str(candidate_path)
    stripe_path = preview_dir / "comparison_stripe_noise.png"
    write_png(stripe_path, np.concatenate(stripe_panels, axis=1))
    output_paths["comparison_stripe_noise"] = str(stripe_path)
    (preview_dir / "legend.json").write_text(json.dumps(legends, ensure_ascii=False, indent=2), encoding="utf-8")
    image.close()
    mask_cube.close()
    return output_paths
