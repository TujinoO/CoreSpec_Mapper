from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image

from corespec_mapper.envi import EnviDataset, write_envi
from corespec_mapper.masking import build_material_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and preview the existing CoreSpec SWIR reflectance foreground baseline."
    )
    parser.add_argument("--swir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="swir_foreground")
    parser.add_argument("--registered-rgb", type=Path)
    parser.add_argument("--rgb-scale", type=int, default=2)
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--minimum-component-pixels", type=int, default=64)
    parser.add_argument("--preview-height", type=int, default=4096)
    return parser.parse_args()


def _stretch(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values) & valid
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    lower, upper = np.percentile(values[finite], [2.0, 98.0])
    if upper <= lower:
        upper = lower + 1e-6
    scaled = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return np.nan_to_num(scaled * 255.0, nan=0.0, posinf=255.0, neginf=0.0).round().astype(np.uint8)


def _registered_rgb_preview(dataset: EnviDataset, rows: np.ndarray) -> np.ndarray:
    if dataset.info.interleave == "bil":
        return np.moveaxis(np.asarray(dataset._array[rows, :3, :]), 1, 2).astype(np.uint8, copy=False)
    if dataset.info.interleave == "bip":
        return np.asarray(dataset._array[rows, :, :3], dtype=np.uint8)
    return np.moveaxis(np.asarray(dataset._array[:3, rows, :]), 0, 2).astype(np.uint8, copy=False)


def main() -> int:
    args = _parse_args()
    if args.chunk_rows < 1 or args.minimum_component_pixels < 1 or args.rgb_scale < 1:
        raise ValueError("chunk-rows, minimum-component-pixels, and rgb-scale must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / f"{args.prefix}_mask.dat"
    audit_path = args.output_dir / f"{args.prefix}_audit.json"
    background_path = args.output_dir / f"{args.prefix}_background.png"
    mask_preview_path = args.output_dir / f"{args.prefix}_mask_preview.png"
    overlay_path = args.output_dir / f"{args.prefix}_overlay.png"
    expected = (
        mask_path,
        mask_path.with_suffix(".hdr"),
        audit_path,
        background_path,
        mask_preview_path,
        overlay_path,
    )
    existing = [path for path in expected if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(map(str, existing)))

    started = monotonic()
    with EnviDataset(args.swir) as swir:
        result = build_material_mask(
            swir,
            chunk_rows=args.chunk_rows,
            minimum_component_pixels=args.minimum_component_pixels,
            minimum_mask_fraction=0.05,
            maximum_mask_fraction=0.85,
            source_override="automatic_swir_reflectance_material_mask",
            strict=False,
        )
        write_envi(
            result.mask.astype(np.uint8),
            mask_path,
            class_names=("background", "core foreground"),
            description="CoreSpec automatic SWIR reflectance foreground baseline",
        )
        preview_height = min(max(1, args.preview_height), swir.info.lines)
        target_rows = np.linspace(0, swir.info.lines - 1, preview_height).round().astype(np.int64)
        wavelengths = swir.info.wavelengths_nm
        if wavelengths is not None:
            background_band = int(np.argmin(np.abs(wavelengths - 1600.0)))
        else:
            background_band = swir.info.bands // 2
        background = np.asarray(
            swir.read_rows(0, swir.info.lines, bands=(background_band,))[target_rows, :, 0],
            dtype=np.float32,
        )
        preview_mask = result.mask[target_rows]
        gray = _stretch(background, preview_mask)
        swir_rgb = np.repeat(gray[..., None], 3, axis=2)
        background_render = swir_rgb
        overlay_background = swir_rgb
        if args.registered_rgb is not None:
            with EnviDataset(args.registered_rgb) as rgb:
                expected_rgb = (swir.info.lines * args.rgb_scale, swir.info.samples * args.rgb_scale)
                if (rgb.info.lines, rgb.info.samples) != expected_rgb:
                    raise ValueError(
                        f"Registered RGB shape {(rgb.info.lines, rgb.info.samples)} does not match {expected_rgb}"
                    )
                rgb_rows = np.linspace(0, rgb.info.lines - 1, preview_height).round().astype(np.int64)
                rgb_preview = _registered_rgb_preview(rgb, rgb_rows)
                columns = np.minimum(
                    (np.arange(rgb.info.samples, dtype=np.float64) * swir.info.samples
                     / rgb.info.samples).astype(np.int64),
                    swir.info.samples - 1,
                )
                preview_mask = result.mask[np.ix_(target_rows, columns)]
                background_render = rgb_preview
                overlay_background = rgb_preview
        overlay = overlay_background.astype(np.float32)
        tint = np.zeros_like(overlay)
        tint[..., 1] = 255.0
        alpha = (0.32 * preview_mask.astype(np.float32))[..., None]
        overlay = np.clip(overlay * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
        Image.fromarray(background_render).save(background_path)
        Image.fromarray(preview_mask.astype(np.uint8) * 255).save(mask_preview_path)
        Image.fromarray(overlay).save(overlay_path)
        report = {
            "status": result.audit.status,
            "source_swir": str(swir.info.data_path),
            "shape": [swir.info.lines, swir.info.samples, swir.info.bands],
            "background_band_index": background_band,
            "background_wavelength_nm": (
                None if wavelengths is None else float(wavelengths[background_band])
            ),
            "mask_audit": result.audit.to_dict(),
            "outputs": {
                "mask": str(mask_path),
                "background_preview": str(background_path),
                "mask_preview": str(mask_preview_path),
                "overlay": str(overlay_path),
            },
            "elapsed_seconds": monotonic() - started,
            "review_required": True,
            "evidence_boundary": (
                "The SWIR reflectance mask is an independent same-grid engineering baseline, "
                "not manually labelled accuracy evidence. Fragmented or tray-like regions require review."
            ),
        }
        audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
