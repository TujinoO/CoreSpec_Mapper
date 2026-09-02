from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from corespec_mapper.envi import EnviDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render native-grid RGB/SWIR/hybrid foreground QA samples."
    )
    parser.add_argument("--registered-rgb", type=Path, required=True)
    parser.add_argument("--swir-mask", type=Path, required=True)
    parser.add_argument("--rgb-mask", type=Path, required=True)
    parser.add_argument("--hybrid-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rgb-scale", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--sample-height", type=int, default=1024)
    return parser.parse_args()


def _read_mask(path: Path) -> np.ndarray:
    with EnviDataset(path) as dataset:
        if dataset.info.bands != 1:
            raise ValueError(f"Expected one-band mask: {path}")
        return np.asarray(dataset.read_rows(0, dataset.info.lines)[..., 0] > 0).copy()


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = rgb.astype(np.float32)
    tint = np.empty_like(result)
    tint[:] = np.asarray(color, dtype=np.float32)
    alpha = (0.34 * mask.astype(np.float32))[..., None]
    return np.clip(result * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(300, panel.shape[1]), 27), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(image)


def main() -> int:
    args = _parse_args()
    if args.rgb_scale < 1 or args.sample_count < 1 or args.sample_height < 1:
        raise ValueError("rgb-scale, sample-count, and sample-height must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.glob("sample_*.jpg"))
    report_path = args.output_dir / "native_sample_manifest.json"
    if existing or report_path.exists():
        raise FileExistsError("Refusing to overwrite existing QA samples")
    swir = _read_mask(args.swir_mask)
    rgb_mask = _read_mask(args.rgb_mask)
    hybrid = _read_mask(args.hybrid_mask)
    if not (swir.shape == rgb_mask.shape == hybrid.shape):
        raise ValueError("All masks must have identical shapes")

    records = []
    with EnviDataset(args.registered_rgb) as rgb:
        if rgb.info.interleave != "bil":
            raise ValueError("Registered RGB proxy must use BIL interleave")
        expected = (swir.shape[0] * args.rgb_scale, swir.shape[1] * args.rgb_scale)
        if (rgb.info.lines, rgb.info.samples) != expected:
            raise ValueError(f"Registered RGB shape does not match masks x scale: {expected}")
        height = min(args.sample_height, rgb.info.lines)
        starts = np.linspace(0, max(0, rgb.info.lines - height), args.sample_count).round().astype(np.int64)
        for index, start in enumerate(starts):
            stop = int(start) + height
            source = np.moveaxis(np.asarray(rgb._array[int(start):stop, :3, :]), 1, 2).astype(np.uint8)
            target_rows = np.minimum(
                (np.arange(int(start), stop, dtype=np.int64) // args.rgb_scale),
                swir.shape[0] - 1,
            )
            target_columns = np.minimum(
                (np.arange(rgb.info.samples, dtype=np.int64) // args.rgb_scale),
                swir.shape[1] - 1,
            )
            swir_local = swir[np.ix_(target_rows, target_columns)]
            rgb_local = rgb_mask[np.ix_(target_rows, target_columns)]
            hybrid_local = hybrid[np.ix_(target_rows, target_columns)]
            agreement = np.zeros(source.shape, dtype=np.uint8)
            both = swir_local & rgb_local
            swir_only = swir_local & ~rgb_local
            rgb_only = ~swir_local & rgb_local
            agreement[both] = (0, 190, 0)
            agreement[swir_only] = (0, 200, 255)
            agreement[rgb_only] = (255, 0, 255)
            panels = (
                _label(source, "registered RGB"),
                _label(_overlay(source, swir_local, (0, 255, 0)), "SWIR primary"),
                _label(_overlay(source, rgb_local, (255, 0, 255)), "RGB U-Net"),
                _label(_overlay(source, hybrid_local, (0, 255, 0)), "conservative hybrid"),
                _label(agreement, "agreement"),
            )
            output = args.output_dir / f"sample_{index + 1:02d}_rgb_row_{int(start):06d}.jpg"
            Image.fromarray(np.concatenate(panels, axis=1)).save(output, quality=92)
            records.append(
                {
                    "rgb_start_row": int(start),
                    "rgb_stop_row": stop,
                    "target_start_row": int(start) // args.rgb_scale,
                    "target_stop_row": (stop - 1) // args.rgb_scale + 1,
                    "output": str(output),
                    "fractions": {
                        "swir": float(np.mean(swir_local)),
                        "rgb": float(np.mean(rgb_local)),
                        "hybrid": float(np.mean(hybrid_local)),
                    },
                }
            )
    report = {"samples": records, "manual_review_required": True}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
