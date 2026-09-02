from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from corespec_mapper.envi import EnviDataset
from corespec_mapper.foreground_model import _axis_positions, _model_class


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fixed radiometric transforms for a registered RGB scan."
    )
    parser.add_argument("--registered-rgb", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--reference-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--sample-height", type=int, default=1024)
    return parser.parse_args()


def _profile(values: np.ndarray) -> dict[str, np.ndarray]:
    pixels = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    return {
        "mean": np.mean(pixels, axis=0),
        "std": np.std(pixels, axis=0),
        "p05": np.percentile(pixels, 5.0, axis=0),
        "p25": np.percentile(pixels, 25.0, axis=0),
        "p50": np.percentile(pixels, 50.0, axis=0),
        "p75": np.percentile(pixels, 75.0, axis=0),
        "p95": np.percentile(pixels, 95.0, axis=0),
    }


def _affine(lower_source, upper_source, lower_target, upper_target) -> tuple[np.ndarray, np.ndarray]:
    gain = np.divide(
        upper_target - lower_target,
        np.maximum(upper_source - lower_source, 1e-6),
    ).astype(np.float32)
    offset = (lower_target - lower_source * gain).astype(np.float32)
    return gain, offset


def _predict(image: np.ndarray, *, torch, model, device, manifest: dict) -> np.ndarray:
    input_config = manifest.get("input", {})
    mean = np.asarray(input_config.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    std = np.asarray(input_config.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
    std[std == 0] = 1.0
    normalized = (np.asarray(image, dtype=np.float32) - mean[None, None, :]) / std[None, None, :]
    height, width = normalized.shape[:2]
    inference = manifest.get("inference", {})
    tile_size = int(inference.get("tile_size", input_config.get("image_size", 512)))
    overlap = float(inference.get("overlap", 0.25))
    step = max(1, int(tile_size * (1.0 - overlap)))
    padded_height, padded_width = max(height, tile_size), max(width, tile_size)
    padded = np.pad(
        normalized,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="edge",
    )
    axis = np.hanning(tile_size)
    weight = np.outer(axis, axis).astype(np.float32)
    positive = weight[weight > 0]
    weight[weight == 0] = float(np.min(positive)) if positive.size else 1.0
    weight /= max(float(np.max(weight)), 1e-6)
    probability_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros_like(probability_sum)
    with torch.inference_mode():
        for y in _axis_positions(padded_height, tile_size, step):
            for x in _axis_positions(padded_width, tile_size, step):
                patch = padded[y : y + tile_size, x : x + tile_size]
                tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)[None])).to(device)
                values = model(tensor)[:, 1].squeeze(0).float().cpu().numpy()
                local = np.s_[y : y + tile_size, x : x + tile_size]
                probability_sum[local] += values * weight
                weight_sum[local] += weight
    return (probability_sum / np.maximum(weight_sum, 1e-6))[:height, :width]


def _panel(image: np.ndarray, probability: np.ndarray, title: str) -> np.ndarray:
    rgb = np.asarray(np.clip(np.rint(image), 0, 255), dtype=np.uint8)
    mask = probability >= 0.5
    overlay = rgb.astype(np.float32)
    tint = np.zeros_like(overlay)
    tint[..., 1] = 255.0
    alpha = (0.35 * mask.astype(np.float32))[..., None]
    overlay = np.clip(overlay * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
    probability_rgb = np.repeat(
        np.clip(np.rint(probability * 255.0), 0, 255).astype(np.uint8)[..., None], 3, axis=2
    )
    combined = np.concatenate([overlay, probability_rgb], axis=1)
    canvas = Image.fromarray(combined)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, min(combined.shape[1], 470), 27), fill=(0, 0, 0))
    draw.text((6, 6), title, fill=(255, 255, 255))
    return np.asarray(canvas)


def main() -> int:
    args = _parse_args()
    if args.sample_count < 1 or args.sample_height < 512:
        raise ValueError("sample-count must be positive and sample-height must be at least 512")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.model_package / "model_manifest.json").read_text(encoding="utf-8"))
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model_class(torch)()
    state = torch.load(args.model_package / str(manifest["weights"]), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict({str(key).removeprefix("module."): value for key, value in state.items()})
    model.to(device).eval()

    reference_paths = sorted(args.reference_images.glob("*.png"))
    if not reference_paths:
        raise FileNotFoundError(f"No PNG reference images under {args.reference_images}")
    reference_pixels = np.concatenate(
        [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).reshape(-1, 3) for path in reference_paths],
        axis=0,
    )
    reference = _profile(reference_pixels)

    records: list[dict[str, object]] = []
    with EnviDataset(args.registered_rgb) as dataset:
        profile_rows = np.linspace(0, dataset.info.lines - 1, min(4096, dataset.info.lines)).round().astype(np.int64)
        if dataset.info.interleave != "bil":
            raise ValueError("This audit expects the registered BIL RGB proxy")
        scan_sample = np.moveaxis(np.asarray(dataset._array[profile_rows, :3, :]), 1, 2)
        scan = _profile(scan_sample)
        range_gain, range_offset = _affine(
            scan["p05"], scan["p95"], reference["p05"], reference["p95"]
        )
        iqr_gain, iqr_offset = _affine(
            scan["p25"], scan["p75"], reference["p25"], reference["p75"]
        )
        std_gain = np.divide(reference["std"], np.maximum(scan["std"], 1e-6)).astype(np.float32)
        std_offset = (reference["mean"] - scan["mean"] * std_gain).astype(np.float32)
        strategies = {
            "identity": (np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32), False),
            "reverse_channels": (np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32), True),
            "reference_p05_p95": (range_gain, range_offset, False),
            "reference_iqr": (iqr_gain, iqr_offset, False),
            "reference_mean_std": (std_gain, std_offset, False),
        }
        starts = np.linspace(
            0,
            max(0, dataset.info.lines - args.sample_height),
            args.sample_count,
        ).round().astype(np.int64)
        for sample_index, start in enumerate(starts):
            source = np.asarray(
                dataset.read_rows(int(start), int(start) + args.sample_height, bands=(0, 1, 2)),
                dtype=np.float32,
            )
            panels: list[np.ndarray] = []
            strategy_records: dict[str, object] = {}
            for name, (gain, offset, reverse) in strategies.items():
                transformed = source[..., ::-1] if reverse else source
                transformed = np.clip(transformed * gain[None, None, :] + offset[None, None, :], 0, 255)
                probability = _predict(
                    transformed,
                    torch=torch,
                    model=model,
                    device=device,
                    manifest=manifest,
                )
                foreground_fraction = float(np.mean(probability >= 0.5))
                quantiles = np.percentile(probability, [1, 25, 50, 75, 95, 99])
                strategy_records[name] = {
                    "gain": gain.tolist(),
                    "offset": offset.tolist(),
                    "reverse_channels": reverse,
                    "foreground_fraction": foreground_fraction,
                    "probability_quantiles": {
                        str(point): float(value)
                        for point, value in zip((1, 25, 50, 75, 95, 99), quantiles)
                    },
                }
                panels.append(
                    _panel(
                        transformed,
                        probability,
                        f"{name} foreground={foreground_fraction:.3f}",
                    )
                )
            contact = np.concatenate(panels, axis=1)
            contact_path = args.output_dir / f"sample_{sample_index + 1:02d}_row_{int(start)}.jpg"
            Image.fromarray(contact).save(contact_path, quality=92)
            records.append(
                {
                    "start_row": int(start),
                    "height": args.sample_height,
                    "contact_sheet": str(contact_path),
                    "strategies": strategy_records,
                }
            )
    report = {
        "registered_rgb": str(args.registered_rgb),
        "model_package": str(args.model_package),
        "reference_images": [str(path) for path in reference_paths],
        "reference_profile": {key: value.tolist() for key, value in reference.items()},
        "scan_profile": {key: value.tolist() for key, value in scan.items()},
        "records": records,
        "evidence_boundary": (
            "Radiometric matching is an input-domain diagnostic. A transform may be adopted only "
            "if RGB/SWIR agreement and representative manual review improve without systematic tray inclusion."
        ),
    }
    report_path = args.output_dir / "preprocessing_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
