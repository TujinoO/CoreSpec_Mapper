from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image

from corespec_mapper.foreground_model import _axis_positions, _model_class


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the packaged CoreMaskUNet decision rule against labelled RGB images."
    )
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--scales",
        default="1.0",
        help="Comma-separated image scales used to audit scale sensitivity, for example 1,0.5,0.25.",
    )
    return parser.parse_args()


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    true_positive = int(np.count_nonzero(prediction & truth))
    false_positive = int(np.count_nonzero(prediction & ~truth))
    false_negative = int(np.count_nonzero(~prediction & truth))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    dice = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "predicted_fraction": float(np.mean(prediction)),
        "truth_fraction": float(np.mean(truth)),
    }


def _load_model(package: Path, device_name: str):
    import torch

    manifest = json.loads((package / "model_manifest.json").read_text(encoding="utf-8"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = _model_class(torch)()
    state = torch.load(package / str(manifest["weights"]), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict({str(key).removeprefix("module."): value for key, value in state.items()})
    model.to(device).eval()
    return torch, model, manifest, device


def _predict(image: np.ndarray, *, torch, model, manifest: dict, device) -> tuple[np.ndarray, np.ndarray]:
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
    padded_height = max(height, tile_size)
    padded_width = max(width, tile_size)
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
    direct_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    softmax_sum = np.zeros_like(direct_sum)
    weight_sum = np.zeros_like(direct_sum)
    with torch.inference_mode():
        for y in _axis_positions(padded_height, tile_size, step):
            for x in _axis_positions(padded_width, tile_size, step):
                patch = padded[y : y + tile_size, x : x + tile_size]
                tensor = torch.from_numpy(patch.transpose(2, 0, 1)[None]).float().to(device)
                output = model(tensor)
                direct = output[:, 1].squeeze(0).detach().cpu().numpy().astype(np.float32)
                softmax = torch.softmax(output, dim=1)[:, 1].squeeze(0).detach().cpu().numpy().astype(np.float32)
                local = np.s_[y : y + tile_size, x : x + tile_size]
                direct_sum[local] += direct * weight
                softmax_sum[local] += softmax * weight
                weight_sum[local] += weight
    denominator = np.maximum(weight_sum, 1e-6)
    return (
        (direct_sum / denominator)[:height, :width],
        (softmax_sum / denominator)[:height, :width],
    )


def main() -> int:
    args = _parse_args()
    started = monotonic()
    torch, model, manifest, device = _load_model(args.model_package, args.device)
    scales = tuple(float(value) for value in args.scales.split(",") if value.strip())
    if not scales or any(value <= 0.0 or value > 1.0 for value in scales):
        raise ValueError("--scales values must be in (0, 1]")
    records: list[dict[str, object]] = []
    for image_path in sorted(args.images.glob("*.png")):
        mask_path = args.masks / image_path.name
        if not mask_path.is_file():
            continue
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.float32)
        with Image.open(mask_path) as opened:
            truth = np.asarray(opened.convert("L")) > 127
        height, width = truth.shape
        scale_records: list[dict[str, object]] = []
        for scale in scales:
            if scale == 1.0:
                scaled_image = image
            else:
                target_size = (
                    max(16, int(round(image.shape[1] * scale))),
                    max(16, int(round(image.shape[0] * scale))),
                )
                scaled_image = np.asarray(
                    Image.fromarray(image.astype(np.uint8), mode="RGB").resize(
                        target_size,
                        resample=Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )
            direct_probability, legacy_probability = _predict(
                scaled_image,
                torch=torch,
                model=model,
                manifest=manifest,
                device=device,
            )
            if direct_probability.shape != truth.shape:
                direct_probability = np.asarray(
                    Image.fromarray(direct_probability.astype(np.float32), mode="F").resize(
                        (width, height),
                        resample=Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )
                legacy_probability = np.asarray(
                    Image.fromarray(legacy_probability.astype(np.float32), mode="F").resize(
                        (width, height),
                        resample=Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )
            direct = direct_probability >= 0.5
            legacy = legacy_probability >= 0.5
            scale_records.append({
                "scale": scale,
                "inference_shape": list(scaled_image.shape[:2]),
                "direct_foreground_channel": _metrics(direct, truth),
                "legacy_softmax_after_sigmoid": _metrics(legacy, truth),
                "decision_disagreement_fraction": float(np.mean(direct != legacy)),
                "direct_probability_range": [
                    float(np.min(direct_probability)),
                    float(np.max(direct_probability)),
                ],
                "legacy_probability_range": [
                    float(np.min(legacy_probability)),
                    float(np.max(legacy_probability)),
                ],
            })
        records.append({
            "image": str(image_path),
            "scales": scale_records,
        })
    report = {
        "model_package": str(args.model_package),
        "model_version": manifest.get("model_version"),
        "device": str(device),
        "labelled_image_count": len(records),
        "records": records,
        "elapsed_seconds": monotonic() - started,
        "evidence_boundary": (
            "The available labelled images were used during fine-tuning; these metrics audit "
            "implementation semantics and are not independent generalisation evidence."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
