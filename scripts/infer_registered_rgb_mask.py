from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image

from corespec_mapper.envi import EnviDataset, write_envi
from corespec_mapper.foreground_model import _axis_positions, _model_class
from corespec_mapper.masking import build_material_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CoreMaskUNet on an RGB image already registered to a multiple of "
            "the target SWIR grid without loading the full RGB image into memory."
        )
    )
    parser.add_argument("--registered-rgb", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="core_foreground")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use CUDA float16 autocast")
    parser.add_argument(
        "--input-gains",
        default="1,1,1",
        help="Fixed comma-separated R,G,B gains applied before model normalization",
    )
    parser.add_argument(
        "--input-offsets",
        default="0,0,0",
        help="Fixed comma-separated R,G,B offsets applied after gains",
    )
    parser.add_argument("--pool-chunk-rows", type=int, default=256)
    parser.add_argument("--minimum-component-pixels", type=int, default=64)
    parser.add_argument("--preview-height", type=int, default=4096)
    return parser.parse_args()


def _parse_triplet(value: str, name: str) -> np.ndarray:
    values = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float32)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain three finite comma-separated values")
    return values


def _write_probability_header(path: Path, *, lines: int, samples: int, band_name: str) -> Path:
    header = path.with_suffix(".hdr")
    header.write_text(
        "ENVI\n"
        "description = {CoreMaskUNet foreground probability}\n"
        f"samples = {samples}\n"
        f"lines = {lines}\n"
        "bands = 1\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 4\n"
        "interleave = bsq\n"
        "byte order = 0\n"
        f"band names = {{{band_name}}}\n",
        encoding="utf-8",
    )
    return header


def _weight_window(tile_size: int) -> np.ndarray:
    axis = np.hanning(tile_size)
    if not np.any(axis):
        axis = np.ones(tile_size, dtype=np.float32)
    weight = np.outer(axis, axis).astype(np.float32)
    positive = weight[weight > 0]
    weight[weight == 0] = float(np.min(positive)) if positive.size else 1.0
    weight /= max(float(np.max(weight)), 1e-6)
    return weight


def _denominator_rows(
    row_start: int,
    row_stop: int,
    *,
    height: int,
    width: int,
    tile_size: int,
    y_positions: list[int],
    x_positions: list[int],
    weight: np.ndarray,
) -> np.ndarray:
    denominator = np.zeros((row_stop - row_start, width), dtype=np.float32)
    for y in y_positions:
        overlap_start = max(row_start, y)
        overlap_stop = min(row_stop, y + tile_size, height)
        if overlap_start >= overlap_stop:
            continue
        output_rows = slice(overlap_start - row_start, overlap_stop - row_start)
        weight_rows = slice(overlap_start - y, overlap_stop - y)
        for x in x_positions:
            overlap_width = min(tile_size, width - x)
            if overlap_width > 0:
                denominator[output_rows, x : x + overlap_width] += weight[weight_rows, :overlap_width]
    return denominator


def _load_model(args: argparse.Namespace, manifest: dict):
    import torch

    requested = args.device
    if requested == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = requested
    if device_name == "cuda" and not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but is not available")
        device_name = "cpu"
    device = torch.device(device_name)
    model = _model_class(torch)()
    weights_path = args.model_package / str(manifest["weights"])
    state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict({str(key).removeprefix("module."): value for key, value in state.items()})
    model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        probe = torch.zeros((1, 3, 32, 32), dtype=torch.float32, device=device)
        try:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=args.amp
            ):
                model(probe)
            torch.cuda.synchronize(device)
        except RuntimeError as exc:
            if requested != "auto":
                raise
            device = torch.device("cpu")
            model.to(device).eval()
            device_name = "cpu"
            args.amp = False
            print(f"CUDA compatibility probe failed; falling back to CPU: {exc}", flush=True)
    return torch, model, device, weights_path


def _save_previews(
    registered_rgb: EnviDataset,
    probability: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    prefix: str,
    preview_height: int,
) -> tuple[Path, Path, Path]:
    height = min(max(1, preview_height), registered_rgb.info.lines)
    rgb_rows = np.linspace(0, registered_rgb.info.lines - 1, height).round().astype(np.int64)
    if registered_rgb.info.interleave == "bil":
        rgb = np.moveaxis(np.asarray(registered_rgb._array[rgb_rows, :3, :]), 1, 2)
    elif registered_rgb.info.interleave == "bip":
        rgb = np.asarray(registered_rgb._array[rgb_rows, :, :3])
    else:
        rgb = np.moveaxis(np.asarray(registered_rgb._array[:3, rgb_rows, :]), 0, 2)
    rgb = np.asarray(rgb, dtype=np.uint8)
    target_rows = np.linspace(0, mask.shape[0] - 1, height).round().astype(np.int64)
    column_indices = np.minimum(
        (np.arange(registered_rgb.info.samples, dtype=np.float64) * mask.shape[1]
         / registered_rgb.info.samples).astype(np.int64),
        mask.shape[1] - 1,
    )
    preview_mask = mask[np.ix_(target_rows, column_indices)]
    preview_probability = np.asarray(probability[np.ix_(target_rows, column_indices)], dtype=np.float32)
    overlay = rgb.astype(np.float32)
    tint = np.zeros_like(overlay)
    tint[..., 1] = 255.0
    alpha = (0.32 * preview_mask.astype(np.float32))[..., None]
    overlay = np.clip(overlay * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
    mask_path = output_dir / f"{prefix}_mask_preview.png"
    probability_path = output_dir / f"{prefix}_probability_preview.png"
    overlay_path = output_dir / f"{prefix}_overlay_preview.png"
    Image.fromarray((preview_mask.astype(np.uint8) * 255)).save(mask_path)
    Image.fromarray(np.clip(np.rint(preview_probability * 255.0), 0, 255).astype(np.uint8)).save(
        probability_path
    )
    Image.fromarray(overlay).save(overlay_path)
    return mask_path, probability_path, overlay_path


def main() -> int:
    args = _parse_args()
    if args.scale < 1 or args.batch_size < 1 or args.pool_chunk_rows < 1:
        raise ValueError("scale, batch-size, and pool-chunk-rows must be positive")
    if args.minimum_component_pixels < 1:
        raise ValueError("minimum-component-pixels must be positive")
    manifest_path = args.model_package / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inference = manifest.get("inference", {})
    input_config = manifest.get("input", {})
    tile_size = int(inference.get("tile_size", input_config.get("image_size", 512)))
    overlap = float(inference.get("overlap", 0.25))
    if tile_size < 16 or tile_size % 16:
        raise ValueError("Model tile size must be at least 16 and divisible by 16")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Model overlap must be in [0, 1)")
    threshold = float(args.threshold if args.threshold is not None else inference.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    input_gains = _parse_triplet(args.input_gains, "input-gains")
    input_offsets = _parse_triplet(args.input_offsets, "input-offsets")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability_2x_path = args.output_dir / f"{args.prefix}_probability_{args.scale}x.dat"
    probability_target_path = args.output_dir / f"{args.prefix}_probability_SWIR.dat"
    raw_mask_path = args.output_dir / f"{args.prefix}_mask_SWIR_raw.dat"
    clean_mask_path = args.output_dir / f"{args.prefix}_mask_SWIR.dat"
    metadata_path = args.output_dir / f"{args.prefix}_audit.json"
    preview_paths = (
        args.output_dir / f"{args.prefix}_mask_preview.png",
        args.output_dir / f"{args.prefix}_probability_preview.png",
        args.output_dir / f"{args.prefix}_overlay_preview.png",
    )
    expected = (
        probability_2x_path,
        probability_2x_path.with_suffix(".hdr"),
        probability_target_path,
        probability_target_path.with_suffix(".hdr"),
        raw_mask_path,
        raw_mask_path.with_suffix(".hdr"),
        clean_mask_path,
        clean_mask_path.with_suffix(".hdr"),
        metadata_path,
        *preview_paths,
    )
    existing = [path for path in expected if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(map(str, existing)))

    started = monotonic()
    torch, model, device, weights_path = _load_model(args, manifest)
    mean = np.asarray(input_config.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    std = np.asarray(input_config.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
    std[std == 0] = 1.0
    weight = _weight_window(tile_size)

    with EnviDataset(args.registered_rgb) as registered_rgb, EnviDataset(args.target) as target:
        height, width = registered_rgb.info.lines, registered_rgb.info.samples
        expected_shape = (target.info.lines * args.scale, target.info.samples * args.scale)
        if (height, width) != expected_shape:
            raise ValueError(
                f"Registered RGB shape {(height, width)} does not equal target x scale {expected_shape}"
            )
        if registered_rgb.info.bands < 3:
            raise ValueError("Registered RGB must have at least three bands")
        step = max(1, int(tile_size * (1.0 - overlap)))
        padded_height = max(height, tile_size)
        padded_width = max(width, tile_size)
        y_positions = _axis_positions(padded_height, tile_size, step)
        x_positions = _axis_positions(padded_width, tile_size, step)
        tile_count = len(y_positions) * len(x_positions)
        probability_sum = np.memmap(
            probability_2x_path, dtype="<f4", mode="w+", shape=(height, width)
        )
        probability_sum[:] = 0.0
        completed_tiles = 0
        inference_started = monotonic()
        with torch.inference_mode():
            for y_index, y in enumerate(y_positions):
                source_stop = min(y + tile_size, height)
                row_block = np.asarray(
                    registered_rgb.read_rows(y, source_stop, bands=(0, 1, 2)), dtype=np.float32
                )
                harmonized = np.clip(
                    row_block * input_gains[None, None, :] + input_offsets[None, None, :],
                    0.0,
                    255.0,
                )
                normalized = (harmonized - mean[None, None, :]) / std[None, None, :]
                if normalized.shape[0] < tile_size or normalized.shape[1] < padded_width:
                    normalized = np.pad(
                        normalized,
                        (
                            (0, tile_size - normalized.shape[0]),
                            (0, padded_width - normalized.shape[1]),
                            (0, 0),
                        ),
                        mode="edge",
                    )
                for batch_start in range(0, len(x_positions), args.batch_size):
                    batch_x = x_positions[batch_start : batch_start + args.batch_size]
                    batch = np.stack(
                        [normalized[:, x : x + tile_size, :].transpose(2, 0, 1) for x in batch_x],
                        axis=0,
                    )
                    tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=(args.amp and device.type == "cuda"),
                    ):
                        output = model(tensor)
                    values = output[:, 1].float().cpu().numpy()
                    for item, x in enumerate(batch_x):
                        valid_height = min(tile_size, height - y)
                        valid_width = min(tile_size, width - x)
                        probability_sum[y : y + valid_height, x : x + valid_width] += (
                            values[item, :valid_height, :valid_width]
                            * weight[:valid_height, :valid_width]
                        )
                    completed_tiles += len(batch_x)
                if device.type == "cuda" and (y_index + 1) % 20 == 0:
                    torch.cuda.synchronize(device)
                if (y_index + 1) % 10 == 0 or y_index + 1 == len(y_positions):
                    print(
                        f"inference {completed_tiles}/{tile_count} tiles "
                        f"({100.0 * completed_tiles / tile_count:.1f}%)",
                        flush=True,
                    )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = monotonic() - inference_started

        probability_target = np.memmap(
            probability_target_path,
            dtype="<f4",
            mode="w+",
            shape=(target.info.lines, target.info.samples),
        )
        pooling_started = monotonic()
        minimum_denominator = float("inf")
        for target_start in range(0, target.info.lines, args.pool_chunk_rows):
            target_stop = min(target.info.lines, target_start + args.pool_chunk_rows)
            source_start = target_start * args.scale
            source_stop = target_stop * args.scale
            denominator = _denominator_rows(
                source_start,
                source_stop,
                height=height,
                width=width,
                tile_size=tile_size,
                y_positions=y_positions,
                x_positions=x_positions,
                weight=weight,
            )
            minimum_denominator = min(minimum_denominator, float(np.min(denominator)))
            if np.any(denominator <= 0.0):
                raise RuntimeError("Tile blending left uncovered pixels")
            normalized_probability = np.asarray(
                probability_sum[source_start:source_stop], dtype=np.float32
            ) / denominator
            probability_sum[source_start:source_stop] = normalized_probability
            pooled = normalized_probability.reshape(
                target_stop - target_start,
                args.scale,
                target.info.samples,
                args.scale,
            ).mean(axis=(1, 3))
            probability_target[target_start:target_stop] = pooled
        probability_sum.flush()
        probability_target.flush()
        pooling_seconds = monotonic() - pooling_started
        _write_probability_header(
            probability_2x_path,
            lines=height,
            samples=width,
            band_name=f"foreground probability {args.scale}x SWIR grid",
        )
        _write_probability_header(
            probability_target_path,
            lines=target.info.lines,
            samples=target.info.samples,
            band_name="foreground probability SWIR grid",
        )

        raw_mask = np.asarray(probability_target >= threshold, dtype=bool)
        material = build_material_mask(
            target,
            raw_mask,
            minimum_component_pixels=args.minimum_component_pixels,
            clean_existing_mask=True,
            minimum_mask_fraction=0.05,
            maximum_mask_fraction=0.85,
            source_override="automatic_registered_rgb_unet_probability_area_pool",
            strict=False,
        )
        clean_mask = material.mask
        write_envi(
            raw_mask.astype(np.uint8),
            raw_mask_path,
            class_names=("background", "core foreground"),
            description="Raw registered RGB CoreMaskUNet mask on SWIR grid",
        )
        write_envi(
            clean_mask.astype(np.uint8),
            clean_mask_path,
            class_names=("background", "core foreground"),
            description="Cleaned registered RGB CoreMaskUNet mask on SWIR grid",
        )
        saved_previews = _save_previews(
            registered_rgb,
            probability_target,
            clean_mask,
            args.output_dir,
            args.prefix,
            args.preview_height,
        )

        probabilities = np.asarray(probability_target)
        quantiles = np.percentile(probabilities, [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5])
        uncertainty_fraction = float(np.mean((probabilities >= 0.4) & (probabilities <= 0.6)))
        cuda_record = None
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            cuda_record = {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": int(properties.total_memory),
                "torch_cuda": torch.version.cuda,
            }
        report = {
            "status": material.audit.status,
            "engine": "integrated_core_foreground_v2_streaming_registered",
            "model_name": manifest.get("model_name"),
            "model_version": manifest.get("model_version"),
            "model_manifest": str(manifest_path.resolve()),
            "weights": str(weights_path.resolve()),
            "registered_rgb": str(registered_rgb.info.data_path),
            "target_swir": str(target.info.data_path),
            "registered_shape": [height, width, registered_rgb.info.bands],
            "target_shape": [target.info.lines, target.info.samples],
            "scale": args.scale,
            "device": str(device),
            "cuda": cuda_record,
            "amp_float16": bool(args.amp and device.type == "cuda"),
            "batch_size": args.batch_size,
            "tile_size": tile_size,
            "overlap": overlap,
            "tile_count": tile_count,
            "decision_rule": "direct_sigmoid_foreground_channel",
            "input_radiometric_harmonization": {
                "channel_order": "RGB",
                "gains": input_gains.tolist(),
                "offsets": input_offsets.tolist(),
                "clipping_range": [0.0, 255.0],
            },
            "threshold": threshold,
            "downsampling": "mean_probability_area_pool",
            "minimum_blend_denominator": minimum_denominator,
            "probability_quantiles": {
                str(point): float(value)
                for point, value in zip((0.5, 1, 5, 25, 50, 75, 95, 99, 99.5), quantiles)
            },
            "uncertainty_fraction_probability_0.4_to_0.6": uncertainty_fraction,
            "raw_foreground_fraction": float(np.mean(raw_mask)),
            "clean_foreground_fraction": float(np.mean(clean_mask)),
            "material_mask_audit": material.audit.to_dict(),
            "outputs": {
                "probability_registered_grid": str(probability_2x_path),
                "probability_swir_grid": str(probability_target_path),
                "raw_mask_swir_grid": str(raw_mask_path),
                "clean_mask_swir_grid": str(clean_mask_path),
                "mask_preview": str(saved_previews[0]),
                "probability_preview": str(saved_previews[1]),
                "overlay_preview": str(saved_previews[2]),
            },
            "timing_seconds": {
                "inference": inference_seconds,
                "normalization_and_pooling": pooling_seconds,
                "total": monotonic() - started,
            },
            "review_required": True,
            "evidence_boundary": (
                "The packaged model was fine-tuned on two labelled boxes. This same-data result "
                "must pass registration, RGB/SWIR consistency, and representative manual review "
                "before it is accepted as an engineering foreground mask."
            ),
        }
        metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
