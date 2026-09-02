from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from corespec_mapper.envi import EnviDataset, write_envi
from corespec_mapper.masking import _component_cleanup, _interior_pixel_fraction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SWIR and registered-RGB foreground masks and build a conservative hybrid."
    )
    parser.add_argument("--swir-mask", type=Path, required=True)
    parser.add_argument("--rgb-mask", type=Path, required=True)
    parser.add_argument("--rgb-probability", type=Path, required=True)
    parser.add_argument("--registered-rgb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="foreground")
    parser.add_argument("--rgb-scale", type=int, default=2)
    parser.add_argument("--closing-size", type=int, default=3)
    parser.add_argument("--maximum-hole-pixels", type=int, default=125)
    parser.add_argument("--minimum-component-pixels", type=int, default=64)
    parser.add_argument("--rgb-confidence", type=float, default=0.90)
    parser.add_argument("--rgb-expansion-pixels", type=int, default=2)
    parser.add_argument("--preview-height", type=int, default=4096)
    parser.add_argument("--depth-block-rows", type=int, default=512)
    return parser.parse_args()


def _read_plane(path: Path) -> np.ndarray:
    with EnviDataset(path) as dataset:
        if dataset.info.bands != 1:
            raise ValueError(f"Expected a one-band ENVI dataset: {path}")
        return np.asarray(dataset.read_rows(0, dataset.info.lines)[..., 0]).copy()


def _fill_small_holes(mask: np.ndarray, maximum_pixels: int) -> tuple[np.ndarray, int, int]:
    if maximum_pixels < 1:
        return np.asarray(mask, dtype=bool).copy(), 0, 0
    source = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(~source, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    border_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    eligible = sizes <= maximum_pixels
    eligible[0] = False
    eligible[border_labels] = False
    fill = eligible[labels]
    return source | fill, int(np.count_nonzero(eligible)), int(np.count_nonzero(fill))


def _metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    intersection = int(np.count_nonzero(first & second))
    first_only = int(np.count_nonzero(first & ~second))
    second_only = int(np.count_nonzero(~first & second))
    union = intersection + first_only + second_only
    dice = 2 * intersection / max(2 * intersection + first_only + second_only, 1)
    return {
        "intersection_pixels": intersection,
        "first_only_pixels": first_only,
        "second_only_pixels": second_only,
        "iou": intersection / max(union, 1),
        "dice": dice,
        "agreement_fraction": float(np.mean(first == second)),
    }


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = rgb.astype(np.float32)
    tint = np.empty_like(result)
    tint[:] = np.asarray(color, dtype=np.float32)
    alpha = (0.32 * np.asarray(mask, dtype=np.float32))[..., None]
    return np.clip(result * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def _rgb_preview(dataset: EnviDataset, rows: np.ndarray) -> np.ndarray:
    if dataset.info.interleave == "bil":
        return np.moveaxis(np.asarray(dataset._array[rows, :3, :]), 1, 2).astype(np.uint8, copy=False)
    if dataset.info.interleave == "bip":
        return np.asarray(dataset._array[rows, :, :3], dtype=np.uint8)
    return np.moveaxis(np.asarray(dataset._array[:3, rows, :]), 0, 2).astype(np.uint8, copy=False)


def main() -> int:
    args = _parse_args()
    if args.closing_size < 1 or args.closing_size % 2 == 0:
        raise ValueError("closing-size must be a positive odd number")
    if args.minimum_component_pixels < 1 or args.rgb_expansion_pixels < 0:
        raise ValueError("minimum-component-pixels must be positive and expansion must be non-negative")
    if not 0.0 <= args.rgb_confidence <= 1.0:
        raise ValueError("rgb-confidence must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "swir_postprocessed": args.output_dir / f"{args.prefix}_SWIR_postprocessed.dat",
        "hybrid": args.output_dir / f"{args.prefix}_hybrid.dat",
        "agreement": args.output_dir / f"{args.prefix}_agreement.dat",
    }
    preview_paths = {
        "swir": args.output_dir / f"{args.prefix}_SWIR_overlay.png",
        "rgb": args.output_dir / f"{args.prefix}_RGB_overlay.png",
        "hybrid": args.output_dir / f"{args.prefix}_hybrid_overlay.png",
        "agreement": args.output_dir / f"{args.prefix}_agreement_preview.png",
        "contact": args.output_dir / f"{args.prefix}_comparison_contact.png",
    }
    audit_path = args.output_dir / f"{args.prefix}_comparison_audit.json"
    expected = [audit_path, *preview_paths.values()]
    for path in output_paths.values():
        expected.extend((path, path.with_suffix(".hdr")))
    existing = [path for path in expected if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(map(str, existing)))

    swir = _read_plane(args.swir_mask) > 0
    rgb = _read_plane(args.rgb_mask) > 0
    probability = np.asarray(_read_plane(args.rgb_probability), dtype=np.float32)
    if swir.shape != rgb.shape or swir.shape != probability.shape:
        raise ValueError("SWIR mask, RGB mask, and RGB probability shapes must match")

    closing_structure = np.ones((args.closing_size, args.closing_size), dtype=bool)
    swir_closed = ndimage.binary_closing(swir, structure=closing_structure)
    swir_filled, filled_hole_count, filled_hole_pixels = _fill_small_holes(
        swir_closed, args.maximum_hole_pixels
    )
    swir_postprocessed, swir_components, swir_kept, swir_removed = _component_cleanup(
        swir_filled, args.minimum_component_pixels
    )
    if args.rgb_expansion_pixels:
        near_swir = ndimage.binary_dilation(
            swir_postprocessed,
            structure=np.ones((3, 3), dtype=bool),
            iterations=args.rgb_expansion_pixels,
        )
    else:
        near_swir = swir_postprocessed
    rgb_high_confidence = probability >= args.rgb_confidence
    rgb_guided_additions = rgb_high_confidence & near_swir & ~swir_postprocessed
    hybrid_seed = swir_postprocessed | rgb_guided_additions
    hybrid_filled, hybrid_hole_count, hybrid_hole_pixels = _fill_small_holes(
        hybrid_seed, args.maximum_hole_pixels
    )
    hybrid, hybrid_components, hybrid_kept, hybrid_removed = _component_cleanup(
        hybrid_filled, args.minimum_component_pixels
    )

    agreement = np.zeros(swir.shape, dtype=np.uint8)
    agreement[swir & rgb] = 1
    agreement[swir & ~rgb] = 2
    agreement[~swir & rgb] = 3
    write_envi(
        swir_postprocessed.astype(np.uint8),
        output_paths["swir_postprocessed"],
        class_names=("background", "core foreground"),
        description="SWIR foreground with model-manifest-compatible spatial postprocessing",
    )
    write_envi(
        hybrid.astype(np.uint8),
        output_paths["hybrid"],
        class_names=("background", "core foreground"),
        description="Conservative SWIR seed plus locally gated high-confidence RGB foreground",
    )
    write_envi(
        agreement,
        output_paths["agreement"],
        class_names=("neither", "both", "SWIR only", "RGB only"),
        class_lookup=(0, 0, 0, 0, 200, 0, 0, 200, 255, 255, 0, 255),
        description="SWIR and registered RGB foreground agreement",
    )

    with EnviDataset(args.registered_rgb) as registered_rgb:
        expected_rgb = (swir.shape[0] * args.rgb_scale, swir.shape[1] * args.rgb_scale)
        if (registered_rgb.info.lines, registered_rgb.info.samples) != expected_rgb:
            raise ValueError(
                f"Registered RGB shape {(registered_rgb.info.lines, registered_rgb.info.samples)} "
                f"does not match expected {expected_rgb}"
            )
        preview_height = min(max(1, args.preview_height), registered_rgb.info.lines)
        rgb_rows = np.linspace(0, registered_rgb.info.lines - 1, preview_height).round().astype(np.int64)
        target_rows = np.linspace(0, swir.shape[0] - 1, preview_height).round().astype(np.int64)
        columns = np.minimum(
            (np.arange(registered_rgb.info.samples, dtype=np.float64) * swir.shape[1]
             / registered_rgb.info.samples).astype(np.int64),
            swir.shape[1] - 1,
        )
        background = _rgb_preview(registered_rgb, rgb_rows)
        swir_preview = swir_postprocessed[np.ix_(target_rows, columns)]
        rgb_preview = rgb[np.ix_(target_rows, columns)]
        hybrid_preview = hybrid[np.ix_(target_rows, columns)]
        agreement_preview = agreement[np.ix_(target_rows, columns)]
        swir_overlay = _overlay(background, swir_preview, (0, 255, 0))
        rgb_overlay = _overlay(background, rgb_preview, (255, 0, 255))
        hybrid_overlay = _overlay(background, hybrid_preview, (0, 255, 0))
        agreement_colors = np.asarray(
            ((0, 0, 0), (0, 200, 0), (0, 200, 255), (255, 0, 255)), dtype=np.uint8
        )
        agreement_render = agreement_colors[agreement_preview]
        Image.fromarray(swir_overlay).save(preview_paths["swir"])
        Image.fromarray(rgb_overlay).save(preview_paths["rgb"])
        Image.fromarray(hybrid_overlay).save(preview_paths["hybrid"])
        Image.fromarray(agreement_render).save(preview_paths["agreement"])
        Image.fromarray(
            np.concatenate([background, swir_overlay, rgb_overlay, hybrid_overlay, agreement_render], axis=1)
        ).save(preview_paths["contact"])

    depth_blocks: list[dict[str, float | int]] = []
    for start in range(0, swir.shape[0], args.depth_block_rows):
        stop = min(swir.shape[0], start + args.depth_block_rows)
        block_metrics = _metrics(swir[start:stop], rgb[start:stop])
        depth_blocks.append(
            {
                "start_row": start,
                "stop_row": stop,
                "swir_fraction": float(np.mean(swir[start:stop])),
                "rgb_fraction": float(np.mean(rgb[start:stop])),
                **block_metrics,
            }
        )
    disagreement = _metrics(swir, rgb)
    report = {
        "shape": list(swir.shape),
        "inputs": {
            "swir_mask": str(args.swir_mask),
            "rgb_mask": str(args.rgb_mask),
            "rgb_probability": str(args.rgb_probability),
            "registered_rgb": str(args.registered_rgb),
        },
        "input_fractions": {
            "swir": float(np.mean(swir)),
            "rgb": float(np.mean(rgb)),
        },
        "swir_rgb_agreement": disagreement,
        "swir_postprocessing": {
            "closing_size": args.closing_size,
            "maximum_hole_pixels": args.maximum_hole_pixels,
            "filled_hole_count": filled_hole_count,
            "filled_hole_pixels": filled_hole_pixels,
            "component_count": swir_components,
            "kept_component_count": swir_kept,
            "removed_component_pixels": swir_removed,
            "foreground_fraction": float(np.mean(swir_postprocessed)),
            "interior_pixel_fraction": _interior_pixel_fraction(swir_postprocessed),
        },
        "hybrid": {
            "rule": "SWIR postprocessed OR (RGB probability >= confidence AND within SWIR dilation)",
            "rgb_confidence": args.rgb_confidence,
            "rgb_expansion_pixels": args.rgb_expansion_pixels,
            "rgb_guided_addition_pixels": int(np.count_nonzero(rgb_guided_additions)),
            "filled_hole_count": hybrid_hole_count,
            "filled_hole_pixels": hybrid_hole_pixels,
            "component_count": hybrid_components,
            "kept_component_count": hybrid_kept,
            "removed_component_pixels": hybrid_removed,
            "foreground_fraction": float(np.mean(hybrid)),
            "interior_pixel_fraction": _interior_pixel_fraction(hybrid),
        },
        "depth_blocks": depth_blocks,
        "outputs": {key: str(value) for key, value in output_paths.items()},
        "previews": {key: str(value) for key, value in preview_paths.items()},
        "quality_gate": {
            "rgb_swir_dice_minimum": 0.80,
            "rgb_swir_dice_observed": disagreement["dice"],
            "rgb_model_accepted_as_primary": bool(disagreement["dice"] >= 0.80),
            "recommended_primary": (
                "registered_rgb_unet" if disagreement["dice"] >= 0.80 else "swir_postprocessed"
            ),
        },
        "review_required": True,
        "evidence_boundary": (
            "Agreement is between two automatic methods and is not ground-truth accuracy. "
            "The hybrid only permits local high-confidence RGB expansion around SWIR seeds."
        ),
    }
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
