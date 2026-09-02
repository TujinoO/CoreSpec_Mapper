from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from corespec_mapper.envi import EnviDataset, write_envi
from corespec_mapper.swir_mask_quality import (
    DepthCoverageRegion,
    analyse_depth_coverage,
    refine_swir_mask_spatially,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively refine a same-grid SWIR foreground mask and apply a "
            "multi-scale per-depth coverage-collapse quality gate."
        )
    )
    parser.add_argument("--swir-mask", type=Path, required=True)
    parser.add_argument("--swir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="SWIR_engineered")
    parser.add_argument("--registered-rgb", type=Path)
    parser.add_argument("--rgb-scale", type=int, default=2)
    parser.add_argument("--closing-size", type=int, default=3)
    parser.add_argument("--maximum-hole-pixels", type=int, default=125)
    parser.add_argument("--minimum-component-pixels", type=int, default=64)
    parser.add_argument("--block-scales", default="256,512")
    parser.add_argument("--neighborhood-blocks", type=int, default=3)
    parser.add_argument("--minimum-expected-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-drop-fraction", type=float, default=0.12)
    parser.add_argument("--maximum-drop-ratio", type=float, default=0.55)
    parser.add_argument("--minimum-robust-z", type=float, default=2.5)
    parser.add_argument("--preview-height", type=int, default=4096)
    parser.add_argument("--start-depth-m", type=float)
    parser.add_argument("--stop-depth-m", type=float)
    return parser.parse_args()


def _read_mask(path: Path) -> np.ndarray:
    with EnviDataset(path) as dataset:
        if dataset.info.bands != 1:
            raise ValueError(f"Expected a one-band mask: {path}")
        return np.asarray(dataset.read_rows(0, dataset.info.lines)[..., 0] > 0).copy()


def _parse_scales(value: str) -> tuple[int, ...]:
    scales = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not scales:
        raise ValueError("At least one block scale is required")
    return scales


def _depth_at(row: int, lines: int, start: float | None, stop: float | None) -> float | None:
    if start is None or stop is None:
        return None
    bounded = min(max(int(row), 0), max(lines - 1, 0))
    return float(start + bounded / max(lines - 1, 1) * (stop - start))


def _add_depths(record: dict, lines: int, start: float | None, stop: float | None) -> dict:
    result = dict(record)
    result["start_depth_m"] = _depth_at(int(record["start_row"]), lines, start, stop)
    result["stop_depth_m"] = _depth_at(int(record["stop_row"]), lines, start, stop)
    return result


def _stretch(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    lower, upper = np.percentile(values[finite], (1.0, 99.0))
    if upper <= lower:
        upper = lower + 1e-6
    scaled = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return np.nan_to_num(scaled * 255.0).round().astype(np.uint8)


def _rgb_preview(dataset: EnviDataset, rows: np.ndarray) -> np.ndarray:
    if dataset.info.interleave == "bil":
        return np.moveaxis(np.asarray(dataset._array[rows, :3, :]), 1, 2).astype(np.uint8)
    if dataset.info.interleave == "bip":
        return np.asarray(dataset._array[rows, :, :3], dtype=np.uint8)
    return np.moveaxis(np.asarray(dataset._array[:3, rows, :]), 0, 2).astype(np.uint8)


def _build_background(
    swir: EnviDataset,
    target_rows: np.ndarray,
    registered_rgb: Path | None,
    rgb_scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    if registered_rgb is None:
        wavelengths = swir.info.wavelengths_nm
        band = swir.info.bands // 2 if wavelengths is None else int(
            np.argmin(np.abs(wavelengths - 1600.0))
        )
        gray = _stretch(
            np.asarray(swir.read_rows(0, swir.info.lines, bands=(band,))[target_rows, :, 0])
        )
        return np.repeat(gray[..., None], 3, axis=2), np.arange(swir.info.samples)
    with EnviDataset(registered_rgb) as rgb:
        expected = (swir.info.lines * rgb_scale, swir.info.samples * rgb_scale)
        if (rgb.info.lines, rgb.info.samples) != expected:
            raise ValueError(
                f"Registered RGB shape {(rgb.info.lines, rgb.info.samples)} does not match {expected}"
            )
        rgb_rows = np.minimum(target_rows * rgb_scale, rgb.info.lines - 1)
        background = _rgb_preview(rgb, rgb_rows)
        columns = np.minimum(
            (np.arange(rgb.info.samples, dtype=np.float64) / rgb_scale).astype(np.int64),
            swir.info.samples - 1,
        )
        return background, columns


def _overlay(background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = background.astype(np.float32)
    tint = np.zeros_like(result)
    tint[..., 1] = 255.0
    alpha = (0.34 * mask.astype(np.float32))[..., None]
    return np.clip(result * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def _mark_regions(
    image: np.ndarray,
    regions: list[DepthCoverageRegion],
    target_rows: np.ndarray,
) -> np.ndarray:
    result = image.copy()
    for region in regions:
        selected = (target_rows >= region.start_row) & (target_rows < region.stop_row)
        color = (
            (255, 32, 32)
            if region.severity == "critical"
            else (255, 170, 0)
            if region.severity == "review"
            else (50, 130, 230)
        )
        result[selected, :4] = color
        result[selected, -4:] = color
    return result


def _coverage_plot(
    mask: np.ndarray,
    regions: list[DepthCoverageRegion],
    output: Path,
    start_depth: float | None,
    stop_depth: float | None,
) -> None:
    width, height = 1500, 760
    left, top, right, bottom = 90, 52, 35, 80
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    plot_w, plot_h = width - left - right, height - top - bottom
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(40, 40, 40), width=2)
    for value in np.linspace(0.0, 1.0, 6):
        y = top + plot_h - int(value * plot_h)
        draw.line((left, y, left + plot_w, y), fill=(225, 225, 225), width=1)
        draw.text((15, y - 7), f"{value:.1f}", fill=(50, 50, 50))
    row_fraction = np.mean(mask, axis=1)
    smoothed = ndimage.uniform_filter1d(row_fraction, size=min(128, mask.shape[0]), mode="nearest")
    expected = ndimage.median_filter(smoothed, size=min(2049, mask.shape[0] | 1), mode="nearest")

    def x_at(row: int) -> int:
        return left + int(row / max(mask.shape[0] - 1, 1) * plot_w)

    def y_at(value: float) -> int:
        return top + plot_h - int(np.clip(value, 0.0, 1.0) * plot_h)

    for region in regions:
        fill = (
            (255, 220, 220)
            if region.severity == "critical"
            else (255, 242, 205)
            if region.severity == "review"
            else (225, 238, 255)
        )
        draw.rectangle(
            (x_at(region.start_row), top, x_at(region.stop_row), top + plot_h), fill=fill
        )
    step = max(1, mask.shape[0] // plot_w)
    rows = np.arange(0, mask.shape[0], step, dtype=np.int64)
    draw.line([(x_at(int(row)), y_at(float(expected[row]))) for row in rows], fill=(70, 120, 210), width=2)
    draw.line([(x_at(int(row)), y_at(float(smoothed[row]))) for row in rows], fill=(20, 150, 60), width=3)
    draw.text((left, 16), "SWIR foreground coverage by depth (128-row smoothing)", fill=(20, 20, 20))
    draw.text((left + 500, 16), "green: observed   blue: local median   red/orange/blue: critical/review/structural gap", fill=(50, 50, 50))
    if start_depth is not None and stop_depth is not None:
        draw.text((left, top + plot_h + 28), f"{start_depth:.2f} m", fill=(30, 30, 30))
        draw.text((left + plot_w - 55, top + plot_h + 28), f"{stop_depth:.2f} m", fill=(30, 30, 30))
        axis_label = "Approximate linear depth; verify against acquisition metadata"
    else:
        draw.text((left, top + plot_h + 28), "row 0", fill=(30, 30, 30))
        draw.text((left + plot_w - 90, top + plot_h + 28), f"row {mask.shape[0]}", fill=(30, 30, 30))
        axis_label = "SWIR image rows"
    draw.text((left + plot_w // 2 - 100, top + plot_h + 52), axis_label, fill=(30, 30, 30))
    canvas.save(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    args = _parse_args()
    started = monotonic()
    scales = _parse_scales(args.block_scales)
    if (args.start_depth_m is None) != (args.stop_depth_m is None):
        raise ValueError("start-depth-m and stop-depth-m must be provided together")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mask": args.output_dir / f"{args.prefix}.dat",
        "audit": args.output_dir / f"{args.prefix}_audit.json",
        "mask_preview": args.output_dir / f"{args.prefix}_mask_preview.png",
        "overlay": args.output_dir / f"{args.prefix}_overlay.png",
        "coverage_profile": args.output_dir / f"{args.prefix}_coverage_profile.png",
    }
    expected_paths = [*paths.values(), paths["mask"].with_suffix(".hdr")]
    existing = [path for path in expected_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(map(str, existing)))

    source_mask = _read_mask(args.swir_mask)
    raw_blocks, raw_regions = analyse_depth_coverage(
        source_mask,
        block_scales=scales,
        neighborhood_blocks=args.neighborhood_blocks,
        minimum_expected_fraction=args.minimum_expected_fraction,
        minimum_drop_fraction=args.minimum_drop_fraction,
        maximum_drop_ratio=args.maximum_drop_ratio,
        minimum_robust_z=args.minimum_robust_z,
    )
    refined, spatial_audit = refine_swir_mask_spatially(
        source_mask,
        closing_size=args.closing_size,
        maximum_hole_pixels=args.maximum_hole_pixels,
        minimum_component_pixels=args.minimum_component_pixels,
    )
    blocks, regions = analyse_depth_coverage(
        refined,
        block_scales=scales,
        neighborhood_blocks=args.neighborhood_blocks,
        minimum_expected_fraction=args.minimum_expected_fraction,
        minimum_drop_fraction=args.minimum_drop_fraction,
        maximum_drop_ratio=args.maximum_drop_ratio,
        minimum_robust_z=args.minimum_robust_z,
    )
    critical = [region for region in regions if region.severity == "critical"]
    review = [region for region in regions if region.severity == "review"]
    informational = [region for region in regions if region.severity == "info"]
    if critical:
        gate_status = "blocked"
    elif review:
        gate_status = "review"
    else:
        gate_status = "passed"

    with EnviDataset(args.swir) as swir:
        if source_mask.shape != (swir.info.lines, swir.info.samples):
            raise ValueError("SWIR mask and SWIR cube dimensions do not match")
        write_envi(
            refined.astype(np.uint8),
            paths["mask"],
            class_names=("background", "core foreground"),
            description="SWIR foreground with conservative spatial cleanup and depth coverage QC",
            metadata={
                "corespec source mask": str(args.swir_mask),
                "corespec depth coverage gate": gate_status,
                "corespec coverage block scales rows": list(scales),
            },
        )
        preview_height = min(max(1, args.preview_height), swir.info.lines)
        target_rows = np.linspace(0, swir.info.lines - 1, preview_height).round().astype(np.int64)
        background, target_columns = _build_background(
            swir, target_rows, args.registered_rgb, args.rgb_scale
        )
    preview_mask = refined[np.ix_(target_rows, target_columns)]
    mask_render = np.repeat((preview_mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    overlay = _overlay(background, preview_mask)
    Image.fromarray(_mark_regions(mask_render, regions, target_rows)).save(paths["mask_preview"])
    Image.fromarray(_mark_regions(overlay, regions, target_rows)).save(paths["overlay"])
    _coverage_plot(
        refined, regions, paths["coverage_profile"], args.start_depth_m, args.stop_depth_m
    )

    report = {
        "status": gate_status,
        "source_swir": str(args.swir),
        "source_mask": str(args.swir_mask),
        "shape": list(refined.shape),
        "method": {
            "name": "conservative_swir_spatial_refinement_plus_multiscale_depth_gate",
            "spectral_region_expansion": False,
            "explanation": (
                "Spatial cleanup can close narrow cracks, fill bounded small holes, and remove small "
                "components. The depth gate detects persistent internal coverage collapses at two "
                "resolutions; it does not automatically turn spectrally unsupported background into core."
            ),
        },
        "spatial_refinement": spatial_audit.to_dict(),
        "depth_gate_parameters": {
            "block_scales_rows": list(scales),
            "neighborhood_blocks": args.neighborhood_blocks,
            "minimum_expected_fraction": args.minimum_expected_fraction,
            "minimum_drop_fraction": args.minimum_drop_fraction,
            "maximum_drop_ratio": args.maximum_drop_ratio,
            "minimum_robust_z": args.minimum_robust_z,
            "critical_definition": "overlapping anomaly at two or more block scales",
            "edge_blank_policy": "leading/trailing low-coverage intervals are not auto-flagged without two-sided support",
        },
        "before_refinement": {
            "foreground_fraction": float(np.mean(source_mask)),
            "anomalous_block_count": sum(item.anomaly for item in raw_blocks),
            "regions": [
                _add_depths(item.to_dict(), refined.shape[0], args.start_depth_m, args.stop_depth_m)
                for item in raw_regions
            ],
        },
        "after_refinement": {
            "foreground_fraction": float(np.mean(refined)),
            "anomalous_block_count": sum(item.anomaly for item in blocks),
            "critical_region_count": len(critical),
            "review_region_count": len(review),
            "structural_gap_region_count": len(informational),
            "regions": [
                _add_depths(item.to_dict(), refined.shape[0], args.start_depth_m, args.stop_depth_m)
                for item in regions
            ],
            "blocks": [
                _add_depths(item.to_dict(), refined.shape[0], args.start_depth_m, args.stop_depth_m)
                for item in blocks
            ],
        },
        "quality_gate": {
            "status": gate_status,
            "mapping_allowed_automatically": gate_status == "passed",
            "action": (
                "inspect_critical_depth_regions_and_supply_manual_or_secondary_evidence"
                if critical
                else "inspect_review_regions_before_approval"
                if review
                else "depth_coverage_stability_gate_passed"
            ),
            "non_blocking_structural_gap_count": len(informational),
        },
        "depth_axis": {
            "start_depth_m": args.start_depth_m,
            "stop_depth_m": args.stop_depth_m,
            "mapping": "linear_approximation" if args.start_depth_m is not None else None,
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "mask_sha256": _sha256(paths["mask"]),
        "elapsed_seconds": monotonic() - started,
        "manual_review_required": gate_status != "passed",
        "evidence_boundary": (
            "Coverage stability is an engineering completeness check, not pixel-level accuracy against "
            "manual ground truth. A stable but consistently biased mask still requires representative visual QA."
        ),
    }
    paths["audit"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": gate_status,
        "foreground_fraction": report["after_refinement"]["foreground_fraction"],
        "critical_region_count": len(critical),
        "review_region_count": len(review),
        "structural_gap_region_count": len(informational),
        "mask": str(paths["mask"]),
        "audit": str(paths["audit"]),
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
