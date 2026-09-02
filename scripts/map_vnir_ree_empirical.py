from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal import find_peaks

from corespec_mapper.algorithms import directional_stripe_mask
from corespec_mapper.empirical_vnir import (
    candidate_neighborhood_support,
    empirical_absorption_features,
    local_radiometric_stability_mask,
    policy_candidate,
    read_envi_ascii_spectrum,
    resample_empirical_spectrum,
    seeded_candidate_region_grow,
    spectral_angles,
    time_normalized_mask,
)
from corespec_mapper.envi import EnviDataset, write_envi
from corespec_mapper.masking import _component_cleanup, build_material_mask


POLICIES = {
    "conservative": {"angle": 0.36, "strength_ratio": 0.40, "margin": 0.025},
    "balanced": {"angle": 0.45, "strength_ratio": 0.30, "margin": 0.018},
    "sensitive": {"angle": 0.55, "strength_ratio": 0.20, "margin": 0.010},
}

RELAXED_GEO_POLICIES = {
    "conservative": {"angle": 0.45, "strength_ratio": 0.30, "margin": 0.018},
    "balanced": {"angle": 0.58, "strength_ratio": 0.16, "margin": 0.008},
    "sensitive": {"angle": 0.68, "strength_ratio": 0.08, "margin": 0.003},
}

# High-recall exploration ladder.  The seed remains substantially narrower
# than the growth domain; permissive pixels are admitted only when connected
# to an accepted seed.  Balanced/sensitive growth angles exceed the observed
# separation of the two user references and must therefore remain review-only.
AGGRESSIVE_GROW_POLICIES = {
    "conservative": {
        "angle": 0.58, "strength_ratio": 0.12, "margin": 0.008,
        "growth_angle": 0.72, "growth_strength_ratio": 0.04,
    },
    "balanced": {
        "angle": 0.68, "strength_ratio": 0.05, "margin": 0.002,
        "growth_angle": 0.84, "growth_strength_ratio": 0.012,
    },
    "sensitive": {
        "angle": 0.72, "strength_ratio": 0.025, "margin": 0.0005,
        "growth_angle": 0.95, "growth_strength_ratio": 0.003,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map user-provided VNIR empirical REE spectral signatures without changing the SWIR expert."
    )
    parser.add_argument("--vnir", type=Path, required=True)
    parser.add_argument("--swir", type=Path, required=True)
    parser.add_argument("--swir-mask", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="ZK5511_VNIR_REE")
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--minimum-component-pixels", type=int, default=6)
    parser.add_argument("--edge-guard-pixels", type=int, default=2)
    parser.add_argument("--vnir-support-erosion-pixels", type=int, default=1)
    parser.add_argument("--local-window", type=int, default=5)
    parser.add_argument("--minimum-local-support", type=float, default=0.80)
    parser.add_argument("--maximum-local-cv", type=float, default=0.30)
    parser.add_argument("--maximum-log-gradient", type=float, default=0.25)
    parser.add_argument("--minimum-candidate-neighbors", type=int, default=4)
    parser.add_argument("--growth-minimum-candidate-neighbors", type=int, default=2)
    parser.add_argument("--preview-height", type=int, default=4096)
    parser.add_argument(
        "--policy-profile",
        choices=("standard", "relaxed-geo", "aggressive-grow"),
        default="standard",
        help="Use the baseline, relaxed, or connectivity-constrained high-recall review ladder.",
    )
    return parser.parse_args()


def _read_mask(path: Path) -> np.ndarray:
    with EnviDataset(path) as dataset:
        if dataset.info.bands != 1:
            raise ValueError(f"Expected one-band mask: {path}")
        return np.asarray(dataset.read_rows(0, dataset.info.lines)[..., 0] > 0).copy()


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _agreement(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    intersection = int(np.count_nonzero(first & second))
    first_only = int(np.count_nonzero(first & ~second))
    second_only = int(np.count_nonzero(~first & second))
    union = intersection + first_only + second_only
    return {
        "intersection_pixels": intersection,
        "first_only_pixels": first_only,
        "second_only_pixels": second_only,
        "iou": intersection / max(union, 1),
        "dice": 2 * intersection / max(2 * intersection + first_only + second_only, 1),
        "agreement_fraction": float(np.mean(first == second)),
    }


def _metadata_float(dataset: EnviDataset, key: str) -> float | None:
    value = dataset.info.metadata.get(key.casefold())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _depth_range(dataset: EnviDataset) -> tuple[float, float] | None:
    value = str(dataset.info.metadata.get("sample deep", ""))
    parts = value.split("_")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _stretch(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    lower, upper = np.percentile(values[finite], (1.0, 99.0))
    if upper <= lower:
        upper = lower + 1e-6
    return np.nan_to_num(np.clip((values - lower) / (upper - lower), 0, 1) * 255).astype(np.uint8)


def _overlay(background: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = background.astype(np.float32)
    tint = np.empty_like(result)
    tint[:] = color
    alpha = (0.50 * np.asarray(mask, dtype=np.float32))[..., None]
    return np.clip(result * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(panel.shape[1] - 1, 300), 27), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(image)


def _write_spectral_library(
    data_path: Path,
    wavelengths_nm: np.ndarray,
    spectra: np.ndarray,
    names: list[str],
) -> None:
    values = np.asarray(spectra, dtype="<f4")
    values[np.newaxis, ...].tofile(data_path)
    header = data_path.with_suffix(".hdr")
    wavelength_text = ", ".join(f"{value:.9f}" for value in wavelengths_nm)
    names_text = ", ".join(names)
    header.write_text(
        "ENVI\n"
        "description = {User-provided empirical VNIR REE reference ensemble; not phase-confirmed}\n"
        f"samples = {wavelengths_nm.size}\n"
        f"lines = {values.shape[0]}\n"
        "bands = 1\n"
        "header offset = 0\n"
        "file type = ENVI Spectral Library\n"
        "data type = 4\n"
        "interleave = bsq\n"
        "byte order = 0\n"
        "wavelength units = Nanometers\n"
        f"wavelength = {{{wavelength_text}}}\n"
        f"spectra names = {{{names_text}}}\n",
        encoding="utf-8",
    )


def _reference_peaks(wavelengths: np.ndarray, features: np.ndarray) -> list[dict[str, float]]:
    peaks, properties = find_peaks(features, prominence=max(0.002, float(np.nanmax(features)) * 0.08), distance=4)
    if not peaks.size:
        return []
    order = np.argsort(properties["prominences"])[::-1][:10]
    records = [
        {
            "center_nm": float(wavelengths[peaks[index]]),
            "relative_absorption": float(features[peaks[index]]),
            "prominence": float(properties["prominences"][index]),
        }
        for index in order
        if features[peaks[index]] > 0.0
    ]
    return records


def _write_reference_plot(
    path: Path,
    wavelengths: np.ndarray,
    spectra: np.ndarray,
    feature_wavelengths: np.ndarray,
    features: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False, constrained_layout=True)
    colors = ("#c43c39", "#674ea7", "#2f8f5b", "#d98b26")
    for index in range(spectra.shape[0]):
        axes[0].plot(wavelengths, spectra[index], color=colors[index % len(colors)], label=f"REE empirical ref {index + 1}")
        axes[1].plot(feature_wavelengths, features[index], color=colors[index % len(colors)], label=f"ref {index + 1}")
    axes[0].set_ylabel("Reflectance")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_title("User-provided empirical VNIR references")
    axes[1].set_ylabel("Narrow absorption feature")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_title("Feature representation used for matching (675-925 nm)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _candidate_components(
    mask: np.ndarray,
    depth: tuple[float, float] | None,
) -> tuple[np.ndarray, list[dict[str, float | int | None]]]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndimage.find_objects(labels)
    records: list[dict[str, float | int | None]] = []
    for label_id in range(1, count + 1):
        location = objects[label_id - 1]
        if location is None:
            continue
        selected = labels[location] == label_id
        rows, columns = np.nonzero(selected)
        row_start, column_start = location[0].start, location[1].start
        centroid_row = float(row_start + np.mean(rows))
        centroid_column = float(column_start + np.mean(columns))
        centroid_depth = None if depth is None else depth[0] + centroid_row / max(mask.shape[0] - 1, 1) * (depth[1] - depth[0])
        records.append({
            "label": label_id,
            "pixels": int(np.count_nonzero(selected)),
            "start_row": int(location[0].start),
            "stop_row": int(location[0].stop),
            "start_column": int(location[1].start),
            "stop_column": int(location[1].stop),
            "centroid_row": centroid_row,
            "centroid_column": centroid_column,
            "centroid_depth_m": centroid_depth,
        })
    records.sort(key=lambda item: int(item["pixels"]), reverse=True)
    return labels, records


def _write_candidate_contact(
    path: Path,
    gray: np.ndarray,
    labels: np.ndarray,
    records: list[dict[str, float | int | None]],
    maximum: int = 12,
) -> None:
    tile_width, tile_height = 620, 210
    selected_records = records[:maximum]
    rows = max(1, (len(selected_records) + 1) // 2)
    canvas = Image.new("RGB", (tile_width * 2, tile_height * rows), "black")
    for index, record in enumerate(selected_records):
        pad_y, pad_x = 24, 18
        y0 = max(0, int(record["start_row"]) - pad_y)
        y1 = min(gray.shape[0], int(record["stop_row"]) + pad_y)
        x0 = max(0, int(record["start_column"]) - pad_x)
        x1 = min(gray.shape[1], int(record["stop_column"]) + pad_x)
        local_gray = gray[y0:y1, x0:x1]
        background = np.repeat(local_gray[..., None], 3, axis=2)
        local_mask = labels[y0:y1, x0:x1] == int(record["label"])
        overlay = _overlay(background, local_mask, (255, 70, 30))
        source_image = Image.fromarray(background).resize((300, 170), Image.Resampling.BILINEAR)
        overlay_image = Image.fromarray(overlay).resize((300, 170), Image.Resampling.NEAREST)
        tile = Image.new("RGB", (tile_width, tile_height), "black")
        tile.paste(source_image, (5, 35))
        tile.paste(overlay_image, (315, 35))
        draw = ImageDraw.Draw(tile)
        depth_text = "n/a" if record["centroid_depth_m"] is None else f"{float(record['centroid_depth_m']):.2f} m"
        draw.text(
            (8, 8),
            f"rank {index + 1} | {int(record['pixels'])} px | row {int(record['centroid_row'])} | {depth_text}",
            fill=(255, 255, 255),
        )
        canvas.paste(tile, ((index % 2) * tile_width, (index // 2) * tile_height))
    canvas.save(path)


def main() -> int:
    args = _parse_args()
    if args.policy_profile == "relaxed-geo":
        policies = RELAXED_GEO_POLICIES
    elif args.policy_profile == "aggressive-grow":
        policies = AGGRESSIVE_GROW_POLICIES
    else:
        policies = POLICIES
    if len(args.reference) < 2:
        raise ValueError("At least two empirical references are required for ensemble matching")
    if args.chunk_rows < 1 or args.minimum_component_pixels < 1:
        raise ValueError("chunk rows and component size must be positive")
    started = monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mapped_mask": args.output_dir / f"{args.prefix}_mapped_SWIR_mask.dat",
        "independent_vnir_mask": args.output_dir / f"{args.prefix}_independent_VNIR_mask.dat",
        "analysis_mask": args.output_dir / f"{args.prefix}_edge_guarded_analysis_mask.dat",
        "neighborhood_guard": args.output_dir / f"{args.prefix}_neighborhood_guard_mask.dat",
        "reference_library": args.output_dir / f"{args.prefix}_empirical_references.sli",
        "reference_plot": args.output_dir / f"{args.prefix}_reference_QA.png",
        "angle_stack": args.output_dir / f"{args.prefix}_reference_angles.dat",
        "best_angle": args.output_dir / f"{args.prefix}_best_angle.dat",
        "feature_strength": args.output_dir / f"{args.prefix}_feature_strength.dat",
        "balanced_confidence": args.output_dir / f"{args.prefix}_balanced_confidence.dat",
        "preview": args.output_dir / f"{args.prefix}_policy_contact.png",
        "registration_contact": args.output_dir / f"{args.prefix}_registration_contact.png",
        "counts": args.output_dir / f"{args.prefix}_counts.csv",
        "depth_blocks": args.output_dir / f"{args.prefix}_depth_blocks.csv",
        "candidate_components": args.output_dir / f"{args.prefix}_balanced_components.csv",
        "candidate_qa": args.output_dir / f"{args.prefix}_balanced_candidate_QA.png",
        "audit": args.output_dir / f"{args.prefix}_audit.json",
    }
    for policy in policies:
        paths[f"family_{policy}"] = args.output_dir / f"{args.prefix}_family_{policy}.dat"
        paths[f"classes_{policy}"] = args.output_dir / f"{args.prefix}_classes_{policy}.dat"
    expected = list(paths.values())
    for key in ("mapped_mask", "independent_vnir_mask", "analysis_mask", "neighborhood_guard", "reference_library", "angle_stack", "best_angle", "feature_strength", "balanced_confidence"):
        expected.append(paths[key].with_suffix(".hdr"))
    for policy in policies:
        expected.extend((paths[f"family_{policy}"].with_suffix(".hdr"), paths[f"classes_{policy}"].with_suffix(".hdr")))
    existing = [path for path in expected if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(map(str, existing)))

    source_mask = _read_mask(args.swir_mask)
    references = [read_envi_ascii_spectrum(path) for path in args.reference]
    with EnviDataset(args.vnir) as vnir, EnviDataset(args.swir) as swir:
        if source_mask.shape != (swir.info.lines, swir.info.samples):
            raise ValueError("SWIR mask and SWIR cube dimensions differ")
        if swir.info.samples != vnir.info.samples:
            raise ValueError("VNIR and SWIR sample counts differ")
        timing_keys = ("start acquisition time", "end acquisition time", "acquisition duration (sec)")
        timing = {
            key: {"vnir": _metadata_float(vnir, key), "swir": _metadata_float(swir, key)}
            for key in timing_keys
        }
        timing_match = all(
            record["vnir"] is not None
            and record["swir"] is not None
            and abs(record["vnir"] - record["swir"]) <= 0.05
            for record in timing.values()
        )
        if not timing_match:
            raise ValueError("VNIR/SWIR acquisition timing does not support normalized line mapping")
        mapped_mask, source_rows = time_normalized_mask(source_mask, vnir.info.lines)
        analysis_mask = ndimage.binary_erosion(
            mapped_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=max(0, int(args.edge_guard_pixels)),
            border_value=0,
        ) if args.edge_guard_pixels else mapped_mask.copy()
        wavelengths = vnir.info.wavelengths_nm
        if wavelengths is None:
            raise ValueError("VNIR image requires wavelengths")
        analysis = (wavelengths >= 650.0) & (wavelengths <= 950.0)
        scoring = (wavelengths[analysis] >= 675.0) & (wavelengths[analysis] <= 925.0)
        selected_bands = tuple(np.flatnonzero(analysis).tolist())
        reference_resampled = np.vstack([
            resample_empirical_spectrum(reference, wavelengths) for reference in references
        ])
        reference_analysis = reference_resampled[:, analysis]
        if not np.all(np.isfinite(reference_analysis) & (reference_analysis > 0)):
            raise ValueError("References contain invalid values in the 650-950 nm analysis interval")
        reference_features_full = empirical_absorption_features(reference_analysis)
        reference_features = reference_features_full[:, scoring]
        reference_strength = np.sqrt(np.mean(reference_features * reference_features, axis=1))
        if np.any(reference_strength < 0.002):
            raise ValueError("At least one reference has insufficient narrow-feature strength")
        reference_angles = spectral_angles(reference_features, reference_features)

        baseline = build_material_mask(
            vnir,
            bands=tuple(np.linspace(selected_bands[0], selected_bands[-1], 9).round().astype(int)),
            chunk_rows=args.chunk_rows,
            minimum_component_pixels=64,
            minimum_mask_fraction=0.05,
            maximum_mask_fraction=0.85,
            source_override="independent_vnir_reflectance_material_mask",
            strict=False,
        )
        mask_agreement = _agreement(mapped_mask, baseline.mask)
        row_profile_correlation = float(np.corrcoef(
            ndimage.uniform_filter1d(mapped_mask.mean(axis=1), 128),
            ndimage.uniform_filter1d(baseline.mask.mean(axis=1), 128),
        )[0, 1])

        band850 = int(np.argmin(np.abs(wavelengths - 850.0)))
        albedo850 = np.asarray(
            vnir.read_rows(0, vnir.info.lines, bands=(band850,))[..., 0], dtype=np.float32
        )
        radiometrically_stable, local_support, local_cv, log_gradient = local_radiometric_stability_mask(
            albedo850,
            analysis_mask,
            window_size=args.local_window,
            minimum_valid_fraction=args.minimum_local_support,
            maximum_coefficient_of_variation=args.maximum_local_cv,
            maximum_log_gradient=args.maximum_log_gradient,
        )
        vnir_support = ndimage.binary_erosion(
            baseline.mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=max(0, int(args.vnir_support_erosion_pixels)),
            border_value=0,
        ) if args.vnir_support_erosion_pixels else baseline.mask.copy()
        neighborhood_guard = radiometrically_stable & vnir_support

        shape = (vnir.info.lines, vnir.info.samples)
        angles = np.full((len(references), *shape), np.nan, dtype=np.float32)
        strength = np.full(shape, np.nan, dtype=np.float32)
        spectrally_valid = np.zeros(shape, dtype=bool)
        for row_start, row_stop, cube in vnir.iter_rows(chunk_rows=args.chunk_rows, bands=selected_bands):
            local_mask = analysis_mask[row_start:row_stop]
            flat = np.asarray(cube, dtype=np.float64).reshape(-1, len(selected_bands))
            chosen = local_mask.ravel() & np.all(np.isfinite(flat) & (flat > 0.0), axis=1)
            if not np.any(chosen):
                continue
            features = empirical_absorption_features(flat[chosen])[:, scoring]
            current_angles = spectral_angles(features, reference_features)
            current_strength = np.sqrt(np.mean(features * features, axis=1))
            local_valid = spectrally_valid[row_start:row_stop].ravel()
            local_strength = strength[row_start:row_stop].ravel()
            local_valid[chosen] = np.all(np.isfinite(current_angles), axis=1)
            local_strength[chosen] = current_strength.astype(np.float32)
            for index in range(len(references)):
                local_angle = angles[index, row_start:row_stop].ravel()
                local_angle[chosen] = current_angles[:, index].astype(np.float32)

        safe = np.where(np.isfinite(angles), angles, np.inf)
        best_index = np.argmin(safe, axis=0)
        best_angle = np.min(safe, axis=0).astype(np.float32)
        best_angle[~np.isfinite(best_angle) | np.isinf(best_angle)] = np.nan
        sorted_angles = np.sort(safe, axis=0)
        margin = np.full(shape, np.nan, dtype=np.float32)
        finite_pair = np.isfinite(sorted_angles[0]) & np.isfinite(sorted_angles[1])
        margin[finite_pair] = (sorted_angles[1, finite_pair] - sorted_angles[0, finite_pair]).astype(np.float32)
        minimum_reference_strength = float(np.min(reference_strength))
        maximum_reference_strength = float(np.max(reference_strength))
        raw_masks: dict[str, np.ndarray] = {}
        family_masks: dict[str, np.ndarray] = {}
        class_maps: dict[str, np.ndarray] = {}
        policy_audits: dict[str, dict] = {}
        sensitive_settings = policies["sensitive"]
        stripe_angle = float(sensitive_settings.get("growth_angle", sensitive_settings["angle"]))
        stripe_strength_ratio = float(
            sensitive_settings.get("growth_strength_ratio", sensitive_settings["strength_ratio"])
        )
        sensitive_before_guard = policy_candidate(
            best_angle,
            strength,
            analysis_mask & spectrally_valid,
            maximum_angle_rad=stripe_angle,
            minimum_strength=minimum_reference_strength * stripe_strength_ratio,
            maximum_strength=maximum_reference_strength * 5.0,
        )
        sensitive_guarded = sensitive_before_guard & neighborhood_guard
        sensitive_raw, _ = candidate_neighborhood_support(
            sensitive_guarded,
            minimum_pixels=(
                args.growth_minimum_candidate_neighbors
                if "growth_angle" in sensitive_settings
                else args.minimum_candidate_neighbors
            ),
        )
        stripe_risk = directional_stripe_mask(
            sensitive_raw,
            analysis_mask,
            vertical_window=61,
            min_vertical_density=0.15,
            column_ratio=1.8,
            column_excess=0.025,
            max_lateral_support=2,
        )
        for policy, settings in policies.items():
            before_guard = policy_candidate(
                best_angle,
                strength,
                analysis_mask & spectrally_valid,
                maximum_angle_rad=settings["angle"],
                minimum_strength=minimum_reference_strength * settings["strength_ratio"],
                maximum_strength=maximum_reference_strength * 5.0,
            )
            guarded = before_guard & neighborhood_guard
            raw, candidate_support_counts = candidate_neighborhood_support(
                guarded, minimum_pixels=args.minimum_candidate_neighbors
            )
            growth_angle = settings.get("growth_angle")
            growth_strength_ratio = settings.get("growth_strength_ratio")
            growth_before_guard = None
            growth_raw = None
            if growth_angle is not None and growth_strength_ratio is not None:
                growth_before_guard = policy_candidate(
                    best_angle,
                    strength,
                    analysis_mask & spectrally_valid,
                    maximum_angle_rad=float(growth_angle),
                    minimum_strength=minimum_reference_strength * float(growth_strength_ratio),
                    maximum_strength=maximum_reference_strength * 5.0,
                )
                growth_guarded = growth_before_guard & neighborhood_guard
                growth_raw, _ = candidate_neighborhood_support(
                    growth_guarded,
                    minimum_pixels=args.growth_minimum_candidate_neighbors,
                )
                without_stripes = seeded_candidate_region_grow(
                    raw & ~stripe_risk,
                    growth_raw & ~stripe_risk,
                )
            else:
                without_stripes = raw & ~stripe_risk
            cleaned, component_count, kept_count, removed_pixels = _component_cleanup(
                without_stripes, args.minimum_component_pixels
            )
            classes = np.zeros(shape, dtype=np.uint8)
            confident_class = cleaned & (margin >= settings["margin"])
            classes[confident_class] = (best_index[confident_class] + 1).astype(np.uint8)
            raw_masks[policy] = raw
            family_masks[policy] = cleaned
            class_maps[policy] = classes
            selected = int(np.count_nonzero(cleaned))
            policy_audits[policy] = {
                **settings,
                "minimum_strength": minimum_reference_strength * settings["strength_ratio"],
                "maximum_strength": maximum_reference_strength * 5.0,
                "raw_pixels": int(np.count_nonzero(raw)),
                "raw_pixels_before_neighborhood_guard": int(np.count_nonzero(before_guard)),
                "neighborhood_guard_pixels_removed": int(np.count_nonzero(before_guard & ~neighborhood_guard)),
                "candidate_neighbor_pixels_removed": int(np.count_nonzero(guarded & ~raw)),
                "minimum_candidate_neighbors_in_3x3": int(args.minimum_candidate_neighbors),
                "growth_enabled": growth_raw is not None,
                "growth_angle": None if growth_angle is None else float(growth_angle),
                "growth_strength_ratio": (
                    None if growth_strength_ratio is None else float(growth_strength_ratio)
                ),
                "growth_minimum_candidate_neighbors_in_3x3": (
                    None if growth_raw is None else int(args.growth_minimum_candidate_neighbors)
                ),
                "growth_pixels_before_neighborhood_guard": (
                    None if growth_before_guard is None else int(np.count_nonzero(growth_before_guard))
                ),
                "growth_pixels_after_candidate_support": (
                    None if growth_raw is None else int(np.count_nonzero(growth_raw))
                ),
                "seed_pixels_before_region_growth": int(np.count_nonzero(raw & ~stripe_risk)),
                "pixels_after_seeded_region_growth": int(np.count_nonzero(without_stripes)),
                "stripe_pixels_removed": int(np.count_nonzero(raw & stripe_risk)),
                "component_count": component_count,
                "kept_component_count": kept_count,
                "small_component_pixels_removed": removed_pixels,
                "family_pixels": selected,
                "fraction_of_foreground": selected / max(int(np.count_nonzero(mapped_mask)), 1),
                "classified_reference_1_pixels": int(np.count_nonzero(classes == 1)),
                "classified_reference_2_pixels": int(np.count_nonzero(classes == 2)),
                "ambiguous_family_pixels": int(np.count_nonzero(cleaned & (classes == 0))),
                "maximum_column_fraction": float(np.max(np.sum(cleaned, axis=0) / np.maximum(np.sum(mapped_mask, axis=0), 1))),
            }

        write_envi(mapped_mask.astype(np.uint8), paths["mapped_mask"], class_names=("background", "core foreground"), description="SWIR foreground transferred to synchronized VNIR grid by normalized acquisition time")
        write_envi(baseline.mask.astype(np.uint8), paths["independent_vnir_mask"], class_names=("background", "VNIR material proxy"), description="Independent VNIR reflectance foreground used only for registration QA")
        write_envi(analysis_mask.astype(np.uint8), paths["analysis_mask"], class_names=("background or edge guard", "interior analysis foreground"), description="Time-mapped SWIR foreground after VNIR empirical-mapping edge guard")
        write_envi(neighborhood_guard.astype(np.uint8), paths["neighborhood_guard"], class_names=("rejected", "stable VNIR interior"), description="VNIR support and scale-invariant local radiometric stability guard")
        _write_spectral_library(paths["reference_library"], wavelengths, reference_resampled, [f"REE empirical reference {index + 1}" for index in range(len(references))])
        _write_reference_plot(paths["reference_plot"], wavelengths, reference_resampled, wavelengths[analysis][scoring], reference_features)
        write_envi(angles, paths["angle_stack"], band_names=[f"REE empirical reference {index + 1} angle rad" for index in range(len(references))], description="VNIR empirical-reference spectral angles; lower is more similar")
        write_envi(best_angle, paths["best_angle"], description="Minimum empirical REE reference angle in radians")
        write_envi(strength, paths["feature_strength"], description="VNIR narrow absorption feature RMS strength")
        balanced_confidence = np.zeros(shape, dtype=np.float32)
        balanced_selected = family_masks["balanced"]
        balanced_confidence[balanced_selected] = np.clip(1.0 - best_angle[balanced_selected] / policies["balanced"]["angle"], 0.0, 1.0)
        write_envi(balanced_confidence, paths["balanced_confidence"], description="Balanced empirical REE similarity confidence; engineering score, not probability")
        for policy in policies:
            write_envi(family_masks[policy].astype(np.uint8), paths[f"family_{policy}"], class_names=("background", "REE empirical spectral-family anomaly"), description=f"{policy} VNIR empirical REE family anomaly")
            write_envi(class_maps[policy], paths[f"classes_{policy}"], class_names=("unclassified or ambiguous", "REE empirical reference 1", "REE empirical reference 2"), class_lookup=(0, 0, 0, 220, 50, 45, 145, 70, 190), description=f"{policy} empirical reference competition; not mineral phase identification")

        preview_height = min(max(1, args.preview_height), vnir.info.lines)
        rows = np.linspace(0, vnir.info.lines - 1, preview_height).round().astype(np.int64)
        full_gray = _stretch(albedo850)
        gray = full_gray[rows]
        background = np.repeat(gray[..., None], 3, axis=2)
        panels = [_label(background, "VNIR 850 nm")]
        colors = {"conservative": (220, 40, 35), "balanced": (255, 155, 20), "sensitive": (255, 225, 30)}
        for policy in policies:
            panels.append(_label(_overlay(background, family_masks[policy][rows], colors[policy]), policy))
        class_render = np.zeros_like(background)
        local_classes = class_maps["balanced"][rows]
        class_render[local_classes == 1] = (220, 50, 45)
        class_render[local_classes == 2] = (145, 70, 190)
        panels.append(_label(class_render, "balanced ref competition"))
        Image.fromarray(np.concatenate(panels, axis=1)).save(paths["preview"])
        mapped_preview = mapped_mask[rows]
        independent_preview = baseline.mask[rows]
        registration_class = np.zeros((*mapped_preview.shape, 3), dtype=np.uint8)
        registration_class[mapped_preview & independent_preview] = (0, 200, 0)
        registration_class[mapped_preview & ~independent_preview] = (255, 170, 0)
        registration_class[~mapped_preview & independent_preview] = (220, 0, 220)
        Image.fromarray(np.concatenate((
            _label(background, "VNIR 850 nm"),
            _label(_overlay(background, mapped_preview, (0, 255, 0)), "time-mapped SWIR mask"),
            _label(_overlay(background, independent_preview, (0, 220, 255)), "independent VNIR proxy"),
            _label(registration_class, "both / SWIR only / VNIR only"),
        ), axis=1)).save(paths["registration_contact"])
        depth = _depth_range(vnir)
        balanced_labels, balanced_components = _candidate_components(family_masks["balanced"], depth)
        _write_candidate_contact(paths["candidate_qa"], full_gray, balanced_labels, balanced_components)
        with paths["candidate_components"].open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(balanced_components[0]) if balanced_components else [
                "label", "pixels", "start_row", "stop_row", "start_column", "stop_column",
                "centroid_row", "centroid_column", "centroid_depth_m",
            ])
            writer.writeheader()
            writer.writerows(balanced_components)

        with paths["counts"].open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("policy", "family_pixels", "fraction_of_foreground", "reference_1", "reference_2", "ambiguous"))
            for policy, record in policy_audits.items():
                writer.writerow((policy, record["family_pixels"], record["fraction_of_foreground"], record["classified_reference_1_pixels"], record["classified_reference_2_pixels"], record["ambiguous_family_pixels"]))
        with paths["depth_blocks"].open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("start_row", "stop_row", "start_depth_m", "stop_depth_m", "foreground_pixels", "balanced_REE_family_pixels", "balanced_fraction_of_foreground"))
            for row_start in range(0, shape[0], 512):
                row_stop = min(shape[0], row_start + 512)
                foreground = int(np.count_nonzero(mapped_mask[row_start:row_stop]))
                selected = int(np.count_nonzero(family_masks["balanced"][row_start:row_stop]))
                start_depth = None if depth is None else depth[0] + row_start / max(shape[0] - 1, 1) * (depth[1] - depth[0])
                stop_depth = None if depth is None else depth[0] + min(row_stop, shape[0] - 1) / max(shape[0] - 1, 1) * (depth[1] - depth[0])
                writer.writerow((row_start, row_stop, start_depth, stop_depth, foreground, selected, selected / max(foreground, 1)))

        warnings: list[str] = []
        if mask_agreement["dice"] < 0.50 or row_profile_correlation < 0.40:
            warnings.append("VNIR/SWIR foreground agreement requires registration review")
        if policy_audits["balanced"]["fraction_of_foreground"] > 0.10:
            warnings.append("Balanced empirical REE anomaly exceeds 10% of foreground and requires threshold review")
        if policy_audits["balanced"]["maximum_column_fraction"] > 0.25:
            warnings.append("Balanced empirical REE result retains a high-density detector column")
        status = "ready" if not warnings else "review"
        reference_records = []
        for index, reference in enumerate(references):
            used_values = reference_resampled[index, analysis]
            reference_records.append({
                "reference_id": index + 1,
                "path": str(reference.path),
                "sha256": _hash(reference.path),
                "source_label": reference.source_label,
                "point_count": int(reference.values.size),
                "wavelength_range_nm": [float(reference.wavelengths_nm[0]), float(reference.wavelengths_nm[-1])],
                "minimum_value": float(np.min(reference.values)),
                "maximum_value": float(np.max(reference.values)),
                "nonpositive_value_count_full_curve": int(np.count_nonzero(reference.values <= 0)),
                "nonpositive_value_count_analysis_interval": int(np.count_nonzero(used_values <= 0)),
                "feature_strength": float(reference_strength[index]),
                "feature_peaks": _reference_peaks(wavelengths[analysis][scoring], reference_features[index]),
                "identity_status": "user_declared_REE_reference_without_phase_name_or_lab_certificate",
            })
        report = {
            "status": status,
            "method": "VNIR narrow-feature empirical-reference spectral-angle ensemble",
            "policy_profile": args.policy_profile,
            "inputs": {
                "vnir": str(args.vnir),
                "swir": str(args.swir),
                "swir_mask": str(args.swir_mask),
                "shape_vnir": list(shape) + [vnir.info.bands],
                "shape_swir": [swir.info.lines, swir.info.samples, swir.info.bands],
            },
            "reference_library": {
                "role": "experimental_user_empirical_reference_ensemble",
                "references": reference_records,
                "analysis_interval_nm": [650.0, 950.0],
                "scoring_interval_nm": [675.0, 925.0],
                "reference_to_reference_angle_rad": float(reference_angles[0, 1]),
                "output": str(paths["reference_library"]),
            },
            "registration": {
                "method": "normalized_line_time_mapping",
                "timing_match": timing_match,
                "timing": timing,
                "source_line_range": [int(source_rows[0]), int(source_rows[-1])],
                "mapped_mask_fraction": float(np.mean(mapped_mask)),
                "edge_guard_pixels": int(args.edge_guard_pixels),
                "analysis_mask_fraction": float(np.mean(analysis_mask)),
                "edge_guard_removed_pixels": int(np.count_nonzero(mapped_mask & ~analysis_mask)),
                "vnir_support_erosion_pixels": int(args.vnir_support_erosion_pixels),
                "neighborhood_guard": {
                    "local_window": int(args.local_window),
                    "minimum_local_support": float(args.minimum_local_support),
                    "maximum_local_coefficient_of_variation": float(args.maximum_local_cv),
                    "maximum_log_reflectance_gradient": float(args.maximum_log_gradient),
                    "guard_pixels": int(np.count_nonzero(neighborhood_guard)),
                    "fraction_of_edge_guarded_analysis_mask": float(np.mean(neighborhood_guard[analysis_mask])) if np.any(analysis_mask) else 0.0,
                    "local_support_quantiles_on_analysis_mask": {
                        str(q): float(np.quantile(local_support[analysis_mask], q)) for q in (0.0, 0.5, 0.95, 1.0)
                    },
                    "local_cv_quantiles_on_guard": {
                        str(q): float(np.quantile(local_cv[neighborhood_guard], q)) for q in (0.0, 0.5, 0.95, 1.0)
                    },
                    "log_gradient_quantiles_on_guard": {
                        str(q): float(np.quantile(log_gradient[neighborhood_guard], q)) for q in (0.0, 0.5, 0.95, 1.0)
                    },
                    "rationale": "Fixed scale-invariant gates reject dark-gap and abrupt-boundary mixed pixels without imposing an anomaly quota.",
                },
                "independent_vnir_mask_fraction": float(np.mean(baseline.mask)),
                "automatic_method_agreement": mask_agreement,
                "smoothed_row_profile_correlation": row_profile_correlation,
                "evidence_boundary": "Automatic VNIR/SWIR mask agreement is registration QA, not manual ground truth accuracy.",
            },
            "spectral_validity": {
                "valid_pixels": int(np.count_nonzero(spectrally_valid)),
                "valid_fraction_of_analysis_foreground": int(np.count_nonzero(spectrally_valid)) / max(int(np.count_nonzero(analysis_mask)), 1),
            },
            "policies": policy_audits,
            "balanced_components": {
                "count": len(balanced_components),
                "largest": balanced_components[:20],
            },
            "score_distribution": {
                "best_angle_rad_quantiles": {
                    str(quantile): float(np.quantile(best_angle[spectrally_valid], quantile))
                    for quantile in (0.0, 0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 1.0)
                },
                "threshold_basis": (
                    "Fixed absolute feature-angle gates, each below the observed inter-reference angle; "
                    "scene quantiles are reported for audit and do not impose a detection quota."
                ),
            },
            "quality_gate": {"status": status, "warnings": warnings},
            "depth_axis": {
                "header_sample_deep": None if depth is None else list(depth),
                "mapping": "linear along synchronized line acquisition",
            },
            "outputs": {key: str(value) for key, value in paths.items()},
            "elapsed_seconds": monotonic() - started,
            "interpretation": (
                "Outputs are spectral-similarity anomalies to user-provided VNIR empirical REE references. "
                "They are not confirmed rare-earth mineral phases, concentrations, grades, or assays."
            ),
        }
        paths["audit"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "mask_dice": report["registration"]["automatic_method_agreement"]["dice"],
        "row_profile_correlation": report["registration"]["smoothed_row_profile_correlation"],
        "balanced": report["policies"]["balanced"],
        "elapsed_seconds": report["elapsed_seconds"],
        "audit": str(paths["audit"]),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
