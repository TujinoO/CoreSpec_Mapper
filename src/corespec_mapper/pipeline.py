from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import csv
import json
import time

import numpy as np

from .algorithms import (
    absorption_depth,
    absorption_feature_metrics,
    classify_sam,
    clean_binary_mask,
    continuum_remove,
    directional_stripe_mask,
    remove_elongated_components,
    robust_column_bias,
    savgol_smooth,
    spectral_angles,
    spectral_feature_fit,
)
from .envi import EnviDataset, SpectralLibrary, derive_mask, wavelength_indices, write_envi


Progress = Callable[[str, float, str], None]


def _default_progress(stage: str, fraction: float, message: str) -> None:
    print(f"[{fraction * 100:6.2f}%] {stage}: {message}", flush=True)


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(value: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _dataset_summary(dataset: EnviDataset) -> dict[str, Any]:
    info = dataset.info
    return {
        "data_path": str(info.data_path),
        "header_path": str(info.header_path),
        "shape": [info.lines, info.samples, info.bands],
        "dtype": str(dataset.dtype),
        "interleave": info.interleave,
        "file_type": info.file_type,
        "wavelength_min_nm": None if info.wavelengths_nm is None else float(info.wavelengths_nm.min()),
        "wavelength_max_nm": None if info.wavelengths_nm is None else float(info.wavelengths_nm.max()),
        "wavelength_warning": info.wavelength_warning,
        "file_size_bytes": info.data_path.stat().st_size,
    }


def audit_project(config: dict[str, Any], progress: Progress = _default_progress) -> dict[str, Any]:
    progress("audit", 0.05, "Opening image and mask datasets")
    image = EnviDataset(config["image"])
    mask_cube = EnviDataset(config["mask"])
    if (image.info.lines, image.info.samples, image.info.bands) != (
        mask_cube.info.lines,
        mask_cube.info.samples,
        mask_cube.info.bands,
    ):
        raise ValueError("Image and mask cube dimensions differ")

    progress("audit", 0.25, "Recovering boolean core mask")
    mask = derive_mask(mask_cube, chunk_rows=int(config.get("chunk_rows", 128)))
    libraries = {}
    for index, group in enumerate(config["groups"]):
        library = SpectralLibrary.open(group["library"])
        if image.info.wavelengths_nm is None:
            raise ValueError("Image has no wavelength vector")
        max_delta = float(np.max(np.abs(image.info.wavelengths_nm - library.wavelengths_nm)))
        libraries[group["name"]] = {
            "path": str(library.data_path),
            "spectra_count": len(library.names),
            "spectra_names": library.names,
            "wavelength_warning": library.warning,
            "max_image_library_wavelength_delta_nm": max_delta,
        }
        progress("audit", 0.35 + 0.45 * (index + 1) / len(config["groups"]), f"Checked {group['name']} library")

    result = {
        "image": _dataset_summary(image),
        "mask_cube": _dataset_summary(mask_cube),
        "mask": {
            "valid_pixels": int(mask.sum()),
            "total_pixels": int(mask.size),
            "valid_fraction": float(mask.mean()),
        },
        "libraries": libraries,
    }
    progress("audit", 1.0, "Audit complete")
    return result


def _window_bounds(dataset: EnviDataset, start: int, stop: int | None) -> tuple[int, int]:
    stop = dataset.info.lines if stop is None else stop
    if not 0 <= start < stop <= dataset.info.lines:
        raise ValueError(f"Invalid line range [{start}, {stop})")
    return start, stop


def _reference_comparison(
    rules: np.ndarray,
    classes: np.ndarray,
    mask: np.ndarray,
    reference_rule_path: str | None,
    reference_class_path: str | None,
    start: int,
    stop: int,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    if reference_rule_path:
        reference_dataset = EnviDataset(reference_rule_path)
        reference = np.array(reference_dataset.read_rows(start, stop), copy=True)
        reference_dataset.close()
        reference = np.moveaxis(reference, -1, 0)
        difference = np.abs(rules[:, mask] - reference[:, mask])
        comparison["rule"] = {
            "mean_absolute_error": float(np.nanmean(difference)),
            "max_absolute_error": float(np.nanmax(difference)),
            "compared_values": int(difference.size),
        }
    if reference_class_path:
        reference_class_dataset = EnviDataset(reference_class_path)
        reference_class = np.array(reference_class_dataset.read_rows(start, stop)[..., 0], copy=True)
        reference_class_dataset.close()
        comparison["class"] = {
            "agreement_valid_pixels": float(np.mean(classes[mask] == reference_class[mask])),
            "compared_pixels": int(mask.sum()),
        }
    return comparison


def run_sam_baseline(
    config: dict[str, Any],
    group_name: str,
    output_dir: str | Path,
    start: int = 0,
    stop: int | None = None,
    progress: Progress = _default_progress,
) -> dict[str, Any]:
    group = next(group for group in config["groups"] if group["name"] == group_name)
    image = EnviDataset(config.get("baseline_image", config["image"]))
    mask_cube = EnviDataset(config.get("baseline_mask", config["mask"]))
    library = SpectralLibrary.open(group["library"])
    start, stop = _window_bounds(image, start, stop)
    mask = derive_mask(mask_cube, chunk_rows=int(config.get("chunk_rows", 128)), start=start, stop=stop)
    threshold = float(group["baseline_sam_threshold"])
    wavelengths = image.info.wavelengths_nm
    if wavelengths is None:
        raise ValueError("Image wavelength vector is required")
    baseline_windows = group.get("baseline_windows", [])
    baseline_indices = np.arange(image.info.bands) if not baseline_windows else wavelength_indices(wavelengths, baseline_windows)
    rows = stop - start
    rules = np.full((library.spectra.shape[0], rows, image.info.samples), np.nan, dtype=np.float32)
    classes = np.full((rows, image.info.samples), library.spectra.shape[0] + 1, dtype=np.uint8)
    classes[mask] = 0
    chunk_rows = int(config.get("chunk_rows", 128))

    for row_start, row_stop, cube in image.iter_rows(chunk_rows, start, stop, bands=baseline_indices):
        local_start = row_start - start
        local_stop = row_stop - start
        chunk_mask = mask[local_start:local_stop]
        if np.any(chunk_mask):
            pixels = np.asarray(cube[chunk_mask], dtype=np.float64)
            angles = spectral_angles(pixels, library.spectra[:, baseline_indices])
            for endpoint in range(library.spectra.shape[0]):
                target = rules[endpoint, local_start:local_stop]
                target[chunk_mask] = angles[:, endpoint].astype(np.float32)
            target_class = classes[local_start:local_stop]
            target_class[chunk_mask] = classify_sam(angles, threshold)
        fraction = (row_stop - start) / rows
        progress("sam-baseline", fraction * 0.9, f"Processed lines {row_start}:{row_stop}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_envi(rules, output_dir / "sam_rules.dat", band_names=[f"Rule ({name})" for name in library.names], description="CoreSpec Mapper SAM rule images")
    class_names = ["Unclassified", *library.names, "Masked Pixels"]
    write_envi(classes, output_dir / "sam_classes.dat", class_names=class_names, description="CoreSpec Mapper SAM classification")
    reference = group.get("reference", {})
    comparison = _reference_comparison(
        rules,
        classes,
        mask,
        reference.get("rule"),
        reference.get("class"),
        start,
        stop,
    )
    ids, counts = np.unique(classes, return_counts=True)
    summary = {
        "group": group_name,
        "line_range": [start, stop],
        "threshold_rad": threshold,
        "spectral_windows_nm": baseline_windows,
        "band_count": int(baseline_indices.size),
        "valid_pixels": int(mask.sum()),
        "class_counts": {str(int(key)): int(value) for key, value in zip(ids, counts)},
        "class_names": class_names,
        "comparison_to_envi": comparison,
    }
    save_json(summary, output_dir / "summary.json")
    progress("sam-baseline", 1.0, "Baseline SAM complete")
    return summary


def diagnose_sam_windows(
    config: dict[str, Any],
    group_name: str,
    output_path: str | Path,
    start: int = 0,
    stop: int | None = None,
    progress: Progress = _default_progress,
) -> dict[str, Any]:
    group = next(group for group in config["groups"] if group["name"] == group_name)
    reference_paths = group.get("reference", {})
    if not reference_paths.get("rule") or not reference_paths.get("class"):
        raise ValueError("SAM window diagnosis requires reference rule and class rasters")
    image = EnviDataset(config.get("baseline_image", config["image"]))
    mask_cube = EnviDataset(config.get("baseline_mask", config["mask"]))
    library = SpectralLibrary.open(group["library"])
    start, stop = _window_bounds(image, start, stop)
    rows = stop - start
    mask = derive_mask(mask_cube, chunk_rows=int(config.get("chunk_rows", 128)), start=start, stop=stop)
    wavelengths = image.info.wavelengths_nm
    if wavelengths is None:
        raise ValueError("Image wavelength vector is required")

    candidates = group.get("diagnostic_windows", [{"name": "full_spectrum", "windows": []}])
    candidate_indices = []
    for candidate in candidates:
        windows = candidate.get("windows", [])
        indices = np.arange(image.info.bands) if not windows else wavelength_indices(wavelengths, windows)
        candidate_indices.append(indices)

    reference_rule_dataset = EnviDataset(reference_paths["rule"])
    reference_rules = np.array(reference_rule_dataset.read_rows(start, stop), copy=True)
    reference_rule_dataset.close()
    reference_rules = np.moveaxis(reference_rules, -1, 0)
    reference_class_dataset = EnviDataset(reference_paths["class"])
    reference_classes = np.array(reference_class_dataset.read_rows(start, stop)[..., 0], copy=True)
    reference_class_dataset.close()

    accumulators = [
        {"absolute_sum": 0.0, "maximum": 0.0, "values": 0, "class_matches": 0, "class_values": 0}
        for _ in candidates
    ]
    threshold = float(group["baseline_sam_threshold"])
    chunk_rows = int(config.get("chunk_rows", 128))
    for row_start, row_stop, cube in image.iter_rows(chunk_rows, start, stop):
        local_start = row_start - start
        local_stop = row_stop - start
        chunk_mask = mask[local_start:local_stop]
        if np.any(chunk_mask):
            pixels = np.asarray(cube[chunk_mask], dtype=np.float64)
            reference_chunk = reference_rules[:, local_start:local_stop]
            reference_values = reference_chunk[:, chunk_mask].T
            reference_class_values = reference_classes[local_start:local_stop][chunk_mask]
            for candidate_index, indices in enumerate(candidate_indices):
                angles = spectral_angles(pixels[:, indices], library.spectra[:, indices])
                finite = np.isfinite(reference_values) & np.isfinite(angles)
                difference = np.abs(angles[finite] - reference_values[finite])
                accumulator = accumulators[candidate_index]
                if difference.size:
                    accumulator["absolute_sum"] += float(np.sum(difference))
                    accumulator["maximum"] = max(accumulator["maximum"], float(np.max(difference)))
                    accumulator["values"] += int(difference.size)
                predicted = classify_sam(angles, threshold)
                accumulator["class_matches"] += int(np.count_nonzero(predicted == reference_class_values))
                accumulator["class_values"] += int(reference_class_values.size)
        progress("sam-diagnose", (row_stop - start) / rows, f"Processed lines {row_start}:{row_stop}")

    results = []
    for candidate, indices, accumulator in zip(candidates, candidate_indices, accumulators):
        results.append(
            {
                "name": candidate["name"],
                "windows_nm": candidate.get("windows", []),
                "band_count": int(indices.size),
                "mean_absolute_error": accumulator["absolute_sum"] / max(accumulator["values"], 1),
                "max_absolute_error": accumulator["maximum"],
                "class_agreement_valid_pixels": accumulator["class_matches"] / max(accumulator["class_values"], 1),
                "compared_rule_values": accumulator["values"],
            }
        )
    results.sort(key=lambda item: item["mean_absolute_error"])
    summary = {
        "group": group_name,
        "line_range": [start, stop],
        "threshold_rad": threshold,
        "valid_pixels": int(mask.sum()),
        "candidates_ranked": results,
    }
    save_json(summary, output_path)
    progress("sam-diagnose", 1.0, f"Best candidate: {results[0]['name']}")
    return summary


def diagnose_sg(
    config: dict[str, Any],
    output_path: str | Path,
    start: int = 0,
    stop: int | None = None,
    progress: Progress = _default_progress,
) -> dict[str, Any]:
    raw = EnviDataset(config["mask"])
    target = EnviDataset(config.get("baseline_mask", config["mask"]))
    if (raw.info.lines, raw.info.samples, raw.info.bands) != (target.info.lines, target.info.samples, target.info.bands):
        raise ValueError("Raw masked cube and SG target dimensions differ")
    start, stop = _window_bounds(raw, start, stop)
    rows = stop - start
    mask = derive_mask(raw, chunk_rows=int(config.get("chunk_rows", 128)), start=start, stop=stop)
    candidates = config.get("sg_candidates", [{"window": 5, "polyorder": 2}])
    accumulators = [
        {"absolute_sum": 0.0, "square_sum": 0.0, "maximum": 0.0, "values": 0}
        for _ in candidates
    ]
    chunk_rows = int(config.get("chunk_rows", 128))
    for row_start in range(start, stop, chunk_rows):
        row_stop = min(row_start + chunk_rows, stop)
        local_start = row_start - start
        local_stop = row_stop - start
        chunk_mask = mask[local_start:local_stop]
        if np.any(chunk_mask):
            raw_pixels = np.asarray(raw.read_rows(row_start, row_stop)[chunk_mask], dtype=np.float64)
            target_pixels = np.asarray(target.read_rows(row_start, row_stop)[chunk_mask], dtype=np.float64)
            for candidate_index, candidate in enumerate(candidates):
                window = int(candidate["window"])
                polyorder = int(candidate["polyorder"])
                predicted = savgol_smooth(raw_pixels, window, polyorder)
                margin = window // 2
                difference = predicted[:, margin:-margin] - target_pixels[:, margin:-margin]
                finite_difference = difference[np.isfinite(difference)]
                if finite_difference.size:
                    accumulator = accumulators[candidate_index]
                    accumulator["absolute_sum"] += float(np.sum(np.abs(finite_difference)))
                    accumulator["square_sum"] += float(np.sum(finite_difference * finite_difference))
                    accumulator["maximum"] = max(accumulator["maximum"], float(np.max(np.abs(finite_difference))))
                    accumulator["values"] += int(finite_difference.size)
        progress("sg-diagnose", (row_stop - start) / rows, f"Processed lines {row_start}:{row_stop}")

    results = []
    for candidate, accumulator in zip(candidates, accumulators):
        count = max(accumulator["values"], 1)
        results.append(
            {
                **candidate,
                "mean_absolute_error": accumulator["absolute_sum"] / count,
                "root_mean_square_error": (accumulator["square_sum"] / count) ** 0.5,
                "max_absolute_error": accumulator["maximum"],
                "compared_values": accumulator["values"],
            }
        )
    results.sort(key=lambda item: item["mean_absolute_error"])
    summary = {
        "line_range": [start, stop],
        "valid_pixels": int(mask.sum()),
        "candidates_ranked": results,
    }
    save_json(summary, output_path)
    progress("sg-diagnose", 1.0, f"Best candidate: window={results[0]['window']}, polyorder={results[0]['polyorder']}")
    return summary


def _mineral_label(name: str) -> str:
    lowered = name.casefold()
    for mineral in ("calcite", "dolomite", "anhydrite", "gypsum", "illite", "montmorillonite", "kaolinite"):
        if mineral in lowered:
            return mineral
    return name.split()[0].casefold()


def _finite_reduce(values: np.ndarray, mode: str) -> np.ndarray:
    finite = np.isfinite(values)
    fill = np.inf if mode == "min" else -np.inf
    reduced = np.min(np.where(finite, values, fill), axis=0) if mode == "min" else np.max(np.where(finite, values, fill), axis=0)
    return np.where(np.any(finite, axis=0), reduced, np.nan)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {}
    points = [1, 5, 25, 50, 75, 95, 99]
    result = np.percentile(finite, points)
    return {f"p{point:02d}": float(value) for point, value in zip(points, result)}


@dataclass
class _PilotBuffers:
    sam: np.ndarray
    destriped_sam: np.ndarray
    destriped_classes: np.ndarray
    scale: np.ndarray
    rms: np.ndarray
    fit: np.ndarray
    depth: np.ndarray
    feature_center: np.ndarray
    feature_center_depth: np.ndarray
    feature_depth_ratio: np.ndarray
    feature_center_distance: np.ndarray
    classes: np.ndarray
    confidence: np.ndarray


def _estimate_column_bias(
    image: EnviDataset,
    mask: np.ndarray,
    bands: np.ndarray,
    *,
    start: int,
    stop: int,
    radius: int,
    sample_lines: int,
    band_chunk: int,
    row_chunk: int,
) -> np.ndarray:
    rows = stop - start
    sample_count = min(max(int(sample_lines), 1), rows)
    sampled_rows = np.unique(np.linspace(0, rows - 1, sample_count, dtype=np.int64))
    sampled_mask = mask[sampled_rows]
    bias = np.zeros((image.info.samples, bands.size), dtype=np.float64)
    for band_start in range(0, bands.size, band_chunk):
        band_stop = min(band_start + band_chunk, bands.size)
        selected_bands = bands[band_start:band_stop]
        blocks: list[np.ndarray] = []
        for row_start, row_stop, cube in image.iter_rows(row_chunk, start, stop, bands=selected_bands):
            local_start = row_start - start
            local_stop = row_stop - start
            selected = sampled_rows[(sampled_rows >= local_start) & (sampled_rows < local_stop)] - local_start
            if selected.size:
                blocks.append(np.array(cube[selected], dtype=np.float64, copy=True))
        sampled_cube = np.concatenate(blocks, axis=0)
        bias[:, band_start:band_stop] = robust_column_bias(sampled_cube, sampled_mask, radius=radius)
    return bias


def run_pilot(
    config: dict[str, Any],
    output_dir: str | Path,
    start: int = 0,
    stop: int | None = None,
    progress: Progress = _default_progress,
) -> dict[str, Any]:
    analysis_image_path = config.get("analysis_image", config["image"])
    analysis_mask_path = config.get("analysis_mask", config["mask"])
    image = EnviDataset(analysis_image_path)
    mask_cube = EnviDataset(analysis_mask_path)
    input_is_smoothed = bool(config.get("analysis_input_is_smoothed", config.get("input_is_smoothed", False)))
    start, stop = _window_bounds(image, start, stop)
    rows = stop - start
    chunk_rows = int(config.get("chunk_rows", 128))
    mask = derive_mask(mask_cube, chunk_rows=chunk_rows, start=start, stop=stop)
    refinement_mask = mask.copy()
    if int(config.get("mask_erosion", 0)) > 0:
        from .algorithms import _binary_erosion

        refinement_mask = _binary_erosion(refinement_mask, iterations=int(config["mask_erosion"]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary: dict[str, Any] = {
        "line_range": [start, stop],
        "analysis_image": str(image.info.data_path),
        "analysis_mask": str(mask_cube.info.data_path),
        "analysis_input_is_smoothed": input_is_smoothed,
        "valid_pixels": int(mask.sum()),
        "refinement_pixels": int(refinement_mask.sum()),
        "groups": {},
    }

    for group_index, group in enumerate(config["groups"]):
        group_start = group_index / len(config["groups"])
        group_span = 1.0 / len(config["groups"])
        library = SpectralLibrary.open(group["library"])
        wavelengths = image.info.wavelengths_nm
        if wavelengths is None:
            raise ValueError("Image wavelength vector is required")
        feature_windows = group.get("feature_windows", group["windows"])
        sam_windows = group.get("sam_windows", group.get("baseline_windows", feature_windows))
        feature_window_indices = [wavelength_indices(wavelengths, [window]) for window in feature_windows]
        sam_indices = wavelength_indices(wavelengths, sam_windows)
        endpoint_count = library.spectra.shape[0]
        destripe_parameters = group.get("destripe", {})
        destripe_enabled = bool(destripe_parameters.get("enabled", False))
        correct_features = destripe_enabled and bool(destripe_parameters.get("apply_to_features", False))
        if destripe_enabled and not input_is_smoothed:
            raise ValueError("Robust column destriping requires a smoothed analysis image")
        correction_indices = (
            np.unique(np.concatenate([sam_indices, *feature_window_indices]))
            if correct_features
            else sam_indices
        )
        column_bias = (
            _estimate_column_bias(
                image,
                mask,
                correction_indices,
                start=start,
                stop=stop,
                radius=int(destripe_parameters.get("radius", 2)),
                sample_lines=int(destripe_parameters.get("sample_lines", 1024)),
                band_chunk=int(destripe_parameters.get("band_chunk", 16)),
                row_chunk=chunk_rows,
            )
            if destripe_enabled
            else np.zeros((image.info.samples, correction_indices.size), dtype=np.float64)
        )
        correction_lookup = {int(band): position for position, band in enumerate(correction_indices)}
        sam_bias_positions = np.asarray([correction_lookup[int(band)] for band in sam_indices])
        sam_column_bias = column_bias[:, sam_bias_positions]
        feature_bias_positions = [
            np.asarray([correction_lookup[int(band)] for band in indices])
            for indices in feature_window_indices
        ] if correct_features else []
        destripe_strength = float(destripe_parameters.get("strength", 0.0 if not destripe_enabled else 0.5))
        feature_gate = group.get("feature_gate", {})
        feature_gate_enabled = bool(feature_gate.get("enabled", False))
        endpoint_minerals = np.asarray([_mineral_label(name) for name in library.names], dtype=object)
        buffers = _PilotBuffers(
            sam=np.full((endpoint_count, rows, image.info.samples), np.nan, dtype=np.float32),
            destriped_sam=np.full((endpoint_count, rows, image.info.samples), np.nan, dtype=np.float32),
            destriped_classes=np.zeros((rows, image.info.samples), dtype=np.uint8),
            scale=np.full((endpoint_count, rows, image.info.samples), np.nan, dtype=np.float32),
            rms=np.full((endpoint_count, rows, image.info.samples), np.nan, dtype=np.float32),
            fit=np.full((endpoint_count, rows, image.info.samples), np.nan, dtype=np.float32),
            depth=np.full((rows, image.info.samples), np.nan, dtype=np.float32),
            feature_center=np.full((rows, image.info.samples), np.nan, dtype=np.float32),
            feature_center_depth=np.full((rows, image.info.samples), np.nan, dtype=np.float32),
            feature_depth_ratio=np.full((rows, image.info.samples), np.nan, dtype=np.float32),
            feature_center_distance=np.full((rows, image.info.samples), np.nan, dtype=np.float32),
            classes=np.zeros((rows, image.info.samples), dtype=np.uint8),
            confidence=np.zeros((rows, image.info.samples), dtype=np.float32),
        )

        reference_removed = []
        for indices in feature_window_indices:
            reference_cr = continuum_remove(library.spectra[:, indices], wavelengths[indices])
            reference_removed.append(reference_cr)
        reference_removed_combined = np.concatenate(reference_removed, axis=1)
        reference_absorption = 1.0 - reference_removed_combined
        feature_wavelengths = np.concatenate([wavelengths[indices] for indices in feature_window_indices])
        if feature_gate_enabled:
            search_window = feature_gate.get("search_window_nm", [float(feature_wavelengths[0]), float(feature_wavelengths[-1])])
            reference_feature_metrics = absorption_feature_metrics(
                reference_removed_combined,
                feature_wavelengths,
                search_window,
            )
            reference_centers = reference_feature_metrics.center_nm
        else:
            reference_centers = np.full(endpoint_count, np.nan, dtype=np.float64)

        for row_start, row_stop, cube in image.iter_rows(chunk_rows, start, stop):
            local_start = row_start - start
            local_stop = row_stop - start
            chunk_mask = mask[local_start:local_stop]
            if np.any(chunk_mask):
                pixels = np.asarray(cube[chunk_mask], dtype=np.float64)
                if not input_is_smoothed:
                    pixels = savgol_smooth(
                        pixels,
                        window_length=int(config.get("sg_window", 5)),
                        polyorder=int(config.get("sg_polyorder", 2)),
                    )
                angles = spectral_angles(pixels[:, sam_indices], library.spectra[:, sam_indices])
                pixel_columns = np.nonzero(chunk_mask)[1]
                corrected_sam_pixels = pixels[:, sam_indices] - destripe_strength * sam_column_bias[pixel_columns]
                destriped_angles = spectral_angles(corrected_sam_pixels, library.spectra[:, sam_indices])
                pixel_removed = []
                for window_index, indices in enumerate(feature_window_indices):
                    feature_pixels = pixels[:, indices]
                    if correct_features:
                        feature_pixels = (
                            feature_pixels
                            - destripe_strength * column_bias[pixel_columns][:, feature_bias_positions[window_index]]
                        )
                    pixel_removed.append(continuum_remove(feature_pixels, wavelengths[indices]))
                pixel_removed_combined = np.concatenate(pixel_removed, axis=1)
                pixel_absorption = 1.0 - pixel_removed_combined
                depth = absorption_depth(pixel_removed_combined)
                sff = spectral_feature_fit(pixel_absorption, reference_absorption)
                if feature_gate_enabled:
                    pixel_feature_metrics = absorption_feature_metrics(
                        pixel_removed_combined,
                        feature_wavelengths,
                        search_window,
                    )
                else:
                    nan_values = np.full(pixels.shape[0], np.nan, dtype=np.float64)
                    pixel_feature_metrics = None

                for endpoint in range(endpoint_count):
                    for array, values in (
                        (buffers.sam, angles[:, endpoint]),
                        (buffers.destriped_sam, destriped_angles[:, endpoint]),
                        (buffers.scale, sff.scale[:, endpoint]),
                        (buffers.rms, sff.rms[:, endpoint]),
                        (buffers.fit, sff.quality[:, endpoint]),
                    ):
                        target = array[endpoint, local_start:local_stop]
                        target[chunk_mask] = values.astype(np.float32)
                depth_target = buffers.depth[local_start:local_stop]
                depth_target[chunk_mask] = depth.astype(np.float32)
                if pixel_feature_metrics is not None:
                    center_target = buffers.feature_center[local_start:local_stop]
                    center_target[chunk_mask] = pixel_feature_metrics.center_nm.astype(np.float32)
                    center_depth_target = buffers.feature_center_depth[local_start:local_stop]
                    center_depth_target[chunk_mask] = pixel_feature_metrics.center_depth.astype(np.float32)
                    ratio_target = buffers.feature_depth_ratio[local_start:local_stop]
                    ratio_target[chunk_mask] = pixel_feature_metrics.depth_ratio.astype(np.float32)

                angle_threshold = float(group["sam_threshold"])
                depth_threshold = float(group["depth_threshold"])
                fit_threshold = float(group.get("fit_threshold", 0.0))
                scale_min = float(group.get("scale_min", 0.1))
                sam_labels = classify_sam(angles, angle_threshold)
                destripe_confirmed = np.zeros(sam_labels.size, dtype=bool)
                for endpoint, mineral in enumerate(endpoint_minerals):
                    selected = sam_labels == endpoint + 1
                    if not np.any(selected):
                        continue
                    same_mineral = np.flatnonzero(endpoint_minerals == mineral)
                    destripe_confirmed[selected] = (
                        np.min(destriped_angles[selected][:, same_mineral], axis=1) <= angle_threshold
                    )
                destriped_labels = np.where(destripe_confirmed, sam_labels, 0).astype(np.uint8)
                destriped_target = buffers.destriped_classes[local_start:local_stop]
                destriped_target[chunk_mask] = destriped_labels
                selected_endpoint = np.maximum(sam_labels.astype(np.int64) - 1, 0)
                pixel_rows = np.arange(sam_labels.size)
                selected_angle = angles[pixel_rows, selected_endpoint]
                selected_fit = sff.quality[pixel_rows, selected_endpoint]
                selected_scale = sff.scale[pixel_rows, selected_endpoint]
                if pixel_feature_metrics is not None:
                    selected_reference_center = reference_centers[selected_endpoint]
                    center_distance = np.abs(pixel_feature_metrics.center_nm - selected_reference_center)
                    center_tolerance = float(feature_gate.get("center_tolerance_nm", np.inf))
                    center_depth_threshold = float(feature_gate.get("min_center_depth", 0.0))
                    depth_ratio_threshold = float(feature_gate.get("min_depth_ratio", 0.0))
                    feature_gate_pass = (
                        (center_distance <= center_tolerance)
                        & (pixel_feature_metrics.center_depth >= center_depth_threshold)
                        & (pixel_feature_metrics.depth_ratio >= depth_ratio_threshold)
                    )
                    distance_target = buffers.feature_center_distance[local_start:local_stop]
                    distance_target[chunk_mask] = center_distance.astype(np.float32)
                else:
                    center_distance = nan_values
                    center_tolerance = np.inf
                    feature_gate_pass = np.ones(sam_labels.size, dtype=bool)
                valid_candidate = (
                    (sam_labels > 0)
                    & destripe_confirmed
                    & (selected_fit >= fit_threshold)
                    & (selected_scale >= scale_min)
                    & (depth >= depth_threshold)
                    & feature_gate_pass
                )
                labels = np.where(valid_candidate, sam_labels, 0).astype(np.uint8)
                angle_score = np.clip(1.0 - selected_angle / angle_threshold, 0.0, 1.0)
                fit_score = np.clip(selected_fit / max(fit_threshold * 2.0, 1e-8), 0.0, 1.0)
                depth_score = np.clip(depth / max(depth_threshold * 2.0, 1e-8), 0.0, 1.0)
                if feature_gate_enabled and np.isfinite(center_tolerance):
                    center_score = np.clip(1.0 - center_distance / max(center_tolerance, 1e-8), 0.0, 1.0)
                else:
                    center_score = 1.0
                best_score = np.where(valid_candidate, angle_score * fit_score * depth_score * center_score, 0.0)
                class_target = buffers.classes[local_start:local_stop]
                class_target[chunk_mask] = labels
                confidence_target = buffers.confidence[local_start:local_stop]
                confidence_target[chunk_mask] = best_score.astype(np.float32)

            chunk_fraction = (row_stop - start) / rows
            overall = group_start + group_span * chunk_fraction
            progress("pilot", overall * 0.92, f"{group['name']}: processed lines {row_start}:{row_stop}")

        group_dir = output_dir / group["name"]
        group_dir.mkdir(parents=True, exist_ok=True)
        names = library.names
        write_envi(buffers.sam, group_dir / "sam_rules.dat", band_names=names, description=f"{group['name']} SAM angles")
        write_envi(
            buffers.destriped_sam,
            group_dir / "destriped_sam_rules.dat",
            band_names=names,
            description=f"{group['name']} robust column-destriped SAM angles",
        )
        write_envi(buffers.scale, group_dir / "sff_scale.dat", band_names=names, description=f"{group['name']} SFF scale")
        write_envi(buffers.rms, group_dir / "sff_rms.dat", band_names=names, description=f"{group['name']} SFF RMS")
        write_envi(buffers.fit, group_dir / "sff_quality.dat", band_names=names, description=f"{group['name']} SFF quality")
        write_envi(buffers.depth, group_dir / "absorption_depth.dat", description=f"{group['name']} maximum absorption depth")
        if feature_gate_enabled:
            write_envi(
                buffers.feature_center,
                group_dir / "feature_center_nm.dat",
                description=f"{group['name']} diagnostic absorption center in nm",
            )
            write_envi(
                buffers.feature_center_depth,
                group_dir / "feature_center_depth.dat",
                description=f"{group['name']} diagnostic absorption-center depth",
            )
            write_envi(
                buffers.feature_depth_ratio,
                group_dir / "feature_depth_ratio.dat",
                description=f"{group['name']} diagnostic depth divided by full-window depth",
            )
            write_envi(
                buffers.feature_center_distance,
                group_dir / "feature_center_distance_nm.dat",
                description=f"{group['name']} absorption-center distance from the SAM-selected reference",
            )
        write_envi(buffers.confidence, group_dir / "confidence.dat", description=f"{group['name']} preliminary confidence")

        raw_classes = buffers.classes.copy()
        raw_classes[~refinement_mask] = 0
        raw_output_classes = raw_classes.copy()
        raw_output_classes[~mask] = endpoint_count + 1
        output_class_names = ["Unclassified", *names, "Masked Pixels"]

        sam_only_classes = np.zeros_like(raw_classes)
        sam_only_classes[mask] = classify_sam(buffers.sam[:, mask].T, float(group["sam_threshold"]))
        sam_only_output = sam_only_classes.copy()
        sam_only_output[~mask] = endpoint_count + 1
        write_envi(
            sam_only_output,
            group_dir / "sam_only_classes.dat",
            class_names=output_class_names,
            description=f"{group['name']} SAM-only classes",
        )
        destriped_output = buffers.destriped_classes.copy()
        destriped_output[~mask] = endpoint_count + 1
        write_envi(
            destriped_output,
            group_dir / "destriped_sam_confirmed_classes.dat",
            class_names=output_class_names,
            description=f"{group['name']} SAM candidates confirmed after robust column correction",
        )
        write_envi(
            raw_output_classes,
            group_dir / "candidate_classes_before_spatial.dat",
            class_names=output_class_names,
            description=f"{group['name']} candidates before spatial cleanup",
        )

        stripe_parameters = group.get("stripe_filter", {})
        stripe_enabled = bool(stripe_parameters.get("enabled", False))
        stripe_noise = np.zeros_like(raw_classes, dtype=bool)
        directional_removed = 0
        small_component_removed = 0
        elongated_removed = 0
        cleaned = np.zeros_like(buffers.classes)
        for endpoint in range(endpoint_count):
            endpoint_mask = raw_classes == endpoint + 1
            if stripe_enabled:
                detected = directional_stripe_mask(
                    endpoint_mask,
                    refinement_mask,
                    vertical_window=int(stripe_parameters.get("vertical_window", 61)),
                    min_vertical_density=float(stripe_parameters.get("min_vertical_density", 0.15)),
                    column_ratio=float(stripe_parameters.get("column_ratio", 1.8)),
                    column_excess=float(stripe_parameters.get("column_excess", 0.04)),
                    max_lateral_support=int(stripe_parameters.get("max_lateral_support", 2)),
                    neighborhood_radius=int(stripe_parameters.get("neighborhood_radius", 6)),
                )
                stripe_noise |= detected
                directional_removed += int(np.count_nonzero(detected))
                endpoint_mask &= ~detected
            before_components = int(np.count_nonzero(endpoint_mask))
            endpoint_mask = clean_binary_mask(
                endpoint_mask,
                median_size=int(group.get("median_size", 3)),
                min_component=int(group.get("min_component", 4)),
            )
            small_component_removed += before_components - int(np.count_nonzero(endpoint_mask))
            if stripe_enabled and bool(stripe_parameters.get("remove_elongated", True)):
                before_elongated = int(np.count_nonzero(endpoint_mask))
                endpoint_mask = remove_elongated_components(
                    endpoint_mask,
                    min_height=int(stripe_parameters.get("min_line_height", 15)),
                    max_width=int(stripe_parameters.get("max_line_width", 3)),
                    min_aspect=float(stripe_parameters.get("min_aspect", 5.0)),
                )
                elongated_removed += before_elongated - int(np.count_nonzero(endpoint_mask))
            cleaned[endpoint_mask] = endpoint + 1
        write_envi(
            stripe_noise.astype(np.uint8),
            group_dir / "directional_stripe_noise_mask.dat",
            description=f"{group['name']} detected directional stripe noise",
        )
        output_classes = cleaned.copy()
        output_classes[~mask] = endpoint_count + 1
        write_envi(
            output_classes,
            group_dir / "preliminary_classes.dat",
            class_names=output_class_names,
            description=f"{group['name']} preliminary classes",
        )

        endpoint_ids, endpoint_counts = np.unique(cleaned[mask], return_counts=True)
        endpoint_summary = {output_class_names[int(idx)]: int(count) for idx, count in zip(endpoint_ids, endpoint_counts)}
        mineral_counts: Counter[str] = Counter()
        for endpoint, name in enumerate(names, start=1):
            mineral_counts[_mineral_label(name)] += int(np.count_nonzero(cleaned == endpoint))
        valid_spectral = mask & np.isfinite(buffers.depth)
        sam_best = _finite_reduce(buffers.sam[:, valid_spectral], "min")
        destriped_sam_best = _finite_reduce(buffers.destriped_sam[:, valid_spectral], "min")
        fit_best = _finite_reduce(buffers.fit[:, valid_spectral], "max")
        scale_best = _finite_reduce(buffers.scale[:, valid_spectral], "max")
        rms_best = _finite_reduce(buffers.rms[:, valid_spectral], "min")
        angle_threshold = float(group["sam_threshold"])
        depth_threshold = float(group["depth_threshold"])
        fit_threshold = float(group.get("fit_threshold", 0.0))
        scale_min = float(group.get("scale_min", 0.1))
        diagnostic_angles = buffers.sam[:, valid_spectral].T
        diagnostic_fit = buffers.fit[:, valid_spectral].T
        diagnostic_scale = buffers.scale[:, valid_spectral].T
        diagnostic_labels = classify_sam(diagnostic_angles, angle_threshold)
        diagnostic_endpoints = np.maximum(diagnostic_labels.astype(np.int64) - 1, 0)
        diagnostic_rows = np.arange(diagnostic_labels.size)
        selected_fit = diagnostic_fit[diagnostic_rows, diagnostic_endpoints]
        selected_scale = diagnostic_scale[diagnostic_rows, diagnostic_endpoints]
        if feature_gate_enabled:
            diagnostic_center_pass = (
                (buffers.feature_center_distance[valid_spectral] <= float(feature_gate.get("center_tolerance_nm", np.inf)))
                & (buffers.feature_center_depth[valid_spectral] >= float(feature_gate.get("min_center_depth", 0.0)))
                & (buffers.feature_depth_ratio[valid_spectral] >= float(feature_gate.get("min_depth_ratio", 0.0)))
            )
        else:
            diagnostic_center_pass = np.ones(diagnostic_labels.size, dtype=bool)
        selected_joint = (
            (diagnostic_labels > 0)
            & (selected_fit >= fit_threshold)
            & (selected_scale >= scale_min)
            & (buffers.depth[valid_spectral] >= depth_threshold)
            & diagnostic_center_pass
            & refinement_mask[valid_spectral]
        )
        diagnostics = {
            "mask_pixels": int(mask.sum()),
            "refinement_mask_pixels": int(refinement_mask.sum()),
            "valid_spectral_pixels": int(valid_spectral.sum()),
            "pass_counts": {
                "sam_any_endpoint": int(np.count_nonzero(sam_best <= angle_threshold)),
                "destriped_sam_same_mineral": int(np.count_nonzero(buffers.destriped_classes[valid_spectral] > 0)),
                "fit_any_endpoint": int(np.count_nonzero(fit_best >= fit_threshold)),
                "fit_selected_sam_endpoint": int(np.count_nonzero((diagnostic_labels > 0) & (selected_fit >= fit_threshold))),
                "scale_any_endpoint": int(np.count_nonzero(scale_best >= scale_min)),
                "scale_selected_sam_endpoint": int(np.count_nonzero((diagnostic_labels > 0) & (selected_scale >= scale_min))),
                "depth": int(np.count_nonzero(buffers.depth[valid_spectral] >= depth_threshold)),
                "feature_gate": int(np.count_nonzero((diagnostic_labels > 0) & diagnostic_center_pass)),
                "joint_before_spatial": int(np.count_nonzero(raw_classes[mask])),
                "directional_stripe_removed": directional_removed,
                "small_component_removed": small_component_removed,
                "elongated_component_removed": elongated_removed,
                "classified_after_spatial": int(np.count_nonzero(cleaned[mask])),
            },
            "percentiles": {
                "best_sam_rad": _percentiles(sam_best),
                "best_destriped_sam_rad": _percentiles(destriped_sam_best),
                "best_sff_quality": _percentiles(fit_best),
                "best_sff_scale": _percentiles(scale_best),
                "best_sff_rms": _percentiles(rms_best),
                "absorption_depth": _percentiles(buffers.depth[valid_spectral]),
                "feature_center_nm": _percentiles(buffers.feature_center[valid_spectral]) if feature_gate_enabled else {},
                "feature_center_distance_nm": _percentiles(buffers.feature_center_distance[valid_spectral]) if feature_gate_enabled else {},
            },
        }
        group_summary = {
            "library": str(library.data_path),
            "sam_windows_nm": sam_windows,
            "sam_band_count": int(sam_indices.size),
            "feature_windows_nm": feature_windows,
            "destripe": {
                "enabled": destripe_enabled,
                "strength": destripe_strength,
                "radius": int(destripe_parameters.get("radius", 2)),
                "sample_lines": int(destripe_parameters.get("sample_lines", 1024)),
                "apply_to_features": correct_features,
            },
            "feature_gate": feature_gate,
            "stripe_filter": stripe_parameters,
            "thresholds": {
                "sam_rad": group["sam_threshold"],
                "depth": group["depth_threshold"],
                "fit": group.get("fit_threshold", 0.0),
                "scale_min": group.get("scale_min", 0.1),
                "feature_gate": feature_gate,
            },
            "endpoint_counts": endpoint_summary,
            "mineral_counts": dict(mineral_counts),
            "diagnostics": diagnostics,
        }
        save_json(group_summary, group_dir / "summary.json")
        run_summary["groups"][group["name"]] = group_summary

    save_json(run_summary, output_dir / "summary.json")
    with (output_dir / "mineral_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "mineral", "pixels"])
        for group_name, summary in run_summary["groups"].items():
            for mineral, count in summary["mineral_counts"].items():
                writer.writerow([group_name, mineral, count])
    from .preview import make_pilot_previews

    make_pilot_previews(config, output_dir, start, stop)
    progress("pilot", 1.0, "Pilot workflow complete")
    return run_summary
