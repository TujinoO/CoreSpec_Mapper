from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence
import csv
import json

import numpy as np

from .algorithms import (
    clean_binary_mask,
    continuum_remove,
    directional_stripe_mask,
    remove_elongated_components,
    savgol_smooth,
    spectral_angles,
)
from .envi import EnviDataset, derive_mask, wavelength_indices, write_envi
from .library_ensemble import (
    MINERAL_DISPLAY_NAMES,
    MINERAL_GROUPS,
    EnsembleLibrary,
    build_library_ensemble,
    write_ensemble_artifacts,
)


Progress = Callable[[str, float, str], None]


def _default_progress(stage: str, fraction: float, message: str) -> None:
    print(f"[{fraction * 100:6.2f}%] {stage}: {message}", flush=True)


def _save_json(value: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return output


def column_percentile_thresholds(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    percentile_fraction: float,
    *,
    min_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or values.shape != valid.shape:
        raise ValueError("Scores and valid mask must be matching two-dimensional arrays")
    if not 0.0 < percentile_fraction < 1.0:
        raise ValueError("Column percentile fraction must be between zero and one")
    sample = values[valid]
    if sample.size == 0:
        raise ValueError("Column calibration contains no finite valid scores")
    percentile = percentile_fraction * 100.0
    global_threshold = float(np.percentile(sample, percentile))
    thresholds = np.full(values.shape[1], global_threshold, dtype=np.float64)
    sample_counts = np.zeros(values.shape[1], dtype=np.int64)
    for column in range(values.shape[1]):
        column_values = values[valid[:, column], column]
        sample_counts[column] = column_values.size
        if column_values.size >= min_samples:
            thresholds[column] = float(np.percentile(column_values, percentile))
    return thresholds, sample_counts


def aggregate_reference_scores(
    reference_scores: np.ndarray,
    reference_minerals: Sequence[str],
    mineral_order: Sequence[str],
    *,
    best_k: int = 2,
) -> np.ndarray:
    scores = np.asarray(reference_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(reference_minerals):
        raise ValueError("Reference score matrix does not match reference labels")
    scores = np.where(np.isfinite(scores), scores, np.inf)
    result = np.full((scores.shape[0], len(mineral_order)), np.inf, dtype=np.float64)
    labels = np.asarray(reference_minerals, dtype=object)
    for mineral_index, mineral in enumerate(mineral_order):
        selected = np.flatnonzero(labels == mineral)
        if selected.size == 0:
            raise ValueError(f"No references are available for {mineral}")
        count = min(max(int(best_k), 1), selected.size)
        mineral_scores = np.sort(scores[:, selected], axis=1)[:, :count]
        result[:, mineral_index] = np.mean(mineral_scores, axis=1)
    return result


def balanced_tie_labels(
    mineral_scores: np.ndarray,
    mineral_order: Sequence[str],
    *,
    tie_tolerance: float,
    tie_bias: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(mineral_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(mineral_order):
        raise ValueError("Mineral scores do not match the mineral order")
    safe = np.where(np.isfinite(scores), scores, np.inf)
    raw_labels = np.argmin(safe, axis=1)
    best = safe[np.arange(safe.shape[0]), raw_labels]
    contenders = safe <= best[:, None] + float(tie_tolerance)
    biases = np.asarray([(tie_bias or {}).get(mineral, 0.0) for mineral in mineral_order], dtype=np.float64)
    adjusted = np.where(contenders, safe - biases[None, :], np.inf)
    labels = np.argmin(adjusted, axis=1)
    labels[~np.isfinite(best)] = -1
    tie_count = np.sum(contenders, axis=1)
    changed = (labels != raw_labels) & (labels >= 0)
    return labels, tie_count, changed


def _unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    return np.divide(array, norms[:, None], out=np.zeros_like(array), where=norms[:, None] > 1e-12)


def _window_depth(absorption: np.ndarray, wavelengths_nm: np.ndarray, window: Sequence[float]) -> np.ndarray:
    lower, upper = (float(value) for value in window)
    selected = (wavelengths_nm >= lower) & (wavelengths_nm <= upper)
    if not np.any(selected):
        return np.zeros(absorption.shape[0], dtype=np.float64)
    return np.max(absorption[:, selected], axis=1)


def _window_center(absorption: np.ndarray, wavelengths_nm: np.ndarray, window: Sequence[float]) -> np.ndarray:
    lower, upper = (float(value) for value in window)
    selected = (wavelengths_nm >= lower) & (wavelengths_nm <= upper)
    selected_wavelengths = wavelengths_nm[selected]
    indices = np.argmax(absorption[:, selected], axis=1)
    return (selected_wavelengths[indices] - lower) / max(upper - lower, 1.0)


def diagnostic_feature_matrix(group_name: str, absorption: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    absorption = np.asarray(absorption, dtype=np.float64)
    full_depth = np.maximum(np.max(absorption, axis=1), 1e-12)
    if group_name == "carbonates":
        features = [
            _window_center(absorption, wavelengths_nm, (2280.0, 2360.0)),
            _window_depth(absorption, wavelengths_nm, (2280.0, 2320.0)) / full_depth,
            _window_depth(absorption, wavelengths_nm, (2320.0, 2360.0)) / full_depth,
        ]
    elif group_name == "sulfates":
        features = [
            _window_depth(absorption, wavelengths_nm, (1410.0, 1490.0)) / full_depth,
            _window_depth(absorption, wavelengths_nm, (1720.0, 1785.0)) / full_depth,
            _window_depth(absorption, wavelengths_nm, (1880.0, 1995.0)) / full_depth,
            _window_depth(absorption, wavelengths_nm, (2140.0, 2250.0)) / full_depth,
            _window_center(absorption, wavelengths_nm, (1880.0, 1995.0)),
        ]
    elif group_name == "clays":
        features = [
            _window_depth(absorption, wavelengths_nm, (2145.0, 2180.0)) / full_depth,
            _window_depth(absorption, wavelengths_nm, (2180.0, 2235.0)) / full_depth,
            _window_depth(absorption, wavelengths_nm, (2300.0, 2380.0)) / full_depth,
            _window_center(absorption, wavelengths_nm, (2160.0, 2235.0)),
        ]
    else:
        raise ValueError(f"Unknown mineral group: {group_name}")
    return np.column_stack(features)


def _continuum_absorption(spectra: np.ndarray, wavelengths_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    removed = continuum_remove(spectra, wavelengths_nm)
    absorption = np.clip(1.0 - removed, 0.0, None)
    depth = np.max(absorption, axis=1)
    return absorption, depth


def _subclass_reference_scores(
    group_name: str,
    pixel_absorption: np.ndarray,
    reference_absorption: np.ndarray,
    wavelengths_nm: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    pixel_shape = _unit_rows(pixel_absorption)
    reference_shape = _unit_rows(reference_absorption)
    shape_angle = spectral_angles(pixel_shape, reference_shape)
    pixel_derivative = _unit_rows(np.gradient(pixel_absorption, wavelengths_nm, axis=1))
    reference_derivative = _unit_rows(np.gradient(reference_absorption, wavelengths_nm, axis=1))
    derivative_angle = spectral_angles(pixel_derivative, reference_derivative)
    pixel_features = diagnostic_feature_matrix(group_name, pixel_absorption, wavelengths_nm)
    reference_features = diagnostic_feature_matrix(group_name, reference_absorption, wavelengths_nm)
    feature_distance = np.sqrt(np.mean((pixel_features[:, None, :] - reference_features[None, :, :]) ** 2, axis=2))
    return (
        float(weights.get("shape", 0.70)) * shape_angle
        + float(weights.get("derivative", 0.15)) * derivative_angle
        + float(weights.get("features", 0.15)) * feature_distance
    )


def _prepare_group(
    group: dict[str, Any],
    ensemble: EnsembleLibrary,
    wavelengths_nm: np.ndarray,
) -> dict[str, Any]:
    name = str(group["name"])
    mineral_order = list(MINERAL_GROUPS[name])
    reference_indices = ensemble.group_indices(name)
    reference_minerals = [ensemble.mineral_labels[index] for index in reference_indices]
    detection_indices = wavelength_indices(wavelengths_nm, group["detection_windows_nm"])
    classification_indices = wavelength_indices(wavelengths_nm, group["classification_windows_nm"])
    references = ensemble.spectra[reference_indices]
    if not np.all(np.isfinite(references[:, detection_indices])):
        raise ValueError(f"Selected {name} references contain gaps in the SAM detection windows")
    if not np.all(np.isfinite(references[:, classification_indices])):
        raise ValueError(f"Selected {name} references contain gaps in the classification windows")
    reference_absorption, _ = _continuum_absorption(
        references[:, classification_indices], wavelengths_nm[classification_indices]
    )
    return {
        "config": group,
        "name": name,
        "mineral_order": mineral_order,
        "reference_indices": reference_indices,
        "reference_minerals": reference_minerals,
        "detection_indices": detection_indices,
        "classification_indices": classification_indices,
        "references": references,
        "reference_absorption": reference_absorption,
    }


def _group_sam_scores(cube: np.ndarray, valid_mask: np.ndarray, prepared: dict[str, Any]) -> np.ndarray:
    result = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    indices = prepared["detection_indices"]
    valid = valid_mask & np.all(np.isfinite(cube[..., indices]), axis=-1) & np.all(cube[..., indices] > 0, axis=-1)
    if not np.any(valid):
        return result
    pixels = np.asarray(cube[..., indices][valid], dtype=np.float64)
    references = np.asarray(prepared["references"][:, indices], dtype=np.float64)
    angles = spectral_angles(pixels, references)
    mineral_scores = aggregate_reference_scores(
        angles,
        prepared["reference_minerals"],
        prepared["mineral_order"],
        best_k=int(prepared["config"].get("detection_best_k", 1)),
    )
    result[valid] = np.min(mineral_scores, axis=1).astype(np.float32)
    return result


def _subclass_scores_for_cube(
    cube: np.ndarray,
    candidate: np.ndarray,
    prepared: dict[str, Any],
    wavelengths_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group = prepared["config"]
    indices = prepared["classification_indices"]
    valid = candidate & np.all(np.isfinite(cube[..., indices]), axis=-1) & np.all(cube[..., indices] > 0, axis=-1)
    if not np.any(valid):
        return (
            valid,
            np.empty((0, len(prepared["mineral_order"])), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        )
    pixel_spectra = cube[..., indices][valid]
    pixel_absorption, depth = _continuum_absorption(pixel_spectra, wavelengths_nm[indices])
    reference_scores = _subclass_reference_scores(
        prepared["name"],
        pixel_absorption,
        prepared["reference_absorption"],
        wavelengths_nm[indices],
        group.get("score_weights", {}),
    )
    mineral_scores = aggregate_reference_scores(
        reference_scores,
        prepared["reference_minerals"],
        prepared["mineral_order"],
        best_k=int(group.get("classification_best_k", 2)),
    )
    pixel_features = diagnostic_feature_matrix(prepared["name"], pixel_absorption, wavelengths_nm[indices])
    return valid, mineral_scores, depth, pixel_features


def _class_feature_gate(
    prepared: dict[str, Any],
    label_indices: np.ndarray,
    depth: np.ndarray,
    pixel_features: np.ndarray,
) -> np.ndarray:
    gates = prepared["config"].get("class_feature_gates", {})
    passed = np.ones(label_indices.size, dtype=bool)
    if not gates:
        return passed
    center_metadata = {
        "carbonates": (0, 2280.0, 2360.0),
        "sulfates": (4, 1880.0, 1995.0),
        "clays": (3, 2160.0, 2235.0),
    }
    feature_index, lower, upper = center_metadata[prepared["name"]]
    center_nm = lower + pixel_features[:, feature_index] * (upper - lower)
    for mineral_index, mineral in enumerate(prepared["mineral_order"]):
        selected = label_indices == mineral_index
        gate = gates.get(mineral, gates.get(MINERAL_DISPLAY_NAMES[mineral], {}))
        if not gate or not np.any(selected):
            continue
        if "center_window_nm" in gate:
            center_lower, center_upper = [float(value) for value in gate["center_window_nm"]]
            passed[selected] &= (center_nm[selected] >= center_lower) & (center_nm[selected] <= center_upper)
        if "minimum_absorption_depth" in gate:
            passed[selected] &= depth[selected] >= float(gate["minimum_absorption_depth"])
    return passed


def _neighborhood_counts(mask: np.ndarray, size: int = 3) -> np.ndarray:
    radius = size // 2
    padded = np.pad(np.asarray(mask, dtype=np.uint8), radius, mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return np.sum(windows, axis=(-2, -1))


def _local_tie_relabel(
    labels: np.ndarray,
    score_cube: np.ndarray,
    *,
    tolerance: float,
    dominant_support: int = 4,
) -> tuple[np.ndarray, int]:
    output = np.asarray(labels, dtype=np.uint8).copy()
    class_count = score_cube.shape[0]
    support = np.stack([_neighborhood_counts(output == class_id) for class_id in range(1, class_count + 1)])
    dominant = np.argmax(support, axis=0)
    dominant_count = np.max(support, axis=0)
    rows, columns = np.indices(output.shape)
    current_index = np.maximum(output.astype(np.int64) - 1, 0)
    current_support = support[current_index, rows, columns]
    current_score = score_cube[current_index, rows, columns]
    dominant_score = score_cube[dominant, rows, columns]
    change = (
        (output > 0)
        & (dominant != current_index)
        & (current_support <= 2)
        & (dominant_count >= dominant_support)
        & np.isfinite(dominant_score)
        & (dominant_score <= current_score + tolerance)
    )
    output[change] = dominant[change].astype(np.uint8) + 1
    return output, int(np.count_nonzero(change))


def _classification_output(labels: np.ndarray, mask: np.ndarray, class_count: int) -> np.ndarray:
    output = np.asarray(labels, dtype=np.uint8).copy()
    output[~mask] = class_count + 1
    return output


def _candidate_output(candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros(candidate.shape, dtype=np.uint8)
    output[candidate] = 1
    output[~mask] = 2
    return output


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"p01": None, "p05": None, "p50": None, "p95": None, "p99": None}
    result = np.percentile(finite, [1, 5, 50, 95, 99])
    return {key: float(value) for key, value in zip(("p01", "p05", "p50", "p95", "p99"), result)}


def run_v3(
    config: dict[str, Any],
    output_dir: str | Path,
    start_line: int = 0,
    stop_line: int | None = None,
    progress: Progress = _default_progress,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    v3_config = config.get("v3")
    if not isinstance(v3_config, dict):
        raise ValueError("The configuration does not contain a V3 section")

    image = EnviDataset(config["analysis_image"])
    wavelengths = image.info.wavelengths_nm
    if wavelengths is None:
        image.close()
        raise ValueError("The analysis image requires an ENVI wavelength vector")
    start = int(start_line)
    stop = image.info.lines if stop_line is None else int(stop_line)
    if start < 0 or stop > image.info.lines or start >= stop:
        image.close()
        raise ValueError(f"Invalid output line interval [{start}, {stop})")
    chunk_rows = int(config.get("chunk_rows", 64))

    progress("v3-mask", 0.01, "Reading the full-depth core mask")
    mask_dataset = EnviDataset(config["analysis_mask"])
    default_mask_bands = [
        min(20, mask_dataset.info.bands - 1),
        min(mask_dataset.info.bands // 2, mask_dataset.info.bands - 1),
        min(190, mask_dataset.info.bands - 1),
    ]
    mask_bands = sorted(set(int(value) for value in v3_config.get("mask_bands", default_mask_bands)))
    full_mask = derive_mask(mask_dataset, bands=mask_bands, chunk_rows=chunk_rows)
    mask_dataset.close()

    progress("v3-library", 0.04, "Building the automatic pure-mineral reference ensemble")
    ensemble_config = v3_config.get("library_ensemble", {})
    scene_config = ensemble_config.get("scene_adaptation", {})
    scene_cube = None
    scene_mask = None
    if bool(scene_config.get("enabled", False)):
        scene_start, scene_stop = [int(value) for value in scene_config.get("selection_lines", [start, stop])]
        if scene_start < 0 or scene_stop > image.info.lines or scene_start >= scene_stop:
            raise ValueError(f"Invalid library scene-selection lines [{scene_start}, {scene_stop})")
        scene_cube = np.asarray(image.read_rows(scene_start, scene_stop), dtype=np.float32)
        scene_mask = full_mask[scene_start:scene_stop]
    ensemble = build_library_ensemble(
        v3_config["spectral_library_root"],
        wavelengths,
        representatives_per_mineral=int(ensemble_config.get("representatives_per_mineral", 4)),
        dedup_angle_rad=float(ensemble_config.get("dedup_angle_rad", 0.02)),
        scene_cube=scene_cube,
        scene_mask=scene_mask,
        scene_support_fraction=float(scene_config.get("support_fraction", 0.05)),
    )
    del scene_cube, scene_mask
    ensemble_paths = write_ensemble_artifacts(ensemble, output / "library_ensemble")
    prepared_groups = [_prepare_group(group, ensemble, wavelengths) for group in v3_config["groups"]]

    progress("v3-calibration", 0.08, "Calibrating SAM distributions for every detector column")
    calibration_scores = {
        prepared["name"]: np.full((image.info.lines, image.info.samples), np.nan, dtype=np.float32)
        for prepared in prepared_groups
    }
    input_is_smoothed = bool(config.get("analysis_input_is_smoothed", False))
    for row_start, row_stop, cube in image.iter_rows(chunk_rows=chunk_rows):
        values = np.asarray(cube, dtype=np.float64)
        if not input_is_smoothed:
            values = savgol_smooth(
                values,
                int(config.get("sg_window", 11)),
                int(config.get("sg_polyorder", 2)),
            )
        valid = full_mask[row_start:row_stop]
        for prepared in prepared_groups:
            calibration_scores[prepared["name"]][row_start:row_stop] = _group_sam_scores(values, valid, prepared)
        fraction = row_stop / image.info.lines
        progress("v3-calibration", 0.08 + 0.42 * fraction, f"Calibrated lines {row_start}-{row_stop}")

    thresholds: dict[str, np.ndarray] = {}
    calibration_counts: dict[str, np.ndarray] = {}
    for prepared in prepared_groups:
        group = prepared["config"]
        name = prepared["name"]
        thresholds[name], calibration_counts[name] = column_percentile_thresholds(
            calibration_scores[name],
            full_mask,
            float(group.get("column_percentile", 0.05)),
            min_samples=int(group.get("column_min_samples", 20)),
        )

    progress("v3-domain-calibration", 0.51, "Estimating full-depth mineral score offsets")
    subclass_samples: dict[str, list[np.ndarray]] = {prepared["name"]: [] for prepared in prepared_groups}
    for row_start, row_stop, cube in image.iter_rows(chunk_rows=chunk_rows):
        values = np.asarray(cube, dtype=np.float64)
        if not input_is_smoothed:
            values = savgol_smooth(
                values,
                int(config.get("sg_window", 11)),
                int(config.get("sg_polyorder", 2)),
            )
        mask_chunk = full_mask[row_start:row_stop]
        for prepared in prepared_groups:
            group = prepared["config"]
            name = prepared["name"]
            score = calibration_scores[name][row_start:row_stop]
            candidate = (
                mask_chunk
                & np.isfinite(score)
                & (score <= float(group.get("absolute_sam_threshold_rad", np.inf)))
                & (score <= thresholds[name][None, :])
            )
            _, mineral_scores, depth, _ = _subclass_scores_for_cube(values, candidate, prepared, wavelengths)
            usable = (
                np.all(np.isfinite(mineral_scores), axis=1)
                & np.isfinite(depth)
                & (depth >= float(group.get("minimum_absorption_depth", 0.01)))
            )
            if np.any(usable):
                subclass_samples[name].append(mineral_scores[usable])
        fraction = row_stop / image.info.lines
        progress("v3-domain-calibration", 0.51 + 0.15 * fraction, f"Calibrated subclass offsets through line {row_stop}")

    subclass_centers: dict[str, np.ndarray] = {}
    for prepared in prepared_groups:
        name = prepared["name"]
        if not subclass_samples[name]:
            raise ValueError(f"No valid full-depth subclass calibration samples for {name}")
        subclass_centers[name] = np.median(np.vstack(subclass_samples[name]), axis=0)

    height = stop - start
    output_mask = full_mask[start:stop]
    buffers: dict[str, dict[str, Any]] = {}
    for prepared in prepared_groups:
        group = prepared["config"]
        name = prepared["name"]
        score = calibration_scores[name][start:stop]
        candidate = (
            output_mask
            & np.isfinite(score)
            & (score <= float(group.get("absolute_sam_threshold_rad", np.inf)))
            & (score <= thresholds[name][None, :])
        )
        mineral_count = len(prepared["mineral_order"])
        buffers[name] = {
            "candidate": candidate,
            "labels": np.zeros((height, image.info.samples), dtype=np.uint8),
            "score_cube": np.full((mineral_count, height, image.info.samples), np.nan, dtype=np.float32),
            "raw_score_cube": np.full((mineral_count, height, image.info.samples), np.nan, dtype=np.float32),
            "depth": np.full((height, image.info.samples), np.nan, dtype=np.float32),
            "margin": np.full((height, image.info.samples), np.nan, dtype=np.float32),
            "tie_count": 0,
            "tie_changed": 0,
        }

    progress("v3-classification", 0.68, "Applying domain-calibrated unlocked mineral classification")
    for row_start, row_stop, cube in image.iter_rows(chunk_rows=chunk_rows, start=start, stop=stop):
        values = np.asarray(cube, dtype=np.float64)
        if not input_is_smoothed:
            values = savgol_smooth(
                values,
                int(config.get("sg_window", 11)),
                int(config.get("sg_polyorder", 2)),
            )
        local_slice = slice(row_start - start, row_stop - start)
        for prepared in prepared_groups:
            group = prepared["config"]
            name = prepared["name"]
            buffer = buffers[name]
            candidate = buffer["candidate"][local_slice]
            valid, raw_mineral_scores, depth, pixel_features = _subclass_scores_for_cube(
                values, candidate, prepared, wavelengths
            )
            if not np.any(valid):
                continue
            calibration_strength = float(group.get("subclass_domain_calibration_strength", 0.0))
            mineral_scores = raw_mineral_scores - calibration_strength * subclass_centers[name][None, :]
            label_indices, tie_count, tie_changed = balanced_tie_labels(
                mineral_scores,
                prepared["mineral_order"],
                tie_tolerance=float(group.get("tie_tolerance", 0.025)),
                tie_bias={str(key).casefold(): float(value) for key, value in group.get("tie_bias", {}).items()},
            )
            sorted_scores = np.sort(mineral_scores, axis=1)
            margin = sorted_scores[:, 1] - sorted_scores[:, 0]
            accepted = (
                (label_indices >= 0)
                & np.isfinite(depth)
                & (depth >= float(group.get("minimum_absorption_depth", 0.01)))
                & (sorted_scores[:, 0] <= float(group.get("maximum_subclass_score", np.inf)))
                & _class_feature_gate(prepared, label_indices, depth, pixel_features)
            )

            label_view = buffer["labels"][local_slice]
            label_view[valid] = np.where(accepted, label_indices + 1, 0).astype(np.uint8)
            depth_view = buffer["depth"][local_slice]
            depth_view[valid] = depth.astype(np.float32)
            margin_view = buffer["margin"][local_slice]
            margin_view[valid] = margin.astype(np.float32)
            score_view = buffer["score_cube"][:, local_slice, :]
            raw_score_view = buffer["raw_score_cube"][:, local_slice, :]
            for mineral_index in range(mineral_scores.shape[1]):
                score_view[mineral_index][valid] = mineral_scores[:, mineral_index].astype(np.float32)
                raw_score_view[mineral_index][valid] = raw_mineral_scores[:, mineral_index].astype(np.float32)
            buffer["tie_count"] += int(np.count_nonzero((tie_count > 1) & accepted))
            buffer["tie_changed"] += int(np.count_nonzero(tie_changed & accepted))
        fraction = (row_stop - start) / height
        progress("v3-classification", 0.68 + 0.10 * fraction, f"Classified output lines {row_start}-{row_stop}")

    run_summary: dict[str, Any] = {
        "version": "V3 Automatic Library Ensemble",
        "profile": str(v3_config.get("profile", "custom")),
        "analysis_image": str(image.info.data_path),
        "analysis_mask": str(config["analysis_mask"]),
        "spectral_library_root": str(v3_config["spectral_library_root"]),
        "library_ensemble_config": ensemble_config,
        "ensemble_artifacts": ensemble_paths,
        "calibration_lines": [0, image.info.lines],
        "output_lines": [start, stop],
        "mask_pixels": int(np.count_nonzero(output_mask)),
        "groups": {},
    }

    progress("v3-spatial", 0.78, "Applying column-aware and mild spatial cleanup")
    for group_index, prepared in enumerate(prepared_groups):
        group = prepared["config"]
        name = prepared["name"]
        buffer = buffers[name]
        group_dir = output / name
        group_dir.mkdir(parents=True, exist_ok=True)
        mineral_order = prepared["mineral_order"]
        class_names = ["Unclassified", *[MINERAL_DISPLAY_NAMES[mineral] for mineral in mineral_order], "Masked Pixels"]
        before_spatial = buffer["labels"].copy()
        relabeled, local_relabels = _local_tie_relabel(
            before_spatial,
            buffer["score_cube"],
            tolerance=float(group.get("local_relabel_tolerance", 0.02)),
            dominant_support=int(group.get("local_dominant_support", 4)),
        )

        stripe_config = group.get("spatial_cleanup", {})
        stripe_noise = np.zeros_like(relabeled, dtype=bool)
        cleaned = np.zeros_like(relabeled)
        directional_removed = 0
        elongated_removed = 0
        component_removed = 0
        for class_id in range(1, len(mineral_order) + 1):
            class_mask = relabeled == class_id
            if bool(stripe_config.get("directional_filter_enabled", True)):
                detected = directional_stripe_mask(
                    class_mask,
                    output_mask,
                    vertical_window=int(stripe_config.get("vertical_window", 61)),
                    min_vertical_density=float(stripe_config.get("min_vertical_density", 0.10)),
                    column_ratio=float(stripe_config.get("column_ratio", 2.0)),
                    column_excess=float(stripe_config.get("column_excess", 0.03)),
                    max_lateral_support=int(stripe_config.get("max_lateral_support", 2)),
                    neighborhood_radius=int(stripe_config.get("neighborhood_radius", 6)),
                )
                stripe_noise |= detected
                directional_removed += int(np.count_nonzero(detected))
                class_mask &= ~detected
            if bool(stripe_config.get("remove_elongated", True)):
                before_count = int(np.count_nonzero(class_mask))
                class_mask = remove_elongated_components(
                    class_mask,
                    min_height=int(stripe_config.get("min_line_height", 15)),
                    max_width=int(stripe_config.get("max_line_width", 3)),
                    min_aspect=float(stripe_config.get("min_aspect", 5.0)),
                )
                elongated_removed += before_count - int(np.count_nonzero(class_mask))
            before_count = int(np.count_nonzero(class_mask))
            class_mask = clean_binary_mask(
                class_mask,
                median_size=1,
                min_component=int(stripe_config.get("min_component", 2)),
            )
            component_removed += before_count - int(np.count_nonzero(class_mask))
            cleaned[class_mask] = class_id

        write_envi(
            _candidate_output(buffer["candidate"], output_mask),
            group_dir / "column_calibrated_candidates.dat",
            class_names=["Unclassified", "Accepted group candidate", "Masked Pixels"],
            description=f"{name} V3 group SAM candidates after full-depth column percentile calibration",
        )
        write_envi(
            _classification_output(before_spatial, output_mask, len(mineral_order)),
            group_dir / "v3_classes_before_spatial.dat",
            class_names=class_names,
            description=f"{name} V3 unlocked mineral classes before spatial cleanup",
        )
        write_envi(
            _classification_output(cleaned, output_mask, len(mineral_order)),
            group_dir / "v3_final_classes.dat",
            class_names=class_names,
            description=f"{name} V3 Automatic Library Ensemble final classes",
        )
        write_envi(
            buffer["score_cube"].astype(np.float32),
            group_dir / "v3_mineral_scores.dat",
            band_names=[MINERAL_DISPLAY_NAMES[mineral] for mineral in mineral_order],
            description=f"{name} domain-calibrated ensemble subclass scores; lower is better",
        )
        write_envi(
            buffer["raw_score_cube"].astype(np.float32),
            group_dir / "v3_raw_mineral_scores.dat",
            band_names=[MINERAL_DISPLAY_NAMES[mineral] for mineral in mineral_order],
            description=f"{name} raw ensemble subclass scores before domain-offset calibration",
        )
        write_envi(
            calibration_scores[name][start:stop].astype(np.float32),
            group_dir / "group_sam_score.dat",
            description=f"{name} best ensemble group SAM score in radians",
        )
        threshold_image = np.broadcast_to(thresholds[name][None, :], (height, image.info.samples)).astype(np.float32)
        write_envi(
            threshold_image,
            group_dir / "column_sam_threshold.dat",
            description=f"{name} full-depth per-column SAM percentile threshold in radians",
        )
        write_envi(buffer["depth"], group_dir / "absorption_depth.dat", description=f"{name} continuum-removed depth")
        write_envi(buffer["margin"], group_dir / "classification_margin.dat", description=f"{name} best versus second score margin")
        write_envi(stripe_noise.astype(np.uint8), group_dir / "stripe_noise_mask.dat", description=f"{name} V3 stripe noise mask")

        candidate_column_valid = np.maximum(np.sum(output_mask, axis=0), 1)
        candidate_column_density = np.sum(buffer["candidate"], axis=0) / candidate_column_valid
        before_counts = {
            MINERAL_DISPLAY_NAMES[mineral]: int(np.count_nonzero(before_spatial == index + 1))
            for index, mineral in enumerate(mineral_order)
        }
        final_counts = {
            MINERAL_DISPLAY_NAMES[mineral]: int(np.count_nonzero(cleaned == index + 1))
            for index, mineral in enumerate(mineral_order)
        }
        calibration_record = {
            "column_percentile": float(group.get("column_percentile", 0.05)),
            "absolute_sam_threshold_rad": float(group.get("absolute_sam_threshold_rad", np.inf)),
            "threshold_percentiles_rad": _percentiles(thresholds[name]),
            "sample_count_percentiles": _percentiles(calibration_counts[name]),
            "output_candidate_column_density": {
                "maximum": float(np.max(candidate_column_density)),
                "median": float(np.median(candidate_column_density)),
                "columns_at_or_above_10_percent": int(np.count_nonzero(candidate_column_density >= 0.10)),
            },
            "thresholds_rad": thresholds[name].tolist(),
            "sample_counts": calibration_counts[name].tolist(),
        }
        _save_json(calibration_record, group_dir / "column_calibration.json")
        group_summary = {
            "minerals": [MINERAL_DISPLAY_NAMES[mineral] for mineral in mineral_order],
            "selected_reference_count_per_mineral": ensemble.group_reference_counts[name],
            "selected_references": [
                ensemble.spectrum_names[index] for index in prepared["reference_indices"]
            ],
            "detection_windows_nm": group["detection_windows_nm"],
            "classification_windows_nm": group["classification_windows_nm"],
            "score_weights": group.get("score_weights", {}),
            "subclass_domain_calibration": {
                "strength": float(group.get("subclass_domain_calibration_strength", 0.0)),
                "full_depth_median_scores": {
                    MINERAL_DISPLAY_NAMES[mineral]: float(subclass_centers[name][index])
                    for index, mineral in enumerate(mineral_order)
                },
                "sample_pixels": int(sum(values.shape[0] for values in subclass_samples[name])),
            },
            "thresholds": {
                "absolute_sam_rad": float(group.get("absolute_sam_threshold_rad", np.inf)),
                "column_percentile": float(group.get("column_percentile", 0.05)),
                "minimum_absorption_depth": float(group.get("minimum_absorption_depth", 0.01)),
                "class_feature_gates": group.get("class_feature_gates", {}),
                "tie_tolerance": float(group.get("tie_tolerance", 0.025)),
                "tie_bias": group.get("tie_bias", {}),
            },
            "counts": {
                "column_calibrated_candidates": int(np.count_nonzero(buffer["candidate"])),
                "classified_before_spatial": int(np.count_nonzero(before_spatial)),
                "local_tie_relabels": local_relabels,
                "near_tie_candidates": int(buffer["tie_count"]),
                "near_tie_bias_changes": int(buffer["tie_changed"]),
                "directional_stripe_removed": directional_removed,
                "elongated_component_removed": elongated_removed,
                "small_component_removed": component_removed,
                "classified_final": int(np.count_nonzero(cleaned)),
            },
            "mineral_counts_before_spatial": before_counts,
            "mineral_counts_final": final_counts,
            "score_percentiles": {
                "group_sam_rad": _percentiles(calibration_scores[name][start:stop][output_mask]),
                "absorption_depth": _percentiles(buffer["depth"][buffer["candidate"]]),
                "classification_margin": _percentiles(buffer["margin"][before_spatial > 0]),
            },
            "column_calibration": calibration_record,
            "spatial_cleanup": stripe_config,
        }
        _save_json(group_summary, group_dir / "summary.json")
        run_summary["groups"][name] = group_summary
        progress("v3-spatial", 0.78 + 0.16 * (group_index + 1) / len(prepared_groups), f"Finished {name}")

    with (output / "mineral_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "mineral", "pixels_before_spatial", "pixels_final"])
        for group_name, summary in run_summary["groups"].items():
            for mineral, count in summary["mineral_counts_final"].items():
                writer.writerow([group_name, mineral, summary["mineral_counts_before_spatial"][mineral], count])
    _save_json(run_summary, output / "summary.json")

    from .preview import make_v3_previews

    make_v3_previews(config, output, start, stop)
    image.close()
    progress("v3", 1.0, "V3 Automatic Library Ensemble pilot complete")
    return run_summary
