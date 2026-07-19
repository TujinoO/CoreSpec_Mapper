from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .algorithms import continuum_remove, spectral_angles, spectral_feature_fit
from .catalog import GroupDefinition, MineralDefinition


def unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    return np.divide(array, norms[:, None], out=np.zeros_like(array), where=norms[:, None] > 1e-12)


def continuum_absorption(spectra: np.ndarray, wavelengths_nm: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    removed = continuum_remove(np.asarray(spectra, dtype=np.float64), wavelengths)
    absorption = np.clip(1.0 - removed, 0.0, None)
    depth = np.max(absorption, axis=1)
    return absorption, depth


def feature_values(
    group: GroupDefinition,
    absorption: np.ndarray,
    wavelengths_nm: Sequence[float],
) -> dict[str, np.ndarray]:
    values = np.asarray(absorption, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    full_depth = np.maximum(np.max(values, axis=1), 1e-12)
    result: dict[str, np.ndarray] = {}
    for feature in group.features:
        lower, upper = feature.window_nm
        selected = (wavelengths >= lower) & (wavelengths <= upper)
        if not np.any(selected):
            result[feature.feature_id] = np.full(values.shape[0], np.nan)
            continue
        subset = values[:, selected]
        if feature.kind == "center_nm":
            indices = np.argmax(subset, axis=1)
            result[feature.feature_id] = wavelengths[selected][indices]
        elif feature.kind == "depth_ratio":
            result[feature.feature_id] = np.max(subset, axis=1) / full_depth
        elif feature.kind == "depth":
            result[feature.feature_id] = np.max(subset, axis=1)
        else:
            raise ValueError(f"Unsupported SWIR feature type: {feature.kind}")
    return result


def feature_matrix(
    group: GroupDefinition,
    absorption: np.ndarray,
    wavelengths_nm: Sequence[float],
) -> np.ndarray:
    values = feature_values(group, absorption, wavelengths_nm)
    columns: list[np.ndarray] = []
    for feature in group.features:
        column = np.asarray(values[feature.feature_id], dtype=np.float64)
        if feature.kind == "center_nm":
            lower, upper = feature.window_nm
            column = (column - lower) / max(upper - lower, 1.0)
        columns.append(column)
    return np.column_stack(columns) if columns else np.empty((np.asarray(absorption).shape[0], 0))


@dataclass(frozen=True)
class ReferenceEvidence:
    total: np.ndarray
    shape: np.ndarray
    derivative: np.ndarray
    feature: np.ndarray
    fit: np.ndarray
    scale: np.ndarray
    rms: np.ndarray
    fit_quality: np.ndarray


def reference_evidence(
    group: GroupDefinition,
    pixel_absorption: np.ndarray,
    reference_absorption: np.ndarray,
    wavelengths_nm: Sequence[float],
) -> ReferenceEvidence:
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    pixels = np.asarray(pixel_absorption, dtype=np.float64)
    references = np.asarray(reference_absorption, dtype=np.float64)
    shape = spectral_angles(unit_rows(pixels), unit_rows(references))
    derivative = spectral_angles(
        unit_rows(np.gradient(pixels, wavelengths, axis=1)),
        unit_rows(np.gradient(references, wavelengths, axis=1)),
    )
    pixel_features = feature_matrix(group, pixels, wavelengths)
    reference_features = feature_matrix(group, references, wavelengths)
    if pixel_features.shape[1]:
        feature = np.sqrt(np.mean((pixel_features[:, None, :] - reference_features[None, :, :]) ** 2, axis=2))
    else:
        feature = np.zeros_like(shape)
    sff = spectral_feature_fit(pixels, references)
    pixel_scale = np.maximum(np.mean(np.abs(pixels), axis=1)[:, None], 1e-8)
    fit = np.clip(sff.rms / pixel_scale, 0.0, np.pi)
    weights = group.score_weights
    total = (
        float(weights.get("shape", 0.60)) * shape
        + float(weights.get("derivative", 0.20)) * derivative
        + float(weights.get("features", 0.20)) * feature
        + float(weights.get("fit", 0.0)) * fit
    )
    return ReferenceEvidence(total, shape, derivative, feature, fit, sff.scale, sff.rms, sff.quality)


def aggregate_reference_scores(
    scores: np.ndarray,
    reference_minerals: Sequence[str],
    mineral_order: Sequence[str],
    *,
    best_k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.where(np.isfinite(scores), scores, np.inf)
    labels = np.asarray(reference_minerals, dtype=object)
    mineral_scores = np.full((values.shape[0], len(mineral_order)), np.inf, dtype=np.float64)
    consensus = np.zeros_like(mineral_scores)
    for mineral_index, mineral in enumerate(mineral_order):
        selected = np.flatnonzero(labels == mineral)
        if selected.size == 0:
            continue
        subset = values[:, selected]
        count = min(max(1, int(best_k)), selected.size)
        ordered = np.sort(subset, axis=1)
        mineral_scores[:, mineral_index] = np.mean(ordered[:, :count], axis=1)
        best = ordered[:, 0]
        tolerance = np.maximum(0.025, best * 0.20)
        consensus[:, mineral_index] = np.mean(subset <= best[:, None] + tolerance[:, None], axis=1)
    return mineral_scores, consensus


def mineral_feature_gate(
    mineral: MineralDefinition,
    expert_id: str,
    policy: str,
    depth: np.ndarray,
    features: Mapping[str, np.ndarray],
) -> np.ndarray:
    passed = np.ones(np.asarray(depth).shape, dtype=bool)
    settings = mineral.experts.get(expert_id, {})
    gate = settings.get("feature_gate", {})
    center_id = gate.get("center_feature")
    if center_id and center_id in features:
        window = gate.get("center_window_nm_by_policy", {}).get(policy, gate.get("center_window_nm"))
        if window:
            lower, upper = (float(value) for value in window)
            passed &= (features[center_id] >= lower) & (features[center_id] <= upper)
    minimum_depth = gate.get("minimum_absorption_depth_by_policy", {}).get(policy)
    if minimum_depth is not None:
        passed &= np.asarray(depth) >= float(minimum_depth)
    for feature_id, minimum in gate.get("minimum_feature_values", {}).items():
        if feature_id in features:
            passed &= features[feature_id] >= float(minimum)
        else:
            passed &= False
    return passed
