from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .algorithms import continuum_remove, continuum_remove_linear, spectral_angles, spectral_feature_fit
from .catalog import GroupDefinition, MineralDefinition


def unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    return np.divide(array, norms[:, None], out=np.zeros_like(array), where=norms[:, None] > 1e-12)


def continuum_absorption(
    spectra: np.ndarray,
    wavelengths_nm: Sequence[float],
    *,
    method: str = "upper_hull",
) -> tuple[np.ndarray, np.ndarray]:
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    values = np.asarray(spectra, dtype=np.float64)
    if method == "upper_hull":
        removed = continuum_remove(values, wavelengths)
    elif method in {"linear", "segmented_linear"}:
        removed = continuum_remove_linear(values, wavelengths)
    else:
        raise ValueError(f"Unsupported continuum method: {method}")
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
        # A catalog feature can be unavailable after the sensor bad-band gate
        # removes its whole diagnostic window.  The former plain mean allowed
        # that optional NaN feature to poison every pixel/reference score in the
        # group (observed for NC-1 carbonates at 1945 nm).  Compare only jointly
        # observable features and leave the feature component unavailable when
        # no feature is shared; the total below then renormalises the remaining
        # physical evidence instead of treating missing data as a perfect fit.
        differences = pixel_features[:, None, :] - reference_features[None, :, :]
        jointly_finite = np.isfinite(differences)
        squared_sum = np.sum(np.where(jointly_finite, differences**2, 0.0), axis=2)
        shared_count = np.sum(jointly_finite, axis=2)
        feature = np.sqrt(
            np.divide(
                squared_sum,
                shared_count,
                out=np.full(squared_sum.shape, np.nan, dtype=np.float64),
                where=shared_count > 0,
            )
        )
    else:
        feature = np.zeros_like(shape)
    weights = group.score_weights
    if float(weights.get("fit", 0.0)) > 0.0:
        sff = spectral_feature_fit(pixels, references)
        pixel_scale = np.maximum(np.mean(np.abs(pixels), axis=1)[:, None], 1e-8)
        fit = np.clip(sff.rms / pixel_scale, 0.0, np.pi)
        scale, rms, fit_quality = sff.scale, sff.rms, sff.quality
    else:
        # V5's default experts do not weight SFF.  Avoid the full pixel-by-
        # reference least-squares matrix and expose neutral diagnostic planes.
        fit = np.zeros_like(shape)
        scale = np.zeros_like(shape)
        rms = np.full_like(shape, np.nan)
        fit_quality = np.zeros_like(shape)
    components = np.stack((shape, derivative, feature, fit), axis=0)
    component_weights = np.asarray(
        [
            float(weights.get("shape", 0.60)),
            float(weights.get("derivative", 0.20)),
            float(weights.get("features", 0.20)),
            float(weights.get("fit", 0.0)),
        ],
        dtype=np.float64,
    )[:, None, None]
    usable = np.isfinite(components) & (component_weights > 0.0)
    available_weight = np.sum(np.where(usable, component_weights, 0.0), axis=0)
    weighted_sum = np.sum(np.where(usable, components * component_weights, 0.0), axis=0)
    total = np.divide(
        weighted_sum,
        available_weight,
        out=np.full(shape.shape, np.nan, dtype=np.float64),
        where=available_weight > 0.0,
    )
    return ReferenceEvidence(total, shape, derivative, feature, fit, scale, rms, fit_quality)


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
    gate_override: Mapping[str, object] | None = None,
) -> np.ndarray:
    passed = np.ones(np.asarray(depth).shape, dtype=bool)
    settings = mineral.experts.get(expert_id, {})
    gate = dict(settings.get("feature_gate", {}))
    if gate_override:
        for key, value in gate_override.items():
            if key == "minimum_feature_values" and isinstance(value, Mapping):
                merged = dict(gate.get(key, {}))
                merged.update(value)
                gate[key] = merged
            else:
                gate[key] = value
    center_id = gate.get("center_feature")
    if center_id and center_id in features:
        window = (
            gate_override.get("center_window_nm")
            if gate_override and gate_override.get("center_window_nm") is not None
            else gate.get("center_window_nm_by_policy", {}).get(policy, gate.get("center_window_nm"))
        )
        if window:
            lower, upper = (float(value) for value in window)
            passed &= (features[center_id] >= lower) & (features[center_id] <= upper)
    minimum_depth = gate.get("minimum_absorption_depth_by_policy", {}).get(policy)
    if minimum_depth is not None:
        passed &= np.asarray(depth) >= float(minimum_depth)
    minimum_feature_checks: list[np.ndarray] = []
    for feature_id, minimum in gate.get("minimum_feature_values", {}).items():
        if feature_id in features:
            minimum_feature_checks.append(features[feature_id] >= float(minimum))
        else:
            minimum_feature_checks.append(np.zeros_like(passed))
    if minimum_feature_checks:
        match_mode = str(gate.get("minimum_feature_match", "all")).casefold()
        if match_mode == "any":
            passed &= np.logical_or.reduce(tuple(minimum_feature_checks))
        else:
            passed &= np.logical_and.reduce(tuple(minimum_feature_checks))
    return passed
