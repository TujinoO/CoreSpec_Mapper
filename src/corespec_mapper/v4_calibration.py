from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .catalog import GroupDefinition
from .v4_models import SensorCapabilityCard


POLICY_ORDER = ("conservative", "balanced", "sensitive")


def calibrated_column_thresholds(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    percentile_fraction: float,
    *,
    minimum_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    sample = values[valid]
    if values.ndim != 2 or values.shape != valid.shape or sample.size == 0:
        raise ValueError("Column calibration requires finite scores in a matching two-dimensional mask")
    percentile = float(percentile_fraction) * 100.0
    scene_threshold = float(np.percentile(sample, percentile))
    thresholds = np.full(values.shape[1], scene_threshold, dtype=np.float64)
    counts = np.zeros(values.shape[1], dtype=np.int32)
    reliable = np.zeros(values.shape[1], dtype=bool)
    for column in range(values.shape[1]):
        column_values = values[valid[:, column], column]
        counts[column] = column_values.size
        if column_values.size >= minimum_samples:
            thresholds[column] = float(np.percentile(column_values, percentile))
            reliable[column] = True
    if np.any(reliable):
        reliable_indices = np.flatnonzero(reliable)
        missing = np.flatnonzero(~reliable)
        thresholds[missing] = np.interp(missing, reliable_indices, thresholds[reliable_indices])
        padded = np.pad(thresholds, 2, mode="edge")
        thresholds = np.asarray([np.median(padded[index : index + 5]) for index in range(thresholds.size)])
    return thresholds, counts, scene_threshold


def estimate_relative_spectral_noise(sample_spectra: np.ndarray) -> float:
    values = np.asarray(sample_spectra, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 5:
        return 0.0
    second = np.diff(values, n=2, axis=1) / np.sqrt(6.0)
    noise = 1.4826 * np.median(np.abs(second - np.median(second, axis=1, keepdims=True)), axis=1)
    signal = np.maximum(np.median(np.abs(values), axis=1), 1e-8)
    relative = noise / signal
    finite = relative[np.isfinite(relative)]
    return 0.0 if finite.size == 0 else float(np.clip(np.median(finite), 0.0, 0.05))


def resolve_group_policies(
    group: GroupDefinition,
    card: SensorCapabilityCard,
    scores: np.ndarray,
    sample_mask: np.ndarray,
    *,
    relative_noise: float,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    minimum_column_samples: int = 20,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    overrides = overrides or {}
    noise_multiplier = {"conservative": 3.0, "balanced": 2.2, "sensitive": 1.5}
    for policy in POLICY_ORDER:
        settings = dict(group.policies[policy])
        settings.update(overrides.get(policy, {}))
        percentile = float(settings["column_percentile"])
        thresholds, counts, scene_threshold = calibrated_column_thresholds(
            scores,
            sample_mask,
            percentile,
            minimum_samples=minimum_column_samples,
        )
        scene_fraction = float(settings.get("scene_percentile", min(0.30, max(0.10, 2.5 * percentile))))
        scene_values = np.asarray(scores, dtype=np.float64)[np.asarray(sample_mask, dtype=bool) & np.isfinite(scores)]
        scene_threshold = float(np.percentile(scene_values, scene_fraction * 100.0))
        catalog_depth = float(settings.get("minimum_absorption_depth", 0.0))
        noise_depth = noise_multiplier[policy] * relative_noise
        settings.update({
            "policy": policy,
            "column_thresholds_rad": thresholds,
            "column_sample_counts": counts,
            "scene_sam_threshold_rad": scene_threshold,
            "scene_percentile": scene_fraction,
            "absolute_sam_threshold_rad": float(settings.get("absolute_sam_threshold_rad", np.inf)),
            "minimum_absorption_depth": max(catalog_depth, noise_depth),
            "catalog_minimum_absorption_depth": catalog_depth,
            "noise_minimum_absorption_depth": noise_depth,
            "estimated_snr": card.estimated_snr,
            "sources": {
                "hard_constraints": "Mineral Evidence Catalog",
                "column_percentile": "scene full-depth stratified calibration",
                "absolute_sam_threshold_rad": "catalog expert safe range",
                "minimum_absorption_depth": "max(catalog floor, spectral noise multiplier)",
                "user_overrides": sorted(overrides.get(policy, {})),
            },
        })
        result[policy] = settings
    return result


def policy_candidate(score: np.ndarray, valid_mask: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    columns = np.asarray(settings["column_thresholds_rad"], dtype=np.float64)
    return (
        valid
        & np.isfinite(values)
        & (values <= float(settings["absolute_sam_threshold_rad"]))
        & (values <= float(settings["scene_sam_threshold_rad"]))
        & (values <= columns[None, :])
    )
