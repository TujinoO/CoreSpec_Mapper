from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .algorithms import clean_binary_mask, directional_stripe_mask, remove_elongated_components
from .v4_models import SamplePlan


def fixed_column_risk_score(candidate: np.ndarray, valid_mask: np.ndarray, plan: SamplePlan) -> np.ndarray:
    selected = np.asarray(candidate, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if selected.shape != valid.shape or selected.shape[0] != plan.total_lines:
        raise ValueError("Candidate, mask, and sample plan dimensions do not match")
    segment_profiles: list[np.ndarray] = []
    for block in plan.blocks:
        block_valid = valid[block.start_line : block.stop_line]
        denominator = np.maximum(np.sum(block_valid, axis=0), 1)
        segment_profiles.append(np.sum(selected[block.start_line : block.stop_line], axis=0) / denominator)
    profiles = np.vstack(segment_profiles)
    center = np.median(profiles, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(profiles - center), axis=1, keepdims=True)
    zscore = np.divide(profiles - center, np.maximum(mad, 0.005))
    repetition = np.mean(zscore >= 2.5, axis=0)
    profile = np.median(profiles, axis=0)
    local = np.empty_like(profile)
    for column in range(profile.size):
        neighborhood = np.concatenate([
            profile[max(0, column - 6) : max(0, column - 1)],
            profile[min(profile.size, column + 2) : min(profile.size, column + 7)],
        ])
        local[column] = float(np.median(neighborhood)) if neighborhood.size else float(np.median(profile))
    excess = np.clip((profile - local) / 0.10, 0.0, 1.0)
    return np.clip(0.60 * repetition + 0.40 * excess, 0.0, 1.0).astype(np.float32)


def edge_risk_score(valid_mask: np.ndarray, width: int = 2) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("Valid mask must be two-dimensional")
    risk = np.zeros(valid.shape, dtype=np.float32)
    for row in range(valid.shape[0]):
        columns = np.flatnonzero(valid[row])
        if columns.size == 0:
            continue
        split_at = np.flatnonzero(np.diff(columns) > 1) + 1
        for segment in np.split(columns, split_at):
            for offset in range(max(int(width), 1)):
                value = 1.0 - offset / max(width, 1)
                if offset < segment.size:
                    risk[row, segment[offset]] = max(risk[row, segment[offset]], value)
                    risk[row, segment[-offset - 1]] = max(risk[row, segment[-offset - 1]], value)
    return risk


def filter_artifacts(
    labels: np.ndarray,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    settings: Mapping[str, Any],
    *,
    policy: str,
    column_risk: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    source = np.asarray(labels, dtype=np.uint8)
    confidence = np.asarray(confidence, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    output = np.zeros_like(source)
    stripe_mask = np.zeros_like(valid)
    profile_factor = {"conservative": 0.85, "balanced": 1.0, "sensitive": 1.15}[policy]
    protection = float(settings.get("strong_evidence_protection", 0.72))
    protection_by_class = {
        int(key): float(value) for key, value in settings.get("strong_evidence_protection_by_class", {}).items()
    }
    if column_risk is None:
        risk = np.zeros(source.shape[1], dtype=np.float64)
    else:
        risk = np.asarray(column_risk, dtype=np.float64)
        if risk.shape != (source.shape[1],):
            raise ValueError("Column risk must contain one value per detector column")
    group_mask = source > 0
    group_directional = directional_stripe_mask(
        group_mask,
        valid,
        vertical_window=int(settings.get("vertical_window", 61)),
        min_vertical_density=min(0.95, float(settings.get("min_vertical_density", 0.10)) * profile_factor),
        column_ratio=float(settings.get("column_ratio", 2.0)),
        column_excess=float(settings.get("column_excess", 0.03)) * profile_factor,
        max_lateral_support=int(settings.get("max_lateral_support", 2)),
        neighborhood_radius=int(settings.get("neighborhood_radius", 6)),
    )
    padded_group = np.pad(group_mask.astype(np.uint8), ((0, 0), (2, 2)), mode="constant")
    group_lateral = np.lib.stride_tricks.sliding_window_view(padded_group, 5, axis=1).sum(axis=-1)
    directional_fraction = float(np.count_nonzero(group_directional) / max(np.count_nonzero(group_mask), 1))
    counts = {
        "directional_removed": 0,
        "elongated_removed": 0,
        "small_component_removed": 0,
        "residual_column_band_removed": 0,
    }
    for class_id in range(1, int(np.max(source)) + 1):
        class_mask = source == class_id
        if not np.any(class_mask):
            continue
        class_protection = protection_by_class.get(class_id, protection)
        strong = class_mask & (confidence >= class_protection)
        detected = group_directional & class_mask & ~strong
        fixed_column = (
            class_mask
            & (risk[None, :] >= float(settings.get("column_risk_threshold", 0.72)))
            & (group_lateral <= int(settings.get("max_fixed_column_lateral_support", 3)))
            & ~strong
        )
        detected |= fixed_column
        stripe_mask |= detected
        counts["directional_removed"] += int(np.count_nonzero(detected))
        class_mask &= ~detected

        weak = class_mask & ~strong
        kept_weak = remove_elongated_components(
            weak,
            min_height=int(settings.get("min_line_height", 15)),
            max_width=int(settings.get("max_line_width", 3)),
            min_aspect=float(settings.get("min_aspect", 5.0)),
        )
        elongated = weak & ~kept_weak
        stripe_mask |= elongated
        counts["elongated_removed"] += int(np.count_nonzero(elongated))
        class_mask &= ~elongated

        weak = class_mask & ~strong
        base_component = int(settings.get("min_component", 2))
        minimum_component = max(1, base_component + {"conservative": 1, "balanced": 0, "sensitive": -1}[policy])
        if policy == "balanced" and directional_fraction >= 0.40:
            minimum_component = 1
        kept_weak = clean_binary_mask(weak, median_size=1, min_component=minimum_component)
        counts["small_component_removed"] += int(np.count_nonzero(weak & ~kept_weak))
        class_mask = strong | kept_weak
        output[class_mask] = class_id

    output_group = output > 0
    column_valid = np.maximum(np.sum(valid, axis=0), 1)
    column_density = np.sum(output_group, axis=0) / column_valid
    density_threshold = {"conservative": 0.08, "balanced": 0.10, "sensitive": 0.14}[policy]
    high_columns = (column_density >= density_threshold) & (risk >= 0.25)
    # Bridge a single-column dip inside an otherwise persistent detector band.
    if high_columns.size >= 3:
        high_columns[1:-1] |= high_columns[:-2] & high_columns[2:]
    selected_columns = np.flatnonzero(high_columns)
    if selected_columns.size:
        split_at = np.flatnonzero(np.diff(selected_columns) > 1) + 1
        for band in np.split(selected_columns, split_at):
            if band.size == 0 or band.size > int(settings.get("maximum_residual_band_width", 12)):
                continue
            row_activity = np.any(output_group[:, band], axis=1)
            segments = np.array_split(row_activity, min(4, row_activity.size))
            repeated_segments = sum(np.mean(segment) >= 0.03 for segment in segments if segment.size)
            if repeated_segments < 2:
                continue
            band_mask = np.zeros_like(output_group)
            band_mask[:, band] = output_group[:, band]
            residual_protection = float(settings.get("residual_band_protection", 0.65))
            removed = band_mask & (confidence < residual_protection)
            stripe_mask |= removed
            output[removed] = 0
            counts["residual_column_band_removed"] += int(np.count_nonzero(removed))
    return output, stripe_mask, counts
