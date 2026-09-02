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


def _material_depth_segments(
    valid_mask: np.ndarray,
    settings: Mapping[str, Any],
) -> list[tuple[int, int]]:
    """Resolve physically separated core-box/depth sections from the mask."""

    valid = np.asarray(valid_mask, dtype=bool)
    minimum_row_pixels = max(
        1,
        int(settings.get("repeated_column_minimum_valid_pixels_per_row", 4)),
    )
    active = np.sum(valid, axis=1) >= minimum_row_pixels
    transitions = np.diff(np.pad(active.astype(np.int8), 1, mode="constant"))
    spans = [
        (int(start), int(stop))
        for start, stop in zip(
            np.flatnonzero(transitions == 1),
            np.flatnonzero(transitions == -1),
        )
    ]
    maximum_gap = max(0, int(settings.get("repeated_column_segment_merge_gap", 12)))
    merged: list[tuple[int, int]] = []
    for start, stop in spans:
        if merged and start - merged[-1][1] <= maximum_gap:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    minimum_height = max(4, int(settings.get("repeated_column_minimum_segment_height", 16)))
    merged = [(start, stop) for start, stop in merged if stop - start >= minimum_height]
    return merged


def repeated_segment_column_stripe_mask(
    candidate: np.ndarray,
    valid_mask: np.ndarray,
    settings: Mapping[str, Any],
    *,
    policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find narrow detector-column peaks repeated in independent core sections.

    A global density statistic misses a common core-imaging artifact: the same
    one-to-few detector columns respond inside several separated core boxes,
    while each individual response is too short to dominate the full image.
    This detector compares every section with neighbouring columns and removes
    only repeated narrow peaks lacking same-class lateral/oblique support.
    """

    selected = np.asarray(candidate, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if selected.shape != valid.shape or selected.ndim != 2:
        raise ValueError("Candidate and valid mask must be matching 2D arrays")
    segments = _material_depth_segments(valid, settings)
    empty = np.zeros_like(selected)
    if len(segments) < 2 or not np.any(selected):
        return empty, {
            "depth_segment_count": len(segments),
            "repeated_column_count": 0,
            "maximum_segment_repetition": 0,
            "candidate_removed_pixels": 0,
            "repeated_columns": [],
            "dominant_corridor_count": 0,
            "dominant_corridors": [],
        }

    density_threshold = float(
        settings.get(
            "repeated_column_segment_density_by_policy",
            {"conservative": 0.025, "balanced": 0.035, "sensitive": 0.045},
        ).get(policy, 0.035)
    )
    ratio_threshold = float(
        settings.get(
            "repeated_column_local_ratio_by_policy",
            {"conservative": 2.5, "balanced": 2.3, "sensitive": 2.1},
        ).get(policy, 2.3)
    )
    excess_threshold = float(
        settings.get(
            "repeated_column_local_excess_by_policy",
            {"conservative": 0.020, "balanced": 0.020, "sensitive": 0.025},
        ).get(policy, 0.020)
    )
    minimum_valid = max(
        4,
        int(settings.get("repeated_column_minimum_valid_pixels_per_segment", 8)),
    )
    radius = max(4, int(settings.get("repeated_column_neighborhood_radius", 8)))
    guard = max(1, min(radius - 1, int(settings.get("repeated_column_peak_guard", 2))))
    hits = np.zeros((len(segments), selected.shape[1]), dtype=bool)
    for segment_index, (start, stop) in enumerate(segments):
        segment_valid = valid[start:stop]
        denominator = np.sum(segment_valid, axis=0)
        density = np.divide(
            np.sum(selected[start:stop], axis=0),
            denominator,
            out=np.zeros(selected.shape[1], dtype=np.float64),
            where=denominator >= minimum_valid,
        )
        for column in range(selected.shape[1]):
            left = density[max(0, column - radius) : max(0, column - guard)]
            right = density[
                min(selected.shape[1], column + guard + 1) :
                min(selected.shape[1], column + radius + 1)
            ]
            neighborhood = np.concatenate((left, right))
            local = float(np.median(neighborhood)) if neighborhood.size else 0.0
            ratio = density[column] / max(local, 0.005)
            hits[segment_index, column] = (
                denominator[column] >= minimum_valid
                and density[column] >= density_threshold
                and density[column] - local >= excess_threshold
                and ratio >= ratio_threshold
            )

    repetition = np.sum(hits, axis=0)
    required_repetition = max(
        2,
        int(
            settings.get(
                "repeated_column_minimum_segments_by_policy",
                {"conservative": 2, "balanced": 2, "sensitive": 2},
            ).get(policy, 2)
        ),
    )
    repeated = repetition >= required_repetition
    if repeated.size >= 3:
        repeated[1:-1] |= repeated[:-2] & repeated[2:]
    columns = np.flatnonzero(repeated)
    removed = np.zeros_like(selected)
    accepted_columns: list[int] = []
    repeated_bands: list[np.ndarray] = []
    if columns.size:
        split_at = np.flatnonzero(np.diff(columns) > 1) + 1
        repeated_bands = [band for band in np.split(columns, split_at) if band.size]
        maximum_width = max(1, int(settings.get("maximum_repeated_column_band_width", 6)))
        support_radius = max(1, int(settings.get("repeated_column_geologic_support_radius", 4)))
        row_tolerance = max(0, int(settings.get("repeated_column_geologic_row_tolerance", 2)))
        for band in repeated_bands:
            if band.size == 0 or band.size > maximum_width:
                continue
            first, last = int(band[0]), int(band[-1])
            left = selected[:, max(0, first - support_radius) : first]
            right = selected[:, last + 1 : min(selected.shape[1], last + support_radius + 1)]
            external_rows = np.zeros(selected.shape[0], dtype=bool)
            if left.size:
                external_rows |= np.any(left, axis=1)
            if right.size:
                external_rows |= np.any(right, axis=1)
            if row_tolerance:
                padded = np.pad(external_rows, (row_tolerance, row_tolerance), mode="constant")
                external_rows = np.lib.stride_tricks.sliding_window_view(
                    padded,
                    2 * row_tolerance + 1,
                ).any(axis=1)
            band_mask = np.zeros_like(selected)
            band_mask[:, band] = selected[:, band]
            band_removed = band_mask & ~external_rows[:, None]
            removed |= band_removed
            accepted_columns.extend(
                int(value) for value in np.flatnonzero(np.any(band_removed, axis=0))
            )
    # A detector artifact can occupy several nearby peak bands rather than one
    # exact column.  The former one-sided support rule then lets bands protect
    # one another, even when a narrow detector corridor contains most of the
    # class across many physically separated core boxes.  Merge nearby peaks
    # only for this dominance audit and require genuine support on both sides
    # of the whole corridor before preserving its peak pixels.
    dominant_corridor_enabled = bool(
        settings.get("dominant_repeated_corridor_enabled", True)
    )
    dominance_threshold = float(
        settings.get(
            "dominant_repeated_corridor_fraction_by_policy",
            {"conservative": 0.60, "balanced": 0.55, "sensitive": 0.55},
        ).get(policy, 0.55)
    )
    corridor_minimum_segments = max(
        2,
        int(settings.get("dominant_repeated_corridor_minimum_segments", 4)),
    )
    corridor_merge_gap = max(
        0,
        int(settings.get("dominant_repeated_corridor_merge_gap", 7)),
    )
    corridor_maximum_width = max(
        1,
        int(settings.get("dominant_repeated_corridor_maximum_width", 24)),
    )
    corridor_support_radius = max(
        1,
        int(settings.get("dominant_repeated_corridor_support_radius", 6)),
    )
    corridor_row_tolerance = max(
        0,
        int(settings.get("dominant_repeated_corridor_row_tolerance", 2)),
    )
    corridor_groups: list[list[np.ndarray]] = []
    for band in repeated_bands if dominant_corridor_enabled else ():
        if (
            corridor_groups
            and int(band[0]) - int(corridor_groups[-1][-1][-1]) - 1
            <= corridor_merge_gap
            and int(band[-1]) - int(corridor_groups[-1][0][0]) + 1
            <= corridor_maximum_width
        ):
            corridor_groups[-1].append(band)
        else:
            corridor_groups.append([band])

    dominant_corridors: list[dict[str, Any]] = []
    selected_count = int(np.count_nonzero(selected))
    for corridor_bands in corridor_groups:
        peak_columns = np.unique(np.concatenate(corridor_bands))
        first, last = int(peak_columns[0]), int(peak_columns[-1])
        width = last - first + 1
        if width > corridor_maximum_width:
            continue
        segment_repetition = int(
            np.count_nonzero(np.any(hits[:, peak_columns], axis=1))
        )
        corridor_pixels = int(np.count_nonzero(selected[:, first : last + 1]))
        corridor_fraction = corridor_pixels / max(selected_count, 1)
        if (
            segment_repetition < corridor_minimum_segments
            or corridor_fraction < dominance_threshold
        ):
            continue

        left = selected[:, max(0, first - corridor_support_radius) : first]
        right = selected[
            :,
            last + 1 : min(selected.shape[1], last + corridor_support_radius + 1),
        ]
        left_rows = (
            np.any(left, axis=1) if left.size else np.zeros(selected.shape[0], dtype=bool)
        )
        right_rows = (
            np.any(right, axis=1) if right.size else np.zeros(selected.shape[0], dtype=bool)
        )
        if corridor_row_tolerance:
            window = 2 * corridor_row_tolerance + 1
            left_rows = np.lib.stride_tricks.sliding_window_view(
                np.pad(left_rows, (corridor_row_tolerance, corridor_row_tolerance)),
                window,
            ).any(axis=1)
            right_rows = np.lib.stride_tricks.sliding_window_view(
                np.pad(right_rows, (corridor_row_tolerance, corridor_row_tolerance)),
                window,
            ).any(axis=1)
        bilateral_rows = left_rows & right_rows
        peak_mask = np.zeros_like(selected)
        peak_mask[:, peak_columns] = selected[:, peak_columns]
        corridor_removed = peak_mask & ~bilateral_rows[:, None]
        if not np.any(corridor_removed):
            continue
        removed |= corridor_removed
        affected_columns = np.flatnonzero(np.any(corridor_removed, axis=0))
        accepted_columns.extend(int(value) for value in affected_columns)
        dominant_corridors.append({
            "start_column": first,
            "stop_column_inclusive": last,
            "width": width,
            "segment_repetition": segment_repetition,
            "corridor_pixels": corridor_pixels,
            "corridor_fraction": float(corridor_fraction),
            "peak_columns": [int(value) for value in peak_columns],
            "removed_pixels": int(np.count_nonzero(corridor_removed)),
        })
    repeated_columns = sorted(set(accepted_columns))
    return removed, {
        "depth_segment_count": len(segments),
        "repeated_column_count": len(repeated_columns),
        "maximum_segment_repetition": int(np.max(repetition)) if repetition.size else 0,
        "candidate_removed_pixels": int(np.count_nonzero(removed)),
        "repeated_columns": repeated_columns,
        "dominant_corridor_count": len(dominant_corridors),
        "dominant_corridors": dominant_corridors,
    }


def low_exposure_high_density_column_mask(
    candidate: np.ndarray,
    valid_mask: np.ndarray,
    settings: Mapping[str, Any],
    *,
    policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove isolated responses in detector columns with very little exposure.

    A column that intersects only a small edge sliver of the foreground can
    evade repetition-based fixed-column detection, yet show a misleadingly
    high classified fraction inside one depth interval. This detector requires
    low valid exposure, enough candidates, strong local density excess, and no
    lateral or oblique candidate support.
    """
    selected = np.asarray(candidate, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if selected.shape != valid.shape or selected.ndim != 2:
        raise ValueError("Candidate and valid mask must be matching 2D arrays")
    removed = np.zeros_like(selected)
    valid_count = np.sum(valid, axis=0)
    positive_valid = valid_count[valid_count > 0]
    if not np.any(selected) or positive_valid.size == 0:
        return removed, {"candidate_removed_pixels": 0, "columns": []}

    reference_exposure = float(np.median(positive_valid))
    exposure_fraction = float(settings.get("low_exposure_column_fraction", 0.10))
    minimum_valid = max(8, int(settings.get("low_exposure_minimum_valid_pixels", 32)))
    minimum_candidate = max(4, int(settings.get("low_exposure_minimum_candidate_pixels", 12)))
    density_threshold = float(
        settings.get(
            "low_exposure_density_by_policy",
            {"conservative": 0.15, "balanced": 0.15, "sensitive": 0.18},
        ).get(policy, 0.15)
    )
    ratio_threshold = float(settings.get("low_exposure_local_ratio", 3.0))
    excess_threshold = float(settings.get("low_exposure_local_excess", 0.10))
    radius = max(3, int(settings.get("low_exposure_neighborhood_radius", 6)))
    guard = max(0, min(radius - 1, int(settings.get("low_exposure_peak_guard", 0))))
    candidate_count = np.sum(selected, axis=0)
    density = np.divide(
        candidate_count,
        valid_count,
        out=np.zeros(selected.shape[1], dtype=np.float64),
        where=valid_count > 0,
    )
    flagged: list[int] = []
    for column in range(selected.shape[1]):
        if (
            valid_count[column] < minimum_valid
            or valid_count[column] > exposure_fraction * reference_exposure
            or candidate_count[column] < minimum_candidate
            or density[column] < density_threshold
        ):
            continue
        left = density[max(0, column - radius) : max(0, column - guard)]
        right = density[
            min(selected.shape[1], column + guard + 1) :
            min(selected.shape[1], column + radius + 1)
        ]
        neighborhood = np.concatenate((left, right))
        local = float(np.median(neighborhood)) if neighborhood.size else 0.0
        if (
            density[column] - local < excess_threshold
            or density[column] / max(local, 0.005) < ratio_threshold
        ):
            continue

        support_radius = max(1, int(settings.get("low_exposure_support_radius", 3)))
        row_tolerance = max(0, int(settings.get("low_exposure_row_tolerance", 2)))
        left_support = selected[:, max(0, column - support_radius) : column]
        right_support = selected[
            :, column + 1 : min(selected.shape[1], column + support_radius + 1)
        ]
        external_rows = np.zeros(selected.shape[0], dtype=bool)
        if left_support.size:
            external_rows |= np.any(left_support, axis=1)
        if right_support.size:
            external_rows |= np.any(right_support, axis=1)
        if row_tolerance:
            window = 2 * row_tolerance + 1
            external_rows = np.lib.stride_tricks.sliding_window_view(
                np.pad(external_rows, (row_tolerance, row_tolerance), mode="constant"),
                window,
            ).any(axis=1)
        column_removed = selected[:, column] & ~external_rows
        if np.any(column_removed):
            removed[:, column] = column_removed
            flagged.append(column)
    return removed, {
        "candidate_removed_pixels": int(np.count_nonzero(removed)),
        "columns": flagged,
        "reference_valid_exposure": reference_exposure,
    }


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
    directional_disabled = {
        int(value) for value in settings.get("disable_directional_filter_by_class", ())
    }
    minimum_component_by_class = {
        int(key): int(value) for key, value in settings.get("minimum_component_by_class", {}).items()
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
    minimum_geologic_lateral_support = int(
        settings.get("minimum_geologic_lateral_support", 3)
    )
    low_column_risk = risk[None, :] < float(
        settings.get("strong_evidence_max_column_risk", 0.35)
    )
    geologic_lateral_support = group_lateral >= minimum_geologic_lateral_support
    directional_fraction = float(np.count_nonzero(group_directional) / max(np.count_nonzero(group_mask), 1))
    counts = {
        "directional_removed": 0,
        "elongated_removed": 0,
        "small_component_removed": 0,
        "residual_column_band_removed": 0,
        "repeated_segment_column_removed": 0,
        "repeated_segment_column_count": 0,
        "dominant_repeated_corridor_count": 0,
        "maximum_segment_repetition": 0,
        "low_exposure_column_removed": 0,
        "low_exposure_column_count": 0,
    }
    for class_id in range(1, int(np.max(source)) + 1):
        class_mask = source == class_id
        if not np.any(class_mask):
            continue
        class_protection = protection_by_class.get(class_id, protection)
        strong = class_mask & (confidence >= class_protection)
        # Strong spectra can protect a local/oblique geological response, but
        # confidence alone cannot legitimise a detector column repeated down
        # the image.  High-risk columns additionally require lateral support.
        strong_geologic = strong & (low_column_risk | geologic_lateral_support)
        detected = (
            np.zeros_like(class_mask)
            if class_id in directional_disabled
            else (
                group_directional
                & class_mask
                & (
                    risk[None, :]
                    >= float(settings.get("directional_column_risk_threshold", 0.20))
                )
                & ~strong_geologic
            )
        )
        fixed_column = (
            class_mask
            & (risk[None, :] >= float(settings.get("column_risk_threshold", 0.72)))
            & (group_lateral <= int(settings.get("max_fixed_column_lateral_support", 3)))
            & ~strong_geologic
        )
        if class_id in directional_disabled:
            fixed_column[:] = False
        detected |= fixed_column
        stripe_mask |= detected
        counts["directional_removed"] += int(np.count_nonzero(detected))
        class_mask &= ~detected

        weak = class_mask & ~strong
        kept_weak = (
            weak
            if class_id in directional_disabled
            else remove_elongated_components(
                weak,
                min_height=int(settings.get("min_line_height", 15)),
                max_width=int(settings.get("max_line_width", 3)),
                min_aspect=float(settings.get("min_aspect", 5.0)),
            )
        )
        elongated = weak & ~kept_weak
        stripe_mask |= elongated
        counts["elongated_removed"] += int(np.count_nonzero(elongated))
        class_mask &= ~elongated

        weak = class_mask & ~strong
        base_component = minimum_component_by_class.get(
            class_id,
            int(settings.get("min_component", 2)),
        )
        minimum_component = max(1, base_component + {"conservative": 1, "balanced": 0, "sensitive": -1}[policy])
        if policy == "balanced" and directional_fraction >= 0.40:
            minimum_component = 1
        kept_weak = clean_binary_mask(weak, median_size=1, min_component=minimum_component)
        counts["small_component_removed"] += int(np.count_nonzero(weak & ~kept_weak))
        class_mask = strong | kept_weak
        output[class_mask] = class_id

    # Complement the repetition detector with a narrow edge-exposure audit.
    # It catches columns that touch only a small foreground sliver in one depth
    # interval and therefore have neither global repetition nor reliable risk.
    for class_id in range(1, int(np.max(output)) + 1):
        class_mask = output == class_id
        low_exposure_mask, low_exposure_record = low_exposure_high_density_column_mask(
            class_mask,
            valid,
            settings,
            policy=policy,
        )
        low_exposure_mask &= class_mask
        stripe_mask |= low_exposure_mask
        output[low_exposure_mask] = 0
        counts["low_exposure_column_removed"] += int(
            low_exposure_record["candidate_removed_pixels"]
        )
        counts["low_exposure_column_count"] += len(low_exposure_record["columns"])

    output_group = output > 0
    column_valid = np.maximum(np.sum(valid, axis=0), 1)
    column_density = np.sum(output_group, axis=0) / column_valid
    density_threshold = {"conservative": 0.06, "balanced": 0.075, "sensitive": 0.09}[policy]
    high_columns = (
        (column_density >= density_threshold)
        & (risk >= float(settings.get("residual_column_risk_threshold", 0.10)))
    )
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
            repeated_segments = sum(np.mean(segment) >= 0.02 for segment in segments if segment.size)
            if repeated_segments < 2:
                continue
            band_mask = np.zeros_like(output_group)
            band_mask[:, band] = output_group[:, band]
            left = output_group[:, max(0, int(band[0]) - 3) : int(band[0])]
            right = output_group[:, int(band[-1]) + 1 : min(output_group.shape[1], int(band[-1]) + 4)]
            external_row_support = np.zeros(output_group.shape[0], dtype=bool)
            if left.size:
                external_row_support |= np.any(left, axis=1)
            if right.size:
                external_row_support |= np.any(right, axis=1)
            # A ±2-row allowance protects oblique fractures crossing the risky
            # detector band while an isolated repeated vertical band has no
            # such outside support and is removed even when locally confident.
            padded_support = np.pad(external_row_support, (2, 2), mode="constant")
            external_geologic_support = np.lib.stride_tricks.sliding_window_view(
                padded_support,
                5,
            ).any(axis=1)
            residual_protection = float(settings.get("residual_band_protection", 0.60))
            protected = (
                band_mask
                & (confidence >= residual_protection)
                & external_geologic_support[:, None]
            )
            removed = band_mask & ~protected
            stripe_mask |= removed
            output[removed] = 0
            counts["residual_column_band_removed"] += int(np.count_nonzero(removed))
    # Final-class audit: detect short responses that recur at the exact same
    # detector columns in multiple separated core boxes.  This runs after the
    # legacy global-density filter because that statistic can remain low even
    # when the montage shows an unmistakably regular vertical pattern.
    for class_id in range(1, int(np.max(output)) + 1):
        class_mask = output == class_id
        repeated_settings = dict(settings)
        # Dominant-corridor suppression is a terminal class-distribution audit.
        # The earlier per-profile cleanup lacks final nesting/competition state.
        repeated_settings["dominant_repeated_corridor_enabled"] = False
        repeated_mask, record = repeated_segment_column_stripe_mask(
            class_mask,
            valid,
            repeated_settings,
            policy=policy,
        )
        repeated_mask &= class_mask
        stripe_mask |= repeated_mask
        output[repeated_mask] = 0
        counts["repeated_segment_column_removed"] += int(np.count_nonzero(repeated_mask))
        counts["repeated_segment_column_count"] += int(record["repeated_column_count"])
        counts["dominant_repeated_corridor_count"] += int(
            record["dominant_corridor_count"]
        )
        counts["maximum_segment_repetition"] = max(
            counts["maximum_segment_repetition"],
            int(record["maximum_segment_repetition"]),
        )
    return output, stripe_mask, counts
