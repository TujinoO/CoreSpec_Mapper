from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .catalog import GroupDefinition
from .v4_models import SamplePlan


@dataclass(frozen=True)
class CalibrationSearchSettings:
    percentile_steps: int = 5
    absolute_threshold_steps: int = 5
    strata: int = 8
    minimum_candidate_pixels: int = 16
    maximum_coverage: float | None = None
    hard_maximum_coverage: float = 0.65
    edge_width: int = 2
    isolated_component_pixels: int = 4
    evidence_weight: float = 1.25
    coverage_utility_weight: float = 0.35
    spatial_support_weight: float = 0.25
    fixed_column_penalty_weight: float = 0.90
    edge_penalty_weight: float = 0.45
    isolated_penalty_weight: float = 0.55
    overcoverage_penalty_weight: float = 1.35

    def __post_init__(self) -> None:
        if self.percentile_steps < 1 or self.absolute_threshold_steps < 1 or self.strata < 1:
            raise ValueError("Calibration grid sizes and strata must be positive")
        if self.minimum_candidate_pixels < 1 or self.edge_width < 1 or self.isolated_component_pixels < 1:
            raise ValueError("Calibration pixel and spatial settings must be positive")
        if not 0.0 < self.hard_maximum_coverage <= 1.0:
            raise ValueError("hard_maximum_coverage must be in (0, 1]")
        if self.maximum_coverage is not None and not 0.0 < self.maximum_coverage <= self.hard_maximum_coverage:
            raise ValueError("maximum_coverage must be positive and no greater than the hard maximum")


@dataclass(frozen=True)
class CalibrationCandidate:
    percentile_fraction: float
    absolute_threshold_rad: float
    objective: float | None
    candidate_pixels: int
    sample_pixels: int
    coverage: float
    evidence_strength: float
    coverage_utility: float
    spatial_support: float
    fixed_column_penalty: float
    edge_penalty: float
    isolated_component_penalty: float
    overcoverage_penalty: float
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V5CalibrationResult:
    status: str
    group_id: str
    policy: str
    safe_percentile_bounds: tuple[float, float]
    safe_absolute_threshold_bounds_rad: tuple[float, float]
    resolved_percentile_fraction: float | None
    resolved_absolute_threshold_rad: float | None
    resolved_column_thresholds_rad: tuple[float, ...]
    sample_pixels: int
    block_ranges: tuple[tuple[int, int], ...]
    candidates: tuple[CalibrationCandidate, ...]
    resolved_reason: str

    @property
    def ready(self) -> bool:
        return self.status == "resolved"

    def require_resolved(self) -> "V5CalibrationResult":
        if not self.ready:
            raise ValueError(f"V5 calibration is blocked: {self.resolved_reason}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "group_id": self.group_id,
            "policy": self.policy,
            "safe_percentile_bounds": list(self.safe_percentile_bounds),
            "safe_absolute_threshold_bounds_rad": list(self.safe_absolute_threshold_bounds_rad),
            "resolved_percentile_fraction": self.resolved_percentile_fraction,
            "resolved_absolute_threshold_rad": self.resolved_absolute_threshold_rad,
            "resolved_column_thresholds_rad": list(self.resolved_column_thresholds_rad),
            "sample_pixels": self.sample_pixels,
            "block_ranges": [list(item) for item in self.block_ranges],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "resolved_reason": self.resolved_reason,
        }


def catalog_safe_search_bounds(group: GroupDefinition) -> tuple[tuple[float, float], tuple[float, float]]:
    percentiles = [
        float(settings["column_percentile"])
        for settings in group.policies.values()
        if "column_percentile" in settings
    ]
    absolutes = [
        float(settings["absolute_sam_threshold_rad"])
        for settings in group.policies.values()
        if "absolute_sam_threshold_rad" in settings
    ]
    if not percentiles or not absolutes:
        raise ValueError(f"Group {group.group_id} does not declare V5-safe percentile and absolute SAM bounds")
    percentile_bounds = (min(percentiles), max(percentiles))
    absolute_bounds = (min(absolutes), max(absolutes))
    if not 0.0 < percentile_bounds[0] <= percentile_bounds[1] < 1.0:
        raise ValueError(f"Group {group.group_id} has invalid percentile bounds")
    if not 0.0 < absolute_bounds[0] <= absolute_bounds[1]:
        raise ValueError(f"Group {group.group_id} has invalid absolute SAM bounds")
    return percentile_bounds, absolute_bounds


def _candidate_values(
    explicit: Sequence[float] | None,
    bounds: tuple[float, float],
    steps: int,
    baseline: float | None,
    name: str,
) -> tuple[float, ...]:
    if explicit is None:
        values = list(np.linspace(bounds[0], bounds[1], steps, dtype=np.float64))
        if baseline is not None:
            values.append(float(baseline))
    else:
        values = [float(item) for item in explicit]
        if not values:
            raise ValueError(f"At least one {name} candidate is required")
    tolerance = 1e-12
    outside = [item for item in values if item < bounds[0] - tolerance or item > bounds[1] + tolerance]
    if outside:
        raise ValueError(f"{name} candidate lies outside Catalog-safe bounds {bounds}: {outside}")
    return tuple(sorted(set(round(item, 12) for item in values)))


def _block_ranges(valid: np.ndarray, sample_plan: SamplePlan | None, strata: int) -> tuple[tuple[int, int], ...]:
    if sample_plan is not None:
        if sample_plan.total_lines != valid.shape[0]:
            raise ValueError("Sample plan and score image line counts do not match")
        ranges = tuple((int(block.start_line), int(block.stop_line)) for block in sample_plan.blocks)
    else:
        edges = np.linspace(0, valid.shape[0], min(strata, valid.shape[0]) + 1, dtype=int)
        ranges = tuple((int(start), int(stop)) for start, stop in zip(edges[:-1], edges[1:]) if stop > start)
    return tuple((start, stop) for start, stop in ranges if np.any(valid[start:stop]))


def _sample_mask(valid: np.ndarray, ranges: Sequence[tuple[int, int]]) -> np.ndarray:
    result = np.zeros_like(valid)
    for start, stop in ranges:
        result[start:stop] = valid[start:stop]
    return result


def _column_thresholds(scores: np.ndarray, selected: np.ndarray, percentile: float) -> np.ndarray:
    thresholds = np.full(scores.shape[1], np.nan, dtype=np.float64)
    for column in range(scores.shape[1]):
        current = scores[selected[:, column], column]
        current = current[np.isfinite(current)]
        if current.size:
            thresholds[column] = float(np.percentile(current, percentile * 100.0))
    reliable = np.flatnonzero(np.isfinite(thresholds))
    if reliable.size == 0:
        raise ValueError("No finite per-column scores are available in the stratified sample")
    missing = np.flatnonzero(~np.isfinite(thresholds))
    thresholds[missing] = np.interp(missing, reliable, thresholds[reliable])
    if thresholds.size >= 3:
        padded = np.pad(thresholds, 1, mode="edge")
        thresholds = np.asarray([np.median(padded[index : index + 3]) for index in range(thresholds.size)])
    return thresholds


def _edge_mask(valid: np.ndarray, width: int) -> np.ndarray:
    result = np.zeros_like(valid)
    for row in range(valid.shape[0]):
        columns = np.flatnonzero(valid[row])
        if columns.size == 0:
            continue
        split = np.flatnonzero(np.diff(columns) > 1) + 1
        for segment in np.split(columns, split):
            count = min(width, segment.size)
            result[row, segment[:count]] = True
            result[row, segment[-count:]] = True
    return result


def _small_component_fraction(mask: np.ndarray, minimum_pixels: int) -> tuple[float, int]:
    source = np.asarray(mask, dtype=bool)
    total = int(np.count_nonzero(source))
    if total == 0:
        return 1.0, 0
    parents: list[int] = []
    sizes: list[int] = []
    previous: list[tuple[int, int, int]] = []

    def find(label: int) -> int:
        root = label
        while parents[root] != root:
            root = parents[root]
        while parents[label] != label:
            parent = parents[label]
            parents[label] = root
            label = parent
        return root

    def union(left: int, right: int) -> int:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return left_root
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        sizes[left_root] += sizes[right_root]
        return left_root

    for row in range(source.shape[0]):
        transitions = np.diff(np.pad(source[row].astype(np.int8, copy=False), 1, mode="constant"))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        current: list[tuple[int, int, int]] = []
        previous_index = 0
        for start, stop in zip(starts, stops):
            label = len(parents)
            parents.append(label)
            sizes.append(int(stop - start))
            while previous_index < len(previous) and previous[previous_index][1] < start:
                previous_index += 1
            overlap_index = previous_index
            while overlap_index < len(previous) and previous[overlap_index][0] <= stop:
                previous_start, previous_stop, previous_label = previous[overlap_index]
                if previous_stop >= start and previous_start <= stop:
                    label = union(label, previous_label)
                overlap_index += 1
            current.append((int(start), int(stop), label))
        previous = current
    roots = {find(label) for label in range(len(parents))}
    small_pixels = sum(sizes[root] for root in roots if sizes[root] < minimum_pixels)
    return small_pixels / total, len(roots)


def _fixed_column_penalty(
    candidate: np.ndarray,
    valid: np.ndarray,
    ranges: Sequence[tuple[int, int]],
) -> float:
    profiles: list[np.ndarray] = []
    for start, stop in ranges:
        block_valid = valid[start:stop]
        denominator = np.sum(block_valid, axis=0)
        density = np.divide(
            np.sum(candidate[start:stop], axis=0),
            denominator,
            out=np.zeros(candidate.shape[1], dtype=np.float64),
            where=denominator > 0,
        )
        profiles.append(density)
    if not profiles:
        return 1.0
    persistent = np.median(np.vstack(profiles), axis=0)
    narrow_excess = np.zeros_like(persistent)
    for column in range(persistent.size):
        neighbors = np.concatenate((
            persistent[max(0, column - 3) : column],
            persistent[column + 1 : min(persistent.size, column + 4)],
        ))
        local = float(np.median(neighbors)) if neighbors.size else float(np.median(persistent))
        narrow_excess[column] = max(0.0, persistent[column] - local)

    # V5.2 used the single worst detector column as the whole-scene penalty.
    # One narrow seam could therefore contribute a penalty of 1.0 even when it
    # represented far below one percent of otherwise coherent candidates.  The
    # V5.3 penalty keeps the peak severity, but scales it by the share of actual
    # candidate pixels carried by suspicious columns.  A map dominated by a
    # fixed column still receives a strong penalty; an isolated seam no longer
    # forces every threshold search to the most conservative boundary.
    severity = np.clip((narrow_excess - 0.03) / 0.20, 0.0, 1.0)
    suspicious = severity > 0.0
    candidate_pixels = int(np.count_nonzero(candidate))
    if candidate_pixels == 0 or not np.any(suspicious):
        return 0.0
    suspicious_share = float(np.count_nonzero(candidate[:, suspicious]) / candidate_pixels)
    peak = float(np.max(severity[suspicious]))
    return float(np.clip(np.sqrt(peak * suspicious_share), 0.0, 1.0))


def _evaluate_candidate(
    scores: np.ndarray,
    sample_mask: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    edge: np.ndarray,
    column_thresholds: np.ndarray,
    percentile: float,
    absolute: float,
    evidence_reference_rad: float,
    settings: CalibrationSearchSettings,
    maximum_coverage: float,
) -> CalibrationCandidate:
    candidate = (
        sample_mask
        & np.isfinite(scores)
        & (scores <= absolute)
        & (scores <= column_thresholds[None, :])
    )
    sample_pixels = int(np.count_nonzero(sample_mask & np.isfinite(scores)))
    candidate_pixels = int(np.count_nonzero(candidate))
    coverage = candidate_pixels / max(sample_pixels, 1)
    reasons: list[str] = []
    if candidate_pixels < settings.minimum_candidate_pixels:
        reasons.append("below_minimum_candidate_pixels")
    if coverage > settings.hard_maximum_coverage:
        reasons.append("hard_maximum_coverage_exceeded")
    if candidate_pixels:
        selected_scores = scores[candidate]
        # Compare evidence on one fixed Catalog-safe scale.  Normalising by the
        # candidate's own absolute threshold would reward a looser threshold
        # even when it selects exactly the same pixels, biasing every search
        # toward the sensitive upper bound.
        evidence_scale = max(float(evidence_reference_rad), 1e-8)
        evidence = float(np.median(np.clip((evidence_scale - selected_scores) / evidence_scale, 0.0, 1.0)))
        edge_penalty = float(np.count_nonzero(candidate & edge) / candidate_pixels)
        isolated_penalty, _ = _small_component_fraction(candidate, settings.isolated_component_pixels)
        spatial_support = 1.0 - isolated_penalty
        fixed_penalty = _fixed_column_penalty(candidate, sample_mask, ranges)
    else:
        evidence = 0.0
        edge_penalty = 1.0
        isolated_penalty = 1.0
        spatial_support = 0.0
        fixed_penalty = 1.0
    coverage_target = max(min(maximum_coverage * 0.25, 0.05), 0.005)
    coverage_utility = float(np.clip(coverage / coverage_target, 0.0, 1.0))
    overcoverage = float(max(0.0, (coverage - maximum_coverage) / max(maximum_coverage, 1e-8)) ** 2)
    accepted = not reasons
    objective = None
    if accepted:
        objective = float(
            settings.evidence_weight * evidence
            + settings.coverage_utility_weight * coverage_utility
            + settings.spatial_support_weight * spatial_support
            - settings.fixed_column_penalty_weight * fixed_penalty
            - settings.edge_penalty_weight * edge_penalty
            - settings.isolated_penalty_weight * isolated_penalty
            - settings.overcoverage_penalty_weight * overcoverage
        )
    return CalibrationCandidate(
        percentile_fraction=float(percentile),
        absolute_threshold_rad=float(absolute),
        objective=objective,
        candidate_pixels=candidate_pixels,
        sample_pixels=sample_pixels,
        coverage=float(coverage),
        evidence_strength=evidence,
        coverage_utility=coverage_utility,
        spatial_support=spatial_support,
        fixed_column_penalty=fixed_penalty,
        edge_penalty=edge_penalty,
        isolated_component_penalty=float(isolated_penalty),
        overcoverage_penalty=overcoverage,
        accepted=accepted,
        reasons=tuple(reasons),
    )


def calibrate_group_thresholds(
    group: GroupDefinition,
    policy: str,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sample_plan: SamplePlan | None = None,
    percentile_candidates: Sequence[float] | None = None,
    absolute_threshold_candidates_rad: Sequence[float] | None = None,
    settings: CalibrationSearchSettings | None = None,
) -> V5CalibrationResult:
    """Resolve percentile and absolute SAM thresholds from stratified scene evidence.

    Search is bounded by the Catalog's conservative-to-sensitive safe envelope.
    It never subtracts a per-mineral scene median or otherwise forces absent minerals
    into competition.  Every searched pair and its proxy-quality terms are returned.
    """
    values = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 2 or values.shape != valid.shape:
        raise ValueError("scores and valid_mask must be matching two-dimensional arrays")
    if policy not in group.policies:
        raise ValueError(f"Unknown policy {policy!r} for group {group.group_id}")
    usable = valid & np.isfinite(values)
    if not np.any(usable):
        raise ValueError("Calibration requires at least one finite score inside the valid mask")
    search = settings or CalibrationSearchSettings()
    percentile_bounds, absolute_bounds = catalog_safe_search_bounds(group)
    baseline = group.policies[policy]
    percentiles = _candidate_values(
        percentile_candidates,
        percentile_bounds,
        search.percentile_steps,
        baseline.get("column_percentile"),
        "percentile",
    )
    absolutes = _candidate_values(
        absolute_threshold_candidates_rad,
        absolute_bounds,
        search.absolute_threshold_steps,
        baseline.get("absolute_sam_threshold_rad"),
        "absolute-threshold",
    )
    ranges = _block_ranges(usable, sample_plan, search.strata)
    if not ranges:
        raise ValueError("No stratified calibration block contains finite valid scores")
    sampled = _sample_mask(usable, ranges)
    sample_pixels = int(np.count_nonzero(sampled))
    maximum_coverage = search.maximum_coverage
    if maximum_coverage is None:
        maximum_coverage = {"conservative": 0.08, "balanced": 0.18, "sensitive": 0.30}.get(policy, 0.18)
        maximum_coverage = min(maximum_coverage, search.hard_maximum_coverage)
    edge = _edge_mask(sampled, search.edge_width)
    thresholds_by_percentile = {
        percentile: _column_thresholds(values, sampled, percentile) for percentile in percentiles
    }
    records: list[CalibrationCandidate] = []
    for percentile in percentiles:
        thresholds = thresholds_by_percentile[percentile]
        for absolute in absolutes:
            records.append(_evaluate_candidate(
                values,
                sampled,
                ranges,
                edge,
                thresholds,
                percentile,
                absolute,
                absolute_bounds[1],
                search,
                float(maximum_coverage),
            ))
    accepted = [record for record in records if record.accepted and record.objective is not None]
    if not accepted:
        reason = (
            "No Catalog-bounded candidate passed the minimum-pixel and hard-coverage gates; "
            "thresholds were not relaxed outside their safe envelope."
        )
        return V5CalibrationResult(
            status="blocked",
            group_id=group.group_id,
            policy=policy,
            safe_percentile_bounds=percentile_bounds,
            safe_absolute_threshold_bounds_rad=absolute_bounds,
            resolved_percentile_fraction=None,
            resolved_absolute_threshold_rad=None,
            resolved_column_thresholds_rad=(),
            sample_pixels=sample_pixels,
            block_ranges=ranges,
            candidates=tuple(records),
            resolved_reason=reason,
        )
    best = max(
        accepted,
        key=lambda item: (
            float(item.objective),
            -item.fixed_column_penalty,
            -item.edge_penalty,
            -item.isolated_component_penalty,
            -item.overcoverage_penalty,
            -item.coverage,
            -item.absolute_threshold_rad,
            -item.percentile_fraction,
        ),
    )
    columns = thresholds_by_percentile[best.percentile_fraction]
    reason = (
        "Selected the highest proxy objective (V5.3 scene-aware) inside Catalog-safe bounds; "
        f"evidence={best.evidence_strength:.4f}, coverage={best.coverage:.4f}, "
        f"fixed_column_penalty={best.fixed_column_penalty:.4f}, edge_penalty={best.edge_penalty:.4f}, "
        f"isolated_penalty={best.isolated_component_penalty:.4f}, "
        f"overcoverage_penalty={best.overcoverage_penalty:.4f}."
    )
    return V5CalibrationResult(
        status="resolved",
        group_id=group.group_id,
        policy=policy,
        safe_percentile_bounds=percentile_bounds,
        safe_absolute_threshold_bounds_rad=absolute_bounds,
        resolved_percentile_fraction=best.percentile_fraction,
        resolved_absolute_threshold_rad=best.absolute_threshold_rad,
        resolved_column_thresholds_rad=tuple(float(item) for item in columns),
        sample_pixels=sample_pixels,
        block_ranges=ranges,
        candidates=tuple(records),
        resolved_reason=reason,
    )


def apply_resolved_thresholds(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    calibration: V5CalibrationResult,
) -> np.ndarray:
    calibration.require_resolved()
    values = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    columns = np.asarray(calibration.resolved_column_thresholds_rad, dtype=np.float64)
    if values.ndim != 2 or values.shape != valid.shape or columns.shape != (values.shape[1],):
        raise ValueError("Score image, valid mask, and resolved column thresholds do not align")
    absolute = float(calibration.resolved_absolute_threshold_rad)
    return valid & np.isfinite(values) & (values <= absolute) & (values <= columns[None, :])
