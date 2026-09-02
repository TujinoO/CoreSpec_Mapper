from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage

from .masking import _component_cleanup, _interior_pixel_fraction


@dataclass(frozen=True)
class SpatialRefinementAudit:
    closing_size: int
    maximum_hole_pixels: int
    minimum_component_pixels: int
    input_pixels: int
    closing_added_pixels: int
    filled_hole_count: int
    filled_hole_pixels: int
    component_count: int
    kept_component_count: int
    removed_component_pixels: int
    output_pixels: int
    foreground_fraction: float
    interior_pixel_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class DepthCoverageBlock:
    scale_rows: int
    start_row: int
    stop_row: int
    foreground_fraction: float
    active_row_fraction: float
    median_active_row_fraction: float
    expected_fraction: float | None
    expected_scale: float | None
    drop_fraction: float | None
    drop_ratio: float | None
    robust_z: float | None
    has_left_support: bool
    has_right_support: bool
    anomaly: bool
    severity: str
    reason: str

    def to_dict(self) -> dict[str, int | float | bool | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class DepthCoverageRegion:
    start_row: int
    stop_row: int
    supporting_scales: tuple[int, ...]
    minimum_observed_fraction: float
    maximum_expected_fraction: float
    severity: str
    classification: str

    def to_dict(self) -> dict[str, int | float | str | list[int]]:
        value = asdict(self)
        value["supporting_scales"] = list(self.supporting_scales)
        return value


def _fill_small_holes(mask: np.ndarray, maximum_pixels: int) -> tuple[np.ndarray, int, int]:
    source = np.asarray(mask, dtype=bool)
    if maximum_pixels < 1 or not np.any(source):
        return source.copy(), 0, 0
    labels, _ = ndimage.label(~source, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    eligible = sizes <= int(maximum_pixels)
    eligible[0] = False
    eligible[border_labels] = False
    fill = eligible[labels]
    return source | fill, int(np.count_nonzero(eligible)), int(np.count_nonzero(fill))


def refine_swir_mask_spatially(
    mask: np.ndarray,
    *,
    closing_size: int = 3,
    maximum_hole_pixels: int = 125,
    minimum_component_pixels: int = 64,
) -> tuple[np.ndarray, SpatialRefinementAudit]:
    """Apply conservative same-grid cleanup without inventing new spectral regions."""
    if closing_size < 1 or closing_size % 2 == 0:
        raise ValueError("closing_size must be a positive odd integer")
    if maximum_hole_pixels < 0 or minimum_component_pixels < 1:
        raise ValueError("hole and component limits must be non-negative/positive")
    source = np.asarray(mask, dtype=bool)
    closed = ndimage.binary_closing(
        source, structure=np.ones((closing_size, closing_size), dtype=bool)
    )
    filled, hole_count, hole_pixels = _fill_small_holes(closed, maximum_hole_pixels)
    cleaned, component_count, kept_count, removed_pixels = _component_cleanup(
        filled, minimum_component_pixels
    )
    audit = SpatialRefinementAudit(
        closing_size=int(closing_size),
        maximum_hole_pixels=int(maximum_hole_pixels),
        minimum_component_pixels=int(minimum_component_pixels),
        input_pixels=int(np.count_nonzero(source)),
        closing_added_pixels=int(np.count_nonzero(closed & ~source)),
        filled_hole_count=hole_count,
        filled_hole_pixels=hole_pixels,
        component_count=component_count,
        kept_component_count=kept_count,
        removed_component_pixels=removed_pixels,
        output_pixels=int(np.count_nonzero(cleaned)),
        foreground_fraction=float(np.mean(cleaned)),
        interior_pixel_fraction=_interior_pixel_fraction(cleaned),
    )
    return cleaned, audit


def _robust_scale(values: np.ndarray, floor: float) -> float:
    if values.size == 0:
        return float(floor)
    center = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - center)))
    q25, q75 = np.percentile(values, (25.0, 75.0))
    return max(mad, float(q75 - q25) / 1.349, float(floor))


def _analyse_one_scale(
    mask: np.ndarray,
    *,
    block_rows: int,
    neighborhood_blocks: int,
    minimum_expected_fraction: float,
    minimum_drop_fraction: float,
    maximum_drop_ratio: float,
    minimum_robust_z: float,
    active_row_minimum_fraction: float,
) -> list[DepthCoverageBlock]:
    row_fraction = np.mean(mask, axis=1)
    slices = [
        (start, min(mask.shape[0], start + block_rows))
        for start in range(0, mask.shape[0], block_rows)
    ]
    fractions = np.asarray(
        [float(np.mean(row_fraction[start:stop])) for start, stop in slices], dtype=np.float64
    )
    records: list[DepthCoverageBlock] = []
    for index, (start, stop) in enumerate(slices):
        left = fractions[max(0, index - neighborhood_blocks):index]
        right = fractions[index + 1:index + 1 + neighborhood_blocks]
        has_left = bool(np.any(left >= minimum_expected_fraction))
        has_right = bool(np.any(right >= minimum_expected_fraction))
        neighbours = np.concatenate((left, right))
        expected = float(np.median(neighbours)) if neighbours.size else None
        scale = _robust_scale(neighbours, floor=0.025) if neighbours.size else None
        if scale is not None and expected is not None:
            # A neighbouring failed block should not inflate the local spread so
            # much that an equally bad adjacent block becomes invisible.
            scale = min(scale, max(0.025, expected * 0.15))
        observed = float(fractions[index])
        drop = None if expected is None else expected - observed
        ratio = None if expected is None or expected <= 0.0 else observed / expected
        robust_z = None if drop is None or scale is None else drop / scale
        two_sided = has_left and has_right
        anomaly = bool(
            two_sided
            and expected is not None
            and expected >= minimum_expected_fraction
            and drop is not None
            and drop >= minimum_drop_fraction
            and ratio is not None
            and ratio <= maximum_drop_ratio
            and robust_z is not None
            and robust_z >= minimum_robust_z
        )
        local_rows = row_fraction[start:stop]
        active = local_rows >= active_row_minimum_fraction
        active_fraction = float(np.mean(active)) if local_rows.size else 0.0
        median_active = float(np.median(local_rows[active])) if np.any(active) else 0.0
        if anomaly:
            severity = "warning"
            reason = "two_sided_robust_coverage_drop"
        elif not two_sided and observed < minimum_expected_fraction:
            severity = "info"
            reason = "edge_or_blank_interval_not_auto_flagged"
        else:
            severity = "normal"
            reason = "coverage_within_local_expectation"
        records.append(DepthCoverageBlock(
            scale_rows=int(block_rows),
            start_row=int(start),
            stop_row=int(stop),
            foreground_fraction=observed,
            active_row_fraction=active_fraction,
            median_active_row_fraction=median_active,
            expected_fraction=expected,
            expected_scale=scale,
            drop_fraction=drop,
            drop_ratio=ratio,
            robust_z=robust_z,
            has_left_support=has_left,
            has_right_support=has_right,
            anomaly=anomaly,
            severity=severity,
            reason=reason,
        ))
    return records


def _overlap(left: DepthCoverageBlock, right: DepthCoverageBlock) -> bool:
    # Adjacent anomalous blocks describe one continuous depth interval even
    # though half-open array slices only touch at their shared boundary.
    return min(left.stop_row, right.stop_row) >= max(left.start_row, right.start_row)


def _merge_regions(blocks: Iterable[DepthCoverageBlock]) -> list[DepthCoverageRegion]:
    anomalous = sorted(
        (block for block in blocks if block.anomaly), key=lambda item: (item.start_row, item.stop_row)
    )
    groups: list[list[DepthCoverageBlock]] = []
    for block in anomalous:
        matching = [group for group in groups if any(_overlap(block, item) for item in group)]
        if not matching:
            groups.append([block])
            continue
        target = matching[0]
        target.append(block)
        for other in matching[1:]:
            target.extend(other)
            groups.remove(other)
    regions: list[DepthCoverageRegion] = []
    for group in groups:
        scales = tuple(sorted({item.scale_rows for item in group}))
        observed = min(item.foreground_fraction for item in group)
        expected = max(float(item.expected_fraction or 0.0) for item in group)
        # Requiring agreement at two resolutions avoids treating a narrow tray
        # separator that happens to align with one block boundary as a core-loss event.
        if len(scales) >= 2:
            severity = "critical"
            classification = "persistent_multiscale_coverage_collapse"
        else:
            active_level = max(item.median_active_row_fraction for item in group)
            if active_level >= expected * 0.75:
                severity = "info"
                classification = "likely_structural_gap_with_normal_active_rows"
            else:
                severity = "review"
                classification = "single_scale_possible_mask_loss"
        regions.append(DepthCoverageRegion(
            start_row=min(item.start_row for item in group),
            stop_row=max(item.stop_row for item in group),
            supporting_scales=scales,
            minimum_observed_fraction=observed,
            maximum_expected_fraction=expected,
            severity=severity,
            classification=classification,
        ))
    return regions


def analyse_depth_coverage(
    mask: np.ndarray,
    *,
    block_scales: Sequence[int] = (256, 512),
    neighborhood_blocks: int = 3,
    minimum_expected_fraction: float = 0.20,
    minimum_drop_fraction: float = 0.12,
    maximum_drop_ratio: float = 0.55,
    minimum_robust_z: float = 2.5,
    active_row_minimum_fraction: float = 0.05,
) -> tuple[list[DepthCoverageBlock], list[DepthCoverageRegion]]:
    """Find internal coverage collapses while leaving leading/trailing blank areas alone."""
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    scales = tuple(dict.fromkeys(int(value) for value in block_scales))
    if not scales or any(value < 16 for value in scales):
        raise ValueError("block scales must contain integers >= 16")
    if neighborhood_blocks < 1:
        raise ValueError("neighborhood_blocks must be positive")
    if not 0.0 < maximum_drop_ratio < 1.0:
        raise ValueError("maximum_drop_ratio must be in (0, 1)")
    records: list[DepthCoverageBlock] = []
    for scale in scales:
        records.extend(_analyse_one_scale(
            source,
            block_rows=scale,
            neighborhood_blocks=neighborhood_blocks,
            minimum_expected_fraction=minimum_expected_fraction,
            minimum_drop_fraction=minimum_drop_fraction,
            maximum_drop_ratio=maximum_drop_ratio,
            minimum_robust_z=minimum_robust_z,
            active_row_minimum_fraction=active_row_minimum_fraction,
        ))
    return records, _merge_regions(records)
