from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .envi import EnviDataset
from .v4_models import SampleBlock, SamplePlan


def build_stratified_sample_plan(
    mask: np.ndarray,
    *,
    desired_blocks: int = 12,
    block_rows: int = 64,
    min_valid_pixels: int = 128,
    forced_lines: Sequence[int] = (),
) -> SamplePlan:
    """Choose non-overlapping, full-width blocks distributed across the full depth."""
    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 2 or not np.any(valid):
        raise ValueError("A non-empty two-dimensional core mask is required")
    lines = valid.shape[0]
    desired = min(max(int(desired_blocks), 1), 32)
    height = min(max(int(block_rows), 8), lines)
    row_counts = np.sum(valid, axis=1)
    blocks: list[SampleBlock] = []

    def add_block(center: int, reason: str) -> None:
        start = min(max(int(center) - height // 2, 0), max(lines - height, 0))
        stop = min(start + height, lines)
        if any(not (stop <= block.start_line or start >= block.stop_line) for block in blocks):
            return
        count = int(np.count_nonzero(valid[start:stop]))
        if count >= min_valid_pixels:
            blocks.append(SampleBlock(start, stop, reason, count))

    for line in forced_lines:
        if 0 <= int(line) < lines:
            add_block(int(line), "forced_roi_or_validation_line")

    edges = np.linspace(0, lines, desired + 1, dtype=int)
    for segment_index, (segment_start, segment_stop) in enumerate(zip(edges[:-1], edges[1:])):
        if segment_stop <= segment_start:
            continue
        best_center = (segment_start + segment_stop) // 2
        best_score = -1
        search_start = max(0, segment_start - height // 2)
        search_stop = min(lines, segment_stop + height // 2)
        step = max(1, height // 4)
        for start in range(search_start, max(search_stop - height + 1, search_start + 1), step):
            stop = min(start + height, lines)
            count = int(np.sum(row_counts[start:stop]))
            if count > best_score:
                best_score = count
                best_center = (start + stop) // 2
        add_block(best_center, f"depth_stratum_{segment_index + 1:02d}")

    if not blocks:
        center = int(np.argmax(np.convolve(row_counts, np.ones(height, dtype=np.int64), mode="same")))
        add_block(center, "validity_fallback")
    if not blocks:
        raise ValueError("No stratified block contains enough valid core pixels")
    blocks.sort(key=lambda item: item.start_line)
    return SamplePlan(
        blocks=tuple(blocks),
        total_lines=lines,
        total_valid_pixels=int(sum(block.valid_pixels for block in blocks)),
    )


def sample_plan_mask(plan: SamplePlan, mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool)
    if valid.shape[0] != plan.total_lines:
        raise ValueError("Sample plan and mask line counts do not match")
    selected = np.zeros_like(valid)
    for block in plan.blocks:
        selected[block.start_line : block.stop_line] = valid[block.start_line : block.stop_line]
    return selected


def read_sample_blocks(
    dataset: EnviDataset,
    mask: np.ndarray,
    plan: SamplePlan,
    *,
    bands: Sequence[int] | None = None,
    maximum_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sampled block rows and their original line indices.

    ``SamplePlan`` records the full calibration regions used for score
    percentiles.  Reading every spectral row from those regions is unnecessary
    for sensor quality, library selection, and detector-column estimation and
    can consume several GiB.  ``maximum_rows`` keeps an evenly distributed,
    deterministic subset for those spectral statistics while leaving the plan
    itself unchanged.
    """
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != (dataset.info.lines, dataset.info.samples):
        raise ValueError("Dataset and mask dimensions do not match")
    available = np.concatenate([
        np.arange(block.start_line, block.stop_line, dtype=np.int32) for block in plan.blocks
    ])
    if maximum_rows is not None and available.size > int(maximum_rows):
        limit = max(int(maximum_rows), len(plan.blocks))
        selected_parts: list[np.ndarray] = []
        remaining = limit
        for index, block in enumerate(plan.blocks):
            block_lines = np.arange(block.start_line, block.stop_line, dtype=np.int32)
            blocks_left = len(plan.blocks) - index
            quota = max(1, remaining // blocks_left)
            quota = min(quota, block_lines.size)
            positions = np.linspace(0, block_lines.size - 1, quota, dtype=np.int64)
            selected_parts.append(block_lines[np.unique(positions)])
            remaining -= selected_parts[-1].size
        available = np.concatenate(selected_parts)
    cubes = [np.asarray(dataset.read_rows(int(line), int(line) + 1, bands=bands)) for line in available]
    sampled_mask = valid[available]
    return np.concatenate(cubes, axis=0), sampled_mask, available


def iter_plan_blocks(plan: SamplePlan) -> Iterable[tuple[int, int]]:
    for block in plan.blocks:
        yield block.start_line, block.stop_line
