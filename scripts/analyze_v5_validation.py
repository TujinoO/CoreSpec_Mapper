#!/usr/bin/env python3
"""Recompute spatial QA metrics for one or more CoreSpec Mapper V5 runs.

The report is deliberately truth-free: it checks output consistency, mask quality,
spatial aggregation, column concentration, and recorded threshold/timing metadata.
It cannot establish mineralogical accuracy without independent pixel or point truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
from typing import Any, Iterable

import numpy as np

try:
    from corespec_mapper.envi import EnviDataset
except ModuleNotFoundError:  # Allow direct execution from an uninstalled checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from corespec_mapper.envi import EnviDataset


POLICIES = ("conservative", "balanced", "sensitive")
SMALL_COMPONENT_MAX_PIXELS = 4


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_single_band(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with EnviDataset(path) as dataset:
        if dataset.info.bands != 1:
            raise ValueError(f"Expected a one-band ENVI raster: {path}")
        data = np.asarray(
            dataset.read_rows(0, dataset.info.lines, bands=[0])[:, :, 0]
        ).copy()
        metadata = dict(dataset.info.metadata)
    return data, metadata


def _component_sizes(mask: np.ndarray) -> list[int]:
    """Return exact 8-connected component sizes using row runs and union-find."""
    source = np.asarray(mask, dtype=bool)
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
        transitions = np.diff(
            np.pad(source[row].astype(np.int8, copy=False), 1, mode="constant")
        )
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        current: list[tuple[int, int, int]] = []
        previous_index = 0
        for start_value, stop_value in zip(starts, stops):
            start, stop = int(start_value), int(stop_value)
            label = len(parents)
            parents.append(label)
            sizes.append(stop - start)
            # Runs touch diagonally when previous_stop == start or
            # previous_start == stop, which implements 8-connectivity.
            while (
                previous_index < len(previous)
                and previous[previous_index][1] < start
            ):
                previous_index += 1
            overlap_index = previous_index
            while (
                overlap_index < len(previous)
                and previous[overlap_index][0] <= stop
            ):
                previous_start, previous_stop, previous_label = previous[overlap_index]
                if previous_stop >= start and previous_start <= stop:
                    label = union(label, previous_label)
                overlap_index += 1
            current.append((start, stop, label))
        previous = current

    roots = {find(label) for label in range(len(parents))}
    return sorted((int(sizes[root]) for root in roots), reverse=True)


def _neighbor_metrics(mask: np.ndarray) -> dict[str, float | int | None]:
    source = np.asarray(mask, dtype=bool)
    pixels = int(np.count_nonzero(source))
    if pixels == 0:
        return {
            "pixels_with_same_class_neighbor": 0,
            "any_same_class_8_neighbor_support_fraction": None,
            "mean_same_class_neighbor_count": None,
            "mean_same_class_8_neighbor_fraction": None,
        }
    counts = np.zeros(source.shape, dtype=np.uint8)
    for row_offset, column_offset in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        target_rows = slice(max(0, row_offset), source.shape[0] + min(0, row_offset))
        target_columns = slice(
            max(0, column_offset), source.shape[1] + min(0, column_offset)
        )
        source_rows = slice(max(0, -row_offset), source.shape[0] - max(0, row_offset))
        source_columns = slice(
            max(0, -column_offset), source.shape[1] - max(0, column_offset)
        )
        counts[target_rows, target_columns] += source[source_rows, source_columns]
    selected = counts[source].astype(np.float64)
    supported = int(np.count_nonzero(selected > 0))
    return {
        "pixels_with_same_class_neighbor": supported,
        "any_same_class_8_neighbor_support_fraction": supported / pixels,
        "mean_same_class_neighbor_count": float(np.mean(selected)),
        "mean_same_class_8_neighbor_fraction": float(np.mean(selected) / 8.0),
    }


def _component_metrics(mask: np.ndarray) -> dict[str, float | int | None]:
    pixels = int(np.count_nonzero(mask))
    sizes = _component_sizes(mask)
    if not sizes:
        return {
            "connectivity": 8,
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_fraction": None,
            "median_component_pixels": None,
            "singleton_component_count": 0,
            "singleton_pixel_fraction": None,
            "small_component_max_pixels": SMALL_COMPONENT_MAX_PIXELS,
            "small_component_count": 0,
            "small_component_pixel_fraction": None,
        }
    small = [size for size in sizes if size <= SMALL_COMPONENT_MAX_PIXELS]
    singletons = sum(size == 1 for size in sizes)
    return {
        "connectivity": 8,
        "component_count": len(sizes),
        "largest_component_pixels": sizes[0],
        "largest_component_fraction": sizes[0] / pixels,
        "median_component_pixels": float(np.median(np.asarray(sizes))),
        "singleton_component_count": int(singletons),
        "singleton_pixel_fraction": singletons / pixels,
        "small_component_max_pixels": SMALL_COMPONENT_MAX_PIXELS,
        "small_component_count": len(small),
        "small_component_pixel_fraction": sum(small) / pixels,
    }


def _column_metrics(
    detected: np.ndarray, material_mask: np.ndarray
) -> dict[str, float | int | None]:
    denominators = np.sum(material_mask, axis=0, dtype=np.int64)
    numerators = np.sum(detected, axis=0, dtype=np.int64)
    valid = denominators > 0
    densities = np.divide(
        numerators,
        denominators,
        out=np.zeros(denominators.shape, dtype=np.float64),
        where=valid,
    )
    valid_densities = densities[valid]
    nonzero_densities = densities[(numerators > 0) & valid]
    detected_pixels = int(np.sum(numerators))
    if valid_densities.size == 0:
        return {
            "material_bearing_columns": 0,
            "detection_bearing_columns": 0,
            "maximum_density": None,
            "maximum_density_column_index": None,
            "maximum_density_column_detection_pixels": 0,
            "maximum_density_column_material_pixels": 0,
            "median_density": None,
            "median_density_all_raster_columns": None,
            "median_density_material_bearing_columns": None,
            "median_nonzero_density": None,
            "p95_density": None,
            "p95_density_all_raster_columns": None,
            "p95_density_material_bearing_columns": None,
            "largest_column_share_of_detections": None,
        }
    maximum_column = int(np.argmax(np.where(valid, densities, -1.0)))
    return {
        "material_bearing_columns": int(np.count_nonzero(valid)),
        "detection_bearing_columns": int(np.count_nonzero((numerators > 0) & valid)),
        "maximum_density": float(densities[maximum_column]),
        "maximum_density_column_index": maximum_column,
        "maximum_density_column_detection_pixels": int(numerators[maximum_column]),
        "maximum_density_column_material_pixels": int(denominators[maximum_column]),
        # Keep the unqualified fields aligned with the built-in quality report,
        # which includes zero-material raster columns as zero density.
        "median_density": float(np.median(densities)),
        "median_density_all_raster_columns": float(np.median(densities)),
        "median_density_material_bearing_columns": float(np.median(valid_densities)),
        "median_nonzero_density": (
            float(np.median(nonzero_densities)) if nonzero_densities.size else None
        ),
        "p95_density": float(np.percentile(densities, 95.0)),
        "p95_density_all_raster_columns": float(np.percentile(densities, 95.0)),
        "p95_density_material_bearing_columns": float(
            np.percentile(valid_densities, 95.0)
        ),
        "largest_column_share_of_detections": (
            float(np.max(numerators) / detected_pixels) if detected_pixels else None
        ),
    }


def _classification_inventory(
    run_dir: Path, material_mask: np.ndarray
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Inspect every ENVI Classification artifact, not just final maps."""
    records: list[dict[str, Any]] = []
    all_ok = True
    legacy_v4_headers: list[str] = []
    for header_path in sorted((run_dir / "groups").rglob("*.hdr")):
        try:
            with EnviDataset(header_path) as dataset:
                if "classification" not in dataset.info.file_type.casefold():
                    continue
                metadata = dataset.info.metadata
                class_names = [str(value) for value in metadata.get("class names", [])]
                class_count = int(metadata.get("classes", len(class_names)))
                values = np.asarray(
                    dataset.read_rows(0, dataset.info.lines, bands=[0])[:, :, 0]
                )
                unique_values, unique_counts = np.unique(values, return_counts=True)
                histogram: dict[str, int] = {}
                for value, count in zip(unique_values, unique_counts):
                    class_id = int(value)
                    label = (
                        class_names[class_id]
                        if 0 <= class_id < len(class_names)
                        else f"INVALID_CLASS_{class_id}"
                    )
                    histogram[f"{class_id}: {label}"] = int(count)
                invalid = (values < 0) | (values >= class_count)
                shape_matches = values.shape == material_mask.shape
                masked_class = next(
                    (
                        index
                        for index, name in enumerate(class_names)
                        if name.casefold() == "masked pixels"
                    ),
                    None,
                )
                masked_mismatch = (
                    int(np.count_nonzero((values == masked_class) != ~material_mask))
                    if masked_class is not None and shape_matches
                    else None
                )
                description = str(metadata.get("description", ""))
                relative_header = header_path.relative_to(run_dir).as_posix()
                if "v4" in description.casefold():
                    legacy_v4_headers.append(relative_header)
                record_ok = (
                    shape_matches
                    and int(np.count_nonzero(invalid)) == 0
                    and masked_mismatch in (None, 0)
                    and class_count == len(class_names)
                )
                all_ok &= record_ok
                records.append(
                    {
                        "header": relative_header,
                        "description": description,
                        "shape": [int(value) for value in values.shape],
                        "shape_matches_material_mask": shape_matches,
                        "class_count": class_count,
                        "class_names": class_names,
                        "class_count_matches_names": class_count == len(class_names),
                        "class_histogram": histogram,
                        "invalid_class_pixels": int(np.count_nonzero(invalid)),
                        "masked_pixel_encoding_mismatch_pixels": masked_mismatch,
                        "integrity_ok": record_ok,
                    }
                )
        except Exception as exc:
            all_ok = False
            records.append(
                {
                    "header": header_path.relative_to(run_dir).as_posix(),
                    "integrity_ok": False,
                    "error": str(exc),
                }
            )
    return records, all_ok, legacy_v4_headers


def _distribution_metrics(
    detected: np.ndarray, material_mask: np.ndarray
) -> dict[str, Any]:
    detected = np.asarray(detected, dtype=bool) & material_mask
    pixels = int(np.count_nonzero(detected))
    material_pixels = int(np.count_nonzero(material_mask))
    return {
        "pixels": pixels,
        "material_mask_fraction": pixels / material_pixels if material_pixels else None,
        "eight_neighbor": _neighbor_metrics(detected),
        "components": _component_metrics(detected),
        "columns": _column_metrics(detected, material_mask),
    }


def _finite_summary(values: Iterable[Any]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _selected_candidate(
    threshold_record: dict[str, Any], percentile: float, absolute: float
) -> dict[str, Any] | None:
    candidates = threshold_record.get("candidates", [])
    matches = [
        candidate
        for candidate in candidates
        if np.isclose(float(candidate.get("percentile_fraction", np.nan)), percentile)
        and np.isclose(float(candidate.get("absolute_threshold_rad", np.nan)), absolute)
    ]
    if not matches:
        return None
    accepted = [candidate for candidate in matches if candidate.get("accepted")]
    return dict((accepted or matches)[0])


def _threshold_metrics(
    group_summary: dict[str, Any], threshold_file_group: dict[str, Any], policy: str
) -> dict[str, Any]:
    resolved = group_summary.get("policies", {}).get(policy, {})
    search = group_summary.get("threshold_search", {}).get(policy, {})
    compact = threshold_file_group.get(policy, {})
    percentile = float(
        search.get(
            "resolved_percentile_fraction",
            compact.get("final", {}).get(
                "column_percentile", resolved.get("scene_percentile", np.nan)
            ),
        )
    )
    absolute = float(
        search.get(
            "resolved_absolute_threshold_rad",
            compact.get("final", {}).get(
                "absolute_sam_rad", resolved.get("scene_sam_threshold_rad", np.nan)
            ),
        )
    )
    selected = _selected_candidate(search, percentile, absolute)
    if selected is None:
        selected = _selected_candidate(compact, percentile, absolute)
    column_thresholds = resolved.get(
        "column_thresholds_rad", search.get("resolved_column_thresholds_rad", [])
    )
    percentile_bounds = search.get(
        "safe_percentile_bounds",
        compact.get("candidate_range", {}).get("column_percentile"),
    )
    absolute_bounds = search.get(
        "safe_absolute_threshold_bounds_rad",
        compact.get("candidate_range", {}).get("absolute_sam_rad"),
    )
    return {
        "column_percentile_fraction": percentile,
        "absolute_sam_threshold_rad": absolute,
        "column_threshold_rad_summary": _finite_summary(column_thresholds),
        "minimum_absorption_depth": resolved.get("minimum_absorption_depth"),
        "minimum_margin": resolved.get(
            "resolved_minimum_margin", resolved.get("minimum_margin")
        ),
        "confidence_threshold": resolved.get("confidence_threshold"),
        "confidence_threshold_by_mineral": resolved.get(
            "resolved_confidence_threshold_by_mineral", {}
        ),
        "minimum_reference_consensus": resolved.get(
            "resolved_consensus_threshold", resolved.get("minimum_reference_consensus")
        ),
        "minimum_evidence_stability": resolved.get(
            "resolved_stability_threshold", resolved.get("minimum_evidence_stability")
        ),
        "source": resolved.get("threshold_source", compact.get("source")),
        "safe_percentile_bounds": percentile_bounds,
        "safe_absolute_threshold_bounds_rad": absolute_bounds,
        "selected_at_percentile_lower_bound": bool(
            percentile_bounds
            and np.isclose(percentile, float(percentile_bounds[0]))
        ),
        "selected_at_percentile_upper_bound": bool(
            percentile_bounds
            and np.isclose(percentile, float(percentile_bounds[-1]))
        ),
        "selected_at_absolute_lower_bound": bool(
            absolute_bounds and np.isclose(absolute, float(absolute_bounds[0]))
        ),
        "selected_at_absolute_upper_bound": bool(
            absolute_bounds and np.isclose(absolute, float(absolute_bounds[-1]))
        ),
        "sample_pixels": search.get("sample_pixels"),
        "evaluated_candidate_count": len(search.get("candidates", compact.get("candidates", []))),
        "selected_proxy_metrics": selected,
        "resolved_reason": search.get("resolved_reason", compact.get("resolved_reason")),
    }


def _preview_inventory(run_dir: Path, groups: Iterable[str]) -> dict[str, Any]:
    preview_dir = run_dir / "previews"
    required = {
        "comparison_balanced.png",
        "comparison_conservative.png",
        "comparison_sensitive.png",
        "comparison_three_profiles.png",
        "stripe_diagnosis.png",
        "swir_background_1600nm.png",
        "legend.json",
    }
    for group in groups:
        for policy in POLICIES:
            required.add(f"group_final_{group}_{policy}.png")
    optional = {"material_mask.png", "aloh_wavelength_subtype.png"}
    actual_paths = {path.name: path for path in preview_dir.iterdir() if path.is_file()}
    actual: dict[str, dict[str, Any]] = {}
    invalid_png_headers: list[str] = []
    for name, path in actual_paths.items():
        record: dict[str, Any] = {"size_bytes": path.stat().st_size}
        if path.suffix.casefold() == ".png":
            with path.open("rb") as stream:
                header = stream.read(24)
            valid_png = (
                len(header) == 24
                and header[:8] == b"\x89PNG\r\n\x1a\n"
                and header[12:16] == b"IHDR"
            )
            record["png_header_valid"] = valid_png
            if valid_png:
                width, height = struct.unpack(">II", header[16:24])
                record["width"] = int(width)
                record["height"] = int(height)
            else:
                invalid_png_headers.append(name)
        actual[name] = record
    return {
        "directory": str(preview_dir.resolve()),
        "required_file_count": len(required),
        "actual_file_count": len(actual),
        "missing_required_files": sorted(required - actual.keys()),
        "optional_files_present": sorted(optional & actual.keys()),
        "optional_files_missing": sorted(optional - actual.keys()),
        "unexpected_files": sorted(actual.keys() - required - optional),
        "zero_byte_files": sorted(
            name for name, record in actual.items() if record["size_bytes"] == 0
        ),
        "invalid_png_headers": sorted(invalid_png_headers),
        "files": dict(sorted(actual.items())),
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary = _read_json(run_dir / "summary.json")
    quality = _read_json(run_dir / "reports" / "quality_report.json")
    thresholds = _read_json(run_dir / "thresholds.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    audit = _read_json(run_dir / "audit.json")

    mask_values, _ = _read_single_band(run_dir / "capability" / "material_mask.dat")
    material_mask = np.isfinite(mask_values) & (mask_values != 0)
    material_pixels = int(np.count_nonzero(material_mask))
    total_pixels = int(material_mask.size)
    mask_manifest = manifest.get("material_mask", {})
    mask_components = _component_sizes(material_mask)
    mask_report = {
        "shape": [int(value) for value in material_mask.shape],
        "pixels": material_pixels,
        "fraction_of_raster": material_pixels / total_pixels if total_pixels else None,
        "source": mask_manifest.get("source"),
        "threshold_method": mask_manifest.get("threshold_method"),
        "threshold": mask_manifest.get("threshold"),
        "spectrally_valid_fraction": mask_manifest.get("spectrally_valid_fraction"),
        "raw_mask_pixels": mask_manifest.get("raw_mask_pixels"),
        "removed_component_pixels": mask_manifest.get("removed_component_pixels"),
        "reported_component_count_before_cleanup": mask_manifest.get("component_count"),
        "reported_component_count_after_cleanup": mask_manifest.get("kept_component_count"),
        "recomputed_component_count": len(mask_components),
        "largest_component_pixels": mask_components[0] if mask_components else 0,
        "largest_component_fraction": (
            mask_components[0] / material_pixels if mask_components and material_pixels else None
        ),
        "flags": mask_manifest.get("flags", []),
        "summary_mask_count_matches_raster": summary.get("mask_pixels") == material_pixels,
        "manifest_mask_count_matches_raster": mask_manifest.get("final_mask_pixels")
        == material_pixels,
    }

    groups_report: dict[str, Any] = {}
    all_integrity_ok = True
    legacy_v4_headers: list[str] = []
    for group, group_summary in summary.get("groups", {}).items():
        group_dir = run_dir / "groups" / group
        policy_arrays: dict[str, np.ndarray] = {}
        policy_metadata: dict[str, dict[str, Any]] = {}
        for policy in POLICIES:
            values, metadata = _read_single_band(group_dir / f"final_{policy}.dat")
            if values.shape != material_mask.shape:
                raise ValueError(
                    f"Classification shape {values.shape} does not match mask "
                    f"{material_mask.shape}: {group}/{policy}"
                )
            policy_arrays[policy] = values
            policy_metadata[policy] = metadata

        class_names = [str(value) for value in policy_metadata["balanced"].get("class names", [])]
        policy_class_names = {
            policy: [str(value) for value in policy_metadata[policy].get("class names", [])]
            for policy in POLICIES
        }
        class_headers_consistent = all(
            names == class_names for names in policy_class_names.values()
        )
        all_integrity_ok &= class_headers_consistent
        excluded_names = {"unclassified", "masked pixels"}
        minerals = [
            (class_index, name)
            for class_index, name in enumerate(class_names)
            if name.casefold() not in excluded_names
        ]
        masked_class = next(
            (
                class_index
                for class_index, name in enumerate(class_names)
                if name.casefold() == "masked pixels"
            ),
            None,
        )

        integrity: dict[str, Any] = {}
        mineral_metrics: dict[str, Any] = {name: {} for _, name in minerals}
        threshold_metrics: dict[str, Any] = {}
        for policy in POLICIES:
            values = policy_arrays[policy]
            detected_classes = np.zeros(values.shape, dtype=bool)
            for class_index, mineral_name in minerals:
                selected = values == class_index
                detected_classes |= selected
                metrics = _distribution_metrics(selected, material_mask)
                reported_count = (
                    group_summary.get("counts", {}).get(policy, {}).get(mineral_name)
                )
                metrics["summary_count"] = reported_count
                metrics["summary_count_matches_raster"] = reported_count == metrics["pixels"]
                mineral_metrics[mineral_name][policy] = metrics
                all_integrity_ok &= bool(metrics["summary_count_matches_raster"])

            outside_detections = int(np.count_nonzero(detected_classes & ~material_mask))
            if masked_class is None:
                masked_encoding_mismatch = None
            else:
                expected_masked = ~material_mask
                actual_masked = values == masked_class
                masked_encoding_mismatch = int(np.count_nonzero(expected_masked != actual_masked))
            integrity[policy] = {
                "detected_mineral_pixels_outside_material_mask": outside_detections,
                "masked_pixel_encoding_mismatch_pixels": masked_encoding_mismatch,
                "class_values": [int(value) for value in np.unique(values)],
            }
            all_integrity_ok &= outside_detections == 0
            all_integrity_ok &= masked_encoding_mismatch in (None, 0)
            threshold_metrics[policy] = _threshold_metrics(
                group_summary, thresholds.get(group, {}), policy
            )

        conservative = policy_arrays["conservative"]
        balanced = policy_arrays["balanced"]
        sensitive = policy_arrays["sensitive"]
        mineral_class_ids = np.asarray([class_id for class_id, _ in minerals])
        conservative_detected = np.isin(conservative, mineral_class_ids)
        balanced_detected = np.isin(balanced, mineral_class_ids)
        policy_nested_conservative = not np.any(
            conservative_detected & (conservative != balanced)
        )
        policy_nested_balanced = not np.any(balanced_detected & (balanced != sensitive))
        all_integrity_ok &= policy_nested_conservative and policy_nested_balanced

        groups_report[group] = {
            "class_names": class_names,
            "minerals": mineral_metrics,
            "thresholds": threshold_metrics,
            "classification_integrity": {
                "profiles": integrity,
                "class_headers_consistent_across_profiles": class_headers_consistent,
                "conservative_is_same_class_subset_of_balanced": policy_nested_conservative,
                "balanced_is_same_class_subset_of_sensitive": policy_nested_balanced,
            },
            "cleanup": group_summary.get("cleanup", {}),
            "stripe_pixels_union": group_summary.get("stripe_pixels_union"),
        }

    timing = dict(manifest.get("timing", {}))
    total_seconds = timing.get("total_seconds")
    audit_seconds = timing.get("audit_snapshot_seconds", audit.get("audit_elapsed_seconds"))
    mapping_seconds = timing.get(
        "mapping_seconds_before_qa", summary.get("elapsed_seconds_before_qa")
    )
    if all(value is not None for value in (total_seconds, audit_seconds, mapping_seconds)):
        timing["qa_preview_manifest_seconds_by_difference"] = max(
            0.0, float(total_seconds) - float(audit_seconds) - float(mapping_seconds)
        )

    quality_markdown = (run_dir / "reports" / "quality_report.md").read_text(
        encoding="utf-8-sig"
    )
    if "v4" in quality_markdown.casefold():
        legacy_v4_headers.append("reports/quality_report.md")

    classification_artifacts, classification_inventory_ok, inventory_legacy = (
        _classification_inventory(run_dir, material_mask)
    )
    legacy_v4_headers.extend(inventory_legacy)
    all_integrity_ok &= classification_inventory_ok
    threshold_records = [
        group_record["thresholds"][policy]
        for group_record in groups_report.values()
        for policy in POLICIES
    ]
    threshold_boundary_summary = {
        "selection_count": len(threshold_records),
        "absolute_upper_bound_count": sum(
            bool(record["selected_at_absolute_upper_bound"])
            for record in threshold_records
        ),
        "absolute_lower_bound_count": sum(
            bool(record["selected_at_absolute_lower_bound"])
            for record in threshold_records
        ),
        "percentile_upper_bound_count": sum(
            bool(record["selected_at_percentile_upper_bound"])
            for record in threshold_records
        ),
        "percentile_lower_bound_count": sum(
            bool(record["selected_at_percentile_lower_bound"])
            for record in threshold_records
        ),
    }

    return {
        "run_directory": str(run_dir),
        "run_name": run_dir.name,
        "software_version": summary.get("version", manifest.get("version")),
        "analysis_image": summary.get("analysis_image"),
        "spectral_domain": summary.get("spectral_domain"),
        "data_physics": summary.get("data_physics"),
        "ground_truth": {
            "available": False,
            "scope_note": (
                "No pixel-level or point-level mineral truth is attached to this run; "
                "reported metrics assess consistency and spatial/artifact behavior only."
            ),
        },
        "material_mask": mask_report,
        "timing_seconds": timing,
        "threshold_boundary_summary": threshold_boundary_summary,
        "quality_report": quality,
        "groups": groups_report,
        "classification_envi_inventory": {
            "artifact_count": len(classification_artifacts),
            "all_artifacts_integrity_ok": classification_inventory_ok,
            "artifacts": classification_artifacts,
        },
        "preview_inventory": _preview_inventory(run_dir, groups_report.keys()),
        "integrity": {
            "all_summary_counts_and_classification_masks_consistent": all_integrity_ok,
            "legacy_v4_provenance_labels": sorted(set(legacy_v4_headers)),
        },
    }


def build_report(run_directories: Iterable[Path]) -> dict[str, Any]:
    runs = [analyze_run(path) for path in run_directories]
    return {
        "schema": "corespec_mapper_v5_validation_analysis",
        "schema_version": 1,
        "metric_definitions": {
            "material_mask_fraction": "Detected mineral pixels / final material-mask pixels.",
            "any_same_class_8_neighbor_support_fraction": (
                "Fraction of detected pixels with at least one same-class pixel in the "
                "eight immediately adjacent cells."
            ),
            "mean_same_class_8_neighbor_fraction": (
                "Mean same-class neighbor count divided by eight."
            ),
            "small_component_pixel_fraction": (
                f"Fraction of detected pixels in 8-connected components of at most "
                f"{SMALL_COMPONENT_MAX_PIXELS} pixels."
            ),
            "column_density": (
                "Per-column detected pixels / per-column material-mask pixels. The "
                "unqualified median includes every raster column (matching built-in QA); "
                "a separate material-bearing-column median is also reported."
            ),
        },
        "limitations": [
            "No pixel-level or point-level mineral ground truth is available.",
            "PPT interpretations and drill-log intervals are qualitative/weak "
            "labels, not truth masks.",
            "Metrics from downsampled harmonized cubes do not replace full-resolution acceptance.",
        ],
        "runs": runs,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", type=Path, nargs="+", help="One or more completed V5 run directories."
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Write JSON to this file instead of stdout."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.runs)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
