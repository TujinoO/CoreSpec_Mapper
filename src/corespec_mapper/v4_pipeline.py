from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping, Sequence
import csv
import json

import numpy as np

from .algorithms import robust_column_bias, savgol_smooth, spectral_angles
from .artifacts import (
    edge_risk_score,
    filter_artifacts,
    fixed_column_risk_score,
    repeated_segment_column_stripe_mask,
)
from .catalog import GroupDefinition, MineralCatalog
from .envi import EnviDataset, derive_mask, subset_spatial_metadata, wavelength_indices, write_envi
from .sampling import build_stratified_sample_plan, read_sample_blocks, sample_plan_mask
from .sensor import build_sensor_capability_card, extract_fwhm_nm, resolve_mineral_support
from .swir_expert import (
    ReferenceEvidence,
    aggregate_reference_scores,
    continuum_absorption,
    feature_values,
    mineral_feature_gate,
    reference_evidence,
)
from .v4_calibration import POLICY_ORDER, estimate_relative_spectral_noise, policy_candidate, resolve_group_policies
from .v4_library import V4EnsembleLibrary, build_v4_library_ensemble, write_v4_library_artifacts
from .v4_models import CancellationToken, ProgressEvent, SamplePlan, SupportLevel
from .v5_calibration import calibrate_group_thresholds
from .v5_project_calibration import (
    calibrate_detection_settings,
    learn_evidence_gate,
    learn_feature_gate_override,
    learn_score_offsets,
)


Progress = Callable[[ProgressEvent], None]


@dataclass(frozen=True)
class PreparedRunInputs:
    """Reusable scientific inputs produced by a single audit pass.

    V5 prepares the mask, stratified sample, sensor card, detector correction,
    and built-in reference ensemble once.  Passing this snapshot into the
    mapper prevents the former hidden second audit and library scan while the
    unchanged V4 caller can continue to let ``run_v4`` prepare everything.
    """

    full_mask: np.ndarray
    sample_plan: SamplePlan
    sample_cube: np.ndarray
    sample_mask: np.ndarray
    processed_sample: np.ndarray
    sensor_card: Any
    column_bias: np.ndarray
    requested_minerals: tuple[str, ...]
    internal_minerals: tuple[str, ...]
    ensemble: V4EnsembleLibrary
    project_weak_labels: Any | None = None


def _default_progress(event: ProgressEvent) -> None:
    print(f"[{event.overall_fraction * 100:6.2f}%] {event.stage}: {event.message}", flush=True)


def _save_json(value: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: None for key in ("p01", "p05", "p10", "p50", "p90", "p95", "p99")}
    result = np.percentile(finite, [1, 5, 10, 50, 90, 95, 99])
    return {key: float(item) for key, item in zip(("p01", "p05", "p10", "p50", "p90", "p95", "p99"), result)}


def _emit(
    progress: Progress,
    started: float,
    stage: str,
    fraction: float,
    message: str,
    *,
    stage_fraction: float | None = None,
    block: tuple[int, int] | None = None,
) -> None:
    elapsed = monotonic() - started
    remaining = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
    progress(ProgressEvent(stage, float(np.clip(fraction, 0.0, 1.0)), message, stage_fraction, block, elapsed, remaining))


def _processed(
    values: np.ndarray,
    input_is_smoothed: bool,
    window: int,
    order: int,
    column_bias: np.ndarray | None = None,
    correction_strength: float = 0.0,
) -> np.ndarray:
    cube = np.asarray(values, dtype=np.float64)
    result = cube if input_is_smoothed else savgol_smooth(cube, window, order)
    if column_bias is not None and correction_strength > 0.0:
        result = result - float(correction_strength) * np.asarray(column_bias, dtype=np.float64)[None, :, :]
    return result


def _valid_indices(
    wavelengths: np.ndarray,
    windows: Sequence[Sequence[float]],
    bad_bands: Sequence[int],
    references: np.ndarray,
) -> np.ndarray:
    indices = wavelength_indices(wavelengths, windows)
    bad = set(int(item) for item in bad_bands)
    indices = np.asarray([item for item in indices if int(item) not in bad and np.all(np.isfinite(references[:, item]))])
    if indices.size < 3:
        raise ValueError(f"Fewer than three usable bands remain in diagnostic windows {windows}")
    return indices


@dataclass(frozen=True)
class _PreparedGroup:
    definition: GroupDefinition
    output_minerals: tuple[str, ...]
    mineral_order: tuple[str, ...]
    reference_indices: np.ndarray
    reference_minerals: tuple[str, ...]
    references: np.ndarray
    detection_indices: np.ndarray
    classification_indices: np.ndarray
    reference_absorption: np.ndarray
    continuum_method: str = "upper_hull"


@dataclass(frozen=True)
class _PreparedDetectionFamily:
    family_id: str
    group_ids: tuple[str, ...]
    references: np.ndarray
    indices: np.ndarray


def _prepare_groups(
    catalog: MineralCatalog,
    ensemble: V4EnsembleLibrary,
    wavelengths: np.ndarray,
    requested: Sequence[str],
    internal: Sequence[str],
    bad_bands: Sequence[int],
    continuum_method: str = "upper_hull",
) -> list[_PreparedGroup]:
    output_by_group = catalog.active_groups(requested)
    internal_by_group = catalog.active_groups(internal)
    result: list[_PreparedGroup] = []
    for group_id, output_minerals in output_by_group.items():
        definition = catalog.group(group_id)
        mineral_order = tuple(
            mineral for mineral in internal_by_group.get(group_id, ()) if mineral in ensemble.mineral_labels
        )
        reference_indices = np.asarray([
            index
            for index, (candidate_group, mineral) in enumerate(zip(ensemble.group_labels, ensemble.mineral_labels))
            if candidate_group == group_id and mineral in mineral_order
        ], dtype=np.int64)
        references = ensemble.spectra[reference_indices]
        detection = _valid_indices(wavelengths, definition.detection_windows_nm, bad_bands, references)
        classification = _valid_indices(wavelengths, definition.classification_windows_nm, bad_bands, references)
        absorption, _ = continuum_absorption(
            references[:, classification],
            wavelengths[classification],
            method=continuum_method,
        )
        result.append(_PreparedGroup(
            definition,
            tuple(output_minerals),
            mineral_order,
            reference_indices,
            tuple(ensemble.mineral_labels[index] for index in reference_indices),
            references,
            detection,
            classification,
            absorption,
            continuum_method,
        ))
    return result


def _prepare_detection_families(
    groups: Sequence[_PreparedGroup],
    wavelengths: np.ndarray,
    bad_bands: Sequence[int],
    settings: Mapping[str, Any] | None,
) -> tuple[_PreparedDetectionFamily, ...]:
    if not settings:
        return ()
    by_id = {group.definition.group_id: group for group in groups}
    result: list[_PreparedDetectionFamily] = []
    for family_id, value in settings.items():
        if not isinstance(value, Mapping):
            continue
        group_ids = tuple(str(item) for item in value.get("groups", ()) if str(item) in by_id)
        if len(group_ids) < 2:
            continue
        references = np.vstack([by_id[group_id].references for group_id in group_ids])
        windows = value.get("windows_nm", ())
        indices = _valid_indices(wavelengths, windows, bad_bands, references)
        result.append(_PreparedDetectionFamily(str(family_id), group_ids, references, indices))
    return tuple(result)


def _shared_detection_sam(
    cube: np.ndarray,
    mask: np.ndarray,
    family: _PreparedDetectionFamily,
) -> np.ndarray:
    result = np.full(mask.shape, np.nan, dtype=np.float32)
    indices = family.indices
    valid = mask & np.all(np.isfinite(cube[..., indices]), axis=-1) & np.all(cube[..., indices] > 0, axis=-1)
    if np.any(valid):
        angles = spectral_angles(cube[..., indices][valid], family.references[:, indices])
        result[valid] = np.min(angles, axis=1).astype(np.float32)
    return result


def _group_sam(cube: np.ndarray, mask: np.ndarray, group: _PreparedGroup) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(mask.shape, np.nan, dtype=np.float32)
    mineral_result = np.full((len(group.mineral_order), *mask.shape), np.nan, dtype=np.float32)
    indices = group.detection_indices
    valid = mask & np.all(np.isfinite(cube[..., indices]), axis=-1) & np.all(cube[..., indices] > 0, axis=-1)
    if not np.any(valid):
        return result, mineral_result
    angles = spectral_angles(cube[..., indices][valid], group.references[:, indices])
    scores, _ = aggregate_reference_scores(
        angles,
        group.reference_minerals,
        group.mineral_order,
        best_k=group.definition.detection_best_k,
    )
    result[valid] = np.min(scores, axis=1).astype(np.float32)
    for mineral_index in range(scores.shape[1]):
        mineral_result[mineral_index][valid] = scores[:, mineral_index].astype(np.float32)
    return result, mineral_result


def _subclass_evidence(
    cube: np.ndarray,
    candidate: np.ndarray,
    group: _PreparedGroup,
    wavelengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], ReferenceEvidence | None]:
    indices = group.classification_indices
    valid = candidate & np.all(np.isfinite(cube[..., indices]), axis=-1) & np.all(cube[..., indices] > 0, axis=-1)
    if not np.any(valid):
        return valid, np.empty((0, len(group.mineral_order))), np.empty((0, len(group.mineral_order))), np.empty(0), {}, None
    absorption, depth = continuum_absorption(
        cube[..., indices][valid],
        wavelengths[indices],
        method=group.continuum_method,
    )
    evidence = reference_evidence(group.definition, absorption, group.reference_absorption, wavelengths[indices])
    scores, consensus = aggregate_reference_scores(
        evidence.total,
        group.reference_minerals,
        group.mineral_order,
        best_k=group.definition.classification_best_k,
    )
    features = feature_values(group.definition, absorption, wavelengths[indices])
    return valid, scores, consensus, depth, features, evidence


def _classification_output(labels: np.ndarray, mask: np.ndarray, class_count: int) -> np.ndarray:
    output = np.asarray(labels, dtype=np.uint8).copy()
    output[~mask] = class_count + 1
    return output


def _candidate_output(candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros(candidate.shape, dtype=np.uint8)
    output[candidate] = 1
    output[~mask] = 2
    return output


def _competition_margin_pass(
    margin: np.ndarray,
    threshold: float,
    *,
    mineral_count: int,
    classified: np.ndarray,
) -> np.ndarray:
    """Apply a winner-margin gate only when a real competitor is available.

    A one-mineral group has no first-versus-second margin.  Requiring that
    non-existent value to be finite made every such group impossible to accept,
    including V5 montmorillonite-only runs.
    """

    labels_available = np.asarray(classified, dtype=bool)
    if mineral_count <= 1:
        return labels_available.copy()
    values = np.asarray(margin, dtype=np.float64)
    return labels_available & np.isfinite(values) & (values >= float(threshold))


def _reference_consensus_threshold(
    catalog_threshold: float,
    policy: str,
    reference_count: int,
) -> float:
    count = max(int(reference_count), 1)
    required = 2 if policy == "conservative" else 1
    return min(float(catalog_threshold), min(required, count) / count)


def _winning_mineral_sff_quality(
    reference_quality: np.ndarray,
    reference_minerals: Sequence[str],
    mineral_order: Sequence[str],
    winning_labels: np.ndarray,
) -> np.ndarray:
    """Return best SFF quality for the winning mineral, not any reference.

    Taking the maximum over the whole library allowed a confuser's excellent
    SFF fit to support the selected mineral.  The late-stage SFF gate must use
    only references belonging to the actual mineral winner.
    """

    quality = np.asarray(reference_quality, dtype=np.float64)
    labels = np.asarray(winning_labels, dtype=np.int64)
    if quality.ndim != 2 or labels.shape != (quality.shape[0],):
        raise ValueError("Reference quality and winning labels have incompatible shapes")
    reference_labels = np.asarray(reference_minerals, dtype=object)
    by_mineral = np.full((quality.shape[0], len(mineral_order)), np.nan, dtype=np.float64)
    for mineral_index, mineral_id in enumerate(mineral_order):
        selected = np.flatnonzero(reference_labels == mineral_id)
        if selected.size:
            subset = quality[:, selected]
            finite = np.isfinite(subset)
            safe = np.where(finite, subset, -np.inf)
            best = np.max(safe, axis=1)
            by_mineral[:, mineral_index] = np.where(np.any(finite, axis=1), best, np.nan)
    valid = (labels >= 0) & (labels < len(mineral_order))
    result = np.full(labels.shape, np.nan, dtype=np.float64)
    rows = np.flatnonzero(valid)
    result[rows] = by_mineral[rows, labels[rows]]
    return result


def _enforce_profile_nesting(final_profiles: Mapping[str, np.ndarray]) -> None:
    balanced = final_profiles["balanced"]
    sensitive = final_profiles["sensitive"]
    conservative = final_profiles["conservative"]
    balanced_present = balanced > 0
    sensitive[balanced_present] = balanced[balanced_present]
    conservative_present = conservative > 0
    balanced[conservative_present] = conservative[conservative_present]
    sensitive[conservative_present] = conservative[conservative_present]


def _apply_post_nesting_repeated_column_filter(
    final_profiles: Mapping[str, np.ndarray],
    valid_mask: np.ndarray,
    settings: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Remove repeated detector-column peaks after policy nesting is restored.

    Profile-specific cleanup can remove a stripe from the sensitive result and
    then nesting can restore it from balanced or conservative.  Detect all
    residuals on the nested maps first, then cascade sensitive removals into
    balanced/conservative and balanced removals into conservative.  This keeps
    the three products nested without allowing nesting to reintroduce noise.
    """

    valid = np.asarray(valid_mask, dtype=bool)
    maximum_passes = max(
        1,
        int(settings.get("maximum_terminal_repeated_column_passes", 4)),
    )
    detection_policy = str(
        settings.get("terminal_repeated_column_detection_policy", "balanced")
    )
    if detection_policy not in POLICY_ORDER:
        raise ValueError(
            "terminal_repeated_column_detection_policy must be conservative, balanced, or sensitive"
        )
    applied_total = {policy: np.zeros_like(valid) for policy in POLICY_ORDER}
    detected_total = {policy: np.zeros_like(valid) for policy in POLICY_ORDER}
    inherited_total = {policy: np.zeros_like(valid) for policy in POLICY_ORDER}
    column_totals = {policy: set() for policy in POLICY_ORDER}
    maximum_repetition = {policy: 0 for policy in POLICY_ORDER}
    dominant_corridor_totals = {policy: 0 for policy in POLICY_ORDER}
    pass_records: dict[str, list[dict[str, Any]]] = {
        policy: [] for policy in POLICY_ORDER
    }
    converged = False

    for pass_index in range(1, maximum_passes + 1):
        pass_settings = dict(settings)
        if pass_index > 1:
            # One dominance decision is sufficient.  Later passes may remove
            # newly exposed exact narrow peaks, but must not peel the same
            # detector corridor inward column by column.
            pass_settings["dominant_repeated_corridor_enabled"] = False
        detected: dict[str, np.ndarray] = {}
        current_records: dict[str, dict[str, Any]] = {}
        for policy in POLICY_ORDER:
            labels = np.asarray(final_profiles[policy])
            policy_mask = np.zeros_like(valid)
            class_records: dict[str, Any] = {}
            repeated_columns: set[int] = set()
            pass_maximum_repetition = 0
            for class_id in range(1, int(np.max(labels)) + 1):
                class_mask = labels == class_id
                if not np.any(class_mask):
                    continue
                class_stripe, record = repeated_segment_column_stripe_mask(
                    class_mask,
                    valid,
                    pass_settings,
                    policy=detection_policy,
                )
                class_stripe &= class_mask
                policy_mask |= class_stripe
                repeated_columns.update(
                    int(value) for value in record["repeated_columns"]
                )
                pass_maximum_repetition = max(
                    pass_maximum_repetition,
                    int(record["maximum_segment_repetition"]),
                )
                class_records[str(class_id)] = record
            detected[policy] = policy_mask
            current_records[policy] = {
                "pass": pass_index,
                "detected_pixels": int(np.count_nonzero(policy_mask)),
                "repeated_column_count": len(repeated_columns),
                "maximum_segment_repetition": pass_maximum_repetition,
                "repeated_columns": sorted(repeated_columns),
                "classes": class_records,
            }
            detected_total[policy] |= policy_mask
            column_totals[policy].update(repeated_columns)
            maximum_repetition[policy] = max(
                maximum_repetition[policy],
                pass_maximum_repetition,
            )
            dominant_corridor_totals[policy] += sum(
                int(record.get("dominant_corridor_count", 0))
                for record in class_records.values()
            )

        inherited = np.zeros_like(valid)
        any_applied = False
        for policy in reversed(POLICY_ORDER):
            inherited |= detected[policy]
            labels = final_profiles[policy]
            policy_applied = inherited & (labels > 0)
            policy_inherited = policy_applied & ~detected[policy]
            labels[policy_applied] = 0
            applied_total[policy] |= policy_applied
            inherited_total[policy] |= policy_inherited
            current_records[policy]["applied_pixels"] = int(
                np.count_nonzero(policy_applied)
            )
            current_records[policy]["inherited_pixels"] = int(
                np.count_nonzero(policy_inherited)
            )
            pass_records[policy].append(current_records[policy])
            any_applied |= bool(np.any(policy_applied))
        if not any_applied:
            converged = True
            break

    records = {
        policy: {
            "detected_pixels": int(np.count_nonzero(detected_total[policy])),
            "applied_pixels": int(np.count_nonzero(applied_total[policy])),
            "inherited_pixels": int(np.count_nonzero(inherited_total[policy])),
            "repeated_column_count": len(column_totals[policy]),
            "maximum_segment_repetition": maximum_repetition[policy],
            "dominant_corridor_count": dominant_corridor_totals[policy],
            "repeated_columns": sorted(column_totals[policy]),
            "passes_evaluated": len(pass_records[policy]),
            "converged": converged,
            "detection_policy": detection_policy,
            "passes": pass_records[policy],
        }
        for policy in POLICY_ORDER
    }
    return applied_total, records


def _arbitrate_cross_group_profiles(
    profiles_by_group: Mapping[str, Mapping[str, np.ndarray]],
    confidence_by_group: Mapping[str, np.ndarray],
    group_order: Sequence[str],
) -> dict[str, Any]:
    """Make related discovery groups mutually exclusive with one shared winner."""

    active = [group_id for group_id in group_order if group_id in profiles_by_group]
    if len(active) < 2:
        return {"status": "not_required", "groups": active}
    shape = profiles_by_group[active[0]]["sensitive"].shape
    if any(profiles_by_group[group_id]["sensitive"].shape != shape for group_id in active):
        raise ValueError("Cross-group competition profiles must share one spatial shape")
    confidence_stack = np.stack(
        [np.asarray(confidence_by_group[group_id], dtype=np.float32) for group_id in active]
    )
    sensitive_stack = np.stack(
        [np.asarray(profiles_by_group[group_id]["sensitive"]) > 0 for group_id in active]
    )
    available_confidence = np.where(sensitive_stack, confidence_stack, -np.inf)
    any_available = np.any(sensitive_stack, axis=0)
    winner = np.argmax(available_confidence, axis=0)
    winner = np.where(any_available, winner, -1)
    record: dict[str, Any] = {
        "status": "resolved",
        "method": "shared_confidence_cross_group_winner",
        "groups": active,
        "profiles": {},
    }
    for policy in POLICY_ORDER:
        before_stack = np.stack(
            [np.asarray(profiles_by_group[group_id][policy]) > 0 for group_id in active]
        )
        before_overlap = int(np.count_nonzero(np.sum(before_stack, axis=0) > 1))
        removed_by_group: dict[str, int] = {}
        for group_index, group_id in enumerate(active):
            labels = profiles_by_group[group_id][policy]
            removed = (labels > 0) & (winner != group_index)
            removed_by_group[group_id] = int(np.count_nonzero(removed))
            labels[removed] = 0
        after_stack = np.stack(
            [np.asarray(profiles_by_group[group_id][policy]) > 0 for group_id in active]
        )
        record["profiles"][policy] = {
            "overlap_pixels_before": before_overlap,
            "overlap_pixels_after": int(np.count_nonzero(np.sum(after_stack, axis=0) > 1)),
            "removed_by_group": removed_by_group,
        }
    for group_id in active:
        _enforce_profile_nesting(profiles_by_group[group_id])
    return record


def _expand_internal_minerals(catalog: MineralCatalog, requested: Sequence[str]) -> tuple[str, ...]:
    result = list(dict.fromkeys(str(item).casefold() for item in requested))
    for mineral_id in tuple(result):
        mineral = catalog.mineral(mineral_id)
        for confuser in mineral.confusers:
            if confuser in catalog.minerals and catalog.mineral(confuser).group_id == mineral.group_id and confuser not in result:
                result.append(confuser)
    return tuple(result)


def _write_capability(
    output: Path,
    card: Any,
    support: Mapping[str, Any],
    sample_plan: SamplePlan,
) -> None:
    capability = output / "capability"
    capability.mkdir(parents=True, exist_ok=True)
    _save_json(card.to_dict(), capability / "sensor_capability_card.json")
    _save_json(sample_plan.to_dict(), capability / "sample_plan.json")
    records = [item.to_dict() for item in support.values()]
    _save_json(records, capability / "mineral_observability.json")
    with (capability / "mineral_observability.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mineral_id", "display_name", "support_level", "score", "expert", "coverage", "valid_bands", "reference_count", "reasons"])
        for item in support.values():
            writer.writerow([
                item.mineral_id,
                item.display_name,
                item.level.value,
                f"{item.score:.6f}",
                item.selected_expert or "",
                f"{item.required_windows_covered:.6f}",
                item.valid_bands_in_required_windows,
                item.reference_count,
                "; ".join(item.reasons),
            ])


def run_v4(
    config: dict[str, Any],
    output_dir: str | Path,
    start_line: int = 0,
    stop_line: int | None = None,
    *,
    progress: Progress = _default_progress,
    cancel_token: CancellationToken | None = None,
    catalog: MineralCatalog | None = None,
    prepared: PreparedRunInputs | None = None,
) -> dict[str, Any]:
    started = monotonic()
    token = cancel_token or CancellationToken()
    catalog = catalog or MineralCatalog.load(config.get("v4", {}).get("catalog_path"))
    v4 = config.get("v4")
    if not isinstance(v4, dict):
        raise ValueError("The configuration does not contain a V4 section")
    algorithm_version = str(v4.get("algorithm_version", "V4.0.0"))
    algorithm_short = algorithm_version.split(".", 1)[0]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False) if not output.exists() else None
    _emit(progress, started, "audit", 0.01, "Opening ENVI image and mask")

    image = EnviDataset(config["analysis_image"])
    mask_dataset: EnviDataset | None = None
    try:
        wavelengths = image.info.wavelengths_nm
        if wavelengths is None:
            raise ValueError("The analysis image requires an ENVI wavelength vector")
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        start = int(start_line)
        stop = image.info.lines if stop_line is None else int(stop_line)
        if start < 0 or stop > image.info.lines or start >= stop:
            raise ValueError(f"Invalid output line interval [{start}, {stop})")
        sampling = v4.get("sampling", {})
        input_is_smoothed = bool(config.get("analysis_input_is_smoothed", False))
        sg_window = int(v4.get("preprocessing", {}).get("sg_window", config.get("sg_window", 11)))
        sg_order = int(v4.get("preprocessing", {}).get("sg_polyorder", config.get("sg_polyorder", 2)))
        preprocessing_state = "already_sg_smoothed" if input_is_smoothed else f"sg_{sg_window}_{sg_order}"
        correction_config = v4.get("artifact_control", {}).get("column_spectral_correction", {})
        correction_enabled = bool(correction_config.get("enabled", True))
        correction_strength = float(correction_config.get("strength", 0.70)) if correction_enabled else 0.0
        if prepared is None:
            if "analysis_mask" not in config:
                raise ValueError("V4 requires analysis_mask when no prepared audit snapshot is supplied")
            mask_dataset = EnviDataset(config["analysis_mask"])
            if (mask_dataset.info.lines, mask_dataset.info.samples) != (image.info.lines, image.info.samples):
                raise ValueError("Analysis mask and image spatial dimensions do not match")
            default_bands = [
                min(20, mask_dataset.info.bands - 1),
                min(mask_dataset.info.bands // 2, mask_dataset.info.bands - 1),
                min(190, mask_dataset.info.bands - 1),
            ]
            mask_bands = sorted(set(int(item) for item in v4.get("mask_bands", default_bands)))
            full_mask = derive_mask(mask_dataset, bands=mask_bands, chunk_rows=int(config.get("chunk_rows", 64)))
            mask_dataset.close()
            mask_dataset = None
            sample_plan = build_stratified_sample_plan(
                full_mask,
                desired_blocks=int(sampling.get("blocks", 12)),
                block_rows=int(sampling.get("block_rows", 64)),
                min_valid_pixels=int(sampling.get("minimum_valid_pixels_per_block", 128)),
                forced_lines=tuple(int(item) for item in sampling.get("forced_lines", ())),
            )
            sample_cube, sample_mask, _ = read_sample_blocks(
                image,
                full_mask,
                sample_plan,
                maximum_rows=int(sampling.get("maximum_sample_rows", 256)),
            )
            processed_sample = _processed(sample_cube, input_is_smoothed, sg_window, sg_order)
            card = build_sensor_capability_card(
                image,
                catalog,
                sample_cube=sample_cube,
                sample_mask=sample_mask,
                data_physics=v4.get("data_physics"),
                preprocessing_state=preprocessing_state,
            )
            column_bias = (
                robust_column_bias(
                    processed_sample,
                    sample_mask,
                    radius=int(correction_config.get("radius", 2)),
                )
                if correction_enabled
                else np.zeros((image.info.samples, image.info.bands), dtype=np.float64)
            )
            if correction_strength > 0.0:
                processed_sample = processed_sample - correction_strength * column_bias[None, :, :]
        else:
            full_mask = np.asarray(prepared.full_mask, dtype=bool)
            if full_mask.shape != (image.info.lines, image.info.samples):
                raise ValueError("Prepared material mask and analysis image dimensions do not match")
            sample_plan = prepared.sample_plan
            sample_cube = np.asarray(prepared.sample_cube)
            sample_mask = np.asarray(prepared.sample_mask, dtype=bool)
            processed_sample = np.asarray(prepared.processed_sample)
            card = prepared.sensor_card
            column_bias = np.asarray(prepared.column_bias, dtype=np.float64)
            if column_bias.shape != (image.info.samples, image.info.bands):
                raise ValueError("Prepared detector-column correction does not match the analysis image")
        if np.count_nonzero(full_mask) < int(v4.get("minimum_mask_pixels", 100)):
            raise ValueError("Core mask is empty or contains too few valid pixels")
        token.raise_if_cancelled()

        provisional = resolve_mineral_support(card, catalog, wavelengths_nm=wavelengths)
        requested = tuple(str(item).casefold() for item in v4.get("requested_minerals", ()))
        if not requested and prepared is not None:
            requested = prepared.requested_minerals
        if not requested:
            requested = tuple(
                mineral.mineral_id
                for mineral in catalog.minerals.values()
                if mineral.support_level == "validated_swir" and provisional[mineral.mineral_id].level != SupportLevel.UNSUPPORTED
            )
        for mineral_id in requested:
            catalog.mineral(mineral_id)
            if provisional[mineral_id].level == SupportLevel.UNSUPPORTED:
                raise ValueError(f"{mineral_id} is unsupported: {'; '.join(provisional[mineral_id].reasons)}")
        if prepared is not None and requested == prepared.requested_minerals:
            internal = prepared.internal_minerals
        else:
            internal = _expand_internal_minerals(catalog, requested)
            internal = tuple(mineral for mineral in internal if provisional[mineral].level != SupportLevel.UNSUPPORTED)
        _emit(progress, started, "library", 0.08, "Building Catalog-driven automatic reference ensemble")
        if prepared is None:
            fwhm, _ = extract_fwhm_nm(image)
            library_config = v4.get("library_ensemble", {})
            ensemble = build_v4_library_ensemble(
                v4["spectral_library_root"],
                wavelengths,
                catalog,
                internal,
                target_fwhm_nm=fwhm,
                scene_cube=processed_sample,
                scene_mask=sample_mask,
                maximum_representatives=int(library_config.get("maximum_representatives_per_mineral", 6)),
                dedup_angle_rad=float(library_config.get("dedup_angle_rad", 0.02)),
            )
        else:
            ensemble = prepared.ensemble
            if ensemble.wavelengths_nm.shape != wavelengths.shape or not np.allclose(
                ensemble.wavelengths_nm,
                wavelengths,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("Prepared reference ensemble wavelength axis does not match the analysis image")
        ensemble_paths = write_v4_library_artifacts(ensemble, output / "library_ensemble")
        support = resolve_mineral_support(card, catalog, wavelengths_nm=wavelengths, reference_counts=ensemble.reference_counts())
        for mineral_id in requested:
            if support[mineral_id].level == SupportLevel.UNSUPPORTED:
                raise ValueError(f"{mineral_id} failed final capability gating: {'; '.join(support[mineral_id].reasons)}")
        _write_capability(output, card, support, sample_plan)
        write_envi(
            column_bias.astype(np.float32),
            output / "capability" / "column_spectral_bias.dat",
            description="Robust full-depth fixed-column spectral bias estimate; detector columns by image bands",
        )
        _save_json(catalog.to_summary(), output / "capability" / "catalog_summary.json")
        token.raise_if_cancelled()

        continuum_method = str(v4.get("preprocessing", {}).get("continuum_method", "upper_hull"))
        groups = _prepare_groups(
            catalog,
            ensemble,
            wavelengths,
            requested,
            internal,
            card.bad_band_indices,
            continuum_method,
        )
        detection_families = _prepare_detection_families(
            groups,
            wavelengths,
            card.bad_band_indices,
            v4.get("shared_detection_families"),
        )
        chunk_rows = int(config.get("chunk_rows", 64))
        group_scores = {
            group.definition.group_id: np.full((image.info.lines, image.info.samples), np.nan, dtype=np.float32)
            for group in groups
        }
        detection_mineral_scores = {
            group.definition.group_id: np.full(
                (len(group.mineral_order), image.info.lines, image.info.samples), np.nan, dtype=np.float32
            )
            for group in groups
        }
        sample_selection = sample_plan_mask(sample_plan, full_mask)
        _emit(progress, started, "group_sam", 0.16, "Computing full-depth group SAM evidence")
        for row_start, row_stop, cube in image.iter_rows(chunk_rows=chunk_rows):
            token.raise_if_cancelled()
            values = _processed(cube, input_is_smoothed, sg_window, sg_order, column_bias, correction_strength)
            mask_chunk = full_mask[row_start:row_stop]
            for group in groups:
                group_score, mineral_scores = _group_sam(values, mask_chunk, group)
                group_scores[group.definition.group_id][row_start:row_stop] = group_score
                detection_mineral_scores[group.definition.group_id][:, row_start:row_stop] = mineral_scores
            for family in detection_families:
                family_score = _shared_detection_sam(values, mask_chunk, family)
                for group_id in family.group_ids:
                    group_scores[group_id][row_start:row_stop] = family_score
            fraction = row_stop / image.info.lines
            _emit(progress, started, "group_sam", 0.16 + 0.20 * fraction, f"Processed SAM through line {row_stop}", stage_fraction=fraction, block=(row_start, row_stop))

        raw_group_scores = {key: value.copy() for key, value in group_scores.items()}
        shared_detection_group_ids = {
            group_id for family in detection_families for group_id in family.group_ids
        }
        detection_calibration: dict[str, dict[str, Any]] = {}
        for group in groups:
            group_id = group.definition.group_id
            mineral_scores = detection_mineral_scores[group_id]
            medians = np.asarray([
                np.nanmedian(mineral_scores[index][sample_selection])
                for index in range(mineral_scores.shape[0])
            ], dtype=np.float64)
            baseline = float(np.nanmin(medians))
            maximum_offset = min(
                0.03,
                0.25 * float(group.definition.policies["balanced"].get("absolute_sam_threshold_rad", 0.12)),
            )
            detection_strength = (
                0.0
                if bool(v4.get("disable_scene_domain_offsets", False))
                else min(float(group.definition.domain_calibration_strength), 0.25)
            )
            offsets = np.clip(
                detection_strength * (medians - baseline),
                0.0,
                maximum_offset,
            )
            calibrated_detection = mineral_scores - offsets[:, None, None]
            finite_detection = np.any(np.isfinite(calibrated_detection), axis=0)
            minimum_detection = np.min(np.where(np.isfinite(calibrated_detection), calibrated_detection, np.inf), axis=0)
            if group_id not in shared_detection_group_ids:
                group_scores[group_id] = np.where(finite_detection, minimum_detection, np.nan).astype(np.float32)
            detection_calibration[group_id] = {
                "sample_median_sam_rad": {
                    mineral: float(medians[index]) for index, mineral in enumerate(group.mineral_order)
                },
                "offsets_rad": {
                    mineral: float(offsets[index]) for index, mineral in enumerate(group.mineral_order)
                },
                "strength": detection_strength,
                "maximum_offset_rad": maximum_offset,
            }
        policies: dict[str, dict[str, dict[str, Any]]] = {}
        threshold_search: dict[str, dict[str, Any]] = {}
        group_noise: dict[str, float] = {}
        use_v5_threshold_search = bool(v4.get("adaptive_threshold_search", False))
        threshold_overrides = v4.get("threshold_overrides", {})
        if threshold_overrides is None:
            threshold_overrides = {}
        if not isinstance(threshold_overrides, Mapping):
            raise ValueError("v4.threshold_overrides must be a mapping")
        for group in groups:
            group_id = group.definition.group_id
            selected_sample_spectra = processed_sample[..., group.classification_indices][sample_mask]
            relative_noise = estimate_relative_spectral_noise(selected_sample_spectra)
            group_noise[group_id] = relative_noise
            overrides = v4.get("policy_overrides", {}).get(group_id, {})
            if use_v5_threshold_search:
                policies[group_id] = {}
                threshold_search[group_id] = {}
                for policy in POLICY_ORDER:
                    group_thresholds = threshold_overrides.get(group_id, {})
                    if not isinstance(group_thresholds, Mapping):
                        raise ValueError(f"Threshold overrides for {group_id} must be a mapping")
                    trial_override = group_thresholds.get(policy, {})
                    if trial_override is None:
                        trial_override = {}
                    if not isinstance(trial_override, Mapping):
                        raise ValueError(f"Threshold override for {group_id}/{policy} must be a mapping")
                    explicit_percentile = trial_override.get(
                        "column_percentile",
                        trial_override.get("resolved_percentile_fraction"),
                    )
                    explicit_absolute = trial_override.get(
                        "absolute_sam_threshold_rad",
                        trial_override.get("absolute_sam_rad", trial_override.get("resolved_absolute_threshold_rad")),
                    )
                    resolved = calibrate_group_thresholds(
                        group.definition,
                        policy,
                        group_scores[group_id],
                        full_mask,
                        sample_plan=sample_plan,
                        percentile_candidates=(
                            None if explicit_percentile is None else [float(explicit_percentile)]
                        ),
                        absolute_threshold_candidates_rad=(
                            None if explicit_absolute is None else [float(explicit_absolute)]
                        ),
                    ).require_resolved()
                    settings = dict(group.definition.policies[policy])
                    # Threshold overrides cannot escape the Catalog-safe V5
                    # search envelope.  Non-threshold expert gates remain
                    # user-configurable for documented advanced workflows.
                    policy_override = dict(overrides.get(policy, {}))
                    for protected in ("column_percentile", "absolute_sam_threshold_rad", "column_thresholds_rad"):
                        policy_override.pop(protected, None)
                    settings.update(policy_override)
                    catalog_depth = float(group.definition.policies[policy].get("minimum_absorption_depth", 0.0))
                    requested_depth = float(settings.get("minimum_absorption_depth", catalog_depth))
                    noise_multiplier = {"conservative": 3.0, "balanced": 2.2, "sensitive": 1.5}[policy]
                    noise_depth = noise_multiplier * relative_noise
                    settings.update({
                        "column_percentile": float(resolved.resolved_percentile_fraction),
                        "absolute_sam_threshold_rad": float(resolved.resolved_absolute_threshold_rad),
                        "scene_sam_threshold_rad": float(resolved.resolved_absolute_threshold_rad),
                        "scene_percentile": float(resolved.resolved_percentile_fraction),
                        "column_thresholds_rad": np.asarray(
                            resolved.resolved_column_thresholds_rad,
                            dtype=np.float64,
                        ),
                        "threshold_source": (
                            "v5_3_reviewed_threshold_trial"
                            if trial_override
                            else "v5_3_catalog_bounded_spatial_search"
                        ),
                        # Adaptive SAM search must not bypass the independent
                        # absorption-depth evidence gate.  User overrides may
                        # tighten this floor, but cannot lower the Catalog or
                        # scene-noise requirement.
                        "minimum_absorption_depth": max(catalog_depth, requested_depth, noise_depth),
                        "catalog_minimum_absorption_depth": catalog_depth,
                        "noise_minimum_absorption_depth": noise_depth,
                        "estimated_snr": card.estimated_snr,
                        "sources": {
                            "hard_constraints": "V5 Mineral Evidence Catalog",
                            "threshold_search": (
                                "reviewed V5.3 threshold-trial value re-evaluated on the full scene"
                                if trial_override
                                else "catalog-bounded stratified spatial proxy objective"
                            ),
                            "minimum_absorption_depth": "max(catalog floor, spectral noise multiplier)",
                            "user_overrides": sorted(policy_override),
                        },
                    })
                    policies[group_id][policy] = settings
                    threshold_search[group_id][policy] = resolved.to_dict()
            else:
                policies[group_id] = resolve_group_policies(
                    group.definition,
                    card,
                    group_scores[group_id],
                    sample_selection,
                    relative_noise=relative_noise,
                    overrides=overrides,
                    minimum_column_samples=int(sampling.get("minimum_column_samples", 20)),
                )

        project_calibration_records: dict[str, dict[str, Any]] = {}
        project_weak_labels = prepared.project_weak_labels if prepared is not None else None
        if project_weak_labels is not None:
            for group in groups:
                group_id = group.definition.group_id
                weak_group = project_weak_labels.labels.get(group_id)
                if not weak_group:
                    continue
                group_record: dict[str, Any] = {"detection": {}}
                for policy in POLICY_ORDER:
                    resolved_settings, record = calibrate_detection_settings(
                        group_scores[group_id],
                        full_mask,
                        weak_group,
                        project_weak_labels.training_mask,
                        project_weak_labels.holdout_mask,
                        policy,
                        project_weak_labels.settings,
                        policies[group_id][policy],
                    )
                    policies[group_id][policy] = resolved_settings
                    group_record["detection"][policy] = record
                project_calibration_records[group_id] = group_record

        _emit(progress, started, "calibration", 0.38, "Calibrating subclass domains from stratified blocks")
        calibration: dict[str, dict[str, Any]] = {}
        for group_index, group in enumerate(groups):
            group_id = group.definition.group_id
            sensitive = policies[group_id]["sensitive"]
            sensitive_full = policy_candidate(group_scores[group_id], full_mask, sensitive)
            score_samples: list[np.ndarray] = []
            depth_samples: list[np.ndarray] = []
            margin_samples: list[np.ndarray] = []
            for block in sample_plan.blocks:
                token.raise_if_cancelled()
                values = _processed(
                    image.read_rows(block.start_line, block.stop_line),
                    input_is_smoothed,
                    sg_window,
                    sg_order,
                )
                candidate = sensitive_full[block.start_line:block.stop_line]
                _, scores, _, depth, _, _ = _subclass_evidence(values, candidate, group, wavelengths)
                usable = np.all(np.isfinite(scores), axis=1) & np.isfinite(depth)
                if np.any(usable):
                    current = scores[usable]
                    score_samples.append(current)
                    depth_samples.append(depth[usable])
                    if current.shape[1] > 1:
                        ordered = np.sort(current, axis=1)
                        margin_samples.append(ordered[:, 1] - ordered[:, 0])
            if score_samples:
                stacked = np.vstack(score_samples)
                centers = np.median(stacked, axis=0)
                best_raw = np.min(stacked, axis=1)
                raw_scale = np.percentile(best_raw, [10, 90])
                margins = np.concatenate(margin_samples) if margin_samples else np.zeros(stacked.shape[0])
            else:
                centers = np.zeros(len(group.mineral_order), dtype=np.float64)
                raw_scale = np.array([0.0, 1.0])
                margins = np.zeros(1)
            calibration[group_id] = {
                "centers": centers,
                "raw_score_p10": float(raw_scale[0]),
                "raw_score_p90": float(max(raw_scale[1], raw_scale[0] + 1e-6)),
                "margin_p50": float(np.percentile(margins, 50)),
                "margin_p75": float(np.percentile(margins, 75)),
                "sample_pixels": int(sum(item.shape[0] for item in score_samples)),
            }
            fraction = (group_index + 1) / max(len(groups), 1)
            _emit(progress, started, "calibration", 0.38 + 0.08 * fraction, f"Calibrated {group_id}", stage_fraction=fraction)

        height = stop - start
        output_mask = full_mask[start:stop]
        spatial_metadata = subset_spatial_metadata(image.info, start_line=start)
        spatial_metadata.update({
            "corespec source data path": str(image.info.data_path.resolve()),
            "corespec source header path": str(image.info.header_path.resolve()),
            "corespec source total lines": int(image.info.lines),
            "corespec source total samples": int(image.info.samples),
            "corespec output stop line exclusive": int(stop),
        })

        def write_spatial(data: np.ndarray, path: str | Path, **kwargs: Any) -> tuple[Path, Path]:
            return write_envi(data, path, metadata=spatial_metadata, **kwargs)

        edge_risk = edge_risk_score(output_mask, width=int(v4.get("artifact_control", {}).get("edge_width", 2)))
        brightness_band = int(np.argmin(np.abs(wavelengths - 1600.0)))
        sample_brightness = processed_sample[..., brightness_band][sample_mask]
        finite_brightness = sample_brightness[np.isfinite(sample_brightness)]
        saturation_threshold = float(np.percentile(finite_brightness, 99.9)) if finite_brightness.size else np.inf
        global_confidence = np.zeros((height, image.info.samples), dtype=np.float32)
        global_stability = np.zeros_like(global_confidence)
        global_rejection = np.zeros((height, image.info.samples), dtype=np.uint8)
        run_groups: dict[str, Any] = {}
        count_rows: list[list[Any]] = []
        cross_group_profiles: dict[str, dict[str, np.ndarray]] = {}
        cross_group_confidence: dict[str, np.ndarray] = {}
        cross_group_class_names: dict[str, list[str]] = {}
        cross_group_class_lookup: dict[str, dict[str, int]] = {}
        cross_group_stripe_unions: dict[str, np.ndarray] = {}
        _emit(progress, started, "classification", 0.47, "Computing shared subclass evidence for all three policies")

        for group_index, group in enumerate(groups):
            token.raise_if_cancelled()
            group_id = group.definition.group_id
            score_full = group_scores[group_id]
            profile_candidates = {
                policy: policy_candidate(score_full[start:stop], output_mask, policies[group_id][policy])
                for policy in POLICY_ORDER
            }
            sensitive_candidate = profile_candidates["sensitive"]
            column_risk = fixed_column_risk_score(
                policy_candidate(score_full, full_mask, policies[group_id]["sensitive"]), full_mask, sample_plan
            )
            mineral_count = len(group.mineral_order)
            raw_scores = np.full((mineral_count, height, image.info.samples), np.nan, dtype=np.float32)
            calibrated_scores = np.full_like(raw_scores, np.nan)
            consensus_cube = np.full_like(raw_scores, np.nan)
            depth_map = np.full((height, image.info.samples), np.nan, dtype=np.float32)
            feature_maps = {
                feature.feature_id: np.full((height, image.info.samples), np.nan, dtype=np.float32)
                for feature in group.definition.features
            }
            best_labels = np.full((height, image.info.samples), -1, dtype=np.int16)
            raw_labels = np.full_like(best_labels, -1)
            margin = np.full((height, image.info.samples), np.nan, dtype=np.float32)
            winning_consensus = np.zeros((height, image.info.samples), dtype=np.float32)
            fit_quality = np.full((height, image.info.samples), np.nan, dtype=np.float32)
            saturation = np.zeros((height, image.info.samples), dtype=bool)
            calibration_record = calibration[group_id]
            centers = np.asarray(calibration_record["centers"], dtype=np.float64)
            strength = (
                0.0
                if bool(v4.get("disable_scene_domain_offsets", False))
                else float(group.definition.domain_calibration_strength)
            )
            for row_start, row_stop, cube in image.iter_rows(chunk_rows=chunk_rows, start=start, stop=stop):
                token.raise_if_cancelled()
                local = slice(row_start - start, row_stop - start)
                # Column correction is a high-recall detector aid. Subclass shape,
                # continuum, and SFF evidence stay on the original spectral domain.
                values = _processed(cube, input_is_smoothed, sg_window, sg_order)
                saturation[local] = np.isfinite(values[..., brightness_band]) & (values[..., brightness_band] >= saturation_threshold)
                valid, scores, consensus, depth, features, evidence = _subclass_evidence(
                    values, sensitive_candidate[local], group, wavelengths
                )
                if np.any(valid):
                    calibrated = scores - strength * centers[None, :]
                    safe_calibrated = np.where(np.isfinite(calibrated), calibrated, np.inf)
                    safe_raw = np.where(np.isfinite(scores), scores, np.inf)
                    score_valid = np.any(np.isfinite(calibrated), axis=1)
                    current_labels = np.argmin(safe_calibrated, axis=1)
                    current_raw_labels = np.argmin(safe_raw, axis=1)
                    ordered = np.sort(safe_calibrated, axis=1)
                    if mineral_count > 1:
                        with np.errstate(invalid="ignore"):
                            current_margin = ordered[:, 1] - ordered[:, 0]
                        current_margin = np.where(
                            np.isfinite(ordered[:, 0]) & np.isfinite(ordered[:, 1]),
                            current_margin,
                            np.where(np.isfinite(ordered[:, 0]), np.inf, np.nan),
                        )
                    else:
                        # No competitor means the margin is not applicable.  It
                        # remains NaN in diagnostics; the explicit margin gate
                        # below bypasses it for a valid one-mineral label.
                        current_margin = np.full(scores.shape[0], np.nan)
                    current_margin[~score_valid] = np.nan
                    for mineral_index in range(mineral_count):
                        raw_scores[mineral_index, local][valid] = scores[:, mineral_index].astype(np.float32)
                        calibrated_scores[mineral_index, local][valid] = calibrated[:, mineral_index].astype(np.float32)
                        consensus_cube[mineral_index, local][valid] = consensus[:, mineral_index].astype(np.float32)
                    best_labels[local][valid] = np.where(score_valid, current_labels, -1).astype(np.int16)
                    raw_labels[local][valid] = np.where(score_valid, current_raw_labels, -1).astype(np.int16)
                    depth_map[local][valid] = depth.astype(np.float32)
                    margin[local][valid] = current_margin.astype(np.float32)
                    winning_consensus[local][valid] = consensus[np.arange(scores.shape[0]), current_labels].astype(np.float32)
                    if evidence is not None:
                        fit_quality[local][valid] = _winning_mineral_sff_quality(
                            evidence.fit_quality,
                            group.reference_minerals,
                            group.mineral_order,
                            current_labels,
                        ).astype(np.float32)
                    for feature_id, feature in features.items():
                        feature_maps[feature_id][local][valid] = feature.astype(np.float32)
                group_fraction = (row_stop - start) / height
                group_span = 0.41 / max(len(groups), 1)
                overall = 0.47 + group_span * (group_index + 0.80 * group_fraction)
                _emit(progress, started, "classification", overall, f"{group_id}: processed lines {row_start}:{row_stop}", stage_fraction=group_fraction, block=(row_start, row_stop))

            weak_group = (
                project_weak_labels.labels.get(group_id)
                if project_weak_labels is not None
                else None
            )
            if weak_group:
                weak_group_local = {
                    mineral_id: np.asarray(positive, dtype=bool)[start:stop]
                    for mineral_id, positive in weak_group.items()
                }
                score_offsets, score_record = learn_score_offsets(
                    raw_scores,
                    group.mineral_order,
                    weak_group_local,
                    project_weak_labels.training_mask[start:stop],
                    project_weak_labels.holdout_mask[start:stop],
                    minimum_pixels=int(project_weak_labels.settings["minimum_pixels_per_mineral"]),
                )
                calibrated_scores = raw_scores - score_offsets[:, None, None].astype(np.float32)
                safe_calibrated = np.where(np.isfinite(calibrated_scores), calibrated_scores, np.inf)
                safe_raw = np.where(np.isfinite(raw_scores), raw_scores, np.inf)
                score_valid = np.any(np.isfinite(calibrated_scores), axis=0)
                best_labels = np.where(score_valid, np.argmin(safe_calibrated, axis=0), -1).astype(np.int16)
                raw_labels = np.where(score_valid, np.argmin(safe_raw, axis=0), -1).astype(np.int16)
                if mineral_count > 1:
                    ordered = np.sort(safe_calibrated, axis=0)
                    with np.errstate(invalid="ignore"):
                        margin = (ordered[1] - ordered[0]).astype(np.float32)
                    margin[~score_valid] = np.nan
                else:
                    margin = np.full(best_labels.shape, np.nan, dtype=np.float32)
                rows_for_consensus, columns_for_consensus = np.indices(best_labels.shape)
                safe_best_labels = np.maximum(best_labels, 0)
                winning_consensus = consensus_cube[
                    safe_best_labels,
                    rows_for_consensus,
                    columns_for_consensus,
                ]
                winning_consensus = np.where(score_valid, winning_consensus, 0.0).astype(np.float32)
                project_calibration_records.setdefault(group_id, {})["classification"] = score_record

            class_lookup = {mineral: index + 1 for index, mineral in enumerate(group.output_minerals)}
            class_names = ["Unclassified", *[catalog.mineral(item).display_name_en for item in group.output_minerals], "Masked Pixels"]
            project_evidence_gates: dict[str, dict[str, np.ndarray]] = {
                policy: {
                    mineral_id: np.ones(best_labels.shape, dtype=bool)
                    for mineral_id in group.mineral_order
                }
                for policy in POLICY_ORDER
            }
            if weak_group:
                evidence_planes = [
                    score_full[start:stop],
                    depth_map,
                    fit_quality,
                    *[raw_scores[index] for index in range(mineral_count)],
                    *[consensus_cube[index] for index in range(mineral_count)],
                    *feature_maps.values(),
                ]
                evidence_records: dict[str, Any] = {}
                evidence_profile_settings = dict(project_weak_labels.settings)
                if len(group.output_minerals) > 1:
                    # In true multi-mineral competitions each mineral already
                    # has a reviewed project-area prior.  A group-domain
                    # fraction can dwarf a rare class (observed as a spurious
                    # 24k-pixel dolomite map), so use only the per-mineral area
                    # multipliers.  Single-output clay experts may retain the
                    # wider discovery domain because SFF/features still gate
                    # the final label.
                    evidence_profile_settings["evidence_maximum_domain_fraction"] = {
                        "conservative": 0.0,
                        "balanced": 0.0,
                        "sensitive": 0.0,
                    }
                else:
                    evidence_profile_settings["evidence_maximum_domain_fraction"] = {
                        "conservative": 0.04,
                        "balanced": 0.25,
                        "sensitive": 0.25,
                    }
                winner_recall = score_record.get("training_winner_recall", {})
                needs_discriminant_assignment = any(
                    value is not None and float(value) < 0.75
                    for value in winner_recall.values()
                )
                for mineral_id, positive in weak_group_local.items():
                    if mineral_id not in group.mineral_order:
                        continue
                    within_group_negative = np.logical_or.reduce(
                        tuple(
                            np.asarray(other_positive, dtype=bool)
                            for other_mineral, other_positive in weak_group_local.items()
                            if other_mineral != mineral_id
                        ),
                    ) if needs_discriminant_assignment and len(weak_group_local) > 1 else None
                    profiled_gates, record = learn_evidence_gate(
                        evidence_planes,
                        positive,
                        sensitive_candidate & (best_labels >= 0),
                        project_weak_labels.training_mask[start:stop],
                        project_weak_labels.holdout_mask[start:stop],
                        minimum_pixels=int(project_weak_labels.settings["minimum_pixels_per_mineral"]),
                        weak_negative=within_group_negative,
                        # All profiles rank the same broad SAM discovery
                        # domain.  Restricting this fit to balanced winners made
                        # the sensitive profile mathematically unable to grow.
                        selection_mask=None,
                        return_profiles=True,
                        profile_settings=evidence_profile_settings,
                    )
                    for policy in POLICY_ORDER:
                        project_evidence_gates[policy][mineral_id] = profiled_gates[policy]
                    evidence_records[mineral_id] = record
                project_calibration_records.setdefault(group_id, {})["evidence_gates"] = evidence_records
                if needs_discriminant_assignment and len(weak_group_local) > 1:
                    gated_scores = np.full_like(calibrated_scores, np.inf)
                    for mineral_id, gate in project_evidence_gates["balanced"].items():
                        if mineral_id not in group.mineral_order:
                            continue
                        mineral_index = group.mineral_order.index(mineral_id)
                        gated_scores[mineral_index] = np.where(
                            gate,
                            calibrated_scores[mineral_index],
                            np.inf,
                        )
                    discriminant_available = np.any(np.isfinite(gated_scores), axis=0)
                    discriminant_labels = np.argmin(gated_scores, axis=0).astype(np.int16)
                    reassigned = discriminant_available & (best_labels != discriminant_labels)
                    best_labels[discriminant_available] = discriminant_labels[discriminant_available]
                    evidence_records["assignment"] = {
                        "status": "resolved",
                        "method": "within_group_evidence_gate_assignment",
                        "reassigned_pixels": int(np.count_nonzero(reassigned)),
                    }

            rows, columns = np.indices(best_labels.shape)
            safe_labels = np.maximum(best_labels, 0)
            winning_consensus = consensus_cube[safe_labels, rows, columns]
            winning_consensus = np.where(best_labels >= 0, winning_consensus, 0.0).astype(np.float32)
            best_raw = raw_scores[safe_labels, rows, columns]
            p10 = float(calibration_record["raw_score_p10"])
            p90 = float(calibration_record["raw_score_p90"])
            score_confidence = np.clip((p90 - best_raw) / max(p90 - p10, 1e-6), 0.0, 1.0)
            margin_scale = max(float(calibration_record["margin_p75"]), 0.015)
            margin_confidence = (
                np.clip(margin / margin_scale, 0.0, 1.0)
                if mineral_count > 1
                else (best_labels >= 0).astype(np.float32)
            )
            sensitive_settings = policies[group_id]["sensitive"]
            sam_confidence = np.clip(1.0 - score_full[start:stop] / max(float(sensitive_settings["absolute_sam_threshold_rad"]), 1e-6), 0.0, 1.0)
            depth_reference = max(2.0 * float(sensitive_settings["minimum_absorption_depth"]), 0.02)
            depth_confidence = np.clip(depth_map / depth_reference, 0.0, 1.0)
            label_agreement = (best_labels == raw_labels).astype(np.float32)
            stability = np.clip(0.40 * label_agreement + 0.35 * margin_confidence + 0.25 * winning_consensus, 0.0, 1.0)
            confidence = (
                0.20 * sam_confidence
                + 0.20 * score_confidence
                + 0.20 * margin_confidence
                + 0.15 * winning_consensus
                + 0.15 * depth_confidence
                + 0.10 * stability
            )
            confidence *= 1.0 - 0.35 * column_risk[None, :]
            confidence *= 1.0 - 0.45 * edge_risk
            confidence *= np.where(saturation, 0.45, 1.0)
            confidence = np.where(
                sensitive_candidate & (best_labels >= 0),
                np.clip(confidence, 0.0, 1.0),
                0.0,
            )
            confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            project_feature_overrides: dict[str, dict[str, Mapping[str, Any]]] = {
                policy: {} for policy in POLICY_ORDER
            }
            if weak_group:
                feature_windows = {
                    feature.feature_id: feature.window_nm for feature in group.definition.features
                }
                feature_records: dict[str, Any] = {}
                for policy in POLICY_ORDER:
                    feature_records[policy] = {}
                    for mineral_id, positive in weak_group_local.items():
                        if mineral_id not in group.mineral_order:
                            continue
                        catalog_gate = catalog.mineral(mineral_id).experts.get(
                            group.definition.expert_id,
                            {},
                        ).get("feature_gate", {})
                        override, record = learn_feature_gate_override(
                            mineral_id,
                            positive,
                            project_weak_labels.training_mask[start:stop],
                            feature_maps,
                            catalog_gate,
                            policy,
                            project_weak_labels.settings,
                            feature_windows,
                        )
                        project_feature_overrides[policy][mineral_id] = override
                        feature_records[policy][mineral_id] = record
                project_calibration_records.setdefault(group_id, {})["feature_gates"] = feature_records
            before_profiles: dict[str, np.ndarray] = {}
            gate_profiles: dict[str, np.ndarray] = {}
            confidence_threshold_maps: dict[str, np.ndarray] = {}
            consensus_threshold_maps: dict[str, np.ndarray] = {}
            stability_threshold_maps: dict[str, np.ndarray] = {}
            sff_threshold_maps: dict[str, np.ndarray] = {}
            sff_enabled = float(group.definition.score_weights.get("fit", 0.0)) > 0.0
            reference_counts_by_mineral = {
                mineral_id: int(sum(item == mineral_id for item in group.reference_minerals))
                for mineral_id in group.mineral_order
            }
            for policy in POLICY_ORDER:
                settings = policies[group_id][policy]
                gate_pass = np.zeros(best_labels.shape, dtype=bool)
                for mineral_index, mineral_id in enumerate(group.mineral_order):
                    selected = best_labels == mineral_index
                    if not np.any(selected):
                        continue
                    mineral_gate = mineral_feature_gate(
                        catalog.mineral(mineral_id),
                        group.definition.expert_id,
                        policy,
                        depth_map,
                        feature_maps,
                        gate_override=project_feature_overrides[policy].get(mineral_id),
                    )
                    gate_pass[selected] = (
                        mineral_gate[selected]
                        & project_evidence_gates[policy].get(
                            mineral_id,
                            np.ones_like(selected),
                        )[selected]
                    )
                gate_profiles[policy] = gate_pass
                adaptive_margin = {
                    "conservative": 0.10,
                    "balanced": 0.0,
                    "sensitive": 0.0,
                }[policy] * float(calibration_record["margin_p50"])
                margin_threshold = max(float(settings.get("minimum_margin", 0.0)), adaptive_margin)
                strict_gates = bool(v4.get("strict_evidence_gates", False))
                consensus_defaults = (
                    {"conservative": 0.50, "balanced": 0.34, "sensitive": 0.20}
                    if strict_gates
                    else {"conservative": 0.20, "balanced": 0.10, "sensitive": 0.0}
                )
                stability_defaults = (
                    {"conservative": 0.60, "balanced": 0.45, "sensitive": 0.20}
                    if strict_gates
                    else {"conservative": 0.45, "balanced": 0.25, "sensitive": 0.0}
                )
                consensus_threshold = float(settings.get(
                    "minimum_reference_consensus", consensus_defaults[policy]
                ))
                # Consensus is a discrete fraction whose step size depends on
                # the selected reference count.  A fixed 0.50 gate meant that
                # one strong representative out of three (1/3) could never be
                # accepted, even though the library intentionally preserves
                # cross-source spectral diversity.  Balanced/sensitive mapping
                # requires one supporting representative; conservative mapping
                # requires two when available.  The Catalog value remains an
                # upper strictness bound, while confidence retains the full
                # continuous consensus contribution.
                consensus_threshold_map = np.ones(best_labels.shape, dtype=np.float32)
                consensus_thresholds_by_mineral: dict[str, float] = {}
                for mineral_index, mineral_id in enumerate(group.mineral_order):
                    reference_count = max(reference_counts_by_mineral.get(mineral_id, 0), 1)
                    resolved_consensus = _reference_consensus_threshold(
                        consensus_threshold,
                        policy,
                        reference_count,
                    )
                    consensus_threshold_map[best_labels == mineral_index] = resolved_consensus
                    consensus_thresholds_by_mineral[mineral_id] = float(resolved_consensus)
                if weak_group:
                    project_minimum = max(
                        20,
                        int(project_weak_labels.settings["minimum_pixels_per_mineral"]) // 4,
                    )
                    consensus_quantile = {
                        "conservative": 35.0,
                        "balanced": 15.0,
                        "sensitive": 5.0,
                    }[policy]
                    for mineral_index, mineral_id in enumerate(group.mineral_order):
                        positive = weak_group_local.get(mineral_id)
                        if positive is None:
                            continue
                        consensus_sample_mask = (
                            profile_candidates[policy]
                            & (best_labels == mineral_index)
                            & positive
                            & project_weak_labels.training_mask[start:stop]
                            & np.isfinite(depth_map)
                            & (depth_map >= float(settings["minimum_absorption_depth"]))
                        )
                        consensus_sample = winning_consensus[consensus_sample_mask]
                        consensus_sample = consensus_sample[np.isfinite(consensus_sample)]
                        if consensus_sample.size >= project_minimum:
                            resolved_consensus = min(
                                consensus_thresholds_by_mineral[mineral_id],
                                float(np.percentile(consensus_sample, consensus_quantile)),
                            )
                            consensus_threshold_map[best_labels == mineral_index] = resolved_consensus
                            consensus_thresholds_by_mineral[mineral_id] = float(resolved_consensus)
                consensus_threshold_maps[policy] = consensus_threshold_map
                stability_threshold = float(settings.get(
                    "minimum_evidence_stability", stability_defaults[policy]
                ))
                requested_winner = np.zeros(best_labels.shape, dtype=bool)
                labels = np.zeros(best_labels.shape, dtype=np.uint8)
                for mineral_index, mineral_id in enumerate(group.mineral_order):
                    if mineral_id not in class_lookup:
                        continue
                    selected = best_labels == mineral_index
                    requested_winner |= selected
                    labels[selected] = class_lookup[mineral_id]
                sff_threshold_map = np.zeros(best_labels.shape, dtype=np.float32)
                sff_thresholds_by_mineral: dict[str, float] = {}
                if sff_enabled:
                    sff_quantile = {
                        "conservative": 50.0,
                        "balanced": 25.0,
                        "sensitive": 10.0,
                    }[policy]
                    for mineral_index, mineral_id in enumerate(group.mineral_order):
                        selected = (
                            profile_candidates[policy]
                            & (best_labels == mineral_index)
                            & np.isfinite(fit_quality)
                            & (fit_quality > 0.0)
                        )
                        positive = weak_group_local.get(mineral_id) if weak_group else None
                        project_selected = (
                            selected
                            & positive
                            & project_weak_labels.training_mask[start:stop]
                            if positive is not None
                            else selected
                        )
                        sff_sample = fit_quality[project_selected]
                        sff_sample = sff_sample[np.isfinite(sff_sample) & (sff_sample > 0.0)]
                        minimum_project = (
                            max(
                                20,
                                int(project_weak_labels.settings["minimum_pixels_per_mineral"]) // 4,
                            )
                            if weak_group
                            else 20
                        )
                        if sff_sample.size < minimum_project:
                            sff_sample = fit_quality[selected]
                            sff_sample = sff_sample[
                                np.isfinite(sff_sample) & (sff_sample > 0.0)
                            ]
                        threshold = (
                            float(np.percentile(sff_sample, sff_quantile))
                            if sff_sample.size
                            else np.inf
                        )
                        threshold = max(
                            float(settings.get("minimum_sff_quality", 0.0)),
                            threshold,
                        )
                        sff_threshold_map[best_labels == mineral_index] = threshold
                        sff_thresholds_by_mineral[mineral_id] = threshold
                else:
                    for mineral_id in group.mineral_order:
                        sff_thresholds_by_mineral[mineral_id] = 0.0
                sff_threshold_maps[policy] = sff_threshold_map
                confidence_quantile = {"conservative": 60.0, "balanced": 25.0, "sensitive": 5.0}[policy]
                confidence_threshold_map = np.ones(best_labels.shape, dtype=np.float32)
                confidence_thresholds_by_mineral: dict[str, float] = {}
                for mineral_index, mineral_id in enumerate(group.mineral_order):
                    selected = profile_candidates[policy] & (best_labels == mineral_index)
                    confidence_sample = confidence[selected]
                    confidence_sample = confidence_sample[np.isfinite(confidence_sample)]
                    catalog_threshold = float(settings.get("confidence_threshold", 0.0))
                    if confidence_sample.size:
                        scene_confidence_threshold = float(
                            np.percentile(confidence_sample, confidence_quantile)
                        )
                        threshold = (
                            max(catalog_threshold, scene_confidence_threshold)
                            if bool(v4.get("safe_confidence_floor", False))
                            else min(catalog_threshold, scene_confidence_threshold)
                        )
                    else:
                        # An empty mineral-specific sample is not evidence for a
                        # perfect-confidence cutoff.  V5.2 substituted 1.0 here,
                        # creating an impossible gate that could silently turn a
                        # selected mineral into a guaranteed zero map.  Retain the
                        # Catalog floor and let the explicit candidate/feature
                        # gates describe the lack of scene support.
                        threshold = catalog_threshold
                    if weak_group and mineral_id in weak_group_local:
                        weak_selected = (
                            selected
                            & weak_group_local[mineral_id]
                            & project_weak_labels.training_mask[start:stop]
                            & np.isfinite(depth_map)
                            & (depth_map >= float(settings["minimum_absorption_depth"]))
                        )
                        weak_confidence = confidence[weak_selected]
                        weak_confidence = weak_confidence[np.isfinite(weak_confidence)]
                        project_minimum = max(
                            20,
                            int(project_weak_labels.settings["minimum_pixels_per_mineral"]) // 4,
                        )
                        if weak_confidence.size >= project_minimum:
                            project_quantile = {
                                "conservative": 35.0,
                                "balanced": 15.0,
                                "sensitive": 5.0,
                            }[policy]
                            project_threshold = float(np.percentile(weak_confidence, project_quantile))
                            threshold = max(0.20, min(threshold, project_threshold))
                    confidence_threshold_map[best_labels == mineral_index] = threshold
                    confidence_thresholds_by_mineral[mineral_id] = threshold
                confidence_threshold_maps[policy] = confidence_threshold_map
                margin_pass = _competition_margin_pass(
                    margin,
                    margin_threshold,
                    mineral_count=mineral_count,
                    classified=best_labels >= 0,
                )
                if weak_group:
                    weak_union = np.logical_or.reduce(
                        tuple(np.asarray(value, dtype=bool) for value in weak_group_local.values())
                    )
                    stability_sample_mask = (
                        profile_candidates[policy]
                        & requested_winner
                        & weak_union
                        & project_weak_labels.training_mask[start:stop]
                        & np.isfinite(depth_map)
                        & (depth_map >= float(settings["minimum_absorption_depth"]))
                        & gate_pass
                        & margin_pass
                        & (winning_consensus >= consensus_threshold_map)
                    )
                    project_stability = stability[stability_sample_mask]
                    project_stability = project_stability[np.isfinite(project_stability)]
                    if project_stability.size >= int(
                        project_weak_labels.settings["minimum_pixels_per_mineral"]
                    ):
                        stability_quantile = {
                            "conservative": 35.0,
                            "balanced": 15.0,
                            "sensitive": 5.0,
                        }[policy]
                        stability_threshold = max(
                            0.20,
                            min(
                                stability_threshold,
                                float(np.percentile(project_stability, stability_quantile)),
                            ),
                        )
                stability_threshold_map = np.full(
                    best_labels.shape,
                    stability_threshold,
                    dtype=np.float32,
                )
                stability_thresholds_by_mineral: dict[str, float] = {}
                if weak_group:
                    project_minimum = max(
                        20,
                        int(project_weak_labels.settings["minimum_pixels_per_mineral"]) // 4,
                    )
                    stability_quantile = {
                        "conservative": 35.0,
                        "balanced": 15.0,
                        "sensitive": 5.0,
                    }[policy]
                    for mineral_index, mineral_id in enumerate(group.mineral_order):
                        resolved_stability = stability_threshold
                        positive = weak_group_local.get(mineral_id)
                        if positive is not None:
                            class_sample = (
                                profile_candidates[policy]
                                & (best_labels == mineral_index)
                                & positive
                                & project_weak_labels.training_mask[start:stop]
                                & np.isfinite(depth_map)
                                & (depth_map >= float(settings["minimum_absorption_depth"]))
                                & margin_pass
                                & (winning_consensus >= consensus_threshold_map)
                            )
                            class_stability = stability[class_sample]
                            class_stability = class_stability[np.isfinite(class_stability)]
                            if class_stability.size >= project_minimum:
                                resolved_stability = max(
                                    0.20,
                                    min(
                                        resolved_stability,
                                        float(np.percentile(class_stability, stability_quantile)),
                                    ),
                                )
                        stability_threshold_map[best_labels == mineral_index] = resolved_stability
                        stability_thresholds_by_mineral[mineral_id] = float(resolved_stability)
                else:
                    for mineral_index, mineral_id in enumerate(group.mineral_order):
                        stability_thresholds_by_mineral[mineral_id] = float(stability_threshold)
                stability_threshold_maps[policy] = stability_threshold_map
                accepted = (
                    profile_candidates[policy]
                    & requested_winner
                    & np.isfinite(depth_map)
                    & (depth_map >= float(settings["minimum_absorption_depth"]))
                    & gate_pass
                    & margin_pass
                    & (winning_consensus >= consensus_threshold_map)
                    & (stability >= stability_threshold_map)
                    & (confidence >= confidence_threshold_map)
                    & (
                        (fit_quality >= sff_threshold_map)
                        if sff_enabled
                        else np.ones_like(requested_winner)
                    )
                )
                before_profiles[policy] = np.where(accepted, labels, 0).astype(np.uint8)
                settings["resolved_minimum_margin"] = margin_threshold
                settings["resolved_consensus_threshold"] = consensus_threshold
                settings["resolved_consensus_threshold_by_mineral"] = consensus_thresholds_by_mineral
                settings["resolved_stability_threshold"] = stability_threshold
                settings["resolved_stability_threshold_by_mineral"] = stability_thresholds_by_mineral
                settings["resolved_confidence_threshold_by_mineral"] = confidence_thresholds_by_mineral
                settings["resolved_sff_threshold_by_mineral"] = sff_thresholds_by_mineral

            final_profiles: dict[str, np.ndarray] = {}
            stripe_profiles: dict[str, np.ndarray] = {}
            cleanup_records: dict[str, Any] = {}
            for policy in POLICY_ORDER:
                cleanup_settings = dict(group.definition.spatial_cleanup)
                # Column risk informs protection but must not erase otherwise
                # strong local/oblique geology before the residual repeated-
                # band test can inspect its lateral support.
                artifact_confidence = confidence * (1.0 - 0.35 * column_risk[None, :])
                low_column_risk = column_risk[None, :] < 0.50
                accepted_confidence = artifact_confidence[(before_profiles[policy] > 0) & low_column_risk]
                protection_quantile = {"conservative": 50.0, "balanced": 65.0, "sensitive": 80.0}[policy]
                if accepted_confidence.size:
                    protection_floor = {"conservative": 0.62, "balanced": 0.58, "sensitive": 0.62}[policy]
                    cleanup_settings["strong_evidence_protection"] = max(
                        protection_floor,
                        min(
                            float(cleanup_settings.get("strong_evidence_protection", 0.72)),
                            float(np.percentile(accepted_confidence, protection_quantile)),
                        ),
                    )
                protection_by_class: dict[int, float] = {}
                project_protection_by_class: dict[int, float] = {}
                protection_floor = {"conservative": 0.55, "balanced": 0.50, "sensitive": 0.55}[policy]
                for class_id in range(1, len(group.output_minerals) + 1):
                    class_values = artifact_confidence[
                        (before_profiles[policy] == class_id) & low_column_risk
                    ]
                    if class_values.size:
                        resolved_protection = max(
                            protection_floor,
                            min(
                                float(cleanup_settings.get("strong_evidence_protection", 0.72)),
                                float(np.percentile(class_values, protection_quantile)),
                            ),
                        )
                        mineral_id = group.output_minerals[class_id - 1]
                        if weak_group and mineral_id in weak_group_local:
                            weak_values = artifact_confidence[
                                (before_profiles[policy] == class_id)
                                & weak_group_local[mineral_id]
                                & project_weak_labels.training_mask[start:stop]
                            ]
                            weak_values = weak_values[np.isfinite(weak_values)]
                            if weak_values.size >= int(
                                project_weak_labels.settings["minimum_pixels_per_mineral"]
                            ):
                                project_quantile = {
                                    "conservative": 35.0,
                                    "balanced": 15.0,
                                    "sensitive": 5.0,
                                }[policy]
                                resolved_protection = max(
                                    0.25,
                                    min(
                                        resolved_protection,
                                        float(np.percentile(weak_values, project_quantile)),
                                    ),
                                )
                                project_protection_by_class[class_id] = resolved_protection
                        protection_by_class[class_id] = resolved_protection
                cleanup_settings["strong_evidence_protection_by_class"] = protection_by_class
                minimum_component_by_class: dict[int, int] = {}
                if weak_group:
                    for class_id, mineral_id in enumerate(group.output_minerals, start=1):
                        positive = weak_group_local.get(mineral_id)
                        if positive is None:
                            continue
                        reviewed_pixels = int(np.count_nonzero(positive))
                        candidate_pixels = int(
                            np.count_nonzero(before_profiles[policy] == class_id)
                        )
                        if 0 < candidate_pixels < max(100, int(round(0.25 * reviewed_pixels))):
                            minimum_component_by_class[class_id] = 1
                if minimum_component_by_class:
                    cleanup_settings["minimum_component_by_class"] = minimum_component_by_class
                final_profiles[policy], stripe_profiles[policy], cleanup_records[policy] = filter_artifacts(
                    before_profiles[policy],
                    artifact_confidence,
                    output_mask,
                    cleanup_settings,
                    policy=policy,
                    column_risk=column_risk,
                )
                cleanup_retention_by_class: dict[int, float] = {}
                retention_warning_classes: list[int] = []
                if weak_group:
                    minimum_training = int(
                        project_weak_labels.settings["minimum_pixels_per_mineral"]
                    )
                    minimum_retention = {
                        "conservative": 0.55,
                        "balanced": 0.60,
                        "sensitive": 0.70,
                    }[policy]
                    for class_id, mineral_id in enumerate(group.output_minerals, start=1):
                        if mineral_id not in weak_group_local:
                            continue
                        reviewed_before = (
                            (before_profiles[policy] == class_id)
                            & weak_group_local[mineral_id]
                            & project_weak_labels.training_mask[start:stop]
                        )
                        reviewed_count = int(np.count_nonzero(reviewed_before))
                        if reviewed_count < minimum_training:
                            continue
                        retention = float(
                            np.count_nonzero(
                                reviewed_before & (final_profiles[policy] == class_id)
                            )
                            / reviewed_count
                        )
                        cleanup_retention_by_class[class_id] = retention
                        if retention < minimum_retention:
                            retention_warning_classes.append(class_id)
                cleanup_records[policy]["resolved_strong_evidence_protection"] = float(
                    cleanup_settings.get("strong_evidence_protection", 0.72)
                )
                cleanup_records[policy]["resolved_strong_evidence_protection_by_class"] = protection_by_class
                cleanup_records[policy]["project_protection_by_class"] = project_protection_by_class
                cleanup_records[policy]["initial_project_retention_by_class"] = cleanup_retention_by_class
                # A low reviewed-pixel retention is an audit warning, never a
                # reason to switch off stripe filtering for an entire clay
                # class.  Upstream evidence breadth must solve low recall.
                cleanup_records[policy]["project_directional_retention_warnings"] = retention_warning_classes
                cleanup_records[policy]["project_relaxed_directional_classes"] = []
                cleanup_records[policy]["project_minimum_component_by_class"] = minimum_component_by_class
            # Cleanup is profile-specific, so independently filtering three
            # already nested evidence sets can break nesting.  The former code
            # restored nesting by deleting balanced/conservative detections when
            # the more sensitive cleanup happened to remove them.  That made a
            # weaker profile veto stronger evidence.  Preserve stronger results
            # in the weaker profiles instead; best labels are shared, so no
            # cross-profile class conflict is introduced.
            _enforce_profile_nesting(final_profiles)
            terminal_stripes, terminal_stripe_records = _apply_post_nesting_repeated_column_filter(
                final_profiles,
                output_mask,
                group.definition.spatial_cleanup,
            )
            for policy in POLICY_ORDER:
                stripe_profiles[policy] |= terminal_stripes[policy]
                terminal_record = terminal_stripe_records[policy]
                cleanup_records[policy]["post_nesting_repeated_column_filter"] = terminal_record
                cleanup_records[policy]["repeated_segment_column_removed"] += int(
                    terminal_record["applied_pixels"]
                )
                cleanup_records[policy]["repeated_segment_column_count"] += int(
                    terminal_record["repeated_column_count"]
                )
                cleanup_records[policy]["dominant_repeated_corridor_count"] += int(
                    terminal_record["dominant_corridor_count"]
                )
                cleanup_records[policy]["maximum_segment_repetition"] = max(
                    int(cleanup_records[policy]["maximum_segment_repetition"]),
                    int(terminal_record["maximum_segment_repetition"]),
                )
            stripe_union = np.logical_or.reduce(tuple(stripe_profiles.values()))

            balanced_settings = policies[group_id]["balanced"]
            balanced_candidate = profile_candidates["balanced"]
            rejection = np.zeros(best_labels.shape, dtype=np.uint8)
            in_mask = output_mask & np.isfinite(score_full[start:stop])
            rejection[in_mask & (score_full[start:stop] > float(balanced_settings["absolute_sam_threshold_rad"]))] = 1
            rejection[in_mask & (rejection == 0) & ~balanced_candidate] = 2
            pending = balanced_candidate & (rejection == 0)
            failed = pending & (best_labels < 0)
            rejection[failed] = 10
            pending &= ~failed
            failed = pending & (depth_map < float(balanced_settings["minimum_absorption_depth"]))
            rejection[failed] = 3
            pending &= ~failed
            failed = pending & ~gate_profiles["balanced"]
            rejection[failed] = 4
            pending &= ~failed
            requested_label = np.isin(best_labels, [group.mineral_order.index(item) for item in group.output_minerals])
            balanced_margin_pass = _competition_margin_pass(
                margin,
                float(balanced_settings["resolved_minimum_margin"]),
                mineral_count=mineral_count,
                classified=best_labels >= 0,
            )
            failed = pending & (
                ~requested_label
                | ~balanced_margin_pass
            )
            rejection[failed] = 5
            pending &= ~failed
            failed = pending & (winning_consensus < consensus_threshold_maps["balanced"])
            rejection[failed] = 6
            pending &= ~failed
            failed = pending & (
                (stability < stability_threshold_maps["balanced"])
                | (confidence < confidence_threshold_maps["balanced"])
                | (
                    (fit_quality < sff_threshold_maps["balanced"])
                    if sff_enabled
                    else np.zeros_like(pending)
                )
            )
            rejection[failed] = 7
            rejection[(before_profiles["balanced"] > 0) & (final_profiles["balanced"] == 0)] = 8
            rejection[balanced_candidate & saturation & (final_profiles["balanced"] == 0)] = 9
            rejection[final_profiles["balanced"] > 0] = 0

            group_dir = output / "groups" / group_id
            group_dir.mkdir(parents=True, exist_ok=True)
            write_spatial(score_full[start:stop].astype(np.float32), group_dir / "group_sam_score.dat", description=f"{group_id} {algorithm_short} group SAM score in radians")
            write_spatial(raw_group_scores[group_id][start:stop].astype(np.float32), group_dir / "raw_group_sam_score.dat", description=f"{group_id} raw minimum group SAM angle before mineral-domain offset calibration")
            write_spatial(detection_mineral_scores[group_id][:, start:stop].astype(np.float32), group_dir / "group_mineral_sam_scores.dat", band_names=[catalog.mineral(item).display_name_en for item in group.mineral_order], description=f"{group_id} mineral-level SAM evidence before group candidate union")
            threshold_cube = np.stack([
                np.broadcast_to(policies[group_id][policy]["column_thresholds_rad"][None, :], (height, image.info.samples))
                for policy in POLICY_ORDER
            ]).astype(np.float32)
            write_spatial(threshold_cube, group_dir / "column_threshold.dat", band_names=[item.title() for item in POLICY_ORDER], description=f"{group_id} {algorithm_short} calibrated column SAM thresholds")
            write_spatial(_candidate_output(balanced_candidate, output_mask), group_dir / "column_candidates.dat", class_names=["Unclassified", "Balanced group candidate", "Masked Pixels"], description=f"{group_id} balanced SAM candidates")
            write_spatial(raw_scores, group_dir / "raw_mineral_scores.dat", band_names=[catalog.mineral(item).display_name_en for item in group.mineral_order], description=f"{group_id} raw shared mineral scores; lower is better")
            write_spatial(calibrated_scores, group_dir / "calibrated_mineral_scores.dat", band_names=[catalog.mineral(item).display_name_en for item in group.mineral_order], description=f"{group_id} domain-calibrated shared mineral scores; lower is better")
            write_spatial(_classification_output(before_profiles["balanced"], output_mask, len(group.output_minerals)), group_dir / "classes_before_artifact_filter.dat", class_names=class_names, description=f"{group_id} balanced classes before artifact filtering")
            write_spatial(stripe_union.astype(np.uint8), group_dir / "stripe_noise_mask.dat", description=f"{group_id} weak-evidence directional artifact mask")
            write_spatial(np.broadcast_to(column_risk[None, :], (height, image.info.samples)).astype(np.float32), group_dir / "column_risk_score.dat", description=f"{group_id} fixed detector-column risk score")
            write_spatial(edge_risk.astype(np.float32), group_dir / "edge_risk_score.dat", description="Core boundary risk score")
            write_spatial(saturation.astype(np.uint8), group_dir / "saturation_mask.dat", description="Scene-relative high-reflectance saturation risk")
            write_spatial(depth_map, group_dir / "absorption_depth.dat", description=f"{group_id} continuum-removed absorption depth")
            write_spatial(margin, group_dir / "classification_margin.dat", description=f"{group_id} first-versus-second mineral score margin")
            write_spatial(fit_quality, group_dir / "sff_quality.dat", description=f"{group_id} best reference SFF scale-to-RMS evidence")
            write_spatial(confidence, group_dir / "confidence.dat", description=f"{group_id} calibrated confidence")
            write_spatial(stability.astype(np.float32), group_dir / "stability.dat", description=f"{group_id} score, label, and reference-consensus stability")
            write_spatial(rejection, group_dir / "rejection_reason.dat", description=f"{group_id} balanced rejection reason code")
            for feature_id, feature_map in feature_maps.items():
                write_spatial(feature_map, group_dir / f"feature_{feature_id}.dat", description=f"{group_id} diagnostic feature {feature_id}")
            profile_counts: dict[str, dict[str, int]] = {}
            for policy in POLICY_ORDER:
                filename = f"final_{policy}.dat"
                write_spatial(
                    _classification_output(final_profiles[policy], output_mask, len(group.output_minerals)),
                    group_dir / filename,
                    class_names=class_names,
                    description=f"CoreSpec Mapper {algorithm_short} {group_id} {policy} classification",
                )
                counts = {
                    catalog.mineral(mineral).display_name_en: int(np.count_nonzero(final_profiles[policy] == class_id))
                    for mineral, class_id in class_lookup.items()
                }
                profile_counts[policy] = counts
                for mineral, count in counts.items():
                    count_rows.append([policy, group_id, mineral, count])
            group_summary = {
                "group_id": group_id,
                "output_minerals": list(group.output_minerals),
                "internal_competitors": list(group.mineral_order),
                "selected_references": [ensemble.spectrum_names[index] for index in group.reference_indices],
                "detection_band_count": int(group.detection_indices.size),
                "classification_band_count": int(group.classification_indices.size),
                "relative_spectral_noise": group_noise[group_id],
                "detection_domain_calibration": detection_calibration[group_id],
                "calibration": calibration_record,
                "policies": policies[group_id],
                "threshold_search": threshold_search.get(group_id),
                "project_calibration": project_calibration_records.get(group_id),
                "counts": profile_counts,
                "cleanup": cleanup_records,
                "stripe_pixels_union": int(np.count_nonzero(stripe_union)),
                "confidence_percentiles": _percentiles(confidence[sensitive_candidate]),
                "stability_percentiles": _percentiles(stability[sensitive_candidate]),
                "column_risk_percentiles": _percentiles(column_risk),
                "rejection_counts_balanced": {
                    str(code): int(np.count_nonzero(rejection == code)) for code in range(0, 11)
                },
            }
            _save_json(group_summary, group_dir / "summary.json")
            run_groups[group_id] = group_summary
            cross_group_profiles[group_id] = {
                policy: final_profiles[policy].copy() for policy in POLICY_ORDER
            }
            cross_group_confidence[group_id] = confidence.copy()
            cross_group_class_names[group_id] = class_names
            cross_group_class_lookup[group_id] = class_lookup
            cross_group_stripe_unions[group_id] = stripe_union.copy()
            update = confidence > global_confidence
            global_confidence[update] = confidence[update]
            global_stability[update] = stability[update]
            global_rejection[update] = rejection[update]
            _emit(progress, started, "outputs", 0.47 + 0.41 * (group_index + 1) / len(groups), f"Wrote {algorithm_short} outputs for {group_id}")

        clay_competition_groups = (
            "white_mica_illite",
            "smectites",
            "kaolin_2170_2205",
        )
        clay_competition = _arbitrate_cross_group_profiles(
            cross_group_profiles,
            cross_group_confidence,
            clay_competition_groups,
        )
        if clay_competition["status"] == "resolved":
            affected = set(clay_competition["groups"])
            count_rows = [row for row in count_rows if row[1] not in affected]
            for group_id in clay_competition["groups"]:
                group_dir = output / "groups" / group_id
                class_lookup = cross_group_class_lookup[group_id]
                competition_stripes, competition_stripe_records = (
                    _apply_post_nesting_repeated_column_filter(
                        cross_group_profiles[group_id],
                        output_mask,
                        {
                            **catalog.group(group_id).spatial_cleanup,
                            # The dominant class-corridor audit already ran on
                            # the nested profiles.  Cross-group competition may
                            # expose exact peaks, but must not trigger a second
                            # dominance peel of the same corridor.
                            "dominant_repeated_corridor_enabled": False,
                        },
                    )
                )
                competition_stripe_union = np.logical_or.reduce(
                    tuple(competition_stripes.values())
                )
                cross_group_stripe_unions[group_id] |= competition_stripe_union
                write_spatial(
                    cross_group_stripe_unions[group_id].astype(np.uint8),
                    group_dir / "stripe_noise_mask.dat",
                    description=(
                        f"{group_id} directional and repeated-column artifact mask "
                        "after shared clay competition"
                    ),
                )
                run_groups[group_id]["post_competition_repeated_column_filter"] = (
                    competition_stripe_records
                )
                run_groups[group_id]["stripe_pixels_union"] = int(
                    np.count_nonzero(cross_group_stripe_unions[group_id])
                )
                for policy in POLICY_ORDER:
                    terminal_record = competition_stripe_records[policy]
                    cleanup_record = run_groups[group_id]["cleanup"][policy]
                    cleanup_record["post_competition_repeated_column_filter"] = (
                        terminal_record
                    )
                    cleanup_record["repeated_segment_column_removed"] += int(
                        terminal_record["applied_pixels"]
                    )
                    cleanup_record["repeated_segment_column_count"] += int(
                        terminal_record["repeated_column_count"]
                    )
                    cleanup_record["dominant_repeated_corridor_count"] += int(
                        terminal_record["dominant_corridor_count"]
                    )
                    cleanup_record["maximum_segment_repetition"] = max(
                        int(cleanup_record["maximum_segment_repetition"]),
                        int(terminal_record["maximum_segment_repetition"]),
                    )
                profile_counts: dict[str, dict[str, int]] = {}
                for policy in POLICY_ORDER:
                    labels = cross_group_profiles[group_id][policy]
                    write_spatial(
                        _classification_output(
                            labels,
                            output_mask,
                            len(class_lookup),
                        ),
                        group_dir / f"final_{policy}.dat",
                        class_names=cross_group_class_names[group_id],
                        description=(
                            f"CoreSpec Mapper {algorithm_short} {group_id} {policy} "
                            "classification after shared clay competition"
                        ),
                    )
                    counts = {
                        catalog.mineral(mineral).display_name_en: int(
                            np.count_nonzero(labels == class_id)
                        )
                        for mineral, class_id in class_lookup.items()
                    }
                    profile_counts[policy] = counts
                    for mineral, count in counts.items():
                        count_rows.append([policy, group_id, mineral, count])
                run_groups[group_id]["counts"] = profile_counts
                run_groups[group_id]["cross_group_competition"] = clay_competition
                _save_json(run_groups[group_id], group_dir / "summary.json")

        confidence_dir = output / "confidence"
        confidence_dir.mkdir(parents=True, exist_ok=True)
        write_spatial(global_confidence, confidence_dir / "confidence.dat", description=f"Maximum {algorithm_short} confidence across mapped mineral groups")
        write_spatial(global_stability, confidence_dir / "stability.dat", description="Stability associated with maximum-confidence mineral group")
        write_spatial(global_rejection, confidence_dir / "rejection_reason.dat", description="Balanced rejection reason associated with maximum-confidence group")
        tables = output / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        with (tables / "mineral_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["policy", "group", "mineral", "pixels"])
            writer.writerows(count_rows)
        summary = {
            "version": f"CoreSpec Mapper {algorithm_version}",
            "analysis_image": str(image.info.data_path),
            "analysis_mask": str(config.get("analysis_mask", "prepared_or_automatic_material_mask")),
            "output_lines": [start, stop],
            "mask_pixels": int(np.count_nonzero(output_mask)),
            "sensor_signature": card.sensor_signature,
            "spectral_domain": card.spectral_domain,
            "data_physics": card.data_physics,
            "requested_minerals": list(requested),
            "internal_competitors": list(internal),
            "sample_plan": sample_plan.to_dict(),
            "preprocessing": {
                "input_is_smoothed": input_is_smoothed,
                "sg_window": sg_window,
                "sg_polyorder": sg_order,
                "continuum_method": continuum_method,
                "column_spectral_correction": {
                    "enabled": correction_enabled,
                    "strength": correction_strength,
                    "radius": int(correction_config.get("radius", 2)),
                    "application": "group_detection_only",
                    "bias_abs_percentiles": _percentiles(np.abs(column_bias)),
                },
            },
            "saturation_threshold": saturation_threshold,
            "ensemble_artifacts": ensemble_paths,
            "groups": run_groups,
            "elapsed_seconds_before_qa": monotonic() - started,
        }
        _save_json(summary, output / "summary.json")
        _emit(progress, started, "complete", 0.90, f"{algorithm_short} scientific outputs are ready for preview and QA")
        return summary
    finally:
        image.close()
        if mask_dataset is not None:
            mask_dataset.close()
