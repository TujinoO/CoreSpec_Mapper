from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .catalog import MineralCatalog
from .swir_expert import mineral_feature_gate
from .v4_calibration import POLICY_ORDER
from .v4_pipeline import PreparedRunInputs, _group_sam, _prepare_groups, _subclass_evidence
from .v5_calibration import (
    CalibrationCandidate,
    _column_thresholds,
    calibrate_group_thresholds,
    catalog_safe_search_bounds,
)


TRIAL_METHOD = "catalog_bounded_scene_proxy_plus_feature_support_v5_3"


def _policy_override(
    overrides: Mapping[str, Any],
    group_id: str,
    policy: str,
) -> dict[str, Any]:
    group = overrides.get(group_id, {})
    if group is None:
        return {}
    if not isinstance(group, Mapping):
        raise ValueError(f"Threshold overrides for {group_id} must be a mapping")
    value = group.get(policy, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Threshold override for {group_id}/{policy} must be a mapping")
    return dict(value)


def _override_value(value: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if value.get(name) is not None:
            return float(value[name])
    return None


def _candidate_mask(
    scores: np.ndarray,
    valid: np.ndarray,
    percentile: float,
    absolute: float,
) -> np.ndarray:
    columns = _column_thresholds(scores, valid & np.isfinite(scores), percentile)
    return (
        valid
        & np.isfinite(scores)
        & (scores <= float(absolute))
        & (scores <= columns[None, :])
    )


def _candidate_key(candidate: CalibrationCandidate) -> tuple[float, float]:
    return (candidate.percentile_fraction, candidate.absolute_threshold_rad)


def _feature_support_context(
    catalog: MineralCatalog,
    group: Any,
    processed_sample: np.ndarray,
    sample_mask: np.ndarray,
    wavelengths_nm: np.ndarray,
    union_candidate: np.ndarray,
) -> dict[str, Any]:
    valid, scores, consensus, depth, features, _ = _subclass_evidence(
        processed_sample,
        union_candidate,
        group,
        wavelengths_nm,
    )
    if not np.any(valid) or scores.size == 0:
        return {
            "valid": valid,
            "labels": np.empty(0, dtype=np.int64),
            "margin": np.empty(0, dtype=np.float64),
            "consensus": np.empty(0, dtype=np.float64),
            "depth": np.empty(0, dtype=np.float64),
            "feature_pass": {policy: np.empty(0, dtype=bool) for policy in POLICY_ORDER},
            "requested_winner": np.empty(0, dtype=bool),
        }

    safe_scores = np.where(np.isfinite(scores), scores, np.inf)
    labels = np.argmin(safe_scores, axis=1)
    ordered = np.sort(safe_scores, axis=1)
    if scores.shape[1] > 1:
        with np.errstate(invalid="ignore"):
            margin = ordered[:, 1] - ordered[:, 0]
        margin = np.where(
            np.isfinite(ordered[:, 0]) & np.isfinite(ordered[:, 1]),
            margin,
            np.where(np.isfinite(ordered[:, 0]), np.inf, np.nan),
        )
    else:
        margin = np.full(scores.shape[0], np.inf, dtype=np.float64)
    winning_consensus = consensus[np.arange(scores.shape[0]), labels]
    requested_indices = {
        group.mineral_order.index(mineral_id)
        for mineral_id in group.output_minerals
        if mineral_id in group.mineral_order
    }
    requested_winner = np.isin(labels, tuple(requested_indices))
    feature_pass: dict[str, np.ndarray] = {}
    for policy in POLICY_ORDER:
        current = np.zeros(labels.shape, dtype=bool)
        for mineral_index, mineral_id in enumerate(group.mineral_order):
            selected = labels == mineral_index
            if not np.any(selected):
                continue
            gate = mineral_feature_gate(
                catalog.mineral(mineral_id),
                group.definition.expert_id,
                policy,
                depth,
                features,
            )
            current[selected] = gate[selected]
        feature_pass[policy] = current
    return {
        "valid": valid,
        "labels": labels,
        "margin": margin,
        "consensus": winning_consensus,
        "depth": depth,
        "feature_pass": feature_pass,
        "requested_winner": requested_winner,
    }


def _enrich_candidate(
    candidate: CalibrationCandidate,
    candidate_mask: np.ndarray,
    context: Mapping[str, Any],
    policy_settings: Mapping[str, Any],
    policy: str,
) -> dict[str, Any]:
    record = candidate.to_dict()
    valid = np.asarray(context["valid"], dtype=bool)
    if not np.any(valid):
        record.update(
            {
                "feature_ready_pixels": 0,
                "feature_ready_fraction": 0.0,
                "joint_objective": None if candidate.objective is None else float(candidate.objective) - 0.25,
            }
        )
        return record

    selected = np.asarray(candidate_mask[valid], dtype=bool)
    depth = np.asarray(context["depth"], dtype=np.float64)
    margin = np.asarray(context["margin"], dtype=np.float64)
    consensus = np.asarray(context["consensus"], dtype=np.float64)
    feature_pass = np.asarray(context["feature_pass"][policy], dtype=bool)
    requested = np.asarray(context["requested_winner"], dtype=bool)
    ready = (
        selected
        & requested
        & np.isfinite(depth)
        & (depth >= float(policy_settings.get("minimum_absorption_depth", 0.0)))
        & feature_pass
        & np.isfinite(margin)
        & (margin >= float(policy_settings.get("minimum_margin", 0.0)))
        & np.isfinite(consensus)
        & (consensus >= float(policy_settings.get("minimum_reference_consensus", 0.0)))
    )
    ready_pixels = int(np.count_nonzero(ready))
    candidate_pixels = max(int(np.count_nonzero(selected)), 1)
    ready_fraction = ready_pixels / candidate_pixels
    support_utility = float(np.clip(np.log1p(ready_pixels) / np.log1p(64.0), 0.0, 1.0))
    joint = None
    if candidate.objective is not None:
        joint = float(
            candidate.objective
            + 0.35 * support_utility
            + 0.15 * np.sqrt(ready_fraction)
            - (0.20 if ready_pixels == 0 else 0.0)
        )
    record.update(
        {
            "feature_ready_pixels": ready_pixels,
            "feature_ready_fraction": float(ready_fraction),
            "feature_support_utility": support_utility,
            "joint_objective": joint,
        }
    )
    return record


def _select_record(
    records: list[dict[str, Any]],
    minimum_percentile: float | None,
    minimum_absolute: float | None,
) -> dict[str, Any] | None:
    eligible = [
        record
        for record in records
        if record.get("accepted")
        and record.get("joint_objective") is not None
        and (minimum_percentile is None or float(record["percentile_fraction"]) >= minimum_percentile - 1e-12)
        and (minimum_absolute is None or float(record["absolute_threshold_rad"]) >= minimum_absolute - 1e-12)
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            float(item["joint_objective"]),
            int(item.get("feature_ready_pixels", 0)),
            -float(item.get("fixed_column_penalty", 1.0)),
            -float(item.get("edge_penalty", 1.0)),
            -float(item["absolute_threshold_rad"]),
            -float(item["percentile_fraction"]),
        ),
    )


def build_v5_threshold_trial(
    catalog: MineralCatalog,
    prepared: PreparedRunInputs,
    wavelengths_nm: np.ndarray,
    *,
    continuum_method: str = "segmented_linear",
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an editable, run-ready threshold ladder on audited scene samples.

    The trial is deliberately truth-free: it searches only inside the Catalog
    envelope, scores spatial/column artefact risk, and then reranks candidates by
    the number of sampled pixels that can also pass the mineral-specific spectral
    feature gates.  It does not claim mineralogical accuracy without independent
    XRD/Raman/point truth.
    """
    override_map = overrides or {}
    if not isinstance(override_map, Mapping):
        raise ValueError("threshold_trial.overrides must be a mapping")
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    groups = _prepare_groups(
        catalog,
        prepared.ensemble,
        wavelengths,
        prepared.requested_minerals,
        prepared.internal_minerals,
        prepared.sensor_card.bad_band_indices,
        continuum_method,
    )
    thresholds: dict[str, Any] = {}
    runtime_overrides: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    all_ready = True

    for group in groups:
        group_id = group.definition.group_id
        group_scores, _ = _group_sam(prepared.processed_sample, prepared.sample_mask, group)
        finite = prepared.sample_mask & np.isfinite(group_scores)
        percentile_bounds, absolute_bounds = catalog_safe_search_bounds(group.definition)
        union_candidate = _candidate_mask(
            group_scores,
            finite,
            percentile_bounds[1],
            absolute_bounds[1],
        )
        context = _feature_support_context(
            catalog,
            group,
            prepared.processed_sample,
            prepared.sample_mask,
            wavelengths,
            union_candidate,
        )
        thresholds[group_id] = {}
        runtime_overrides[group_id] = {}
        minimum_percentile: float | None = None
        minimum_absolute: float | None = None

        for policy in POLICY_ORDER:
            explicit = _policy_override(override_map, group_id, policy)
            explicit_percentile = _override_value(
                explicit,
                "column_percentile",
                "resolved_percentile_fraction",
            )
            explicit_absolute = _override_value(
                explicit,
                "absolute_sam_threshold_rad",
                "absolute_sam_rad",
                "resolved_absolute_threshold_rad",
            )
            calibration = calibrate_group_thresholds(
                group.definition,
                policy,
                group_scores,
                finite,
                percentile_candidates=(None if explicit_percentile is None else [explicit_percentile]),
                absolute_threshold_candidates_rad=(None if explicit_absolute is None else [explicit_absolute]),
            )
            candidate_lookup = {_candidate_key(candidate): candidate for candidate in calibration.candidates}
            enriched: list[dict[str, Any]] = []
            for key, candidate in candidate_lookup.items():
                mask = _candidate_mask(group_scores, finite, key[0], key[1])
                enriched.append(
                    _enrich_candidate(
                        candidate,
                        mask,
                        context,
                        group.definition.policies[policy],
                        policy,
                    )
                )
            selected = _select_record(enriched, minimum_percentile, minimum_absolute)
            if selected is None:
                all_ready = False
                reason = calibration.resolved_reason
                thresholds[group_id][policy] = {
                    "status": "blocked",
                    "parameter": "SAM 角度与列分位联合阈值",
                    "method": TRIAL_METHOD,
                    "candidate_range": {
                        "column_percentile": list(percentile_bounds),
                        "absolute_sam_rad": list(absolute_bounds),
                        "evaluated_candidates": len(enriched),
                    },
                    "final": None,
                    "editable": False,
                    "source": "V5.3 场景阈值联合试算",
                    "resolved_reason": reason,
                    "candidates": enriched,
                }
                warnings.append(
                    {
                        "code": "THRESHOLD_TRIAL_BLOCKED",
                        "severity": "error",
                        "group_id": group_id,
                        "policy": policy,
                        "message": reason,
                    }
                )
                continue

            percentile = float(selected["percentile_fraction"])
            absolute = float(selected["absolute_threshold_rad"])
            minimum_percentile = percentile
            minimum_absolute = absolute
            runtime_overrides[group_id][policy] = {
                "column_percentile": percentile,
                "absolute_sam_threshold_rad": absolute,
                "source": "user_reviewed" if explicit else "automatic_threshold_trial",
            }
            feature_ready = int(selected.get("feature_ready_pixels", 0))
            if feature_ready == 0:
                warnings.append(
                    {
                        "code": "NO_FEATURE_READY_SAMPLE",
                        "severity": "warning",
                        "group_id": group_id,
                        "policy": policy,
                        "message": (
                            "阈值包络内存在组级候选，但分层样本中没有像元同时通过矿物特征门；"
                            "运行前应复核目标矿物、标准谱与数据物理。"
                        ),
                    }
                )
            thresholds[group_id][policy] = {
                "status": "resolved",
                "parameter": "SAM 角度与列分位联合阈值",
                "method": TRIAL_METHOD,
                "candidate_range": {
                    "column_percentile": list(percentile_bounds),
                    "absolute_sam_rad": list(absolute_bounds),
                    "evaluated_candidates": len(enriched),
                },
                "final": {
                    "column_percentile": percentile,
                    "absolute_sam_rad": absolute,
                },
                "editable": True,
                "source": "用户复核值" if explicit else "V5.3 场景联合优选",
                "selected_proxy_metrics": {
                    key: selected.get(key)
                    for key in (
                        "objective",
                        "joint_objective",
                        "candidate_pixels",
                        "sample_pixels",
                        "coverage",
                        "evidence_strength",
                        "spatial_support",
                        "fixed_column_penalty",
                        "edge_penalty",
                        "isolated_component_penalty",
                        "feature_ready_pixels",
                        "feature_ready_fraction",
                    )
                },
                "resolved_reason": (
                    "在 Catalog 安全边界内联合最大化场景证据、空间支持和矿物特征可通过率；"
                    f"样本候选 {selected.get('candidate_pixels', 0)}，"
                    f"特征可通过 {feature_ready}。"
                ),
                "candidates": enriched,
            }

    return {
        "status": "resolved" if all_ready else "blocked",
        "method": TRIAL_METHOD,
        "thresholds": thresholds,
        "runtime_overrides": runtime_overrides,
        "warnings": warnings,
        "disclaimer": (
            "阈值试算基于无标签场景代理目标，只证明参数位于安全边界并具有可解释的场景支持；"
            "矿物学准确性仍需 XRD、拉曼、薄片或点光谱独立验证。"
        ),
    }
