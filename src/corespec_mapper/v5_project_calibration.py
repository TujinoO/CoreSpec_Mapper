from __future__ import annotations

"""Project-scoped calibration from reviewed, same-grid weak mineral labels.

Only compact evidence parameters are learned: group detection thresholds,
per-mineral score offsets, and sensor-aware feature gates.  Classification
pixels are never copied into a new result.  Interleaved depth blocks are held
out so every learned parameter has an out-of-sample project regression check.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .envi import EnviDataset


DEFAULT_WEAK_LABEL_SOURCES: Mapping[str, tuple[str, Mapping[str, int]]] = {
    "carbonate_2300": (
        "carbonates/v3_final_classes.dat",
        {"calcite": 1, "dolomite": 2},
    ),
    "calcium_sulfates": (
        "sulfates/v3_final_classes.dat",
        {"anhydrite": 1, "gypsum": 2},
    ),
    "white_mica_illite": (
        "clays/v3_final_classes.dat",
        {"illite": 1},
    ),
    "smectites": (
        "clays/v3_final_classes.dat",
        {"montmorillonite": 2},
    ),
    "kaolin_2170_2205": (
        "clays/v3_final_classes.dat",
        {"kaolinite": 3},
    ),
}


@dataclass(frozen=True)
class ProjectWeakLabelSet:
    root: Path
    labels: Mapping[str, Mapping[str, np.ndarray]]
    training_mask: np.ndarray
    holdout_mask: np.ndarray
    settings: Mapping[str, Any]
    audit: Mapping[str, Any]


def _read_single_band(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with EnviDataset(path) as dataset:
        if dataset.info.bands != 1:
            raise ValueError(f"Weak-label classification must have one band: {path}")
        if (dataset.info.lines, dataset.info.samples) != expected_shape:
            raise ValueError(
                f"Weak-label grid {dataset.info.lines}x{dataset.info.samples} does not match "
                f"analysis grid {expected_shape[0]}x{expected_shape[1]}: {path}"
            )
        return np.asarray(dataset.read_rows(0, dataset.info.lines, bands=[0])[:, :, 0]).copy()


def load_project_weak_labels(
    config: Mapping[str, Any] | None,
    expected_shape: tuple[int, int],
    material_mask: np.ndarray,
) -> ProjectWeakLabelSet | None:
    value = {} if config is None else dict(config)
    if not bool(value.get("enabled", False)):
        return None
    root_value = value.get("baseline_root")
    if not root_value:
        raise ValueError("Project weak-label calibration requires baseline_root")
    root = Path(str(root_value)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    block_rows = max(16, int(value.get("holdout_block_rows", 256)))
    modulo = max(3, int(value.get("holdout_modulo", 5)))
    row_blocks = np.arange(expected_shape[0], dtype=np.int64) // block_rows
    holdout_rows = row_blocks % modulo == 0
    material = np.asarray(material_mask, dtype=bool)
    holdout = material & holdout_rows[:, None]
    training = material & ~holdout_rows[:, None]
    labels: dict[str, dict[str, np.ndarray]] = {}
    files: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    cache: dict[Path, np.ndarray] = {}
    for group_id, (relative, class_map) in DEFAULT_WEAK_LABEL_SOURCES.items():
        path = root / relative
        if not path.is_file():
            continue
        if path not in cache:
            cache[path] = _read_single_band(path, expected_shape)
        values = cache[path]
        group_labels: dict[str, np.ndarray] = {}
        counts[group_id] = {}
        for mineral_id, class_id in class_map.items():
            positive = material & (values == int(class_id))
            if np.any(positive):
                group_labels[mineral_id] = positive
                counts[group_id][mineral_id] = int(np.count_nonzero(positive))
        if group_labels:
            labels[group_id] = group_labels
            files[group_id] = str(path)
    if not labels:
        raise ValueError(f"No recognised weak-label classifications were found below {root}")
    settings = {
        "role": str(value.get("role", "human_reviewed_project_weak_labels")),
        "minimum_pixels_per_mineral": max(20, int(value.get("minimum_pixels_per_mineral", 100))),
        "holdout_block_rows": block_rows,
        "holdout_modulo": modulo,
        "detection_recall_targets": dict(
            value.get(
                "detection_recall_targets",
                {"conservative": 0.80, "balanced": 0.93, "sensitive": 0.985},
            )
        ),
        "maximum_candidate_coverage": dict(
            value.get(
                "maximum_candidate_coverage",
                {"conservative": 0.20, "balanced": 0.38, "sensitive": 0.60},
            )
        ),
        # SAM is deliberately a high-recall discovery stage.  These later
        # evidence profiles carry the precision burden and are learned from
        # the reviewed mineral pixels without copying their classifications.
        "evidence_recall_targets": dict(
            value.get(
                "evidence_recall_targets",
                {"conservative": 0.55, "balanced": 0.82, "sensitive": 0.95},
            )
        ),
        "evidence_area_multipliers": dict(
            value.get(
                "evidence_area_multipliers",
                {"conservative": 0.85, "balanced": 1.75, "sensitive": 3.00},
            )
        ),
        "evidence_maximum_domain_fraction": dict(
            value.get(
                "evidence_maximum_domain_fraction",
                {"conservative": 0.04, "balanced": 0.25, "sensitive": 0.25},
            )
        ),
        "classification_target_recall": float(value.get("classification_target_recall", 0.85)),
        "feature_lower_quantile": float(value.get("feature_lower_quantile", 0.02)),
        "feature_upper_quantile": float(value.get("feature_upper_quantile", 0.98)),
        "feature_quantiles_by_policy": dict(
            value.get(
                "feature_quantiles_by_policy",
                {
                    "conservative": [0.10, 0.90],
                    "balanced": [0.02, 0.98],
                    "sensitive": [0.005, 0.995],
                },
            )
        ),
        "feature_catalog_floor_fraction": dict(
            value.get(
                "feature_catalog_floor_fraction",
                {"conservative": 0.35, "balanced": 0.10, "sensitive": 0.02},
            )
        ),
    }
    audit = {
        "enabled": True,
        "role": settings["role"],
        "baseline_root": str(root),
        "files": files,
        "positive_counts": counts,
        "training_material_pixels": int(np.count_nonzero(training)),
        "holdout_material_pixels": int(np.count_nonzero(holdout)),
        "holdout_design": "interleaved contiguous depth blocks",
        "parameters_only": True,
        "classification_pixels_copied": False,
    }
    return ProjectWeakLabelSet(root, labels, training, holdout, settings, audit)


def calibrate_detection_settings(
    scores: np.ndarray,
    material_mask: np.ndarray,
    weak_labels: Mapping[str, np.ndarray],
    training_mask: np.ndarray,
    holdout_mask: np.ndarray,
    policy: str,
    settings: Mapping[str, Any],
    current: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    material = np.asarray(material_mask, dtype=bool) & np.isfinite(values)
    positive = np.logical_or.reduce(tuple(np.asarray(item, dtype=bool) for item in weak_labels.values()))
    train = positive & training_mask & np.isfinite(values)
    holdout = positive & holdout_mask & np.isfinite(values)
    minimum = int(settings["minimum_pixels_per_mineral"])
    if np.count_nonzero(train) < minimum:
        return dict(current), {"status": "insufficient_training_labels", "training_positive_pixels": int(np.count_nonzero(train))}
    target = float(settings["detection_recall_targets"][policy])
    maximum_coverage = float(settings["maximum_candidate_coverage"][policy])
    positive_threshold = float(np.quantile(values[train], target))
    scene_threshold = float(np.quantile(values[material], maximum_coverage))
    learned_threshold = min(positive_threshold, scene_threshold)
    resolved = dict(current)
    resolved["absolute_sam_threshold_rad"] = learned_threshold
    resolved["scene_sam_threshold_rad"] = learned_threshold
    resolved["column_thresholds_rad"] = np.full(values.shape[1], learned_threshold, dtype=np.float64)
    resolved["threshold_source"] = "project_weak_label_train_with_heldout_depth_blocks"
    candidate = material & (values <= learned_threshold)

    def recall(mask: np.ndarray) -> float | None:
        count = int(np.count_nonzero(mask))
        return float(np.count_nonzero(candidate & mask) / count) if count else None

    record = {
        "status": "resolved",
        "policy": policy,
        "target_training_recall": target,
        "positive_quantile_threshold_rad": positive_threshold,
        "coverage_cap_threshold_rad": scene_threshold,
        "resolved_threshold_rad": learned_threshold,
        "training_positive_pixels": int(np.count_nonzero(train)),
        "holdout_positive_pixels": int(np.count_nonzero(holdout)),
        "training_recall": recall(train),
        "holdout_recall": recall(holdout),
        "candidate_coverage": float(np.count_nonzero(candidate) / max(np.count_nonzero(material), 1)),
    }
    return resolved, record


def learn_score_offsets(
    raw_scores: np.ndarray,
    mineral_order: Sequence[str],
    weak_labels: Mapping[str, np.ndarray],
    training_mask: np.ndarray,
    holdout_mask: np.ndarray,
    *,
    minimum_pixels: int,
    maximum_offset: float = 0.45,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[0] != len(mineral_order):
        raise ValueError("raw_scores must be mineral x lines x samples")
    sample_scores: list[np.ndarray] = []
    sample_labels: list[np.ndarray] = []
    per_class: dict[str, int] = {}
    for index, mineral_id in enumerate(mineral_order):
        positive = weak_labels.get(mineral_id)
        if positive is None:
            continue
        selected = np.asarray(positive, dtype=bool) & training_mask & np.all(np.isfinite(scores), axis=0)
        count = int(np.count_nonzero(selected))
        per_class[mineral_id] = count
        if count < minimum_pixels:
            continue
        current = scores[:, selected].T
        # Bound runtime and balance the much larger gypsum/illite classes.
        if current.shape[0] > 5000:
            take = np.linspace(0, current.shape[0] - 1, 5000, dtype=np.int64)
            current = current[take]
        sample_scores.append(current)
        sample_labels.append(np.full(current.shape[0], index, dtype=np.int64))
    if not sample_scores or len(mineral_order) <= 1:
        return np.zeros(len(mineral_order), dtype=np.float64), {
            "status": "not_required_or_insufficient_labels",
            "training_pixels_by_mineral": per_class,
        }
    x = np.vstack(sample_scores)
    y = np.concatenate(sample_labels)
    class_counts = np.bincount(y, minlength=len(mineral_order)).astype(np.float64)
    weights = np.divide(1.0, class_counts[y], out=np.zeros(y.shape, dtype=np.float64), where=class_counts[y] > 0)
    weights /= max(np.sum(weights), 1e-12)
    finite_differences = []
    if x.shape[1] > 1:
        ordered = np.sort(x, axis=1)
        finite_differences = ordered[:, 1] - ordered[:, 0]
    temperature = max(float(np.median(finite_differences)) if len(finite_differences) else 0.05, 0.03)
    intercepts = np.zeros(x.shape[1], dtype=np.float64)
    learning_rate = 0.25
    for _ in range(800):
        logits = -x / temperature + intercepts[None, :]
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        target = np.zeros_like(probabilities)
        target[np.arange(y.size), y] = 1.0
        gradient = np.sum(weights[:, None] * (target - probabilities), axis=0)
        gradient -= 0.02 * intercepts
        intercepts += learning_rate * gradient
        intercepts -= np.mean(intercepts)
    offsets = np.clip(intercepts * temperature, -maximum_offset, maximum_offset)
    offsets -= np.mean(offsets)
    calibrated = scores - offsets[:, None, None]

    def winner_recall(mineral_id: str, evaluation_mask: np.ndarray) -> float | None:
        if mineral_id not in mineral_order or mineral_id not in weak_labels:
            return None
        selected = weak_labels[mineral_id] & evaluation_mask & np.all(np.isfinite(calibrated), axis=0)
        count = int(np.count_nonzero(selected))
        if not count:
            return None
        return float(np.count_nonzero(np.argmin(calibrated[:, selected], axis=0) == mineral_order.index(mineral_id)) / count)

    record = {
        "status": "resolved",
        "method": "class_balanced_multinomial_intercept_calibration",
        "temperature": temperature,
        "offsets": {mineral_id: float(offsets[index]) for index, mineral_id in enumerate(mineral_order)},
        "training_pixels_by_mineral": per_class,
        "training_winner_recall": {
            mineral_id: winner_recall(mineral_id, training_mask) for mineral_id in weak_labels
        },
        "holdout_winner_recall": {
            mineral_id: winner_recall(mineral_id, holdout_mask) for mineral_id in weak_labels
        },
    }
    return offsets, record


def learn_feature_gate_override(
    mineral_id: str,
    weak_positive: np.ndarray,
    training_mask: np.ndarray,
    feature_maps: Mapping[str, np.ndarray],
    catalog_gate: Mapping[str, Any],
    policy: str,
    settings: Mapping[str, Any],
    feature_windows: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = np.asarray(weak_positive, dtype=bool) & training_mask
    policy_quantiles = settings.get("feature_quantiles_by_policy", {}).get(policy)
    if policy_quantiles is None:
        lower_q = float(settings["feature_lower_quantile"])
        upper_q = float(settings["feature_upper_quantile"])
    else:
        lower_q, upper_q = (float(item) for item in policy_quantiles)
    lower_q = float(np.clip(lower_q, 0.0, 0.49))
    upper_q = float(np.clip(upper_q, 0.51, 1.0))
    catalog_floor_fraction = float(
        settings.get("feature_catalog_floor_fraction", {}).get(policy, 0.0)
    )
    override: dict[str, Any] = {}
    record: dict[str, Any] = {
        "mineral_id": mineral_id,
        "policy": policy,
        "status": "resolved",
        "training_quantiles": [lower_q, upper_q],
        "catalog_floor_fraction": catalog_floor_fraction,
    }
    center_id = catalog_gate.get("center_feature")
    if center_id in feature_maps:
        values = np.asarray(feature_maps[center_id], dtype=np.float64)[selected]
        values = values[np.isfinite(values)]
        if values.size:
            learned = [float(np.quantile(values, lower_q)), float(np.quantile(values, upper_q))]
            bounds = feature_windows.get(str(center_id))
            if bounds is not None:
                learned = [max(float(bounds[0]), learned[0]), min(float(bounds[1]), learned[1])]
            if learned[0] <= learned[1]:
                override["center_feature"] = center_id
                override["center_window_nm"] = learned
                record["center_window_nm"] = learned
    minimum_values: dict[str, float] = {}
    for feature_id, catalog_minimum in catalog_gate.get("minimum_feature_values", {}).items():
        if feature_id not in feature_maps:
            continue
        values = np.asarray(feature_maps[feature_id], dtype=np.float64)[selected]
        values = values[np.isfinite(values)]
        if values.size:
            learned_minimum = float(np.quantile(values, lower_q))
            minimum_values[str(feature_id)] = max(
                0.0,
                learned_minimum,
                catalog_floor_fraction * float(catalog_minimum),
            )
    if minimum_values:
        override["minimum_feature_values"] = minimum_values
        override["minimum_feature_match"] = "any" if policy == "sensitive" else "all"
        record["minimum_feature_values"] = minimum_values
        record["minimum_feature_match"] = override["minimum_feature_match"]
    record["training_positive_pixels"] = int(np.count_nonzero(selected))
    return override, record


def learn_evidence_gate(
    feature_planes: Sequence[np.ndarray],
    weak_positive: np.ndarray,
    eligible_mask: np.ndarray,
    training_mask: np.ndarray,
    holdout_mask: np.ndarray,
    *,
    minimum_pixels: int,
    weak_negative: np.ndarray | None = None,
    selection_mask: np.ndarray | None = None,
    return_profiles: bool = False,
    profile_settings: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
    """Fit one QDA ranking and resolve nested evidence gates for three policies.

    The group SAM gate is intentionally a broad discovery domain.  Precision
    is recovered here with mineral-specific SFF/feature evidence.  A single
    continuous ranking is learned once, then conservative/balanced/sensitive
    thresholds and area guards are resolved independently so the profiles no
    longer collapse onto one shared Boolean project gate.
    """

    if not feature_planes:
        raise ValueError("At least one evidence plane is required")
    shape = np.asarray(feature_planes[0]).shape
    if any(np.asarray(plane).shape != shape for plane in feature_planes):
        raise ValueError("All evidence planes must share one spatial shape")
    planes = [np.asarray(plane).reshape(-1) for plane in feature_planes]
    observable_pixel = np.zeros(planes[0].shape, dtype=bool)
    for plane in planes:
        observable_pixel |= np.isfinite(plane)
    eligible = np.asarray(eligible_mask, dtype=bool).reshape(-1) & observable_pixel
    selection = (
        eligible
        if selection_mask is None
        else eligible & np.asarray(selection_mask, dtype=bool).reshape(-1)
    )
    positive = np.asarray(weak_positive, dtype=bool).reshape(-1)
    training = np.asarray(training_mask, dtype=bool).reshape(-1)
    holdout = np.asarray(holdout_mask, dtype=bool).reshape(-1)
    train_positive = eligible & positive & training
    if weak_negative is None:
        train_negative = eligible & ~positive & training
        negative_source = "eligible_scene_complement"
    else:
        negative = np.asarray(weak_negative, dtype=bool).reshape(-1)
        train_negative = eligible & negative & ~positive & training
        negative_source = "reviewed_within_group_competitors"
        if np.count_nonzero(train_negative) < minimum_pixels:
            train_negative = eligible & ~positive & training
            negative_source = "eligible_scene_complement_fallback"
    positive_count = int(np.count_nonzero(train_positive))
    negative_indices = np.flatnonzero(train_negative)
    if positive_count < minimum_pixels or negative_indices.size < minimum_pixels:
        all_pass = np.ones(shape, dtype=bool)
        resolved = (
            {policy: all_pass.copy() for policy in ("conservative", "balanced", "sensitive")}
            if return_profiles
            else all_pass
        )
        return resolved, {
            "status": "insufficient_training_labels",
            "training_positive_pixels": positive_count,
            "training_negative_pixels": int(negative_indices.size),
        }
    maximum_negative = min(12000, max(4000, positive_count * 4))
    if negative_indices.size > maximum_negative:
        take = np.linspace(0, negative_indices.size - 1, maximum_negative, dtype=np.int64)
        negative_indices = negative_indices[take]
    positive_indices = np.flatnonzero(train_positive)
    fit_indices = np.concatenate((positive_indices, negative_indices))
    fit = np.column_stack(
        [np.asarray(plane[fit_indices], dtype=np.float64) for plane in planes]
    )
    observable_columns = np.any(np.isfinite(fit), axis=0)
    if not np.any(observable_columns):
        all_pass = np.ones(shape, dtype=bool)
        resolved = (
            {policy: all_pass.copy() for policy in ("conservative", "balanced", "sensitive")}
            if return_profiles
            else all_pass
        )
        return resolved, {
            "status": "no_observable_training_features",
            "training_positive_pixels": positive_count,
            "training_negative_pixels": int(negative_indices.size),
        }
    fit = fit[:, observable_columns]
    observable_planes = [plane for plane, keep in zip(planes, observable_columns) if keep]
    medians = np.nanmedian(fit, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    fit = np.where(np.isfinite(fit), fit, medians[None, :])
    q25, q75 = np.percentile(fit, [25.0, 75.0], axis=0)
    scales = np.where(q75 - q25 > 1e-8, q75 - q25, np.std(fit, axis=0))
    scales = np.where(scales > 1e-8, scales, 1.0)
    standard = (fit - medians[None, :]) / scales[None, :]
    positive_fit = standard[: positive_indices.size]
    negative_fit = standard[positive_indices.size :]
    positive_mean = np.mean(positive_fit, axis=0)
    negative_mean = np.mean(negative_fit, axis=0)
    positive_variance = np.maximum(np.var(positive_fit, axis=0), 0.20)
    negative_variance = np.maximum(np.var(negative_fit, axis=0), 0.20)
    separation = float(np.linalg.norm(positive_mean - negative_mean))
    if not np.isfinite(separation) or separation < 1e-8:
        all_pass = np.ones(shape, dtype=bool)
        resolved = (
            {policy: all_pass.copy() for policy in ("conservative", "balanced", "sensitive")}
            if return_profiles
            else all_pass
        )
        return resolved, {
            "status": "non_discriminative_training_evidence",
            "training_positive_pixels": positive_count,
            "training_negative_pixels": int(negative_indices.size),
        }
    def score_rows(rows: np.ndarray) -> np.ndarray:
        values = np.where(np.isfinite(rows), rows, medians[None, :])
        standard_rows = (values - medians[None, :]) / scales[None, :]
        positive_log_likelihood = -0.5 * np.sum(
            ((standard_rows - positive_mean[None, :]) ** 2) / positive_variance[None, :]
            + np.log(positive_variance[None, :]),
            axis=1,
        )
        negative_log_likelihood = -0.5 * np.sum(
            ((standard_rows - negative_mean[None, :]) ** 2) / negative_variance[None, :]
            + np.log(negative_variance[None, :]),
            axis=1,
        )
        return positive_log_likelihood - negative_log_likelihood

    positive_scores = score_rows(
        np.column_stack(
            [np.asarray(plane[positive_indices], dtype=np.float64) for plane in observable_planes]
        )
    )
    negative_scores = score_rows(
        np.column_stack(
            [np.asarray(plane[negative_indices], dtype=np.float64) for plane in observable_planes]
        )
    )
    prevalence = positive_count / max(positive_count + int(np.count_nonzero(train_negative)), 1)
    candidates = np.unique(np.quantile(np.concatenate((positive_scores, negative_scores)), np.linspace(0.02, 0.98, 161)))
    best: tuple[float, float, float, float] | None = None
    best_threshold = float(np.median(positive_scores))
    for threshold in candidates:
        recall = float(np.mean(positive_scores >= threshold))
        false_positive_rate = float(np.mean(negative_scores >= threshold))
        precision = (prevalence * recall) / max(
            prevalence * recall + (1.0 - prevalence) * false_positive_rate,
            1e-12,
        )
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        utility = f1 - 0.15 * max(0.0, 0.55 - recall)
        candidate = (utility, recall, precision, -false_positive_rate)
        if best is None or candidate > best:
            best = candidate
            best_threshold = float(threshold)
    all_scores = np.full(planes[0].shape[0], -np.inf, dtype=np.float32)
    eligible_indices = np.flatnonzero(eligible)
    for start in range(0, eligible_indices.size, 100000):
        indices = eligible_indices[start : start + 100000]
        rows = np.column_stack(
            [np.asarray(plane[indices], dtype=np.float64) for plane in observable_planes]
        )
        all_scores[indices] = score_rows(rows).astype(np.float32)
    weak_positive_pixels = int(np.count_nonzero(positive))
    resolved_profile_settings = profile_settings or {}
    recall_targets = {
        "conservative": 0.55,
        "balanced": 0.82,
        "sensitive": 0.95,
        **dict(resolved_profile_settings.get("evidence_recall_targets", {})),
    }
    area_multipliers = {
        "conservative": 0.85,
        "balanced": 1.75,
        "sensitive": 3.00,
        **dict(resolved_profile_settings.get("evidence_area_multipliers", {})),
    }
    maximum_domain_fractions = {
        "conservative": 0.04,
        "balanced": 0.25,
        "sensitive": 0.25,
        **dict(resolved_profile_settings.get("evidence_maximum_domain_fraction", {})),
    }

    def build_profile_masks(
        score_values: np.ndarray,
        positive_score_values: np.ndarray,
        baseline_threshold: float,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | int]]]:
        masks: dict[str, np.ndarray] = {}
        records: dict[str, dict[str, float | int]] = {}
        for policy in ("conservative", "balanced", "sensitive"):
            target_recall = float(np.clip(recall_targets[policy], 0.01, 0.999))
            recall_threshold = float(
                np.quantile(positive_score_values, 1.0 - target_recall)
            )
            # Conservative remains on the precision side of the F1 optimum;
            # balanced/sensitive may lower the evidence cutoff to meet their
            # requested reviewed-pixel recall.  Together with increasing area
            # guards this guarantees nested evidence domains by construction.
            resolved_threshold = (
                max(float(baseline_threshold), recall_threshold)
                if policy == "conservative"
                else min(float(baseline_threshold), recall_threshold)
            )
            maximum_selected = max(
                1,
                int(round(float(area_multipliers[policy]) * weak_positive_pixels)),
                int(
                    round(
                        float(maximum_domain_fractions[policy])
                        * int(np.count_nonzero(selection))
                    )
                ),
            )
            selected_indices = np.flatnonzero(
                selection & np.isfinite(score_values) & (score_values >= resolved_threshold)
            )
            selected = np.zeros(selection.shape, dtype=bool)
            if selected_indices.size > maximum_selected:
                strengths = score_values[selected_indices]
                keep = selected_indices[
                    np.argpartition(strengths, -maximum_selected)[-maximum_selected:]
                ]
                selected[keep] = True
            else:
                selected[selected_indices] = True
            masks[policy] = selected.reshape(shape)
            records[policy] = {
                "target_training_recall": target_recall,
                "positive_recall_threshold": recall_threshold,
                "resolved_threshold": resolved_threshold,
                "weak_label_area_multiplier": float(area_multipliers[policy]),
                "maximum_evidence_domain_fraction": float(maximum_domain_fractions[policy]),
                "maximum_selected_pixels_from_weak_label_area": maximum_selected,
                "selected_pixels": int(np.count_nonzero(selected)),
            }
        return masks, records

    profile_gates, profile_records = build_profile_masks(
        all_scores,
        positive_scores,
        best_threshold,
    )
    fallback_record: dict[str, Any] | None = None
    qda_training_recall = float(
        np.count_nonzero(profile_gates["balanced"].reshape(-1) & train_positive)
        / max(positive_count, 1)
    )
    if weak_negative is not None and qda_training_recall < 0.10:
        axis_candidates: list[tuple[float, float, float, int, int, float]] = []
        for plane_index, plane in enumerate(planes):
            positive_values = np.asarray(plane[train_positive], dtype=np.float64)
            negative_values = np.asarray(plane[train_negative], dtype=np.float64)
            positive_values = positive_values[np.isfinite(positive_values)]
            negative_values = negative_values[np.isfinite(negative_values)]
            if positive_values.size < minimum_pixels or negative_values.size < minimum_pixels:
                continue
            thresholds = np.unique(
                np.quantile(
                    np.concatenate((positive_values, negative_values)),
                    np.linspace(0.02, 0.98, 97),
                )
            )
            for direction in (-1, 1):
                best_direction: tuple[float, float, float, int, int, float] | None = None
                for threshold in thresholds:
                    recall = float(
                        np.mean(direction * positive_values >= direction * threshold)
                    )
                    false_positive_rate = float(
                        np.mean(direction * negative_values >= direction * threshold)
                    )
                    if recall < 0.40:
                        continue
                    utility = recall - false_positive_rate
                    candidate_axis = (
                        utility,
                        recall,
                        -false_positive_rate,
                        plane_index,
                        direction,
                        float(threshold),
                    )
                    if best_direction is None or candidate_axis > best_direction:
                        best_direction = candidate_axis
                if best_direction is not None:
                    axis_candidates.append(best_direction)
        resolved_axis: tuple[
            tuple[float, float, float],
            dict[str, np.ndarray],
            dict[str, dict[str, float | int]],
            tuple[float, float, float, int, int, float],
        ] | None = None
        for candidate_axis in axis_candidates:
            utility, _, negative_fpr, plane_index, direction, threshold = candidate_axis
            plane = np.asarray(planes[plane_index], dtype=np.float64)
            score_values = np.where(np.isfinite(plane), direction * plane, -np.inf)
            positive_axis_scores = score_values[train_positive]
            positive_axis_scores = positive_axis_scores[np.isfinite(positive_axis_scores)]
            if positive_axis_scores.size < minimum_pixels:
                continue
            axis_profiles, axis_records = build_profile_masks(
                score_values,
                positive_axis_scores,
                direction * threshold,
            )
            axis_training_recall = float(
                np.count_nonzero(axis_profiles["balanced"].reshape(-1) & train_positive)
                / max(positive_count, 1)
            )
            resolved_score = (axis_training_recall, utility, negative_fpr)
            if resolved_axis is None or resolved_score > resolved_axis[0]:
                resolved_axis = (
                    resolved_score,
                    axis_profiles,
                    axis_records,
                    candidate_axis,
                )
        if resolved_axis is not None and resolved_axis[0][0] > qda_training_recall:
            (axis_training_recall, _, _), axis_profiles, axis_records, candidate_axis = resolved_axis
            utility, _, negative_fpr, plane_index, direction, threshold = candidate_axis
            profile_gates = axis_profiles
            profile_records = axis_records
            fallback_record = {
                "method": "bounded_axis_aligned_physical_evidence_fallback",
                "feature_plane_index": int(plane_index),
                "direction": "minimum" if direction > 0 else "maximum",
                "threshold": threshold,
                "training_recall": axis_training_recall,
                "training_false_positive_rate_against_reviewed_competitor": -negative_fpr,
                "utility": utility,
            }

    def metrics(mask: np.ndarray, selected: np.ndarray) -> dict[str, float | int | None]:
        evaluation = eligible & np.asarray(mask, dtype=bool).reshape(-1)
        positives = evaluation & positive
        negatives = evaluation & ~positive
        positive_pixels = int(np.count_nonzero(positives))
        negative_pixels = int(np.count_nonzero(negatives))
        selected = np.asarray(selected, dtype=bool).reshape(-1)
        return {
            "positive_pixels": positive_pixels,
            "negative_pixels": negative_pixels,
            "recall": (
                float(np.count_nonzero(selected & positives) / positive_pixels) if positive_pixels else None
            ),
            "false_positive_rate": (
                float(np.count_nonzero(selected & negatives) / negative_pixels) if negative_pixels else None
            ),
        }

    for policy, gate in profile_gates.items():
        profile_records[policy]["training"] = metrics(training_mask, gate)
        profile_records[policy]["holdout"] = metrics(holdout_mask, gate)
    balanced_record = profile_records["balanced"]
    balanced_gate = profile_gates["balanced"]
    record = {
        "status": "resolved",
        "method": "robust_scaled_diagonal_qda_profiled_evidence_gate",
        "training_negative_source": negative_source,
        "feature_count": int(np.count_nonzero(observable_columns)),
        "threshold": float(balanced_record["resolved_threshold"]),
        "training": balanced_record["training"],
        "holdout": balanced_record["holdout"],
        "selected_pixels": int(np.count_nonzero(balanced_gate)),
        "selection_domain_pixels": int(np.count_nonzero(selection)),
        "maximum_selected_pixels_from_weak_label_area": int(
            balanced_record["maximum_selected_pixels_from_weak_label_area"]
        ),
        "profiles": profile_records,
        "fallback": fallback_record,
    }
    return (profile_gates if return_profiles else balanced_gate), record
