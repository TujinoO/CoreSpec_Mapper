from __future__ import annotations

"""Audit V5 evidence gates against a same-grid, human-reviewed weak-label map.

The weak labels are used only for diagnostics: this script does not alter or
copy classification values.  It reports where known positive pixels first lose
recall in the adaptive evidence chain and summarises the evidence distributions
needed to calibrate project-specific, physically bounded gates.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corespec_mapper.catalog import MineralCatalog, default_v5_runtime_catalog_path  # noqa: E402
from corespec_mapper.envi import EnviDataset  # noqa: E402
from corespec_mapper.swir_expert import mineral_feature_gate  # noqa: E402


TARGETS = {
    "calcite": ("carbonates", 1, "carbonate_2300", "Calcite"),
    "dolomite": ("carbonates", 2, "carbonate_2300", "Dolomite"),
    "anhydrite": ("sulfates", 1, "calcium_sulfates", "Anhydrite"),
    "gypsum": ("sulfates", 2, "calcium_sulfates", "Gypsum"),
    "illite": ("clays", 1, "white_mica_illite", "Illite"),
    "montmorillonite": ("clays", 2, "smectites", "Montmorillonite"),
    "kaolinite": ("clays", 3, "kaolin_2170_2205", "Kaolinite"),
}


def _read(path: Path) -> tuple[np.ndarray, list[str]]:
    with EnviDataset(path) as dataset:
        values = np.asarray(dataset.read_rows(0, dataset.info.lines)).copy()
        names = [str(value) for value in dataset.info.metadata.get("band names", [])]
    return values, names


def _single(path: Path) -> np.ndarray:
    values, _ = _read(path)
    if values.shape[2] != 1:
        raise ValueError(f"Expected one band: {path}")
    return values[:, :, 0]


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {key: None for key in ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99")}
    levels = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    return {f"p{level:02d}": float(value) for level, value in zip(levels, np.percentile(finite, levels))}


def _fraction(mask: np.ndarray, positive: np.ndarray) -> float:
    count = int(np.count_nonzero(positive))
    return float(np.count_nonzero(mask & positive) / count) if count else 0.0


def _agreement_metrics(predicted: np.ndarray, positive: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(predicted, dtype=bool)
    reference = np.asarray(positive, dtype=bool)
    true_positive = int(np.count_nonzero(prediction & reference))
    predicted_count = int(np.count_nonzero(prediction))
    positive_count = int(np.count_nonzero(reference))
    union = int(np.count_nonzero(prediction | reference))
    return {
        "predicted_pixels": predicted_count,
        "weak_label_pixels": positive_count,
        "intersection_pixels": true_positive,
        "recall": float(true_positive / positive_count) if positive_count else 0.0,
        "precision": float(true_positive / predicted_count) if predicted_count else 0.0,
        "iou": float(true_positive / union) if union else 0.0,
    }


def _column_profile_metrics(
    predicted: np.ndarray,
    positive: np.ndarray,
    material: np.ndarray,
) -> dict[str, float | int | None]:
    denominator = np.maximum(np.sum(np.asarray(material, dtype=bool), axis=0), 1)
    predicted_profile = np.sum(np.asarray(predicted, dtype=bool), axis=0) / denominator
    reference_profile = np.sum(np.asarray(positive, dtype=bool), axis=0) / denominator
    if np.std(predicted_profile) > 0.0 and np.std(reference_profile) > 0.0:
        correlation = float(np.corrcoef(predicted_profile, reference_profile)[0, 1])
    else:
        correlation = None
    predicted_column = int(np.argmax(predicted_profile))
    reference_column = int(np.argmax(reference_profile))
    neighbor_columns = np.concatenate(
        (
            np.arange(max(0, predicted_column - 6), max(0, predicted_column - 1)),
            np.arange(
                min(predicted_profile.size, predicted_column + 2),
                min(predicted_profile.size, predicted_column + 7),
            ),
        )
    )
    local_density = (
        float(np.median(predicted_profile[neighbor_columns]))
        if neighbor_columns.size
        else 0.0
    )
    repeated_segments = 0
    for rows in np.array_split(np.arange(predicted.shape[0]), min(8, predicted.shape[0])):
        if rows.size == 0:
            continue
        segment_material = int(np.count_nonzero(material[rows, predicted_column]))
        segment_density = (
            float(np.count_nonzero(predicted[rows, predicted_column]) / segment_material)
            if segment_material
            else 0.0
        )
        repeated_segments += segment_density >= 0.05
    return {
        "profile_correlation": correlation,
        "predicted_maximum_density": float(predicted_profile[predicted_column]),
        "predicted_maximum_column": predicted_column,
        "predicted_local_neighbor_density": local_density,
        "predicted_local_spike_ratio": float(
            predicted_profile[predicted_column] / max(local_density, 0.005)
        ),
        "predicted_repeated_depth_segments": int(repeated_segments),
        "reference_maximum_density": float(reference_profile[reference_column]),
        "reference_maximum_column": reference_column,
    }


def _class_names(path: Path) -> list[str]:
    with EnviDataset(path) as dataset:
        return [str(value) for value in dataset.info.metadata.get("class names", [])]


def audit(baseline_root: Path, run_root: Path) -> dict[str, Any]:
    catalog = MineralCatalog.load(default_v5_runtime_catalog_path())
    report: dict[str, Any] = {
        "baseline_root": str(baseline_root),
        "run_root": str(run_root),
        "weak_label_role": "same-grid human-reviewed project-level diagnostic labels",
        "targets": {},
    }
    baseline_cache: dict[str, np.ndarray] = {}
    prediction_cache: dict[str, np.ndarray] = {}
    for mineral_id, (baseline_group, baseline_class, v5_group, display_name) in TARGETS.items():
        if baseline_group not in baseline_cache:
            baseline_cache[baseline_group] = _single(
                baseline_root / baseline_group / "v3_final_classes.dat"
            )
        positive = baseline_cache[baseline_group] == baseline_class
        group_dir = run_root / "groups" / v5_group
        raw_scores, raw_names = _read(group_dir / "raw_mineral_scores.dat")
        calibrated_scores, calibrated_names = _read(group_dir / "calibrated_mineral_scores.dat")
        names = calibrated_names or raw_names
        target_index = next(
            index for index, name in enumerate(names) if name.casefold() == display_name.casefold()
        )
        safe_scores = np.where(np.isfinite(calibrated_scores), calibrated_scores, np.inf)
        finite_competition = np.any(np.isfinite(calibrated_scores), axis=2)
        winner = np.where(finite_competition, np.argmin(safe_scores, axis=2), -1)
        depth = _single(group_dir / "absorption_depth.dat")
        margin = _single(group_dir / "classification_margin.dat")
        confidence = _single(group_dir / "confidence.dat")
        stability = _single(group_dir / "stability.dat")
        group_score = _single(group_dir / "group_sam_score.dat")
        column_thresholds, column_threshold_names = _read(group_dir / "column_threshold.dat")
        balanced_threshold_index = next(
            (index for index, name in enumerate(column_threshold_names) if name.casefold() == "balanced"),
            1,
        )
        column_threshold = column_thresholds[:, :, balanced_threshold_index]
        candidate = _single(group_dir / "column_candidates.dat") == 1
        before = _single(group_dir / "classes_before_artifact_filter.dat")
        final = _single(group_dir / "final_balanced.dat")
        rejection = _single(group_dir / "rejection_reason.dat").astype(np.uint8)
        output_classes = _class_names(group_dir / "final_balanced.dat")
        material = final != (len(output_classes) - 1)
        output_class = next(
            index for index, name in enumerate(output_classes) if name.casefold() == display_name.casefold()
        )
        row_blocks = np.arange(final.shape[0], dtype=np.int64) // 256
        holdout_rows = np.broadcast_to((row_blocks % 5 == 0)[:, None], final.shape)
        training_rows = ~holdout_rows
        predicted_before = before == output_class
        predicted_final = final == output_class
        prediction_cache[mineral_id] = predicted_final
        retained = predicted_before & predicted_final
        summary = json.loads((group_dir / "summary.json").read_text(encoding="utf-8-sig"))
        balanced = summary["policies"]["balanced"]
        features = {
            path.stem.removeprefix("feature_"): _single(path)
            for path in group_dir.glob("feature_*.dat")
        }
        feature_pass = mineral_feature_gate(
            catalog.mineral(mineral_id),
            catalog.group(v5_group).expert_id,
            "balanced",
            depth,
            features,
        )
        depth_floor = float(balanced["minimum_absorption_depth"])
        direct_depth_pass = np.isfinite(depth) & (depth >= depth_floor)
        target_winner = winner == target_index
        target_raw = raw_scores[:, :, target_index]
        target_calibrated = calibrated_scores[:, :, target_index]
        best_calibrated = np.min(safe_scores, axis=2)
        target = {
            "weak_label_pixels": int(np.count_nonzero(positive)),
            "agreement_metrics": {
                "full_scene": _agreement_metrics(predicted_final, positive),
                "training_depth_blocks": _agreement_metrics(
                    predicted_final & training_rows,
                    positive & training_rows,
                ),
                "holdout_depth_blocks": _agreement_metrics(
                    predicted_final & holdout_rows,
                    positive & holdout_rows,
                ),
                "before_spatial_cleanup": _agreement_metrics(predicted_before, positive),
                "column_profiles": _column_profile_metrics(
                    predicted_final,
                    positive,
                    material,
                ),
                "spatial_cleanup": {
                    "before_pixels": int(np.count_nonzero(predicted_before)),
                    "final_pixels": int(np.count_nonzero(predicted_final)),
                    "retained_from_before_pixels": int(np.count_nonzero(retained)),
                    "retention_fraction": float(
                        np.count_nonzero(retained) / max(np.count_nonzero(predicted_before), 1)
                    ),
                    "weak_positive_pixels_lost": int(
                        np.count_nonzero(positive & predicted_before & ~predicted_final)
                    ),
                },
            },
            "score_finite_pixels": {
                "scene_any_mineral": int(np.count_nonzero(finite_competition)),
                "weak_positive_any_mineral": int(np.count_nonzero(finite_competition & positive)),
                "weak_positive_target": int(np.count_nonzero(np.isfinite(target_calibrated) & positive)),
            },
            "scene_candidate_counts": {
                "balanced_group_candidates": int(np.count_nonzero(candidate)),
                "target_winner": int(np.count_nonzero(target_winner)),
                "target_winner_within_balanced_candidates": int(
                    np.count_nonzero(target_winner & candidate)
                ),
            },
            "v5_score_bands": names,
            "v5_output_classes": output_classes,
            "balanced_settings": {
                key: value
                for key, value in balanced.items()
                if key not in {
                    "column_thresholds_rad",
                    "resolved_confidence_threshold_by_mineral",
                    "resolved_consensus_threshold_by_mineral",
                }
            },
            "direct_gate_recall": {
                "group_sam_candidate": _fraction(candidate, positive),
                "absorption_depth": _fraction(direct_depth_pass, positive),
                "mineral_feature_gate": _fraction(feature_pass, positive),
                "target_is_best_mineral": _fraction(target_winner, positive),
                "accepted_before_spatial_cleanup": _fraction(before == output_class, positive),
                "final_balanced": _fraction(final == output_class, positive),
            },
            "sequential_rejection_reason_counts": {
                str(code): int(np.count_nonzero(positive & (rejection == code)))
                for code in range(11)
            },
            "sequential_rejection_reason_fraction": {
                str(code): _fraction(rejection == code, positive) for code in range(11)
            },
            "weak_positive_evidence_quantiles": {
                "target_raw_score": _quantiles(target_raw[positive]),
                "target_calibrated_score": _quantiles(target_calibrated[positive]),
                "best_calibrated_score": _quantiles(best_calibrated[positive]),
                "target_minus_best_score": _quantiles(
                    target_calibrated[positive & finite_competition]
                    - best_calibrated[positive & finite_competition]
                ),
                "group_sam_score": _quantiles(group_score[positive]),
                "balanced_column_threshold": _quantiles(column_threshold[positive]),
                "group_score_minus_column_threshold": _quantiles(
                    (group_score - column_threshold)[positive]
                ),
                "absorption_depth": _quantiles(depth[positive]),
                "classification_margin": _quantiles(margin[positive]),
                "confidence": _quantiles(confidence[positive]),
                "stability": _quantiles(stability[positive]),
                **{feature_id: _quantiles(values[positive]) for feature_id, values in features.items()},
            },
            "scene_material_evidence_quantiles": {
                "group_sam_score": _quantiles(group_score[material]),
            },
        }
        report["targets"][mineral_id] = target
    clay_ids = ("illite", "montmorillonite", "kaolinite")
    if all(mineral_id in prediction_cache for mineral_id in clay_ids):
        clay_stack = np.stack([prediction_cache[mineral_id] for mineral_id in clay_ids])
        predicted_union = np.any(clay_stack, axis=0)
        reference_union = np.isin(baseline_cache["clays"], (1, 2, 3))
        report["cross_group_clay_competition"] = {
            "predicted_union_pixels": int(np.count_nonzero(predicted_union)),
            "reference_union_pixels": int(np.count_nonzero(reference_union)),
            "pixels_with_multiple_clay_predictions": int(
                np.count_nonzero(np.sum(clay_stack, axis=0) > 1)
            ),
            "maximum_predictions_at_one_pixel": int(np.max(np.sum(clay_stack, axis=0))),
            "union_agreement": _agreement_metrics(predicted_union, reference_union),
            "pairwise_overlap_pixels": {
                f"{left}__{right}": int(
                    np.count_nonzero(prediction_cache[left] & prediction_cache[right])
                )
                for left_index, left in enumerate(clay_ids)
                for right in clay_ids[left_index + 1 :]
            },
        }
    return report


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.baseline_root, args.run_root)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
