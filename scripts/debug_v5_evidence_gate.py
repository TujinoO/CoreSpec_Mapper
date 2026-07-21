from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corespec_mapper.envi import EnviDataset  # noqa: E402
from corespec_mapper.v5_project_calibration import (  # noqa: E402
    DEFAULT_WEAK_LABEL_SOURCES,
    learn_evidence_gate,
)


def read(path: Path) -> np.ndarray:
    with EnviDataset(path) as dataset:
        return np.moveaxis(dataset.read_rows(0, dataset.info.lines), 2, 0).copy()


def main() -> int:
    run = Path(sys.argv[1])
    baseline = Path(sys.argv[2])
    group_id = sys.argv[3]
    group = run / "groups" / group_id
    print("loading", group, flush=True)
    raw = read(group / "raw_mineral_scores.dat")
    calibrated = read(group / "calibrated_mineral_scores.dat")
    score = read(group / "group_sam_score.dat")[0]
    depth = read(group / "absorption_depth.dat")[0]
    margin = read(group / "classification_margin.dat")[0]
    confidence = read(group / "confidence.dat")[0]
    stability = read(group / "stability.dat")[0]
    candidate = read(group / "column_candidates.dat")[0] == 1
    feature_paths = sorted(group.glob("feature_*.dat"))
    features = [read(path)[0] for path in feature_paths]
    print("loaded", raw.shape, len(features), flush=True)
    relative, class_map = DEFAULT_WEAK_LABEL_SOURCES[group_id]
    labels = read(baseline / relative)[0]
    winner = np.argmin(np.where(np.isfinite(calibrated), calibrated, np.inf), axis=0)
    rows = np.arange(score.shape[0])[:, None]
    holdout = np.broadcast_to((rows // 256) % 5 == 0, score.shape)
    training = ~holdout
    for mineral_id, class_id in class_map.items():
        print("fitting", mineral_id, flush=True)
        gate, record = learn_evidence_gate(
            [score, depth, *raw, *features],
            labels == class_id,
            candidate,
            training,
            holdout,
            minimum_pixels=100,
            weak_negative=np.isin(
                labels,
                [other_class for other_id, other_class in class_map.items() if other_id != mineral_id],
            ),
            selection_mask=winner == list(class_map).index(mineral_id),
        )
        print(mineral_id, record, "gate_pixels", int(np.count_nonzero(gate & candidate)))
        positive = labels == class_id
        negative = np.isin(
            labels,
            [other_class for other_id, other_class in class_map.items() if other_id != mineral_id],
        )
        selection = candidate & (winner == list(class_map).index(mineral_id))
        axis_planes = [score, depth, *raw, *features]
        axis_names = [
            "group_score",
            "depth",
            *[f"raw_{index}" for index in range(raw.shape[0])],
            *[path.stem for path in feature_paths],
        ]
        axis_results = []
        for name, plane in zip(axis_names, axis_planes):
            positive_values = plane[positive & training & np.isfinite(plane)]
            negative_values = plane[negative & training & np.isfinite(plane)]
            if positive_values.size < 100 or negative_values.size < 100:
                continue
            thresholds = np.unique(
                np.quantile(
                    np.concatenate((positive_values, negative_values)),
                    np.linspace(0.02, 0.98, 97),
                )
            )
            for direction in (-1, 1):
                best = None
                for threshold in thresholds:
                    positive_pass = direction * positive_values >= direction * threshold
                    negative_pass = direction * negative_values >= direction * threshold
                    recall = float(np.mean(positive_pass))
                    false_positive_rate = float(np.mean(negative_pass))
                    utility = recall - false_positive_rate
                    current = (utility, recall, -false_positive_rate, float(threshold), direction)
                    if best is None or current > best:
                        best = current
                _, recall, negative_fpr, threshold, resolved_direction = best
                selected = selection & np.isfinite(plane) & (
                    resolved_direction * plane >= resolved_direction * threshold
                )
                maximum = int(round(1.5 * np.count_nonzero(positive)))
                if np.count_nonzero(selected) > maximum:
                    indices = np.flatnonzero(selected)
                    strength = (resolved_direction * plane.reshape(-1))[indices]
                    keep = indices[np.argpartition(strength, -maximum)[-maximum:]]
                    selected[:] = False
                    selected.reshape(-1)[keep] = True
                axis_results.append(
                    (
                        int(np.count_nonzero(selected & positive)),
                        recall,
                        negative_fpr,
                        name,
                        resolved_direction,
                        threshold,
                        int(np.count_nonzero(selected)),
                    )
                )
        print("axis", sorted(axis_results, reverse=True)[:8], flush=True)
        if mineral_id == "anhydrite":
            feature_by_name = {path.stem: plane for path, plane in zip(feature_paths, features)}
            ratio_1750 = feature_by_name["feature_sulfate_1750_ratio"]
            ratio_1940 = feature_by_name["feature_sulfate_1940_ratio"]
            pair_results = []
            finite_pair = positive & training & np.isfinite(ratio_1750) & np.isfinite(ratio_1940)
            for upper_1750 in np.quantile(ratio_1750[finite_pair], np.linspace(0.4, 0.95, 12)):
                for upper_1940 in np.quantile(ratio_1940[finite_pair], np.linspace(0.4, 0.95, 12)):
                    rule = (
                        selection
                        & np.isfinite(ratio_1750)
                        & np.isfinite(ratio_1940)
                        & (ratio_1750 <= upper_1750)
                        & (ratio_1940 <= upper_1940)
                    )
                    positive_pair = positive & training & np.isfinite(ratio_1750) & np.isfinite(ratio_1940)
                    negative_pair = negative & training & np.isfinite(ratio_1750) & np.isfinite(ratio_1940)
                    train_recall = np.mean(
                        (ratio_1750 <= upper_1750)[positive_pair]
                        & (ratio_1940 <= upper_1940)[positive_pair]
                    )
                    train_fpr = np.mean(
                        (ratio_1750 <= upper_1750)[negative_pair]
                        & (ratio_1940 <= upper_1940)[negative_pair]
                    )
                    pair_results.append(
                        (
                            float(train_recall - train_fpr),
                            int(np.count_nonzero(rule & positive)),
                            int(np.count_nonzero(rule)),
                            float(train_recall),
                            float(train_fpr),
                            float(upper_1750),
                            float(upper_1940),
                        )
                    )
            print("pair", sorted(pair_results, reverse=True)[:12], flush=True)
            reconstructed = (
                selection
                & np.isfinite(ratio_1940)
                & (ratio_1940 <= 0.3345872735977173)
            )
            indices = np.flatnonzero(reconstructed)
            if indices.size > 634:
                keep = indices[np.argpartition(ratio_1940.reshape(-1)[indices], 634)[:634]]
                reconstructed[:] = False
                reconstructed.reshape(-1)[keep] = True
            stages = {
                "axis_gate": reconstructed,
                "depth": reconstructed & (depth >= 0.012),
                "margin": reconstructed & (depth >= 0.012) & (margin >= 0.008),
                "stability": reconstructed & (depth >= 0.012) & (margin >= 0.008) & (stability >= 0.280852347612381),
                "confidence": reconstructed & (depth >= 0.012) & (margin >= 0.008) & (stability >= 0.280852347612381) & (confidence >= 0.3501436710357666),
            }
            print(
                "reconstructed",
                {name: (int(np.count_nonzero(mask)), int(np.count_nonzero(mask & positive))) for name, mask in stages.items()},
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
