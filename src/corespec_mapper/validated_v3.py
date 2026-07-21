from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping
import json

from .pipeline import load_config
from .preview import make_v3_previews
from .v3_pipeline import run_v3
from .v4_models import CancellationToken
from .v5_validation import RegressionGate, compare_classification_rasters, read_nonzero_mask


ValidatedProgress = Callable[[str, float, str], None]


def _validation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    advanced = config.get("advanced", {}) if isinstance(config.get("advanced"), Mapping) else {}
    value = advanced.get("validation", {})
    if not isinstance(value, Mapping):
        raise ValueError("advanced.validation must be a JSON object")
    result = dict(value)
    profiles = result.get("v3_profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("validated_v3 mode requires advanced.validation.v3_profiles")
    for policy, path in profiles.items():
        if policy not in {"conservative", "balanced", "sensitive"}:
            raise ValueError(f"Unsupported validated V3 policy: {policy}")
        if not path:
            raise ValueError(f"Validated V3 profile {policy} has no configuration path")
    return result


def _classification_gate(value: Mapping[str, Any] | None) -> RegressionGate:
    defaults = {
        "minimum_iou": 0.90,
        "minimum_precision": 0.90,
        "minimum_recall": 0.90,
        "maximum_area_difference_fraction": 0.10,
    }
    defaults.update(dict(value or {}))
    return RegressionGate.from_mapping(defaults)


def _profile_reference_path(settings: Mapping[str, Any], policy: str, group: str) -> Path:
    baseline_root = Path(str(settings["baseline_root"]))
    subdirs = settings.get("baseline_profile_subdirs", {})
    if not isinstance(subdirs, Mapping) or policy not in subdirs:
        raise ValueError(f"No baseline profile directory is configured for {policy}")
    return baseline_root / str(subdirs[policy]) / group / "v3_final_classes.dat"


def _rejection_waterfalls(profile_summaries: Mapping[str, Any]) -> dict[str, Any]:
    waterfalls: dict[str, Any] = {}
    for policy, profile in profile_summaries.items():
        groups: dict[str, Any] = {}
        material_pixels = int(profile.get("mask_pixels", 0))
        for group_id, group in profile.get("groups", {}).items():
            counts = group.get("counts", {})
            candidates = int(counts.get("column_calibrated_candidates", 0))
            classified = int(counts.get("classified_before_spatial", 0))
            final = int(counts.get("classified_final", 0))
            groups[group_id] = {
                "material_pixels": material_pixels,
                "sam_or_column_candidates": candidates,
                "rejected_by_spectral_depth_or_class_evidence": max(0, candidates - classified),
                "classified_before_spatial": classified,
                "directional_stripe_removed": int(counts.get("directional_stripe_removed", 0)),
                "elongated_component_removed": int(counts.get("elongated_component_removed", 0)),
                "small_component_removed": int(counts.get("small_component_removed", 0)),
                "other_spatial_or_overlap_removed": max(
                    0,
                    classified
                    - final
                    - int(counts.get("directional_stripe_removed", 0))
                    - int(counts.get("elongated_component_removed", 0))
                    - int(counts.get("small_component_removed", 0)),
                ),
                "classified_final": final,
                "mineral_counts_final": dict(group.get("mineral_counts_final", {})),
            }
        waterfalls[policy] = groups
    return waterfalls


def run_validated_v3_profiles(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    start_line: int = 0,
    stop_line: int | None = None,
    progress: ValidatedProgress | None = None,
    cancel_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """Run project-locked V3 recipes and enforce weak-label regression gates."""
    settings = _validation_settings(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    token = cancel_token or CancellationToken()
    callback = progress or (lambda _stage, _fraction, _message: None)
    configured_profiles = dict(settings["v3_profiles"])
    profile_count = len(configured_profiles)
    summaries: dict[str, Any] = {}
    previews: dict[str, str] = {}
    profile_configs: dict[str, dict[str, Any]] = {}

    for profile_index, (policy, config_path) in enumerate(configured_profiles.items()):
        token.raise_if_cancelled()
        profile_config = load_config(config_path)
        profile_configs[policy] = profile_config
        profile_output = output / "validated_profiles" / policy

        def profile_progress(stage: str, fraction: float, message: str) -> None:
            token.raise_if_cancelled()
            callback(
                f"validated_{policy}_{stage}",
                (profile_index + float(fraction)) / profile_count,
                f"{policy}: {message}",
            )

        summary = run_v3(
            profile_config,
            profile_output,
            start_line,
            stop_line,
            progress=profile_progress,
        )
        summaries[policy] = summary
        start, stop = (int(item) for item in summary["output_lines"])
        for key, path in make_v3_previews(profile_config, profile_output, start, stop).items():
            previews[f"{policy}_{key}"] = path

    reference_mask_path = Path(str(settings["baseline_mask"]))
    reference_mask = read_nonzero_mask(reference_mask_path)
    gate = _classification_gate(settings.get("classification_gate"))
    group_minerals = settings.get("group_minerals", {})
    if not isinstance(group_minerals, Mapping):
        raise ValueError("advanced.validation.group_minerals must be a JSON object")
    failures: list[str] = []
    profile_reports: dict[str, Any] = {}
    for policy, summary in summaries.items():
        group_reports: dict[str, Any] = {}
        profile_output = output / "validated_profiles" / policy
        for group, minerals in group_minerals.items():
            comparisons = compare_classification_rasters(
                _profile_reference_path(settings, policy, str(group)),
                profile_output / str(group) / "v3_final_classes.dat",
                tuple(str(item) for item in minerals),
                domain=reference_mask,
            )
            mineral_records: dict[str, Any] = {}
            for mineral, agreement in comparisons.items():
                current_failures = gate.failures(agreement)
                mineral_records[mineral] = {
                    "passed": not current_failures,
                    "agreement": agreement.to_dict(),
                    "failures": list(current_failures),
                }
                failures.extend(f"{policy}/{group}/{mineral}: {item}" for item in current_failures)
            group_reports[str(group)] = {"minerals": mineral_records}
        profile_reports[policy] = {"groups": group_reports}

    report = {
        "format": "CoreSpec Mapper V5.3 Validated Recipe Report",
        "format_version": 1,
        "passed": not failures,
        "evidence_level": "historical_workflow_weak_label",
        "evidence_notice": (
            "The frozen classifications verify workflow regression only and are not independent XRD/Raman truth."
        ),
        "baseline_mask": str(reference_mask_path),
        "classification_gate": gate.to_dict(),
        "profiles": profile_reports,
        "failures": failures,
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    thresholds = {
        policy: {
            group: summary["thresholds"]
            for group, summary in profile_summary.get("groups", {}).items()
        }
        for policy, profile_summary in summaries.items()
    }
    rejection_waterfall = _rejection_waterfalls(summaries)
    (output / "rejection_waterfall.json").write_text(
        json.dumps(rejection_waterfall, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "engine": "validated_v3",
        "passed": report["passed"],
        "profiles": summaries,
        "validation": report,
        "previews": previews,
        "thresholds": thresholds,
        "rejection_waterfall": rejection_waterfall,
    }


__all__ = ["run_validated_v3_profiles"]
