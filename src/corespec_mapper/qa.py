from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
import json

import numpy as np

from .artifacts import repeated_segment_column_stripe_mask
from .envi import EnviDataset
from .v4_calibration import POLICY_ORDER


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _issue(code: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "details": details}


def _repeated_column_residual_metrics(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Audit removable narrow-column residuals separately for every class."""

    valid = np.asarray(valid_mask, dtype=bool)
    values = np.asarray(labels)
    classes: dict[str, Any] = {}
    maximum_repetition = 0
    repeated_column_count = 0
    dominant_corridor_count = 0
    residual_pixels = 0
    for class_id, class_name in enumerate(class_names[1:-1], start=1):
        candidate = values == class_id
        if not np.any(candidate):
            continue
        _, record = repeated_segment_column_stripe_mask(
            candidate,
            valid,
            {},
            policy="balanced",
        )
        class_metrics = {
            "maximum_segment_repetition": int(record["maximum_segment_repetition"]),
            "repeated_narrow_column_count": int(record["repeated_column_count"]),
            "dominant_repeated_corridor_count": int(record["dominant_corridor_count"]),
            "repeated_stripe_like_pixels": int(record["candidate_removed_pixels"]),
            "repeated_columns": [int(value) for value in record["repeated_columns"]],
        }
        classes[str(class_name)] = class_metrics
        maximum_repetition = max(
            maximum_repetition,
            class_metrics["maximum_segment_repetition"],
        )
        repeated_column_count += class_metrics["repeated_narrow_column_count"]
        dominant_corridor_count += class_metrics["dominant_repeated_corridor_count"]
        residual_pixels += class_metrics["repeated_stripe_like_pixels"]
    return {
        "maximum_segment_repetition": maximum_repetition,
        "repeated_narrow_column_count": repeated_column_count,
        "dominant_repeated_corridor_count": dominant_corridor_count,
        "repeated_stripe_like_pixels": residual_pixels,
        "classes": classes,
    }


def validate_v4_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    summary_path = run / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = _load_json(summary_path)
    software_version = str(summary.get("version", "CoreSpec Mapper V4.0.0"))
    expected_lines = int(summary["output_lines"][1] - summary["output_lines"][0])
    issues: list[dict[str, Any]] = []
    checks: dict[str, Any] = {"groups": {}}
    for group_id, group_summary in summary["groups"].items():
        group_dir = run / "groups" / group_id
        arrays: dict[str, np.ndarray] = {}
        names: list[str] = []
        masked_id = -1
        for policy in POLICY_ORDER:
            path = group_dir / f"final_{policy}.dat"
            try:
                dataset = EnviDataset(path)
                if dataset.info.lines != expected_lines or dataset.info.bands != 1:
                    issues.append(_issue("OUTPUT_DIMENSION_MISMATCH", "error", f"{group_id}/{policy} has unexpected dimensions"))
                current_names = [str(item) for item in dataset.info.metadata.get("class names", [])]
                if len(current_names) < 3 or current_names[0] != "Unclassified" or current_names[-1] != "Masked Pixels":
                    issues.append(_issue("CLASS_HEADER_INVALID", "error", f"{group_id}/{policy} classification header is incomplete"))
                arrays[policy] = np.array(dataset.read_rows(0, dataset.info.lines)[..., 0], copy=True)
                names = current_names
                masked_id = len(current_names) - 1
                dataset.close()
            except Exception as exc:
                issues.append(_issue("OUTPUT_REOPEN_FAILED", "error", f"Cannot reopen {path.name}: {exc}"))
        if len(arrays) != len(POLICY_ORDER):
            continue
        outside = arrays["balanced"] == masked_id
        for policy in POLICY_ORDER:
            if not np.array_equal(arrays[policy] == masked_id, outside):
                issues.append(_issue("MASK_INCONSISTENT", "error", f"{group_id} mask differs between policy outputs"))
        conservative = arrays["conservative"]
        balanced = arrays["balanced"]
        sensitive = arrays["sensitive"]
        if np.any((conservative > 0) & (conservative != masked_id) & (conservative != balanced)):
            issues.append(_issue("POLICY_NOT_NESTED", "error", f"{group_id} conservative output is not a subset of balanced"))
        if np.any((balanced > 0) & (balanced != masked_id) & (balanced != sensitive)):
            issues.append(_issue("POLICY_NOT_NESTED", "error", f"{group_id} balanced output is not a subset of sensitive"))
        confidence_dataset = EnviDataset(group_dir / "confidence.dat")
        confidence = np.array(confidence_dataset.read_rows(0, expected_lines)[..., 0], copy=True)
        confidence_dataset.close()
        if np.any(~np.isfinite(confidence)) or float(np.min(confidence)) < 0.0 or float(np.max(confidence)) > 1.0:
            issues.append(_issue("CONFIDENCE_RANGE_INVALID", "error", f"{group_id} confidence is outside [0, 1]"))
        valid = ~outside
        classified = (balanced > 0) & (balanced != masked_id)
        same_class_neighbour = np.zeros(classified.shape, dtype=bool)
        for row_shift, column_shift in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            shifted = np.roll(np.roll(balanced, row_shift, axis=0), column_shift, axis=1)
            neighbour = shifted == balanced
            if row_shift < 0:
                neighbour[row_shift:] = False
            elif row_shift > 0:
                neighbour[:row_shift] = False
            if column_shift < 0:
                neighbour[:, column_shift:] = False
            elif column_shift > 0:
                neighbour[:, :column_shift] = False
            same_class_neighbour |= neighbour
        classified_count = int(np.count_nonzero(classified))
        clustered_fraction = (
            float(np.count_nonzero(classified & same_class_neighbour) / classified_count)
            if classified_count
            else None
        )
        if classified_count >= 20 and clustered_fraction is not None and clustered_fraction < 0.25:
            issues.append(_issue(
                "SCATTERED_DETECTIONS",
                "warning",
                f"{group_id} balanced detections have weak same-class spatial support",
                clustered_fraction=clustered_fraction,
                pixels=classified_count,
            ))
        column_density = np.sum(classified, axis=0) / np.maximum(np.sum(valid, axis=0), 1)
        median_density = float(np.median(column_density))
        maximum_density = float(np.max(column_density))
        if maximum_density >= max(0.15, median_density + 0.10) and maximum_density >= 3.0 * max(median_density, 0.01):
            issues.append(_issue("FIXED_COLUMN_RESIDUAL", "warning", f"{group_id} retains a high-response detector column", maximum=maximum_density, median=median_density))
        repeated_column_metrics = _repeated_column_residual_metrics(
            balanced,
            valid,
            names,
        )
        if repeated_column_metrics["repeated_stripe_like_pixels"] > 0:
            issues.append(_issue(
                "REPEATED_NARROW_COLUMN_RESIDUAL",
                "warning",
                f"{group_id} retains removable narrow detector-column responses repeated across core sections",
                maximum_segment_repetition=repeated_column_metrics["maximum_segment_repetition"],
                repeated_narrow_column_count=repeated_column_metrics["repeated_narrow_column_count"],
                dominant_repeated_corridor_count=repeated_column_metrics["dominant_repeated_corridor_count"],
                repeated_stripe_like_pixels=repeated_column_metrics["repeated_stripe_like_pixels"],
                classes=repeated_column_metrics["classes"],
            ))
        zero_minerals = [name for class_id, name in enumerate(names[1:-1], start=1) if np.count_nonzero(balanced == class_id) == 0]
        if zero_minerals:
            issues.append(_issue("ZERO_DETECTION", "warning", f"{group_id} has no balanced detections for: {', '.join(zero_minerals)}"))
        checks["groups"][group_id] = {
            "dimensions": [expected_lines, int(balanced.shape[1])],
            "mask_pixels": int(np.count_nonzero(valid)),
            "balanced_classified_pixels": int(np.count_nonzero(classified)),
            "same_class_neighbour_fraction": clustered_fraction,
            "maximum_balanced_column_density": maximum_density,
            "median_balanced_column_density": median_density,
            "maximum_segment_repetition": repeated_column_metrics["maximum_segment_repetition"],
            "repeated_narrow_column_count": repeated_column_metrics["repeated_narrow_column_count"],
            "dominant_repeated_corridor_count": repeated_column_metrics["dominant_repeated_corridor_count"],
            "repeated_stripe_like_pixels": repeated_column_metrics["repeated_stripe_like_pixels"],
            "repeated_column_classes": repeated_column_metrics["classes"],
            "zero_detection_minerals": zero_minerals,
        }

    card_path = run / "capability" / "sensor_capability_card.json"
    if card_path.exists():
        card = _load_json(card_path)
        if card.get("fwhm_status") == "unknown":
            issues.append(_issue("FWHM_UNKNOWN", "warning", "Reference resampling used constrained interpolation because FWHM is unavailable"))
        for flag in card.get("quality_flags", []):
            if flag.get("severity") == "error":
                issues.append(_issue(flag.get("code", "CAPABILITY_ERROR"), "error", flag.get("message", "Capability error")))
    else:
        issues.append(_issue("CAPABILITY_CARD_MISSING", "error", "Sensor capability card is missing"))
    issues.append(_issue("NO_GROUND_TRUTH", "info", "This run measures evidence consistency and artifact control; it does not establish mineralogical accuracy without XRD, Raman, thin-section, or point-spectral validation."))
    error_count = sum(item["severity"] == "error" for item in issues)
    high_risk = sum(item["code"] in {
        "FIXED_COLUMN_RESIDUAL",
        "POLICY_NOT_NESTED",
        "REPEATED_NARROW_COLUMN_RESIDUAL",
    } for item in issues)
    warning_count = sum(item["severity"] == "warning" for item in issues)
    if error_count:
        grade, status = "D", "Blocked"
    elif high_risk:
        grade, status = "C", "Warning"
    elif warning_count:
        grade, status = "B", "Warning"
    else:
        grade, status = "A", "Completed"
    report = {
        "quality_grade": grade,
        "status": status,
        "publishable": grade in {"A", "B"},
        "checks": checks,
        "issues": issues,
    }
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {software_version} Quality Report",
        "",
        f"- Quality grade: **{grade}**",
        f"- Run status: **{status}**",
        f"- Automatic publish gate: **{'Passed' if report['publishable'] else 'Not passed'}**",
        "",
        "## Checks",
        "",
        "| Group | Balanced pixels | Same-class neighbour fraction | Maximum column density | Maximum segment repetition | Repeated narrow columns | Dominant repeated corridors | Residual stripe pixels | Zero-detection minerals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group_id, value in checks["groups"].items():
        clustered = value['same_class_neighbour_fraction']
        clustered_text = "—" if clustered is None else f"{clustered:.4f}"
        lines.append(f"| {group_id} | {value['balanced_classified_pixels']} | {clustered_text} | {value['maximum_balanced_column_density']:.4f} | {value['maximum_segment_repetition']} | {value['repeated_narrow_column_count']} | {value['dominant_repeated_corridor_count']} | {value['repeated_stripe_like_pixels']} | {', '.join(value['zero_detection_minerals']) or 'None'} |")
    lines.extend(["", "## Issues", ""])
    for item in issues:
        lines.append(f"- **{item['severity'].upper()} / {item['code']}**: {item['message']}")
    markdown = "\n".join(lines) + "\n"
    (reports / "quality_report.md").write_text(markdown, encoding="utf-8")
    issue_html = "".join(f"<li><strong>{escape(item['severity'].upper())} / {escape(item['code'])}</strong>: {escape(item['message'])}</li>" for item in issues)
    rows_html = "".join(
        f"<tr><td>{escape(group_id)}</td><td>{value['balanced_classified_pixels']}</td><td>{'—' if value['same_class_neighbour_fraction'] is None else format(value['same_class_neighbour_fraction'], '.4f')}</td><td>{value['maximum_balanced_column_density']:.4f}</td><td>{value['maximum_segment_repetition']}</td><td>{value['repeated_narrow_column_count']}</td><td>{value['dominant_repeated_corridor_count']}</td><td>{value['repeated_stripe_like_pixels']}</td><td>{escape(', '.join(value['zero_detection_minerals']) or 'None')}</td></tr>"
        for group_id, value in checks["groups"].items()
    )
    html = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>{escape(software_version)} Quality Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:32px auto;color:#202124}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cfd4da;padding:8px;text-align:left}}th{{background:#f3f5f7}}.grade{{font-size:28px;font-weight:700}}</style></head>
<body><h1>{escape(software_version)} Quality Report</h1><p class=\"grade\">Grade {escape(grade)} / {escape(status)}</p><h2>Checks</h2><table><thead><tr><th>Group</th><th>Balanced pixels</th><th>Same-class neighbour fraction</th><th>Maximum column density</th><th>Maximum segment repetition</th><th>Repeated narrow columns</th><th>Dominant repeated corridors</th><th>Residual stripe pixels</th><th>Zero detections</th></tr></thead><tbody>{rows_html}</tbody></table><h2>Issues</h2><ul>{issue_html}</ul></body></html>"""
    (reports / "quality_report.html").write_text(html, encoding="utf-8")
    return report
