from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
import json

import numpy as np

from .envi import EnviDataset
from .v4_calibration import POLICY_ORDER


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _issue(code: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "details": details}


def validate_v4_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    summary_path = run / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = _load_json(summary_path)
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
        column_density = np.sum(classified, axis=0) / np.maximum(np.sum(valid, axis=0), 1)
        median_density = float(np.median(column_density))
        maximum_density = float(np.max(column_density))
        if maximum_density >= max(0.15, median_density + 0.10) and maximum_density >= 3.0 * max(median_density, 0.01):
            issues.append(_issue("FIXED_COLUMN_RESIDUAL", "warning", f"{group_id} retains a high-response detector column", maximum=maximum_density, median=median_density))
        zero_minerals = [name for class_id, name in enumerate(names[1:-1], start=1) if np.count_nonzero(balanced == class_id) == 0]
        if zero_minerals:
            issues.append(_issue("ZERO_DETECTION", "warning", f"{group_id} has no balanced detections for: {', '.join(zero_minerals)}"))
        checks["groups"][group_id] = {
            "dimensions": [expected_lines, int(balanced.shape[1])],
            "mask_pixels": int(np.count_nonzero(valid)),
            "balanced_classified_pixels": int(np.count_nonzero(classified)),
            "maximum_balanced_column_density": maximum_density,
            "median_balanced_column_density": median_density,
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
    high_risk = sum(item["code"] in {"FIXED_COLUMN_RESIDUAL", "POLICY_NOT_NESTED"} for item in issues)
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
        "# CoreSpec Mapper V4 Quality Report",
        "",
        f"- Quality grade: **{grade}**",
        f"- Run status: **{status}**",
        f"- Automatic publish gate: **{'Passed' if report['publishable'] else 'Not passed'}**",
        "",
        "## Checks",
        "",
        "| Group | Balanced pixels | Maximum column density | Zero-detection minerals |",
        "|---|---:|---:|---|",
    ]
    for group_id, value in checks["groups"].items():
        lines.append(f"| {group_id} | {value['balanced_classified_pixels']} | {value['maximum_balanced_column_density']:.4f} | {', '.join(value['zero_detection_minerals']) or 'None'} |")
    lines.extend(["", "## Issues", ""])
    for item in issues:
        lines.append(f"- **{item['severity'].upper()} / {item['code']}**: {item['message']}")
    markdown = "\n".join(lines) + "\n"
    (reports / "quality_report.md").write_text(markdown, encoding="utf-8")
    issue_html = "".join(f"<li><strong>{escape(item['severity'].upper())} / {escape(item['code'])}</strong>: {escape(item['message'])}</li>" for item in issues)
    rows_html = "".join(
        f"<tr><td>{escape(group_id)}</td><td>{value['balanced_classified_pixels']}</td><td>{value['maximum_balanced_column_density']:.4f}</td><td>{escape(', '.join(value['zero_detection_minerals']) or 'None')}</td></tr>"
        for group_id, value in checks["groups"].items()
    )
    html = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>CoreSpec Mapper V4 Quality Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:32px auto;color:#202124}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cfd4da;padding:8px;text-align:left}}th{{background:#f3f5f7}}.grade{{font-size:28px;font-weight:700}}</style></head>
<body><h1>CoreSpec Mapper V4 Quality Report</h1><p class=\"grade\">Grade {escape(grade)} / {escape(status)}</p><h2>Checks</h2><table><thead><tr><th>Group</th><th>Balanced pixels</th><th>Maximum column density</th><th>Zero detections</th></tr></thead><tbody>{rows_html}</tbody></table><h2>Issues</h2><ul>{issue_html}</ul></body></html>"""
    (reports / "quality_report.html").write_text(html, encoding="utf-8")
    return report
