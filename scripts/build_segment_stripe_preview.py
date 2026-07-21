from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from corespec_mapper.catalog import MineralCatalog, default_v5_runtime_catalog_path
from corespec_mapper.envi import EnviDataset
from corespec_mapper.preview import _overlay, write_png
from corespec_mapper.qa import _repeated_column_residual_metrics
from corespec_mapper.v4_calibration import POLICY_ORDER
from corespec_mapper.v4_pipeline import _apply_post_nesting_repeated_column_filter


def _read_classification(path: Path) -> tuple[np.ndarray, list[str]]:
    dataset = EnviDataset(path)
    try:
        labels = np.array(
            dataset.read_rows(0, dataset.info.lines)[..., 0],
            copy=True,
        )
        names = [str(value) for value in dataset.info.metadata.get("class names", [])]
        return labels, names
    finally:
        dataset.close()


def build_preview(run_dir: Path) -> Path:
    run = run_dir.resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    group_ids = list(summary["groups"])
    background = np.asarray(
        Image.open(run / "previews" / "swir_background_1600nm.png").convert("RGB")
    )
    catalog = MineralCatalog.load(default_v5_runtime_catalog_path())
    output = run / "previews" / "segment_stripe_terminal_audit"
    output.mkdir(parents=True, exist_ok=True)

    valid: np.ndarray | None = None
    profile_panels: dict[str, list[np.ndarray]] = {
        policy: [background] for policy in POLICY_ORDER
    }
    stripe_panels = [background]
    audit: dict[str, object] = {"groups": {}}
    for group_id in group_ids:
        group_dir = run / "groups" / group_id
        profiles: dict[str, np.ndarray] = {}
        names: list[str] = []
        for policy in POLICY_ORDER:
            profiles[policy], names = _read_classification(
                group_dir / f"final_{policy}.dat"
            )
        masked_id = len(names) - 1
        group_valid = profiles["balanced"] != masked_id
        if valid is None:
            valid = group_valid
        elif not np.array_equal(valid, group_valid):
            raise ValueError(f"Material mask differs for group {group_id}")

        before_counts = {
            policy: {
                names[class_id]: int(np.count_nonzero(profiles[policy] == class_id))
                for class_id in range(1, len(names) - 1)
            }
            for policy in POLICY_ORDER
        }
        applied, records = _apply_post_nesting_repeated_column_filter(
            profiles,
            group_valid,
            catalog.group(group_id).spatial_cleanup,
        )
        after_counts = {
            policy: {
                names[class_id]: int(np.count_nonzero(profiles[policy] == class_id))
                for class_id in range(1, len(names) - 1)
            }
            for policy in POLICY_ORDER
        }
        residual = {
            policy: _repeated_column_residual_metrics(
                profiles[policy],
                group_valid,
                names,
            )
            for policy in POLICY_ORDER
        }
        audit["groups"][group_id] = {
            "before_counts": before_counts,
            "after_counts": after_counts,
            "terminal_filter": records,
            "residual": residual,
        }

        group_stripes = np.logical_or.reduce(tuple(applied.values()))
        stripe_overlay = background.astype(np.float64)
        stripe_overlay[group_stripes] = (
            0.15 * stripe_overlay[group_stripes]
            + 0.85 * np.array([255.0, 0.0, 255.0])
        )
        stripe_panels.append(np.clip(stripe_overlay, 0, 255).astype(np.uint8))
        for policy in POLICY_ORDER:
            overlay = _overlay(background, profiles[policy], names)
            profile_panels[policy].append(overlay)
            write_png(output / f"group_{group_id}_{policy}_after.png", overlay)

    if valid is None:
        raise ValueError("Run contains no groups")
    for policy in POLICY_ORDER:
        write_png(
            output / f"comparison_{policy}_after.png",
            np.concatenate(profile_panels[policy], axis=1),
        )
    write_png(
        output / "removed_repeated_columns.png",
        np.concatenate(stripe_panels, axis=1),
    )
    audit_path = output / "offline_segment_stripe_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview the post-nesting repeated-column terminal filter"
    )
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(build_preview(args.run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
