from __future__ import annotations

"""Build the disabled mineral-name reserve from uncatalogued pure references."""

from argparse import ArgumentParser
from datetime import date
from pathlib import Path
import json
import sqlite3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "src" / "corespec_mapper" / "resources" / "corespec_spectral_v5.sqlite3"
DEFAULT_OUTPUT = ROOT / "src" / "corespec_mapper" / "resources" / "mineral_recognition_reserve_v5.json"


def build(database_path: Path) -> dict:
    connection = sqlite3.connect(f"file:{database_path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        release = connection.execute("SELECT value FROM metadata WHERE key='database_release'").fetchone()[0]
        rows = connection.execute(
            """
            SELECT
                m.primary_phase_id AS phase_id,
                COUNT(*) AS measurement_count,
                COUNT(DISTINCT m.sample_id) AS independent_sample_count,
                COUNT(DISTINCT m.source_id) AS source_count,
                GROUP_CONCAT(DISTINCT m.source_id) AS source_ids
            FROM measurement AS m
            JOIN source AS s ON s.source_id = m.source_id
            WHERE s.source_role = 'mineral_reference'
              AND m.purity_status = 'declared_pure'
              AND m.catalogued = 0
              AND m.primary_phase_id IS NOT NULL
            GROUP BY m.primary_phase_id
            ORDER BY measurement_count DESC, phase_id
            """
        ).fetchall()
    finally:
        connection.close()

    entries = [
        {
            "phase_id": str(row["phase_id"]),
            "measurement_count": int(row["measurement_count"]),
            "independent_sample_count": int(row["independent_sample_count"]),
            "source_count": int(row["source_count"]),
            "source_ids": sorted(str(row["source_ids"]).split(",")),
            "status": "reserved_expert_required",
            "recognition_enabled": False,
            "required_expert_definition": True,
            "review_note": "Automatically parsed primary label; mineral identity, aliases, spectral windows, confusers, gates, and truth validation are still required.",
        }
        for row in rows
    ]
    return {
        "schema_version": 1,
        "database_release": str(release),
        "generated_on": date.today().isoformat(),
        "status": "disabled_reserve_only",
        "purpose": "Complete name-level reserve for future expert-reviewed recognition targets; entries are intentionally unavailable to the runtime and desktop selector.",
        "activation_contract": {
            "required_fields": [
                "expert_id",
                "taxonomy_group",
                "spectral_family",
                "primary_aliases",
                "required_windows_nm",
                "key_features_nm",
                "confusers",
                "feature_gate",
                "validation_record",
            ],
            "required_actions": [
                "expert_review",
                "catalog_and_runtime_definition",
                "reference_qc",
                "confuser_competition_test",
                "sensor_coverage_test",
                "ground_truth_or_geological_acceptance",
            ],
        },
        "summary": {
            "reserved_phase_count": len(entries),
            "reserved_measurement_count": sum(item["measurement_count"] for item in entries),
            "reserved_independent_sample_count_sum": sum(item["independent_sample_count"] for item in entries),
        },
        "entries": entries,
    }


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build(args.database)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **value["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
