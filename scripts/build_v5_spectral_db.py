from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corespec_mapper.spectral_db import build_v5_database  # noqa: E402


def main() -> int:
    parser = ArgumentParser(description="Build the versioned CoreSpec Mapper V5 private spectral database")
    parser.add_argument("--library-root", type=Path, default=ROOT / "spec_lib")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "corespec_mapper" / "resources" / "corespec_spectral_v5.sqlite3",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "src" / "corespec_mapper" / "resources" / "mineral_catalog_v5.json",
    )
    args = parser.parse_args()
    summary = build_v5_database(args.library_root, args.output, catalog_path=args.catalog)
    print(json.dumps({"output": str(args.output.resolve()), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
