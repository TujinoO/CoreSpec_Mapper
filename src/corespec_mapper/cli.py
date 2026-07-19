from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import MineralCatalog
from .pipeline import audit_project, diagnose_sam_windows, diagnose_sg, load_config, run_pilot, run_sam_baseline, save_json
from .v3_pipeline import run_v3
from .v4_service import audit_v4_project, run_v4_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corespec", description="CoreSpec Mapper backend CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate ENVI images, mask cube, and spectral libraries")
    audit.add_argument("--config", required=True)
    audit.add_argument("--output", required=True)

    baseline = subparsers.add_parser("sam-baseline", help="Run ENVI-comparable full-spectrum SAM")
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--group", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--start-line", type=int, default=0)
    baseline.add_argument("--stop-line", type=int)

    diagnose = subparsers.add_parser("sam-diagnose", help="Compare candidate spectral windows against ENVI SAM rules")
    diagnose.add_argument("--config", required=True)
    diagnose.add_argument("--group", required=True)
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument("--start-line", type=int, default=0)
    diagnose.add_argument("--stop-line", type=int)

    sg_diagnose = subparsers.add_parser("sg-diagnose", help="Compare candidate SG parameters against an existing SG cube")
    sg_diagnose.add_argument("--config", required=True)
    sg_diagnose.add_argument("--output", required=True)
    sg_diagnose.add_argument("--start-line", type=int, default=0)
    sg_diagnose.add_argument("--stop-line", type=int)

    pilot = subparsers.add_parser("pilot", help="Run SG + continuum removal + SAM + SFF pilot workflow")
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--start-line", type=int, default=0)
    pilot.add_argument("--stop-line", type=int)

    v3 = subparsers.add_parser("v3-pilot", help="Run V3 Automatic Library Ensemble with full-depth column calibration")
    v3.add_argument("--config", required=True)
    v3.add_argument("--output", required=True)
    v3.add_argument("--start-line", type=int, default=0)
    v3.add_argument("--stop-line", type=int)

    v4_audit = subparsers.add_parser("v4-audit", help="Build a V4 capability card and audit automatic references")
    v4_audit.add_argument("--config", required=True)
    v4_audit.add_argument("--output", required=True)
    v4_audit.add_argument("--skip-library", action="store_true", help="Skip the slower spectral-library scan")

    v4_run = subparsers.add_parser("v4-run", help="Run the V4 Adaptive Mineral Evidence Engine")
    v4_run.add_argument("--config", required=True)
    v4_run.add_argument("--output-root", required=True)
    v4_run.add_argument("--run-id")
    v4_run.add_argument("--start-line", type=int, default=0)
    v4_run.add_argument("--stop-line", type=int)

    catalog = subparsers.add_parser("v4-catalog", help="Print the V4 mineral and expert catalog as JSON")
    catalog.add_argument("--output")

    subparsers.add_parser("desktop", help="Launch the CoreSpec Mapper V4 desktop application")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        config = load_config(args.config)
        result = audit_project(config)
        save_json(result, args.output)
    elif args.command == "sam-baseline":
        config = load_config(args.config)
        run_sam_baseline(config, args.group, args.output, args.start_line, args.stop_line)
    elif args.command == "sam-diagnose":
        config = load_config(args.config)
        diagnose_sam_windows(config, args.group, args.output, args.start_line, args.stop_line)
    elif args.command == "sg-diagnose":
        config = load_config(args.config)
        diagnose_sg(config, args.output, args.start_line, args.stop_line)
    elif args.command == "pilot":
        config = load_config(args.config)
        run_pilot(config, args.output, args.start_line, args.stop_line)
    elif args.command == "v3-pilot":
        config = load_config(args.config)
        run_v3(config, args.output, args.start_line, args.stop_line)
    elif args.command == "v4-audit":
        config = load_config(args.config)
        save_json(audit_v4_project(config, scan_library=not args.skip_library), args.output)
    elif args.command == "v4-run":
        config = load_config(args.config)
        result = run_v4_project(
            config,
            args.output_root,
            run_id=args.run_id,
            start_line=args.start_line,
            stop_line=args.stop_line,
        )
        print(result["run_directory"])
    elif args.command == "v4-catalog":
        result = MineralCatalog.load().to_summary()
        if args.output:
            save_json(result, args.output)
        else:
            import json

            print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "desktop":
        from .desktop import launch_desktop

        return launch_desktop()
    else:
        raise AssertionError(args.command)
    return 0
