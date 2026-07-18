from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import audit_project, diagnose_sam_windows, diagnose_sg, load_config, run_pilot, run_sam_baseline, save_json
from .v3_pipeline import run_v3


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "audit":
        result = audit_project(config)
        save_json(result, args.output)
    elif args.command == "sam-baseline":
        run_sam_baseline(config, args.group, args.output, args.start_line, args.stop_line)
    elif args.command == "sam-diagnose":
        diagnose_sam_windows(config, args.group, args.output, args.start_line, args.stop_line)
    elif args.command == "sg-diagnose":
        diagnose_sg(config, args.output, args.start_line, args.stop_line)
    elif args.command == "pilot":
        run_pilot(config, args.output, args.start_line, args.stop_line)
    elif args.command == "v3-pilot":
        run_v3(config, args.output, args.start_line, args.stop_line)
    else:
        raise AssertionError(args.command)
    return 0
