from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GeoCoreFusion coarse registration on an RGB/VNIR/SWIR scan."
    )
    parser.add_argument("--geocorefusion-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--preview-max-height", type=int, default=4096)
    parser.add_argument("--refine-roi", action="store_true")
    parser.add_argument("--roi-x", type=int)
    parser.add_argument("--roi-y", type=int)
    parser.add_argument(
        "--roi-y-list",
        help="Comma-separated manual RGB start rows; coarse registration is estimated only once",
    )
    parser.add_argument("--roi-width", type=int, default=4096)
    parser.add_argument("--roi-height", type=int, default=8192)
    return parser.parse_args()


def _find_header(directory: Path, prefixes: tuple[str, ...]) -> Path:
    for path in sorted(directory.glob("*.hdr")):
        upper = path.name.upper()
        if any(upper.startswith(prefix) for prefix in prefixes):
            return path
    raise FileNotFoundError(f"No ENVI header with prefixes {prefixes!r} under {directory}")


def _preview_uint8(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    lower, upper = np.percentile(values[finite], [2.0, 98.0])
    if upper <= lower:
        upper = lower + 1e-6
    scaled = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32) * 255.0


def _save_overlay(path: Path, reference: np.ndarray, moving: np.ndarray) -> None:
    ref = _preview_uint8(reference).round().astype(np.uint8)
    mov = _preview_uint8(moving).round().astype(np.uint8)
    blue = ((ref.astype(np.uint16) + mov.astype(np.uint16)) // 2).astype(np.uint8)
    Image.fromarray(np.stack([ref, mov, blue], axis=2), mode="RGB").save(path)


def _save_checkerboard(path: Path, reference: np.ndarray, moving: np.ndarray, block: int = 24) -> None:
    ref = _preview_uint8(reference).round().astype(np.uint8)
    mov = _preview_uint8(moving).round().astype(np.uint8)
    yy, xx = np.indices(ref.shape)
    select = ((yy // block) + (xx // block)) % 2 == 0
    Image.fromarray(np.where(select, ref, mov)).save(path)


def main() -> int:
    args = _parse_args()
    package_root = (args.geocorefusion_root / "src").resolve()
    sys.path.insert(0, str(package_root))

    from geocorefusion.config import RegistrationConfig, RoiConfig
    from geocorefusion.dataset import DatasetTriplet, SensorData
    from geocorefusion.envi import open_cube, parse_header
    from geocorefusion.registration import estimate_registration, estimate_roi_registration
    from geocorefusion.roi import choose_roi

    def sensor(name: str, prefixes: tuple[str, ...]) -> SensorData:
        header = _find_header(args.data_dir, prefixes)
        metadata = parse_header(header)
        cube, _ = open_cube(metadata)
        return SensorData(name=name, meta=metadata, cube=cube)

    started = monotonic()
    dataset = DatasetTriplet(
        root=args.data_dir,
        rgb=sensor("RGB", ("RGB-",)),
        nir=sensor("VNIR", ("VNIR-", "NIR-")),
        swir=sensor("SWIR", ("SWIR-",)),
    )
    config = RegistrationConfig(
        preview_width=args.preview_width,
        preview_max_height=args.preview_max_height,
        motion="auto_physical",
        enable_strip_drift=False,
        enable_roi_refinement=args.refine_roi,
    )
    registration = estimate_registration(dataset, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_preview_uint8(registration.preview_rgb).round().astype(np.uint8)).save(
        args.output_dir / "rgb_structure.png"
    )
    _save_overlay(
        args.output_dir / "rgb_vnir_coarse_overlay.png",
        registration.preview_rgb,
        registration.preview_nir_aligned,
    )
    _save_overlay(
        args.output_dir / "rgb_swir_coarse_overlay.png",
        registration.preview_rgb,
        registration.preview_swir_aligned,
    )
    roi_report = None
    roi_reports = None
    if args.refine_roi:
        if args.roi_y_list and args.roi_y is not None:
            raise ValueError("Use either --roi-y or --roi-y-list, not both")
        requested_rows = None
        if args.roi_y_list:
            requested_rows = [
                int(value.strip()) for value in args.roi_y_list.split(",") if value.strip()
            ]
            if not requested_rows:
                raise ValueError("--roi-y-list did not contain any rows")
        manual = requested_rows is not None or args.roi_x is not None or args.roi_y is not None
        if manual and args.roi_x is None:
            raise ValueError("--roi-x is required for manual ROI refinement")
        if manual and requested_rows is None and args.roi_y is None:
            raise ValueError("--roi-y or --roi-y-list is required for manual ROI refinement")
        roi_rows = requested_rows if requested_rows is not None else [args.roi_y]
        if not manual:
            roi_rows = [None]

        def scales(matrix: np.ndarray) -> tuple[float, float]:
            return float(np.linalg.norm(matrix[:2, 0])), float(np.linalg.norm(matrix[:2, 1]))

        nir_x, nir_y = scales(registration.nir.rgb_to_sensor_matrix)
        swir_x, swir_y = scales(registration.swir.rgb_to_sensor_matrix)
        collected = []
        for index, requested_y in enumerate(roi_rows):
            roi_config = RoiConfig(
                mode="manual" if manual else "auto",
                x=args.roi_x,
                y=requested_y,
                width=args.roi_width,
                height=args.roi_height,
                auto_candidates=160,
            )
            roi = choose_roi(roi_config, registration, dataset.rgb.meta.shape[:2])
            analysis_width = min(
                max(24, int(round(roi["width"] * min(nir_x, swir_x)))),
                dataset.nir.meta.samples,
                dataset.swir.meta.samples,
            )
            analysis_height = min(
                max(24, int(round(roi["height"] * min(nir_y, swir_y)))),
                dataset.nir.meta.lines,
                dataset.swir.meta.lines,
            )
            refined = estimate_roi_registration(
                dataset,
                registration,
                roi,
                (analysis_height, analysis_width),
                config,
            )
            suffix = "" if len(roi_rows) == 1 else f"_{index + 1:02d}_y{int(roi['y']):06d}"
            _save_overlay(
                args.output_dir / f"rgb_swir_roi{suffix}_before.png",
                refined.reference_structure,
                refined.swir_initial,
            )
            _save_overlay(
                args.output_dir / f"rgb_swir_roi{suffix}_after.png",
                refined.reference_structure,
                refined.swir_aligned,
            )
            _save_checkerboard(
                args.output_dir / f"rgb_swir_roi{suffix}_checkerboard.png",
                refined.reference_structure,
                refined.swir_aligned,
            )
            current = refined.to_dict()
            current["requested_rgb_roi"] = {
                "x": args.roi_x,
                "y": requested_y,
                "width": args.roi_width,
                "height": args.roi_height,
            }
            collected.append(current)
        if len(collected) == 1:
            roi_report = collected[0]
        else:
            roi_reports = collected
    report = {
        "status": "coarse_registration_estimated",
        "data_dir": str(args.data_dir),
        "software_source": str(args.geocorefusion_root),
        "preview_shape": list(registration.preview_rgb.shape),
        "registration": registration.to_dict(),
        "roi_refinement": roi_report,
        "roi_refinements": roi_reports,
        "elapsed_seconds": monotonic() - started,
        "evidence_boundary": (
            "Scores are same-data cross-modal structure diagnostics. They do not establish "
            "native-pixel registration error without independent landmarks."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    (args.output_dir / "coarse_registration.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
