from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image

from corespec_mapper.envi import EnviDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream an ENVI RGB scan onto a registered multiple of a target ENVI grid."
    )
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--registration-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output ENVI .dat path")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--preview-height", type=int, default=4096)
    return parser.parse_args()


def _registration_matrix(record: dict) -> np.ndarray:
    candidates = (
        record.get("registration", {}).get("swir", {}),
        record.get("full_scan_coarse", {}).get("swir", {}),
        record.get("swir", {}),
    )
    for candidate in candidates:
        value = candidate.get("rgb_to_sensor_matrix") if isinstance(candidate, dict) else None
        if value is not None:
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError("RGB-to-SWIR registration matrix must be a finite 3 x 3 matrix")
            return matrix
    raise ValueError("Registration JSON does not contain registration.swir.rgb_to_sensor_matrix")


def _rgb_band_indices(dataset: EnviDataset) -> tuple[int, int, int]:
    wavelengths = dataset.info.wavelengths_nm
    if wavelengths is not None and wavelengths.size == dataset.info.bands:
        # ENVI RGB monitor files in this project store 450/550/650 nm in B/G/R
        # order.  The network was trained on conventional R/G/B images.
        return tuple(int(np.argmin(np.abs(wavelengths - target))) for target in (650.0, 550.0, 450.0))
    if dataset.info.bands < 3:
        return tuple(list(range(dataset.info.bands)) + [dataset.info.bands - 1] * (3 - dataset.info.bands))
    return (0, 1, 2)


def _write_header(path: Path, *, lines: int, samples: int, scale: int) -> Path:
    header = path.with_suffix(".hdr")
    header.write_text(
        "ENVI\n"
        "description = {RGB registered to the SWIR grid for CoreMaskUNet inference}\n"
        f"samples = {samples}\n"
        f"lines = {lines}\n"
        "bands = 3\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 1\n"
        "interleave = bil\n"
        "byte order = 0\n"
        "band names = {Red, Green, Blue}\n"
        "wavelength units = Nanometers\n"
        "wavelength = {650.0, 550.0, 450.0}\n"
        f"registered target scale = {scale}\n",
        encoding="utf-8",
    )
    return header


def _save_preview(data: np.memmap, path: Path, height: int) -> None:
    rows = np.linspace(0, data.shape[0] - 1, min(height, data.shape[0])).round().astype(np.int64)
    preview = np.moveaxis(np.asarray(data[rows, :, :]), 1, 2)
    Image.fromarray(preview, mode="RGB").save(path)


def main() -> int:
    args = _parse_args()
    if args.scale < 1:
        raise ValueError("--scale must be positive")
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    output = args.output.resolve()
    header = output.with_suffix(".hdr")
    metadata_path = output.with_suffix(".registration.json")
    preview_path = output.with_suffix(".preview.png")
    existing = [path for path in (output, header, metadata_path, preview_path) if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing registered RGB outputs: " + ", ".join(map(str, existing)))
    output.parent.mkdir(parents=True, exist_ok=True)

    registration_record = json.loads(args.registration_json.read_text(encoding="utf-8"))
    rgb_to_target = _registration_matrix(registration_record)
    target_to_rgb = np.linalg.inv(rgb_to_target)
    started = monotonic()
    valid_pixels = 0
    total_pixels = 0
    with EnviDataset(args.rgb) as rgb, EnviDataset(args.target) as target:
        output_lines = target.info.lines * args.scale
        output_samples = target.info.samples * args.scale
        band_indices = _rgb_band_indices(rgb)
        writer = np.memmap(
            output,
            dtype=np.uint8,
            mode="w+",
            shape=(output_lines, 3, output_samples),
        )
        sensor_x = np.linspace(
            0.0,
            target.info.samples - 1,
            output_samples,
            dtype=np.float64,
        )[None, :]
        for row_start in range(0, output_lines, args.chunk_rows):
            row_stop = min(output_lines, row_start + args.chunk_rows)
            sensor_y = np.linspace(
                row_start * (target.info.lines - 1) / max(output_lines - 1, 1),
                (row_stop - 1) * (target.info.lines - 1) / max(output_lines - 1, 1),
                row_stop - row_start,
                dtype=np.float64,
            )[:, None]
            raw_x = target_to_rgb[0, 0] * sensor_x + target_to_rgb[0, 1] * sensor_y + target_to_rgb[0, 2]
            raw_y = target_to_rgb[1, 0] * sensor_x + target_to_rgb[1, 1] * sensor_y + target_to_rgb[1, 2]
            valid = (
                (raw_x >= 0.0)
                & (raw_x <= rgb.info.samples - 1)
                & (raw_y >= 0.0)
                & (raw_y <= rgb.info.lines - 1)
            )
            clipped_x = np.clip(raw_x, 0.0, rgb.info.samples - 1.0)
            clipped_y = np.clip(raw_y, 0.0, rgb.info.lines - 1.0)
            x0_global = np.floor(clipped_x).astype(np.int64)
            y0_global = np.floor(clipped_y).astype(np.int64)
            x1_global = np.minimum(x0_global + 1, rgb.info.samples - 1)
            y1_global = np.minimum(y0_global + 1, rgb.info.lines - 1)
            source_y0 = int(np.min(y0_global))
            source_y1 = int(np.max(y1_global)) + 1
            source_x0 = int(np.min(x0_global))
            source_x1 = int(np.max(x1_global)) + 1
            source = np.asarray(
                rgb.read_rows(source_y0, source_y1, bands=band_indices)[:, source_x0:source_x1, :],
                dtype=np.float32,
            )
            local_y0 = y0_global - source_y0
            local_y1 = y1_global - source_y0
            local_x0 = x0_global - source_x0
            local_x1 = x1_global - source_x0
            fraction_x = (clipped_x - x0_global)[..., None].astype(np.float32)
            fraction_y = (clipped_y - y0_global)[..., None].astype(np.float32)
            top = source[local_y0, local_x0] * (1.0 - fraction_x) + source[local_y0, local_x1] * fraction_x
            bottom = source[local_y1, local_x0] * (1.0 - fraction_x) + source[local_y1, local_x1] * fraction_x
            sampled = top * (1.0 - fraction_y) + bottom * fraction_y
            sampled[~valid] = 0.0
            writer[row_start:row_stop] = np.moveaxis(np.clip(np.rint(sampled), 0, 255).astype(np.uint8), 2, 1)
            valid_pixels += int(np.count_nonzero(valid))
            total_pixels += int(valid.size)
        writer.flush()
        _write_header(output, lines=output_lines, samples=output_samples, scale=args.scale)
        _save_preview(writer, preview_path, args.preview_height)
        metadata = {
            "format": "CoreSpec registered RGB proxy",
            "source_rgb": str(rgb.info.data_path),
            "target_grid": str(target.info.data_path),
            "output_data": str(output),
            "output_header": str(header),
            "source_shape": [rgb.info.lines, rgb.info.samples, rgb.info.bands],
            "target_shape": [target.info.lines, target.info.samples],
            "registered_shape": [output_lines, output_samples, 3],
            "registered_scale": args.scale,
            "rgb_band_indices": list(band_indices),
            "rgb_to_target_matrix": rgb_to_target.tolist(),
            "target_to_rgb_matrix": target_to_rgb.tolist(),
            "resampling": "bilinear_pixel_center",
            "valid_fraction": valid_pixels / max(total_pixels, 1),
            "elapsed_seconds": monotonic() - started,
            "registration_level": "full_scan_coarse_affine",
            "manual_review_required": True,
            "evidence_boundary": (
                "The transform is estimated from same-data cross-modal structures and requires "
                "local registration review before a final mineral map is approved."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
