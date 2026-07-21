from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import importlib
import json
import sys
import tempfile

import numpy as np

from .foreground_model import (
    default_foreground_model_package,
    inspect_foreground_model,
    run_foreground_model,
)


DEFAULT_GEOCOR_M12_ROOT = Path(
    r"E:\Code\Geocore_M0&1_Preprocessing\modules\M1-2_foreground_mask"
)


@dataclass(frozen=True)
class GeocoreM12Availability:
    module_root: str
    code_complete: bool
    model_package: str | None
    model_manifest: str | None
    weights: str | None
    available: bool
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeocoreM12Result:
    mask: np.ndarray
    metadata: Mapping[str, Any]


def inspect_geocore_m12(
    module_root: str | Path | None = None,
    model_package: str | Path | None = None,
) -> GeocoreM12Availability:
    # Compatibility entry point retained for existing configs and tests.  V5.3
    # uses the packaged model by default and no longer depends on another source
    # checkout or exposes the former project name in the desktop UI.
    if module_root is None:
        status = inspect_foreground_model(model_package)
        package = Path(str(status["package"]))
        return GeocoreM12Availability(
            module_root=str(package),
            code_complete=True,
            model_package=str(package),
            model_manifest=status.get("manifest"),
            weights=status.get("weights"),
            available=bool(status["available"]),
            missing=tuple(str(item) for item in status["missing"]),
        )
    root = Path(module_root or DEFAULT_GEOCOR_M12_ROOT)
    required_code = (
        root / "geocore_mask" / "inference" / "predictor.py",
        root / "geocore_mask" / "inference" / "tiler.py",
        root / "geocore_mask" / "inference" / "merger.py",
        root / "geocore_mask" / "postprocess" / "refine_mask.py",
        root / "geocore_mask" / "models" / "registry.py",
    )
    missing = [str(path) for path in required_code if not path.is_file()]
    package = Path(model_package) if model_package else root / "models" / "core_mask_unet_v1"
    manifest_path = package / "model_manifest.json"
    weights_path: Path | None = None
    if not manifest_path.is_file():
        missing.append(str(manifest_path))
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            weights_value = manifest.get("weights")
            if not weights_value:
                missing.append(f"{manifest_path}:weights")
            else:
                raw_weights = Path(str(weights_value))
                weights_path = raw_weights if raw_weights.is_absolute() else package / raw_weights
                if not weights_path.is_file():
                    missing.append(str(weights_path))
        except (OSError, ValueError, TypeError) as exc:
            missing.append(f"{manifest_path}: {exc}")
    return GeocoreM12Availability(
        module_root=str(root),
        code_complete=all(path.is_file() for path in required_code),
        model_package=str(package) if manifest_path.is_file() else None,
        model_manifest=str(manifest_path) if manifest_path.is_file() else None,
        weights=str(weights_path) if weights_path is not None and weights_path.is_file() else None,
        available=not missing,
        missing=tuple(missing),
    )


def resize_mask_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("GeoCore M1-2 output mask must be two dimensional")
    rows, columns = (int(target_shape[0]), int(target_shape[1]))
    if rows < 1 or columns < 1:
        raise ValueError("Target mask dimensions must be positive")
    if source.shape == (rows, columns):
        return source.copy()
    row_indices = np.minimum(
        (np.arange(rows, dtype=np.float64) * source.shape[0] / rows).astype(np.int64),
        source.shape[0] - 1,
    )
    column_indices = np.minimum(
        (np.arange(columns, dtype=np.float64) * source.shape[1] / columns).astype(np.int64),
        source.shape[1] - 1,
    )
    return source[np.ix_(row_indices, column_indices)]


def _read_binary_image(path: str | Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - desktop distribution includes Pillow through Qt tooling
        raise RuntimeError("Reading the GeoCore M1-2 mask requires Pillow") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def run_geocore_m12(
    rgb_path: str | Path,
    target_shape: tuple[int, int],
    *,
    module_root: str | Path | None = None,
    model_package: str | Path | None = None,
    threshold: float | None = None,
    enable_postprocess: bool = True,
) -> GeocoreM12Result:
    """Run the existing GeoCore M1-2 model and adapt its RGB mask to a SWIR grid.

    Full-frame nearest-neighbour registration is intentionally reported in the
    metadata and must be reviewed.  Projects with a calibrated transform should
    supply an already registered external mask until a sensor-pair transform is
    available.
    """
    if module_root is None:
        mask, metadata = run_foreground_model(
            rgb_path,
            target_shape,
            model_package=model_package or default_foreground_model_package(),
            threshold=threshold,
        )
        return GeocoreM12Result(mask, metadata)

    availability = inspect_geocore_m12(module_root, model_package)
    if not availability.available:
        raise FileNotFoundError(
            "GeoCore M1-2 is not deployable; missing: " + "; ".join(availability.missing)
        )
    root = Path(availability.module_root)
    inserted = str(root) not in sys.path
    if inserted:
        sys.path.insert(0, str(root))
    try:
        predictor_module = importlib.import_module("geocore_mask.inference.predictor")
        package_module = importlib.import_module("geocore_mask.utils.model_package")
        package = package_module.load_model_package(model_package=availability.model_package)
        predictor = predictor_module.CoreMaskPredictor(package)
        with tempfile.TemporaryDirectory(prefix="corespec_geocore_m12_") as directory:
            result = predictor.predict(
                rgb_path,
                directory,
                threshold=threshold,
                enable_postprocess=enable_postprocess,
                output_preview=False,
            )
            output_files = result.get("output_files", {})
            mask_path = output_files.get("mask_png") or output_files.get("mask_tif")
            if not mask_path:
                raise RuntimeError("GeoCore M1-2 did not return a mask output")
            rgb_mask = _read_binary_image(mask_path)
            swir_mask = resize_mask_nearest(rgb_mask, target_shape)
        metadata = {
            "engine": "geocore_m12",
            "availability": availability.to_dict(),
            "rgb_shape": list(rgb_mask.shape),
            "target_shape": [int(target_shape[0]), int(target_shape[1])],
            "registration": "normalized_full_frame_nearest",
            "registration_review_required": rgb_mask.shape != tuple(target_shape),
            "model_metrics": result.get("metrics", {}),
            "model_warnings": result.get("warnings", ()),
        }
        return GeocoreM12Result(swir_mask, metadata)
    finally:
        if inserted:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass
