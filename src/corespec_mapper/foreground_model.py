from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from time import monotonic
from typing import Any
import json

import numpy as np

from .envi import EnviDataset


def default_foreground_model_package() -> Path:
    return Path(str(files("corespec_mapper.resources").joinpath("core_mask_model_v2")))


def inspect_foreground_model(model_package: str | Path | None = None) -> dict[str, Any]:
    package = Path(model_package) if model_package else default_foreground_model_package()
    manifest_path = package / "model_manifest.json"
    missing: list[str] = []
    manifest: dict[str, Any] = {}
    weights_path: Path | None = None
    if not manifest_path.is_file():
        missing.append(str(manifest_path))
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            value = manifest.get("weights")
            if not value:
                missing.append(f"{manifest_path}:weights")
            else:
                candidate = Path(str(value))
                weights_path = candidate if candidate.is_absolute() else package / candidate
                if not weights_path.is_file():
                    missing.append(str(weights_path))
        except (OSError, ValueError, TypeError) as exc:
            missing.append(f"{manifest_path}: {exc}")
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on desktop deployment
        missing.append(f"PyTorch runtime: {exc}")
    return {
        "engine": "integrated_core_foreground_v2",
        "package": str(package),
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "weights": str(weights_path) if weights_path is not None and weights_path.is_file() else None,
        "model_name": manifest.get("model_name"),
        "model_version": manifest.get("model_version"),
        "available": not missing,
        "missing": missing,
    }


def _read_rgb(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.casefold() in {".dat", ".img", ".hdr", ".sli"}:
        with EnviDataset(source) as dataset:
            bands = list(range(min(3, dataset.info.bands)))
            image = np.asarray(dataset.read_rows(0, dataset.info.lines, bands=bands), dtype=np.float32).copy()
    else:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - desktop extra normally supplies Pillow
            raise RuntimeError("智能岩心前景模型读取普通影像需要 Pillow") from exc
        with Image.open(source) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.float32)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[2] < 1:
        raise ValueError("智能岩心前景模型需要二维 RGB 或三波段 ENVI 影像")
    if image.shape[2] < 3:
        image = np.concatenate([image] + [image[..., -1:]] * (3 - image.shape[2]), axis=2)
    image = image[..., :3]
    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("RGB 影像没有有限像元")
    finite_values = image[finite]
    if float(np.nanmax(finite_values)) <= 2.0 or float(np.nanmin(finite_values)) < 0.0:
        stretched = np.zeros_like(image, dtype=np.float32)
        for band in range(3):
            values = image[..., band]
            usable = values[np.isfinite(values)]
            if usable.size == 0:
                continue
            lower, upper = np.percentile(usable, [2.0, 98.0])
            if upper > lower:
                stretched[..., band] = np.clip((values - lower) / (upper - lower), 0.0, 1.0) * 255.0
        image = stretched
    return np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0).astype(np.float32)


def _resize_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    rows, columns = (int(target_shape[0]), int(target_shape[1]))
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


def _model_class(torch):
    nn = torch.nn
    functional = torch.nn.functional

    class DoubleConv(nn.Module):
        def __init__(self, inputs: int, outputs: int, middle: int | None = None):
            super().__init__()
            middle = middle or outputs
            self.double_conv = nn.Sequential(
                nn.Conv2d(inputs, middle, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(middle),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle, outputs, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
            )

        def forward(self, value):
            return self.double_conv(value)

    class Down(nn.Module):
        def __init__(self, inputs: int, outputs: int):
            super().__init__()
            self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(inputs, outputs))

        def forward(self, value):
            return self.maxpool_conv(value)

    class Up(nn.Module):
        def __init__(self, inputs: int, outputs: int):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(inputs, outputs, inputs // 2)

        def forward(self, first, second):
            first = self.up(first)
            delta_y = second.size()[2] - first.size()[2]
            delta_x = second.size()[3] - first.size()[3]
            first = functional.pad(
                first,
                [delta_x // 2, delta_x - delta_x // 2, delta_y // 2, delta_y - delta_y // 2],
            )
            return self.conv(torch.cat([second, first], dim=1))

    class OutConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1), nn.Sigmoid())

        def forward(self, value):
            return self.conv(value)

    class CoreMaskUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inc = DoubleConv(3, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 512)
            self.up1 = Up(1024, 256)
            self.up2 = Up(512, 128)
            self.up3 = Up(256, 64)
            self.up4 = Up(128, 64)
            self.outc = OutConv()

        def forward(self, value):
            first = self.inc(value)
            second = self.down1(first)
            third = self.down2(second)
            fourth = self.down3(third)
            fifth = self.down4(fourth)
            value = self.up1(fifth, fourth)
            value = self.up2(value, third)
            value = self.up3(value, second)
            value = self.up4(value, first)
            return self.outc(value)

    return CoreMaskUNet


def _axis_positions(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def run_foreground_model(
    rgb_path: str | Path,
    target_shape: tuple[int, int],
    *,
    model_package: str | Path | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    availability = inspect_foreground_model(model_package)
    if not availability["available"]:
        raise FileNotFoundError(
            "智能岩心前景模型资产不完整：" + "; ".join(str(item) for item in availability["missing"])
        )
    package = Path(str(availability["package"]))
    manifest = json.loads((package / "model_manifest.json").read_text(encoding="utf-8"))
    started = monotonic()
    import torch

    device_name = str(manifest.get("inference", {}).get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model = _model_class(torch)()
    state = torch.load(str(availability["weights"]), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()

    image = _read_rgb(rgb_path)
    input_config = manifest.get("input", {})
    mean = np.asarray(input_config.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    std = np.asarray(input_config.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
    std[std == 0] = 1.0
    normalized = (image - mean[None, None, :]) / std[None, None, :]
    height, width = normalized.shape[:2]
    inference = manifest.get("inference", {})
    tile_size = int(inference.get("tile_size", input_config.get("image_size", 512)))
    overlap = float(inference.get("overlap", 0.25))
    step = max(1, int(tile_size * (1.0 - overlap)))
    padded_height = max(height, tile_size)
    padded_width = max(width, tile_size)
    padded = np.pad(
        normalized,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="edge",
    )
    axis = np.hanning(tile_size)
    if not np.any(axis):
        axis = np.ones(tile_size, dtype=np.float32)
    weight = np.outer(axis, axis).astype(np.float32)
    positive = weight[weight > 0]
    weight[weight == 0] = float(np.min(positive)) if positive.size else 1.0
    weight /= max(float(np.max(weight)), 1e-6)
    probability_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros_like(probability_sum)
    with torch.no_grad():
        for y in _axis_positions(padded_height, tile_size, step):
            for x in _axis_positions(padded_width, tile_size, step):
                patch = padded[y : y + tile_size, x : x + tile_size]
                tensor = torch.from_numpy(patch.transpose(2, 0, 1)[None]).float().to(device)
                output = model(tensor)
                probability = torch.softmax(output, dim=1)[:, 1]
                values = probability.squeeze(0).detach().cpu().numpy().astype(np.float32)
                probability_sum[y : y + tile_size, x : x + tile_size] += values * weight
                weight_sum[y : y + tile_size, x : x + tile_size] += weight
    probability = probability_sum / np.maximum(weight_sum, 1e-6)
    threshold_value = float(threshold if threshold is not None else inference.get("threshold", 0.5))
    rgb_mask = probability[:height, :width] >= threshold_value
    target_mask = _resize_nearest(rgb_mask, target_shape)
    metadata = {
        "engine": "integrated_core_foreground_v2",
        "model_version": manifest.get("model_version"),
        "device": str(device),
        "threshold": threshold_value,
        "rgb_shape": [height, width],
        "target_shape": [int(target_shape[0]), int(target_shape[1])],
        "registration": "normalized_full_frame_nearest",
        "registration_review_required": (height, width) != tuple(target_shape),
        "foreground_fraction_rgb": float(np.mean(rgb_mask)),
        "elapsed_seconds": monotonic() - started,
    }
    return target_mask, metadata
