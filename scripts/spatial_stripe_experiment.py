from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np

from corespec_mapper.envi import EnviDataset, derive_mask, write_envi
from corespec_mapper.preview import _overlay, _stretched_gray, write_png


CONFIG_PATH = Path(r"E:\Code\CoreSpec_Mapper\configs\nc1.json")
PILOT = Path(r"E:\Code\CoreSpec_Mapper\output\pilot_corrected_v2_0000_0800")
DESTRIPED_ROOT = Path(r"E:\GRP Docs\2026.07 岩心影像矿物智能识别\destripe_experiment")
OUTPUT = Path(r"E:\GRP Docs\2026.07 岩心影像矿物智能识别\spatial_stripe_experiment")
START, STOP = 0, 800


VARIANTS = {
    "balanced": {
        "source_strength": 1.0,
        "vertical_window": 61,
        "min_vertical_density": 0.16,
        "column_ratio": 1.8,
        "column_excess": 0.04,
        "max_lateral_support": 2,
        "min_component": 8,
        "min_line_height": 15,
        "max_line_width": 3,
        "min_aspect": 5.0,
        "strict_sam": {"carbonates": 0.055, "sulfates": 0.05, "clays": 0.025},
    },
    "strong": {
        "source_strength": 1.0,
        "vertical_window": 41,
        "min_vertical_density": 0.12,
        "column_ratio": 1.5,
        "column_excess": 0.025,
        "max_lateral_support": 3,
        "min_component": 12,
        "min_line_height": 10,
        "max_line_width": 4,
        "min_aspect": 4.0,
        "strict_sam": {"carbonates": 0.05, "sulfates": 0.045, "clays": 0.0225},
    },
    "recommended": {
        "source_strength": 0.5,
        "vertical_window": 61,
        "min_vertical_density": 0.16,
        "column_ratio": 1.8,
        "column_excess": 0.04,
        "max_lateral_support": 2,
        "min_component": 6,
        "min_line_height": 15,
        "max_line_width": 3,
        "min_aspect": 5.0,
        "strict_sam": {"carbonates": 0.055, "sulfates": 0.05, "clays": 0.0275},
        "group_overrides": {
            "carbonates": {
                "min_vertical_density": 0.18,
                "column_ratio": 2.0,
                "column_excess": 0.05,
            },
            "clays": {
                "min_vertical_density": 0.22,
                "column_ratio": 2.2,
                "column_excess": 0.06,
                "min_component": 4,
                "min_line_height": 20,
            },
        },
    },
    "directional_clean": {
        "source_strength": 0.5,
        "vertical_window": 61,
        "min_vertical_density": 0.15,
        "column_ratio": 1.8,
        "column_excess": 0.04,
        "max_lateral_support": 2,
        "min_component": 6,
        "min_line_height": 15,
        "max_line_width": 3,
        "min_aspect": 5.0,
        "preserve_strong_directional": False,
        "preserve_strong_lines": False,
        "strict_sam": {"carbonates": 0.055, "sulfates": 0.05, "clays": 0.0275},
        "group_overrides": {
            "carbonates": {
                "min_vertical_density": 0.18,
                "column_ratio": 2.0,
                "column_excess": 0.05,
            },
            "clays": {
                "min_vertical_density": 0.22,
                "column_ratio": 2.2,
                "column_excess": 0.06,
                "min_component": 4,
                "min_line_height": 20,
            },
        },
    },
    "final_tuned": {
        "source_strength": 0.5,
        "vertical_window": 61,
        "min_vertical_density": 0.15,
        "column_ratio": 1.8,
        "column_excess": 0.04,
        "max_lateral_support": 2,
        "min_component": 6,
        "min_line_height": 15,
        "max_line_width": 3,
        "min_aspect": 5.0,
        "preserve_strong_directional": False,
        "preserve_strong_lines": False,
        "strict_sam": {"carbonates": 0.055, "sulfates": 0.05, "clays": 0.0275},
        "group_overrides": {
            "carbonates": {
                "min_vertical_density": 0.18,
                "column_ratio": 2.0,
                "column_excess": 0.05,
            },
            "sulfates": {
                "vertical_window": 41,
                "min_vertical_density": 0.12,
                "column_ratio": 1.5,
                "column_excess": 0.025,
                "max_lateral_support": 3,
                "min_component": 8,
                "min_line_height": 10,
                "max_line_width": 4,
                "min_aspect": 4.0,
            },
            "clays": {
                "min_vertical_density": 0.22,
                "column_ratio": 2.2,
                "column_excess": 0.06,
                "min_component": 4,
                "min_line_height": 20,
            },
        },
    },
}


def mineral_label(name: str) -> str:
    lowered = name.casefold()
    for mineral in ("calcite", "dolomite", "anhydrite", "gypsum", "illite", "montmorillonite", "kaolinite"):
        if mineral in lowered:
            return mineral
    return lowered


def moving_sum(values: np.ndarray, size: int, axis: int) -> np.ndarray:
    radius = size // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="constant")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.int32)
    zeros_shape = list(cumulative.shape)
    zeros_shape[axis] = 1
    cumulative = np.concatenate([np.zeros(zeros_shape, dtype=np.int32), cumulative], axis=axis)
    upper = [slice(None)] * values.ndim
    lower = [slice(None)] * values.ndim
    upper[axis] = slice(size, size + values.shape[axis])
    lower[axis] = slice(0, values.shape[axis])
    return cumulative[tuple(upper)] - cumulative[tuple(lower)]


def local_column_baseline(profile: np.ndarray, radius: int = 6, exclusion: int = 1) -> np.ndarray:
    result = np.zeros_like(profile, dtype=np.float64)
    for column in range(profile.size):
        left = profile[max(0, column - radius) : max(0, column - exclusion)]
        right = profile[min(profile.size, column + exclusion + 1) : min(profile.size, column + radius + 1)]
        neighborhood = np.concatenate([left, right])
        result[column] = float(np.median(neighborhood)) if neighborhood.size else 0.0
    return result


def component_filter(
    mask: np.ndarray,
    angles: np.ndarray,
    *,
    min_component: int,
    min_line_height: int,
    max_line_width: int,
    min_aspect: float,
    strict_angle: float,
    preserve_strong_lines: bool,
) -> tuple[np.ndarray, int, int]:
    visited = np.zeros_like(mask, dtype=bool)
    output = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    removed_small = 0
    removed_lines = 0
    for row, column in np.argwhere(mask):
        row, column = int(row), int(column)
        if visited[row, column]:
            continue
        queue = deque([(row, column)])
        visited[row, column] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            current_row, current_column = queue.popleft()
            pixels.append((current_row, current_column))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    nr, nc = current_row + dr, current_column + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        if len(pixels) < min_component:
            removed_small += len(pixels)
            continue
        rows, columns = zip(*pixels)
        component_height = max(rows) - min(rows) + 1
        component_width = max(columns) - min(columns) + 1
        aspect = component_height / max(component_width, 1)
        rr, cc = np.asarray(rows), np.asarray(columns)
        is_line = component_height >= min_line_height and component_width <= max_line_width and aspect >= min_aspect
        is_strong = float(np.nanmedian(angles[rr, cc])) <= strict_angle
        if is_line and (not preserve_strong_lines or not is_strong):
            removed_lines += len(pixels)
            continue
        output[rr, cc] = True
    return output, removed_small, removed_lines


config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
image = EnviDataset(config["analysis_image"])
mask_dataset = EnviDataset(config["analysis_mask"])
core_mask = derive_mask(mask_dataset, start=START, stop=STOP)
wavelengths = image.info.wavelengths_nm
assert wavelengths is not None
background_band = int(np.argmin(np.abs(wavelengths - 1600.0)))
background_values = np.array(image.read_rows(START, STOP, bands=[background_band])[..., 0], copy=True)
gray = _stretched_gray(background_values, core_mask)
background = np.repeat(gray[..., None], 3, axis=2)

for variant_name, parameters in VARIANTS.items():
    panels = [background]
    print(f"\n{variant_name}")
    for group in config["groups"]:
        group_name = group["name"]
        group_parameters = dict(parameters)
        group_parameters.update(parameters.get("group_overrides", {}).get(group_name, {}))
        source_root = DESTRIPED_ROOT / f"strength_{float(parameters.get('source_strength', 1.0)):.1f}"
        source_dataset = EnviDataset(source_root / group_name / "destripe_confirmed_classes.dat")
        classes = np.array(source_dataset.read_rows(0, STOP - START)[..., 0], dtype=np.uint8, copy=True)
        class_names = [str(name) for name in source_dataset.info.metadata["class names"]]
        source_dataset.close()
        endpoint_names = class_names[1:-1]
        endpoint_count = len(endpoint_names)
        endpoint_minerals = np.array([mineral_label(name) for name in endpoint_names], dtype=object)

        sam_dataset = EnviDataset(PILOT / group_name / "sam_rules.dat")
        sam_angles = np.array(sam_dataset.read_rows(0, STOP - START), dtype=np.float64, copy=True)
        sam_dataset.close()
        output_classes = np.zeros_like(classes)
        total_input = 0
        total_directional = 0
        total_small = 0
        total_lines = 0
        for endpoint, mineral in enumerate(endpoint_minerals):
            binary = core_mask & (classes == endpoint + 1)
            if not np.any(binary):
                continue
            same_mineral = np.flatnonzero(endpoint_minerals == mineral)
            angle = np.min(sam_angles[..., same_mineral], axis=-1)
            total_input += int(binary.sum())
            core_column_count = np.maximum(core_mask.sum(axis=0), 1)
            profile = binary.sum(axis=0) / core_column_count
            baseline = local_column_baseline(profile)
            column_peak = (
                (profile >= parameters["min_vertical_density"])
                & (profile >= baseline * group_parameters["column_ratio"])
                & (profile >= baseline + group_parameters["column_excess"])
            )
            column_peak &= profile >= group_parameters["min_vertical_density"]
            vertical_count = moving_sum(binary.astype(np.uint8), group_parameters["vertical_window"], axis=0)
            vertical_valid = np.maximum(
                moving_sum(core_mask.astype(np.uint8), group_parameters["vertical_window"], axis=0),
                1,
            )
            vertical_density = vertical_count / vertical_valid
            lateral_support = moving_sum(binary.astype(np.uint8), 5, axis=1)
            directional = (
                binary
                & column_peak[None, :]
                & (vertical_density >= group_parameters["min_vertical_density"])
                & (lateral_support <= group_parameters["max_lateral_support"])
            )
            strict_angle = group_parameters["strict_sam"][group_name]
            strong = angle <= strict_angle
            preserve_strong_directional = bool(group_parameters.get("preserve_strong_directional", True))
            directional_remove = directional & (~strong if preserve_strong_directional else True)
            total_directional += int(np.count_nonzero(directional_remove))
            filtered = binary & ~directional_remove
            filtered, removed_small, removed_lines = component_filter(
                filtered,
                angle,
                min_component=group_parameters["min_component"],
                min_line_height=group_parameters["min_line_height"],
                max_line_width=group_parameters["max_line_width"],
                min_aspect=group_parameters["min_aspect"],
                strict_angle=strict_angle,
                preserve_strong_lines=bool(group_parameters.get("preserve_strong_lines", True)),
            )
            total_small += removed_small
            total_lines += removed_lines
            output_classes[filtered] = endpoint + 1

        output_classes[~core_mask] = endpoint_count + 1
        variant_dir = OUTPUT / variant_name / group_name
        write_envi(
            output_classes,
            variant_dir / "stripe_suppressed_classes.dat",
            class_names=class_names,
            description=f"{group_name} robust spectral and directional stripe suppression",
        )
        overlay = _overlay(background, output_classes, class_names)
        write_png(OUTPUT / variant_name / f"{group_name}.png", overlay)
        panels.append(overlay)
        print(
            f"  {group_name}: input={total_input}, removed_directional={total_directional}, "
            f"removed_small={total_small}, removed_line_components={total_lines}, "
            f"retained={int(np.count_nonzero((output_classes > 0) & (output_classes <= endpoint_count)))}"
        )
    write_png(OUTPUT / f"comparison_{variant_name}.png", np.concatenate(panels, axis=1))

mask_dataset.close()
image.close()
