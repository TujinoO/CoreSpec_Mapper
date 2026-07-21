from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np

from .envi import EnviDataset


@dataclass(frozen=True)
class MaskQualityFlag:
    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaskAudit:
    status: str
    source: str
    lines: int
    samples: int
    selected_band_indices: tuple[int, ...]
    selected_wavelengths_nm: tuple[float, ...]
    spectrally_valid_fraction: float | None
    threshold_method: str
    threshold: float | None
    border_brightness_median: float | None
    border_brightness_scale: float | None
    raw_mask_pixels: int
    raw_mask_fraction: float
    final_mask_pixels: int
    final_mask_fraction: float
    component_count: int
    kept_component_count: int
    removed_component_pixels: int
    interior_pixel_fraction: float | None = None
    flags: tuple[MaskQualityFlag, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["flags"] = [flag.to_dict() for flag in self.flags]
        return value


@dataclass(frozen=True)
class MaterialMaskResult:
    mask: np.ndarray = field(repr=False)
    audit: MaskAudit

    @property
    def ready(self) -> bool:
        return self.audit.ready

    def require_ready(self) -> "MaterialMaskResult":
        if not self.ready:
            raise MaskGateError(self.audit)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_shape": [int(item) for item in self.mask.shape],
            "audit": self.audit.to_dict(),
        }


class MaskGateError(ValueError):
    def __init__(self, audit: MaskAudit):
        self.audit = audit
        codes = ", ".join(flag.code for flag in audit.flags if flag.severity == "error")
        super().__init__(f"Material-mask quality gate failed: {codes or 'unknown reason'}")


def _band_indices(dataset: EnviDataset, bands: Sequence[int] | None, maximum: int) -> tuple[int, ...]:
    if bands is not None:
        selected = tuple(dict.fromkeys(int(item) for item in bands))
        if not selected:
            raise ValueError("At least one mask-analysis band is required")
        if any(item < 0 or item >= dataset.info.bands for item in selected):
            raise ValueError(f"Mask-analysis band is outside [0, {dataset.info.bands})")
        return selected
    count = min(max(int(maximum), 3), dataset.info.bands)
    if count == dataset.info.bands:
        return tuple(range(dataset.info.bands))
    wavelengths = dataset.info.wavelengths_nm
    if wavelengths is not None and wavelengths.size == dataset.info.bands:
        targets = np.linspace(float(wavelengths[0]), float(wavelengths[-1]), count + 2)[1:-1]
        return tuple(dict.fromkeys(int(np.argmin(np.abs(wavelengths - target))) for target in targets))
    return tuple(dict.fromkeys(int(item) for item in np.linspace(0, dataset.info.bands - 1, count)))


def _robust_scale(values: np.ndarray, center: float, relative_floor: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return max(abs(center) * relative_floor, 1e-6)
    mad = 1.4826 * float(np.median(np.abs(finite - center)))
    q25, q75 = np.percentile(finite, [25.0, 75.0])
    iqr_scale = float(q75 - q25) / 1.349
    return max(mad, iqr_scale, abs(center) * relative_floor, 1e-6)


def _border(shape: tuple[int, int], width: int) -> np.ndarray:
    rows, columns = shape
    size = min(max(int(width), 1), max(1, min(rows, columns) // 2))
    result = np.zeros(shape, dtype=bool)
    result[:size] = True
    result[-size:] = True
    result[:, :size] = True
    result[:, -size:] = True
    return result


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if upper <= lower:
        return lower
    histogram, edges = np.histogram(finite, bins=max(int(bins), 16), range=(lower, upper))
    probability = histogram.astype(np.float64)
    probability /= max(float(np.sum(probability)), 1.0)
    centers = (edges[:-1] + edges[1:]) * 0.5
    cumulative_probability = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * centers)
    total_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    between = np.divide(
        (total_mean * cumulative_probability - cumulative_mean) ** 2,
        denominator,
        out=np.full_like(denominator, -np.inf),
        where=denominator > 1e-12,
    )
    return float(centers[int(np.argmax(between))])


def _component_cleanup(mask: np.ndarray, minimum_pixels: int) -> tuple[np.ndarray, int, int, int]:
    """Remove small exact 8-connected components using row runs, not pixel queues."""
    source = np.asarray(mask, dtype=bool)
    output = np.zeros_like(source)
    parents: list[int] = []
    sizes: list[int] = []
    runs: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, int]] = []

    def find(label: int) -> int:
        root = label
        while parents[root] != root:
            root = parents[root]
        while parents[label] != label:
            parent = parents[label]
            parents[label] = root
            label = parent
        return root

    def union(left: int, right: int) -> int:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return left_root
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        sizes[left_root] += sizes[right_root]
        return left_root

    for row in range(source.shape[0]):
        padded = np.pad(source[row].astype(np.int8, copy=False), 1, mode="constant")
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        current: list[tuple[int, int, int]] = []
        previous_index = 0
        for start, stop in zip(starts, stops):
            label = len(parents)
            parents.append(label)
            sizes.append(int(stop - start))
            while previous_index < len(previous) and previous[previous_index][1] < start:
                previous_index += 1
            overlap_index = previous_index
            while overlap_index < len(previous) and previous[overlap_index][0] <= stop:
                previous_start, previous_stop, previous_label = previous[overlap_index]
                if previous_stop >= start and previous_start <= stop:
                    label = union(label, previous_label)
                overlap_index += 1
            current.append((int(start), int(stop), label))
            runs.append((row, int(start), int(stop), label))
        previous = current

    roots = {find(label) for label in range(len(parents))}
    kept_roots = {root for root in roots if sizes[root] >= minimum_pixels}
    for row, start, stop, label in runs:
        if find(label) in kept_roots:
            output[row, start:stop] = True
    removed_pixels = sum(sizes[root] for root in roots if root not in kept_roots)
    return output, len(roots), len(kept_roots), int(removed_pixels)


def _interior_pixel_fraction(mask: np.ndarray, minimum_neighbors: int = 8) -> float:
    """Measure how much of a mask is filled material rather than an edge network."""
    source = np.asarray(mask, dtype=bool)
    selected = int(np.count_nonzero(source))
    if selected == 0:
        return 0.0
    padded = np.pad(source.astype(np.uint8, copy=False), 1, mode="constant")
    neighbours = np.zeros(source.shape, dtype=np.uint8)
    for row_offset in range(3):
        for column_offset in range(3):
            neighbours += padded[
                row_offset : row_offset + source.shape[0],
                column_offset : column_offset + source.shape[1],
            ]
    interior = source & (neighbours >= max(1, min(int(minimum_neighbors), 9)))
    return float(np.count_nonzero(interior) / selected)


def _existing_mask_array(
    image: EnviDataset,
    existing_mask: str | Path | EnviDataset | np.ndarray,
    *,
    chunk_rows: int,
    epsilon: float,
) -> np.ndarray:
    if isinstance(existing_mask, np.ndarray):
        values = np.asarray(existing_mask)
        if values.shape != (image.info.lines, image.info.samples):
            raise ValueError("Existing mask and reflectance image dimensions do not match")
        return values if values.dtype == np.bool_ else np.isfinite(values) & (np.abs(values) > epsilon)
    owned = not isinstance(existing_mask, EnviDataset)
    dataset = EnviDataset(existing_mask) if owned else existing_mask
    try:
        if (dataset.info.lines, dataset.info.samples) != (image.info.lines, image.info.samples):
            raise ValueError("Existing mask and reflectance image dimensions do not match")
        if dataset.info.bands == 1:
            bands = (0,)
        else:
            bands = tuple(dict.fromkeys((0, dataset.info.bands // 2, dataset.info.bands - 1)))
        result = np.zeros((image.info.lines, image.info.samples), dtype=bool)
        for row_start, row_stop, cube in dataset.iter_rows(chunk_rows=chunk_rows, bands=bands):
            values = np.asarray(cube)
            result[row_start:row_stop] = np.any(np.isfinite(values) & (np.abs(values) > epsilon), axis=-1)
        return result
    finally:
        if owned:
            dataset.close()


def _automatic_mask(
    image: EnviDataset,
    selected_bands: tuple[int, ...],
    *,
    chunk_rows: int,
    minimum_valid_band_fraction: float,
    epsilon: float,
    border_width: int,
) -> tuple[np.ndarray, dict[str, float]]:
    shape = (image.info.lines, image.info.samples)
    brightness = np.full(shape, np.nan, dtype=np.float32)
    contrast = np.full(shape, np.nan, dtype=np.float32)
    spectral_quality = np.zeros(shape, dtype=bool)
    required = max(1, int(np.ceil(len(selected_bands) * minimum_valid_band_fraction)))
    for row_start, row_stop, cube in image.iter_rows(chunk_rows=chunk_rows, bands=selected_bands):
        values = np.asarray(cube, dtype=np.float32)
        finite_positive = np.isfinite(values) & (values > epsilon)
        enough = np.sum(finite_positive, axis=-1) >= required
        safe = np.where(finite_positive, values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            current_brightness = np.nanmedian(safe, axis=-1)
            # The selected band set is intentionally small.  A finite range is a
            # useful material/background contrast proxy and avoids nanpercentile's
            # per-pixel sort, which is disproportionately slow on full core scans.
            lower = np.nanmin(safe, axis=-1)
            upper = np.nanmax(safe, axis=-1)
        current_contrast = np.divide(
            upper - lower,
            np.maximum(np.abs(current_brightness), epsilon),
            out=np.full_like(current_brightness, np.nan),
            where=np.isfinite(current_brightness),
        )
        local = slice(row_start, row_stop)
        spectral_quality[local] = enough & np.isfinite(current_brightness) & np.isfinite(current_contrast)
        brightness[local] = current_brightness.astype(np.float32, copy=False)
        contrast[local] = current_contrast.astype(np.float32, copy=False)

    border = _border(shape, border_width)
    background = border & spectral_quality
    if np.count_nonzero(background) < 16:
        background = spectral_quality
    background_brightness = brightness[background]
    background_contrast = contrast[background]
    brightness_center = float(np.median(background_brightness)) if background_brightness.size else 0.0
    contrast_center = float(np.median(background_contrast)) if background_contrast.size else 0.0
    brightness_scale = _robust_scale(background_brightness, brightness_center, 0.02)
    contrast_scale = _robust_scale(background_contrast, contrast_center, 0.05)
    border_heterogeneity = brightness_scale / max(abs(brightness_center), 1e-3)
    if border_heterogeneity > 0.25:
        # A cropped ROI may contain core pieces along every border, making the
        # border population unusable as a background model.  In that case the
        # high-reflectance side of a global Otsu split is the conservative
        # material proxy; component cleanup still removes isolated labels and
        # tray highlights.  The branch is explicit in the audit record.
        threshold = _otsu_threshold(brightness[spectral_quality])
        raw = spectral_quality & np.isfinite(brightness) & (brightness >= threshold)
        return raw, {
            "spectrally_valid_fraction": float(np.mean(spectral_quality)),
            "threshold": float(threshold),
            "threshold_method": "global_brightness_otsu_cropped_roi_fallback",
            "border_brightness_median": brightness_center,
            "border_brightness_scale": brightness_scale,
        }
    brightness_distance = np.abs(brightness.astype(np.float64) - brightness_center) / brightness_scale
    contrast_excess = np.clip((contrast.astype(np.float64) - contrast_center) / contrast_scale, 0.0, None)
    score = np.clip(0.70 * brightness_distance + 0.30 * contrast_excess, 0.0, 1000.0)
    score[~spectral_quality] = np.nan
    usable_scores = score[spectral_quality]
    threshold = _otsu_threshold(usable_scores)
    border_scores = score[background]
    if border_scores.size:
        # A few bright objects or reconstruction seams may touch a cropped ROI
        # border.  Let them raise the Otsu cut, but cap that guard at the global
        # 80th percentile so a handful of border outliers cannot collapse the
        # material mask to <1% of the scene (observed in fused low-res cubes).
        border_guard = float(np.percentile(border_scores, 99.5))
        global_guard_cap = float(np.percentile(usable_scores, 80.0))
        threshold = max(threshold, min(border_guard, global_guard_cap))
    raw = spectral_quality & np.isfinite(score) & (score >= threshold)
    return raw, {
        "spectrally_valid_fraction": float(np.mean(spectral_quality)),
        "threshold": float(threshold),
        "threshold_method": "border_robust_score_plus_otsu",
        "border_brightness_median": brightness_center,
        "border_brightness_scale": brightness_scale,
    }


def build_material_mask(
    image: str | Path | EnviDataset,
    existing_mask: str | Path | EnviDataset | np.ndarray | None = None,
    *,
    bands: Sequence[int] | None = None,
    maximum_analysis_bands: int = 9,
    chunk_rows: int = 64,
    minimum_valid_band_fraction: float = 0.70,
    minimum_mask_fraction: float = 0.005,
    maximum_mask_fraction: float = 0.85,
    minimum_mask_pixels: int = 128,
    minimum_component_pixels: int | None = None,
    clean_existing_mask: bool = False,
    source_override: str | None = None,
    additional_flags: Sequence[MaskQualityFlag] = (),
    minimum_interior_pixel_fraction: float = 0.65,
    border_width: int | None = None,
    epsilon: float = 1e-10,
    strict: bool = False,
) -> MaterialMaskResult:
    """Build or validate a same-grid material mask without loading the reflectance cube.

    Automatic mode reads at most ``maximum_analysis_bands`` in row chunks and retains
    only two float32 diagnostic planes.  The returned audit is JSON serialisable and
    carries hard gates for an almost-full-frame or materially undersized mask.
    """
    if not 0.0 < minimum_valid_band_fraction <= 1.0:
        raise ValueError("minimum_valid_band_fraction must be in (0, 1]")
    if not 0.0 <= minimum_mask_fraction < maximum_mask_fraction <= 1.0:
        raise ValueError("Mask-fraction gates must satisfy 0 <= minimum < maximum <= 1")
    if chunk_rows < 1 or minimum_mask_pixels < 1:
        raise ValueError("chunk_rows and minimum_mask_pixels must be positive")
    if not 0.0 <= minimum_interior_pixel_fraction <= 1.0:
        raise ValueError("minimum_interior_pixel_fraction must be in [0, 1]")
    owned = not isinstance(image, EnviDataset)
    dataset = EnviDataset(image) if owned else image
    try:
        total_pixels = dataset.info.lines * dataset.info.samples
        component_floor = (
            max(8, int(np.ceil(total_pixels * 0.0001)))
            if minimum_component_pixels is None
            else max(1, int(minimum_component_pixels))
        )
        selected = _band_indices(dataset, bands, maximum_analysis_bands)
        wavelengths = dataset.info.wavelengths_nm
        selected_wavelengths = (
            tuple(float(wavelengths[index]) for index in selected)
            if wavelengths is not None and wavelengths.size == dataset.info.bands
            else ()
        )
        if existing_mask is None:
            raw, metrics = _automatic_mask(
                dataset,
                selected,
                chunk_rows=chunk_rows,
                minimum_valid_band_fraction=minimum_valid_band_fraction,
                epsilon=epsilon,
                border_width=border_width or max(2, min(dataset.info.lines, dataset.info.samples) // 32),
            )
            source = source_override or "automatic_reflectance_material_mask"
            threshold_method = str(metrics["threshold_method"])
            cleanup_floor = component_floor
        else:
            raw = _existing_mask_array(dataset, existing_mask, chunk_rows=chunk_rows, epsilon=epsilon)
            metrics = {
                "spectrally_valid_fraction": None,
                "threshold": None,
                "border_brightness_median": None,
                "border_brightness_scale": None,
            }
            source = source_override or "existing_same_grid_mask"
            threshold_method = "existing_nonzero"
            cleanup_floor = component_floor if clean_existing_mask else 1

        raw_count = int(np.count_nonzero(raw))
        cleaned, component_count, kept_count, removed_pixels = _component_cleanup(raw, cleanup_floor)
        final_count = int(np.count_nonzero(cleaned))
        raw_fraction = raw_count / max(total_pixels, 1)
        final_fraction = final_count / max(total_pixels, 1)
        interior_fraction = _interior_pixel_fraction(cleaned)
        flags: list[MaskQualityFlag] = list(additional_flags)
        required_pixels = max(int(minimum_mask_pixels), int(np.ceil(total_pixels * minimum_mask_fraction)))
        if final_count < required_pixels:
            flags.append(MaskQualityFlag(
                "MASK_TOO_SMALL",
                "error",
                "The material mask contains too few pixels for calibration and mapping.",
                {"pixels": final_count, "required_pixels": required_pixels, "fraction": final_fraction},
            ))
        if final_fraction > maximum_mask_fraction:
            flags.append(MaskQualityFlag(
                "MASK_TOO_LARGE",
                "error",
                "The material mask covers almost the full frame and likely includes tray or background.",
                {"fraction": final_fraction, "maximum_fraction": maximum_mask_fraction},
            ))
        valid_fraction = metrics["spectrally_valid_fraction"]
        if valid_fraction is not None and float(valid_fraction) < minimum_mask_fraction:
            flags.append(MaskQualityFlag(
                "INSUFFICIENT_VALID_SPECTRA",
                "error",
                "Too few pixels contain enough finite positive analysis bands.",
                {"fraction": float(valid_fraction)},
            ))
        automatic_source = existing_mask is None or bool(
            source_override and source_override.startswith("automatic_")
        )
        if automatic_source and interior_fraction < minimum_interior_pixel_fraction:
            flags.append(MaskQualityFlag(
                "MASK_EDGE_LIKE",
                "error",
                "The automatic mask is dominated by outlines or cracks instead of filled core material.",
                {
                    "interior_pixel_fraction": interior_fraction,
                    "minimum_interior_pixel_fraction": minimum_interior_pixel_fraction,
                },
            ))
        if removed_pixels:
            flags.append(MaskQualityFlag(
                "SMALL_COMPONENTS_REMOVED",
                "info",
                "Small disconnected material-mask components were removed.",
                {"pixels": int(removed_pixels), "minimum_component_pixels": cleanup_floor},
            ))
        fragmentation_limit = max(32, dataset.info.lines // 64)
        if kept_count > fragmentation_limit:
            flags.append(MaskQualityFlag(
                "MASK_HIGHLY_FRAGMENTED",
                "warning",
                "The material mask contains many disconnected components and should be reviewed before mapping.",
                {"components": kept_count, "review_threshold": fragmentation_limit},
            ))
        status = "blocked" if any(flag.severity == "error" for flag in flags) else "ready"
        audit = MaskAudit(
            status=status,
            source=source,
            lines=dataset.info.lines,
            samples=dataset.info.samples,
            selected_band_indices=selected if existing_mask is None else (),
            selected_wavelengths_nm=selected_wavelengths if existing_mask is None else (),
            spectrally_valid_fraction=None if valid_fraction is None else float(valid_fraction),
            threshold_method=threshold_method,
            threshold=None if metrics["threshold"] is None else float(metrics["threshold"]),
            border_brightness_median=(
                None if metrics["border_brightness_median"] is None else float(metrics["border_brightness_median"])
            ),
            border_brightness_scale=(
                None if metrics["border_brightness_scale"] is None else float(metrics["border_brightness_scale"])
            ),
            raw_mask_pixels=raw_count,
            raw_mask_fraction=float(raw_fraction),
            final_mask_pixels=final_count,
            final_mask_fraction=float(final_fraction),
            component_count=component_count,
            kept_component_count=kept_count,
            removed_component_pixels=int(removed_pixels),
            interior_pixel_fraction=interior_fraction,
            flags=tuple(flags),
        )
        result = MaterialMaskResult(np.asarray(cleaned, dtype=bool), audit)
        if strict:
            result.require_ready()
        return result
    finally:
        if owned:
            dataset.close()
