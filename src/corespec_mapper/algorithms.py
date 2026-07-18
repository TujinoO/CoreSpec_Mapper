from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import warnings

import numpy as np


def _solve_small_system(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    augmented = np.column_stack([np.asarray(matrix, dtype=np.float64), np.asarray(target, dtype=np.float64)])
    size = augmented.shape[0]
    for column in range(size):
        pivot = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[pivot, column]) < 1e-14:
            raise ValueError("Singular Savitzky-Golay design matrix")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] /= augmented[column, column]
        for row in range(size):
            if row != column:
                augmented[row] -= augmented[row, column] * augmented[column]
    return augmented[:, -1]


def _savgol_weights(window_length: int, polyorder: int) -> np.ndarray:
    half = window_length // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    design = np.column_stack([offsets**power for power in range(polyorder + 1)])
    normal = design.T @ design
    target = np.zeros(polyorder + 1, dtype=np.float64)
    target[0] = 1.0
    coefficients = _solve_small_system(normal, target)
    return design @ coefficients


def savgol_smooth(spectra: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float64)
    if window_length % 2 != 1 or window_length <= polyorder:
        raise ValueError("Savitzky-Golay window must be odd and greater than polyorder")
    if spectra.shape[-1] < window_length:
        raise ValueError("Spectrum has fewer bands than the smoothing window")
    weights = _savgol_weights(window_length, polyorder)
    half = window_length // 2
    pad_width = [(0, 0)] * spectra.ndim
    pad_width[-1] = (half, half)
    padded = np.pad(spectra, pad_width, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_length, axis=-1)
    return np.sum(windows * weights, axis=-1)


def spectral_angles(pixels: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    endmembers = np.asarray(endmembers, dtype=np.float64)
    if pixels.ndim != 2 or endmembers.ndim != 2 or pixels.shape[1] != endmembers.shape[1]:
        raise ValueError("Pixels and endmembers must be 2D arrays with the same band count")
    pixel_norm = np.linalg.norm(pixels, axis=1)
    endmember_norm = np.linalg.norm(endmembers, axis=1)
    denominator = pixel_norm[:, None] * endmember_norm[None, :]
    cosine = np.divide(
        pixels @ endmembers.T,
        denominator,
        out=np.full((pixels.shape[0], endmembers.shape[0]), np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def classify_sam(angles: np.ndarray, threshold: float) -> np.ndarray:
    angles = np.asarray(angles)
    safe = np.where(np.isfinite(angles), angles, np.inf)
    best = np.argmin(safe, axis=1)
    best_angle = safe[np.arange(safe.shape[0]), best]
    return np.where(best_angle <= threshold, best + 1, 0).astype(np.uint8)


def _upper_hull_indices(x: np.ndarray, y: np.ndarray) -> list[int]:
    hull: list[int] = []
    for idx in range(x.size):
        while len(hull) >= 2:
            i, j = hull[-2], hull[-1]
            cross = (x[j] - x[i]) * (y[idx] - y[j]) - (y[j] - y[i]) * (x[idx] - x[j])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(idx)
    return hull


def continuum_remove(spectra: np.ndarray, wavelengths_nm: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    if spectra.ndim == 1:
        spectra = spectra[None, :]
        squeeze = True
    else:
        squeeze = False
    if spectra.ndim != 2 or spectra.shape[1] != wavelengths.size:
        raise ValueError("Spectra must have shape (n, bands) matching wavelengths")
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Wavelengths must be strictly increasing")

    result = np.full_like(spectra, np.nan, dtype=np.float64)
    for row_idx, spectrum in enumerate(spectra):
        if not np.all(np.isfinite(spectrum)) or np.any(spectrum <= 0):
            continue
        hull = _upper_hull_indices(wavelengths, spectrum)
        continuum = np.interp(wavelengths, wavelengths[hull], spectrum[hull])
        result[row_idx] = np.divide(spectrum, continuum, out=np.full_like(spectrum, np.nan), where=np.abs(continuum) > epsilon)
    return result[0] if squeeze else result


@dataclass(frozen=True)
class SffResult:
    scale: np.ndarray
    rms: np.ndarray
    quality: np.ndarray


def spectral_feature_fit(pixel_absorption: np.ndarray, reference_absorption: np.ndarray) -> SffResult:
    pixels = np.asarray(pixel_absorption, dtype=np.float64)
    references = np.asarray(reference_absorption, dtype=np.float64)
    if pixels.ndim != 2 or references.ndim != 2 or pixels.shape[1] != references.shape[1]:
        raise ValueError("Pixel and reference absorptions must be 2D with matching feature counts")
    reference_energy = np.sum(references * references, axis=1)
    dot = pixels @ references.T
    scale = np.divide(dot, reference_energy[None, :], out=np.zeros_like(dot), where=reference_energy[None, :] > 0)
    pixel_energy = np.sum(pixels * pixels, axis=1)[:, None]
    residual_energy = pixel_energy - 2.0 * scale * dot + scale * scale * reference_energy[None, :]
    rms = np.sqrt(np.maximum(residual_energy, 0.0) / pixels.shape[1])
    # ENVI-style SFF favors a large feature scale and a small residual error.
    quality = np.divide(
        scale,
        np.maximum(rms, 1e-12),
        out=np.zeros_like(scale),
        where=(scale > 0) & np.isfinite(rms),
    )
    return SffResult(scale=scale, rms=rms, quality=quality)


def absorption_depth(continuum_removed: np.ndarray) -> np.ndarray:
    continuum_removed = np.asarray(continuum_removed, dtype=np.float64)
    absorption = 1.0 - continuum_removed
    finite = np.isfinite(absorption)
    filled = np.where(finite, absorption, -np.inf)
    depth = np.max(filled, axis=-1)
    return np.where(np.any(finite, axis=-1), depth, np.nan)


@dataclass(frozen=True)
class AbsorptionFeatureMetrics:
    full_depth: np.ndarray
    center_depth: np.ndarray
    center_nm: np.ndarray
    depth_ratio: np.ndarray


def absorption_feature_metrics(
    continuum_removed: np.ndarray,
    wavelengths_nm: np.ndarray,
    search_window_nm: tuple[float, float] | list[float],
) -> AbsorptionFeatureMetrics:
    spectra = np.asarray(continuum_removed, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    if spectra.ndim != 2 or spectra.shape[1] != wavelengths.size:
        raise ValueError("Continuum-removed spectra must be 2D and match wavelengths")
    lower, upper = (float(value) for value in search_window_nm)
    selected = (wavelengths >= lower) & (wavelengths <= upper)
    if np.count_nonzero(selected) < 1:
        raise ValueError(f"Feature search window contains no bands: {search_window_nm}")

    absorption = 1.0 - spectra
    finite = np.isfinite(absorption)
    safe = np.where(finite, absorption, -np.inf)
    full_depth = np.max(safe, axis=1)
    has_full = np.any(finite, axis=1)
    full_depth = np.where(has_full, full_depth, np.nan)

    selected_safe = safe[:, selected]
    selected_finite = finite[:, selected]
    center_indices = np.argmax(selected_safe, axis=1)
    center_depth = selected_safe[np.arange(spectra.shape[0]), center_indices]
    has_center = np.any(selected_finite, axis=1)
    center_depth = np.where(has_center, center_depth, np.nan)
    center_nm = np.where(has_center, wavelengths[selected][center_indices], np.nan)
    depth_ratio = np.divide(
        center_depth,
        np.maximum(full_depth, 1e-12),
        out=np.full_like(center_depth, np.nan),
        where=has_center & np.isfinite(full_depth) & (full_depth > 0),
    )
    return AbsorptionFeatureMetrics(
        full_depth=full_depth,
        center_depth=center_depth,
        center_nm=center_nm,
        depth_ratio=depth_ratio,
    )


def robust_column_bias(cube: np.ndarray, mask: np.ndarray, radius: int = 2) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if cube.ndim != 3 or cube.shape[:2] != mask.shape:
        raise ValueError("Cube must have shape (lines, samples, bands) matching the mask")
    if radius < 1:
        raise ValueError("Column-bias radius must be at least one")
    masked = np.where(mask[..., None], cube, np.nan)
    padded = np.pad(masked, ((0, 0), (radius, radius), (0, 0)), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        local = np.nanmedian(windows, axis=-1)
        residual = masked - local
        bias = np.nanmedian(residual, axis=0)
        center = np.nanmedian(bias, axis=0)
    bias = bias - center[None, :]
    return np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)


def _moving_sum(values: np.ndarray, size: int, axis: int) -> np.ndarray:
    if size % 2 != 1 or size < 1:
        raise ValueError("Moving-sum size must be a positive odd number")
    radius = size // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="constant")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.int32)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate([np.zeros(zero_shape, dtype=np.int32), cumulative], axis=axis)
    upper = [slice(None)] * values.ndim
    lower = [slice(None)] * values.ndim
    upper[axis] = slice(size, size + values.shape[axis])
    lower[axis] = slice(0, values.shape[axis])
    return cumulative[tuple(upper)] - cumulative[tuple(lower)]


def directional_stripe_mask(
    candidate: np.ndarray,
    valid_mask: np.ndarray,
    *,
    vertical_window: int = 61,
    min_vertical_density: float = 0.15,
    column_ratio: float = 1.8,
    column_excess: float = 0.04,
    max_lateral_support: int = 2,
    neighborhood_radius: int = 6,
) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if candidate.shape != valid_mask.shape or candidate.ndim != 2:
        raise ValueError("Candidate and valid masks must be matching 2D arrays")
    column_valid = np.maximum(valid_mask.sum(axis=0), 1)
    profile = candidate.sum(axis=0) / column_valid
    baseline = np.zeros_like(profile, dtype=np.float64)
    for column in range(profile.size):
        left = profile[max(0, column - neighborhood_radius) : max(0, column - 1)]
        right = profile[min(profile.size, column + 2) : min(profile.size, column + neighborhood_radius + 1)]
        neighborhood = np.concatenate([left, right])
        baseline[column] = float(np.median(neighborhood)) if neighborhood.size else 0.0
    column_peak = (
        (profile >= min_vertical_density)
        & (profile >= baseline * column_ratio)
        & (profile >= baseline + column_excess)
    )
    vertical_count = _moving_sum(candidate.astype(np.uint8), vertical_window, axis=0)
    vertical_valid = np.maximum(_moving_sum(valid_mask.astype(np.uint8), vertical_window, axis=0), 1)
    vertical_density = vertical_count / vertical_valid
    lateral_support = _moving_sum(candidate.astype(np.uint8), 5, axis=1)
    return (
        candidate
        & column_peak[None, :]
        & (vertical_density >= min_vertical_density)
        & (lateral_support <= max_lateral_support)
    )


def remove_elongated_components(
    mask: np.ndarray,
    *,
    min_height: int = 15,
    max_width: int = 3,
    min_aspect: float = 5.0,
) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(source)
    output = np.zeros_like(source)
    height, width = source.shape
    for row, column in np.argwhere(source):
        row, column = int(row), int(column)
        if visited[row, column]:
            continue
        queue = deque([(row, column)])
        visited[row, column] = True
        component: list[tuple[int, int]] = []
        while queue:
            current_row, current_column = queue.popleft()
            component.append((current_row, current_column))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    nr, nc = current_row + dr, current_column + dc
                    if 0 <= nr < height and 0 <= nc < width and source[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        rows, columns = zip(*component)
        component_height = max(rows) - min(rows) + 1
        component_width = max(columns) - min(columns) + 1
        aspect = component_height / max(component_width, 1)
        if component_height >= min_height and component_width <= max_width and aspect >= min_aspect:
            continue
        rr, cc = np.asarray(rows), np.asarray(columns)
        output[rr, cc] = True
    return output


def _binary_erosion(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        result = np.all(windows, axis=(-2, -1))
    return result


def _majority_filter(mask: np.ndarray, size: int) -> np.ndarray:
    if size % 2 != 1:
        raise ValueError("Majority-filter size must be odd")
    radius = size // 2
    padded = np.pad(np.asarray(mask, dtype=bool), radius, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return np.sum(windows, axis=(-2, -1)) >= (size * size // 2 + 1)


def _remove_small_components(mask: np.ndarray, min_component: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(source)
    output = np.zeros_like(source)
    height, width = source.shape
    for row, column in np.argwhere(source):
        row = int(row)
        column = int(column)
        if visited[row, column]:
            continue
        queue = deque([(row, column)])
        visited[row, column] = True
        component: list[tuple[int, int]] = []
        while queue:
            current_row, current_column = queue.popleft()
            component.append((current_row, current_column))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = current_row + dr, current_column + dc
                    if 0 <= nr < height and 0 <= nc < width and source[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        if len(component) >= min_component:
            rr, cc = zip(*component)
            output[np.asarray(rr), np.asarray(cc)] = True
    return output


def clean_binary_mask(mask: np.ndarray, erode_pixels: int = 0, median_size: int = 3, min_component: int = 4) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool)
    if erode_pixels > 0:
        cleaned = _binary_erosion(cleaned, erode_pixels)
    if median_size > 1:
        cleaned = _majority_filter(cleaned, median_size)
    if min_component > 1:
        cleaned = _remove_small_components(cleaned, min_component)
    return cleaned
