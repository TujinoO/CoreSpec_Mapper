from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter


_PAIR = re.compile(
    r"(?m)^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    r"\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$"
)


@dataclass(frozen=True)
class EnviAsciiSpectrum:
    path: Path
    wavelengths_nm: np.ndarray
    values: np.ndarray
    source_label: str


def read_envi_ascii_spectrum(path: str | Path) -> EnviAsciiSpectrum:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    pairs = [(float(left), float(right)) for left, right in _PAIR.findall(text)]
    if len(pairs) < 3:
        raise ValueError(f"Fewer than three spectral points found in {source}")
    array = np.asarray(pairs, dtype=np.float64)
    wavelengths = array[:, 0]
    if not np.all(np.isfinite(array)) or np.any(np.diff(wavelengths) <= 0):
        raise ValueError(f"Spectrum wavelengths must be finite and strictly increasing: {source}")
    label_match = re.search(r"(?m)^Column\s+2:\s*(.+?)\s*$", text)
    label = label_match.group(1).strip() if label_match else source.stem
    return EnviAsciiSpectrum(source, wavelengths, array[:, 1], label)


def resample_empirical_spectrum(
    spectrum: EnviAsciiSpectrum,
    target_wavelengths_nm: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_wavelengths_nm, dtype=np.float64)
    return np.interp(
        target,
        spectrum.wavelengths_nm,
        spectrum.values,
        left=np.nan,
        right=np.nan,
    )


def _window(size: int, requested: int, polyorder: int) -> int:
    maximum = size if size % 2 == 1 else size - 1
    value = min(int(requested), maximum)
    if value % 2 == 0:
        value -= 1
    if value <= polyorder:
        raise ValueError("Not enough bands for the requested Savitzky-Golay transform")
    return value


def empirical_absorption_features(
    spectra: np.ndarray,
    *,
    fine_window: int = 5,
    baseline_window: int = 31,
    polyorder: int = 2,
) -> np.ndarray:
    """Extract narrow absorption structure while suppressing albedo and broad slope.

    The same deterministic transform is applied to user references and scene
    pixels. Rows containing a non-positive or non-finite value remain NaN.
    """
    values = np.asarray(spectra, dtype=np.float64)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("spectra must be one- or two-dimensional")
    fine = _window(values.shape[1], fine_window, polyorder)
    broad = _window(values.shape[1], baseline_window, polyorder)
    if broad <= fine:
        raise ValueError("baseline_window must resolve to a wider window than fine_window")
    valid = np.all(np.isfinite(values) & (values > 0.0), axis=1)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        selected = values[valid]
        smooth = savgol_filter(selected, fine, polyorder, axis=1, mode="interp")
        baseline = savgol_filter(smooth, broad, polyorder, axis=1, mode="interp")
        result[valid] = np.divide(
            baseline - smooth,
            np.maximum(np.abs(baseline), 1e-8),
        )
    return result[0] if squeeze else result


def time_normalized_mask(mask: np.ndarray, target_lines: int) -> tuple[np.ndarray, np.ndarray]:
    """Map a synchronized line-scan mask by normalized acquisition time."""
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2 or target_lines < 1:
        raise ValueError("mask must be 2D and target_lines must be positive")
    indices = np.linspace(0, source.shape[0] - 1, target_lines).round().astype(np.int64)
    return source[indices].copy(), indices


def spectral_angles(features: np.ndarray, references: np.ndarray) -> np.ndarray:
    pixels = np.asarray(features, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    if pixels.ndim != 2 or refs.ndim != 2 or pixels.shape[1] != refs.shape[1]:
        raise ValueError("features and references must be 2D with matching band count")
    denominator = np.linalg.norm(pixels, axis=1)[:, None] * np.linalg.norm(refs, axis=1)[None, :]
    dot = np.einsum("ij,kj->ik", pixels, refs, optimize=False)
    cosine = np.divide(dot, denominator, out=np.full_like(dot, np.nan), where=denominator > 0)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def policy_candidate(
    best_angle: np.ndarray,
    strength: np.ndarray,
    valid_mask: np.ndarray,
    *,
    maximum_angle_rad: float,
    minimum_strength: float,
    maximum_strength: float,
) -> np.ndarray:
    return (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(best_angle)
        & np.isfinite(strength)
        & (best_angle <= float(maximum_angle_rad))
        & (strength >= float(minimum_strength))
        & (strength <= float(maximum_strength))
    )


def local_radiometric_stability_mask(
    reflectance: np.ndarray,
    valid_mask: np.ndarray,
    *,
    window_size: int = 5,
    minimum_valid_fraction: float = 0.80,
    maximum_coefficient_of_variation: float = 0.30,
    maximum_log_gradient: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reject mixed pixels at dark gaps and abrupt radiometric boundaries.

    The coefficient of variation and the gradient of log-reflectance are both
    scale invariant.  This makes the fixed engineering gates portable across
    normally exposed reflectance scenes while avoiding a scene-percentile
    quota that would force a non-zero anomaly result.
    """
    values = np.asarray(reflectance, dtype=np.float32)
    foreground = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 2 or foreground.shape != values.shape:
        raise ValueError("reflectance and valid_mask must be matching 2D arrays")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least 3")
    if not 0.0 <= minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must be in [0, 1]")
    if maximum_coefficient_of_variation <= 0.0 or maximum_log_gradient <= 0.0:
        raise ValueError("radiometric stability limits must be positive")

    valid = foreground & np.isfinite(values) & (values > 0.0)
    support = ndimage.uniform_filter(
        valid.astype(np.float32), size=window_size, mode="constant", cval=0.0
    )
    selected = np.where(valid, values, 0.0)
    local_mean = ndimage.uniform_filter(
        selected, size=window_size, mode="constant", cval=0.0
    ) / np.maximum(support, 1e-6)
    local_square_mean = ndimage.uniform_filter(
        selected * selected, size=window_size, mode="constant", cval=0.0
    ) / np.maximum(support, 1e-6)
    local_cv = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0)) / np.maximum(
        local_mean, 1e-6
    )
    log_reflectance = np.zeros_like(values, dtype=np.float32)
    log_reflectance[valid] = np.log(np.maximum(values[valid], 1e-6))
    log_gradient = ndimage.gaussian_gradient_magnitude(log_reflectance, sigma=1.0)
    stable = (
        valid
        & (support >= float(minimum_valid_fraction))
        & (local_cv <= float(maximum_coefficient_of_variation))
        & (log_gradient <= float(maximum_log_gradient))
    )
    return stable, support, local_cv, log_gradient


def candidate_neighborhood_support(
    candidate_mask: np.ndarray,
    *,
    window_size: int = 3,
    minimum_pixels: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep candidates supported by a compact local candidate neighborhood."""
    candidate = np.asarray(candidate_mask, dtype=bool)
    if candidate.ndim != 2:
        raise ValueError("candidate_mask must be 2D")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least 3")
    maximum = window_size * window_size
    if minimum_pixels < 1 or minimum_pixels > maximum:
        raise ValueError("minimum_pixels must fit inside the support window")
    counts = ndimage.convolve(
        candidate.astype(np.uint16),
        np.ones((window_size, window_size), dtype=np.uint16),
        mode="constant",
        cval=0,
    )
    return candidate & (counts >= int(minimum_pixels)), counts


def seeded_candidate_region_grow(
    seed_mask: np.ndarray,
    growth_mask: np.ndarray,
) -> np.ndarray:
    """Grow spectral seeds only through connected pixels in a wider candidate mask.

    This is deliberately connectivity constrained: a permissive spectral pixel
    cannot create a new anomaly component unless it touches an accepted seed.
    """
    seeds = np.asarray(seed_mask, dtype=bool)
    domain = np.asarray(growth_mask, dtype=bool)
    if seeds.ndim != 2 or domain.shape != seeds.shape:
        raise ValueError("seed_mask and growth_mask must be matching 2D arrays")
    marker = seeds & domain
    if not np.any(marker):
        return np.zeros_like(seeds)
    return ndimage.binary_propagation(
        marker,
        structure=np.ones((3, 3), dtype=bool),
        mask=domain,
    )
