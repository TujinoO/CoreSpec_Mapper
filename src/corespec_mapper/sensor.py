from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence
import json

import numpy as np

from .catalog import MineralCatalog
from .envi import EnviDataset
from .v4_models import MineralSupport, QualityFlag, SensorCapabilityCard, Severity, SupportLevel


def _normalise_spectral_values(values: Any, units: str) -> np.ndarray | None:
    if not isinstance(values, list) or not values:
        return None
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        return None
    unit = units.casefold()
    if "micro" in unit and float(np.nanmax(result)) <= 100.0:
        result = result * 1000.0
    elif float(np.nanmax(result)) <= 20.0:
        result = result * 1000.0
    return result


def extract_fwhm_nm(dataset: EnviDataset) -> tuple[np.ndarray | None, str]:
    metadata = dataset.info.metadata
    values = metadata.get("fwhm")
    units = str(metadata.get("fwhm units", metadata.get("wavelength units", "unknown")))
    fwhm = _normalise_spectral_values(values, units)
    if fwhm is None or fwhm.size != dataset.info.bands or np.any(fwhm <= 0):
        return None, "unknown"
    return fwhm, "measured"


def infer_spectral_domain(wavelengths_nm: Sequence[float]) -> str:
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    lower, upper = float(np.nanmin(wavelengths)), float(np.nanmax(wavelengths))
    domains: list[str] = []
    if lower <= 750.0 and upper >= 450.0:
        domains.append("vnir")
    if lower <= 1700.0 and upper >= 900.0:
        domains.append("nir")
    if lower <= 2450.0 and upper >= 2000.0:
        domains.append("swir")
    if lower <= 12000.0 and upper >= 8000.0:
        domains.append("tir")
    if not domains:
        return "unknown"
    return domains[0] if len(domains) == 1 else "+".join(domains)


def infer_data_physics(dataset: EnviDataset, override: str | None = None) -> str:
    if override:
        return str(override).strip().casefold()
    metadata = dataset.info.metadata
    text = " ".join(
        str(metadata.get(key, ""))
        for key in ("data units", "units", "description", "band names", "file type")
    ).casefold()
    if "brightness temperature" in text or "temperature" in text:
        return "brightness_temperature"
    if "radiance" in text or "w/(" in text:
        return "radiance"
    if "emissiv" in text:
        return "emissivity"
    if "reflect" in text:
        return "reflectance"
    return "unknown"


def _sample_quality(sample_cube: np.ndarray | None, sample_mask: np.ndarray | None) -> tuple[float | None, tuple[int, ...]]:
    if sample_cube is None or sample_mask is None:
        return None, ()
    cube = np.asarray(sample_cube, dtype=np.float64)
    mask = np.asarray(sample_mask, dtype=bool)
    if cube.ndim != 3 or cube.shape[:2] != mask.shape or not np.any(mask):
        return None, ()
    values = cube[mask]
    finite_fraction = np.mean(np.isfinite(values), axis=0)
    filled = np.where(np.isfinite(values), values, np.nan)
    with np.errstate(invalid="ignore"):
        band_median = np.nanmedian(filled, axis=0)
    if values.shape[1] >= 5:
        smooth = np.empty_like(band_median)
        smooth[0] = band_median[0]
        smooth[-1] = band_median[-1]
        smooth[1:-1] = (band_median[:-2] + 2.0 * band_median[1:-1] + band_median[2:]) / 4.0
        residual = band_median - smooth
        noise = 1.4826 * float(np.nanmedian(np.abs(residual - np.nanmedian(residual))))
    else:
        noise = float("nan")
    signal = float(np.nanmedian(np.abs(band_median)))
    snr = signal / noise if np.isfinite(noise) and noise > 1e-12 else None
    per_band_noise = 1.4826 * np.nanmedian(np.abs(filled - band_median[None, :]), axis=0)
    noise_limit = float(np.nanmedian(per_band_noise) + 6.0 * np.nanmedian(np.abs(per_band_noise - np.nanmedian(per_band_noise))))
    bad = (finite_fraction < 0.98) | ~np.isfinite(band_median)
    if np.isfinite(noise_limit) and noise_limit > 0:
        bad |= per_band_noise > noise_limit
    return (None if snr is None or not np.isfinite(snr) else float(snr)), tuple(int(index) for index in np.flatnonzero(bad))


def build_sensor_capability_card(
    dataset: EnviDataset,
    catalog: MineralCatalog,
    *,
    sample_cube: np.ndarray | None = None,
    sample_mask: np.ndarray | None = None,
    data_physics: str | None = None,
    preprocessing_state: str = "unknown",
) -> SensorCapabilityCard:
    wavelengths = dataset.info.wavelengths_nm
    if wavelengths is None or wavelengths.size != dataset.info.bands:
        raise ValueError("A valid ENVI wavelength vector is required")
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if not np.all(np.isfinite(wavelengths)) or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Image wavelengths must be finite and strictly increasing")
    fwhm, fwhm_status = extract_fwhm_nm(dataset)
    snr, bad = _sample_quality(sample_cube, sample_mask)
    flags: list[QualityFlag] = []
    if fwhm is None:
        flags.append(QualityFlag("FWHM_UNKNOWN", Severity.WARNING, "FWHM is unavailable; constrained interpolation will be used."))
    if snr is None:
        flags.append(QualityFlag("SNR_NOT_ESTIMATED", Severity.WARNING, "SNR could not be estimated from the available sample."))
    elif snr < 50.0:
        flags.append(QualityFlag("LOW_ESTIMATED_SNR", Severity.WARNING, f"Estimated SNR is low ({snr:.1f})."))
    physics = infer_data_physics(dataset, data_physics)
    if physics == "unknown":
        flags.append(QualityFlag("DATA_PHYSICS_ASSUMED", Severity.WARNING, "Data physics is not declared; reflectance must be confirmed."))
        physics = "reflectance"
    signature_value = {
        "wavelengths_nm": np.round(wavelengths, 6).tolist(),
        "fwhm_nm": None if fwhm is None else np.round(fwhm, 6).tolist(),
        "band_count": dataset.info.bands,
        "spectral_domain": infer_spectral_domain(wavelengths),
        "data_type": dataset.info.data_type,
        "interleave": dataset.info.interleave,
        "data_physics": physics,
        "scanner_model": str(dataset.info.metadata.get("sensor type", dataset.info.metadata.get("sensor", "unknown"))),
    }
    signature = sha256(json.dumps(signature_value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    implemented = tuple(
        expert.expert_id
        for expert in catalog.experts.values()
        if expert.implemented and physics in expert.data_physics
    )
    return SensorCapabilityCard(
        sensor_signature=signature,
        spectral_domain=infer_spectral_domain(wavelengths),
        data_physics=physics,
        wavelength_min_nm=float(wavelengths[0]),
        wavelength_max_nm=float(wavelengths[-1]),
        band_count=int(wavelengths.size),
        valid_band_count=int(wavelengths.size - len(bad)),
        median_spacing_nm=float(np.median(np.diff(wavelengths))),
        fwhm_status=fwhm_status,
        fwhm_median_nm=None if fwhm is None else float(np.median(fwhm)),
        estimated_snr=snr,
        bad_band_indices=bad,
        detector_axis="samples",
        preprocessing_state=preprocessing_state,
        implemented_experts=implemented,
        quality_flags=tuple(flags),
    )


def _window_coverage(wavelengths: np.ndarray, windows: Sequence[Sequence[float]], bad: set[int]) -> tuple[float, int]:
    required_width = 0.0
    covered_width = 0.0
    valid_bands = 0
    for lower, upper in windows:
        lower, upper = float(lower), float(upper)
        required_width += upper - lower
        covered_width += max(0.0, min(float(wavelengths[-1]), upper) - max(float(wavelengths[0]), lower))
        selected = np.flatnonzero((wavelengths >= lower) & (wavelengths <= upper))
        valid_bands += sum(int(index) not in bad for index in selected)
    return covered_width / max(required_width, 1e-12), valid_bands


def resolve_mineral_support(
    card: SensorCapabilityCard,
    catalog: MineralCatalog,
    *,
    wavelengths_nm: Sequence[float],
    reference_counts: Mapping[str, int] | None = None,
) -> dict[str, MineralSupport]:
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    bad = set(card.bad_band_indices)
    counts = {str(key).casefold(): int(value) for key, value in (reference_counts or {}).items()}
    result: dict[str, MineralSupport] = {}
    for mineral in catalog.minerals.values():
        reasons: list[str] = []
        chosen_expert: str | None = None
        coverage = 0.0
        valid_bands = 0
        sampling_score = 0.0
        for expert_id, requirements in mineral.experts.items():
            expert = catalog.experts[expert_id]
            if not expert.implemented:
                reasons.append(f"{expert.display_name} is registered but not implemented")
                continue
            if card.data_physics not in expert.data_physics:
                reasons.append(f"{card.data_physics} data are incompatible with {expert.display_name}")
                continue
            current_coverage, current_bands = _window_coverage(
                wavelengths, requirements.get("required_windows_nm", ()), bad
            )
            if current_coverage > coverage:
                coverage, valid_bands, chosen_expert = current_coverage, current_bands, expert_id
                minimum_bands = int(requirements.get("minimum_valid_bands", 3))
                sampling_score = min(1.0, current_bands / max(minimum_bands, 1))
        reference_count = counts.get(mineral.mineral_id, 0)
        if chosen_expert is None:
            level = SupportLevel.UNSUPPORTED
            score = 0.0
        else:
            requirements = mineral.experts[chosen_expert]
            minimum_bands = int(requirements.get("minimum_valid_bands", 3))
            if coverage < 0.97:
                reasons.append(f"required wavelength coverage is only {coverage:.1%}")
            if valid_bands < minimum_bands:
                reasons.append(f"only {valid_bands} valid diagnostic bands; {minimum_bands} required")
            max_fwhm = float(requirements.get("maximum_effective_fwhm_nm", np.inf))
            if card.fwhm_median_nm is not None and card.fwhm_median_nm > max_fwhm:
                reasons.append(f"median FWHM {card.fwhm_median_nm:.1f} nm exceeds {max_fwhm:.1f} nm")
            if reference_counts is not None and reference_count < 1:
                reasons.append("no usable pure reference spectrum")
            snr_score = 0.65 if card.estimated_snr is None else min(1.0, card.estimated_snr / 150.0)
            reference_score = (
                0.50
                if reference_counts is None
                else 0.0 if reference_count == 0 else min(1.0, reference_count / 3.0)
            )
            score = 0.30 * coverage + 0.20 * sampling_score + 0.20 * snr_score + 0.15 * 0.75 + 0.15 * reference_score
            hard_pass = coverage >= 0.97 and valid_bands >= minimum_bands and (reference_counts is None or reference_count > 0)
            if card.fwhm_median_nm is not None:
                hard_pass &= card.fwhm_median_nm <= max_fwhm
            if reference_counts is not None and reference_count < 1:
                level = SupportLevel.UNSUPPORTED
            elif hard_pass and score >= 0.75:
                level = SupportLevel.SUPPORTED
            elif hard_pass or (coverage >= 0.90 and valid_bands >= max(3, minimum_bands // 2)):
                level = SupportLevel.CONDITIONAL
            else:
                level = SupportLevel.UNSUPPORTED
            if reference_counts is None and level == SupportLevel.SUPPORTED:
                level = SupportLevel.CONDITIONAL
            if reference_counts is None and level != SupportLevel.UNSUPPORTED:
                reasons.append("reference library availability has not yet been audited")
        if level == SupportLevel.SUPPORTED and not reasons:
            reasons.append("required diagnostic coverage, sampling, data physics, and references are available")
        result[mineral.mineral_id] = MineralSupport(
            mineral_id=mineral.mineral_id,
            display_name=f"{mineral.display_name_en} {mineral.display_name_zh}".strip(),
            level=level,
            score=float(np.clip(score, 0.0, 1.0)),
            selected_expert=chosen_expert,
            reasons=tuple(reasons),
            required_windows_covered=float(coverage),
            valid_bands_in_required_windows=int(valid_bands),
            reference_count=reference_count,
        )
    return result
