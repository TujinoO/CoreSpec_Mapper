from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json

import numpy as np

from .algorithms import spectral_angles
from .catalog import MineralCatalog
from .envi import SpectralLibrary
from .library_ensemble import resample_spectrum
from .swir_expert import continuum_absorption, feature_values, unit_rows


@dataclass
class V4LibraryCandidate:
    mineral_id: str
    group_id: str
    source_library: str
    source_index: int
    source_name: str
    resampling_method: str = ""
    rejection_reason: str | None = None
    coverage: float = float("nan")
    absorption_depth: float = float("nan")
    primary_center_nm: float = float("nan")
    roughness: float = float("nan")
    intrinsic_quality: float = float("nan")
    diagnostic_quality: float = float("nan")
    confuser_separability: float = float("nan")
    scene_support: float = float("nan")
    fixed_column_risk: float = float("nan")
    edge_risk: float = float("nan")
    selection_score: float = float("nan")
    anchor: bool = False
    subclass_stable: bool = True
    selected: bool = False
    selected_rank: int | None = None
    duplicate_of: str | None = None
    resampled: np.ndarray | None = field(default=None, repr=False)
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def identifier(self) -> str:
        return f"{self.source_library}#{self.source_index}"

    @property
    def source_family(self) -> str:
        return self.source_library.replace("/", "\\").split("\\", 1)[0].casefold()

    def to_record(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "mineral_id": self.mineral_id,
            "group_id": self.group_id,
            "source_library": self.source_library,
            "source_index": self.source_index,
            "source_name": self.source_name,
            "resampling_method": self.resampling_method,
            "anchor": self.anchor,
            "coverage": _finite_or_none(self.coverage),
            "absorption_depth": _finite_or_none(self.absorption_depth),
            "primary_center_nm": _finite_or_none(self.primary_center_nm),
            "roughness": _finite_or_none(self.roughness),
            "intrinsic_quality": _finite_or_none(self.intrinsic_quality),
            "diagnostic_quality": _finite_or_none(self.diagnostic_quality),
            "confuser_separability": _finite_or_none(self.confuser_separability),
            "scene_support": _finite_or_none(self.scene_support),
            "fixed_column_risk": _finite_or_none(self.fixed_column_risk),
            "edge_risk": _finite_or_none(self.edge_risk),
            "selection_score": _finite_or_none(self.selection_score),
            "subclass_stable": self.subclass_stable,
            "selected": self.selected,
            "selected_rank": self.selected_rank,
            "rejection_reason": self.rejection_reason,
            "duplicate_of": self.duplicate_of,
        }


@dataclass(frozen=True)
class V4EnsembleLibrary:
    wavelengths_nm: np.ndarray
    spectra: np.ndarray
    mineral_labels: tuple[str, ...]
    group_labels: tuple[str, ...]
    spectrum_names: tuple[str, ...]
    selected_candidates: tuple[V4LibraryCandidate, ...]
    candidates: tuple[V4LibraryCandidate, ...]
    libraries: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]

    def group_indices(self, group_id: str) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.group_labels, dtype=object) == group_id)

    def reference_counts(self) -> dict[str, int]:
        return {
            mineral: sum(candidate.mineral_id == mineral for candidate in self.selected_candidates)
            for mineral in sorted(set(self.mineral_labels))
        }


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _window_mask(wavelengths: np.ndarray, windows: Sequence[Sequence[float]]) -> np.ndarray:
    selected = np.zeros(wavelengths.size, dtype=bool)
    for lower, upper in windows:
        selected |= (wavelengths >= float(lower)) & (wavelengths <= float(upper))
    return selected


def resample_with_fwhm(
    source_wavelengths_nm: Sequence[float],
    source_values: Sequence[float],
    target_wavelengths_nm: Sequence[float],
    target_fwhm_nm: Sequence[float] | None,
) -> tuple[np.ndarray, str]:
    source_wavelengths = np.asarray(source_wavelengths_nm, dtype=np.float64)
    source = np.asarray(source_values, dtype=np.float64)
    target = np.asarray(target_wavelengths_nm, dtype=np.float64)
    if target_fwhm_nm is None:
        return resample_spectrum(source_wavelengths, source, target), "gap_constrained_interpolation"
    fwhm = np.asarray(target_fwhm_nm, dtype=np.float64)
    if fwhm.shape != target.shape or np.any(~np.isfinite(fwhm)) or np.any(fwhm <= 0):
        raise ValueError("FWHM must contain one positive value per target band")
    valid = np.isfinite(source_wavelengths) & np.isfinite(source) & (source > 0)
    source_wavelengths = source_wavelengths[valid]
    source = source[valid]
    order = np.argsort(source_wavelengths, kind="stable")
    source_wavelengths, source = source_wavelengths[order], source[order]
    output = np.full(target.shape, np.nan, dtype=np.float64)
    for index, (center, width) in enumerate(zip(target, fwhm)):
        sigma = width / 2.354820045
        selected = np.abs(source_wavelengths - center) <= 3.0 * sigma
        if np.count_nonzero(selected) < 2:
            continue
        local_wavelengths = source_wavelengths[selected]
        if np.max(np.diff(local_wavelengths)) > max(45.0, 8.0 * float(np.median(np.diff(source_wavelengths)))):
            continue
        weights = np.exp(-0.5 * ((local_wavelengths - center) / sigma) ** 2)
        output[index] = float(np.sum(source[selected] * weights) / np.sum(weights))
    return output, "gaussian_fwhm_convolution"


def _candidate_diagnostics(
    candidate: V4LibraryCandidate,
    catalog: MineralCatalog,
    wavelengths: np.ndarray,
) -> None:
    if candidate.resampled is None:
        candidate.rejection_reason = "resampling_failed"
        return
    group = catalog.group(candidate.group_id)
    selected = _window_mask(wavelengths, group.classification_windows_nm)
    finite = np.isfinite(candidate.resampled[selected])
    candidate.coverage = float(np.mean(finite)) if finite.size else 0.0
    if candidate.coverage < 0.97:
        candidate.rejection_reason = "insufficient_spectral_coverage"
        return
    spectrum = candidate.resampled[selected]
    if np.any(~np.isfinite(spectrum)) or np.any(spectrum <= 0):
        candidate.rejection_reason = "invalid_values_in_analysis_window"
        return
    absorption, depth = continuum_absorption(spectrum[None, :], wavelengths[selected])
    candidate.absorption_depth = float(depth[0])
    if candidate.absorption_depth < 0.003 or np.linalg.norm(absorption[0]) <= 1e-10:
        candidate.rejection_reason = "weak_or_invalid_absorption"
        return
    second = np.diff(absorption[0], n=2)
    candidate.roughness = float(np.median(np.abs(second)) / max(candidate.absorption_depth, 1e-12)) if second.size else 0.0
    if candidate.roughness > 0.15:
        candidate.rejection_reason = "rough_reference_shape"
        return
    features = feature_values(group, absorption, wavelengths[selected])
    mineral = catalog.mineral(candidate.mineral_id)
    gate = mineral.experts[group.expert_id].get("feature_gate", {})
    center_id = gate.get("center_feature")
    if center_id and center_id in features:
        candidate.primary_center_nm = float(features[center_id][0])
        window = gate.get("center_window_nm")
        if window and not float(window[0]) <= candidate.primary_center_nm <= float(window[1]):
            candidate.rejection_reason = "implausible_diagnostic_center"
            return
    elif group.features:
        center_feature = next((item for item in group.features if item.kind == "center_nm"), None)
        if center_feature is not None:
            candidate.primary_center_nm = float(features[center_feature.feature_id][0])
    candidate.intrinsic_quality = float(
        np.clip(0.55 * candidate.coverage + 0.30 * min(candidate.absorption_depth / 0.15, 1.0) + 0.15 * (1.0 - min(candidate.roughness / 0.15, 1.0)), 0.0, 1.0)
    )
    candidate.diagnostic_quality = float(np.clip(candidate.absorption_depth / 0.10, 0.0, 1.0))
    candidate.embedding = unit_rows(absorption)[0]


def _mark_confuser_separability(candidates: list[V4LibraryCandidate], catalog: MineralCatalog) -> None:
    valid = [candidate for candidate in candidates if candidate.rejection_reason is None and candidate.embedding is not None]
    for candidate in valid:
        confusers = set(catalog.mineral(candidate.mineral_id).confusers)
        comparisons = [item for item in valid if item.mineral_id in confusers and item.group_id == candidate.group_id]
        if not comparisons:
            candidate.confuser_separability = 0.5
            continue
        angles = spectral_angles(candidate.embedding[None, :], np.vstack([item.embedding for item in comparisons]))[0]
        separation = float(np.min(angles))
        candidate.confuser_separability = float(np.clip(separation / 0.12, 0.0, 1.0))
        candidate.subclass_stable = separation >= 0.012


def _scene_scores(
    candidates: list[V4LibraryCandidate],
    catalog: MineralCatalog,
    wavelengths: np.ndarray,
    scene_cube: np.ndarray,
    scene_mask: np.ndarray,
) -> None:
    cube = np.asarray(scene_cube, dtype=np.float64)
    mask = np.asarray(scene_mask, dtype=bool)
    for group_id in sorted({candidate.group_id for candidate in candidates}):
        group = catalog.group(group_id)
        indices = _window_mask(wavelengths, group.detection_windows_nm)
        valid = mask & np.all(np.isfinite(cube[..., indices]), axis=-1) & np.all(cube[..., indices] > 0, axis=-1)
        if not np.any(valid):
            continue
        pixels = cube[..., indices][valid]
        column_valid = np.maximum(np.sum(valid, axis=0), 1)
        for candidate in candidates:
            if candidate.group_id != group_id or candidate.rejection_reason is not None or candidate.resampled is None:
                continue
            reference = candidate.resampled[indices]
            if np.any(~np.isfinite(reference)):
                continue
            angles = spectral_angles(pixels, reference[None, :])[:, 0]
            cutoff = float(np.percentile(angles, 5.0))
            support = np.zeros(valid.shape, dtype=bool)
            support[valid] = angles <= cutoff
            density = np.sum(support, axis=0) / column_valid
            median_density = float(np.median(density))
            maximum_density = float(np.max(density))
            candidate.fixed_column_risk = float(np.clip((maximum_density - median_density) / 0.20, 0.0, 1.0))
            candidate.scene_support = float(np.clip(1.0 - cutoff / 0.20, 0.0, 1.0))
            edge_columns = max(1, cube.shape[1] // 32)
            edge_hits = np.count_nonzero(support[:, :edge_columns]) + np.count_nonzero(support[:, -edge_columns:])
            candidate.edge_risk = float(edge_hits / max(np.count_nonzero(support), 1))


def _deduplicate(candidates: list[V4LibraryCandidate], threshold: float) -> list[V4LibraryCandidate]:
    ordered = sorted(candidates, key=lambda item: (not item.anchor, -item.intrinsic_quality, item.identifier.casefold()))
    kept: list[V4LibraryCandidate] = []
    for candidate in ordered:
        duplicate = next(
            (
                other
                for other in kept
                if not (candidate.anchor and other.anchor)
                if float(spectral_angles(candidate.embedding[None, :], other.embedding[None, :])[0, 0]) <= threshold
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
        else:
            candidate.rejection_reason = "near_duplicate_spectrum"
            candidate.duplicate_of = duplicate.identifier
    return kept


def _select_representatives(candidates: list[V4LibraryCandidate], maximum: int) -> list[V4LibraryCandidate]:
    if not candidates:
        return []
    desired = min(maximum, max(1, int(np.ceil(np.sqrt(len(candidates))))))
    anchor_count = min(sum(candidate.anchor for candidate in candidates), maximum)
    desired = max(desired, anchor_count)
    if anchor_count >= 4:
        desired = anchor_count
    for candidate in candidates:
        candidate.selection_score = float(np.clip(
            0.25 * candidate.intrinsic_quality
            + 0.20 * candidate.diagnostic_quality
            + 0.20 * (candidate.confuser_separability if np.isfinite(candidate.confuser_separability) else 0.5)
            + 0.15 * (candidate.scene_support if np.isfinite(candidate.scene_support) else 0.5)
            + 0.10 * 0.75
            + 0.10 * 0.5
            - 0.20 * (candidate.fixed_column_risk if np.isfinite(candidate.fixed_column_risk) else 0.0)
            - 0.10 * (candidate.edge_risk if np.isfinite(candidate.edge_risk) else 0.0),
            0.0,
            1.0,
        ))
    if anchor_count:
        useful_non_anchors = sum(
            (not candidate.anchor)
            and (
                not np.isfinite(candidate.fixed_column_risk)
                or candidate.fixed_column_risk < 0.80
                or (np.isfinite(candidate.scene_support) and candidate.scene_support >= 0.35)
            )
            for candidate in candidates
        )
        desired = min(desired, anchor_count + useful_non_anchors)
    selected = sorted(
        [candidate for candidate in candidates if candidate.anchor],
        key=lambda item: (-item.selection_score, item.identifier.casefold()),
    )[:desired]
    families = {candidate.source_family for candidate in selected}
    while len(selected) < desired:
        remaining = [candidate for candidate in candidates if candidate not in selected]
        if not remaining:
            break
        best: V4LibraryCandidate | None = None
        best_score = -np.inf
        for candidate in remaining:
            diversity = min(
                (float(spectral_angles(candidate.embedding[None, :], item.embedding[None, :])[0, 0]) for item in selected),
                default=0.12,
            )
            source_gain = 0.12 if candidate.source_family not in families else 0.0
            score = candidate.selection_score + source_gain + 0.12 * min(diversity / 0.20, 1.0)
            if score > best_score:
                best, best_score = candidate, score
        assert best is not None
        selected.append(best)
        families.add(best.source_family)
    return selected


def build_v4_library_ensemble(
    library_root: str | Path,
    target_wavelengths_nm: Sequence[float],
    catalog: MineralCatalog,
    mineral_ids: Sequence[str],
    *,
    target_fwhm_nm: Sequence[float] | None = None,
    scene_cube: np.ndarray | None = None,
    scene_mask: np.ndarray | None = None,
    maximum_representatives: int = 6,
    dedup_angle_rad: float = 0.02,
    allow_missing: bool = False,
) -> V4EnsembleLibrary:
    root = Path(library_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    wavelengths = np.asarray(target_wavelengths_nm, dtype=np.float64)
    allowed = tuple(dict.fromkeys(str(item).casefold() for item in mineral_ids))
    if not allowed:
        raise ValueError("At least one mineral is required for library construction")
    candidates: list[V4LibraryCandidate] = []
    libraries: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in sorted(root.rglob("*.sli"), key=lambda item: str(item).casefold()):
        relative = str(path.relative_to(root))
        try:
            library = SpectralLibrary.open(path)
        except Exception as exc:
            libraries.append({"path": relative, "error": str(exc)})
            continue
        matches = 0
        for index, name in enumerate(library.names):
            mineral_id, rejection = catalog.match_spectrum_name(name, allowed)
            if mineral_id is None:
                continue
            matches += 1
            mineral = catalog.mineral(mineral_id)
            candidate = V4LibraryCandidate(mineral_id, mineral.group_id, relative, index, name, rejection_reason=rejection)
            candidate.anchor = any(
                (
                    "#" in anchor
                    and anchor.replace("/", "\\").casefold() == candidate.identifier.replace("/", "\\").casefold()
                )
                or ("#" not in anchor and anchor.casefold() in name.casefold())
                for anchor in mineral.canonical_anchors
            )
            candidates.append(candidate)
            if rejection is not None:
                continue
            candidate.resampled, candidate.resampling_method = resample_with_fwhm(
                library.wavelengths_nm, library.spectra[index], wavelengths, target_fwhm_nm
            )
            _candidate_diagnostics(candidate, catalog, wavelengths)
        libraries.append({
            "path": relative,
            "spectra": len(library.names),
            "samples": int(library.wavelengths_nm.size),
            "wavelength_min_nm": float(np.min(library.wavelengths_nm)),
            "wavelength_max_nm": float(np.max(library.wavelengths_nm)),
            "target_name_matches": matches,
            "warning": library.warning,
        })
    _mark_confuser_separability(candidates, catalog)
    if (scene_cube is None) != (scene_mask is None):
        raise ValueError("Scene cube and scene mask must be supplied together")
    if scene_cube is not None and scene_mask is not None:
        _scene_scores(candidates, catalog, wavelengths, scene_cube, scene_mask)
    if target_fwhm_nm is None:
        warnings.append("FWHM unavailable: reference spectra used gap-constrained interpolation")

    selected: list[V4LibraryCandidate] = []
    for mineral_id in allowed:
        valid = [candidate for candidate in candidates if candidate.mineral_id == mineral_id and candidate.rejection_reason is None]
        unique = _deduplicate(valid, dedup_angle_rad)
        representatives = _select_representatives(unique, min(max(int(maximum_representatives), 1), 6))
        for rank, candidate in enumerate(representatives, start=1):
            candidate.selected = True
            candidate.selected_rank = rank
            selected.append(candidate)
        selected_ids = {candidate.identifier for candidate in representatives}
        for candidate in unique:
            if candidate.identifier not in selected_ids and candidate.rejection_reason is None:
                candidate.rejection_reason = "not_selected_as_representative"
    missing = [mineral for mineral in allowed if not any(item.mineral_id == mineral for item in selected)]
    if missing and not allow_missing:
        raise ValueError(f"No usable pure reference spectra remain for: {', '.join(missing)}")
    if missing:
        warnings.append(f"No usable pure reference spectra remain for: {', '.join(missing)}")
    if not selected:
        raise ValueError("No usable pure reference spectra remain for the requested mineral set")
    selected.sort(key=lambda item: (allowed.index(item.mineral_id), item.selected_rank or 0, item.identifier.casefold()))
    spectra = np.vstack([candidate.resampled for candidate in selected]).astype(np.float64)
    names = tuple(
        f"{catalog.mineral(candidate.mineral_id).display_name_en} | {candidate.source_library} #{candidate.source_index} | {candidate.source_name}"
        for candidate in selected
    )
    return V4EnsembleLibrary(
        wavelengths_nm=wavelengths,
        spectra=spectra,
        mineral_labels=tuple(candidate.mineral_id for candidate in selected),
        group_labels=tuple(candidate.group_id for candidate in selected),
        spectrum_names=names,
        selected_candidates=tuple(selected),
        candidates=tuple(candidates),
        libraries=tuple(libraries),
        warnings=tuple(warnings),
    )


def write_v4_library_artifacts(ensemble: V4EnsembleLibrary, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "automatic_library_ensemble.sli"
    header_path = data_path.with_suffix(".hdr")
    np.asarray(ensemble.spectra, dtype="<f4").tofile(data_path)
    safe_names = [name.replace(",", ";") for name in ensemble.spectrum_names]
    values = lambda items: "{\n  " + ", ".join(str(item) for item in items) + "}"
    header_path.write_text("\n".join([
        "ENVI",
        "description = {CoreSpec Mapper V4 automatic spectral library ensemble}",
        f"samples = {ensemble.wavelengths_nm.size}",
        f"lines = {ensemble.spectra.shape[0]}",
        "bands = 1",
        "header offset = 0",
        "file type = ENVI Spectral Library",
        "data type = 4",
        "interleave = bsq",
        "byte order = 0",
        "wavelength units = Nanometers",
        "wavelength = " + values([f"{item:.6f}" for item in ensemble.wavelengths_nm]),
        "spectra names = " + values(safe_names),
    ]) + "\n", encoding="utf-8")
    records = [candidate.to_record() for candidate in ensemble.candidates]
    manifest = {
        "version": "CoreSpec Mapper V4",
        "selected_spectra": len(ensemble.selected_candidates),
        "selected_counts": ensemble.reference_counts(),
        "selection_modes": {
            mineral: (
                "canonical_anchor_fallback"
                if selected and all(candidate.anchor for candidate in selected)
                else "anchor_plus_automatic" if any(candidate.anchor for candidate in selected) else "automatic_ensemble"
            )
            for mineral in sorted(set(ensemble.mineral_labels))
            for selected in [[candidate for candidate in ensemble.selected_candidates if candidate.mineral_id == mineral]]
        },
        "warnings": list(ensemble.warnings),
        "libraries": list(ensemble.libraries),
        "candidates": records,
    }
    manifest_path = output / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output / "selection_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return {"library": str(data_path), "header": str(header_path), "manifest": str(manifest_path), "csv": str(csv_path)}
