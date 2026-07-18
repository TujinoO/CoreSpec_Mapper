from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import csv
import json
import re

import numpy as np

from .algorithms import continuum_remove, spectral_angles
from .envi import SpectralLibrary


MINERAL_DISPLAY_NAMES = {
    "calcite": "Calcite",
    "dolomite": "Dolomite",
    "anhydrite": "Anhydrite",
    "gypsum": "Gypsum",
    "illite": "Illite",
    "montmorillonite": "Montmorillonite",
    "kaolinite": "Kaolinite",
}

MINERAL_GROUPS = {
    "carbonates": ("calcite", "dolomite"),
    "sulfates": ("anhydrite", "gypsum"),
    "clays": ("illite", "montmorillonite", "kaolinite"),
}

SELECTION_WINDOWS_NM = {
    "carbonates": ((2200.0, 2450.0),),
    "sulfates": ((1350.0, 1800.0), (1850.0, 2400.0)),
    "clays": ((1850.0, 2400.0),),
}

PRIMARY_FEATURE_WINDOWS_NM = {
    "carbonates": (2280.0, 2360.0),
    "sulfates": (1880.0, 1995.0),
    "clays": (2160.0, 2235.0),
}

SCENE_WINDOWS_NM = {
    "carbonates": ((2248.0, 2402.0),),
    "sulfates": ((1350.0, 1800.0), (1850.0, 2351.0)),
    "clays": ((2100.0, 2256.0),),
}

REFERENCE_FEATURE_CENTER_GATES_NM = {
    "calcite": (2325.0, 2350.0),
    "dolomite": (2295.0, 2330.0),
}

_TARGET_PATTERNS = {
    "calcite": re.compile(r"\bcalcite\b", re.IGNORECASE),
    "dolomite": re.compile(r"\bdolomit(?:e|ic)?\b", re.IGNORECASE),
    "anhydrite": re.compile(r"\banhydrit(?:e|ic)?\b", re.IGNORECASE),
    "gypsum": re.compile(r"\bgypsum\b", re.IGNORECASE),
    "illite": re.compile(r"\billite\b", re.IGNORECASE),
    "montmorillonite": re.compile(r"\bmontmorillonite\b|\bmontmor\w*\b", re.IGNORECASE),
    "kaolinite": re.compile(r"\bkaolinite\b|\bkaolini\w*\b", re.IGNORECASE),
}

_ROCK_TERMS = re.compile(
    r"\b(?:bearing|shale|marble|limestone|dolostone|rock|soil|mixture|mixed)\b",
    re.IGNORECASE,
)


@dataclass
class LibraryCandidate:
    mineral: str
    source_library: str
    source_index: int
    source_name: str
    resampled: np.ndarray | None = None
    embedding: np.ndarray | None = None
    coverage: float = 0.0
    absorption_depth: float = float("nan")
    feature_center_nm: float = float("nan")
    roughness: float = float("nan")
    quality: float = float("-inf")
    scene_angle_p05: float = float("nan")
    scene_bright_fraction: float = float("nan")
    scene_max_column_density: float = float("nan")
    scene_score: float = float("nan")
    selection_score: float = float("nan")
    rejection_reason: str | None = None
    duplicate_of: str | None = None
    selected: bool = False
    selected_rank: int | None = None

    @property
    def identifier(self) -> str:
        return f"{self.source_library}#{self.source_index}"

    def manifest_record(self) -> dict[str, Any]:
        return {
            "mineral": self.mineral,
            "mineral_name": MINERAL_DISPLAY_NAMES[self.mineral],
            "source_library": self.source_library,
            "source_index": self.source_index,
            "source_name": self.source_name,
            "coverage": self.coverage,
            "absorption_depth": self.absorption_depth,
            "feature_center_nm": self.feature_center_nm,
            "roughness": self.roughness,
            "quality": self.quality,
            "source_family": self.source_library.split("\\", 1)[0],
            "scene_angle_p05": self.scene_angle_p05,
            "scene_bright_fraction": self.scene_bright_fraction,
            "scene_max_column_density": self.scene_max_column_density,
            "scene_score": self.scene_score,
            "selection_score": self.selection_score,
            "rejection_reason": self.rejection_reason,
            "duplicate_of": self.duplicate_of,
            "selected": self.selected,
            "selected_rank": self.selected_rank,
        }


@dataclass(frozen=True)
class EnsembleLibrary:
    wavelengths_nm: np.ndarray
    spectra: np.ndarray
    mineral_labels: list[str]
    spectrum_names: list[str]
    selected_candidates: list[LibraryCandidate]
    candidates: list[LibraryCandidate]
    libraries: list[dict[str, Any]]
    group_reference_counts: dict[str, int]

    def group_indices(self, group_name: str) -> np.ndarray:
        minerals = set(MINERAL_GROUPS[group_name])
        return np.asarray([index for index, name in enumerate(self.mineral_labels) if name in minerals], dtype=np.int64)


def match_pure_target(name: str) -> tuple[str | None, str | None]:
    matches = [mineral for mineral, pattern in _TARGET_PATTERNS.items() if pattern.search(name)]
    if not matches:
        return None, None
    if len(matches) > 1:
        return matches[0], "mixed_target_minerals"
    mineral = matches[0]
    lowered = name.casefold()
    if _ROCK_TERMS.search(name):
        return mineral, "rock_or_mixture_name"
    if mineral == "dolomite" and "dolomitic" in lowered:
        return mineral, "rock_or_mixture_name"
    if mineral == "illite" and re.search(r"smec|illsmec|illite\s*[-+/]\s*smect", lowered):
        return mineral, "mixed_clay_name"
    if mineral == "montmorillonite" and re.search(r"\+\s*illi|with\s+illi", lowered):
        return mineral, "mixed_clay_name"
    if mineral == "kaolinite" and "halloysite" in lowered:
        return mineral, "mixed_clay_name"
    return mineral, None


def resample_spectrum(
    source_wavelengths_nm: Sequence[float],
    source_spectrum: Sequence[float],
    target_wavelengths_nm: Sequence[float],
    *,
    max_gap_nm: float | None = None,
) -> np.ndarray:
    wavelengths = np.asarray(source_wavelengths_nm, dtype=np.float64)
    values = np.asarray(source_spectrum, dtype=np.float64)
    target = np.asarray(target_wavelengths_nm, dtype=np.float64)
    valid = np.isfinite(wavelengths) & np.isfinite(values) & (values > 0.0) & (np.abs(values) < 1e6)
    wavelengths = wavelengths[valid]
    values = values[valid]
    if wavelengths.size < 2:
        return np.full(target.shape, np.nan, dtype=np.float64)

    order = np.argsort(wavelengths, kind="stable")
    wavelengths = wavelengths[order]
    values = values[order]
    unique, inverse = np.unique(wavelengths, return_inverse=True)
    if unique.size != wavelengths.size:
        sums = np.zeros(unique.size, dtype=np.float64)
        counts = np.zeros(unique.size, dtype=np.int64)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1)
        wavelengths = unique
        values = sums / np.maximum(counts, 1)

    spacing = np.diff(wavelengths)
    typical_spacing = float(np.median(spacing[spacing > 0])) if np.any(spacing > 0) else 1.0
    allowed_gap = float(max_gap_nm) if max_gap_nm is not None else max(45.0, 8.0 * typical_spacing)
    result = np.interp(target, wavelengths, values, left=np.nan, right=np.nan)
    right = np.searchsorted(wavelengths, target, side="left")
    exact = (right < wavelengths.size) & np.isclose(wavelengths[np.minimum(right, wavelengths.size - 1)], target)
    bracketed = (right > 0) & (right < wavelengths.size)
    left_index = np.clip(right - 1, 0, wavelengths.size - 1)
    right_index = np.clip(right, 0, wavelengths.size - 1)
    gap = wavelengths[right_index] - wavelengths[left_index]
    supported = exact | (bracketed & (gap <= allowed_gap))
    result[~supported] = np.nan
    return result


def _window_mask(wavelengths_nm: np.ndarray, windows: Sequence[Sequence[float]]) -> np.ndarray:
    selected = np.zeros(wavelengths_nm.size, dtype=bool)
    for lower, upper in windows:
        selected |= (wavelengths_nm >= float(lower)) & (wavelengths_nm <= float(upper))
    return selected


def _candidate_diagnostics(candidate: LibraryCandidate, wavelengths_nm: np.ndarray, group_name: str) -> None:
    assert candidate.resampled is not None
    selected = _window_mask(wavelengths_nm, SELECTION_WINDOWS_NM[group_name])
    candidate.coverage = float(np.mean(np.isfinite(candidate.resampled[selected])))
    if candidate.coverage < 0.97:
        candidate.rejection_reason = "insufficient_spectral_coverage"
        return

    lower = min(window[0] for window in SELECTION_WINDOWS_NM[group_name])
    upper = max(window[1] for window in SELECTION_WINDOWS_NM[group_name])
    contiguous = (wavelengths_nm >= lower) & (wavelengths_nm <= upper)
    spectrum = candidate.resampled[contiguous]
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum <= 0):
        candidate.rejection_reason = "invalid_values_in_analysis_window"
        return
    removed = continuum_remove(spectrum, wavelengths_nm[contiguous])
    absorption = np.clip(1.0 - removed, 0.0, None)
    depth = float(np.max(absorption))
    norm = float(np.linalg.norm(absorption))
    if not np.isfinite(depth) or depth < 0.003 or norm <= 1e-10:
        candidate.rejection_reason = "weak_or_invalid_absorption"
        return

    primary_lower, primary_upper = PRIMARY_FEATURE_WINDOWS_NM[group_name]
    primary_wavelengths = wavelengths_nm[contiguous]
    primary = (primary_wavelengths >= primary_lower) & (primary_wavelengths <= primary_upper)
    center = float(primary_wavelengths[primary][int(np.argmax(absorption[primary]))])
    second_difference = np.diff(absorption, n=2)
    roughness = float(np.median(np.abs(second_difference)) / max(depth, 1e-12)) if second_difference.size else 0.0
    candidate.absorption_depth = depth
    candidate.feature_center_nm = center
    candidate.roughness = roughness
    candidate.quality = candidate.coverage + min(depth, 0.35) - 0.15 * min(roughness, 2.0)
    candidate.embedding = absorption / norm


def _apply_reference_quality_gates(candidate: LibraryCandidate, group_name: str) -> None:
    if candidate.rejection_reason is not None:
        return
    if group_name == "carbonates":
        if candidate.absorption_depth < 0.05:
            candidate.rejection_reason = "weak_reference_absorption"
            return
        if candidate.roughness > 0.08:
            candidate.rejection_reason = "rough_reference_shape"
            return
    center_gate = REFERENCE_FEATURE_CENTER_GATES_NM.get(candidate.mineral)
    if center_gate is not None and not center_gate[0] <= candidate.feature_center_nm <= center_gate[1]:
        candidate.rejection_reason = "implausible_diagnostic_center"


def _rank_score(values: np.ndarray, *, lower_is_better: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, 0.5, dtype=np.float64)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) <= 1:
        return result
    finite_indices = np.flatnonzero(finite)
    order = np.argsort(values[finite], kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, order.size)
    if lower_is_better:
        ranks = 1.0 - ranks
    result[finite_indices] = ranks
    return result


def _score_scene_compatibility(
    candidates: list[LibraryCandidate],
    wavelengths_nm: np.ndarray,
    scene_cube: np.ndarray,
    scene_mask: np.ndarray,
    *,
    support_fraction: float,
) -> None:
    cube = np.asarray(scene_cube, dtype=np.float64)
    mask = np.asarray(scene_mask, dtype=bool)
    if cube.ndim != 3 or cube.shape[:2] != mask.shape or cube.shape[2] != wavelengths_nm.size:
        raise ValueError("Scene-selection cube and mask do not match target wavelengths")
    if not 0.0 < support_fraction < 0.5:
        raise ValueError("Scene support fraction must be between zero and 0.5")
    brightness_index = int(np.argmin(np.abs(wavelengths_nm - 1600.0)))

    for group_name, minerals in MINERAL_GROUPS.items():
        indices = _window_mask(wavelengths_nm, SCENE_WINDOWS_NM[group_name])
        valid = mask & np.all(np.isfinite(cube[..., indices]), axis=-1) & np.all(cube[..., indices] > 0, axis=-1)
        if not np.any(valid):
            continue
        pixels = cube[..., indices][valid]
        brightness = cube[..., brightness_index]
        bright_threshold = float(np.percentile(brightness[valid], 85.0))
        bright = valid & (brightness >= bright_threshold)
        column_valid = np.maximum(np.sum(valid, axis=0), 1)
        group_candidates = [
            candidate
            for candidate in candidates
            if candidate.mineral in minerals
            and candidate.rejection_reason is None
            and candidate.resampled is not None
            and np.all(np.isfinite(candidate.resampled[indices]))
        ]
        for candidate in group_candidates:
            angles = spectral_angles(pixels, candidate.resampled[indices][None])[:, 0]
            cutoff = float(np.percentile(angles, support_fraction * 100.0))
            support = np.zeros(valid.shape, dtype=bool)
            support[valid] = angles <= cutoff
            candidate.scene_angle_p05 = cutoff
            candidate.scene_bright_fraction = float(np.count_nonzero(support & bright) / max(np.count_nonzero(support), 1))
            candidate.scene_max_column_density = float(np.max(np.sum(support, axis=0) / column_valid))

        for mineral in minerals:
            mineral_candidates = [candidate for candidate in group_candidates if candidate.mineral == mineral]
            if not mineral_candidates:
                continue
            quality = _rank_score(np.asarray([candidate.quality for candidate in mineral_candidates]))
            angle = _rank_score(
                np.asarray([candidate.scene_angle_p05 for candidate in mineral_candidates]), lower_is_better=True
            )
            stripe = _rank_score(
                np.asarray([candidate.scene_max_column_density for candidate in mineral_candidates]), lower_is_better=True
            )
            bright_score = _rank_score(np.asarray([candidate.scene_bright_fraction for candidate in mineral_candidates]))
            if group_name == "carbonates":
                combined = 0.30 * quality + 0.25 * angle + 0.25 * stripe + 0.20 * bright_score
            else:
                combined = 0.40 * quality + 0.30 * angle + 0.30 * stripe
            for candidate, score in zip(mineral_candidates, combined):
                candidate.scene_score = float(score)


def _angle_matrix(embeddings: np.ndarray) -> np.ndarray:
    cosine = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    return np.arccos(cosine)


def _deduplicate(candidates: list[LibraryCandidate], threshold: float) -> list[LibraryCandidate]:
    if len(candidates) <= 1:
        return candidates.copy()
    embeddings = np.vstack([candidate.embedding for candidate in candidates])
    distances = _angle_matrix(embeddings)
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if distances[left, right] <= threshold:
                union(left, right)

    clusters: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        clusters.setdefault(find(index), []).append(index)
    representatives: list[LibraryCandidate] = []
    for members in clusters.values():
        representative_index = max(
            members,
            key=lambda index: (candidates[index].quality, -len(candidates[index].source_name), candidates[index].identifier),
        )
        representative = candidates[representative_index]
        representatives.append(representative)
        for index in members:
            if index != representative_index:
                candidates[index].rejection_reason = "near_duplicate_spectrum"
                candidates[index].duplicate_of = representative.identifier
    return sorted(representatives, key=lambda candidate: candidate.identifier.casefold())


def _select_medoids(candidates: list[LibraryCandidate], count: int) -> list[LibraryCandidate]:
    if count >= len(candidates):
        return candidates.copy()
    embeddings = np.vstack([candidate.embedding for candidate in candidates])
    distances = _angle_matrix(embeddings)
    medoids = [int(np.argmin(np.sum(distances, axis=1)))]
    while len(medoids) < count:
        nearest = np.min(distances[:, medoids], axis=1)
        nearest[medoids] = -np.inf
        medoids.append(int(np.argmax(nearest)))

    for _ in range(20):
        assignment = np.argmin(distances[:, medoids], axis=1)
        updated: list[int] = []
        for cluster_index, current in enumerate(medoids):
            members = np.flatnonzero(assignment == cluster_index)
            if members.size == 0:
                updated.append(current)
                continue
            within = distances[np.ix_(members, members)]
            updated.append(int(members[int(np.argmin(np.sum(within, axis=1)))]))
        if updated == medoids:
            break
        medoids = updated
    return [candidates[index] for index in sorted(set(medoids))]


def _select_source_diverse_representatives(
    candidates: list[LibraryCandidate],
    count: int,
) -> list[LibraryCandidate]:
    if count >= len(candidates):
        for candidate in candidates:
            candidate.selection_score = candidate.scene_score if np.isfinite(candidate.scene_score) else candidate.quality
        return candidates.copy()
    embeddings = np.vstack([candidate.embedding for candidate in candidates])
    distances = _angle_matrix(embeddings)
    typicality = _rank_score(np.median(distances, axis=1), lower_is_better=True)
    quality = _rank_score(np.asarray([candidate.quality for candidate in candidates]))
    scene = np.asarray([candidate.scene_score for candidate in candidates], dtype=np.float64)
    if np.any(np.isfinite(scene)):
        scene = np.where(np.isfinite(scene), scene, 0.5)
        base = 0.60 * scene + 0.25 * quality + 0.15 * typicality
    else:
        base = 0.60 * quality + 0.40 * typicality
    for candidate, score in zip(candidates, base):
        candidate.selection_score = float(score)

    families: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        family = candidate.source_library.split("\\", 1)[0].casefold()
        families.setdefault(family, []).append(index)
    family_best = [max(indices, key=lambda index: (base[index], candidates[index].quality)) for indices in families.values()]
    family_best.sort(key=lambda index: (base[index], candidates[index].quality), reverse=True)
    selected = family_best[:count]
    while len(selected) < count:
        best_index = -1
        best_score = -np.inf
        for index in range(len(candidates)):
            if index in selected:
                continue
            diversity = float(np.min(distances[index, selected])) if selected else 0.0
            score = float(base[index] + 0.20 * min(diversity / 0.5, 1.0))
            if score > best_score:
                best_index, best_score = index, score
        selected.append(best_index)
    return [candidates[index] for index in selected]


def build_library_ensemble(
    library_root: str | Path,
    target_wavelengths_nm: Sequence[float],
    *,
    representatives_per_mineral: int = 4,
    dedup_angle_rad: float = 0.02,
    scene_cube: np.ndarray | None = None,
    scene_mask: np.ndarray | None = None,
    scene_support_fraction: float = 0.05,
) -> EnsembleLibrary:
    root = Path(library_root)
    wavelengths = np.asarray(target_wavelengths_nm, dtype=np.float64)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if wavelengths.ndim != 1 or wavelengths.size < 3 or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Target wavelengths must be a strictly increasing one-dimensional vector")

    candidates: list[LibraryCandidate] = []
    libraries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.sli"), key=lambda item: str(item).casefold()):
        relative = str(path.relative_to(root))
        try:
            library = SpectralLibrary.open(path)
        except Exception as exc:
            libraries.append({"path": relative, "error": str(exc)})
            continue
        target_count = 0
        for index, name in enumerate(library.names):
            mineral, rejection = match_pure_target(name)
            if mineral is None:
                continue
            target_count += 1
            candidate = LibraryCandidate(
                mineral=mineral,
                source_library=relative,
                source_index=index,
                source_name=name,
                rejection_reason=rejection,
            )
            candidates.append(candidate)
            if rejection is not None:
                continue
            candidate.resampled = resample_spectrum(library.wavelengths_nm, library.spectra[index], wavelengths)
            group_name = next(group for group, minerals in MINERAL_GROUPS.items() if mineral in minerals)
            _candidate_diagnostics(candidate, wavelengths, group_name)
            _apply_reference_quality_gates(candidate, group_name)
        libraries.append(
            {
                "path": relative,
                "spectra": len(library.names),
                "samples": int(library.wavelengths_nm.size),
                "wavelength_min_nm": float(np.min(library.wavelengths_nm)),
                "wavelength_max_nm": float(np.max(library.wavelengths_nm)),
                "target_name_matches": target_count,
                "warning": library.warning,
            }
        )

    if (scene_cube is None) != (scene_mask is None):
        raise ValueError("Scene cube and scene mask must be provided together")
    if scene_cube is not None and scene_mask is not None:
        _score_scene_compatibility(
            candidates,
            wavelengths,
            scene_cube,
            scene_mask,
            support_fraction=scene_support_fraction,
        )

    selected: list[LibraryCandidate] = []
    group_counts: dict[str, int] = {}
    for group_name, minerals in MINERAL_GROUPS.items():
        unique_by_mineral: dict[str, list[LibraryCandidate]] = {}
        for mineral in minerals:
            valid = [candidate for candidate in candidates if candidate.mineral == mineral and candidate.rejection_reason is None]
            unique_by_mineral[mineral] = _deduplicate(valid, dedup_angle_rad)
            if not unique_by_mineral[mineral]:
                raise ValueError(f"No usable pure reference spectra remain for {mineral}")
        equal_count = min(representatives_per_mineral, *(len(unique_by_mineral[mineral]) for mineral in minerals))
        group_counts[group_name] = equal_count
        for mineral in minerals:
            representatives = _select_source_diverse_representatives(unique_by_mineral[mineral], equal_count)
            for rank, candidate in enumerate(representatives, start=1):
                candidate.selected = True
                candidate.selected_rank = rank
                selected.append(candidate)
            representative_ids = {candidate.identifier for candidate in representatives}
            for candidate in unique_by_mineral[mineral]:
                if candidate.identifier not in representative_ids and candidate.rejection_reason is None:
                    candidate.rejection_reason = "not_selected_as_representative"

    mineral_order = list(MINERAL_DISPLAY_NAMES)
    selected.sort(key=lambda candidate: (mineral_order.index(candidate.mineral), candidate.selected_rank or 0))
    spectra = np.vstack([candidate.resampled for candidate in selected])
    mineral_labels = [candidate.mineral for candidate in selected]
    spectrum_names = [
        f"{MINERAL_DISPLAY_NAMES[candidate.mineral]} | {candidate.source_library} #{candidate.source_index} | {candidate.source_name}"
        for candidate in selected
    ]
    return EnsembleLibrary(
        wavelengths_nm=wavelengths,
        spectra=spectra,
        mineral_labels=mineral_labels,
        spectrum_names=spectrum_names,
        selected_candidates=selected,
        candidates=candidates,
        libraries=libraries,
        group_reference_counts=group_counts,
    )


def _format_header_list(values: Sequence[Any]) -> str:
    return "{\n  " + ", ".join(str(value) for value in values) + "}"


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def write_ensemble_artifacts(ensemble: EnsembleLibrary, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "v3_automatic_library_ensemble.sli"
    header_path = data_path.with_suffix(".hdr")
    np.asarray(ensemble.spectra, dtype="<f4").tofile(data_path)
    safe_names = [name.replace(",", ";") for name in ensemble.spectrum_names]
    header = [
        "ENVI",
        "description = {CoreSpec Mapper V3 automatic pure-mineral library ensemble}",
        f"samples = {ensemble.wavelengths_nm.size}",
        f"lines = {ensemble.spectra.shape[0]}",
        "bands = 1",
        "header offset = 0",
        "file type = ENVI Spectral Library",
        "data type = 4",
        "interleave = bsq",
        "byte order = 0",
        "wavelength units = Nanometers",
        "wavelength = " + _format_header_list([f"{value:.6f}" for value in ensemble.wavelengths_nm]),
        "spectra names = " + _format_header_list(safe_names),
    ]
    header_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    records = [candidate.manifest_record() for candidate in ensemble.candidates]
    for record in records:
        for key in (
            "coverage",
            "absorption_depth",
            "feature_center_nm",
            "roughness",
            "quality",
            "scene_angle_p05",
            "scene_bright_fraction",
            "scene_max_column_density",
            "scene_score",
            "selection_score",
        ):
            record[key] = _json_number(record[key])
    manifest = {
        "version": "V3 Automatic Library Ensemble",
        "selected_spectra": len(ensemble.selected_candidates),
        "group_reference_count_per_mineral": ensemble.group_reference_counts,
        "selected_counts": {
            mineral: sum(candidate.selected and candidate.mineral == mineral for candidate in ensemble.candidates)
            for mineral in MINERAL_DISPLAY_NAMES
        },
        "libraries": ensemble.libraries,
        "candidates": records,
    }
    manifest_path = output / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = output / "selection_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return {"library": str(data_path), "header": str(header_path), "manifest": str(manifest_path), "csv": str(csv_path)}
