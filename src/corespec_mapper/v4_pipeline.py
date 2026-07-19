from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping, Sequence
import csv
import json

import numpy as np

from .algorithms import robust_column_bias, savgol_smooth, spectral_angles
from .artifacts import edge_risk_score, filter_artifacts, fixed_column_risk_score
from .catalog import GroupDefinition, MineralCatalog
from .envi import EnviDataset, derive_mask, wavelength_indices, write_envi
from .sampling import build_stratified_sample_plan, read_sample_blocks, sample_plan_mask
from .sensor import build_sensor_capability_card, extract_fwhm_nm, resolve_mineral_support
from .swir_expert import (
    ReferenceEvidence,
    aggregate_reference_scores,
    continuum_absorption,
    feature_values,
    mineral_feature_gate,
    reference_evidence,
)
from .v4_calibration import POLICY_ORDER, estimate_relative_spectral_noise, policy_candidate, resolve_group_policies
from .v4_library import V4EnsembleLibrary, build_v4_library_ensemble, write_v4_library_artifacts
from .v4_models import CancellationToken, ProgressEvent, SamplePlan, SupportLevel


Progress = Callable[[ProgressEvent], None]


def _default_progress(event: ProgressEvent) -> None:
    print(f"[{event.overall_fraction * 100:6.2f}%] {event.stage}: {event.message}", flush=True)


def _save_json(value: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: None for key in ("p01", "p05", "p10", "p50", "p90", "p95", "p99")}
    result = np.percentile(finite, [1, 5, 10, 50, 90, 95, 99])
    return {key: float(item) for key, item in zip(("p01", "p05", "p10", "p50", "p90", "p95", "p99"), result)}


def _emit(
    progress: Progress,
    started: float,
    stage: str,
    fraction: float,
    message: str,
    *,
    stage_fraction: float | None = None,
    block: tuple[int, int] | None = None,
) -> None:
    elapsed = monotonic() - started
    remaining = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
    progress(ProgressEvent(stage, float(np.clip(fraction, 0.0, 1.0)), message, stage_fraction, block, elapsed, remaining))


def _processed(
    values: np.ndarray,
    input_is_smoothed: bool,
    window: int,
    order: int,
    column_bias: np.ndarray | None = None,
    correction_strength: float = 0.0,
) -> np.ndarray:
    cube = np.asarray(values, dtype=np.float64)
    result = cube if input_is_smoothed else savgol_smooth(cube, window, order)
    if column_bias is not None and correction_strength > 0.0:
        result = result - float(correction_strength) * np.asarray(column_bias, dtype=np.float64)[None, :, :]
    return result


def _valid_indices(
    wavelengths: np.ndarray,
    windows: Sequence[Sequence[float]],
    bad_bands: Sequence[int],
    references: np.ndarray,
) -> np.ndarray:
    indices = wavelength_indices(wavelengths, windows)
    bad = set(int(item) for item in bad_bands)
    indices = np.asarray([item for item in indices if int(item) not in bad and np.all(np.isfinite(references[:, item]))])
    if indices.size < 3:
        raise ValueError(f"Fewer than three usable bands remain in diagnostic windows {windows}")
    return indices


@dataclass(frozen=True)
class _PreparedGroup:
    definition: GroupDefinition
    output_minerals: tuple[str, ...]
    mineral_order: tuple[str, ...]
    reference_indices: np.ndarray
    reference_minerals: tuple[str, ...]
    references: np.ndarray
    detection_indices: np.ndarray
    classification_indices: np.ndarray
    reference_absorption: np.ndarray


def _prepare_groups(
    catalog: MineralCatalog,
    ensemble: V4EnsembleLibrary,
    wavelengths: np.ndarray,
    requested: Sequence[str],
    internal: Sequence[str],
    bad_bands: Sequence[int],
) -> list[_PreparedGroup]:
    output_by_group = catalog.active_groups(requested)
    internal_by_group = catalog.active_groups(internal)
    result: list[_PreparedGroup] = []
    for group_id, output_minerals in output_by_group.items():
        definition = catalog.group(group_id)
        mineral_order = tuple(
            mineral for mineral in internal_by_group.get(group_id, ()) if mineral in ensemble.mineral_labels
        )
        reference_indices = np.asarray([
            index
            for index, (candidate_group, mineral) in enumerate(zip(ensemble.group_labels, ensemble.mineral_labels))
            if candidate_group == group_id and mineral in mineral_order
        ], dtype=np.int64)
        references = ensemble.spectra[reference_indices]
        detection = _valid_indices(wavelengths, definition.detection_windows_nm, bad_bands, references)
        classification = _valid_indices(wavelengths, definition.classification_windows_nm, bad_bands, references)
        absorption, _ = continuum_absorption(references[:, classification], wavelengths[classification])
        result.append(_PreparedGroup(
            definition,
            tuple(output_minerals),
            mineral_order,
            reference_indices,
            tuple(ensemble.mineral_labels[index] for index in reference_indices),
            references,
            detection,
            classification,
            absorption,
        ))
    return result


def _group_sam(cube: np.ndarray, mask: np.ndarray, group: _PreparedGroup) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(mask.shape, np.nan, dtype=np.float32)
    mineral_result = np.full((len(group.mineral_order), *mask.shape), np.nan, dtype=np.float32)
    indices = group.detection_indices
    valid = mask & np.all(np.isfinite(cube[..., indices]), axis=-1) & np.all(cube[..., indices] > 0, axis=-1)
    if not np.any(valid):
        return result, mineral_result
    angles = spectral_angles(cube[..., indices][valid], group.references[:, indices])
    scores, _ = aggregate_reference_scores(
        angles,
        group.reference_minerals,
        group.mineral_order,
        best_k=group.definition.detection_best_k,
    )
    result[valid] = np.min(scores, axis=1).astype(np.float32)
    for mineral_index in range(scores.shape[1]):
        mineral_result[mineral_index][valid] = scores[:, mineral_index].astype(np.float32)
    return result, mineral_result


def _subclass_evidence(
    cube: np.ndarray,
    candidate: np.ndarray,
    group: _PreparedGroup,
    wavelengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], ReferenceEvidence | None]:
    indices = group.classification_indices
    valid = candidate & np.all(np.isfinite(cube[..., indices]), axis=-1) & np.all(cube[..., indices] > 0, axis=-1)
    if not np.any(valid):
        return valid, np.empty((0, len(group.mineral_order))), np.empty((0, len(group.mineral_order))), np.empty(0), {}, None
    absorption, depth = continuum_absorption(cube[..., indices][valid], wavelengths[indices])
    evidence = reference_evidence(group.definition, absorption, group.reference_absorption, wavelengths[indices])
    scores, consensus = aggregate_reference_scores(
        evidence.total,
        group.reference_minerals,
        group.mineral_order,
        best_k=group.definition.classification_best_k,
    )
    features = feature_values(group.definition, absorption, wavelengths[indices])
    return valid, scores, consensus, depth, features, evidence


def _classification_output(labels: np.ndarray, mask: np.ndarray, class_count: int) -> np.ndarray:
    output = np.asarray(labels, dtype=np.uint8).copy()
    output[~mask] = class_count + 1
    return output


def _candidate_output(candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros(candidate.shape, dtype=np.uint8)
    output[candidate] = 1
    output[~mask] = 2
    return output


def _expand_internal_minerals(catalog: MineralCatalog, requested: Sequence[str]) -> tuple[str, ...]:
    result = list(dict.fromkeys(str(item).casefold() for item in requested))
    for mineral_id in tuple(result):
        mineral = catalog.mineral(mineral_id)
        for confuser in mineral.confusers:
            if confuser in catalog.minerals and catalog.mineral(confuser).group_id == mineral.group_id and confuser not in result:
                result.append(confuser)
    return tuple(result)


def _write_capability(
    output: Path,
    card: Any,
    support: Mapping[str, Any],
    sample_plan: SamplePlan,
) -> None:
    capability = output / "capability"
    capability.mkdir(parents=True, exist_ok=True)
    _save_json(card.to_dict(), capability / "sensor_capability_card.json")
    _save_json(sample_plan.to_dict(), capability / "sample_plan.json")
    records = [item.to_dict() for item in support.values()]
    _save_json(records, capability / "mineral_observability.json")
    with (capability / "mineral_observability.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mineral_id", "display_name", "support_level", "score", "expert", "coverage", "valid_bands", "reference_count", "reasons"])
        for item in support.values():
            writer.writerow([
                item.mineral_id,
                item.display_name,
                item.level.value,
                f"{item.score:.6f}",
                item.selected_expert or "",
                f"{item.required_windows_covered:.6f}",
                item.valid_bands_in_required_windows,
                item.reference_count,
                "; ".join(item.reasons),
            ])


def run_v4(
    config: dict[str, Any],
    output_dir: str | Path,
    start_line: int = 0,
    stop_line: int | None = None,
    *,
    progress: Progress = _default_progress,
    cancel_token: CancellationToken | None = None,
    catalog: MineralCatalog | None = None,
) -> dict[str, Any]:
    started = monotonic()
    token = cancel_token or CancellationToken()
    catalog = catalog or MineralCatalog.load(config.get("v4", {}).get("catalog_path"))
    v4 = config.get("v4")
    if not isinstance(v4, dict):
        raise ValueError("The configuration does not contain a V4 section")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False) if not output.exists() else None
    _emit(progress, started, "audit", 0.01, "Opening ENVI image and mask")

    image = EnviDataset(config["analysis_image"])
    mask_dataset: EnviDataset | None = None
    try:
        wavelengths = image.info.wavelengths_nm
        if wavelengths is None:
            raise ValueError("The analysis image requires an ENVI wavelength vector")
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        start = int(start_line)
        stop = image.info.lines if stop_line is None else int(stop_line)
        if start < 0 or stop > image.info.lines or start >= stop:
            raise ValueError(f"Invalid output line interval [{start}, {stop})")
        mask_dataset = EnviDataset(config["analysis_mask"])
        if (mask_dataset.info.lines, mask_dataset.info.samples) != (image.info.lines, image.info.samples):
            raise ValueError("Analysis mask and image spatial dimensions do not match")
        default_bands = [min(20, mask_dataset.info.bands - 1), min(mask_dataset.info.bands // 2, mask_dataset.info.bands - 1), min(190, mask_dataset.info.bands - 1)]
        mask_bands = sorted(set(int(item) for item in v4.get("mask_bands", default_bands)))
        full_mask = derive_mask(mask_dataset, bands=mask_bands, chunk_rows=int(config.get("chunk_rows", 64)))
        if np.count_nonzero(full_mask) < int(v4.get("minimum_mask_pixels", 100)):
            raise ValueError("Core mask is empty or contains too few valid pixels")
        mask_dataset.close()
        mask_dataset = None
        token.raise_if_cancelled()

        sampling = v4.get("sampling", {})
        sample_plan = build_stratified_sample_plan(
            full_mask,
            desired_blocks=int(sampling.get("blocks", 12)),
            block_rows=int(sampling.get("block_rows", 64)),
            min_valid_pixels=int(sampling.get("minimum_valid_pixels_per_block", 128)),
            forced_lines=tuple(int(item) for item in sampling.get("forced_lines", ())),
        )
        sample_cube, sample_mask, _ = read_sample_blocks(image, full_mask, sample_plan)
        input_is_smoothed = bool(config.get("analysis_input_is_smoothed", False))
        sg_window = int(v4.get("preprocessing", {}).get("sg_window", config.get("sg_window", 11)))
        sg_order = int(v4.get("preprocessing", {}).get("sg_polyorder", config.get("sg_polyorder", 2)))
        processed_sample = _processed(sample_cube, input_is_smoothed, sg_window, sg_order)
        preprocessing_state = "already_sg_smoothed" if input_is_smoothed else f"sg_{sg_window}_{sg_order}"
        card = build_sensor_capability_card(
            image,
            catalog,
            sample_cube=sample_cube,
            sample_mask=sample_mask,
            data_physics=v4.get("data_physics"),
            preprocessing_state=preprocessing_state,
        )
        correction_config = v4.get("artifact_control", {}).get("column_spectral_correction", {})
        correction_enabled = bool(correction_config.get("enabled", True))
        correction_strength = float(correction_config.get("strength", 0.70)) if correction_enabled else 0.0
        column_bias = (
            robust_column_bias(
                processed_sample,
                sample_mask,
                radius=int(correction_config.get("radius", 2)),
            )
            if correction_enabled
            else np.zeros((image.info.samples, image.info.bands), dtype=np.float64)
        )
        processed_sample = _processed(
            sample_cube,
            input_is_smoothed,
            sg_window,
            sg_order,
            column_bias,
            correction_strength,
        )
        provisional = resolve_mineral_support(card, catalog, wavelengths_nm=wavelengths)
        requested = tuple(str(item).casefold() for item in v4.get("requested_minerals", ()))
        if not requested:
            requested = tuple(
                mineral.mineral_id
                for mineral in catalog.minerals.values()
                if mineral.support_level == "validated_swir" and provisional[mineral.mineral_id].level != SupportLevel.UNSUPPORTED
            )
        for mineral_id in requested:
            catalog.mineral(mineral_id)
            if provisional[mineral_id].level == SupportLevel.UNSUPPORTED:
                raise ValueError(f"{mineral_id} is unsupported: {'; '.join(provisional[mineral_id].reasons)}")
        internal = _expand_internal_minerals(catalog, requested)
        internal = tuple(mineral for mineral in internal if provisional[mineral].level != SupportLevel.UNSUPPORTED)
        _emit(progress, started, "library", 0.08, "Building Catalog-driven automatic reference ensemble")
        fwhm, _ = extract_fwhm_nm(image)
        library_config = v4.get("library_ensemble", {})
        ensemble = build_v4_library_ensemble(
            v4["spectral_library_root"],
            wavelengths,
            catalog,
            internal,
            target_fwhm_nm=fwhm,
            scene_cube=processed_sample,
            scene_mask=sample_mask,
            maximum_representatives=int(library_config.get("maximum_representatives_per_mineral", 6)),
            dedup_angle_rad=float(library_config.get("dedup_angle_rad", 0.02)),
        )
        ensemble_paths = write_v4_library_artifacts(ensemble, output / "library_ensemble")
        support = resolve_mineral_support(card, catalog, wavelengths_nm=wavelengths, reference_counts=ensemble.reference_counts())
        for mineral_id in requested:
            if support[mineral_id].level == SupportLevel.UNSUPPORTED:
                raise ValueError(f"{mineral_id} failed final capability gating: {'; '.join(support[mineral_id].reasons)}")
        _write_capability(output, card, support, sample_plan)
        write_envi(
            column_bias.astype(np.float32),
            output / "capability" / "column_spectral_bias.dat",
            description="Robust full-depth fixed-column spectral bias estimate; detector columns by image bands",
        )
        _save_json(catalog.to_summary(), output / "capability" / "catalog_summary.json")
        token.raise_if_cancelled()

        groups = _prepare_groups(catalog, ensemble, wavelengths, requested, internal, card.bad_band_indices)
        chunk_rows = int(config.get("chunk_rows", 64))
        group_scores = {
            group.definition.group_id: np.full((image.info.lines, image.info.samples), np.nan, dtype=np.float32)
            for group in groups
        }
        detection_mineral_scores = {
            group.definition.group_id: np.full(
                (len(group.mineral_order), image.info.lines, image.info.samples), np.nan, dtype=np.float32
            )
            for group in groups
        }
        sample_selection = sample_plan_mask(sample_plan, full_mask)
        _emit(progress, started, "group_sam", 0.16, "Computing full-depth group SAM evidence")
        for row_start, row_stop, cube in image.iter_rows(chunk_rows=chunk_rows):
            token.raise_if_cancelled()
            values = _processed(cube, input_is_smoothed, sg_window, sg_order, column_bias, correction_strength)
            mask_chunk = full_mask[row_start:row_stop]
            for group in groups:
                group_score, mineral_scores = _group_sam(values, mask_chunk, group)
                group_scores[group.definition.group_id][row_start:row_stop] = group_score
                detection_mineral_scores[group.definition.group_id][:, row_start:row_stop] = mineral_scores
            fraction = row_stop / image.info.lines
            _emit(progress, started, "group_sam", 0.16 + 0.20 * fraction, f"Processed SAM through line {row_stop}", stage_fraction=fraction, block=(row_start, row_stop))

        raw_group_scores = {key: value.copy() for key, value in group_scores.items()}
        detection_calibration: dict[str, dict[str, Any]] = {}
        for group in groups:
            group_id = group.definition.group_id
            mineral_scores = detection_mineral_scores[group_id]
            medians = np.asarray([
                np.nanmedian(mineral_scores[index][sample_selection])
                for index in range(mineral_scores.shape[0])
            ], dtype=np.float64)
            baseline = float(np.nanmin(medians))
            maximum_offset = min(
                0.03,
                0.25 * float(group.definition.policies["balanced"].get("absolute_sam_threshold_rad", 0.12)),
            )
            detection_strength = min(float(group.definition.domain_calibration_strength), 0.25)
            offsets = np.clip(
                detection_strength * (medians - baseline),
                0.0,
                maximum_offset,
            )
            calibrated_detection = mineral_scores - offsets[:, None, None]
            finite_detection = np.any(np.isfinite(calibrated_detection), axis=0)
            minimum_detection = np.min(np.where(np.isfinite(calibrated_detection), calibrated_detection, np.inf), axis=0)
            group_scores[group_id] = np.where(finite_detection, minimum_detection, np.nan).astype(np.float32)
            detection_calibration[group_id] = {
                "sample_median_sam_rad": {
                    mineral: float(medians[index]) for index, mineral in enumerate(group.mineral_order)
                },
                "offsets_rad": {
                    mineral: float(offsets[index]) for index, mineral in enumerate(group.mineral_order)
                },
                "strength": detection_strength,
                "maximum_offset_rad": maximum_offset,
            }
        policies: dict[str, dict[str, dict[str, Any]]] = {}
        group_noise: dict[str, float] = {}
        for group in groups:
            group_id = group.definition.group_id
            selected_sample_spectra = processed_sample[..., group.classification_indices][sample_mask]
            relative_noise = estimate_relative_spectral_noise(selected_sample_spectra)
            group_noise[group_id] = relative_noise
            overrides = v4.get("policy_overrides", {}).get(group_id, {})
            policies[group_id] = resolve_group_policies(
                group.definition,
                card,
                group_scores[group_id],
                sample_selection,
                relative_noise=relative_noise,
                overrides=overrides,
                minimum_column_samples=int(sampling.get("minimum_column_samples", 20)),
            )

        _emit(progress, started, "calibration", 0.38, "Calibrating subclass domains from stratified blocks")
        calibration: dict[str, dict[str, Any]] = {}
        for group_index, group in enumerate(groups):
            group_id = group.definition.group_id
            sensitive = policies[group_id]["sensitive"]
            sensitive_full = policy_candidate(group_scores[group_id], full_mask, sensitive)
            score_samples: list[np.ndarray] = []
            depth_samples: list[np.ndarray] = []
            margin_samples: list[np.ndarray] = []
            for block in sample_plan.blocks:
                token.raise_if_cancelled()
                values = _processed(
                    image.read_rows(block.start_line, block.stop_line),
                    input_is_smoothed,
                    sg_window,
                    sg_order,
                )
                candidate = sensitive_full[block.start_line:block.stop_line]
                _, scores, _, depth, _, _ = _subclass_evidence(values, candidate, group, wavelengths)
                usable = np.all(np.isfinite(scores), axis=1) & np.isfinite(depth)
                if np.any(usable):
                    current = scores[usable]
                    score_samples.append(current)
                    depth_samples.append(depth[usable])
                    if current.shape[1] > 1:
                        ordered = np.sort(current, axis=1)
                        margin_samples.append(ordered[:, 1] - ordered[:, 0])
            if score_samples:
                stacked = np.vstack(score_samples)
                centers = np.median(stacked, axis=0)
                best_raw = np.min(stacked, axis=1)
                raw_scale = np.percentile(best_raw, [10, 90])
                margins = np.concatenate(margin_samples) if margin_samples else np.zeros(stacked.shape[0])
            else:
                centers = np.zeros(len(group.mineral_order), dtype=np.float64)
                raw_scale = np.array([0.0, 1.0])
                margins = np.zeros(1)
            calibration[group_id] = {
                "centers": centers,
                "raw_score_p10": float(raw_scale[0]),
                "raw_score_p90": float(max(raw_scale[1], raw_scale[0] + 1e-6)),
                "margin_p50": float(np.percentile(margins, 50)),
                "margin_p75": float(np.percentile(margins, 75)),
                "sample_pixels": int(sum(item.shape[0] for item in score_samples)),
            }
            fraction = (group_index + 1) / max(len(groups), 1)
            _emit(progress, started, "calibration", 0.38 + 0.08 * fraction, f"Calibrated {group_id}", stage_fraction=fraction)

        height = stop - start
        output_mask = full_mask[start:stop]
        edge_risk = edge_risk_score(output_mask, width=int(v4.get("artifact_control", {}).get("edge_width", 2)))
        brightness_band = int(np.argmin(np.abs(wavelengths - 1600.0)))
        sample_brightness = processed_sample[..., brightness_band][sample_mask]
        finite_brightness = sample_brightness[np.isfinite(sample_brightness)]
        saturation_threshold = float(np.percentile(finite_brightness, 99.9)) if finite_brightness.size else np.inf
        global_confidence = np.zeros((height, image.info.samples), dtype=np.float32)
        global_stability = np.zeros_like(global_confidence)
        global_rejection = np.zeros((height, image.info.samples), dtype=np.uint8)
        run_groups: dict[str, Any] = {}
        count_rows: list[list[Any]] = []
        _emit(progress, started, "classification", 0.47, "Computing shared subclass evidence for all three policies")

        for group_index, group in enumerate(groups):
            token.raise_if_cancelled()
            group_id = group.definition.group_id
            score_full = group_scores[group_id]
            profile_candidates = {
                policy: policy_candidate(score_full[start:stop], output_mask, policies[group_id][policy])
                for policy in POLICY_ORDER
            }
            sensitive_candidate = profile_candidates["sensitive"]
            column_risk = fixed_column_risk_score(
                policy_candidate(score_full, full_mask, policies[group_id]["sensitive"]), full_mask, sample_plan
            )
            mineral_count = len(group.mineral_order)
            raw_scores = np.full((mineral_count, height, image.info.samples), np.nan, dtype=np.float32)
            calibrated_scores = np.full_like(raw_scores, np.nan)
            consensus_cube = np.full_like(raw_scores, np.nan)
            depth_map = np.full((height, image.info.samples), np.nan, dtype=np.float32)
            feature_maps = {
                feature.feature_id: np.full((height, image.info.samples), np.nan, dtype=np.float32)
                for feature in group.definition.features
            }
            best_labels = np.full((height, image.info.samples), -1, dtype=np.int16)
            raw_labels = np.full_like(best_labels, -1)
            margin = np.full((height, image.info.samples), np.nan, dtype=np.float32)
            winning_consensus = np.zeros((height, image.info.samples), dtype=np.float32)
            fit_quality = np.full((height, image.info.samples), np.nan, dtype=np.float32)
            saturation = np.zeros((height, image.info.samples), dtype=bool)
            calibration_record = calibration[group_id]
            centers = np.asarray(calibration_record["centers"], dtype=np.float64)
            strength = float(group.definition.domain_calibration_strength)
            for row_start, row_stop, cube in image.iter_rows(chunk_rows=chunk_rows, start=start, stop=stop):
                token.raise_if_cancelled()
                local = slice(row_start - start, row_stop - start)
                # Column correction is a high-recall detector aid. Subclass shape,
                # continuum, and SFF evidence stay on the original spectral domain.
                values = _processed(cube, input_is_smoothed, sg_window, sg_order)
                saturation[local] = np.isfinite(values[..., brightness_band]) & (values[..., brightness_band] >= saturation_threshold)
                valid, scores, consensus, depth, features, evidence = _subclass_evidence(
                    values, sensitive_candidate[local], group, wavelengths
                )
                if np.any(valid):
                    calibrated = scores - strength * centers[None, :]
                    current_labels = np.argmin(calibrated, axis=1)
                    current_raw_labels = np.argmin(scores, axis=1)
                    ordered = np.sort(calibrated, axis=1)
                    current_margin = ordered[:, 1] - ordered[:, 0] if mineral_count > 1 else np.full(scores.shape[0], np.inf)
                    for mineral_index in range(mineral_count):
                        raw_scores[mineral_index, local][valid] = scores[:, mineral_index].astype(np.float32)
                        calibrated_scores[mineral_index, local][valid] = calibrated[:, mineral_index].astype(np.float32)
                        consensus_cube[mineral_index, local][valid] = consensus[:, mineral_index].astype(np.float32)
                    best_labels[local][valid] = current_labels.astype(np.int16)
                    raw_labels[local][valid] = current_raw_labels.astype(np.int16)
                    depth_map[local][valid] = depth.astype(np.float32)
                    margin[local][valid] = current_margin.astype(np.float32)
                    winning_consensus[local][valid] = consensus[np.arange(scores.shape[0]), current_labels].astype(np.float32)
                    if evidence is not None:
                        fit_quality[local][valid] = np.max(evidence.fit_quality, axis=1).astype(np.float32)
                    for feature_id, feature in features.items():
                        feature_maps[feature_id][local][valid] = feature.astype(np.float32)
                group_fraction = (row_stop - start) / height
                group_span = 0.41 / max(len(groups), 1)
                overall = 0.47 + group_span * (group_index + 0.80 * group_fraction)
                _emit(progress, started, "classification", overall, f"{group_id}: processed lines {row_start}:{row_stop}", stage_fraction=group_fraction, block=(row_start, row_stop))

            rows, columns = np.indices(best_labels.shape)
            safe_labels = np.maximum(best_labels, 0)
            best_raw = raw_scores[safe_labels, rows, columns]
            p10 = float(calibration_record["raw_score_p10"])
            p90 = float(calibration_record["raw_score_p90"])
            score_confidence = np.clip((p90 - best_raw) / max(p90 - p10, 1e-6), 0.0, 1.0)
            margin_scale = max(float(calibration_record["margin_p75"]), 0.015)
            margin_confidence = np.clip(margin / margin_scale, 0.0, 1.0)
            sensitive_settings = policies[group_id]["sensitive"]
            sam_confidence = np.clip(1.0 - score_full[start:stop] / max(float(sensitive_settings["absolute_sam_threshold_rad"]), 1e-6), 0.0, 1.0)
            depth_reference = max(2.0 * float(sensitive_settings["minimum_absorption_depth"]), 0.02)
            depth_confidence = np.clip(depth_map / depth_reference, 0.0, 1.0)
            label_agreement = (best_labels == raw_labels).astype(np.float32)
            stability = np.clip(0.40 * label_agreement + 0.35 * margin_confidence + 0.25 * winning_consensus, 0.0, 1.0)
            confidence = (
                0.20 * sam_confidence
                + 0.20 * score_confidence
                + 0.20 * margin_confidence
                + 0.15 * winning_consensus
                + 0.15 * depth_confidence
                + 0.10 * stability
            )
            confidence *= 1.0 - 0.35 * column_risk[None, :]
            confidence *= 1.0 - 0.45 * edge_risk
            confidence *= np.where(saturation, 0.45, 1.0)
            confidence = np.where(sensitive_candidate & (best_labels >= 0), np.clip(confidence, 0.0, 1.0), 0.0).astype(np.float32)

            class_lookup = {mineral: index + 1 for index, mineral in enumerate(group.output_minerals)}
            class_names = ["Unclassified", *[catalog.mineral(item).display_name_en for item in group.output_minerals], "Masked Pixels"]
            before_profiles: dict[str, np.ndarray] = {}
            gate_profiles: dict[str, np.ndarray] = {}
            confidence_threshold_maps: dict[str, np.ndarray] = {}
            for policy in POLICY_ORDER:
                settings = policies[group_id][policy]
                gate_pass = np.zeros(best_labels.shape, dtype=bool)
                for mineral_index, mineral_id in enumerate(group.mineral_order):
                    selected = best_labels == mineral_index
                    if not np.any(selected):
                        continue
                    mineral_gate = mineral_feature_gate(
                        catalog.mineral(mineral_id), group.definition.expert_id, policy, depth_map, feature_maps
                    )
                    gate_pass[selected] = mineral_gate[selected]
                gate_profiles[policy] = gate_pass
                adaptive_margin = {
                    "conservative": 0.10,
                    "balanced": 0.0,
                    "sensitive": 0.0,
                }[policy] * float(calibration_record["margin_p50"])
                margin_threshold = max(float(settings.get("minimum_margin", 0.0)), adaptive_margin)
                consensus_threshold = {"conservative": 0.20, "balanced": 0.10, "sensitive": 0.0}[policy]
                stability_threshold = {"conservative": 0.45, "balanced": 0.25, "sensitive": 0.0}[policy]
                requested_winner = np.zeros(best_labels.shape, dtype=bool)
                labels = np.zeros(best_labels.shape, dtype=np.uint8)
                for mineral_index, mineral_id in enumerate(group.mineral_order):
                    if mineral_id not in class_lookup:
                        continue
                    selected = best_labels == mineral_index
                    requested_winner |= selected
                    labels[selected] = class_lookup[mineral_id]
                confidence_quantile = {"conservative": 60.0, "balanced": 25.0, "sensitive": 5.0}[policy]
                confidence_threshold_map = np.ones(best_labels.shape, dtype=np.float32)
                confidence_thresholds_by_mineral: dict[str, float] = {}
                for mineral_index, mineral_id in enumerate(group.mineral_order):
                    selected = profile_candidates[policy] & (best_labels == mineral_index)
                    confidence_sample = confidence[selected]
                    confidence_sample = confidence_sample[np.isfinite(confidence_sample)]
                    scene_confidence_threshold = (
                        float(np.percentile(confidence_sample, confidence_quantile)) if confidence_sample.size else 1.0
                    )
                    threshold = min(
                        float(settings.get("confidence_threshold", 0.0)), scene_confidence_threshold
                    )
                    confidence_threshold_map[best_labels == mineral_index] = threshold
                    confidence_thresholds_by_mineral[mineral_id] = threshold
                confidence_threshold_maps[policy] = confidence_threshold_map
                accepted = (
                    profile_candidates[policy]
                    & requested_winner
                    & np.isfinite(depth_map)
                    & (depth_map >= float(settings["minimum_absorption_depth"]))
                    & gate_pass
                    & np.isfinite(margin)
                    & (margin >= margin_threshold)
                    & (winning_consensus >= consensus_threshold)
                    & (stability >= stability_threshold)
                    & (confidence >= confidence_threshold_map)
                )
                before_profiles[policy] = np.where(accepted, labels, 0).astype(np.uint8)
                settings["resolved_minimum_margin"] = margin_threshold
                settings["resolved_consensus_threshold"] = consensus_threshold
                settings["resolved_stability_threshold"] = stability_threshold
                settings["resolved_confidence_threshold_by_mineral"] = confidence_thresholds_by_mineral

            final_profiles: dict[str, np.ndarray] = {}
            stripe_profiles: dict[str, np.ndarray] = {}
            cleanup_records: dict[str, Any] = {}
            for policy in POLICY_ORDER:
                cleanup_settings = dict(group.definition.spatial_cleanup)
                artifact_confidence = confidence * (1.0 - 0.80 * column_risk[None, :])
                low_column_risk = column_risk[None, :] < 0.50
                accepted_confidence = artifact_confidence[(before_profiles[policy] > 0) & low_column_risk]
                protection_quantile = {"conservative": 50.0, "balanced": 65.0, "sensitive": 80.0}[policy]
                if accepted_confidence.size:
                    protection_floor = {"conservative": 0.62, "balanced": 0.58, "sensitive": 0.62}[policy]
                    cleanup_settings["strong_evidence_protection"] = max(
                        protection_floor,
                        min(
                            float(cleanup_settings.get("strong_evidence_protection", 0.72)),
                            float(np.percentile(accepted_confidence, protection_quantile)),
                        ),
                    )
                protection_by_class: dict[int, float] = {}
                protection_floor = {"conservative": 0.55, "balanced": 0.50, "sensitive": 0.55}[policy]
                for class_id in range(1, len(group.output_minerals) + 1):
                    class_values = artifact_confidence[
                        (before_profiles[policy] == class_id) & low_column_risk
                    ]
                    if class_values.size:
                        protection_by_class[class_id] = max(
                            protection_floor,
                            min(
                                float(cleanup_settings.get("strong_evidence_protection", 0.72)),
                                float(np.percentile(class_values, protection_quantile)),
                            ),
                        )
                cleanup_settings["strong_evidence_protection_by_class"] = protection_by_class
                final_profiles[policy], stripe_profiles[policy], cleanup_records[policy] = filter_artifacts(
                    before_profiles[policy],
                    artifact_confidence,
                    output_mask,
                    cleanup_settings,
                    policy=policy,
                    column_risk=column_risk,
                )
                cleanup_records[policy]["resolved_strong_evidence_protection"] = float(
                    cleanup_settings.get("strong_evidence_protection", 0.72)
                )
                cleanup_records[policy]["resolved_strong_evidence_protection_by_class"] = protection_by_class
            balanced_mismatch = (final_profiles["balanced"] > 0) & (final_profiles["sensitive"] != final_profiles["balanced"])
            final_profiles["balanced"][balanced_mismatch] = 0
            conservative_mismatch = (final_profiles["conservative"] > 0) & (final_profiles["balanced"] != final_profiles["conservative"])
            final_profiles["conservative"][conservative_mismatch] = 0
            stripe_union = np.logical_or.reduce(tuple(stripe_profiles.values()))

            balanced_settings = policies[group_id]["balanced"]
            balanced_candidate = profile_candidates["balanced"]
            rejection = np.zeros(best_labels.shape, dtype=np.uint8)
            in_mask = output_mask & np.isfinite(score_full[start:stop])
            rejection[in_mask & (score_full[start:stop] > float(balanced_settings["absolute_sam_threshold_rad"]))] = 1
            rejection[in_mask & (rejection == 0) & ~balanced_candidate] = 2
            pending = balanced_candidate & (rejection == 0)
            failed = pending & (depth_map < float(balanced_settings["minimum_absorption_depth"]))
            rejection[failed] = 3
            pending &= ~failed
            failed = pending & ~gate_profiles["balanced"]
            rejection[failed] = 4
            pending &= ~failed
            requested_label = np.isin(best_labels, [group.mineral_order.index(item) for item in group.output_minerals])
            failed = pending & (
                ~requested_label
                | (margin < float(balanced_settings["resolved_minimum_margin"]))
            )
            rejection[failed] = 5
            pending &= ~failed
            failed = pending & (winning_consensus < float(balanced_settings["resolved_consensus_threshold"]))
            rejection[failed] = 6
            pending &= ~failed
            failed = pending & (
                (stability < float(balanced_settings["resolved_stability_threshold"]))
                | (confidence < confidence_threshold_maps["balanced"])
            )
            rejection[failed] = 7
            rejection[(before_profiles["balanced"] > 0) & (final_profiles["balanced"] == 0)] = 8
            rejection[balanced_candidate & saturation & (final_profiles["balanced"] == 0)] = 9
            rejection[final_profiles["balanced"] > 0] = 0

            group_dir = output / "groups" / group_id
            group_dir.mkdir(parents=True, exist_ok=True)
            write_envi(score_full[start:stop].astype(np.float32), group_dir / "group_sam_score.dat", description=f"{group_id} V4 group SAM score in radians")
            write_envi(raw_group_scores[group_id][start:stop].astype(np.float32), group_dir / "raw_group_sam_score.dat", description=f"{group_id} raw minimum group SAM angle before mineral-domain offset calibration")
            write_envi(detection_mineral_scores[group_id][:, start:stop].astype(np.float32), group_dir / "group_mineral_sam_scores.dat", band_names=[catalog.mineral(item).display_name_en for item in group.mineral_order], description=f"{group_id} mineral-level SAM evidence before group candidate union")
            threshold_cube = np.stack([
                np.broadcast_to(policies[group_id][policy]["column_thresholds_rad"][None, :], (height, image.info.samples))
                for policy in POLICY_ORDER
            ]).astype(np.float32)
            write_envi(threshold_cube, group_dir / "column_threshold.dat", band_names=[item.title() for item in POLICY_ORDER], description=f"{group_id} V4 calibrated column SAM thresholds")
            write_envi(_candidate_output(balanced_candidate, output_mask), group_dir / "column_candidates.dat", class_names=["Unclassified", "Balanced group candidate", "Masked Pixels"], description=f"{group_id} balanced SAM candidates")
            write_envi(raw_scores, group_dir / "raw_mineral_scores.dat", band_names=[catalog.mineral(item).display_name_en for item in group.mineral_order], description=f"{group_id} raw shared mineral scores; lower is better")
            write_envi(calibrated_scores, group_dir / "calibrated_mineral_scores.dat", band_names=[catalog.mineral(item).display_name_en for item in group.mineral_order], description=f"{group_id} domain-calibrated shared mineral scores; lower is better")
            write_envi(_classification_output(before_profiles["balanced"], output_mask, len(group.output_minerals)), group_dir / "classes_before_artifact_filter.dat", class_names=class_names, description=f"{group_id} balanced classes before artifact filtering")
            write_envi(stripe_union.astype(np.uint8), group_dir / "stripe_noise_mask.dat", description=f"{group_id} weak-evidence directional artifact mask")
            write_envi(np.broadcast_to(column_risk[None, :], (height, image.info.samples)).astype(np.float32), group_dir / "column_risk_score.dat", description=f"{group_id} fixed detector-column risk score")
            write_envi(edge_risk.astype(np.float32), group_dir / "edge_risk_score.dat", description="Core boundary risk score")
            write_envi(saturation.astype(np.uint8), group_dir / "saturation_mask.dat", description="Scene-relative high-reflectance saturation risk")
            write_envi(depth_map, group_dir / "absorption_depth.dat", description=f"{group_id} continuum-removed absorption depth")
            write_envi(margin, group_dir / "classification_margin.dat", description=f"{group_id} first-versus-second mineral score margin")
            write_envi(fit_quality, group_dir / "sff_quality.dat", description=f"{group_id} best reference SFF scale-to-RMS evidence")
            write_envi(confidence, group_dir / "confidence.dat", description=f"{group_id} calibrated confidence")
            write_envi(stability.astype(np.float32), group_dir / "stability.dat", description=f"{group_id} score, label, and reference-consensus stability")
            write_envi(rejection, group_dir / "rejection_reason.dat", description=f"{group_id} balanced rejection reason code")
            for feature_id, feature_map in feature_maps.items():
                write_envi(feature_map, group_dir / f"feature_{feature_id}.dat", description=f"{group_id} diagnostic feature {feature_id}")
            profile_counts: dict[str, dict[str, int]] = {}
            for policy in POLICY_ORDER:
                filename = f"final_{policy}.dat"
                write_envi(
                    _classification_output(final_profiles[policy], output_mask, len(group.output_minerals)),
                    group_dir / filename,
                    class_names=class_names,
                    description=f"CoreSpec Mapper V4 {group_id} {policy} classification",
                )
                counts = {
                    catalog.mineral(mineral).display_name_en: int(np.count_nonzero(final_profiles[policy] == class_id))
                    for mineral, class_id in class_lookup.items()
                }
                profile_counts[policy] = counts
                for mineral, count in counts.items():
                    count_rows.append([policy, group_id, mineral, count])
            group_summary = {
                "group_id": group_id,
                "output_minerals": list(group.output_minerals),
                "internal_competitors": list(group.mineral_order),
                "selected_references": [ensemble.spectrum_names[index] for index in group.reference_indices],
                "detection_band_count": int(group.detection_indices.size),
                "classification_band_count": int(group.classification_indices.size),
                "relative_spectral_noise": group_noise[group_id],
                "detection_domain_calibration": detection_calibration[group_id],
                "calibration": calibration_record,
                "policies": policies[group_id],
                "counts": profile_counts,
                "cleanup": cleanup_records,
                "stripe_pixels_union": int(np.count_nonzero(stripe_union)),
                "confidence_percentiles": _percentiles(confidence[sensitive_candidate]),
                "stability_percentiles": _percentiles(stability[sensitive_candidate]),
                "column_risk_percentiles": _percentiles(column_risk),
                "rejection_counts_balanced": {
                    str(code): int(np.count_nonzero(rejection == code)) for code in range(0, 11)
                },
            }
            _save_json(group_summary, group_dir / "summary.json")
            run_groups[group_id] = group_summary
            update = confidence > global_confidence
            global_confidence[update] = confidence[update]
            global_stability[update] = stability[update]
            global_rejection[update] = rejection[update]
            _emit(progress, started, "outputs", 0.47 + 0.41 * (group_index + 1) / len(groups), f"Wrote V4 outputs for {group_id}")

        confidence_dir = output / "confidence"
        confidence_dir.mkdir(parents=True, exist_ok=True)
        write_envi(global_confidence, confidence_dir / "confidence.dat", description="Maximum V4 confidence across mapped mineral groups")
        write_envi(global_stability, confidence_dir / "stability.dat", description="Stability associated with maximum-confidence mineral group")
        write_envi(global_rejection, confidence_dir / "rejection_reason.dat", description="Balanced rejection reason associated with maximum-confidence group")
        tables = output / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        with (tables / "mineral_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["policy", "group", "mineral", "pixels"])
            writer.writerows(count_rows)
        summary = {
            "version": "CoreSpec Mapper V4.0.0",
            "analysis_image": str(image.info.data_path),
            "analysis_mask": str(config["analysis_mask"]),
            "output_lines": [start, stop],
            "mask_pixels": int(np.count_nonzero(output_mask)),
            "sensor_signature": card.sensor_signature,
            "spectral_domain": card.spectral_domain,
            "data_physics": card.data_physics,
            "requested_minerals": list(requested),
            "internal_competitors": list(internal),
            "sample_plan": sample_plan.to_dict(),
            "preprocessing": {
                "input_is_smoothed": input_is_smoothed,
                "sg_window": sg_window,
                "sg_polyorder": sg_order,
                "column_spectral_correction": {
                    "enabled": correction_enabled,
                    "strength": correction_strength,
                    "radius": int(correction_config.get("radius", 2)),
                    "application": "group_detection_only",
                    "bias_abs_percentiles": _percentiles(np.abs(column_bias)),
                },
            },
            "saturation_threshold": saturation_threshold,
            "ensemble_artifacts": ensemble_paths,
            "groups": run_groups,
            "elapsed_seconds_before_qa": monotonic() - started,
        }
        _save_json(summary, output / "summary.json")
        _emit(progress, started, "complete", 0.90, "V4 scientific outputs are ready for preview and QA")
        return summary
    finally:
        image.close()
        if mask_dataset is not None:
            mask_dataset.close()
