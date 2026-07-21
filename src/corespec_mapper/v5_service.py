from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any, Callable, Mapping
import json
import tempfile

import numpy as np

from .algorithms import robust_column_bias, savgol_smooth
from .catalog import MineralCatalog, default_v5_runtime_catalog_path
from .envi import EnviDataset, resolve_envi_paths, subset_spatial_metadata, write_envi
from .geocore_mask_adapter import inspect_geocore_m12, run_geocore_m12
from .masking import MaskQualityFlag, MaterialMaskResult, build_material_mask
from .preview import _overlay, _stretched_gray, write_png
from .qa import validate_v4_run
from .sampling import build_stratified_sample_plan, read_sample_blocks
from .sensor import build_sensor_capability_card, extract_fwhm_nm, resolve_mineral_support
from .spectral_db import default_v5_database_path
from .spectral_v5 import default_v5_catalog_path, load_v5_catalog
from .v4_models import CancellationToken, ProgressEvent, SupportLevel
from .v4_pipeline import PreparedRunInputs, run_v4
from .v4_preview import make_v4_previews
from .v5_calibration import catalog_safe_search_bounds
from .v5_library import V5ReferenceLibrary, build_v5_library_ensemble
from .v5_threshold_trial import build_v5_threshold_trial
from .v5_project_calibration import load_project_weak_labels
from .validated_v3 import run_validated_v3_profiles


Progress = Callable[[ProgressEvent], None]


@dataclass(frozen=True)
class V5AuditSnapshot:
    cache_key: str
    runtime_config: dict[str, Any]
    catalog: MineralCatalog
    knowledge_catalog: dict[str, Any]
    prepared: PreparedRunInputs
    material_mask: MaterialMaskResult
    reference_library: V5ReferenceLibrary
    support: Mapping[str, Any]
    audit_record: dict[str, Any]


_CACHE_LOCK = Lock()
_SNAPSHOT_CACHE: "OrderedDict[str, V5AuditSnapshot]" = OrderedDict()
_MAX_CACHED_SNAPSHOTS = 1


def _save_json(value: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _emit(
    callback: Progress,
    started: float,
    stage: str,
    fraction: float,
    message: str,
    *,
    base: float = 0.0,
    span: float = 1.0,
) -> None:
    overall = float(np.clip(base + span * fraction, 0.0, 1.0))
    elapsed = monotonic() - started
    remaining = elapsed * (1.0 - overall) / overall if overall > 0 else None
    callback(ProgressEvent(stage, overall, message, elapsed_seconds=elapsed, estimated_remaining_seconds=remaining))


def _input_record(role: str, path: str | Path, *, required: bool = False) -> dict[str, Any]:
    try:
        dataset = EnviDataset(path)
    except (FileNotFoundError, ValueError) as exc:
        if required:
            raise
        source = Path(path)
        header_only = source.suffix.casefold() == ".hdr" and source.is_file()
        return {
            "role": role,
            "path": str(source),
            "status": "header_only_missing_data" if header_only else "unreadable",
            "blocking": False,
            "message": str(exc),
        }
    try:
        wavelengths = dataset.info.wavelengths_nm
        record = {
            "role": role,
            "data_path": str(dataset.info.data_path.resolve()),
            "header_path": str(dataset.info.header_path.resolve()),
            "lines": dataset.info.lines,
            "samples": dataset.info.samples,
            "bands": dataset.info.bands,
            "shape": [dataset.info.lines, dataset.info.samples, dataset.info.bands],
            "interleave": dataset.info.interleave,
            "data_size": dataset.info.data_path.stat().st_size,
            "data_modified_ns": dataset.info.data_path.stat().st_mtime_ns,
            "header_size": dataset.info.header_path.stat().st_size,
            "header_sha256": _sha256(dataset.info.header_path),
        }
        if wavelengths is not None and wavelengths.size:
            finite = np.asarray(wavelengths, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
        else:
            finite = np.empty(0, dtype=np.float64)
        if finite.size:
            lower = float(np.min(finite))
            upper = float(np.max(finite))
            record.update({
                "wavelength_min_nm": lower,
                "wavelength_max_nm": upper,
                "wavelength_range_nm": [lower, upper],
                "wavelength_count": int(wavelengths.size),
            })
        record["status"] = "readable"
        return record
    finally:
        dataset.close()


def _cache_key(config: Mapping[str, Any]) -> str:
    # Desktop-only state changes after a successful audit (for example, the
    # wizard advances from Evidence Review to Run).  It must not invalidate the
    # scientific snapshot and trigger the former hidden second audit.
    cache_config = deepcopy(dict(config))
    cache_config.pop("desktop", None)
    project = cache_config.get("project")
    if isinstance(project, Mapping):
        project = dict(project)
        project.pop("output_root", None)
        project.pop("description", None)
        cache_config["project"] = project
    canonical = json.dumps(_json_safe(cache_config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = sha256(canonical.encode("utf-8"))
    inputs = config.get("inputs", {}) if isinstance(config.get("inputs"), Mapping) else {}
    paths = [
        inputs.get("analysis_image", config.get("analysis_image")),
        inputs.get("rgb"),
        inputs.get("nir"),
        inputs.get("swir"),
    ]
    mask = config.get("mask", {}) if isinstance(config.get("mask"), Mapping) else {}
    paths.append(mask.get("path"))
    for value in paths:
        if not value:
            continue
        try:
            data, header = resolve_envi_paths(value)
            for path in (data, header):
                stat = path.stat()
                digest.update(str(path.resolve()).casefold().encode("utf-8"))
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        except (FileNotFoundError, ValueError):
            digest.update(str(value).casefold().encode("utf-8"))
    # A snapshot contains references and expert definitions derived from these
    # packaged assets.  Include their identities so a same-process upgrade or
    # development rebuild cannot silently reuse an incompatible ensemble.
    for path in (default_v5_database_path(), default_v5_catalog_path(), default_v5_runtime_catalog_path()):
        stat = path.stat()
        digest.update(str(path.resolve()).casefold().encode("utf-8"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


def _put_cache(snapshot: V5AuditSnapshot) -> None:
    with _CACHE_LOCK:
        _SNAPSHOT_CACHE.clear()
        _SNAPSHOT_CACHE[snapshot.cache_key] = snapshot
        while len(_SNAPSHOT_CACHE) > _MAX_CACHED_SNAPSHOTS:
            _SNAPSHOT_CACHE.popitem(last=False)


def _get_cache(key: str) -> V5AuditSnapshot | None:
    with _CACHE_LOCK:
        snapshot = _SNAPSHOT_CACHE.get(key)
        if snapshot is not None:
            _SNAPSHOT_CACHE.move_to_end(key)
        return snapshot


def _mask_settings(config: Mapping[str, Any]) -> tuple[dict[str, Any], str, Any]:
    value = config.get("mask", {})
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("V5 mask configuration must be a JSON object")
    mask = dict(value)
    mode = str(mask.get("mode", "automatic")).strip().casefold()
    if mode not in {"automatic", "external"}:
        raise ValueError("V5 mask.mode must be 'automatic' or 'external'")
    path = mask.get("path")
    if mode == "external" and not path:
        raise ValueError("V5 external mask mode requires mask.path")
    engine = str(mask.get("engine", "auto")).strip().casefold()
    legacy_aliases = {"auto", "geocore_m12", "integrated", "integrated_core_foreground_v2"}
    if engine not in legacy_aliases:
        raise ValueError("V5.3 automatic masking uses the integrated foreground model")
    mask["engine"] = "integrated_core_foreground_v2"
    return mask, mode, path


def _mask_approval(mask_config: Mapping[str, Any]) -> dict[str, Any]:
    value = mask_config.get("approval", {})
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("V5 mask.approval must be a JSON object")
    required = bool(value.get("required", False))
    approved = bool(value.get("approved", False))
    return {
        "required": required,
        "approved": approved,
        "ready": not required or approved,
        "reviewer": value.get("reviewer"),
        "reviewed_at": value.get("reviewed_at"),
        "notes": value.get("notes"),
    }


def _build_v5_material_mask(
    image: EnviDataset,
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[MaterialMaskResult, dict[str, Any]]:
    mask_config, mask_mode, mask_path = _mask_settings(config)
    engine = str(mask_config["engine"])
    source_override: str | None = None
    additional_flags: list[MaskQualityFlag] = []
    engine_record: dict[str, Any] = {
        "requested": "external" if mask_mode == "external" else engine,
        "used": "external" if mask_mode == "external" else None,
    }
    existing_mask: Any = mask_path if mask_mode == "external" else None

    inputs_config = config.get("inputs", {}) if isinstance(config.get("inputs"), Mapping) else {}
    rgb_path = inputs_config.get("rgb")
    if mask_mode == "automatic":
        # V5.3 owns the model asset.  Legacy external-source fields are ignored
        # so an old configuration cannot silently switch the production UI back
        # to a checkout-dependent implementation.
        module_root = None
        model_package = None
        availability = inspect_geocore_m12(module_root, model_package)
        engine_record["integrated_foreground_model"] = availability.to_dict()
        if availability.available and rgb_path:
            adapted = run_geocore_m12(
                rgb_path,
                (image.info.lines, image.info.samples),
                module_root=module_root,
                model_package=model_package,
                threshold=(None if mask_config.get("threshold") is None else float(mask_config["threshold"])),
                enable_postprocess=bool(mask_config.get("enable_postprocess", True)),
            )
            existing_mask = adapted.mask
            source_override = "automatic_integrated_rgb_foreground_registered"
            engine_record.update({"used": "integrated_core_foreground_v2", "result": dict(adapted.metadata)})
            if adapted.metadata.get("registration_review_required", False):
                additional_flags.append(MaskQualityFlag(
                    "MASK_REGISTRATION_REVIEW_REQUIRED",
                    "warning",
                    "The RGB M1-2 mask was scaled to the SWIR grid and requires visual registration approval.",
                    {"registration": adapted.metadata.get("registration")},
                ))
        elif not rgb_path:
            raise ValueError("自动掩膜需要在数据页提供 RGB 影像")
        else:
            raise FileNotFoundError(
                "应用内智能岩心前景模型资产不完整：" + "; ".join(availability.missing)
            )

    result = build_material_mask(
        image,
        existing_mask,
        chunk_rows=int(runtime["chunk_rows"]),
        minimum_component_pixels=int(mask_config.get("minimum_component_pixels", 64)),
        clean_existing_mask=(mask_mode == "automatic"),
        minimum_mask_fraction=float(mask_config.get("minimum_mask_fraction", 0.05)),
        source_override=source_override,
        additional_flags=additional_flags,
        minimum_interior_pixel_fraction=float(mask_config.get("minimum_interior_pixel_fraction", 0.65)),
        border_width=max(1, int(mask_config.get("edge_guard_pixels", 2))),
        strict=True,
    )
    return result, engine_record


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config.get("inputs", {}) if isinstance(config.get("inputs"), Mapping) else {}
    image = inputs.get("analysis_image", config.get("analysis_image"))
    if not image:
        raise ValueError("V5 configuration requires inputs.analysis_image")
    project = config.get("project", {}) if isinstance(config.get("project"), Mapping) else {}
    minerals = config.get("minerals", {}) if isinstance(config.get("minerals"), Mapping) else {}
    advanced = config.get("advanced", {}) if isinstance(config.get("advanced"), Mapping) else {}
    preprocessing = advanced.get("preprocessing", {}) if isinstance(advanced.get("preprocessing"), Mapping) else {}
    sampling = advanced.get("sampling", {}) if isinstance(advanced.get("sampling"), Mapping) else {}
    execution = advanced.get("execution", {}) if isinstance(advanced.get("execution"), Mapping) else {}
    library = advanced.get("library_ensemble", {}) if isinstance(advanced.get("library_ensemble"), Mapping) else {}
    artifact_user = advanced.get("artifact_control", {}) if isinstance(advanced.get("artifact_control"), Mapping) else {}
    correction_user = artifact_user.get("column_spectral_correction", {})
    if correction_user is None:
        correction_user = {}
    if not isinstance(correction_user, Mapping):
        raise ValueError("advanced.artifact_control.column_spectral_correction must be a JSON object")
    correction = {
        "enabled": True,
        "strength": 0.70,
        "radius": 2,
    }
    correction.update(dict(correction_user))
    artifact = {
        "edge_width": int(artifact_user.get("edge_width", 2)),
        "column_spectral_correction": correction,
    }
    artifact.update({key: value for key, value in artifact_user.items() if key != "column_spectral_correction"})
    runtime = {
        "analysis_image": str(image),
        "analysis_input_is_smoothed": bool(inputs.get("input_is_smoothed", False)),
        "chunk_rows": int(execution.get("chunk_rows", 64)),
        "v4": {
            "algorithm_version": "V5.3.0",
            "algorithm_name": "V5.3 Adaptive Mineral Evidence Engine",
            "project_name": str(project.get("name", "CoreSpec_Project")),
            "catalog_path": str(default_v5_runtime_catalog_path()),
            "requested_minerals": [],
            "data_physics": str(inputs.get("data_physics", "reflectance")).casefold(),
            "minimum_mask_pixels": 100,
            "preprocessing": {
                "sg_window": int(preprocessing.get("sg_window", 11)),
                "sg_polyorder": int(preprocessing.get("sg_polyorder", 2)),
                "continuum_method": "segmented_linear",
            },
            "sampling": {
                "blocks": int(sampling.get("blocks", 12)),
                "block_rows": int(sampling.get("block_rows", 64)),
                "minimum_valid_pixels_per_block": int(sampling.get("minimum_valid_pixels_per_block", 128)),
                "minimum_column_samples": int(sampling.get("minimum_column_samples", 20)),
                "maximum_sample_rows": int(sampling.get("maximum_sample_rows", 128)),
            },
            "library_ensemble": {
                "maximum_representatives_per_mineral": int(
                    library.get("maximum_representatives_per_mineral", 6)
                ),
            },
            "artifact_control": artifact,
            "adaptive_threshold_search": True,
            "disable_scene_domain_offsets": True,
            "safe_confidence_floor": True,
            "strict_evidence_gates": True,
            "shared_detection_families": {
                "phyllosilicate_2200": {
                    "groups": ["white_mica_illite", "smectites", "kaolin_2170_2205"],
                    "windows_nm": [[2100.0, 2256.0]],
                }
            },
        },
    }
    requested = minerals.get("requested", ())
    if requested is None:
        requested = ()
    if isinstance(requested, (str, bytes)) or not isinstance(requested, (tuple, list)):
        raise ValueError("minerals.requested must be a JSON array of mineral identifiers")
    runtime["v4"]["requested_minerals"] = [str(item).strip().casefold() for item in requested]
    mask, mask_mode, mask_path = _mask_settings(config)
    if mask_mode == "external":
        runtime["analysis_mask"] = str(mask_path)
    return runtime


def _requested_minerals(
    runtime: dict[str, Any],
    knowledge: Mapping[str, Any],
) -> tuple[str, ...]:
    definitions = knowledge.get("minerals", {})
    requested = tuple(dict.fromkeys(runtime["v4"].get("requested_minerals", ())))
    if not requested:
        requested = tuple(
            mineral_id
            for mineral_id, item in definitions.items()
            if item.get("support_level") == "validated_swir" and item.get("recognition_enabled", False)
        )
        runtime["v4"]["requested_minerals"] = list(requested)
    unknown = [item for item in requested if item not in definitions]
    if unknown:
        raise KeyError(f"Unknown V5 mineral(s): {', '.join(unknown)}")
    disabled = [item for item in requested if not definitions[item].get("recognition_enabled", False)]
    if disabled:
        details = "; ".join(
            f"{item}: {definitions[item].get('limitation', 'recognition is disabled')}" for item in disabled
        )
        raise ValueError(f"Selected mineral targets are not currently recognisable: {details}")
    return requested


def _internal_minerals(
    catalog: MineralCatalog,
    requested: tuple[str, ...],
    include_confusers: bool,
) -> tuple[str, ...]:
    result = list(requested)
    if include_confusers:
        for mineral_id in tuple(result):
            mineral = catalog.mineral(mineral_id)
            for confuser in mineral.confusers:
                if (
                    confuser in catalog.minerals
                    and catalog.mineral(confuser).group_id == mineral.group_id
                    and confuser not in result
                ):
                    result.append(confuser)
    return tuple(result)


def _selected_reference_records(library: V5ReferenceLibrary) -> list[dict[str, Any]]:
    wavelengths = [float(item) for item in library.wavelengths_nm]
    records: list[dict[str, Any]] = []
    for candidate in library.selected_candidates:
        record = candidate.to_record(include_curve=True)
        record.update({
            "name": candidate.source_name,
            "raw_name": candidate.source_name,
            "source": candidate.source_id,
            "score": None if not np.isfinite(candidate.selection_score) else float(candidate.selection_score),
            "wavelengths_nm": wavelengths,
        })
        records.append(record)
    return records


def _write_audit_mask_preview(image: EnviDataset, mask: np.ndarray, cache_key: str) -> str | None:
    wavelengths = image.info.wavelengths_nm
    if wavelengths is None or not wavelengths.size:
        return None
    try:
        band = int(np.argmin(np.abs(np.asarray(wavelengths) - 1600.0)))
        reflectance = image.read_rows(0, image.info.lines, bands=[band])[..., 0]
        material = np.asarray(mask, dtype=bool)
        gray = _stretched_gray(reflectance, material)
        rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float64)
        rgb[~material] *= 0.12
        rgb[material] = 0.82 * rgb[material] + 0.18 * np.array([80.0, 190.0, 255.0])
        output = Path(tempfile.gettempdir()) / "corespec_mapper_v5_audit" / cache_key[:20] / "material_mask.png"
        write_png(output, np.clip(rgb, 0, 255).astype(np.uint8))
        return str(output)
    except (OSError, ValueError):
        return None


def _threshold_preview(catalog: MineralCatalog, requested: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_id in catalog.active_groups(requested):
        group = catalog.group(group_id)
        percentile_bounds, absolute_bounds = catalog_safe_search_bounds(group)
        result[group_id] = {
            policy: {
                "parameter": f"{policy} SAM threshold",
                "candidate_range": {
                    "column_percentile": list(percentile_bounds),
                    "absolute_sam_rad": list(absolute_bounds),
                },
                "final": "运行阶段按空间证据自动解析",
                "source": "V5 Catalog 安全包络 + 分层空间代理目标",
            }
            for policy in ("conservative", "balanced", "sensitive")
        }
    return result


def _prepare_snapshot(
    config: Mapping[str, Any],
    *,
    callback: Progress,
    token: CancellationToken,
    progress_base: float,
    progress_span: float,
) -> V5AuditSnapshot:
    started = monotonic()
    runtime = _runtime_config(config)
    key = _cache_key(config)
    catalog = MineralCatalog.load(default_v5_runtime_catalog_path())
    knowledge = load_v5_catalog()
    requested = _requested_minerals(runtime, knowledge)
    include_confusers = bool(
        (config.get("minerals", {}) if isinstance(config.get("minerals"), Mapping) else {}).get(
            "include_internal_confusers", True
        )
    )
    internal = _internal_minerals(catalog, requested, include_confusers)
    _emit(callback, started, "inputs", 0.03, "正在核对主分析立方体与多源伴随数据", base=progress_base, span=progress_span)
    token.raise_if_cancelled()

    inputs_config = config.get("inputs", {}) if isinstance(config.get("inputs"), Mapping) else {}
    input_records: dict[str, Any] = {
        "analysis": _input_record("analysis", runtime["analysis_image"], required=True)
    }
    for role in ("rgb", "nir", "swir"):
        value = inputs_config.get(role)
        if value:
            input_records[role] = _input_record(role, value)

    image = EnviDataset(runtime["analysis_image"])
    try:
        wavelengths = image.info.wavelengths_nm
        if wavelengths is None:
            raise ValueError("The V5 analysis image requires an ENVI wavelength vector")
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        mask_config, _, _ = _mask_settings(config)
        _emit(callback, started, "mask", 0.12, "正在构建或核验岩心材料掩膜", base=progress_base, span=progress_span)
        mask_result, mask_engine = _build_v5_material_mask(image, config, runtime)
        mask_preview = _write_audit_mask_preview(image, mask_result.mask, key)
        token.raise_if_cancelled()

        sampling = runtime["v4"]["sampling"]
        plan = build_stratified_sample_plan(
            mask_result.mask,
            desired_blocks=int(sampling["blocks"]),
            block_rows=int(sampling["block_rows"]),
            min_valid_pixels=int(sampling["minimum_valid_pixels_per_block"]),
        )
        sample_cube, sample_mask, _ = read_sample_blocks(
            image,
            mask_result.mask,
            plan,
            maximum_rows=int(sampling["maximum_sample_rows"]),
        )
        preprocessing = runtime["v4"]["preprocessing"]
        input_smoothed = runtime["analysis_input_is_smoothed"]
        processed = sample_cube if input_smoothed else savgol_smooth(
            sample_cube,
            int(preprocessing["sg_window"]),
            int(preprocessing["sg_polyorder"]),
        )
        card = build_sensor_capability_card(
            image,
            catalog,
            sample_cube=sample_cube,
            sample_mask=sample_mask,
            data_physics=runtime["v4"]["data_physics"],
            preprocessing_state=(
                "already_sg_smoothed"
                if input_smoothed
                else f"sg_{preprocessing['sg_window']}_{preprocessing['sg_polyorder']}"
            ),
        )
        provisional = resolve_mineral_support(card, catalog, wavelengths_nm=wavelengths)
        for mineral_id in requested:
            if provisional[mineral_id].level == SupportLevel.UNSUPPORTED:
                raise ValueError(
                    f"{mineral_id} is unsupported by this sensor: {'; '.join(provisional[mineral_id].reasons)}"
                )
        correction = runtime["v4"]["artifact_control"]["column_spectral_correction"]
        column_bias = robust_column_bias(processed, sample_mask, radius=int(correction["radius"]))
        processed = processed - float(correction["strength"]) * column_bias[None, :, :]
        _emit(callback, started, "capability", 0.38, "传感器能力与全深度分层样本已解析", base=progress_base, span=progress_span)
        token.raise_if_cancelled()

        fwhm, _ = extract_fwhm_nm(image)
        library_settings = runtime["v4"]["library_ensemble"]
        reference_library = build_v5_library_ensemble(
            wavelengths,
            internal,
            target_fwhm_nm=fwhm,
            data_physics=runtime["v4"]["data_physics"],
            maximum_representatives=int(library_settings["maximum_representatives_per_mineral"]),
            require_all_available=True,
        )
        ensemble = reference_library.to_v4_ensemble()
        support = resolve_mineral_support(
            card,
            catalog,
            wavelengths_nm=wavelengths,
            reference_counts=ensemble.reference_counts(),
        )
        for mineral_id in requested:
            if support[mineral_id].level == SupportLevel.UNSUPPORTED:
                raise ValueError(
                    f"{mineral_id} failed reference gating: {'; '.join(support[mineral_id].reasons)}"
                )
        advanced_config = config.get("advanced", {}) if isinstance(config.get("advanced"), Mapping) else {}
        weak_label_config = advanced_config.get("project_calibration", {})
        if weak_label_config is not None and not isinstance(weak_label_config, Mapping):
            raise ValueError("advanced.project_calibration must be a JSON object")
        weak_label_set = load_project_weak_labels(
            weak_label_config,
            (image.info.lines, image.info.samples),
            mask_result.mask,
        )
        prepared = PreparedRunInputs(
            full_mask=mask_result.mask,
            sample_plan=plan,
            sample_cube=sample_cube,
            sample_mask=sample_mask,
            processed_sample=processed,
            sensor_card=card,
            column_bias=column_bias,
            requested_minerals=requested,
            internal_minerals=internal,
            ensemble=ensemble,
            project_weak_labels=weak_label_set,
        )
        _emit(callback, started, "library", 0.78, "内置私有光谱库参考谱已优选并重采样", base=progress_base, span=progress_span)
    finally:
        image.close()

    threshold_config = config.get("threshold_trial", {})
    if threshold_config is None:
        threshold_config = {}
    if not isinstance(threshold_config, Mapping):
        raise ValueError("threshold_trial must be a JSON object")
    threshold_overrides = threshold_config.get("overrides", {})
    threshold_trial = build_v5_threshold_trial(
        catalog,
        prepared,
        wavelengths,
        continuum_method=str(runtime["v4"]["preprocessing"].get("continuum_method", "segmented_linear")),
        overrides=threshold_overrides,
    )
    runtime["v4"]["threshold_overrides"] = deepcopy(threshold_trial["runtime_overrides"])

    support_records = []
    for mineral_id, item in support.items():
        record = item.to_dict()
        knowledge_item = knowledge["minerals"].get(mineral_id, {})
        # ``support_level`` is the current sensor/runtime verdict consumed by
        # the desktop selector.  Keep the Catalog maturity label separate so
        # an unobservable mineral is never left enabled merely because it has
        # been validated on another SWIR sensor.
        record["catalog_support_level"] = knowledge_item.get("support_level", "unknown")
        record["support_level"] = item.level.value
        record["engine_support_level"] = item.level.value
        record["reference_count"] = reference_library.reference_counts().get(mineral_id, 0)
        support_records.append(record)
    mask_record = mask_result.audit.to_dict()
    approval = _mask_approval(mask_config)
    mask_record.update({
        "blocked": not mask_result.ready,
        "valid_pixels": mask_result.audit.final_mask_pixels,
        "valid_fraction": mask_result.audit.final_mask_fraction,
        "quality_flags": [flag.to_dict() for flag in mask_result.audit.flags],
        "preview_path": mask_preview,
        "engine": mask_engine,
        "approval": approval,
    })
    quality_flags = [flag.to_dict() for flag in mask_result.audit.flags]
    quality_flags.extend(
        {
            "code": "OPTIONAL_INPUT_UNREADABLE",
            "severity": "warning",
            "message": f"Optional {role} input is not readable: {record.get('message', record.get('status'))}",
            "details": dict(record),
        }
        for role, record in input_records.items()
        if role != "analysis" and record.get("status") != "readable"
    )
    quality_flags.extend(flag.to_dict() for flag in prepared.sensor_card.quality_flags)
    quality_flags.extend(
        {"code": "LIBRARY_WARNING", "severity": "warning", "message": warning, "details": {}}
        for warning in reference_library.warnings
    )
    quality_flags.extend(
        {
            "code": str(item.get("code", "THRESHOLD_TRIAL_WARNING")),
            "severity": str(item.get("severity", "warning")),
            "message": str(item.get("message", "阈值试算需要复核")),
            "details": {
                key: value
                for key, value in item.items()
                if key not in {"code", "severity", "message"}
            },
        }
        for item in threshold_trial["warnings"]
    )
    risk_summary = {
        "mask_fraction": mask_result.audit.final_mask_fraction,
        "mask_components": mask_result.audit.kept_component_count,
        "fwhm_status": prepared.sensor_card.fwhm_status,
        "fixed_column_control": "启用全深度稳健列偏差估计；只用于组级候选检测",
        "threshold_control": "阈值仅在 Catalog 安全包络内搜索；不使用无标签矿物中位数扶正",
        "threshold_trial": threshold_trial["status"],
        "ground_truth": "PPT/编录仅作定性地质约束，不作为像元级真值",
    }
    threshold_ready = threshold_trial["status"] == "resolved"
    audit_ready = bool(approval["ready"] and threshold_ready)
    audit_status = (
        "Ready"
        if audit_ready
        else "ReviewRequired"
        if threshold_ready and not approval["ready"]
        else "Blocked"
    )
    audit_record = {
        "format": "CoreSpec Mapper V5 Audit",
        "format_version": 1,
        "version": "5.3.0",
        "status": audit_status,
        "ready": audit_ready,
        "summary": (
            f"审计通过：{len(requested)} 个目标矿物、{len(reference_library.selected_candidates)} 条入选标准谱；"
            f"材料掩膜覆盖 {mask_result.audit.final_mask_fraction:.2%}。"
        ),
        "inputs": input_records,
        "sensor_capability_card": prepared.sensor_card.to_dict(),
        "mask": mask_record,
        "sample_plan": plan.to_dict(),
        "requested_minerals": list(requested),
        "internal_competitors": list(internal),
        "mineral_support": support_records,
        "library_ensemble": reference_library.to_dict(include_candidates=True, include_curves=False),
        "selected_references": _selected_reference_records(reference_library),
        "threshold_trial": {
            "status": threshold_trial["status"],
            "method": threshold_trial["method"],
            "warnings": threshold_trial["warnings"],
            "disclaimer": threshold_trial["disclaimer"],
        },
        "thresholds": threshold_trial["thresholds"],
        "project_calibration": (
            None
            if prepared.project_weak_labels is None
            else dict(prepared.project_weak_labels.audit)
        ),
        "risk_summary": risk_summary,
        "quality_flags": quality_flags,
        "audit_elapsed_seconds": monotonic() - started,
    }
    snapshot = V5AuditSnapshot(
        key,
        runtime,
        catalog,
        knowledge,
        prepared,
        mask_result,
        reference_library,
        support,
        audit_record,
    )
    _emit(callback, started, "complete", 1.0, "V5 项目审计完成，可进入证据审查", base=progress_base, span=progress_span)
    return snapshot


def audit_v5_project(
    config: Mapping[str, Any],
    *,
    progress: Progress | None = None,
    cancel_token: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancel_token or CancellationToken()
    callback = progress or (lambda event: None)
    snapshot = _prepare_snapshot(
        deepcopy(dict(config)),
        callback=callback,
        token=token,
        progress_base=0.0,
        progress_span=1.0,
    )
    _put_cache(snapshot)
    return deepcopy(snapshot.audit_record)


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _safe_project_name(value: str) -> str:
    name = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    name = name.strip(" .") or "CoreSpec_Project"
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if name.split(".", 1)[0].upper() in reserved:
        name = f"_{name}"
    return name


def _validated_run_id(value: str | None) -> str:
    if value is None:
        return _run_id()
    identifier = str(value).strip()
    if (
        not identifier
        or identifier in {".", ".."}
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in identifier)
    ):
        raise ValueError("run_id may contain only letters, digits, '-' and '_'")
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if identifier.upper() in reserved:
        raise ValueError(f"run_id is a reserved Windows path component: {identifier}")
    return identifier


def _output_inventory(run: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    critical = {"summary.json", "quality_report.json", "project.csmproj", "thresholds.json"}
    for path in sorted((item for item in run.rglob("*") if item.is_file()), key=lambda item: str(item).casefold()):
        if path.name == "run_manifest.json":
            continue
        relative = str(path.relative_to(run)).replace("\\", "/")
        record: dict[str, Any] = {"path": relative, "size": path.stat().st_size}
        if path.name in critical or path.suffix.casefold() in {".hdr", ".csmproj"}:
            record["sha256"] = _sha256(path)
        records.append(record)
    return records


def _resolved_thresholds(summary: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_id, group in summary.get("groups", {}).items():
        result[group_id] = {}
        searches = group.get("threshold_search") or {}
        for policy, search in searches.items():
            policy_settings = (
                group.get("policies", {}).get(policy, {})
                if isinstance(group.get("policies"), Mapping)
                else {}
            )
            source_code = str(policy_settings.get("threshold_source", ""))
            source = (
                "V5.3 已复核场景阈值；全图重新验证"
                if source_code == "v5_3_reviewed_threshold_trial"
                else "V5.3 Catalog 安全包络 + 场景空间代理目标"
            )
            result[group_id][policy] = {
                "parameter": f"{policy} SAM threshold",
                "candidate_range": {
                    "column_percentile": search.get("safe_percentile_bounds"),
                    "absolute_sam_rad": search.get("safe_absolute_threshold_bounds_rad"),
                    "evaluated_candidates": len(search.get("candidates", ())),
                },
                "final": {
                    "column_percentile": search.get("resolved_percentile_fraction"),
                    "absolute_sam_rad": search.get("resolved_absolute_threshold_rad"),
                },
                "source": source,
                "resolved_reason": search.get("resolved_reason"),
                "candidates": search.get("candidates", ()),
            }
    return result


def _write_aloh_subtypes(
    run: Path,
    summary: dict[str, Any],
    material_mask: np.ndarray,
    image_path: str | Path,
    start: int,
    stop: int,
) -> dict[str, Any] | None:
    group_id = "white_mica_illite"
    group_dir = run / "groups" / group_id
    feature_path = group_dir / "feature_aloh_2200_center_nm.dat"
    classes_path = group_dir / "final_balanced.dat"
    if not feature_path.is_file() or not classes_path.is_file():
        return None
    feature_dataset = EnviDataset(feature_path)
    classes_dataset = EnviDataset(classes_path)
    image = EnviDataset(image_path)
    try:
        centers = np.asarray(feature_dataset.read_rows(0, stop - start)[..., 0], dtype=np.float64)
        classes = np.asarray(classes_dataset.read_rows(0, stop - start)[..., 0])
        class_names = [str(item) for item in classes_dataset.info.metadata.get("class names", ())]
        masked_class = max(len(class_names) - 1, 1)
        core = np.asarray(material_mask[start:stop], dtype=bool)
        accepted = core & (classes > 0) & (classes < masked_class) & np.isfinite(centers)
        subtypes = np.zeros(core.shape, dtype=np.uint8)
        subtypes[accepted & (centers < 2201.0)] = 1
        subtypes[accepted & (centers >= 2201.0) & (centers < 2209.0)] = 2
        subtypes[accepted & (centers >= 2209.0)] = 3
        subtypes[~core] = 4
        names = [
            "Unclassified",
            "Short-wave Al-OH (<2201 nm)",
            "Medium-wave Al-OH (2201-2209 nm)",
            "Long-wave Al-OH (>=2209 nm)",
            "Masked Pixels",
        ]
        write_envi(
            subtypes,
            group_dir / "aloh_wavelength_subtype.dat",
            class_names=names,
            description=(
                "V5 Al-OH wavelength subtype; a spectral-composition descriptor, not certain "
                "illite-versus-muscovite phase separation"
            ),
            metadata=subset_spatial_metadata(image.info, start_line=start),
        )
        record = {
            "artifact": "groups/white_mica_illite/aloh_wavelength_subtype.dat",
            "feature": "aloh_2200_center_nm",
            "classes": names,
            "counts": {
                "short_wave": int(np.count_nonzero(subtypes == 1)),
                "medium_wave": int(np.count_nonzero(subtypes == 2)),
                "long_wave": int(np.count_nonzero(subtypes == 3)),
            },
            "interpretation": (
                "Al-OH peak-position subtype only; do not treat it as certain illite/muscovite phase separation."
            ),
        }
        if group_id in summary.get("groups", {}):
            summary["groups"][group_id]["aloh_wavelength_subtype"] = record
        return record
    finally:
        feature_dataset.close()
        classes_dataset.close()
        image.close()


def _add_v5_previews(
    run: Path,
    previews: dict[str, str],
    image_path: str | Path,
    material_mask: np.ndarray,
    start: int,
    stop: int,
) -> None:
    image = EnviDataset(image_path)
    try:
        wavelengths = image.info.wavelengths_nm
        if wavelengths is None:
            return
        band = int(np.argmin(np.abs(np.asarray(wavelengths) - 1600.0)))
        mask = np.asarray(material_mask[start:stop], dtype=bool)
        reflectance = image.read_rows(start, stop, bands=[band])[..., 0]
        gray = _stretched_gray(reflectance, mask)
        background = np.repeat(gray[..., None], 3, axis=2)
        output = run / "previews"
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        mask_rgb[mask] = np.array([235, 245, 255], dtype=np.uint8)
        mask_path = output / "material_mask.png"
        write_png(mask_path, mask_rgb)
        previews["material_mask"] = str(mask_path)

        subtype_path = run / "groups" / "white_mica_illite" / "aloh_wavelength_subtype.dat"
        if subtype_path.is_file():
            dataset = EnviDataset(subtype_path)
            try:
                classes = np.array(dataset.read_rows(0, stop - start)[..., 0], copy=True)
                names = [str(item) for item in dataset.info.metadata.get("class names", ())]
            finally:
                dataset.close()
            overlay = _overlay(background, classes, names)
            output_path = output / "aloh_wavelength_subtype.png"
            write_png(output_path, overlay)
            previews["aloh_wavelength_subtype"] = str(output_path)
    finally:
        image.close()


def _run_validated_v3_engine(
    source_config: Mapping[str, Any],
    snapshot: V5AuditSnapshot,
    run: Path,
    *,
    project_name: str,
    identifier: str,
    start_line: int,
    stop_line: int | None,
    callback: Progress,
    token: CancellationToken,
    started: float,
) -> dict[str, Any]:
    def validated_progress(stage: str, fraction: float, message: str) -> None:
        overall = 0.15 + 0.78 * float(fraction)
        elapsed = monotonic() - started
        remaining = elapsed * (1.0 - overall) / overall if overall > 0 else None
        callback(ProgressEvent(
            stage,
            overall,
            message,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=remaining,
        ))

    image = EnviDataset(snapshot.runtime_config["analysis_image"])
    try:
        resolved_stop = image.info.lines if stop_line is None else int(stop_line)
        write_envi(
            snapshot.material_mask.mask[start_line:resolved_stop].astype(np.uint8),
            run / "capability" / "material_mask.dat",
            description="V5.3 approved core-material mask",
            metadata=subset_spatial_metadata(image.info, start_line=start_line),
        )
    finally:
        image.close()

    validated = run_validated_v3_profiles(
        source_config,
        run,
        start_line=start_line,
        stop_line=stop_line,
        progress=validated_progress,
        cancel_token=token,
    )
    token.raise_if_cancelled()
    representative = validated["profiles"].get("balanced") or next(iter(validated["profiles"].values()))
    summary = {
        "version": "5.3.0",
        "algorithm": "V5.3 validated project recipe",
        "engine": "validated_v3",
        "project_name": project_name,
        "requested_minerals": list(snapshot.prepared.requested_minerals),
        "internal_competitors": list(snapshot.prepared.internal_minerals),
        "output_lines": representative["output_lines"],
        "mask_pixels": representative["mask_pixels"],
        "profiles": validated["profiles"],
        "validation_passed": bool(validated["passed"]),
    }
    quality = {
        "status": "Completed" if validated["passed"] else "Warning",
        "quality_grade": "A" if validated["passed"] else "C",
        "publishable": bool(validated["passed"]),
        "validation_engine": "historical_workflow_weak_label_regression",
        "failures": list(validated["validation"]["failures"]),
    }
    _save_json(summary, run / "summary.json")
    _save_json(validated["thresholds"], run / "thresholds.json")
    _save_json(quality, run / "quality_report.json")
    manifest = {
        "software": "CoreSpec Mapper",
        "version": "5.3.0",
        "algorithm": "V5.3 validated project recipe",
        "engine": "validated_v3",
        "project_name": project_name,
        "run_id": identifier,
        "status": quality["status"],
        "quality_grade": quality["quality_grade"],
        "publishable": quality["publishable"],
        "inputs": dict(snapshot.audit_record["inputs"]),
        "material_mask": snapshot.material_mask.audit.to_dict(),
        "mask_approval": snapshot.audit_record["mask"].get("approval", {}),
        "requested_minerals": summary["requested_minerals"],
        "internal_competitors": summary["internal_competitors"],
        "selected_references": _selected_reference_records(snapshot.reference_library),
        "threshold_trial": snapshot.audit_record.get("threshold_trial", {}),
        "thresholds": validated["thresholds"],
        "validation": validated["validation"],
        "rejection_waterfall": validated["rejection_waterfall"],
        "previews": validated["previews"],
        "quality": quality,
        "timing": {"total_seconds": monotonic() - started},
    }
    manifest["outputs"] = _output_inventory(run)
    _save_json(manifest, run / "run_manifest.json")
    callback(ProgressEvent("complete", 1.0, f"V5.3 validated run complete; quality grade {quality['quality_grade']}"))
    return {
        "run_directory": str(run),
        "status": quality["status"],
        "quality_grade": quality["quality_grade"],
        "summary": summary,
        "quality": quality,
        "manifest": manifest,
        "previews": validated["previews"],
        "threshold_trial": manifest["threshold_trial"],
        "thresholds": validated["thresholds"],
        "selected_references": manifest["selected_references"],
        "validation": validated["validation"],
        "rejection_waterfall": validated["rejection_waterfall"],
    }


def run_v5_project(
    config: Mapping[str, Any],
    output_root: str | Path,
    *,
    progress: Progress | None = None,
    cancel_token: CancellationToken | None = None,
    run_id: str | None = None,
    start_line: int = 0,
    stop_line: int | None = None,
) -> dict[str, Any]:
    started = monotonic()
    token = cancel_token or CancellationToken()
    callback = progress or (lambda event: print(f"[{event.overall_fraction * 100:6.2f}%] {event.message}", flush=True))
    source_config = deepcopy(dict(config))
    identifier = _validated_run_id(run_id)
    key = _cache_key(source_config)
    snapshot = _get_cache(key)
    snapshot_reused = snapshot is not None
    if snapshot is None:
        snapshot = _prepare_snapshot(
            source_config,
            callback=callback,
            token=token,
            progress_base=0.0,
            progress_span=0.15,
        )
        _put_cache(snapshot)
    else:
        _emit(callback, started, "audit_reuse", 0.03, "复用已审计的掩膜、样本与入选标准谱快照")
    token.raise_if_cancelled()
    if not bool(snapshot.audit_record.get("ready", False)):
        approval = snapshot.audit_record.get("mask", {}).get("approval", {})
        if approval.get("required") and not approval.get("approved"):
            raise ValueError("The material mask must be visually reviewed and approved before mapping")
        raise ValueError("The V5 project audit is not ready for mapping")

    project = source_config.get("project", {}) if isinstance(source_config.get("project"), Mapping) else {}
    project_name = str(project.get("name", "CoreSpec_Project")).strip() or "CoreSpec_Project"
    run = Path(output_root).expanduser().resolve() / _safe_project_name(project_name) / identifier
    run.mkdir(parents=True, exist_ok=False)
    project_record = {
        "format": "CoreSpec Mapper Project",
        "format_version": 3,
        "application_version": "5.3.0",
        "project_name": project_name,
        "run_id": identifier,
        "analysis_image": snapshot.runtime_config["analysis_image"],
        "mask_mode": snapshot.material_mask.audit.source,
        "output_root": str(Path(output_root).resolve()),
        "non_destructive": True,
    }
    _save_json(project_record, run / "project.csmproj")
    resolved_config = deepcopy(source_config)
    resolved_config["resolved_runtime"] = snapshot.runtime_config
    _save_json(resolved_config, run / "config.resolved.json")
    _save_json(snapshot.audit_record, run / "audit.json")
    _save_json(
        snapshot.reference_library.to_dict(include_candidates=True, include_curves=True),
        run / "library_ensemble" / "v5_selection.json",
    )

    try:
        mode = source_config.get("mode", {}) if isinstance(source_config.get("mode"), Mapping) else {}
        if str(mode.get("engine", "adaptive_v5")).strip().casefold() == "validated_v3":
            return _run_validated_v3_engine(
                source_config,
                snapshot,
                run,
                project_name=project_name,
                identifier=identifier,
                start_line=start_line,
                stop_line=stop_line,
                callback=callback,
                token=token,
                started=started,
            )

        def pipeline_progress(event: ProgressEvent) -> None:
            overall = 0.15 + 0.80 * event.overall_fraction
            elapsed = monotonic() - started
            remaining = elapsed * (1.0 - overall) / overall if overall > 0 else None
            mapped = ProgressEvent(
                event.stage,
                overall,
                event.message.replace("V4", "V5"),
                event.stage_fraction,
                event.current_block,
                elapsed,
                remaining,
            )
            callback(mapped)

        summary = run_v4(
            snapshot.runtime_config,
            run,
            start_line=start_line,
            stop_line=stop_line,
            progress=pipeline_progress,
            cancel_token=token,
            catalog=snapshot.catalog,
            prepared=snapshot.prepared,
        )
        start, stop = (int(item) for item in summary["output_lines"])
        subtype = _write_aloh_subtypes(
            run,
            summary,
            snapshot.material_mask.mask,
            snapshot.runtime_config["analysis_image"],
            start,
            stop,
        )
        _save_json(summary, run / "summary.json")
        image = EnviDataset(snapshot.runtime_config["analysis_image"])
        try:
            write_envi(
                snapshot.material_mask.mask[start:stop].astype(np.uint8),
                run / "capability" / "material_mask.dat",
                description="V5 validated core-material mask",
                metadata=subset_spatial_metadata(image.info, start_line=start),
            )
        finally:
            image.close()
        _emit(callback, started, "preview", 0.92, "正在生成三档分类与伪影诊断预览")
        previews = make_v4_previews(
            snapshot.runtime_config,
            run,
            list(summary["groups"]),
            start,
            stop,
            material_mask=snapshot.material_mask.mask,
        )
        _add_v5_previews(
            run,
            previews,
            snapshot.runtime_config["analysis_image"],
            snapshot.material_mask.mask,
            start,
            stop,
        )
        token.raise_if_cancelled()
        quality = validate_v4_run(run)
        thresholds = _resolved_thresholds(summary)
        _save_json(thresholds, run / "thresholds.json")
        _emit(callback, started, "qa", 0.97, "成果完整性、空间聚集与固定列风险门禁已完成")

        inputs = dict(snapshot.audit_record["inputs"])
        manifest = {
            "software": "CoreSpec Mapper",
            "version": "5.3.0",
            "algorithm": "V5 Adaptive Mineral Evidence Engine",
            "project_name": project_name,
            "run_id": identifier,
            "status": quality["status"],
            "quality_grade": quality["quality_grade"],
            "sensor_signature": summary["sensor_signature"],
            "inputs": inputs,
            "material_mask": snapshot.material_mask.audit.to_dict(),
            "requested_minerals": summary["requested_minerals"],
            "internal_competitors": summary["internal_competitors"],
            "spectral_database": {
                "filename": snapshot.reference_library.database_path.name,
                "sha256": snapshot.reference_library.database_sha256,
                "release": snapshot.reference_library.database_release,
            },
            "knowledge_catalog": {
                "filename": snapshot.reference_library.catalog_path.name,
                "sha256": snapshot.reference_library.catalog_sha256,
                "version": snapshot.reference_library.catalog_version,
            },
            "runtime_catalog": {
                "filename": default_v5_runtime_catalog_path().name,
                "sha256": _sha256(default_v5_runtime_catalog_path()),
                "version": snapshot.catalog.version,
            },
            "selected_references": _selected_reference_records(snapshot.reference_library),
            "threshold_trial": snapshot.audit_record.get("threshold_trial", {}),
            "thresholds": thresholds,
            "aloh_wavelength_subtype": subtype,
            "previews": previews,
            "quality": quality,
            "timing": {
                "audit_snapshot_reused": snapshot_reused,
                "audit_snapshot_seconds": snapshot.audit_record.get("audit_elapsed_seconds"),
                "mapping_seconds_before_qa": summary.get("elapsed_seconds_before_qa"),
                "total_seconds": monotonic() - started,
            },
            "manifest_inventory_policy": (
                "run_manifest.json is intentionally excluded from its own inventory to avoid a recursive, stale self-hash."
            ),
        }
        manifest["outputs"] = _output_inventory(run)
        _save_json(manifest, run / "run_manifest.json")
        callback(ProgressEvent("complete", 1.0, f"V5 运行完成，质量等级 {quality['quality_grade']}"))
        return {
            "run_directory": str(run),
            "status": quality["status"],
            "quality_grade": quality["quality_grade"],
            "summary": summary,
            "quality": quality,
            "manifest": manifest,
            "previews": previews,
            "threshold_trial": manifest["threshold_trial"],
            "thresholds": thresholds,
            "selected_references": manifest["selected_references"],
        }
    except Exception as exc:
        failure = {
            "software": "CoreSpec Mapper",
            "version": "5.3.0",
            "project_name": project_name,
            "run_id": identifier,
            "status": "Cancelled" if token.cancelled else "Failed",
            "error": str(exc),
            "elapsed_seconds": monotonic() - started,
        }
        _save_json(failure, run / "run_manifest.json")
        raise
