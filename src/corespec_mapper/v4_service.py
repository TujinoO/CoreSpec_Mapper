from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any, Callable
import json

from .algorithms import robust_column_bias, savgol_smooth
from .catalog import MineralCatalog
from .envi import EnviDataset, derive_mask, resolve_envi_paths
from .qa import validate_v4_run
from .sampling import build_stratified_sample_plan, read_sample_blocks
from .sensor import build_sensor_capability_card, extract_fwhm_nm, resolve_mineral_support
from .v4_library import build_v4_library_ensemble
from .v4_models import CancellationToken, ProgressEvent, SupportLevel
from .v4_pipeline import run_v4
from .v4_preview import make_v4_previews


def _save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _requested_minerals(config: dict[str, Any], catalog: MineralCatalog, provisional: dict[str, Any]) -> tuple[str, ...]:
    requested = tuple(str(item).casefold() for item in config["v4"].get("requested_minerals", ()))
    if requested:
        return requested
    return tuple(
        mineral.mineral_id
        for mineral in catalog.minerals.values()
        if mineral.support_level == "validated_swir" and provisional[mineral.mineral_id].level != SupportLevel.UNSUPPORTED
    )


def _internal_minerals(catalog: MineralCatalog, requested: tuple[str, ...]) -> tuple[str, ...]:
    values = list(dict.fromkeys(requested))
    for mineral_id in tuple(values):
        mineral = catalog.mineral(mineral_id)
        for confuser in mineral.confusers:
            if confuser in catalog.minerals and catalog.mineral(confuser).group_id == mineral.group_id and confuser not in values:
                values.append(confuser)
    return tuple(values)


def audit_v4_project(
    config: dict[str, Any],
    *,
    scan_library: bool = True,
    progress: Callable[[ProgressEvent], None] | None = None,
    cancel_token: CancellationToken | None = None,
) -> dict[str, Any]:
    started = monotonic()
    token = cancel_token or CancellationToken()
    callback = progress or (lambda event: None)

    def emit(stage: str, fraction: float, message: str) -> None:
        elapsed = monotonic() - started
        remaining = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
        callback(ProgressEvent(stage, fraction, message, elapsed_seconds=elapsed, estimated_remaining_seconds=remaining))

    v4 = config.get("v4")
    if not isinstance(v4, dict):
        raise ValueError("The configuration does not contain a V4 section")
    emit("audit", 0.02, "Opening ENVI image, mask, and mineral catalog")
    token.raise_if_cancelled()
    catalog = MineralCatalog.load(v4.get("catalog_path"))
    image = EnviDataset(config["analysis_image"])
    mask_dataset = EnviDataset(config["analysis_mask"])
    try:
        if (image.info.lines, image.info.samples) != (mask_dataset.info.lines, mask_dataset.info.samples):
            raise ValueError("Analysis image and mask dimensions do not match")
        wavelengths = image.info.wavelengths_nm
        if wavelengths is None:
            raise ValueError("Analysis image has no wavelength vector")
        default_bands = [min(20, mask_dataset.info.bands - 1), min(mask_dataset.info.bands // 2, mask_dataset.info.bands - 1), min(190, mask_dataset.info.bands - 1)]
        mask_bands = sorted(set(int(item) for item in v4.get("mask_bands", default_bands)))
        mask = derive_mask(mask_dataset, bands=mask_bands, chunk_rows=int(config.get("chunk_rows", 64)))
        token.raise_if_cancelled()
        emit("audit", 0.20, "Validated image dimensions, wavelengths, and core mask")
        sampling = v4.get("sampling", {})
        plan = build_stratified_sample_plan(
            mask,
            desired_blocks=int(sampling.get("blocks", 12)),
            block_rows=int(sampling.get("block_rows", 64)),
            min_valid_pixels=int(sampling.get("minimum_valid_pixels_per_block", 128)),
        )
        sample_cube, sample_mask, _ = read_sample_blocks(image, mask, plan)
        token.raise_if_cancelled()
        emit("sampling", 0.38, f"Read {len(plan.blocks)} full-depth calibration blocks")
        input_is_smoothed = bool(config.get("analysis_input_is_smoothed", False))
        sg_window = int(v4.get("preprocessing", {}).get("sg_window", config.get("sg_window", 11)))
        sg_order = int(v4.get("preprocessing", {}).get("sg_polyorder", config.get("sg_polyorder", 2)))
        processed = sample_cube if input_is_smoothed else savgol_smooth(sample_cube, sg_window, sg_order)
        correction = v4.get("artifact_control", {}).get("column_spectral_correction", {})
        if bool(correction.get("enabled", True)):
            bias = robust_column_bias(processed, sample_mask, radius=int(correction.get("radius", 2)))
            processed = processed - float(correction.get("strength", 0.70)) * bias[None, :, :]
        card = build_sensor_capability_card(
            image,
            catalog,
            sample_cube=sample_cube,
            sample_mask=sample_mask,
            data_physics=v4.get("data_physics"),
            preprocessing_state="already_sg_smoothed" if input_is_smoothed else f"sg_{sg_window}_{sg_order}",
        )
        provisional = resolve_mineral_support(card, catalog, wavelengths_nm=wavelengths)
        requested = _requested_minerals(config, catalog, provisional)
        internal = tuple(item for item in _internal_minerals(catalog, requested) if provisional[item].level != SupportLevel.UNSUPPORTED)
        token.raise_if_cancelled()
        emit("capability", 0.55, "Resolved sensor capability and mineral observability")
        reference_counts: dict[str, int] | None = None
        library_summary: dict[str, Any] | None = None
        if scan_library:
            emit("library", 0.62, "Scanning and resampling candidate reference spectra")
            fwhm, _ = extract_fwhm_nm(image)
            scan_minerals = tuple(
                mineral_id
                for mineral_id, item in provisional.items()
                if item.level != SupportLevel.UNSUPPORTED
                and item.selected_expert == "swir_reflectance"
            )
            ensemble = build_v4_library_ensemble(
                v4["spectral_library_root"],
                wavelengths,
                catalog,
                scan_minerals,
                target_fwhm_nm=fwhm,
                scene_cube=processed,
                scene_mask=sample_mask,
                maximum_representatives=int(v4.get("library_ensemble", {}).get("maximum_representatives_per_mineral", 6)),
                dedup_angle_rad=float(v4.get("library_ensemble", {}).get("dedup_angle_rad", 0.02)),
                allow_missing=True,
            )
            reference_counts = ensemble.reference_counts()
            library_summary = {
                "selected_spectra": len(ensemble.selected_candidates),
                "selected_counts": reference_counts,
                "scanned_libraries": len(ensemble.libraries),
                "warnings": list(ensemble.warnings),
            }
            token.raise_if_cancelled()
            emit("library", 0.92, f"Selected {len(ensemble.selected_candidates)} reference spectra")
        support = resolve_mineral_support(card, catalog, wavelengths_nm=wavelengths, reference_counts=reference_counts)
        ready = all(support[mineral].level != SupportLevel.UNSUPPORTED for mineral in requested)
        result = {
            "ready": ready,
            "image": {
                "path": str(image.info.data_path),
                "lines": image.info.lines,
                "samples": image.info.samples,
                "bands": image.info.bands,
                "interleave": image.info.interleave,
            },
            "mask": {
                "path": str(mask_dataset.info.data_path),
                "valid_pixels": int(mask.sum()),
                "valid_fraction": float(mask.mean()),
            },
            "sensor_capability_card": card.to_dict(),
            "sample_plan": plan.to_dict(),
            "column_spectral_correction": {
                "enabled": bool(correction.get("enabled", True)),
                "strength": float(correction.get("strength", 0.70)),
                "radius": int(correction.get("radius", 2)),
            },
            "requested_minerals": list(requested),
            "internal_competitors": list(internal),
            "mineral_support": {key: value.to_dict() for key, value in support.items()},
            "library": library_summary,
        }
        emit("complete", 1.0, "V4 project audit completed")
        return result
    finally:
        image.close()
        mask_dataset.close()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _input_record(path: str | Path) -> dict[str, Any]:
    data, header = resolve_envi_paths(path)
    return {
        "data_path": str(data.resolve()),
        "header_path": str(header.resolve()),
        "data_size": data.stat().st_size,
        "data_modified_ns": data.stat().st_mtime_ns,
        "header_size": header.stat().st_size,
        "header_sha256": _sha256(header),
    }


def _output_inventory(run: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    critical_names = {"summary.json", "quality_report.json", "final_conservative.dat", "final_balanced.dat", "final_sensitive.dat"}
    for path in sorted((item for item in run.rglob("*") if item.is_file()), key=lambda item: str(item).casefold()):
        relative = str(path.relative_to(run))
        record = {"path": relative, "size": path.stat().st_size}
        if path.name in critical_names or path.suffix.casefold() in {".hdr", ".csmproj"}:
            record["sha256"] = _sha256(path)
        records.append(record)
    return records


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def run_v4_project(
    config: dict[str, Any],
    output_root: str | Path,
    *,
    progress: Callable[[ProgressEvent], None] | None = None,
    cancel_token: CancellationToken | None = None,
    run_id: str | None = None,
    start_line: int = 0,
    stop_line: int | None = None,
) -> dict[str, Any]:
    v4 = config.get("v4", {})
    project_name = str(v4.get("project_name", "CoreSpec_Project")).strip() or "CoreSpec_Project"
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in project_name)
    identifier = run_id or _run_id()
    run = Path(output_root).expanduser().resolve() / safe_name / identifier
    if run.exists():
        raise FileExistsError(f"Run directory already exists: {run}")
    run.mkdir(parents=True)
    token = cancel_token or CancellationToken()
    callback = progress or (lambda event: print(f"[{event.overall_fraction * 100:6.2f}%] {event.stage}: {event.message}", flush=True))
    resolved = json.loads(json.dumps(config))
    resolved.setdefault("v4", {})["run_id"] = identifier
    project_record = {
        "format": "CoreSpec Mapper Project",
        "format_version": 1,
        "project_name": project_name,
        "run_id": identifier,
        "analysis_image": str(config["analysis_image"]),
        "analysis_mask": str(config["analysis_mask"]),
        "output_root": str(Path(output_root).resolve()),
    }
    _save_json(project_record, run / "project.csmproj")
    _save_json(resolved, run / "config.resolved.json")
    try:
        audit = audit_v4_project(config, scan_library=False, cancel_token=token)
        _save_json(audit, run / "audit.json")
        if not audit["ready"]:
            raise ValueError("V4 project audit did not pass capability gating")
        summary = run_v4(
            resolved,
            run,
            start_line=start_line,
            stop_line=stop_line,
            progress=callback,
            cancel_token=token,
        )
        group_ids = list(summary["groups"])
        previews = make_v4_previews(resolved, run, group_ids, int(summary["output_lines"][0]), int(summary["output_lines"][1]))
        quality = validate_v4_run(run)
        manifest = {
            "software": "CoreSpec Mapper",
            "version": "4.0.0",
            "algorithm": "Adaptive Mineral Evidence Engine",
            "project_name": project_name,
            "run_id": identifier,
            "status": quality["status"],
            "quality_grade": quality["quality_grade"],
            "sensor_signature": summary["sensor_signature"],
            "inputs": {
                "analysis_image": _input_record(config["analysis_image"]),
                "analysis_mask": _input_record(config["analysis_mask"]),
                "spectral_library_root": str(Path(v4["spectral_library_root"]).resolve()),
            },
            "requested_minerals": summary["requested_minerals"],
            "previews": previews,
            "quality": quality,
        }
        _save_json(manifest, run / "run_manifest.json")
        manifest["outputs"] = _output_inventory(run)
        _save_json(manifest, run / "run_manifest.json")
        callback(ProgressEvent("complete", 1.0, f"V4 run completed with quality grade {quality['quality_grade']}"))
        return {"run_directory": str(run), "summary": summary, "quality": quality, "manifest": manifest}
    except Exception as exc:
        failure = {
            "software": "CoreSpec Mapper",
            "version": "4.0.0",
            "project_name": project_name,
            "run_id": identifier,
            "status": "Cancelled" if token.cancelled else "Failed",
            "error": str(exc),
        }
        _save_json(failure, run / "run_manifest.json")
        raise
