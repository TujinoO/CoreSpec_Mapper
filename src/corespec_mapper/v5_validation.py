from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np

from .envi import EnviDataset


@dataclass(frozen=True)
class BinaryAgreement:
    reference_pixels: int
    candidate_pixels: int
    intersection_pixels: int
    union_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    iou: float
    precision: float
    recall: float
    area_difference_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionGate:
    minimum_iou: float = 0.90
    minimum_precision: float = 0.90
    minimum_recall: float = 0.95
    maximum_area_difference_fraction: float = 0.05

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RegressionGate":
        return cls(**dict(value or {}))

    def failures(self, agreement: BinaryAgreement) -> tuple[str, ...]:
        failures: list[str] = []
        if agreement.iou < self.minimum_iou:
            failures.append(f"IoU {agreement.iou:.4f} < {self.minimum_iou:.4f}")
        if agreement.precision < self.minimum_precision:
            failures.append(f"precision {agreement.precision:.4f} < {self.minimum_precision:.4f}")
        if agreement.recall < self.minimum_recall:
            failures.append(f"recall {agreement.recall:.4f} < {self.minimum_recall:.4f}")
        if agreement.area_difference_fraction > self.maximum_area_difference_fraction:
            failures.append(
                "area difference "
                f"{agreement.area_difference_fraction:.4f} > {self.maximum_area_difference_fraction:.4f}"
            )
        return tuple(failures)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def binary_agreement(reference: np.ndarray, candidate: np.ndarray, domain: np.ndarray | None = None) -> BinaryAgreement:
    reference_mask = np.asarray(reference, dtype=bool)
    candidate_mask = np.asarray(candidate, dtype=bool)
    if reference_mask.shape != candidate_mask.shape:
        raise ValueError(
            f"Reference and candidate shapes differ: {reference_mask.shape} != {candidate_mask.shape}"
        )
    if domain is not None:
        domain_mask = np.asarray(domain, dtype=bool)
        if domain_mask.shape != reference_mask.shape:
            raise ValueError("Comparison domain and classification shapes differ")
        reference_mask = reference_mask & domain_mask
        candidate_mask = candidate_mask & domain_mask
    true_positive = int(np.count_nonzero(reference_mask & candidate_mask))
    false_positive = int(np.count_nonzero(~reference_mask & candidate_mask))
    false_negative = int(np.count_nonzero(reference_mask & ~candidate_mask))
    reference_pixels = true_positive + false_negative
    candidate_pixels = true_positive + false_positive
    union = true_positive + false_positive + false_negative
    reference_scale = max(reference_pixels, 1)
    return BinaryAgreement(
        reference_pixels=reference_pixels,
        candidate_pixels=candidate_pixels,
        intersection_pixels=true_positive,
        union_pixels=union,
        false_positive_pixels=false_positive,
        false_negative_pixels=false_negative,
        iou=true_positive / union if union else 1.0,
        precision=true_positive / candidate_pixels if candidate_pixels else (1.0 if reference_pixels == 0 else 0.0),
        recall=true_positive / reference_pixels if reference_pixels else (1.0 if candidate_pixels == 0 else 0.0),
        area_difference_fraction=abs(candidate_pixels - reference_pixels) / reference_scale,
    )


def read_nonzero_mask(path: str | Path, *, chunk_rows: int = 64) -> np.ndarray:
    dataset = EnviDataset(path)
    try:
        bands = (
            (0,)
            if dataset.info.bands == 1
            else tuple(dict.fromkeys((0, dataset.info.bands // 2, dataset.info.bands - 1)))
        )
        result = np.zeros((dataset.info.lines, dataset.info.samples), dtype=bool)
        for start, stop, cube in dataset.iter_rows(chunk_rows=chunk_rows, bands=bands):
            values = np.asarray(cube)
            result[start:stop] = np.any(np.isfinite(values) & (np.abs(values) > 1e-10), axis=-1)
        return result
    finally:
        dataset.close()


def read_classification(path: str | Path) -> tuple[np.ndarray, tuple[str, ...]]:
    dataset = EnviDataset(path)
    try:
        if dataset.info.bands != 1:
            raise ValueError(f"Classification raster must contain one band: {path}")
        values = np.array(dataset.read_rows(0, dataset.info.lines)[..., 0], copy=True)
        names = tuple(str(item) for item in dataset.info.metadata.get("class names", ()))
        if not names:
            raise ValueError(f"Classification raster has no ENVI class names: {path}")
        return values, names
    finally:
        dataset.close()


def _class_id(names: Sequence[str], mineral: str) -> int:
    lookup = {str(name).strip().casefold(): index for index, name in enumerate(names)}
    try:
        return lookup[mineral.strip().casefold()]
    except KeyError as exc:
        raise KeyError(f"Classification does not contain class {mineral!r}; available={tuple(names)!r}") from exc


def compare_classification_rasters(
    reference_path: str | Path,
    candidate_path: str | Path,
    minerals: Sequence[str],
    *,
    domain: np.ndarray | None = None,
) -> dict[str, BinaryAgreement]:
    reference, reference_names = read_classification(reference_path)
    candidate, candidate_names = read_classification(candidate_path)
    if reference.shape != candidate.shape:
        raise ValueError(f"Classification raster shapes differ: {reference.shape} != {candidate.shape}")
    return {
        mineral: binary_agreement(
            reference == _class_id(reference_names, mineral),
            candidate == _class_id(candidate_names, mineral),
            domain,
        )
        for mineral in minerals
    }


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def validate_v5_regression(config: Mapping[str, Any], *, config_dir: str | Path | None = None) -> dict[str, Any]:
    """Compare a candidate V5/V5.3 run with frozen NC-style weak-label references.

    The report intentionally labels historical classifications as weak labels.  A
    byte-identical old workflow is useful for regression, but is not independent
    XRD/Raman mineralogical truth.
    """
    base = Path(config_dir or ".")
    mask_config = config.get("mask")
    if not isinstance(mask_config, Mapping):
        raise ValueError("Regression configuration requires a mask object")
    reference_mask_path = _resolve_path(str(mask_config["reference"]), base)
    candidate_mask_path = _resolve_path(str(mask_config["candidate"]), base)
    reference_mask = read_nonzero_mask(reference_mask_path)
    candidate_mask = read_nonzero_mask(candidate_mask_path)
    mask_agreement = binary_agreement(reference_mask, candidate_mask)
    mask_gate = RegressionGate.from_mapping(mask_config.get("gate"))
    mask_failures = mask_gate.failures(mask_agreement)

    classification_results: dict[str, Any] = {}
    failed_items: list[str] = [f"mask: {item}" for item in mask_failures]
    for item in config.get("classifications", ()):
        if not isinstance(item, Mapping):
            raise ValueError("Each classification regression item must be an object")
        name = str(item["name"])
        domain_name = str(item.get("domain", "reference_mask"))
        if domain_name == "reference_mask":
            domain = reference_mask
        elif domain_name == "shared_mask":
            domain = reference_mask & candidate_mask
        elif domain_name == "full_grid":
            domain = None
        else:
            raise ValueError(f"Unknown classification comparison domain: {domain_name}")
        gate = RegressionGate.from_mapping(item.get("gate"))
        comparisons = compare_classification_rasters(
            _resolve_path(str(item["reference"]), base),
            _resolve_path(str(item["candidate"]), base),
            tuple(str(value) for value in item["minerals"]),
            domain=domain,
        )
        mineral_records: dict[str, Any] = {}
        for mineral, agreement in comparisons.items():
            failures = gate.failures(agreement)
            mineral_records[mineral] = {
                "passed": not failures,
                "agreement": agreement.to_dict(),
                "failures": list(failures),
            }
            failed_items.extend(f"{name}/{mineral}: {failure}" for failure in failures)
        classification_results[name] = {
            "evidence_level": "historical_workflow_weak_label",
            "comparison_domain": domain_name,
            "gate": gate.to_dict(),
            "minerals": mineral_records,
        }

    return {
        "format": "CoreSpec Mapper V5.3 Regression Report",
        "format_version": 1,
        "passed": not failed_items,
        "evidence_notice": (
            "ENVI rule/class rasters test numerical workflow equivalence; historical final classes are weak labels, "
            "not independent mineralogical ground truth."
        ),
        "mask": {
            "reference": str(reference_mask_path),
            "candidate": str(candidate_mask_path),
            "passed": not mask_failures,
            "gate": mask_gate.to_dict(),
            "agreement": mask_agreement.to_dict(),
            "failures": list(mask_failures),
        },
        "classifications": classification_results,
        "failures": failed_items,
    }


def validate_v5_regression_file(config_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    report = validate_v5_regression(config, config_dir=path.parent)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
