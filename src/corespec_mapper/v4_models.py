from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from threading import Event
from typing import Any


class SupportLevel(str, Enum):
    SUPPORTED = "supported"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class QualityFlag:
    code: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["severity"] = self.severity.value
        return value


@dataclass(frozen=True)
class SensorCapabilityCard:
    sensor_signature: str
    spectral_domain: str
    data_physics: str
    wavelength_min_nm: float
    wavelength_max_nm: float
    band_count: int
    valid_band_count: int
    median_spacing_nm: float
    fwhm_status: str
    fwhm_median_nm: float | None
    estimated_snr: float | None
    bad_band_indices: tuple[int, ...]
    detector_axis: str
    preprocessing_state: str
    implemented_experts: tuple[str, ...]
    quality_flags: tuple[QualityFlag, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["quality_flags"] = [flag.to_dict() for flag in self.quality_flags]
        return value


@dataclass(frozen=True)
class MineralSupport:
    mineral_id: str
    display_name: str
    level: SupportLevel
    score: float
    selected_expert: str | None
    reasons: tuple[str, ...]
    required_windows_covered: float
    valid_bands_in_required_windows: int
    reference_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["level"] = self.level.value
        return value


@dataclass(frozen=True)
class SampleBlock:
    start_line: int
    stop_line: int
    reason: str
    valid_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SamplePlan:
    blocks: tuple[SampleBlock, ...]
    total_lines: int
    total_valid_pixels: int
    strategy: str = "full_depth_stratified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "total_lines": self.total_lines,
            "total_valid_pixels": self.total_valid_pixels,
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    overall_fraction: float
    message: str
    stage_fraction: float | None = None
    current_block: tuple[int, int] | None = None
    elapsed_seconds: float | None = None
    estimated_remaining_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunCancelled(RuntimeError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelled("CoreSpec Mapper run cancelled")
