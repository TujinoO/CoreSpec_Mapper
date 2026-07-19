from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import re


@dataclass(frozen=True)
class ExpertDefinition:
    expert_id: str
    display_name: str
    implemented: bool
    data_physics: tuple[str, ...]
    nominal_domain: str


@dataclass(frozen=True)
class FeatureDefinition:
    feature_id: str
    kind: str
    window_nm: tuple[float, float]


@dataclass(frozen=True)
class GroupDefinition:
    group_id: str
    display_name: str
    expert_id: str
    detection_windows_nm: tuple[tuple[float, float], ...]
    classification_windows_nm: tuple[tuple[float, float], ...]
    features: tuple[FeatureDefinition, ...]
    score_weights: Mapping[str, float]
    policies: Mapping[str, Mapping[str, Any]]
    domain_calibration_strength: float
    detection_best_k: int
    classification_best_k: int
    spatial_cleanup: Mapping[str, Any]


@dataclass(frozen=True)
class MineralDefinition:
    mineral_id: str
    display_name_en: str
    display_name_zh: str
    formula: str
    group_id: str
    name_patterns: tuple[str, ...]
    reject_patterns: tuple[str, ...]
    confusers: tuple[str, ...]
    canonical_anchors: tuple[str, ...]
    experts: Mapping[str, Mapping[str, Any]]
    support_level: str

    @property
    def display_name(self) -> str:
        return self.display_name_en


@dataclass(frozen=True)
class MineralCatalog:
    version: str
    experts: Mapping[str, ExpertDefinition]
    groups: Mapping[str, GroupDefinition]
    minerals: Mapping[str, MineralDefinition]
    global_reject_patterns: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MineralCatalog":
        source = default_catalog_path() if path is None else Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MineralCatalog":
        experts: dict[str, ExpertDefinition] = {}
        for expert_id, record in value.get("experts", {}).items():
            experts[expert_id] = ExpertDefinition(
                expert_id=expert_id,
                display_name=str(record.get("display_name", expert_id)),
                implemented=bool(record.get("implemented", False)),
                data_physics=tuple(str(item).casefold() for item in record.get("data_physics", [])),
                nominal_domain=str(record.get("nominal_domain", "unknown")),
            )

        groups: dict[str, GroupDefinition] = {}
        for group_id, record in value.get("groups", {}).items():
            features = tuple(
                FeatureDefinition(
                    feature_id=str(item["id"]),
                    kind=str(item["type"]),
                    window_nm=_window(item["window_nm"]),
                )
                for item in record.get("features", [])
            )
            groups[group_id] = GroupDefinition(
                group_id=group_id,
                display_name=str(record.get("display_name", group_id)),
                expert_id=str(record["expert"]),
                detection_windows_nm=_windows(record["detection_windows_nm"]),
                classification_windows_nm=_windows(record["classification_windows_nm"]),
                features=features,
                score_weights={str(key): float(number) for key, number in record.get("score_weights", {}).items()},
                policies={
                    str(policy): dict(settings) for policy, settings in record.get("policies", {}).items()
                },
                domain_calibration_strength=float(record.get("domain_calibration_strength", 0.0)),
                detection_best_k=int(record.get("detection_best_k", 1)),
                classification_best_k=int(record.get("classification_best_k", 2)),
                spatial_cleanup=dict(record.get("spatial_cleanup", {})),
            )

        minerals: dict[str, MineralDefinition] = {}
        for mineral_id, record in value.get("minerals", {}).items():
            experts_record = {
                str(expert_id): dict(settings) for expert_id, settings in record.get("experts", {}).items()
            }
            minerals[mineral_id] = MineralDefinition(
                mineral_id=mineral_id,
                display_name_en=str(record.get("display_name_en", mineral_id.title())),
                display_name_zh=str(record.get("display_name_zh", "")),
                formula=str(record.get("formula", "")),
                group_id=str(record["group"]),
                name_patterns=tuple(str(pattern) for pattern in record.get("name_patterns", [])),
                reject_patterns=tuple(str(pattern) for pattern in record.get("reject_patterns", [])),
                confusers=tuple(str(item) for item in record.get("confusers", [])),
                canonical_anchors=tuple(str(item) for item in record.get("canonical_anchors", [])),
                experts=experts_record,
                support_level=str(record.get("support_level", "experimental")),
            )

        catalog = cls(
            version=str(value.get("version", "unknown")),
            experts=experts,
            groups=groups,
            minerals=minerals,
            global_reject_patterns=tuple(str(pattern) for pattern in value.get("global_reject_patterns", [])),
        )
        catalog.validate()
        return catalog

    def validate(self) -> None:
        if not self.experts or not self.groups or not self.minerals:
            raise ValueError("Mineral catalog requires experts, groups, and minerals")
        for group in self.groups.values():
            if group.expert_id not in self.experts:
                raise ValueError(f"Group {group.group_id} references unknown expert {group.expert_id}")
            if set(group.policies) != {"conservative", "balanced", "sensitive"}:
                raise ValueError(f"Group {group.group_id} requires conservative, balanced, and sensitive policies")
            for policy, settings in group.policies.items():
                percentile = float(settings.get("column_percentile", 0.0))
                if not 0.0 < percentile < 0.5:
                    raise ValueError(f"Invalid {policy} column percentile for {group.group_id}")
        for mineral in self.minerals.values():
            if mineral.group_id not in self.groups:
                raise ValueError(f"Mineral {mineral.mineral_id} references unknown group {mineral.group_id}")
            for expert_id in mineral.experts:
                if expert_id not in self.experts:
                    raise ValueError(f"Mineral {mineral.mineral_id} references unknown expert {expert_id}")
            for pattern in (*mineral.name_patterns, *mineral.reject_patterns):
                re.compile(pattern, re.IGNORECASE)
        for pattern in self.global_reject_patterns:
            re.compile(pattern, re.IGNORECASE)

    def mineral(self, mineral_id: str) -> MineralDefinition:
        key = str(mineral_id).casefold()
        if key not in self.minerals:
            raise KeyError(f"Unknown mineral: {mineral_id}")
        return self.minerals[key]

    def group(self, group_id: str) -> GroupDefinition:
        if group_id not in self.groups:
            raise KeyError(f"Unknown mineral group: {group_id}")
        return self.groups[group_id]

    def group_minerals(self, group_id: str, *, expert_id: str | None = None) -> tuple[str, ...]:
        return tuple(
            mineral.mineral_id
            for mineral in self.minerals.values()
            if mineral.group_id == group_id and (expert_id is None or expert_id in mineral.experts)
        )

    def active_groups(self, mineral_ids: Iterable[str]) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {}
        for mineral_id in mineral_ids:
            mineral = self.mineral(mineral_id)
            result.setdefault(mineral.group_id, []).append(mineral.mineral_id)
        return {group: tuple(values) for group, values in result.items()}

    def match_spectrum_name(
        self,
        name: str,
        allowed_minerals: Iterable[str] | None = None,
    ) -> tuple[str | None, str | None]:
        allowed = set(self.minerals) if allowed_minerals is None else {str(item).casefold() for item in allowed_minerals}
        matches: list[str] = []
        for mineral_id in allowed:
            mineral = self.minerals.get(mineral_id)
            if mineral is None:
                continue
            if any(re.search(pattern, name, re.IGNORECASE) for pattern in mineral.name_patterns):
                matches.append(mineral_id)
        if not matches:
            return None, None
        matches.sort()
        if len(matches) > 1:
            return matches[0], "mixed_target_minerals"
        mineral = self.minerals[matches[0]]
        if any(re.search(pattern, name, re.IGNORECASE) for pattern in self.global_reject_patterns):
            return mineral.mineral_id, "rock_or_mixture_name"
        if any(re.search(pattern, name, re.IGNORECASE) for pattern in mineral.reject_patterns):
            return mineral.mineral_id, "mineral_specific_rejected_name"
        return mineral.mineral_id, None

    def to_summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "experts": {
                key: {
                    "display_name": value.display_name,
                    "implemented": value.implemented,
                    "data_physics": list(value.data_physics),
                    "nominal_domain": value.nominal_domain,
                }
                for key, value in self.experts.items()
            },
            "groups": list(self.groups),
            "minerals": list(self.minerals),
        }


def _window(value: Sequence[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"Spectral window must contain two values: {value}")
    lower, upper = float(value[0]), float(value[1])
    if lower >= upper:
        raise ValueError(f"Spectral window must be increasing: {value}")
    return lower, upper


def _windows(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    result = tuple(_window(value) for value in values)
    if not result:
        raise ValueError("At least one spectral window is required")
    return result


def default_catalog_path() -> Path:
    return Path(__file__).parent / "resources" / "mineral_catalog_v4.json"
