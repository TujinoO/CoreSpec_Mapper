from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import re
import unicodedata


class SourceRole(str, Enum):
    MINERAL_REFERENCE = "mineral_reference"
    ROCK = "rock"
    SOIL = "soil"
    VEGETATION = "vegetation"
    WATER = "water"
    SNOW_ICE = "snow_ice"
    METEORITE = "meteorite"
    LUNAR = "lunar"
    MANMADE = "manmade"
    UNKNOWN = "unknown"


class PurityStatus(str, Enum):
    DECLARED_PURE = "declared_pure"
    MIXTURE = "mixture"
    AMBIGUOUS = "ambiguous"
    NOT_MINERAL_REFERENCE = "not_mineral_reference"


@dataclass(frozen=True)
class SourceDescriptor:
    source_family: str
    role: SourceRole
    title: str
    data_physics: str
    measurement_geometry: str
    license_status: str = "unverified"


@dataclass(frozen=True)
class ParsedSpectrumName:
    raw_name: str
    normalized_name: str
    display_name: str
    primary_phase_id: str | None
    primary_phase_raw: str | None
    constituent_phase_ids: tuple[str, ...]
    material_kind: str
    purity_status: PurityStatus
    sample_code: str | None
    parse_confidence: str
    parse_reasons: tuple[str, ...]
    catalogued: bool


def default_v5_catalog_path() -> Path:
    return Path(__file__).parent / "resources" / "mineral_catalog_v5.json"


def load_v5_catalog(path: str | Path | None = None) -> dict[str, Any]:
    source = default_v5_catalog_path() if path is None else Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def _normalise(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return " ".join(text.strip().strip('"').split())


def _slug(value: str) -> str | None:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or None


def describe_source(relative_path: str | Path) -> SourceDescriptor:
    relative = str(relative_path).replace("/", "\\").casefold()
    if relative.startswith("igcp264\\"):
        return SourceDescriptor("igcp264", SourceRole.MINERAL_REFERENCE, "IGCP-264 mineral standards", "reflectance", "mixed_instrument_reflectance")
    if relative.startswith("jpl_lib\\"):
        return SourceDescriptor("jpl", SourceRole.MINERAL_REFERENCE, "JPL mineral spectral library", "reflectance", "hemispherical_reflectance")
    if relative == "jhu_lib\\minerals.sli":
        return SourceDescriptor("jhu_minerals", SourceRole.MINERAL_REFERENCE, "JHU mineral spectral library", "reflectance", "biconical_reflectance")
    if relative.startswith("usgs_min\\"):
        return SourceDescriptor("usgs_v1", SourceRole.MINERAL_REFERENCE, "USGS digital mineral spectral library v1", "reflectance", "laboratory_reflectance")
    if relative.startswith("veg_lib\\") or relative == "jhu_lib\\veg.sli":
        return SourceDescriptor("vegetation", SourceRole.VEGETATION, "Vegetation spectral library", "reflectance", "mixed")
    if relative in {"jhu_lib\\ign_crs.sli", "jhu_lib\\ign_fn.sli"}:
        return SourceDescriptor("jhu_igneous", SourceRole.ROCK, "JHU igneous rocks", "reflectance", "directional_hemispherical_reflectance")
    if relative in {"jhu_lib\\meta_crs.sli", "jhu_lib\\meta_fn.sli"}:
        return SourceDescriptor("jhu_metamorphic", SourceRole.ROCK, "JHU metamorphic rocks", "reflectance", "directional_hemispherical_reflectance")
    if relative in {"jhu_lib\\sed_crs.sli", "jhu_lib\\sed_fn.sli"}:
        return SourceDescriptor("jhu_sedimentary", SourceRole.ROCK, "JHU sedimentary rocks", "reflectance", "directional_hemispherical_reflectance")
    if relative == "jhu_lib\\soils.sli":
        return SourceDescriptor("jhu_soils", SourceRole.SOIL, "JHU soils", "reflectance", "directional_hemispherical_reflectance")
    if relative == "jhu_lib\\water.sli":
        return SourceDescriptor("jhu_water", SourceRole.WATER, "JHU water", "reflectance", "directional_hemispherical_reflectance")
    if relative == "jhu_lib\\snow.sli":
        return SourceDescriptor("jhu_snow", SourceRole.SNOW_ICE, "JHU snow and ice", "reflectance", "mixed")
    if relative == "jhu_lib\\meteor.sli":
        return SourceDescriptor("jhu_meteorite", SourceRole.METEORITE, "JHU meteorites", "reflectance", "biconical_reflectance")
    if relative == "jhu_lib\\lunar.sli":
        return SourceDescriptor("jhu_lunar", SourceRole.LUNAR, "JHU lunar materials", "reflectance", "biconical_reflectance")
    if relative in {"jhu_lib\\manmade1.sli", "jhu_lib\\manmade2.sli"}:
        return SourceDescriptor("jhu_manmade", SourceRole.MANMADE, "JHU manmade materials", "reflectance", "mixed")
    return SourceDescriptor("unknown", SourceRole.UNKNOWN, "Unknown spectral source", "unknown", "unknown")


class MineralAliasIndex:
    def __init__(self, aliases: Mapping[str, str]):
        normalised: dict[str, str] = {}
        for alias, mineral_id in aliases.items():
            key = _normalise(alias).casefold().replace("_", " ")
            normalised[key] = str(mineral_id).casefold()
        self._aliases = tuple(sorted(normalised.items(), key=lambda item: (-len(item[0]), item[0])))
        self._mineral_ids = frozenset(normalised.values())

    @classmethod
    def from_catalog(cls, catalog: Mapping[str, Any]) -> "MineralAliasIndex":
        aliases: dict[str, str] = {}
        for mineral_id, record in catalog.get("minerals", {}).items():
            aliases[str(mineral_id)] = str(mineral_id)
            aliases[str(record.get("display_name_en", mineral_id))] = str(mineral_id)
            for alias in record.get("primary_aliases", ()):
                aliases[str(alias)] = str(mineral_id)
        return cls(aliases)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MineralAliasIndex":
        return cls.from_catalog(load_v5_catalog(path))

    @property
    def mineral_ids(self) -> frozenset[str]:
        return self._mineral_ids

    def match_prefix(self, value: str) -> tuple[str | None, str | None]:
        candidate = _normalise(value).casefold().replace("_", " ")
        for alias, mineral_id in self._aliases:
            if not candidate.startswith(alias):
                continue
            if len(candidate) == len(alias) or candidate[len(alias)] in " -/+(\t:[":
                return mineral_id, alias
        return None, None

    def phases_in(self, value: str) -> tuple[str, ...]:
        text = _normalise(value).casefold().replace("_", " ")
        found: list[str] = []
        for alias, mineral_id in self._aliases:
            if mineral_id in found:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text):
                found.append(mineral_id)
        return tuple(found)


_MIXTURE_WORDS = re.compile(r"\b(?:bearing|mixture|mixed|with)\b", re.IGNORECASE)
_PERCENT_MIXTURE = re.compile(r"\d+(?:\.\d+)?\s*%\s*\+", re.IGNORECASE)
_KNOWN_SPELLING_ALIASES = {
    "notronite": "nontronite",
    "flourite": "fluorite",
}
_IGCP_VEGETATION_LABELS = {"drygrass", "greenveg"}


def _source_display_and_sample(source: SourceDescriptor, raw_name: str) -> tuple[str, str | None]:
    display = _normalise(raw_name)
    sample_code: str | None = None
    if source.source_family == "usgs_v1":
        match = re.match(r"^\S+\.spc\s+(.+)$", display, re.IGNORECASE)
        if match:
            display = match.group(1)
    elif source.source_family == "igcp264":
        if "_" in display:
            label, sample_code = display.rsplit("_", 1)
            display = f"{label} {sample_code}"
    elif source.source_family == "jhu_minerals":
        match = re.search(r"([A-Za-z][A-Za-z0-9_-]*\.\d+)\)+$", display)
        if match:
            sample_code = match.group(1)
    return display, sample_code


def _leading_phase_token(display: str) -> str | None:
    if not display or display.startswith("("):
        return None
    token = display.split(maxsplit=1)[0].strip(" ,;:()[]")
    token = _KNOWN_SPELLING_ALIASES.get(token.casefold(), token)
    return token or None


def _phase_segment(source: SourceDescriptor, display: str, matched_alias: str | None) -> str:
    if source.source_family == "jhu_minerals":
        return display.split(maxsplit=1)[0] if display else ""
    if matched_alias is None:
        return display.split(maxsplit=1)[0] if display else ""
    # USGS and JPL place the phase at the front and sample metadata later. The
    # short prefix is enough for slash/hyphen mixtures; percentages may follow.
    words = display.split()
    return " ".join(words[: min(4, len(words))])


def parse_spectrum_name(
    relative_path: str | Path,
    raw_name: str,
    alias_index: MineralAliasIndex,
) -> ParsedSpectrumName:
    source = describe_source(relative_path)
    display, source_sample = _source_display_and_sample(source, raw_name)
    normalised = _normalise(display)
    mineral_id, matched_alias = alias_index.match_prefix(normalised)
    leading = _leading_phase_token(normalised)
    phase_segment = _phase_segment(source, normalised, matched_alias)
    phases = alias_index.phases_in(phase_segment)

    mixture = False
    reasons: list[str] = []
    if _MIXTURE_WORDS.search(phase_segment) or _PERCENT_MIXTURE.search(phase_segment):
        mixture = True
    first_token = phase_segment.split(maxsplit=1)[0].casefold() if phase_segment else ""
    if "+" in first_token or "/" in first_token:
        mixture = True
    if matched_alias and first_token.startswith(matched_alias + "-"):
        mixture = True
    if "-" in first_token and len(phases) > 1:
        mixture = True
    if len(phases) > 1:
        mixture = True
    if mixture:
        reasons.append("multiple_or_mixed_phase_name")

    if mineral_id is None and leading is not None:
        primary_phase = _slug(leading)
        primary_raw = leading
        catalogued = False
        confidence = "provisional"
        reasons.append("phase_not_in_v5_catalog")
    elif mineral_id is None:
        primary_phase = None
        primary_raw = None
        catalogued = False
        confidence = "unresolved"
        reasons.append("missing_or_unparseable_primary_phase")
    else:
        primary_phase = mineral_id
        primary_raw = matched_alias
        catalogued = True
        confidence = "high"

    record_role = source.role
    if source.source_family == "igcp264" and (primary_phase or "") in _IGCP_VEGETATION_LABELS:
        record_role = SourceRole.VEGETATION
        reasons.append("record_role_vegetation")
    if record_role != SourceRole.MINERAL_REFERENCE:
        purity = PurityStatus.NOT_MINERAL_REFERENCE
        material_kind = record_role.value
        reasons.append(f"source_role_{record_role.value}")
    elif mixture:
        purity = PurityStatus.MIXTURE
        material_kind = "mineral_mixture"
    elif primary_phase is None:
        purity = PurityStatus.AMBIGUOUS
        material_kind = "mineral_unknown"
    else:
        purity = PurityStatus.DECLARED_PURE
        material_kind = "mineral"

    sample_code = source_sample
    if sample_code is None and primary_raw:
        remainder = normalised[len(primary_raw) :].strip(" _-:()")
        if remainder:
            sample_code = remainder

    constituent = phases
    if primary_phase and primary_phase not in constituent:
        constituent = (primary_phase, *constituent)
    return ParsedSpectrumName(
        raw_name=str(raw_name),
        normalized_name=normalised.casefold(),
        display_name=normalised,
        primary_phase_id=primary_phase,
        primary_phase_raw=primary_raw,
        constituent_phase_ids=tuple(dict.fromkeys(constituent)),
        material_kind=material_kind,
        purity_status=purity,
        sample_code=sample_code,
        parse_confidence=confidence,
        parse_reasons=tuple(dict.fromkeys(reasons)),
        catalogued=catalogued,
    )


def parse_many(
    relative_path: str | Path,
    names: Iterable[str],
    alias_index: MineralAliasIndex,
) -> tuple[ParsedSpectrumName, ...]:
    return tuple(parse_spectrum_name(relative_path, name, alias_index) for name in names)
