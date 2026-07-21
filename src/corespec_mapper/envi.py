from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
import re

import numpy as np


ENVI_DTYPES = {
    1: "u1",
    2: "i2",
    3: "i4",
    4: "f4",
    5: "f8",
    12: "u2",
    13: "u4",
    14: "i8",
    15: "u8",
}


def _find_case_insensitive(parent: Path, name: str) -> Path | None:
    target = name.casefold()
    for candidate in parent.iterdir():
        if candidate.name.casefold() == target:
            return candidate
    return None


def resolve_envi_paths(path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.casefold() == ".hdr":
        header = path
        stem = path.stem
        names = [stem, stem + ".dat", stem + ".img", stem + ".sli"]
        data = next((_find_case_insensitive(path.parent, name) for name in names if _find_case_insensitive(path.parent, name)), None)
        if data is None:
            raise FileNotFoundError(f"No ENVI data file found for header: {path}")
        return data, header

    data = path
    header_names = []
    if path.suffix:
        header_names.append(path.stem + ".hdr")
        header_names.append(path.name + ".hdr")
    else:
        header_names.append(path.name + ".hdr")
    header = next((_find_case_insensitive(path.parent, name) for name in header_names if _find_case_insensitive(path.parent, name)), None)
    if header is None:
        raise FileNotFoundError(f"No ENVI header found for data file: {path}")
    return data, header


def _parse_atom(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _parse_value(value: str) -> Any:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        body = value[1:-1].replace("\r", " ").replace("\n", " ")
        parts = [part.strip() for part in body.split(",")]
        return [_parse_atom(part) for part in parts if part.strip()]
    return _parse_atom(value)


def parse_envi_header(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines or lines[0].strip().casefold() != "envi":
        raise ValueError(f"Not an ENVI header: {path}")

    metadata: dict[str, Any] = {}
    key: str | None = None
    value_parts: list[str] = []
    brace_balance = 0

    def commit() -> None:
        nonlocal key, value_parts, brace_balance
        if key is not None:
            metadata[key.casefold()] = _parse_value("\n".join(value_parts))
        key = None
        value_parts = []
        brace_balance = 0

    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        if key is None:
            if "=" not in line:
                continue
            key, first = line.split("=", 1)
            key = key.strip()
            value_parts = [first.strip()]
            brace_balance = first.count("{") - first.count("}")
            if brace_balance <= 0:
                commit()
        else:
            value_parts.append(line)
            brace_balance += line.count("{") - line.count("}")
            if brace_balance <= 0:
                commit()
    commit()
    return metadata


@dataclass(frozen=True)
class EnviInfo:
    data_path: Path
    header_path: Path
    samples: int
    lines: int
    bands: int
    interleave: str
    data_type: int
    byte_order: int
    header_offset: int
    file_type: str
    wavelengths_nm: np.ndarray | None
    wavelength_warning: str | None
    metadata: dict[str, Any]


def _normalize_wavelengths(metadata: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    values = metadata.get("wavelength")
    if not isinstance(values, list) or not values:
        return None, None
    wavelengths = np.asarray(values, dtype=np.float64)
    units = str(metadata.get("wavelength units", "unknown")).casefold()
    warning = None
    maximum = float(np.nanmax(wavelengths))
    if "micro" in units and maximum <= 100:
        wavelengths = wavelengths * 1000.0
    elif "micro" in units and maximum > 100:
        warning = "Header says micrometers but wavelength values are in the nanometer range; values treated as nm."
    elif maximum <= 20:
        wavelengths = wavelengths * 1000.0
        if "nano" in units:
            warning = "Wavelength values look like micrometers although the header says nanometers; converted to nm."
    return wavelengths, warning


class EnviDataset:
    def __init__(self, path: str | Path):
        data_path, header_path = resolve_envi_paths(path)
        metadata = parse_envi_header(header_path)
        data_type = int(metadata["data type"])
        if data_type not in ENVI_DTYPES:
            raise ValueError(f"Unsupported ENVI data type {data_type}: {header_path}")
        byte_order = int(metadata.get("byte order", 0))
        endian = "<" if byte_order == 0 else ">"
        dtype = np.dtype(endian + ENVI_DTYPES[data_type])
        lines = int(metadata["lines"])
        samples = int(metadata["samples"])
        bands = int(metadata["bands"])
        interleave = str(metadata["interleave"]).strip().casefold()
        header_offset = int(metadata.get("header offset", 0))
        file_type = str(metadata.get("file type", "ENVI Standard"))
        wavelengths, warning = _normalize_wavelengths(metadata)
        if wavelengths is not None and wavelengths.size not in (bands, samples):
            raise ValueError(f"Wavelength count {wavelengths.size} does not match bands/samples in {header_path}")

        if interleave == "bip":
            shape = (lines, samples, bands)
        elif interleave == "bil":
            shape = (lines, bands, samples)
        elif interleave == "bsq":
            shape = (bands, lines, samples)
        else:
            raise ValueError(f"Unsupported interleave {interleave}: {header_path}")

        expected = header_offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        actual = data_path.stat().st_size
        if actual != expected:
            raise ValueError(f"Data length mismatch for {data_path}: expected {expected}, got {actual}")

        self.info = EnviInfo(
            data_path=data_path,
            header_path=header_path,
            samples=samples,
            lines=lines,
            bands=bands,
            interleave=interleave,
            data_type=data_type,
            byte_order=byte_order,
            header_offset=header_offset,
            file_type=file_type,
            wavelengths_nm=wavelengths if wavelengths is None or wavelengths.size == bands else None,
            wavelength_warning=warning,
            metadata=metadata,
        )
        self.dtype = dtype
        self._array = np.memmap(data_path, dtype=dtype, mode="r", offset=header_offset, shape=shape)

    def close(self) -> None:
        array = getattr(self, "_array", None)
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self._array = None

    def __enter__(self) -> "EnviDataset":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def read_rows(self, start: int, stop: int, bands: Sequence[int] | None = None) -> np.ndarray:
        if start < 0 or stop > self.info.lines or start >= stop:
            raise ValueError(f"Invalid row window [{start}, {stop})")
        band_index: slice | Sequence[int] = slice(None) if bands is None else list(bands)
        if self.info.interleave == "bip":
            data = self._array[start:stop, :, band_index]
        elif self.info.interleave == "bil":
            data = np.moveaxis(self._array[start:stop, band_index, :], 1, 2)
        else:
            data = np.moveaxis(self._array[band_index, start:stop, :], 0, 2)
        return np.asarray(data)

    def iter_rows(
        self,
        chunk_rows: int = 128,
        start: int = 0,
        stop: int | None = None,
        bands: Sequence[int] | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        stop = self.info.lines if stop is None else stop
        for row_start in range(start, stop, chunk_rows):
            row_stop = min(row_start + chunk_rows, stop)
            yield row_start, row_stop, self.read_rows(row_start, row_stop, bands=bands)


@dataclass(frozen=True)
class SpectralLibrary:
    spectra: np.ndarray
    names: list[str]
    wavelengths_nm: np.ndarray
    warning: str | None
    data_path: Path
    header_path: Path

    @classmethod
    def open(cls, path: str | Path) -> "SpectralLibrary":
        dataset = EnviDataset(path)
        info = dataset.info
        if "spectral library" not in info.file_type.casefold():
            raise ValueError(f"Not an ENVI spectral library: {info.header_path}")
        if info.interleave != "bsq" or info.bands != 1:
            raise ValueError("Only one-band BSQ ENVI spectral libraries are currently supported")
        spectra = np.asarray(dataset._array[0, :, :], dtype=np.float64)
        wavelengths, warning = _normalize_wavelengths(info.metadata)
        if wavelengths is None or wavelengths.size != info.samples:
            raise ValueError(f"Invalid spectral-library wavelength vector: {info.header_path}")
        raw_names = info.metadata.get("spectra names", [])
        names = [str(name).strip() for name in raw_names]
        if len(names) != info.lines:
            names = [f"Spectrum {idx + 1}" for idx in range(info.lines)]
        result = cls(spectra, names, wavelengths, warning, info.data_path, info.header_path)
        dataset.close()
        return result


def wavelength_indices(wavelengths_nm: np.ndarray, windows: Sequence[Sequence[float]]) -> np.ndarray:
    selected = np.zeros(wavelengths_nm.size, dtype=bool)
    for lower, upper in windows:
        selected |= (wavelengths_nm >= float(lower)) & (wavelengths_nm <= float(upper))
    indices = np.flatnonzero(selected)
    if indices.size < 3:
        raise ValueError(f"Spectral windows selected fewer than three bands: {windows}")
    return indices


def derive_mask(
    masked_cube: EnviDataset,
    bands: Sequence[int] | None = None,
    epsilon: float = 1e-10,
    chunk_rows: int = 128,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    stop = masked_cube.info.lines if stop is None else stop
    result = np.zeros((stop - start, masked_cube.info.samples), dtype=bool)
    for row_start, row_stop, cube in masked_cube.iter_rows(chunk_rows, start, stop, bands=bands):
        valid = np.isfinite(cube) & (np.abs(cube) > epsilon)
        result[row_start - start : row_stop - start] = np.any(valid, axis=-1)
    return result


def _format_list(values: Sequence[Any]) -> str:
    return "{\n  " + ", ".join(str(value) for value in values) + "}"


def _format_header_value(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return _format_list(value)
    if isinstance(value, str):
        text = value.strip()
        return text if text.startswith("{") and text.endswith("}") else "{" + text + "}"
    return str(value)


def subset_spatial_metadata(
    info: EnviInfo,
    *,
    start_line: int = 0,
    start_sample: int = 0,
) -> dict[str, Any]:
    """Return georeferencing metadata adjusted for a spatial subset."""
    metadata = info.metadata
    result = {
        key: metadata[key]
        for key in (
            "coordinate system string",
            "projection info",
            "rpc info",
            "geo points",
        )
        if key in metadata
    }
    map_info = metadata.get("map info")
    if isinstance(map_info, list) and len(map_info) >= 7:
        adjusted = list(map_info)
        try:
            reference_sample = float(adjusted[1])
            reference_line = float(adjusted[2])
            map_x = float(adjusted[3])
            map_y = float(adjusted[4])
            pixel_x = float(adjusted[5])
            pixel_y = float(adjusted[6])
            adjusted[1] = 1.0
            adjusted[2] = 1.0
            adjusted[3] = map_x + (start_sample + 1.0 - reference_sample) * pixel_x
            adjusted[4] = map_y - (start_line + 1.0 - reference_line) * pixel_y
        except (TypeError, ValueError):
            # Retain the source record when a vendor-specific map-info layout
            # cannot be interpreted safely.
            pass
        result["map info"] = adjusted
    if "x start" in metadata:
        try:
            result["x start"] = int(metadata["x start"]) + int(start_sample)
        except (TypeError, ValueError):
            result["x start"] = metadata["x start"]
    if "y start" in metadata:
        try:
            result["y start"] = int(metadata["y start"]) + int(start_line)
        except (TypeError, ValueError):
            result["y start"] = metadata["y start"]
    result["corespec source start line"] = int(start_line)
    result["corespec source start sample"] = int(start_sample)
    return result


def classification_palette(class_names: Sequence[str]) -> list[int]:
    fallback = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (255, 127, 14),
        (148, 103, 189),
        (23, 190, 207),
        (227, 119, 194),
    ]
    colors: list[int] = []
    for index, name in enumerate(class_names):
        lowered = name.casefold()
        if "unclassified" in lowered:
            color = (0, 0, 0)
        elif "masked" in lowered:
            color = (64, 64, 64)
        elif "calcite" in lowered:
            color = (255, 0, 0)
        elif "dolomite" in lowered:
            color = (145, 44, 238)
        elif "anhydrite" in lowered:
            color = (255, 165, 0)
        elif "gypsum" in lowered:
            color = (0, 102, 255)
        elif "illite" in lowered:
            color = (0, 180, 0)
        elif "montmorillonite" in lowered:
            color = (255, 140, 0)
        elif "kaolinite" in lowered:
            color = (255, 230, 0)
        else:
            color = fallback[(index - 1) % len(fallback)]
        colors.extend(color)
    return colors


def write_envi(
    data: np.ndarray,
    data_path: str | Path,
    *,
    band_names: Sequence[str] | None = None,
    description: str | None = None,
    class_names: Sequence[str] | None = None,
    class_lookup: Sequence[int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data)
    if array.ndim == 2:
        lines, samples = array.shape
        bands = 1
        bsq = array[np.newaxis, ...]
    elif array.ndim == 3:
        bands, lines, samples = array.shape
        bsq = array
    else:
        raise ValueError("ENVI output must be a 2D array or a BSQ (bands, lines, samples) array")

    dtype_to_code = {
        np.dtype("uint8"): 1,
        np.dtype("int16"): 2,
        np.dtype("int32"): 3,
        np.dtype("float32"): 4,
        np.dtype("float64"): 5,
        np.dtype("uint16"): 12,
        np.dtype("uint32"): 13,
        np.dtype("int64"): 14,
        np.dtype("uint64"): 15,
    }
    native = np.dtype(array.dtype.name)
    if native not in dtype_to_code:
        raise ValueError(f"Unsupported output dtype: {array.dtype}")
    code = dtype_to_code[native]
    little = bsq.astype(native.newbyteorder("<"), copy=False)
    little.tofile(data_path)

    header_path = data_path.with_suffix(".hdr")
    lines_out = [
        "ENVI",
        f"description = {{{description or 'CoreSpec Mapper output'}}}",
        f"samples = {samples}",
        f"lines = {lines}",
        f"bands = {bands}",
        "header offset = 0",
        f"file type = {'ENVI Classification' if class_names else 'ENVI Standard'}",
        f"data type = {code}",
        "interleave = bsq",
        "byte order = 0",
    ]
    if band_names:
        lines_out.append("band names = " + _format_list(band_names))
    if class_names:
        lines_out.append(f"classes = {len(class_names)}")
        if class_lookup is None:
            class_lookup = classification_palette(class_names)
        if len(class_lookup) != len(class_names) * 3:
            raise ValueError("Class lookup must contain exactly three RGB values per class")
        lines_out.append("class lookup = " + _format_list(class_lookup))
        lines_out.append("class names = " + _format_list(class_names))
        lines_out.append("wavelength units = Unknown")
        if not band_names:
            lines_out.append("band names = " + _format_list(["CoreSpec classification"]))
    reserved = {
        "description", "samples", "lines", "bands", "header offset", "file type",
        "data type", "interleave", "byte order", "band names", "classes",
        "class lookup", "class names",
    }
    for key, value in (metadata or {}).items():
        normalized = str(key).strip().casefold()
        if not normalized or normalized in reserved or value is None:
            continue
        lines_out.append(f"{normalized} = {_format_header_value(value)}")
    header_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return data_path, header_path
