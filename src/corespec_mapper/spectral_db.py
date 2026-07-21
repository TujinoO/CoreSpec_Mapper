from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence
import json
import os
import sqlite3
import tempfile
import zlib

import numpy as np

from .envi import SpectralLibrary, parse_envi_header
from .spectral_v5 import (
    MineralAliasIndex,
    PurityStatus,
    SourceRole,
    describe_source,
    load_v5_catalog,
    parse_spectrum_name,
)


DATABASE_SCHEMA_VERSION = 1
DATABASE_RELEASE = "5.1.0"


def default_v5_database_path() -> Path:
    return Path(__file__).parent / "resources" / "corespec_spectral_v5.sqlite3"


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_blob(values: np.ndarray, dtype: str) -> tuple[bytes, bytes]:
    data = np.ascontiguousarray(values, dtype=np.dtype(dtype)).tobytes()
    return zlib.compress(data, level=9), data


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sample_key(source_family: str, source_id: str, source_index: int, phase: str | None, sample_code: str | None) -> str:
    if sample_code:
        compact = "_".join(sample_code.casefold().split())[:240]
        return f"{source_family}:{phase or 'unknown'}:{compact}"
    return f"{source_id}#{source_index}"


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE taxonomy_group (
            group_id TEXT PRIMARY KEY,
            display_name_en TEXT NOT NULL,
            display_name_zh TEXT NOT NULL
        );
        CREATE TABLE spectral_family (
            family_id TEXT PRIMARY KEY,
            display_name_en TEXT NOT NULL,
            display_name_zh TEXT NOT NULL,
            expert_id TEXT NOT NULL,
            definition_json TEXT NOT NULL
        );
        CREATE TABLE mineral (
            mineral_id TEXT PRIMARY KEY,
            display_name_en TEXT NOT NULL,
            display_name_zh TEXT NOT NULL,
            formula TEXT NOT NULL,
            taxonomy_group_id TEXT NOT NULL,
            spectral_family_id TEXT NOT NULL,
            alteration_roles_json TEXT NOT NULL,
            support_level TEXT NOT NULL,
            recognition_enabled INTEGER NOT NULL,
            definition_json TEXT NOT NULL,
            FOREIGN KEY(taxonomy_group_id) REFERENCES taxonomy_group(group_id),
            FOREIGN KEY(spectral_family_id) REFERENCES spectral_family(family_id)
        );
        CREATE TABLE source (
            source_id TEXT PRIMARY KEY,
            source_family TEXT NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            source_role TEXT NOT NULL,
            data_physics TEXT NOT NULL,
            measurement_geometry TEXT NOT NULL,
            license_status TEXT NOT NULL,
            declared_wavelength_units TEXT NOT NULL,
            band_count INTEGER NOT NULL,
            spectrum_count INTEGER NOT NULL,
            wavelength_min_nm REAL NOT NULL,
            wavelength_max_nm REAL NOT NULL,
            wavelength_sha256 TEXT NOT NULL,
            wavelength_zlib BLOB NOT NULL,
            data_file_sha256 TEXT NOT NULL,
            header_file_sha256 TEXT NOT NULL,
            warning TEXT
        );
        CREATE TABLE sample (
            sample_id TEXT PRIMARY KEY,
            source_family TEXT NOT NULL,
            primary_phase_id TEXT,
            primary_phase_raw TEXT,
            constituent_phase_ids_json TEXT NOT NULL,
            material_kind TEXT NOT NULL,
            purity_status TEXT NOT NULL,
            sample_code TEXT,
            catalogued INTEGER NOT NULL,
            parse_confidence TEXT NOT NULL,
            parse_reasons_json TEXT NOT NULL
        );
        CREATE TABLE measurement (
            measurement_id INTEGER PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_index INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            raw_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            primary_phase_id TEXT,
            purity_status TEXT NOT NULL,
            catalogued INTEGER NOT NULL,
            point_count INTEGER NOT NULL,
            finite_count INTEGER NOT NULL,
            positive_count INTEGER NOT NULL,
            zero_count INTEGER NOT NULL,
            negative_count INTEGER NOT NULL,
            value_min REAL,
            value_max REAL,
            values_sha256 TEXT NOT NULL,
            spectrum_sha256 TEXT NOT NULL,
            values_zlib BLOB NOT NULL,
            FOREIGN KEY(source_id) REFERENCES source(source_id),
            FOREIGN KEY(sample_id) REFERENCES sample(sample_id),
            UNIQUE(source_id, source_index)
        );
        CREATE TABLE qc_result (
            measurement_id INTEGER PRIMARY KEY,
            qc_status TEXT NOT NULL,
            numeric_status TEXT NOT NULL,
            reasons_json TEXT NOT NULL,
            label_conflict INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(measurement_id) REFERENCES measurement(measurement_id)
        );
        CREATE INDEX measurement_phase_idx ON measurement(primary_phase_id, purity_status, catalogued);
        CREATE INDEX measurement_source_idx ON measurement(source_id);
        CREATE INDEX measurement_spectrum_hash_idx ON measurement(spectrum_sha256);
        CREATE INDEX qc_status_idx ON qc_result(qc_status);
        """
    )


def _insert_catalog(connection: sqlite3.Connection, catalog: dict[str, Any]) -> None:
    for group_id, record in catalog["taxonomy_groups"].items():
        connection.execute(
            "INSERT INTO taxonomy_group VALUES (?, ?, ?)",
            (group_id, record["display_name_en"], record["display_name_zh"]),
        )
    for family_id, record in catalog["spectral_families"].items():
        connection.execute(
            "INSERT INTO spectral_family VALUES (?, ?, ?, ?, ?)",
            (family_id, record["display_name_en"], record["display_name_zh"], record["expert"], _json(record)),
        )
    for mineral_id, record in catalog["minerals"].items():
        connection.execute(
            "INSERT INTO mineral VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mineral_id,
                record["display_name_en"],
                record["display_name_zh"],
                record.get("formula", ""),
                record["taxonomy_group"],
                record["spectral_family"],
                _json(record.get("alteration_roles", [])),
                record["support_level"],
                int(bool(record.get("recognition_enabled", False))),
                _json(record),
            ),
        )


def _numeric_qc(values: np.ndarray) -> tuple[str, list[str], dict[str, int | float | None]]:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    finite_values = array[finite]
    positive = finite & (array > 0) & (np.abs(array) < 1e6)
    zero = finite & (array == 0)
    negative = finite & (array < 0)
    reasons: list[str] = []
    if finite_values.size != array.size:
        reasons.append("non_finite_values")
    if np.count_nonzero(positive) < 2:
        reasons.append("fewer_than_two_positive_values")
    if finite_values.size >= 2 and float(np.ptp(finite_values)) <= 1e-8:
        reasons.append("constant_curve")
    if np.any(zero):
        reasons.append("contains_zero_values")
    if np.any(negative):
        reasons.append("contains_negative_values")
    hard = {"non_finite_values", "fewer_than_two_positive_values", "constant_curve"}
    status = "invalid" if hard.intersection(reasons) else "review" if reasons else "pass"
    metrics: dict[str, int | float | None] = {
        "finite_count": int(np.count_nonzero(finite)),
        "positive_count": int(np.count_nonzero(positive)),
        "zero_count": int(np.count_nonzero(zero)),
        "negative_count": int(np.count_nonzero(negative)),
        "value_min": None if not finite_values.size else float(np.min(finite_values)),
        "value_max": None if not finite_values.size else float(np.max(finite_values)),
    }
    return status, reasons, metrics


def _mark_label_conflicts(connection: sqlite3.Connection) -> int:
    conflicts = connection.execute(
        """
        SELECT spectrum_sha256
        FROM measurement
        GROUP BY spectrum_sha256
        HAVING COUNT(DISTINCT COALESCE(primary_phase_id, '')) > 1
        """
    ).fetchall()
    count = 0
    for (spectrum_hash,) in conflicts:
        rows = connection.execute(
            "SELECT measurement_id FROM measurement WHERE spectrum_sha256 = ?", (spectrum_hash,)
        ).fetchall()
        for (measurement_id,) in rows:
            current = connection.execute(
                "SELECT reasons_json FROM qc_result WHERE measurement_id = ?", (measurement_id,)
            ).fetchone()
            reasons = json.loads(current[0]) if current else []
            if "identical_curve_with_conflicting_phase_label" not in reasons:
                reasons.append("identical_curve_with_conflicting_phase_label")
            connection.execute(
                "UPDATE qc_result SET qc_status='excluded', label_conflict=1, reasons_json=? WHERE measurement_id=?",
                (_json(reasons), measurement_id),
            )
            count += 1
    return count


def build_v5_database(
    library_root: str | Path,
    output_path: str | Path,
    *,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(library_root).resolve()
    output = Path(output_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    catalog = load_v5_catalog(catalog_path)
    alias_index = MineralAliasIndex.from_catalog(catalog)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(prefix=output.stem + "_", suffix=".sqlite3", dir=output.parent, delete=False)
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    connection = sqlite3.connect(temporary)
    try:
        _create_schema(connection)
        _insert_catalog(connection, catalog)
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            (
                ("database_release", DATABASE_RELEASE),
                ("schema_version", str(DATABASE_SCHEMA_VERSION)),
                ("catalog_version", str(catalog["version"])),
            ),
        )
        measurement_id = 0
        role_counts: dict[str, int] = {}
        for path in sorted(root.rglob("*.sli"), key=lambda item: str(item).casefold()):
            relative = path.relative_to(root).as_posix()
            source_id = relative
            descriptor = describe_source(relative)
            role_counts[descriptor.role.value] = role_counts.get(descriptor.role.value, 0) + 1
            library = SpectralLibrary.open(path)
            header = parse_envi_header(library.header_path)
            wavelength_blob, wavelength_bytes = _array_blob(library.wavelengths_nm, "<f8")
            connection.execute(
                """
                INSERT INTO source VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    descriptor.source_family,
                    relative,
                    descriptor.title,
                    descriptor.role.value,
                    descriptor.data_physics,
                    descriptor.measurement_geometry,
                    descriptor.license_status,
                    str(header.get("wavelength units", "unknown")),
                    int(library.wavelengths_nm.size),
                    int(len(library.names)),
                    float(np.min(library.wavelengths_nm)),
                    float(np.max(library.wavelengths_nm)),
                    sha256(wavelength_bytes).hexdigest(),
                    wavelength_blob,
                    _hash_file(library.data_path),
                    _hash_file(library.header_path),
                    library.warning,
                ),
            )
            for source_index, (raw_name, values) in enumerate(zip(library.names, library.spectra)):
                measurement_id += 1
                parsed = parse_spectrum_name(relative, raw_name, alias_index)
                sample_id = _sample_key(
                    descriptor.source_family,
                    source_id,
                    source_index,
                    parsed.primary_phase_id,
                    parsed.sample_code,
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO sample VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id,
                        descriptor.source_family,
                        parsed.primary_phase_id,
                        parsed.primary_phase_raw,
                        _json(parsed.constituent_phase_ids),
                        parsed.material_kind,
                        parsed.purity_status.value,
                        parsed.sample_code,
                        int(parsed.catalogued),
                        parsed.parse_confidence,
                        _json(parsed.parse_reasons),
                    ),
                )
                values_blob, values_bytes = _array_blob(values, "<f4")
                spectrum_digest = sha256(wavelength_bytes + values_bytes).hexdigest()
                numeric_status, numeric_reasons, metrics = _numeric_qc(values)
                if descriptor.role != SourceRole.MINERAL_REFERENCE or parsed.purity_status in {
                    PurityStatus.MIXTURE,
                    PurityStatus.NOT_MINERAL_REFERENCE,
                }:
                    qc_status = "excluded"
                elif numeric_status == "invalid":
                    qc_status = "excluded"
                elif parsed.catalogued and parsed.purity_status == PurityStatus.DECLARED_PURE:
                    qc_status = "eligible"
                else:
                    qc_status = "review"
                reasons = list(parsed.parse_reasons) + numeric_reasons
                connection.execute(
                    """
                    INSERT INTO measurement VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        measurement_id,
                        source_id,
                        source_index,
                        sample_id,
                        raw_name,
                        parsed.normalized_name,
                        parsed.primary_phase_id,
                        parsed.purity_status.value,
                        int(parsed.catalogued),
                        int(np.asarray(values).size),
                        metrics["finite_count"],
                        metrics["positive_count"],
                        metrics["zero_count"],
                        metrics["negative_count"],
                        metrics["value_min"],
                        metrics["value_max"],
                        sha256(values_bytes).hexdigest(),
                        spectrum_digest,
                        values_blob,
                    ),
                )
                connection.execute(
                    "INSERT INTO qc_result VALUES (?, ?, ?, ?, 0)",
                    (measurement_id, qc_status, numeric_status, _json(reasons)),
                )
        label_conflicts = _mark_label_conflicts(connection)
        summary = {
            "database_release": DATABASE_RELEASE,
            "schema_version": DATABASE_SCHEMA_VERSION,
            "catalog_version": catalog["version"],
            "source_count": int(connection.execute("SELECT COUNT(*) FROM source").fetchone()[0]),
            "sample_count": int(connection.execute("SELECT COUNT(*) FROM sample").fetchone()[0]),
            "measurement_count": int(connection.execute("SELECT COUNT(*) FROM measurement").fetchone()[0]),
            "mineral_count": int(connection.execute("SELECT COUNT(*) FROM mineral").fetchone()[0]),
            "catalogued_pure_measurements": int(connection.execute("SELECT COUNT(*) FROM measurement WHERE catalogued=1 AND purity_status='declared_pure'").fetchone()[0]),
            "eligible_measurements": int(connection.execute("SELECT COUNT(*) FROM qc_result WHERE qc_status='eligible'").fetchone()[0]),
            "label_conflict_measurements": label_conflicts,
            "source_role_library_counts": role_counts,
        }
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("summary_json", _json(summary)))
        connection.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        os.replace(temporary, output)
        return summary
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class StoredSpectrum:
    measurement_id: int
    source_id: str
    source_index: int
    sample_id: str
    raw_name: str
    primary_phase_id: str | None
    purity_status: str
    qc_status: str
    source_role: str
    data_physics: str
    measurement_geometry: str
    wavelengths_nm: np.ndarray
    values: np.ndarray


class SpectralOverlayProvider(Protocol):
    """Reserved extension point for user-managed V5 overlay databases."""

    def database_paths(self) -> Sequence[Path]: ...


@dataclass(frozen=True)
class NoUserOverlayProvider:
    def database_paths(self) -> Sequence[Path]:
        return ()


class V5SpectralDatabase:
    def __init__(self, path: str | Path | None = None, *, overlay_provider: SpectralOverlayProvider | None = None):
        self.path = (default_v5_database_path() if path is None else Path(path)).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.overlay_provider = overlay_provider or NoUserOverlayProvider()
        uri = self.path.as_uri() + "?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True)
        self._connection.row_factory = sqlite3.Row
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version != DATABASE_SCHEMA_VERSION:
            self.close()
            raise ValueError(f"Unsupported V5 spectral database schema {version}")

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def __enter__(self) -> "V5SpectralDatabase":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def summary(self) -> dict[str, Any]:
        row = self._connection.execute("SELECT value FROM metadata WHERE key='summary_json'").fetchone()
        return json.loads(row[0])

    def mineral_definitions(self) -> dict[str, dict[str, Any]]:
        rows = self._connection.execute("SELECT mineral_id, definition_json FROM mineral ORDER BY mineral_id")
        return {row["mineral_id"]: json.loads(row["definition_json"]) for row in rows}

    def candidate_counts(self) -> dict[str, int]:
        rows = self._connection.execute(
            """
            SELECT m.primary_phase_id, COUNT(*) AS count
            FROM measurement m JOIN qc_result q USING(measurement_id)
            WHERE m.catalogued=1 AND m.purity_status='declared_pure' AND q.qc_status='eligible'
            GROUP BY m.primary_phase_id ORDER BY m.primary_phase_id
            """
        )
        return {row["primary_phase_id"]: int(row["count"]) for row in rows}

    def iter_spectra(self, mineral_id: str | None = None, *, eligible_only: bool = True) -> Iterator[StoredSpectrum]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if mineral_id is not None:
            clauses.append("m.primary_phase_id = ?")
            parameters.append(str(mineral_id).casefold())
        if eligible_only:
            clauses.append("q.qc_status = 'eligible'")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = (
            "SELECT m.*, q.qc_status, s.band_count, s.wavelength_zlib, "
            "s.source_role, s.data_physics, s.measurement_geometry "
            "FROM measurement m JOIN qc_result q USING(measurement_id) JOIN source s USING(source_id)"
            + where
            + " ORDER BY m.measurement_id"
        )
        for row in self._connection.execute(query, parameters):
            wavelengths = np.frombuffer(zlib.decompress(row["wavelength_zlib"]), dtype="<f8", count=row["band_count"]).copy()
            values = np.frombuffer(zlib.decompress(row["values_zlib"]), dtype="<f4", count=row["point_count"]).copy()
            yield StoredSpectrum(
                measurement_id=int(row["measurement_id"]),
                source_id=row["source_id"],
                source_index=int(row["source_index"]),
                sample_id=row["sample_id"],
                raw_name=row["raw_name"],
                primary_phase_id=row["primary_phase_id"],
                purity_status=row["purity_status"],
                qc_status=row["qc_status"],
                source_role=row["source_role"],
                data_physics=row["data_physics"],
                measurement_geometry=row["measurement_geometry"],
                wavelengths_nm=wavelengths,
                values=values,
            )

    def iter_compatible_spectra(
        self,
        mineral_id: str,
        required_windows_nm: Sequence[Sequence[float]] | None = None,
        *,
        data_physics: str = "reflectance",
        minimum_coverage: float = 0.97,
    ) -> Iterator[StoredSpectrum]:
        if required_windows_nm is None:
            definitions = self.mineral_definitions()
            if mineral_id not in definitions:
                raise KeyError(f"Unknown V5 mineral: {mineral_id}")
            required_windows_nm = definitions[mineral_id].get("required_windows_nm", ())
        windows = tuple((float(window[0]), float(window[1])) for window in required_windows_nm)
        required_width = sum(upper - lower for lower, upper in windows)
        if required_width <= 0:
            raise ValueError("At least one increasing required wavelength window is required")
        for spectrum in self.iter_spectra(mineral_id):
            if spectrum.data_physics.casefold() != data_physics.casefold():
                continue
            lower_available = float(spectrum.wavelengths_nm[0])
            upper_available = float(spectrum.wavelengths_nm[-1])
            covered = sum(
                max(0.0, min(upper_available, upper) - max(lower_available, lower))
                for lower, upper in windows
            )
            if covered / required_width >= float(minimum_coverage):
                yield spectrum
