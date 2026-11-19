"""Identify supported input layouts without fitting or inferring experiment metadata."""

import csv
import gzip
from pathlib import Path
import re
import struct
import zipfile
import zlib

from .utils.compound_file import SIGNATURE, read_compound_file
from .utils.text_io import DELIMITERS

__all__ = ["detect_file_format"]

_HEADER_BYTES = 65536
_AXIS = re.compile(r"^(.+?)_([XY])$", re.IGNORECASE)


def _table_format(fields):
    fields = [field.strip() for field in fields]
    while fields and not fields[-1]:
        fields.pop()
    if len(fields) < 2:
        return None
    if fields[0] == "Label":
        expected = ["Label"] + [
            f"{axis}{i}"
            for i in range(1, (len(fields) + 1) // 2)
            for axis in ("X", "Y")
        ]
        return "carterra_csv" if len(fields) >= 3 and fields == expected else None
    matches = [_AXIS.fullmatch(field) for field in fields]
    if all(matches):
        pairs = {}
        for match in matches:
            name, axis = match[1].strip(), match[2].upper()
            if not name or axis in pairs.setdefault(name, set()):
                return None
            pairs[name].add(axis)
        return (
            "spr_text" if all(axes == {"X", "Y"} for axes in pairs.values()) else None
        )
    names = {re.sub(r"\s+", " ", field).casefold() for field in fields}
    if {"time", "signal", "smax", "analyte_concentration_micromolar_constant"} <= names:
        return "kingenie_surface"
    if {"time", "signal"} <= names and names & {
        "protein_concentration_micromolar",
        "ligand_concentration_micromolar",
    }:
        return "kingenie_solution"
    if {"finalvalue", "time", "smax", "tmax", "tmin"} <= names:
        return "gator_csv"
    if names & {"step time", "time", "duration", "step duration"} and names & {
        "step type name",
        "step type",
        "type",
    }:
        return "surface_csv_steps"
    # A generic Time/Signal table is ambiguous between surface and solution.
    if re.sub(r"\s+", " ", fields[0]).casefold() == "x values":
        return "surface_csv_traces"
    return None


def detect_file_format(file):
    """Return a supported format identifier, or ``'unknown'`` when ambiguous.

    Identify content rather than the filename extension. Text headers are searched
    within the first 64 KiB of uncompressed data, including metadata preambles,
    UTF-8 BOMs, Windows-1252 text, and comma/tab/semicolon delimiters. Gzip text
    exports are recognized by their compression signature. Numeric data and
    injection boundaries are not validated or inferred.

    Returns one of ``octet_frd``, ``biacore_blr``, ``gator_csv``, ``gator_steps``,
    ``gator_settings``, ``gator_zip``, ``kingenie_surface``, ``kingenie_solution``,
    ``surface_csv_steps``, ``surface_csv_traces``, ``spr_text``, ``carterra_csv``,
    or ``unknown``. All paired-column SPR header variants use ``spr_text``;
    this identifies their common reader, not the instrument manufacturer.

    Missing or inaccessible files raise the underlying filesystem error. Generic
    Time/Signal data and unsupported native XLSX exports return ``unknown``.
    Detection does not imply that all required companion files or fit metadata
    are present, or that the full file will pass reader validation.
    """
    file = Path(file)
    with file.open("rb") as stream:
        prefix = stream.read(_HEADER_BYTES)
    if prefix.startswith(SIGNATURE):
        try:
            streams = read_compound_file(file)
        except (ValueError, struct.error, IndexError):
            return "unknown"
        if any(re.fullmatch(r"_Cycle \d+/EventLog", name) for name in streams):
            return "biacore_blr"
        return "unknown"
    if prefix.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(file) as archive:
                names = {Path(name).name.casefold() for name in archive.namelist()}
        except zipfile.BadZipFile:
            return "unknown"
        if {"setting.ini", "experimentstep.ini"} <= names and any(
            re.fullmatch(r"assay_\d+_channel\d+\.csv", name) for name in names
        ):
            return "gator_zip"
        return "unknown"
    if prefix.startswith(b"\x1f\x8b"):
        try:
            with gzip.open(file, "rb") as stream:
                prefix = stream.read(_HEADER_BYTES)
        except (gzip.BadGzipFile, EOFError, zlib.error):
            return "unknown"
    try:
        text = prefix.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = prefix.decode("cp1252", errors="replace")
    if "<ExperimentResults>" in text and "<ExperimentInfo " in text:
        return "octet_frd"
    if "[ExperimentLog]" in text and '"StepType"' in text:
        return "gator_steps"
    if "[BasicInformation]" in text and "PreExperiment=" in text:
        return "gator_settings"
    for line in text.splitlines():
        for delimiter in DELIMITERS:
            fields = next(
                csv.reader([line], delimiter=delimiter, skipinitialspace=True)
            )
            result = _table_format(fields)
            if result:
                return result
    return "unknown"
