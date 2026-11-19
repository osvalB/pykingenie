"""Read TitrationAnalysis "Data Info Sheet" tables that describe SPR runs.

The sheet is the input table of the TitrationAnalysis tool (Duke University).
It has one row per sensor position and is used here to fill the settings that
sensorgram exports (``SprTextExperiment``, ``CarterraExperiment``) lack.

What the file provides
----------------------
- Timing of each cycle in seconds: ``Bsl Start`` (time of the first baseline
  point), ``Baseline``, ``Association`` and ``Dissociation`` durations, and the
  ``Asso. Skip`` / ``Disso. Skip`` times left out of the fit.
- ``All Concentrations``: the molar analyte concentrations, lowest first, and
  ``Incl. Conc.``: the ones used in the analysis (``All`` or a subset).
- The analyte and ligand names, the sensor position, and Y/N flags for bulk
  shift, regeneration, baseline correction and global Rmax.

What the readers take from it
-----------------------------
- ``association_start`` = ``Bsl Start`` + ``Baseline``;
  ``association_end`` = ``association_start`` + ``Association``.
- ``concentrations`` (converted to µM) and ``sample_id`` = ``Analyte``.
- For Carterra single-cycle imports, ``cycle_duration`` =
  ``Baseline`` + ``Association`` + ``Dissociation``.
- Curves whose concentration is not in ``Incl. Conc.`` start unselected
  (``Select`` column) for fitting. Everything else is kept in
  ``exp_info["info_sheet"]``, for example the skip times to pass as
  ``trim_time``.

What the user must still provide
--------------------------------
- Which row applies, when the included rows of a sheet differ in timing,
  analyte or concentrations.
- ``concentrations`` when the data file holds a different number of curves
  than the sheet lists concentrations (for example an excerpt of the run).
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .utils.text_io import read_delimited_table

__all__ = ["read_info_sheet"]

# Normalised column -> accepted headers (case and spacing ignored).
_COLUMNS = {
    "Included": ("incl.",),
    "Block": ("block/chip/tray",),
    "Position": ("position/channel/sensor",),
    "Analyte": ("analyte",),
    "Ligand": ("ligand",),
    "Baseline_start_s": ("bsl start",),
    "Baseline_s": ("baseline",),
    "Association_s": ("association",),
    "Dissociation_s": ("dissociation",),
    "Association_skip_s": ("asso. skip",),
    "Dissociation_skip_s": ("disso. skip",),
    "Concentrations_micromolar": ("all concentrations",),
    "Included_concentrations_micromolar": ("incl. conc.",),
    "Bulk_shift": ("bulkshift",),
    "Regeneration": ("regen.",),
    # The T200 example sheet spells this header "Bass Corr".
    "Baseline_correction": ("base corr", "bass corr"),
    "Global_Rmax": ("global rmax",),
}
_REQUIRED = ("Analyte", "Baseline_s", "Association_s", "Dissociation_s",
             "Concentrations_micromolar")
_DEFAULTS = {"Included": "Y", "Baseline_start_s": "0", "Association_skip_s": "0",
             "Dissociation_skip_s": "0", "Included_concentrations_micromolar": "All"}
_SECONDS = ("Baseline_start_s", "Baseline_s", "Association_s", "Dissociation_s",
            "Association_skip_s", "Dissociation_skip_s")
_FLAGS = ("Included", "Bulk_shift", "Regeneration", "Baseline_correction", "Global_Rmax")
# Settings that must agree before a whole sheet can stand for one experiment.
_SHARED = ("Analyte", "Baseline_start_s", "Baseline_s", "Association_s",
           "Dissociation_s", "Concentrations_micromolar",
           "Included_concentrations_micromolar")


def _text(value):
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value).strip()


def _seconds(value, column, row):
    try:
        result = float(_text(value))
    except ValueError as error:
        raise ValueError(f"Info sheet row {row}: {column} must be a number of seconds, "
                         f"got {value!r}.") from error
    if not np.isfinite(result):
        raise ValueError(f"Info sheet row {row}: {column} must be finite.")
    return result


def _molar_list(value, column, row):
    text = _text(value)
    if text.casefold() in ("", "all"):
        return None
    try:
        values = [float(item) for item in re.split(r"[,;\s]+", text) if item]
    except ValueError as error:
        raise ValueError(f"Info sheet row {row}: {column} must list molar "
                         f"concentrations, got {text!r}.") from error
    values = np.asarray(values) * 1e6
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Info sheet row {row}: {column} must be finite and non-negative.")
    return values.tolist()


def _flag(value):
    text = _text(value).casefold()
    return None if not text else text in ("y", "yes", "true", "1")


def read_info_sheet(file):
    """Read a TitrationAnalysis Data Info Sheet (``.xlsx``, CSV or TSV).

    Returns one row per sensor position with normalised columns: times in
    seconds (``Baseline_s``, ``Association_s`` ...), concentrations as lists in
    µM (``Included_concentrations_micromolar`` is ``None`` for ``All``) and Y/N
    flags as booleans. Pass the path, or one row, to the ``info`` argument of
    ``SprTextExperiment.read_file`` or ``CarterraExperiment.read_csv``.
    """
    file = Path(file)
    if file.suffix.casefold() in (".xlsx", ".xlsm"):
        raw = pd.read_excel(file, dtype=str)
    else:
        raw = read_delimited_table(file).astype(object)
    raw = raw.dropna(how="all")

    headers = {re.sub(r"\s+", " ", str(column)).strip().casefold(): column
               for column in raw.columns}
    found = {}
    for name, accepted in _COLUMNS.items():
        for header in accepted:
            if header in headers:
                found[name] = headers[header]
                break
    missing = [name for name in _REQUIRED if name not in found]
    if missing:
        raise ValueError("Not a Data Info Sheet: missing columns "
                         + ", ".join(_COLUMNS[name][0].title() for name in missing) + ".")
    if raw.empty:
        raise ValueError("The info sheet has no rows.")

    rows = []
    for number, (_, source) in enumerate(raw.iterrows(), start=2):
        row = {}
        for name in _COLUMNS:
            value = source[found[name]] if name in found else _DEFAULTS.get(name)
            if name in _SECONDS:
                value = _seconds(value, name, number)
            elif name in ("Concentrations_micromolar", "Included_concentrations_micromolar"):
                value = _molar_list(value, name, number)
            elif name in _FLAGS:
                value = _flag(value)
            else:
                value = _text(value)
            row[name] = value
        if row["Concentrations_micromolar"] is None:
            raise ValueError(f"Info sheet row {number}: All Concentrations is empty.")
        if not row["Analyte"]:
            raise ValueError(f"Info sheet row {number}: Analyte is empty.")
        if min(row["Association_s"], row["Dissociation_s"]) <= 0:
            raise ValueError(f"Info sheet row {number}: Association and Dissociation "
                             "must be positive.")
        rows.append(row)
    return pd.DataFrame(rows, columns=list(_COLUMNS))


def info_settings(info):
    """Translate an info sheet (path) or one of its rows into reader settings."""
    if isinstance(info, (str, Path)):
        sheet = read_info_sheet(info)
        rows = sheet[sheet["Included"].fillna(True).astype(bool)]
        rows = rows if len(rows) else sheet
        if len(rows[list(_SHARED)].astype(str).drop_duplicates()) > 1:
            raise ValueError("The info sheet's included rows differ in timing, analyte or "
                             "concentrations. Pass one row instead, for example "
                             "info=read_info_sheet(path).iloc[0].")
        row = rows.iloc[0]
    elif isinstance(info, (pd.Series, dict)):
        row = info
    else:
        raise TypeError("info must be an info-sheet path or one row of read_info_sheet().")
    missing = [name for name in _SHARED if name not in row]
    if missing:
        raise ValueError("info is missing " + ", ".join(missing)
                         + "; use a row of read_info_sheet().")

    start = row["Baseline_start_s"] + row["Baseline_s"]
    included = row["Included_concentrations_micromolar"]
    return {
        "association_start": float(start),
        "association_end": float(start + row["Association_s"]),
        "cycle_duration": float(row["Baseline_s"] + row["Association_s"] + row["Dissociation_s"]),
        "concentrations": list(row["Concentrations_micromolar"]),
        "sample_id": str(row["Analyte"]),
        "included_concentrations": None if included is None else list(included),
        "record": {name: row[name] for name in _COLUMNS if name in row},
    }
