"""Shared parsing for sensorgram text exports (paired-column SPR and Carterra).

These exports contain sensorgrams, rather than a complete instrument run log.
Injection boundaries must be supplied by the caller. Responses and individual
time grids are retained without interpolation or subtraction.
"""

import csv
from dataclasses import dataclass
import gzip
import re
import warnings

import numpy as np
import pandas as pd

from ..surface_exp import SurfaceBasedExperiment
from .text_io import DELIMITERS

__all__ = [
    "SensorgramTextExperiment",
    "Trace",
    "data_rows",
    "find_header",
    "parse_trace",
    "text_lines",
    "values_per_curve",
]

_CONCENTRATION_UNITS = {
    "M": 1e6,
    "mM": 1e3,
    "uM": 1.0,
    "µM": 1.0,
    "μM": 1.0,
    "nM": 1e-3,
    "pM": 1e-6,
}
_TIME_UNITS = {"s": 1.0, "min": 60.0}
_CONCENTRATION = re.compile(
    r"\bconc\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(M|mM|[uµμ]M|nM|pM)?\s*$",
    re.IGNORECASE,
)


@dataclass
class Trace:
    name: str
    time: np.ndarray
    response: np.ndarray


def text_lines(file, encoding):
    opener = gzip.open if str(file).lower().endswith(".gz") else open
    with opener(file, "rt", encoding=encoding, newline="") as stream:
        return stream.read().splitlines()


def _number(value, decimal, name):
    try:
        result = float(value.strip().replace(decimal, "."))
    except ValueError as error:
        raise ValueError(
            f"Non-numeric measurement in curve {name!r}: {value!r}."
        ) from error
    if not np.isfinite(result):
        raise ValueError(f"Non-finite measurement in curve {name!r}.")
    return result


def parse_trace(name, points, decimal):
    numeric = []
    for x, y in points:
        if not x.strip() and not y.strip():
            continue
        if not x.strip() or not y.strip():
            raise ValueError(f"Incomplete X/Y measurement in curve {name!r}.")
        numeric.append((_number(x, decimal, name), _number(y, decimal, name)))
    if len(numeric) < 2:
        raise ValueError(f"Curve {name!r} must contain at least two measurements.")
    time, response = np.asarray(numeric, dtype=float).T
    if not np.all(np.diff(time) > 0):
        raise ValueError(f"Time must increase strictly in curve {name!r}.")
    return Trace(name, time, response)


def find_header(lines, predicate):
    for i, line in enumerate(lines):
        for delimiter in DELIMITERS:
            fields = [
                field.strip()
                for field in next(
                    csv.reader([line], delimiter=delimiter, skipinitialspace=True)
                )
            ]
            while fields and not fields[-1]:
                fields.pop()
            if predicate(fields):
                return i, delimiter, fields
    raise ValueError("No supported SPR export header found.")


def data_rows(lines, start, delimiter, width):
    for row in csv.reader(lines[start:], delimiter=delimiter, skipinitialspace=True):
        if not any(field.strip() for field in row):
            continue
        if any(field.strip() for field in row[width:]):
            raise ValueError("Data row contains more fields than the export header.")
        yield row[:width] + [""] * max(0, width - len(row))


def values_per_curve(value, names, label):
    if isinstance(value, dict):
        if set(value) != set(names):
            raise ValueError(f"{label} must contain exactly the exported curve names.")
        return [value[name] for name in names]
    if isinstance(value, str):
        return [value] * len(names)
    values = list(value)
    if len(values) != len(names):
        raise ValueError(f"Expected {len(names)} {label} values, got {len(values)}.")
    return values


class SensorgramTextExperiment(SurfaceBasedExperiment):
    """Common import for sensorgram exports without an instrument run log.

    Subclasses supply a reader returning ``Trace`` objects. Each trace becomes
    a sensor with Baseline, Association and Dissociation steps defined by
    caller-supplied boundaries; responses and time grids are kept unchanged.
    """

    def _load(
        self,
        file,
        reader,
        association_end,
        association_start,
        concentrations,
        concentration_unit,
        time_unit,
        sample_id,
        decimal,
        encoding,
    ):
        if decimal not in (".", ","):
            raise ValueError("decimal must be '.' or ','.")
        if time_unit not in _TIME_UNITS:
            raise ValueError("time_unit must be 's' or 'min'.")
        if (
            concentration_unit is not None
            and concentration_unit not in _CONCENTRATION_UNITS
        ):
            raise ValueError(
                "Unsupported concentration unit; use M, mM, uM, µM, nM or pM."
            )
        start, end = float(association_start), float(association_end)
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError(
                "Association boundaries must be finite, with end greater than start."
            )
        traces = reader(file, decimal, encoding)
        names = [trace.name for trace in traces]
        samples = values_per_curve(sample_id, names, "sample_id")
        if any(not isinstance(sample, str) or not sample.strip() for sample in samples):
            raise ValueError("sample_id values must be non-empty strings.")
        if concentrations is not None:
            if isinstance(concentrations, (str, bytes)):
                raise ValueError(
                    "concentrations must be a sequence or curve-name mapping."
                )
            values = values_per_curve(concentrations, names, "concentrations")
            conc = (
                np.asarray(values, dtype=float)
                * _CONCENTRATION_UNITS[concentration_unit or "uM"]
            )
            if conc.ndim != 1:
                raise ValueError("concentrations must contain one scalar per curve.")
        else:
            conc = []
            for name in names:
                match = _CONCENTRATION.search(name)
                unit = (match[2] if match else None) or concentration_unit
                # Unitless labels and Carterra "Conc. 13" indices do not define molarity.
                conc.append(
                    float(match[1]) * _CONCENTRATION_UNITS[unit]
                    if match and unit in _CONCENTRATION_UNITS
                    else np.nan
                )
            conc = np.asarray(conc)
            if np.isnan(conc).any():
                warnings.warn(
                    "Some analyte concentrations are unavailable or have no unit. "
                    "Provide concentrations or edit ligand_conc_df before fitting.",
                    UserWarning,
                )
        if (
            np.isinf(conc).any()
            or (conc < 0).any()
            or (concentrations is not None and np.isnan(conc).any())
        ):
            raise ValueError("Provided concentrations must be finite and non-negative.")
        xs, ys = [], []
        for trace in traces:
            time = trace.time * _TIME_UNITS[time_unit]
            masks = (time < start, (time >= start) & (time < end), time >= end)
            if any(mask.sum() < 2 for mask in masks[1:]):
                raise ValueError(
                    f"Curve {trace.name!r} needs at least two points in each "
                    "association/dissociation phase; check boundaries."
                )
            xs.append([time[mask].copy() for mask in masks])
            ys.append([trace.response[mask].copy() for mask in masks])
        # Commit only a fully validated import, preserving existing data on failure.
        self.sensor_names, self.no_sensors, self.no_steps = names, len(names), 3
        self.xs, self.ys = xs, ys
        self.df_steps = pd.DataFrame(
            {
                "#Step": [1, 2, 3],
                "Name": ["Baseline", "Association", "Dissociation"],
                "Type": ["BASELINE", "ASSOC", "DISASSOC"],
                "Start": [
                    min(
                        start,
                        *(trace.time[0] * _TIME_UNITS[time_unit] for trace in traces),
                    ),
                    start,
                    end,
                ],
                "Column_location": ["0", "1", "1"],
                "Loading_location": [np.nan, "0", np.nan],
            }
        )
        self.ligand_conc_df = pd.DataFrame(
            {
                "Sensor": names,
                "Analyte_location": "1",
                "Concentration_micromolar": conc,
                "SampleID": samples,
                "Replicate": 1,
                "Loading_location": "0",
                "Experiment": self.name,
            }
        )
        self.exp_info = {
            "format": self.type,
            "time_unit_in_file": time_unit,
            "association_start_seconds": start,
            "association_end_seconds": end,
            "concentration_unit_in_file": concentration_unit,
            "curve_labels": names.copy(),
        }
        self.fns = [str(file)]
        self.create_unique_sensor_names()
        self.traces_loaded = True
