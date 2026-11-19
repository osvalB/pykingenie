"""Import verified paired-column SPR and row-oriented Carterra text exports.

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

from .surface_exp import SurfaceBasedExperiment
from .utils.text_io import DELIMITERS

__all__ = ["SprTextExperiment", "CarterraExperiment"]

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
_AXIS = re.compile(r"^(.*?)_([XY])\s*$", re.IGNORECASE)
_CONCENTRATION = re.compile(
    r"\bconc\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(M|mM|[uµμ]M|nM|pM)?\s*$",
    re.IGNORECASE,
)


@dataclass
class _Trace:
    name: str
    time: np.ndarray
    response: np.ndarray


def _text_lines(file, encoding):
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


def _trace(name, points, decimal):
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
    return _Trace(name, time, response)


def _find_header(lines, predicate):
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


def _rows(lines, start, delimiter, width):
    for row in csv.reader(lines[start:], delimiter=delimiter, skipinitialspace=True):
        if not any(field.strip() for field in row):
            continue
        if any(field.strip() for field in row[width:]):
            raise ValueError("Data row contains more fields than the export header.")
        yield row[:width] + [""] * max(0, width - len(row))


def _paired_columns(file, decimal, encoding):
    lines = _text_lines(file, encoding)
    start, delimiter, header = _find_header(
        lines,
        lambda fields: len(fields) >= 2
        and all(_AXIS.fullmatch(field) for field in fields),
    )
    pairs = {}
    for index, column in enumerate(header):
        match = _AXIS.fullmatch(column)
        name, axis = match[1].strip(), match[2].upper()
        # Exported fitting curves are not experimental observations.
        if re.search(r"\bfitted\s+curve\b", name, flags=re.IGNORECASE):
            continue
        if not name or axis in pairs.setdefault(name, {}):
            raise ValueError(f"Empty or duplicate X/Y curve header: {column!r}.")
        pairs[name][axis] = index
    if not pairs:
        raise ValueError("The export contains no measured sensorgrams.")
    if any(set(axes) != {"X", "Y"} for axes in pairs.values()):
        raise ValueError(
            "Each measured curve must have exactly one X and one Y column."
        )
    rows = list(_rows(lines, start + 1, delimiter, len(header)))
    return [
        _trace(name, [(row[axes["X"]], row[axes["Y"]]) for row in rows], decimal)
        for name, axes in pairs.items()
    ]


def _carterra_rows(file, decimal, encoding):
    lines = _text_lines(file, encoding)
    start, delimiter, header = _find_header(
        lines, lambda fields: len(fields) >= 3 and fields[0] == "Label"
    )
    expected = ["Label"] + [
        f"{axis}{i}" for i in range(1, (len(header) + 1) // 2) for axis in ("X", "Y")
    ]
    if header != expected:
        raise ValueError("Carterra headers must be Label, X1, Y1, X2, Y2, ... .")
    traces = []
    names = set()
    for row in _rows(lines, start + 1, delimiter, len(header)):
        name = row[0].strip()
        if not name or name in names:
            raise ValueError("Carterra curve labels must be non-empty and unique.")
        names.add(name)
        traces.append(_trace(name, zip(row[1::2], row[2::2]), decimal))
    if not traces:
        raise ValueError("The export contains no measured sensorgrams.")
    return traces


def _values_per_curve(value, names, label):
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


class _SensorgramExport(SurfaceBasedExperiment):
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
        samples = _values_per_curve(sample_id, names, "sample_id")
        if any(not isinstance(sample, str) or not sample.strip() for sample in samples):
            raise ValueError("sample_id values must be non-empty strings.")
        if concentrations is not None:
            if isinstance(concentrations, (str, bytes)):
                raise ValueError(
                    "concentrations must be a sequence or curve-name mapping."
                )
            values = _values_per_curve(concentrations, names, "concentrations")
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


class SprTextExperiment(_SensorgramExport):
    """SPR text exports containing one named _X/_Y column pair per sensorgram.

    Verified with Biacore T200 Cycle/Fc and Cycle/sample headers and the public
    rstoolbox Run/Ch/Cy/conc example. Fitted-curve columns are excluded.
    """

    def __init__(self, name="spr_text"):
        super().__init__(name, "spr_text")

    def read_file(
        self,
        file,
        *,
        association_end,
        association_start=0,
        concentrations=None,
        concentration_unit=None,
        time_unit="s",
        sample_id="sample",
        decimal=".",
        encoding="utf-8-sig",
    ):
        """Read paired-column text/CSV/TSV, optionally gzip compressed.

        Comma, semicolon and tab delimiters are detected automatically.
        Standard CSV quoting and spaces after delimiters are accepted.

        Boundaries are seconds on the exported time axis; association is
        [start, end) and dissociation starts at end. Missing baseline points
        are allowed. Concentrations are a list or curve-name mapping, in µM
        unless concentration_unit is supplied. Without concentrations, only
        explicit conc= labels with a known unit are used; other values stay
        NaN with a warning. sample_id accepts a common string or one value per
        curve (list or mapping), to distinguish different binding interactions.
        time_unit describes the file ('s' or 'min'); decimal is '.' or ','.
        encoding can be changed for older instrument exports. No times, units
        or concentrations are inferred from signal shape or curve position.
        """
        self._load(
            file,
            _paired_columns,
            association_end,
            association_start,
            concentrations,
            concentration_unit,
            time_unit,
            sample_id,
            decimal,
            encoding,
        )


class CarterraExperiment(_SensorgramExport):
    """Carterra row-oriented CSV export: Label, X1, Y1, X2, Y2, ... .

    Each row is a sensorgram. Conc. N in a label is an index, not a molar value.
    XLSX exports must first be saved as CSV; no Excel dependency is required.
    """

    def __init__(self, name="carterra"):
        super().__init__(name, "carterra_csv")

    def read_csv(
        self,
        file,
        *,
        association_end,
        association_start=0,
        concentrations=None,
        concentration_unit=None,
        time_unit="s",
        sample_id="sample",
        decimal=".",
        encoding="utf-8-sig",
        single_cycle=False,
        cycle_duration=None,
    ):
        """Read Carterra CSV with caller-supplied phase boundaries in seconds.

        Comma, semicolon and tab delimiters are detected automatically.
        Standard CSV quoting and spaces after delimiters are accepted.

        Concentrations and sample_id follow SprTextExperiment.read_file().
        Original row labels and each curve's time grid are retained. Labels
        marked NotFitted remain measured data and are included in the import.

        With single_cycle=True, rows must be consecutive injections on one
        physical sensor, in acquisition order. Supply cycle_duration in seconds:
        each row's exported clock is translated by this period. This timing is
        caller supplied, not inferred from labels or signal shape. Rows must not
        overlap on the resulting clock. The next row's pre-association baseline
        extends the preceding dissociation, yielding BASELINE, ASSOC, DISASSOC,
        ASSOC, DISASSOC, ... . This lets align_association preserve occupied
        states after the first injection and lets the existing fitter carry the
        bound state between injections. No response is shifted or interpolated
        during import. Different physical sensors need separate imports.
        """
        if not isinstance(single_cycle, (bool, np.bool_)):
            raise ValueError("single_cycle must be a boolean.")
        if not single_cycle and cycle_duration is not None:
            raise ValueError("cycle_duration requires single_cycle=True.")
        target = CarterraExperiment(self.name) if single_cycle else self
        target._load(
            file,
            _carterra_rows,
            association_end,
            association_start,
            concentrations,
            concentration_unit,
            time_unit,
            sample_id,
            decimal,
            encoding,
        )
        if single_cycle:
            target._combine_cycles(cycle_duration)
            # Keep the existing import intact if sequence validation fails.
            self.__dict__.update(target.__dict__)

    def _combine_cycles(self, cycle_duration):
        """Represent row-wise injections as one sensor with alternating steps."""
        try:
            period = float(cycle_duration)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "single_cycle requires a positive cycle_duration in seconds."
            ) from error
        if not np.isfinite(period) or period <= 0:
            raise ValueError(
                "single_cycle requires a positive cycle_duration in seconds."
            )
        if self.ligand_conc_df.SampleID.nunique() != 1:
            raise ValueError(
                "A single-cycle import must contain one sample_id on one sensor."
            )
        if not np.isfinite(self.ligand_conc_df.Concentration_micromolar).all():
            raise ValueError("Provide concentrations for every single-cycle injection.")

        times = [
            [phase + i * period for phase in curve] for i, curve in enumerate(self.xs)
        ]
        clock = np.concatenate([np.concatenate(curve) for curve in times])
        if not np.all(np.diff(clock) > 0):
            raise ValueError(
                "Single-cycle rows overlap in time; check cycle_duration and row order."
            )

        start = self.exp_info["association_start_seconds"]
        end = self.exp_info["association_end_seconds"]
        xs, ys = [times[0][0]], [self.ys[0][0].copy()]
        steps = [
            {
                "Name": "Baseline",
                "Type": "BASELINE",
                "Start": self.df_steps.Start.iloc[0],
                "Column_location": "0",
                "Loading_location": np.nan,
            }
        ]
        for i, (time, signal) in enumerate(zip(times, self.ys)):
            if i:
                # With no analyte present, the next baseline belongs to the
                # preceding dissociation, not a fresh empty-surface baseline.
                xs[-1] = np.concatenate([xs[-1], time[0]])
                ys[-1] = np.concatenate([ys[-1], signal[0]])
            xs.extend([time[1], time[2]])
            ys.extend([signal[1].copy(), signal[2].copy()])
            for name, kind, boundary in [
                ("Association", "ASSOC", start),
                ("Dissociation", "DISASSOC", end),
            ]:
                steps.append(
                    {
                        "Name": f"{name} {i + 1}",
                        "Type": kind,
                        "Start": i * period + boundary,
                        "Column_location": str(i + 1),
                        "Loading_location": "0" if kind == "ASSOC" else np.nan,
                    }
                )

        labels = self.sensor_names.copy()
        count = len(labels)
        sensor = "single_cycle"
        self.sensor_names, self.no_sensors, self.no_steps = [sensor], 1, len(steps)
        self.xs, self.ys = [xs], [ys]
        self.df_steps = pd.DataFrame(steps)
        self.df_steps.insert(0, "#Step", np.arange(len(steps)) + 1)
        self.ligand_conc_df["Source_curve"] = labels
        self.ligand_conc_df["Sensor"] = sensor
        self.ligand_conc_df["Analyte_location"] = [str(i + 1) for i in range(count)]
        self.exp_info.update(
            single_cycle=True,
            cycle_duration_seconds=period,
            cycle_order=labels,
            cycle_count=count,
        )
        self.create_unique_sensor_names()
