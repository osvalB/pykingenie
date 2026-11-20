"""Import Carterra row-oriented CSV exports: ``Label, X1, Y1, X2, Y2, ...``.

What the file provides
----------------------
- One sensorgram per row: a ``Label`` and its time/response pairs, each row on
  its own time grid. Rows marked ``NotFitted`` are still imported.
- The row label (for example ``[Ligand 1-CH31 Conc. 13] [Fitted]``). The
  ``Conc. N`` in it is an index, not a molar concentration.

What the file does not provide
------------------------------
- When each injection starts and ends, or the time between injections.
- Analyte concentrations: every concentration is ``NaN`` unless supplied.
- Which spot/ROI the rows come from, beyond what the label says.
- Native Carterra ``.xlsx`` files are not read: save the worksheet as CSV first.

What the user must provide
--------------------------
A TitrationAnalysis Data Info Sheet (``info=``, see ``pykingenie.info_sheet``)
can supply the timing, concentrations, ``sample_id`` and ``cycle_duration``
below.

- To import: ``association_end`` and, if not 0 s, ``association_start``, in
  seconds on each row's time axis. The same boundaries apply to every row.
  Also ``time_unit="min"`` if time is in minutes. The decimal mark (``.`` or
  ``,``) is detected; pass ``decimal`` only if the file uses both.
- To fit: one concentration per row (``concentrations``, in µM unless
  ``concentration_unit`` is given), or for a dilution series just
  ``top_concentration`` and ``dilution_factor``; and ``sample_id`` to group
  rows of the same analyte (by default all rows form one group).
- For a single-cycle (titration) series on one sensor: ``single_cycle=True``,
  ``cycle_duration`` (seconds from one injection to the next, used to rebuild
  the run clock), rows in acquisition order, one ``sample_id`` and a
  concentration for every row. Different sensors need separate imports.
- Alignment is not done on import; call ``align_association`` before fitting.
"""

import numpy as np
import pandas as pd

from .utils.sensorgram_text import (
    SensorgramTextExperiment,
    data_rows,
    detect_decimal,
    find_header,
    is_per_curve,
    parse_trace,
    text_lines,
)

__all__ = ["CarterraExperiment"]


def is_header(fields):
    """Whether fields are exactly Label, X1, Y1, X2, Y2, ... ."""
    return len(fields) >= 3 and fields == ["Label"] + [
        f"{axis}{i}" for i in range(1, (len(fields) + 1) // 2) for axis in ("X", "Y")
    ]


def _carterra_rows(file, decimal, encoding):
    lines = text_lines(file, encoding)
    start, delimiter, header = find_header(
        lines, lambda fields: len(fields) >= 3 and fields[0] == "Label"
    )
    if not is_header(header):
        raise ValueError("Carterra headers must be Label, X1, Y1, X2, Y2, ... .")
    rows = list(data_rows(lines, start + 1, delimiter, len(header)))
    if not rows:
        raise ValueError("The export contains no measured sensorgrams.")
    decimal = detect_decimal((cell for row in rows for cell in row[1:]), decimal)
    traces = []
    names = set()
    for row in rows:
        name = row[0].strip()
        if not name or name in names:
            raise ValueError("Carterra curve labels must be non-empty and unique.")
        names.add(name)
        traces.append(parse_trace(name, zip(row[1::2], row[2::2]), decimal))
    return traces, decimal


def _dilution(top_concentration, dilution_factor, highest_last, concentrations):
    """Validate the dilution-series arguments; None when they are not used."""
    if top_concentration is None and dilution_factor is None:
        return None
    if top_concentration is None or dilution_factor is None:
        raise ValueError("A dilution series needs both top_concentration and dilution_factor.")
    if concentrations is not None:
        raise ValueError("Pass either concentrations or a dilution series, not both.")
    if not isinstance(highest_last, (bool, np.bool_)):
        raise ValueError("highest_last must be a boolean.")
    try:
        top, factor = float(top_concentration), float(dilution_factor)
    except (TypeError, ValueError) as error:
        raise ValueError("top_concentration and dilution_factor must be numbers.") from error
    if not np.isfinite(top) or top <= 0:
        raise ValueError("top_concentration must be positive and finite.")
    if not np.isfinite(factor) or factor <= 1:
        raise ValueError("dilution_factor must be greater than 1.")
    return top, factor, bool(highest_last)


class CarterraExperiment(SensorgramTextExperiment):
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
        association_end=None,
        association_start=None,
        concentrations=None,
        concentration_unit=None,
        time_unit="s",
        sample_id=None,
        decimal=None,
        encoding="utf-8-sig",
        single_cycle=False,
        cycle_duration=None,
        info=None,
        top_concentration=None,
        dilution_factor=None,
        highest_last=True,
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

        info is a TitrationAnalysis Data Info Sheet path, or one row of
        read_info_sheet(). It supplies the association boundaries, the molar
        concentrations, sample_id (the analyte) and, with single_cycle=True,
        cycle_duration = Baseline + Association + Dissociation. Arguments you
        pass take precedence. Without info, association_start defaults to 0 s
        and sample_id to "sample".

        For a dilution series, pass top_concentration (in concentration_unit,
        µM by default) and dilution_factor instead of concentrations: row k
        from the highest gets top_concentration / dilution_factor**k. By
        default the highest concentration is the last row, as in a titration
        run from low to high; set highest_last=False if it is the first row.
        The series is assigned by row order only, not from the labels.
        """
        if not isinstance(single_cycle, (bool, np.bool_)):
            raise ValueError("single_cycle must be a boolean.")
        if not single_cycle and cycle_duration is not None:
            raise ValueError("cycle_duration requires single_cycle=True.")
        if is_per_curve(association_start) or is_per_curve(association_end):
            raise ValueError(
                "Carterra imports use one association_start and association_end "
                "for every row."
            )
        dilution = _dilution(top_concentration, dilution_factor, highest_last,
                             concentrations)
        settings, sheet = self._settings(
            info,
            association_start=association_start,
            association_end=association_end,
            concentrations=concentrations,
            concentration_unit=concentration_unit,
            sample_id=sample_id,
            dilution=dilution,
        )
        if cycle_duration is None:
            cycle_duration = sheet.get("cycle_duration")
        target = CarterraExperiment(self.name) if single_cycle else self
        target._load(
            file,
            _carterra_rows,
            time_unit=time_unit,
            decimal=decimal,
            encoding=encoding,
            **settings,
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
