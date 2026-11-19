"""Import Carterra row-oriented CSV exports: ``Label, X1, Y1, X2, Y2, ...``."""

import numpy as np
import pandas as pd

from .utils.sensorgram_text import (
    SensorgramTextExperiment,
    data_rows,
    find_header,
    parse_trace,
    text_lines,
)

__all__ = ["CarterraExperiment"]


def _carterra_rows(file, decimal, encoding):
    lines = text_lines(file, encoding)
    start, delimiter, header = find_header(
        lines, lambda fields: len(fields) >= 3 and fields[0] == "Label"
    )
    expected = ["Label"] + [
        f"{axis}{i}" for i in range(1, (len(header) + 1) // 2) for axis in ("X", "Y")
    ]
    if header != expected:
        raise ValueError("Carterra headers must be Label, X1, Y1, X2, Y2, ... .")
    traces = []
    names = set()
    for row in data_rows(lines, start + 1, delimiter, len(header)):
        name = row[0].strip()
        if not name or name in names:
            raise ValueError("Carterra curve labels must be non-empty and unique.")
        names.add(name)
        traces.append(parse_trace(name, zip(row[1::2], row[2::2]), decimal))
    if not traces:
        raise ValueError("The export contains no measured sensorgrams.")
    return traces


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
