"""Import paired-column SPR text exports (one ``_X``/``_Y`` pair per curve).

What the file provides
----------------------
- One sensorgram per ``<label>_X`` / ``<label>_Y`` column pair: time and
  response, each curve on its own time grid. Columns of fitted curves are ignored.
- The curve label from the header (for example ``Cycle=1_Fc=4-3`` or
  ``Run 1; Ch 1; Cy 14; conc=15``). It becomes the sensor name.
- A concentration only when the label ends in ``conc=<value><unit>`` with a
  molar unit (M, mM, uM, nM, pM). Other labels give ``NaN`` and a warning.

What the file does not provide
------------------------------
- When each injection starts and ends.
- Concentrations, in most exports.
- Which curves belong to the same analyte, and which flow cell is the reference.
- Plate positions, ligand loading or regeneration steps.

What the user must provide
--------------------------
A TitrationAnalysis Data Info Sheet (``info=``, see ``pykingenie.info_sheet``)
can supply the timing, concentrations and ``sample_id`` below.

- To import: ``association_end`` and, if not 0 s, ``association_start``, in
  seconds on the exported time axis. The same boundaries apply to every curve
  in the file, so curves with different timing need separate imports. Also
  ``decimal=","`` for decimal-comma files and ``time_unit="min"`` if time is
  in minutes.
- To fit: one concentration per curve (``concentrations``, in µM unless
  ``concentration_unit`` is given), and ``sample_id`` to group curves of the
  same analyte (by default all curves form one group). Reference subtraction
  and alignment are not done on import; use ``subtraction`` and
  ``align_association`` on the experiment.
"""

import re

from .utils.sensorgram_text import (
    SensorgramTextExperiment,
    data_rows,
    find_header,
    parse_trace,
    text_lines,
)

__all__ = ["SprTextExperiment"]

_AXIS = re.compile(r"^(.*?)_([XY])\s*$", re.IGNORECASE)


def _paired_columns(file, decimal, encoding):
    lines = text_lines(file, encoding)
    start, delimiter, header = find_header(
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
    rows = list(data_rows(lines, start + 1, delimiter, len(header)))
    return [
        parse_trace(name, [(row[axes["X"]], row[axes["Y"]]) for row in rows], decimal)
        for name, axes in pairs.items()
    ]


class SprTextExperiment(SensorgramTextExperiment):
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
        association_end=None,
        association_start=None,
        concentrations=None,
        concentration_unit=None,
        time_unit="s",
        sample_id=None,
        decimal=".",
        encoding="utf-8-sig",
        info=None,
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

        info is a TitrationAnalysis Data Info Sheet path, or one row of
        read_info_sheet(). It supplies the association boundaries, the molar
        concentrations and sample_id (the analyte); arguments you pass take
        precedence. Without info, association_start defaults to 0 s and
        sample_id to "sample".
        """
        settings, _ = self._settings(
            info,
            association_start=association_start,
            association_end=association_end,
            concentrations=concentrations,
            concentration_unit=concentration_unit,
            sample_id=sample_id,
        )
        self._load(
            file,
            _paired_columns,
            time_unit=time_unit,
            decimal=decimal,
            encoding=encoding,
            **settings,
        )
