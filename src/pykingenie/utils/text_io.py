"""Shared delimiter handling for exported text tables."""

import csv
from itertools import islice

import pandas as pd
from pandas.io.common import get_handle

DELIMITERS = ("\t", ",", ";")


def read_delimited_table(file, *, skiprows=0, header="infer"):
    """Read a comma-, tab- or semicolon-separated table with CSV quoting.

    Detect the separator from up to ten non-empty rows, after ``skiprows``.
    Fall back to the header when optional trailing fields make rows ragged.
    Limit inference to supported separators so a one-column file cannot be
    silently split on letters in its header. Pandas' input handle preserves its
    normal compression support. Spaces after separators are ignored outside
    quoted fields; separators within quoted names or values are preserved.
    """
    with get_handle(file, "r", encoding="utf-8-sig", compression="infer") as handles:
        stream = handles.handle
        start = stream.tell()
        for _ in range(skiprows):
            next(stream, None)
        lines = list(islice((line for line in stream if line.strip()), 10))
        # A comma inside a semicolon-separated header must not outweigh the
        # consistent separators in its data rows. Header-only inference still
        # handles tables with missing optional trailing fields.
        delimiter = ","
        for sample in ("".join(lines), lines[0] if lines else ""):
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
                break
            except csv.Error:
                # Preserve single-column input for the reader's normal validation.
                continue
        stream.seek(start)
        return pd.read_csv(
            stream,
            sep=delimiter,
            skiprows=skiprows,
            header=header,
            skipinitialspace=True,
        )
