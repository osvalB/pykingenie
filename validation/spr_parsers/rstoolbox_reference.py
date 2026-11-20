"""Run the installed external parser, without modifying its implementation."""

import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rstoolbox
from rstoolbox.io import read_SPR


def main():
    frame = read_SPR(sys.argv[1])
    curves = []
    incomplete = {}
    fitted_count = 0
    for kind, concentration, axis in frame.columns:
        if axis != "X":
            continue
        if kind == "fitted":
            fitted_count += 1
            continue
        xy = frame.loc[:, [(kind, concentration, "X"), (kind, concentration, "Y")]]
        if xy.shape[1] != 2:
            raise ValueError("Ambiguous curve identity in the reference output.")
        xy = xy.dropna(how="all")
        missing = int(xy.isna().any(axis=1).sum())
        if missing:
            incomplete[concentration] = missing
        # The reference retains NaNs. Export only complete observations for
        # numeric comparison and explicitly report every omitted partial pair.
        xy = xy.dropna(how="any")
        curves.append(
            {
                "name": concentration,
                "time": xy.iloc[:, 0].tolist(),
                "response": xy.iloc[:, 1].tolist(),
            }
        )
    source = Path(inspect.getsourcefile(read_SPR))
    result = {
        "reference": "rstoolbox.io.read_SPR",
        "version": rstoolbox.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fitted_curves_excluded": fitted_count,
        "incomplete_pairs": incomplete,
        "curves": curves,
    }
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
