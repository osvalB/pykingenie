"""Verify the metadata and published references used by the SPR examples."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "key,filename,entries",
    [
        ("biacore_cycle_sample", "Example T200 Data Info Sheet.xlsx", [5, 6, 7]),
        ("carterra_rows", "Example Data Info Sheet - Carterra.xlsx", [7, 10, 12]),
    ],
)
def test_verified_concentrations_match_author_list_and_original_workbook(
    key, filename, entries
):
    folder = ROOT / "notebooks/data/spr_exports"
    metadata = json.loads((folder / "fit_metadata.json").read_text())[key]
    assert (
        hashlib.sha256((folder / "metadata" / filename).read_bytes()).hexdigest()
        == metadata["source_sha256"]
    )
    molar = [
        float(x) for x in metadata["source_sheet_row"]["All Concentrations"].split(",")
    ]
    np.testing.assert_allclose(
        metadata["concentrations"], np.array(molar)[entries] * 1e6
    )


@pytest.mark.parametrize("key", ["biacore_cycle_sample", "carterra_rows"])
def test_literature_values_match_unmodified_author_report(key):
    folder = ROOT / "notebooks/data/spr_exports"
    reference = json.loads((folder / "fit_metadata.json").read_text())[key][
        "literature_reference"
    ]
    path = folder / reference["local_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["source_sha256"]
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if all(row[k] == v for k, v in reference["row_identity"].items())
        ]
    assert len(rows) == 1
    expected = {
        "kon": float(rows[0]["ka"]),
        "koff": float(rows[0]["kd"]),
        "Kd": float(rows[0]["KD"]),
    }
    assert reference["parameters_molar"] == expected
