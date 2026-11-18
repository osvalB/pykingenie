"""Verify the metadata and published references used by the SPR examples."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pykingenie import CarterraExperiment, SprTextExperiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "key,filename,entries",
    [
        ("biacore_cycle_sample", "Example T200 Data Info Sheet.xlsx", [3, 4, 5, 6, 7]),
        (
            "carterra_rows",
            "Example Data Info Sheet - Carterra.xlsx",
            [8, 9, 10, 11, 12],
        ),
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
    assert metadata["source_concentration_entries_one_based"] == [
        i + 1 for i in entries
    ]


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


@pytest.mark.parametrize(
    "key,reader_type,published_nm",
    [
        ("biacore_cycle_sample", SprTextExperiment, [221.7, 443.4, 886.8, 1774, 3547]),
        ("carterra_rows", CarterraExperiment, [62.5, 125, 250, 500, 1000]),
    ],
)
def test_imported_five_curves_match_published_selection_and_windows(
    key, reader_type, published_nm
):
    folder = ROOT / "notebooks/data/spr_exports"
    metadata = json.loads((folder / "fit_metadata.json").read_text())[key]
    manifest = json.loads((folder / "manifest.json").read_text())
    entry = next(x for x in manifest["examples"] if x["file"] == metadata["file"])
    path = folder / metadata["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["output_sha256"]
    experiment = reader_type()
    read = experiment.read_csv if key == "carterra_rows" else experiment.read_file
    read(
        path,
        association_start=metadata["association_start"],
        association_end=metadata["association_end"],
        concentrations=metadata["concentrations"],
        concentration_unit="uM",
    )
    assert (
        experiment.no_sensors == metadata["literature_reference"]["curves_fitted"] == 5
    )
    np.testing.assert_allclose(
        metadata["literature_reference"]["concentrations_nM"], published_nm
    )
    np.testing.assert_allclose(
        experiment.ligand_conc_df["Concentration_micromolar"] * 1000, published_nm
    )
    assert (
        metadata["trim_time"]
        == metadata["source_sheet_row"]["Asso. Skip"]
        == metadata["source_sheet_row"]["Disso. Skip"]
        == 0
    )
    lo, hi = entry["retained_time_window_seconds"]
    for i, curve in enumerate(entry["curves"]):
        time = np.concatenate(experiment.xs[i])
        assert len(time) == curve["points"]
        assert time.min() >= lo and time.max() <= hi
        assert np.isfinite(np.concatenate(experiment.ys[i])).all()
        a, d = experiment.xs[i][1:]
        assert a.min() >= metadata["fitting_window_seconds"]["association"][0]
        assert a.max() < metadata["fitting_window_seconds"]["association"][1]
        assert d.min() >= metadata["fitting_window_seconds"]["dissociation"][0]
        assert d.max() <= metadata["fitting_window_seconds"]["dissociation"][1]
