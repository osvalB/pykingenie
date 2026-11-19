"""A dilution series can replace the per-row Carterra concentration list."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pykingenie import CarterraExperiment

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "notebooks" / "data" / "spr_exports"


def write_rows(path, count=4):
    time = np.arange(0, 50, 0.5)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Label"] + [f"{axis}{i + 1}" for i in range(len(time)) for axis in "XY"])
        for i in range(count):
            response = (i + 1) * (time >= 10)
            writer.writerow([f"row {i + 1}"] + list(np.column_stack([time, response]).ravel()))
    return path


def read(path, **kwargs):
    experiment = CarterraExperiment()
    experiment.read_csv(path, association_start=10, association_end=30, **kwargs)
    return experiment.ligand_conc_df["Concentration_micromolar"].to_numpy()


def test_highest_concentration_is_the_last_row_by_default(tmp_path):
    conc = read(write_rows(tmp_path / "rows.csv"), top_concentration=8, dilution_factor=2)
    np.testing.assert_allclose(conc, [1, 2, 4, 8])


def test_highest_concentration_can_be_the_first_row(tmp_path):
    conc = read(write_rows(tmp_path / "rows.csv"), top_concentration=9,
                dilution_factor=3, highest_last=False)
    np.testing.assert_allclose(conc, [9, 3, 1, 1 / 3])


def test_top_concentration_uses_concentration_unit(tmp_path):
    conc = read(write_rows(tmp_path / "rows.csv", count=2), top_concentration=500,
                dilution_factor=2, concentration_unit="nM")
    np.testing.assert_allclose(conc, [0.25, 0.5])


def test_example_titration_matches_its_published_concentrations():
    metadata = json.loads((EXAMPLES / "fit_metadata.json").read_text())["carterra_rows"]
    experiment = CarterraExperiment()
    experiment.read_csv(EXAMPLES / "carterra_rows.csv", association_start=120,
                        association_end=420, single_cycle=True, cycle_duration=1140,
                        top_concentration=1, dilution_factor=2)
    # The published values are rounded to three significant digits.
    np.testing.assert_allclose(experiment.ligand_conc_df["Concentration_micromolar"],
                               metadata["cycle_concentrations"], rtol=2e-3)


def test_dilution_series_overrides_info_sheet_concentrations(tmp_path):
    sheet = tmp_path / "info.csv"
    with sheet.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Analyte", "Baseline", "Association", "Dissociation",
                         "All Concentrations", "Incl. Conc."])
        writer.writerow(["AN1", "10", "20", "20", "1E-9, 2E-9, 3E-9, 4E-9", "4E-06, 8E-06"])
    experiment = CarterraExperiment()
    experiment.read_csv(write_rows(tmp_path / "rows.csv"), info=sheet,
                        top_concentration=8, dilution_factor=2)
    conc = experiment.ligand_conc_df
    np.testing.assert_allclose(conc["Concentration_micromolar"], [1, 2, 4, 8])
    assert conc["Select"].tolist() == [False, False, True, True]
    assert experiment.exp_info["association_start_seconds"] == 10


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"top_concentration": 1}, "needs both"),
        ({"dilution_factor": 2}, "needs both"),
        ({"top_concentration": 1, "dilution_factor": 2, "concentrations": [1, 2, 3, 4]},
         "not both"),
        ({"top_concentration": 1, "dilution_factor": 2, "highest_last": "yes"}, "boolean"),
        ({"top_concentration": "high", "dilution_factor": 2}, "must be numbers"),
        ({"top_concentration": 0, "dilution_factor": 2}, "positive"),
        ({"top_concentration": np.inf, "dilution_factor": 2}, "positive"),
        ({"top_concentration": 1, "dilution_factor": 1}, "greater than 1"),
        ({"top_concentration": 1, "dilution_factor": np.nan}, "greater than 1"),
    ],
)
def test_invalid_dilution_series_is_rejected(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        read(write_rows(tmp_path / "rows.csv"), **kwargs)
