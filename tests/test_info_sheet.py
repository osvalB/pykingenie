"""TitrationAnalysis Data Info Sheets supply the settings sensorgram exports lack."""

import csv
from pathlib import Path

import numpy as np
import pytest

from pykingenie import (
    CarterraExperiment,
    KineticsAnalyzer,
    SprTextExperiment,
    read_info_sheet,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "notebooks" / "data" / "spr_exports"
CARTERRA_SHEET = EXAMPLES / "metadata" / "Example Data Info Sheet - Carterra.xlsx"
T200_SHEET = EXAMPLES / "metadata" / "Example T200 Data Info Sheet.xlsx"

HEADER = ["Incl.", "Block/Chip/Tray", "Position/Channel/Sensor", "Analyte", "Ligand",
          "Baseline", "Association", "Dissociation", "Asso. Skip", "Disso. Skip",
          "All Concentrations", "Incl. Conc.", "Bulkshift", "Regen.", "Bsl Start",
          "Bass Corr", "Global Rmax"]
ROW = ["Y", "1", "A1", "AN1", "LIG", "10", "20", "20", "2", "3",
       "1E-07, 4E-07, 1E-06", "4E-07, 1E-06", "N", "N", "0", "Y", "N"]


def write_sheet(path, rows=(ROW,), header=HEADER):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_cycles(path, labels=("cycle 1", "cycle 2", "cycle 3")):
    time = np.arange(0, 50, 0.5)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Label"] + [f"{axis}{i + 1}" for i in range(len(time)) for axis in "XY"])
        for i, label in enumerate(labels):
            response = (i + 1) * (time >= 10)
            writer.writerow([label] + list(np.column_stack([time, response]).ravel()))
    return path


def write_paired(path, count=3):
    time = np.arange(0, 50, 0.5)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"curve {i}_{axis}" for i in range(count) for axis in "XY"])
        for t in time:
            writer.writerow([value for i in range(count) for value in (t, i * (t >= 10))])
    return path


def test_csv_sheet_is_normalised(tmp_path):
    sheet = read_info_sheet(write_sheet(tmp_path / "info.csv"))
    row = sheet.iloc[0]
    assert row["Included"] and row["Baseline_correction"] and not row["Global_Rmax"]
    assert (row["Baseline_s"], row["Association_s"], row["Dissociation_s"]) == (10, 20, 20)
    assert (row["Association_skip_s"], row["Dissociation_skip_s"]) == (2, 3)
    np.testing.assert_allclose(row["Concentrations_micromolar"], [0.1, 0.4, 1.0])
    np.testing.assert_allclose(row["Included_concentrations_micromolar"], [0.4, 1.0])


def test_optional_columns_take_defaults(tmp_path):
    header = ["Analyte", "Baseline", "Association", "Dissociation", "All Concentrations"]
    row = read_info_sheet(write_sheet(tmp_path / "info.csv", [["AN1", "5", "6", "7", "1E-6"]],
                                      header)).iloc[0]
    assert row["Included"] and row["Baseline_start_s"] == 0
    assert row["Included_concentrations_micromolar"] is None
    assert row["Bulk_shift"] is None and row["Ligand"] == ""


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("Baseline", "ten", "number of seconds"),
        ("Baseline", "inf", "finite"),
        ("All Concentrations", "", "All Concentrations is empty"),
        ("All Concentrations", "1E-6, high", "molar concentrations"),
        ("All Concentrations", "-1E-6", "non-negative"),
        ("Analyte", "", "Analyte is empty"),
        ("Association", "0", "must be positive"),
    ],
)
def test_invalid_rows_are_rejected(tmp_path, column, value, message):
    row = list(ROW)
    row[HEADER.index(column)] = value
    with pytest.raises(ValueError, match=message):
        read_info_sheet(write_sheet(tmp_path / "info.csv", [row]))


def test_other_tables_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing columns Baseline"):
        read_info_sheet(write_sheet(tmp_path / "info.csv", [["AN1"]], ["Analyte"]))
    with pytest.raises(ValueError, match="no rows"):
        read_info_sheet(write_sheet(tmp_path / "empty.csv", []))


def test_carterra_example_sheet_reproduces_the_notebook_settings():
    experiment = CarterraExperiment("carterra_rows")
    experiment.read_csv(EXAMPLES / "carterra_rows.csv", info=CARTERRA_SHEET, single_cycle=True)
    assert experiment.exp_info["association_start_seconds"] == 120
    assert experiment.exp_info["association_end_seconds"] == 420
    assert experiment.exp_info["cycle_duration_seconds"] == 1140
    assert experiment.exp_info["info_sheet"]["Position"] == "A1"
    conc = experiment.ligand_conc_df
    np.testing.assert_allclose(conc["Concentration_micromolar"].iloc[-5:],
                               [0.0625, 0.125, 0.25, 0.5, 1.0])
    assert (conc["SampleID"] == "AE.A244").all()
    assert conc["Select"].tolist() == [False] * 8 + [True] * 5


def test_t200_excerpt_needs_its_own_concentrations():
    experiment = SprTextExperiment("t200")
    with pytest.raises(ValueError, match="lists 8 concentrations but the file has 5"):
        experiment.read_file(EXAMPLES / "biacore_cycle_sample.tsv", info=T200_SHEET)
    experiment.read_file(EXAMPLES / "biacore_cycle_sample.tsv", info=T200_SHEET,
                         concentrations=[0.2217, 0.4434, 0.8868, 1.774, 3.547])
    assert experiment.exp_info["association_start_seconds"] == 0
    assert experiment.exp_info["association_end_seconds"] == 180
    assert "Select" not in experiment.ligand_conc_df  # "Incl. Conc." is All
    assert (experiment.ligand_conc_df["SampleID"] == "AE.A244").all()


def test_sheet_fills_a_single_cycle_import_and_fitting_selection(tmp_path):
    sheet = write_sheet(tmp_path / "info.csv")
    experiment = CarterraExperiment()
    experiment.read_csv(write_cycles(tmp_path / "cycles.csv"), info=sheet, single_cycle=True)
    assert experiment.exp_info["cycle_duration_seconds"] == 50
    assert experiment.df_steps["Start"].tolist()[1:3] == [10, 30]
    np.testing.assert_allclose(experiment.ligand_conc_df["Concentration_micromolar"], [0.1, 0.4, 1.0])
    assert experiment.exp_info["info_sheet"]["Association_skip_s"] == 2

    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(experiment, experiment.name)
    analyzer.merge_ligand_conc_df()
    assert analyzer.combined_ligand_conc_df["Select"].tolist() == [False, True, True]


def test_explicit_arguments_override_the_sheet(tmp_path):
    sheet = write_sheet(tmp_path / "info.csv")
    experiment = CarterraExperiment()
    # The overriding concentrations are not in the sheet's "Incl. Conc." list.
    with pytest.warns(UserWarning, match="No curve matches"):
        experiment.read_csv(write_cycles(tmp_path / "cycles.csv"), info=sheet,
                            single_cycle=True, association_start=12, association_end=28,
                            cycle_duration=60, concentrations=[1, 2, 3],
                            concentration_unit="nM", sample_id="other")
    assert experiment.exp_info["association_start_seconds"] == 12
    assert experiment.exp_info["association_end_seconds"] == 28
    assert experiment.exp_info["cycle_duration_seconds"] == 60
    np.testing.assert_allclose(experiment.ligand_conc_df["Concentration_micromolar"],
                               [0.001, 0.002, 0.003])
    assert (experiment.ligand_conc_df["SampleID"] == "other").all()


def test_sheet_without_single_cycle_ignores_the_cycle_period(tmp_path):
    experiment = CarterraExperiment()
    experiment.read_csv(write_cycles(tmp_path / "cycles.csv"), info=write_sheet(tmp_path / "i.csv"))
    assert experiment.no_sensors == 3 and "cycle_duration_seconds" not in experiment.exp_info


def test_unmatched_included_concentrations_select_everything(tmp_path):
    row = list(ROW)
    row[HEADER.index("Incl. Conc.")] = "5E-06"
    experiment = SprTextExperiment()
    with pytest.warns(UserWarning, match="No curve matches"):
        experiment.read_file(write_paired(tmp_path / "p.csv"),
                             info=write_sheet(tmp_path / "info.csv", [row]))
    assert experiment.ligand_conc_df["Select"].all()


def test_unit_for_sheet_concentrations_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="already molar"):
        SprTextExperiment().read_file(write_paired(tmp_path / "p.csv"),
                                      info=write_sheet(tmp_path / "info.csv"),
                                      concentration_unit="nM")


def test_sheet_rows_that_differ_need_a_choice(tmp_path):
    other = list(ROW)
    other[HEADER.index("Association")] = "15"
    excluded = list(ROW)
    excluded[HEADER.index("Incl.")] = "N"
    excluded[HEADER.index("Association")] = "99"
    data = write_paired(tmp_path / "p.csv")

    agreeing = write_sheet(tmp_path / "agree.csv", [ROW, excluded])
    SprTextExperiment().read_file(data, info=agreeing)

    differing = write_sheet(tmp_path / "differ.csv", [ROW, other])
    with pytest.raises(ValueError, match="Pass one row instead"):
        SprTextExperiment().read_file(data, info=differing)
    experiment = SprTextExperiment()
    experiment.read_file(data, info=read_info_sheet(differing).iloc[1])
    assert experiment.exp_info["association_end_seconds"] == 25


def test_sheet_with_only_excluded_rows_uses_them(tmp_path):
    row = list(ROW)
    row[HEADER.index("Incl.")] = "N"
    experiment = SprTextExperiment()
    experiment.read_file(write_paired(tmp_path / "p.csv"),
                         info=write_sheet(tmp_path / "info.csv", [row]))
    assert experiment.exp_info["association_end_seconds"] == 30


def test_info_must_be_a_path_or_a_sheet_row(tmp_path):
    data = write_paired(tmp_path / "p.csv")
    with pytest.raises(TypeError, match="info must be"):
        SprTextExperiment().read_file(data, info=3)
    with pytest.raises(ValueError, match="info is missing"):
        SprTextExperiment().read_file(data, info={"Analyte": "AN1"})


def test_boundaries_are_required_without_a_sheet(tmp_path):
    with pytest.raises(ValueError, match="Provide association_end"):
        SprTextExperiment().read_file(write_paired(tmp_path / "p.csv"))


def test_merge_keeps_selection_from_one_experiment_only(tmp_path):
    marked = SprTextExperiment("marked")
    marked.read_file(write_paired(tmp_path / "a.csv"), info=write_sheet(tmp_path / "info.csv"))
    plain = SprTextExperiment("plain")
    plain.read_file(write_paired(tmp_path / "b.csv"), association_start=10,
                    association_end=30, concentrations=[1, 2, 3])
    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(marked, "marked")
    analyzer.add_experiment(plain, "plain")
    analyzer.merge_ligand_conc_df()
    assert analyzer.combined_ligand_conc_df["Select"].tolist() == [False, True, True,
                                                                   True, True, True]
    assert analyzer.combined_ligand_conc_df["Select"].dtype == bool
