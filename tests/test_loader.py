"""KineticsAnalyzer.load picks the reader from the detected file format."""

import shutil
from pathlib import Path

import pytest

from pykingenie import (
    BiacoreExperiment,
    CarterraExperiment,
    GatorExperiment,
    KineticsAnalyzer,
    KinGenieCsv,
    KinGenieCsvSolution,
    OctetExperiment,
    SprTextExperiment,
    SurfaceCsvExperiment,
)

DATA = Path(__file__).resolve().parents[1] / "test_files"
FRD = sorted(DATA.glob("230309_0*.frd"))
FMF = DATA / "230309_ExpMethod.fmf"
SPR = DATA / "spr_exports"


def load(files, **kwargs):
    analyzer = KineticsAnalyzer()
    experiment = analyzer.load(files, **kwargs)
    assert analyzer.experiments[experiment.name] is experiment
    return experiment


def test_octet_run_with_plate_layout():
    experiment = load(FRD + [FMF])
    assert isinstance(experiment, OctetExperiment)
    assert len(experiment.sensor_names) == 8
    assert experiment.sample_row is not None
    assert experiment.name.startswith("5th interaction")


def test_octet_run_without_plate_layout():
    experiment = load([str(file) for file in FRD])
    assert isinstance(experiment, OctetExperiment) and experiment.sample_row is None


@pytest.mark.parametrize(
    "file,cls",
    [
        ("SPR_sensograms_with_Streptavidin_1.blr", BiacoreExperiment),
        ("multi_cycle_kingenie.csv", KinGenieCsv),
        ("kingenie_solution.csv", KinGenieCsvSolution),
    ],
)
def test_single_files_without_options(file, cls):
    experiment = load(DATA / file)
    assert isinstance(experiment, cls) and experiment.name == Path(file).stem


def test_gator_zip(tmp_path):
    archive = shutil.copy(DATA / "gator_example_folder.zip", tmp_path)
    experiment = load(archive)
    assert isinstance(experiment, GatorExperiment) and experiment.traces_loaded
    assert experiment.name == "gator_example_folder"


def test_gator_files(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    files = [shutil.copy(DATA / name, folder)
             for name in ("Assay_1_Channel1.csv", "ExperimentStep.ini", "Setting.ini")]
    experiment = load(files)
    assert isinstance(experiment, GatorExperiment) and experiment.traces_loaded
    assert experiment.name == "run"


def test_steps_and_traces_in_any_order():
    experiment = load([DATA / "surface_csv_traces.csv", DATA / "surface_csv_steps.txt"],
                      sample_id="analyte")
    assert isinstance(experiment, SurfaceCsvExperiment)
    assert (experiment.ligand_conc_df["SampleID"] == "analyte").all()


def test_spr_text_receives_its_options():
    experiment = load(SPR / "biacore_cycle_sample.tsv", association_end=180,
                      concentrations=[0.2, 0.4], name="t200")
    assert isinstance(experiment, SprTextExperiment) and experiment.name == "t200"
    assert experiment.exp_info["association_end_seconds"] == 180


def test_carterra_receives_its_options():
    experiment = load(SPR / "carterra_rows.csv", association_start=120, association_end=420,
                      top_concentration=1, dilution_factor=2)
    assert isinstance(experiment, CarterraExperiment)
    assert experiment.ligand_conc_df["Concentration_micromolar"].iloc[-1] == 1


def test_automatic_names_are_numbered_and_explicit_names_replace():
    analyzer = KineticsAnalyzer()
    first = analyzer.load(DATA / "multi_cycle_kingenie.csv")
    second = analyzer.load(DATA / "multi_cycle_kingenie.csv")
    third = analyzer.load(DATA / "multi_cycle_kingenie.csv")
    assert [first.name, second.name, third.name] == [
        "multi_cycle_kingenie", "multi_cycle_kingenie (2)", "multi_cycle_kingenie (3)"]
    replacement = analyzer.load(DATA / "single_cycle_kingenie.csv", name="multi_cycle_kingenie")
    assert analyzer.experiments["multi_cycle_kingenie"] is replacement
    assert len(analyzer.experiment_names) == 3


@pytest.mark.parametrize(
    "files,message",
    [
        ([], "No files"),
        ([DATA / "two_columns_solution.csv"], "Unrecognized input.*two_columns_solution.csv: unknown"),
        ([FMF], "do not form one experiment.*octet_fmf"),
        ([DATA / "multi_cycle_kingenie.csv", DATA / "single_cycle_kingenie.csv"],
         "do not form one experiment"),
        ([FRD[0], DATA / "SPR_sensograms_with_Streptavidin_1.blr"], "do not form one experiment"),
        ([DATA / "surface_csv_steps.txt"], "do not form one experiment"),
        ([DATA / "ExperimentStep.ini", DATA / "Setting.ini"], "needs one ExperimentStep.ini"),
        ([DATA / "ExperimentStep.ini", DATA / "ExperimentStep.ini", DATA / "Setting.ini",
          DATA / "Assay_1_Channel1.csv"], "needs one ExperimentStep.ini"),
    ],
)
def test_files_that_do_not_make_one_experiment(files, message):
    with pytest.raises(ValueError, match=message):
        KineticsAnalyzer().load(files)


def test_options_for_readers_without_options_are_rejected():
    analyzer = KineticsAnalyzer()
    with pytest.raises(TypeError, match="biacore_blr reader takes no options; got association_end"):
        analyzer.load(DATA / "SPR_sensograms_with_Streptavidin_1.blr", association_end=10)
    assert analyzer.experiment_names == []


def test_unknown_reader_options_are_reported_by_the_reader():
    with pytest.raises(TypeError, match="unexpected keyword argument 'association_stop'"):
        KineticsAnalyzer().load(SPR / "paired_run.csv", association_stop=10)
