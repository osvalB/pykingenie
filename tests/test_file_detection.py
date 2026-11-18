"""Real-file routing checks and ambiguity regressions for format detection."""

import gzip
from pathlib import Path
import shutil
import zipfile

import pytest

from pykingenie import detect_file_format, guess_experiment_type
from pykingenie.utils.compound_file import SIGNATURE

DATA = Path(__file__).resolve().parents[1] / "test_files"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("spr_exports/paired_run.csv", "spr_text"),
        ("spr_exports/biacore_cycle_fc.tsv", "spr_text"),
        ("spr_exports/biacore_cycle_sample.tsv", "spr_text"),
        ("spr_exports/carterra_rows.csv", "carterra_csv"),
        ("230309_001.frd", "octet_frd"),
        ("SPR_sensograms_with_Streptavidin_1.blr", "biacore_blr"),
        ("Assay_1_Channel1.csv", "gator_csv"),
        ("ExperimentStep.ini", "gator_steps"),
        ("Setting.ini", "gator_settings"),
        ("gator_example_folder.zip", "gator_zip"),
        ("multi_cycle_kingenie.csv", "kingenie_surface"),
        ("single_cycle_kingenie.csv", "kingenie_surface"),
        ("kingenie_solution.csv", "kingenie_solution"),
        ("three_columns_solution.csv", "kingenie_solution"),
        ("surface_csv_steps.txt", "surface_csv_steps"),
        ("surface_csv_traces.csv", "surface_csv_traces"),
        ("surface_csv_mck_steps.txt", "surface_csv_steps"),
        ("surface_csv_mck_traces.csv", "surface_csv_traces"),
        # A two-column trace alone cannot establish surface versus solution.
        ("two_columns_solution.csv", "unknown"),
        ("one_column_solution.csv", "unknown"),
        ("empty_file.frd", "unknown"),
    ],
)
def test_real_input_formats(filename, expected):
    assert detect_file_format(DATA / filename) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("paired_run.csv", "spr_text"),
        ("biacore_cycle_fc.tsv", "spr_text"),
        ("biacore_cycle_sample.tsv", "spr_text"),
        ("carterra_rows.csv", "carterra_csv"),
    ],
)
def test_new_formats_independent_of_extension(tmp_path, filename, expected):
    file = tmp_path / "unlabelled.data"
    shutil.copyfile(DATA / "spr_exports" / filename, file)
    assert detect_file_format(file) == expected
    assert guess_experiment_type(file) == "surface"
    compressed = tmp_path / "compressed.data"
    compressed.write_bytes(gzip.compress(file.read_bytes()))
    assert detect_file_format(compressed) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ('Report metadata\n"a, quoted_X","a, quoted_Y"\n0,1', "spr_text"),
        ("Export metadata\na_Y;a_X;b_X;b_Y;;\n1;0;0;2;;", "spr_text"),
        ("Label;X1;Y1;X2;Y2\na;0;1;1;2", "carterra_csv"),
        ("Time,Signal,Smax,Analyte_concentration_micromolar_constant", "kingenie_surface"),
        ("Time,Signal,Ligand_concentration_micromolar", "kingenie_solution"),
        ("Duration,Step Type\n60,Baseline", "surface_csv_steps"),
        ("x values,A1\n0,1", "surface_csv_traces"),
        ("Label,X1,Y1,X3,Y3", "unknown"),
        ("Label,X1,Y1,X2", "unknown"),
        ("a_X,b_Y", "unknown"),
        ("a_X,a_X,a_Y", "unknown"),
        (" _X, _Y", "unknown"),
        ("Time,Signal\n0,1", "unknown"),
        ("a_X,Response\n0,1", "unknown"),
        ("generic,random,csv\n0,1,2", "unknown"),
        ("", "unknown"),
    ],
)
def test_header_layouts_and_ambiguous_inputs(tmp_path, text, expected):
    file = tmp_path / "report.CSV"
    file.write_text(text, encoding="utf-8-sig")
    assert detect_file_format(file) == expected


def test_windows_encoding_and_header_preamble(tmp_path):
    file = tmp_path / "report.txt"
    file.write_bytes("Legacy instrument export\nµ sample_X\tµ sample_Y\n0\t1".encode("cp1252"))
    assert detect_file_format(file) == "spr_text"


def test_detector_does_not_parse_the_measurements(tmp_path):
    file = tmp_path / "report.csv"
    file.write_text("a_X,a_Y\n0,NaN\n1,not-a-number")
    assert detect_file_format(file) == "spr_text"


def test_gzip_solution_and_legacy_api(tmp_path):
    file = tmp_path / "solution.CSV.GZ"
    file.write_bytes(gzip.compress((DATA / "kingenie_solution.csv").read_bytes()))
    assert detect_file_format(file) == "kingenie_solution"
    assert guess_experiment_type(file) == "solution"
    # Metadata or an unrelated CSV earlier in a batch must not hide solution data.
    unknown = tmp_path / "notes.csv"
    unknown.write_text("Notes,comment\n1,2")
    assert guess_experiment_type([unknown, file]) == "solution"
    assert guess_experiment_type([file, "run.BLR"]) == "surface"
    assert guess_experiment_type([unknown, DATA / "Assay_1_Channel1.csv"]) == "surface"


def test_uppercase_solution_with_metadata(tmp_path):
    file = tmp_path / "solution.CSV"
    file.write_text(
        "Instrument export\n\n" + (DATA / "kingenie_solution.csv").read_text(), encoding="utf-8-sig"
    )
    assert guess_experiment_type(str(file)) == "solution"


def test_legacy_unknown_and_missing_inputs(tmp_path):
    assert guess_experiment_type(tmp_path / "missing.csv") == "surface"
    assert guess_experiment_type([]) == "surface"
    assert guess_experiment_type("RUN.FRD") == "surface"
    assert guess_experiment_type("EXPERIMENTSTEP.INI") == "surface"
    with pytest.raises(FileNotFoundError):
        detect_file_format(tmp_path / "missing.csv")


def test_native_biacore_renamed(tmp_path):
    file = tmp_path / "native.data"
    shutil.copyfile(DATA / "SPR_sensograms_with_Streptavidin_1.blr", file)
    assert detect_file_format(file) == "biacore_blr"


def test_non_biacore_compound_container(tmp_path, monkeypatch):
    file = tmp_path / "spreadsheet.blr"
    file.write_bytes(SIGNATURE + b"\x00" * 504)
    assert detect_file_format(file) == "unknown"
    # The OLE container signature alone does not identify a Biacore result file.
    monkeypatch.setattr("pykingenie.file_detection.read_compound_file", lambda _: {"Workbook": b""})
    assert detect_file_format(file) == "unknown"


@pytest.mark.parametrize(
    "content",
    [
        b"PK\x03\x04broken",
        b"\x1f\x8b",
        b"\x1f\x8b\x00" + b"\x00" * 50,
        b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff" + b"\xff" * 8,
    ],
)
def test_corrupt_containers_are_unknown(tmp_path, content):
    file = tmp_path / "corrupt.data"
    file.write_bytes(content)
    assert detect_file_format(file) == "unknown"


@pytest.mark.parametrize(
    "names",
    [
        ["[Content_Types].xml", "xl/workbook.xml"],
        ["Setting.ini", "ExperimentStep.ini"],
    ],
)
def test_xlsx_and_incomplete_gator_archives_are_unknown(tmp_path, names):
    file = tmp_path / "archive.zip"
    with zipfile.ZipFile(file, "w") as archive:
        for name in names:
            archive.writestr(name, "metadata")
    assert detect_file_format(file) == "unknown"
