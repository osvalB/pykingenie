"""Delimiter changes must preserve imported measurements and experiment metadata."""

import bz2
import csv
import gzip
import io
import lzma
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from pykingenie import (
    CarterraExperiment,
    GatorExperiment,
    KinGenieCsv,
    KinGenieCsvSolution,
    SprTextExperiment,
    SurfaceCsvExperiment,
    detect_file_format,
)
from pykingenie.surface_csv import read_steps_file, read_traces_file
from pykingenie.utils.text_io import read_delimited_table

DATA = Path(__file__).resolve().parents[1] / "test_files"
DELIMITERS = [",", ";", "\t"]


def rewrite_table(source, target, source_delimiter, delimiter, header_row=0):
    """Re-encode fields with quotes and separator whitespace, without changing values."""
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    rows = csv.reader(lines[header_row:], delimiter=source_delimiter)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        for line in lines[:header_row]:
            stream.write(line + "\n")
        for row in rows:
            quoted = ['"' + field.replace('"', '""') + '"' for field in row]
            stream.write((delimiter + " ").join(quoted) + "\n")
    return target


def assert_surface_equal(actual, expected):
    assert actual.traces_loaded and expected.traces_loaded
    assert actual.sensor_names == expected.sensor_names
    assert actual.no_steps == expected.no_steps
    pd.testing.assert_frame_equal(actual.df_steps, expected.df_steps)
    pd.testing.assert_frame_equal(actual.ligand_conc_df, expected.ligand_conc_df)
    for observed, reference in [(actual.xs, expected.xs), (actual.ys, expected.ys)]:
        assert len(observed) == len(reference)
        for sensor, expected_sensor in zip(observed, reference):
            assert len(sensor) == len(expected_sensor)
            for phase, expected_phase in zip(sensor, expected_sensor):
                np.testing.assert_array_equal(phase, expected_phase)


@pytest.mark.parametrize("delimiter", DELIMITERS)
@pytest.mark.parametrize(
    "filename,source_delimiter,decimal,count,kind",
    [
        ("paired_run.csv", ",", ".", 2, "paired"),
        ("biacore_cycle_fc.tsv", "\t", ",", 2, "paired"),
        ("biacore_cycle_sample.tsv", "\t", ".", 2, "paired"),
        ("carterra_rows.csv", ",", ".", 3, "carterra"),
        ("carterra_rows.csv", ",", ".", 3, "single_cycle"),
    ],
)
def test_spr_layouts_preserve_values_across_delimiters(
    tmp_path, filename, source_delimiter, decimal, count, kind, delimiter
):
    source = DATA / "spr_exports" / filename
    converted = rewrite_table(
        source, tmp_path / "export.data", source_delimiter, delimiter
    )
    experiments = []
    for path in [source, converted]:
        exp = (
            SprTextExperiment("test")
            if kind == "paired"
            else CarterraExperiment("test")
        )
        reader = exp.read_file if kind == "paired" else exp.read_csv
        settings = {
            "decimal": decimal,
            "association_end": 300,
            "concentrations": [0.1] * count,
        }
        if kind == "single_cycle":
            settings.update(
                association_start=120,
                association_end=420,
                single_cycle=True,
                cycle_duration=1200,
            )
        reader(path, **settings)
        experiments.append(exp)
    assert_surface_equal(experiments[1], experiments[0])
    assert detect_file_format(converted) == (
        "spr_text" if kind == "paired" else "carterra_csv"
    )


@pytest.mark.parametrize("delimiter", DELIMITERS)
@pytest.mark.parametrize(
    "filename", ["multi_cycle_kingenie.csv", "single_cycle_kingenie.csv"]
)
def test_kingenie_surface_delimiters(tmp_path, filename, delimiter):
    source = DATA / filename
    converted = rewrite_table(source, tmp_path / "simulation.csv", ",", delimiter)
    expected, actual = KinGenieCsv("test"), KinGenieCsv("test")
    expected.read_csv(source)
    actual.read_csv(converted)
    assert_surface_equal(actual, expected)
    assert detect_file_format(converted) == "kingenie_surface"


@pytest.mark.parametrize("delimiter", DELIMITERS)
@pytest.mark.parametrize(
    "filename",
    ["kingenie_solution.csv", "two_columns_solution.csv", "three_columns_solution.csv"],
)
def test_kingenie_solution_delimiters(tmp_path, filename, delimiter):
    source = DATA / filename
    converted = rewrite_table(source, tmp_path / "solution.csv", ",", delimiter)
    expected, actual = KinGenieCsvSolution("test"), KinGenieCsvSolution("test")
    expected.read_csv(source)
    actual.read_csv(converted)
    assert actual.traces_loaded and actual.traces_names == expected.traces_names
    pd.testing.assert_frame_equal(actual.conc_df, expected.conc_df)
    for observed, reference in [(actual.xs, expected.xs), (actual.ys, expected.ys)]:
        assert len(observed) == len(reference)
        for trace, expected_trace in zip(observed, reference):
            np.testing.assert_array_equal(trace, expected_trace)


@pytest.mark.parametrize("steps_delimiter", DELIMITERS)
@pytest.mark.parametrize("traces_delimiter", DELIMITERS)
def test_generic_surface_files_detect_delimiters_independently(
    tmp_path, steps_delimiter, traces_delimiter
):
    steps = rewrite_table(
        DATA / "surface_csv_steps.txt", tmp_path / "steps.txt", "\t", steps_delimiter
    )
    traces = rewrite_table(
        DATA / "surface_csv_traces.csv",
        tmp_path / "traces.csv",
        ",",
        traces_delimiter,
        header_row=3,
    )
    expected, actual = SurfaceCsvExperiment("test"), SurfaceCsvExperiment("test")
    settings = {"concentrations": [0.025, 0.05, 0.1, 0.2, 0.4]}
    expected.read_files(
        DATA / "surface_csv_steps.txt", DATA / "surface_csv_traces.csv", **settings
    )
    actual.read_files(steps, traces, **settings)
    assert_surface_equal(actual, expected)
    assert detect_file_format(steps) == "surface_csv_steps"
    assert detect_file_format(traces) == "surface_csv_traces"


@pytest.mark.parametrize("delimiter", DELIMITERS)
def test_gator_delimiters_preserve_every_phase(tmp_path, delimiter):
    source = DATA / "Assay_1_Channel1.csv"
    converted = rewrite_table(source, tmp_path / source.name, ",", delimiter)
    experiments = []
    for path in [source, converted]:
        exp = GatorExperiment("test")
        exp.read_all_gator_data(
            [str(DATA / "ExperimentStep.ini"), str(DATA / "Setting.ini"), str(path)]
        )
        experiments.append(exp)
    assert_surface_equal(experiments[1], experiments[0])
    assert any(len(phase) for sensor in experiments[1].ys for phase in sensor)
    assert detect_file_format(converted) == "gator_csv"


@pytest.mark.parametrize("delimiter", DELIMITERS)
def test_generic_traces_and_steps_preserve_quoted_names(tmp_path, delimiter):
    name = 'sensor, A; "quoted"\tlabel'
    traces = tmp_path / "traces.csv"
    with traces.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter, quoting=csv.QUOTE_ALL)
        writer.writerow(["X Values", name])
        writer.writerows([[0, 1.5], [1, 2.5]])
    df = read_traces_file(traces)
    assert list(df.columns) == ["X Values", name]
    np.testing.assert_array_equal(df.to_numpy(), [[0, 1.5], [1, 2.5]])

    steps = tmp_path / "steps.csv"
    with steps.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter, quoting=csv.QUOTE_ALL)
        writer.writerow(["Step Time", "Step Type Name", "Step Name"])
        writer.writerow([60, "Association", name])
    assert read_steps_file(steps)["Name"].tolist() == [name]


@pytest.mark.parametrize("compression", ["gz", "bz2", "xz", "zip"])
def test_compressed_tables_keep_delimiter_detection(tmp_path, compression):
    content = '\ufeff\n\n"Time"; "Signal"\n"0"; "1.25"\n"1"; "2.5"\n'.encode()
    path = tmp_path / f"table.csv.{compression}"
    compressors = {"gz": gzip.compress, "bz2": bz2.compress, "xz": lzma.compress}
    if compression == "zip":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("table.csv", content)
    else:
        path.write_bytes(compressors[compression](content))
    pd.testing.assert_frame_equal(
        read_delimited_table(path),
        pd.DataFrame({"Time": [0, 1], "Signal": [1.25, 2.5]}),
    )


def test_existing_stream_position_is_preserved_for_table_read():
    stream = io.StringIO('ignored preamble\n"Time"; "Signal"\n0;1.25\n1;2.5\n')
    stream.readline()
    actual = read_delimited_table(stream)
    pd.testing.assert_frame_equal(
        actual, pd.DataFrame({"Time": [0, 1], "Signal": [1.25, 2.5]})
    )
    assert not stream.closed


@pytest.mark.parametrize("delimiter", [";", "\t"])
def test_delimiter_in_unquoted_optional_header_is_not_the_table_separator(
    tmp_path, delimiter
):
    path = tmp_path / "steps.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(["Step Time", "Step Type Name", "Note, detail"])
        writer.writerow([60, "Association", "a"])
        writer.writerow([60, "Dissociation", "b"])
    steps = read_steps_file(path)
    assert steps["Type"].tolist() == ["ASSOC", "DISASSOC"]
    assert steps["Note, detail"].tolist() == ["a", "b"]


@pytest.mark.parametrize("delimiter", DELIMITERS)
def test_optional_trailing_step_fields_can_be_missing(tmp_path, delimiter):
    path = tmp_path / "steps.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(["Step Time", "Step Type Name", "Sample ID"])
        writer.writerow([60, "Baseline"])
        writer.writerow([90, "Association", "analyte"])
    steps = read_steps_file(path)
    assert steps["Time"].tolist() == [60.0, 90.0]
    assert pd.isna(steps["SampleID"].iloc[0])
    assert steps["SampleID"].iloc[1] == "analyte"
