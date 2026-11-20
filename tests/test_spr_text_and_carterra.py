"""Regression checks against public exports and small explicit malformed files."""

import csv
import gzip
from pathlib import Path

import numpy as np
import pytest

from pykingenie import CarterraExperiment, KineticsAnalyzer, SprTextExperiment

FIXTURES = Path(__file__).resolve().parents[1] / "test_files" / "spr_exports"
TIMES = [-1, 0, 1, 2, 3, 4]


def paired_file(tmp_path, names=("a", "b"), delimiter=",", decimal=".", encoding="utf-8-sig"):
    path = tmp_path / "curves.txt"
    with path.open("w", encoding=encoding, newline="") as stream:
        stream.write("Exported sensorgrams\n")
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow([f"{name}_{axis}" for name in names for axis in ("X", "Y")])
        for i, t in enumerate(TIMES):
            writer.writerow(
                [
                    str(v).replace(".", decimal)
                    for j in range(len(names))
                    for v in (t + j * 0.1, i + j * 0.25)
                ]
            )
    return path


def carterra_file(tmp_path, labels=("[Conc. 1] [NotFitted]", "[Conc. 2] [NotFitted]")):
    path = tmp_path / "carterra.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Label"] + [f"{axis}{i}" for i in range(1, 7) for axis in ("X", "Y")])
        for j, label in enumerate(labels):
            writer.writerow(
                [label] + [v for i, t in enumerate(TIMES) for v in (t + 0.1 * j, i + j * 0.25)]
            )
    return path


def load_text(path, **kwargs):
    exp = SprTextExperiment("test")
    exp.read_file(path, association_end=2, concentrations=[0.1, 0.2], **kwargs)
    return exp


@pytest.mark.parametrize(
    "filename,decimal,count,first_response",
    [
        ("paired_run.csv", ".", 2, -0.011209273),
        ("biacore_cycle_fc.tsv", ",", 2, 232.221),
        ("biacore_cycle_sample.tsv", ".", 2, 0.213257),
        ("carterra_rows.csv", ".", 3, -0.2542114),
    ],
)
def test_public_sample_regressions(filename, decimal, count, first_response):
    exp = CarterraExperiment() if filename.startswith("carterra") else SprTextExperiment()
    reader = exp.read_csv if isinstance(exp, CarterraExperiment) else exp.read_file
    with pytest.warns(UserWarning, match="concentrations"):
        reader(FIXTURES / filename, decimal=decimal, association_end=300)
    assert exp.no_sensors == count
    assert exp.no_steps == 3 and exp.traces_loaded
    assert exp.ligand_conc_df.Concentration_micromolar.isna().all()
    assert np.concatenate(exp.ys[0])[0] == pytest.approx(first_response)
    assert not any("fitted curve" in name.lower() for name in exp.sensor_names)
    assert exp.df_steps.Start.is_monotonic_increasing
    # Each curve retains its original time grid, even when it has no baseline.
    if filename != "biacore_cycle_fc.tsv":
        assert np.concatenate(exp.xs[0])[0] != np.concatenate(exp.xs[1])[0]
    if filename == "carterra_rows.csv":
        assert all("NotFitted" in name for name in exp.sensor_names)
        assert len(np.concatenate(exp.xs[0])) == 534
    if filename == "paired_run.csv":
        exp.read_file(FIXTURES / filename, association_end=300, concentration_unit="uM")
        np.testing.assert_allclose(exp.ligand_conc_df.Concentration_micromolar, [0.0585, 0.117])


@pytest.mark.parametrize("delimiter,decimal", [("\t", ","), (",", "."), (";", ",")])
def test_delimiters_values_and_independent_times(tmp_path, delimiter, decimal):
    exp = load_text(
        paired_file(tmp_path, ("one, quoted", "two"), delimiter, decimal), decimal=decimal
    )
    assert exp.sensor_names == ["one, quoted", "two"]
    np.testing.assert_array_equal(np.concatenate(exp.xs[0]), TIMES)
    np.testing.assert_allclose(np.concatenate(exp.xs[1]), np.array(TIMES) + 0.1)
    np.testing.assert_allclose(np.concatenate(exp.ys[1]), np.arange(6) + 0.25)
    assert exp.sensor_names_unique == ["test one, quoted", "test two"]
    assert exp.exp_info["association_end_seconds"] == 2


def test_gzip_legacy_encoding_and_ragged_pairs(tmp_path):
    p = paired_file(tmp_path, ("µ sample", "b"), encoding="cp1252")
    gz = tmp_path / "curves.txt.gz"
    with gzip.open(gz, "wb") as stream:
        stream.write(p.read_bytes() + b"5,6,,\n,,,\n")
    exp = load_text(gz, encoding="cp1252")
    assert exp.sensor_names[0] == "µ sample"
    assert len(np.concatenate(exp.xs[0])) == 7
    assert len(np.concatenate(exp.xs[1])) == 6


def test_units_and_explicit_metadata(tmp_path):
    p = paired_file(tmp_path, ("a conc=1e2 nM", "b conc=.2uM"))
    exp = SprTextExperiment()
    exp.read_file(
        p,
        association_end=120,
        time_unit="min",
        sample_id={"b conc=.2uM": "B", "a conc=1e2 nM": "A"},
    )
    np.testing.assert_allclose(exp.ligand_conc_df.Concentration_micromolar, [0.1, 0.2])
    np.testing.assert_array_equal(exp.xs[0][1], [0, 60])
    assert exp.ligand_conc_df.SampleID.tolist() == ["A", "B"]
    exp.read_file(
        p,
        association_end=2,
        concentrations={"b conc=.2uM": 200, "a conc=1e2 nM": 100},
        concentration_unit="nM",
        sample_id=["A", "B"],
    )
    np.testing.assert_allclose(exp.ligand_conc_df.Concentration_micromolar, [0.1, 0.2])


def test_uncertain_labels_are_not_molar_values(tmp_path):
    p = paired_file(tmp_path, ("Cycle: 19 sample 25 µg", "[Conc. 13] [NotFitted]"))
    exp = SprTextExperiment()
    with pytest.warns(UserWarning):
        exp.read_file(p, association_end=2, concentration_unit="uM")
    assert exp.ligand_conc_df.Concentration_micromolar.isna().all()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"decimal": ":"}, "decimal"),
        ({"time_unit": "hours"}, "time_unit"),
        ({"concentration_unit": "mg/mL"}, "concentration unit"),
        ({"association_start": 2}, "boundaries"),
        ({"association_start": np.nan}, "boundaries"),
        ({"association_start": 3}, "boundaries"),
        ({"sample_id": ["", "B"]}, "non-empty strings"),
        ({"sample_id": [1, 2]}, "non-empty strings"),
        ({"sample_id": ["A"]}, "Expected 2"),
        ({"sample_id": {"wrong": "A"}}, "exactly"),
        ({"concentrations": [1]}, "Expected 2"),
        ({"concentrations": {"wrong": 1}}, "exactly"),
        ({"concentrations": "1"}, "sequence"),
        ({"concentrations": [[1], [2]]}, "scalar"),
        ({"concentrations": [-1, 2]}, "finite and non-negative"),
        ({"concentrations": [np.nan, 2]}, "finite and non-negative"),
        ({"concentrations": [np.inf, 2]}, "finite and non-negative"),
        ({"association_start": 1.5}, "two points"),
    ],
)
def test_metadata_validation(tmp_path, kwargs, message):
    settings = dict(association_end=2, concentrations=[0.1, 0.2])
    settings.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SprTextExperiment().read_file(paired_file(tmp_path), **settings)


@pytest.mark.parametrize(
    "content,message",
    [
        ("", "header"),
        ("Time,signal\n0,1", "header"),
        ("_X,_Y\n0,1", "Empty or duplicate"),
        ("a_X,a_X\n0,1", "Empty or duplicate"),
        ("a_X,b_Y\n0,1", "exactly one"),
        ("a fitted curve_X,a fitted curve_Y\n0,1", "no measured"),
        ("a_X,a_Y\n", "at least two"),
        ("a_X,a_Y\n0,1\n1,", "Incomplete"),
        ("a_X,a_Y\n,1\n1,1", "Incomplete"),
        ("a_X,a_Y\n0,bad\n1,1", "Non-numeric"),
        ("a_X,a_Y\n0,inf\n1,1", "Non-finite"),
        ("a_X,a_Y\n0,1\n0,2", "strictly"),
        ("a_X,a_Y\n1,1\n0,2", "strictly"),
        ("a_X,a_Y\n0,1,extra\n1,2", "more fields"),
    ],
)
def test_paired_file_validation(tmp_path, content, message):
    p = tmp_path / "invalid.txt"
    p.write_text(content)
    with pytest.raises(ValueError, match=message):
        SprTextExperiment().read_file(p, association_end=2)


@pytest.mark.parametrize(
    "content,message",
    [
        ("Label,X1,X2\n", "Carterra headers"),
        ("Label,X1,Y1,X2\n", "Carterra headers"),
        ("Label,X1,Y1\n", "no measured"),
        ("Label,X1,Y1\n,0,1", "non-empty and unique"),
        ("Label,X1,Y1,X2,Y2\na,0,1,1,2\na,0,1,1,2", "non-empty and unique"),
    ],
)
def test_carterra_file_validation(tmp_path, content, message):
    p = tmp_path / "invalid.csv"
    p.write_text(content)
    with pytest.raises(ValueError, match=message):
        CarterraExperiment().read_csv(p, association_end=2)


def test_carterra_metadata_and_reread_is_transactional(tmp_path):
    p = carterra_file(tmp_path)
    exp = CarterraExperiment("rows")
    exp.read_csv(p, association_end=2, concentrations=[0.1, 0.2])
    previous_xs, previous_df = exp.xs, exp.ligand_conc_df
    with pytest.raises(ValueError, match="two points"):
        exp.read_csv(p, association_end=100, concentrations=[0.1, 0.2])
    assert exp.xs is previous_xs and exp.ligand_conc_df is previous_df
    np.testing.assert_allclose(exp.xs[1][1], [0.1, 1.1])


@pytest.mark.parametrize("kind", ["paired", "carterra"])
def test_analyzer_integration(tmp_path, kind):
    if kind == "paired":
        exp = load_text(paired_file(tmp_path))
    else:
        exp = CarterraExperiment("test")
        exp.read_csv(carterra_file(tmp_path), association_end=2, concentrations=[0.1, 0.2])
    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(exp, "test")
    analyzer.merge_ligand_conc_df()
    analyzer.init_fittings()
    analyzer.generate_fittings(analyzer.combined_ligand_conc_df)
    fitter = next(iter(analyzer.fittings.values()))
    np.testing.assert_allclose(fitter.lig_conc_lst, [0.1, 0.2])
    np.testing.assert_array_equal(fitter.assoc_lst[0], exp.ys[0][1])
    np.testing.assert_allclose(fitter.time_disso_lst[1], [2, 3, 4])


def test_trailing_header_fields_and_missing_trailing_pairs(tmp_path):
    p = paired_file(tmp_path)
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    lines[1] += ",,"
    p.write_text("\n".join(lines) + "\n5,6\n", encoding="utf-8-sig")
    exp = load_text(p)
    assert len(np.concatenate(exp.xs[0])) == 7
    assert len(np.concatenate(exp.xs[1])) == 6


def test_empty_baseline_and_phase_boundary_membership(tmp_path):
    p = paired_file(tmp_path)
    exp = SprTextExperiment()
    exp.read_file(p, association_start=-2, association_end=2, concentrations=[1, 2])
    assert exp.xs[0][0].size == 0
    assert exp.df_steps.Start.tolist() == [-2, -2, 2]
    np.testing.assert_array_equal(exp.xs[0][1], [-1, 0, 1])
    np.testing.assert_array_equal(exp.xs[0][2], [2, 3, 4])
