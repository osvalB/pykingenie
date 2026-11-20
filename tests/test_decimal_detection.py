"""SPR readers find the decimal mark themselves unless one is given."""

import csv

import numpy as np
import pytest

from pykingenie import CarterraExperiment, SprTextExperiment
from pykingenie.utils.sensorgram_text import detect_decimal

TIME = np.arange(0, 50, 0.5)


def number(value, decimal):
    return f"{value:.4f}".replace(".", decimal)


def write_paired(path, delimiter, decimal):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["a_X", "a_Y", "b_X", "b_Y"])
        for t in TIME:
            writer.writerow([number(t, decimal), number(0.25 * t, decimal)] * 2)
    return path


def write_carterra(path, delimiter, decimal):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(["Label"] + [f"{axis}{i + 1}" for i in range(len(TIME)) for axis in "XY"])
        for label in ("row 1", "row 2"):
            writer.writerow([label] + [number(v, decimal) for t in TIME for v in (t, 0.25 * t)])
    return path


@pytest.mark.parametrize(
    "cells,expected",
    [
        (["1.5", "2", "-3.25e-2"], "."),
        (["1,5", "2", "-3,25E-2"], ","),
        ([" 1,5 ", ",5"], ","),
        (["1", "2", "-3"], "."),
        ([], "."),
    ],
)
def test_decimal_mark_is_detected(cells, expected):
    assert detect_decimal(cells) == expected


def test_both_marks_are_ambiguous():
    with pytest.raises(ValueError, match="both '.' and ','"):
        detect_decimal(["1.5", "2,5"])


def test_given_decimal_is_kept_and_checked():
    assert detect_decimal(["1,5"], ".") == "."
    with pytest.raises(ValueError, match="decimal must be"):
        detect_decimal(["1.5"], ";")


@pytest.mark.parametrize("delimiter", ["\t", ";", ","])
@pytest.mark.parametrize("decimal", [".", ","])
def test_paired_columns_detect_their_decimal_mark(tmp_path, delimiter, decimal):
    experiment = SprTextExperiment()
    experiment.read_file(write_paired(tmp_path / "p.csv", delimiter, decimal),
                         association_start=10, association_end=30, concentrations=[1, 2])
    assert experiment.exp_info["decimal_in_file"] == decimal
    np.testing.assert_allclose(np.concatenate(experiment.xs[0]), TIME)
    np.testing.assert_allclose(np.concatenate(experiment.ys[0]), 0.25 * TIME)


@pytest.mark.parametrize("delimiter,decimal", [("\t", ","), (";", ","), (",", ".")])
def test_carterra_rows_detect_their_decimal_mark(tmp_path, delimiter, decimal):
    experiment = CarterraExperiment()
    experiment.read_csv(write_carterra(tmp_path / "c.csv", delimiter, decimal),
                        association_start=10, association_end=30, concentrations=[1, 2])
    assert experiment.exp_info["decimal_in_file"] == decimal
    np.testing.assert_allclose(np.concatenate(experiment.ys[1]), 0.25 * TIME)


def test_wrong_explicit_decimal_still_fails(tmp_path):
    with pytest.raises(ValueError, match="Non-numeric measurement"):
        SprTextExperiment().read_file(write_paired(tmp_path / "p.csv", "\t", ","),
                                      association_end=30, decimal=".")


def test_invalid_decimal_argument_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="decimal must be"):
        CarterraExperiment().read_csv(write_carterra(tmp_path / "c.csv", ";", ","),
                                      association_end=30, decimal=";")
