"""Physical-clock and occupied-state checks for the SPR teaching analyses."""

import hashlib
import csv
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pykingenie import CarterraExperiment, SprTextExperiment

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spr_notebook_fitting", ROOT / "notebooks" / "spr_fitting.py"
)
fitting = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fitting
SPEC.loader.exec_module(fitting)


def test_masked_one_site_fit_keeps_true_clock_and_occupied_state():
    """Compare against independent integration, including an unobserved switch."""
    times = np.array([15.0, 30.0, 45.0, 63.0, 100.0])
    trace = fitting.Trace("occupied", times, times * 0, 0, 20, 0.4, np.arange(5))
    meta = dict(association_start=10, association_end=60, fit_end=100)
    params = dict(Kd=0.2, k_off=0.01, Rmax_1=100)
    kon = params["k_off"] / params["Kd"]

    def ode(concentration):
        return lambda t, y: kon * concentration * (100 - y) - 0.01 * y

    assoc = solve_ivp(
        ode(0.4), (10, 60), [20], rtol=1e-11, atol=1e-12, dense_output=True
    )
    disso = solve_ivp(
        ode(0), (60, 100), assoc.y[:, -1], rtol=1e-11, atol=1e-12, dense_output=True
    )
    expected = np.r_[assoc.sol(times[:3])[0], disso.sol(times[3:])[0]]
    actual = fitting.predict(trace, meta, params, 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_recovers_one_site_parameters_with_prebound_signal():
    """A non-regenerative fit must retain measured initial binding."""
    concentrations = [0.1, 0.4, 1.0]
    capacities, initial = [100, 110, 120], [10, 25, 50]
    times = np.arange(-20.0, 281.0)
    rng = np.random.default_rng(821)
    responses = []
    for c, rmax, r0 in zip(concentrations, capacities, initial):
        equilibrium = rmax * c / (0.2 + c)
        response = np.full_like(times, r0)
        assoc = (times >= 0) & (times < 80)
        response[assoc] = equilibrium + (r0 - equilibrium) * np.exp(
            -0.008 * (1 + c / 0.2) * times[assoc]
        )
        switch = equilibrium + (r0 - equilibrium) * np.exp(-0.008 * (1 + c / 0.2) * 80)
        response[times >= 80] = switch * np.exp(-0.008 * (times[times >= 80] - 80))
        responses.append(response + rng.normal(0, 0.005, len(times)))
    experiment = SimpleNamespace(
        sensor_names=["one", "two", "three"],
        xs=[[times] for _ in responses],
        ys=[[response] for response in responses],
    )
    metadata = dict(
        association_start=0,
        association_end=80,
        fit_end=280,
        baseline_window=[-20, -5],
        concentrations=concentrations,
        regenerated=False,
        molar_concentrations_verified=True,
        bulk_shift=False,
    )
    offsets = {
        name: float(np.mean(response[times < 0][-10:]))
        for name, response in zip(experiment.sensor_names, responses)
    }
    experiment.ys = [
        [response - offsets[name]]
        for name, response in zip(experiment.sensor_names, responses)
    ]
    metadata["molar_concentrations_verified"] = True
    metadata["association_alignment"] = {"offsets_by_sensor": offsets}
    traces, meta = fitting.prepare(experiment, metadata)
    assert all(trace.initial > 9 for trace in traces)
    assert all(trace.background == 0 for trace in traces)
    result = fitting.fit_one_site(experiment, metadata, n_starts=2)
    assert result["params"]["Kd"] == pytest.approx(0.2, rel=0.002)
    assert result["params"]["k_off"] == pytest.approx(0.008, rel=0.002)
    for i, capacity in enumerate(capacities):
        assert result["params"][f"Rmax_{i + 1}"] == pytest.approx(capacity, rel=0.002)


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


def test_carterra_alignment_preserves_occupied_state_and_fit_residuals():
    folder = ROOT / "notebooks/data/spr_exports"
    metadata = json.loads((folder / "fit_metadata.json").read_text())["carterra_rows"]
    experiment = CarterraExperiment()
    experiment.read_csv(
        folder / metadata["file"],
        association_start=120,
        association_end=420,
        concentrations=metadata["concentrations"],
        concentration_unit="uM",
    )
    aligned = deepcopy(experiment)
    offsets = {
        name: float(np.mean(experiment.ys[i][0][-10:]))
        for i, name in enumerate(experiment.sensor_names)
    }
    aligned.align_association(inplace=True, npoints=10)
    aligned_meta = dict(
        metadata,
        association_alignment={
            "method": "align_association",
            "npoints": 10,
            "offsets_by_sensor": offsets,
        },
    )
    aligned_traces, aligned_meta = fitting.prepare(aligned, aligned_meta)
    params = dict(Kd=0.02, k_off=8e-5, Rmax_1=220, Rmax_2=220, Rmax_3=220)
    for i, trace in enumerate(aligned_traces):
        time = np.concatenate(experiment.xs[i])
        response = np.concatenate(experiment.ys[i])
        initial = float(np.median(response[(time >= 90) & (time <= 115)]))
        raw = fitting.Trace(
            trace.name, time, response, 0, initial, trace.concentration, trace.indices
        )
        offset = offsets[trace.name]
        assert abs(np.mean(aligned.ys[i][0][-10:])) < 1e-10
        assert trace.initial == pytest.approx(raw.initial)
        assert trace.initial > 0
        assert trace.alignment_offset == offset
        np.testing.assert_array_equal(trace.time, raw.time)
        np.testing.assert_allclose(trace.response, raw.response - offset)
        time = trace.time[trace.indices]
        original = fitting.predict(raw, metadata, params, i, time=time)
        shifted = fitting.predict(trace, aligned_meta, params, i, time=time)
        np.testing.assert_allclose(shifted, original - offset, atol=1e-10)
        np.testing.assert_allclose(
            trace.response[trace.indices] - shifted,
            raw.response[raw.indices] - original,
            atol=1e-10,
        )


def test_aligned_regenerated_data_are_not_baseline_subtracted_twice():
    folder = ROOT / "notebooks/data/spr_exports"
    metadata = json.loads((folder / "fit_metadata.json").read_text())[
        "biacore_cycle_sample"
    ]
    experiment = SprTextExperiment()
    experiment.read_file(
        folder / metadata["file"],
        association_start=0,
        association_end=180,
        concentrations=metadata["concentrations"],
        concentration_unit="uM",
    )
    offsets = {
        name: float(np.mean(experiment.ys[i][0][-10:]))
        for i, name in enumerate(experiment.sensor_names)
    }
    experiment.align_association(inplace=True, npoints=10)
    metadata["association_alignment"] = {"offsets_by_sensor": offsets}
    traces, _ = fitting.prepare(experiment, metadata)
    for i, trace in enumerate(traces):
        np.testing.assert_array_equal(trace.response, np.concatenate(experiment.ys[i]))
        assert trace.initial == 0
        assert trace.background == 0
        assert trace.alignment_offset == 0


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


def test_comparison_converts_micromolar_fit_to_published_units():
    fit = {
        "params": {"Kd": 0.02, "k_off": 1e-4, "Rmax_1": 100},
        "meta": {
            "literature_reference": {
                "parameters_molar": {"kon": 4000, "koff": 1e-4, "Kd": 25e-9}
            }
        },
    }
    comparison = fitting.literature_comparison(fit).set_index("Parameter")
    assert comparison.loc["kon", "This notebook"] == pytest.approx(5000)
    assert comparison.loc["Kd", "This notebook"] == pytest.approx(20)
    assert comparison.loc["kon", "Difference (%)"] == pytest.approx(25)
    assert comparison.loc["Kd", "Difference (%)"] == pytest.approx(-20)
