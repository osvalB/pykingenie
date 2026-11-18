"""Physical-clock and occupied-state checks for the SPR teaching analyses."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spr_notebook_fitting", ROOT / "notebooks" / "spr_fitting.py"
)
fitting = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fitting
SPEC.loader.exec_module(fitting)


def test_masked_transport_fit_keeps_true_clock_and_occupied_state():
    """Compare against independent integration, including an unobserved switch."""
    times = np.array([15.0, 30.0, 45.0, 63.0, 100.0])
    trace = fitting.Trace("occupied", times, times * 0, 0, 20, 0.4, np.arange(5))
    meta = dict(association_start=10, association_end=60, fit_end=100)
    params = dict(Kd=0.2, k_off=0.01, Ktr=2, Rmax_1=100)
    kon = params["k_off"] / params["Kd"]

    def ode(concentration):
        return lambda t, y: (kon * concentration * (100 - y) - 0.01 * y) / (
            1 + kon * (100 - y) / 2
        )

    assoc = solve_ivp(
        ode(0.4), (10, 60), [20], rtol=1e-11, atol=1e-12, dense_output=True
    )
    disso = solve_ivp(
        ode(0), (60, 100), assoc.y[:, -1], rtol=1e-11, atol=1e-12, dense_output=True
    )
    expected = np.r_[assoc.sol(times[:3])[0], disso.sol(times[3:])[0]]
    actual = fitting.predict(trace, meta, "mass_transport", True, params, 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_heterogeneous_dissociation_uses_each_state_at_unobserved_switch():
    times = np.array([5.0, 22.0, 48.0, 55.0, 120.0])
    trace = fitting.Trace("two states", times, times * 0, 0, 0, 0.3, np.arange(5))
    meta = dict(association_start=0, association_end=50, fit_end=120)
    params = dict(Kd1=0.1, k_off1=0.003, Kd2=0.8, k_off2=0.05, fraction=0.7, Rmax_1=150)
    expected = np.zeros(5)
    for kd, off, fraction in [(0.1, 0.003, 0.7), (0.8, 0.05, 0.3)]:
        equilibrium = 150 * fraction * 0.3 / (kd + 0.3)
        rate = off * (1 + 0.3 / kd)
        expected[:3] += equilibrium * (1 - np.exp(-rate * times[:3]))
        switch = equilibrium * (1 - np.exp(-rate * 50))
        expected[3:] += switch * np.exp(-off * (times[3:] - 50))
    actual = fitting.predict(trace, meta, "heterogeneous", True, params, 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


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
    traces, meta = fitting.prepare(experiment, metadata)
    assert all(trace.initial > 9 for trace in traces)
    assert all(trace.background == 0 for trace in traces)
    result = fitting.fit_candidate(traces, meta, "one_site", True, n_starts=2)
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
