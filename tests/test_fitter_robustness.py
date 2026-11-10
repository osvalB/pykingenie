# This code is AI generated and reviewed by a human (PR #18, PR #XX).
"""
Fits that must recover known parameters from simulated data with noise:
starting points for mass transport, carry-over between steps, and an
induced-fit model whose rates are all determined by the curves, or only
partly when the conformational step is fast.
"""
import numpy as np
import pytest

import pykingenie.fitter_surface as fitter_surface
import pykingenie.utils.fitting_surface as fitting_surface
from pykingenie.fitter_surface import KineticsFitter
from pykingenie.utils.fitting_surface import fit_one_site_assoc_and_disso_ktr
from pykingenie.utils.signal_surface import (
    solve_induced_fit_association,
    solve_induced_fit_dissociation,
    solve_ode_one_site_mass_transport_association,
    solve_ode_one_site_mass_transport_dissociation,
)

SMAX = 5.0
T_ASSOC = np.arange(0, 300.0, 3.0)
T_DISSO = np.arange(300.0, 900.0, 6.0)


def _max_rel_error(actual, expected):
    """Largest difference relative to the largest expected value."""
    return np.max(np.abs(np.asarray(actual) - expected)) / np.max(np.abs(expected))


def _mass_transport_fitter(Kd, koff, ktr, seed=1, noise=0.005):
    """Multi-cycle mass-transport data at 0.1 to 10 times Kd, with noise relative to Smax."""
    rng = np.random.default_rng(seed)
    concs = [Kd * x for x in (0.1, 0.3, 1, 3, 10)]
    assoc, disso = [], []
    for c in concs:
        a = solve_ode_one_site_mass_transport_association(T_ASSOC, 0.0, c, Kd, koff, ktr, SMAX)
        d = solve_ode_one_site_mass_transport_dissociation(T_DISSO - T_DISSO[0], a[-1], Kd, koff, ktr, SMAX)
        assoc.append(a + rng.normal(0, noise * SMAX, len(a)))
        disso.append(d + rng.normal(0, noise * SMAX, len(d)))
    fitter = KineticsFitter([T_ASSOC] * 5, assoc, concs, [T_DISSO] * 5, disso, smax_id=[0] * 5)
    fitter.fit_steady_state()
    return fitter


# With slow dissociation and binding twice as fast as transport, the dissociation
# fit gives a Kd and koff guess about twice too high and too low. From that
# start alone the fit stops at Kd 10.8 and koff 0.00055; the grid of starting
# points finds the true values.
def test_mass_transport_grid_start_recovers_parameters():
    Kd, koff = 5.0, 0.001
    ktr = koff / Kd * SMAX / 2
    fitter = _mass_transport_fitter(Kd, koff, ktr)
    fitter.fit_one_site_assoc_and_disso(fit_ktr=True)
    np.testing.assert_allclose([fitter.Kd, fitter.k_off], [Kd, koff], rtol=0.05)


# The grid only adds a second start: where the first start already works,
# the result is unchanged.
def test_mass_transport_grid_start_keeps_good_fits():
    Kd, koff = 0.5, 0.01
    ktr = koff / Kd * SMAX / 20
    fitter = _mass_transport_fitter(Kd, koff, ktr)
    fitter.fit_one_site_assoc_and_disso(fit_ktr=True)
    np.testing.assert_allclose([fitter.Kd, fitter.k_off], [Kd, koff], rtol=0.1)


# A mass-transport trace that does not follow straight after the previous
# dissociation starts from an empty surface, even when its time does not
# start near 0.
def test_mass_transport_fit_restarts_after_gap():
    Kd, koff, ktr = 0.5, 0.01, 0.05
    concs = [0.5, 1.0]
    t_assoc = [np.arange(0, 300.0, 2.0), np.arange(1000.0, 1300.0, 2.0)]
    t_disso = [np.arange(300.0, 600.0, 2.0), np.arange(1300.0, 1600.0, 2.0)]
    assoc, disso = [], []
    for c, ta, td in zip(concs, t_assoc, t_disso):
        a = solve_ode_one_site_mass_transport_association(ta - ta[0], 0.0, c, Kd, koff, ktr, SMAX)
        assoc.append(a)
        disso.append(solve_ode_one_site_mass_transport_dissociation(td - td[0], a[-1], Kd, koff, ktr, SMAX))

    params, _, fit_assoc, _ = fit_one_site_assoc_and_disso_ktr(
        assoc, t_assoc, concs, disso, t_disso,
        initial_parameters=[0.04, 4.0], low_bounds=[1e-3, 1.0], high_bounds=[1.0, 20.0],
        smax_idx=[0, 0], shared_smax=True,
        fixed_Kd=True, Kd_value=Kd, fixed_koff=True, koff_value=koff)

    np.testing.assert_allclose(params, [ktr, SMAX], rtol=1e-4)
    assert _max_rel_error(fit_assoc[1], assoc[1]) < 1e-4


def _two_start_fit(Kd=0.5, koff=0.01, ktr=0.05):
    """Fit noise-free mass-transport data from a start and one start candidate."""
    concs = [0.25, 1.0]
    assoc, disso = [], []
    for c in concs:
        a = solve_ode_one_site_mass_transport_association(T_ASSOC, 0.0, c, Kd, koff, ktr, SMAX)
        assoc.append(a)
        disso.append(solve_ode_one_site_mass_transport_dissociation(T_DISSO - T_DISSO[0], a[-1], Kd, koff, ktr, SMAX))
    return fit_one_site_assoc_and_disso_ktr(
        assoc, [T_ASSOC] * 2, concs, disso, [T_DISSO] * 2,
        initial_parameters=[0.4, 0.012, 0.04, 4.5],
        low_bounds=[1e-3, 1e-4, 1e-4, 1.0], high_bounds=[10, 1, 1, 20],
        smax_idx=[0, 0], shared_smax=True,
        start_candidates=[[0.6, 0.008, 0.06, 5.5]])


# If the fit from one start fails, the fit from the other start is returned.
def test_two_start_fit_survives_one_failed_start(monkeypatch):
    original = fitting_surface.curve_fit
    calls = []

    def fail_first(*args, **kwargs):
        calls.append(kwargs["p0"])
        if len(calls) == 1:
            raise RuntimeError("Optimal parameters not found")
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting_surface, "curve_fit", fail_first)
    params = _two_start_fit()[0]
    assert len(calls) == 2
    np.testing.assert_allclose(params, [0.5, 0.01, 0.05, SMAX], rtol=1e-4)


# If the fits from both starts fail, the error is raised.
def test_two_start_fit_raises_when_all_starts_fail(monkeypatch):
    def always_fail(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found")

    monkeypatch.setattr(fitting_surface, "curve_fit", always_fail)
    with pytest.raises(RuntimeError, match="Optimal parameters not found"):
        _two_start_fit()


# With a slow conformational change (kc, krev below koff), the curves show
# it, and all four induced-fit rates are recovered from data with 0.5 % noise.
@pytest.mark.parametrize("seed", [0, 1])
def test_induced_fit_recovers_all_rates(seed):
    kon, koff, kc, krev = 0.1, 0.05, 0.01, 0.005
    concs = [0.05, 0.15, 0.45, 1.35, 4.05]
    t_assoc = np.arange(0, 600.0, 2.0)
    t_disso = np.arange(600.0, 1800.0, 4.0)
    rng = np.random.default_rng(seed)
    assoc, disso = [], []
    for c in concs:
        y = solve_induced_fit_association(t_assoc, c, kon, koff, kc, krev, smax=SMAX)
        d = solve_induced_fit_dissociation(t_disso, koff, kc, krev, s0=y[-1, 0], sP2L=y[-1, 2], smax=SMAX)[:, 0]
        assoc.append(y[:, 0] + rng.normal(0, 0.005 * SMAX, len(t_assoc)))
        disso.append(d + rng.normal(0, 0.005 * SMAX, len(t_disso)))

    fitter = KineticsFitter([t_assoc] * 5, assoc, concs, [t_disso] * 5, disso, smax_id=[0] * 5)
    fitter.fit_steady_state()
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=True)

    np.testing.assert_allclose([fitter.k_on, fitter.k_off, fitter.k_c, fitter.k_rev],
                               [kon, koff, kc, krev], rtol=0.05)


def _induced_fit_data(kon, koff, kc, krev, concs, seed):
    """Induced-fit fitter on data with noise of 0.5 % of Smax, and the residual of the true curves."""
    t_assoc = np.arange(0, 600.0, 2.0)
    t_disso = np.arange(600.0, 1800.0, 4.0)
    rng = np.random.default_rng(seed)
    assoc, disso, rss_true = [], [], 0.0
    for c in concs:
        y = solve_induced_fit_association(t_assoc, c, kon, koff, kc, krev, smax=SMAX)
        d = solve_induced_fit_dissociation(t_disso, koff, kc, krev, s0=y[-1, 0], sP2L=y[-1, 2], smax=SMAX)[:, 0]
        noise_a = rng.normal(0, 0.005 * SMAX, len(t_assoc))
        noise_d = rng.normal(0, 0.005 * SMAX, len(t_disso))
        assoc.append(y[:, 0] + noise_a)
        disso.append(d + noise_d)
        rss_true += np.sum(noise_a ** 2) + np.sum(noise_d ** 2)
    n = len(concs)
    fitter = KineticsFitter([t_assoc] * n, assoc, concs, [t_disso] * n, disso, smax_id=[0] * n)
    fitter.fit_steady_state()
    return fitter, rss_true


def _fit_rss(fitter):
    return sum(np.sum((y - f) ** 2) for y, f in zip(fitter.assoc_lst + fitter.disso_lst,
                                                    fitter.signal_assoc_fit + fitter.signal_disso_fit))


def _kd_app(kon, koff, kc, krev):
    """Equilibrium dissociation constant of the induced-fit model."""
    return koff / kon * krev / (kc + krev)


# With a conformational step faster than dissociation the curves only fix kon
# and an apparent koff, and many (kc, krev) fit equally well. With this noise
# the fit used to stop at the evaluation limit and raise; it now ends as close
# to the data as the true curves, with the right equilibrium Kd.
@pytest.mark.parametrize("shared_smax", [True, False])
def test_induced_fit_fast_conformational_step_converges(shared_smax):
    rates = (0.1, 0.01, 0.1, 1.0)
    fitter, rss_true = _induced_fit_data(*rates, [0.01, 0.03, 0.09, 0.27, 0.81], seed=0)
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=shared_smax)
    assert _fit_rss(fitter) < 1.01 * rss_true
    np.testing.assert_allclose(_kd_app(fitter.k_on, fitter.k_off, fitter.k_c, fitter.k_rev),
                               _kd_app(*rates), rtol=0.02)


def _final_fit_calls(monkeypatch, n_failures):
    """Make the first n_failures full-data induced-fit fits raise; return the starts tried."""
    original = fitter_surface.fit_induced_fit_sites_assoc_and_disso
    starts = []

    def fail_final(*args, **kwargs):
        if kwargs.get("max_nfev") is None:  # grid fits on subsampled data
            return original(*args, **kwargs)
        starts.append(list(args[5]))
        if len(starts) <= n_failures:
            raise RuntimeError("Optimal parameters not found")
        return original(*args, **kwargs)

    monkeypatch.setattr(fitter_surface, "fit_induced_fit_sites_assoc_and_disso", fail_final)
    return starts


# If the fit from the best grid point fails, the fit starts again from the
# next-best grid point.
def test_induced_fit_falls_back_to_next_grid_point(monkeypatch):
    fitter, rss_true = _induced_fit_data(0.1, 0.05, 0.01, 0.005, [0.05, 0.15, 0.45, 1.35, 4.05], seed=0)
    starts = _final_fit_calls(monkeypatch, n_failures=1)
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=True)
    assert len(starts) == 2
    assert starts[0][2:4] != starts[1][2:4]  # a different (kc, krev) start
    assert fitter.p0 == starts[1]
    assert _fit_rss(fitter) < 1.01 * rss_true


# If the fits from all tried grid points fail, the error is raised.
def test_induced_fit_raises_when_all_starts_fail(monkeypatch):
    fitter, _ = _induced_fit_data(0.1, 0.05, 0.01, 0.005, [0.05, 0.15, 0.45, 1.35, 4.05], seed=0)
    starts = _final_fit_calls(monkeypatch, n_failures=100)
    with pytest.raises(RuntimeError, match="Optimal parameters not found"):
        fitter.fit_one_site_if_assoc_and_disso(shared_smax=True)
    assert len(starts) == 3


# If no grid point can be fitted, there is no start for the full fit and an
# error says so.
def test_induced_fit_raises_without_grid_start(monkeypatch):
    fitter, _ = _induced_fit_data(0.1, 0.05, 0.01, 0.005, [0.05, 0.15, 0.45, 1.35, 4.05], seed=0)

    def always_fail(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found")

    fitter.fit_one_site_assoc_and_disso(shared_smax=True)  # starting values, before patching
    monkeypatch.setattr(fitter, "fit_one_site_assoc_and_disso", lambda **kwargs: None)
    monkeypatch.setattr(fitter_surface, "fit_induced_fit_sites_assoc_and_disso", always_fail)
    with pytest.raises(RuntimeError, match="could not be fitted from any starting point"):
        fitter.fit_one_site_if_assoc_and_disso(shared_smax=True)
