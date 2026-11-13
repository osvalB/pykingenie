# This code is AI generated.
"""
Induced-fit model (surface) where the induced complex ES produces a different
signal than the intermediate complex E·S.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pykingenie.fitter_surface import KineticsFitter
from pykingenie.utils.fitting_surface import fit_induced_fit_sites_assoc_and_disso
from pykingenie.utils.signal_surface import (
    solve_induced_fit_association,
    solve_induced_fit_dissociation,
)

SMAX = 5.0
RATES = (0.1, 0.05, 0.01, 0.005)  # kon, koff, kc, krev
CONCS = [0.05, 0.15, 0.45, 1.35, 4.05]
T_ASSOC = np.arange(0, 600.0, 2.0)
T_DISSO = np.arange(600.0, 1800.0, 4.0)


def _simulate(ratio, smax=SMAX):
    kon, koff, kc, krev = RATES
    assoc, disso = [], []
    for c in CONCS:
        y = solve_induced_fit_association(T_ASSOC, c, kon, koff, kc, krev, smax=smax, signal_ratio=ratio)
        d = solve_induced_fit_dissociation(T_DISSO, koff, kc, krev, s0=y[-1, 0], sP2L=y[-1, 2],
                                           smax=smax, signal_ratio=ratio)
        assoc.append(y)
        disso.append(d)
    return assoc, disso


def _noisy_fitter(ratio, seed=0, smax_values=None):
    rng = np.random.default_rng(seed)
    smax_values = smax_values or [SMAX] * len(CONCS)
    kon, koff, kc, krev = RATES
    assoc, disso = [], []
    for c, smax in zip(CONCS, smax_values):
        y = solve_induced_fit_association(T_ASSOC, c, kon, koff, kc, krev, smax=smax, signal_ratio=ratio)
        d = solve_induced_fit_dissociation(T_DISSO, koff, kc, krev, s0=y[-1, 0], sP2L=y[-1, 2],
                                           smax=smax, signal_ratio=ratio)[:, 0]
        assoc.append(y[:, 0] + rng.normal(0, 0.005 * SMAX, len(T_ASSOC)))
        disso.append(d + rng.normal(0, 0.005 * SMAX, len(T_DISSO)))
    n = len(CONCS)
    fitter = KineticsFitter([T_ASSOC] * n, assoc, CONCS, [T_DISSO] * n, disso, smax_id=[0] * n)
    fitter.fit_steady_state()
    return fitter


def test_unit_ratio_matches_default_signal():
    kon, koff, kc, krev = RATES
    a_default = solve_induced_fit_association(T_ASSOC, 0.5, kon, koff, kc, krev, smax=SMAX)
    a_unit = solve_induced_fit_association(T_ASSOC, 0.5, kon, koff, kc, krev, smax=SMAX, signal_ratio=1.0)
    np.testing.assert_allclose(a_unit, a_default)

    d_default = solve_induced_fit_dissociation(T_DISSO, koff, kc, krev, s0=a_default[-1, 0],
                                               sP2L=a_default[-1, 2], smax=SMAX)
    d_unit = solve_induced_fit_dissociation(T_DISSO, koff, kc, krev, s0=a_default[-1, 0],
                                            sP2L=a_default[-1, 2], smax=SMAX, signal_ratio=1.0)
    np.testing.assert_allclose(d_unit, d_default)


@pytest.mark.parametrize("ratio", [0.0, 0.5, 2.0])
def test_total_signal_weights_induced_complex(ratio):
    """The species are the same for any ratio; only the ES contribution to the total changes."""
    assoc_ref, disso_ref = _simulate(1.0)
    assoc, disso = _simulate(ratio)
    for ref, mat in zip(assoc_ref + disso_ref, assoc + disso):
        np.testing.assert_allclose(mat[:, 1:], ref[:, 1:], atol=1e-10)
        np.testing.assert_allclose(mat[:, 0], mat[:, 1] + ratio * mat[:, 2], atol=1e-10)


def test_dissociation_continues_from_association():
    """The dissociation starts from the bound amounts reached at the end of the association."""
    assoc, disso = _simulate(2.5)
    for a, d in zip(assoc, disso):
        np.testing.assert_allclose(d[0], a[-1], rtol=1e-8, atol=1e-10)


def test_global_fit_with_fixed_and_fitted_ratio():
    """A fixed ratio equal to the true one, or a fitted one, both reproduce noise-free data."""
    ratio = 2.0
    assoc, disso = _simulate(ratio)
    assoc_sig = [a[:, 0] for a in assoc]
    disso_sig = [d[:, 0] for d in disso]
    p0 = [0.08, 0.06, 0.012, 0.004, SMAX * 0.9]
    low = [1e-4, 1e-4, 1e-5, 1e-5, 0.1]
    high = [10, 10, 1, 1, 50]

    fit, _, fa, fd = fit_induced_fit_sites_assoc_and_disso(
        assoc_sig, [T_ASSOC] * 5, CONCS, disso_sig, [T_DISSO] * 5,
        p0, low, high, smax_idx=[0] * 5, shared_smax=True,
        signal_ratio_value=ratio)
    np.testing.assert_allclose(fit, list(RATES) + [SMAX], rtol=1e-3)

    p0.insert(4, 1.0)
    low.insert(4, 1e-2)
    high.insert(4, 1e2)
    fit, _, fa, fd = fit_induced_fit_sites_assoc_and_disso(
        assoc_sig, [T_ASSOC] * 5, CONCS, disso_sig, [T_DISSO] * 5,
        p0, low, high, smax_idx=[0] * 5, shared_smax=True,
        fit_signal_ratio=True)
    np.testing.assert_allclose(fit, list(RATES) + [ratio, SMAX], rtol=1e-3)


@pytest.mark.parametrize("ratio", [0.4, 1.8])
def test_fitter_recovers_signal_ratio(ratio):
    fitter = _noisy_fitter(ratio)
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=True, fit_signal_ratio=True)

    np.testing.assert_allclose([fitter.k_on, fitter.k_off, fitter.k_c, fitter.k_rev],
                               RATES, rtol=0.05)
    assert fitter.signal_ratio == pytest.approx(ratio, rel=0.02)
    assert fitter.Smax[0] == pytest.approx(SMAX, rel=0.02)

    assert 'Signal ratio (ES/E·S)' in fitter.fit_params_kinetics.columns
    assert 'Signal ratio (ES/E·S)' in fitter.fit_params_kinetics_error.columns
    assert 'Signal ratio (ES/E·S)' in fitter.fit_params_kinetics_correlation.columns
    assert len(fitter.params) == len(fitter.low_bounds) == len(fitter.high_bounds) == 6
    assert (fitter.low_bounds[4], fitter.high_bounds[4]) == (0.1, 10.0)


def test_fitter_signal_ratio_per_trace_smax_and_loo():
    smax_values = [4.0, 4.5, 5.0, 5.5, 6.0]
    fitter = _noisy_fitter(1.8, smax_values=smax_values)
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=False, fit_signal_ratio=True, calc_loo=True)

    assert fitter.signal_ratio == pytest.approx(1.8, rel=0.02)
    np.testing.assert_allclose(fitter.Smax, smax_values, rtol=0.02)
    assert len(fitter.signal_ratio_loo) == len(CONCS)
    assert fitter.signal_ratio_loo_mean == pytest.approx(1.8, rel=0.02)
    assert 'Signal ratio (ES/E·S)' in fitter.df_fit_loo.columns


def _ode_rhs(t, y, a, kon, koff, kc, krev, smax):
    """Mass-action ODEs for R[E·S] and R[ES], in E·S signal units."""
    c1, c2 = y
    free = smax - c1 - c2
    return [kon * a * free - (koff + kc) * c1 + krev * c2,
            kc * c1 - krev * c2]


def _ode_solve(t, y0, a, rates, smax):
    sol = solve_ivp(_ode_rhs, (t[0], t[-1]), y0, t_eval=t, args=(a, *rates, smax),
                    method="LSODA", rtol=1e-11, atol=1e-13)
    return sol.y.T


ODE_CASES = {
    "slow_conformational_step": (0.1, 0.05, 0.01, 0.005),
    "fast_conformational_step": (0.1, 0.01, 0.1, 1.0),
    "stable_induced_complex": (1.0, 0.5, 0.2, 0.001),
    "near_degenerate_rates": (0.05, 0.02, 0.02, 0.02),
}


@pytest.mark.parametrize("rates", ODE_CASES.values(), ids=ODE_CASES.keys())
@pytest.mark.parametrize("ratio", [0.1, 1.0, 2.0, 10.0])
@pytest.mark.parametrize("single_cycle", [False, True])
def test_signal_matches_numerical_ode(rates, ratio, single_cycle):
    """The closed-form solution matches a direct ODE integration, species and total signal."""
    kon, koff, kc, krev = rates
    t_assoc = np.linspace(0, 300, 301)
    t_disso = np.linspace(300, 900, 301)
    y_ode, last = np.zeros(2), None

    for c in (0.05, 0.3, 2.0):
        if not single_cycle:
            y_ode, last = np.zeros(2), None

        ode_a = _ode_solve(t_assoc, y_ode, c, rates, SMAX)
        ode_d = _ode_solve(t_disso, ode_a[-1], 0.0, rates, SMAX)

        sP1L, sP2L = (last[1], last[2]) if last is not None else (0, 0)
        code_a = solve_induced_fit_association(t_assoc, c, kon, koff, kc, krev, sP1L=sP1L, sP2L=sP2L,
                                               smax=SMAX, signal_ratio=ratio)
        code_d = solve_induced_fit_dissociation(t_disso, koff, kc, krev, s0=code_a[-1, 0],
                                                sP2L=code_a[-1, 2], smax=SMAX, signal_ratio=ratio)

        for code, ode in ((code_a, ode_a), (code_d, ode_d)):
            np.testing.assert_allclose(code[:, 1:], ode, rtol=0, atol=1e-8 * SMAX)
            np.testing.assert_allclose(code[:, 0], ode[:, 0] + ratio * ode[:, 1], rtol=0, atol=1e-8 * SMAX)

        y_ode, last = ode_d[-1], code_d[-1]


@pytest.mark.parametrize("single_cycle", [False, True])
def test_fit_recovers_parameters_from_ode_data(single_cycle):
    """Fitting noise-free ODE data recovers the rates, the signal ratio and Smax."""
    rates, ratio = RATES, 2.0
    concs = [0.05, 0.15, 0.45, 1.35]
    y0, t = np.zeros(2), 0.0
    t_a_lst, t_d_lst, s_a_lst, s_d_lst = [], [], [], []

    for c in concs:
        if single_cycle:
            t_a = np.arange(t, t + 300.5, 1.0)
        else:
            y0, t_a = np.zeros(2), np.arange(0, 300.5, 1.0)
        t_d = np.arange(t_a[-1], t_a[-1] + 600.5, 2.0)

        ode_a = _ode_solve(t_a, y0, c, rates, SMAX)
        ode_d = _ode_solve(t_d, ode_a[-1], 0.0, rates, SMAX)

        t_a_lst.append(t_a)
        t_d_lst.append(t_d)
        s_a_lst.append(ode_a[:, 0] + ratio * ode_a[:, 1])
        s_d_lst.append(ode_d[:, 0] + ratio * ode_d[:, 1])
        y0, t = ode_d[-1], t_d[-1]

    fit, _, _, _ = fit_induced_fit_sites_assoc_and_disso(
        s_a_lst, t_a_lst, concs, s_d_lst, t_d_lst,
        [0.07, 0.07, 0.02, 0.002, 1.0, 4.0], [1e-4] * 4 + [0.1, 0.1], [10] * 4 + [10, 50],
        smax_idx=[0] * len(concs), shared_smax=True, fit_signal_ratio=True)

    np.testing.assert_allclose(fit, list(rates) + [ratio, SMAX], rtol=1e-4)


def test_fitter_default_keeps_equal_signals():
    fitter = _noisy_fitter(1.0)
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=True)
    assert fitter.signal_ratio == 1.0
    assert not fitter.fit_signal_ratio
    assert 'Signal ratio (ES/E·S)' not in fitter.fit_params_kinetics.columns
    assert len(fitter.params) == 5
