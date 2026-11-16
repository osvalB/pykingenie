# This code is AI generated.
"""
Time offset (t0) and time-axis handling:

- solution-phase induced-fit and conformational-selection models: t0 is the
  time between the start of the reaction and the first time point, as in the
  one-site solution model. Before, odeint started at the first time point and
  t0 had no effect.
- surface one-site association-only fit: traces start at time zero before t0
  is applied, as in the joint fit. Before, absolute times gave wrong rates.
- surface confidence intervals use the settings of the fit, not the caller's.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from pykingenie.fitter_surface import KineticsFitter
from pykingenie.utils.fitting_solution import (
    fit_conformational_selection_solution,
    fit_induced_fit_solution,
)
from pykingenie.utils.fitting_surface import fit_one_site_association
from pykingenie.utils.signal_solution import (
    get_initial_concentration_conformational_selection,
    ode_conformational_selection_insolution,
    ode_induced_fit_insolution,
    signal_ode_conformational_selection_insolution,
    signal_ode_induced_fit_insolution,
    solve_ode_conformational_selection_insolution,
    solve_ode_induced_fit_insolution,
)
from pykingenie.utils.signal_surface import (
    one_site_association_analytical,
    one_site_dissociation_analytical,
)

# Induced fit: E + S <-> E·S <-> ES. Conformational selection: E1 <-> E2, E2 + S <-> E2S.
IF_RATES = (1.0, 0.5, 0.2, 0.05)       # k_on, k_off, k_c, k_rev
CS_RATES = (0.5, 0.2, 1.0, 0.3)        # k_c (E1->E2), k_rev, k_on, k_off, in solver order
E_TOT, S_TOT = 1.0, 2.0
T_SOL = np.linspace(0.5, 30.5, 301)    # first point not at zero, as in measured data


def _reference(fun, y0, t, t0, rates):
    """Tight solve_ivp from the start of the reaction, which is t0 before the first point."""
    elapsed = t - t[0] + t0
    out = np.empty((len(t), len(y0)))
    for mask, end in ((elapsed >= 0, max(elapsed.max(), 0.0)), (elapsed < 0, min(elapsed.min(), 0.0))):
        if not np.any(mask) or end == 0.0:
            out[mask] = y0
            continue
        # solve_ivp needs t_eval ordered in the direction of integration
        order = np.argsort(elapsed[mask]) if end > 0 else np.argsort(elapsed[mask])[::-1]
        sol = solve_ivp(fun, (0.0, end), y0, t_eval=elapsed[mask][order], args=(*rates, E_TOT, S_TOT),
                        method="LSODA", rtol=1e-11, atol=1e-13)
        states = np.empty_like(sol.y.T)
        states[order] = sol.y.T
        out[mask] = states
    return out


CASES = {
    "induced_fit": (solve_ode_induced_fit_insolution, ode_induced_fit_insolution, [0.0, 0.0], IF_RATES, [2, 3]),
    "conformational_selection": (solve_ode_conformational_selection_insolution,
                                 ode_conformational_selection_insolution,
                                 [get_initial_concentration_conformational_selection(E_TOT, 0.5, 0.2)[1], 0.0],
                                 CS_RATES, [1, 3]),
}


@pytest.mark.parametrize("case", CASES.keys())
@pytest.mark.parametrize("t0", [0.0, 0.3, 2.0, -0.3])
def test_solution_t0_matches_reference(case, t0):
    """The state is y0 at t0 before the first point; negative t0 runs backward for earlier points."""
    solver, fun, y0, rates, rows = CASES[case]
    species = solver(T_SOL, y0, *rates, E_TOT, S_TOT, t0=t0)
    ref = _reference(fun, y0, T_SOL, t0, rates)
    np.testing.assert_allclose(species[rows].T, ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("case", CASES.keys())
def test_solution_t0_zero_starts_at_first_point(case):
    """With t0 = 0 the first point holds y0 exactly, wherever the time axis starts (unchanged behaviour)."""
    solver, _, y0, rates, rows = CASES[case]
    species = solver(T_SOL, y0, *rates, E_TOT, S_TOT)
    np.testing.assert_array_equal(species[rows][:, 0], y0)


@pytest.mark.parametrize("case", CASES.keys())
def test_solution_t0_shifts_the_curve(case):
    """A positive t0 is the same curve, t0 further along."""
    solver, _, y0, rates, rows = CASES[case]
    dt = T_SOL[1] - T_SOL[0]
    shifted = solver(T_SOL, y0, *rates, E_TOT, S_TOT, t0=10 * dt)
    unshifted = solver(T_SOL, y0, *rates, E_TOT, S_TOT)
    np.testing.assert_allclose(shifted[:, :-10], unshifted[:, 10:], rtol=0, atol=1e-6)


def test_solution_repeated_time_points():
    """Repeated time points get the same state (odeint needs a strictly ordered sequence)."""
    t = np.array([0.0, 1.0, 1.0, 2.0, 5.0, 5.0])
    s = solve_ode_induced_fit_insolution(t, [0.0, 0.0], *IF_RATES, E_TOT, S_TOT, t0=0.2)
    np.testing.assert_array_equal(s[:, 1], s[:, 2])
    np.testing.assert_array_equal(s[:, 4], s[:, 5])


LIG_CONCS = [0.5, 1.0, 2.0, 4.0]
T_FIT = np.arange(0.0, 20.0, 0.05)
DEAD_TIMES = [0.06, 0.03, 0.08, 0.05]   # within the fitter's t0 bounds (-0.1, 0.1)


def test_fit_induced_fit_solution_recovers_dead_time():
    k_on, k_off, k_c, k_rev = IF_RATES
    signal_es = 2.0
    signals = [signal_ode_induced_fit_insolution(T_FIT, [0, 0], k_on, k_off, k_c, k_rev, E_TOT, c,
                                                 t0=t0, signal_ES_int=signal_es, signal_ES=signal_es)
               for c, t0 in zip(LIG_CONCS, DEAD_TIMES)]

    p0 = [1.8, 1.2, 0.6, 0.25, 0.06] + [0.0] * 4
    fit, _, _, _ = fit_induced_fit_solution(
        signals, [T_FIT] * 4, LIG_CONCS, [E_TOT] * 4,
        p0, [0.1, 1e-3, 1e-3, 1e-3, 1e-3] + [-0.1] * 4, [10, 100, 100, 100, 100] + [0.1] * 4,
        fixed_t0=False)

    np.testing.assert_allclose(fit[:5], [signal_es, *IF_RATES], rtol=1e-3)
    np.testing.assert_allclose(fit[5:], DEAD_TIMES, atol=1e-3)


def test_fit_conformational_selection_solution_recovers_dead_time():
    k_c, k_rev, k_on, k_off = CS_RATES
    signal_e2s = 2.0
    _, e2 = get_initial_concentration_conformational_selection(E_TOT, k_c, k_rev)
    signals = [signal_ode_conformational_selection_insolution(T_FIT, [e2, 0], k_c, k_rev, k_on, k_off,
                                                              E_TOT, c, t0=t0, signal_E2S=signal_e2s)
               for c, t0 in zip(LIG_CONCS, DEAD_TIMES)]

    # Parameter order: signal_E2S, k_on, k_off, k_c, k_rev, t0...
    p0 = [1.8, 1.2, 0.4, 0.6, 0.25] + [0.0] * 4
    fit, _, _, _ = fit_conformational_selection_solution(
        signals, [T_FIT] * 4, LIG_CONCS, [E_TOT] * 4,
        p0, [0.1, 1e-3, 1e-3, 1e-3, 1e-3] + [-0.1] * 4, [10, 100, 100, 100, 100] + [0.1] * 4,
        fixed_t0=False)

    np.testing.assert_allclose(fit[:5], [signal_e2s, k_on, k_off, k_c, k_rev], rtol=1e-3)
    np.testing.assert_allclose(fit[5:], DEAD_TIMES, atol=1e-3)


# Surface one-site association-only fit

KD, KOFF, SMAX = 0.5, 0.01, 5.0
SURF_CONCS = [0.1, 0.5, 2.0]
T_ASSOC = np.arange(0.0, 300.0, 2.0)


def _surface_fitter(offset, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    assoc = [one_site_association_analytical(T_ASSOC, 0, SMAX, KOFF, KD, c) for c in SURF_CONCS]
    disso = [one_site_dissociation_analytical(T_ASSOC, a[-1], KOFF) for a in assoc]
    assoc = [a + rng.normal(0, noise * SMAX, len(a)) for a in assoc]
    disso = [d + rng.normal(0, noise * SMAX, len(d)) for d in disso]
    fitter = KineticsFitter([T_ASSOC + offset] * 3, assoc, SURF_CONCS,
                            [T_ASSOC + 300 + offset] * 3, disso, smax_id=[0] * 3)
    fitter.fit_steady_state()
    return fitter


@pytest.mark.parametrize("offset", [0.0, 300.0, 1234.5])
def test_association_only_fit_does_not_depend_on_time_origin(offset):
    """Before, association times starting at 300 s gave Kd 0.0045 and koff 1e-5 instead of 0.5 and 0.01."""
    fitter = _surface_fitter(offset)
    fitter.fit_one_site_association(shared_smax=True)
    np.testing.assert_allclose([fitter.Kd, fitter.k_off, fitter.Smax[0]], [KD, KOFF, SMAX], rtol=1e-4)


def test_association_only_fit_recovers_t0_with_absolute_times():
    """t0 shifts the start from the first time point of each trace, not from absolute zero."""
    t0 = 2.0
    signals = [one_site_association_analytical(T_ASSOC, 0, SMAX, KOFF, KD, c, t0) for c in SURF_CONCS]
    fit, _, _ = fit_one_site_association(
        signals, [T_ASSOC + 300.0] * 3, SURF_CONCS,
        [0.3, 0.02, 0.0, 4.0], [1e-3, 1e-4, -5.0, 0.5], [10.0, 1.0, 5.0, 50.0],
        smax_idx=[0, 0, 0], shared_smax=True, fixed_t0=False)
    np.testing.assert_allclose(fit, [KD, KOFF, t0, SMAX], rtol=1e-4)


# Surface confidence intervals

def test_ci95_uses_the_settings_of_the_fit():
    """Without arguments the CI uses the fit's settings, and gives the same table as matching arguments."""
    fitter = _surface_fitter(0.0, noise=0.005)
    fitter.fit_one_site_assoc_and_disso(shared_smax=False, fixed_t0=True)

    fitter.calculate_ci95()
    default = fitter.fit_params_kinetics_ci95
    assert not default.empty
    assert np.all(default.iloc[:, 1] < default.iloc[:, 2])

    fitter.calculate_ci95(shared_smax=False, fixed_t0=True, fit_ktr=False)
    pd.testing.assert_frame_equal(fitter.fit_params_kinetics_ci95, default)


@pytest.mark.parametrize("kwargs", [{"shared_smax": True}, {"fixed_t0": False}, {"fit_ktr": True}])
def test_ci95_rejects_settings_that_differ_from_the_fit(kwargs):
    """Before, a mismatch read the parameters in the wrong order and gave an empty table without warning."""
    fitter = _surface_fitter(0.0, noise=0.005)
    fitter.fit_one_site_assoc_and_disso(shared_smax=False, fixed_t0=True)
    with pytest.raises(ValueError, match="differs from the fit"):
        fitter.calculate_ci95(**kwargs)


def test_ci95_with_fitted_t0():
    fitter = _surface_fitter(0.0, noise=0.005)
    fitter.fit_one_site_assoc_and_disso(shared_smax=True, fixed_t0=False)
    fitter.calculate_ci95()
    ci = fitter.fit_params_kinetics_ci95
    assert list(ci['Parameter']) == ['Kd [µM]', 'k_off [1/s]']
    assert ci.iloc[0, 1] < fitter.Kd < ci.iloc[0, 2]
    assert ci.iloc[1, 1] < fitter.k_off < ci.iloc[1, 2]


def test_ci95_is_empty_for_unsupported_fits():
    """Mass transport fits are not supported: the table is emptied, not left from an earlier fit."""
    fitter = _surface_fitter(0.0, noise=0.005)
    fitter.fit_one_site_assoc_and_disso(shared_smax=True)
    fitter.calculate_ci95()
    assert not fitter.fit_params_kinetics_ci95.empty

    fitter.fit_one_site_assoc_and_disso(shared_smax=True, fit_ktr=True)
    fitter.calculate_ci95()
    assert fitter.fit_params_kinetics_ci95.empty
