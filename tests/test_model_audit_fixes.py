# This code is AI generated and reviewed by a human (PR #16).
# AI-generated additions extend the induced-fit relaxation-rate regression tests.
"""
Checks for fixes found by comparing the surface and solution models with their
reaction equations solved directly.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pykingenie.utils.signal_surface import (
    solve_all_states_fast,
    solve_two_site_association,
    solve_two_site_cooperative_association,
    solve_two_site_cooperative_dissociation,
    solve_induced_fit_association,
    solve_induced_fit_dissociation,
    solve_ode_mixture_analyte_association,
    solve_ode_mixture_analyte_dissociation,
)
from pykingenie.utils.signal_solution import (
    ode_induced_fit_insolution,
    ode_conformational_selection_insolution,
    get_kobs_induced_fit,
    get_kobs_conformational_selection,
)
from pykingenie.utils.fitting_surface import (
    fit_two_site_assoc_and_disso,
    fit_induced_fit_sites_assoc_and_disso,
)


def _reference(f, y0, t):
    """Solve an ODE directly with tight tolerances."""
    return solve_ivp(f, [t[0], t[-1]], y0, t_eval=t, method="LSODA",
                     rtol=1e-11, atol=1e-14).y.T


def _max_rel_error(actual, expected):
    """Largest difference relative to the largest expected value."""
    return np.max(np.abs(np.asarray(actual) - expected)) / np.max(np.abs(expected))


def _jacobian(f, y, h=1e-7):
    """Central-difference Jacobian of f at y."""
    y = np.asarray(y, dtype=float)
    cols = [(np.array(f(0, y + e)) - np.array(f(0, y - e))) / (2 * h)
            for e in np.eye(len(y)) * h]
    return np.column_stack(cols)


# A 2x2 matrix with equal eigenvalues that cannot be diagonalised:
# exp(A·t) has a t·e^(λt) term that an eigendecomposition cannot represent.
def test_solve_all_states_fast_defective_matrix():
    k = 0.02
    A = np.array([[-k, 0.5], [0.0, -k]])
    t = np.linspace(0, 300, 61)
    x0 = np.array([1.0, 2.0])
    states = solve_all_states_fast(t, A, x0, np.zeros(2))
    expected = np.column_stack(((x0[0] + 0.5 * x0[1] * t) * np.exp(-k * t),
                                x0[1] * np.exp(-k * t)))
    np.testing.assert_allclose(states, expected, rtol=1e-10)


# sigma = 4 makes both dissociation rates equal (koff and 2·koff/√σ).
# The result must still match the reaction equations, also just around 4.
@pytest.mark.parametrize("sigma", [4.0, 4.0 * (1 + 1e-9), 4.0 * (1 + 1e-5), 3.9])
def test_cooperative_two_site_dissociation_equal_rates(sigma):
    koff = 0.01
    s = np.sqrt(sigma)
    t = np.linspace(0, 600, 301)
    y0 = [0.3, 0.5]
    ref = _reference(lambda _, y: [-koff * y[0] + 2 * koff / s * y[1],
                                   -2 * koff / s * y[1]], y0, t)
    lib = solve_two_site_cooperative_dissociation(t, koff, sigma, 1.0, 1.7, *y0)
    np.testing.assert_allclose(lib[:, 0], ref @ [1.0, 1.7], rtol=1e-8)


# t0 delays the start of the association: the curve at time t equals the
# curve without t0 at time t - t0. Before the fix t0 had no effect.
@pytest.mark.parametrize("cooperative", [False, True])
def test_two_site_association_honours_t0(cooperative):
    t = np.linspace(0, 300, 301)
    t0 = 20.0
    k = int(t0)  # t[k] == t0

    def solve(time, **kw):
        if cooperative:
            return solve_two_site_cooperative_association(time, 0.5, 0.02, 0.01, 3.0, 1.0, 1.5, **kw)
        return solve_two_site_association(time, 0.5, 0.02, 0.01, 1.0, 1.5, **kw)

    shifted = solve(t, t0=t0)
    expected = solve(t[k:] - t0)
    np.testing.assert_allclose(shifted[k:], expected, rtol=1e-10)
    assert not np.allclose(shifted[:, 0], solve(t)[:, 0], rtol=1e-3)


# Same t0 behaviour for the induced-fit association.
def test_induced_fit_association_honours_t0():
    t = np.linspace(0, 300, 301)
    t0 = 20.0
    k = int(t0)
    args = (0.5, 0.02, 0.01, 0.03, 0.005)
    shifted = solve_induced_fit_association(t, *args, smax=2.0, t0=t0)
    expected = solve_induced_fit_association(t[k:] - t0, *args, smax=2.0)
    np.testing.assert_allclose(shifted[k:], expected, rtol=1e-10)


# The two-site fitter recovers a non-zero t0 when it is fitted.
def test_two_site_fit_recovers_t0():
    Kd, koff, t0 = 0.4, 0.01, 6.0
    concs = [0.2, 0.8, 3.2]
    t_assoc = np.arange(0, 300.0, 1.0)
    t_disso = np.arange(300.0, 900.0, 2.0)
    assoc, disso = [], []
    for c in concs:
        y = solve_two_site_association(t_assoc, c, koff / Kd, koff, 1.0, 1.6, t0=t0)
        assoc.append(y[:, 0])
        f_pl, f_lpl = y[-1, 1] / 1.0, y[-1, 2] / 1.6
        disso.append(solve_two_site_cooperative_dissociation(t_disso, koff, 1.0, 1.0, 1.6, f_pl, f_lpl)[:, 0])

    p0 = [0.6, 0.008, 1.0, 1.2, 1.2]
    low = [1e-3, 1e-4, -20.0, 0.1, 0.1]
    high = [10.0, 1.0, 20.0, 10.0, 10.0]
    params, *_ = fit_two_site_assoc_and_disso(
        assoc, [t_assoc] * 3, concs, disso, [t_disso] * 3,
        p0, low, high, smax_idx=[0, 0, 0], shared_smax=True, fixed_t0=False)

    np.testing.assert_allclose(params[2], t0, rtol=1e-3)
    np.testing.assert_allclose(params[:2], [Kd, koff], rtol=1e-3)


# A trace that does not follow straight after the previous dissociation
# starts from an empty surface, even when its time does not start near 0.
def test_induced_fit_fitter_restarts_after_gap():
    kon, koff, kc, krev, smax = 0.05, 0.01, 0.03, 0.005, 2.0
    concs = [0.5, 1.0]
    t_assoc = [np.arange(0, 300.0, 2.0), np.arange(1000.0, 1300.0, 2.0)]
    t_disso = [np.arange(300.0, 600.0, 2.0), np.arange(1300.0, 1600.0, 2.0)]
    assoc, disso = [], []
    for c, ta, td in zip(concs, t_assoc, t_disso):
        y = solve_induced_fit_association(ta, c, kon, koff, kc, krev, smax=smax)
        assoc.append(y[:, 0])
        disso.append(solve_induced_fit_dissociation(td, koff, kc, krev, s0=y[-1, 0], sP2L=y[-1, 2], smax=smax)[:, 0])

    params, _, fit_assoc, _ = fit_induced_fit_sites_assoc_and_disso(
        assoc, t_assoc, concs, disso, t_disso,
        [1.5], [0.1], [10.0], smax_idx=[0, 0], shared_smax=True,
        fixed_kon1=True, kon1_value=kon, fixed_koff1=True, koff1_value=koff,
        fixed_kon2=True, kon2_value=kc, fixed_koff2=True, koff2_value=krev)

    np.testing.assert_allclose(params[0], smax, rtol=1e-6)
    assert _max_rel_error(fit_assoc[1], assoc[1]) < 1e-6


# Induced fit in solution: the two kobs values are the two relaxation rates
# of the reaction equations linearised around equilibrium.
@pytest.mark.parametrize("k1, km1, k2, km2, E, L", [
    (0.3, 0.02, 0.05, 0.004, 1.0, 3.0),
    (0.3, 0.02, 0.004, 0.05, 1.0, 3.0),
    (0.3, 0.02, 0.05, 0.05, 1.0, 3.0),
    (1.3, 0.4, 0.7, 0.2, 2.4, 1.2),
    (1.0, 0.1, 0.1, 0.01, 1.0, 1.0),
    (0.5, 0.2, 0.02, 0.3, 0.1, 10.0),
])
def test_kobs_induced_fit_both_rates(k1, km1, k2, km2, E, L):
    # AI-generated regression: compare correctly named rates with the ODE Jacobian.
    f = lambda t, y: ode_induced_fit_insolution(t, y, k1, km1, k2, km2, E, L)
    y_eq = _reference(f, [0.0, 0.0], np.array([0.0, 1e6]))[-1]
    rates = np.sort(-np.linalg.eigvals(_jacobian(f, y_eq)).real)
    slow = get_kobs_induced_fit(L, E, kr=km2, ke=k2, kon=k1, koff=km1, dominant=True)
    fast = get_kobs_induced_fit(L, E, kr=km2, ke=k2, kon=k1, koff=km1, dominant=False)
    assert 0 < slow <= fast
    np.testing.assert_allclose([slow, fast], rates, rtol=1e-6)


# Conformational selection in solution: same check for both kobs values.
def test_kobs_conformational_selection_both_rates():
    k1, km1, k2, km2, E, L = 0.02, 0.05, 0.3, 0.01, 1.0, 3.0
    f = lambda t, y: ode_conformational_selection_insolution(t, y, k1, km1, k2, km2, E, L)
    y0 = [E * k1 / (k1 + km1), 0.0]
    y_eq = _reference(f, y0, np.array([0.0, 1e6]))[-1]
    rates = np.sort(-np.linalg.eigvals(_jacobian(f, y_eq)).real)
    slow = get_kobs_conformational_selection(L, E, km1, k1, k2, km2, dominant=True)
    fast = get_kobs_conformational_selection(L, E, km1, k1, k2, km2, dominant=False)
    np.testing.assert_allclose([slow, fast], rates, rtol=1e-6)


# The analyte-mixture model is solved tightly enough for finite-difference
# fitting: it matches the reaction equations to about 1e-8.
def test_mixture_analyte_tolerance():
    C, F = 1.0, np.array([0.6, 0.4])
    Rm = np.array([1.0, 2.0])
    koffs, Kds = np.array([0.01, 0.002]), np.array([0.5, 0.05])
    kons = koffs / Kds
    t = np.linspace(0, 600, 301)

    ref = _reference(lambda _, R: kons * F * C * Rm * (1 - np.sum(R / Rm)) - koffs * R, [0.0, 0.0], t)
    lib = solve_ode_mixture_analyte_association(t, [0.0, 0.0], C, F, Rm, koffs, Kds).T
    assert _max_rel_error(lib, ref) < 1e-6

    ref_d = ref[-1] * np.exp(-np.outer(t, koffs))
    lib_d = solve_ode_mixture_analyte_dissociation(t, ref[-1], koffs).T
    np.testing.assert_allclose(lib_d, ref_d, rtol=1e-6)
