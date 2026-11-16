# This code is AI generated.
"""
Closed-form (Lambert W) solution of the mass transport-limited one-site model,
checked against numerical integration of the same ODE over a wide range of
k_on, k_off and k_tr, and used by the mass transport fit.
"""
import itertools

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import pykingenie.utils.fitting_surface as fitting_surface
from pykingenie.utils.fitting_surface import fit_one_site_assoc_and_disso_ktr
from pykingenie.utils.signal_surface import (
    ode_one_site_mass_transport,
    one_site_association_analytical,
    one_site_dissociation_analytical,
    one_site_mass_transport_analytical,
    one_site_mass_transport_association_analytical,
    one_site_mass_transport_dissociation_analytical,
    solve_ode_one_site_mass_transport_association,
    solve_ode_one_site_mass_transport_dissociation,
)

SMAX = 1.0
T = np.linspace(0, 600, 301)

KON  = [1e-2, 1.0, 1e2]                   # 1/(µM s)
KOFF = [1e-4, 1e-2, 1.0]                  # 1/s
KTR  = [1e-3, 1e-1, 1e1, 1e3, 1e6]        # signal/(µM s): from strongly transport-limited to none
CONC = [0.01, 1.0, 100.0]                 # µM


def _reference_ode(t, s0, conc, Kd, koff, ktr, smax):
    """The ODE integrated with tight tolerances, as reference."""
    out = solve_ivp(ode_one_site_mass_transport, [t[0], t[-1]], [s0], t_eval=t,
                    args=([Kd, koff, ktr, smax], conc), method="Radau",
                    rtol=1e-12, atol=1e-14 * smax)
    return out.y[0]


@pytest.mark.parametrize("kon,koff,ktr", list(itertools.product(KON, KOFF, KTR)))
def test_matches_tight_ode(kon, koff, ktr):
    """Association from an empty surface, then dissociation, match the ODE to 1e-9 of Smax."""
    Kd = koff / kon
    for conc in CONC:
        assoc = one_site_mass_transport_association_analytical(T, 0.0, conc, Kd, koff, ktr, SMAX)
        ref_a = _reference_ode(T, 0.0, conc, Kd, koff, ktr, SMAX)
        np.testing.assert_allclose(assoc, ref_a, rtol=0, atol=1e-9 * SMAX)

        disso = one_site_mass_transport_dissociation_analytical(T, assoc[-1], Kd, koff, ktr, SMAX)
        ref_d = _reference_ode(T, assoc[-1], 0.0, Kd, koff, ktr, SMAX)
        np.testing.assert_allclose(disso, ref_d, rtol=0, atol=1e-9 * SMAX)


@pytest.mark.parametrize("kon,koff,ktr", list(itertools.product(KON, KOFF, KTR)))
def test_matches_package_ode_solvers(kon, koff, ktr):
    """Drop-in replacement for the ODE solvers previously used by the fit."""
    Kd = koff / kon
    for conc in CONC:
        a_ode = solve_ode_one_site_mass_transport_association(T, 0.0, conc, Kd, koff, ktr, SMAX)
        a_ana = one_site_mass_transport_association_analytical(T, 0.0, conc, Kd, koff, ktr, SMAX)
        np.testing.assert_allclose(a_ana, a_ode, rtol=0, atol=1e-6 * SMAX)

        d_ode = solve_ode_one_site_mass_transport_dissociation(T, a_ode[-1], Kd, koff, ktr, SMAX)
        d_ana = one_site_mass_transport_dissociation_analytical(T, a_ode[-1], Kd, koff, ktr, SMAX)
        np.testing.assert_allclose(d_ana, d_ode, rtol=0, atol=1e-6 * SMAX)


@pytest.mark.parametrize("kon,koff,ktr", list(itertools.product(KON, KOFF, KTR)))
def test_derivative_satisfies_ode(kon, koff, ktr):
    """The time derivative of the closed form equals the right-hand side of the ODE."""
    Kd = koff / kon
    conc = Kd
    # Time step small compared with the fastest possible rate, k_on * c + k_off
    t = np.linspace(0, 5 / (kon * conc + koff), 4001)
    s = one_site_mass_transport_analytical(t, 0.1 * SMAX, conc, Kd, koff, ktr, SMAX)
    ds_dt = np.gradient(s, t, edge_order=2)
    rhs = ode_one_site_mass_transport(t, s, [Kd, koff, ktr, SMAX], conc)
    scale = max(np.max(np.abs(rhs)), 1e-12)
    # Central differences: error of order h^2 * S''' relative to the largest rate
    assert np.max(np.abs(ds_dt - rhs))[()] / scale < 1e-3


def test_initial_value_and_equilibrium():
    """Starts at s0 and ends at the 1:1 equilibrium, whatever the transport rate."""
    Kd, koff, conc = 0.5, 0.05, 1.0
    for ktr in KTR:
        s = one_site_mass_transport_association_analytical(np.linspace(0, 1e6, 50), 0.2, conc, Kd, koff, ktr, SMAX)
        assert s[0] == pytest.approx(0.2, abs=1e-12)
        assert s[-1] == pytest.approx(SMAX * conc / (conc + Kd), rel=1e-9)


def test_starting_at_equilibrium_stays_constant():
    Kd, koff, conc = 0.5, 0.05, 1.0
    s_eq = SMAX * conc / (conc + Kd)
    s = one_site_mass_transport_association_analytical(T, s_eq, conc, Kd, koff, 0.01, SMAX)
    np.testing.assert_allclose(s, s_eq, rtol=1e-12)


def test_infinite_ktr_is_the_one_to_one_model():
    Kd, koff, conc = 0.5, 0.05, 1.0
    assoc = one_site_mass_transport_association_analytical(T, 0.0, conc, Kd, koff, np.inf, SMAX)
    np.testing.assert_allclose(assoc, one_site_association_analytical(T, 0.0, SMAX, koff, Kd, conc), atol=1e-14)
    disso = one_site_mass_transport_dissociation_analytical(T, 0.7, Kd, koff, np.inf, SMAX)
    np.testing.assert_allclose(disso, one_site_dissociation_analytical(T, 0.7, koff), atol=1e-14)


def test_very_large_ktr_approaches_one_to_one_model():
    """k_on / k_tr around 1e-14: no loss of precision from dividing by a tiny k."""
    Kd, koff, conc = 0.5, 0.05, 1.0
    s = one_site_mass_transport_association_analytical(T, 0.0, conc, Kd, koff, 1e13, SMAX)
    np.testing.assert_allclose(s, one_site_association_analytical(T, 0.0, SMAX, koff, Kd, conc), atol=1e-10)


def test_extreme_transport_limitation_does_not_overflow():
    """k_on * Smax / k_tr = 1e6: the direct Lambert W argument exp(1e6) would overflow."""
    Kd, koff, ktr, smax = 1e-3, 1e-2, 1e-2, 1e3
    t = np.linspace(0, 3000, 301)
    for conc in (1e-3, 1.0):
        s = one_site_mass_transport_association_analytical(t, 0.0, conc, Kd, koff, ktr, smax)
        assert np.all(np.isfinite(s))
        np.testing.assert_allclose(s, _reference_ode(t, 0.0, conc, Kd, koff, ktr, smax), rtol=0, atol=1e-8 * smax)


def test_dissociation_from_saturation_with_strong_transport_limitation():
    """Starts next to the Lambert W branch point (y close to -1), where direct evaluation loses precision."""
    Kd, koff, ktr = 1e-3, 1e-2, 1e-3
    for s0 in (SMAX, SMAX * (1 - 1e-9), 0.999 * SMAX):
        s = one_site_mass_transport_dissociation_analytical(T, s0, Kd, koff, ktr, SMAX)
        np.testing.assert_allclose(s, _reference_ode(T, s0, 0.0, Kd, koff, ktr, SMAX), rtol=0, atol=1e-9 * SMAX)


def test_times_need_not_start_at_zero():
    """With t0 = 0 the initial signal is at the first time point, as in the ODE solvers."""
    Kd, koff, ktr = 0.5, 0.05, 0.02
    t = np.arange(300.0, 900.0, 3.0)
    np.testing.assert_allclose(
        one_site_mass_transport_dissociation_analytical(t, 0.6, Kd, koff, ktr, SMAX),
        solve_ode_one_site_mass_transport_dissociation(t, 0.6, Kd, koff, ktr, SMAX),
        rtol=0, atol=1e-6)


@pytest.mark.parametrize("s0,conc,ktr", [(0.2, 1.0, 0.02), (0.9, 0.1, 0.02), (0.2, 1.0, 1e3)])
def test_t0_shifts_the_start(s0, conc, ktr):
    """The signal is s0 at time t0, as in the 1:1 model. Earlier points follow the ODE backward in time."""
    Kd, koff, t0 = 0.5, 0.05, 30.0
    t = np.linspace(0, 300, 301)
    s = one_site_mass_transport_association_analytical(t, s0, conc, Kd, koff, ktr, SMAX, t0)

    after = t >= t0
    np.testing.assert_allclose(
        s[after], one_site_mass_transport_association_analytical(t[after] - t0, s0, conc, Kd, koff, ktr, SMAX),
        rtol=0, atol=1e-12)

    t_back = t[~after][::-1]
    ref = solve_ivp(ode_one_site_mass_transport, [t0, 0.0], [s0], t_eval=t_back,
                    args=([Kd, koff, ktr, SMAX], conc), method="Radau", rtol=1e-12, atol=1e-14).y[0]
    np.testing.assert_allclose(s[~after][::-1], ref, rtol=0, atol=1e-9)


def test_backward_past_full_surface_stays_finite():
    """Backward in time from above equilibrium the surface fills; before that the signal follows the clipped ODE."""
    Kd, koff, ktr, conc, s0, t0 = 0.5, 0.05, 1e-3, 0.01, 0.99, 200.0
    for k_tr in (ktr, np.inf):
        t = np.linspace(0, 300, 301)
        s = one_site_mass_transport_association_analytical(t, s0, conc, Kd, koff, k_tr, SMAX, t0)
        assert np.all(np.isfinite(s))
        assert np.max(s) > SMAX
        before = t <= t0
        ref = solve_ivp(ode_one_site_mass_transport, [t0, 0.0], [s0], t_eval=t[before][::-1],
                        args=([Kd, koff, k_tr, SMAX], conc), method="Radau", rtol=1e-12, atol=1e-14).y[0]
        np.testing.assert_allclose(s[before][::-1], ref, rtol=1e-7)


def test_fit_uses_closed_form(monkeypatch):
    """The fit no longer calls the ODE solver, and recovers parameters from ODE-simulated data."""
    Kd, koff, ktr, smax = 0.5, 0.01, 0.05, 5.0
    concs = [0.1, 0.5, 2.0]
    t_assoc = [np.arange(0, 300.0, 2.0)] * 3
    t_disso = [np.arange(300.0, 900.0, 3.0)] * 3
    assoc, disso = [], []
    for c, ta, td in zip(concs, t_assoc, t_disso):
        a = solve_ode_one_site_mass_transport_association(ta, 0.0, c, Kd, koff, ktr, smax)
        assoc.append(a)
        disso.append(solve_ode_one_site_mass_transport_dissociation(td, a[-1], Kd, koff, ktr, smax))

    def _fail(*args, **kwargs):
        raise AssertionError("the fit should not integrate the ODE")
    monkeypatch.setattr(fitting_surface, "solve_ode_one_site_mass_transport_association", _fail, raising=False)
    monkeypatch.setattr(fitting_surface, "solve_ode_one_site_mass_transport_dissociation", _fail, raising=False)

    params, _, fit_assoc, fit_disso = fit_one_site_assoc_and_disso_ktr(
        assoc, t_assoc, concs, disso, t_disso,
        initial_parameters=[1.0, 0.005, 0.1, 4.0],
        low_bounds=[1e-3, 1e-5, 1e-4, 0.5], high_bounds=[1e2, 1.0, 10.0, 50.0],
        smax_idx=[0, 0, 0], shared_smax=True)

    np.testing.assert_allclose(params, [Kd, koff, ktr, smax], rtol=1e-3)
    for fitted, data in zip(fit_assoc + fit_disso, assoc + disso):
        np.testing.assert_allclose(fitted, data, atol=1e-5 * smax)
