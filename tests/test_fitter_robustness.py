# This code is AI generated and reviewed by a human (PR #18).
"""
Fits that must recover known parameters from simulated data with noise:
starting points for mass transport, carry-over between steps, and an
induced-fit model whose rates are all determined by the curves.
"""
import numpy as np
import pytest

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
