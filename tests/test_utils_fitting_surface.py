import numpy as np
import pytest

from numpy.testing import assert_almost_equal

from pykingenie.utils.fitting_surface import (
    guess_initial_signal,
    fit_one_site_dissociation,
    fit_two_site_assoc_and_disso,
    fit_induced_fit_sites_assoc_and_disso,
    fit_one_site_assoc_and_disso_ktr,
)
from pykingenie.utils.signal_surface import (
    solve_two_site_association,
    solve_two_site_dissociation,
    solve_two_site_cooperative_association,
    solve_two_site_cooperative_dissociation,
    solve_induced_fit_association,
    solve_induced_fit_dissociation,
    solve_ode_one_site_mass_transport_association,
    solve_ode_one_site_mass_transport_dissociation,
)

def test_guess_initial_signal_exception():

    assoc_time_lst = [[10]]
    assoc_signal_lst = [[10]]

    s0s = guess_initial_signal(assoc_time_lst,assoc_signal_lst)

    assert s0s[0] == 10, "The initial signal should be the same as the signal at the first time point"

def test_fit_one_site_dissociation_fixed_s0():

    t = np.linspace(0, 10, 50)
    s0 = 1.0
    k_off = 0.1

    y = s0 * np.exp(-k_off*t)

    initial_parameters = [0.5]

    low_bounds = [0.005]
    high_bounds = [10]

    fit_params, _, _ = fit_one_site_dissociation(
        [y], [t], initial_parameters, low_bounds, high_bounds,
        fit_s0=False)

    assert_almost_equal(fit_params[0],k_off,decimal=2)


def test_fit_two_site_assoc_and_disso_non_cooperative_recovery():

    time_assoc = np.linspace(0, 120, 80)
    time_disso = np.linspace(0, 120, 80)

    a = 1.2
    Kd_true = 0.2
    koff_true = 0.05
    kon_true = koff_true / Kd_true
    rmax_pl_true = 5.0
    rmax_lpl_true = 10.0

    assoc_mat = solve_two_site_association(
        time_assoc, a, kon_true, koff_true,
        Rmax_PL=rmax_pl_true, Rmax_LPL=rmax_lpl_true,
    )
    disso_mat = solve_two_site_dissociation(
        time_disso, koff_true,
        Rmax_PL=rmax_pl_true, Rmax_LPL=rmax_lpl_true,
        fPL_0=assoc_mat[-1, 1] / rmax_pl_true,
        fLPL_0=assoc_mat[-1, 2] / rmax_lpl_true,
    )

    fit, _, _, _ = fit_two_site_assoc_and_disso(
        assoc_signal_lst=[assoc_mat[:, 0]],
        assoc_time_lst=[time_assoc],
        analyte_conc_lst=[a],
        disso_signal_lst=[disso_mat[:, 0]],
        disso_time_lst=[time_disso],
        initial_parameters=[0.4, 0.08, 4.0, 9.0],
        low_bounds=[1e-6, 1e-6, 0.0, 0.0],
        high_bounds=[1e3, 1e2, 1e2, 1e2],
        smax_idx=[0],
        shared_smax=False,
        fixed_t0=True,
    )

    assert np.isclose(fit[0], Kd_true, rtol=0.1)
    assert np.isclose(fit[1], koff_true, rtol=0.1)
    assert np.isclose(fit[2], rmax_pl_true, rtol=0.1)
    assert np.isclose(fit[3], rmax_lpl_true, rtol=0.1)


def test_fit_two_site_assoc_and_disso_cooperative_recovery():

    time_assoc = np.linspace(0, 120, 80)
    time_disso = np.linspace(0, 120, 80)

    a = 0.9
    Kd_true = 0.3
    koff_true = 0.04
    sigma_true = 3.0
    kon_true = koff_true / Kd_true
    rmax_pl_true = 4.0
    rmax_lpl_true = 9.0

    assoc_mat = solve_two_site_cooperative_association(
        time_assoc, a, kon_true, koff_true, sigma_true,
        Rmax_PL=rmax_pl_true, Rmax_LPL=rmax_lpl_true,
    )
    disso_mat = solve_two_site_cooperative_dissociation(
        time_disso, koff_true, sigma_true,
        Rmax_PL=rmax_pl_true, Rmax_LPL=rmax_lpl_true,
        fPL_0=assoc_mat[-1, 1] / rmax_pl_true,
        fLPL_0=assoc_mat[-1, 2] / rmax_lpl_true,
    )

    fit, _, _, _ = fit_two_site_assoc_and_disso(
        assoc_signal_lst=[assoc_mat[:, 0]],
        assoc_time_lst=[time_assoc],
        analyte_conc_lst=[a],
        disso_signal_lst=[disso_mat[:, 0]],
        disso_time_lst=[time_disso],
        initial_parameters=[0.5, 0.06, 1.5, 3.0, 7.0],
        low_bounds=[1e-6, 1e-6, 1e-6, 0.0, 0.0],
        high_bounds=[1e3, 1e2, 1e2, 1e2, 1e2],
        smax_idx=[0],
        shared_smax=False,
        fixed_t0=True,
        fit_sigma=True,
    )

    assert np.isclose(fit[0], Kd_true, rtol=0.2)
    assert np.isclose(fit[1], koff_true, rtol=0.2)
    assert np.isclose(fit[2], sigma_true, rtol=0.2)
    assert np.isclose(fit[3], rmax_pl_true, rtol=0.2)
    assert np.isclose(fit[4], rmax_lpl_true, rtol=0.2)


def test_fit_two_site_assoc_and_disso_default_smax_idx_and_t0_and_shared_smax():

    Kd_true = 0.25
    koff_true = 0.04
    kon_true = koff_true / Kd_true

    t0_1, t0_2 = 0.2, -0.1
    a1, a2 = 1.0, 0.8

    rmax_pl_1, rmax_lpl_1 = 5.0, 10.0
    rmax_pl_2, rmax_lpl_2 = 6.0, 11.0

    t_assoc_1 = np.linspace(0, 60, 70)
    t_disso_1 = np.linspace(0, 60, 70)
    # continuous with previous dissociation and >2 to trigger the elif branch
    t_assoc_2 = np.linspace(61, 121, 70)
    t_disso_2 = np.linspace(0, 60, 70)

    assoc_1 = solve_two_site_association(
        t_assoc_1 - t_assoc_1[0] - t0_1, a1, kon_true, koff_true,
        Rmax_PL=rmax_pl_1, Rmax_LPL=rmax_lpl_1,
    )
    disso_1 = solve_two_site_dissociation(
        t_disso_1, koff_true,
        Rmax_PL=rmax_pl_1, Rmax_LPL=rmax_lpl_1,
        fPL_0=assoc_1[-1, 1] / rmax_pl_1,
        fLPL_0=assoc_1[-1, 2] / rmax_lpl_1,
    )

    fpl2_0 = disso_1[-1, 1] / rmax_pl_1
    flpl2_0 = disso_1[-1, 2] / rmax_lpl_1

    assoc_2 = solve_two_site_association(
        t_assoc_2 - t_assoc_2[0] - t0_2, a2, kon_true, koff_true,
        Rmax_PL=rmax_pl_2, Rmax_LPL=rmax_lpl_2,
        fPL_0=fpl2_0, fLPL_0=flpl2_0,
    )
    disso_2 = solve_two_site_dissociation(
        t_disso_2, koff_true,
        Rmax_PL=rmax_pl_2, Rmax_LPL=rmax_lpl_2,
        fPL_0=assoc_2[-1, 1] / rmax_pl_2,
        fLPL_0=assoc_2[-1, 2] / rmax_lpl_2,
    )

    # smax_idx omitted on purpose to hit default assignment
    fit, _, _, _ = fit_two_site_assoc_and_disso(
        assoc_signal_lst=[assoc_1[:, 0], assoc_2[:, 0]],
        assoc_time_lst=[t_assoc_1, t_assoc_2],
        analyte_conc_lst=[a1, a2],
        disso_signal_lst=[disso_1[:, 0], disso_2[:, 0]],
        disso_time_lst=[t_disso_1, t_disso_2],
        initial_parameters=[0.3, 0.05, 0.0, 0.0, 4.0, 9.0, 5.0, 10.0],
        low_bounds=[1e-6, 1e-6, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        high_bounds=[1e3, 1e2, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0],
        shared_smax=True,
        fixed_t0=False,
        fit_sigma=False,
    )

    assert len(fit) == 8


def test_fit_two_site_assoc_and_disso_noncontinuous_branch():

    Kd_true = 0.2
    koff_true = 0.03
    kon_true = koff_true / Kd_true

    t_assoc_1 = np.linspace(0, 60, 70)
    t_disso_1 = np.linspace(0, 60, 70)
    # far from previous dissociation end -> non-continuous branch
    t_assoc_2 = np.linspace(120, 180, 70)
    t_disso_2 = np.linspace(0, 60, 70)

    assoc_1 = solve_two_site_association(
        t_assoc_1 - t_assoc_1[0], 1.0, kon_true, koff_true,
        Rmax_PL=5.0, Rmax_LPL=10.0,
    )
    disso_1 = solve_two_site_dissociation(
        t_disso_1, koff_true,
        Rmax_PL=5.0, Rmax_LPL=10.0,
        fPL_0=assoc_1[-1, 1] / 5.0,
        fLPL_0=assoc_1[-1, 2] / 10.0,
    )
    assoc_2 = solve_two_site_association(
        t_assoc_2 - t_assoc_2[0], 1.0, kon_true, koff_true,
        Rmax_PL=6.0, Rmax_LPL=11.0,
    )
    disso_2 = solve_two_site_dissociation(
        t_disso_2, koff_true,
        Rmax_PL=6.0, Rmax_LPL=11.0,
        fPL_0=assoc_2[-1, 1] / 6.0,
        fLPL_0=assoc_2[-1, 2] / 11.0,
    )

    fit, _, _, _ = fit_two_site_assoc_and_disso(
        assoc_signal_lst=[assoc_1[:, 0], assoc_2[:, 0]],
        assoc_time_lst=[t_assoc_1, t_assoc_2],
        analyte_conc_lst=[1.0, 1.0],
        disso_signal_lst=[disso_1[:, 0], disso_2[:, 0]],
        disso_time_lst=[t_disso_1, t_disso_2],
        initial_parameters=[0.25, 0.04, 5.0, 10.0, 6.0, 11.0],
        low_bounds=[1e-6, 1e-6, 0.0, 0.0, 0.0, 0.0],
        high_bounds=[1e3, 1e2, 100.0, 100.0, 100.0, 100.0],
        smax_idx=[0, 1],
        shared_smax=False,
        fixed_t0=True,
        fit_sigma=False,
    )

    assert len(fit) == 6


def test_fit_induced_fit_assoc_and_disso_uses_previous_state_branch():

    kon1, koff1, kon2, koff2 = 0.1, 0.02, 1.0, 10.0
    smax1, smax2 = 5.0, 5.0
    a1, a2 = 1.0, 1.0

    t_assoc_1 = np.linspace(0, 40, 60)
    t_disso_1 = np.linspace(0, 40, 60)
    t_assoc_2 = np.linspace(45, 85, 60)
    t_disso_2 = np.linspace(0, 40, 60)

    assoc_1 = solve_induced_fit_association(t_assoc_1, a1, kon1, koff1, kon2, koff2, smax=smax1)
    disso_1 = solve_induced_fit_dissociation(
        t_disso_1, koff1, kon2, koff2,
        s0=assoc_1[-1, 0], sP2L=assoc_1[-1, 2], smax=smax1,
    )

    assoc_2 = solve_induced_fit_association(
        t_assoc_2, a2, kon1, koff1, kon2, koff2,
        sP1L=disso_1[-1, 1], sP2L=disso_1[-1, 2], smax=smax2,
    )
    disso_2 = solve_induced_fit_dissociation(
        t_disso_2, koff1, kon2, koff2,
        s0=assoc_2[-1, 0], sP2L=assoc_2[-1, 2], smax=smax2,
    )

    true_params = np.array([kon1, koff1, kon2, koff2, smax1, smax2], dtype=float)
    eps = np.array([1e-4, 1e-4, 1e-3, 1e-2, 1e-3, 1e-3], dtype=float)

    fit, _, _, _ = fit_induced_fit_sites_assoc_and_disso(
        assoc_signal_lst=[assoc_1[:, 0], assoc_2[:, 0]],
        assoc_time_lst=[t_assoc_1, t_assoc_2],
        analyte_conc_lst=[a1, a2],
        disso_signal_lst=[disso_1[:, 0], disso_2[:, 0]],
        disso_time_lst=[t_disso_1, t_disso_2],
        initial_parameters=true_params.tolist(),
        low_bounds=(true_params - eps).tolist(),
        high_bounds=(true_params + eps).tolist(),
        smax_idx=[0, 1],
        shared_smax=False,
        fixed_t0=True,
    )

    assert len(fit) == 6


def test_fit_one_site_assoc_and_disso_ktr_uses_previous_disso_end_branch():

    Kd_true = 0.2
    koff_true = 0.02
    ktr1 = 0.005
    ktr2 = 0.005
    smax1 = 5.0
    smax2 = 5.0

    t_assoc_1 = np.linspace(0, 40, 60)
    t_disso_1 = np.linspace(0, 40, 60)
    t_assoc_2 = np.linspace(45, 85, 60)
    t_disso_2 = np.linspace(0, 40, 60)

    assoc_1 = solve_ode_one_site_mass_transport_association(
        t_assoc_1 - t_assoc_1[0], 0.0, 0.5, 1.0, Kd_true, koff_true, ktr1, smax1
    )
    disso_1 = solve_ode_one_site_mass_transport_dissociation(
        t_disso_1, assoc_1[-1], Kd_true, koff_true, ktr1, smax1
    )

    assoc_2 = solve_ode_one_site_mass_transport_association(
        t_assoc_2 - t_assoc_2[0], disso_1[-1], 0.5, 1.0, Kd_true, koff_true, ktr2, smax2
    )
    disso_2 = solve_ode_one_site_mass_transport_dissociation(
        t_disso_2, assoc_2[-1], Kd_true, koff_true, ktr2, smax2
    )

    fit, _, _, _ = fit_one_site_assoc_and_disso_ktr(
        assoc_signal_lst=[assoc_1, assoc_2],
        assoc_time_lst=[t_assoc_1, t_assoc_2],
        analyte_conc_lst=[1.0, 1.0],
        disso_signal_lst=[disso_1, disso_2],
        disso_time_lst=[t_disso_1, t_disso_2],
        initial_parameters=[0.25, 0.03, 0.004, 0.004, 4.0, 4.0],
        low_bounds=[1e-6, 1e-6, 1e-8, 1e-8, 0.0, 0.0],
        high_bounds=[1e2, 1e2, 1.0, 1.0, 100.0, 100.0],
        smax_idx=[0, 1],
        shared_smax=False,
        fixed_t0=True,
    )

    assert len(fit) == 6