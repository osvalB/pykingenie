import numpy as np

from pykingenie.utils.fitting_surface import fit_steady_state_two_site
from pykingenie.utils.signal_surface import (
    steady_state_two_site,
    steady_state_two_site_cooperative,
)


def _noisy(y, rel_noise=0.01, seed=123):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, rel_noise * np.max(np.abs(y)), size=len(y))
    return y + noise


def test_fit_two_site_non_cooperative_single_trace_noise_free():
    C = np.logspace(-3, 2, 60)
    Kd_true = 0.25
    Rmax_PL_true = 7.0
    Rmax_LPL_true = 11.0

    y = steady_state_two_site(C, Rmax_PL_true, Rmax_LPL_true, Kd_true)
    fit, _, fit_vals = fit_steady_state_two_site(
        signal_lst=[y],
        ligand_lst=[C],
        initial_parameters=[0.6, 4.0, 8.0],
        low_bounds=[1e-6, 0.0, 0.0],
        high_bounds=[1e3, 100.0, 100.0],
    )

    assert np.isclose(fit[0], Kd_true, rtol=1e-3)
    assert np.isclose(fit[1], Rmax_PL_true, rtol=1e-3)
    assert np.isclose(fit[2], Rmax_LPL_true, rtol=1e-3)
    np.testing.assert_allclose(fit_vals[0], y, atol=1e-8)


def test_fit_two_site_non_cooperative_multi_trace_shared_kd_noise_free():
    C1 = np.logspace(-3, 2, 45)
    C2 = np.logspace(-2, 1.5, 35)

    Kd_true = 0.15
    Rmax_PL_1, Rmax_LPL_1 = 5.0, 9.0
    Rmax_PL_2, Rmax_LPL_2 = 8.0, 14.0

    y1 = steady_state_two_site(C1, Rmax_PL_1, Rmax_LPL_1, Kd_true)
    y2 = steady_state_two_site(C2, Rmax_PL_2, Rmax_LPL_2, Kd_true)
    fit, _, _ = fit_steady_state_two_site(
        signal_lst=[y1, y2],
        ligand_lst=[C1, C2],
        initial_parameters=[0.3, 4.0, 8.0, 6.0, 12.0],
        low_bounds=[1e-6, 0.0, 0.0, 0.0, 0.0],
        high_bounds=[1e3, 100.0, 100.0, 100.0, 100.0],
    )

    assert np.isclose(fit[0], Kd_true, rtol=1e-3)
    assert np.isclose(fit[1], Rmax_PL_1, rtol=1e-3)
    assert np.isclose(fit[2], Rmax_LPL_1, rtol=1e-3)
    assert np.isclose(fit[3], Rmax_PL_2, rtol=1e-3)
    assert np.isclose(fit[4], Rmax_LPL_2, rtol=1e-3)


def test_fit_two_site_non_cooperative_with_noise_parameters_close():
    C1 = np.logspace(-3, 2, 55)
    C2 = np.logspace(-3, 2, 55)

    Kd_true = 0.3
    Rmax_PL_1, Rmax_LPL_1 = 6.0, 10.0
    Rmax_PL_2, Rmax_LPL_2 = 9.0, 15.0

    y1 = steady_state_two_site(C1, Rmax_PL_1, Rmax_LPL_1, Kd_true)
    y2 = steady_state_two_site(C2, Rmax_PL_2, Rmax_LPL_2, Kd_true)

    y1n = _noisy(y1, rel_noise=0.01, seed=11)
    y2n = _noisy(y2, rel_noise=0.01, seed=22)
    fit, _, _ = fit_steady_state_two_site(
        signal_lst=[y1n, y2n],
        ligand_lst=[C1, C2],
        initial_parameters=[0.8, 5.0, 9.0, 7.0, 13.0],
        low_bounds=[1e-6, 0.0, 0.0, 0.0, 0.0],
        high_bounds=[1e3, 100.0, 100.0, 100.0, 100.0],
    )

    assert np.isclose(fit[0], Kd_true, rtol=0.20)
    assert np.isclose(fit[1], Rmax_PL_1, rtol=0.15)
    assert np.isclose(fit[2], Rmax_LPL_1, rtol=0.12)
    assert np.isclose(fit[3], Rmax_PL_2, rtol=0.15)
    assert np.isclose(fit[4], Rmax_LPL_2, rtol=0.12)


def test_fit_two_site_cooperative_single_trace_noise_free():
    C = np.logspace(-3, 2, 60)

    Kd_true = 0.2
    sigma_true = 4.0
    Rmax_PL_true = 6.5
    Rmax_LPL_true = 12.0

    y = steady_state_two_site_cooperative(C, Rmax_PL_true, Rmax_LPL_true, Kd_true, sigma_true)
    fit, _, fit_vals = fit_steady_state_two_site(
        signal_lst=[y],
        ligand_lst=[C],
        initial_parameters=[0.5, 2.0, 5.0, 10.0],
        low_bounds=[1e-6, 1e-6, 0.0, 0.0],
        high_bounds=[1e3, 1e3, 100.0, 100.0],
        fit_sigma=True,
    )

    assert np.isclose(fit[0], Kd_true, rtol=1e-3)
    assert np.isclose(fit[1], sigma_true, rtol=1e-3)
    assert np.isclose(fit[2], Rmax_PL_true, rtol=1e-3)
    assert np.isclose(fit[3], Rmax_LPL_true, rtol=1e-3)
    np.testing.assert_allclose(fit_vals[0], y, atol=1e-8)


def test_fit_two_site_cooperative_multi_trace_with_noise_parameters_close():
    C1 = np.logspace(-3, 2, 50)
    C2 = np.logspace(-2.5, 1.7, 40)

    Kd_true = 0.35
    sigma_true = 2.5
    Rmax_PL_1, Rmax_LPL_1 = 5.5, 10.5
    Rmax_PL_2, Rmax_LPL_2 = 8.5, 14.5

    y1 = steady_state_two_site_cooperative(C1, Rmax_PL_1, Rmax_LPL_1, Kd_true, sigma_true)
    y2 = steady_state_two_site_cooperative(C2, Rmax_PL_2, Rmax_LPL_2, Kd_true, sigma_true)

    y1n = _noisy(y1, rel_noise=0.01, seed=101)
    y2n = _noisy(y2, rel_noise=0.01, seed=202)
    fit, _, _ = fit_steady_state_two_site(
        signal_lst=[y1n, y2n],
        ligand_lst=[C1, C2],
        initial_parameters=[0.7, 1.2, 4.0, 9.0, 7.0, 13.0],
        low_bounds=[1e-6, 1e-6, 0.0, 0.0, 0.0, 0.0],
        high_bounds=[1e3, 1e3, 100.0, 100.0, 100.0, 100.0],
        fit_sigma=True,
    )

    assert np.isclose(fit[0], Kd_true, rtol=0.25)
    assert np.isclose(fit[1], sigma_true, rtol=0.35)
    assert np.isclose(fit[2], Rmax_PL_1, rtol=0.20)
    assert np.isclose(fit[3], Rmax_LPL_1, rtol=0.15)
    assert np.isclose(fit[4], Rmax_PL_2, rtol=0.20)
    assert np.isclose(fit[5], Rmax_LPL_2, rtol=0.15)


def test_fit_two_site_cooperative_fixed_kd_recovers_sigma_and_rmax():
    C = np.logspace(-3, 2, 60)

    Kd_true = 0.18
    sigma_true = 3.0
    Rmax_PL_true = 7.0
    Rmax_LPL_true = 13.0

    y = steady_state_two_site_cooperative(C, Rmax_PL_true, Rmax_LPL_true, Kd_true, sigma_true)
    fit, _, _ = fit_steady_state_two_site(
        signal_lst=[y],
        ligand_lst=[C],
        initial_parameters=[2.0, 5.0, 10.0],
        low_bounds=[1e-6, 0.0, 0.0],
        high_bounds=[1e3, 100.0, 100.0],
        fixed_Kd=True,
        Kd_value=Kd_true,
        fit_sigma=True,
    )

    assert np.isclose(fit[0], sigma_true, rtol=1e-3)
    assert np.isclose(fit[1], Rmax_PL_true, rtol=1e-3)
    assert np.isclose(fit[2], Rmax_LPL_true, rtol=1e-3)
