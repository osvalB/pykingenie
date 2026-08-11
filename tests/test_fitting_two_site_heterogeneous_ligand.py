import numpy as np

from pykingenie.utils.fitting_surface import fit_steady_state_two_site_heterogeneous_ligand
from pykingenie.utils.signal_surface import steady_state_two_site_heterogeneous_ligand


def _noisy(y, rel_noise=0.01, seed=123):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, rel_noise * np.max(np.abs(y)), size=len(y))
    return y + noise


def test_fit_heterogeneous_ligand_single_trace_noise_free():
    C = np.logspace(-3, 2, 80)
    Kd1_true = 0.08
    Kd2_true = 4.0
    fraction_true = 0.35
    Rmax_true = 12.0

    y = steady_state_two_site_heterogeneous_ligand(
        C, Rmax_true, Kd1_true, Kd2_true, fraction_true
    )
    fit, _, fit_vals = fit_steady_state_two_site_heterogeneous_ligand(
        signal_lst=[y],
        ligand_lst=[C],
        initial_parameters=[0.15, 2.0, 0.5, 10.0],
        low_bounds=[1e-3, 1.0, 0.0, 0.0],
        high_bounds=[0.5, 50.0, 1.0, 100.0],
    )

    assert np.isclose(fit[0], Kd1_true, rtol=1e-3)
    assert np.isclose(fit[1], Kd2_true, rtol=1e-3)
    assert np.isclose(fit[2], fraction_true, rtol=1e-3)
    assert np.isclose(fit[3], Rmax_true, rtol=1e-3)
    np.testing.assert_allclose(fit_vals[0], y, atol=1e-8)


def test_fit_heterogeneous_ligand_multi_trace_shared_shape_noise_free():
    C1 = np.logspace(-3, 2, 80)
    C2 = np.logspace(-2.5, 1.8, 65)

    Kd1_true = 0.12
    Kd2_true = 6.0
    fraction_true = 0.4
    Rmax_1_true = 8.0
    Rmax_2_true = 14.0

    y1 = steady_state_two_site_heterogeneous_ligand(
        C1, Rmax_1_true, Kd1_true, Kd2_true, fraction_true
    )
    y2 = steady_state_two_site_heterogeneous_ligand(
        C2, Rmax_2_true, Kd1_true, Kd2_true, fraction_true
    )
    fit, _, fit_vals = fit_steady_state_two_site_heterogeneous_ligand(
        signal_lst=[y1, y2],
        ligand_lst=[C1, C2],
        initial_parameters=[0.2, 3.0, 0.6, 7.0, 12.0],
        low_bounds=[1e-3, 1.0, 0.0, 0.0, 0.0],
        high_bounds=[0.5, 50.0, 1.0, 100.0, 100.0],
    )

    assert np.isclose(fit[0], Kd1_true, rtol=1e-3)
    assert np.isclose(fit[1], Kd2_true, rtol=1e-3)
    assert np.isclose(fit[2], fraction_true, rtol=1e-3)
    assert np.isclose(fit[3], Rmax_1_true, rtol=1e-3)
    assert np.isclose(fit[4], Rmax_2_true, rtol=1e-3)
    np.testing.assert_allclose(fit_vals[0], y1, atol=1e-8)
    np.testing.assert_allclose(fit_vals[1], y2, atol=1e-8)


def test_fit_heterogeneous_ligand_with_noise_parameters_close():
    C1 = np.logspace(-3, 2, 90)
    C2 = np.logspace(-3, 2, 90)

    Kd1_true = 0.1
    Kd2_true = 5.0
    fraction_true = 0.35
    Rmax_1_true = 9.0
    Rmax_2_true = 15.0

    y1 = steady_state_two_site_heterogeneous_ligand(
        C1, Rmax_1_true, Kd1_true, Kd2_true, fraction_true
    )
    y2 = steady_state_two_site_heterogeneous_ligand(
        C2, Rmax_2_true, Kd1_true, Kd2_true, fraction_true
    )

    fit, _, _ = fit_steady_state_two_site_heterogeneous_ligand(
        signal_lst=[_noisy(y1, rel_noise=0.005, seed=11), _noisy(y2, rel_noise=0.005, seed=22)],
        ligand_lst=[C1, C2],
        initial_parameters=[0.2, 3.0, 0.55, 8.0, 13.0],
        low_bounds=[1e-3, 1.0, 0.0, 0.0, 0.0],
        high_bounds=[0.5, 50.0, 1.0, 100.0, 100.0],
    )

    assert np.isclose(fit[0], Kd1_true, rtol=0.25)
    assert np.isclose(fit[1], Kd2_true, rtol=0.30)
    assert np.isclose(fit[2], fraction_true, atol=0.12)
    assert np.isclose(fit[3], Rmax_1_true, rtol=0.15)
    assert np.isclose(fit[4], Rmax_2_true, rtol=0.15)


def test_fit_heterogeneous_ligand_fixed_kds_and_fraction_recovers_rmax():
    C1 = np.logspace(-3, 2, 80)
    C2 = np.logspace(-2.5, 1.8, 65)

    Kd1_true = 0.12
    Kd2_true = 6.0
    fraction_true = 0.4
    Rmax_1_true = 8.0
    Rmax_2_true = 14.0

    y1 = steady_state_two_site_heterogeneous_ligand(
        C1, Rmax_1_true, Kd1_true, Kd2_true, fraction_true
    )
    y2 = steady_state_two_site_heterogeneous_ligand(
        C2, Rmax_2_true, Kd1_true, Kd2_true, fraction_true
    )
    fit, _, fit_vals = fit_steady_state_two_site_heterogeneous_ligand(
        signal_lst=[y1, y2],
        ligand_lst=[C1, C2],
        initial_parameters=[7.0, 12.0],
        low_bounds=[0.0, 0.0],
        high_bounds=[100.0, 100.0],
        fixed_Kd1=True,
        Kd1_value=Kd1_true,
        fixed_Kd2=True,
        Kd2_value=Kd2_true,
        fixed_fraction_site1=True,
        fraction_site1_value=fraction_true,
    )

    assert np.isclose(fit[0], Rmax_1_true, rtol=1e-3)
    assert np.isclose(fit[1], Rmax_2_true, rtol=1e-3)
    np.testing.assert_allclose(fit_vals[0], y1, atol=1e-8)
    np.testing.assert_allclose(fit_vals[1], y2, atol=1e-8)
