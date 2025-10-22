import pytest
import numpy as np

from pykingenie.utils.signal_solution import (
    signal_ode_one_site_insolution,
    signal_ode_induced_fit_insolution,  # now the reduced version
    signal_ode_conformational_selection_insolution,
    get_initial_concentration_conformational_selection,
    get_kobs_induced_fit,
    get_kobs_conformational_selection
)

def test_solve_ode_one_site_insolution():

    t = np.linspace(0, 10, 10)
    koff = 0.1
    Kd = 0.5
    a_total = 1.0
    b_total = 1.0
    t0 = 0

    signal_a=0
    signal_b=0
    signal_complex=1

    result = signal_ode_one_site_insolution(t, koff, Kd, a_total, b_total, t0,signal_a, signal_b, signal_complex)

    expected = [0,0.17265383,0.2791352,0.34808851,0.39423549,0.42570054,0.44748987,0.46274577,0.47348593,0.48107122]

    np.testing.assert_array_almost_equal(result, expected, decimal=5, err_msg="The result of the ODE solution does not match the expected values.")

    # Check that the response increases with time
    assert np.all(np.diff(result) >= 0), "The response should be non-decreasing with increasing time."


def test_signal_ode_induced_fit_insolution():

    t = np.linspace(0, 10, 10)
    y = [0,0] # Initial concentrations of E and S
    k1 = 100
    k_minus1 = 100
    k2 = 10
    k_minus2 = 1

    E_tot = 0.5
    S_tot = 0.5

    result = signal_ode_induced_fit_insolution(
        t, y, k1, k_minus1, k2, k_minus2,
        E_tot, S_tot,
        t0=0, signal_E=0, signal_S=0, signal_ES_int=1, signal_ES=1)

    expected = np.array([0.    , 0.3234, 0.3274, 0.3275, 0.3275, 0.3275, 0.3275, 0.3275, 0.3275, 0.3275])

    np.testing.assert_array_almost_equal(result, expected, decimal=4, err_msg="The result of the ODE solution (induced fit) does not match the expected values.")

    # Check that the response increases with time
    assert np.all(np.diff(result) >= -1e-6), "The response should be non-decreasing with increasing time. Tolerance is set to -1e-6 to account for numerical precision issues."


def test_signal_ode_conformational_selection_insolution():

    k1 = 100
    k_minus1 = 1
    k2 = 10
    k_minus2 = 100

    E_tot = 1
    S_tot = 0.5

    E1, E2 = get_initial_concentration_conformational_selection(E_tot,k1,k_minus1)

    t = np.linspace(0, 0.05, 10)

    y = [E2,0]

    result = signal_ode_conformational_selection_insolution(
        t, y, k1, k_minus1, k2,k_minus2,
        E_tot, S_tot,
        t0=0,signal_E1=0,signal_E2=0,signal_S=0,signal_E2S=1)

    expected = np.array([0.    , 0.0203, 0.0311, 0.0368, 0.0398, 0.0414, 0.0423, 0.0428,0.043 , 0.0431])

    np.testing.assert_array_almost_equal(result, expected, decimal=4, err_msg="The result of the ODE solution (conformational selection) does not match the expected values.")

def test_get_kobs_induced_fit():

    tot_lig = 0.5
    tot_prot = 0.5
    kr = 1
    ke= 10
    kon = 100
    koff = 100
    dominant=True

    result = get_kobs_induced_fit(tot_lig, tot_prot, kr, ke, kon, koff, dominant)

    assert np.isclose(result, 10.38016, atol=1e-3), "The kobs value for induced fit does not match the expected value."

def test_get_kobs_conformational_selection():
    
    tot_lig = 0.5
    tot_prot = 0.5
    kr = 100
    ke= 10
    kon = 10
    koff = 10
    dominant= True

    result = get_kobs_conformational_selection(tot_lig, tot_prot, kr, ke, kon, koff, dominant)

    assert np.isclose(result, 10.4155, atol=1e-3), "The kobs value for conformational selection does not match the expected value."

