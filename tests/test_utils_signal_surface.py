import pytest
import numpy as np

from pykingenie.utils.signal_surface import (
    steady_state_one_site,
    one_site_association_analytical,
    one_site_dissociation_analytical,
    solve_ode_one_site_mass_transport_association,
    solve_ode_one_site_mass_transport_dissociation,
    solve_induced_fit_association,
    solve_induced_fit_dissociation,
    solve_conformational_selection_association,
    solve_conformational_selection_dissociation,
    solve_ode_mixture_analyte_association,
    solve_ode_mixture_analyte_dissociation
)

def test_steady_state_one_site():

    C = np.logspace(-9, -3, 20)  # Concentration range from 10^-9 to 10^-3
    Kd = 1e-6  # Dissociation constant
    R_max = 1.0  # Maximum response

    response = steady_state_one_site(C, R_max, Kd)

    # Check if the response is a numpy array
    assert isinstance(response, np.ndarray), "The response should be a numpy array."

    # Check that the last value is close to R_max
    assert np.isclose(response[-1], R_max, atol=1e-3), "The last value of the response should be close to R_max."

    # Check that the response is non-negative
    assert np.all(response >= 0), "The response should be non-negative."

    # Check that the response increases with concentration
    assert np.all(np.diff(response) >= 0), "The response should be non-decreasing with increasing concentration."

def test_one_site_association_analytical():

    t = np.linspace(0, 10, 10)
    s0 = 0 # initial signal
    s_max = 1.0 # maximum signal
    k_off = 0.05 # dissociation rate constant
    Kd = 0.5 # equilibrium dissociation constant
    A = 1.0 # total concentration of A
    t0=0

    result = one_site_association_analytical(t, s0, s_max, k_off, Kd, A, t0)

    # Check if the result is a numpy array
    assert isinstance(result, np.ndarray), "The result should be a numpy array."

    # Check the length of the result
    assert len(result) == len(t), "The length of the result should match the length of t."

    # Check the first value of the result
    assert np.isclose(result[0], s0), "The first value of the result should match the initial signal."

    # Check increasing behavior
    assert np.all(np.diff(result) >= 0), "The result should be non-decreasing with increasing time."

    # Assuming infinite concentration of A, the last value should be close to s_max
    result = one_site_association_analytical(t, s0, s_max, k_off, Kd, 1e6, t0)

    assert np.isclose(result[-1], s_max), "The last value of the result should match the expected value based on the maximum signal and dissociation"

def test_one_site_dissociation_analytical():
    t = np.linspace(0, 10, 10)
    s0 = 1.0 # initial signal
    k_off = 0.05 # dissociation rate constant

    result = one_site_dissociation_analytical(t,s0,k_off,t0=0)

    # Check if the result is a numpy array
    assert isinstance(result, np.ndarray), "The result should be a numpy array."

    # Check the length of the result
    assert len(result) == len(t), "The length of the result should match the length of t."

    # Check the first value of the result
    assert np.isclose(result[0], s0), "The first value of the result should match the initial signal."

    # Check decreasing behavior
    assert np.all(np.diff(result) <= 0), "The result should be non-increasing with increasing time."

def test_ode_one_site_mass_transport_association():

    # Parameters
    t = np.linspace(0, 100, 50)  # time points
    s1_0 = 0.0                   # initial signal
    cs_0 = 0.5                   # initial analyte surface concentration
    analyte_conc = 0.5           # high analyte concentration
    Kd = 1e-6                    # equilibrium dissociation constant
    k_off = 0.1                  # dissociation rate constant
    k_tr = 10.0                  # mass transport rate constant
    s_max = 1.0                  # maximum signal
    t0 = 0

    # Solve ODE
    signal = solve_ode_one_site_mass_transport_association(
        t, s1_0, cs_0, analyte_conc, Kd, k_off, k_tr, s_max, t0
    )

    # Check output type
    assert isinstance(signal, np.ndarray), "The output should be a numpy array."
    # Check monotonic increase
    assert np.all(np.diff(signal) >= -1e-8), "Signal should be non-decreasing with time."
    # Check initial value
    assert np.isclose(signal[0], s1_0), "Signal at t=0 should match initial value."
    # Check saturation at high concentration (should approach s_max)
    assert np.isclose(signal[-1], s_max, atol=1e-2), "Signal at high t and high analyte concentration should approach s_max."

def test_ode_one_site_mass_transport_dissociation():

    # Parameters
    t = np.linspace(0, 100, 50)  # time points
    s1_0 = 0.8                   # initial signal (start at max)
    Kd = 0.5                    # equilibrium dissociation constant
    k_off = 0.1                  # dissociation rate constant
    k_tr = 0.1                  # mass transport rate constant
    s_max = 1.0                  # maximum signal
    t0 = 0

    # Solve ODE
    signal = solve_ode_one_site_mass_transport_dissociation(
        t, s1_0, Kd, k_off, k_tr, s_max, t0
    )

    # Check output type
    assert isinstance(signal, np.ndarray), "The output should be a numpy array."
    # Check monotonic decrease
    assert np.all(np.diff(signal) <= 1e-8), "Signal should be non-increasing with time."
    # Check initial value
    assert np.isclose(signal[0], s1_0), "Signal at t=0 should match initial value."
    # Check that the signal approaches zero at long times
    assert signal[-1] < 0.05, "Signal at long times should approach zero."

def test_solve_induced_fit_association():

    # Parameters
    t = np.linspace(0, 0.5, 5)
    a_conc = 1.0
    kon = 100
    koff = 100
    kc = 10
    krev = 1
    sP1L = 0.0
    sP2L = 0.0
    smax = 1.0

    arr = solve_induced_fit_association(t, a_conc, kon, koff, kc, krev, sP1L, sP2L, smax)

    # Check output type and shape
    assert isinstance(arr, np.ndarray), "Output should be a numpy array."
    assert arr.shape == (len(t), 3), "Output should have shape (len(t), 3)."

    total_signal = arr[:, 1] + arr[:, 2]

    # Check that the signal is non-decreasing in the first column (total signal)
    assert np.all(np.diff(total_signal) >= -1e-8), "Signal should be non-decreasing with time."

    # Check that the signal starts at zero 
    assert np.isclose(total_signal[0], 0, atol=1e-6), "Initial signal should be zero."
    
    # Assuming infinite concentration of A, the last value should be close to smax
    arr = solve_induced_fit_association(t, 1e8, kon, koff, kc, krev, sP1L, sP2L, smax)
    total_signal = arr[:, 1] + arr[:, 2]

    assert np.isclose(total_signal[-1], smax, atol=1e-2), "Total signal at long times should approach smax."

def test_solve_induced_fit_dissociation():

    # Parameters
    t = np.linspace(0, 0.5, 5)
    koff = 100
    kc = 10
    krev = 1
    sP2L = 0.5
    smax = 10
    s0 = 0.7

    arr = solve_induced_fit_dissociation(t, koff, kc, krev, s0=s0, sP2L=sP2L, smax=smax)

    # Check output type and shape
    assert isinstance(arr, np.ndarray), "Output should be a numpy array."
    assert arr.shape == (len(t), 3), "Output should have shape (len(t), 3)."

    total_signal = arr[:, 1] + arr[:, 2]

    # Check that the signal is decreasing in the first column (total signal)
    assert np.all(np.diff(total_signal) <= 1e-6), "Signal should be non-increasing with time."

    # Check that the signal starts at s0 
    assert np.isclose(total_signal[0], s0, atol=1e-6), "Initial signal should match s0."
    
    # Assuming infinite time, the signal should approach zero
    arr = solve_induced_fit_dissociation([0,1,1e2,1e8], koff, kc, krev, s0=s0, sP2L=sP2L, smax=smax)

    total_signal = arr[:, 1] + arr[:, 2]

    assert np.isclose(total_signal[-1], 0, atol=1e-6), "Total signal at long times should approach zero."

def test_solve_conformational_selection_association():

    time = np.linspace(0, 1, 10)
    a_conc = 1.0
    kon = 10
    koff = 10
    kc = 10
    krev = 10
    smax = 1

    P_tot = 2

    # Compute the conformational equilibrium constant from kc and krev
    Kc = kc / krev

    # Compute the initial concentrations of P1 and P2
    P1_conc = P_tot / (1 + Kc)

    result = solve_conformational_selection_association(time, a_conc, kon, koff, kc, krev,smax=smax,sP1=P1_conc,sP2L=0)

    # Check output type and shape
    assert isinstance(result, np.ndarray), "Output should be a numpy array."
    assert result.shape == (len(time), 3), "Output should have shape (len(time), 3)."

    # Check that the signal is non-decreasing in the first column (total signal)
    total_signal = result[:, 0] 
    assert np.all(np.diff(total_signal) >= -1e-8), "Signal should be non-decreasing with time."

    # Check that at infinite concentration of A, the last value should be close to smax
    result = solve_conformational_selection_association(time, 1e8, kon, koff, kc, krev,smax=smax,sP1=P1_conc,sP2L=0)
    total_signal = result[:, 0]
    assert np.isclose(total_signal[-1], smax, atol=1e-2), "Total signal at long times should approach smax."

def test_solve_conformational_selection_dissociation():
    
    time = np.linspace(0, 1, 10)
    koff = 10
    kc = 10
    krev = 10
    smax = 10

    result = solve_conformational_selection_dissociation(time, koff, kc, krev,smax=smax,sP1=2,sP2L=5)

    # Check output type and shape
    assert isinstance(result, np.ndarray), "Output should be a numpy array."
    assert result.shape == (len(time), 3), "Output should have shape (len(time), 3)."

    total_signal = result[:, 0] 

    # Check that the signal is decreasing in the first column (total signal)
    assert np.all(np.diff(total_signal) <= 1e-6), "Signal should be non-increasing with time."

    
    # Assuming infinite time, the signal should approach zero
    result = solve_conformational_selection_dissociation([0,1e2,1e8], koff, kc, krev,smax=smax,sP1=2,sP2L=5)

    total_signal = result[:, 0]

    assert np.isclose(total_signal[-1], 0, atol=1e-6), "Total signal at long times should approach zero."

def test_solve_ode_mixture_analyte_association():

    t = np.linspace(0, 200, 100)
    Ris0 = [0,0]
    C_tot = 0.1
    Fis = np.array([0.5, 0.5])  # Example values for Fis
    Ris_max = np.array([1.0, 2])  # Example values for Ris_max
    koffs = np.array([0.1, 0.2])  # Example values for koffs
    Kds = np.array([0.5, 0.1])    # Example values for Kds

    result = solve_ode_mixture_analyte_association(t,Ris0,C_tot,Fis,Ris_max,koffs,Kds,t0=0)

    # Check output type and shape
    assert isinstance(result, np.ndarray), "Output should be a numpy array."
    assert result.shape == (len(Fis), len(t)), "Output should have shape (len(Fis), len(t))."

    # Check that the first value matches Ris0
    assert np.allclose(result[:, 0], Ris0), "The first value of the result should match Ris0."

    # Sum the responses - column axis
    total_response = np.sum(result, axis=0)

    # Check that the response increases with time
    assert np.all(np.diff(total_response) >= -1e-4), "The response should be non-decreasing with increasing time."

    # Test case with Fis[0] = 0 and infinite C_tot. Then the max value should be close to Ris_max[1]
    result = solve_ode_mixture_analyte_association(t, Ris0, 1e8, [0,1], Ris_max, koffs, Kds, t0=0)

    # Sum the responses - column axis
    total_response = np.sum(result, axis=0)

    assert np.isclose(total_response[-1], Ris_max[1], atol=1e-2), "The last values should approach the sum of Ris_max."

    # Test case with Fis[1] = 0 and infinite C_tot. Then the max value should be close to Ris_max[0]
    result = solve_ode_mixture_analyte_association(t, Ris0, 1e8, [1,0], Ris_max, koffs, Kds, t0=0)
    # Sum the responses - column axis
    total_response = np.sum(result, axis=0) 
    assert np.isclose(total_response[-1], Ris_max[0], atol=1e-2), "The last values should approach the sum of Ris_max."

def test_solve_ode_mixture_analyte_dissociation():

    Ris0 = [100, 20]
    koffs = np.array([0.1, 0.2])  # Example values for koffs
    t = np.linspace(0, 200, 100)

    # Solve the ODE for mixture analyte dissociation
    result = solve_ode_mixture_analyte_dissociation(t,Ris0,koffs,t0=0)

    # Check output type and shape
    assert isinstance(result, np.ndarray), "Output should be a numpy array."
    assert result.shape == (len(Ris0), len(t)), "Output should have shape (len(Ris0), len(t))."

    # Check that the first value matches Ris0
    assert np.allclose(result[:, 0], Ris0), "The first value of the result should match Ris0."

    # Check that the response decreases with time
    total_response = np.sum(result, axis=0)
    assert np.all(np.diff(total_response) <= 1e-4), "The response should be non-increasing with increasing time."

    # Check that the last value approaches zero - infinite time
    result = solve_ode_mixture_analyte_dissociation(np.array([0,1e2,1e8]), Ris0, koffs, t0=0)
    total_response = np.sum(result, axis=0)

    assert np.isclose(total_response[-1], 0, atol=1e-2), "The last value of the response should approach zero."