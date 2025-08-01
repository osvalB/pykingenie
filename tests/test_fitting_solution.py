import pytest
import numpy as np

from pykingenie.utils.signal_solution import (
    signal_ode_one_site_insolution,
    signal_ode_induced_fit_insolution
)

from pykingenie.utils.fitting_solution import(

    fit_one_site_solution,
    fit_induced_fit_solution,
    find_initial_parameters_induced_fit_solution   
)

def test_fit_one_site_solution():

    t = np.linspace(0, 30, 30)
    koff = 0.1
    Kd = 0.5
    a_total = 1.0
    t0 = 0

    signal_a=0
    signal_b=0
    signal_complex=5

    signals = []
    for b_total in [0.5, 1.0, 1.5]:

        result = signal_ode_one_site_insolution(t, koff, Kd, a_total, b_total, t0,signal_a, signal_b, signal_complex)
        # Add small gaussian error 
        noise = np.random.normal(0, 0.008, len(result))
        result += noise
        signals.append(result)

    signal_lst = signals
    time_lst = [t] * len(signals)
    ligand_conc_lst = [0.5, 1.0, 1.5]
    protein_conc_lst = [a_total] * len(signals)

    initial_parameters = [1,1,1]
    low_bounds = [0.0, 0.0, 0.0] # Lower bounds for signal ES, K_d and k_off
    high_bounds = [100.0, 100.0, 100.0] # Upper bounds for signal ES, K_d and k_off

    fit_signal_E = False
    fit_signal_S = False
    fit_signal_ES = True
    fixed_Kd = False
    fixed_koff = False

    global_fit_params, _, _, _ = fit_one_site_solution(
        signal_lst=signal_lst,
        time_lst=time_lst,
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst=protein_conc_lst,
        fit_signal_E=fit_signal_E,
        fit_signal_S=fit_signal_S,
        fit_signal_ES=fit_signal_ES,
        fixed_Kd=fixed_Kd,
        fixed_koff=fixed_koff,
        initial_parameters=initial_parameters,
        low_bounds=low_bounds,
        high_bounds=high_bounds
    )

    # Assert that the  signal_ES, K_d and k_off parameters are within expected ranges
    expected_params = [signal_complex, Kd, koff]

    np.testing.assert_array_almost_equal(global_fit_params, expected_params, decimal=1,
                                         err_msg="The fitted parameters do not match the expected values.")
    
def test_fit_one_site_solution_with_signal_b():

    t = np.linspace(0, 30, 30)
    koff = 0.1
    Kd = 0.5
    a_total = 1.0
    t0 = 0

    signal_a=0
    signal_b=2
    signal_complex=5

    signals = []
    for b_total in [0.5, 1.0, 1.5]:

        result = signal_ode_one_site_insolution(t, koff, Kd, a_total, b_total, t0,signal_a, signal_b, signal_complex)
        # Add small gaussian error 
        noise = np.random.normal(0, 0.008, len(result))
        result += noise
        signals.append(result)

    signal_lst = signals
    time_lst = [t] * len(signals)
    ligand_conc_lst = [0.5, 1.0, 1.5]
    protein_conc_lst = [a_total] * len(signals)

    initial_parameters = [1,1,1,1]
    low_bounds = [0] * len(initial_parameters)  
    high_bounds = [100] * len(initial_parameters)

    fit_signal_E = False
    fit_signal_S = True
    fit_signal_ES = True
    fixed_Kd = False
    fixed_koff = False

    global_fit_params, _, _, _ = fit_one_site_solution(
        signal_lst=signal_lst,
        time_lst=time_lst,
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst=protein_conc_lst,
        fit_signal_E=fit_signal_E,
        fit_signal_S=fit_signal_S,
        fit_signal_ES=fit_signal_ES,
        fixed_Kd=fixed_Kd,
        fixed_koff=fixed_koff,
        initial_parameters=initial_parameters,
        low_bounds=low_bounds,
        high_bounds=high_bounds
    )

    # Assert that the  signal_ES, K_d and k_off parameters are within expected ranges
    expected_params = [signal_b,signal_complex, Kd, koff]

    np.testing.assert_array_almost_equal(global_fit_params, expected_params, decimal=1,
                                         err_msg="The fitted parameters do not match the expected values.")
    
def test_fit_one_site_solution_with_signal_a():

    t = np.linspace(0, 30, 30)
    koff = 0.1
    Kd = 0.5
    a_total = 1.0
    t0 = 0

    signal_a=8
    signal_b=0
    signal_complex=5

    signals = []
    for b_total in [0.5, 1.0, 1.5]:

        result = signal_ode_one_site_insolution(t, koff, Kd, a_total, b_total, t0,signal_a, signal_b, signal_complex)
        # Add small gaussian error 
        noise = np.random.normal(0, 0.008, len(result))
        result += noise
        signals.append(result)

    signal_lst = signals
    time_lst = [t] * len(signals)
    ligand_conc_lst = [0.5, 1.0, 1.5]
    protein_conc_lst = [a_total] * len(signals)

    initial_parameters = [1,1,1,1]
    low_bounds = [0] * len(initial_parameters)  
    high_bounds = [100] * len(initial_parameters)

    fit_signal_E = True
    fit_signal_S = False
    fit_signal_ES = True
    fixed_Kd = False
    fixed_koff = False

    global_fit_params, _, _, _ = fit_one_site_solution(
        signal_lst=signal_lst,
        time_lst=time_lst,
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst=protein_conc_lst,
        fit_signal_E=fit_signal_E,
        fit_signal_S=fit_signal_S,
        fit_signal_ES=fit_signal_ES,
        fixed_Kd=fixed_Kd,
        fixed_koff=fixed_koff,
        initial_parameters=initial_parameters,
        low_bounds=low_bounds,
        high_bounds=high_bounds
    )

    # Assert that the  signal_ES, K_d and k_off parameters are within expected ranges
    expected_params = [signal_a,signal_complex, Kd, koff]

    np.testing.assert_array_almost_equal(global_fit_params, expected_params, decimal=1,
                                         err_msg="The fitted parameters do not match the expected values.")
    

def test_fit_one_site_solution_with_fixed_constants():

    t = np.linspace(0, 30, 30)
    koff = 0.1
    Kd = 0.5
    a_total = 1.0
    t0 = 0

    signal_a=8
    signal_b=2
    signal_complex=5

    signals = []
    for b_total in [0.5, 1.0, 1.5]:

        result = signal_ode_one_site_insolution(t, koff, Kd, a_total, b_total, t0,signal_a, signal_b, signal_complex)
        # Add small gaussian error 
        noise = np.random.normal(0, 0.008, len(result))
        result += noise
        signals.append(result)

    signal_lst = signals
    time_lst = [t] * len(signals)
    ligand_conc_lst = [0.5, 1.0, 1.5]
    protein_conc_lst = [a_total] * len(signals)

    initial_parameters = [1,1,1]
    low_bounds = [0] * len(initial_parameters)  
    high_bounds = [100] * len(initial_parameters)

    fit_signal_E = True
    fit_signal_S = True
    fit_signal_ES = True
    fixed_Kd = True
    fixed_koff = True

    global_fit_params, _, _, _ = fit_one_site_solution(
        signal_lst=signal_lst,
        time_lst=time_lst,
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst=protein_conc_lst,
        fit_signal_E=fit_signal_E,
        fit_signal_S=fit_signal_S,
        fit_signal_ES=fit_signal_ES,
        fixed_Kd=fixed_Kd,
        fixed_koff=fixed_koff,
        initial_parameters=initial_parameters,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        Kd_value= Kd,
        koff_value= koff
    )

    # Assert that the  signal_ES, K_d and k_off parameters are within expected ranges
    expected_params = [signal_a,signal_b, signal_complex]

    np.testing.assert_array_almost_equal(global_fit_params, expected_params, decimal=1,
                                         err_msg="The fitted parameters do not match the expected values.")
    

def test_fit_one_site_solution_with_t0():

    t = np.linspace(0, 30, 30)
    koff = 0.1
    Kd = 0.5
    a_total = 1.0
    t0 = 0

    signal_a=0
    signal_b=0
    signal_complex=5

    signals = []
    for b_total in [0.5, 1.0, 1.5]:

        result = signal_ode_one_site_insolution(t, koff, Kd, a_total, b_total, t0,signal_a, signal_b, signal_complex)
        # Add small gaussian error 
        noise = np.random.normal(0, 0.008, len(result))
        result += noise
        signals.append(result)

    signal_lst = signals
    time_lst = [t+0.01] * len(signals) # We add a small offset to the time to simulate t0
    ligand_conc_lst = [0.5, 1.0, 1.5]
    protein_conc_lst = [a_total] * len(signals)

    initial_parameters = [1,1,1,0,0,0] # Initial parameters for signal ES, K_d, k_off, t0_1, t0_2, t0_3
    low_bounds = [0.0, 0.0, 0.0, -0.1, -0.1, -0.1] # Lower bounds for signal ES, K_d, k_off, t0_1, t0_2, t0_3
    high_bounds = [100.0, 100.0, 100.0,0.1,0.1,0.1] # Upper bounds for signal ES, K_d and k_off

    fit_signal_E = False
    fit_signal_S = False
    fit_signal_ES = True
    fixed_Kd = False
    fixed_koff = False
    fixed_t0 = False

    global_fit_params, _, _, _ = fit_one_site_solution(
        signal_lst=signal_lst,
        time_lst=time_lst,
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst=protein_conc_lst,
        fit_signal_E=fit_signal_E,
        fit_signal_S=fit_signal_S,
        fit_signal_ES=fit_signal_ES,
        fixed_Kd=fixed_Kd,
        fixed_koff=fixed_koff,
        initial_parameters=initial_parameters,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        fixed_t0=fixed_t0
    )

    # Assert that the  signal_ES, K_d and k_off parameters are within expected ranges
    expected_params = [signal_complex, Kd, koff] + [0.01, 0.01, 0.01]  # t0 values should be close to the offset we added

    np.testing.assert_array_almost_equal(global_fit_params, expected_params, decimal=1,
                                         err_msg="The fitted parameters do not match the expected values.")
    
def test_fit_induced_fit_solution():

    kon  = 100
    koff = 100
    kc   = 10
    krev = 1

    t = np.linspace(0, 0.25, 50)

    E_tot = 1  # Total enzyme concentration
    S_tot = np.logspace(-1.5, 1, 15)  # Total substrate concentrations

    signal_E = 0
    signal_S = 0
    signal_ES_int = 1
    signal_ES = 1

    ys = []

    for i in range(len(S_tot)):

        y0 = [0, 0.0]   # y0 (list): initial concentrations of E·S and ES
        y = signal_ode_induced_fit_insolution(
            t, y0, kon, koff, kc, krev, E_tot, S_tot[i], 0,
            signal_E, signal_S, signal_ES_int, signal_ES)

        # Add small noise to the signal
        noise = np.random.normal(0, 0.001, len(t))
        y += noise

        ys.append(y)


    signal_lst       = ys
    time_lst         = [t] * len(signal_lst)
    ligand_conc_lst  = S_tot
    protein_conc_lst = [E_tot] * len(signal_lst)

    best_params = find_initial_parameters_induced_fit_solution(
        signal_lst, 
        time_lst, 
        ligand_conc_lst,
        protein_conc_lst,
        np_linspace_low=-1,
        np_linspace_high=2,
        np_linspace_num=3)

    # Fit using as initial parameters the best found parameters
    initial_parameters = np.array(best_params)
    low_bounds         = best_params / 1e3
    high_bounds        = best_params * 1e3

    global_fit_params, _, _, _ = fit_induced_fit_solution(signal_lst, time_lst, ligand_conc_lst,
                                                                 protein_conc_lst, initial_parameters, low_bounds,
                                                                 high_bounds)
    
    expected_params = [signal_ES,kon, koff, kc, krev]

    np.testing.assert_array_almost_equal(global_fit_params, expected_params, decimal=0,
                                         err_msg="The fitted parameters do not match the expected values.")
    

def test_fit_induced_fit_solution_fixed_constants():

    kon  = 100
    koff = 100
    kc   = 10
    krev = 1

    t = np.linspace(0, 0.25, 50)

    E_tot = 1  # Total enzyme concentration
    S_tot = np.logspace(-1.5, 1, 15)  # Total substrate concentrations

    signal_E = 0
    signal_S = 0
    signal_ES_int = 1
    signal_ES = 1

    ys = []

    for i in range(len(S_tot)):

        y0 = [0, 0.0]   # y0 (list): initial concentrations of E·S and ES
        y = signal_ode_induced_fit_insolution(
            t, y0, kon, koff, kc, krev, E_tot, S_tot[i], 0,
            signal_E, signal_S, signal_ES_int, signal_ES)

        # Add small noise to the signal
        noise = np.random.normal(0, 0.001, len(t))
        y += noise

        ys.append(y)


    signal_lst       = ys
    time_lst         = [t] * len(signal_lst)
    ligand_conc_lst  = S_tot
    protein_conc_lst = [E_tot] * len(signal_lst)

    initial_parameters = [1]
    low_bounds         = [1e-3]
    high_bounds        = [1e3]

    global_fit_params, _, _, _ = fit_induced_fit_solution(
        signal_lst=signal_lst, 
        time_lst=time_lst, 
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst= protein_conc_lst, 
        initial_parameters= initial_parameters, 
        low_bounds= low_bounds,
        high_bounds= high_bounds,
        fixed_kon=True,
        fixed_koff=True,
        fixed_kc=True,
        fixed_krev=True,
        kon_value=kon,
        koff_value=koff,
        kc_value=kc,
        krev_value=krev,
        fit_signal_E=False,
        fit_signal_S=False,
        ESint_equals_ES=True,
        fit_signal_ES=True)
    
    expected_params = [signal_ES]

    np.testing.assert_array_almost_equal(global_fit_params, expected_params, decimal=0,
                                         err_msg="The fitted parameters do not match the expected values.")

def test_fit_induced_fit_solution_fixed_constants_different_ES_signal():

    kon  = 100
    koff = 100
    kc   = 10
    krev = 1

    t = np.linspace(0, 0.25, 50)

    E_tot = 1  # Total enzyme concentration
    S_tot = np.logspace(-1.5, 1, 15)  # Total substrate concentrations

    signal_E = 0
    signal_S = 0
    signal_ES_int = 1
    signal_ES = 5

    ys = []

    for i in range(len(S_tot)):

        y0 = [0, 0.0]   # y0 (list): initial concentrations of E·S and ES
        y = signal_ode_induced_fit_insolution(
            t, y0, kon, koff, kc, krev, E_tot, S_tot[i], 0,
            signal_E, signal_S, signal_ES_int, signal_ES)

        # Add small noise to the signal
        noise = np.random.normal(0, 0.001, len(t))
        y += noise

        ys.append(y)

    signal_lst       = ys
    time_lst         = [t] * len(signal_lst)
    ligand_conc_lst  = S_tot
    protein_conc_lst = [E_tot] * len(signal_lst)

    initial_parameters = [1,1]
    low_bounds         = [1e-3, 1e-3]
    high_bounds        = [1e3, 1e3]

    global_fit_params, _, _, params_names = fit_induced_fit_solution(
        signal_lst=signal_lst, 
        time_lst=time_lst, 
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst= protein_conc_lst, 
        initial_parameters= initial_parameters, 
        low_bounds= low_bounds,
        high_bounds= high_bounds,
        fixed_kon=True,
        fixed_koff=True,
        fixed_kc=True,
        fixed_krev=True,
        kon_value=kon,
        koff_value=koff,
        kc_value=kc,
        krev_value=krev,
        fit_signal_E=False,
        fit_signal_S=False,
        ESint_equals_ES=False,
        fit_signal_ES=True)
    
    expected_params = [signal_ES,signal_ES_int]

    np.testing.assert_array_almost_equal(global_fit_params, expected_params, decimal=0,
                                         err_msg="The fitted parameters do not match the expected values.")
    
    expected_params_names = ["Signal of the trapped complex", "Signal of the intermediate complex"]

    assert params_names == expected_params_names, "The parameter names do not match the expected names."

def test_fit_induced_fit_solution_fixed_constants_different_signals():

    kon  = 100
    koff = 100
    kc   = 10
    krev = 1

    t = np.linspace(0, 0.25, 50)

    E_tot = 1  # Total enzyme concentration
    S_tot = np.logspace(-1.5, 1, 15)  # Total substrate concentrations

    signal_E = 1
    signal_S = 2
    signal_ES_int = 10
    signal_ES = 10

    ys = []

    for i in range(len(S_tot)):

        y0 = [0, 0.0]   # y0 (list): initial concentrations of E·S and ES
        y = signal_ode_induced_fit_insolution(
            t, y0, kon, koff, kc, krev, E_tot, S_tot[i], 0,
            signal_E, signal_S, signal_ES_int, signal_ES)

        # Add small noise to the signal
        noise = np.random.normal(0, 0.001, len(t))
        y += noise

        ys.append(y)

    signal_lst       = ys
    time_lst         = [t] * len(signal_lst)
    ligand_conc_lst  = S_tot
    protein_conc_lst = [E_tot] * len(signal_lst)

    initial_parameters = [1,1,1]
    low_bounds         = [1e-3, 1e-3, 1e-3]
    high_bounds        = [1e3, 1e3, 1e3]

    global_fit_params, _, _, params_names = fit_induced_fit_solution(
        signal_lst=signal_lst, 
        time_lst=time_lst, 
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst= protein_conc_lst, 
        initial_parameters= initial_parameters, 
        low_bounds= low_bounds,
        high_bounds= high_bounds,
        fixed_kon=True,
        fixed_koff=True,
        fixed_kc=True,
        fixed_krev=True,
        kon_value=kon,
        koff_value=koff,
        kc_value=kc,
        krev_value=krev,
        fit_signal_E=True,
        fit_signal_S=True,
        ESint_equals_ES=True,
        fit_signal_ES=True)
    
    expected_params = [signal_E,signal_S,signal_ES]

    np.testing.assert_array_almost_equal(global_fit_params, expected_params, decimal=0,
                                         err_msg="The fitted parameters do not match the expected values.")
        
    expected_params_names = [
        "Signal of the free protein", 
        "Signal of the free ligand",
        "Signal of the complex"]

    assert params_names == expected_params_names, "The parameter names do not match the expected names."


def test_fit_induced_fit_solution_fixed_constants_t0():

    kon  = 100
    koff = 100
    kc   = 10
    krev = 1

    t = np.linspace(0, 0.25, 50)

    E_tot = 1  # Total enzyme concentration
    S_tot = np.logspace(-1.5, 1, 15)  # Total substrate concentrations

    signal_E = 0
    signal_S = 0
    signal_ES_int = 20
    signal_ES = 20

    ys = []

    for i in range(len(S_tot)):

        y0 = [0, 0.0]   # y0 (list): initial concentrations of E·S and ES
        y = signal_ode_induced_fit_insolution(
            t, y0, kon, koff, kc, krev, E_tot, S_tot[i], 0,
            signal_E, signal_S, signal_ES_int, signal_ES)

        # Add small noise to the signal
        noise = np.random.normal(0, 0.001, len(t))
        y += noise

        ys.append(y)

    signal_lst       = ys
    time_lst         = [t+0.001] * len(signal_lst)
    ligand_conc_lst  = S_tot
    protein_conc_lst = [E_tot] * len(signal_lst)

    initial_parameters = [1] + ([0.001] * len(time_lst))
    low_bounds         = [1e-3] + ([-0.001] * len(time_lst))
    high_bounds        = [1e3] + ([0.01] * len(time_lst))

    global_fit_params, _, _, params_names = fit_induced_fit_solution(
        signal_lst=signal_lst, 
        time_lst=time_lst, 
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst= protein_conc_lst, 
        initial_parameters= initial_parameters, 
        low_bounds= low_bounds,
        high_bounds= high_bounds,
        fixed_kon=True,
        fixed_koff=True,
        fixed_kc=True,
        fixed_krev=True,
        kon_value=kon,
        koff_value=koff,
        kc_value=kc,
        krev_value=krev,
        fit_signal_E=False,
        fit_signal_S=False,
        ESint_equals_ES=True,
        fit_signal_ES=True,
        fixed_t0=False)
    
    expected_params = [signal_ES] + ([0.001] * len(time_lst))  # t0 values should be close to the offset we added

    np.testing.assert_array_almost_equal(global_fit_params, expected_params, decimal=0,
                                         err_msg="The fitted parameters do not match the expected values.")
        
    expected_params_names = ["Signal of the complex"] + ['t0_' + str(i+1) for i in range(len(time_lst))]

    assert params_names == expected_params_names, "The parameter names do not match the expected names."