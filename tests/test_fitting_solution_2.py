import pytest
import numpy as np

from pykingenie.utils.signal_solution  import (
    signal_ode_conformational_selection_insolution,
    get_initial_concentration_conformational_selection
)

from pykingenie.utils.fitting_solution import fit_conformational_selection_solution

from src.pykingenie.utils.fitting_solution import find_initial_parameters_conformational_selection_solution


t = np.linspace(0, 0.5, 20)

E_tot = 1  # Total enzyme concentration
S_tot = np.logspace(-1.5, 1, 6)  # Total substrate concentrations

kon = 100
koff = 1
kc = 10
krev = 100

noise = np.random.normal(0, 0.0001, len(t))

def generate_ys(E_tot,S_tot,kc,krev,kon,koff,signal_E1, signal_E2, signal_S, signal_E2S):

    ys = []

    for i in range(len(S_tot)):

        E1, E2 = get_initial_concentration_conformational_selection(E_tot,kc,krev)

        y0 = [E2, 0]

        y = signal_ode_conformational_selection_insolution(
            t, y0, kc,krev,kon,koff, E_tot, S_tot[i], 0,
            signal_E1, signal_E2, signal_S, signal_E2S)

        y += noise

        ys.append(y)

    return ys


def test_fit_cs_solution_fixed_constants_t0():

    signal_E = 0
    signal_S = 0
    signal_E2S = 20

    signal_lst = generate_ys(E_tot,S_tot[-3:],kc,krev,kon,koff,signal_E, signal_E, signal_S, signal_E2S)

    time_lst = [t + 0.03] * len(signal_lst)
    ligand_conc_lst = S_tot[-3:]
    protein_conc_lst = [E_tot] * len(signal_lst)

    initial_parameters = [20] + ([0.03] * len(time_lst))
    low_bounds = [10] + ([0.003] * len(time_lst))
    high_bounds = [30] + ([0.3] * len(time_lst))

    global_fit_params, _, _, params_names = fit_conformational_selection_solution(
        signal_lst=signal_lst,
        time_lst=time_lst,
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst=protein_conc_lst,
        initial_parameters=initial_parameters,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
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
        fit_signal_E2S=True,
        fixed_t0=False)

    expected_params = [signal_E2S] + ([0.03] * len(time_lst))  # t0 values should be close to the offset we added

    np.testing.assert_allclose(
        global_fit_params, expected_params, rtol=0.2,
        err_msg="The fitted parameters do not match the expected values."
    )

    expected_params_names = ["Signal of the complex"] + ['t0_' + str(i + 1) for i in range(len(time_lst))]

    assert params_names == expected_params_names, "The parameter names do not match the expected names."

def test_conformational_selection_insolution():

    signal_E1 = 0
    signal_S = 0
    signal_E2 = 0
    signal_E2S = 1

    signal_lst = generate_ys(E_tot,S_tot,kc,krev,kon,koff,signal_E1, signal_E2, signal_S, signal_E2S)

    time_lst = [t] * len(signal_lst)
    ligand_conc_lst = S_tot
    protein_conc_lst = [E_tot] * len(signal_lst)

    best_params = find_initial_parameters_conformational_selection_solution(
        signal_lst,
        time_lst,
        ligand_conc_lst,
        protein_conc_lst,
        np_linspace_low=0,
        np_linspace_high=3,
        np_linspace_num=4)

    # Fit using as initial parameters the best found parameters
    initial_parameters = np.array(best_params)
    low_bounds = best_params / 1e2
    high_bounds = best_params * 1e2

    global_fit_params, _, _, _ = fit_conformational_selection_solution(
        signal_lst, time_lst, ligand_conc_lst,
        protein_conc_lst, initial_parameters,
        low_bounds,high_bounds
    )

    expected_params = [signal_E2S, kon, koff, kc, krev]

    np.testing.assert_allclose(
        global_fit_params, expected_params, rtol=0.1,
        err_msg="The fitted parameters do not match the expected values.")


def test_fit_conformational_selection_solution_fixed_constants():

    signal_E1 = 0
    signal_E2 = 0
    signal_S = 0
    signal_E2S = 1

    signal_lst = generate_ys(E_tot,S_tot,kc,krev,kon,koff,signal_E1, signal_E2, signal_S, signal_E2S)

    time_lst = [t] * len(signal_lst)
    ligand_conc_lst = S_tot
    protein_conc_lst = [E_tot] * len(signal_lst)

    initial_parameters = [10]
    low_bounds = [1e-3]
    high_bounds = [1e3]

    global_fit_params, _, _, _ = fit_conformational_selection_solution(
        signal_lst=signal_lst,
        time_lst=time_lst,
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst=protein_conc_lst,
        initial_parameters=initial_parameters,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        fixed_kon=True,
        fixed_koff=True,
        fixed_kc=True,
        fixed_krev=True,
        kon_value=kon,
        koff_value=koff,
        kc_value=kc,
        krev_value=krev,
        fit_signal_E=False,
        E1_equals_E2=True,
        fit_signal_S=False,
        fit_signal_E2S=True)

    expected_params = [signal_E2S]

    np.testing.assert_allclose(
        global_fit_params, expected_params, rtol=0.1,
        err_msg="The fitted parameters do not match the expected values.")


def test_fit_conf_sel_solution_fixed_constants_different_E_signal():

    signal_E1 = 1
    signal_E2 = 4
    signal_S = 0
    signal_E2S = 0

    signal_lst = generate_ys(E_tot,S_tot,kc,krev,kon,koff,signal_E1, signal_E2, signal_S, signal_E2S)

    time_lst = [t] * len(signal_lst)
    ligand_conc_lst = S_tot
    protein_conc_lst = [E_tot] * len(signal_lst)

    best_params = find_initial_parameters_conformational_selection_solution(
        signal_lst,
        time_lst,
        ligand_conc_lst,
        protein_conc_lst,
        np_linspace_low=0,
        np_linspace_high=3,
        np_linspace_num=4,
        fit_signal_E=True,
        E1_equals_E2=False,
        fit_signal_S=False,
        fit_signal_E2S=False,
        fixed_t0=True
        )

    # Fit using as initial parameters the best found parameters
    initial_parameters = np.array(best_params[:2])
    low_bounds = initial_parameters / 1e2
    high_bounds = initial_parameters * 1e2

    global_fit_params, _, _, params_names = fit_conformational_selection_solution(
        signal_lst=signal_lst,
        time_lst=time_lst,
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst=protein_conc_lst,
        initial_parameters=initial_parameters,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        fixed_kon=True,
        fixed_koff=True,
        fixed_kc=True,
        fixed_krev=True,
        kon_value=kon,
        koff_value=koff,
        kc_value=kc,
        krev_value=krev,
        fit_signal_E=True,
        E1_equals_E2=False,
        fit_signal_S=False,
        fit_signal_E2S=False)

    expected_params = [signal_E1, signal_E2]

    np.testing.assert_allclose(
        global_fit_params, expected_params, rtol=0.1,
        err_msg="The fitted parameters do not match the expected values.")

    expected_params_names = [
        "Signal of the inactive protein (E1)",
        "Signal of the active protein (E2)"
    ]

    assert params_names == expected_params_names, "The parameter names do not match the expected names."


def test_fit_cs_solution_fixed_constants_different_signals():

    signal_E1 = 1
    signal_E2 = 1
    signal_S = 10
    signal_E2S = 5

    signal_lst = generate_ys(E_tot,S_tot,kc,krev,kon,koff,signal_E1, signal_E2, signal_S, signal_E2S)

    time_lst = [t] * len(signal_lst)
    ligand_conc_lst = S_tot
    protein_conc_lst = [E_tot] * len(signal_lst)

    best_params = find_initial_parameters_conformational_selection_solution(
        signal_lst,
        time_lst,
        ligand_conc_lst,
        protein_conc_lst,
        np_linspace_low=0,
        np_linspace_high=3,
        np_linspace_num=4,
        fit_signal_E=True,
        E1_equals_E2=True,
        fit_signal_S=True,
        fit_signal_E2S=True,
        fixed_t0=True
        )

    # Fit using as initial parameters the best found parameters
    initial_parameters = np.array(best_params[:3])
    low_bounds = initial_parameters / 1e2
    high_bounds = initial_parameters * 1e2

    global_fit_params, _, _, params_names = fit_conformational_selection_solution(
        signal_lst=signal_lst,
        time_lst=time_lst,
        ligand_conc_lst=ligand_conc_lst,
        protein_conc_lst=protein_conc_lst,
        initial_parameters=initial_parameters,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        fixed_kon=True,
        fixed_koff=True,
        fixed_kc=True,
        fixed_krev=True,
        kon_value=kon,
        koff_value=koff,
        kc_value=kc,
        krev_value=krev,
        fit_signal_E=True,
        E1_equals_E2=True,
        fit_signal_S=True,
        fit_signal_E2S=True)

    expected_params = [signal_E1, signal_S, signal_E2S]

    np.testing.assert_allclose(
        global_fit_params, expected_params, rtol=0.1,
        err_msg="The fitted parameters do not match the expected values."
    )

    expected_params_names = [
        "Signal of the free protein",
        "Signal of the free ligand",
        "Signal of the complex"]

    assert params_names == expected_params_names, "The parameter names do not match the expected names."