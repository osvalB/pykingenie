import numpy as np
import pandas as pd
import pytest
import os

from pykingenie.main import KineticsAnalyzer
from pykingenie.kingenie_solution import KinGenieCsvSolution

pyKinetics = KineticsAnalyzer()

kingenie = KinGenieCsvSolution('test_kingenie_csv')

test_file = "./test_files/kingenie_solution.csv"
kingenie.read_csv(test_file)

test_file_2 = "./test_files/sim_solution_kon-100_koff-100_kc-1_krev-10.csv"
kingenie2  = KinGenieCsvSolution('test_kingenie_csv_2')
kingenie2.read_csv(test_file_2)

test_file_3 = "./test_files/sim_solution_kon-100_koff-100_kc-1_krev-10_ESint-15_ES-5.csv"
kingenie3  = KinGenieCsvSolution('test_kingenie_csv_3')
kingenie3.read_csv(test_file_3)

test_file_4 = "./test_files/sim_solution_Kd-0.1_koff-0.01_E-20_S-0_ES-0.csv"
kingenie4  = KinGenieCsvSolution('test_kingenie_csv_4')
kingenie4.read_csv(test_file_4)

test_file_5 = "./test_files/sim_solution_CS_kon-100_koff-1_kc-10_krev-100.csv"
kingenie5  = KinGenieCsvSolution('test_kingenie_csv_5')
kingenie5.read_csv(test_file_5)

test_file_6 = "./test_files/sim_cs_kon-100_koff-1_kc-10_krev-100_fit_signal_E.csv"
kingenie6  = KinGenieCsvSolution('test_kingenie_csv_6')
kingenie6.read_csv(test_file_6)


def test_load_solution_data():

    pyKinetics.add_experiment(kingenie, 'test_kingenie_csv')
    pyKinetics.merge_conc_df_solution()

    df = pyKinetics.combined_conc_df.iloc[4:,:].copy() # Last 4 curves only to make it faster

    assert not df.empty, "Combined concentration DataFrame should not be empty after merging."

    # generate fittings
    pyKinetics.generate_fittings_solution(df)

    # Remove the 'Experiment' column to trigger code execution when generating fittings
    df = df.drop(columns=["Experiment"], errors="ignore")

    pyKinetics.init_fittings()
    pyKinetics.generate_fittings_solution(df)

    assert len(pyKinetics.fittings_names) > 0

    # check value error with invalid fitting model
    with pytest.raises(ValueError):
        pyKinetics.submit_fitting_solution(fitting_model='invalid_model')

    pyKinetics.submit_fitting_solution(fitting_model='single')

    k_obs = pyKinetics.get_experiment_properties('k_obs', fittings=True)[0]

    expected = [0.22084, 0.28767, 0.39540,0.56919]

    assert np.allclose(k_obs, expected,rtol=0.0001), f"Expected {expected}, got {k_obs}"

    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site')

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert len(fit_params_kinetics) == 4, "There should be 4 sets of fit parameters for kinetics."

    assert np.allclose(fit_params_kinetics['Kd [µM]'][0],0.1,rtol=0.01)


def test_submit_fitting_solution_simple_fit_t0():

    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site',fixed_t0=False)

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert len(fit_params_kinetics) == 4, "There should be 4 sets of fit parameters for kinetics."


def test_submit_fitting_solution_simple_fit_2():

    pyKinetics.submit_fitting_solution(
        fitting_model='one_binding_site',
        fixed_t0=True,
        fit_signal_E=True, # should be close to zero
        fit_signal_S=False,
        fit_signal_ES=True)

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert np.allclose(fit_params_kinetics['Kd [µM]'][0],0.1,rtol=0.01)


def test_submit_fitting_solution_simple_fit_3():

    pyKinetics.submit_fitting_solution(
        fitting_model='one_binding_site',
        fixed_t0=True,
        fit_signal_E=False,
        fit_signal_S=True, # should be close to zero
        fit_signal_ES=True)

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert np.allclose(fit_params_kinetics['Kd [µM]'][0],0.1,rtol=0.01)


def test_submit_fitting_solution_if():

    pyKinetics.delete_experiment('test_kingenie_csv')
    pyKinetics.add_experiment(kingenie2, 'test_kingenie_csv_2')
    pyKinetics.merge_conc_df_solution()

    df = pyKinetics.combined_conc_df.iloc[5:,:].copy() # Last 3 curves only to make it faster

    assert not df.empty, "Combined concentration DataFrame should not be empty after merging."

    pyKinetics.generate_fittings_solution(df)

    pyKinetics.submit_fitting_solution(fitting_model='double')

    fitting_names   = pyKinetics.fittings_names
    fitter_solution = pyKinetics.fittings[fitting_names[0]]

    # Check that self.kobs_x is not filled with NaN values
    assert not np.isnan(fitter_solution.k_obs_1).all(), "kobs_1 should not be all NaN values."
    assert not np.isnan(fitter_solution.k_obs_2).all(), "kobs_2 should not be all NaN values."

    kwargs = {
        "fit_signal_E": False,  # E alone does not produce a signal
        "fit_signal_S": False, # S alone does not produce a signal
        "fit_signal_ES": True, # The complex ES and ES_int produce a signal
        "ESint_equals_ES": True, # ES_int produces a signal equal to ES
        "fixed_t0": True # t0 is fixed to 0
    }

    # fit one binding site with induced fit and verify parameters
    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site_if',**kwargs)

    assert np.isclose(fitter_solution.fit_params_kinetics['k_c [1/s]'].iloc[0], 1, rtol=0.1), "k_c should be close to 1 (within 10%)."
    assert np.isclose(fitter_solution.fit_params_kinetics['k_rev [1/s]'].iloc[0], 10, rtol=0.1), "k_rev should be close to 10 (within 10%)."
    assert np.isclose(fitter_solution.fit_params_kinetics['k_on [1/(µM·s)]'].iloc[0], 100, rtol=0.1), "k_on should be close to 100 (within 10%)."
    assert np.isclose(fitter_solution.fit_params_kinetics['k_off [1/s]'].iloc[0], 100, rtol=0.1), "k_off should be close to 100 (within 10%)."


def test_submit_fitting_solution_if_2():

    kwargs = {
        "fit_signal_E": True,  # E does not produce a signal in the simulated data - we use it to trigger code execution
        "fit_signal_S": True, # S does not produce a signal in the simulated data - we use it to trigger code execution
        "fit_signal_ES": True, # The complex ES and ES_int produce a signal
        "ESint_equals_ES": True, # ES_int produces a signal equal to ES
        "fixed_t0": True # t0 is fixed to 0
    }

    # fit one binding site with induced fit and verify parameters
    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site_if',**kwargs)

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert np.isclose(fit_params_kinetics['k_c [1/s]'].iloc[0], 1, rtol=0.1), "k_c should be close to 1 (within 10%)."
    assert np.isclose(fit_params_kinetics['k_rev [1/s]'].iloc[0], 10, rtol=0.1), "k_rev should be close to 10 (within 10%)."
    assert np.isclose(fit_params_kinetics['k_on [1/(µM·s)]'].iloc[0], 100, rtol=0.1), "k_on should be close to 100 (within 10%)."
    assert np.isclose(fit_params_kinetics['k_off [1/s]'].iloc[0], 100, rtol=0.1), "k_off should be close to 100 (within 10%)."


def test_submit_fitting_solution_if_with_t0():

    kwargs = {
        "fit_signal_E": False,  # E alone does not produce a signal
        "fit_signal_S": False, # S alone does not produce a signal
        "fit_signal_ES": True, # The complex ES and ES_int produce a signal
        "ESint_equals_ES": True, # ES_int produces a signal equal to ES
        "fixed_t0": False # t0 is fixed to 0
    }

    # fit one binding site with induced fit and verify parameters
    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site_if',**kwargs)

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert np.isclose(fit_params_kinetics['k_c [1/s]'].iloc[0], 1,
                      rtol=0.1), "k_c should be close to 1 (within 10%)."
    assert np.isclose(fit_params_kinetics['k_rev [1/s]'].iloc[0], 10,
                      rtol=0.1), "k_rev should be close to 10 (within 10%)."
    assert np.isclose(fit_params_kinetics['k_on [1/(µM·s)]'].iloc[0], 100,
                      rtol=0.1), "k_on should be close to 100 (within 10%)."
    assert np.isclose(fit_params_kinetics['k_off [1/s]'].iloc[0], 100,
                      rtol=0.1), "k_off should be close to 100 (within 10%)."


def test_submit_fitting_solution_if_ESint():

    pyKinetics.delete_experiment('test_kingenie_csv_2')
    pyKinetics.add_experiment(kingenie3, 'test_kingenie_csv_3')
    pyKinetics.merge_conc_df_solution()

    df = pyKinetics.combined_conc_df

    assert not df.empty, "Combined concentration DataFrame should not be empty after merging."

    pyKinetics.generate_fittings_solution(df)

    kwargs = {
        "fit_signal_E": False,  # E alone does not produce a signal
        "fit_signal_S": False, # S alone does not produce a signal
        "fit_signal_ES": True, # The complex ES and ES_int produce a signal
        "ESint_equals_ES": False, # ES_int produces a signal not equal to ES
        "fixed_t0": True # t0 is fixed to 0
    }

    # fit one binding site with induced fit and verify parameters
    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site_if',**kwargs)

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert np.isclose(fit_params_kinetics['k_c [1/s]'].iloc[0], 1, rtol=0.1), "k_c should be close to 1 (within 10%)."
    assert np.isclose(fit_params_kinetics['k_rev [1/s]'].iloc[0], 10, rtol=0.1), "k_rev should be close to 10 (within 10%)."
    assert np.isclose(fit_params_kinetics['k_on [1/(µM·s)]'].iloc[0], 100, rtol=0.1), "k_on should be close to 100 (within 10%)."
    assert np.isclose(fit_params_kinetics['k_off [1/s]'].iloc[0], 100, rtol=0.1), "k_off should be close to 100 (within 10%)."

def test_submit_fitting_solution_simple_fit_E():

    pyKinetics.delete_experiment('test_kingenie_csv_3')
    pyKinetics.add_experiment(kingenie4, 'test_kingenie_csv_4')
    pyKinetics.merge_conc_df_solution()

    df = pyKinetics.combined_conc_df

    pyKinetics.generate_fittings_solution(df)

    kwargs = {
        "fit_signal_E": True,  # E alone does produce a signal
        "fit_signal_S": False, # S alone does not produce a signal
        "fit_signal_ES": False, # The complex ES and ES_int do not produce a signal
        "fixed_t0": True # t0 is fixed to 0
    }

    # fit one binding site with induced fit and verify parameters
    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site',**kwargs)

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert np.allclose(fit_params_kinetics['Kd [µM]'][0],0.1,rtol=0.01)

def test_submit_fitting_solution_cs():

    pyKinetics.delete_experiment('test_kingenie_csv_4')
    pyKinetics.add_experiment(kingenie5, 'test_kingenie_csv_5')
    pyKinetics.merge_conc_df_solution()

    df = pyKinetics.combined_conc_df

    pyKinetics.generate_fittings_solution(df)

    kwargs = {
        "fit_signal_E": False,  # E alone does not produce a signal
        "E1_equals_E2": True, # E1 and E2 produce the same signal, in this case it does not matter because fit_signal_E is false
        "fit_signal_S": False, # S alone does not produce a signal
        "fit_signal_E2S": True, # The complex ES and ES_int produce a signal
        "fixed_t0": True # t0 is fixed to 0
    }

    # fit one binding site with induced fit and verify parameters
    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site_cs',**kwargs)

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert np.allclose(fit_params_kinetics['k_off [1/s]'][0],1,rtol=0.1)

    assert np.allclose(fit_params_kinetics['k_rev [1/s]'][0],100,rtol=0.1)

def test_submit_fitting_solution_cs_with_t0():

    kwargs = {
        "fit_signal_E": False,  # E alone does not produce a signal
        "E1_equals_E2": True, # E1 and E2 produce the same signal, in this case it does not matter because fit_signal_E is false
        "fit_signal_S": False, # S alone does not produce a signal
        "fit_signal_E2S": True, # The complex ES and ES_int produce a signal
        "fixed_t0": False # t0 is fitted
    }

    # fit one binding site with induced fit and verify parameters
    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site_cs',**kwargs)

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert np.allclose(fit_params_kinetics['k_off [1/s]'][0],1,rtol=0.1)

    assert np.allclose(fit_params_kinetics['k_rev [1/s]'][0],100,rtol=0.1)

def test_submit_fitting_solution_cs_fit_signal_E():

    pyKinetics.delete_experiment('test_kingenie_csv_5')
    pyKinetics.add_experiment(kingenie6, 'test_kingenie_csv_6')
    pyKinetics.merge_conc_df_solution()

    df = pyKinetics.combined_conc_df

    pyKinetics.generate_fittings_solution(df)

    kwargs = {
        "fit_signal_E": True,  # E alone does not produce a signal
        "E1_equals_E2": True, # just to force code execution
        "fit_signal_S": False, # S alone does not produce a signal
        "fit_signal_E2S": False, # The complex ES and ES_int produce a signal
        "fixed_t0": True # t0 is fixed to 0
    }

    # fit one binding site with induced fit and verify parameters
    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site_cs',**kwargs)

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert np.allclose(fit_params_kinetics['k_off [1/s]'][0],1,rtol=0.1)

    assert np.allclose(fit_params_kinetics['k_rev [1/s]'][0],100,rtol=0.1)