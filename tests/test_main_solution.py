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


def test_load_solution_data():

    pyKinetics.add_experiment(kingenie, 'test_kingenie_csv')
    pyKinetics.merge_conc_df_solution()

    df = pyKinetics.combined_conc_df

    assert not df.empty, "Combined concentration DataFrame should not be empty after merging."

    pyKinetics.generate_fittings_solution(df)

    assert len(pyKinetics.fittings_names) > 0

    pyKinetics.submit_fitting_solution(fitting_model='single')

    k_obs = pyKinetics.get_experiment_properties('k_obs', fittings=True)[0]

    expected = [0.12707, 0.13733, 0.15369, 0.17947, 0.22084, 0.28767, 0.39540,0.56919]

    assert np.allclose(k_obs, expected,rtol=0.0001), f"Expected {expected}, got {k_obs}"

    pyKinetics.submit_fitting_solution(fitting_model='one_binding_site')

    fit_params_kinetics = pyKinetics.get_experiment_properties('fit_params_kinetics', fittings=True)[0]

    assert len(fit_params_kinetics) == 8, "There should be 8 sets of fit parameters for kinetics."

    assert np.allclose(fit_params_kinetics['Kd [µM]'][0],0.103944,rtol=0.001)








