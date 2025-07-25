import numpy as np
import pandas as pd
import pytest
import os

from src.pykingenie.main  import KineticsAnalyzer
from src.pykingenie.octet import OctetExperiment

pyKinetics = KineticsAnalyzer()

folder = "./test_files/"
frd_files = os.listdir(folder)

frd_files = [os.path.join(folder, file) for file in frd_files if file.endswith('.frd')]
frd_files.sort()

bli = OctetExperiment('test')
bli.read_sensor_data(frd_files)

def test_instance_creation():
    assert isinstance(pyKinetics, KineticsAnalyzer)

def test_add_experiment():

    pyKinetics.add_experiment(bli, 'test_octet')

    assert len(pyKinetics.experiments) == 1

def test_get_experiment_properties():

    sensor_names = pyKinetics.get_experiment_properties('sensor_names',fittings=False)[0]

    assert sensor_names == ['A1','B1','C1','D1','E1','F1','G1','H1']

def test_merge_ligand_conc_df():

    pyKinetics.merge_ligand_conc_df()

    assert len(pyKinetics.combined_ligand_conc_df) > 0, "The ligand concentration DataFrame should not be empty after merging."

def test_remove_experiment():

    pyKinetics.delete_experiment('test_octet')

    assert len(pyKinetics.experiments) == 0

def test_generate_fittings():

    bli.align_association(bli.sensor_names)
    bli.align_dissociation(bli.sensor_names)
    bli.subtraction(['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1'],'H1')

    pyKinetics.add_experiment(bli, 'test_octet')

    pyKinetics.merge_ligand_conc_df()

    df = pyKinetics.combined_ligand_conc_df

    # Select only the first 8 rows for testing
    df = df.iloc[:8, :].copy()

    pyKinetics.generate_fittings(df)

    assert len(pyKinetics.fittings_names) > 0

def test_submit_steady_state_fitting():

    pyKinetics.submit_steady_state_fitting()

    Kd_ss = pyKinetics.get_experiment_properties('Kd_ss', fittings=True)

    expected = [0.019602607912092396]

    assert np.allclose(Kd_ss, expected), f"Expected {expected}, got {Kd_ss}"

def test_submit_kinetic_fitting():

    pyKinetics.submit_kinetics_fitting(fitting_model='one_to_one',
                                       fitting_region='dissociation',
                                       linkedSmax=False)

    k_off = pyKinetics.get_experiment_properties('Koff', fittings=True)[0]

    assert np.round(k_off, decimals=3) == 0.002

    pyKinetics.submit_kinetics_fitting(fitting_model='one_to_one',
                                       fitting_region='association_dissociation',
                                       linkedSmax=False)

    Kd = pyKinetics.get_experiment_properties('Kd', fittings=True)[0]

    assert np.round(Kd,decimals=3) == 0.006

    pyKinetics.submit_kinetics_fitting(fitting_model='one_to_one',
                                       fitting_region='association_dissociation',
                                       linkedSmax=True)

    Kd = pyKinetics.get_experiment_properties('Kd', fittings=True)[0]

    assert np.round(Kd,decimals=3) == 0.01

def test_calculate_asymmetric_error():

    pyKinetics.calculate_asymmetric_error(shared_smax=True, fixed_t0=True, fit_ktr=False)

    print('here')

    asymmetric_error_df = pyKinetics.get_experiment_properties('fit_params_kinetics_ci95', fittings=True)[0]

    assert np.round(float(asymmetric_error_df.iloc[0,1]),3) == 0.01

def test_get_fitting_results():

    pyKinetics.get_fitting_results()

    df = pyKinetics.fit_params_kinetics_all

    assert isinstance(df, pd.DataFrame), "Fitting results should be a dataframe."
    assert not df.empty, "Fitting results dataframe should not be empty."


