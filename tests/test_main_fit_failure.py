import numpy as np
import pandas as pd
import pytest
import os

from pykingenie.main  import KineticsAnalyzer
from pykingenie.octet import OctetExperiment
from pykingenie.fitter_surface import KineticsFitter

pyKinetics = KineticsAnalyzer()

folder = "./test_files/"
frd_files = os.listdir(folder)

frd_files = [os.path.join(folder, file) for file in frd_files if file.endswith('.frd') and file.startswith('230309')]
frd_files.sort()

bli = OctetExperiment('test')
bli.read_sensor_data(frd_files)
pyKinetics.add_experiment(bli, 'test_octet')
pyKinetics.init_fittings()

def test_fitting_failure():

    # comment processing steps to trigger fitting failure...

    #bli.align_association(bli.sensor_names)
    #bli.align_dissociation(bli.sensor_names)
    #bli.subtraction(['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1'],'H1')

    pyKinetics.add_experiment(bli, 'test_octet')

    pyKinetics.merge_ligand_conc_df()

    df = pyKinetics.combined_ligand_conc_df

    # Select only the first 8 rows for testing
    df = df.iloc[:8, :].copy()

    pyKinetics.generate_fittings(df)

    good_fits, bad_fits = pyKinetics.submit_kinetics_fitting(fitting_model='one_to_one',
                                       fitting_region='association_dissociation',
                                       shared_smax=False)

    assert len(bad_fits) == 1, "Expected one fitting failure but none were found."
    assert len(good_fits) == 0, "Expected none good fits but some were found."