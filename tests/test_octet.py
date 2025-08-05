import numpy as np
import pytest
import os

from pykingenie.octet import OctetExperiment

folder = "./test_files/"
frd_files = os.listdir(folder)

frd_files = [os.path.join(folder, file) for file in frd_files if file.endswith('.frd')]
fmf_file  = os.path.join(folder, '230309_ExpMethod.fmf')


def test_instance_creation():

    bli = OctetExperiment('test_octet')
    assert isinstance(bli, OctetExperiment), "The instance should be of type OctetExperiment."
    assert bli.name == "test_octet"

def test_import_str_file():

    bli = OctetExperiment('test_octet')
    bli.read_sensor_data(frd_files[0])

    assert len(bli.xs) > 0, "The xs array should not be empty after reading a single file."

def test_import_no_frd_files():

    bli = OctetExperiment('test_octet')
    bli.read_sensor_data(fmf_file)

    assert bli.xs is None, "The xs list should be still None when no .frd files are provided."

def test_import_file_with_no_loading_data():

    # Create an instance of OctetExperiment
    bli = OctetExperiment('test_octet')
    bli.read_sensor_data(frd_files)

    bli.df_steps.loc[bli.df_steps['Type'] == 'LOADING', 'Type'] = 'BASELINE'
    bli.generate_ligand_conc_df()

    assert len(bli.ligand_conc_df) > 0

def test_merge_consecutive_steps():

    # Create an instance of OctetExperiment
    bli = OctetExperiment('test_octet')

    bli.read_sensor_data(frd_files)

    x_ref      = bli.xs[0][2]
    x_to_merge = bli.xs[0][3]

    bli.merge_consecutive_steps(3,4)

    x_merged = bli.xs[0][2]

    assert all(np.concatenate((x_ref, x_to_merge)) == x_merged), "The consecutive steps should be merged correctly."

    # Test reading sensor data with a non-empty list (mocked)
    # This part would normally read files, but we can mock it for testing purposes
    # Here we assume that the method works correctly when provided with valid file paths

def test_merge_consecutive_steps_2():

    # Create an instance of OctetExperiment
    bli = OctetExperiment('test_octet')

    bli.read_sensor_data(frd_files)

    # modify t0 so t0 ref is higher than t0 to merge
    bli.xs[0][2] += 100

    x_ref      = bli.xs[0][2]
    x_to_merge = bli.xs[0][3]

    bli.merge_consecutive_steps(3,4)

    x_merged = bli.xs[0][2]

    assert all(np.concatenate((x_to_merge, x_ref)) == x_merged), "The consecutive steps should be merged correctly."

def test_merge_steps_by_name():

    # Create an instance of OctetExperiment
    bli = OctetExperiment('test_octet')

    bli.read_sensor_data(frd_files)

    # Merge consecutive steps by name
    bli.merge_consecutive_steps_by_name('Association')

    # Check if the xs list has been updated correctly
    assert len(bli.xs) > 0, "The xs list should not be empty after merging steps by name."

def test_read_sample_plate_info():

    # Create an instance of OctetExperiment
    bli = OctetExperiment('test_octet')

    bli.read_sensor_data(frd_files)

    # run read_sample_plate_info without a valid fmf file
    bli.read_sample_plate_info('test')

    assert not bli.sample_plate_loaded, "The sample plate should not be loaded when an invalid fmf file is provided."

    # Read the plate information
    bli.read_sample_plate_info(fmf_file)

    # Check if sample_row is not None
    assert bli.sample_row is not None, "The sample_row should not be None after reading plate info."



def test_convert_to_numbers():

    # Create an instance of OctetExperiment
    bli = OctetExperiment('test_octet')

    bli.read_sensor_data(frd_files)

    # List of entries in step info
    entries = ['Concentration', 'MolarConcentration', 'MolecularWeight', 'Temperature', 'StartTime',
               'AssayTime', 'FlowRate', 'ActualTime', 'CycleTime']

    # Change type to trigger exception
    bli.step_info[0]['Concentration'] = ['hello', 'world']

    bli.convert_to_numbers()

    assert all(bli.step_info[0]['Concentration'] == np.array([-1,-1]))