import numpy as np
import pytest
import os

from src.pykingenie.octet import OctetExperiment

folder = "./test_files/"
frd_files = os.listdir(folder)
frd_files = [os.path.join(folder, file) for file in frd_files if file.endswith('.frd')]

frd_files.sort() # from sensor A1 to H1

# Create an instance of OctetExperiment
bli = OctetExperiment('test_octet')

bli.read_sensor_data(frd_files)

sensor_names = bli.sensor_names.copy()

def test_align_association():

    bli.align_association(sensor_names = sensor_names,inplace = True,new_names = False)

    assert len(bli.sensor_names) == 8, "The sensor_names list should have 8 sensors after aligning association."
    assert bli.xs is not None, "The xs list should not be None after aligning association."

    bli.align_association(sensor_names=sensor_names, inplace=False, new_names=True)

    assert len(bli.sensor_names) == (len(sensor_names)*2), "The sensor_names list should have twice the number of sensors after aligning with new names."

    # Delete the newly created sensors to avoid conflicts in other tests
    bli.sensor_names = bli.sensor_names[:len(sensor_names)]
    bli.xs = bli.xs[:len(sensor_names)]
    bli.ys = bli.ys[:len(sensor_names)]


def test_align_dissociation():

    bli.align_dissociation(sensor_names = sensor_names,inplace = True,new_names = False)

    assert len(bli.sensor_names) == 8, "The sensor_names list should have 8 sensors after aligning dissociation."
    assert bli.xs is not None, "The xs list should not be None after aligning dissociation."

    bli.align_dissociation(sensor_names=sensor_names, inplace=False, new_names=True)

    assert len(bli.sensor_names) == (len(sensor_names)*2), "The sensor_names list should have twice the number of sensors after aligning with new names."

    # Delete the newly created sensors to avoid conflicts in other tests
    bli.sensor_names = bli.sensor_names[:len(sensor_names)]
    bli.xs = bli.xs[:len(sensor_names)]
    bli.ys = bli.ys[:len(sensor_names)]


def test_subtract():

    sensor_names_2 = [x for x in bli.sensor_names if x != 'H1']

    y_1 = bli.ys[0][0][0].copy()
    y_2 = bli.ys[-1][0][0]

    bli.subtraction(list_of_sensor_names = sensor_names_2,reference_sensor='H1', inplace = True)

    y_new = bli.ys[0][0][0]

    assert np.allclose(y_new, y_1 - y_2), "The subtraction should be correctly calculated."

    sensor_names_2 = [x for x in bli.sensor_names if x != 'H1']

    bli.subtraction(list_of_sensor_names=sensor_names_2, reference_sensor='H1', inplace=False)

    assert len(bli.sensor_names) == len(sensor_names)*2 - 1 # Double minus one because the reference sensor is not included

def test_average():

    bli.average(list_of_sensor_names=bli.sensor_names[:2],new_sensor_name='Average')

    y_1 = bli.ys[0][0][0]
    y_2 = bli.ys[1][0][0]

    assert bli.xs is not None, "The xs list should not be None after average."

    y_avg = bli.ys[-1][0][0]

    assert np.allclose(y_avg, (y_1 + y_2) / 2), "The average should be correctly calculated."

def test_discard_steps():

    bli.discard_steps(sensor_names=bli.sensor_names, step_types=['KREGENERATION','LOADING'])

    # Check for NaNs in all subarrays of bli.ys[0]
    nan_count = sum(np.isnan(subarr).sum() for subarr in bli.ys[0] if isinstance(subarr, np.ndarray))
    assert nan_count > 0, "The ys list should contain NaN values after discarding steps."


def test_get_step_xy():

    x = bli.get_step_xy(sensor_name = bli.sensor_names[0],
                    location_loading = 12,
                    location_sample = 5,
                    step_type='ASSOC',
                    replicate=1,
                    type='x')

    # Check if x is a numpy array
    assert isinstance(x, np.ndarray), "The x should be a numpy array."

    y = bli.get_step_xy(sensor_name=bli.sensor_names[0],
                        location_loading=12,
                        location_sample=5,
                        step_type='ASSOC',
                        replicate=1,
                        type='y')

    # Check if y is a numpy array
    assert isinstance(y, np.ndarray), "The y should be a numpy array."

    x = bli.get_step_xy(sensor_name = bli.sensor_names[0],
                    location_loading = 12,
                    location_sample = 5,
                    step_type='DISASSOC',
                    replicate=1,
                    type='x')

    # Check if x is a numpy array
    assert isinstance(x, np.ndarray), "The x should be a numpy array."

    y = bli.get_step_xy(sensor_name=bli.sensor_names[0],
                        location_loading=12,
                        location_sample=5,
                        step_type='DISASSOC',
                        replicate=1,
                        type='y')

    # Check if y is a numpy array
    assert isinstance(y, np.ndarray), "The y should be a numpy array."
