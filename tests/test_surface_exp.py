import numpy as np
import pytest
import os

from pykingenie.octet import OctetExperiment
from pykingenie.kingenie_surface import KinGenieCsv

folder = "./test_files/"
frd_files = os.listdir(folder)
frd_files = [os.path.join(folder, file) for file in frd_files if file.endswith('.frd') and file.startswith('230309')]

frd_files.sort() # from sensor A1 to H1

# Create an instance of OctetExperiment
bli = OctetExperiment('test_octet')

bli.read_sensor_data(frd_files)

sensor_names = bli.sensor_names.copy()

csv_test_file = "./test_files/single_cycle_kingenie.csv"
kingenie = KinGenieCsv()
kingenie.read_csv(csv_test_file)

def test_subtract_no_traces_loaded():

    # bli_temp has no traces loaded
    bli_temp = OctetExperiment('test_octet')
    with pytest.raises(RuntimeError, match="No traces loaded. Cannot perform subtraction."):
        bli_temp.subtraction_one_to_one(sensor_name1='A1', sensor_name2='B1')

def test_average_no_traces_loaded():
    
    # bli_temp has no traces loaded
    bli_temp = OctetExperiment('test_octet')
    with pytest.raises(RuntimeError, match="No traces loaded. Cannot perform averaging."):
        bli_temp.average(list_of_sensor_names=['A1', 'B1'], new_sensor_name='Average')

def test_subtract_incompatible():
    
    # bli_temp has no traces loaded
    bli_temp = OctetExperiment('test_octet')
    bli_temp.read_sensor_data(frd_files[:2])  # Load only A1, B1

    bli_temp.xs[0] = bli_temp.xs[0][:5] # Reduce the number of steps for sensor A1

    with pytest.raises(RuntimeError, match="Sensors have different number of steps"):
        bli_temp.subtraction_one_to_one(sensor_name1='A1', sensor_name2='B1')    

    bli_temp.xs[0] = bli_temp.xs[1].copy() # Reduce the number of data points for sensor A1
    bli_temp.xs[0][0] = bli_temp.xs[0][0][:5] # Reduce the number of steps for sensor A1

    with pytest.raises(RuntimeError, match="Sensors have different number of points"):
        bli_temp.subtraction_one_to_one(sensor_name1='A1', sensor_name2='B1')


def test_subtract_repeated_names():

    bli.subtraction_one_to_one(sensor_name1='A1', sensor_name2='H1',inplace=False)
    assert len(bli.sensor_names) == len(sensor_names) + 1, "The sensor_names list should have one more sensor after subtraction."

    bli.subtraction_one_to_one(sensor_name1='A1', sensor_name2='H1', inplace=False)
    assert len(bli.sensor_names) == len(sensor_names) + 2, "The sensor_names list should have two more sensors after repeated subtraction."

    # Delete the newly created sensor to avoid conflicts in other tests
    bli.sensor_names = bli.sensor_names[:len(sensor_names)]
    bli.xs = bli.xs[:len(sensor_names)]
    bli.ys = bli.ys[:len(sensor_names)]


def test_align_association():

    # Sensor names equal None means all sensors
    bli.align_association(sensor_names = None,inplace = True,new_names = False)

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

def test_subtraction_string_as_sensor_names():

    bli.subtraction(list_of_sensor_names=bli.sensor_names[0], reference_sensor=bli.sensor_names[-1], inplace=False)

    assert bli.xs is not None, "The xs list should not be None after subtraction."

## Now we test the KinGenieCsv 
def test_align_single_cycle():

    sensor_names = kingenie.sensor_names.copy()

    kingenie.align_association(sensor_names=sensor_names, inplace=False, new_names=True)

    assert len(kingenie.sensor_names) == 2, "The sensor_names list should have two sensors after aligning association with new names."

    # Align again to force the use of 'rep' in the new sensor names
    kingenie.align_association(sensor_names=sensor_names, inplace=False, new_names=True)

    assert len(kingenie.sensor_names) == 3, "The sensor_names list should have three sensors after aligning association with new names."

    # Now align in place with new names

    kingenie.align_association(sensor_names=sensor_names, inplace=True, new_names=True)

    assert len(kingenie.sensor_names) == 3, "The sensor_names list should have three sensors after aligning association in place with new names."

    # align the dissociation steps - not inplace and with new names twice to force the use of 'rep' in the new sensor names

    sensor_names = kingenie.sensor_names.copy()[:1]

    kingenie.align_dissociation(sensor_names=sensor_names, inplace=False, new_names=True)
    kingenie.align_dissociation(sensor_names=sensor_names, inplace=False, new_names=True)

    assert len(kingenie.sensor_names) == (len(sensor_names)*5), "The sensor_names list should have five times the number of sensors after aligning dissociation with new names."

def test_get_step_xy_kingenie():

    # check raise value error if location_sample is not an integer
    with pytest.raises(ValueError, match="location_sample, location_loading and replicate must be integers"):
        kingenie.get_step_xy(
            sensor_name=kingenie.sensor_names[0],
            location_loading=12,
            location_sample='hola',
            step_type='ASSOC',
            replicate=1,
            type='x')
        
    # check raise  Type error if sensor_name is not a string
    with pytest.raises(TypeError, match="sensor_name must be a str"):
        kingenie.get_step_xy(
            sensor_name=123,
            location_loading=12,
            location_sample=5,
            step_type='ASSOC',
            replicate=1,
            type='x')
    
    # Extract x and y values for four association steps
    sensor_name = kingenie.sensor_names[0]

    for i in range(1, 4):

        x = kingenie.get_step_xy(
            sensor_name=sensor_name,
            location_loading=1,
            location_sample=1+i,
            step_type='ASSOC',
            replicate=1,
            type='x')  
        
        # Check if x is a numpy array
        assert isinstance(x, np.ndarray), "The x should be a numpy array."
    
        y = kingenie.get_step_xy(
            sensor_name=sensor_name,
            location_loading=1,
            location_sample=1+i,
            step_type='ASSOC',
            replicate=1,
            type='y')
        
        # Check if y is a numpy array
        assert isinstance(y, np.ndarray), "The y should be a numpy array."

        x = kingenie.get_step_xy(
            sensor_name=sensor_name,
            location_loading=1,
            location_sample=1+i,
            step_type='DISASSOC',
            replicate=1,
            type='x')  
        
        # Check if x is a numpy array
        assert isinstance(x, np.ndarray), "The x should be a numpy array."
    
        y = kingenie.get_step_xy(
            sensor_name=sensor_name,
            location_loading=1,
            location_sample=1+i,
            step_type='DISASSOC',
            replicate=1,
            type='y')
        
        # Check if y is a numpy array

def test_subtraction_by_column():

    bli_temp = OctetExperiment('test_octet')
    bli_temp.read_sensor_data(frd_files[:2])  # Load only A1, B1

    bli_temp.sensor_names = ['A1', 'A2'] # Rename sensors to allow subtraction by column

    result = bli_temp.subtraction_by_column(1, 2)

    assert result is not None, "The subtraction by column should return a result."

    assert isinstance(result, list), "The result should be a list."

    assert isinstance(result[0], str), "The first element of the result should be a string."