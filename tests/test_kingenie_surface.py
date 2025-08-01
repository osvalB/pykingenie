import numpy as np
import pytest
import os

from pykingenie.kingenie_surface import KinGenieCsv

csv_file_single_cycle = "./test_files/single_cycle_kingenie.csv"
csv_file_multi_cycle  = "./test_files/multi_cycle_kingenie.csv"

kingenie_csv = KinGenieCsv('test_kingenie_csv')

def test_instance_creation():
    """
    Test the creation of a KinGenieCsv instance.
    """
    assert isinstance(kingenie_csv, KinGenieCsv), "The instance should be of type KinGenieCsv."
    assert kingenie_csv.name == "test_kingenie_csv"

def test_read_csv():

    kingenie_csv.read_csv(csv_file_single_cycle)

    assert len(kingenie_csv.xs) > 0, "The xs list should not be empty after reading a single cycle CSV file."

    kingenie_csv.read_csv(csv_file_multi_cycle)

    assert len(kingenie_csv.xs) > 0, "The xs list should not be empty after reading a multi-cycle CSV file."

def test_get_step_xy():
    """
    Test the get_step_xy method.
    """

    kingenie_csv.read_csv(csv_file_single_cycle)

    x = kingenie_csv.get_step_xy(kingenie_csv.sensor_names[0],
                                 location_loading=1, #  loading location for this simulated file
                                 location_sample=2,  # sample location for this simulated file
                                 step_type='ASSOC',
                                 replicate=1,
                                 type='x')
    
    # Check if x is a numpy array
    assert isinstance(x, np.ndarray), "The x should be a numpy array."

    y = kingenie_csv.get_step_xy(kingenie_csv.sensor_names[0],
                                    location_loading=1, #  loading location for this simulated file 
                                    location_sample=2,  # sample location for this simulated file
                                    step_type='ASSOC',
                                    replicate=1,
                                    type='y')
    
    # Check if y is a numpy array
    assert isinstance(y, np.ndarray), "The y should be a numpy array."

    x = kingenie_csv.get_step_xy(kingenie_csv.sensor_names[0],
                                    location_loading=1, #  loading location for this simulated file
                                    location_sample=3,  # sample location for this simulated file
                                    step_type='DISASSOC',   
                                    replicate=1,
                                    type='x')

    # Check if x is a numpy array
    assert isinstance(x, np.ndarray), "The x should be a numpy array."

    y = kingenie_csv.get_step_xy(kingenie_csv.sensor_names[0],
                                    location_loading=1, #  loading location for this simulated file
                                    location_sample=3,  # sample location for this simulated file
                                    step_type='DISASSOC',
                                    replicate=1,
                                    type='y')   
    # Check if y is a numpy array
    assert isinstance(y, np.ndarray), "The y should be a numpy array."
