import numpy as np
import pytest
import os

from pykingenie.gator import GatorExperiment

gator_zip_folder = "./test_files/gator_example_folder.zip"
csv_file         = "./test_files/Assay_1_Channel1.csv"

exp_ini_file = "./test_files/ExperimentStep.ini"
set_ini_file = "./test_files/Setting.ini"

def test_instance_creation():

    gator = GatorExperiment('test_gator')
    assert isinstance(gator, GatorExperiment), "The instance should be of type GatorExperiment."
    assert gator.name == "test_gator"

def test_read_all_gator_data():

    gator = GatorExperiment('test_gator')
    gator.read_all_gator_data(gator_zip_folder)

    assert len(gator.xs) > 0, "The xs array should not be empty after reading the Gator data."
