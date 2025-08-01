import numpy as np
import pytest
import os

from pykingenie.kingenie_solution import KinGenieCsvSolution

kingenie = KinGenieCsvSolution('test_kingenie_csv')

test_file = "./test_files/kingenie_solution.csv"

def test_instance_creation():
    """
    Test the creation of a KinGenieCsv instance.
    """
    assert isinstance(kingenie, KinGenieCsvSolution), "The instance should be of type KinGenieCsv."
    assert kingenie.name == "test_kingenie_csv"

def test_read_csv():

    kingenie.read_csv(test_file)

    # check kingenie.xs has some data
    assert len(kingenie.xs) > 0, "The xs list should not be empty after reading the CSV file."
