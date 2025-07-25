import numpy as np
import pytest
import os

from src.pykingenie.kingenie_surface import KinGenieCsv

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
