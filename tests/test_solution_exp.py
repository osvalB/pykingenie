import numpy as np
import pytest
import os

from pykingenie.kingenie_solution import KinGenieCsvSolution

kingenie = KinGenieCsvSolution('test_kingenie_csv')

test_file = "./test_files/kingenie_solution.csv"

kingenie.read_csv(test_file)

def test_get_trace_xy():

    x = kingenie.get_trace_xy(kingenie.traces_names[0], type='x')

    assert x is not None, "The xs list should not be empty after reading the CSV file."

    y = kingenie.get_trace_xy(kingenie.traces_names[0], type='y')

    assert y is not None, "The ys list should not be empty after reading the CSV file."

def test_cut_off_time():

    x1 = kingenie.get_trace_xy(kingenie.traces_names[0], type='x')

    kingenie.cut_off_time(0.1)

    x2 = kingenie.get_trace_xy(kingenie.traces_names[0], type='x')

    assert len(x1) > len(x2), "The cut off time should remove some data points."

