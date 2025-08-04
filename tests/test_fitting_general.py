import numpy as np
import pytest

from pykingenie.utils.fitting_general import convert_to_numpy_array

def test_convert_to_numpy_array():

    l = [1, 2, 3]
    arr = convert_to_numpy_array(l)
    assert isinstance(arr, np.ndarray), "The output should be a numpy array"

    l = np.array([1, 2, 3])
    arr = convert_to_numpy_array(l)
    assert isinstance(arr, np.ndarray), "The output should be a numpy array"