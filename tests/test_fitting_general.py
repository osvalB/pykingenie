import numpy as np
import pytest

from numpy.testing import assert_allclose

from pykingenie.utils.fitting_general import (
    convert_to_numpy_array,
    fit_single_exponential
)

def test_convert_to_numpy_array():

    l = [1, 2, 3]
    arr = convert_to_numpy_array(l)
    assert isinstance(arr, np.ndarray), "The output should be a numpy array"

    l = np.array([1, 2, 3])
    arr = convert_to_numpy_array(l)
    assert isinstance(arr, np.ndarray), "The output should be a numpy array"

def test_fit_single_exponential():

    t = np.arange(0, 5, 0.1)
    k_obs = 5
    a0 = 0.5
    a1 = 10
    y = a0 + a1 * np.exp(-k_obs * t)

    fit_params, cov, fit_y = fit_single_exponential(y,t)

    assert_allclose(fit_params, [a0,a1,k_obs], rtol=0.01)