import numpy as np
import pytest
import os

from pykingenie.utils.math import single_exponential, double_exponential, median_filter, rss_p, get_rss, get_desired_rss

def test_single_exponential():
    t = np.array([0, 1, 2, 3, 4])
    a0 = 1.0
    a1 = 2.0
    kobs = 0.5

    result = single_exponential(t, a0, a1, kobs)
    expected = a0 + a1 * np.exp(-kobs * t)

    np.testing.assert_array_almost_equal(result, expected)

def test_double_exponential():
    t = np.array([0, 1, 2, 3, 4])
    a0 = 1.0
    a1 = 2.0
    kobs1 = 0.5
    a2 = 3.0
    kobs2 = 0.3

    result = double_exponential(t, a0, a1, kobs1, a2, kobs2)
    expected = a0 + a1 * np.exp(-kobs1 * t) + a2 * np.exp(-kobs2 * t)

    np.testing.assert_array_almost_equal(result, expected)

def test_median_filter():
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    x = np.arange(len(y)) / 100

    rolling_window = 3 / 100

    filtered_y = median_filter(y, x, rolling_window)

    # Check if the filtered_y is a numpy array
    assert isinstance(filtered_y, np.ndarray), "The filtered_y should be a numpy array."

    # Check the length of the filtered_y
    assert len(filtered_y) == len(y), "The length of filtered_y should match the length of y."

    expected_values_after_padding = np.array([2, 3, 4, 5, 6, 7, 8])

    # Check if the filtered values match the expected values
    np.testing.assert_array_almost_equal(filtered_y[2:], expected_values_after_padding)

def test_rss_p():

    rrs0 = 100
    n = 10
    p = 5
    alfa = 0.05

    result = rss_p(rrs0, n, p, alfa)

    np.testing.assert_almost_equal(result, 232.1578194)

def test_get_rss():

    y     = np.arange(10)
    y_fit = y + 1

    rss_expected = 10

    rss = get_rss(y, y_fit)

    np.testing.assert_equal(rss, rss_expected)

def test_get_desired_rss():

    y = np.arange(10)
    y_fit = y + 1
    n = len(y)
    p = 2
    alpha=0.05

    rss = get_desired_rss(y, y_fit, n, p, alpha)

    np.testing.assert_almost_equal(rss, 16.647068,decimal=5)
