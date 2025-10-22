import numpy as np
import pytest

from numpy.testing import assert_almost_equal

from pykingenie.utils.fitting_surface import (
    guess_initial_signal,
    fit_one_site_dissociation
)

def test_guess_initial_signal_exception():

    assoc_time_lst = [[10]]
    assoc_signal_lst = [[10]]

    s0s = guess_initial_signal(assoc_time_lst,assoc_signal_lst)

    assert s0s[0] == 10, "The initial signal should be the same as the signal at the first time point"

def test_fit_one_site_dissociation_fixed_s0():

    t = np.linspace(0, 10, 50)
    s0 = 1.0
    k_off = 0.1

    y = s0 * np.exp(-k_off*t)

    initial_parameters = [0.5]

    low_bounds = [0.005]
    high_bounds = [10]

    fit_params, _, _ = fit_one_site_dissociation(
        [y], [t], initial_parameters, low_bounds, high_bounds,
        fit_s0=False)

    assert_almost_equal(fit_params[0],k_off,decimal=2)