import pytest
import numpy as np

from pykingenie.utils.fitting_general import fit_many_double_exponential
from pykingenie.utils.math import double_exponential

from numpy.testing import assert_allclose

t = np.linspace(0, 10, 100)

a0 = 1
a1 = 0.1
a2 = 0.2
kobs1 = 0.5
kobs2 = 5

y = double_exponential(t, a0, a1, kobs1, a2, kobs2)

def test_fit_many_double_exponential():

    k_obs_1_fit, k_obs_2_fit, _,_,_ = fit_many_double_exponential(
        signal_lst=[y],
        time_lst=[t]
    )

    # assert that k_obs_1 is a list of NaN values - same length as signal_lst
    assert len(k_obs_1_fit) == 1, "k_obs_1 should have the same length as signal_lst."

    assert_allclose(k_obs_1_fit[0],kobs1,rtol=0.1)

    assert_allclose(k_obs_2_fit[0],kobs2,rtol=0.1)


def test_fit_many_double_exponential_no_data():

    k_obs_1, _, _, _,_ = fit_many_double_exponential(
        signal_lst=[1,2],
        time_lst=[1,2]
    )

    # assert that k_obs_1 is a list of NaN values - same length as signal_lst
    assert len(k_obs_1) == 2, "k_obs_1 should have the same length as signal_lst."

    # Verift that the first value is NaN
    assert k_obs_1[0] is np.nan, "The first value of k_obs_1 should be NaN."