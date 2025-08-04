import pytest
import numpy as np

from pykingenie.utils.fitting_general import fit_many_double_exponential

def test_fit_many_double_exponential():

    k_obs_1, _, _ = fit_many_double_exponential(
        signal_lst=[1,2],
        time_lst=[1,2]
    )

    # assert that k_obs_1 is a list of NaN values - same length as signal_lst
    assert len(k_obs_1) == 2, "k_obs_1 should have the same length as signal_lst."

    # Verift that the first value is NaN
    assert k_obs_1[0] is np.nan, "The first value of k_obs_1 should be NaN."