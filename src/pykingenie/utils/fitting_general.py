import itertools
import numpy as np
from scipy.optimize import curve_fit
from ..utils.math import *

def fit_single_exponential(y,t,min_log_k=-5, max_log_k=5,log_k_points=50):

    """

    Fit a single exponential to a signal

    Args:

        y (np.ndarray): signal
        t (np.ndarray): time

    Returns:

        fit_params (np.ndarray): fitted parameters

    """


    # Convert to numpy array, if needed,
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(t, np.ndarray):
        t = np.array(t)

    t = t - np.min(t)  # Start at zero

    possible_k = np.logspace(min_log_k, max_log_k, log_k_points)

    # define a single exponential function reduced to fit only a0 and a1
    def single_exponential_reduced(t, a0, a1):
        return a0 + a1 * np.exp(-k_obs * t)

    rss         = np.inf
    best_params = None
    best_k_obs  = None

    # Loop through each k_obs value
    for k_obs in possible_k:

        try:
            # Fit the reduced model
            params, cov = curve_fit(single_exponential_reduced, t, y, p0=[0, np.max(y)])

            # Calculate the residual sum of squares
            rss_temp = np.sum((y - single_exponential_reduced(t, *params)) ** 2)

            if rss_temp < rss:
                rss         = rss_temp
                best_params = params
                best_k_obs  = k_obs

        except Exception as e:
            # If fitting fails, continue to the next k_obs
            continue

    p0 = [best_params[0], best_params[1], best_k_obs]

    fit_params, cov = curve_fit(single_exponential, t, y,p0=p0)

    fit_y = single_exponential(t, *fit_params)

    return fit_params, cov, fit_y

def fit_double_exponential(y,t,min_log_k=-4, max_log_k=4,log_k_points=22):

    """

    Fit a double exponential to a signal

    Args:

        y (np.ndarray): signal
        t (np.ndarray): time

    Returns:

        fit_params (np.ndarray): fitted parameters

    """

    # Convert to numpy array, if needed,
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(t, np.ndarray):
        t = np.array(t)

    # Define a two columns dataframe with possible values for k obs 1 and k obs 2
    # Evenly spaced in logarithmic scale
    k_obs_1 = np.logspace(min_log_k, max_log_k, log_k_points)

    combinations = np.array(list(itertools.product(k_obs_1, k_obs_1)))

    combinations_df = pd.DataFrame(combinations, columns=['k_obs_1', 'k_obs_2'])

    # Remove all combinations where k_obs_1 is larger than k_obs_2
    combinations_df = combinations_df[combinations_df['k_obs_1'] < combinations_df['k_obs_2']]

    rss_min = np.inf

    # Loop through each combination of k_obs_1 and k_obs_2
    for index, row in combinations_df.iterrows():

        k_obs1 = row['k_obs_1']
        k_obs2 = row['k_obs_2']

        def double_exponential_reduced(t, a0, a1, a2):

            return a0 + a1 * np.exp(-k_obs1 * t) + a2 * np.exp(-k_obs2 * t)

        try:

            params, cov = curve_fit(double_exponential_reduced, t, y, p0=[0, np.max(y), np.max(y)])

            rss = np.sum((y - double_exponential_reduced(t, *params)) ** 2)

            if rss < rss_min:
                rss_min = rss
                best_params = params
                best_k_obs1 = k_obs1
                best_k_obs2 = k_obs2

        except:

            pass

    # Now fit the full double exponential model
    a0 = best_params[0]
    a1 = best_params[1]
    a2 = best_params[2]

    p0 = [a0, a1, best_k_obs1, a2, best_k_obs2]

    params, cov = curve_fit(double_exponential, t, y, p0=p0)
    fitted_y    = double_exponential(t, *params)

    return params, cov, fitted_y