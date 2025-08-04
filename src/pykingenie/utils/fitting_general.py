import itertools
import numpy as np
from scipy.optimize import curve_fit
from ..utils.math import *

__all__ = ['fit_single_exponential', 'fit_double_exponential','re_fit','re_fit_2']

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

def expand_high_bound(value,factor=10):

    """
    Multiply/divide the value by a factor and return the new value.
    If the parameter is negative, we divide by the factor, otherwise we multiply.

    Args:
        value (float): Parameter to expand
    Returns:
        new_param (float): New parameter value after expansion
    """

    new_value = value * factor if value >= 0 else value / factor

    return new_value

def expand_low_bound(value,factor=10):

    """
    Multiply/divide the value by a factor and return the new value.
    If the parameter is negative, we multiply by the factor, otherwise we divide.

    Args:
        value (float): Parameter to expand
    Returns:
        new_param (float): New parameter value after expansion
    """

    new_value = value * factor if value < 0 else value / factor

    return new_value

def re_fit(fit, cov, fit_vals,fit_fx,low_bounds,high_bounds,times,**kwargs):

    """
    Evaluate the difference between the fitted parameters and the initial parameters
    If the difference is less than 2 percent, the bounds are relaxed by a factor of 10
    and the fitting is repeated

    Args:
        fit (list): Fitted parameters
        cov (np.ndarray): Covariance matrix of the fitted parameters
        fit_vals (list): Fitted values for each signal, same dimensions as signal_lst
        fit_fx (function): Function to fit the data - returns fit, cov, fit_vals
        low_bounds (list): Lower bounds for the parameters
        high_bounds (list): Upper bounds for the parameters
        times (int): Number of times to re-fit the data
        **kwargs: Additional arguments to pass to the fitting function

    Returns:
        fit (list): Fitted parameters after re-fitting
        cov (np.ndarray): Covariance matrix of the fitted parameters after re-fitting
        fit_vals (list): Fitted values for each signal after re-fitting, same dimensions as signal_lst
        low_bounds (list): Lower bounds for the parameters after re-fitting
        high_bounds (list): Upper bounds for the parameters after re-fitting
    """

    fit         = np.array(fit).astype(float)
    low_bounds  = np.array(low_bounds).astype(float)
    high_bounds = np.array(high_bounds).astype(float)

    for _ in range(times):

        difference_to_upper = np.abs( (high_bounds - fit) / high_bounds )
        difference_to_lower = np.abs( (fit - low_bounds) / fit )

        c1 = any(difference_to_upper < 0.02)
        c2 = any(difference_to_lower < 0.02)

        if c1:

            # Relax bounds by a factor of 10 - only those that are too close to the upper bound
            high_bounds = [expand_high_bound(value, factor=10) if diff < 0.02 else value for value, diff in zip(high_bounds, difference_to_upper)]
            high_bounds = np.array(high_bounds)

        if c2:

            low_bounds = [expand_low_bound(value, factor=10) if diff < 0.02 else value for value, diff in zip(low_bounds, difference_to_lower)]
            low_bounds = np.array(low_bounds)

        if c1 or c2:

            fit, cov, fit_vals = fit_fx(
                initial_parameters=fit,
                low_bounds=low_bounds,
                high_bounds=high_bounds,
                **kwargs)

    return fit, cov, fit_vals, low_bounds, high_bounds

def re_fit_2(fit, cov, fit_vals_a,fit_vals_b,fit_fx,low_bounds,high_bounds,times,**kwargs):

    """
    Evaluate the difference between the fitted parameters and the initial parameters
    If the difference is less than 2 percent, the bounds are relaxed by a factor of 10
    and the fitting is repeated.
    The difference with re_fit() is that this function is used for fitting two signals at once,
    e.g. fitting association and dissociation signals simultaneously.

    Args:
        fit (list): Fitted parameters
        cov (np.ndarray): Covariance matrix of the fitted parameters
        fit_vals_a (list): Fitted values for each signal, same dimensions as signal_lst
        fit_vals_b (list): Fitted values for each signal, same dimensions as signal_lst
        fit_fx (function): Function to fit the data - returns fit, cov, fit_vals_a, fit_vals_b
        low_bounds (list): Lower bounds for the parameters
        high_bounds (list): Upper bounds for the parameters
        times (int): Number of times to re-fit the data
        **kwargs: Additional arguments to pass to the fitting function

    Returns:
        fit (list): Fitted parameters after re-fitting
        cov (np.ndarray): Covariance matrix of the fitted parameters after re-fitting
        fit_vals (list): Fitted values for each signal after re-fitting, same dimensions as signal_lst
        low_bounds (list): Lower bounds for the parameters after re-fitting
        high_bounds (list): Upper bounds for the parameters after re-fitting
    """

    fit         = np.array(fit).astype(float)
    low_bounds  = np.array(low_bounds).astype(float)
    high_bounds = np.array(high_bounds).astype(float)

    for _ in range(times):

        difference_to_upper = (high_bounds - fit) / high_bounds
        difference_to_lower = (fit - low_bounds) / fit

        c1 = any(difference_to_upper < 0.02)
        c2 = any(difference_to_lower < 0.02)

        if c1:

            # Relax bounds by a factor of 10 - only those that are too close to the upper bound
            high_bounds = [expand_high_bound(value, factor=10) if diff < 0.02 else value for value, diff in zip(high_bounds, difference_to_upper)]
            high_bounds = np.array(high_bounds)

        if c2:

            low_bounds = [expand_low_bound(value, factor=10) if diff < 0.02 else value for value, diff in zip(low_bounds, difference_to_lower)]
            low_bounds = np.array(low_bounds)

        if c1 or c2:

            fit, cov, fit_vals_a, fit_vals_b  = fit_fx(
                initial_parameters=fit,
                low_bounds=low_bounds,
                high_bounds=high_bounds,
                **kwargs)

    return fit, cov, fit_vals_a, fit_vals_b, low_bounds, high_bounds