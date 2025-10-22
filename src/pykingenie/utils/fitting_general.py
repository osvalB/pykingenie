import itertools
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..utils.math import (
    single_exponential,
    double_exponential
)

__all__ = ['fit_single_exponential', 'fit_double_exponential','re_fit','re_fit_2']

def convert_to_numpy_array(data):
    """
    Convert input data to a numpy array if it is not already one.

    Parameters
    ----------
    data : list or np.ndarray
        Input data to be converted.

    Returns
    -------
    np.ndarray
        Converted numpy array.
    """
    if not isinstance(data, np.ndarray):
        return np.array(data)

    return data

def fit_single_exponential(y, t, min_log_k=-5, max_log_k=5, log_k_points=50):
    """
    Fit a single exponential to a signal.

    Parameters
    ----------
    y : np.ndarray
        Signal values.
    t : np.ndarray
        Time values.
    min_log_k : float, optional
        Minimum logarithmic value for k_obs initial estimation, default is -5.
    max_log_k : float, optional
        Maximum logarithmic value for k_obs initial estimation, default is 5.
    log_k_points : int, optional
        Number of points in the logarithmic scale for k_obs initial estimation, default is 50.

    Returns
    -------
    np.ndarray
        Fitted parameters (a0, a1, k_obs).
    np.ndarray
        Covariance matrix of the fitted parameters.
    np.ndarray
        Fitted y values.
    """
    # Convert to numpy array, if needed,
    y = convert_to_numpy_array(y)
    t = convert_to_numpy_array(t)

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

        # Fit the reduced model
        params, cov = curve_fit(single_exponential_reduced, t, y, p0=[0, np.max(y)])

        # Calculate the residual sum of squares
        rss_temp = np.sum((y - single_exponential_reduced(t, *params)) ** 2)

        if rss_temp < rss:
            rss         = rss_temp
            best_params = params
            best_k_obs  = k_obs

    p0 = [best_params[0], best_params[1], best_k_obs]

    fit_params, cov = curve_fit(single_exponential, t, y, p0=p0)

    fit_y = single_exponential(t, *fit_params)

    return fit_params, cov, fit_y

def fit_double_exponential(y, t, min_log_k=-4, max_log_k=4, log_k_points=24):
    """
    Fit a double exponential to a signal.

    Parameters
    ----------
    y : np.ndarray
        Signal values.
    t : np.ndarray
        Time values.
    min_log_k : float, optional
        Minimum logarithmic value for k_obs initial estimation, default is -4.
    max_log_k : float, optional
        Maximum logarithmic value for k_obs initial estimation, default is 4.
    log_k_points : int, optional
        Number of points in the logarithmic scale for k_obs initial estimation, default is 22.

    Returns
    -------
    np.ndarray
        Fitted parameters (a0, a1, k_obs1, a2, k_obs2).
    np.ndarray
        Covariance matrix of the fitted parameters.
    np.ndarray
        Fitted y values.
    """
    # Convert to numpy array, if needed,
    y = convert_to_numpy_array(y)
    t = convert_to_numpy_array(t)

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

def fit_many_double_exponential(signal_lst, time_lst, min_log_k=-4, max_log_k=4, log_k_points=22):
    """
    Fit a double exponential to many signals.
    
    Parameters
    ----------
    signal_lst : list of np.ndarray
        List of signals to fit.
    time_lst : list of np.ndarray
        List of time arrays corresponding to each signal.
    min_log_k : float, optional
        Minimum logarithmic value for k_obs initial estimation, default is -4.
    max_log_k : float, optional
        Maximum logarithmic value for k_obs initial estimation, default is 4.
    log_k_points : int, optional
        Number of points in the logarithmic scale for k_obs initial estimation, default is 22.
        
    Returns
    -------
    list
        List of slowest k_obs values for each signal.
    list
        List of second slowest k_obs values for each signal.
    list
        List of fitted values for each signal.
    list
        List of the relative errors for the slowest k_obs values.
    list
        List of the relative errors for the second slowest k_obs values.
        
    Notes
    -----
    This function iterates over each signal and time pair, fits a double exponential model,
    and returns the slowest and second slowest k_obs values along with the fitted values.
    If fitting fails for a signal, it will skip that signal and return NaN for k_obs values 
    and None for fitted values. The function assumes that the input signals and time arrays 
    are of the same length.
    """

    k_obs_1 = [np.nan for _ in range(len(signal_lst))]
    k_obs_2 = [np.nan for _ in range(len(signal_lst))]
    y_pred =  [None   for _ in range(len(signal_lst))]

    k_obs_1_err = [np.nan for _ in range(len(signal_lst))]
    k_obs_2_err = [np.nan for _ in range(len(signal_lst))]

    i = 0
    for y, t in zip(signal_lst, time_lst):

        try:

            params, cov, fitted_y = fit_double_exponential(y, t, min_log_k=min_log_k, max_log_k=max_log_k,
                                                           log_k_points=log_k_points)

            errors = (np.sqrt(cov) / params) * 100

            k_obs_both = [params[2], params[4]]
            k_obs_both_err = [errors[2], errors[4]]

            # Find the index of the slowest k_obs
            min_idx = np.argmin(k_obs_both)
            max_idx = 1 - min_idx

            slowest_k = k_obs_both[min_idx]
            second_k = k_obs_both[max_idx]

            slowest_k_err = k_obs_both_err[min_idx]
            second_k_err = k_obs_both_err[max_idx]

            k_obs_1[i] = slowest_k
            k_obs_2[i] = second_k
            y_pred[i] = fitted_y

            k_obs_1_err[i] = slowest_k_err
            k_obs_2_err[i] = second_k_err

        except:

            pass

        i += 1

    return k_obs_1, k_obs_2, y_pred, k_obs_1_err, k_obs_2_err

def expand_high_bound(value, factor=10):
    """
    Multiply/divide the value by a factor and return the new value.
    
    If the parameter is negative, we divide by the factor, otherwise we multiply.
    
    Parameters
    ----------
    value : float
        Parameter to expand.
    factor : float, optional
        Factor to multiply/divide by, default is 10.
        
    Returns
    -------
    float
        New parameter value after expansion.
    """
    new_value = value * factor if value >= 0 else value / factor

    return new_value

def expand_low_bound(value, factor=10):
    """
    Multiply/divide the value by a factor and return the new value.
    
    If the parameter is negative, we multiply by the factor, otherwise we divide.
    
    Parameters
    ----------
    value : float
        Parameter to expand.
    factor : float, optional
        Factor to multiply/divide by, default is 10.
        
    Returns
    -------
    float
        New parameter value after expansion.
    """
    new_value = value * factor if value < 0 else value / factor

    return new_value

def re_fit(fit, cov, fit_vals, fit_fx, low_bounds, high_bounds, times, **kwargs):
    """
    Re-fit data by evaluating the difference between fitted parameters and bounds.
    
    If the difference is less than 2 percent, the bounds are relaxed by a factor of 10
    and the fitting is repeated.
    
    Parameters
    ----------
    fit : list
        Fitted parameters.
    cov : np.ndarray
        Covariance matrix of the fitted parameters.
    fit_vals : list
        Fitted values for each signal, same dimensions as signal_lst.
    fit_fx : callable
        Function to fit the data - returns fit, cov, fit_vals.
    low_bounds : list
        Lower bounds for the parameters.
    high_bounds : list
        Upper bounds for the parameters.
    times : int
        Number of times to re-fit the data.
    **kwargs
        Additional arguments to pass to the fitting function.
        
    Returns
    -------
    np.ndarray
        Fitted parameters after re-fitting.
    np.ndarray
        Covariance matrix of the fitted parameters after re-fitting.
    list
        Fitted values for each signal after re-fitting.
    np.ndarray
        Lower bounds for the parameters after re-fitting.
    np.ndarray
        Upper bounds for the parameters after re-fitting.
    """
    fit         = np.array(fit).astype(float)
    low_bounds  = np.array(low_bounds).astype(float)
    high_bounds = np.array(high_bounds).astype(float)

    for _ in range(times):

        # upper bounds may contain np.inf, which will cause a division error
        difference_to_upper = np.abs(np.array([(a-b)/a if a != np.inf else np.inf for a, b in zip(high_bounds, fit)]))
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

           # Some fitting functions return 3 elements: fit, cov, fit_vals, while others return those 3 and the parameter names
            result = fit_fx(
                initial_parameters=fit,
                low_bounds=low_bounds,
                high_bounds=high_bounds,
                **kwargs)

            if len(result) == 4:
                fit, cov, fit_vals, param_names = result
            else:
                fit, cov, fit_vals = result

    return fit, cov, fit_vals, low_bounds, high_bounds

def re_fit_2(fit, cov, fit_vals_a, fit_vals_b, fit_fx, low_bounds, high_bounds, times, **kwargs):
    """
    Re-fit data for two signals by evaluating the difference between fitted parameters and bounds.
    
    If the difference is less than 2 percent, the bounds are relaxed by a factor of 10
    and the fitting is repeated. This function is used for fitting two signals at once,
    e.g. fitting association and dissociation signals simultaneously.
    
    Parameters
    ----------
    fit : list
        Fitted parameters.
    cov : np.ndarray
        Covariance matrix of the fitted parameters.
    fit_vals_a : list
        Fitted values for the first set of signals.
    fit_vals_b : list
        Fitted values for the second set of signals.
    fit_fx : callable
        Function to fit the data - returns fit, cov, fit_vals_a, fit_vals_b.
    low_bounds : list
        Lower bounds for the parameters.
    high_bounds : list
        Upper bounds for the parameters.
    times : int
        Number of times to re-fit the data.
    **kwargs
        Additional arguments to pass to the fitting function.
        
    Returns
    -------
    np.ndarray
        Fitted parameters after re-fitting.
    np.ndarray
        Covariance matrix of the fitted parameters after re-fitting.
    list
        Fitted values for the first set of signals after re-fitting.
    list
        Fitted values for the second set of signals after re-fitting.
    np.ndarray
        Lower bounds for the parameters after re-fitting.
    np.ndarray
        Upper bounds for the parameters after re-fitting.
    """
    fit         = np.array(fit).astype(float)
    low_bounds  = np.array(low_bounds).astype(float)
    high_bounds = np.array(high_bounds).astype(float)

    for _ in range(times):

        difference_to_upper = np.abs(np.array([(a-b)/a if a != np.inf else np.inf for a, b in zip(high_bounds, fit)]))
        difference_to_lower = np.abs( (fit - low_bounds) / fit)

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