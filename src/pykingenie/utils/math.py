import numpy as np
import pandas as pd

from scipy           import stats

___all__ = ['single_exponential',
           'double_exponential',
           'median_filter',
           'rss_p',
           'get_rss',
           'get_desired_rss']

def single_exponential(t, a0, a1, kobs):
    """
    Single exponential function for fitting.

    Parameters
    ----------
    t : np.ndarray
        Time values.
    a0 : float
        Offset.
    a1 : float
        Amplitude.
    kobs : float
        Observed rate constant.

    Returns
    -------
    np.ndarray
        Computed values of the single exponential function.
    """
    return a0 + a1 * np.exp(-kobs * t)

def double_exponential(t, a0, a1, kobs1, a2, kobs2):
    """
    Double exponential function for fitting.

    Parameters
    ----------
    t : np.ndarray
        Time values.
    a0 : float
        Offset.
    a1 : float
        Amplitude of the first exponential.
    kobs1 : float
        Observed rate constant of the first exponential.
    a2 : float
        Amplitude of the second exponential.
    kobs2 : float
        Observed rate constant of the second exponential.

    Returns
    -------
    np.ndarray
        Computed values of the double exponential function.
    """
    return a0 + a1 * np.exp(-kobs1 * t) + a2 * np.exp(-kobs2 * t)

def median_filter(y, x, rolling_window):
    """
    Compute the median filter of the input signal using a rolling window.

    The x vector is converted into an integer vector and then into a time variable to take advantage of pandas rolling().median().
    Returns the y vector passed through the median filter.

    Parameters
    ----------
    y : np.ndarray
        Input signal values.
    x : np.ndarray
        Time or index values corresponding to the signal.
    rolling_window : int or float
        Size of the median filter window.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """

    # Check the average time step in the x vector
    dx = np.min(np.diff(x))

    scaling_factor = 1
    if dx <= 1:
        scaling_factor = 1/dx * 10 # Rescale 
    
    temp_vec     =  np.multiply(x,scaling_factor).astype(int)
    series       =  pd.Series(y,index=temp_vec,dtype=float)
    series.index =  pd.to_datetime(series.index,unit='s') # Convert to datetime index - allows rolling window

    roll_window  = str(int(rolling_window*scaling_factor))+"s"

    y_filt = series.rolling(
        window=roll_window).median().to_numpy()

    return y_filt

def rss_p(rrs0, n, p, alfa):
    """
    Compute the desired residual sum of squares for a 1-alpha confidence interval.

    Given the residuals of the best fitted model, this is used to compute asymmetric confidence intervals for the fitted parameters.

    Parameters
    ----------
    rrs0 : float
        Residual sum of squares of the model with the best fit.
    n : int
        Number of data points.
    p : int
        Number of parameters.
    alfa : float
        Desired confidence interval (1 - alpha).

    Returns
    -------
    float
        Residual sum of squares for the desired confidence interval.
    """

    critical_value = stats.f.ppf(q=1 - alfa, dfn=1, dfd=n - p)

    return rrs0 * (1 + critical_value / (n - p))

def get_rss(y, y_fit):
    """
    Compute the residual sum of squares.

    Parameters
    ----------
    y : np.ndarray
        Observed values.
    y_fit : np.ndarray
        Fitted values.

    Returns
    -------
    float
        Residual sum of squares.
    """

    residuals = y - y_fit
    rss       = np.sum(residuals ** 2)

    return rss

def get_desired_rss(y, y_fit, n, p, alpha=0.05):
    """
    Find the residual sum of squares required for a 1-alpha confidence interval.

    Given the observed and fitted data, computes the RSS for the desired confidence interval.

    Parameters
    ----------
    y : np.ndarray
        Observed values.
    y_fit : np.ndarray
        Fitted values.
    n : int
        Number of data points.
    p : int
        Number of parameters.
    alpha : float, optional
        Desired confidence interval (default is 0.05).

    Returns
    -------
    float
        Residual sum of squares for the desired confidence interval.
    """

    rss = get_rss(y, y_fit)

    return rss_p(rss, n, p, alpha)
