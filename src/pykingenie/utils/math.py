import numpy as np
import pandas as pd

from scipy           import stats

def single_exponential(t, a0, a1, kobs):

    """
    Single exponential function for fitting

    Args:
        t (np.ndarray): time
        a0 (float): offset
        a1 (float): amplitude
        kobs (float): observed rate constant

    Returns:
        np.ndarray: computed values of the single exponential function
    """
    return a0 + a1 * np.exp(-kobs * t)

def double_exponential(t, a0, a1, kobs1, a2, kobs2):
    """
    Double exponential function for fitting

    Args:
        t (np.ndarray): time
        a0 (float): offset
        a1 (float): amplitude of the first exponential
        kobs1 (float): observed rate constant of the first exponential
        a2 (float): amplitude of the second exponential
        kobs2 (float): observed rate constant of the second exponential

    Returns:
        np.ndarray: computed values of the double exponential function
    """
    return a0 + a1 * np.exp(-kobs1 * t) + a2 * np.exp(-kobs2 * t)

def median_filter(y,x,rolling_window):

    """

    Compute the median filter of the x vector using a rolling window

    First, we convert the x vector into an integer vector and then
        into time variable to take advantage of pandas function
            rolling().median()

	Returns the y vector passed through the median filter

    """

    scaling_factor = 1e4

    temp_vec     =  np.multiply(x,scaling_factor).astype(int)
    series       =  pd.Series(y,index=temp_vec,dtype=float)
    series.index =  pd.to_datetime(series.index,unit='s')

    roll_window  = str(int(rolling_window*scaling_factor))+"s"

    y_filt = series.rolling(roll_window).median().to_numpy()

    return y_filt

def rss_p(rrs0, n, p, alfa):

    """
    Given the residuals of the best fitted model,
    compute the desired residual sum of squares for a 1-alpha confidence interval
    This is used to compute asymmetric confidence intervals for the fitted parameters

    Args:
        rrs0 (float): residual sum of squares of the model with the best fit
        n (int): number of data points
        p (int): number of parameters
        alfa (float): desired confidence interval

    Returns:
        rss (float): residual sum of squares for the desired confidence interval
    """

    critical_value = stats.f.ppf(q=1 - alfa, dfn=1, dfd=n - p)

    return rrs0 * (1 + critical_value / (n - p))

def get_rss(y, y_fit):

    """
    Compute the residual sum of squares

    Args:

        y (np.ndarray): observed values
        y_fit (np.ndarray): fitted values

    Returns:

        rss (np.ndarray): residual sum of squares

    """

    residuals = y - y_fit
    rss       = np.sum(residuals ** 2)

    return rss

def get_desired_rss(y, y_fit, n, p,alpha=0.05):

    """
    Given the observed and fitted data,
    find the residual sum of squares required for a 1-alpha confidence interval

    Args:

        y (np.ndarray): observed values
        y_fit (np.ndarray): fitted values
        n (int): number of data points
        p (int): number of parameters
        alpha (float): desired confidence interval

    Returns:

        rss (np.ndarray): residual sum of squares

    """

    rss = get_rss(y, y_fit)

    return rss_p(rss, n, p, alpha)
