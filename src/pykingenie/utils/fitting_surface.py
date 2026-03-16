import numpy as np

from .processing import concat_signal_lst, detect_time_list_continuos

from .signal_surface import (
    steady_state_one_site,
    one_site_association_analytical,
    one_site_dissociation_analytical,
    solve_ode_one_site_mass_transport_association,
    solve_ode_one_site_mass_transport_dissociation,
    solve_induced_fit_association,
    solve_induced_fit_dissociation
)

from .math import get_rss

from .fitting_general import fit_single_exponential

from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar

__all__ = [
    'guess_initial_signal',
    'fit_steady_state_one_site',
    'steady_state_one_site_asymmetric_ci95',
    'fit_one_site_association',
    'fit_one_site_dissociation',
    'fit_one_site_assoc_and_disso',
    'fit_induced_fit_sites_assoc_and_disso',
    'fit_one_site_assoc_and_disso_ktr',
    'one_site_assoc_and_disso_asymmetric_ci95',
    'one_site_assoc_and_disso_asymmetric_ci95_koff',
    'get_smax_upper_bound_factor'
]

def guess_initial_signal(assoc_time_lst, assoc_signal_lst, time_limit=30):
    """
    Guess the initial signal for each signal in the list by fitting single exponentials.
    
    Used only for the one-to-one binding model in case one of the steps is not present.

    Parameters
    ----------
    assoc_time_lst : list
        List of association time arrays.
    assoc_signal_lst : list
        List of association signal arrays.
    time_limit : float, optional
        Time limit to consider for the fit, default is 30 seconds.

    Returns
    -------
    list
        List of initial signals for each association signal.
    """
    s0s = []

    for t,y in zip(assoc_time_lst,assoc_signal_lst):

        try:

            t = t - t[0]
            y = y[t < time_limit]
            t = t[t < time_limit]

            a0,a1,kobs = fit_single_exponential(y,t)

            fit_params, _, _ = fit_single_exponential(y,t)

            a0, a1, kobs = fit_params

            s0s.append(a0 + a1)

        except:

            s0s.append(y[0])

    return s0s

def get_smax_upper_bound_factor(Kd_ss):
    """
    Get a factor to determine the upper bound for Smax based on Kd_ss value.
    
    Parameters
    ----------
    Kd_ss : float
        Steady state Kd value.
        
    Returns
    -------
    float or None
        Factor to multiply for setting Smax upper bound, or None if no matching range found.
    """
    factor_dict = {
        (10, float('inf')): 1e3,
        (1, 10): 1e2,
        (float('-inf'), 1): 50
    }
    for (low, high), factor in factor_dict.items():
        if low <= Kd_ss < high:
            return factor

def fit_steady_state_one_site(signal_lst, ligand_lst, initial_parameters,
                              low_bounds, high_bounds, fixed_Kd=False, Kd_value=None):
    """
    Fit a one-site binding model to a set of steady state signals.
    
    Parameters
    ----------
    signal_lst : list
        List of signals to fit, each signal is a numpy array.
    ligand_lst : list
        List of ligand concentrations, each concentration is a numpy array.
    initial_parameters : list
        Initial guess for the parameters.
    low_bounds : list
        Lower bounds for the parameters.
    high_bounds : list
        Upper bounds for the parameters.
    fixed_Kd : bool, optional
        If True, Kd is fixed to Kd_value, default is False.
    Kd_value : float, optional
        Value of Kd to use if fixed_Kd is True.
        
    Returns
    -------
    list
        Fitted parameters.
    np.ndarray
        Covariance matrix of the fitted parameters.
    list
        Fitted values for each signal, same dimensions as signal_lst.
    """
    all_signal = concat_signal_lst(signal_lst)

    start = 1
    if fixed_Kd:

        # relax bounds
        low_bounds  = [x / 5 for x in low_bounds]
        high_bounds = [x * 5 for x in high_bounds]
        start       = 0

    # Pre-compute slice boundaries for output array
    _lengths = [len(C) for C in ligand_lst]
    _total   = sum(_lengths)
    _offsets = np.empty(len(_lengths) + 1, dtype=int)
    _offsets[0] = 0
    for _k, _n in enumerate(_lengths):
        _offsets[_k + 1] = _offsets[_k] + _n

    def fit_fx(dummyVariable, *args):

        Kd = Kd_value if fixed_Kd else args[0]

        Rmax_all = args[start:]

        out = np.empty(_total)
        for _k, (C, Rmax) in enumerate(zip(ligand_lst, Rmax_all)):
            out[_offsets[_k]:_offsets[_k + 1]] = steady_state_one_site(C, Rmax, Kd)
        return out

    global_fit_params, cov = curve_fit(fit_fx, 1, all_signal,
                                       p0=initial_parameters,
                                       bounds=(low_bounds, high_bounds))

    Kd = Kd_value if fixed_Kd else global_fit_params[0]

    Rmax_all      = global_fit_params[start:]
    fitted_values = [steady_state_one_site(C, Rmax, Kd) for C, Rmax in zip(ligand_lst, Rmax_all)]

    return global_fit_params, cov, fitted_values

def steady_state_one_site_asymmetric_ci95(kd_estimated, signal_lst, ligand_lst, initial_parameters,
                                          low_bounds, high_bounds, rss_desired):
    """
    Calculate the asymmetric confidence interval for the steady-state signal.
    
    Parameters
    ----------
    kd_estimated : float
        Estimated Kd value.
    signal_lst : list
        List of signals to fit, each signal is a numpy array.
    ligand_lst : list
        List of ligand concentrations, each concentration is a numpy array.
    initial_parameters : list
        Initial guess for the parameters (without the Kd!).
    low_bounds : list
        Lower bounds for the parameters (without the Kd!).
    high_bounds : list
        Upper bounds for the parameters (without the Kd!).
    rss_desired : float
        Maximum residual sum of squares.

    Returns
    -------
    np.ndarray
        95% asymmetric confidence interval for the Kd value [lower_bound, upper_bound].
    """
    # Pre-concatenate the static reference signal once
    _all_signal_ref = concat_signal_lst(signal_lst)

    def f_to_optimize(Kd):

        fit_params, _, fit_vals = fit_steady_state_one_site(signal_lst, ligand_lst,
                                                  initial_parameters,
                                                  low_bounds,
                                                  high_bounds,
                                                  fixed_Kd = True,
                                                  Kd_value = Kd)

        rss = get_rss(_all_signal_ref, concat_signal_lst(fit_vals))

        return np.abs(rss - rss_desired)

    boundsMin = np.array([kd_estimated/5e4,kd_estimated])
    boundsMax = np.array([kd_estimated,kd_estimated*5e4])

    kd_min95 = minimize_scalar(f_to_optimize, bounds=boundsMin,method='bounded')
    kd_max95 = minimize_scalar(f_to_optimize, bounds=boundsMax,method='bounded')

    kd_min95, kd_max95 = kd_min95.x, kd_max95.x

    ci95 = np.array([kd_min95, kd_max95])

    return ci95

def fit_one_site_association(signal_lst, time_lst, analyte_conc_lst,
                             initial_parameters, low_bounds, high_bounds,
                             smax_idx=None,
                             shared_smax=False,
                             fixed_t0=True,
                             fixed_Kd=False, Kd_value=None,
                             fixed_koff=False, koff_value=None):
    """
    Global fit to a list of association traces - one-to-one binding model.

    Parameters
    ----------
    signal_lst : list
        List of signals to fit, each signal is a numpy array.
    time_lst : list
        List of time arrays.
    analyte_conc_lst : list
        List of analyte concentrations, each element is a numpy array.
    initial_parameters : list
        Initial guess for the parameters.
    low_bounds : list
        Lower bounds for the parameters.
    high_bounds : list
        Upper bounds for the parameters.
    smax_idx : list, optional
        List of indices for the s_max parameters, used if shared_smax is True.
    shared_smax : bool, optional
        If True, the s_max parameters are shared between the signals, default is False.
    fixed_t0 : bool, optional
        If True, t0 is fixed to 0, otherwise we fit it, default is True.
    fixed_Kd : bool, optional
        If True, Kd is fixed to Kd_value, default is False.
    Kd_value : float, optional
        Value of Kd to use if fixed_Kd is True.
    fixed_koff : bool, optional
        If True, koff is fixed to koff_value, default is False.
    koff_value : float, optional
        Value of koff to use if fixed_koff is True.

    Returns
    -------
    list
        Fitted parameters.
    np.ndarray
        Covariance matrix of the fitted parameters.
    list
        Fitted values for each signal, same dimensions as signal_lst.
    """
    all_signal = concat_signal_lst(signal_lst)

    time_lst = [np.array(t) for t in time_lst]

    start = 3 - sum([fixed_t0, fixed_Kd, fixed_koff])

    # Pre-compute slice boundaries
    _n_traces = len(time_lst)
    _lengths  = [len(t) for t in time_lst]
    _total    = sum(_lengths)
    _offsets  = np.empty(_n_traces + 1, dtype=int)
    _offsets[0] = 0
    for _k, _n in enumerate(_lengths):
        _offsets[_k + 1] = _offsets[_k] + _n

    def fit_fx(dummyVariable, *args):
        Kd = Kd_value if fixed_Kd else args[0]
        Koff = koff_value if fixed_koff else args[1 - sum([fixed_Kd])]
        t0 = args[2 - sum([fixed_Kd, fixed_koff])] if not fixed_t0 else 0

        out = np.empty(_total)
        for i in range(_n_traces):
            t            = time_lst[i]
            analyte_conc = analyte_conc_lst[i]
            s_max        = args[start+smax_idx[i]] if shared_smax else args[start + i]
            out[_offsets[i]:_offsets[i + 1]] = one_site_association_analytical(t,0,s_max,Koff,Kd,analyte_conc,t0)
        return out

    global_fit_params, cov = curve_fit(fit_fx, 1, all_signal,
                                       p0=initial_parameters,
                                       bounds=(low_bounds, high_bounds))

    predicted_curve = fit_fx(1, *global_fit_params)

    fitted_values_assoc = []
    for _k in range(_n_traces):
        fitted_values_assoc.append(predicted_curve[_offsets[_k]:_offsets[_k + 1]])

    return global_fit_params, cov, fitted_values_assoc

def fit_one_site_dissociation(signal_lst, time_lst,
                             initial_parameters, low_bounds, high_bounds,
                             fixed_t0=True,
                             fixed_koff=False, koff_value=None,
                             fit_s0=True):
    """
    Global fit to a list of dissociation traces - one-to-one binding model.

    Parameters
    ----------
    signal_lst : list
        List of signals to fit, each signal is a numpy array.
    time_lst : list
        List of time arrays.
    initial_parameters : list
        Initial guess for the parameters.
    low_bounds : list
        Lower bounds for the parameters.
    high_bounds : list
        Upper bounds for the parameters.
    fixed_t0 : bool, optional
        If True, t0 is fixed to 0, otherwise we fit it, default is True.
    fixed_koff : bool, optional
        If True, koff is fixed to koff_value, default is False.
    koff_value : float, optional
        Value of koff to use if fixed_koff is True.
    fit_s0 : bool, optional
        If True, s0 is fitted, otherwise it is fixed to the first value of the signal, default is True.

    Returns
    -------
    list
        Fitted parameters.
    np.ndarray
        Covariance matrix of the fitted parameters.
    list
        Fitted values for each signal, same dimensions as signal_lst.
    """
    all_signal = concat_signal_lst(signal_lst)

    time_lst = [np.array(t) for t in time_lst]

    # Start at time zero
    time_lst = [t - t[0] for t in time_lst]

    if not fit_s0:
        s0_all = [s[0] for s in signal_lst]

    # Pre-compute slice boundaries
    _n_traces = len(time_lst)
    _lengths  = [len(t) for t in time_lst]
    _total    = sum(_lengths)
    _offsets  = np.empty(_n_traces + 1, dtype=int)
    _offsets[0] = 0
    for _k, _n in enumerate(_lengths):
        _offsets[_k + 1] = _offsets[_k] + _n

    def fit_fx(dummyVariable, *args):
        Koff = koff_value if fixed_koff else args[0]
        t0 = args[1 - sum([fixed_koff])] if not fixed_t0 else 0

        if fit_s0:
            s0_vals = args[( 2 - sum([fixed_koff,fixed_t0]) ):]
        else:
            s0_vals = s0_all

        out = np.empty(_total)
        for _k, (t, s0) in enumerate(zip(time_lst, s0_vals)):
            out[_offsets[_k]:_offsets[_k + 1]] = one_site_dissociation_analytical(t, s0, Koff, t0)
        return out

    global_fit_params, cov = curve_fit(fit_fx, 1, all_signal,
                                       p0=initial_parameters,
                                       bounds=(low_bounds, high_bounds))

    predicted_curve = fit_fx(1, *global_fit_params)

    fitted_values_disso = []
    for _k in range(_n_traces):
        fitted_values_disso.append(predicted_curve[_offsets[_k]:_offsets[_k + 1]])

    return global_fit_params, cov, fitted_values_disso

def fit_one_site_assoc_and_disso(assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
                                 disso_signal_lst, disso_time_lst,
                                 initial_parameters, low_bounds, high_bounds,
                                 smax_idx=None,
                                 shared_smax=False,
                                 fixed_t0=True,
                                 fixed_Kd=False, Kd_value=None,
                                 fixed_koff=False, koff_value=None):
    """
    Global fit to a set of association and dissociation traces - one-to-one binding model.
    
    Parameters
    ----------
    assoc_signal_lst : list
        List of association signals to fit, each signal is a numpy array.
    assoc_time_lst : list
        List of association time arrays.
    analyte_conc_lst : list
        List of analyte concentrations, each element is a numpy array.
    disso_signal_lst : list
        List of dissociation signals to fit, each signal is a numpy array.
    disso_time_lst : list
        List of dissociation time arrays.
    initial_parameters : list
        Initial guess for the parameters.
    low_bounds : list
        Lower bounds for the parameters.
    high_bounds : list
        Upper bounds for the parameters.
    smax_idx : list, optional
        List of indices for the s_max parameters, used if shared_smax is True.
    shared_smax : bool, optional
        If True, the s_max parameters are shared between traces, default is False.
    fixed_t0 : bool, optional
        If True, t0 is fixed to 0, otherwise we fit it, default is True.
    fixed_Kd : bool, optional
        If True, Kd is fixed to Kd_value, default is False.
    Kd_value : float, optional
        Value of Kd to use if fixed_Kd is True.
    fixed_koff : bool, optional
        If True, koff is fixed to koff_value, default is False.
    koff_value : float, optional
        Value of koff to use if fixed_koff is True.
        
    Returns
    -------
    list
        Fitted parameters.
    np.ndarray
        Covariance matrix of the fitted parameters.
    list
        Fitted values for each association signal, same dimensions as assoc_signal_lst.
    list
        Fitted values for each dissociation signal, same dimensions as disso_signal_lst.
    """
    # Set a flag for the association that was done after a dissociation step
    initial_signal_at_zero = [time[0] < 2 for time in assoc_time_lst]

    all_signal_assoc = concat_signal_lst(assoc_signal_lst)
    all_signal_disso = concat_signal_lst(disso_signal_lst)

    time_lst_assoc = [np.array(t) for t in assoc_time_lst]
    time_lst_disso = [np.array(t) for t in disso_time_lst]

    time_lst_disso = [t - t[0] for t in time_lst_disso]

    start = 2 - sum([fixed_Kd, fixed_koff])

    n_t0s = len(np.unique(smax_idx))*(not fixed_t0)

    continuos_time = detect_time_list_continuos(time_lst_assoc,disso_time_lst)
    s0s            = guess_initial_signal(assoc_time_lst, assoc_signal_lst)

    # Pre-compute slice boundaries for output array
    _n_traces = len(time_lst_assoc)
    _a_lengths = [len(t) for t in time_lst_assoc]
    _d_lengths = [len(t) for t in time_lst_disso]
    _total_a = sum(_a_lengths)
    _total_d = sum(_d_lengths)
    _total   = _total_a + _total_d

    _a_offsets = np.empty(_n_traces + 1, dtype=int)
    _a_offsets[0] = 0
    for _k in range(_n_traces):
        _a_offsets[_k + 1] = _a_offsets[_k] + _a_lengths[_k]

    _d_offsets = np.empty(_n_traces + 1, dtype=int)
    _d_offsets[0] = _total_a
    for _k in range(_n_traces):
        _d_offsets[_k + 1] = _d_offsets[_k] + _d_lengths[_k]

    def fit_fx(dummyVariable, *args):
        Kd   = Kd_value if fixed_Kd else args[0]
        Koff = koff_value if fixed_koff else args[1 - sum([fixed_Kd])]

        out = np.empty(_total)
        prev_disso_end = 0.0

        for i in range(_n_traces):
            t_assoc = time_lst_assoc[i]
            t_dissoc = time_lst_disso[i]
            analyte_conc = analyte_conc_lst[i]

            t0    = args[start + smax_idx[i]] if not fixed_t0 else 0
            s_max = args[start+smax_idx[i]+n_t0s] if shared_smax else args[start + i + n_t0s]

            if np.logical_or(i == 0, initial_signal_at_zero[i]) and continuos_time[i]:
                s0 = 0
            elif continuos_time[i]:
                s0 = prev_disso_end
            else:
                s0 = s0s[i]

            y_assoc = one_site_association_analytical(t_assoc-t_assoc[0],s0,s_max,Koff,Kd,analyte_conc,t0)
            out[_a_offsets[i]:_a_offsets[i + 1]] = y_assoc

            s0_d = y_assoc[-1]
            y_disso = one_site_dissociation_analytical(t_dissoc, s0_d, Koff)
            out[_d_offsets[i]:_d_offsets[i + 1]] = y_disso

            prev_disso_end = y_disso[-1]

        return out

    all_signal = np.concatenate([all_signal_assoc,all_signal_disso], axis=0)

    global_fit_params, cov = curve_fit(fit_fx, 1, all_signal,
                                       p0=initial_parameters,
                                       bounds=(low_bounds, high_bounds))

    predicted_curve = fit_fx(1, *global_fit_params)

    fitted_values_assoc = []
    for _k in range(_n_traces):
        fitted_values_assoc.append(predicted_curve[_a_offsets[_k]:_a_offsets[_k + 1]])

    fitted_values_disso = []
    for _k in range(_n_traces):
        fitted_values_disso.append(predicted_curve[_d_offsets[_k]:_d_offsets[_k + 1]])

    return global_fit_params, cov, fitted_values_assoc, fitted_values_disso

def fit_induced_fit_sites_assoc_and_disso(
    assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
    disso_signal_lst, disso_time_lst,
    initial_parameters, low_bounds, high_bounds,
    smax_idx=None,
    shared_smax=False,
    fixed_t0=True,
    fixed_kon1=False, kon1_value=None,
    fixed_koff1=False, koff1_value=None,
    fixed_kon2=False, kon2_value=None,
    fixed_koff2=False, koff2_value=None,
    max_nfev=None
):
    """
    Global fit to association and dissociation traces - one-to-one binding model with induced fit.
    
    Parameters
    ----------
    assoc_signal_lst : list
        List of association signals to fit, each signal is a numpy array.
    assoc_time_lst : list
        List of association time arrays.
    analyte_conc_lst : list
        List of analyte concentrations, each element is a numpy array.
    disso_signal_lst : list
        List of dissociation signals to fit, each signal is a numpy array.
    disso_time_lst : list
        List of dissociation time arrays.
    initial_parameters : list
        Initial guess for the parameters.
    low_bounds : list
        Lower bounds for the parameters.
    high_bounds : list
        Upper bounds for the parameters.
    smax_idx : list, optional
        List of indices for the s_max parameters, used if shared_smax is True.
    shared_smax : bool, optional
        If True, the s_max parameters are shared between traces, default is False.
    fixed_t0 : bool, optional
        If True, t0 is fixed to 0, otherwise we fit it, default is True.
    fixed_kon1 : bool, optional
        If True, kon1 is fixed to kon1_value, default is False.
    kon1_value : float, optional
        Value of kon1 to use if fixed_kon1 is True.
    fixed_koff1 : bool, optional
        If True, koff1 is fixed to koff1_value, default is False.
    koff1_value : float, optional
        Value of koff1 to use if fixed_koff1 is True.
    fixed_kon2 : bool, optional
        If True, kon2 is fixed to kon2_value, default is False.
    kon2_value : float, optional
        Value of kon2 to use if fixed_kon2 is True.
    fixed_koff2 : bool, optional
        If True, koff2 is fixed to koff2_value, default is False.
    koff2_value : float, optional
        Value of koff2 to use if fixed_koff2 is True.
    max_nfev : int, optional
        Maximum number of function evaluations.
    
    Returns
    -------
    list
        Fitted parameters.
    np.ndarray
        Covariance matrix of the fitted parameters.
    list
        Fitted values for each association signal, same dimensions as assoc_signal_lst.
    list
        Fitted values for each dissociation signal, same dimensions as disso_signal_lst.
    """
    # Set a flag for the association that was done after a dissociation step
    initial_signal_at_zero = [time[0] < 2 for time in assoc_time_lst]

    # Preprocess time
    time_lst_assoc = [np.asarray(t) for t in assoc_time_lst]
    time_lst_disso = [np.asarray(t) - t[0] for t in disso_time_lst]  # normalize

    # Indexing logic
    start = 4 - sum([fixed_kon1, fixed_koff1, fixed_kon2, fixed_koff2])
    n_unq_smax = len(np.unique(smax_idx))
    n_t0s = n_unq_smax * (not fixed_t0)

    # Pre-compute slice boundaries for output array
    _n_traces  = len(time_lst_assoc)
    _a_lengths = [len(t) for t in time_lst_assoc]
    _d_lengths = [len(t) for t in time_lst_disso]
    _total_a   = sum(_a_lengths)
    _total_d   = sum(_d_lengths)
    _total     = _total_a + _total_d

    _a_offsets = np.empty(_n_traces + 1, dtype=int)
    _a_offsets[0] = 0
    for _k in range(_n_traces):
        _a_offsets[_k + 1] = _a_offsets[_k] + _a_lengths[_k]

    _d_offsets = np.empty(_n_traces + 1, dtype=int)
    _d_offsets[0] = _total_a
    for _k in range(_n_traces):
        _d_offsets[_k + 1] = _d_offsets[_k] + _d_lengths[_k]

    # Flatten signals once
    all_signal = np.concatenate(
        [np.concatenate(assoc_signal_lst), np.concatenate(disso_signal_lst)]
    )

    def fit_fx(_, *args):
        # Efficient argument unpacking
        idx = 0
        kon1 = kon1_value if fixed_kon1 else args[idx]; idx += not fixed_kon1
        koff1 = koff1_value if fixed_koff1 else args[idx]; idx += not fixed_koff1
        kon2 = kon2_value if fixed_kon2 else args[idx]; idx += not fixed_kon2
        koff2 = koff2_value if fixed_koff2 else args[idx]; idx += not fixed_koff2

        out = np.empty(_total)

        i = 0
        for t_assoc, t_dissoc in zip(time_lst_assoc, time_lst_disso):

            if i == 0 or initial_signal_at_zero[i]:
                sP1L, sP2l = 0, 0
            else:
                s0, sP1L, sP2l = last

            conc = analyte_conc_lst[i]
            smax = args[start + smax_idx[i] + n_t0s] if shared_smax else args[start + i + n_t0s]

            mat_assoc = solve_induced_fit_association(t_assoc, conc, kon1, koff1, kon2, koff2, sP1L=sP1L, sP2L=sP2l, smax=smax)
            out[_a_offsets[i]:_a_offsets[i + 1]] = mat_assoc[:, 0]
            last = mat_assoc[-1]

            s0, sP1L, sP2l = last
            mat_disso = solve_induced_fit_dissociation(t_dissoc, koff1, kon2, koff2, s0=s0, sP2L=sP2l, smax=smax)
            out[_d_offsets[i]:_d_offsets[i + 1]] = mat_disso[:, 0]

            last = mat_disso[-1]

            i += 1

        return out

    # Run fitting
    global_fit_params, cov = curve_fit(
        fit_fx, xdata=1, ydata=all_signal,
        p0=initial_parameters,
        bounds=(low_bounds, high_bounds),
        max_nfev=max_nfev
    )

    # Predict
    predicted = fit_fx(1, *global_fit_params)

    # Split fitted values using pre-computed offsets
    fitted_values_assoc = []
    for _k in range(_n_traces):
        fitted_values_assoc.append(predicted[_a_offsets[_k]:_a_offsets[_k + 1]])

    fitted_values_disso = []
    for _k in range(_n_traces):
        fitted_values_disso.append(predicted[_d_offsets[_k]:_d_offsets[_k + 1]])

    return global_fit_params, cov, fitted_values_assoc, fitted_values_disso


def fit_one_site_assoc_and_disso_ktr(assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
                                 disso_signal_lst, disso_time_lst,
                                 initial_parameters, low_bounds, high_bounds,
                                 smax_idx=None,
                                 shared_smax=False,
                                 fixed_t0=True,
                                 fixed_Kd=False, Kd_value=None,
                                 fixed_koff=False, koff_value=None):
    """
    Global fit to a set of association and dissociation traces - one-to-one with mass transport limitation binding model.
    
    Parameters
    ----------
    assoc_signal_lst : list
        List of association signals to fit, each signal is a numpy array.
    assoc_time_lst : list
        List of association time arrays.
    analyte_conc_lst : list
        List of analyte concentrations, each element is a numpy array.
    disso_signal_lst : list
        List of dissociation signals to fit, each signal is a numpy array.
    disso_time_lst : list
        List of dissociation time arrays.
    initial_parameters : list
        Initial guess for the parameters.
    low_bounds : list
        Lower bounds for the parameters.
    high_bounds : list
        Upper bounds for the parameters.
    smax_idx : list, optional
        List of indices for the s_max parameters, used if shared_smax is True.
    shared_smax : bool, optional
        If True, the s_max parameters are shared between traces, default is False.
    fixed_t0 : bool, optional
        If True, t0 is fixed to 0, otherwise we fit it, default is True.
    fixed_Kd : bool, optional
        If True, Kd is fixed to Kd_value, default is False.
    Kd_value : float, optional
        Value of Kd to use if fixed_Kd is True.
    fixed_koff : bool, optional
        If True, koff is fixed to koff_value, default is False.
    koff_value : float, optional
        Value of koff to use if fixed_koff is True.

    Returns
    -------
    list
        Fitted parameters.
    np.ndarray
        Covariance matrix of the fitted parameters.
    list
        Fitted values for each association signal, same dimensions as assoc_signal_lst.
    list
        Fitted values for each dissociation signal, same dimensions as disso_signal_lst.
    """
    all_signal_assoc = concat_signal_lst(assoc_signal_lst)
    all_signal_disso = concat_signal_lst(disso_signal_lst)

    time_lst_assoc = [np.array(t) for t in assoc_time_lst]
    time_lst_disso = [np.array(t) for t in disso_time_lst]

    initial_signal_at_zero = [time[0] < 2 for time in assoc_time_lst]

    time_lst_disso = [t - t[0] for t in time_lst_disso]

    start = 2 - sum([fixed_Kd, fixed_koff])

    n_t0s = len(np.unique(smax_idx))*(not fixed_t0)
    n_ktr = len(np.unique(smax_idx))

    # Pre-compute slice boundaries
    _n_traces  = len(time_lst_assoc)
    _a_lengths = [len(t) for t in time_lst_assoc]
    _d_lengths = [len(t) for t in time_lst_disso]
    _total_a   = sum(_a_lengths)
    _total_d   = sum(_d_lengths)
    _total     = _total_a + _total_d

    _a_offsets = np.empty(_n_traces + 1, dtype=int)
    _a_offsets[0] = 0
    for _k in range(_n_traces):
        _a_offsets[_k + 1] = _a_offsets[_k] + _a_lengths[_k]

    _d_offsets = np.empty(_n_traces + 1, dtype=int)
    _d_offsets[0] = _total_a
    for _k in range(_n_traces):
        _d_offsets[_k + 1] = _d_offsets[_k] + _d_lengths[_k]

    def fit_fx(dummyVariable, *args):
        Kd   = Kd_value if fixed_Kd else args[0]
        Koff = koff_value if fixed_koff else args[1 - sum([fixed_Kd])]

        out = np.empty(_total)
        prev_disso_end = 0.0

        for i in range(_n_traces):
            t_assoc  = time_lst_assoc[i]
            t_dissoc = time_lst_disso[i]
            analyte_conc = analyte_conc_lst[i]

            t0     = args[start + smax_idx[i]] if not fixed_t0 else 0
            k_tr   = args[start + smax_idx[i]] if fixed_t0 else args[start + n_t0s + smax_idx[i]]
            s_max  = args[start+smax_idx[i]+n_t0s+n_ktr] if shared_smax else args[start + i + n_t0s + n_ktr]

            if np.logical_or(i == 0, initial_signal_at_zero[i]):
                s0 = 0
            else:
                s0 = prev_disso_end

            y_assoc = solve_ode_one_site_mass_transport_association(t_assoc-t_assoc[0],s0,analyte_conc/2,analyte_conc,Kd,Koff,k_tr,s_max,t0)
            out[_a_offsets[i]:_a_offsets[i + 1]] = y_assoc

            s0_d = y_assoc[-1]
            y_disso = solve_ode_one_site_mass_transport_dissociation(t_dissoc,s0_d,Kd,Koff,k_tr,s_max)
            out[_d_offsets[i]:_d_offsets[i + 1]] = y_disso

            prev_disso_end = y_disso[-1]

        return out

    all_signal = np.concatenate([all_signal_assoc,all_signal_disso], axis=0)

    global_fit_params, cov = curve_fit(fit_fx, 1, all_signal,
                                       p0=initial_parameters,
                                       bounds=(low_bounds, high_bounds))

    predicted_curve = fit_fx(1, *global_fit_params)

    fitted_values_assoc = []
    for _k in range(_n_traces):
        fitted_values_assoc.append(predicted_curve[_a_offsets[_k]:_a_offsets[_k + 1]])

    fitted_values_disso = []
    for _k in range(_n_traces):
        fitted_values_disso.append(predicted_curve[_d_offsets[_k]:_d_offsets[_k + 1]])

    return global_fit_params, cov, fitted_values_assoc, fitted_values_disso


def one_site_assoc_and_disso_asymmetric_ci95(kd_estimated, rss_desired,
                                             assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
                                             disso_signal_lst, disso_time_lst,
                                             initial_parameters, low_bounds, high_bounds,
                                             smax_idx=None,
                                             shared_smax=False,
                                             fixed_t0=True,
                                             fixed_koff=False, koff_value=None):
    """
    Calculate the asymmetric confidence interval for the Kd value, given a desired RSS value.
    
    Parameters
    ----------
    kd_estimated : float
        Estimated Kd value.
    rss_desired : float
        Desired RSS value.
    assoc_signal_lst : list
        List of association signals to fit, each signal is a numpy array.
    assoc_time_lst : list
        List of association time arrays.
    analyte_conc_lst : list
        List of analyte concentrations, each element is a numpy array.
    disso_signal_lst : list
        List of dissociation signals to fit, each signal is a numpy array.
    disso_time_lst : list
        List of dissociation time arrays.
    initial_parameters : list
        Initial guess for the parameters, without the Kd value!
    low_bounds : list
        Lower bounds for the parameters, without the Kd value!
    high_bounds : list
        Upper bounds for the parameters, without the Kd value!
    smax_idx : list, optional
        List of indices for the s_max parameters, used if shared_smax is True.
    shared_smax : bool, optional
        If True, the s_max parameters are shared between traces, default is False.
    fixed_t0 : bool, optional
        If True, t0 is fixed to 0, otherwise we fit it, default is True.
    fixed_koff : bool, optional
        If True, koff is fixed to koff_value, default is False.
    koff_value : float, optional
        Value of koff to use if fixed_koff is True.
        
    Returns
    -------
    float
        Minimum Kd value for the 95% confidence interval.
    float
        Maximum Kd value for the 95% confidence interval.
    """
    # Pre-concatenate the static reference signals once
    _ref_assoc = concat_signal_lst(assoc_signal_lst)
    _ref_disso = concat_signal_lst(disso_signal_lst)

    boundsMax = np.array([kd_estimated*1e2, kd_estimated * 1e4]) * 1e3

    # Guess starting point for the upper bound
    test_factors = [1,1.1,1.5, 2, 5, 10, 25,50,100]
    for i,test_factor in enumerate(test_factors):

        if test_factor == 1:
            continue

        _, _, fitted_values_assoc, fitted_values_disso = fit_one_site_assoc_and_disso(
            assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
            disso_signal_lst, disso_time_lst,
            initial_parameters, low_bounds, high_bounds,
            smax_idx=smax_idx,
            shared_smax=shared_smax,
            fixed_t0=fixed_t0,
            fixed_Kd=True, Kd_value=kd_estimated*test_factor,
            fixed_koff=fixed_koff, koff_value=koff_value)

        rss1 = get_rss(_ref_assoc, concat_signal_lst(fitted_values_assoc))
        rss2 = get_rss(_ref_disso, concat_signal_lst(fitted_values_disso))

        if rss1 + rss2 > rss_desired:
            boundsMax = np.array([kd_estimated*test_factors[i-1],
                                  kd_estimated *test_factor]) * 1e3
            break

    boundsMin = np.array([kd_estimated/1e4, kd_estimated / 1e2]) * 1e3

    # Guess starting point for the lower bound
    for i,test_factor in enumerate(test_factors):

        if test_factor == 1:
            continue

        _, _, fitted_values_assoc, fitted_values_disso = fit_one_site_assoc_and_disso(
            assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
            disso_signal_lst, disso_time_lst,
            initial_parameters, low_bounds, high_bounds,
            smax_idx=smax_idx,
            shared_smax=shared_smax,
            fixed_t0=fixed_t0,
            fixed_Kd=True, Kd_value=kd_estimated/test_factor,
            fixed_koff=fixed_koff, koff_value=koff_value)

        rss1 = get_rss(_ref_assoc, concat_signal_lst(fitted_values_assoc))
        rss2 = get_rss(_ref_disso, concat_signal_lst(fitted_values_disso))

        if rss1 + rss2 > rss_desired:
            boundsMin = np.array([kd_estimated/test_factor,
                                  kd_estimated *test_factors[i-1]]) * 1e3
            break

    def f_to_optimize(Kd):

        Kd = Kd / 1e3

        _, _, fitted_values_assoc, fitted_values_disso = fit_one_site_assoc_and_disso(
                                                            assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
                                                            disso_signal_lst, disso_time_lst,
                                                            initial_parameters,low_bounds, high_bounds,
                                                            smax_idx=smax_idx,
                                                            shared_smax=shared_smax,
                                                            fixed_t0=fixed_t0,
                                                            fixed_Kd=True,Kd_value=Kd,
                                                            fixed_koff=fixed_koff,koff_value=koff_value)

        rss1 = get_rss(_ref_assoc, concat_signal_lst(fitted_values_assoc))
        rss2 = get_rss(_ref_disso, concat_signal_lst(fitted_values_disso))

        return np.abs(rss1 + rss2 - rss_desired)

    kd_min95 = minimize_scalar(f_to_optimize, bounds=boundsMin,method='bounded')
    kd_max95 = minimize_scalar(f_to_optimize, bounds=boundsMax,method='bounded')

    kd_min95, kd_max95 = kd_min95.x, kd_max95.x

    # Rescale back the Kd to micromolar
    kd_min95, kd_max95 = kd_min95 / 1e3, kd_max95 / 1e3

    return kd_min95, kd_max95

def one_site_assoc_and_disso_asymmetric_ci95_koff(koff_estimated, rss_desired,
                                             assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
                                             disso_signal_lst, disso_time_lst,
                                             initial_parameters, low_bounds, high_bounds,
                                             smax_idx=None,
                                             shared_smax=False,
                                             fixed_t0=True):
    """
    Calculate the asymmetric confidence interval for the koff value, given a desired RSS value.
    
    Parameters
    ----------
    koff_estimated : float
        Estimated koff value.
    rss_desired : float
        Desired RSS value.
    assoc_signal_lst : list
        List of association signals to fit, each signal is a numpy array.
    assoc_time_lst : list
        List of association time arrays.
    analyte_conc_lst : list
        List of analyte concentrations, each element is a numpy array.
    disso_signal_lst : list
        List of dissociation signals to fit, each signal is a numpy array.
    disso_time_lst : list
        List of dissociation time arrays.
    initial_parameters : list
        Initial guess for the parameters, without the Kd value!
    low_bounds : list
        Lower bounds for the parameters, without the Kd value!
    high_bounds : list
        Upper bounds for the parameters, without the Kd value!
    smax_idx : list, optional
        List of indices for the s_max parameters, used if shared_smax is TRUE
    shared_smax : bool, optional
        If True, the s_max parameters are shared between traces
    fixed_t0 : bool, optional
        If True, t0 is fixed to 0, otherwise we fit it, default is True
        
    Returns
    -------
    float
        Minimum koff value for the 95% confidence interval.
    float
        Maximum koff value for the 95% confidence interval.
    """
    # Pre-concatenate the static reference signals once
    _ref_assoc = concat_signal_lst(assoc_signal_lst)
    _ref_disso = concat_signal_lst(disso_signal_lst)

    boundsMax = np.array([koff_estimated*1e2, koff_estimated * 1e4]) * 1e3

    # Guess starting point for the upper bound
    test_factors = [1,1.02,1.1,1.5, 2, 5, 10, 25,50,100]
    for i,test_factor in enumerate(test_factors):

        if test_factor == 1:
            continue

        _, _, fitted_values_assoc, fitted_values_disso = fit_one_site_assoc_and_disso(
            assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
            disso_signal_lst, disso_time_lst,
            initial_parameters, low_bounds, high_bounds,
            smax_idx=smax_idx,
            shared_smax=shared_smax,
            fixed_t0=fixed_t0,
            fixed_koff=True, koff_value=koff_estimated*test_factor)

        rss1 = get_rss(_ref_assoc, concat_signal_lst(fitted_values_assoc))
        rss2 = get_rss(_ref_disso, concat_signal_lst(fitted_values_disso))

        if rss1 + rss2 > rss_desired:
            boundsMax = np.array([koff_estimated*test_factors[i-1],
                                  koff_estimated *test_factor]) * 1e3
            break

    boundsMin = np.array([koff_estimated/1e4, koff_estimated / 1e2]) * 1e3

    # Guess starting point for the lower bound
    for i,test_factor in enumerate(test_factors):

        if test_factor == 1:
            continue

        _, _, fitted_values_assoc, fitted_values_disso = fit_one_site_assoc_and_disso(
            assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
            disso_signal_lst, disso_time_lst,
            initial_parameters, low_bounds, high_bounds,
            smax_idx=smax_idx,
            shared_smax=shared_smax,
            fixed_t0=fixed_t0,
            fixed_koff=True, koff_value=koff_estimated*test_factor)

        rss1 = get_rss(_ref_assoc, concat_signal_lst(fitted_values_assoc))
        rss2 = get_rss(_ref_disso, concat_signal_lst(fitted_values_disso))

        if rss1 + rss2 > rss_desired:
            boundsMin = np.array([koff_estimated/test_factor,
                                  koff_estimated *test_factors[i-1]]) * 1e3
            break

    def f_to_optimize(Koff):

        Koff = Koff / 1e3

        _, _, fitted_values_assoc, fitted_values_disso = fit_one_site_assoc_and_disso(
                                                            assoc_signal_lst, assoc_time_lst, analyte_conc_lst,
                                                            disso_signal_lst, disso_time_lst,
                                                            initial_parameters,low_bounds, high_bounds,
                                                            smax_idx=smax_idx,
                                                            shared_smax=shared_smax,
                                                            fixed_t0=fixed_t0,
                                                            fixed_koff=True,
                                                            koff_value=Koff)

        rss1 = get_rss(_ref_assoc, concat_signal_lst(fitted_values_assoc))
        rss2 = get_rss(_ref_disso, concat_signal_lst(fitted_values_disso))

        return np.abs(rss1 + rss2 - rss_desired)

    k_min95 = minimize_scalar(f_to_optimize, bounds=boundsMin,method='bounded')
    k_max95 = minimize_scalar(f_to_optimize, bounds=boundsMax,method='bounded')

    k_min95, k_max95 = k_min95.x, k_max95.x

    # Rescale back the k_off
    k_min95, k_max95 = k_min95 / 1e3, k_max95 / 1e3

    return k_min95, k_max95

