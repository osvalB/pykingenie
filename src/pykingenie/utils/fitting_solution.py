import numpy as np
import pandas as pd
import itertools

from scipy.optimize import curve_fit

from ..utils.signal_solution import (
    signal_ode_one_site_insolution,
    signal_ode_induced_fit_insolution,
    signal_ode_conformational_selection_insolution,
    get_initial_concentration_conformational_selection
)

from ..utils.fitting_general import re_fit

__all__ = [
    'fit_one_site_solution',
    'fit_induced_fit_solution',
    'fit_conformational_selection_solution',
    'find_initial_parameters_induced_fit_solution',
    'find_initial_parameters_conformational_selection_solution'
]

def fit_one_site_solution(
        signal_lst,
        time_lst,
        ligand_conc_lst,
        protein_conc_lst,
        initial_parameters,
        low_bounds,
        high_bounds,
        fit_signal_E=False,
        fit_signal_S=False,
        fit_signal_ES=True,
        fixed_t0=True,
        fixed_Kd=False,
        Kd_value=None,
        fixed_koff=False,
        koff_value=None):
    """
    Global fit to association and dissociation traces - one-to-one binding model.

    Parameters
    ----------
    signal_lst : list
        List of signals. We assume initial values as follows: E = E_tot, S = S_tot, ES = 0.
    time_lst : list
        List of time points for the association signals.
    ligand_conc_lst : list
        List of ligand concentrations, one per element in signal_lst.
    protein_conc_lst : list
        List of protein concentrations, one per element in signal_lst.
    initial_parameters : list
        Initial parameters for the fit.
    low_bounds : list
        Lower bounds for the fit parameters.
    high_bounds : list
        Upper bounds for the fit parameters.
    fit_signal_E : bool, optional
        If True, fit the signal of the free protein, default is False.
    fit_signal_S : bool, optional
        If True, fit the signal of the free ligand, default is False.
    fit_signal_ES : bool, optional
        If True, fit the signal of the complex, default is True.
    fixed_t0 : bool, optional
        If True, the initial time point is zero, default is True.
    fixed_Kd : bool, optional
        If True, the equilibrium dissociation constant is fixed, default is False.
    Kd_value : float, optional
        Value of the equilibrium dissociation constant if fixed_Kd is True.
    fixed_koff : bool, optional
        If True, the dissociation rate constant is fixed, default is False.
    koff_value : float, optional
        Value of the dissociation rate constant if fixed_koff is True.
        
    Returns
    -------
    list
        Fitted parameters.
    np.ndarray
        Covariance matrix of the fitted parameters.
    list
        Fitted values for each signal, same dimensions as signal_lst.
    list
        Names of the fitted parameters.
    """
    # Flatten signals once
    all_signal = np.concatenate(signal_lst)
    # Preprocess time
    time_lst = [np.asarray(t) for t in time_lst]

    # Create an empty list that will contain the parameter names
    parameter_names = []
    if fit_signal_E: parameter_names.append("Signal of free protein")
    if fit_signal_S: parameter_names.append("Signal of free ligand")
    if fit_signal_ES: parameter_names.append("Signal of the complex")
    if not fixed_Kd: parameter_names.append("Kd [µM]")
    if not fixed_koff: parameter_names.append("k_off [1/s]")
    if not fixed_t0:
        for i in range(len(time_lst)):
            parameter_names.append(f't0_{i+1}')

    def fit_fx(_, *args):
        # Efficient argument unpacking
        idx = 0

        signal_E  = 0 if not fit_signal_E  else args[idx]
        idx += fit_signal_E

        signal_S  = 0 if not fit_signal_S  else args[idx]
        idx += fit_signal_S

        signal_ES = 0 if not fit_signal_ES else args[idx]
        idx += fit_signal_ES

        Kd     = Kd_value     if fixed_Kd    else args[idx]
        idx += not fixed_Kd

        k_off  = koff_value   if fixed_koff  else args[idx]
        idx += not fixed_koff

        # Preallocate lists
        signal_a = [None] * len(time_lst)

        # Association phase
        for i, t in enumerate(time_lst):

            t0 = 0 if fixed_t0 else args[idx + i]

            lig_conc  = ligand_conc_lst[i]
            prot_conc = protein_conc_lst[i]
            signal = signal_ode_one_site_insolution(t,k_off,Kd,prot_conc,lig_conc,t0=t0,signal_a=signal_E,signal_b=signal_S,signal_complex=signal_ES)
            signal_a[i] = signal

        return np.concatenate(signal_a)

    # Run fitting
    global_fit_params, cov = curve_fit(
        fit_fx, xdata=1, ydata=all_signal,
        p0=initial_parameters,
        bounds=(low_bounds, high_bounds),
        max_nfev=None
    )
    # Predict
    predicted = fit_fx(1, *global_fit_params)
    # Split fitted values
    fitted_values = []
    idx = 0
    for t in time_lst:
        n = len(t)
        fitted_values.append(predicted[idx:idx + n])
        idx += n

    return global_fit_params, cov, fitted_values, parameter_names

def fit_induced_fit_solution(
        signal_lst,
        time_lst,
        ligand_conc_lst,
        protein_conc_lst,
        initial_parameters,
        low_bounds,
        high_bounds,
        fit_signal_E=False,
        fit_signal_S=False,
        fit_signal_ES=True,
        ESint_equals_ES=True,
        fixed_t0=True,
        fixed_kon=False,
        kon_value=None,
        fixed_koff=False,
        koff_value=None,
        fixed_kc=False,
        kc_value=None,
        fixed_krev=False,
        krev_value=None,
        max_nfev=None
):
    """
    Global fit to association and dissociation traces - one-to-one binding model with induced fit.

    Parameters
    ----------
    signal_lst : list
        List of signals. We assume initial values as follows: E = E_tot, S = S_tot, ES = 0, ESint = 0.
    time_lst : list
        List of time points for the association signals.
    ligand_conc_lst : list
        List of ligand concentrations, one per element in signal_lst.
    protein_conc_lst : list
        List of protein concentrations, one per element in signal_lst.
    initial_parameters : list
        Initial parameters for the fit.
    low_bounds : list
        Lower bounds for the fit parameters.
    high_bounds : list
        Upper bounds for the fit parameters.
    fit_signal_E : bool, optional
        If True, fit the signal of the free protein, default is False.
    fit_signal_S : bool, optional
        If True, fit the signal of the free ligand, default is False.
    fit_signal_ES : bool, optional
        If True, fit the signal of the complex, default is True.
    ESint_equals_ES : bool, optional
        If True, the signal of the intermediate complex is equal to the signal of the trapped complex, default is True.
    fixed_t0 : bool, optional
        If True, the initial time point is zero, default is True.
    fixed_kon : bool, optional
        If True, the association rate constant is fixed, default is False.
    kon_value : float, optional
        Value of the association rate constant if fixed_kon is True.
    fixed_koff : bool, optional
        If True, the dissociation rate constant is fixed, default is False.
    koff_value : float, optional
        Value of the dissociation rate constant if fixed_koff is True.
    fixed_kc : bool, optional
        If True, the induced fit rate constant is fixed, default is False.
    kc_value : float, optional
        Value of the induced fit rate constant if fixed_kc is True.
    fixed_krev : bool, optional
        If True, the reverse induced fit rate constant is fixed, default is False.
    krev_value : float, optional
        Value of the reverse induced fit rate constant if fixed_krev is True.
    max_nfev : int, optional
        Maximum number of function evaluations for the fit.
        
    Returns
    -------
    list
        Fitted parameters.
    np.ndarray
        Covariance matrix of the fitted parameters.
    list
        Fitted values for each signal, same dimensions as signal_lst.
    list
        Names of the fitted parameters.
        
    Notes
    -----
    The initial_parameters are given in the following order:
        
    - signal_E     if fit_signal_E                            (signal of the free protein)
    - signal_S     if fit_signal_S                            (signal of the free ligand)
    - signal_ES    if fit_signal_ES                           (signal of the trapped complex)
    - signal_ESint if fit_signal_ES and not ESint_equals_ES   (signal of the intermediate complex)
      if ESint_equals_ES is True, then the trapped complex signal is used for both ES and ESint
    - k_on         if not fixed_kon                           (association rate constant)
    - k_off        if not fixed_koff                          (dissociation rate constant)
    - k_c          if not fixed_kc                            (induced fit rate constant)
    - k_rev        if not fixed_krev                          (reverse induced fit rate constant)
    - t0_1         if not fixed_t0                            (initial time point for the first signal array, default is 0)
    - t0_2         if not fixed_t0                            (initial time point for the second signal array, default is 0)
    """
    # Flatten signals once
    all_signal = np.concatenate(signal_lst)

    # Preprocess time
    time_lst = [np.asarray(t) for t in time_lst]

    # Create an empty list that will contain the parameter names
    parameter_names = []
    if fit_signal_E: parameter_names.append("Signal of the free protein")
    if fit_signal_S: parameter_names.append("Signal of the free ligand")
    if fit_signal_ES and ESint_equals_ES: parameter_names.append("Signal of the complex")
    
    if fit_signal_ES and not ESint_equals_ES:
        parameter_names.append("Signal of the trapped complex")
        parameter_names.append("Signal of the intermediate complex")

    if not fixed_kon: parameter_names.append("k_on [1/(µM·s)]")
    if not fixed_koff: parameter_names.append("k_off [1/s]")
    if not fixed_kc: parameter_names.append("k_c [1/s]")
    if not fixed_krev: parameter_names.append("k_rev [1/s]")

    if not fixed_t0:
        for i in range(len(time_lst)):
            parameter_names.append(f't0_{i+1}')

    def fit_fx(_, *args):
        # Efficient argument unpacking
        idx = 0

        signal_E  = 0 if not fit_signal_E  else args[idx]
        idx += fit_signal_E

        signal_S  = 0 if not fit_signal_S  else args[idx]
        idx += fit_signal_S

        if fit_signal_ES:
            signal_ES = args[idx]
            idx += 1

            signal_ESint = signal_ES if ESint_equals_ES else args[idx]
            idx += not ESint_equals_ES

        k_on    = kon_value     if fixed_kon    else args[idx]
        idx += not fixed_kon

        k_off   = koff_value    if fixed_koff   else args[idx]
        idx += not fixed_koff

        k_c     = kc_value      if fixed_kc     else args[idx]
        idx += not fixed_kc

        k_rev   = krev_value    if fixed_krev   else args[idx]
        idx += not fixed_krev

        # Preallocate lists
        signal_a = [None] * len(time_lst)

        # Association phase
        for i, t in enumerate(time_lst):

            t0 = 0 if fixed_t0 else args[idx + i]  # Initial time point for the current signal

            lig_conc  = ligand_conc_lst[i]
            prot_conc = protein_conc_lst[i]

            signal = signal_ode_induced_fit_insolution(

                t,
                y = [0,0], # Initial concentrations of E·S (aka ES_int) and ES
                k1 = k_on,
                k_minus1 = k_off,
                k2 = k_c,
                k_minus2 = k_rev,
                E_tot = prot_conc,
                S_tot = lig_conc,
                t0=t0,
                signal_E= signal_E,
                signal_S= signal_S,
                signal_ES_int= signal_ESint,
                signal_ES= signal_ES)

            signal_a[i] = signal

        return np.concatenate(signal_a)

    # Run fitting
    global_fit_params, cov = curve_fit(
        fit_fx, xdata=1, ydata=all_signal,
        p0=initial_parameters,
        bounds=(low_bounds, high_bounds),
        max_nfev=max_nfev
    )

    # Predict
    predicted = fit_fx(1, *global_fit_params)

    # Split fitted values
    fitted_values = []
    idx = 0
    for t in time_lst:
        n = len(t)
        fitted_values.append(predicted[idx:idx + n])
        idx += n

    return global_fit_params, cov, fitted_values, parameter_names


def fit_conformational_selection_solution(
        signal_lst,
        time_lst,
        ligand_conc_lst,
        protein_conc_lst,
        initial_parameters,
        low_bounds,
        high_bounds,
        fit_signal_E=False,
        E1_equals_E2=True,
        fit_signal_S=False,
        fit_signal_E2S=True,
        fixed_t0=True,
        fixed_kon=False,
        kon_value=None,
        fixed_koff=False,
        koff_value=None,
        fixed_kc=False,
        kc_value=None,
        fixed_krev=False,
        krev_value=None,
        max_nfev=None
):
    """
    Global fit to association and dissociation traces - one-to-one binding model with conformational selection.

    Parameters
    ----------
    signal_lst : list
        List of signals. We assume initial values as follows: E = E_tot, S = S_tot, ES = 0, ESint = 0.
    time_lst : list
        List of time points for the association signals.
    ligand_conc_lst : list
        List of ligand concentrations, one per element in signal_lst.
    protein_conc_lst : list
        List of protein concentrations, one per element in signal_lst.
    initial_parameters : list
        Initial parameters for the fit.
    low_bounds : list
        Lower bounds for the fit parameters.
    high_bounds : list
        Upper bounds for the fit parameters.
    fit_signal_E : bool, optional
        If True, fit the signal of the free protein, default is False.
    E1_equals_E2 : bool, optional
        If true the signal in protein state 1 is equal to the signal in protein state 2, default is True.
    fit_signal_S : bool, optional
        If True, fit the signal of the free ligand, default is False
    fit_signal_E2S : bool, optional
        If true fit the signal of the complex, default is True.
    fixed_t0 : bool, optional
        If True, the initial time point is zero, default is True.
    fixed_kon : bool, optional
        If True, the association rate constant is fixed, default is False.
    kon_value : float, optional
        Value of the association rate constant if fixed_kon is True.
    fixed_koff : bool, optional
        If True, the dissociation rate constant is fixed, default is False.
    koff_value : float, optional
        Value of the dissociation rate constant if fixed_koff is True.
    fixed_kc : bool, optional
        If True, the induced fit rate constant is fixed, default is False.
    kc_value : float, optional
        Value of the induced fit rate constant if fixed_kc is True.
    fixed_krev : bool, optional
        If True, the reverse induced fit rate constant is fixed, default is False.
    krev_value : float, optional
        Value of the reverse induced fit rate constant if fixed_krev is True.
    max_nfev : int, optional
        Maximum number of function evaluations for the fit.

    Returns
    -------
    list
        Fitted parameters.
    np.ndarray
        Covariance matrix of the fitted parameters.
    list
        Fitted values for each signal, same dimensions as signal_lst.
    list
        Names of the fitted parameters.

    Notes
    -----
    The initial_parameters are given in the following order:

    - signal_E1    if fit_signal_E                           (signal of the free protein state 1)
    - signal_E2    if fit_signal_E and E1_equals_E2          (signal of the free protein state 2)
    - signal_S     if fit_signal_S                         (signal of the free ligand)
    - signal_E2S    if fit_signal_E2S                           (signal of the complex)
    - k_on         if not fixed_kon                           (association rate constant)
    - k_off        if not fixed_koff                          (dissociation rate constant)
    - k_c          if not fixed_kc                            (conformational rate constant)
    - k_rev        if not fixed_krev                          (reverse conformational rate constant)
    - t0_1         if not fixed_t0                            (initial time point for the first signal array, default is 0)
    - t0_2         if not fixed_t0                            (initial time point for the second signal array, default is 0)
    """
    # Flatten signals once
    all_signal = np.concatenate(signal_lst)

    # Preprocess time
    time_lst = [np.asarray(t) for t in time_lst]

    # Create an empty list that will contain the parameter names
    parameter_names = []
    if fit_signal_E:

        if E1_equals_E2:

            parameter_names.append("Signal of the free protein")

        else:

            parameter_names.append("Signal of the inactive protein (E1)")
            parameter_names.append("Signal of the active protein (E2)")

    if fit_signal_S: parameter_names.append("Signal of the free ligand")
    if fit_signal_E2S: parameter_names.append("Signal of the complex")

    if not fixed_kon: parameter_names.append("k_on [1/(µM·s)]")
    if not fixed_koff: parameter_names.append("k_off [1/s]")
    if not fixed_kc: parameter_names.append("k_c [1/s]")
    if not fixed_krev: parameter_names.append("k_rev [1/s]")

    if not fixed_t0:
        for i in range(len(time_lst)):
            parameter_names.append(f't0_{i + 1}')

    def fit_fx(_, *args):
        # Efficient argument unpacking
        idx = 0

        signal_E1 = 0 if not fit_signal_E else args[idx]

        idx += fit_signal_E * (not E1_equals_E2)

        # fit_signal_E and not E1_equals_E2, idx = 1
        # fit_signal_E and E1_equals_E2, idx = 0

        signal_E2 = 0 if not fit_signal_E else args[idx]

        idx += fit_signal_E

        signal_S = 0 if not fit_signal_S else args[idx]
        idx += fit_signal_S

        signal_E2S = 0 if not fit_signal_E2S else args[idx]
        idx += fit_signal_E2S

        k_on = kon_value if fixed_kon else args[idx]
        idx += not fixed_kon

        k_off = koff_value if fixed_koff else args[idx]
        idx += not fixed_koff

        k_c = kc_value if fixed_kc else args[idx]
        idx += not fixed_kc

        k_rev = krev_value if fixed_krev else args[idx]
        idx += not fixed_krev

        # Preallocate lists
        signal_a = [None] * len(time_lst)

        # Association phase
        for i, t in enumerate(time_lst):

            t0 = 0 if fixed_t0 else args[idx + i]  # Initial time point for the current signal

            lig_conc  = ligand_conc_lst[i]
            prot_conc = protein_conc_lst[i]

            E1, E2 = get_initial_concentration_conformational_selection(prot_conc,k_c,k_rev)

            signal = signal_ode_conformational_selection_insolution(

                t,
                y=[E2, 0],  # Initial concentrations of E2 and E2S (complex)
                k1=k_c,
                k_minus1=k_rev,
                k2=k_on,
                k_minus2=k_off,
                E_tot=prot_conc,
                S_tot=lig_conc,
                t0=t0,
                signal_E1=signal_E1,
                signal_E2=signal_E2,
                signal_S=signal_S,
                signal_E2S=signal_E2S)

            signal_a[i] = signal

        return np.concatenate(signal_a)

    # Run fitting
    global_fit_params, cov = curve_fit(
        fit_fx, xdata=1, ydata=all_signal,
        p0=initial_parameters,
        bounds=(low_bounds, high_bounds),
        max_nfev=max_nfev
    )

    # Predict
    predicted = fit_fx(1, *global_fit_params)

    # Split fitted values
    fitted_values = []
    idx = 0
    for t in time_lst:
        n = len(t)
        fitted_values.append(predicted[idx:idx + n])
        idx += n

    return global_fit_params, cov, fitted_values, parameter_names

def find_initial_parameters_induced_fit_solution(
        signal_lst,
        time_lst,
        ligand_conc_lst,
        protein_conc_lst,
        fit_signal_E=False,
        fit_signal_S=False,
        fit_signal_ES=True,
        ESint_equals_ES=True,
        fixed_t0=True,
        np_linspace_low=-2,
        np_linspace_high=2,
        np_linspace_num=5
        ):

    """
    Find initial parameters for the global fit to association and dissociation traces - one-to-one binding model with induced fit.
    
    Heuristic algorithm to explore a range of fixed kc and krev values, and fit kon and koff.
    This function is used to find the initial parameters for the fit_induced_fit_solution function.

    Parameters
    ----------
    signal_lst : list
        List of signals. We assume initial values as follows: E = E_tot, S = S_tot, ES = 0, ESint = 0.
    time_lst : list
        List of time points for the association signals.
    ligand_conc_lst : list
        List of ligand concentrations, one per element in signal_lst.
    protein_conc_lst : list
        List of protein concentrations, one per element in signal_lst.
    fit_signal_E : bool, optional
        If True, fit the signal of the free protein, default is False.
    fit_signal_S : bool, optional
        If True, fit the signal of the free ligand, default is False.
    fit_signal_ES : bool, optional
        If True, fit the signal of the complex, default is True.
    ESint_equals_ES : bool, optional
        If True, the signal of the intermediate complex is equal to the signal of the trapped complex, default is True.
    fixed_t0 : bool, optional
        If True, the initial time point is zero, default is True.
    np_linspace_low : float, optional
        Lower bound for the logarithmic space to sample kc and krev, default is -2.
    np_linspace_high : float, optional
        Upper bound for the logarithmic space to sample kc and krev, default is 2.
    np_linspace_num : int, optional
        Number of points to sample in the logarithmic space, default is 5.
        
    Returns
    -------
    np.ndarray
        Best initial parameters found by the heuristic algorithm.
    """
    kc_seq   = 10 ** (np.linspace(np_linspace_low, np_linspace_high, np_linspace_num))
    krev_seq = kc_seq

    # Find combinations of kc and krev - create a dataframe with all combinations
    combinations    = list(itertools.product(kc_seq, krev_seq))

    # Create a DataFrame with the combinations
    df_combinations = pd.DataFrame(combinations, columns=['kc', 'krev'])

    # Filter by Kc >= Krev*0.1
    df_combinations = df_combinations[df_combinations['kc'] >= df_combinations['krev'] * 0.1]

    rss_init    = np.inf
    best_params = None

    # Initial parameters are as follows:
    # 1. Signal amplitude of the free protein - removed if not fit_signal_E
    # 2. Signal amplitude of the free ligand - removed if not fit_signal_S
    # 3. Signal amplitude of the complex - removed if not fit_signal_ES
    # 4. Signal amplitude of the intermediate complex - only included if fit_signal_ES and not ESint_equals_ES
    # 5. Association rate constant
    # 6. Dissociation rate constant
    # 7. Induced fit forward rate constant - not included in the next loop with fixed_kc and fixed_krev
    # 8. Induced fit reverse rate constant - not included in the next loop with fixed_kc and fixed_krev
    # 9. t0 of the first curve
    # ... t0 of the last curve

    max_signal = np.max(signal_lst)
    max_prot   = np.max(protein_conc_lst)
    max_lig    = np.max(ligand_conc_lst)

    max_signal_P = max_signal / max_prot
    max_signal_L = max_signal / max_lig

    initial_parameters = []
    low_bounds = []
    high_bounds = []

    if fit_signal_E:
        initial_parameters.append(max_signal_P)
        low_bounds.append(0)
        high_bounds.append(np.inf)

    if fit_signal_S:
        initial_parameters.append(max_signal_L)
        low_bounds.append(0)
        high_bounds.append(np.inf)

    if fit_signal_ES:
        initial_parameters.append(max_signal_P)
        low_bounds.append(0)
        high_bounds.append(np.inf)

        if not ESint_equals_ES:
            initial_parameters.append(max_signal_P)
            low_bounds.append(0)
            high_bounds.append(np.inf)

    # Append kon and koff
    initial_parameters += [1, 1]
    low_bounds         += [1e-4, 1e-4]
    high_bounds        += [1e4, 1e4]

    # Include the t0 parameter
    if not fixed_t0:
        n_t0 = len(time_lst)
        initial_parameters += [0] * n_t0
        low_bounds         += [-0.1] * n_t0
        high_bounds        += [0.1] * n_t0

    for index, row in df_combinations.iterrows():

        kc   = row['kc']
        krev = row['krev']

        global_fit_params, _, fitted_values, _ = fit_induced_fit_solution(signal_lst, time_lst, ligand_conc_lst,
                                                                         protein_conc_lst, initial_parameters,
                                                                         low_bounds,
                                                                         high_bounds,
                                                                         fixed_kc=True, kc_value=kc,
                                                                         fixed_krev=True, krev_value=krev,
                                                                         fit_signal_E=fit_signal_E,
                                                                         fit_signal_S=fit_signal_S,
                                                                         fit_signal_ES=fit_signal_ES,
                                                                         ESint_equals_ES=ESint_equals_ES,
                                                                         fixed_t0=fixed_t0)

        # We fit k_on and k_off, using fixed kc and krev
        rss = np.sum((np.concatenate(signal_lst) - np.concatenate(fitted_values)) ** 2)

        if rss < rss_init:
            rss_init = rss

            # We need to include kc and krev before the t0 parameter
            if fixed_t0:
                best_params = np.concatenate((global_fit_params, [kc, krev]))
            else:
                best_params = np.concatenate((global_fit_params[:-n_t0], [kc, krev],global_fit_params[-n_t0:]))

            initial_parameters = global_fit_params.tolist() 

    return best_params


def find_initial_parameters_conformational_selection_solution(
        signal_lst,
        time_lst,
        ligand_conc_lst,
        protein_conc_lst,
        fit_signal_E=False,
        E1_equals_E2=False,
        fit_signal_S=False,
        fit_signal_E2S=True,
        fixed_t0=True,
        np_linspace_low=-2,
        np_linspace_high=2,
        np_linspace_num=5
):
    """
    Find initial parameters for the global fit to association and dissociation traces
    one-to-one binding model with conformational selection.

    Heuristic algorithm to explore a range of fixed kc and krev values, and fit kon and koff.
    This function is used to find the initial parameters for the fit_conformational_selection_solution function.

    Parameters
    ----------
    signal_lst : list
        List of signals
    time_lst : list
        List of time points for the association signals.
    ligand_conc_lst : list
        List of ligand concentrations, one per element in signal_lst.
    protein_conc_lst : list
        List of protein concentrations, one per element in signal_lst.
    fit_signal_E : bool, optional
        If True, fit the signal of the free protein, default is False.
    fit_signal_S : bool, optional
        If True, fit the signal of the free ligand, default is False.
    fit_signal_ES : bool, optional
        If True, fit the signal of the complex, default is True.
    ESint_equals_ES : bool, optional
        If True, the signal of the intermediate complex is equal to the signal of the trapped complex, default is True.
    fixed_t0 : bool, optional
        If True, the initial time point is zero, default is True.
    np_linspace_low : float, optional
        Lower bound for the logarithmic space to sample kc and krev, default is -2.
    np_linspace_high : float, optional
        Upper bound for the logarithmic space to sample kc and krev, default is 2.
    np_linspace_num : int, optional
        Number of points to sample in the logarithmic space, default is 5.

    Returns
    -------
    np.ndarray
        Best initial parameters found by the heuristic algorithm.
    """
    kc_seq   = 10 ** (np.linspace(np_linspace_low, np_linspace_high, np_linspace_num))
    krev_seq = kc_seq

    # Find combinations of kc and krev - create a dataframe with all combinations
    combinations = list(itertools.product(kc_seq, krev_seq))

    # Create a DataFrame with the combinations
    df_combinations = pd.DataFrame(combinations, columns=['kc', 'krev'])

    # Filter by Krev >= Kc
    df_combinations = df_combinations[df_combinations['krev'] >= df_combinations['kc'] * 0.1]

    rss_init    = np.inf
    best_params = None

    # Initial parameters are as follows:
    # 1. Signal amplitude of the free protein - removed if not fit_signal_E
    # If E1_equals_E2, then the signal of the inactive and active state are the same
    # 2. Signal amplitude of the free ligand - removed if not fit_signal_S
    # 3. Signal amplitude of the complex - removed if not fit_signal_E2S
    # 4. Association rate constant
    # 5. Dissociation rate constant
    # 6. Conf. selection forward rate constant - not included in the next loop with fixed_kc and fixed_krev
    # 7. Conf. selection reverse rate constant - not included in the next loop with fixed_kc and fixed_krev
    # 9. t0 of the first curve
    # ... t0 of the last curve

    max_signal = np.max(signal_lst)
    max_prot   = np.max(protein_conc_lst)
    max_lig    = np.max(ligand_conc_lst)

    max_signal_P = max_signal / max_prot
    max_signal_L = max_signal / max_lig

    initial_parameters = []
    low_bounds  = []
    high_bounds = []

    if fit_signal_E:
        initial_parameters.append(max_signal_P)
        low_bounds.append(0)
        high_bounds.append(np.inf)

        if not E1_equals_E2:
            initial_parameters.append(max_signal_P)
            low_bounds.append(0)
            high_bounds.append(np.inf)

    if fit_signal_S:
        initial_parameters.append(max_signal_L)
        low_bounds.append(0)
        high_bounds.append(np.inf)

    if fit_signal_E2S:
        initial_parameters.append(max_signal_P)
        low_bounds.append(0)
        high_bounds.append(np.inf)

    # Append kon and koff
    initial_parameters += [1, 1]
    low_bounds += [1e-4, 1e-4]
    high_bounds += [1e4, 1e4]

    # Include the t0 parameter
    if not fixed_t0:
        n_t0 = len(time_lst)
        initial_parameters += [0] * n_t0
        low_bounds += [-0.1] * n_t0
        high_bounds += [0.1] * n_t0

    for index, row in df_combinations.iterrows():

        kc   = row['kc']
        krev = row['krev']

        global_fit_params, _, fitted_values, _ = fit_conformational_selection_solution(
            signal_lst, time_lst, ligand_conc_lst,
            protein_conc_lst, initial_parameters,
            low_bounds,
            high_bounds,
            fixed_kc=True, kc_value=kc,
            fixed_krev=True, krev_value=krev,
            fit_signal_E=fit_signal_E,
            E1_equals_E2=E1_equals_E2,
            fit_signal_S=fit_signal_S,
            fit_signal_E2S=fit_signal_E2S,
            fixed_t0=fixed_t0)

        # We fit k_on and k_off, using fixed kc and krev
        rss = np.sum((np.concatenate(signal_lst) - np.concatenate(fitted_values)) ** 2)

        if rss < rss_init:
            rss_init = rss

            # We need to include kc and krev before the t0 parameter
            if fixed_t0:
                best_params = np.concatenate((global_fit_params, [kc, krev]))
            else:
                best_params = np.concatenate((global_fit_params[:-n_t0], [kc, krev], global_fit_params[-n_t0:]))

            initial_parameters = global_fit_params.tolist()

    return best_params