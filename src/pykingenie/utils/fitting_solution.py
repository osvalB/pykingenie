from scipy.optimize import curve_fit
from ..utils.signal_solution  import *
import numpy as np

def fit_induced_fit_solution(
        signal_lst,
        time_lst,
        ligand_conc_lst,
        protein_conc_lst,
        initial_parameters,
        low_bounds,
        high_bounds,
        fit_signal_E    =   False,
        fit_signal_S    =   False,
        fit_signal_ES   =   True,
        ESint_equals_ES =   True,
        fixed_t0        =   True,
        fixed_kon       =   False,
        kon_value       =   None,
        fixed_koff      =   False,
        koff_value      =   None,
        fixed_kc        =   False,
        kc_value        =   None,
        fixed_krev      =   False,
        krev_value      =   None,
        max_nfev        =   None
):
    """
    Global fit to association and dissociation traces - one-to-one binding model with induced fit

    Arguments:
        signal_lst:          list of signals. We assume initial values as follows: E = E_tot, S = S_tot, ES = 0, ESint = 0
        time_lst:            list of time points for the association signals
        ligand_conc_lst:     list of ligand concentrations, one per element in signal_lst
        protein_conc_lst:    list of protein concentrations, one per element in signal_lst
        initial_parameters:  initial parameters for the fit
        low_bounds:          lower bounds for the fit parameters
        high_bounds:         upper bounds for the fit parameters
        fit_signal_E:        if True, fit the signal of the free protein
        fit_signal_S:        if True, fit the signal of the free ligand
        fit_signal_ES:       if True, fit the signal of the complex
        ESint_equals_ES:     if True, the signal of the intermediate complex is equal to the signal of the trapped complex
        fixed_t0:            if True, the initial time point is zero
        fixed_kon:           if True, the association rate constant is fixed
        kon_value:           value of the association rate constant if fixed_kon is True
        fixed_koff:          if True, the dissociation rate constant is fixed
        koff_value:          value of the dissociation rate constant if fixed_koff is True
        fixed_kc:            if True, the induced fit rate constant is fixed
        kc_value:            value of the induced fit rate constant if fixed_kc is True
        fixed_krev:          if True, the reverse induced fit rate constant is fixed
        krev_value:          value of the reverse induced fit rate constant if fixed_krev is True
        max_nfev:            maximum number of function evaluations for the fit

    The initial_parameters are given in the following order:
        signal_E     if fit_signal_E                            (signal of the free rotein)
        signal_S     if fit_signal_S                            (signal of the free ligand)
        signal_ES    if fit_signal_ES                           (signal of the trapped complex)
        signal_ESint if fit_signal_ES and not ESint_equals_ES   (signal of the intermediate complex)
                                                                if ESint_equals_ES is True, then the trapped complex signal is used for both ES and ESint

        k_on         if not fixed_kon                           (association rate constant)
        k_off        if not fixed_koff                          (dissociation rate constant)
        k_c          if not fixed_kc                            (induced fit rate constant)
        k_rev        if not fixed_krev                          (reverse induced fit rate constant)

        t0_1         if not fixed_t0                             (initial time point for the first  signal array, default is 0)
        t0_2         if not fixed_t0                             (initial time point for the second signal array, default is 0)
        ...

    """

    # Flatten signals once
    all_signal = np.concatenate(signal_lst)

    # Preprocess time
    time_lst = [np.asarray(t) for t in time_lst]

    # Indexing logic
    start       = 4 - sum([fixed_kon, fixed_koff, fixed_kc, fixed_krev])

    n_t0s       = len(time_lst) * (not fixed_t0)

    def fit_fx(_, *args):
        # Efficient argument unpacking
        idx = 0

        signal_E  = 0 if not fit_signal_E  else args[idx]; idx += fit_signal_E
        signal_S  = 0 if not fit_signal_S  else args[idx]; idx += fit_signal_S

        if fit_signal_ES:
            signal_ES = args[idx]
            idx += 1
            if ESint_equals_ES:
                signal_ESint = signal_ES
            else:
                signal_ESint = args[idx]
                idx += 1

        k_on    = kon_value     if fixed_kon    else args[idx]; idx += not fixed_kon
        k_off   = koff_value    if fixed_koff   else args[idx]; idx += not fixed_koff
        k_c     = kc_value      if fixed_kc     else args[idx]; idx += not fixed_kc
        k_rev   = krev_value    if fixed_krev   else args[idx]; idx += not fixed_krev

        # Preallocate lists
        signal_a = [None] * len(time_lst)

        # Association phase
        for i, t in enumerate(time_lst):

            t0 = 0 if fixed_t0 else args[start + i]  # Initial time point for the current signal

            lig_conc  = ligand_conc_lst[i]
            prot_conc = protein_conc_lst[i]

            signal = signal_ode_induced_fit_insolution_reduced(

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

    return global_fit_params, cov, fitted_values