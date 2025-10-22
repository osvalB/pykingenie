import itertools
import numpy as np
import pandas as pd

from .utils.fitting_surface import (
    guess_initial_signal,
    fit_steady_state_one_site,
    steady_state_one_site_asymmetric_ci95,
    fit_one_site_association,
    fit_one_site_dissociation,
    fit_one_site_assoc_and_disso,
    fit_induced_fit_sites_assoc_and_disso,
    fit_one_site_assoc_and_disso_ktr,
    one_site_assoc_and_disso_asymmetric_ci95,
    one_site_assoc_and_disso_asymmetric_ci95_koff,
    get_smax_upper_bound_factor
)

from .utils.fitting_general import (
    fit_single_exponential,
    re_fit,
    re_fit_2
)

from .utils.processing      import (
    subset_data,
    concat_signal_lst,
    expand_parameter_list
)

from .utils.math import  get_desired_rss

from .fitter import KineticsFitterGeneral

class KineticsFitter(KineticsFitterGeneral):
    """
    A class used to fit kinetics data with shared thermodynamic parameters.

    Parameters
    ----------
    time_assoc_lst : list
        List of time points for the association signals, one per replicate.
    association_signal_lst : list
        List of association signals, one per replicate.
    lig_conc_lst : list
        List of ligand concentrations, one per replicate.
    time_diss_lst : list, optional
        List of time points for the dissociation signals, one per replicate.
    dissociation_signal_lst : list, optional
        List of dissociation signals, one per replicate.
    smax_id : list, optional
        List containing the Smax IDs (maximum amplitude identifiers).
    name_lst : list, optional
        List of experiment names.
    is_single_cycle : bool, optional
        Whether the experiment is a single cycle kinetics experiment.

    Attributes
    ----------
    names : list
        List of experiment names.
    assoc_lst : list
        List of association signals, one per replicate.
        Each signal is a numpy matrix of size n*m where n is the number of time points and m
        is the number of ligand concentrations.
    disso_lst : list
        List of dissociation signals, one per replicate.
        Each signal is a numpy matrix of size n*m where n is the number of time points and m
        is the number of ligand concentrations (different from zero).
    lig_conc_lst : list
        List of ligand concentrations, one per replicate.
    time_assoc_lst : list
        List of time points for the association signals, one per replicate.
    time_disso_lst : list
        List of time points for the dissociation signals, one per replicate.
    is_single_cycle : bool
        Whether the experiment is a single cycle kinetics experiment.
    signal_ss : list
        List of steady state signals, one per replicate.
    signal_ss_fit : list
        List of steady state fitted signals, one per replicate.
    signal_assoc_fit : list
        List of association kinetics fitted signals, one per replicate.
    signal_disso_fit : list
        List of dissociation kinetics fitted signals, one per replicate.
    lig_conc_lst_per_id : list
        Ligand concentrations per Smax ID.
    smax_guesses_unq : list
        Smax guesses per association signal.
    smax_guesses_shared : list
        Smax guesses per Smax ID.
    fit_params_kinetics : pd.DataFrame
        DataFrame with the fitted parameters for the association/dissociation kinetics.
    fit_params_ss : pd.DataFrame
        DataFrame with the values of the fitted parameters - steady state.
    """

    def __init__(self,time_assoc_lst,association_signal_lst,lig_conc_lst,
                 time_diss_lst=None,dissociation_signal_lst=None,
                 smax_id=None,name_lst=None,is_single_cycle=False):
        """
        Initialize the KineticsFitter class.
        
        Parameters
        ----------
        time_assoc_lst : list
            List of time points for the association signals, one per replicate.
        association_signal_lst : list
            List of association signals, one per replicate.
        lig_conc_lst : list
            List of ligand concentrations, one per replicate.
        time_diss_lst : list, optional
            List of time points for the dissociation signals, one per replicate.
        dissociation_signal_lst : list, optional
            List of dissociation signals, one per replicate.
        smax_id : list, optional
            List containing the Smax IDs (maximum amplitude identifiers).
        name_lst : list, optional
            List of experiment names.
        is_single_cycle : bool, optional
            Whether the experiment is a single cycle kinetics experiment, default is False.
        """

        super().__init__

        self.names            = name_lst
        self.assoc_lst        = association_signal_lst
        self.disso_lst        = dissociation_signal_lst
        self.lig_conc_lst     = lig_conc_lst
        self.time_assoc_lst   = time_assoc_lst
        self.time_disso_lst   = time_diss_lst
        self.is_single_cycle  = is_single_cycle

        self.lig_conc_lst_per_id = None  # Ligand concentrations per Smax ID
        self.smax_guesses_unq    = None  # Smax guesses per association signal
        self.smax_guesses_shared = None  # Smax guesses per Smax ID

        self.Kd_ss = None

        # If smax_id is None, set it to a list of zeros
        if smax_id is None:
            smax_id = [0 for _ in range(len(association_signal_lst))]

        # We need to rearrange smax_id to start at 0, then 1, then 2, etc.
        smax_id_unq, idx = np.unique(smax_id, return_index=True)
        smax_id_unq = smax_id_unq[np.argsort(idx)]

        smax_id_new = []
        for i,unq in enumerate(smax_id_unq):
            smax_id_new += [i for _ in range(len(np.where(smax_id == unq)[0]))]

        self.smax_id = smax_id_new

        self.signal_assoc_fit = None
        self.signal_disso_fit = None
        self.signal_ss_fit    = None

        self.Smax_upper_bound_factor = 1e2  # Normal values for lower than micromolar affinity

    def get_steady_state(self):

        """
        Calculate the steady state signal and group it by Smax ID.
        
        This function calculates the steady state signal for each association curve
        and groups the signals by Smax ID. It also calculates the Smax guesses
        for each association signal.
        
        Returns
        -------
        None
            Updates the following instance attributes:
            - signal_ss: List of steady state signals grouped by Smax ID
            - lig_conc_lst_per_id: List of ligand concentrations grouped by Smax ID
            - smax_guesses_unq: Smax guesses for each association signal
            - smax_guesses_shared: Smax guesses for each Smax ID
        """

        signals_steady_state = [np.median(assoc[-10:]) for assoc in self.assoc_lst]

        # Create a new list, that will contain one element per Smax ID
        # each element will be a list of steady-state signals

        self.signal_ss            = [] # convert to list of lists, one list per unique smax id
        self.lig_conc_lst_per_id  = [] # convert to list of lists, one list per unique smax id

        # Obtain the smax guesses, the maximum signal
        smax_guesses_unq    = [] # One element per association signal
        smax_guesses_shared = [] # One element per Smax ID

        for i,smax_id in enumerate(np.unique(self.smax_id)):

            idx = np.where(self.smax_id == smax_id)[0]

            self.signal_ss.append([signals_steady_state[i] for i in idx])
            self.lig_conc_lst_per_id.append([self.lig_conc_lst[i] for i in idx])

            smax = np.max(self.signal_ss[i])
            smax_guesses_unq.append(smax*1.5)
            smax_guesses_shared += [smax*1.5 for _ in range(len(idx))]

        self.smax_guesses_unq    = smax_guesses_unq
        self.smax_guesses_shared = smax_guesses_shared

        return None

    def fit_steady_state(self):

        """
        Fit the steady state signal using the one-site binding model.
        
        This function fits the steady state signal to a one-site binding model
        to estimate the equilibrium dissociation constant (Kd) and maximum signal (Smax).
        It also calculates the 95% confidence intervals for the Kd.
        
        Returns
        -------
        None
            Updates the following instance attributes:
            - params: The fitted parameters [Kd, Smax1, Smax2, ...]
            - signal_ss_fit: The fitted steady state signals
            - fit_params_ss: DataFrame with the fitted parameters
            - Kd_ss: The equilibrium dissociation constant
            - Smax_upper_bound_factor: Factor to determine the upper bound for Smax
        """

        self.clear_fittings()

        if self.signal_ss is None:

            self.get_steady_state()

        Kd_init = np.median(self.lig_conc_lst_per_id[0])
        p0      = [Kd_init] + [np.max(signal) for signal in self.signal_ss]

        kd_min  = np.min(self.lig_conc_lst_per_id[0]) / 1e3
        kd_max  = np.max(self.lig_conc_lst_per_id[0]) * 1e3

        # Find the upper bound for the Kd
        upper_bound = 1e3 if Kd_init >= 1 else 1e2

        low_bounds  = [kd_min]  + [x*0.5          for x in p0[1:]]
        high_bounds = [kd_max]  + [x*upper_bound  for x in p0[1:]]

        # testing - set upper bound for smax to smax
        high_bounds = [kd_max]  + [x*1  for x in p0[1:]]

        fit, cov, fit_vals = fit_steady_state_one_site(
            self.signal_ss,self.lig_conc_lst_per_id,
            p0,low_bounds,high_bounds)

        # Prepare the arguments for re-fitting fit_steady_state_one_site
        kwargs = {
            "signal_lst": self.signal_ss,
            "ligand_lst": self.lig_conc_lst_per_id
        }

        fit, cov, fit_vals, low_bounds, high_bounds = re_fit(
            fit=fit,
            cov=cov,
            fit_vals=fit_vals,
            fit_fx=fit_steady_state_one_site,
            low_bounds=low_bounds,
            high_bounds= high_bounds,
            times=3,
            **kwargs)

        self.Kd_ss   = fit[0]
        Smax         = fit[1:]

        self.params      = fit
        self.p0          = p0
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds

        self.signal_ss_fit = fit_vals

        n = np.sum([len(signal) for signal in self.signal_ss])
        p = len(p0)

        rss_desired = get_desired_rss(concat_signal_lst(self.signal_ss), concat_signal_lst(fit_vals), n, p)

        minKd, maxKd = steady_state_one_site_asymmetric_ci95(
            self.Kd_ss, self.signal_ss, self.lig_conc_lst_per_id, p0[1:],
            low_bounds[1:], high_bounds[1:], rss_desired)

        # Create a dataframe with the fitted parameters
        df_fit = pd.DataFrame({'Kd [µM]': self.Kd_ss,
                               'Kd_min95': minKd,
                               'Kd_max95': maxKd,
                               'Smax': Smax,
                               'Name': self.names})

        self.fit_params_ss = df_fit

        self.Smax_upper_bound_factor = get_smax_upper_bound_factor(self.Kd_ss)

        self.fit_params_ss = df_fit

        # Fit the steady state signal
        return None

    def get_k_off_initial_guess(self):
        """
        Get initial guess and bounds for k_off parameter.
        
        This method determines initial values and fitting bounds for Kd and k_off.
        If dissociation data is available, it fits the dissociation curves to get
        a better estimate of k_off.
        
        Returns
        -------
        tuple
            A tuple containing:
            - p0: Initial parameter guesses [Kd, k_off]
            - low_bounds: Lower bounds for the parameters
            - high_bounds: Upper bounds for the parameters
            
        Raises
        ------
        ValueError
            If Kd_ss is not set and fit_steady_state has not been run
        """

        # Initial guess for Kd_ss for single_cycle_kinetics
        if self.is_single_cycle:
            self.Kd_ss = np.median(self.lig_conc_lst)

        # Requires that Kd_ss is set
        if self.Kd_ss is None:
            raise ValueError("Kd_ss is not set. Please run fit_steady_state() first.")

        # We check if we have dissociation curves and fit them to get a better estimate of k_off
        if self.disso_lst is not None and self.time_disso_lst is not None:

            self.fit_one_site_dissociation()
            p0 = [self.Kd_ss, self.k_off]
            low_bounds  = [self.Kd_ss/7e2, self.k_off/7e2]
            high_bounds = [self.Kd_ss*7e2, self.k_off*7e2]

            self.clear_fittings()

        else:

            p0 = [self.Kd_ss,0.01]

            low_bounds  = [self.Kd_ss/7e2,1e-7]
            high_bounds = [self.Kd_ss*7e2,10]

        return p0, low_bounds, high_bounds

    def fit_one_site_association(self, shared_smax=True):
        """
        Fit the association curves using the one-site binding model.
        
        This function fits the association curves to a one-site binding model
        to estimate the equilibrium dissociation constant (Kd), dissociation
        rate constant (k_off), and maximum signal (Smax).
        
        Parameters
        ----------
        shared_smax : bool, optional
            Whether to share Smax across curves with the same Smax ID, default is True.
            
        Returns
        -------
        None
            Updates the following instance attributes:
            - params: The fitted parameters [Kd, k_off, Smax1, Smax2, ...]
            - signal_assoc_fit: The fitted association signals
            - Kd: The equilibrium dissociation constant
            - k_off: The dissociation rate constant
            - Smax: The maximum signal values
        """

        self.clear_fittings()

        p0, low_bounds, high_bounds = self.get_k_off_initial_guess()

        smax_guesses = self.smax_guesses_unq if shared_smax else self.smax_guesses_shared

        p0 += smax_guesses

        low_bounds  += [x/15 for x in p0[2:]]
        high_bounds += [x*self.Smax_upper_bound_factor for x in p0[2:]]

        self.p0           = p0
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds

        fit, cov, fit_vals = fit_one_site_association(
            self.assoc_lst,self.time_assoc_lst,self.lig_conc_lst,
            p0,low_bounds,high_bounds,smax_idx=self.smax_id,shared_smax=shared_smax
        )

        self.params = fit

        self.signal_assoc_fit = fit_vals

        self.Kd   = fit[0]
        self.k_off = fit[1]
        self.Smax = fit[2:]

        # Create a dataframe with the fitted parameters
        df_fit = pd.DataFrame({'Kd [µM]':   self.Kd,
                               'k_off [1/s]': self.k_off,
                               'Smax': self.Smax})

        error     = np.sqrt(np.diag(cov))
        rel_error = error/fit * 100

        df_error = pd.DataFrame({'Kd [µM]':   rel_error[0],
                                 'k_off': rel_error[1],
                                 'Smax': rel_error[2:]})

        self.fit_params_kinetics       = df_fit
        self.fit_params_kinetics_error = df_error

        return  None

    def fit_one_site_dissociation(self, time_limit=0):
        """
        Fit the dissociation curves using the one-site binding model.
        
        This function fits the dissociation curves to a one-site binding model
        to estimate the dissociation rate constant (k_off) and the initial signal (S0).
        
        Parameters
        ----------
        time_limit : float, optional
            Time limit for fitting (in seconds). If > 0, only fit data up to this time, 
            default is 0 (fit all data).
            
        Returns
        -------
        None
            Updates the following instance attributes:
            - params: The fitted parameters [k_off, S0_1, S0_2, ...]
            - signal_disso_fit: The fitted dissociation signals
            - k_off: The dissociation rate constant
            - fit_params_kinetics: DataFrame with the fitted parameters
            - fit_params_kinetics_error: DataFrame with the relative errors
        """

        self.clear_fittings()

        disso_lst     = self.disso_lst
        time_disso_lst = self.time_disso_lst

        # Fit only some data. If time_limit = 0, fit all data
        if time_limit > 0:

            disso_lst      = [x[t < (np.min(t)+time_limit)] for x,t in zip(disso_lst,time_disso_lst)]
            time_disso_lst = [t[t < (np.min(t)+time_limit)] for t in time_disso_lst]

        p0 = [0.1] + [np.max(signal) for signal in disso_lst]

        low_bounds  = [1e-7] + [x/5 for x in p0[1:]]
        high_bounds = [10]   + [x*5 for x in p0[1:]]

        self.p0           = p0
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds

        fit, cov, fit_vals = fit_one_site_dissociation(disso_lst,time_disso_lst,
                                                       p0,low_bounds,high_bounds)

        self.signal_disso_fit = fit_vals

        self.k_off = fit[0]

        # generate dataframe with the fitted parameters
        df_fit = pd.DataFrame({'k_off': self.k_off,
                               'S0': fit[1:]})

        self.fit_params_kinetics = df_fit

        # Add the fitted errors to the dataframe
        error     = np.sqrt(np.diag(cov))
        rel_error = error/fit * 100

        df_error = pd.DataFrame({'k_off [1/s]': rel_error[0],
                                 'S0':          rel_error[1:]})

        self.fit_params_kinetics_error = df_error
        self.params = fit

        return  None

    def fit_single_exponentials(self):
        """
        Fit single exponentials to the association signals.
        
        This method fits each association curve with a single exponential function
        and extracts the observed rate constants (k_obs).
        
        Returns
        -------
        None
            Updates the k_obs attribute with a list of observed rate constants
            for each association curve.
        """

        # Fit one exponential to each signal

        # Initialize a list to store the k_obs values - filld with NaNs
        k_obs = [np.nan for _ in range(len(self.assoc_lst))]

        i = 0
        for y,t in zip(self.assoc_lst,self.time_assoc_lst):

            fit_params, _, _ = fit_single_exponential(y,t)

            # Append the k_obs value to the list
            k_obs[i] = fit_params[2]

            i += 1

        self.k_obs = k_obs

        return None

    def fit_one_site_assoc_and_disso(self,shared_smax=True,fixed_t0=True,fit_ktr=False):

        self.clear_fittings()

        p0, low_bounds, high_bounds = self.get_k_off_initial_guess()

        # Extend the lower bound for Kd_ss
        low_bounds[0] = np.min([self.Kd_ss/1e3] + self.lig_conc_lst)  # Kd lower bound

        smax_guesses = self.smax_guesses_unq if shared_smax else self.smax_guesses_shared

        p0 += smax_guesses

        low_bounds  += [x/4 for x in p0[2:]]
        high_bounds += [x*self.Smax_upper_bound_factor for x in p0[2:]]

        smax_param_start = 2

        if fit_ktr:

            for i, _ in enumerate(self.smax_guesses_unq):
                p0.insert(2, 1e-4)
                low_bounds.insert(2, 1e-7)
                high_bounds.insert(2, 1)
                smax_param_start += 1

        if not fixed_t0:

            for i,_ in enumerate(self.smax_guesses_unq):
                p0.insert(2,0)
                low_bounds.insert(2,-0.01)
                high_bounds.insert(2,0.1)
                smax_param_start += 1

        if fit_ktr:

            fit, cov, fit_vals_assoc, fit_vals_disso = fit_one_site_assoc_and_disso_ktr(
                self.assoc_lst,self.time_assoc_lst,self.lig_conc_lst,
                self.disso_lst,self.time_disso_lst,
                p0,low_bounds,high_bounds,
                smax_idx=self.smax_id,
                shared_smax=shared_smax,
                fixed_t0=fixed_t0
            )

            kwargs = {
                'assoc_signal_lst' :self.assoc_lst,
                'assoc_time_lst' : self.time_assoc_lst,
                'analyte_conc_lst' : self.lig_conc_lst,
                'disso_signal_lst' :self.disso_lst,
                'disso_time_lst' : self.time_disso_lst,
                'shared_smax' : shared_smax,
                'smax_idx' : self.smax_id,
                'fixed_t0' : fixed_t0
            }

            fit, cov, fit_vals_assoc, fit_vals_disso,low_bounds, high_bounds = re_fit_2(
                fit=fit,
                cov=cov,
                fit_vals_a=fit_vals_assoc,
                fit_vals_b=fit_vals_disso,
                fit_fx=fit_one_site_assoc_and_disso_ktr,
                low_bounds=low_bounds,
                high_bounds=high_bounds,
                times=2,
                **kwargs
            )

        else:

            fit, cov, fit_vals_assoc, fit_vals_disso = fit_one_site_assoc_and_disso(
                self.assoc_lst,self.time_assoc_lst,self.lig_conc_lst,
                self.disso_lst,self.time_disso_lst,
                p0,low_bounds,high_bounds,
                smax_idx=self.smax_id,
                shared_smax=shared_smax,
                fixed_t0=fixed_t0)

        self.signal_assoc_fit = fit_vals_assoc
        self.signal_disso_fit = fit_vals_disso

        self.Kd   = fit[0]
        self.k_off = fit[1]
        self.Smax = fit[smax_param_start:]

        # Create a dataframe with the fitted parameters
        df_fit = pd.DataFrame({'Kd [µM]':     self.Kd,
                               'k_off [1/s]': self.k_off,
                               'Smax':        self.Smax})

        # Include the Kon, derived from the Kd and Koff
        df_fit['(Derived) k_on [1/µM/s]'] = df_fit['k_off [1/s]'] / df_fit['Kd [µM]']

        error     = np.sqrt(np.diag(cov))
        rel_error = error/fit * 100

        df_error = pd.DataFrame({'Kd [µM]':   rel_error[0],
                                 'k_off [1/s]': rel_error[1],
                                 'Smax': rel_error[smax_param_start:]})

        # Add the t0 parameter
        if not fixed_t0:

            t0       = fit[3:3+len(np.unique(self.smax_id))]
            t0_error = rel_error[3:3+len(np.unique(self.smax_id))]

            if not shared_smax:

                t0_all       = expand_parameter_list(t0,       self.smax_id)
                t0_error_all = expand_parameter_list(t0_error, self.smax_id)

                df_fit['t0']   = t0_all
                df_error['t0'] = t0_error_all

            else:

                df_fit['t0']   = t0
                df_error['t0'] = t0_error

        if fit_ktr:

            n_ktr      = len(np.unique(self.smax_id))
            idx_start  = 2+(not fixed_t0)*n_ktr
            ktrs       = fit[idx_start:smax_param_start]
            ktrs_error = rel_error[idx_start:smax_param_start]

            if not shared_smax:

                ktr_all       = expand_parameter_list(ktrs,       self.smax_id)
                ktr_error_all = expand_parameter_list(ktrs_error, self.smax_id)

                df_fit['Ktr']   = ktr_all
                df_error['Ktr'] = ktr_error_all

            else:

                df_fit['Ktr']   = ktrs
                df_error['Ktr'] = ktrs_error

        self.fit_params_kinetics       = df_fit
        self.fit_params_kinetics_error = df_error

        self.params      = fit
        self.p0          = p0
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds

        return  None

    def calculate_ci95(self, shared_smax=True, fixed_t0=True, fit_ktr=False):
        """
        Calculate 95% confidence intervals for the fitted parameters.
        
        This method computes asymmetrical 95% confidence intervals for the
        equilibrium dissociation constant (Kd) and the dissociation rate constant (k_off).
        
        Parameters
        ----------
        shared_smax : bool, optional
            Whether Smax was shared across curves with the same Smax ID, default is True.
        fixed_t0 : bool, optional
            Whether the time offset (t0) was fixed to zero, default is True.
        fit_ktr : bool, optional
            Whether mass transport rate constants were included in the model, default is False.
            
        Returns
        -------
        None
            Updates the fit_params_kinetics_ci95 attribute with a DataFrame containing
            the 95% confidence intervals for Kd and k_off.
            
        Notes
        -----
        This method will create an empty DataFrame if the confidence interval calculation fails.
        """

        try:

        # Compute asymmetrical 95% confidence intervals
            if not fit_ktr:

                exp_signal_concat = concat_signal_lst(self.assoc_lst+self.disso_lst)
                fit_signal_concat = concat_signal_lst(self.signal_assoc_fit+self.signal_disso_fit)
                n                 = len(exp_signal_concat)
                p                 = len(self.p0)

                rss_desired = get_desired_rss(exp_signal_concat,
                                              fit_signal_concat,
                                              n, p)

                Kd_min, Kd_max = one_site_assoc_and_disso_asymmetric_ci95(
                    self.Kd,rss_desired,
                    self.assoc_lst,self.time_assoc_lst,
                    self.lig_conc_lst,
                    self.disso_lst,self.time_disso_lst,
                    self.p0[1:],self.low_bounds[1:],self.high_bounds[1:],
                    self.smax_id,shared_smax=shared_smax,fixed_t0=fixed_t0)

                p0 = self.p0[:1] + self.p0[2:]
                low_bounds = self.low_bounds[:1] + self.low_bounds[2:]
                high_bounds = self.high_bounds[:1] + self.high_bounds[2:]

                koff_min, koff_max  = one_site_assoc_and_disso_asymmetric_ci95_koff(
                    self.k_off,rss_desired,
                    self.assoc_lst,self.time_assoc_lst,
                    self.lig_conc_lst,
                    self.disso_lst,self.time_disso_lst,
                    p0,low_bounds,high_bounds,
                    self.smax_id,shared_smax=shared_smax,fixed_t0=fixed_t0)

                # Convert ci95_Kd and ci95_koff to a nice Table
                header = ['Parameter','95% CI lower','95% CI upper']
                row1   = ['Kd [µM]',Kd_min,Kd_max]
                row2   = ['k_off [1/s]',koff_min,koff_max]

                df = pd.DataFrame([row1,row2],columns=header)

                self.fit_params_kinetics_ci95 = df

        except:

            # Generate an empty dataframe if the procedure did not work
            self.fit_params_kinetics_ci95 = pd.DataFrame()

        return  None

    def fit_one_site_if_assoc_and_disso(self, shared_smax=True):
        """
        Fit the association and dissociation signals using the induced fit model.
        
        This method first fits a simple one-site model to get initial parameter estimates,
        then explores different values of the conformational change rate constants (kc and krev)
        to find good initial guesses for the induced fit model. Finally, it fits the full
        induced fit model with these initial guesses.
        
        Parameters
        ----------
        shared_smax : bool, optional
            Whether to share Smax across curves with the same Smax ID, default is True.
            
        Returns
        -------
        None
            Updates the following instance attributes:
            - params: The fitted parameters [k_on, k_off, k_c, k_rev, Smax1, Smax2, ...]
            - signal_assoc_fit: The fitted association signals
            - signal_disso_fit: The fitted dissociation signals
            - k_on: The association rate constant
            - k_off: The dissociation rate constant
            - k_c: The forward conformational change rate constant
            - k_rev: The reverse conformational change rate constant
            - Smax: The maximum signal values
            - fit_params_kinetics: DataFrame with the fitted parameters
        """

        # Fit first a model without induced-fit

        self.fit_one_site_assoc_and_disso(shared_smax=shared_smax)

        # Get the initial parameters from the single site fit
        p0          = self.p0
        low_bounds  = self.low_bounds
        high_bounds = self.high_bounds

        # Find kon from Kd (1st fitted param) and koff (2nd fitted param)
        kon = p0[1] / p0[0]

        p0[0]          = kon
        low_bounds[0]  = kon / 1e3
        high_bounds[0] = kon * 1e3

        # We need to find good initial guesses for the kc and krev parameters
        kc_init    = np.logspace(-4, 1, 6)
        k_rev_init = kc_init

        combinations    = np.array(list(itertools.product(kc_init, k_rev_init)))
        df_combinations = pd.DataFrame(combinations, columns=['kc_init', 'kc_rev'])

        # We need kc > krev/100 - otherwise there is no detectable induced fit
        df_combinations = df_combinations[df_combinations['kc_init'] >= df_combinations['kc_rev']/100]

        rss_init    = np.inf
        best_kc     = None
        best_krev   = None
        best_params = None

        # We create a subsample of the time points to speed up the fitting for the initial guess
        time_assoc_lst_subsampled = [subset_data(t) for t in self.time_assoc_lst]
        time_disso_lst_subsampled = [subset_data(t) for t in self.time_disso_lst]
        assoc_lst_subsampled      = [subset_data(y) for y in self.assoc_lst]
        disso_lst_subsampled      = [subset_data(y) for y in self.disso_lst]

        # Loop through each combination of kc and krev
        # Apply fit_induced_fit_sites_assoc_and_disso with fixed kon2 and koff2 (corresponding to kc and krev)
        for index, row in df_combinations.iterrows():

            kc     = row['kc_init']
            krev   = row['kc_rev']

            try:

                params, cov, fit_vals_assoc, fit_vals_disso = fit_induced_fit_sites_assoc_and_disso(
                    assoc_lst_subsampled,time_assoc_lst_subsampled,self.lig_conc_lst,
                    disso_lst_subsampled,time_disso_lst_subsampled,
                    p0,low_bounds,high_bounds,
                    smax_idx=self.smax_id,
                    shared_smax=shared_smax,
                    fixed_t0=True,
                    fixed_kon2 = True, # k_on2 refers here to the forward rate constant of the induced fit
                    kon2_value=kc,
                    fixed_koff2 = True, # k_off2 refers here to the reverse rate constant of the induced fit
                    koff2_value=krev
                )

                # Calculate the residuals for the signals - use the subsampled data
                rss_asso  = np.sum([np.sum((y - fit_y)**2) for y, fit_y in zip(assoc_lst_subsampled, fit_vals_assoc)])
                rss_disso = np.sum([np.sum((y - fit_y)**2) for y, fit_y in zip(disso_lst_subsampled, fit_vals_disso)])

                rss = rss_asso + rss_disso

                if rss < rss_init:

                    rss_init = rss
                    best_kc  = kc
                    best_krev = krev
                    best_params = params

            except:

                # If the fit fails, just continue to the next combination
                continue

        # Insert the kc and krev parameters into the p0, low_bounds and high_bounds lists

        factor = 1e3  # Factor to scale the bounds for kc and krev

        p0.insert(2, best_kc)
        low_bounds.insert(2, best_kc/factor)
        high_bounds.insert(2, best_kc*factor)
        p0.insert(3, best_krev)
        low_bounds.insert(3, best_krev/factor)
        high_bounds.insert(3, best_krev*factor)

        # Replace the kon and koff with the best parameters
        p0[0]  = best_params[0]  # kon
        p0[1]  = best_params[1]  # koff

        low_bounds[0]  = best_params[0] / factor  # kon
        high_bounds[0] = best_params[0] * factor

        low_bounds[1]  = best_params[1] / factor  # koff
        high_bounds[1] = best_params[1] * factor

        fit, cov, fit_vals_assoc, fit_vals_disso = fit_induced_fit_sites_assoc_and_disso(
            self.assoc_lst,self.time_assoc_lst,self.lig_conc_lst,
            self.disso_lst,self.time_disso_lst,
            p0,low_bounds,high_bounds,
            smax_idx=self.smax_id,
            shared_smax=shared_smax,
            fixed_t0=True)

        self.signal_assoc_fit = fit_vals_assoc
        self.signal_disso_fit = fit_vals_disso

        self.k_on, self.k_off, self.k_c, self.k_rev, *self.Smax = fit

        # Create a dataframe with the fitted parameters
        df_fit = pd.DataFrame({'k_on [1/(s*µM)]':  self.k_on,
                               'k_off [1/s]':      self.k_off,
                               'k_c [1/s]':        self.k_c,
                               'k_rev [1/s]':      self.k_rev,
                               'Smax':             self.Smax})

        self.fit_params_kinetics = df_fit

        self.params      = fit
        self.p0          = p0
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds

        return None

