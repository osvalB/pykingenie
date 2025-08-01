from .utils.fitting_surface import *
from .utils.fitting_general import fit_single_exponential
from .utils.processing      import *
from .utils.math            import *

class KineticsFitter:

    """

    A class used to fit kinetics data with shared thermodynamic parameters

    Attributes
    ----------

        names (list):                   list of experiment names
        assoc (list):                   list of association signals, one per replicate,
                                        each signal is a numpy matrix of size n*m where n is the number of time points and m
        disso (list):                   list containing the dissociation signals
        smax_id (list):                 list containing the Smax IDs (maximum amplitude identifiers)


        disso (list):                   list of dissociation signals
                                        each signal is a numpy matrix of size n*m where n is the number of time points and m
                                        is the number of ligand concentrations (different from zero)
        lig_conc (list):                list of ligand concentrations, one per replicate
                                        each element contains a numpy array with the ligand concentrations
        time_assoc (list):              list of time points for the association signals, one per replicate
                                        each element contains a numpy array with the time points
        time_disso (list):              list of time points for the dissociation signals, one per replicate
                                        each element contains a numpy array with the time points
        signal_ss (list):               list of steady state signals, one per replicate
                                        each element contains a numpy array with the steady state signals
        signal_ss_fit (list):           list of steady state fitted signals, one per replicate
                                        each element contains a numpy array with the steady state fitted signals
        signal_assoc_fit (list):        list of association kinetics fitted signals, one per replicate
                                        each element contains a numpy array with the association kinetics fitted signals
        signal_disso_fit (list):        list of dissociation kinetics fitted signals, one per replicate
                                        each element contains a numpy array with the dissociation kinetics fitted signals
        fit_params_kinetics (pd.Dataframe):     dataframe with the fitted parameters for the association / dissociation kinetics
        fit_params_ss (pd.Dataframe):           dataframe with the values of the fitted parameters - steady state

    """

    def __init__(self,time_assoc_lst,association_signal_lst,lig_conc_lst,time_diss_lst,
                 dissociation_signal_lst=None,smax_id=None,name_lst=None,is_single_cycle=False):

        self.names            = name_lst
        self.assoc_lst        = association_signal_lst
        self.disso_lst        = dissociation_signal_lst
        self.lig_conc_lst     = lig_conc_lst
        self.time_assoc_lst   = time_assoc_lst
        self.time_disso_lst   = time_diss_lst
        self.is_single_cycle  = is_single_cycle

        self.signal_ss        = None  # Steady state signal
        self.signal_ss_fit    = None  # Steady state fitted signal
        self.signal_assoc_fit = None  # Association kinetics fitted signal
        self.signal_disso_fit = None  # Association kinetics fitted signal

        self.lig_conc_lst_per_id = None  # Ligand concentrations per Smax ID
        self.smax_guesses_unq    = None  # Smax guesses per association signal
        self.smax_guesses_shared = None  # Smax guesses per Smax ID

        self.Kd_ss = None

        self.fit_params_kinetics  = None

        # We need to rearrange smax_id to start at 0, then 1, then 2, etc.
        smax_id_unq, idx = np.unique(smax_id, return_index=True)
        smax_id_unq = smax_id_unq[np.argsort(idx)]

        smax_id_new = []
        for i,unq in enumerate(smax_id_unq):
            smax_id_new += [i for _ in range(len(np.where(smax_id == unq)[0]))]

        self.smax_id = smax_id_new

    def get_steady_state(self):

        """
        This function calculates the steady state signal and groups it by smax ID
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

    def clear_fittings(self):

        self.signal_ss_fit        = None  # Steady state fitted signal
        self.signal_assoc_fit     = None  # Association kinetics fitted signal
        self.signal_disso_fit     = None  # Association kinetics fitted signal
        self.fit_params_kinetics  = None  # Fitted parameters for the association / dissociation kinetics
        self.fit_params_ss        = None  # Values of the fitted parameters - steady state

        return None

    def create_fitting_bounds_table(self):

        """
        Create a dataframe with the fitting bounds and the fitted parameters.
        It uses self.params, self.low_bounds and self.high_bounds
        """

        df = pd.DataFrame({
            'Fitted_parameter_value':   self.params,
            'Lower_limit_for_fitting':  self.low_bounds,
            'Upper_limit_for_fitting':  self.high_bounds
        })

        self.fitted_params_boundaries = df

        return None

    def fit_steady_state(self):

        """
        This function fits the steady state signal
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
            fit, cov, fit_vals,
            fit_steady_state_one_site, low_bounds, high_bounds,
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

    def fit_one_site_association(self,shared_smax=True):

        # Initial guess for Kd_ss for single_cycle_kinetics
        if self.is_single_cycle:
            self.Kd_ss = np.median(self.lig_conc_lst)
        else:
            if self.Kd_ss is None:
                raise ValueError("Kd_ss is not set. Please run fit_steady_state() first.")

        self.clear_fittings()

        # Try to fit first the dissociation curves to get a better estimate of Koff
        try:

            self.fit_one_site_dissociation()

            p0          = [self.Kd_ss,self.Koff]
            low_bounds  = [self.Kd_ss/7e2,self.Koff/7e2]
            high_bounds = [self.Kd_ss*7e2,self.Koff*7e2]

            self.clear_fittings()

        except:

            p0 = [self.Kd_ss,0.01]

            low_bounds  = [self.Kd_ss/7e2,1e-7]
            high_bounds = [self.Kd_ss*7e2,10]

        if shared_smax:

            smax_guesses = self.smax_guesses_unq
        else:
            smax_guesses = self.smax_guesses_shared

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
        self.Koff = fit[1]
        self.Smax = fit[2:]

        # Create a dataframe with the fitted parameters
        df_fit = pd.DataFrame({'Kd [µM]':   self.Kd,
                               'k_off [1/s]': self.Koff,
                               'Smax': self.Smax})

        error     = np.sqrt(np.diag(cov))
        rel_error = error/fit * 100

        df_error = pd.DataFrame({'Kd [µM]':   rel_error[0],
                                 'k_off': rel_error[1],
                                 'Smax': rel_error[2:]})

        self.fit_params_kinetics       = df_fit
        self.fit_params_kinetics_error = df_error

        return  None

    def fit_one_site_dissociation(self,time_limit=0):

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

        self.Koff = fit[0]

        # generate dataframe with the fitted parameters
        df_fit = pd.DataFrame({'k_off': self.Koff,
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

        # Fit one exponential to each signal
        k_obs = []

        for y,t in zip(self.assoc_lst,self.time_assoc_lst):

            try:

                fit_params, cov, fit_y = fit_single_exponential(y,t)
                # Append the k_obs value to the list
                k_obs.append(fit_params[2])

            except:

                k_obs.append(np.nan)

        self.k_obs = k_obs

        return None

    def fit_one_site_assoc_and_disso(self,shared_smax=True,fixed_t0=True,fit_ktr=False):

        # Initial guess for Kd_ss for single_cycle_kinetics
        if self.is_single_cycle:
            self.Kd_ss = np.median(self.lig_conc_lst)

        self.clear_fittings()

        # check that we have Kd_ss
        if self.Kd_ss is None:
            raise ValueError("Kd_ss is not set. Please run fit_steady_state() first.")

        kd_low_bound = np.min([self.Kd_ss/1e3, np.min(self.lig_conc_lst)])

        # Try to fit first the dissociation curves to get a better estimate of Koff
        try:

            self.fit_one_site_dissociation()

            p0          = [self.Kd_ss,self.Koff]
            low_bounds  = [kd_low_bound,self.Koff/7.5e2]
            high_bounds = [self.Kd_ss*7.5e2,self.Koff*7.5e2]

        except:

            p0 = [self.Kd_ss,0.01]

            low_bounds  = [kd_low_bound,1e-7]
            high_bounds = [self.Kd_ss*7e2,10]

        if shared_smax:

            smax_guesses = self.smax_guesses_unq

        else:

            smax_guesses = self.smax_guesses_shared

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

            # Refit the data if the Ktr is at the upper bound
            if 1 - fit[2] < 0.01:

                p0[2] = 1
                low_bounds[2]  = 0.01
                high_bounds[2] = 100
                fit, cov, fit_vals_assoc, fit_vals_disso = fit_one_site_assoc_and_disso_ktr(
                    self.assoc_lst,self.time_assoc_lst,self.lig_conc_lst,
                    self.disso_lst,self.time_disso_lst,
                    p0,low_bounds,high_bounds,
                    smax_idx=self.smax_id,
                    shared_smax=shared_smax,
                    fixed_t0=fixed_t0
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
        self.Koff = fit[1]
        self.Smax = fit[smax_param_start:]

        # Create a dataframe with the fitted parameters
        df_fit = pd.DataFrame({'Kd [µM]':     self.Kd,
                               'k_off [1/s]': self.Koff,
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

    def calculate_ci95(self,shared_smax=True,fixed_t0=True,fit_ktr=False):

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
                    self.Koff,rss_desired,
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

    def fit_one_site_if_assoc_and_disso(self,shared_smax=True):

        """
        Fit the association and dissociation signals using the induced fit model.
        This function is a wrapper for the fit_induced_fit_sites_assoc_and_disso method.
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
                    fixed_kon2 = True,
                    kon2_value=kc,
                    fixed_koff2 = True,
                    koff2_value=krev
                )

                # Calculate the residuals for the signals - use the subsampled data
                rss_asso = np.sum([np.sum((y - fit_y)**2) for y, fit_y in zip(assoc_lst_subsampled, fit_vals_assoc)])
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

        self.Kon   = fit[0]
        self.Koff  = fit[1]
        self.Kc    = fit[2]
        self.Krev  = fit[3]

        self.Smax  = fit[4:]

        # Create a dataframe with the fitted parameters
        df_fit = pd.DataFrame({'k_on [1/(s*µM)]':  self.Kon,
                               'k_off [1/s]':      self.Koff,
                               'k_c [1/s]':        self.Kc,
                               'k_rev [1/s]':      self.Krev,
                               'Smax':             self.Smax})

        self.fit_params_kinetics = df_fit

        self.params      = fit
        self.p0          = p0
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds

        return None

