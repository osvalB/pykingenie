from .utils.math            import *
from .utils.fitting_general import *
from .utils.fitting_solution import *

class KineticsFitterSolution:

    """
    A class used to fit solution-based kinetics data with shared thermodynamic parameters

        Attributes
    ----------

        name (str):                     name of the experiment
        assoc_lst (list):                   list containing the association signals

        lig_conc (list):                list of ligand concentrations, one per element in assoc
        prot_conc (list):               list of protein concentrations, one per element in assoc

        time_assoc_lst (list):              list of time points for the association signals

        signal_ss (list):               list of steady state signals
        signal_ss_fit (list):           list of steady state fitted signals
        signal_assoc_fit (list):        list of association kinetics fitted signals
        fit_params_kinetics (pd.Dataframe):     dataframe with the fitted parameters
        fit_params_ss (pd.Dataframe):           dataframe with the values of the fitted parameters - steady state

    """

    def __init__(self, name, assoc, lig_conc, protein_conc, time_assoc):
        """
        Initialize the KineticsFitterSolution class

        Args:
            name (str):                     name of the experiment
            assoc (list):                   list containing the association signals
            lig_conc (list):                list of ligand concentrations, one per element in assoc
            protein_conc (list):            list of protein concentrations, one per element in assoc
            time_assoc (list):              list of time points for the association signals, one per replicate

        """
        self.name  = name
        self.assoc_lst = assoc

        self.lig_conc     = lig_conc
        self.prot_conc    = protein_conc

        self.time_assoc_lst = time_assoc
        self.signal_ss        = None
        self.signal_ss_fit    = None
        self.signal_assoc_fit = None

        # Create empty dataframes for the fitted parameters
        self.fit_params_kinetics = None
        self.fit_params_ss       = None

    def get_steady_state(self):

        """
        Get the steady state signals from the association signals.
        The steady state signal is the last value of each association signal.

        We calculate the steady signal grouped by protein concentration.

        """

        signal_ss_per_protein   = []
        ligand_conc_per_protein = []

        unq_prot_conc = pd.unique(np.array(self.prot_conc)) # Follow the order of appearance in the list

        for prot in unq_prot_conc:

            # Get the indices of the association signals for the current protein concentration
            indices = [i for i, p in enumerate(self.prot_conc) if p == prot]

            # Get the corresponding association signals
            assoc_signals = [self.assoc_lst[i] for i in indices]

            # Get the corresponding ligand concentrations
            lig_conc = [self.lig_conc[i] for i in indices]

            # Calculate the steady state signal as the median of the last 5 values of each signal
            signal_ss = [np.median(y[-5:]) if len(y) >= 5 else np.nan for y in assoc_signals]

            signal_ss_per_protein.append(np.array(signal_ss))
            ligand_conc_per_protein.append(np.array(lig_conc))

        self.unq_prot_conc         = unq_prot_conc
        self.signal_ss_per_protein = signal_ss_per_protein
        self.lig_conc_per_protein  = ligand_conc_per_protein

        return None

    def clear_fittings(self):

        self.signal_assoc_fit    = None
        self.fit_params_kinetics = None
        self.fit_params_ss       = None

        return None

    def fit_single_exponentials(self):

        """
        Fit single exponentials to the association signals in the solution kinetics experiment.
        """
        self.clear_fittings()

        k_obs  = []
        y_pred = []

        for y,t in zip(self.assoc_lst,self.time_assoc_lst):

            try:

                fit_params, cov, fit_y = fit_single_exponential(y,t)

                k_obs.append(fit_params[2])
                y_pred.append(fit_y)

            except:

                k_obs.append(np.nan)
                y_pred.append(None)

        self.k_obs            = k_obs
        self.signal_assoc_fit = y_pred

        self.group_k_obs_by_protein_concentration()

        # Create a DataFrame with the fitted parameters and assign it to fit_params_kinetics
        self.fit_params_kinetics = pd.DataFrame({
            'Protein [µM]': self.prot_conc,
            'Ligand [µM]':  self.lig_conc,
            'k_obs [1/s]':  self.k_obs
        })

        return None

    def group_k_obs_by_protein_concentration(self):

        """
        Group the observed rate constants by protein concentration.
        This is useful for plotting the rate constants against the protein concentration.
        """
        k_obs_per_prot = {}

        for prot, k in zip(self.prot_conc, self.k_obs):
            if prot not in k_obs_per_prot:
                k_obs_per_prot[prot] = []
            k_obs_per_prot[prot].append(k)

        # Convert to numpy arrays for easier handling later
        for prot in k_obs_per_prot:
            k_obs_per_prot[prot] = np.array(k_obs_per_prot[prot])

        self.k_obs_per_prot = k_obs_per_prot

        return None

    def fit_double_exponentials(self, min_log_k=-4, max_log_k=4, log_k_points=22):

        """
        Fit double exponentials to the association signals in the solution kinetics experiment.
        """
        k_obs_1 = []
        k_obs_2 = []
        y_pred  = []

        for y,t in zip(self.assoc_lst,self.time_assoc_lst):

            try:

                params, cov, fitted_y = fit_double_exponential(y,t,min_log_k=min_log_k,max_log_k=max_log_k,log_k_points=log_k_points)

                slowest_k = np.min([params[2], params[4]])
                second_k  = np.max([params[2], params[4]])

                k_obs_1.append(slowest_k)
                k_obs_2.append(second_k)
                y_pred.append(fitted_y)

            except:

                k_obs_1.append(np.nan)
                k_obs_2.append(np.nan)
                y_pred.append(None)

        self.k_obs_1 = k_obs_1
        self.k_obs_2 = k_obs_2
        self.signal_assoc_fit = y_pred

        self.group_double_exponential_k_obs_by_protein_concentration()

        # Create a DataFrame with the fitted parameters and assign it to fit_params_kinetics
        self.fit_params_kinetics = pd.DataFrame({
            'Protein [µM]': self.prot_conc,
            'Ligand [µM]':  self.lig_conc,
            'k_obs_1 [1/s]': self.k_obs_1,
            'k_obs_2 [1/s]': self.k_obs_2
        })

        return None

    def group_double_exponential_k_obs_by_protein_concentration(self):

        """
        Group the observed rate constants by protein concentration for double exponential fits.
        This is useful for plotting the rate constants against the protein concentration.
        """
        k_obs_1_per_prot = {}
        k_obs_2_per_prot = {}

        for prot, k1, k2 in zip(self.prot_conc, self.k_obs_1, self.k_obs_2):
            if prot not in k_obs_1_per_prot:
                k_obs_1_per_prot[prot] = []
                k_obs_2_per_prot[prot] = []
            k_obs_1_per_prot[prot].append(k1)
            k_obs_2_per_prot[prot].append(k2)

        # Convert to numpy arrays for easier handling later
        for prot in k_obs_1_per_prot:
            k_obs_1_per_prot[prot] = np.array(k_obs_1_per_prot[prot])
            k_obs_2_per_prot[prot] = np.array(k_obs_2_per_prot[prot])

        self.k_obs_1_per_prot = k_obs_1_per_prot
        self.k_obs_2_per_prot = k_obs_2_per_prot

        return None

    def fit_one_binding_site(self):

        """
        Fit the association signals assuming one binding site.
        This is a simplified model that assumes a single binding site for the ligand on the protein.
        """

        initial_parameters = [1,np.mean(self.lig_conc),1] # Signal amplitude of the complex, Kd, and koff
        low_bounds         = [0,np.min(self.lig_conc)/1e2,1e-2]
        high_bounds        = [np.inf,np.max(self.lig_conc)*1e2,np.inf]

        global_fit_params, cov, fitted_values, parameter_names = fit_one_site_solution(
            signal_lst=self.assoc_lst,
            time_lst=self.time_assoc_lst,
            ligand_conc_lst=self.lig_conc,
            protein_conc_lst=self.prot_conc,
            initial_parameters=initial_parameters,
            low_bounds=low_bounds,
            high_bounds=high_bounds,
            fit_signal_E=False,
            fit_signal_S=False,
            fit_signal_ES=True,
            fixed_t0=True,
            fixed_Kd=False,
            Kd_value=None,
            fixed_koff=False,
            koff_value=None
        )

        self.signal_assoc_fit = fitted_values

        # Create a DataFrame with the fitted parameters and assign it to fit_params_kinetics

        self.fit_params_kinetics = pd.DataFrame({
            'Protein [µM]': self.prot_conc,
            'Ligand [µM]':  self.lig_conc,
        })

        for i, param in enumerate(parameter_names):
            self.fit_params_kinetics[param] = global_fit_params[i]

        return None

    def find_initial_params_if(self,
                               fit_signal_E=False,
                               fit_signal_S=False,
                               fit_signal_ES=True,
                               ESint_equals_ES=True,
                               fixed_t0=True
                               ):

        # Heuristically find the best initial parameters for the fit
        # We explore fixed values of kc and krev and fit kon and koff (and the signal of the complex)
        params_guess = find_initial_parameters_induced_fit_solution(
            signal_lst=self.assoc_lst,
            time_lst=self.time_assoc_lst,
            ligand_conc_lst=self.lig_conc,
            protein_conc_lst=self.prot_conc,
            fit_signal_E=fit_signal_E,
            fit_signal_S=fit_signal_S,
            fit_signal_ES=fit_signal_ES,
            ESint_equals_ES=ESint_equals_ES,
            fixed_t0=fixed_t0)

        self.params_guess = params_guess

        return None

    def fit_induced_fit(self,
                        fit_signal_E=False,
                        fit_signal_S=False,
                        fit_signal_ES=True,
                        ESint_equals_ES=True,
                        fixed_t0=True
                        ):

        """
        Fit the association signals assuming induced fit.
        This model accounts for conformational changes in the protein upon ligand binding.
        Args:
            fit_signal_E (bool): If True, fit the signal of the free protein E.
            fit_signal_S (bool): If True, fit the signal of the free ligand S.
            fit_signal_ES (bool): If True, fit the signal of the complex ES.
            ESint_equals_ES (bool): If True, assume that the signal of the intermediate ESint is equal to the signal of the complex ES.
            fixed_t0 (bool): If True, fix the t0 parameter to 0.
        """

        # fit using as initial parameters the best found parameters
        initial_parameters = np.array(self.params_guess )
        low_bounds  = initial_parameters / 1e3
        high_bounds = initial_parameters * 1e3

        global_fit_params, cov, fitted_values, parameter_names = fit_induced_fit_solution(
            signal_lst=self.assoc_lst,
            time_lst=self.time_assoc_lst,
            ligand_conc_lst=self.lig_conc,
            protein_conc_lst=self.prot_conc,
            initial_parameters= initial_parameters,
            low_bounds=low_bounds,
            high_bounds=high_bounds,
            fit_signal_E=fit_signal_E,
            fit_signal_S=fit_signal_S,
            fit_signal_ES=fit_signal_ES,
            ESint_equals_ES=ESint_equals_ES,
            fixed_t0=fixed_t0
        )

        self.signal_assoc_fit = fitted_values

        # Create a DataFrame with the fitted parameters and assign it to fit_params_kinetics
        self.fit_params_kinetics = pd.DataFrame({
            'Protein [µM]': self.prot_conc,
            'Ligand [µM]':  self.lig_conc,
        })

        for i, param in enumerate(parameter_names):
            self.fit_params_kinetics[param] = global_fit_params[i]

        return None

