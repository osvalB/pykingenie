import numpy as np
import pandas as pd

from .utils.fitting_general import (
    fit_many_double_exponential,
    fit_single_exponential,
    re_fit
)

from .utils.fitting_solution import (
    fit_one_site_solution,
    fit_induced_fit_solution,
    find_initial_parameters_induced_fit_solution, find_initial_parameters_conformational_selection_solution,
    fit_conformational_selection_solution
)

from .fitter import KineticsFitterGeneral

class KineticsFitterSolution(KineticsFitterGeneral):
    """
    A class used to fit solution-based kinetics data with shared thermodynamic parameters.
    
    Parameters
    ----------
    name : str
        Name of the experiment.
    assoc : list
        List containing the association signals.
    lig_conc : list
        List of ligand concentrations, one per element in assoc.
    protein_conc : list
        List of protein concentrations, one per element in assoc.
    time_assoc : list
        List of time points for the association signals.
        
    Attributes
    ----------
    name : str
        Name of the experiment.
    assoc_lst : list
        List containing the association signals.
    lig_conc : list
        List of ligand concentrations, one per element in assoc.
    prot_conc : list
        List of protein concentrations, one per element in assoc.
    time_assoc_lst : list
        List of time points for the association signals.
    signal_ss : list
        List of steady state signals.
    signal_ss_fit : list
        List of steady state fitted signals.
    signal_assoc_fit : list
        List of association kinetics fitted signals.
    fit_params_kinetics : pd.DataFrame
        DataFrame with the fitted parameters.
    fit_params_ss : pd.DataFrame
        DataFrame with the values of the fitted parameters - steady state.
    """

    def __init__(self, name, assoc, lig_conc, protein_conc, time_assoc):
        """
        Initialize the KineticsFitterSolution class.
        
        Parameters
        ----------
        name : str
            Name of the experiment.
        assoc : list
            List containing the association signals.
        lig_conc : list
            List of ligand concentrations, one per element in assoc.
        protein_conc : list
            List of protein concentrations, one per element in assoc.
        time_assoc : list
            List of time points for the association signals.
        """

        super().__init__()

        self.name  = name
        self.assoc_lst = assoc

        self.lig_conc     = lig_conc
        self.prot_conc    = protein_conc

        self.time_assoc_lst = time_assoc

    def get_steady_state(self):
        """
        Get the steady state signals from the association signals.
        
        The steady state signal is calculated as the median of the last 5 values
        of each association signal. The signals are grouped by protein concentration.
        
        Returns
        -------
        None
            Updates the following instance attributes:
            - unq_prot_conc: Unique protein concentrations
            - signal_ss_per_protein: Steady state signals grouped by protein concentration
            - lig_conc_per_protein: Ligand concentrations grouped by protein concentration
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

    def fit_single_exponentials(self):

        """
        Fit single exponentials to the association signals in the solution kinetics experiment.
        
        This method fits each association curve with a single exponential function
        and extracts the observed rate constants (k_obs). The results are then grouped
        by protein concentration.
        
        Returns
        -------
        None
            Updates the following instance attributes:
            - k_obs: List of observed rate constants for each association curve
            - signal_assoc_fit: List of fitted association signals
            - fit_params_kinetics: DataFrame with the fitted parameters
        """
        self.clear_fittings()

        k_obs  = [np.nan for _ in range(len(self.assoc_lst))]
        y_pred = [None for _ in range(len(self.assoc_lst))]

        i = 0
        for y,t in zip(self.assoc_lst,self.time_assoc_lst):

            fit_params, cov, fit_y = fit_single_exponential(y,t)

            k_obs[i] = fit_params[2]
            y_pred[i] = fit_y

            i += 1

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
        
        This is useful for plotting the rate constants against the protein concentration
        and for subsequent analysis of the concentration dependence of the kinetics.
        
        Returns
        -------
        None
            Updates the k_obs_per_prot attribute with a dictionary mapping
            protein concentrations to arrays of observed rate constants.
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
        
        This method fits each association curve with a double exponential function
        and extracts two observed rate constants (k_obs_1 and k_obs_2). The results 
        are then grouped by protein concentration.
        
        Parameters
        ----------
        min_log_k : float, optional
            Minimum value of log10(k) to search, default is -4.
        max_log_k : float, optional
            Maximum value of log10(k) to search, default is 4.
        log_k_points : int, optional
            Number of points to sample in the log(k) space, default is 22.
            
        Returns
        -------
        None
            Updates the following instance attributes:
            - k_obs_1: List of first observed rate constants for each association curve
            - k_obs_2: List of second observed rate constants for each association curve
            - signal_assoc_fit: List of fitted association signals
            - fit_params_kinetics: DataFrame with the fitted parameters
        """
        self.clear_fittings()

        double_exp_results = fit_many_double_exponential(self.assoc_lst,self.time_assoc_lst,min_log_k, max_log_k, log_k_points)

        k_obs_1, k_obs_2, y_pred, k_obs_1_err, k_obs_2_err = double_exp_results

        self.k_obs_1 = k_obs_1
        self.k_obs_2 = k_obs_2
        self.signal_assoc_fit = y_pred

        self.k_obs_1_err = k_obs_1_err
        self.k_obs_2_err = k_obs_2_err

        self.group_double_exponential_k_obs_by_protein_concentration()

        # Create a DataFrame with the fitted parameters and assign it to fit_params_kinetics
        self.fit_params_kinetics = pd.DataFrame({
            'Protein [µM]': self.prot_conc,
            'Ligand [µM]':  self.lig_conc,
            'k_obs_1 [1/s]': self.k_obs_1,
            'k_obs_1 relative error (%)': self.k_obs_1_err,
            'k_obs_2 [1/s]': self.k_obs_2,
            'k_obs_2 relative error (%)': self.k_obs_2_err
        })

        return None

    def group_double_exponential_k_obs_by_protein_concentration(self):

        """
        Group the observed rate constants by protein concentration for double exponential fits.
        
        This is useful for plotting the rate constants against the protein concentration
        and for subsequent analysis of the concentration dependence of the kinetics
        when using double exponential fits.
        
        Returns
        -------
        None
            Updates the following instance attributes:
            - k_obs_1_per_prot: Dictionary mapping protein concentrations to arrays of first observed rate constants
            - k_obs_2_per_prot: Dictionary mapping protein concentrations to arrays of second observed rate constants
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

    def fit_one_binding_site(
            self,fit_signal_E=False,
            fit_signal_S=False,
            fit_signal_ES=True,
            fixed_t0=True):
        """
        Fit the association signals assuming one binding site.
        
        This is a simplified model that assumes a single binding site for the ligand 
        on the protein. The model fits global parameters to all curves simultaneously.

        Parameters
        ----------
        fit_signal_E : bool, optional
            If True, fit the signal of the free protein E, default is False.
        fit_signal_S : bool, optional
            If True, fit the signal of the free ligand S, default is False.
        fit_signal_ES : bool, optional
            If True, fit the signal of the complex ES, default is True.
        fixed_t0 : bool, optional
            If True, fix the t0 parameter to 0, default is True.

        Returns
        -------
        None
            Updates the following instance attributes:
            - signal_assoc_fit: List of fitted association signals
            - fit_params_kinetics: DataFrame with the fitted parameters
        """

        # Initial parameters are as follows:
        # 1. Signal amplitude of the free protein
        # 2. Signal amplitude of the free ligand
        # 3. Signal amplitude of the complex
        # 4. Kd of the complex
        # 5. koff of the complex
        # 6. t0 of the first trace
        # 7. t0 of the second trace
        # n. t0 of the last trace

        self.clear_fittings()

        max_signal = np.max(self.assoc_lst)
        max_prot   = np.max(self.prot_conc)
        max_lig    = np.max(self.lig_conc)

        max_signal_P = max_signal / max_prot
        max_signal_L = max_signal / max_lig

        min_Kd =  np.min(self.lig_conc)/1e2
        max_Kd = np.max(self.lig_conc)*1e2

        initial_parameters = [max_signal_P,max_signal_L,max_signal_P,np.mean(self.lig_conc),1]
        low_bounds  = [0,0,0,min_Kd,1e-4]
        high_bounds = [np.inf,np.inf,np.inf,max_Kd,1e5]

        # Remove the third parameter if we do not fit the signal of the complex
        if not fit_signal_ES:
            initial_parameters.pop(2)
            low_bounds.pop(2)
            high_bounds.pop(2)

        # Remove the second parameter if we do not fit the signal of the free ligand
        if not fit_signal_S:
            initial_parameters.pop(1)
            low_bounds.pop(1)
            high_bounds.pop(1)

        # Remove the first parameter if we do not fit the signal of the free protein
        if not fit_signal_E:
            initial_parameters.pop(0)
            low_bounds.pop(0)
            high_bounds.pop(0)

        # If t0 is fitted, we need to include it in the initial parameters
        if not fixed_t0:

            for _ in range(len(self.assoc_lst)):

                initial_parameters.append(0)
                low_bounds.append(-0.1)
                high_bounds.append(0.1)

        fit_params, cov, fitted_values, parameter_names = fit_one_site_solution(
            signal_lst=self.assoc_lst,
            time_lst=self.time_assoc_lst,
            ligand_conc_lst=self.lig_conc,
            protein_conc_lst=self.prot_conc,
            initial_parameters=initial_parameters,
            low_bounds=low_bounds,
            high_bounds=high_bounds,
            fit_signal_E=fit_signal_E,
            fit_signal_S=fit_signal_S,
            fit_signal_ES=fit_signal_ES,
            fixed_t0=fixed_t0,
            fixed_Kd=False,
            Kd_value=None,
            fixed_koff=False,
            koff_value=None
        )

        kwargs = {
            'signal_lst': self.assoc_lst,
            'time_lst': self.time_assoc_lst,
            'ligand_conc_lst': self.lig_conc,
            'protein_conc_lst': self.prot_conc,
            'fit_signal_E': fit_signal_E,
            'fit_signal_S': fit_signal_S,
            'fit_signal_ES': fit_signal_ES,
            'fixed_t0': fixed_t0
        }

        # Re-fit the parameters if they are close to the constraints
        fit_params, cov, fit_vals, low_bounds, high_bounds = re_fit(
            fit=fit_params,
            cov=cov,
            fit_vals=fitted_values,
            fit_fx=fit_one_site_solution,
            low_bounds=low_bounds,
            high_bounds=high_bounds,
            times = 2,
            **kwargs
        )

        self.params = fit_params
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds
        self.create_fitting_bounds_table()

        self.signal_assoc_fit = fit_vals

        # Create a DataFrame with the fitted parameters and assign it to fit_params_kinetics

        self.fit_params_kinetics = pd.DataFrame({
            'Protein [µM]': self.prot_conc,
            'Ligand [µM]':  self.lig_conc,
        })

        for i, param in enumerate(parameter_names):

            if "t0" not in param:

                self.fit_params_kinetics[param] = fit_params[i]

        # If we fit t0, we need to restructure the fit_params_kinetics DataFrame
        if not fixed_t0:
            n_t0 = len(self.assoc_lst)
            t0_params = fit_params[-n_t0:]

            # Include a column named t0
            self.fit_params_kinetics['t0'] = t0_params

        return None

    def find_initial_params_if(self,
                               fit_signal_E=False,
                               fit_signal_S=False,
                               fit_signal_ES=True,
                               ESint_equals_ES=True,
                               fixed_t0=True
                               ):
        """
        Find optimal initial parameters for the induced fit model.
        
        Heuristically finds the best initial parameters for the fit by exploring
        fixed values of kc and krev and fitting kon and koff (and the signal of the complex).
        
        Parameters
        ----------
        fit_signal_E : bool, optional
            If True, fit the signal of the free protein E, default is False.
        fit_signal_S : bool, optional
            If True, fit the signal of the free ligand S, default is False.
        fit_signal_ES : bool, optional
            If True, fit the signal of the complex ES, default is True.
        ESint_equals_ES : bool, optional
            If True, assume that the signal of the intermediate ESint is equal 
            to the signal of the complex ES, default is True.
        fixed_t0 : bool, optional
            If True, fix the t0 parameter to 0, default is True.
            
        Returns
        -------
        None
            Updates the params_guess attribute with the best initial parameters.
        """

        self.clear_fittings()

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
        Fit the association signals assuming an induced-fit mechanism.

        A + B <-> AB_intermediate <-> AB_trapped

        This model accounts for conformational changes in the protein upon ligand binding.

        Parameters
        ----------
        fit_signal_E : bool, optional
            If True, fit the signal of the free protein E, default is False.
        fit_signal_S : bool, optional
            If True, fit the signal of the free ligand S, default is False.
        fit_signal_ES : bool, optional
            If True, fit the signal of the complex ES, default is True.
        ESint_equals_ES : bool, optional
            If True, assume that the signal of the intermediate ESint is equal to the signal of the complex ES, default is True.
        fixed_t0 : bool, optional
            If True, fix the t0 parameter to 0, default is True.
            
        Returns
        -------
        None
            Updates the following instance attributes:
            - signal_assoc_fit: List of fitted association signals
            - fit_params_kinetics: DataFrame with the fitted parameters
        """

        # fit using as initial parameters the best found parameters
        initial_parameters = np.array(self.params_guess )
        low_bounds  = [x / 1e3 if x > 0 else x*1e3 for x in initial_parameters ]
        high_bounds = [x * 1e3 if x > 0 else x*1e-3 for x in initial_parameters ]

        # Set the bounds for the t0 parameter - between -0.1 and 0.1
        if not fixed_t0:
            n_params = len(initial_parameters)
            for i in range(len(self.assoc_lst)):
                low_bounds[n_params-i-1] = -0.1
                high_bounds[n_params-i-1] = 0.1

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

        kwargs = {
            'signal_lst': self.assoc_lst,
            'time_lst': self.time_assoc_lst,
            'ligand_conc_lst': self.lig_conc,
            'protein_conc_lst': self.prot_conc,
            'fit_signal_E': fit_signal_E,
            'fit_signal_S': fit_signal_S,
            'fit_signal_ES': fit_signal_ES,
            'ESint_equals_ES': ESint_equals_ES,
            'fixed_t0': fixed_t0
        }

        # Re-fit the parameters if they are close to the constraints
        global_fit_params, cov, fitted_values, low_bounds, high_bounds = re_fit(
            fit=global_fit_params,
            cov=cov,
            fit_vals=fitted_values,
            fit_fx=fit_induced_fit_solution,
            low_bounds=low_bounds,
            high_bounds=high_bounds,
            times=2,
            **kwargs
        )

        self.params  = global_fit_params
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds

        self.create_fitting_bounds_table()

        self.signal_assoc_fit = fitted_values

        # Create a DataFrame with the fitted parameters and assign it to fit_params_kinetics
        self.fit_params_kinetics = pd.DataFrame({
            'Protein [µM]': self.prot_conc,
            'Ligand [µM]':  self.lig_conc,
        })

        for i, param in enumerate(parameter_names):
            if "t0" not in param:
                self.fit_params_kinetics[param] = global_fit_params[i]

        # Include the t0 column in the dataframe if we fitted t0
        if not fixed_t0:
            n_t0 = len(self.assoc_lst)
            t0_params = global_fit_params[-n_t0:]
            self.fit_params_kinetics['t0'] = t0_params

        return None

    def find_initial_params_cs(self,
                               fit_signal_E=False,
                               E1_equals_E2=True,
                               fit_signal_S=False,
                               fit_signal_E2S=True,
                               fixed_t0=True
                               ):
        """
        Find optimal initial parameters for the conformational selection model.

        Heuristically finds the best initial parameters for the fit by exploring
        fixed values of kc and krev and fitting kon and koff (and the signal of the complex).

        Parameters
        ----------
        fit_signal_E : bool, optional
            If True, fit the signal of the free protein E, default is False.
        E1_equals_E2 : bool, optional
            If True, assume that the signal of the two conformations of the free protein E1
            and E2 are equal, default is True.
        fit_signal_S : bool, optional
            If True, fit the signal of the free ligand S, default is False.
        fit_signal_E2S : bool, optional
            If True, fit the signal of the complex E2S, default is True.
        fixed_t0 : bool, optional
            If True, fix the t0 parameter to 0, default is True.

        Returns
        -------
        None
            Updates the params_guess attribute with the best initial parameters.
        """

        self.clear_fittings()

        # Heuristically find the best initial parameters for the fit
        # We explore fixed values of kc and krev and fit kon and koff (and the signal of the complex)
        params_guess = find_initial_parameters_conformational_selection_solution(
            signal_lst=self.assoc_lst,
            time_lst=self.time_assoc_lst,
            ligand_conc_lst=self.lig_conc,
            protein_conc_lst=self.prot_conc,
            fit_signal_E=fit_signal_E,
            E1_equals_E2=E1_equals_E2,
            fit_signal_S=fit_signal_S,
            fit_signal_E2S=fit_signal_E2S,
            fixed_t0=fixed_t0)

        self.params_guess = params_guess

        print(f"Initial parameters found for the conformational selection model: {params_guess}")

        return None

    def fit_conformational_selection(
            self,
            fit_signal_E=False,
            E1_equals_E2=True,
            fit_signal_S=False,
            fit_signal_E2S=True,
            fixed_t0=True
            ):
        """
        Fit the association signals assuming an conformational selection mechanism.

        A_active <-> A_inactive

        A_active + B <-> AB

        This model accounts for conformational changes in the protein before ligand binding.

        Parameters
        ----------
        fit_signal_E : bool, optional
            If True, fit the signal of the free protein E, default is False.
        E1_equals_E2 : bool, optional
            If True, assume that the signal of the two conformations of the free protein E1
            and E2 are equal, default is True.
        fit_signal_S : bool, optional
            If True, fit the signal of the free ligand S, default is False.
        fit_signal_E2S : bool, optional
            If True, fit the signal of the complex E2S, default is True.
        fixed_t0 : bool, optional
            If True, fix the t0 parameter to 0, default is True.

        Returns
        -------
        None
            Updates the following instance attributes:
            - signal_assoc_fit: List of fitted association signals
            - fit_params_kinetics: DataFrame with the fitted parameters
        """

        # fit using as initial parameters the best found parameters
        initial_parameters = np.array(self.params_guess)
        low_bounds = [x / 1e3 if x > 0 else x * 1e3 for x in initial_parameters]
        high_bounds = [x * 1e3 if x > 0 else x * 1e-3 for x in initial_parameters]

        # Set the bounds for the t0 parameter - between -0.1 and 0.1
        if not fixed_t0:
            n_params = len(initial_parameters)
            for i in range(len(self.assoc_lst)):
                low_bounds[n_params - i - 1] = -0.1
                high_bounds[n_params - i - 1] = 0.1

        global_fit_params, cov, fitted_values, parameter_names = fit_conformational_selection_solution(
            signal_lst=self.assoc_lst,
            time_lst=self.time_assoc_lst,
            ligand_conc_lst=self.lig_conc,
            protein_conc_lst=self.prot_conc,
            initial_parameters=initial_parameters,
            low_bounds=low_bounds,
            high_bounds=high_bounds,
            fit_signal_E=fit_signal_E,
            E1_equals_E2=E1_equals_E2,
            fit_signal_S=fit_signal_S,
            fit_signal_E2S=fit_signal_E2S,
            fixed_t0=fixed_t0
        )

        kwargs = {
            'signal_lst': self.assoc_lst,
            'time_lst': self.time_assoc_lst,
            'ligand_conc_lst': self.lig_conc,
            'protein_conc_lst': self.prot_conc,
            'fit_signal_E': fit_signal_E,
            'E1_equals_E2' : E1_equals_E2,
            'fit_signal_S': fit_signal_S,
            'fit_signal_E2S': fit_signal_E2S,
            'fixed_t0': fixed_t0
        }

        # Re-fit the parameters if they are close to the constraints
        global_fit_params, cov, fitted_values, low_bounds, high_bounds = re_fit(
            fit=global_fit_params,
            cov=cov,
            fit_vals=fitted_values,
            fit_fx=fit_conformational_selection_solution,
            low_bounds=low_bounds,
            high_bounds=high_bounds,
            times=2,
            **kwargs
        )

        self.params = global_fit_params
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds

        self.create_fitting_bounds_table()

        self.signal_assoc_fit = fitted_values

        # Create a DataFrame with the fitted parameters and assign it to fit_params_kinetics
        self.fit_params_kinetics = pd.DataFrame({
            'Protein [µM]': self.prot_conc,
            'Ligand [µM]': self.lig_conc,
        })

        for i, param in enumerate(parameter_names):
            if "t0" not in param:
                self.fit_params_kinetics[param] = global_fit_params[i]

        # Include the t0 column in the dataframe if we fitted t0
        if not fixed_t0:
            n_t0 = len(self.assoc_lst)
            t0_params = global_fit_params[-n_t0:]
            self.fit_params_kinetics['t0'] = t0_params

        return None