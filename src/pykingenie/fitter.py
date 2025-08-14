import pandas as pd

class KineticsFitterGeneral:

    """
    A class used to handle kinetics data both for solution-based and surface-based experiments.
    """

    def __init__(self):
        """
        Initialize the KineticsFitterGeneral instance.

        Attributes
        ----------
        signal_ss_fit : list of numpy.ndarray or None
            Fitted steady state signals.
        signal_assoc_fit : list of numpy.ndarray or None
            Fitted association kinetics signals.
        signal_disso_fit : list of numpy.ndarray or None
            Fitted dissociation kinetics signals.
        fit_params_kinetics : pandas.DataFrame or None
            Fitted parameters for the association / dissociation kinetics.
        fit_params_ss : pandas.DataFrame or None
            Fitted parameters for the steady state.
        fitted_params_boundaries : pandas.DataFrame or None
            DataFrame with fitting boundaries and fitted parameters.
        params : numpy.ndarray or None
            Parameters used for fitting.
        low_bounds : numpy.ndarray or None
            Lower bounds for fitting parameters.
        high_bounds : numpy.ndarray or None
            Upper bounds for fitting parameters.
        """

        self.signal_ss_fit = None  # Steady state fitted signal
        self.signal_assoc_fit = None  # Association kinetics fitted signal
        self.signal_disso_fit = None  # Dissociation kinetics fitted signal
        self.fit_params_kinetics = None  # Fitted parameters for the association / dissociation kinetics
        self.fit_params_ss = None  # Values of the fitted parameters - steady state
        self.fitted_params_boundaries = None  # DataFrame with fitting boundaries and fitted parameters
        self.params = None  # Parameters used for fitting
        self.low_bounds = None  # Lower bounds for fitting parameters
        self.high_bounds = None  # Upper bounds for fitting parameters

    def clear_fittings(self):
        """
        Clear all fitting results.

        Resets all fitted signal arrays and parameter dataframes to None.

        Returns
        -------
        None
        """
        self.signal_ss_fit = None  # Steady state fitted signal
        self.signal_assoc_fit = None  # Association kinetics fitted signal
        self.signal_disso_fit = None  # Association kinetics fitted signal
        self.fit_params_kinetics = None  # Fitted parameters for the association / dissociation kinetics
        self.fit_params_ss = None  # Values of the fitted parameters - steady state

        return None

    def create_fitting_bounds_table(self):
        """
        Create a dataframe with the fitting bounds and the fitted parameters.

        Uses the class attributes params, low_bounds and high_bounds to create
        a DataFrame with fitting boundaries.

        Returns
        -------
        None
            Updates the fitted_params_boundaries attribute with a DataFrame.
        """
        df = pd.DataFrame({
            'Fitted_parameter_value': self.params,
            'Lower_limit_for_fitting': self.low_bounds,
            'Upper_limit_for_fitting': self.high_bounds
        })

        self.fitted_params_boundaries = df

        return None