"""Shared base class for kinetics fitting result state and utilities."""

import copy
from functools import wraps
from inspect import signature

import numpy as np
import pandas as pd


def _same_fit_argument(left, right):
    """Compare scalar or array-valued fit arguments without ambiguous truth values."""
    try:
        return np.array_equal(left, right, equal_nan=True)
    except TypeError:
        return np.array_equal(left, right)


def _matching_fit_settings(settings, arguments):
    return all(name in settings and _same_fit_argument(settings[name], value)
               for name, value in arguments.items())


def store_fit_result(model, **variants):
    """Archive successful public fits, excluding internal fits for initial guesses.

    ``variants`` maps boolean option names to distinct model keys. Other fitting
    options are recorded in the snapshot's ``fit_settings`` dictionary.
    """
    def decorate(method):
        method_signature = signature(method)

        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if self._fit_in_progress:
                return method(self, *args, **kwargs)

            arguments = method_signature.bind(self, *args, **kwargs)
            arguments.apply_defaults()
            settings = {k: v for k, v in arguments.arguments.items() if k != 'self'}
            key = next((name for option, name in variants.items() if settings[option]), model)

            self._save_active_fit_result()
            previous = self._capture_fit_state()
            previous_key = self.active_fit_result
            self._fit_in_progress = True
            try:
                self._clear_active_fit_state()
                value = method(self, *args, **kwargs)
                self.fit_model = key
                self.fit_settings = copy.deepcopy(settings)
                if not hasattr(self, 'model'):
                    self.model = key
                self.active_fit_result = key
                self.save_fit_result()
                return value
            except Exception:
                self._restore_fit_state(previous)
                self.active_fit_result = previous_key
                raise
            finally:
                self._fit_in_progress = False

        return wrapped
    return decorate


def update_fit_result(method):
    """Keep the active snapshot current after optional diagnostics are added."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        value = method(self, *args, **kwargs)
        if not self._fit_in_progress:
            self._save_active_fit_result()
        return value
    return wrapped


class KineticsFitterGeneral:

    """
    A class used to handle kinetics data both for solution-based and surface-based experiments.
    """

    # Preparation state is retained when starting a new fit and included when
    # restoring one. The archive contains fitting state, not copies of the fitter.
    _fit_context_attributes = (
        'Kd_ss', 'Kd1_ss', 'Kd2_ss', 'fraction_site1_ss', 'sigma_ss',
        'Smax_upper_bound_factor', 'signal_ss', 'lig_conc_lst_per_id',
        'smax_guesses_unq', 'smax_guesses_shared', 'params_guess',
        'unq_prot_conc', 'signal_ss_per_protein', 'lig_conc_per_protein',
    )
    _fit_output_attributes = (
        'model', 'fit_model', 'fit_settings', 'signal_ss_fit', 'signal_assoc_fit',
        'signal_disso_fit', 'fit_params_kinetics', 'fit_params_ss',
        'fit_params_kinetics_correlation', 'fit_params_ss_correlation',
        'fit_params_kinetics_error', 'fit_params_kinetics_ci95',
        'fitted_params_boundaries', 'params', 'p0', 'low_bounds', 'high_bounds',
        'Kd', 'Kd1', 'Kd2', 'k_off', 'k_off1', 'k_off2', 'k_on', 'k_c', 'k_rev',
        'Smax', 'fraction_site1', 'signal_ratio', 'shared_smax', 'fixed_t0',
        'fit_ktr', 'fit_sigma', 'fit_signal_ratio', 'smax_param_start',
        'disso_lst_subset', 'time_disso_lst_subset',
        'heterogeneous_ligand_grid_search', 'best_Kd1_grid', 'best_Kd2_grid',
        'k_obs', 'k_obs_per_prot', 'k_obs_1', 'k_obs_2',
        'k_obs_1_err', 'k_obs_2_err', 'k_obs_1_per_prot', 'k_obs_2_per_prot',
        'df_fit_loo', 'ktr_loo_means', 'ktr_loo_stds',
    ) + tuple(
        f'{name}_loo{suffix}'
        for name in (
            'Kd', 'Kd1', 'Kd2', 'k_off', 'k_off1', 'k_off2', 'k_on', 'k_c',
            'k_rev', 'fraction_site1', 'sigma', 'signal_ratio',
        )
        for suffix in ('', '_mean', '_std')
    )

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
        fit_params_kinetics_correlation : pandas.DataFrame or None
            Correlation matrix of the fitted kinetics parameters.
        fit_params_ss_correlation : pandas.DataFrame or None
            Correlation matrix of the fitted steady-state parameters.
        fitted_params_boundaries : pandas.DataFrame or None
            DataFrame with fitting boundaries and fitted parameters.
        params : numpy.ndarray or None
            Parameters used for fitting.
        low_bounds : numpy.ndarray or None
            Lower bounds for fitting parameters.
        high_bounds : numpy.ndarray or None
            Upper bounds for fitting parameters.
        fit_results : dict of str to dict
            Latest snapshots keyed by model, plus explicitly named snapshots.
        fit_result_history : dict of str to list of dict
            Successful snapshots for each model and argument combination.
            Refitting the same configuration replaces that configuration only.
        active_fit_result : str or None
            Key of the result currently exposed through the fitting attributes.
        """
        # This function contains AI-generated changes (PR #19).

        self.signal_ss_fit = None  # Steady state fitted signal
        self.signal_assoc_fit = None  # Association kinetics fitted signal
        self.signal_disso_fit = None  # Dissociation kinetics fitted signal
        self.fit_params_kinetics = None  # Fitted parameters for the association / dissociation kinetics
        self.fit_params_ss = None  # Values of the fitted parameters - steady state
        self.fit_params_kinetics_correlation = None  # Correlation of the fitted kinetics parameters
        self.fit_params_ss_correlation = None  # Correlation of the fitted steady-state parameters
        self.fitted_params_boundaries = None  # DataFrame with fitting boundaries and fitted parameters
        self.params = None  # Parameters used for fitting
        self.low_bounds = None  # Lower bounds for fitting parameters
        self.high_bounds = None  # Upper bounds for fitting parameters
        self.fit_results = {}
        self.fit_result_history = {}
        self.active_fit_result = None
        self._fit_in_progress = False

    def _capture_fit_state(self):
        attributes = self._fit_output_attributes + self._fit_context_attributes
        return copy.deepcopy({name: getattr(self, name) for name in attributes
                              if hasattr(self, name)})

    def _restore_fit_state(self, state):
        for name in self._fit_output_attributes + self._fit_context_attributes:
            self.__dict__.pop(name, None)
        self.__dict__.update(copy.deepcopy(state))

    def _clear_active_fit_state(self):
        for name in self._fit_output_attributes:
            self.__dict__.pop(name, None)
        self.clear_fittings()

    def _save_active_fit_result(self):
        if self.active_fit_result is not None:
            self.save_fit_result()

    def save_fit_result(self, name=None, **fit_arguments):
        """Save an independent snapshot of the active fit and return its key.

        Parameters
        ----------
        name : str, optional
            Custom key, for example 'one_site_shared_smax'. Defaults to the
            active result's key. Existing entries at that key are replaced.
        **fit_arguments
            Arguments used for the active fit, for example shared_smax=True,
            fixed_t0=False or time_limit=100. Supplied values are checked
            against the fitted configuration. All actual arguments, including
            defaults, are saved even when none are supplied here. A mismatch
            raises ValueError; unknown argument names raise TypeError.

        Notes
        -----
        Snapshots contain fitted signals, tables, errors, correlations, bounds,
        initial guesses, settings and any calculated LOO/CI diagnostics. They
        live in memory on this fitter; this does not write a file.

        Each distinct argument combination is retained in fit_result_history.
        Use restore_fit_result(model, **fit_arguments) to select a configuration
        or restore_fit_result(name) to load an explicitly named version.
        """
        if self.active_fit_result is None:
            raise ValueError('No active fit result to save. Run a fit first.')
        key = self.active_fit_result if name is None else name
        if not isinstance(key, str) or not key:
            raise ValueError('The fit result name must be a non-empty string.')
        self._validate_fit_arguments(self.fit_settings, fit_arguments)
        snapshot = self._capture_fit_state()
        history = self.fit_result_history.setdefault(self.fit_model, [])
        for i, saved in enumerate(history):
            if _matching_fit_settings(saved['fit_settings'], self.fit_settings):
                history[i] = copy.deepcopy(snapshot)
                break
        else:
            history.append(copy.deepcopy(snapshot))
        self.fit_results[key] = snapshot
        return key

    @staticmethod
    def _validate_fit_arguments(settings, arguments):
        for name, value in arguments.items():
            if name not in settings:
                raise TypeError(f'Unknown fit argument: {name!r}')
            if not _same_fit_argument(settings[name], value):
                raise ValueError(
                    f'{name}={value!r} differs from the fitted value '
                    f'{settings[name]!r}. Run a new fit to change its arguments.')

    def restore_fit_result(self, name, **fit_arguments):
        """Restore a saved result for plotting, export or further diagnostics.

        Switching to another result saves changes to the current one first.
        Restoring the active result discards its unsaved edits. Restored arrays
        and tables are copied, so editing them cannot change the saved snapshot.
        Raises KeyError if ``name`` has not been saved on this fitter.

        Parameters
        ----------
        name : str
            Model key or explicitly saved name.
        **fit_arguments
            Select a saved configuration of this model, for example
            shared_smax=True and fixed_t0=False. Unspecified arguments may be
            omitted if the selection is unique. Raises ValueError when no
            configuration matches or the selection is ambiguous. With no
            arguments, restores the latest snapshot at the given key.
        """
        if name not in self.fit_results and name not in self.fit_result_history:
            raise KeyError(f'Unknown fit result: {name!r}')

        if fit_arguments and name in self.fit_result_history:
            history = self.fit_result_history[name]
            for argument in fit_arguments:
                if argument not in history[-1]['fit_settings']:
                    raise TypeError(f'Unknown fit argument: {argument!r}')
            matches = [saved for saved in history
                       if _matching_fit_settings(saved['fit_settings'], fit_arguments)]
            if not matches:
                raise ValueError(f'No saved fit for {name!r} matches {fit_arguments!r}.')
            if len(matches) > 1:
                raise ValueError(
                    f'Multiple saved fits for {name!r} match {fit_arguments!r}. '
                    'Specify more fitting arguments or restore a named result.')
            state = copy.deepcopy(matches[0])
        else:
            state = copy.deepcopy(self.fit_results[name])
            self._validate_fit_arguments(state['fit_settings'], fit_arguments)

        if name != self.active_fit_result or fit_arguments:
            self._save_active_fit_result()
        self._restore_fit_state(state)
        self.active_fit_result = name
        if fit_arguments:
            self.fit_results[name] = copy.deepcopy(state)

    def clear_fittings(self):
        """
        Clear active fitting results, retaining the saved model snapshots.

        Resets all fitted signal arrays and parameter dataframes to None.
        Outside a fit, saves the active result before clearing it.

        Returns
        -------
        None
        """
        # This function contains AI-generated changes (PR #19).
        if not self._fit_in_progress:
            self._save_active_fit_result()
            self.active_fit_result = None
            for name in self._fit_output_attributes:
                self.__dict__.pop(name, None)
        self.signal_ss_fit = None  # Steady state fitted signal
        self.signal_assoc_fit = None  # Association kinetics fitted signal
        self.signal_disso_fit = None  # Association kinetics fitted signal
        self.fit_params_kinetics = None  # Fitted parameters for the association / dissociation kinetics
        self.fit_params_ss = None  # Values of the fitted parameters - steady state
        self.fit_params_kinetics_correlation = None
        self.fit_params_ss_correlation = None
        self.fit_params_kinetics_error = None
        self.fit_params_kinetics_ci95 = None
        self.fitted_params_boundaries = None
        self.params = None
        self.low_bounds = None
        self.high_bounds = None

        return None

    @update_fit_result
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
