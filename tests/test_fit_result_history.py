"""Model changes retain independent results on each surface/solution fitter."""

import copy

import numpy as np
import pandas as pd
import pytest

import pykingenie.fitter_surface as surface_module
from pykingenie.fitter_solution import KineticsFitterSolution
from pykingenie.fitter_surface import KineticsFitter
from pykingenie.main import KineticsAnalyzer
from pykingenie.utils.signal_surface import one_site_association_analytical


ONE_SITE = 'one_site_association_and_dissociation'


@pytest.fixture
def fitter():
    rng = np.random.default_rng(42)
    ta = np.linspace(0, 300, 50)
    td = np.linspace(300, 600, 50)
    concentrations = [0.2, 0.5, 1.0, 2.0]
    assoc, disso = [], []
    for concentration in concentrations:
        signal = one_site_association_analytical(ta, 0, 5, 0.01, 0.5, concentration)
        assoc.append(signal + rng.normal(0, 0.001, len(ta)))
        disso.append(signal[-1] * np.exp(-0.01 * (td - td[0])) +
                     rng.normal(0, 0.001, len(td)))
    fit = KineticsFitter([ta] * 4, assoc, concentrations, [td] * 4, disso,
                         name_lst=['sample'] * 4)
    fit.get_steady_state()
    fit.Kd_ss = 0.5
    return fit


def test_model_switch_and_restore_for_export(fitter):
    fitter.fit_one_site_assoc_and_disso()
    expected_params = fitter.params.copy()
    expected_table = fitter.fit_params_kinetics.copy(deep=True)
    expected_export = fitter.create_export_df(type='fit')
    fitter.create_fitting_bounds_table()
    expected_bounds = fitter.fitted_params_boundaries.copy(deep=True)

    fitter.fit_one_site_dissociation(time_limit=100)
    assert set(fitter.fit_results) == {ONE_SITE, 'one_site_dissociation'}
    assert fitter.active_fit_result == 'one_site_dissociation'
    assert fitter.fit_results[ONE_SITE]['fit_settings']['shared_smax'] is True
    pd.testing.assert_frame_equal(fitter.fit_results[ONE_SITE]['fit_params_kinetics'],
                                  expected_table)

    fitter.restore_fit_result(ONE_SITE)
    assert fitter.active_fit_result == ONE_SITE
    assert fitter.model == ONE_SITE
    np.testing.assert_array_equal(fitter.params, expected_params)
    pd.testing.assert_frame_equal(fitter.fit_params_kinetics, expected_table)
    pd.testing.assert_frame_equal(fitter.create_export_df(type='fit'), expected_export)
    pd.testing.assert_frame_equal(fitter.fitted_params_boundaries, expected_bounds)

    # Restoring the dissociation-only model removes the previous model's rates.
    fitter.restore_fit_result('one_site_dissociation')
    assert not hasattr(fitter, 'Kd')
    assert fitter.signal_assoc_fit is None
    assert fitter.fit_settings['time_limit'] == 100
    assert len(fitter.time_disso_lst_subset[0]) < len(fitter.time_disso_lst[0])


def test_saved_arrays_tables_and_settings_are_independent(fitter):
    fitter.fit_one_site_assoc_and_disso()
    saved = copy.deepcopy(fitter.fit_results[ONE_SITE])
    fitter.params[0] = 99
    fitter.signal_assoc_fit[0][0] = 99
    fitter.fit_params_kinetics.iloc[0, 0] = 99
    fitter.fit_settings['shared_smax'] = False
    np.testing.assert_array_equal(fitter.fit_results[ONE_SITE]['params'], saved['params'])
    pd.testing.assert_frame_equal(fitter.fit_results[ONE_SITE]['fit_params_kinetics'],
                                  saved['fit_params_kinetics'])
    assert fitter.fit_results[ONE_SITE]['fit_settings']['shared_smax'] is True

    fitter.restore_fit_result(ONE_SITE)
    np.testing.assert_array_equal(fitter.params, saved['params'])
    fitter.signal_assoc_fit[0][0] = -99
    np.testing.assert_array_equal(fitter.fit_results[ONE_SITE]['signal_assoc_fit'][0],
                                  saved['signal_assoc_fit'][0])


def test_named_versions_and_repeated_fits(fitter):
    fitter.fit_one_site_assoc_and_disso(shared_smax=True)
    fitter.save_fit_result('shared_smax')
    saved = fitter.params.copy()
    fitter.fit_one_site_assoc_and_disso(shared_smax=False)
    assert set(fitter.fit_results) == {ONE_SITE, 'shared_smax'}
    assert fitter.fit_results[ONE_SITE]['fit_settings']['shared_smax'] is False
    assert len(fitter.fit_results[ONE_SITE]['params']) > len(saved)
    np.testing.assert_array_equal(fitter.fit_results['shared_smax']['params'], saved)
    fitter.restore_fit_result('shared_smax')
    assert fitter.shared_smax is True
    assert fitter.fit_settings['shared_smax'] is True


def test_failed_fit_preserves_history_and_active_result(fitter, monkeypatch):
    fitter.fit_one_site_assoc_and_disso()
    saved = copy.deepcopy(fitter.fit_results[ONE_SITE])

    def fail(*args, **kwargs):
        raise RuntimeError('optimizer failed')

    monkeypatch.setattr(surface_module, 'fit_one_site_dissociation', fail)
    with pytest.raises(RuntimeError, match='optimizer failed'):
        fitter.fit_one_site_dissociation()
    assert set(fitter.fit_results) == {ONE_SITE}
    assert fitter.active_fit_result == ONE_SITE
    assert fitter.model == ONE_SITE
    np.testing.assert_array_equal(fitter.params, saved['params'])
    pd.testing.assert_frame_equal(fitter.fit_params_kinetics, saved['fit_params_kinetics'])
    assert fitter._fit_in_progress is False

    # The same guarantee applies to a failed refit of an existing key.
    with pytest.raises(RuntimeError, match='optimizer failed'):
        fitter.fit_one_site_assoc_and_disso(shared_smax=False)
    np.testing.assert_array_equal(fitter.fit_results[ONE_SITE]['params'], saved['params'])
    assert fitter.shared_smax is True


def test_initial_guess_fit_does_not_overwrite_explicit_dissociation(fitter):
    fitter.fit_one_site_dissociation(time_limit=60)
    saved = copy.deepcopy(fitter.fit_results['one_site_dissociation'])
    fitter.fit_one_site_assoc_and_disso()
    assert fitter.fit_results['one_site_dissociation']['fit_settings']['time_limit'] == 60
    np.testing.assert_array_equal(fitter.fit_results['one_site_dissociation']['params'],
                                  saved['params'])


def test_failed_complex_model_rolls_back_after_initial_guess_fit(fitter, monkeypatch):
    fitter.fit_one_site_assoc_and_disso(shared_smax=False)
    saved = copy.deepcopy(fitter.fit_results[ONE_SITE])

    def fail(*args, **kwargs):
        raise RuntimeError('two-site optimizer failed')

    monkeypatch.setattr(surface_module, 'fit_two_site_assoc_and_disso', fail)
    with pytest.raises(RuntimeError, match='two-site optimizer failed'):
        fitter.fit_two_site_assoc_and_disso(shared_smax=True)
    assert set(fitter.fit_results) == {ONE_SITE}
    assert fitter.active_fit_result == ONE_SITE
    assert fitter.shared_smax is False
    np.testing.assert_array_equal(fitter.params, saved['params'])
    np.testing.assert_array_equal(fitter.fit_results[ONE_SITE]['params'], saved['params'])


def test_steady_state_results_survive_kinetics_fit(fitter):
    # The steady-state table takes one name per Smax group.
    fitter.names = ['sample']
    fitter.fit_steady_state()
    expected = fitter.fit_params_ss.copy(deep=True)
    expected_signals = copy.deepcopy(fitter.signal_ss_fit)
    fitter.fit_one_site_assoc_and_disso()
    assert set(fitter.fit_results) == {'steady_state_one_site', ONE_SITE}
    fitter.restore_fit_result('steady_state_one_site')
    pd.testing.assert_frame_equal(fitter.fit_params_ss, expected)
    np.testing.assert_array_equal(fitter.signal_ss_fit, expected_signals)
    assert fitter.signal_assoc_fit is None
    assert fitter.fit_params_kinetics is None


def test_mass_transport_has_its_own_key(fitter, monkeypatch):
    fitter.fit_one_site_assoc_and_disso()
    saved = fitter.params.copy()

    # Exercise the fit entry point/key without running the expensive MTL search.
    def mass_transport(*args, **kwargs):
        values = np.array([0.5, 0.01, 0.05, 5.0])
        return values, np.eye(4), fitter.assoc_lst, fitter.disso_lst

    monkeypatch.setattr(surface_module, 'fit_one_site_assoc_and_disso_ktr', mass_transport)
    monkeypatch.setattr(fitter, '_mtl_start_candidates', lambda *args: [])
    fitter.fit_one_site_assoc_and_disso(fit_ktr=True)
    assert set(fitter.fit_results) == {ONE_SITE, 'one_site_mass_transport'}
    assert fitter.fit_results['one_site_mass_transport']['fit_settings']['fit_ktr'] is True
    np.testing.assert_array_equal(fitter.fit_results[ONE_SITE]['params'], saved)
    fitter.restore_fit_result(ONE_SITE)
    assert fitter.fit_ktr is False


def test_diagnostics_are_saved_and_do_not_leak_to_next_model(fitter):
    fitter.fit_one_site_assoc_and_disso()
    fitter.calc_leave_one_out_errors()
    fitter.calculate_ci95()
    saved = fitter.fit_results[ONE_SITE]
    pd.testing.assert_frame_equal(saved['df_fit_loo'], fitter.df_fit_loo)
    pd.testing.assert_frame_equal(saved['fit_params_kinetics_ci95'],
                                  fitter.fit_params_kinetics_ci95)

    fitter.fit_one_site_dissociation()
    assert 'df_fit_loo' not in fitter.fit_results['one_site_dissociation']
    assert fitter.fit_params_kinetics_ci95 is None
    fitter.restore_fit_result(ONE_SITE)
    pd.testing.assert_frame_equal(fitter.df_fit_loo, saved['df_fit_loo'])


def test_clear_keeps_history_and_missing_restore_does_not_modify_active(fitter):
    with pytest.raises(ValueError, match='No active fit'):
        fitter.save_fit_result('empty')
    fitter.fit_one_site_assoc_and_disso()
    with pytest.raises(KeyError, match='Unknown fit result'):
        fitter.restore_fit_result('missing')
    assert fitter.active_fit_result == ONE_SITE
    with pytest.raises(ValueError, match='non-empty string'):
        fitter.save_fit_result('')
    fitter.clear_fittings()
    assert fitter.active_fit_result is None
    assert fitter.fit_params_kinetics is None
    assert fitter.params is None
    assert ONE_SITE in fitter.fit_results
    fitter.restore_fit_result(ONE_SITE)
    assert fitter.fit_params_kinetics is not None


def test_analyzer_updates_archived_bounds_and_kobs(fitter):
    analyzer = KineticsAnalyzer()
    analyzer.add_fitting(fitter, 'sample')
    good, bad = analyzer.submit_kinetics_fitting(shared_smax=True)
    assert good == [fitter]
    assert bad == []
    saved = fitter.fit_results[ONE_SITE]
    pd.testing.assert_frame_equal(saved['fitted_params_boundaries'],
                                  fitter.fitted_params_boundaries)
    np.testing.assert_array_equal(saved['k_obs'], fitter.k_obs)
    analyzer.submit_kinetics_fitting(fitting_region='dissociation')
    assert set(fitter.fit_results) == {ONE_SITE, 'one_site_dissociation'}


def test_solution_models_preserve_results_on_the_same_fitter():
    t = np.linspace(0, 50, 80)
    concentrations = [0.5, 1.0, 2.0]
    signals = [2 * (1 - np.exp(-0.05 * c * t)) for c in concentrations]
    fitter = KineticsFitterSolution('sample', signals, concentrations, [1] * 3, [t] * 3)
    analyzer = KineticsAnalyzer()
    analyzer.add_fitting(fitter, 'sample')
    analyzer.submit_fitting_solution('single')
    expected = fitter.fit_params_kinetics.copy(deep=True)
    expected_signal = copy.deepcopy(fitter.signal_assoc_fit)
    fitter.fit_double_exponentials(log_k_points=6)
    assert set(fitter.fit_results) == {'single_exponential', 'double_exponential'}
    assert fitter.fit_results['double_exponential']['fit_settings']['log_k_points'] == 6
    fitter.restore_fit_result('single_exponential')
    pd.testing.assert_frame_equal(fitter.fit_params_kinetics, expected)
    np.testing.assert_array_equal(fitter.signal_assoc_fit, expected_signal)
    assert not hasattr(fitter, 'k_obs_1')


def test_histories_belong_to_each_fitter(fitter):
    fitter.fit_one_site_dissociation()
    other = KineticsFitter(fitter.time_assoc_lst, fitter.assoc_lst, fitter.lig_conc_lst)
    assert other.fit_results == {}
    assert other.fit_result_history == {}
    assert other.active_fit_result is None


def test_save_records_and_checks_explicit_fitting_arguments(fitter):
    fitter.fit_one_site_assoc_and_disso(shared_smax=False, fixed_t0=True)
    key = fitter.save_fit_result('per_trace', shared_smax=False, fixed_t0=True)
    assert key == 'per_trace'
    assert fitter.fit_results[key]['fit_settings'] == {
        'shared_smax': False, 'fixed_t0': True, 'fit_ktr': False, 'calc_loo': False,
    }
    assert fitter.fit_results[key]['shared_smax'] is False
    assert fitter.fit_results[key]['fit_model'] == ONE_SITE

    with pytest.raises(ValueError, match='differs from the fitted value'):
        fitter.save_fit_result('wrong_settings', shared_smax=True)
    with pytest.raises(TypeError, match='Unknown fit argument'):
        fitter.save_fit_result('typo', shared_smx=False)
    assert 'wrong_settings' not in fitter.fit_results
    assert 'typo' not in fitter.fit_results
    assert len(fitter.fit_result_history[ONE_SITE]) == 1


def test_restore_distinguishes_shared_and_per_trace_smax_without_custom_names(fitter):
    fitter.fit_one_site_assoc_and_disso(shared_smax=True)
    shared = fitter.params.copy()
    fitter.fit_one_site_assoc_and_disso(shared_smax=False)
    per_trace = fitter.params.copy()
    assert len(fitter.fit_result_history[ONE_SITE]) == 2

    fitter.restore_fit_result(ONE_SITE, shared_smax=True)
    assert fitter.shared_smax is True
    np.testing.assert_array_equal(fitter.params, shared)
    assert fitter.fit_results[ONE_SITE]['fit_settings']['shared_smax'] is True
    fitter.restore_fit_result(ONE_SITE, shared_smax=False)
    assert fitter.shared_smax is False
    np.testing.assert_array_equal(fitter.params, per_trace)
    assert len(fitter.fit_result_history[ONE_SITE]) == 2


def test_restore_selects_time_limit_and_its_fitted_subset(fitter):
    fitter.fit_one_site_dissociation(time_limit=60)
    params = fitter.params.copy()
    subset = copy.deepcopy(fitter.time_disso_lst_subset)
    fitter.fit_one_site_dissociation(time_limit=200)
    fitter.restore_fit_result('one_site_dissociation', time_limit=60)
    assert fitter.fit_settings['time_limit'] == 60
    np.testing.assert_array_equal(fitter.params, params)
    np.testing.assert_array_equal(fitter.time_disso_lst_subset, subset)


def test_restore_requires_unique_matching_arguments(fitter):
    fitter.fit_one_site_assoc_and_disso(shared_smax=True, fixed_t0=True)
    fixed = fitter.params.copy()
    fitter.fit_one_site_assoc_and_disso(shared_smax=True, fixed_t0=False)
    active = fitter.params.copy()

    with pytest.raises(ValueError, match='Multiple saved fits'):
        fitter.restore_fit_result(ONE_SITE, shared_smax=True)
    with pytest.raises(ValueError, match='No saved fit'):
        fitter.restore_fit_result(ONE_SITE, shared_smax=False)
    with pytest.raises(TypeError, match='Unknown fit argument'):
        fitter.restore_fit_result(ONE_SITE, shared_smx=True)
    np.testing.assert_array_equal(fitter.params, active)

    fitter.restore_fit_result(ONE_SITE, shared_smax=True, fixed_t0=True)
    assert fitter.fixed_t0 is True
    np.testing.assert_array_equal(fitter.params, fixed)


def test_named_result_arguments_are_checked_without_selecting_another_fit(fitter):
    fitter.fit_one_site_assoc_and_disso(shared_smax=True)
    fitter.save_fit_result('shared', shared_smax=True)
    fitter.fit_one_site_assoc_and_disso(shared_smax=False)
    with pytest.raises(ValueError, match='differs from the fitted value'):
        fitter.restore_fit_result('shared', shared_smax=False)
    assert fitter.shared_smax is False
    fitter.restore_fit_result('shared', shared_smax=True)
    assert fitter.shared_smax is True


def test_refit_replaces_only_its_configuration_and_clear_retains_all(fitter):
    fitter.fit_one_site_assoc_and_disso(shared_smax=True)
    shared = fitter.params.copy()
    fitter.fit_one_site_assoc_and_disso(shared_smax=False)
    fitter.fit_one_site_assoc_and_disso(shared_smax=False)
    assert len(fitter.fit_result_history[ONE_SITE]) == 2
    fitter.clear_fittings()
    fitter.restore_fit_result(ONE_SITE, shared_smax=True)
    np.testing.assert_array_equal(fitter.params, shared)
    assert len(fitter.fit_result_history[ONE_SITE]) == 2
