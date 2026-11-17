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


@pytest.mark.parametrize('attribute', [
    'assoc_lst', 'disso_lst', 'time_assoc_lst', 'time_disso_lst',
    'lig_conc_lst', 'smax_id', 'is_single_cycle', 'trim_time',
])
def test_validation_rejects_changed_data_before_restore_or_save(fitter, attribute):
    fitter.fit_one_site_assoc_and_disso()
    fitter.save_fit_result('original')
    params = fitter.params.copy()
    fingerprint = fitter.fit_results['original']['fit_data_fingerprint']
    if attribute == 'is_single_cycle':
        fitter.is_single_cycle = True
    elif attribute == 'trim_time':
        fitter.trim_time += 1
    elif attribute in ('lig_conc_lst', 'smax_id'):
        getattr(fitter, attribute)[0] += 1
    else:
        getattr(fitter, attribute)[0][0] += 1

    for operation in (
        lambda: fitter.validate_fit_result('original'),
        lambda: fitter.restore_fit_result('original'),
        lambda: fitter.save_fit_result('stale'),
        fitter.create_fitting_bounds_table,
        fitter.compare_saved_fits,
    ):
        with pytest.raises(ValueError, match='Fitting data have changed'):
            operation()
    assert fitter.active_fit_result == ONE_SITE
    np.testing.assert_array_equal(fitter.params, params)
    assert 'stale' not in fitter.fit_results
    assert fitter.fit_results['original']['fit_data_fingerprint'] == fingerprint


def test_validation_accepts_equivalent_containers_and_recovered_data(fitter):
    fitter.fit_one_site_assoc_and_disso()
    fitter.assoc_lst = [trace.tolist() for trace in fitter.assoc_lst]
    fitter.lig_conc_lst = np.asarray(fitter.lig_conc_lst)
    fitter.names = ['renamed'] * 4
    assert fitter.validate_fit_result() is True
    assert fitter.validate_fit_result(ONE_SITE, shared_smax=True) is True
    original = fitter.assoc_lst[0][0]
    fitter.assoc_lst[0][0] += 1
    with pytest.raises(ValueError, match='Fitting data have changed'):
        fitter.restore_fit_result(ONE_SITE)
    fitter.assoc_lst[0][0] = original
    fitter.restore_fit_result(ONE_SITE)
    assert fitter.validate_fit_result() is True


def test_refitting_new_data_does_not_relabel_old_snapshot(fitter):
    fitter.fit_one_site_assoc_and_disso()
    fitter.save_fit_result('old_data')
    fingerprint = fitter.fit_results['old_data']['fit_data_fingerprint']
    fitter.assoc_lst[0] = fitter.assoc_lst[0] + 0.001
    fitter.fit_one_site_assoc_and_disso()
    assert fitter.validate_fit_result() is True
    assert fitter.fit_data_fingerprint != fingerprint
    assert fitter.fit_results['old_data']['fit_data_fingerprint'] == fingerprint
    with pytest.raises(ValueError, match='Fitting data have changed'):
        fitter.restore_fit_result('old_data')


def test_validation_rejects_legacy_results_without_fingerprints(fitter):
    with pytest.raises(ValueError, match='No active fit'):
        fitter.validate_fit_result()
    fitter.fit_one_site_dissociation()
    fitter.save_fit_result('legacy')
    del fitter.fit_results['legacy']['fit_data_fingerprint']
    with pytest.raises(ValueError, match='no data fingerprint'):
        fitter.restore_fit_result('legacy')


def test_comparison_scores_and_optional_bic_sorting_preserve_active_fit(fitter):
    import math

    fitter.fit_one_site_assoc_and_disso(shared_smax=True)
    fitter.save_fit_result('shared')
    fitter.fit_one_site_assoc_and_disso(shared_smax=False)
    fitter.save_fit_result('per_trace')
    active = fitter.active_fit_result
    params = fitter.params.copy()
    comparison = fitter.compare_saved_fits(['shared', 'per_trace'])
    assert list(comparison.index) == ['shared', 'per_trace']
    assert list(fitter.compare_saved_fits(['per_trace', 'shared']).index) == ['per_trace', 'shared']
    ranked = fitter.compare_saved_fits(['per_trace', 'shared'], sort_by='BIC')
    assert ranked['BIC'].is_monotonic_increasing
    pd.testing.assert_frame_equal(ranked, comparison.sort_values('BIC', kind='stable'))
    assert comparison['Parameter count'].to_dict() == {'shared': 3, 'per_trace': 6}
    for name in comparison.index:
        state = fitter.fit_results[name]
        observed = np.concatenate(fitter.assoc_lst + fitter.disso_lst)
        predicted = np.concatenate(state['signal_assoc_fit'] + state['signal_disso_fit'])
        rss = np.sum((observed - predicted) ** 2)
        n, k = len(observed), len(state['params'])
        expected_bic = n * math.log(rss / n) + k * math.log(n)
        assert comparison.loc[name, 'Reduced χ²'] == pytest.approx(rss / (n - k))
        assert comparison.loc[name, 'BIC'] == pytest.approx(expected_bic)
    assert fitter.active_fit_result == active
    np.testing.assert_array_equal(fitter.params, params)


def test_default_comparison_includes_each_configuration_once(fitter):
    fitter.fit_one_site_assoc_and_disso(shared_smax=True)
    fitter.save_fit_result('alias')
    fitter.fit_one_site_assoc_and_disso(shared_smax=False)
    comparison = fitter.compare_saved_fits()
    assert len(comparison) == 2
    assert set(comparison['Parameter count']) == {3, 6}
    comparison.iloc[0]['Fit settings']['shared_smax'] = 'changed'
    assert all(isinstance(state['fit_settings']['shared_smax'], bool)
               for state in fitter.fit_result_history[ONE_SITE])


def test_comparison_refuses_different_regions_or_time_limits(fitter):
    fitter.fit_one_site_assoc_and_disso()
    fitter.fit_one_site_dissociation(time_limit=100)
    fitter.save_fit_result('short')
    fitter.fit_one_site_dissociation(time_limit=200)
    fitter.save_fit_result('long')
    with pytest.raises(ValueError, match='different observations or time ranges'):
        fitter.compare_saved_fits([ONE_SITE, 'short'])
    with pytest.raises(ValueError, match='different observations or time ranges'):
        fitter.compare_saved_fits(['short', 'long'])
    state = fitter.fit_results['short']
    n = sum(len(time) for time in state['time_disso_lst_subset'])
    rss = sum(np.sum((observed[time < time.min() + 100] - predicted) ** 2)
              for observed, time, predicted in zip(
                  fitter.disso_lst, fitter.time_disso_lst, state['signal_disso_fit']))
    comparison = fitter.compare_saved_fits('short')
    assert comparison.loc['short', 'Reduced χ²'] == pytest.approx(rss / (n - 5))


def test_comparison_rejects_missing_or_invalid_predictions(fitter):
    fitter.fit_one_site_dissociation()
    fitter.save_fit_result('missing')
    fitter.fit_results['missing']['signal_disso_fit'][0] = None
    with pytest.raises(ValueError, match='missing predictions'):
        fitter.compare_saved_fits('missing')
    fitter.save_fit_result('invalid')
    fitter.fit_results['invalid']['signal_disso_fit'][0][0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        fitter.compare_saved_fits('invalid')


def test_empty_comparison_has_expected_columns(fitter):
    comparison = fitter.compare_saved_fits()
    assert comparison.empty
    assert comparison.columns.tolist() == [
        'Model', 'Fit settings', 'Parameter count', 'Reduced χ²', 'BIC',
    ]


@pytest.mark.parametrize('method, kwargs, parameter_count', [
    ('fit_single_exponentials', {}, 6),
    ('fit_double_exponentials', {'min_log_k': -3, 'max_log_k': 0, 'log_k_points': 8}, 10),
])
def test_solution_comparison_counts_offsets_amplitudes_and_rates(method, kwargs, parameter_count):
    t = np.linspace(0, 50, 80)
    concentrations = [0.5, 1.0]
    signals = [2 * (1 - np.exp(-0.05 * c * t)) for c in concentrations]
    if method == 'fit_double_exponentials':
        signals = [1 + 1.5 * np.exp(-0.02 * c * t) + 0.8 * np.exp(-0.2 * c * t)
                   for c in concentrations]
    solution = KineticsFitterSolution('sample', signals, concentrations, [1] * 2, [t] * 2)
    getattr(solution, method)(**kwargs)
    comparison = solution.compare_saved_fits()
    assert comparison['Parameter count'].iloc[0] == parameter_count
    rss = sum(np.sum((observed - predicted) ** 2)
              for observed, predicted in zip(solution.assoc_lst, solution.signal_assoc_fit))
    assert comparison['Reduced χ²'].iloc[0] == pytest.approx(rss / (160 - parameter_count))
    solution.prot_conc[0] = 2
    with pytest.raises(ValueError, match='Fitting data have changed'):
        solution.restore_fit_result(solution.active_fit_result)


def test_steady_state_comparison_uses_steady_state_points(fitter):
    fitter.names = ['sample']
    fitter.fit_steady_state_one_site()
    comparison = fitter.compare_saved_fits()
    observed = np.concatenate(fitter.signal_ss)
    predicted = np.concatenate(fitter.signal_ss_fit)
    rss = np.sum((observed - predicted) ** 2)
    assert comparison['Parameter count'].iloc[0] == 2
    assert comparison['Reduced χ²'].iloc[0] == pytest.approx(rss / (4 - 2))


def test_comparison_counts_fitted_time_offset_and_rejects_nonpositive_dof(fitter):
    fitter.fit_one_site_assoc_and_disso(fixed_t0=False)
    assert fitter.compare_saved_fits()['Parameter count'].iloc[0] == 4
    fitter.save_fit_result('no_degrees_of_freedom')
    state = fitter.fit_results['no_degrees_of_freedom']
    state['fit_parameter_count'] = 400
    with pytest.raises(ValueError, match='more observations than fitted parameters'):
        fitter.compare_saved_fits('no_degrees_of_freedom')


def test_comparison_handles_perfect_fit_without_runtime_warnings(fitter):
    fitter.fit_one_site_dissociation()
    fitter.save_fit_result('perfect')
    state = fitter.fit_results['perfect']
    state['signal_disso_fit'] = copy.deepcopy(fitter.disso_lst)
    with np.errstate(all='raise'):
        comparison = fitter.compare_saved_fits('perfect')
    assert comparison.loc['perfect', 'Reduced χ²'] == 0
    assert comparison.loc['perfect', 'BIC'] == -np.inf


def test_bic_sorting_preserves_ties_and_rejects_unknown_scores(fitter):
    fitter.fit_one_site_assoc_and_disso()
    fitter.save_fit_result('first')
    fitter.save_fit_result('second')
    assert list(fitter.compare_saved_fits(['second', 'first'], sort_by='BIC').index) == ['second', 'first']
    with pytest.raises(ValueError, match='sort_by must be'):
        fitter.compare_saved_fits(sort_by='unknown')


def test_trimmed_points_are_excluded_from_comparison(fitter):
    fitter.trim_time = 20
    fitter.fit_one_site_assoc_and_disso()
    rss, n = 0, 0
    for observed, times, predictions in (
        (fitter.assoc_lst, fitter.time_assoc_lst, fitter.signal_assoc_fit),
        (fitter.disso_lst, fitter.time_disso_lst, fitter.signal_disso_fit),
    ):
        for y, t, prediction in zip(observed, times, predictions):
            keep = t - t[0] >= fitter.trim_time
            rss += np.sum((y[keep] - prediction[keep]) ** 2)
            n += keep.sum()
    comparison = fitter.compare_saved_fits()
    assert comparison['Reduced χ²'].iloc[0] == pytest.approx(rss / (n - 3))
    assert comparison['BIC'].iloc[0] == pytest.approx(n * np.log(rss / n) + 3 * np.log(n))


def test_comparison_rejects_missing_parameter_count(fitter):
    fitter.fit_one_site_dissociation()
    fitter.save_fit_result('missing_count')
    del fitter.fit_results['missing_count']['fit_parameter_count']
    with pytest.raises(ValueError, match='has no parameter count'):
        fitter.compare_saved_fits('missing_count')


@pytest.mark.parametrize('corruption, message', [
    ('trace_count', 'do not match the observed traces'),
    ('point_count', 'do not match the fitted time range'),
    ('no_predictions', 'no fitted observations'),
])
def test_comparison_rejects_incomplete_prediction_layouts(fitter, corruption, message):
    fitter.fit_one_site_dissociation()
    fitter.save_fit_result('incomplete')
    state = fitter.fit_results['incomplete']
    if corruption == 'trace_count':
        state['signal_disso_fit'].pop()
    elif corruption == 'point_count':
        state['signal_disso_fit'][0] = state['signal_disso_fit'][0][:-1]
    else:
        state['signal_disso_fit'] = None
    with pytest.raises(ValueError, match=message):
        fitter.compare_saved_fits('incomplete')


def test_comparison_rejects_overflow_from_finite_predictions(fitter):
    fitter.fit_one_site_dissociation()
    fitter.save_fit_result('overflow')
    fitter.fit_results['overflow']['signal_disso_fit'] = [
        np.full_like(signal, 1e200) for signal in fitter.disso_lst
    ]
    with np.errstate(over='ignore'), pytest.raises(ValueError, match='finite, non-negative residual'):
        fitter.compare_saved_fits('overflow')
