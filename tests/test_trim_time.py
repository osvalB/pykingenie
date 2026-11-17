# This file is AI generated.
"""
Tests for trim_time: the first seconds of each association and dissociation are left
out of the kinetic fits, while the models still start at the beginning of each step.

The data are simulated with the one-to-one model (Kd 15 µM, koff 0.15 1/s, Smax 10).
Some traces carry a spike during the first second of each step, like the bulk
refractive index jump at the start and end of an SPR injection.
"""
import warnings

import numpy as np
import pytest

from pykingenie import BiacoreExperiment, KineticsAnalyzer, KineticsFitter
from pykingenie.utils.fitting_surface import (
    _concat_masks,
    fit_one_site_assoc_and_disso,
    trim_masks,
)
from pykingenie.utils.signal_surface import (
    one_site_association_analytical,
    one_site_dissociation_analytical,
)

KD, KOFF, SMAX = 15.0, 0.15, 10.0
CONCS = [0.78, 3.125, 12.5, 50.0]
T_ASSOC = np.arange(0, 120.05, 0.25)
T_DISSO = np.arange(120.25, 360.05, 0.25)
BLR_FILE = "./test_files/SPR_sensograms_with_Streptavidin_1.blr"


def simulate(spike=0.0, seed=1, noise=0.05):
    """Association and dissociation traces, with a spike of the given height (at 50 µM)."""
    rng = np.random.default_rng(seed)
    assoc, disso = [], []
    for conc in CONCS:
        y_assoc = one_site_association_analytical(T_ASSOC, 0, SMAX, KOFF, KD, conc, 0)
        y_disso = one_site_dissociation_analytical(T_DISSO - T_DISSO[0], y_assoc[-1], KOFF, 0)
        y_assoc = y_assoc + spike * conc / 50 * (T_ASSOC < 1)
        y_disso = y_disso - spike * conc / 50 * (T_DISSO - T_DISSO[0] < 1)
        assoc.append(y_assoc + rng.normal(0, noise, y_assoc.size))
        disso.append(y_disso + rng.normal(0, noise, y_disso.size))
    return assoc, disso


def fitter(assoc, disso, trim_time=0):
    n = len(CONCS)
    fit = KineticsFitter([T_ASSOC] * n, assoc, CONCS, [T_DISSO] * n, disso,
                         smax_id=[0] * n, name_lst=['sim'], trim_time=trim_time)
    fit.get_steady_state()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit.fit_steady_state()
    return fit


def test_trim_masks():
    assert trim_masks([np.arange(5)], 0) is None
    assert trim_masks(None, 1) is None
    # Counted from the first point of each step, also when the step starts later
    masks = trim_masks([np.array([0, 0.5, 1, 1.5]), np.array([100, 100.5, 101, 101.5])], 1)
    for mask in masks:
        assert mask.tolist() == [False, False, True, True]


def test_concat_masks():
    signals = [np.zeros(3), np.zeros(2)]
    assert _concat_masks([None], [signals]) is None
    mask = _concat_masks([[np.array([False, True, True]), np.array([True, True])], None],
                         [signals, [np.zeros(2)]])
    assert mask.tolist() == [False, True, True, True, True, True, True]
    with pytest.raises(ValueError, match="one value per data point"):
        _concat_masks([[np.ones(2, dtype=bool), np.ones(2, dtype=bool)]], [signals])
    with pytest.raises(ValueError, match="No data points"):
        _concat_masks([[np.zeros(3, dtype=bool), np.zeros(2, dtype=bool)]], [signals])


def test_negative_trim_time():
    assoc, disso = simulate()
    with pytest.raises(ValueError, match="trim_time"):
        fitter(assoc, disso, trim_time=-1)


def test_no_trim_is_unchanged():
    """trim_time=0, and masks that keep every point, give the fit without masks."""
    assoc, disso = simulate(spike=3)
    default = KineticsFitter([T_ASSOC] * 4, assoc, CONCS, [T_DISSO] * 4, disso,
                             smax_id=[0] * 4, name_lst=['sim'])
    default.get_steady_state()
    default.fit_steady_state()
    default.fit_one_site_assoc_and_disso()
    zero = fitter(assoc, disso, trim_time=0)
    zero.fit_one_site_assoc_and_disso()
    np.testing.assert_array_equal(default.params, zero.params)

    args = (assoc, [T_ASSOC] * 4, CONCS, disso, [T_DISSO] * 4,
            zero.p0, zero.low_bounds, zero.high_bounds)
    kwargs = dict(smax_idx=[0] * 4, shared_smax=True)
    plain = fit_one_site_assoc_and_disso(*args, **kwargs)[0]
    all_true = fit_one_site_assoc_and_disso(
        *args, **kwargs,
        assoc_mask_lst=[np.ones(T_ASSOC.size, dtype=bool)] * 4,
        disso_mask_lst=[np.ones(T_DISSO.size, dtype=bool)] * 4)[0]
    np.testing.assert_allclose(plain, all_true, rtol=1e-12)


def test_trimmed_points_are_ignored():
    """Whatever the first second holds, the trimmed fit is the same."""
    assoc, disso = simulate()
    garbage_assoc = [y.copy() for y in assoc]
    garbage_disso = [y.copy() for y in disso]
    for y in garbage_assoc:
        y[T_ASSOC < 1] = 1e3
    for y in garbage_disso:
        y[T_DISSO - T_DISSO[0] < 1] = -1e3

    clean = fitter(assoc, disso, trim_time=1)
    garbage = fitter(garbage_assoc, garbage_disso, trim_time=1)
    # The steady state uses the end of the association, so it is the same too
    assert clean.Kd_ss == pytest.approx(garbage.Kd_ss, rel=1e-12)
    clean.fit_one_site_assoc_and_disso()
    garbage.fit_one_site_assoc_and_disso()
    np.testing.assert_allclose(clean.params, garbage.params, rtol=1e-6)


def test_trim_removes_spike_bias():
    """A spike biases the fit and its confidence interval; trimming it recovers the truth."""
    assoc, disso = simulate(spike=3)

    biased = fitter(assoc, disso)
    biased.fit_one_site_assoc_and_disso()
    biased.calculate_ci95()
    assert abs(biased.k_off / KOFF - 1) > 0.02

    trimmed = fitter(assoc, disso, trim_time=1)
    trimmed.fit_one_site_assoc_and_disso()
    trimmed.calculate_ci95()
    assert trimmed.k_off == pytest.approx(KOFF, rel=0.01)
    assert trimmed.Kd == pytest.approx(KD, rel=0.01)

    # Without trimming, the 95% interval of koff misses the true value.
    # With trimming, the errors are within the interval widths.
    def width(fit, i):
        ci = fit.fit_params_kinetics_ci95.iloc[i]
        return ci['95% CI upper'] - ci['95% CI lower']

    ci_biased = biased.fit_params_kinetics_ci95.iloc[1]
    assert not ci_biased['95% CI lower'] <= KOFF <= ci_biased['95% CI upper']
    assert abs(trimmed.Kd - KD) < width(trimmed, 0)
    assert abs(trimmed.k_off - KOFF) < width(trimmed, 1)

    # The fitted curves still cover every point, from the start of each step
    assert [len(y) for y in trimmed.signal_assoc_fit] == [T_ASSOC.size] * 4
    assert [len(y) for y in trimmed.signal_disso_fit] == [T_DISSO.size] * 4
    assert trimmed.signal_assoc_fit[0][0] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize('trim_time', [1, 2, 5])
def test_trim_without_spike_is_unbiased(trim_time):
    """Leaving points out does not shift time zero: clean data give the same rates."""
    assoc, disso = simulate()
    fit = fitter(assoc, disso, trim_time=trim_time)
    fit.fit_one_site_assoc_and_disso()
    assert fit.k_off == pytest.approx(KOFF, rel=0.01)
    assert fit.Kd == pytest.approx(KD, rel=0.01)


@pytest.mark.parametrize('model', ['association', 'dissociation', 'mass_transport',
                                   'two_site', 'induced_fit'])
def test_trim_other_models(model):
    """Each model and its leave-one-out errors run on trimmed data."""
    assoc, disso = simulate(spike=3)
    fit = fitter(assoc, disso, trim_time=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if model == 'association':
            fit.fit_one_site_association(calc_loo=True)
        elif model == 'dissociation':
            fit.fit_one_site_dissociation(calc_loo=True)
        elif model == 'mass_transport':
            fit.fit_one_site_assoc_and_disso(fit_ktr=True, calc_loo=True)
        elif model == 'two_site':
            fit.fit_two_site_assoc_and_disso(calc_loo=True)
        else:
            fit.fit_one_site_if_assoc_and_disso(calc_loo=True)

    assert len(fit.k_off_loo) == len(CONCS)
    if model == 'induced_fit':
        # On one-to-one data koff, kc and krev are not identifiable. Check that the
        # curves fit the points that are kept to the noise level (0.05).
        masks = trim_masks(fit.time_assoc_lst, 1) + trim_masks(fit.time_disso_lst, 1)
        signals = fit.assoc_lst + fit.disso_lst
        fitted = fit.signal_assoc_fit + fit.signal_disso_fit
        residuals = np.concatenate([(y - f)[m] for y, f, m in zip(signals, fitted, masks)])
        assert np.sqrt(np.mean(residuals ** 2)) < 0.055
        assert np.all(np.isfinite(fit.k_off_loo))
        return
    if model != 'two_site':
        assert fit.k_off == pytest.approx(KOFF, rel=0.02)
    rtol = {'two_site': 0.2, 'mass_transport': 0.1}.get(model, 0.05)
    assert np.allclose(fit.k_off_loo, KOFF, rtol=rtol)


def test_single_exponentials_trimmed():
    assoc, disso = simulate(spike=3, noise=0)
    expected = [KOFF + KOFF / KD * c for c in CONCS]
    trimmed = fitter(assoc, disso, trim_time=1)
    trimmed.fit_single_exponentials()
    untrimmed = fitter(assoc, disso)
    untrimmed.fit_single_exponentials()
    err_trimmed = np.abs(np.array(trimmed.k_obs) / expected - 1)
    err_untrimmed = np.abs(np.array(untrimmed.k_obs) / expected - 1)
    assert err_trimmed.max() < 1e-3
    assert err_untrimmed.max() > 0.05


def test_generate_fittings_trim_time():
    exp = BiacoreExperiment('trim')
    exp.read_blr(BLR_FILE)
    exp.align_association(['Fc4-3'])

    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(exp, 'trim')
    analyzer.merge_ligand_conc_df()
    df = analyzer.combined_ligand_conc_df
    df = df[(df['Sensor'] == 'Fc4-3') & (df['[Analyte] (μM)'] > 0) & (df['[Analyte] (μM)'] < 50)]

    with pytest.raises(ValueError, match="trim_time"):
        analyzer.generate_fittings(df, trim_time=-0.5)

    analyzer.generate_fittings(df, trim_time=1.5)
    fit = list(analyzer.fittings.values())[0]
    assert fit.trim_time == 1.5
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        analyzer.submit_steady_state_fitting()
        analyzer.submit_kinetics_fitting()
    # Fitted curves cover the whole steps, trimmed points included
    assert [len(y) for y in fit.signal_assoc_fit] == [len(t) for t in fit.time_assoc_lst]
    assert np.isfinite(fit.k_off) and fit.k_off > 0

    analyzer.generate_fittings(df)
    assert list(analyzer.fittings.values())[0].trim_time == 0
