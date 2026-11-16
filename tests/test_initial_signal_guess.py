# This code is AI generated.
"""
The one-to-one fit guesses the initial signal (single-exponential fits) only for
association traces that are not continuous with the previous dissociation, and the
confidence intervals compute that guess once instead of at every refit.
"""
import numpy as np
import pytest

import pykingenie.utils.fitting_surface as fitting_surface
from pykingenie.utils.fitting_surface import (
    fit_one_site_assoc_and_disso,
    guess_initial_signal,
    initial_signal_guesses_where_needed,
    one_site_assoc_and_disso_asymmetric_ci95,
    one_site_assoc_and_disso_asymmetric_ci95_koff,
)
from pykingenie.utils.math import get_desired_rss
from pykingenie.utils.signal_surface import (
    one_site_association_analytical,
    one_site_dissociation_analytical,
)

KD, KOFF, SMAX = 0.5, 0.01, 5.0
CONCS = [0.1, 0.3, 1.0, 3.0]


def _cycles(gap, s0_assoc=0.4, noise=0.02):
    """Association and dissociation traces on one time axis. Each association starts
    `gap` seconds after the previous dissociation (and at t = gap for the first one),
    so with a gap above the 3 s tolerance no trace is continuous. Continuous traces
    start from zero (the first) or from the end of the previous dissociation."""
    rng = np.random.default_rng(1)
    t_a, y_a, t_d, y_d = [], [], [], []
    start, s0 = gap, 0.0
    for conc in CONCS:
        ta = start + np.arange(0, 300, 1.0)
        if gap >= 3:
            s0 = s0_assoc
        ya = one_site_association_analytical(ta - ta[0], s0, SMAX, KOFF, KD, conc)
        td = ta[-1] + 1 + np.arange(0, 400, 1.0)
        yd = one_site_dissociation_analytical(td - td[0], ya[-1], KOFF)
        t_a.append(ta); y_a.append(ya + rng.normal(0, noise, len(ta)))
        t_d.append(td); y_d.append(yd + rng.normal(0, noise, len(td)))
        start, s0 = td[-1] + gap, yd[-1]
    return t_a, y_a, t_d, y_d


def _fit(t_a, y_a, t_d, y_d, **kwargs):
    p0 = [1.0, 0.02, 4.0]
    return fit_one_site_assoc_and_disso(y_a, t_a, CONCS, y_d, t_d, p0, [1e-3, 1e-4, 1.0], [1e3, 1.0, 50.0],
                                        smax_idx=[0] * len(CONCS), shared_smax=True, **kwargs)


def _forbid_guess(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("guess_initial_signal should not be called")
    monkeypatch.setattr(fitting_surface, "guess_initial_signal", fail)


def test_guesses_only_for_traces_that_are_not_continuous():
    t_a, y_a, t_d, _ = _cycles(gap=10)
    # Make the third association continuous with the second dissociation
    t_a[2] = t_a[2] - (t_a[2][0] - t_d[1][-1]) + 1

    guesses = initial_signal_guesses_where_needed(t_a, y_a, t_d)
    expected = guess_initial_signal(t_a, y_a)

    assert guesses[2] is None
    for i in (0, 1, 3):
        assert guesses[i] == expected[i]


def test_continuous_data_does_not_guess(monkeypatch):
    t_a, y_a, t_d, y_d = _cycles(gap=1)
    assert initial_signal_guesses_where_needed(t_a, y_a, t_d) == [None] * len(CONCS)

    _forbid_guess(monkeypatch)
    params, _, _, _ = _fit(t_a, y_a, t_d, y_d)
    assert params[0] == pytest.approx(KD, rel=0.05)


def test_fit_is_unchanged_for_traces_that_are_not_continuous():
    """Same result as before, when the guess was computed for every trace."""
    t_a, y_a, t_d, y_d = _cycles(gap=10)
    params, _, fit_a, fit_d = _fit(t_a, y_a, t_d, y_d)
    params_all, _, fit_a_all, fit_d_all = _fit(t_a, y_a, t_d, y_d, s0_guesses=guess_initial_signal(t_a, y_a))

    np.testing.assert_array_equal(params, params_all)
    np.testing.assert_array_equal(np.concatenate(fit_a + fit_d), np.concatenate(fit_a_all + fit_d_all))
    assert params[0] == pytest.approx(KD, rel=0.05)


def test_given_guesses_are_used(monkeypatch):
    t_a, y_a, t_d, y_d = _cycles(gap=10)
    guesses = initial_signal_guesses_where_needed(t_a, y_a, t_d)
    expected = _fit(t_a, y_a, t_d, y_d)[0]

    _forbid_guess(monkeypatch)
    np.testing.assert_array_equal(_fit(t_a, y_a, t_d, y_d, s0_guesses=guesses)[0], expected)


@pytest.mark.parametrize("ci95", [one_site_assoc_and_disso_asymmetric_ci95,
                                  one_site_assoc_and_disso_asymmetric_ci95_koff])
def test_confidence_interval_guesses_once(monkeypatch, ci95):
    t_a, y_a, t_d, y_d = _cycles(gap=10)
    params, _, fit_a, fit_d = _fit(t_a, y_a, t_d, y_d)
    y, y_fit = np.concatenate(y_a + y_d), np.concatenate(fit_a + fit_d)
    rss_desired = get_desired_rss(y, y_fit, len(y), len(params))

    calls = []
    original = fitting_surface.guess_initial_signal
    monkeypatch.setattr(fitting_surface, "guess_initial_signal",
                        lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))

    if ci95 is one_site_assoc_and_disso_asymmetric_ci95:
        estimate, rest = params[0], [params[1], params[2]]
        low, high = [1e-4, 1.0], [1.0, 50.0]
    else:
        estimate, rest = params[1], [params[0], params[2]]
        low, high = [1e-3, 1.0], [1e3, 50.0]

    lower, upper = ci95(estimate, rss_desired, y_a, t_a, CONCS, y_d, t_d, rest, low, high,
                        smax_idx=[0] * len(CONCS), shared_smax=True)

    assert len(calls) == 1
    assert lower < estimate < upper
