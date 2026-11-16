# This code is AI generated.
"""
Asymmetric 95% confidence limits of Kd and koff (one-to-one model): each limit is
where the RSS of the refit with the parameter fixed reaches the desired RSS. The
limits are found with a root search and warm-started refits.
"""
import numpy as np
import pytest

import pykingenie.utils.fitting_surface as fitting_surface
from pykingenie.utils.fitting_surface import (
    _asymmetric_ci95_limits,
    fit_one_site_assoc_and_disso,
    one_site_assoc_and_disso_asymmetric_ci95,
    one_site_assoc_and_disso_asymmetric_ci95_koff,
)
from pykingenie.utils.math import get_desired_rss, get_rss
from pykingenie.utils.signal_surface import (
    one_site_association_analytical,
    one_site_dissociation_analytical,
)

ESTIMATE = 2.0
FACTORS = [1.1, 1.5, 2, 5, 10, 25, 50, 100]


def _profile(rss_at_estimate=0.0):
    """Synthetic profile RSS(v) = rss_at_estimate + ln(v / ESTIMATE)^2, recording each refit."""
    calls = []

    def fit_fixed(value, start):
        calls.append((value, list(start)))
        return rss_at_estimate + np.log(value / ESTIMATE) ** 2, [value]

    return fit_fixed, calls


def test_limits_where_the_profile_crosses_the_target():
    fit_fixed, _ = _profile()
    lower, upper = _asymmetric_ci95_limits(fit_fixed, ESTIMATE, np.log(3.0) ** 2, [0.0], FACTORS)
    assert upper == pytest.approx(3 * ESTIMATE, rel=1e-8)
    assert lower == pytest.approx(ESTIMATE / 3, rel=1e-8)


def test_each_refit_starts_from_the_nearest_value_tried():
    fit_fixed, calls = _profile()
    _asymmetric_ci95_limits(fit_fixed, ESTIMATE, np.log(3.0) ** 2, [0.0], FACTORS)

    assert calls[0][1] == [0.0]  # the first refit starts from the initial parameters
    for i, (value, start) in enumerate(calls[1:], start=1):
        tried = [v for v, _ in calls[:i]]
        nearest = min(tried, key=lambda v: abs(np.log(v / value)))
        assert start == [nearest]


def test_limits_beyond_the_test_factors_use_the_wide_search():
    """The RSS stays below the target up to 100 times the estimate: the limits, at 1000
    times and 1/1000 of it, are searched between 100 and 10,000 times (and inverse)."""
    fit_fixed, _ = _profile()
    lower, upper = _asymmetric_ci95_limits(fit_fixed, ESTIMATE, np.log(1000.0) ** 2, [0.0], FACTORS)
    assert upper == pytest.approx(1000 * ESTIMATE, rel=1e-4)
    assert lower == pytest.approx(ESTIMATE / 1000, rel=1e-4)


def test_rounding_above_the_target_at_the_estimate():
    """Noise-free data: the RSS is at rounding level, and the refit at the estimate can
    land above the target. The estimate is the optimum, so it still counts as below the
    target, and the limits stay on each side of it, close to it."""
    fit_fixed, _ = _profile(rss_at_estimate=1.0)
    lower, upper = _asymmetric_ci95_limits(fit_fixed, ESTIMATE, 0.5, [0.0], FACTORS)
    assert lower <= ESTIMATE <= upper
    assert upper == pytest.approx(ESTIMATE, rel=1e-6) and lower == pytest.approx(ESTIMATE, rel=1e-6)


# One-to-one data with noise

KD, KOFF, SMAX = 0.5, 0.01, 5.0
CONCS = [0.1, 0.3, 1.0, 3.0]
LOW, HIGH = [1e-3, 1e-4, 1.0], [1e3, 1.0, 50.0]


@pytest.fixture(scope="module")
def noisy_fit():
    rng = np.random.default_rng(3)
    t_a = [np.arange(0, 300, 2.0) for _ in CONCS]
    t_d = [np.arange(0, 400, 2.0) for _ in CONCS]
    y_a = [one_site_association_analytical(t, 0.0, SMAX, KOFF, KD, c) for t, c in zip(t_a, CONCS)]
    y_d = [one_site_dissociation_analytical(t, y[-1], KOFF) for t, y in zip(t_d, y_a)]
    y_a = [y + rng.normal(0, 0.05, len(y)) for y in y_a]
    y_d = [y + rng.normal(0, 0.05, len(y)) for y in y_d]

    data = (y_a, t_a, CONCS, y_d, t_d)
    params, _, fit_a, fit_d = fit_one_site_assoc_and_disso(*data, [1.0, 0.02, 4.0], LOW, HIGH,
                                                          smax_idx=[0] * len(CONCS), shared_smax=True)
    y, y_fit = np.concatenate(y_a + y_d), np.concatenate(fit_a + fit_d)
    return data, params, get_desired_rss(y, y_fit, len(y), len(params))


def _rss_with_fixed(data, rss_args, **fixed):
    _, _, fit_a, fit_d = fit_one_site_assoc_and_disso(*data, *rss_args, smax_idx=[0] * len(CONCS),
                                                      shared_smax=True, **fixed)
    y_a, _, _, y_d, _ = data
    return get_rss(np.concatenate(y_a + y_d), np.concatenate(fit_a + fit_d))


@pytest.mark.parametrize("parameter", ["Kd", "koff"])
def test_refit_at_each_limit_reaches_the_desired_rss(noisy_fit, parameter, monkeypatch):
    data, params, rss_desired = noisy_fit

    refits = []
    original = fitting_surface.fit_one_site_assoc_and_disso
    monkeypatch.setattr(fitting_surface, "fit_one_site_assoc_and_disso",
                        lambda *args, **kwargs: refits.append(1) or original(*args, **kwargs))

    if parameter == "Kd":
        rest = ([params[1], params[2]], LOW[1:], HIGH[1:])
        limits = one_site_assoc_and_disso_asymmetric_ci95(params[0], rss_desired, *data, *rest,
                                                          smax_idx=[0] * len(CONCS), shared_smax=True)
        fixed = lambda value: dict(fixed_Kd=True, Kd_value=value)
        estimate = params[0]
    else:
        rest = ([params[0], params[2]], [LOW[0], LOW[2]], [HIGH[0], HIGH[2]])
        limits = one_site_assoc_and_disso_asymmetric_ci95_koff(params[1], rss_desired, *data, *rest,
                                                               smax_idx=[0] * len(CONCS), shared_smax=True)
        fixed = lambda value: dict(fixed_koff=True, koff_value=value)
        estimate = params[1]

    lower, upper = limits
    assert lower < estimate < upper
    # Fewer refits than the bounded minimisation of |RSS - target| needed (about 80)
    assert len(refits) < 60

    monkeypatch.undo()
    for limit in limits:
        # Refit from the global optimum, independent of the warm starts
        rss = _rss_with_fixed(data, rest, **fixed(limit))
        assert rss == pytest.approx(rss_desired, rel=1e-6)
