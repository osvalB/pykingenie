# This code is AI generated.
"""
Global surface kinetic fits run on a log scale for the strictly positive parameters.
At the same optimum the parameters, standard errors and correlations are those of a
linear-scale fit; the fit only needs fewer evaluations.
"""
import numpy as np
import pytest
from scipy.optimize import curve_fit

import pykingenie.utils.fitting_surface as fitting_surface
from pykingenie.kingenie_surface import KinGenieCsv
from pykingenie.main import KineticsAnalyzer
from pykingenie.utils.fitting_general import curve_fit_log_scale

T = np.linspace(0, 100, 200)


def _rise(t, a, k, t0):
    return a * (1 - np.exp(-k * (t - t0)))


def _noisy(f, *params):
    return f(T, *params) + np.random.default_rng(0).normal(0, 0.05, len(T))


def _assert_same_fit(fit_a, fit_b, rtol=1e-5):
    (p_a, cov_a), (p_b, cov_b) = fit_a, fit_b
    np.testing.assert_allclose(p_a, p_b, rtol=rtol, atol=1e-6)
    se = np.sqrt(np.outer(np.diag(cov_b), np.diag(cov_b)))
    np.testing.assert_allclose(cov_a / se, cov_b / se, rtol=0, atol=1e-4)


def test_same_fit_as_curve_fit_with_a_parameter_that_can_be_negative():
    """t0 has a negative lower bound, so it stays on a linear scale."""
    y = _noisy(_rise, 5.0, 0.05, -2.0)
    kwargs = dict(p0=[1.0, 0.01, 0.0], bounds=([0.1, 1e-4, -10.0], [50.0, 1.0, 10.0]))
    params, cov = curve_fit_log_scale(_rise, T, y, **kwargs)
    _assert_same_fit((params, cov), curve_fit(_rise, T, y, **kwargs))
    assert params[2] < 0


def test_jacobian_method_name_is_passed_to_curve_fit():
    # This test is AI generated.
    """A jac given by name, not as a function, needs no conversion to the log scale."""
    y = _noisy(_rise, 5.0, 0.05, -2.0)
    kwargs = dict(p0=[1.0, 0.01, 0.0], bounds=([0.1, 1e-4, -10.0], [50.0, 1.0, 10.0]))
    _assert_same_fit(curve_fit_log_scale(_rise, T, y, jac='3-point', **kwargs),
                     curve_fit(_rise, T, y, jac='3-point', **kwargs))


def test_parameter_with_zero_lower_bound_stays_linear():
    """A baseline whose best value is on its lower bound of zero."""
    def decay(t, a, k, c):
        return a * np.exp(-k * t) + c

    y = decay(T, 3.0, 0.03, 0.0) - 0.01
    kwargs = dict(p0=[1.0, 0.1, 0.5], bounds=([1e-3, 1e-4, 0.0], [10.0, 1.0, 1.0]))
    params, cov = curve_fit_log_scale(decay, T, y, **kwargs)
    _assert_same_fit((params, cov), curve_fit(decay, T, y, **kwargs), rtol=1e-4)
    assert params[2] == pytest.approx(0.0, abs=1e-8)


def test_scalar_bounds_and_infinite_upper_bound():
    y = _noisy(lambda t, a, k: a * np.exp(-k * t), 4.0, 0.02)
    params, _ = curve_fit_log_scale(lambda t, a, k: a * np.exp(-k * t), T, y, p0=[1.0, 0.1], bounds=(1e-6, np.inf))
    np.testing.assert_allclose(params, [4.0, 0.02], rtol=0.02)


@pytest.mark.parametrize("p0", [[0.05, 0.01, 0.0], [-1.0, 0.01, 0.0]])
def test_start_outside_the_bounds_raises_as_curve_fit(p0):
    y = _noisy(_rise, 5.0, 0.05, -2.0)
    with pytest.raises(ValueError):
        curve_fit_log_scale(_rise, T, y, p0=p0, bounds=([0.1, 1e-4, -10.0], [50.0, 1.0, 10.0]))


# Global fits of the surface models, compared with the same fits on a linear scale

FILES = {
    "1:1": "./test_files/simulation_Kd-10_koff-0.01.csv",
    "MTL": "./test_files/simulation_Kd-0.5_koff-0.01_Ktr-0.005.csv",
    "two-site": "./test_files/simulation_KinGenie_kon-0.5_koff-0.005_sigma-1_plRmax-3.5_pl2_Rmax-7.csv",
}
FITS = {
    "1:1": lambda f: f.fit_one_site_assoc_and_disso(),
    "1:1 Smax per trace": lambda f: f.fit_one_site_assoc_and_disso(shared_smax=False),
    "1:1 t0": lambda f: f.fit_one_site_assoc_and_disso(fixed_t0=False),
    "MTL": lambda f: f.fit_one_site_assoc_and_disso(fit_ktr=True),
    "two-site": lambda f: f.fit_two_site_assoc_and_disso(),
}


def _noisy_fitter(file):
    kingenie = KinGenieCsv()
    kingenie.read_csv(file)
    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(kingenie, "test")
    analyzer.merge_ligand_conc_df()
    analyzer.generate_fittings(analyzer.combined_ligand_conc_df)
    fitter = list(analyzer.fittings.values())[0]

    rng = np.random.default_rng(0)
    amp = 0.01 * max(np.max(np.abs(y)) for y in fitter.assoc_lst)
    fitter.time_assoc_lst = [t[::4] for t in fitter.time_assoc_lst]
    fitter.time_disso_lst = [t[::4] for t in fitter.time_disso_lst]
    fitter.assoc_lst = [y[::4] + rng.normal(0, amp, len(y[::4])) for y in fitter.assoc_lst]
    fitter.disso_lst = [y[::4] + rng.normal(0, amp, len(y[::4])) for y in fitter.disso_lst]
    fitter.get_steady_state()
    fitter.fit_steady_state()
    return fitter


def _counting(fit_function, counter):
    def wrapped(f, *args, **kwargs):
        def counted(x, *params):
            counter[0] += 1
            return f(x, *params)
        return fit_function(counted, *args, **kwargs)
    return wrapped


def _run(name, monkeypatch, linear):
    counter = [0]
    fit_function = (lambda f, *args, **kwargs: curve_fit(f, *args, **kwargs)) if linear else curve_fit_log_scale
    monkeypatch.setattr(fitting_surface, "curve_fit_log_scale", _counting(fit_function, counter))
    fitter = _noisy_fitter(FILES[name.split(" ")[0]])
    FITS[name](fitter)
    results = [fitter.fit_params_kinetics, fitter.fit_params_kinetics_error, fitter.fit_params_kinetics_correlation]
    return [r.select_dtypes("number").to_numpy(float) for r in results], counter[0]


@pytest.mark.parametrize("name", FITS)
def test_surface_fits_match_the_linear_scale_fit(name, monkeypatch):
    (params, errors, corr), _ = _run(name, monkeypatch, linear=False)
    (params_lin, errors_lin, corr_lin), _ = _run(name, monkeypatch, linear=True)

    # Within the optimizer tolerance: t0 is the least constrained (relative difference 9e-5)
    np.testing.assert_allclose(params, params_lin, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(errors, errors_lin, rtol=1e-3)
    np.testing.assert_allclose(corr, corr_lin, rtol=0, atol=1e-4)


def test_mass_transport_fit_needs_fewer_evaluations(monkeypatch):
    _, n_log = _run("MTL", monkeypatch, linear=False)
    _, n_linear = _run("MTL", monkeypatch, linear=True)
    assert n_log < 0.75 * n_linear
