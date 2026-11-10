# This code is AI generated and reviewed by a human (PR #XX).
"""
Correlation matrices of fitted surface-binding parameters: the helper that
builds them, the labels each fit stores, and access from KineticsAnalyzer.
"""
import numpy as np
import pandas as pd
import pytest

from pykingenie.fitter_surface import KineticsFitter
from pykingenie.kingenie_surface import KinGenieCsv
from pykingenie.main import KineticsAnalyzer
from pykingenie.utils.math import correlation_from_covariance
from pykingenie.utils.signal_surface import (
    solve_induced_fit_association,
    solve_induced_fit_dissociation,
)

ONE_SITE = "./test_files/simulation_Kd-10_koff-0.01.csv"
MASS_TRANSPORT = "./test_files/simulation_Kd-0.5_koff-0.01_Ktr-0.005.csv"
INDUCED_FIT = "./test_files/simulation_kon-0.1_koff-0.01_kc-1_krev-10.csv"
TWO_SITE = "./test_files/simulation_KinGenie_kon-0.5_koff-0.005_sigma-1_plRmax-3.5_pl2_Rmax-7.csv"


def _analyzer(path):
    """KineticsAnalyzer with one fitting, every 4th point kept for speed."""
    kingenie = KinGenieCsv()
    kingenie.read_csv(path)
    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(kingenie, "test_surface")
    analyzer.merge_ligand_conc_df()
    analyzer.generate_fittings(analyzer.combined_ligand_conc_df)
    fitter = list(analyzer.fittings.values())[0]
    for name in ("assoc_lst", "disso_lst", "time_assoc_lst", "time_disso_lst"):
        setattr(fitter, name, [x[::4] for x in getattr(fitter, name)])
    return analyzer


def _fitter(path):
    return list(_analyzer(path).fittings.values())[0]


def _check_matrix(corr, labels):
    """A correlation matrix: labelled, symmetric, ones on the diagonal, within [-1, 1]."""
    assert isinstance(corr, pd.DataFrame)
    assert list(corr.index) == labels
    assert list(corr.columns) == labels
    values = corr.to_numpy()
    np.testing.assert_allclose(values, values.T, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(np.diag(values), 1.0, rtol=1e-12)
    finite = values[np.isfinite(values)]
    assert np.all((finite >= -1.0) & (finite <= 1.0))


# The helper turns a known covariance matrix into the expected correlations.
def test_correlation_from_covariance_known_values():
    sd = np.array([2.0, 0.5, 3.0])
    rho = np.array([[1.0, 0.9, -0.3], [0.9, 1.0, 0.0], [-0.3, 0.0, 1.0]])
    cov = rho * np.outer(sd, sd)
    corr = correlation_from_covariance(cov, ["a", "b", "c"])
    np.testing.assert_allclose(corr.to_numpy(), rho, rtol=1e-12)
    assert list(corr.columns) == ["a", "b", "c"]


# Parameters whose variance cannot be estimated (inf, as curve_fit returns
# when the fit is singular, or zero) get NaN correlations, also on the diagonal.
@pytest.mark.parametrize("bad_variance", [np.inf, 0.0, np.nan])
def test_correlation_from_covariance_undefined_variance(bad_variance):
    cov = np.array([[4.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, bad_variance]])
    corr = correlation_from_covariance(cov, ["a", "b", "c"]).to_numpy()
    np.testing.assert_allclose(corr[:2, :2], [[1.0, 0.5], [0.5, 1.0]], rtol=1e-12)
    assert np.all(np.isnan(corr[2, :])) and np.all(np.isnan(corr[:, 2]))


# A covariance matrix that does not match the labels is refused.
def test_correlation_from_covariance_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        correlation_from_covariance(np.eye(3), ["a", "b"])


# Parameters fitted per group are numbered from 1, with paired names interleaved.
def test_group_labels():
    assert KineticsFitter._group_labels("Smax", 2) == ["Smax 1", "Smax 2"]
    assert KineticsFitter._group_labels(("Rmax_PL", "Rmax_LPL"), 2) == [
        "Rmax_PL 1", "Rmax_LPL 1", "Rmax_PL 2", "Rmax_LPL 2"]


# Each steady-state fit stores a matrix labelled in the order of its parameters.
@pytest.mark.parametrize("model, kwargs, path, head", [
    ("one_site", {}, ONE_SITE, ["Kd [µM]"]),
    ("two_site", {"fit_sigma": False}, TWO_SITE, ["Kd [µM]"]),
    ("two_site", {"fit_sigma": True}, TWO_SITE, ["Kd [µM]", "sigma"]),
    ("two_site_heterogeneous_ligand", {}, TWO_SITE, ["Kd1 [µM]", "Kd2 [µM]", "fraction_site1"]),
])
def test_steady_state_correlation_labels(model, kwargs, path, head):
    fitter = _fitter(path)
    fitter.fit_steady_state(model=model, **kwargs)
    n = len(fitter.params) - len(head)
    if model == "two_site":
        tail = KineticsFitter._group_labels(("Rmax_PL", "Rmax_LPL"), n // 2)
    elif model == "one_site":
        tail = KineticsFitter._group_labels("Smax", n)
    else:
        tail = KineticsFitter._group_labels("Rmax", n)
    _check_matrix(fitter.fit_params_ss_correlation, head + tail)


# Each kinetics fit stores a matrix labelled in the order of its parameters,
# including t0 and Ktr when they are fitted.
@pytest.mark.parametrize("method, kwargs, path, head, per_group", [
    ("fit_one_site_association", {}, ONE_SITE, ["Kd [µM]", "k_off [1/s]"], []),
    ("fit_one_site_dissociation", {}, ONE_SITE, ["k_off [1/s]"], []),
    ("fit_one_site_assoc_and_disso", {}, ONE_SITE, ["Kd [µM]", "k_off [1/s]"], []),
    ("fit_one_site_assoc_and_disso", {"fixed_t0": False}, ONE_SITE, ["Kd [µM]", "k_off [1/s]"], ["t0"]),
    ("fit_one_site_assoc_and_disso", {"fit_ktr": True}, MASS_TRANSPORT, ["Kd [µM]", "k_off [1/s]"], ["Ktr"]),
    ("fit_one_site_assoc_and_disso", {"fit_ktr": True, "fixed_t0": False}, MASS_TRANSPORT,
     ["Kd [µM]", "k_off [1/s]"], ["t0", "Ktr"]),
    ("fit_two_site_assoc_and_disso", {"fit_sigma": True, "fixed_t0": False}, TWO_SITE,
     ["Kd [µM]", "k_off [1/s]", "sigma"], ["t0"]),
    ("fit_two_site_heterogeneous_ligand_assoc_and_disso", {}, ONE_SITE,
     ["Kd1 [µM]", "k_off1 [1/s]", "Kd2 [µM]", "k_off2 [1/s]", "fraction_site1"], []),
    ("fit_one_site_if_assoc_and_disso", {}, INDUCED_FIT,
     ["k_on [1/(s*µM)]", "k_off [1/s]", "k_c [1/s]", "k_rev [1/s]"], []),
])
def test_kinetics_correlation_labels(method, kwargs, path, head, per_group):
    fitter = _fitter(path)
    fitter.fit_steady_state()
    getattr(fitter, method)(**kwargs)
    n_groups = len(np.unique(fitter.smax_id))
    labels = list(head)
    for name in per_group:
        labels += KineticsFitter._group_labels(name, n_groups)
    n_rest = len(fitter.params) - len(labels)
    if method == "fit_two_site_assoc_and_disso":
        labels += KineticsFitter._group_labels(("Rmax_PL", "Rmax_LPL"), n_rest // 2)
    elif method == "fit_one_site_dissociation":
        labels += KineticsFitter._group_labels("S0", n_rest)
    elif method == "fit_two_site_heterogeneous_ligand_assoc_and_disso":
        labels += KineticsFitter._group_labels("Rmax", n_rest)
    else:
        labels += KineticsFitter._group_labels("Smax", n_rest)
    _check_matrix(fitter.fit_params_kinetics_correlation, labels)


# A new fit clears the previous matrices.
def test_correlation_cleared_by_new_fit():
    fitter = _fitter(ONE_SITE)
    fitter.fit_steady_state()
    fitter.fit_one_site_association()
    assert fitter.fit_params_kinetics_correlation is not None
    fitter.clear_fittings()
    assert fitter.fit_params_kinetics_correlation is None
    assert fitter.fit_params_ss_correlation is None


# KineticsAnalyzer returns one matrix per fitting, for kinetics or steady state.
def test_analyzer_get_correlation_matrices():
    analyzer = _analyzer(ONE_SITE)
    assert analyzer.get_correlation_matrices() == {}

    analyzer.submit_steady_state_fitting()
    ss = analyzer.get_correlation_matrices(kind="steady_state")
    assert list(ss) == analyzer.fittings_names
    assert list(ss[analyzer.fittings_names[0]].columns)[0] == "Kd [µM]"

    analyzer.submit_kinetics_fitting(fitting_model="one_to_one", fitting_region="association_dissociation")
    kinetics = analyzer.get_correlation_matrices()
    corr = kinetics[analyzer.fittings_names[0]]
    assert list(corr.columns)[:2] == ["Kd [µM]", "k_off [1/s]"]

    # A copy: changing it does not change the stored matrix
    corr.iloc[0, 1] = 99.0
    assert analyzer.get_correlation_matrices()[analyzer.fittings_names[0]].iloc[0, 1] != 99.0

    with pytest.raises(ValueError, match="kind must be"):
        analyzer.get_correlation_matrices(kind="other")


# The matrix flags underdetermined rates: in the test file kc and krev
# correlate almost perfectly with the other rates, while with a slow
# conformational step they do not.
def test_correlation_flags_underdetermined_induced_fit_rates():
    def max_rate_correlation(fitter):
        corr = fitter.fit_params_kinetics_correlation.abs()
        rates = ["k_on [1/(s*µM)]", "k_off [1/s]", "k_c [1/s]", "k_rev [1/s]"]
        sub = corr.loc[["k_c [1/s]", "k_rev [1/s]"], rates].to_numpy()
        return np.nanmax(np.where(sub < 1.0 - 1e-12, sub, np.nan))

    fitter = _fitter(INDUCED_FIT)
    fitter.fit_steady_state()
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=True)
    assert max_rate_correlation(fitter) > 0.95

    kon, koff, kc, krev, smax = 0.1, 0.05, 0.01, 0.005, 5.0
    concs = [0.05, 0.15, 0.45, 1.35, 4.05]
    t_assoc = np.arange(0, 600.0, 2.0)
    t_disso = np.arange(600.0, 1800.0, 4.0)
    rng = np.random.default_rng(0)
    assoc, disso = [], []
    for c in concs:
        y = solve_induced_fit_association(t_assoc, c, kon, koff, kc, krev, smax=smax)
        d = solve_induced_fit_dissociation(t_disso, koff, kc, krev, s0=y[-1, 0], sP2L=y[-1, 2], smax=smax)[:, 0]
        assoc.append(y[:, 0] + rng.normal(0, 0.005 * smax, len(t_assoc)))
        disso.append(d + rng.normal(0, 0.005 * smax, len(t_disso)))
    fitter = KineticsFitter([t_assoc] * 5, assoc, concs, [t_disso] * 5, disso, smax_id=[0] * 5)
    fitter.fit_steady_state()
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=True)
    assert max_rate_correlation(fitter) < 0.9
