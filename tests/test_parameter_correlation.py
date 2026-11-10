# This code is AI generated and reviewed by a human (PR #19).
"""
Correlation matrices of fitted surface-binding parameters: the helper that
builds them, the labels each fit stores, access from KineticsAnalyzer, and
the concentration column of tables with one row per trace.
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
# including t0 and Ktr when they are fitted. Amplitudes shared by a group
# are numbered; amplitudes fitted per trace are named by concentration.
@pytest.mark.parametrize("shared_smax", [True, False])
@pytest.mark.parametrize("method, kwargs, path, head, per_group", [
    ("fit_one_site_association", {}, ONE_SITE, ["Kd [µM]", "k_off [1/s]"], []),
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
def test_kinetics_correlation_labels(method, kwargs, path, head, per_group, shared_smax):
    fitter = _fitter(path)
    fitter.fit_steady_state()
    getattr(fitter, method)(shared_smax=shared_smax, **kwargs)
    n_groups = len(np.unique(fitter.smax_id))
    labels = list(head)
    for name in per_group:
        labels += KineticsFitter._group_labels(name, n_groups)
    if method == "fit_two_site_assoc_and_disso":
        amplitude = ("Rmax_PL", "Rmax_LPL")
    elif method == "fit_two_site_heterogeneous_ligand_assoc_and_disso":
        amplitude = "Rmax"
    else:
        amplitude = "Smax"
    if shared_smax:
        labels += KineticsFitter._group_labels(amplitude, n_groups)
    else:
        labels += fitter._trace_labels(amplitude)
    _check_matrix(fitter.fit_params_kinetics_correlation, labels)

    # Tables with one row per trace start with the trace's concentration
    table = fitter.fit_params_kinetics
    if shared_smax:
        assert "Concentration [µM]" not in table
    else:
        assert list(table.columns)[0] == "Concentration [µM]"
        np.testing.assert_allclose(table["Concentration [µM]"], fitter.lig_conc_lst, rtol=1e-12)


# The dissociation fit has one initial signal per trace, named by concentration,
# and its tables start with the concentration of each trace.
def test_dissociation_correlation_labels():
    fitter = _fitter(ONE_SITE)
    fitter.fit_one_site_dissociation()
    _check_matrix(fitter.fit_params_kinetics_correlation,
                  ["k_off [1/s]"] + fitter._trace_labels("S0"))
    for table in (fitter.fit_params_kinetics, fitter.fit_params_kinetics_error):
        assert list(table.columns)[0] == "Concentration [µM]"
        np.testing.assert_allclose(table["Concentration [µM]"], fitter.lig_conc_lst, rtol=1e-12)


# Per-trace labels show the concentration with 3 significant figures,
# numbering traces that share a concentration so every label is unique.
def test_trace_labels():
    fitter = _fitter(ONE_SITE)
    fitter.lig_conc_lst = [0.2, 0.44097, 1.0, 1.0]
    assert fitter._trace_labels("Smax") == [
        "Smax (0.2 µM)", "Smax (0.441 µM)", "Smax (1 µM, 1)", "Smax (1 µM, 2)"]
    assert fitter._trace_labels(("Rmax_PL", "Rmax_LPL"))[:2] == ["Rmax_PL (0.2 µM)", "Rmax_LPL (0.2 µM)"]


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


# The matrix flags underdetermined rates. With a conformational step faster
# than dissociation, the curves only fix kon and an apparent koff, so kc and
# krev correlate almost perfectly with the other rates; with a slow step they
# do not. Both use data with noise: on noise-free data the covariance is
# set by round-off and changes between platforms.
def test_correlation_flags_underdetermined_induced_fit_rates():
    def max_rate_correlation(fitter):
        corr = fitter.fit_params_kinetics_correlation.abs()
        rates = ["k_on [1/(s*µM)]", "k_off [1/s]", "k_c [1/s]", "k_rev [1/s]"]
        sub = corr.loc[["k_c [1/s]", "k_rev [1/s]"], rates].to_numpy()
        return np.nanmax(np.where(sub < 1.0 - 1e-12, sub, np.nan))

    fitter = _induced_fit_fitter(0.1, 0.01, 0.2, 1.0, [0.01, 0.03, 0.09, 0.27, 0.81], seed=2)
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=True)
    assert max_rate_correlation(fitter) > 0.98

    fitter = _slow_induced_fit_fitter()
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=True)
    assert max_rate_correlation(fitter) < 0.9


def _induced_fit_fitter(kon, koff, kc, krev, concs, seed, smax=5.0):
    """Induced-fit data with noise of 0.5 % of Smax."""
    t_assoc = np.arange(0, 600.0, 2.0)
    t_disso = np.arange(600.0, 1800.0, 4.0)
    rng = np.random.default_rng(seed)
    assoc, disso = [], []
    for c in concs:
        y = solve_induced_fit_association(t_assoc, c, kon, koff, kc, krev, smax=smax)
        d = solve_induced_fit_dissociation(t_disso, koff, kc, krev, s0=y[-1, 0], sP2L=y[-1, 2], smax=smax)[:, 0]
        assoc.append(y[:, 0] + rng.normal(0, 0.005 * smax, len(t_assoc)))
        disso.append(d + rng.normal(0, 0.005 * smax, len(t_disso)))
    n = len(concs)
    fitter = KineticsFitter([t_assoc] * n, assoc, concs, [t_disso] * n, disso, smax_id=[0] * n)
    fitter.fit_steady_state()
    return fitter


def _slow_induced_fit_fitter():
    """Induced-fit data with a slow conformational step, so all four rates are determined."""
    return _induced_fit_fitter(0.1, 0.05, 0.01, 0.005, [0.05, 0.15, 0.45, 1.35, 4.05], seed=0)


# The induced-fit fit stores its own relative errors, not those of the
# one-to-one fit it runs first for starting values.
@pytest.mark.parametrize("shared_smax", [True, False])
def test_induced_fit_error_table(shared_smax):
    fitter = _slow_induced_fit_fitter()
    fitter.fit_one_site_if_assoc_and_disso(shared_smax=shared_smax)
    errors = fitter.fit_params_kinetics_error
    rates = ["k_on [1/(s*µM)]", "k_off [1/s]", "k_c [1/s]", "k_rev [1/s]"]
    assert list(errors.columns) == list(fitter.fit_params_kinetics.columns)
    assert all(rate in errors for rate in rates) and "Kd [µM]" not in errors
    assert len(errors) == len(fitter.fit_params_kinetics)
    # Well-determined rates: small, positive relative errors (%)
    values = errors[rates].to_numpy()
    assert np.all((values > 0) & (values < 10))
