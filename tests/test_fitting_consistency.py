# This file is AI generated (PR #15).
"""Regressions for grouping, confidence limits, and optional fitting inputs."""

import numpy as np
import pytest

import pykingenie.utils.fitting_surface as surface_fits
from pykingenie.fitter_solution import KineticsFitterSolution
from pykingenie.fitter_surface import KineticsFitter
from pykingenie.utils.fitting_solution import (
    fit_induced_fit_solution,
    find_initial_parameters_conformational_selection_solution,
    find_initial_parameters_induced_fit_solution,
)
from pykingenie.utils.signal_solution import (
    get_initial_concentration_conformational_selection,
    signal_ode_conformational_selection_insolution,
    signal_ode_induced_fit_insolution,
    signal_ode_one_site_insolution,
)
from pykingenie.utils.signal_surface import (
    one_site_association_analytical,
    one_site_dissociation_analytical,
    solve_induced_fit_association,
    solve_induced_fit_dissociation,
    solve_ode_one_site_mass_transport_association,
    solve_ode_one_site_mass_transport_dissociation,
    solve_two_site_cooperative_association,
    solve_two_site_cooperative_dissociation,
    solve_two_site_heterogeneous_ligand_association,
    solve_two_site_heterogeneous_ligand_dissociation,
)


@pytest.mark.parametrize("group_ids", [[7, 2, 7, 2, 7, 2], np.array([7, 2, 7, 2, 7, 2])])
def test_interleaved_smax_groups_recover_distinct_capacities(group_ids):
    # This function is AI generated (PR #15).
    t = np.linspace(0, 150, 41)
    concs = [0.2, 0.2, 1.0, 1.0, 5.0, 5.0]
    capacities = [3.0, 8.0, 3.0, 8.0, 3.0, 8.0]
    signals = [one_site_association_analytical(t, 0, s, 0.03, 0.5, c)
               for s, c in zip(capacities, concs)]
    fitter = KineticsFitter([t] * 6, signals, concs, smax_id=group_ids)
    fitter.Kd_ss = 0.5
    fitter.get_steady_state()
    fitter.fit_one_site_association(shared_smax=True)

    assert fitter.smax_id == [0, 1, 0, 1, 0, 1]
    np.testing.assert_allclose(fitter.Smax, [3.0, 8.0], rtol=1e-4)
    np.testing.assert_allclose(fitter.signal_assoc_fit, signals, atol=1e-7)


@pytest.mark.parametrize("parameter", ["Kd", "koff"])
@pytest.mark.parametrize("target_rss", [0.51, 2.0, 50.0])
def test_confidence_limits_reach_profile_rss_threshold(parameter, target_rss):
    """Refit real traces at both limits and check the RSS crossing on each side."""
    # This function is AI generated (PR #15).
    t = np.linspace(0, 60, 25)
    concs = [0.1, 0.5, 2.0]
    assoc = [one_site_association_analytical(t, 0, 5.0, 0.03, 0.5, c) for c in concs]
    disso = [one_site_dissociation_analytical(t, a[-1], 0.03) for a in assoc]
    data = dict(
        assoc_signal_lst=assoc, assoc_time_lst=[t]*3, analyte_conc_lst=concs,
        disso_signal_lst=disso, disso_time_lst=[t+60]*3,
        smax_idx=[0, 0, 0], shared_smax=True,
    )
    if parameter == "Kd":
        estimate = 0.5
        nuisance = dict(initial_parameters=[0.03, 5.0], low_bounds=[0.001, 0.1],
                        high_bounds=[1.0, 20.0])
        confidence_limits = surface_fits.one_site_assoc_and_disso_asymmetric_ci95
    else:
        estimate = 0.03
        nuisance = dict(initial_parameters=[0.5, 5.0], low_bounds=[0.01, 0.1],
                        high_bounds=[10.0, 20.0])
        confidence_limits = surface_fits.one_site_assoc_and_disso_asymmetric_ci95_koff

    def profile_rss(value):
        # This function is AI generated (PR #15).
        fixed = (dict(fixed_Kd=True, Kd_value=value) if parameter == "Kd"
                 else dict(fixed_koff=True, koff_value=value))
        _, _, fitted_assoc, fitted_disso = surface_fits.fit_one_site_assoc_and_disso(
            **data, **nuisance, **fixed)
        return sum(np.sum((observed-predicted)**2)
                   for observed, predicted in zip(assoc+disso, fitted_assoc+fitted_disso))

    lower, upper = confidence_limits(estimate, target_rss, **data, **nuisance)
    assert 0 < lower < estimate < upper
    np.testing.assert_allclose([profile_rss(lower), profile_rss(upper)], target_rss, rtol=1e-4)
    assert profile_rss(lower*0.99) > target_rss
    assert profile_rss(lower*1.01) < target_rss
    assert profile_rss(upper*0.99) < target_rss
    assert profile_rss(upper*1.01) > target_rss


@pytest.mark.parametrize("equal_complex_signals", [False, True])
def test_induced_fit_can_measure_only_free_protein(equal_complex_signals):
    # This function is AI generated (PR #15).
    times = [np.linspace(0, 30, 41), np.linspace(0, 40, 55)]
    concs = [0.5, 2.0]
    rates = dict(k1=0.4, k_minus1=0.03, k2=0.1, k_minus2=0.02)
    signals = [signal_ode_induced_fit_insolution(t, [0, 0], **rates,
               E_tot=1.0, S_tot=c, signal_E=3.0) for t, c in zip(times, concs)]
    params, _, predictions, names = fit_induced_fit_solution(
        signals, times, concs, [1.0, 1.0], [2.0], [0.0], [10.0],
        fit_signal_E=True, fit_signal_ES=False, ESint_equals_ES=equal_complex_signals,
        fixed_kon=True, kon_value=0.4, fixed_koff=True, koff_value=0.03,
        fixed_kc=True, kc_value=0.1, fixed_krev=True, krev_value=0.02,
    )
    np.testing.assert_allclose(params, [3.0], rtol=1e-6)
    assert names == ["Signal of the free protein"]
    for prediction, signal in zip(predictions, signals):
        np.testing.assert_allclose(prediction, signal, rtol=1e-6)


def test_solution_one_site_wrapper_accepts_unequal_trace_lengths():
    # This function is AI generated (PR #15).
    times = [np.linspace(0, 5, 31), np.linspace(0, 8, 53)]
    concs = [0.5, 2.0]
    signals = [signal_ode_one_site_insolution(t, 0.5, 0.5, 1.0, c, signal_complex=3.0)
               for t, c in zip(times, concs)]
    fitter = KineticsFitterSolution("unequal", signals, concs, [1.0, 1.0], times)
    fitter.fit_one_binding_site()
    np.testing.assert_allclose(fitter.params, [3.0, 0.5, 0.5], rtol=1e-4)
    for prediction, signal in zip(fitter.signal_assoc_fit, signals):
        np.testing.assert_allclose(prediction, signal, atol=1e-7)


@pytest.mark.parametrize("model", ["induced_fit", "conformational_selection"])
def test_solution_initial_search_accepts_unequal_trace_lengths(model):
    # This function is AI generated (PR #15).
    times = [np.linspace(0, 30, 31), np.linspace(0, 50, 47)]
    concs = [0.5, 2.0]
    if model == "induced_fit":
        signals = [signal_ode_induced_fit_insolution(
            t, [0, 0], 1.0, 1.0, 0.1, 0.1, 1.0, c,
            signal_ES=3.0, signal_ES_int=3.0) for t, c in zip(times, concs)]
        search = find_initial_parameters_induced_fit_solution
    else:
        _, active = get_initial_concentration_conformational_selection(1.0, 0.1, 0.1)
        signals = [signal_ode_conformational_selection_insolution(
            t, [active, 0], 0.1, 0.1, 1.0, 1.0, 1.0, c,
            signal_E2S=3.0) for t, c in zip(times, concs)]
        search = find_initial_parameters_conformational_selection_solution
    params = search(signals, times, concs, [1.0, 1.0],
                    np_linspace_low=-1, np_linspace_high=-1, np_linspace_num=1)
    np.testing.assert_allclose(params, [3.0, 1.0, 1.0, 0.1, 0.1], rtol=1e-3)


@pytest.mark.parametrize("model, fixed_t0", [
    ("association", True), ("one_site", True), ("one_site", False),
    ("transport", True), ("transport", False),
    ("two_site", True), ("two_site", False),
    ("heterogeneous", True), ("heterogeneous", False), ("induced", True),
])
@pytest.mark.parametrize("shared_smax", [False, True])
@pytest.mark.parametrize("groups", [[0, 1, 2], [0, 1, 0, 2]])
def test_loo_keeps_group_parameters_aligned(model, shared_smax, fixed_t0, groups):
    """Run every LOO optimization and recover the simulated rates and group Ktr values."""
    # This function is AI generated (PR #15).
    n_groups = len(set(groups))
    n_traces = len(groups)
    t = np.linspace(0, 60, 41)
    concs = [0.2, 1.0, 3.0, 0.5][:n_traces]
    offsets = [0.0] * n_groups if fixed_t0 else [-0.02, -0.04, -0.06]
    ktrs = [0.02, 0.04, 0.08]
    responses = [3.0, 5.0, 7.0] if shared_smax else [3.0, 5.0, 7.0, 9.0][:n_traces]
    assoc, disso = [], []
    for i, (c, group) in enumerate(zip(concs, groups)):
        smax = responses[group] if shared_smax else responses[i]
        t0 = offsets[group]
        if model in ("association", "one_site"):
            a = one_site_association_analytical(t, 0, smax, 0.03, 0.5, c, t0)
            d = one_site_dissociation_analytical(t, a[-1], 0.03)
        elif model == "transport":
            a = solve_ode_one_site_mass_transport_association(
                t, 0, c, 0.5, 0.03, ktrs[group], smax, t0)
            d = solve_ode_one_site_mass_transport_dissociation(
                t, a[-1], 0.5, 0.03, ktrs[group], smax)
        elif model == "two_site":
            states = solve_two_site_cooperative_association(
                t, c, 0.06, 0.03, 1.5, smax, 2*smax, t0=t0)
            a = states[:, 0]
            d = solve_two_site_cooperative_dissociation(
                t, 0.03, 1.5, smax, 2*smax,
                states[-1, 1]/smax, states[-1, 2]/(2*smax))[:, 0]
        elif model == "heterogeneous":
            states = solve_two_site_heterogeneous_ligand_association(
                t, c, 0.1, 0.03, 2.0, 0.1, smax, 0.4, t0=t0)
            a = states[:, 0]
            d = solve_two_site_heterogeneous_ligand_dissociation(
                t, 0.03, 0.1, fraction_site1=0.4,
                s1_0=states[-1, 1], s2_0=states[-1, 2])[:, 0]
        else:
            states = solve_induced_fit_association(t, c, 0.06, 0.03, 0.1, 0.02, smax=smax)
            a = states[:, 0]
            d = solve_induced_fit_dissociation(t, 0.03, 0.1, 0.02, a[-1], states[-1, 2], smax)[:, 0]
        assoc.append(a)
        disso.append(d)

    fitter = KineticsFitter([t]*n_traces, assoc, concs, [t+60]*n_traces, disso, smax_id=groups)
    fitter.shared_smax = shared_smax
    fitter.fixed_t0 = fixed_t0
    fitter.fit_ktr = model == "transport"
    fitter.fit_sigma = model == "two_site"
    rates = {"association": [0.5, 0.03], "one_site": [0.5, 0.03],
             "transport": [0.5, 0.03], "two_site": [0.5, 0.03, 1.5],
             "heterogeneous": [0.1, 0.03, 2.0, 0.1, 0.4],
             "induced": [0.06, 0.03, 0.1, 0.02]}[model]
    params = rates + ([] if fixed_t0 else offsets)
    if model == "transport":
        params += ktrs
    params += [value for s in responses for value in ([s, 2*s] if model == "two_site" else [s])]
    fitter.params = params
    fitter.low_bounds = [p - 0.001 for p in params]
    fitter.high_bounds = [p + 0.001 for p in params]
    method = {"association": "one_site_association_loo", "one_site": "one_site_assoc_and_disso_loo",
              "transport": "one_site_assoc_and_disso_loo", "two_site": "two_site_assoc_and_disso_loo",
              "heterogeneous": "two_site_heterogeneous_ligand_assoc_and_disso_loo",
              "induced": "one_site_if_assoc_and_disso_loo"}[model]
    getattr(fitter, method)()

    attributes = {
        "association": ["Kd_loo", "k_off_loo"],
        "one_site": ["Kd_loo", "k_off_loo"],
        "transport": ["Kd_loo", "k_off_loo"],
        "two_site": ["Kd_loo", "k_off_loo", "sigma_loo"],
        "heterogeneous": ["Kd1_loo", "k_off1_loo", "Kd2_loo", "k_off2_loo", "fraction_site1_loo"],
        "induced": ["k_on_loo", "k_off_loo", "k_c_loo", "k_rev_loo"],
    }[model]
    for attribute, expected in zip(attributes, rates):
        values = getattr(fitter, attribute)
        assert len(values) == n_traces
        np.testing.assert_allclose(values, expected, rtol=1e-5)
    if model == "transport":
        np.testing.assert_allclose(fitter.ktr_loo_means, ktrs, rtol=1e-5)
