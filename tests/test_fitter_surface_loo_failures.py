# This file contains AI-generated changes (PR #15).
"""Leave-one-out fits skip real invalid-data failures and retain valid results."""

import numpy as np
import pytest

from pykingenie.fitter_surface import KineticsFitter
from pykingenie.utils.signal_surface import (
    one_site_association_analytical,
    one_site_dissociation_analytical,
    solve_induced_fit_association,
    solve_induced_fit_dissociation,
    solve_two_site_association,
    solve_two_site_dissociation,
    solve_two_site_heterogeneous_ligand_association,
    solve_two_site_heterogeneous_ligand_dissociation,
)


def _create_loo_fitter(model="one_site"):
    # This function contains AI-generated changes (PR #15).
    t = np.linspace(0, 60, 41)
    concs = [0.2, 1.0, 3.0]
    assoc, disso = [], []
    for i, conc in enumerate(concs):
        if model == "two_site":
            states = solve_two_site_association(t, conc, 0.06, 0.03, 5.0, 10.0)
            a = states[:, 0]
            d = solve_two_site_dissociation(
                t, 0.03, 5.0, 10.0, states[-1, 1]/5.0, states[-1, 2]/10.0)[:, 0]
        elif model == "heterogeneous":
            states = solve_two_site_heterogeneous_ligand_association(
                t, conc, 0.1, 0.03, 2.0, 0.1, Rmax=5.0, fraction_site1=0.4)
            a = states[:, 0]
            d = solve_two_site_heterogeneous_ligand_dissociation(
                t, 0.03, 0.1, fraction_site1=0.4,
                s1_0=states[-1, 1], s2_0=states[-1, 2])[:, 0]
        elif model == "induced":
            states = solve_induced_fit_association(t, conc, 0.06, 0.03, 0.1, 0.02, smax=5.0)
            a = states[:, 0]
            d = solve_induced_fit_dissociation(
                t, 0.03, 0.1, 0.02, a[-1], states[-1, 2], 5.0)[:, 0]
        else:
            a = one_site_association_analytical(t, 0, 5.0, 0.03, 0.5, conc)
            s0 = 3.0+i if model == "dissociation" else a[-1]
            d = one_site_dissociation_analytical(t, s0, 0.03)
        assoc.append(a)
        disso.append(d)

    # Only the fold omitting this trace has entirely finite fitting data.
    assoc[-1] = np.full_like(t, np.nan)
    disso[-1] = np.full_like(t, np.nan)
    fitter = KineticsFitter([t]*3, assoc, concs, [t+60]*3, disso, smax_id=[0, 0, 0])
    fitter.disso_lst_subset = disso
    fitter.time_disso_lst_subset = [t+60]*3
    fitter.shared_smax = True
    fitter.fixed_t0 = True
    fitter.fit_ktr = False
    fitter.fit_sigma = False
    fitter.params = {
        "association": [0.5, 0.03, 5.0],
        "one_site": [0.5, 0.03, 5.0],
        "dissociation": [0.03, 3.0, 4.0, 5.0],
        "two_site": [0.5, 0.03, 5.0, 10.0],
        "heterogeneous": [0.1, 0.03, 2.0, 0.1, 0.4, 5.0],
        "induced": [0.06, 0.03, 0.1, 0.02, 5.0],
    }[model]
    fitter.low_bounds = [p-0.001 for p in fitter.params]
    fitter.high_bounds = [p+0.001 for p in fitter.params]
    return fitter


@pytest.mark.parametrize("model", [None, "unsupported_model"])
def test_calc_leave_one_out_errors_rejects_unavailable_model(model):
    fitter = _create_loo_fitter()
    if model is not None:
        fitter.model = model

    with pytest.raises(ValueError) as exc_info:
        fitter.calc_leave_one_out_errors()

    assert str(exc_info.value) == f"No LOO calculation is available for model: {model!r}"


@pytest.mark.parametrize("model, method_name, expected", [
    ("association", "one_site_association_loo", {"Kd_loo": 0.5, "k_off_loo": 0.03}),
    ("dissociation", "one_site_dissociation_loo", {"k_off_loo": 0.03}),
    ("one_site", "one_site_assoc_and_disso_loo", {"Kd_loo": 0.5, "k_off_loo": 0.03}),
    ("two_site", "two_site_assoc_and_disso_loo", {"Kd_loo": 0.5, "k_off_loo": 0.03}),
    ("heterogeneous", "two_site_heterogeneous_ligand_assoc_and_disso_loo", {
        "Kd1_loo": 0.1, "k_off1_loo": 0.03, "Kd2_loo": 2.0,
        "k_off2_loo": 0.1, "fraction_site1_loo": 0.4,
    }),
    ("induced", "one_site_if_assoc_and_disso_loo", {
        "k_on_loo": 0.06, "k_off_loo": 0.03, "k_c_loo": 0.1, "k_rev_loo": 0.02,
    }),
])
def test_loo_skips_invalid_data_and_keeps_successful_fit(model, method_name, expected):
    # This function is AI generated (PR #15).
    fitter = _create_loo_fitter(model)
    getattr(fitter, method_name)()

    for attribute, value in expected.items():
        estimates = getattr(fitter, attribute)
        assert len(estimates) == 1
        np.testing.assert_allclose(estimates, [value], rtol=1e-5)
