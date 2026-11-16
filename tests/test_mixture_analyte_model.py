# This code is AI generated.
"""
Mixture-analyte surface model (several analytes compete for one ligand),
solved exactly as a linear system instead of with solve_ivp.
"""
import itertools

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pykingenie.utils.signal_surface import (
    ode_mixture_analyte_association,
    one_site_association_analytical,
    solve_ode_mixture_analyte_association,
    solve_ode_mixture_analyte_dissociation,
)

T = np.linspace(0, 600, 301)

MIXTURES = {
    "one analyte": dict(Fis=[1.0], Ris_max=[2.0], koffs=[0.01], Kds=[0.5]),
    "two analytes": dict(Fis=[0.6, 0.4], Ris_max=[1.0, 2.0], koffs=[0.01, 0.002], Kds=[0.5, 0.05]),
    "three analytes": dict(Fis=[0.5, 0.3, 0.2], Ris_max=[1.0, 2.0, 3.0], koffs=[0.1, 0.01, 1e-3], Kds=[1.0, 0.1, 0.01]),
    "stiff": dict(Fis=[0.5, 0.5], Ris_max=[1.0, 1.0], koffs=[10.0, 1e-4], Kds=[1e-3, 10.0]),
}


def _reference(mix, conc, R0, t=T):
    m = {k: np.asarray(v, dtype=float) for k, v in mix.items()}
    sol = solve_ivp(ode_mixture_analyte_association, (t[0], t[-1]), R0, t_eval=t,
                    args=(conc, m["Fis"], m["Ris_max"], m["koffs"], m["Kds"]),
                    method="Radau", rtol=1e-12, atol=1e-14)
    return sol.y


@pytest.mark.parametrize("name,conc", list(itertools.product(MIXTURES, [0.01, 1.0, 100.0])))
def test_association_matches_tight_ode(name, conc):
    mix = MIXTURES[name]
    R0 = np.zeros(len(mix["Fis"]))
    R = solve_ode_mixture_analyte_association(T, R0, conc, **mix)
    assert R.shape == (len(mix["Fis"]), len(T))
    np.testing.assert_allclose(R, _reference(mix, conc, R0), rtol=0, atol=1e-9 * max(mix["Ris_max"]))


def test_association_from_a_partly_bound_surface():
    mix = MIXTURES["three analytes"]
    R0 = np.array([0.2, 0.1, 0.4])
    R = solve_ode_mixture_analyte_association(T, R0, 0.5, **mix)
    np.testing.assert_allclose(R[:, 0], R0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(R, _reference(mix, 0.5, R0), rtol=0, atol=1e-9 * 3)


@pytest.mark.parametrize("name", MIXTURES)
def test_equilibrium_is_competitive_langmuir(name):
    """At long times R_i = Rmax_i·x_i / (1 + Σ x_j), with x_i = F_i·C / Kd_i."""
    mix = {k: np.asarray(v, dtype=float) for k, v in MIXTURES[name].items()}
    conc = 0.3
    x = mix["Fis"] * conc / mix["Kds"]
    expected = mix["Ris_max"] * x / (1 + x.sum())
    R = solve_ode_mixture_analyte_association(np.array([0.0, 1e7]), np.zeros(len(x)), conc, **MIXTURES[name])
    np.testing.assert_allclose(R[:, -1], expected, rtol=1e-10)


def test_single_analyte_is_the_one_to_one_model():
    mix = MIXTURES["one analyte"]
    R = solve_ode_mixture_analyte_association(T, [0.3], 0.7, **mix)
    expected = one_site_association_analytical(T, 0.3, mix["Ris_max"][0], mix["koffs"][0], mix["Kds"][0], 0.7)
    np.testing.assert_allclose(R[0], expected, rtol=1e-12)


def test_dissociation_is_exact():
    R0, koffs = np.array([1.0, 2.0, 0.5]), np.array([0.1, 0.01, 1e-4])
    t = np.arange(300.0, 900.0, 3.0)  # need not start at zero
    R = solve_ode_mixture_analyte_dissociation(t, R0, koffs)
    np.testing.assert_allclose(R, R0[:, None] * np.exp(-np.outer(koffs, t - t[0])), rtol=1e-14)


@pytest.mark.parametrize("phase", ["association", "dissociation"])
def test_t0_shifts_the_start(phase):
    """The phase starts t0 after the first time point, as in the other surface models.
    Before, solve_ivp always started at the first time point and t0 had no effect."""
    mix, t0 = MIXTURES["two analytes"], 30.0
    R0 = np.array([0.1, 0.3])
    if phase == "association":
        solve = lambda t, shift: solve_ode_mixture_analyte_association(t, R0, 0.5, **mix, t0=shift)
    else:
        solve = lambda t, shift: solve_ode_mixture_analyte_dissociation(t, R0, mix["koffs"], t0=shift)

    shifted = solve(T, t0)
    after = T >= t0
    np.testing.assert_allclose(shifted[:, after], solve(T[after] - t0, 0.0), rtol=1e-12, atol=1e-15)
    assert not np.allclose(shifted, solve(T, 0.0))
