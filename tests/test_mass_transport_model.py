# This code is AI generated and reviewed by a human.
"""
Checks of the two-compartment mass-transport model (Svitel et al. 2007, Eq. 5,
https://doi.org/10.1529/biophysj.106.094615) against limits with known answers,
conservation, expected behaviour and fits.

    dS/dt  = kon * cs * (Smax - S) - koff * S
    dcs/dt = ktr * (c0 - cs) - dS/dt
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pykingenie.utils.signal_surface import (
    one_site_association_analytical,
    one_site_dissociation_analytical,
    solve_ode_one_site_mass_transport_association as assoc,
    solve_ode_one_site_mass_transport_dissociation as disso,
)
from pykingenie.utils.fitting_surface import fit_one_site_assoc_and_disso_ktr

KD, KOFF, KTR, SMAX = 0.5, 0.01, 0.005, 5.0


def eq8_ode(c0, Kd, koff, ktr, smax):
    """Quasi-steady-state (Eq. 8) form: cs follows S instantly."""
    kon = koff / Kd
    return lambda t, y: [(kon * c0 * (smax - y[0]) - koff * y[0]) / (1 + kon / ktr * (smax - y[0]))]


# Limits with known answers

def test_large_ktr_matches_one_site_analytical():
    """With very fast transport (ktr = 1e4) there is no transport limit, so both
    phases must match the analytical one-site (1:1) association and dissociation."""
    t = np.linspace(0, 300, 301)
    c0 = 1.0
    s, _ = assoc(t, 0.0, 0.0, c0, KD, KOFF, 1e4, SMAX, return_cs=True)
    expected = one_site_association_analytical(t, 0.0, SMAX, KOFF, KD, c0)
    assert np.allclose(s, expected, rtol=1e-2, atol=0)

    d = disso(t, s[-1], KD, KOFF, 1e4, SMAX)
    assert np.allclose(d, one_site_dissociation_analytical(t, s[-1], KOFF), rtol=1e-2, atol=0)


@pytest.mark.parametrize("ktr", [0.005, 0.05, 5.0])
@pytest.mark.parametrize("c0", [0.1, 0.5, 5.0])
def test_equilibrium_does_not_depend_on_ktr(ktr, c0):
    """Transport changes how fast binding happens, never where it ends: after a long
    association S reaches Smax * c0 / (Kd + c0) and cs reaches the bulk c0, for any ktr."""
    t = np.linspace(0, 2e5, 401)
    s, cs = assoc(t, 0.0, 0.0, c0, KD, KOFF, ktr, SMAX, return_cs=True)
    assert np.isclose(s[-1], SMAX * c0 / (KD + c0), rtol=1e-2, atol=0)
    assert np.isclose(cs[-1], c0, rtol=1e-2, atol=0)


def test_fast_surface_layer_follows_eq8():
    """When the surface layer settles much faster than S changes, the full model
    reduces to the quasi-steady-state form (Eq. 8), in association and dissociation."""
    # Surface layer settles at >= ktr = 0.5 /s, while S changes at <= ~0.01 /s
    Kd, koff, ktr, smax, c0 = 0.01, 0.001, 0.5, 5.0, 0.02
    t = np.linspace(0, 2000, 2001)

    # Start the surface layer at its quasi-steady-state value, as Eq. 8 assumes
    kon = koff / Kd
    cs0 = ktr * c0 / (ktr + kon * smax)
    s = assoc(t, 0.0, cs0, c0, Kd, koff, ktr, smax)
    ref = solve_ivp(eq8_ode(c0, Kd, koff, ktr, smax), (0, 2000), [0.0], t_eval=t, rtol=1e-10, atol=1e-12).y[0]
    assert np.allclose(s, ref, rtol=1e-2, atol=0)

    d = disso(t, s[-1], Kd, koff, ktr, smax)
    ref = solve_ivp(eq8_ode(0.0, Kd, koff, ktr, smax), (0, 2000), [s[-1]], t_eval=t, rtol=1e-10, atol=1e-12).y[0]
    assert np.allclose(d, ref, rtol=1e-2, atol=0)


def test_transport_only_without_binding():
    """With no binding (huge Kd), nothing binds and the surface layer fills from
    the bulk as c0 * (1 - exp(-ktr * t))."""
    t = np.linspace(0, 200, 201)
    ktr, c0 = 0.05, 2.0
    s, cs = assoc(t, 0.0, 0.0, c0, 1e9, 1e-9, ktr, 1.0, return_cs=True)
    # Nothing binds: S stays a negligible fraction of Smax
    assert np.all(np.abs(s) <= 1e-9 * 1.0)  # Smax = 1
    assert np.allclose(cs, c0 * (1 - np.exp(-ktr * t)), rtol=1e-2, atol=0)


# Conservation and bounds

@pytest.mark.parametrize("phase", ["association", "dissociation"])
def test_mass_balance(phase):
    """Analyte that enters the surface layer either binds or stays free:
    S + cs changes by exactly the integrated transport flux ktr * (c0 - cs)."""
    t = np.linspace(0, 300, 3001)
    if phase == "association":
        c0 = 1.0
        s, cs = assoc(t, 0.0, 0.0, c0, KD, KOFF, KTR, SMAX, return_cs=True)
    else:
        c0 = 0.0
        s, cs = disso(t, 3.0, KD, KOFF, KTR, SMAX, cs_0=0.8, return_cs=True)

    total = s + cs
    inflow = np.concatenate([[0.0], np.cumsum(np.diff(t) * KTR * (c0 - (cs[1:] + cs[:-1]) / 2))])
    assert np.allclose(total, total[0] + inflow, rtol=1e-3, atol=0)


@pytest.mark.parametrize("Kd", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("koff", [1e-3, 0.1])
@pytest.mark.parametrize("ktr", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("c0", [0.1, 10.0])
def test_signal_and_surface_concentration_stay_in_bounds(Kd, koff, ktr, c0):
    """Across a grid of parameters, 0 <= S <= Smax and 0 <= cs <= c0 in association,
    and cs never rises above its starting value in dissociation."""
    t = np.linspace(0, 600, 301)
    # Relative to each bound, at the solvers' LSODA default rtol of 1e-3
    rtol = 1e-3
    s, cs = assoc(t, 0.0, 0.0, c0, Kd, koff, ktr, SMAX, return_cs=True)
    d, cs_d = disso(t, s[-1], Kd, koff, ktr, SMAX, cs_0=cs[-1], return_cs=True)
    for signal in (s, d):
        assert np.all((signal >= -rtol * SMAX) & (signal <= SMAX * (1 + rtol)))
    assert np.all((cs >= -rtol * c0) & (cs <= c0 * (1 + rtol)))
    assert np.all((cs_d >= -rtol * cs[-1]) & (cs_d <= cs[-1] * (1 + rtol)))


# Mass-transport behaviour

def test_slower_transport_slows_binding_and_dissociation():
    """A smaller ktr gives a lower signal at every association time and, through
    more rebinding, a higher signal at every dissociation time."""
    t = np.linspace(0, 300, 301)
    ktrs = [0.5, 0.05, 0.005]

    assoc_traces = [assoc(t, 0.0, 0.0, 1.0, KD, KOFF, k, SMAX) for k in ktrs]
    for faster, slower in zip(assoc_traces, assoc_traces[1:]):
        assert np.all(slower[1:] < faster[1:])

    # Same start, nothing free near the surface: more rebinding with slower transport
    disso_traces = [disso(t, 3.0, KD, KOFF, k, SMAX, cs_0=0.0) for k in ktrs]
    for faster, slower in zip(disso_traces, disso_traces[1:]):
        assert np.all(slower[1:] > faster[1:])


def test_binding_is_transport_limited_when_kon_smax_dominates():
    """When kon * Smax >> ktr, every molecule that reaches the surface binds, so the
    early binding rate equals the transport flux ktr * c0, not kon."""
    # kon * Smax = 10 /s >> ktr = 0.01 /s: every molecule that arrives binds
    Kd, koff, ktr, smax, c0 = 0.01, 0.01, 0.01, 10.0, 0.1
    t = np.linspace(0, 100, 1001)
    s = assoc(t, 0.0, 0.0, c0, Kd, koff, ktr, smax)
    early = t >= 10
    slope = np.polyfit(t[early], s[early], 1)[0]
    assert np.isclose(slope, ktr * c0, rtol=0.05, atol=0)


def test_rebinding_lowers_apparent_off_rate():
    """Late in dissociation (S << Smax), rebinding slows the decay to an apparent
    off-rate of koff / (1 + kon * (Smax - S) / ktr)."""
    Kd, koff, ktr, smax = 0.1, 0.01, 0.5, 10.0
    kon = koff / Kd
    t = np.linspace(0, 1500, 1501)
    d = disso(t, 0.05, Kd, koff, ktr, smax)
    late = t >= 300
    slope = -np.polyfit(t[late], np.log(d[late]), 1)[0]
    expected = koff / (1 + kon * (smax - d[late].mean()) / ktr)
    assert np.isclose(slope, expected, rtol=0.03, atol=0)
    assert slope < koff / 2


# State carried between steps (single-cycle kinetics)

def test_restarting_from_returned_state_matches_one_solve():
    """Single-cycle kinetics: solving in two pieces, restarting from the returned
    (S, cs), must give the same curve as one solve, in both phases."""
    t = np.linspace(0, 200, 401)
    half = t <= 100

    s_full, cs_full = assoc(t, 0.0, 0.0, 1.0, KD, KOFF, KTR, SMAX, return_cs=True)
    s1, cs1 = assoc(t[half], 0.0, 0.0, 1.0, KD, KOFF, KTR, SMAX, return_cs=True)
    s2, cs2 = assoc(t[~half] - 100, s1[-1], cs1[-1], 1.0, KD, KOFF, KTR, SMAX, return_cs=True)
    # Each solve has LSODA's default rtol of 1e-3, so the two paths differ by a few times that
    assert np.allclose(np.r_[s1, s2], np.r_[s_full[half], s_full[~half]], rtol=1e-2, atol=0)
    assert np.allclose(np.r_[cs1, cs2], np.r_[cs_full[half], cs_full[~half]], rtol=1e-2, atol=0)

    d_full, csd_full = disso(t, s_full[-1], KD, KOFF, KTR, SMAX, cs_0=cs_full[-1], return_cs=True)
    d1, csd1 = disso(t[half], s_full[-1], KD, KOFF, KTR, SMAX, cs_0=cs_full[-1], return_cs=True)
    d2 = disso(t[~half] - 100, d1[-1], KD, KOFF, KTR, SMAX, cs_0=csd1[-1])
    assert np.allclose(np.r_[d1, d2], np.r_[d_full[half], d_full[~half]], rtol=1e-2, atol=0)


# Fits

CONCS = [0.25, 0.5, 1.0, 2.0]
T_ASSOC = np.linspace(0, 300, 151)
T_DISSO = np.linspace(0, 300, 151)


def simulate(ktr, single_cycle=False, noise=0.0, seed=0):
    """Traces for CONCS; single-cycle runs carry signal and cs from one step to the next."""
    rng = np.random.default_rng(seed)
    out = {"assoc_signal_lst": [], "assoc_time_lst": [], "disso_signal_lst": [], "disso_time_lst": []}
    s0 = cs0 = 0.0
    for i, c0 in enumerate(CONCS):
        offset = i * (T_ASSOC[-1] + T_DISSO[-1]) if single_cycle else 0.0
        a, cs_a = assoc(T_ASSOC, s0, cs0, c0, KD, KOFF, ktr, SMAX, return_cs=True)
        d, cs_d = disso(T_DISSO, a[-1], KD, KOFF, ktr, SMAX, cs_0=cs_a[-1], return_cs=True)
        if single_cycle:
            s0, cs0 = d[-1], cs_d[-1]
        out["assoc_signal_lst"].append(a + rng.normal(0, noise, a.size))
        out["assoc_time_lst"].append(T_ASSOC + offset)
        out["disso_signal_lst"].append(d + rng.normal(0, noise, d.size))
        out["disso_time_lst"].append(T_DISSO + offset + T_ASSOC[-1])
    return out


def fit(data, ktr_guess=0.01, ktr_upper=10.0, with_errors=False):
    params, cov, _, _ = fit_one_site_assoc_and_disso_ktr(
        **data,
        analyte_conc_lst=CONCS,
        initial_parameters=[1.0, 0.02, ktr_guess, 4.0],
        low_bounds=[1e-3, 1e-4, 1e-5, 0.5],
        high_bounds=[100.0, 1.0, ktr_upper, 50.0],
        smax_idx=[0, 0, 0, 0],
        shared_smax=True,
        fixed_t0=True,
    )
    names = ["Kd", "koff", "ktr", "smax"]
    if with_errors:
        return dict(zip(names, params)), dict(zip(names, np.sqrt(np.diag(cov))))
    return dict(zip(names, params))


@pytest.mark.parametrize("seed", range(5))
def test_fit_recovers_parameters_from_noisy_data(seed):
    """On data with 1% noise, the fit recovers Kd, koff, ktr and Smax within
    3 standard errors, and those errors stay small (under 25%)."""
    # Kd and Smax are correlated, so a fixed tolerance would fail by chance
    p, err = fit(simulate(KTR, noise=0.01 * SMAX, seed=seed), with_errors=True)
    for name, true in [("Kd", KD), ("koff", KOFF), ("ktr", KTR), ("smax", SMAX)]:
        assert abs(p[name] - true) < 3 * err[name], (name, p[name], err[name])
        assert err[name] < 0.25 * true, (name, err[name])


def test_fit_does_not_invent_transport_limit():
    """On data with no transport limit, the fit still recovers Kd and koff and
    the fitted ktr is too large to matter."""
    p = fit(simulate(ktr=1e3), ktr_guess=0.05, ktr_upper=1e4)
    assert np.isclose(p["Kd"], KD, rtol=0.05, atol=0)
    assert np.isclose(p["koff"], KOFF, rtol=0.05, atol=0)
    # kon * Smax / ktr << 1: the fitted transport limit is negligible
    assert (KOFF / KD) * SMAX / p["ktr"] < 0.05


def test_single_and_multi_cycle_fits_agree():
    """The same parameters fitted from multi-cycle and single-cycle runs agree
    with each other and with the true values."""
    multi = fit(simulate(KTR))
    single = fit(simulate(KTR, single_cycle=True))
    for name, true in [("Kd", KD), ("koff", KOFF), ("ktr", KTR), ("smax", SMAX)]:
        assert np.isclose(multi[name], true, rtol=0.05, atol=0), (name, multi[name])
        assert np.isclose(single[name], true, rtol=0.05, atol=0), (name, single[name])
        assert np.isclose(single[name], multi[name], rtol=0.05, atol=0)
