# This code is AI generated and reviewed by a human.
"""
Checks of the mass-transport model against limits with known answers, physical
behaviour and fits.

The model is the quasi-steady-state form (Svitel et al. 2007, Eq. 8,
https://doi.org/10.1529/biophysj.106.094615) with a lumped transport coefficient
k_tr in signal per µM per second:

    dS/dt = (kon * c0 * (Smax - S) - koff * S) / (1 + kon * (Smax - S) / k_tr)
"""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from pykingenie.kingenie_surface import KinGenieCsv
from pykingenie.main import KineticsAnalyzer
from pykingenie.utils.signal_surface import (
    one_site_association_analytical,
    one_site_dissociation_analytical,
    solve_ode_one_site_mass_transport_association as assoc,
    solve_ode_one_site_mass_transport_dissociation as disso,
)
from pykingenie.utils.fitting_surface import fit_one_site_assoc_and_disso_ktr

KD, KOFF, KTR, SMAX = 0.5, 0.01, 0.005, 5.0
TEST_FILE = "./test_files/simulation_Kd-0.5_koff-0.01_Ktr-0.005.csv"


# Limits with known answers

def test_large_ktr_matches_one_site_analytical():
    """With very fast transport (k_tr = 1e4) there is no transport limit, so both
    phases must match the analytical one-site (1:1) association and dissociation."""
    t = np.linspace(0, 300, 301)
    c0 = 1.0
    s = assoc(t, 0.0, c0, KD, KOFF, 1e4, SMAX)
    expected = one_site_association_analytical(t, 0.0, SMAX, KOFF, KD, c0)
    assert np.allclose(s, expected, rtol=1e-2, atol=0)

    d = disso(t, s[-1], KD, KOFF, 1e4, SMAX)
    assert np.allclose(d, one_site_dissociation_analytical(t, s[-1], KOFF), rtol=1e-2, atol=0)


@pytest.mark.parametrize("ktr", [0.005, 0.05, 5.0])
@pytest.mark.parametrize("c0", [0.1, 0.5, 5.0])
def test_equilibrium_does_not_depend_on_ktr(ktr, c0):
    """Transport changes how fast binding happens, never where it ends: after a long
    association S reaches Smax * c0 / (Kd + c0), for any k_tr."""
    t = np.linspace(0, 2e5, 401)
    s = assoc(t, 0.0, c0, KD, KOFF, ktr, SMAX)
    assert np.isclose(s[-1], SMAX * c0 / (KD + c0), rtol=1e-2, atol=0)


@pytest.mark.parametrize("c0", [1.0, 0.0])
def test_thin_surface_layer_limit_of_two_compartment_model(c0):
    """The model is the limit of the two-compartment model (Svitel et al. 2007, Eq. 5)
    when the layer next to the surface holds a negligible amount of analyte.

    With layer capacity eps (signal per µM), the two-compartment model is
        dS/dt  = kon * cs * (Smax - S) - koff * S
        dcs/dt = (k_tr * (c0 - cs) - dS/dt) / eps
    and for small eps it must follow the quasi-steady-state solvers."""
    eps = 1e-4
    kon = KOFF / KD

    def two_compartment(t, y):
        s, cs = y
        ds = kon * cs * (SMAX - s) - KOFF * s
        return [ds, (KTR * (c0 - cs) - ds) / eps]

    t = np.linspace(0, 600, 601)
    s0 = 0.0 if c0 else 3.0
    # Start the layer at its quasi-steady-state value, as the solvers assume
    cs0 = (KTR * c0 + KOFF * s0) / (KTR + kon * (SMAX - s0))
    ref = solve_ivp(two_compartment, (0, 600), [s0, cs0], t_eval=t, method="LSODA", rtol=1e-10, atol=1e-12).y[0]

    s = assoc(t, s0, c0, KD, KOFF, KTR, SMAX) if c0 else disso(t, s0, KD, KOFF, KTR, SMAX)
    assert np.allclose(s, ref, rtol=1e-2, atol=0)


# Units

@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_solvers_scale_with_signal_units(scale):
    """k_tr is in signal per µM per second: expressing the signal in other units
    (Smax and k_tr times `scale`) must scale the curves by exactly that factor."""
    t = np.linspace(0, 300, 301)
    s = assoc(t, 0.0, 1.0, KD, KOFF, KTR, SMAX)
    s_scaled = assoc(t, 0.0, 1.0, KD, KOFF, KTR * scale, SMAX * scale)
    assert np.allclose(s_scaled, s * scale, rtol=1e-2, atol=0)

    d = disso(t, s[-1], KD, KOFF, KTR, SMAX)
    d_scaled = disso(t, s[-1] * scale, KD, KOFF, KTR * scale, SMAX * scale)
    assert np.allclose(d_scaled, d * scale, rtol=1e-2, atol=0)


def fit_test_file(path):
    """Fit the KinGenie test file with the mass-transport model, as in test_fitter_surface."""
    kingenie = KinGenieCsv()
    kingenie.read_csv(str(path))
    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(kingenie, "test_surface")
    analyzer.merge_ligand_conc_df()
    analyzer.generate_fittings(analyzer.combined_ligand_conc_df)
    fitter = list(analyzer.fittings.values())[0]
    for name in ("assoc_lst", "disso_lst", "time_assoc_lst", "time_disso_lst"):
        setattr(fitter, name, [x[::4] for x in getattr(fitter, name)])
    fitter.Kd_ss = KD
    fitter.Smax_upper_bound_factor = 50
    fitter.fit_one_site_assoc_and_disso(shared_smax=True, fixed_t0=True, fit_ktr=True)
    return fitter.fit_params_kinetics.iloc[0]


def test_fit_does_not_depend_on_signal_units(tmp_path):
    """Fitting the same data in signal units 1000 times larger (for example nm to pm)
    must give the same Kd and koff, with k_tr and Smax 1000 times larger."""
    scale = 1000.0
    data = pd.read_csv(TEST_FILE)
    data["Signal"] *= scale
    data["Smax"] *= scale
    scaled_file = tmp_path / "scaled.csv"
    data.to_csv(scaled_file, index=False)

    base = fit_test_file(TEST_FILE)
    scaled = fit_test_file(scaled_file)

    assert np.isclose(scaled["Kd [µM]"], base["Kd [µM]"], rtol=1e-2, atol=0)
    assert np.isclose(scaled["k_off [1/s]"], base["k_off [1/s]"], rtol=1e-2, atol=0)
    assert np.isclose(scaled["Ktr"], base["Ktr"] * scale, rtol=1e-2, atol=0)
    assert np.isclose(scaled["Smax"], base["Smax"] * scale, rtol=1e-2, atol=0)
    assert np.isclose(base["Ktr"], KTR, rtol=0.05, atol=0)


# Bounds and mass-transport behaviour

@pytest.mark.parametrize("Kd", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("koff", [1e-3, 0.1])
@pytest.mark.parametrize("ktr", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("c0", [0.1, 10.0])
def test_signal_stays_in_bounds(Kd, koff, ktr, c0):
    """Across a grid of parameters, 0 <= S <= Smax in association and dissociation."""
    t = np.linspace(0, 600, 301)
    # Relative to Smax, well above the solvers' rtol of 1e-8
    rtol = 1e-6
    s = assoc(t, 0.0, c0, Kd, koff, ktr, SMAX)
    d = disso(t, s[-1], Kd, koff, ktr, SMAX)
    for signal in (s, d):
        assert np.all((signal >= -rtol * SMAX) & (signal <= SMAX * (1 + rtol)))


def test_slower_transport_slows_binding_and_dissociation():
    """A smaller k_tr gives a lower signal at every association time and, through
    more rebinding, a higher signal at every dissociation time."""
    t = np.linspace(0, 300, 301)
    ktrs = [0.5, 0.05, 0.005]

    assoc_traces = [assoc(t, 0.0, 1.0, KD, KOFF, k, SMAX) for k in ktrs]
    for faster, slower in zip(assoc_traces, assoc_traces[1:]):
        assert np.all(slower[1:] < faster[1:])

    disso_traces = [disso(t, 3.0, KD, KOFF, k, SMAX) for k in ktrs]
    for faster, slower in zip(disso_traces, disso_traces[1:]):
        assert np.all(slower[1:] > faster[1:])


def test_binding_is_transport_limited_when_kon_smax_dominates():
    """When kon * Smax >> k_tr, every molecule that reaches the surface binds, so the
    early binding rate equals the transport flux k_tr * c0, not kon."""
    Kd, koff, ktr, smax, c0 = 0.01, 0.01, 0.01, 10.0, 0.1
    t = np.linspace(0, 100, 1001)
    s = assoc(t, 0.0, c0, Kd, koff, ktr, smax)
    slope = np.polyfit(t, s, 1)[0]
    assert np.isclose(slope, ktr * c0, rtol=0.05, atol=0)


def test_rebinding_lowers_apparent_off_rate():
    """Late in dissociation (S << Smax), rebinding slows the decay to an apparent
    off-rate of koff / (1 + kon * (Smax - S) / k_tr)."""
    Kd, koff, ktr, smax = 0.1, 0.01, 0.5, 10.0
    kon = koff / Kd
    t = np.linspace(0, 1500, 1501)
    d = disso(t, 0.05, Kd, koff, ktr, smax)
    late = t >= 300
    slope = -np.polyfit(t[late], np.log(d[late]), 1)[0]
    expected = koff / (1 + kon * (smax - d[late].mean()) / ktr)
    assert np.isclose(slope, expected, rtol=0.03, atol=0)
    assert slope < koff / 2


# Single-cycle kinetics

def test_restarting_from_returned_signal_matches_one_solve():
    """Single-cycle kinetics: solving in two pieces, restarting from the returned
    signal, must give the same curve as one solve, in both phases."""
    t = np.linspace(0, 200, 401)
    first, second = t <= 100, t >= 100  # both pieces include t = 100, where the second restarts

    s_full = assoc(t, 0.0, 1.0, KD, KOFF, KTR, SMAX)
    s1 = assoc(t[first], 0.0, 1.0, KD, KOFF, KTR, SMAX)
    s2 = assoc(t[second] - 100, s1[-1], 1.0, KD, KOFF, KTR, SMAX)
    # Each solve has rtol 1e-8, so the two paths agree far more closely than this
    assert np.allclose(np.r_[s1, s2[1:]], s_full, rtol=1e-6, atol=0)

    d_full = disso(t, s_full[-1], KD, KOFF, KTR, SMAX)
    d1 = disso(t[first], s_full[-1], KD, KOFF, KTR, SMAX)
    d2 = disso(t[second] - 100, d1[-1], KD, KOFF, KTR, SMAX)
    assert np.allclose(np.r_[d1, d2[1:]], d_full, rtol=1e-6, atol=0)


# Fits

CONCS = [0.25, 0.5, 1.0, 2.0]
T_ASSOC = np.linspace(0, 300, 151)
T_DISSO = np.linspace(0, 300, 151)


def simulate(ktr, single_cycle=False, noise=0.0, seed=0):
    """Traces for CONCS; single-cycle runs carry the signal from one step to the next."""
    rng = np.random.default_rng(seed)
    out = {"assoc_signal_lst": [], "assoc_time_lst": [], "disso_signal_lst": [], "disso_time_lst": []}
    s0 = 0.0
    for i, c0 in enumerate(CONCS):
        offset = i * (T_ASSOC[-1] + T_DISSO[-1]) if single_cycle else 0.0
        a = assoc(T_ASSOC, s0, c0, KD, KOFF, ktr, SMAX)
        d = disso(T_DISSO, a[-1], KD, KOFF, ktr, SMAX)
        if single_cycle:
            s0 = d[-1]
        out["assoc_signal_lst"].append(a + rng.normal(0, noise, a.size))
        out["assoc_time_lst"].append(T_ASSOC + offset)
        out["disso_signal_lst"].append(d + rng.normal(0, noise, d.size))
        out["disso_time_lst"].append(T_DISSO + offset + T_ASSOC[-1])
    return out


def fit(data, with_errors=False):
    # Same starting point as KineticsFitter: k_tr = kon * Smax from the Kd, koff and Smax guesses
    kd_guess, koff_guess, smax_guess = 1.0, 0.02, 4.0
    ktr_guess = koff_guess / kd_guess * smax_guess
    params, cov, _, _ = fit_one_site_assoc_and_disso_ktr(
        **data,
        analyte_conc_lst=CONCS,
        initial_parameters=[kd_guess, koff_guess, ktr_guess, smax_guess],
        low_bounds=[1e-3, 1e-4, ktr_guess * 1e-4, 0.5],
        high_bounds=[100.0, 1.0, ktr_guess * 1e4, 50.0],
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
    """On data with 1% noise and a moderate transport limit (kon * Smax / k_tr = 2),
    the fit recovers Kd, koff, k_tr and Smax within 3 standard errors, and those
    errors stay small (under 25%).

    With a strong limit (k_tr = 0.005, ratio 20) binding is set by transport and koff
    barely affects the curves, so noisy data cannot determine it."""
    ktr = 0.05
    # Kd and Smax are correlated, so a fixed tolerance would fail by chance
    p, err = fit(simulate(ktr, noise=0.01 * SMAX, seed=seed), with_errors=True)
    for name, true in [("Kd", KD), ("koff", KOFF), ("ktr", ktr), ("smax", SMAX)]:
        assert abs(p[name] - true) < 3 * err[name], (name, p[name], err[name])
        assert err[name] < 0.25 * true, (name, err[name])


def test_fit_does_not_invent_transport_limit():
    """On data with no transport limit, the fit still recovers Kd and koff and
    the fitted k_tr is too large to matter."""
    p = fit(simulate(ktr=1e3))
    assert np.isclose(p["Kd"], KD, rtol=0.05, atol=0)
    assert np.isclose(p["koff"], KOFF, rtol=0.05, atol=0)
    # kon * Smax / k_tr << 1: the fitted transport limit is negligible
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
