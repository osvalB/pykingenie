# This file is AI generated.
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from pykingenie.fitter_surface import KineticsFitter
from pykingenie.main import KineticsAnalyzer
from pykingenie.utils.signal_surface import (
    ode_bivalent_analyte,
    solve_bivalent_analyte_association,
    solve_bivalent_analyte_dissociation,
    one_site_association_analytical,
)

TRUTH = dict(Kd1=2.0, koff1=0.05, ka2=0.002, kd2=0.01, Rmax=40.0)
CONCS = [0.5, 2.0, 8.0, 16.0]
TIME_ASSOC = np.arange(0, 180.1, 2.0)
TIME_DISSO = np.arange(182, 500.1, 2.0)
# Small grid that contains a start near the truth, to keep the tests fast
COARSE_GRID = dict(ka2_rmax_values=[0.01, 0.1, 1.0], kd2_factors=[1.0], koff1_factors=[1.0, 10.0])


def _simulate(p=TRUTH, noise=0.0, seed=0, rmax_values=None, single_cycle=False):
    rng = np.random.default_rng(seed)
    t_assoc_lst, t_disso_lst, assoc_lst, disso_lst = [], [], [], []
    x0, offset = (0.0, 0.0), 0.0
    for i, conc in enumerate(CONCS):
        rmax = p['Rmax'] if rmax_values is None else rmax_values[i]
        if not single_cycle:
            x0 = (0.0, 0.0)
        assoc = solve_bivalent_analyte_association(TIME_ASSOC, conc, p['Kd1'], p['koff1'],
                                                   p['ka2'], p['kd2'], rmax, *x0)
        disso = solve_bivalent_analyte_dissociation(TIME_DISSO, p['koff1'], p['ka2'], p['kd2'],
                                                    rmax, *assoc[-1, 1:])
        x0 = tuple(disso[-1, 1:])
        assoc_lst.append(assoc[:, 0] + rng.normal(0, noise, TIME_ASSOC.size))
        disso_lst.append(disso[:, 0] + rng.normal(0, noise, TIME_DISSO.size))
        t_assoc_lst.append(TIME_ASSOC + offset)
        t_disso_lst.append(TIME_DISSO + offset)
        if single_cycle:
            offset += TIME_DISSO[-1] + 2.0
    return t_assoc_lst, assoc_lst, t_disso_lst, disso_lst


def _fitter(smax_id=None, single_cycle=False, trim_time=0, steady_state=True, **sim_kwargs):
    t_assoc_lst, assoc_lst, t_disso_lst, disso_lst = _simulate(single_cycle=single_cycle, **sim_kwargs)
    fitter = KineticsFitter(
        time_assoc_lst=t_assoc_lst,
        association_signal_lst=assoc_lst,
        lig_conc_lst=CONCS,
        time_diss_lst=t_disso_lst,
        dissociation_signal_lst=disso_lst,
        smax_id=[0] * len(CONCS) if smax_id is None else smax_id,
        name_lst=['bivalent'],
        is_single_cycle=single_cycle,
        trim_time=trim_time,
    )
    if steady_state:
        fitter.get_steady_state()
    return fitter


def _assert_recovered(fitter, p=TRUTH, rtol=0.02):
    for value, key in zip(fitter.params[:4], ['Kd1', 'koff1', 'ka2', 'kd2']):
        assert np.isclose(value, p[key], rtol=rtol), (key, value, p[key])


# Solver

def test_without_second_arm_is_one_to_one_with_half_kd():
    # ka2 = 0: two equivalent arms give kon = 2 * ka1 and Kd = Kd1 / 2
    t = np.linspace(0, 200, 101)
    sol = solve_bivalent_analyte_association(t, 3.0, 2.0, 0.05, 0.0, 0.01, 40.0)
    expected = one_site_association_analytical(t, 0, 40.0, 0.05, 1.0, 3.0)
    np.testing.assert_allclose(sol[:, 0], expected, atol=1e-5)
    assert np.allclose(sol[:, 2], 0)


def test_matches_reference_integration_and_conserves_ligand():
    t = np.linspace(0, 300, 61)
    Kd1, koff1, ka2, kd2, rmax, conc = 2.0, 0.05, 0.01, 0.01, 40.0, 4.0
    sol = solve_bivalent_analyte_association(t, conc, Kd1, koff1, ka2, kd2, rmax)

    ref = solve_ivp(ode_bivalent_analyte, (0, 300), [0.0, 0.0], t_eval=t, method='Radau',
                    args=(koff1 / Kd1, koff1, ka2, kd2, conc, rmax), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sol[:, 1], ref.y[0], atol=1e-5)
    np.testing.assert_allclose(sol[:, 2], ref.y[1], atol=1e-5)
    np.testing.assert_allclose(sol[:, 0], sol[:, 1] + sol[:, 2])

    free_ligand = rmax - sol[:, 1] - 2 * sol[:, 2]
    assert np.all(free_ligand >= -1e-8)


def test_t0_shift_and_dissociation_decay():
    t = np.linspace(0, 100, 51)
    shifted = solve_bivalent_analyte_association(t, 4.0, 2.0, 0.05, 0.01, 0.01, 40.0, t0=10)
    assert np.allclose(shifted[t < 10], 0)
    unshifted = solve_bivalent_analyte_association(t[t >= 10] - 10, 4.0, 2.0, 0.05, 0.01, 0.01, 40.0)
    np.testing.assert_allclose(shifted[t >= 10], unshifted, atol=1e-6)

    disso = solve_bivalent_analyte_dissociation(t, 0.05, 0.01, 0.01, 40.0, *shifted[-1, 1:])
    assert np.isclose(disso[0, 0], shifted[-1, 0])
    assert np.all(np.diff(disso[:, 0]) < 0)


def test_solver_failure_raises(monkeypatch):
    import pykingenie.utils.signal_surface as ss

    def failed_odeint(fun, y0, t, **kwargs):
        return np.zeros((len(t), 2)), {'message': 'Excess work done on this call.'}

    monkeypatch.setattr(ss, 'odeint', failed_odeint)
    with pytest.raises(RuntimeError):
        solve_bivalent_analyte_association(np.linspace(0, 10, 5), 1.0, 1.0, 0.1, 0.01, 0.01, 10.0)


# Fitting

def test_fit_recovers_parameters():
    fitter = _fitter(noise=0.05)
    fitter.fit_bivalent_analyte_assoc_and_disso(**COARSE_GRID)

    assert fitter.model == 'bivalent_analyte_association_and_dissociation'
    _assert_recovered(fitter)
    assert np.isclose(fitter.Smax[0], TRUTH['Rmax'], rtol=0.02)

    df = fitter.fit_params_kinetics
    for col in ['Kd1 [µM]', 'k_off1 [1/s]', 'k_a2 [1/(signal·s)]', 'k_d2 [1/s]', 'Rmax',
                '(Derived) k_on1 [1/µM/s]', '(Derived) Kd2 [signal]']:
        assert col in df.columns
    assert np.isclose(df['(Derived) k_on1 [1/µM/s]'].iloc[0], fitter.k_off1 / fitter.Kd1)
    assert np.isclose(df['(Derived) Kd2 [signal]'].iloc[0], fitter.k_d2 / fitter.k_a2)

    grid = fitter.bivalent_analyte_grid_search
    assert isinstance(grid, pd.DataFrame)
    assert len(grid) == 6
    assert list(grid.columns) == ['koff1_start', 'ka2_value', 'kd2_value', 'rss']

    assert 'bivalent_analyte_association_and_dissociation' in fitter.fit_results


def test_fit_strong_avidity():
    p = {**TRUTH, 'ka2': 0.05, 'kd2': 0.005}
    fitter = _fitter(p=p, noise=0.05)
    fitter.fit_bivalent_analyte_assoc_and_disso(
        ka2_rmax_values=[0.1, 1.0, 10.0], kd2_factors=[1.0], koff1_factors=[1.0, 10.0, 100.0])
    _assert_recovered(fitter, p, rtol=0.05)


def test_fit_rmax_per_curve_and_t0():
    rmax_values = [38.0, 40.0, 42.0, 39.0]
    fitter = _fitter(rmax_values=rmax_values)
    fitter.fit_bivalent_analyte_assoc_and_disso(shared_smax=False, fixed_t0=False, **COARSE_GRID)
    _assert_recovered(fitter)
    np.testing.assert_allclose(fitter.Smax, rmax_values, rtol=0.02)
    assert any(col.startswith('t0') for col in fitter.fit_params_kinetics.columns)


def test_fit_shared_t0_without_steady_state():
    fitter = _fitter(steady_state=False)
    fitter.fit_bivalent_analyte_assoc_and_disso(fixed_t0=False, **COARSE_GRID)
    _assert_recovered(fitter)
    assert abs(fitter.fit_params_kinetics['t0'].iloc[0]) < 0.01


def test_fit_function_default_smax_index():
    from pykingenie.utils.fitting_surface import fit_bivalent_analyte_assoc_and_disso

    t_assoc_lst, assoc_lst, t_disso_lst, disso_lst = _simulate()
    p = TRUTH
    # Without smax_idx, each curve gets its own Rmax
    p0 = [p['Kd1'], p['koff1'], p['ka2'], p['kd2']] + [p['Rmax']] * len(CONCS)
    params, _, fit_assoc, _ = fit_bivalent_analyte_assoc_and_disso(
        assoc_lst, t_assoc_lst, CONCS, disso_lst, t_disso_lst,
        p0, [x / 10 for x in p0], [x * 10 for x in p0])
    assert len(params) == 4 + len(CONCS)
    np.testing.assert_allclose(fit_assoc[-1], assoc_lst[-1], atol=1e-4)


def test_fit_single_cycle_and_trim():
    fitter = _fitter(single_cycle=True)
    fitter.fit_bivalent_analyte_assoc_and_disso(**COARSE_GRID)
    _assert_recovered(fitter)

    fitter = _fitter(trim_time=4)
    fitter.fit_bivalent_analyte_assoc_and_disso(**COARSE_GRID)
    _assert_recovered(fitter)


def test_final_fits_start_from_distinct_ka2_kd2_pairs(monkeypatch):
    import pykingenie.fitter_surface as fs

    original = fs.fit_bivalent_analyte_assoc_and_disso
    final_starts = []

    def record(*args, **kwargs):
        if not kwargs.get('fixed_ka2', False):
            final_starts.append(tuple(args[5][2:4]))
        return original(*args, **kwargs)

    monkeypatch.setattr(fs, 'fit_bivalent_analyte_assoc_and_disso', record)
    fitter = _fitter(noise=0.05)
    # Two k_off1 starts per (k_a2, k_d2) pair often converge to the same grid point
    fitter.fit_bivalent_analyte_assoc_and_disso(n_starts=3, **COARSE_GRID)
    assert len(final_starts) == 3
    assert len(set(final_starts)) == 3


def test_fit_fast_kd1_slow_kd2():
    # Fast first arm held by a slow second arm: the observed dissociation is
    # hundreds of times slower than k_off1 (the Duke CH31/CH505 well 7 regime)
    p = dict(Kd1=28.0, koff1=0.08, ka2=1.3e-4, kd2=7.6e-5, Rmax=550.0)
    fitter = _fitter(p=p, noise=0.5)
    fitter.fit_bivalent_analyte_assoc_and_disso(
        ka2_rmax_values=[0.01, 0.1], kd2_factors=[0.1, 1.0], koff1_factors=[1.0, 1000.0])
    _assert_recovered(fitter, p, rtol=0.05)


def test_fit_leave_one_out():
    fitter = _fitter(noise=0.05)
    fitter.fit_bivalent_analyte_assoc_and_disso(calc_loo=True, **COARSE_GRID)
    for name in ['Kd1', 'k_off1', 'k_a2', 'k_d2']:
        assert len(getattr(fitter, f'{name}_loo')) == len(CONCS)
        assert np.isclose(getattr(fitter, f'{name}_loo_mean'), getattr(fitter, name), rtol=0.05)
    assert isinstance(fitter.df_fit_loo, pd.DataFrame)

    # The generic dispatch runs the same calculation
    fitter.calc_leave_one_out_errors()
    assert np.isclose(fitter.Kd1_loo_mean, fitter.Kd1, rtol=0.05)


def test_grid_failure_raises(monkeypatch):
    import pykingenie.fitter_surface as fs

    def fail(*args, **kwargs):
        raise RuntimeError('solver failed')

    fitter = _fitter()
    monkeypatch.setattr(fs, 'fit_bivalent_analyte_assoc_and_disso', fail)
    with pytest.raises(RuntimeError):
        fitter.fit_bivalent_analyte_assoc_and_disso(**COARSE_GRID)


def test_final_fit_failure_raises_and_loo_skips_failed_fits(monkeypatch):
    import pykingenie.fitter_surface as fs

    original = fs.fit_bivalent_analyte_assoc_and_disso

    def fail_without_fixed_ka2(*args, **kwargs):
        if not kwargs.get('fixed_ka2', False):
            raise RuntimeError('solver failed')
        return original(*args, **kwargs)

    fitter = _fitter()
    monkeypatch.setattr(fs, 'fit_bivalent_analyte_assoc_and_disso', fail_without_fixed_ka2)
    with pytest.raises(RuntimeError, match='solver failed'):
        fitter.fit_bivalent_analyte_assoc_and_disso(**COARSE_GRID)

    # A fitted model whose leave-one-out refits all fail gives empty estimates
    monkeypatch.setattr(fs, 'fit_bivalent_analyte_assoc_and_disso', original)
    fitter.fit_bivalent_analyte_assoc_and_disso(**COARSE_GRID)
    monkeypatch.setattr(fs, 'fit_bivalent_analyte_assoc_and_disso', fail_without_fixed_ka2)
    fitter.bivalent_analyte_assoc_and_disso_loo()
    assert len(fitter.Kd1_loo) == 0


def test_submit_kinetics_fitting_bivalent_analyte_routing():
    # Runs the default grid
    fitter = _fitter()
    ka = KineticsAnalyzer()
    ka.add_fitting(fitter, 'bivalent')
    ka.submit_kinetics_fitting(fitting_model='bivalent_analyte',
                               fitting_region='association_dissociation',
                               shared_smax=True)
    assert fitter.model == 'bivalent_analyte_association_and_dissociation'
    _assert_recovered(fitter)
