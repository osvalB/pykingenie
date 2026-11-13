# Note: this file contains AI-generated code, written with AI coding assistants (PR #24).
"""
Single-cycle parameter recovery for every kinetic model, using the steps/traces file format.

For each model a single-cycle titration (one sensor, five increasing analyte
concentrations, short dissociations in between and a long final one) is
simulated with the model's own signal functions, written as a steps file plus a
column-formatted traces file, imported with SurfaceCsvExperiment and fitted
with a global Smax. The state of every bound species is carried over from one
cycle to the next, so these tests check that each fitter treats single-cycle
data consistently with its model.
"""

import numpy as np
import pytest

from pykingenie import SurfaceCsvExperiment
from pykingenie.main import KineticsAnalyzer
from pykingenie.utils.signal_surface import (
    solve_ode_one_site_mass_transport_association,
    solve_ode_one_site_mass_transport_dissociation,
    solve_induced_fit_association,
    solve_induced_fit_dissociation,
    solve_two_site_association,
    solve_two_site_dissociation,
    solve_two_site_cooperative_association,
    solve_two_site_cooperative_dissociation,
    solve_two_site_heterogeneous_ligand_association,
    solve_two_site_heterogeneous_ligand_dissociation,
)

CONCS      = [0.01, 0.03, 0.1, 0.3, 1.0]  # µM
T_ASSOC    = 150                          # s
T_DISSO    = 60                           # s, between associations
T_DISSO_END = 600                         # s, final dissociation
DT         = 1.0
NOISE_SD   = 0.002
LOAD_LEVEL = 1.0


# --- Phase simulators: each returns (signal, new_state) ---------------------

def mtl_model(p):
    def assoc(t, c, s):
        y = solve_ode_one_site_mass_transport_association(t, s, c, p['Kd'], p['koff'], p['ktr'], p['smax'])
        return y, y[-1]
    def disso(t, s):
        y = solve_ode_one_site_mass_transport_dissociation(t, s, p['Kd'], p['koff'], p['ktr'], p['smax'])
        return y, y[-1]
    return assoc, disso, 0.0


def induced_fit_model(p):
    def assoc(t, c, state):
        m = solve_induced_fit_association(t, c, p['kon'], p['koff'], p['kc'], p['krev'],
                                          sP1L=state[0], sP2L=state[1], smax=p['smax'])
        return m[:, 0], (m[-1, 1], m[-1, 2])
    def disso(t, state):
        s0 = state[0] + state[1]
        m = solve_induced_fit_dissociation(t, p['koff'], p['kc'], p['krev'], s0=s0, sP2L=state[1], smax=p['smax'])
        return m[:, 0], (m[-1, 1], m[-1, 2])
    return assoc, disso, (0.0, 0.0)


def two_site_model(p, cooperative):
    kon = p['koff'] / p['Kd']
    def to_fractions(row):
        return row[1] / p['Rmax_PL'], row[2] / p['Rmax_LPL']
    def assoc(t, c, state):
        if cooperative:
            m = solve_two_site_cooperative_association(t, c, kon, p['koff'], p['sigma'], Rmax_PL=p['Rmax_PL'],
                                                       Rmax_LPL=p['Rmax_LPL'], fPL_0=state[0], fLPL_0=state[1])
        else:
            m = solve_two_site_association(t, c, kon, p['koff'], Rmax_PL=p['Rmax_PL'],
                                           Rmax_LPL=p['Rmax_LPL'], fPL_0=state[0], fLPL_0=state[1])
        return m[:, 0], to_fractions(m[-1])
    def disso(t, state):
        if cooperative:
            m = solve_two_site_cooperative_dissociation(t, p['koff'], p['sigma'], Rmax_PL=p['Rmax_PL'],
                                                        Rmax_LPL=p['Rmax_LPL'], fPL_0=state[0], fLPL_0=state[1])
        else:
            m = solve_two_site_dissociation(t, p['koff'], Rmax_PL=p['Rmax_PL'],
                                            Rmax_LPL=p['Rmax_LPL'], fPL_0=state[0], fLPL_0=state[1])
        return m[:, 0], to_fractions(m[-1])
    return assoc, disso, (0.0, 0.0)


def heterogeneous_ligand_model(p):
    def assoc(t, c, state):
        m = solve_two_site_heterogeneous_ligand_association(t, c, p['Kd1'], p['koff1'], p['Kd2'], p['koff2'],
                                                            Rmax=p['Rmax'], fraction_site1=p['fraction_site1'],
                                                            s1_0=state[0], s2_0=state[1])
        return m[:, 0], (m[-1, 1], m[-1, 2])
    def disso(t, state):
        m = solve_two_site_heterogeneous_ligand_dissociation(t, p['koff1'], p['koff2'],
                                                             fraction_site1=p['fraction_site1'],
                                                             s1_0=state[0], s2_0=state[1])
        return m[:, 0], (m[-1, 1], m[-1, 2])
    return assoc, disso, (0.0, 0.0)


# --- Test cases: true parameters, fit options and expected fitted values -----

CASES = {
    'one_to_one_mtl': dict(
        sim=lambda: mtl_model(dict(Kd=0.05, koff=0.01, ktr=0.1, smax=0.6)),
        fit=dict(fitting_model='one_to_one_mtl'),
        expected={'Kd [µM]': 0.05, 'k_off [1/s]': 0.01, 'Ktr': 0.1},
        rtol={'Ktr': 0.2}),
    'one_to_one_if': dict(
        sim=lambda: induced_fit_model(dict(kon=0.2, koff=0.02, kc=0.005, krev=0.0005, smax=0.6)),
        fit=dict(fitting_model='one_to_one_if'),
        expected={'k_on [1/(s*µM)]': 0.2, 'k_off [1/s]': 0.02, 'k_c [1/s]': 0.005, 'k_rev [1/s]': 0.0005},
        rtol={'k_rev [1/s]': 0.2}),
    'two_to_one': dict(
        sim=lambda: two_site_model(dict(Kd=0.05, koff=0.01, Rmax_PL=0.35, Rmax_LPL=0.6), cooperative=False),
        fit=dict(fitting_model='two_to_one', fit_sigma=False),
        expected={'Kd [µM]': 0.05, 'k_off [1/s]': 0.01},
        rtol={}),
    'two_to_one_cooperative': dict(
        sim=lambda: two_site_model(dict(Kd=0.05, koff=0.01, sigma=3.0, Rmax_PL=0.35, Rmax_LPL=0.6), cooperative=True),
        fit=dict(fitting_model='two_to_one', fit_sigma=True),
        expected={'Kd [µM]': 0.05, 'k_off [1/s]': 0.01, 'sigma': 3.0},
        rtol={'sigma': 0.2}),
    'heterogeneous_ligand': dict(
        sim=lambda: heterogeneous_ligand_model(dict(Kd1=0.01, koff1=0.001, Kd2=0.1, koff2=0.02,
                                                    Rmax=0.6, fraction_site1=0.6)),
        fit=dict(fitting_model='heterogeneous_ligand'),
        expected={'Kd1 [µM]': 0.01, 'k_off1 [1/s]': 0.001, 'Kd2 [µM]': 0.1,
                  'k_off2 [1/s]': 0.02, 'fraction_site1': 0.6},
        rtol={}),
}

DEFAULT_RTOL = 0.1


def write_single_cycle_files(folder, assoc, disso, state, seed=0):
    """Simulate a single-cycle titration and write it as steps + traces files."""

    steps = [('Baseline', 60, 'Baseline', 'p1A1', ''),
             ('Loading', 120, 'Loading', 'p1A2', ''),
             ('Baseline2', 60, 'Baseline', 'p1A3', '')]
    for k, c in enumerate(CONCS):
        steps.append((f'Association{k + 1}', T_ASSOC, 'Association', f'p1A{4 + k}', c))
        last = k == len(CONCS) - 1
        steps.append((f'Dissociation{k + 1}', T_DISSO_END if last else T_DISSO, 'Dissociation', 'p1A9', ''))

    steps_file = folder / 'steps.txt'
    with open(steps_file, 'w') as f:
        f.write('Step Name\tStep Time\tStep Type Name\tSample Location\tConcentration\n')
        for name, t, typ, loc, conc in steps:
            f.write(f'{name}\t{t}\t{typ}\t{loc}\t{conc}\n')

    signal = []
    for name, dur, typ, loc, conc in steps:
        t = np.arange(0, dur, DT)
        if typ == 'Loading':
            y = LOAD_LEVEL * (1 - np.exp(-0.05 * t)) / (1 - np.exp(-0.05 * dur))
        elif typ == 'Association':
            b, state = assoc(t, conc, state)
            y = LOAD_LEVEL + b
        elif typ == 'Dissociation':
            b, state = disso(t, state)
            y = LOAD_LEVEL + b
        else:
            y = np.full(t.size, 0.0 if name == 'Baseline' else LOAD_LEVEL)
        signal.append(y)

    signal = np.concatenate(signal)
    signal = signal + np.random.default_rng(seed).normal(0, NOISE_SD, signal.size)
    time = np.arange(signal.size) * DT

    traces_file = folder / 'traces.csv'
    with open(traces_file, 'w') as f:
        f.write(' Data Alignment: None,\n')
        f.write('X Values, A1\n')
        for t, y in zip(time, signal):
            f.write(f'{t:g},{y:.6f}\n')

    return str(steps_file), str(traces_file)


@pytest.mark.parametrize('case', list(CASES.keys()))
def test_single_cycle_parameter_recovery(case, tmp_path):

    spec = CASES[case]
    assoc, disso, state0 = spec['sim']()
    steps_file, traces_file = write_single_cycle_files(tmp_path, assoc, disso, state0)

    exp = SurfaceCsvExperiment(case)
    exp.read_files(steps_file, traces_file)  # concentrations from the steps file
    exp.align_association(exp.sensor_names, inplace=True)

    assert exp.ligand_conc_df['Concentration_micromolar'].tolist() == CONCS

    ka = KineticsAnalyzer()
    ka.add_experiment(exp, case)
    ka.merge_ligand_conc_df()
    ka.generate_fittings(ka.combined_ligand_conc_df.copy())

    fitter = list(ka.fittings.values())[0]
    assert fitter.is_single_cycle

    _, failed = ka.submit_kinetics_fitting(shared_smax=True, **spec['fit'])
    assert not failed, f'{case}: fit failed'

    params = fitter.fit_params_kinetics.iloc[0]

    for name, true_value in spec['expected'].items():
        rtol = spec['rtol'].get(name, DEFAULT_RTOL)
        fitted = float(params[name])
        assert np.isclose(fitted, true_value, rtol=rtol), \
            f'{case}: {name} fitted {fitted:.4g}, expected {true_value:.4g} (rtol {rtol})'

    # The fit must also reproduce the data to within the noise level
    residuals = np.concatenate([np.concatenate(fitter.assoc_lst) - np.concatenate(fitter.signal_assoc_fit),
                                np.concatenate(fitter.disso_lst) - np.concatenate(fitter.signal_disso_fit)])
    assert np.sqrt(np.mean(residuals ** 2)) < 2 * NOISE_SD
