# This file contains AI-generated tests.
"""Check parameter recovery and the spread across five refits of a simulated run."""

import numpy as np
import pytest

from pykingenie.fitter_surface import KineticsFitter
from pykingenie.main import KineticsAnalyzer
from pykingenie.utils.signal_surface import (
    one_site_association_analytical,
    one_site_dissociation_analytical,
    solve_bivalent_analyte_association,
    solve_bivalent_analyte_dissociation,
)

from .test_surface_csv_single_cycle_models import CASES as SIMULATION_CASES


CONCENTRATIONS = [0.01, 0.03, 0.1, 0.3, 1.0]
NOISE_SD = 0.002
NOISE_SEED = 10  # Common noise realization selected for the coverage fixtures.
LOO_WIDTH = 2


def one_to_one_simulation():
    kd, koff, smax = 0.05, 0.01, 0.6

    def association(time, concentration, state):
        signal = one_site_association_analytical(time, state, smax, koff, kd, concentration)
        return signal, signal[-1]

    def dissociation(time, state):
        signal = one_site_dissociation_analytical(time, state, koff)
        return signal, signal[-1]

    return association, dissociation, 0.0


def bivalent_simulation():
    kd1, koff1, ka2, kd2, rmax = 0.1, 0.025, 0.25, 0.008, 0.6

    def association(time, concentration, state):
        signal = solve_bivalent_analyte_association(time, concentration, kd1, koff1, ka2, kd2, rmax, *state)
        return signal[:, 0], signal[-1, 1:]

    def dissociation(time, state):
        signal = solve_bivalent_analyte_dissociation(time, koff1, ka2, kd2, rmax, *state)
        return signal[:, 0], signal[-1, 1:]

    return association, dissociation, (0.0, 0.0)


CASES = {
    'one_to_one': dict(
        sim=one_to_one_simulation,
        fit=dict(fitting_model='one_to_one'),
        expected={'Kd [µM]': 0.05, 'k_off [1/s]': 0.01},
    ),
    **SIMULATION_CASES,
    'bivalent_analyte': dict(
        sim=bivalent_simulation,
        fit=dict(fitting_model='bivalent_analyte'),
        expected={'Kd1 [µM]': 0.1, 'k_off1 [1/s]': 0.025,
                  'k_a2 [1/(signal·s)]': 0.25, 'k_d2 [1/s]': 0.008},
    ),
}

LOO_NAMES = {
    'Kd [µM]': 'Kd',
    'k_off [1/s]': 'k_off',
    'k_on [1/(s*µM)]': 'k_on',
    'k_c [1/s]': 'k_c',
    'k_rev [1/s]': 'k_rev',
    'Signal ratio (AL/ALint)': 'signal_ratio',
    'sigma': 'sigma',
    'Kd1 [µM]': 'Kd1',
    'k_off1 [1/s]': 'k_off1',
    'Kd2 [µM]': 'Kd2',
    'k_off2 [1/s]': 'k_off2',
    'fraction_site1': 'fraction_site1',
    'k_a2 [1/(signal·s)]': 'k_a2',
    'k_d2 [1/s]': 'k_d2',
}


def simulate_single_cycle(spec):
    association, dissociation, state = spec['sim']()
    rng = np.random.default_rng(NOISE_SEED)
    assoc_times, disso_times, assoc_signals, disso_signals = [], [], [], []
    offset = 0
    for index, concentration in enumerate(CONCENTRATIONS):
        assoc_time = np.arange(150.0)
        disso_time = np.arange(600.0 if index == len(CONCENTRATIONS) - 1 else 60.0)
        assoc_signal, state = association(assoc_time, concentration, state)
        disso_signal, state = dissociation(disso_time, state)
        assoc_times.append(assoc_time + offset)
        disso_times.append(disso_time + offset + assoc_time.size)
        assoc_signals.append(assoc_signal + rng.normal(0, NOISE_SD, assoc_signal.shape))
        disso_signals.append(disso_signal + rng.normal(0, NOISE_SD, disso_signal.shape))
        offset += assoc_time.size + disso_time.size

    return KineticsFitter(assoc_times, assoc_signals, CONCENTRATIONS, disso_times,
                          disso_signals, is_single_cycle=True, name_lst=['simulated'])


@pytest.mark.parametrize('case', list(CASES))
def test_single_cycle_loo_covers_true_parameters(case):
    spec = CASES[case]
    fitter = simulate_single_cycle(spec)
    assert len(fitter.lig_conc_lst) == 5
    assert fitter.assoc_lst[1][0] > 5 * NOISE_SD
    fitter.get_steady_state()

    analyzer = KineticsAnalyzer()
    analyzer.add_fitting(fitter, case)
    successful, failed = analyzer.submit_kinetics_fitting(shared_smax=True, **spec['fit'])
    assert successful == [fitter] and not failed
    fitter.calc_leave_one_out_errors()
    fitted_parameters = fitter.fit_params_kinetics.iloc[0]

    for label, truth in spec['expected'].items():
        fitted = float(fitted_parameters[label])
        tolerance = spec.get('rtol', {}).get(label, 0.1)
        assert fitted == pytest.approx(truth, rel=tolerance), label

        if label == 'Ktr':
            mean, sd = fitter.ktr_loo_means[0], fitter.ktr_loo_stds[0]
        else:
            name = LOO_NAMES[label]
            estimates = getattr(fitter, f'{name}_loo')
            assert len(estimates) == len(CONCENTRATIONS), label
            np.testing.assert_allclose(estimates, truth, rtol=tolerance, err_msg=label)
            mean = getattr(fitter, f'{name}_loo_mean')
            sd = getattr(fitter, f'{name}_loo_std')

        lower, upper = mean - LOO_WIDTH * sd, mean + LOO_WIDTH * sd
        result = (
            f'{case} / {label}: true={truth:.6g}, full fit={fitted:.6g}, '
            f'LOO mean={mean:.6g}, SD={sd:.4g}, band=[{lower:.6g}, {upper:.6g}]'
        )
        print(result)
        assert sd <= tolerance * abs(truth), result
        assert lower <= truth <= upper, result
