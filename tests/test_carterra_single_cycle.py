"""Single-cycle import must preserve responses, chronology and occupied states."""

import csv

import numpy as np
import pytest

from pykingenie import CarterraExperiment, KineticsAnalyzer
from pykingenie.utils.processing import detect_time_list_continuos


def write_cycles(path):
    time = np.arange(0, 50, 0.1)
    concentrations = [0.1, 0.4, 1.0]
    state = 0.0
    responses = []
    for concentration in concentrations:
        response = state * np.exp(-0.02 * np.minimum(time, 10))
        initial = state * np.exp(-0.02 * 10)
        equilibrium = 10 * concentration / (0.2 + concentration)
        rate = 0.02 * (1 + concentration / 0.2)
        association = (time >= 10) & (time < 30)
        response[association] = equilibrium + (initial - equilibrium) * np.exp(
            -rate * (time[association] - 10)
        )
        end = equilibrium + (initial - equilibrium) * np.exp(-rate * 20)
        response[time >= 30] = end * np.exp(-0.02 * (time[time >= 30] - 30))
        state = end * np.exp(-0.02 * 20)
        responses.append(response)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["Label"] + [f"{axis}{i + 1}" for i in range(len(time)) for axis in "XY"]
        )
        for i, response in enumerate(responses):
            writer.writerow(
                [f"cycle {i + 1}"] + list(np.column_stack([time, response]).ravel())
            )
    return time, concentrations, responses


def read_cycles(path, concentrations, **kwargs):
    experiment = CarterraExperiment("sequence")
    experiment.read_csv(
        path,
        association_start=10,
        association_end=30,
        concentrations=concentrations,
        single_cycle=True,
        cycle_duration=50,
        **kwargs,
    )
    return experiment


def test_rows_become_one_clock_and_one_continuous_response(tmp_path):
    path = tmp_path / "cycles.csv"
    time, concentrations, responses = write_cycles(path)
    experiment = read_cycles(path, concentrations)
    assert experiment.no_sensors == 1
    assert experiment.df_steps.Type.tolist() == [
        "BASELINE",
        "ASSOC",
        "DISASSOC",
        "ASSOC",
        "DISASSOC",
        "ASSOC",
        "DISASSOC",
    ]
    np.testing.assert_allclose(
        np.concatenate(experiment.xs[0]),
        np.concatenate([time + i * 50 for i in range(3)]),
    )
    np.testing.assert_array_equal(
        np.concatenate(experiment.ys[0]), np.concatenate(responses)
    )
    assert experiment.ligand_conc_df.Source_curve.tolist() == [
        "cycle 1",
        "cycle 2",
        "cycle 3",
    ]
    # The next pre-association baseline extends the preceding dissociation.
    assert experiment.xs[0][2][-1] == pytest.approx(59.9)
    before = [experiment.ys[0][i].copy() for i in [1, 3, 5]]
    experiment.align_association(inplace=True)
    for i, signal in zip([1, 3, 5], before):
        np.testing.assert_array_equal(experiment.ys[0][i], signal)
    assert experiment.ys[0][3][0] > 0


def test_existing_fitter_recovers_single_cycle_parameters(tmp_path):
    path = tmp_path / "cycles.csv"
    _, concentrations, _ = write_cycles(path)
    experiment = read_cycles(path, concentrations)
    experiment.align_association(inplace=True)
    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(experiment, experiment.name)
    analyzer.merge_ligand_conc_df()
    analyzer.generate_fittings(analyzer.combined_ligand_conc_df.copy())
    fitter = next(iter(analyzer.fittings.values()))
    assert fitter.is_single_cycle
    assert all(detect_time_list_continuos(fitter.time_assoc_lst, fitter.time_disso_lst))
    analyzer.submit_steady_state_fitting()
    fitter.fit_one_site_assoc_and_disso(shared_smax=True, calc_loo=False)
    assert fitter.Kd == pytest.approx(0.2, rel=0.05)
    assert fitter.k_off == pytest.approx(0.02, rel=0.05)
    assert fitter.Smax[0] == pytest.approx(10, rel=0.05)


@pytest.mark.parametrize("period", [None, 0, -1, np.nan, "unknown", 10])
def test_invalid_sequence_is_transactional(tmp_path, period):
    path = tmp_path / "cycles.csv"
    _, concentrations, _ = write_cycles(path)
    experiment = CarterraExperiment()
    experiment.read_csv(
        path, association_start=10, association_end=30, concentrations=concentrations
    )
    original = experiment.xs
    with pytest.raises(ValueError, match="cycle_duration|overlap"):
        experiment.read_csv(
            path,
            association_start=10,
            association_end=30,
            concentrations=concentrations,
            single_cycle=True,
            cycle_duration=period,
        )
    assert experiment.xs is original


def test_single_cycle_requires_one_sample_and_explicit_mode(tmp_path):
    path = tmp_path / "cycles.csv"
    _, concentrations, _ = write_cycles(path)
    with pytest.raises(ValueError, match="one sample_id"):
        read_cycles(path, concentrations, sample_id=["one", "two", "three"])
    with pytest.raises(ValueError, match="requires single_cycle"):
        CarterraExperiment().read_csv(path, association_end=30, cycle_duration=50)
    with pytest.raises(ValueError, match="boolean"):
        CarterraExperiment().read_csv(path, association_end=30, single_cycle="yes")
    with pytest.warns(UserWarning, match="concentrations"):
        with pytest.raises(ValueError, match="Provide concentrations"):
            read_cycles(path, None)
