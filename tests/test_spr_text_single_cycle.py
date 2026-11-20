"""SprTextExperiment(single_cycle=True): several injections on one exported curve."""

import csv

import numpy as np
import pytest

from pykingenie import KineticsAnalyzer, SprTextExperiment
from pykingenie.utils.processing import detect_time_list_continuos

KD, KOFF = 0.1, 0.01
CONCENTRATIONS = [0.025, 0.05, 0.1, 0.2, 0.4]
STARTS = [60.0 + 300 * i for i in range(5)]
ENDS = [start + 120 for start in STARTS]
SMAX = {"Fc=2-1": 50.0, "Fc=4-3": 30.0}


def single_cycle(time, smax):
    """Noise-free 1:1 response to the five injections, bound state carried over."""
    kon = KOFF / KD
    response = np.zeros_like(time)
    level, last = 0.0, 0.0
    for start, end, conc in zip(STARTS + [np.inf], ENDS + [np.inf], CONCENTRATIONS + [0]):
        # Dissociation from the previous injection end until this start.
        gap = (time >= last) & (time < start)
        response[gap] = level * np.exp(-KOFF * (time[gap] - last))
        if np.isinf(start):
            break
        level *= np.exp(-KOFF * (start - last))
        plateau = smax * conc / (conc + KD)
        kobs = kon * conc + KOFF
        on = (time >= start) & (time < end)
        response[on] = plateau + (level - plateau) * np.exp(-kobs * (time[on] - start))
        level = plateau + (level - plateau) * np.exp(-kobs * (end - start))
        last = end
    return response


def write_export(path):
    time = np.arange(0, 1800, 1.0)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow([f"Cycle=1_{fc}_{axis}" for fc in SMAX for axis in "XY"])
        responses = [single_cycle(time, smax) for smax in SMAX.values()]
        for row in zip(time, *responses):
            writer.writerow([value for r in row[1:] for value in (row[0], r)])
    return path


def read(path, **kwargs):
    settings = dict(
        single_cycle=True,
        association_start=STARTS,
        association_end=ENDS,
        concentrations=CONCENTRATIONS,
        sample_id={"Cycle=1_Fc=2-1": "low", "Cycle=1_Fc=4-3": "high"},
    )
    settings.update(kwargs)
    experiment = SprTextExperiment("sck")
    experiment.read_file(path, **settings)
    return experiment


def test_each_curve_is_split_into_one_step_pair_per_injection(tmp_path):
    experiment = read(write_export(tmp_path / "sck.txt"))
    steps = experiment.df_steps
    assert steps.Type.tolist() == ["BASELINE"] + ["ASSOC", "DISASSOC"] * 5
    assert steps.Name.tolist()[1:3] == ["Association 1", "Dissociation 1"]
    assert steps.Column_location.tolist() == ["0", "1", "1", "2", "2", "3", "3", "4", "4", "5", "5"]
    assert steps.Start.tolist()[1:] == [t for pair in zip(STARTS, ENDS) for t in pair]
    assert experiment.no_steps == 11 and experiment.no_sensors == 2
    for xs in experiment.xs:
        assert len(xs) == 11
        # Each dissociation lasts until the next injection starts.
        assert xs[2][0] == ENDS[0] and xs[2][-1] == STARTS[1] - 1
        assert xs[-1][-1] == 1799
    conc = experiment.ligand_conc_df
    assert conc.Sensor.tolist() == ["Cycle=1_Fc=2-1"] * 5 + ["Cycle=1_Fc=4-3"] * 5
    assert conc.Analyte_location.tolist() == ["1", "2", "3", "4", "5"] * 2
    assert conc.Concentration_micromolar.tolist() == CONCENTRATIONS * 2
    assert conc.SampleID.tolist() == ["low"] * 5 + ["high"] * 5
    assert experiment.exp_info["single_cycle"] is True
    assert experiment.exp_info["injection_count"] == 5
    assert experiment.exp_info["association_start_seconds"] == STARTS


def test_fit_recovers_constants_for_each_curve(tmp_path):
    experiment = read(write_export(tmp_path / "sck.txt"))
    experiment.align_association(inplace=True)
    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(experiment, experiment.name)
    analyzer.merge_ligand_conc_df()
    analyzer.generate_fittings(analyzer.combined_ligand_conc_df.copy())
    analyzer.submit_steady_state_fitting()
    assert len(analyzer.fittings) == 2
    for fitter, smax in zip(analyzer.fittings.values(), SMAX.values()):
        assert fitter.is_single_cycle
        assert all(detect_time_list_continuos(fitter.time_assoc_lst, fitter.time_disso_lst))
        fitter.fit_one_site_assoc_and_disso(shared_smax=True, calc_loo=False)
        assert fitter.Kd == pytest.approx(KD, rel=0.02)
        assert fitter.k_off == pytest.approx(KOFF, rel=0.02)
        assert fitter.Smax[0] == pytest.approx(smax, rel=0.02)


def test_per_curve_injection_times_and_concentrations(tmp_path):
    path = write_export(tmp_path / "sck.txt")
    names = ["Cycle=1_Fc=2-1", "Cycle=1_Fc=4-3"]
    later = [start + 1 for start in STARTS]
    experiment = read(
        path,
        association_start=dict(zip(names, [STARTS, later])),
        concentrations=dict(zip(names, [CONCENTRATIONS, CONCENTRATIONS[::-1]])),
    )
    assert experiment.exp_info["association_start_seconds"] is None
    assert experiment.exp_info["association_start_seconds_by_curve"] == dict(
        zip(names, [STARTS, later])
    )
    assert experiment.xs[1][1][0] == STARTS[0] + 1
    conc = experiment.ligand_conc_df.Concentration_micromolar.tolist()
    assert conc == CONCENTRATIONS + CONCENTRATIONS[::-1]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"association_start": 60}, "same number of injections"),
        ({"concentrations": CONCENTRATIONS[:4]}, "5 numbers per curve"),
        ({"concentrations": None}, "needs concentrations"),
        ({"sample_id": "same"}, "its own sample_id"),
        ({"association_end": [190] + ENDS[1:], "association_start": STARTS[:1] + [180] + STARTS[2:]},
         r"after the previous one ends \(curve"),
        ({"association_start": {"Cycle=1_Fc=2-1": STARTS, "Cycle=1_Fc=4-3": STARTS[:4]}},
         "same number of injections, one number each"),
        ({"single_cycle": "yes"}, "must be a boolean"),
        ({"association_start": ["a"] * 5}, "must be numbers of seconds"),
        ({"concentrations": ["a"] * 5}, "concentrations must be numbers"),
    ],
)
def test_invalid_single_cycle_settings(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        read(write_export(tmp_path / "sck.txt"), **kwargs)
