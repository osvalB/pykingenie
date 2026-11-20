"""SprTextExperiment accepts different injection times for each curve."""

import csv

import numpy as np
import pandas as pd
import pytest

from pykingenie import CarterraExperiment, KineticsAnalyzer, SprTextExperiment

KD, KOFF, SMAX = 0.1, 0.01, 50.0
CONCENTRATIONS = [0.05, 0.1, 0.2, 0.4]
OFFSETS = [0.0, 20.0, 40.0, 60.0]
DURATION = 150.0


def one_to_one(time, start, conc):
    """Noise-free 1:1 response for an injection from start to start + DURATION."""
    kon = KOFF / KD
    kobs = kon * conc + KOFF
    plateau = SMAX * conc / (conc + KD)
    t = time - start
    response = np.where(t > 0, plateau * (1 - np.exp(-kobs * np.clip(t, 0, None))), 0.0)
    end_level = plateau * (1 - np.exp(-kobs * DURATION))
    return np.where(t >= DURATION, end_level * np.exp(-KOFF * (t - DURATION)), response)


def write_curves(path, offsets=OFFSETS, concentrations=CONCENTRATIONS):
    time = np.arange(-30, 500, 1.0)
    names = [f"curve {i}" for i in range(len(offsets))]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"{name}_{axis}" for name in names for axis in "XY"])
        for t in time:
            writer.writerow([value for start, conc in zip(offsets, concentrations)
                             for value in (t, one_to_one(t, start, conc))])
    return path, names


def read(path, **kwargs):
    experiment = SprTextExperiment()
    experiment.read_file(path, concentrations=CONCENTRATIONS, **kwargs)
    return experiment


def test_each_curve_is_split_at_its_own_boundaries(tmp_path):
    path, names = write_curves(tmp_path / "curves.csv")
    starts = OFFSETS
    ends = [start + DURATION for start in OFFSETS]
    experiment = read(path, association_start=starts, association_end=ends)
    for i, (start, end) in enumerate(zip(starts, ends)):
        assert experiment.xs[i][1][0] == start and experiment.xs[i][2][0] == end
        assert experiment.xs[i][0][-1] < start
    assert experiment.exp_info["association_start_seconds"] is None
    assert experiment.exp_info["association_end_seconds"] is None
    assert experiment.exp_info["association_start_seconds_by_curve"] == dict(zip(names, starts))
    assert experiment.df_steps["Start"].tolist() == [-30, 0, 150]


def test_mapping_series_and_shared_values_can_be_mixed(tmp_path):
    path, names = write_curves(tmp_path / "curves.csv")
    by_label = dict(zip(names, OFFSETS))
    experiment = read(path, association_start=by_label, association_end=300)
    assert [x[2][0] for x in experiment.xs] == [300] * 4
    assert experiment.exp_info["association_end_seconds"] == 300
    series = read(path, association_start=pd.Series(by_label), association_end=300)
    assert [x[1][0] for x in series.xs] == OFFSETS


def test_series_settings_are_matched_by_curve_label_not_position(tmp_path):
    path, names = write_curves(tmp_path / "curves.csv")
    reordered = pd.Series(dict(zip(names, CONCENTRATIONS))).iloc[::-1]
    samples = pd.Series({name: f"S{i}" for i, name in enumerate(names)}).iloc[::-1]
    experiment = SprTextExperiment()
    experiment.read_file(path, association_end=150, concentrations=reordered,
                         sample_id=samples)
    conc = experiment.ligand_conc_df
    assert conc.Concentration_micromolar.tolist() == CONCENTRATIONS
    assert conc.SampleID.tolist() == ["S0", "S1", "S2", "S3"]
    with pytest.raises(ValueError, match="pass a list for values in curve order"):
        SprTextExperiment().read_file(path, association_end=150,
                                      concentrations=pd.Series(CONCENTRATIONS))


def test_equal_per_curve_values_keep_the_shared_keys(tmp_path):
    path, _ = write_curves(tmp_path / "curves.csv")
    experiment = read(path, association_start=[0] * 4, association_end=(150,) * 4)
    assert experiment.exp_info["association_start_seconds"] == 0
    assert experiment.exp_info["association_end_seconds"] == 150
    assert "association_start_seconds_by_curve" not in experiment.exp_info


def test_fit_recovers_constants_from_staggered_injections(tmp_path):
    path, _ = write_curves(tmp_path / "curves.csv")
    experiment = read(path, association_start=OFFSETS,
                      association_end=[start + DURATION for start in OFFSETS])
    experiment.align_association(npoints=10)
    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(experiment, experiment.name)
    analyzer.merge_ligand_conc_df()
    analyzer.generate_fittings(analyzer.combined_ligand_conc_df)
    analyzer.submit_steady_state_fitting()
    analyzer.submit_kinetics_fitting(fitting_model="one_to_one", shared_smax=True)
    fitter = analyzer.fittings[analyzer.fittings_names[0]]
    assert fitter.Kd == pytest.approx(KD, rel=0.02)
    assert fitter.k_off == pytest.approx(KOFF, rel=0.02)


@pytest.mark.parametrize(
    "start,end,message",
    [
        ([0, 20], 300, "Expected 4 association_start values, got 2"),
        ({"curve 0": 0}, 300, "exactly the exported curve names"),
        (["a", 0, 0, 0], 300, "numbers of seconds"),
        ([[0], [0], [0], [0]], 300, "one number per curve"),
        ([0, 20, 40, 60], [150, 170, 30, 210], r"start \(curve 'curve 2'\)"),
        ([0, 20, np.nan, 60], 300, r"finite.*curve 'curve 2'"),
    ],
)
def test_invalid_per_curve_boundaries_are_rejected(tmp_path, start, end, message):
    path, _ = write_curves(tmp_path / "curves.csv")
    with pytest.raises(ValueError, match=message):
        read(path, association_start=start, association_end=end)


def test_invalid_shared_boundaries_fail_before_reading(tmp_path):
    with pytest.raises(ValueError, match=r"greater than start\.$"):
        read(tmp_path / "missing.csv", association_start=10, association_end=5)


def test_carterra_keeps_one_timing_for_all_rows(tmp_path):
    with pytest.raises(ValueError, match="one association_start and association_end"):
        CarterraExperiment().read_csv(tmp_path / "rows.csv", association_start=[0, 10],
                                      association_end=30)
