# This code is AI generated.
"""
Biacore result files (.blr) and the compound file reader underneath.

Test data: 'SPR sensograms with Streptavidin_1.blr' from Vummidi et al., "A mating
mechanism to generate diversity for the Darwinian selection of DNA-encoded synthetic
molecules", Zenodo, https://doi.org/10.5281/zenodo.5011513, licensed CC-BY 4.0.
"""
import struct
import warnings

import numpy as np
import pytest

from pykingenie import BiacoreExperiment, KineticsAnalyzer
from pykingenie.biacore import _cycle_date, _injection_type, _parse_events, _read_segment, _read_xy_data
from pykingenie.utils.compound_file import (
    SIGNATURE,
    _read_compound_bytes,
    is_compound_file,
    read_compound_file,
)
from pykingenie.utils.processing import guess_experiment_type

BLR_FILE = "./test_files/SPR_sensograms_with_Streptavidin_1.blr"
CSV_FILE = "./test_files/surface_csv_traces.csv"


@pytest.fixture(scope="module")
def streams():
    return read_compound_file(BLR_FILE)


@pytest.fixture(scope="module")
def experiment():
    exp = BiacoreExperiment('streptavidin')
    exp.read_blr(BLR_FILE)
    return exp


# Compound file reader

def test_signature_detection():
    assert is_compound_file(BLR_FILE)
    assert not is_compound_file(CSV_FILE)


def test_reads_every_stream(streams):
    assert len(streams) == 404
    # Control characters at the start of names are dropped ('\x03Keywords' -> 'Keywords')
    assert '_Cycle 1/_Window 1/_Curve 1/Keywords' in streams
    assert all(ord(ch) >= 32 for key in streams for ch in key)
    # Small streams come from the mini stream, large ones from regular sectors
    assert streams['FileTag'].decode('cp1252').count('ConcUnit=') == 1
    assert len(streams['_Cycle 1/EventLog']) < 4096
    assert max(len(v) for v in streams.values()) >= 4096


def _header_with(data, offset, fmt, value):
    data = bytearray(data)
    struct.pack_into(fmt, data, offset, value)
    return bytes(data)


@pytest.mark.parametrize("change, message", [
    (lambda d: d[:100], "signature"),
    (lambda d: b'\x00' * 8 + d[8:], "signature"),
    (lambda d: _header_with(d, 0x1C, '<H', 0x1234), "byte order"),
    (lambda d: _header_with(d, 0x1A, '<H', 5), "version"),
    (lambda d: _header_with(d, 0x30, '<I', 10 ** 6), "Broken sector chain"),
    (lambda d: d[:len(d) // 2], "outside the file"),
])
def test_invalid_compound_files(change, message):
    with open(BLR_FILE, 'rb') as f:
        data = f.read()
    with pytest.raises(ValueError, match=message):
        _read_compound_bytes(change(data))


def test_sector_chain_loop_is_detected():
    """A FAT entry pointing to itself would loop forever without the guard."""
    with open(BLR_FILE, 'rb') as f:
        data = bytearray(f.read())
    first_fat_sector = struct.unpack_from('<I', data, 0x4C)[0]
    first_dir_sector = struct.unpack_from('<I', data, 0x30)[0]
    sector_size = 1 << struct.unpack_from('<H', data, 0x1E)[0]
    entry = (first_fat_sector + 1) * sector_size + 4 * first_dir_sector
    struct.pack_into('<I', data, entry, first_dir_sector)
    with pytest.raises(ValueError, match="Broken sector chain"):
        _read_compound_bytes(bytes(data))


# Binary curve formats

def test_segment_applies_offset_and_scale():
    y_raw = np.array([0.0, 1.0, 2.0], dtype='<f4')
    raw = struct.pack('<ii4di', 0, 0, 5.0, 0.5, 100.0, 2.0, 3) + y_raw.tobytes()
    t, y = _read_segment(raw)
    np.testing.assert_allclose(t, [5.0, 5.5, 6.0])
    np.testing.assert_allclose(y, [100.0, 102.0, 104.0])


def test_xy_data():
    values = np.array([1.0, 2.0, 10.0, 20.0], dtype='<f4')
    t, y = _read_xy_data(struct.pack('<iii', 0, 0, 2) + values.tobytes())
    np.testing.assert_allclose(t, [1.0, 2.0])
    np.testing.assert_allclose(y, [10.0, 20.0])


def test_both_curve_formats_in_the_file(streams):
    """Flow cells are stored as segments, the subtracted pair (Fc4-3) as XYData."""
    assert sum(key.endswith('/XYData') for key in streams) == 14
    assert sum('/Segment ' in key for key in streams) == 28


# Event log

def test_parse_events():
    raw = b'2\nF0;93;bA\nF63300;33;pR1C12;f30;iSample 1\nnot an event\nF183400;34;pR1C12;v60;iSample 1\n'
    events = _parse_events(raw)
    assert events[0] == (0.0, 93, {'b': 'A'})
    assert events[1] == (63.3, 33, {'p': 'R1C12', 'f': '30', 'i': 'Sample 1'})
    assert len(events) == 3


@pytest.mark.parametrize("label, expected", [
    ("Sample 1", "ASSOC"), ("Regeneration 2", "KREGENERATION"),
    ("Capture 1", "LOADING"), ("Ligand", "LOADING"), ("Wash", "INJECTION"),
])
def test_injection_type(label, expected):
    assert _injection_type(label) == expected


def _event(time, code, label):
    return (time, code, {'i': label, 'p': 'R1A1'})


def test_cycle_steps_with_report_end():
    events = [_event(60, 33, 'Sample 1'), _event(180, 34, 'Sample 1'), _event(420, 37, 'Sample 1'),
              _event(500, 33, 'Regeneration 1'), _event(510, 34, 'Regeneration 1')]
    steps = BiacoreExperiment._cycle_steps(events, 550.0)
    assert [(s[0], s[1], s[2]) for s in steps] == [
        (0.0, 60, 'BASELINE'), (60, 180, 'ASSOC'), (180, 420, 'DISASSOC'),
        (420, 500, 'BASELINE'), (500, 510, 'KREGENERATION'), (510, 550.0, 'BASELINE')]


def test_cycle_steps_without_report_end():
    """Without event 37 the dissociation lasts until the next injection, or the cycle end."""
    events = [_event(10, 33, 'Capture 1'), _event(40, 34, 'Capture 1'),
              _event(60, 33, 'Sample 1'), _event(180, 34, 'Sample 1'),
              _event(300, 33, 'Sample 2'), _event(400, 34, 'Sample 2')]
    steps = BiacoreExperiment._cycle_steps(events, 600.0)
    assert [(s[0], s[1], s[2]) for s in steps] == [
        (0.0, 10, 'BASELINE'), (10, 40, 'LOADING'), (40, 60, 'BASELINE'),
        (60, 180, 'ASSOC'), (180, 300, 'DISASSOC'), (300, 400, 'ASSOC'), (400, 600.0, 'DISASSOC')]


def test_concentration_rules():
    conc = BiacoreExperiment._concentration
    assert conc({}, {'Sample_1_Conc#': '50'}, 1, 1, 1e-3, '1', '0')['Concentration_micromolar'] == 0.05
    assert conc({}, {'AssayStep': 'Startup'}, 1, 1, 1.0, '1', '0')['Concentration_micromolar'] == 0.0
    assert conc({}, {'Sample_1_Sample': 'Buffer'}, 1, 1, 1.0, '1', '0')['Concentration_micromolar'] == 0.0
    with pytest.warns(UserWarning, match="No concentration"):
        row = conc({}, {'Sample_1_Sample': 'protein'}, 1, 3, 1.0, '1', '0')
    assert np.isnan(row['Concentration_micromolar'])


# Whole file

def test_guess_experiment_type():
    assert guess_experiment_type(BLR_FILE) == 'surface'
    assert guess_experiment_type('run.BLR') == 'surface'


def test_sensors_and_steps(experiment):
    assert experiment.sensor_names == ['Fc3', 'Fc4', 'Fc4-3']
    assert experiment.no_sensors == 3
    assert experiment.no_steps == len(experiment.df_steps) == 84
    assert experiment.traces_loaded

    counts = experiment.df_steps['Type'].value_counts().to_dict()
    assert counts == {'BASELINE': 42, 'ASSOC': 14, 'DISASSOC': 14, 'KREGENERATION': 14}

    # Every association is followed by its dissociation
    types = experiment.df_steps['Type'].tolist()
    assert all(types[i + 1] == 'DISASSOC' for i, t in enumerate(types) if t == 'ASSOC')


def test_first_cycle_steps(experiment):
    first = experiment.df_steps[experiment.df_steps['Cycle'] == 1]
    assert first['Type'].tolist() == ['BASELINE', 'ASSOC', 'DISASSOC', 'BASELINE', 'KREGENERATION', 'BASELINE']
    # Step starts within the cycle, shifted by the cycle start in the run (464.915 s)
    np.testing.assert_allclose(first['Start'] - 464.915, [0.0, 63.3, 183.4, 423.4, 495.1, 505.2], atol=1e-5)
    assert first['Sample_position'].iloc[1] == 'R1C12'


# Cycle starts in seconds after the run start, as OpenReadout 0.4.0 reports them
# (its 'cycles' table, rounded to the millisecond)
CYCLE_STARTS = [464.915, 1035.86, 1606.806, 2177.832, 2746.419, 3315.992, 3884.592,
                4454.971, 5023.797, 5595.367, 6163.961, 6735.583, 7304.667, 7875.396]


def test_cycles_at_their_start_in_the_run(experiment):
    starts = experiment.df_steps.groupby('Cycle')['Start'].min().to_numpy()
    np.testing.assert_allclose(starts, CYCLE_STARTS, atol=1e-3)
    assert experiment.exp_info['Run_start'] == '2017-09-21T19:27:47'
    assert experiment.exp_info['Run_end'] == '2017-09-21T21:48:50'


def test_continuous_time_axis(experiment):
    for s in range(experiment.no_sensors):
        t = np.concatenate(experiment.xs[s])
        assert np.all(np.diff(t) > 0)
        assert t[0] == pytest.approx(CYCLE_STARTS[0] + 0.1, abs=1e-3)
        assert t[-1] == pytest.approx(8430.996, abs=1e-3)
        assert len(t) == sum(len(y) for y in experiment.ys[s])
    # All sensors share the same times
    for x_a, x_b in zip(experiment.xs[0], experiment.xs[2]):
        np.testing.assert_array_equal(x_a, x_b)


def test_concentrations(experiment):
    df = experiment.ligand_conc_df
    assert len(df) == 3 * 14
    concs = df[df['Sensor'] == 'Fc4-3']['Concentration_micromolar'].tolist()
    assert concs == [0, 0, 0, 0, 0.78125, 1.5625, 3.125, 6.25, 12.5, 25, 50, 100, 0, 0.78125]
    assert df['SampleID'].iloc[0] == 'buffer'
    assert set(df['Experiment']) == {'streptavidin'}


def test_run_info(experiment):
    info = experiment.exp_info
    assert info['Application'] == 'Biacore T200 Control Software'
    assert info['RunType'] == 'Kinetics/Affinity'
    assert info['Chip'] == 'CM5'
    assert info['Concentration_unit'] == 'µM'
    assert info['Fc4_ligand'] == 'Streptavidin'
    assert info['Fc4_immobilization_level_RU'] == pytest.approx(1000.86, abs=0.01)
    assert np.isnan(info['Fc3_immobilization_level_RU'])


def test_fitting_pipeline(experiment):
    """The imported data go through alignment and fitting like other surface data."""
    exp = BiacoreExperiment('pipeline')
    exp.read_blr(BLR_FILE)
    exp.align_association(['Fc4-3'])

    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(exp, 'pipeline')
    analyzer.merge_ligand_conc_df()
    df = analyzer.combined_ligand_conc_df
    analyzer.generate_fittings(df[(df['Sensor'] == 'Fc4-3') & (df['[Analyte] (μM)'] > 0)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        analyzer.submit_steady_state_fitting()
    assert len(analyzer.fittings) == 1


def test_sample_plate(experiment):
    """Rack positions as a sample plate: rows carry the rack, one entry per position."""
    exp = experiment
    assert exp.sample_plate_loaded
    positions = [f'{r.replace(" ", "")}{c}' for r, c in zip(exp.sample_row, exp.sample_column)]
    assert positions == ['R1A12'] + [f'R1C{i}' for i in range(1, 13)] + ['R2A1']
    plate = dict(zip(positions, zip(exp.sample_type, exp.sample_id, exp.sample_conc_labeled)))
    assert plate['R1C5'] == ('KSAMPLE', 'Streptag-Gly loop', '6.25 µM')
    assert plate['R1C1'] == ('BUFFER', 'Streptag-Gly loop', '0 µM')
    assert plate['R1C12'] == ('BUFFER', 'buffer', '0 µM (2 injections)')
    assert plate['R2A1'] == ('REGENERATION', 'Regeneration 1', '14 injections')
    assert exp.sample_conc[positions.index('R1C9')] == 100
    assert np.isnan(exp.sample_conc[positions.index('R2A1')])


def test_plate_plots(experiment):
    from pykingenie import plot_plate_info
    from pykingenie.utils import plotting_xy
    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(experiment, 'streptavidin')
    fig = plot_plate_info(analyzer, 'streptavidin')
    assert list(fig.data[0].text) == [{'KSAMPLE': 'S', 'BUFFER': 'B', 'REGENERATION': 'R'}[t]
                                      for t in experiment.sample_type]
    assert plotting_xy.plot_plate_info(analyzer, 'streptavidin') is not None


def test_sample_plate_builder():
    exp = BiacoreExperiment()
    exp._build_sample_plate([
        {'position': 'R2B3', 'step_type': 'LOADING', 'id': 'Capture 1', 'conc': None},
        {'position': 'R1A1', 'step_type': 'ASSOC', 'id': 'protein', 'conc': 1.0},
        {'position': 'R1A1', 'step_type': 'ASSOC', 'id': 'protein', 'conc': 2.0},
        {'position': 'R1A2', 'step_type': 'ASSOC', 'id': 'protein', 'conc': np.nan},
        {'position': 'vial 7', 'step_type': 'ASSOC', 'id': 'other', 'conc': 1.0},
    ])
    assert list(exp.sample_row) == ['R1 A', 'R1 A', 'R2 B']
    assert list(exp.sample_column) == [1, 2, 3]
    assert exp.sample_type == ['KSAMPLE', 'KSAMPLE', 'LOADING']
    assert exp.sample_conc_labeled == ['1 µM, 2 µM (2 injections)', '', '']
    assert np.isnan(exp.sample_conc[0]) and np.isnan(exp.sample_conc[1])

    empty = BiacoreExperiment()
    empty._build_sample_plate([{'position': 'vial 7', 'step_type': 'ASSOC', 'id': 'x', 'conc': 1.0}])
    assert not empty.sample_plate_loaded and empty.sample_row is None


def _read_without(monkeypatch, drop):
    """Read the test file with some streams removed."""
    import pykingenie.biacore as biacore
    original = biacore.read_compound_file
    monkeypatch.setattr(biacore, 'read_compound_file',
                        lambda path: {k: v for k, v in original(path).items() if not drop(k, v)})
    exp = BiacoreExperiment()
    exp.read_blr(BLR_FILE)
    return exp.df_steps.groupby('Cycle')['Start'].min().to_numpy(), exp


def test_without_run_start_cycles_start_from_the_first(monkeypatch):
    starts, exp = _read_without(monkeypatch, lambda key, value: key == 'Environment')
    np.testing.assert_allclose(starts, np.array(CYCLE_STARTS) - CYCLE_STARTS[0], atol=1e-3)
    assert exp.exp_info['Run_start'] is None


def test_without_date_stamps_cycles_follow_each_other(monkeypatch):
    """Without the cycle date stamps (event 10), each cycle starts where the previous ends."""
    import pykingenie.biacore as biacore
    monkeypatch.setattr(biacore, '_cycle_date', lambda events: None)
    exp = BiacoreExperiment()
    exp.read_blr(BLR_FILE)
    starts = exp.df_steps.groupby('Cycle')['Start'].min().to_numpy()
    assert starts[0] == 0.0
    ends = [np.concatenate([x for x, c in zip(exp.xs[0], exp.df_steps['Cycle']) if c == cycle])[-1]
            for cycle in range(1, 15)]
    np.testing.assert_allclose(starts[1:], ends[:-1])


def test_cycle_date():
    events = _parse_events(b'F0;93;bA\nF0;10;d42999.8163440162\nF500;10;d1\n')
    assert _cycle_date(events) == 42999.8163440162
    assert _cycle_date(_parse_events(b'F0;10;dnot a date\n')) is None
    assert _cycle_date([]) is None


def test_not_a_biacore_file(tmp_path):
    """A compound file without cycles is refused."""
    with open(BLR_FILE, 'rb') as f:
        data = f.read()
    assert data[:8] == SIGNATURE

    exp = BiacoreExperiment()
    empty = tmp_path / 'empty.blr'
    empty.write_bytes(b'not a compound file' * 40)
    with pytest.raises(ValueError, match="signature"):
        exp.read_blr(str(empty))
