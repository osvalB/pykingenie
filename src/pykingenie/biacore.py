# This file is AI generated.
"""Import surface plasmon resonance (SPR) data from Biacore result files (.blr).

A .blr file is a compound file (see utils/compound_file.py) written by the Biacore
control software (checked with Biacore T200 Control Software 2.0.2). It holds one
storage per cycle, with:

- an event log (``_Cycle N/EventLog``), with the injections of the cycle;
- one curve per flow cell and per reference-subtracted pair
  (``_Cycle N/_Window 1/_Curve K``), with keywords (flow cell, sample, concentration,
  ligand) and the sensorgram, stored as ``Segment`` (equally spaced times) or
  ``XYData`` (explicit times).

Each flow cell (and subtracted pair, e.g. ``Fc4-3``) becomes a sensor. The cycles are
placed on a single time axis at their start in the run (seconds after the run start,
from the date stamps of the event logs), and each cycle is split into steps from its
event log:

- sample injection start to end: association ('ASSOC');
- from the end of the sample injection to the end of the report period (event 37) or
  the next injection: dissociation ('DISASSOC');
- regeneration injections: 'KREGENERATION'; capture injections: 'LOADING';
- the periods of buffer flow in between: 'BASELINE'.
"""
import re
import struct
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .surface_exp import SurfaceBasedExperiment
from .utils.compound_file import read_compound_file

__all__ = ['BiacoreExperiment']

# Event log codes (time in ms; code; fields)
_INJECTION_START = 33
_INJECTION_END = 34
_REPORT_END = 37
_CYCLE_DATE = 10  # at time 0 of the cycle, field 'd': date stamp

# Date stamps are OLE automation dates: days since 1899-12-30
_OLE_EPOCH = datetime(1899, 12, 30)
_SECONDS_PER_DAY = 86400.0

# Concentration units of the file, to µM
_CONC_UNIT_TO_MICROMOLAR = {'M': 1e6, 'mM': 1e3, 'µM': 1.0, 'uM': 1.0, 'nM': 1e-3, 'pM': 1e-6}

# Injection label (event field 'i', e.g. 'Sample 1') to step type
_INJECTION_TYPES = [('regeneration', 'KREGENERATION'), ('capture', 'LOADING'),
                    ('ligand', 'LOADING'), ('sample', 'ASSOC')]

# Step type of an injection to the sample type of its rack position, as in Octet plates
_PLATE_SAMPLE_TYPES = {'ASSOC': 'KSAMPLE', 'KREGENERATION': 'REGENERATION',
                       'LOADING': 'LOADING', 'INJECTION': 'INJECTION'}

# Rack position, e.g. 'R1C12': rack 1, row C, column 12
_RACK_POSITION = re.compile(r'^R(\d+)([A-Z]+)(\d+)$')


def _text(raw):
    """Decode a text stream (the software writes Windows-1252 text)."""
    return raw.decode('cp1252', errors='replace')


def _key_values(raw):
    """Parse 'key=value' lines into a dict."""
    pairs = (line.split('=', 1) for line in _text(raw).splitlines() if '=' in line)
    return {key.strip(): value.strip() for key, value in pairs}


def _read_xy_data(raw):
    """Curve stored with explicit times: two int32 flags, n, n float32 times, n float32 responses."""
    n = struct.unpack_from('<i', raw, 8)[0]
    values = np.frombuffer(raw, dtype='<f4', count=2 * n, offset=12).astype(float)
    return values[:n], values[n:]


def _read_segment(raw):
    """Equally spaced curve: two int32 flags, x0, dx, y offset, y scale (float64), n, n float32."""
    x0, dx, y_offset, y_scale = struct.unpack_from('<4d', raw, 8)
    n = struct.unpack_from('<i', raw, 40)[0]
    y = np.frombuffer(raw, dtype='<f4', count=n, offset=44).astype(float)
    return x0 + dx * np.arange(n), y_offset + y_scale * y


def _parse_events(raw):
    """Event log lines 'F<time ms>;<code>;<fields>' as (time in s, code, {field letter: value})."""
    events = []
    for line in _text(raw).splitlines():
        match = re.match(r'F(\d+);(\d+);?(.*)$', line.strip())
        if not match:
            continue
        fields = {item[0]: item[1:] for item in match.group(3).split(';') if item}
        events.append((int(match.group(1)) / 1000, int(match.group(2)), fields))
    return events


def _ole_date(value):
    """OLE automation date (days since 1899-12-30) from text, or None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cycle_date(events):
    """Date stamp of the cycle start (OLE date), or None if the event log has none."""
    for time, code, fields in events:
        if code == _CYCLE_DATE and time == 0:
            date = _ole_date(fields.get('d'))
            if date is not None:
                return date
    return None


def _ole_date_to_iso(date):
    return (_OLE_EPOCH + timedelta(days=date)).isoformat(timespec='seconds') if date is not None else None


def _injection_type(label):
    label = label.lower()
    for word, step_type in _INJECTION_TYPES:
        if word in label:
            return step_type
    return 'INJECTION'


def _sensor_name(flow_cell):
    """'4-3' -> 'Fc4-3', '3' -> 'Fc3'."""
    return 'Fc' + flow_cell.replace(' ', '')


class BiacoreExperiment(SurfaceBasedExperiment):
    """
    Surface plasmon resonance experiment read from a Biacore result file (.blr).

    Parameters
    ----------
    name : str, optional
        Name of the experiment. Default is 'biacore'.

    Attributes
    ----------
    xs : list of list of numpy.ndarray
        Time values, one list per sensor (flow cell) with one array per step.
    ys : list of list of numpy.ndarray
        Response values (RU), one list per sensor with one array per step.
    df_steps : pandas.DataFrame
        Step information with columns '#Step', 'Name', 'Type', 'Start', 'Cycle',
        'Sample_position', 'Column_location' and 'Loading_location'.
    ligand_conc_df : pandas.DataFrame
        One row per sensor and association step, with the analyte concentration.
    exp_info : dict
        Run information: instrument software, run type, run start and end, sensor
        chip, ligand and immobilization level per flow cell, and the concentration
        unit of the file.
    sample_row, sample_column, sample_type, sample_id, sample_conc, sample_conc_labeled
        Rack positions used by the injections, one entry per position, for
        ``plot_plate_info``. Rows carry the rack (e.g. 'R1 C' for position R1C5).
    """

    def __init__(self, name='biacore'):
        """Initialize a Biacore experiment named ``name``."""

        super().__init__(name, 'biacore')

    def read_blr(self, file):
        """
        Read a Biacore result file (.blr) and populate the experiment.

        Parameters
        ----------
        file : str
            Path to the .blr file.

        Returns
        -------
        None
        """
        streams = read_compound_file(file)

        cycles = sorted({int(m.group(1)) for key in streams for m in [re.match(r'_Cycle (\d+)/', key)] if m})
        if not cycles:
            raise ValueError(f"No cycles found in {file}; is it a Biacore result file?")

        conc_unit = _key_values(streams.get('FileTag', b'')).get('ConcUnit', 'µM')
        if conc_unit not in _CONC_UNIT_TO_MICROMOLAR:
            warnings.warn(f"Unknown concentration unit '{conc_unit}'; concentrations are read as µM.")
        to_micromolar = _CONC_UNIT_TO_MICROMOLAR.get(conc_unit, 1.0)

        self.fns = [file]
        self.exp_info = self._run_info(streams, conc_unit)

        # Curves of every cycle: sensor name -> (time, response, keywords)
        curves_per_cycle = [self._read_cycle_curves(streams, c) for c in cycles]
        self.sensor_names = list(curves_per_cycle[0])
        for c, curves in zip(cycles, curves_per_cycle):
            if list(curves) != self.sensor_names:
                raise ValueError(f"Cycle {c} has curves {list(curves)}, expected {self.sensor_names}.")
        self.no_sensors = len(self.sensor_names)

        self.xs = [[] for _ in self.sensor_names]
        self.ys = [[] for _ in self.sensor_names]
        steps, conc_rows, injections = [], [], []
        previous_end, n_assoc, n_loading, loading_location = 0.0, 0, 0, '0'

        # Cycles start at their time in the run: seconds after the run start, or after
        # the first cycle if the run start is missing
        events_per_cycle = [_parse_events(streams.get(f'_Cycle {c}/EventLog', b'')) for c in cycles]
        cycle_dates = [_cycle_date(events) for events in events_per_cycle]
        run_start = _ole_date(_key_values(streams.get('Environment', b'')).get('Timestamp'))
        if run_start is None:
            run_start = next((d for d in cycle_dates if d is not None), None)

        for c, curves, events, date in zip(cycles, curves_per_cycle, events_per_cycle, cycle_dates):

            keywords = next(iter(curves.values()))[2]
            cycle_end = max(t[-1] for t, _, _ in curves.values())

            # Without a date stamp, or if it would overlap the previous cycle, the cycle
            # follows the previous one
            offset = previous_end
            if date is not None and run_start is not None:
                offset = max((date - run_start) * _SECONDS_PER_DAY, previous_end)

            sample_index = 0
            for start, end, step_type, name, fields in self._cycle_steps(events, cycle_end):

                column_location = str(c)
                if step_type == 'LOADING':
                    n_loading += 1
                    loading_location = column_location = str(n_loading)
                if step_type == 'ASSOC':
                    n_assoc += 1
                    sample_index += 1
                    column_location = str(n_assoc)
                    conc_rows.append(self._concentration(curves, keywords, sample_index, c,
                                                         to_micromolar, column_location, loading_location))

                steps.append({'Name': f'Cycle {c} {name}', 'Type': step_type, 'Start': offset + start,
                              'Cycle': c, 'Sample_position': fields.get('p', ''),
                              'Column_location': column_location,
                              'Loading_location': loading_location if step_type == 'ASSOC' else np.nan})

                if step_type in _PLATE_SAMPLE_TYPES and fields.get('p'):
                    row = conc_rows[-1] if step_type == 'ASSOC' else {}
                    injections.append({'position': fields['p'], 'step_type': step_type,
                                       'id': row.get('SampleID', name),
                                       'conc': row.get('Concentration_micromolar')})

                for s, sensor in enumerate(self.sensor_names):
                    t, y, _ = curves[sensor]
                    last_step = end >= cycle_end
                    sel = (t >= start) & ((t <= end) if last_step else (t < end))
                    self.xs[s].append(offset + t[sel])
                    self.ys[s].append(y[sel])

            previous_end = offset + cycle_end

        if n_assoc == 0:
            raise ValueError(f"No sample injections found in {file}.")

        self.df_steps = pd.DataFrame(steps)
        self.df_steps.insert(0, '#Step', np.arange(len(steps)) + 1)
        self.no_steps = len(steps)

        self._build_ligand_conc_df(conc_rows)
        self._build_sample_plate(injections)
        self.create_unique_sensor_names()
        self.traces_loaded = True

        return None

    @staticmethod
    def _run_info(streams, conc_unit):
        """Instrument, run and sensor chip information."""
        environment = _key_values(streams.get('Environment', b''))
        chip = _key_values(streams.get('Chip', b''))

        info = {key: environment.get(key) for key in ('Application', 'Version', 'RunType', 'InstrumentId')}
        info['Run_start'] = _ole_date_to_iso(_ole_date(environment.get('Timestamp')))
        info['Run_end'] = _ole_date_to_iso(_ole_date(environment.get('EndTime')))
        info['Chip'] = chip.get('Name')
        info['Concentration_unit'] = conc_unit

        n_flow_cells = int(chip.get('NoFcs', 0) or 0)
        for fc in range(1, n_flow_cells + 1):
            ligand = chip.get(f'Ligand{fc},1', '')
            level = chip.get(f'Level{fc},1')
            info[f'Fc{fc}_ligand'] = ligand
            info[f'Fc{fc}_immobilization_level_RU'] = float(level) if level else np.nan

        return info

    @staticmethod
    def _read_cycle_curves(streams, cycle):
        """Sensorgrams of one cycle: sensor name -> (time, response, keywords)."""
        prefix = f'_Cycle {cycle}/_Window 1/'
        curve_ids = sorted({int(m.group(1)) for key in streams
                            for m in [re.match(re.escape(prefix) + r'_Curve (\d+)/', key)] if m})
        curves = {}
        for k in curve_ids:
            base = f'{prefix}_Curve {k}/'
            keywords = _key_values(streams.get(base + 'Keywords', b''))
            if base + 'XYData' in streams:
                t, y = _read_xy_data(streams[base + 'XYData'])
            else:
                segments = sorted((int(m.group(1)), key) for key in streams
                                  for m in [re.match(re.escape(base) + r'Segment (\d+)$', key)] if m)
                if not segments:
                    continue
                parts = [_read_segment(streams[key]) for _, key in segments]
                t = np.concatenate([p[0] for p in parts])
                y = np.concatenate([p[1] for p in parts])
            # Times are stored as float32 in XYData: round so all sensors share the same times
            curves[_sensor_name(keywords.get('Fc', str(k)))] = (np.round(t, 4), y, keywords)
        return curves

    @staticmethod
    def _cycle_steps(events, cycle_end):
        """Steps of a cycle as (start, end, type, name, event fields), times in s from the cycle start."""
        injections, report_ends, open_start = [], [], None
        for time, code, fields in events:
            if code == _INJECTION_START:
                open_start = (time, fields)
            elif code == _INJECTION_END and open_start is not None:
                injections.append((open_start[0], time, open_start[1]))
                open_start = None
            elif code == _REPORT_END:
                report_ends.append(time)

        steps, previous_end = [], 0.0
        for i, (start, end, fields) in enumerate(injections):
            step_type = _injection_type(fields.get('i', ''))
            label = fields.get('i', 'Injection')
            if start > previous_end:
                steps.append((previous_end, start, 'BASELINE', 'Baseline', {}))
            steps.append((start, end, step_type, label, fields))
            previous_end = end

            if step_type == 'ASSOC':
                next_start = injections[i + 1][0] if i + 1 < len(injections) else cycle_end
                report_end = [t for t in report_ends if end < t <= next_start]
                disso_end = report_end[0] if report_end else next_start
                steps.append((end, disso_end, 'DISASSOC', 'Dissociation', fields))
                previous_end = disso_end

        if previous_end < cycle_end:
            steps.append((previous_end, cycle_end, 'BASELINE', 'Baseline', {}))

        return steps

    @staticmethod
    def _concentration(curves, keywords, sample_index, cycle, to_micromolar, column_location, loading_location):
        """Concentration (µM) and sample name of the sample_index-th sample injection of a cycle."""
        value = keywords.get(f'Sample_{sample_index}_Conc#', '')
        sample = keywords.get(f'Sample_{sample_index}_Sample', '') or 'sample'
        if value != '':
            conc = float(value) * to_micromolar
        elif keywords.get('AssayStep', '').lower() == 'startup' or sample.lower() in ('buffer', 'blank'):
            conc = 0.0
        else:
            warnings.warn(f"No concentration for the sample injection of cycle {cycle}; set to NaN.")
            conc = np.nan
        return {'Analyte_location': column_location, 'Concentration_micromolar': conc,
                'SampleID': sample, 'Loading_location': loading_location}

    def _build_ligand_conc_df(self, conc_rows):
        """One row per sensor and association step."""
        rows = [{'Sensor': sensor, **row} for sensor in self.sensor_names for row in conc_rows]
        df = pd.DataFrame(rows)
        df['Replicate'] = df.groupby(['Sensor', 'Loading_location', 'Analyte_location']).cumcount() + 1
        df['Experiment'] = self.name
        self.ligand_conc_df = df[['Sensor', 'Analyte_location', 'Concentration_micromolar',
                                  'SampleID', 'Replicate', 'Loading_location', 'Experiment']]
        return None

    def _build_sample_plate(self, injections):
        """
        Rack positions used by the injections, in the form plot_plate_info expects.

        One entry per position (e.g. 'R1C12'), sorted by rack, row and column. Rows are labelled
        with their rack ('R1 C'), so positions of different racks do not overlap. A
        position used by sample injections is 'KSAMPLE', or 'BUFFER' if all of them are
        0 µM; regeneration and capture positions are 'REGENERATION' and 'LOADING'.
        """
        positions = {}
        for injection in injections:
            match = _RACK_POSITION.match(injection['position'])
            if match is None:
                continue
            positions.setdefault(injection['position'], (match, []))[1].append(injection)

        rows, columns, types, ids, concs, labels = [], [], [], [], [], []
        def rack_order(item):
            rack, row, column = item[1][0].groups()
            return int(rack), row, int(column)

        for position, (match, uses) in sorted(positions.items(), key=rack_order):
            rack, row, column = match.groups()
            sample_type = _PLATE_SAMPLE_TYPES[uses[0]['step_type']]
            values = [u['conc'] for u in uses if u['conc'] is not None and not np.isnan(u['conc'])]
            if sample_type == 'KSAMPLE' and values and all(v == 0 for v in values):
                sample_type = 'BUFFER'

            names = list(dict.fromkeys(u['id'] for u in uses))
            unique_values = list(dict.fromkeys(values))
            label = ', '.join(f'{v:g} µM' for v in unique_values)
            if len(uses) > 1:
                label = f'{label} ({len(uses)} injections)' if label else f'{len(uses)} injections'

            rows.append(f'R{rack} {row}')
            columns.append(int(column))
            types.append(sample_type)
            ids.append(', '.join(names))
            concs.append(unique_values[0] if len(unique_values) == 1 else np.nan)
            labels.append(label)

        if not rows:
            return None

        self.sample_row = np.array(rows)
        self.sample_column = np.array(columns)
        self.sample_type = types
        self.sample_id = ids
        self.sample_conc = np.array(concs, dtype=float)
        self.sample_conc_labeled = labels
        self.sample_plate_loaded = True

        return None
