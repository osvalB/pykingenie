# Note: this file contains AI-generated code, written with AI coding assistants (PR #XX).
"""Import surface-based binding data from a steps table and a column-formatted traces file.

This importer is instrument-agnostic. It needs two plain-text files:

1. A **steps file** (tab-, comma- or semicolon-delimited) with one row per
   assay step, for example::

       Assay  Step Number  Step Name     Step Time  Step Type Name  Sample Location
       1      1            Baseline      60         Baseline        p1A1
       1      2            Loading       120        Loading         p1A2
       1      3            Association   120        Association     p1A3
       1      4            Dissociation  60         Dissociation    p1A4

   Required columns are the step duration (``Step Time``, in seconds) and the
   step type (``Step Type Name``). ``Step Name`` and ``Sample Location`` are
   optional. An optional ``Concentration`` column (in µM) can hold the analyte
   concentration of each association step.

2. A **traces file** (comma-, tab- or semicolon-delimited) with the time in
   the first column and one sensor per remaining column, for example::

       X Values, A1, B1
       0, 0, 0
       0.2, -0.0016, 0.0085

   Any metadata lines before the header row are skipped automatically. The
   time axis must be continuous across steps, i.e. step ``i`` spans the time
   interval ``[sum(step_time[:i]), sum(step_time[:i+1]))``.
"""

import re
import warnings

import numpy  as np
import pandas as pd

from .surface_exp import SurfaceBasedExperiment

# Map common step type names to the internal codes used by pykingenie
STEP_TYPE_MAP = {
    'baseline'          : 'BASELINE',
    'loading'           : 'LOADING',
    'association'       : 'ASSOC',
    'assoc'             : 'ASSOC',
    'dissociation'      : 'DISASSOC',
    'disassoc'          : 'DISASSOC',
    'regeneration'      : 'KREGENERATION',
    'kregeneration'     : 'KREGENERATION',
    'neutralization'    : 'NEUTRALIZATION',
    'activation'        : 'ACTIVATION',
    'quenching'         : 'QUENCHING',
}

# Accepted (lower-case, whitespace-collapsed) aliases for the columns of the steps file
_STEP_COLUMN_ALIASES = {
    'Name'          : ['step name', 'name'],
    'Time'          : ['step time', 'time', 'duration', 'step duration'],
    'Type'          : ['step type name', 'step type', 'type'],
    'Location'      : ['sample location', 'location', 'well'],
    'Concentration' : ['concentration', 'concentration (µm)', 'concentration (um)',
                       'concentration_micromolar', 'conc', 'conc (µm)', 'conc (um)'],
    'SampleID'      : ['sample id', 'sampleid', 'sample'],
}


def _normalise(text):
    """Lower-case ``text`` and collapse whitespace, for tolerant column matching."""
    return re.sub(r'\s+', ' ', str(text)).strip().lower()


def map_step_type(step_type):
    """
    Convert a step type name (e.g. 'Association') to the internal pykingenie code (e.g. 'ASSOC').

    Parameters
    ----------
    step_type : str
        Step type name as written in the steps file.

    Returns
    -------
    str
        Internal step type code. Unknown names are returned upper-cased.
    """
    key = _normalise(step_type)
    return STEP_TYPE_MAP.get(key, key.upper())


def location_to_column(location, default):
    """
    Extract the plate column number from a well location string.

    Parameters
    ----------
    location : str
        Well location, e.g. 'A7', 'p1A7' or '7'.
    default : int
        Value returned if no column number can be parsed.

    Returns
    -------
    str
        Column number as a string (e.g. '7').
    """
    if location is None or (isinstance(location, float) and np.isnan(location)):
        return str(default)

    match = re.search(r'(\d+)\s*$', str(location).strip())

    return str(int(match.group(1))) if match else str(default)


def _is_numeric_row(fields):
    """Return True if all (non-empty) fields can be converted to float."""
    fields = [f.strip() for f in fields if f.strip() != '']
    if len(fields) < 2:
        return False
    try:
        [float(f) for f in fields]
        return True
    except ValueError:
        return False


def _detect_delimiter(line):
    """Guess the delimiter of a line (tab, semicolon or comma)."""
    for delim in ['\t', ';', ',']:
        if delim in line:
            return delim
    return ','


def read_traces_file(file):
    """
    Read a column-formatted traces file (time in the first column, one sensor per column).

    Metadata lines before the header are skipped. The header is the last
    non-numeric line before the first fully numeric line.

    Parameters
    ----------
    file : str
        Path to the traces file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the time in the first column and one column per sensor.
    """
    with open(file, 'r', encoding='utf-8-sig', errors='replace') as f:
        lines = f.read().splitlines()

    first_data_idx = None
    delimiter      = ','

    for i, line in enumerate(lines):
        delim = _detect_delimiter(line)
        if _is_numeric_row(line.split(delim)):
            first_data_idx = i
            delimiter      = delim
            break

    if first_data_idx is None:
        raise ValueError(f"No numeric data found in the traces file: {file}")

    if first_data_idx > 0:
        header = [h.strip() for h in lines[first_data_idx - 1].split(delimiter)]
    else:
        header = None

    rows = [l.split(delimiter) for l in lines[first_data_idx:] if l.strip() != '']
    n_cols = len(rows[0])
    rows   = [r[:n_cols] + [''] * (n_cols - len(r)) for r in rows]

    df = pd.DataFrame(rows).replace(r'^\s*$', np.nan, regex=True).astype(float)

    if header is None or len(header) < n_cols:
        header = ['Time'] + [f'Sensor {i}' for i in range(1, n_cols)]

    df.columns = header[:n_cols]

    # Drop columns that are completely empty (e.g. from trailing delimiters)
    df = df.dropna(axis=1, how='all')

    if df.shape[1] < 2:
        raise ValueError("The traces file must contain a time column and at least one sensor column.")

    return df


def read_steps_file(file):
    """
    Read a steps file describing the assay steps.

    Parameters
    ----------
    file : str
        Path to the steps file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardised columns: 'Name', 'Time', 'Type' (internal code),
        'Location' and, if present, 'Concentration' and 'SampleID'.
    """
    df = pd.read_csv(file, sep=None, engine='python', encoding='utf-8-sig')

    # Drop fully empty rows/columns (e.g. trailing tabs or blank lines)
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    rename = {}
    normalised_cols = {_normalise(c): c for c in df.columns}

    for standard, aliases in _STEP_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalised_cols:
                rename[normalised_cols[alias]] = standard
                break

    df = df.rename(columns=rename)

    for required in ['Time', 'Type']:
        if required not in df.columns:
            raise ValueError(f"The steps file must contain a '{required}' column "
                             f"(accepted names: {_STEP_COLUMN_ALIASES[required]}).")

    df = df[df['Type'].notna()].reset_index(drop=True)

    df['Time'] = df['Time'].astype(float)
    df['Type'] = [map_step_type(t) for t in df['Type']]

    if 'Name' not in df.columns:
        df['Name'] = df['Type']
    df['Name'] = df['Name'].astype(str).str.strip()

    if 'Location' not in df.columns:
        df['Location'] = np.nan

    return df


class SurfaceCsvExperiment(SurfaceBasedExperiment):
    """
    Surface-based experiment imported from a steps file and a column-formatted traces file.

    Useful for data exported from any instrument (e.g. Octet 'Time Interpolated Data'
    exports) where the native binary format is not available.

    Parameters
    ----------
    name : str, optional
        Name of the experiment. Default is 'surface_csv'.

    Attributes
    ----------
    xs : list of list of numpy.ndarray
        Time values, one list per sensor with one array per step.
    ys : list of list of numpy.ndarray
        Signal values, one list per sensor with one array per step.
    df_steps : pandas.DataFrame
        Step information with columns '#Step', 'Name', 'Type', 'Start',
        'Column_location' and 'Loading_location'.
    ligand_conc_df : pandas.DataFrame
        One row per sensor and association step, with the analyte concentration.
    """

    def __init__(self, name='surface_csv'):
        """Initialize a surface-based experiment named ``name``."""

        super().__init__(name, 'surface_csv')

    def read_files(self, steps_file, traces_file, concentrations=None, sample_id='sample'):
        """
        Read the steps file and the traces file and populate the experiment.

        Parameters
        ----------
        steps_file : str
            Path to the steps file (one row per step).
        traces_file : str
            Path to the traces file (time column followed by one column per sensor).
        concentrations : list, dict or None, optional
            Analyte concentrations in µM. Either a list with one value per association
            step (shared by all sensors) or a dict mapping sensor name to such a list.
            If None, the 'Concentration' column of the steps file is used; if that is
            also missing, concentrations are set to NaN and a warning is issued
            (they can be edited later in ``ligand_conc_df``).
        sample_id : str, optional
            Sample ID used when the steps file has no 'Sample ID' column (or it is empty
            for an association step). Default 'sample'. If the loading steps also have a
            Sample ID and more than one ligand was loaded, the association Sample IDs
            become '<analyte> - <ligand>', so each ligand is fitted separately.

        Returns
        -------
        None
        """

        steps  = read_steps_file(steps_file)
        traces = read_traces_file(traces_file)

        self.fns = [steps_file, traces_file]

        time = traces.iloc[:, 0].to_numpy(dtype=float)

        self.sensor_names = [str(c) for c in traces.columns[1:]]
        self.no_sensors   = len(self.sensor_names)
        self.no_steps     = len(steps)

        # Step boundaries, relative to the first time point
        durations = steps['Time'].to_numpy(dtype=float)
        ends      = np.cumsum(durations)
        starts    = np.concatenate([[0.0], ends[:-1]])

        t_rel = time - time[0]

        # Assign each time point to a step. Points beyond the last boundary
        # (e.g. rounding of the step times) are assigned to the last step.
        step_idx = np.searchsorted(ends, t_rel, side='right')
        step_idx = np.clip(step_idx, 0, self.no_steps - 1)

        self.xs = [[] for _ in range(self.no_sensors)]
        self.ys = [[] for _ in range(self.no_sensors)]

        for s, sensor in enumerate(traces.columns[1:]):
            y = traces[sensor].to_numpy(dtype=float)
            for i in range(self.no_steps):
                sel = step_idx == i
                self.xs[s].append(time[sel])
                self.ys[s].append(y[sel])

        empty_steps = [steps['Name'].iloc[i] for i in range(self.no_steps) if len(self.xs[0][i]) == 0]
        if empty_steps:
            warnings.warn(f"No data points found for steps: {empty_steps}. "
                          "Check that the step times match the traces time axis.")

        # Build the steps dataframe
        column_location = [location_to_column(loc, i + 1) for i, loc in enumerate(steps['Location'])]

        loading_location = []
        last_loading     = None
        for typ, col in zip(steps['Type'], column_location):
            if typ == 'LOADING':
                last_loading = col
            if typ == 'ASSOC':
                loading_location.append(last_loading if last_loading is not None else '0')
            else:
                loading_location.append(np.nan)

        self.df_steps = pd.DataFrame({'#Step'           : np.arange(self.no_steps) + 1,
                                      'Name'            : steps['Name'].to_numpy(),
                                      'Type'            : steps['Type'].to_numpy(),
                                      'Start'           : starts + time[0],
                                      'Column_location' : column_location,
                                      'Loading_location': loading_location})

        self._generate_ligand_conc_df(steps, concentrations, sample_id)

        self.create_unique_sensor_names()

        self.traces_loaded = True

        return None

    def _generate_ligand_conc_df(self, steps, concentrations, sample_id):
        """Build ``ligand_conc_df`` with one row per sensor and association step."""

        assoc_idx = np.where(self.df_steps['Type'] == 'ASSOC')[0]
        n_assoc   = len(assoc_idx)

        if n_assoc == 0:
            raise ValueError("The steps file does not contain any association step.")

        # Resolve concentrations: dict sensor -> list of n_assoc values
        if concentrations is None:
            if 'Concentration' in steps.columns:
                values = steps['Concentration'].to_numpy()[assoc_idx].astype(float)
            else:
                warnings.warn("No analyte concentrations were provided. They are set to NaN; "
                              "pass 'concentrations' or add a 'Concentration' column (µM) to the steps file.")
                values = np.full(n_assoc, np.nan)
            conc_per_sensor = {s: list(values) for s in self.sensor_names}

        elif isinstance(concentrations, dict):
            missing = [s for s in self.sensor_names if s not in concentrations]
            if missing:
                raise ValueError(f"Missing concentrations for sensors: {missing}")
            conc_per_sensor = {s: list(concentrations[s]) for s in self.sensor_names}

        else:
            conc_per_sensor = {s: list(concentrations) for s in self.sensor_names}

        for s, values in conc_per_sensor.items():
            if len(values) != n_assoc:
                raise ValueError(f"Expected {n_assoc} concentrations (one per association step) "
                                 f"for sensor {s}, got {len(values)}.")

        sample_ids = [sample_id] * n_assoc

        if 'SampleID' in steps.columns:

            ids = [None if pd.isna(x) or str(x).strip() == '' else str(x).strip() for x in steps['SampleID']]

            sample_ids = [ids[idx] if ids[idx] is not None else sample_id for idx in assoc_idx]

            # Sample ID of the most recent loading step before each association
            loading_ids, last_loading_id = [], None
            for i, typ in enumerate(steps['Type']):
                if typ == 'LOADING' and ids[i] is not None:
                    last_loading_id = ids[i]
                if typ == 'ASSOC':
                    loading_ids.append(last_loading_id)

            # If different ligands were loaded, combine analyte and ligand IDs
            # (same convention as the Octet importer) so each gets its own fit
            if len(set(x for x in loading_ids if x is not None)) > 1:
                sample_ids = [s if l is None else f'{s} - {l}' for s, l in zip(sample_ids, loading_ids)]

        rows = []
        for sensor in self.sensor_names:
            for k, idx in enumerate(assoc_idx):
                rows.append({'Sensor'                  : sensor,
                             'Analyte_location'        : self.df_steps['Column_location'].iloc[idx],
                             'Concentration_micromolar': float(conc_per_sensor[sensor][k]),
                             'SampleID'                : sample_ids[k],
                             'Loading_location'        : self.df_steps['Loading_location'].iloc[idx]})

        df = pd.DataFrame(rows)

        # Replicate number: repeated (loading, analyte location) combinations per sensor,
        # in the same order as the association steps (consistent with get_step_xy)
        df['Replicate'] = df.groupby(['Sensor', 'Loading_location', 'Analyte_location']).cumcount() + 1

        df['Experiment'] = self.name

        self.ligand_conc_df = df[['Sensor', 'Analyte_location', 'Concentration_micromolar',
                                  'SampleID', 'Replicate', 'Loading_location', 'Experiment']]

        return None
