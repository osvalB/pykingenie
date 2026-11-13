# Note: this file contains AI-generated code, written with AI coding assistants (PR #24).
import numpy as np
import pandas as pd
import pytest

from pykingenie import SurfaceCsvExperiment
from pykingenie.main import KineticsAnalyzer
from pykingenie.surface_csv import (
    map_step_type,
    location_to_column,
    read_steps_file,
    read_traces_file,
)

steps_file  = "./test_files/surface_csv_steps.txt"
traces_file = "./test_files/surface_csv_traces.csv"

# Values used to simulate the synthetic test files
concs     = [0.025, 0.05, 0.1, 0.2, 0.4]   # µM
kd_true   = 0.05                            # µM
koff_true = 0.005                           # 1/s


def load(**kwargs):
    exp = SurfaceCsvExperiment('surface_csv_test')
    exp.read_files(steps_file, traces_file, **kwargs)
    return exp


def test_helpers():

    assert map_step_type('Association') == 'ASSOC'
    assert map_step_type(' dissociation ') == 'DISASSOC'
    assert map_step_type('Loading') == 'LOADING'
    assert map_step_type('Baseline') == 'BASELINE'
    assert map_step_type('Regeneration') == 'KREGENERATION'
    assert map_step_type('Custom step') == 'CUSTOM STEP'

    assert location_to_column('p1A7', 1) == '7'
    assert location_to_column('B12', 1) == '12'
    assert location_to_column('3', 1) == '3'
    assert location_to_column('plate', 4) == '4'
    assert location_to_column(np.nan, 5) == '5'


def test_read_traces_file_skips_metadata():

    df = read_traces_file(traces_file)

    assert list(df.columns) == ['X Values', 'A1', 'B1']
    assert df.iloc[0, 0] == 0
    assert np.isclose(df.iloc[1, 0], 0.5)
    assert df.shape[0] == 2760


def test_read_steps_file():

    df = read_steps_file(steps_file)

    assert len(df) == 13
    assert df['Type'].tolist()[:4] == ['BASELINE', 'LOADING', 'BASELINE', 'ASSOC']
    assert np.isclose(df['Time'].sum(), 1380)


def test_read_files():

    exp = load(concentrations=concs)

    assert exp.traces_loaded
    assert exp.sensor_names == ['A1', 'B1']
    assert exp.sensor_names_unique == ['surface_csv_test A1', 'surface_csv_test B1']
    assert exp.no_sensors == 2
    assert exp.no_steps == 13

    # One array per step and per sensor, covering all data points
    assert len(exp.xs) == 2 and len(exp.xs[0]) == 13
    assert sum(len(x) for x in exp.xs[0]) == 2760
    assert len(exp.xs[0][0]) == 120   # 60 s baseline sampled every 0.5 s
    assert len(exp.xs[0][-1]) == 600  # 300 s final dissociation

    # Steps dataframe
    assert (exp.df_steps['Type'] == 'ASSOC').sum() == 5
    assert (exp.df_steps['Type'] == 'DISASSOC').sum() == 5
    assert exp.df_steps['Column_location'].tolist()[3] == '4'
    assert exp.df_steps['Start'].tolist()[3] == 240
    assoc = exp.df_steps[exp.df_steps['Type'] == 'ASSOC']
    assert (assoc['Loading_location'] == '2').all()

    # Ligand concentration dataframe: one row per sensor and association step
    df = exp.ligand_conc_df
    assert len(df) == 10
    assert df[df['Sensor'] == 'A1']['Concentration_micromolar'].tolist() == concs
    assert df['Analyte_location'].tolist()[:5] == ['4', '5', '6', '7', '8']
    assert (df['Replicate'] == 1).all()
    assert (df['Experiment'] == 'surface_csv_test').all()


def test_concentrations_per_sensor():

    exp = load(concentrations={'A1': concs, 'B1': [1, 2, 3, 4, 5]})
    df  = exp.ligand_conc_df

    assert df[df['Sensor'] == 'B1']['Concentration_micromolar'].tolist() == [1, 2, 3, 4, 5]


def test_missing_concentrations_warns():

    with pytest.warns(UserWarning, match='No analyte concentrations'):
        exp = load()

    assert exp.ligand_conc_df['Concentration_micromolar'].isna().all()


def test_wrong_number_of_concentrations():

    with pytest.raises(ValueError, match='Expected 5 concentrations'):
        load(concentrations=[1, 2])

    with pytest.raises(ValueError, match='Missing concentrations'):
        load(concentrations={'A1': concs})


def test_concentration_and_sample_id_columns_in_steps_file(tmp_path):

    df = pd.read_csv(steps_file, sep='\t')
    conc_col = [np.nan] * len(df)
    for i, c in zip([3, 5, 7, 9, 11], concs):
        conc_col[i] = c
    df['Concentration'] = conc_col
    df['Sample ID'] = 'analyte_X'

    new_steps = tmp_path / 'steps.csv'
    df.to_csv(new_steps, index=False)  # comma-delimited this time

    exp = SurfaceCsvExperiment('col_conc')
    exp.read_files(str(new_steps), traces_file)

    assert exp.ligand_conc_df['Concentration_micromolar'].tolist()[:5] == concs
    assert (exp.ligand_conc_df['SampleID'] == 'analyte_X').all()


def test_missing_required_column(tmp_path):

    bad = tmp_path / 'bad_steps.txt'
    bad.write_text('Step Name\tStep Type Name\nA\tAssociation\n')

    with pytest.raises(ValueError, match="'Time'"):
        read_steps_file(str(bad))


def test_no_association_step(tmp_path):

    bad = tmp_path / 'no_assoc.txt'
    bad.write_text('Step Time\tStep Type Name\n60\tBaseline\n1320\tDissociation\n')

    with pytest.raises(ValueError, match='association'):
        SurfaceCsvExperiment().read_files(str(bad), traces_file, concentrations=[])


def test_traces_without_header(tmp_path):

    f = tmp_path / 'traces.csv'
    f.write_text('0,1,2\n1,2,3\n2,3,4\n')

    df = read_traces_file(str(f))

    assert list(df.columns) == ['Time', 'Sensor 1', 'Sensor 2']


def test_traces_without_data(tmp_path):

    f = tmp_path / 'empty.csv'
    f.write_text('Time,A1\n')

    with pytest.raises(ValueError, match='No numeric data'):
        read_traces_file(str(f))


def test_traces_metadata_without_delimiter_and_trailing_columns(tmp_path):

    f = tmp_path / 'traces.csv'
    f.write_text('Exported traces\nTime,A1,\n0,1,\n1,2,\n')

    df = read_traces_file(str(f))

    # The metadata line has no delimiter; the empty trailing column is dropped
    assert list(df.columns) == ['Time', 'A1']
    assert df['A1'].tolist() == [1, 2]


def test_steps_longer_than_traces_warns(tmp_path):

    df = pd.read_csv(steps_file, sep='\t')
    extra = df.iloc[[-1]].copy()
    extra['Step Name '] = 'Extra'
    df = pd.concat([df, extra], ignore_index=True)
    steps = tmp_path / 'steps.txt'
    df.to_csv(steps, sep='\t', index=False)

    with pytest.warns(UserWarning, match='No data points found'):
        SurfaceCsvExperiment().read_files(str(steps), traces_file, concentrations=concs)


def test_get_step_xy_single_cycle():

    exp = load(concentrations=concs)

    # Third association step, located in column 6, loaded from column 2
    x_assoc = exp.get_step_xy('A1', 2, 6, 'ASSOC', 1, 'x')
    x_diss  = exp.get_step_xy('A1', 2, 6, 'DISASSOC', 1, 'x')
    y_assoc = exp.get_step_xy('A1', 2, 6, 'ASSOC', 1, 'y')

    # Single-cycle: time is relative to the start of the first association
    assert np.isclose(x_assoc[0], 360)
    assert np.isclose(x_diss[0], 480)
    assert len(y_assoc) == 240


def test_kinetics_fit_recovers_parameters():

    exp = load(concentrations=concs)
    exp.subtraction(['A1'], 'B1', inplace=True)
    exp.align_association(exp.sensor_names, inplace=True)

    ka = KineticsAnalyzer()
    ka.add_experiment(exp, 'surface_csv_test')
    ka.merge_ligand_conc_df()

    df = ka.combined_ligand_conc_df.copy()
    df = df[df['Sensor'] == 'A1 - B1']

    ka.generate_fittings(df)
    ka.submit_kinetics_fitting(fitting_model='one_to_one')

    fitter = list(ka.fittings.values())[0]
    params = fitter.fit_params_kinetics

    assert np.isclose(float(params['Kd [µM]'].iloc[0]), kd_true, rtol=0.15)
    assert np.isclose(float(params['k_off [1/s]'].iloc[0]), koff_true, rtol=0.15)


# ---------------------------------------------------------------------------
# Multi-cycle kinetics (MCK): 8 sensors, one concentration per sensor row,
# two ligands loaded from different columns, regeneration in between.
# ---------------------------------------------------------------------------

mck_steps_file  = "./test_files/surface_csv_mck_steps.txt"
mck_traces_file = "./test_files/surface_csv_mck_traces.csv"

mck_sensors  = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1']
mck_row_conc = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.0]  # µM, H1 = buffer reference
mck_concs    = {s: [c, c] for s, c in zip(mck_sensors, mck_row_conc)}

# Values used to simulate the MCK test files
mck_truth = {'sampleAB - ligandA': (0.05, 0.01),    # Kd [µM], koff [1/s]
             'sampleAB - ligandB': (0.004, 0.002)}


def load_mck():
    exp = SurfaceCsvExperiment('mck_test')
    exp.read_files(mck_steps_file, mck_traces_file, concentrations=mck_concs)
    return exp


def test_read_mck_files():

    exp = load_mck()

    assert exp.sensor_names == mck_sensors
    assert exp.no_steps == 11
    assert 'KREGENERATION' in exp.df_steps['Type'].tolist()

    assoc = exp.df_steps[exp.df_steps['Type'] == 'ASSOC']
    assert assoc['Column_location'].tolist() == ['4', '4']
    assert assoc['Loading_location'].tolist() == ['2', '5']

    df = exp.ligand_conc_df
    assert len(df) == 16
    # Different ligands loaded -> sample ID combines analyte and ligand
    assert sorted(df['SampleID'].unique()) == ['sampleAB - ligandA', 'sampleAB - ligandB']
    assert (df['Replicate'] == 1).all()
    a1 = df[df['Sensor'] == 'A1']
    assert a1['Concentration_micromolar'].tolist() == [0.8, 0.8]
    assert a1['Loading_location'].tolist() == ['2', '5']


def test_mck_get_step_xy_uses_loading_location():

    exp = load_mck()

    # Same analyte column (4), different loading columns (2 and 5)
    x1 = exp.get_step_xy('A1', 2, 4, 'ASSOC', 1, 'x')
    x2 = exp.get_step_xy('A1', 5, 4, 'ASSOC', 1, 'x')
    xd = exp.get_step_xy('A1', 5, 4, 'DISASSOC', 1, 'x')

    # Multi-cycle: each association starts at zero, dissociation continues from it
    assert x1[0] == 0 and x2[0] == 0
    assert np.isclose(xd[0], 180)

    y1 = exp.get_step_xy('A1', 2, 4, 'ASSOC', 1, 'y')
    y2 = exp.get_step_xy('A1', 5, 4, 'ASSOC', 1, 'y')
    assert not np.allclose(y1, y2)


def test_single_loading_id_is_not_appended(tmp_path):

    df = pd.read_csv(mck_steps_file, sep='\t')
    df.loc[df['Step Type Name'] == 'Loading', 'Sample ID'] = 'ligandA'
    steps = tmp_path / 'steps.txt'
    df.to_csv(steps, sep='\t', index=False)

    exp = SurfaceCsvExperiment()
    exp.read_files(str(steps), mck_traces_file, concentrations=mck_concs)

    assert (exp.ligand_conc_df['SampleID'] == 'sampleAB').all()


def test_mck_kinetics_fit_recovers_parameters():

    exp = load_mck()
    exp.subtraction(mck_sensors[:-1], 'H1', inplace=True)
    exp.align_association(exp.sensor_names, inplace=True)

    ka = KineticsAnalyzer()
    ka.add_experiment(exp, 'mck_test')
    ka.merge_ligand_conc_df()
    ka.generate_fittings(ka.combined_ligand_conc_df.copy())

    assert sorted(ka.fittings.keys()) == sorted(mck_truth.keys())

    ka.submit_steady_state_fitting()
    ka.submit_kinetics_fitting(fitting_model='one_to_one', shared_smax=True)

    for name, fitter in ka.fittings.items():

        assert not fitter.is_single_cycle
        assert len(fitter.lig_conc_lst) == 7  # zero-concentration reference excluded

        kd_true, koff_true = mck_truth[name]
        params = fitter.fit_params_kinetics

        assert np.isclose(float(params['Kd [µM]'].iloc[0]), kd_true, rtol=0.1)
        assert np.isclose(float(params['k_off [1/s]'].iloc[0]), koff_true, rtol=0.1)
