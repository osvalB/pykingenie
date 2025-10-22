import pytest
import os
import numpy as np

from pykingenie.main import KineticsAnalyzer
from pykingenie.octet import OctetExperiment

from pykingenie.utils.plotting import (
    plot_plate_info,
    plot_traces_all,
    config_fig,
    plot_steady_state,
    plot_association_dissociation
)

pyKinetics = KineticsAnalyzer()

folder = "./test_files/"
frd_files = os.listdir(folder)

frd_files = [os.path.join(folder, file) for file in frd_files if file.endswith('.frd') and file.startswith('230309')]
frd_files.sort()

bli = OctetExperiment('test')
bli.read_sensor_data(frd_files)

pyKinetics.add_experiment(bli, 'test_octet')

def test_plot_plate_info():

    # Check raise error if experiment does not have the required attributes
    with pytest.raises(AttributeError):
        bli.sample_row = None
        plot_plate_info(pyKinetics, 'test_octet', font_size=18)

    fmf_file  = os.path.join(folder, '230309_ExpMethod.fmf')

    # Read the plate information
    bli.read_sample_plate_info(fmf_file)

    fig = plot_plate_info(pyKinetics, 'test_octet', font_size=18)

    # Check if the figure is created
    assert fig is not None, "The figure should be created successfully."

def test_plot_traces_all():

    legends_df = pyKinetics.get_legends_table()

    # Deselect one sensor to test the plotting function
    legends_df.iloc[0, legends_df.columns.get_loc('Show')] = False

    fig = plot_traces_all(pyKinetics, legends_df,
                          plot_width=16, plot_height=14, plot_type='png',
                          font_size=18, show_grid_x=False, show_grid_y=False,
                          marker_size=1, line_width=2)
    
    # Check if the figure is created
    assert fig is not None, "The figure should be created successfully."
    
    # Test config_fig function to export the figure
    config_fig(fig, "fig_test", export_format='png',
               plot_width=10, plot_height=6, scale_factor=50, 
               save=True)
    
    # check that the file was created
    assert os.path.exists("fig_test.png"), "The figure should be saved as 'fig_test.png'."

    # check runtime error if saving fails
    with pytest.raises(RuntimeError):
        config_fig(fig, "fig_test", export_format='invalid_format',
                   plot_width=10, plot_height=6, scale_factor=50, 
                   save=True)

    # remove the saved file after the test
    os.remove("fig_test.png")

def test_plot_steady_state():

    bli.align_association(bli.sensor_names)
    bli.align_dissociation(bli.sensor_names)
    bli.subtraction(['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1'], 'H1')

    pyKinetics.merge_ligand_conc_df()

    df = pyKinetics.combined_ligand_conc_df

    # Select only the first 8 rows for testing
    df = df.iloc[:8, :].copy()

    pyKinetics.generate_fittings(df)

    pyKinetics.submit_steady_state_fitting()

    fig = plot_steady_state(
        pyKinetics,
        plot_width=30,
        plot_height=20,
        plot_type='png',
        font_size=18,
        show_grid_x=True,
        show_grid_y=True,
        marker_size=5,
        line_width=2,
        plot_fit=False)
    
    # Check if the figure is created
    assert fig is not None, "The figure should be created successfully."

    # Check plotting with fit
    fig = plot_steady_state(
        pyKinetics,
        plot_fit=True)
    
    # Check if the figure is created
    assert fig is not None, "The figure with fit should be created successfully."

def test_plot_association_dissociation():

    pyKinetics.submit_kinetics_fitting(fitting_model='one_to_one',
                                fitting_region='association_dissociation',
                                linkedSmax=True)

    fig = plot_association_dissociation(
        pyKinetics,
        plot_width=26,
        plot_height=20,
        plot_type='png',
        font_size=14,
        show_grid_x=False,
        show_grid_y=False,
        marker_size=4,
        line_width=2,
        split_by_smax_id=True,
        max_points_per_plot=2000,
        smooth_curves_fit=False,
        rolling_window=0.1,
        vertical_spacing=0.08)
    
    # Check if the figure is created
    assert fig is not None, "The figure should be created successfully."

    # Test plotting fit smoothing and non split by Smax ID

    fig = plot_association_dissociation(
        pyKinetics,
        split_by_smax_id=False,
        smooth_curves_fit=True)
    
    # Check if the figure is created
    assert fig is not None, "The figure with fit smoothing should be created successfully."


