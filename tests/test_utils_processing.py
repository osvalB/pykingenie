import numpy as np
import pandas as pd
import pytest
import os

from pykingenie.utils.processing import (
    guess_experiment_name,
    guess_experiment_type,
    detect_time_list_continuos,
    get_palette,
    subset_data,
    sample_type_to_letter,
    combine_sequences,
    get_colors_from_numeric_values,
    if_string_to_list,
    find_loading_column,
    find_index_of_previous_step_type
    )

def test_guess_experiment_name():
    # Create a temporary file with known content
    test_file = './test_files/230309_001.frd'

    name = guess_experiment_name(test_file)
    assert name == '5th interaction with antibodie reg and imd - wt mut and wtDTT', "The guessed experiment name should be '5th interaction with antibodie reg and imd - wt mut and wtDTT'"

    test_file = './test_files/empty_file.frd'
    name = guess_experiment_name(test_file)
    assert name == 'Experiment'

def test_guess_experiment_type():

    non_existing_file = 'sarasaa'
    file_type = guess_experiment_type(non_existing_file)
    assert file_type == 'surface', "The default guessed experiment type should be 'surface'"

    mock_file = 'ExperimentStep.ini'
    file_type = guess_experiment_type(mock_file)
    assert file_type == 'surface', "The guessed experiment type should be 'surface'"

    mock_file = 'xxxx.frd'
    file_type = guess_experiment_type(mock_file)
    assert file_type == 'surface', "The guessed experiment type should be 'surface'"

    test_file = "./test_files/multi_cycle_kingenie.csv"
    file_type = guess_experiment_type(test_file)
    assert file_type == 'surface', "The guessed experiment type should be 'solution'"

    test_file = "./test_files/kingenie_solution.csv"
    file_type = guess_experiment_type(test_file)
    assert file_type == 'solution', "The guessed experiment type should be 'solution'"

def test_if_string_to_list():

    test_lst = if_string_to_list('test_string')
    assert isinstance(test_lst, list), "The output should be a list"

    test_lst = if_string_to_list(['test_string'])
    assert isinstance(test_lst, list), "The output should be a list"

    # check type error if input is not a string or a list
    with pytest.raises(TypeError):
        if_string_to_list(123)

def test_detect_time_list_continuos():

    assoc_time_lst = [ np.arange(0, 10, 1),  np.arange(20,30, 1)]
    disso_time_lst = [ np.arange(10, 20, 1)]

    result = detect_time_list_continuos(assoc_time_lst, disso_time_lst,tolerance=3)

    assert result == [True,True], "The first is True because it is the first association, the second is True because the association comes directly after the dissociation"

    assoc_time_lst = [ np.arange(0, 10, 1), np.arange(40, 50, 1)]
    disso_time_lst = [ np.arange(10, 20, 1)]

    result = detect_time_list_continuos(assoc_time_lst, disso_time_lst,tolerance=3)

    assert result == [True,False], "The first is True because it is the first association, the second is False because the association does not come directly after the dissociation"

def test_get_palette():
    # Test with a valid number of colors
    palette = get_palette(5)
    assert len(palette) == 5, "The palette should contain 5 colors"

    palette = get_palette(10)
    assert len(palette) == 10, "The palette should contain 10 colors"


    # Test with a number of colors greater than the available colors
    palette = get_palette(20)
    assert len(palette) == 20, "The palette should contain 20 colors even if it exceeds the default size"

def test_find_loading_column():

    # Create a mock DataFrame
    data = {
        'StepNumber': [1,1],
        'Type': ['hello', 'world']
    }

    df = pd.DataFrame(data)

    # Test with a valid step number
    loading_col = find_loading_column(df, step_number=1)
    assert loading_col is None, "There should be no loading column"

def test_subset_data():

    data = np.arange(200)

    subset = subset_data(data, max_points=100)

    assert len(subset) == 100, "The subset should contain 100 points"

def test_get_colors_from_numeric_values():

    values = [1,10,100]
    min_value = 0.1
    max_value = 1000

    colors = get_colors_from_numeric_values(values, min_value, max_value)

    assert len(colors) == len(values), "The number of colors should match the number of values" 
    
def test_sample_type_to_letter():

    assert sample_type_to_letter('SAMPLE') == 'S', "The word SAMPLE should map to the letter S"
    assert sample_type_to_letter('') == '', "An empty string should return an empty string"

def test_combine_sequences():

    seq1 = [1]
    seq2 = [3, 4]
    combined = combine_sequences(seq1, seq2)

    expected = [(1, 3), (1, 4)]
    np.testing.assert_array_equal(combined, expected, "The combined sequence should match the expected result")

def test_find_index_of_previous_step_type():

    # Create a mock DataFrame
    data = {
        'Type': ['BASELINE','ASSOC', 'DISASSOC','ASSOC', 'DISASSOC']
    }

    df_steps = pd.DataFrame(data)

    # Test with a valid step index and type
    step_index = 3
    step_type = 'DISASSOC'
    index = find_index_of_previous_step_type(df_steps, step_index, step_type)

    assert index == 1, "The index of the previous association should be 1"
