import re
import math
import itertools

import numpy  as np
import pandas as pd

import matplotlib.colors as mcolors

from ..utils.palettes import set1_colors, set3_colors, VIRIDIS

__all__ = [
    'guess_experiment_name', 'etree_to_dict', 'combine_dicts','guess_experiment_type',
    'detect_time_list_continuos', 'find_loading_column', 'concat_signal_lst',
    'expand_parameter_list', 'combine_sequences', 'get_palette', 'get_plotting_df',
    'subset_data', 'sample_type_to_letter', 'get_colors_from_numeric_values',
    'if_string_to_list','find_index_of_previous_step_type','number_is_in_word']

def if_string_to_list(string_or_list):
    """
    Convert a string to a list if necessary.

    Parameters
    ----------
    string_or_list : str or list
        Input string or list to convert.

    Returns
    -------
    list
        Converted list.
        
    Raises
    ------
    TypeError
        If input is neither a string nor a list.
    """
    if isinstance(string_or_list, str):
        return [string_or_list]
    elif isinstance(string_or_list, list):
        return string_or_list
    else:
        raise TypeError("Input must be a string or a list of strings.")

def guess_experiment_name(frd_file):
    """
    Given a certain frd file, try to guess the experiment name.
    
    The frd file is the format exported by the Biacore software.

    Parameters
    ----------
    frd_file : str
        Path to the FRD file.
        
    Returns
    -------
    str
        Experiment name extracted from the file, or 'Experiment' if no name is found.
    """

    # Read the file content
    with open(frd_file, 'r') as file:
        content = file.read()

    # Define the regex pattern
    pattern = r'<ExperimentInfo Name="([^"]+)">'

    # Find the first match
    match = re.search(pattern, content)

    if match:
        experiment_name = match.group(1)  # Get the captured group

        # Remove any sequence of six numbers in a row (we assume it is a timestamp)
        experiment_name = re.sub(r'\b\d{6}\b', '', experiment_name)

        # Remove all extra spaces
        experiment_name = re.sub(r'\s+', ' ', experiment_name).strip()

        return experiment_name

    else:

        return 'Experiment'

def etree_to_dict(tree):
    """
    Converts XML tree to dictionary.
    
    Parameters
    ----------
    tree : ElementTree
        XML tree to convert.
        
    Returns
    -------
    dict
        Dictionary representation of the XML tree.
    """
    tree_dict = {}
    for elem in tree:
        tree_dict[elem.tag] = elem.text
    return tree_dict

def combine_dicts(list_dicts):
    """
    Combines dictionaries with the same keys into a dictionary with lists as values.
    
    Parameters
    ----------
    list_dicts : list
        List of dictionaries to combine.
        
    Returns
    -------
    dict
        Combined dictionary with lists as values.
        
    Notes
    -----
    All dictionaries must have the same keys.
    """
    new_dict = {}
    for key in list_dicts[0].keys():
        new_dict[key] = [one_dict[key] for one_dict in list_dicts]
    return new_dict

def guess_experiment_type(files):
    """
    Given file paths, try to guess if they correspond to surface-based or solution-based binding experiments.

    Parameters
    ----------
    files : str or list
        Path or list of paths to the experiment files.
        
    Returns
    -------
    str
        'surface' or 'solution' indicating the experiment type.
    """

    # convert to list if it is a string
    if isinstance(files, str):
        files = [files]

    for file in files:
        if file.endswith('.frd'):
            return 'surface'

    for file in files:
        if 'ExperimentStep.ini' in file:
            return 'surface'

    for file in files:
        if file.endswith('.csv'):

            # Find if we have a line with the protein concentration
            with open(file, 'r') as f:
                first_line = f.read().splitlines()[0]
                if 'Protein_concentration_micromolar' in first_line:
                    return 'solution'
                else:
                    return 'surface'

    return 'surface'

def detect_time_list_continuos(assoc_time_lst, disso_time_lst, tolerance=3):
    """
    Detect which association steps come directly after a dissociation step.
    
    Useful for single-cycle kinetics.

    Parameters
    ----------
    assoc_time_lst : list
        List of association time arrays.
    disso_time_lst : list
        List of dissociation time arrays.
    tolerance : float, optional
        Tolerance for the time difference (in seconds), default is 3.
        
    Returns
    -------
    list
        List of booleans indicating if the association phase has a dissociation phase just before.
    """

    continuos = []

    for i, element in enumerate(assoc_time_lst):

        if i == 0:

            continuos.append(element[0] < tolerance)

        else:

            prev_time = disso_time_lst[i-1][-1]
            continuos.append(element[0] < prev_time+tolerance)

    return continuos

def find_loading_column(df, step_number):
    """
    Find the loading column before the given step number.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns 'Column', 'Type' and 'StepNumber'.
    step_number : int
        The step number to find the loading column before.
        
    Returns
    -------
    str or None
        The name of the loading column before the given step number, or None if no loading column is found.
    """

    # Filter the dataframe to get the loading columns before the given step number
    loading_columns = df[(df['Type'] == 'LOADING') & (df['StepNumber'] < step_number)]

    # If there are no loading columns, return None
    if loading_columns.empty:
        return None

    # Get the last loading column before the given step number
    last_loading_column = loading_columns.iloc[-1]

    # Return the column name
    return last_loading_column['Column']

def concat_signal_lst(signal_lst):
    """
    Concatenate a list of signals into a single array.

    Parameters
    ----------
    signal_lst : list
        List of signals to concatenate, each signal is a numpy array.

    Returns
    -------
    np.ndarray
        Concatenated signal.
    """

    return np.concatenate(signal_lst)

def expand_parameter_list(parameter_lst, id_list):
    """
    Expand a parameter list based on ID mapping.
    
    Given a list of n-parameters, such as [1,3] and another list
    containing IDs, such as [0,0,0,0,0,1,1],
    create a new list where the elements are repeated
    according to the IDs. In this case: [1,1,1,1,1,3,3]

    Parameters
    ----------
    parameter_lst : list
        List of n-Parameters.
    id_list : list
        List of m-IDs mapping to parameter indices.
        
    Returns
    -------
    list
        m-Parameters list expanded according to the IDs.
    """

    expanded_parameters = [parameter_lst[i] for i in id_list]
    return expanded_parameters

def combine_sequences(seq1, seq2):
    """
    Combine two sequences to generate all possible combinations of their elements.
    
    Parameters
    ----------
    seq1 : list
        First sequence of elements.
    seq2 : list
        Second sequence of elements.
        
    Returns
    -------
    list
        A list of tuples, where each tuple contains one element from seq1 and one from seq2.
    """
    return list(itertools.product(seq1, seq2))


def get_palette(n_colors):
    """
    Get a color palette with the specified number of colors.
    
    Parameters
    ----------
    n_colors : int
        Number of colors to include in the palette.
        
    Returns
    -------
    list
        List of hex color codes.
        
    Notes
    -----
    Uses Set1 from ColorBrewer for n_colors <= 9,
    Set3 for n_colors <= 12, and interpolated colors for n_colors > 12.
    """
    if n_colors <= 9:
        # Use Set1 from ColorBrewer
        return set1_colors[:n_colors]
    elif n_colors <= 12:
        # Use Set3 from ColorBrewer
        return set3_colors[:n_colors]
    else:

        # Create a colormap from the original palette
        cmap = mcolors.LinearSegmentedColormap.from_list("set3_interp", set3_colors)
        interpolated_colors = [mcolors.to_hex(cmap(i / (n_colors - 1))) for i in range(n_colors)]

        return interpolated_colors

def get_plotting_df(ID, labels=None):
    """
    Create a DataFrame for plotting purposes with colors and labels.
    
    Parameters
    ----------
    ID : list
        List of unique identifiers for each entry.
    labels : list, optional
        List of labels corresponding to each ID. Defaults to None, in which case the ID values are used as labels.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing Internal_ID, Color, Legend, and Show columns.
    """
    if labels is None:
        labels = ID
    n_total = len(labels)
    df = pd.DataFrame({
        "Internal_ID": ID,
        "Color": get_palette(n_total),
        "Legend": labels,
        "Show": [True] * n_total
    })
    return df

def subset_data(data, max_points=200):
    """
    Subset data to a maximum number of points for plotting.
    
    Parameters
    ----------
    data : list or np.ndarray
        Data to be subsetted.
    max_points : int, optional
        Maximum number of points to retain, default is 200.
        
    Returns
    -------
    list or np.ndarray
        Subsetted data with at most max_points points.
    """
    total_points = len(data)
    if total_points > max_points:
        step = math.ceil(total_points / max_points)
        data = data[::step]
    return data

def sample_type_to_letter(string):
    """
    Convert sample type string to a single letter code.
    
    Parameters
    ----------
    string : str
        Sample type string.
        
    Returns
    -------
    str
        Single letter code representing the sample type.
        
    Notes
    -----
    Mapping:
    - KSAMPLE/SAMPLE -> S
    - LOADING/KLOADING -> L
    - NEUTRALIZATION -> N
    - REGENERATION -> R
    - BASELINE/BUFFER -> B
    - Others -> First letter of the string
    """

    mapping = {
        "KSAMPLE": "S",
        "SAMPLE": "S",
        "LOADING": "L",
        "KLOADING": "L",
        "NEUTRALIZATION": "N",
        "REGENERATION": "R",
        "BASELINE": "B",
        "BUFFER": "B"
    }
    if string not in mapping:
        return string[0] if string else ""
    return mapping[string]

def get_colors_from_numeric_values(values, min_val, max_val, use_log_scale=True):
    """
    Map numeric values to colors in the VIRIDIS palette based on a specified range.
    
    Parameters
    ----------
    values : list or np.ndarray
        Numeric values to map to colors.
    min_val : float
        Minimum value of the range.
    max_val : float
        Maximum value of the range.
    use_log_scale : bool, optional
        Whether to use logarithmic scaling for the values, default is True.
        
    Returns
    -------
    list
        List of hex color codes corresponding to the input values.
    """
    values = np.array(values)
    if use_log_scale:
        min_val = np.log10(min_val)
        max_val = np.log10(max_val)
        values = np.log10(values)
    seq = np.linspace(min_val, max_val, 21)
    idx = [np.argmin(np.abs(v - seq)) for v in values]
    return [VIRIDIS[i] for i in idx]

def find_index_of_previous_step_type(df_steps, step_index, step_type):
    """
    Find the index of the first step of a given type before a specified step index.
    
    Parameters
    ----------
    df_steps : pd.DataFrame
        DataFrame containing step information with a 'Type' column.
    step_index : int
        Current step index.
    step_type : str
        Type of step to search for.
        
    Returns
    -------
    int or None
        Index of the first step of the specified type before the given step index,
        or None if no matching step is found.
    """
    i = None
    # Find the index of the first step_type, from the single cycle
    # Iterate over the previous steps, two at a time, until we find a step that is not a step of the same type

    for idx in range(step_index-2, 0, -2):
        if df_steps['Type'][idx] != step_type:
            i = idx
            break

    return i