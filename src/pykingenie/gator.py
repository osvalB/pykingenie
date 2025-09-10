from .surface_exp      import SurfaceBasedExperiment
import os
import json
import zipfile

import numpy as np
import pandas as pd

from .utils.processing import find_loading_column

class GatorExperiment(SurfaceBasedExperiment):
    """
    Class to handle Gator Bio-Layer Interferometry (BLI) experiments.
    
    This class extends the SurfaceBasedExperiment to provide functionality
    specific to Gator BLI instruments, including parsing experiment files
    and loading sensor data.
    
    Parameters
    ----------
    name : str, optional
        Name of the experiment, default is 'Gator_experiment'.
    """

    def __init__(self, name = 'Gator_experiment'):

        super().__init__(name, 'Gator_experiment')

    def read_all_gator_data(self, files, names=None):
        """
        Read all Gator data files for the experiment.
        
        This method reads all necessary files for a Gator experiment, including
        the experiment settings, step information, and sensor data.
        
        Parameters
        ----------
        files : str or list
            Path to a zip file containing Gator data files or a list of file paths.
            If a zip file is provided, it will be extracted to the same directory.
        names : list, optional
            List of file names corresponding to the file paths in `files`.
            If None, the file names will be extracted from the file paths.
            
        Returns
        -------
        None
            Updates instance attributes with the loaded experiment data.
        """

        # Check if files is a string
        if isinstance(files, str):
            # Check if files is a zipfile and extract it
            if files.endswith('.zip'):
                with zipfile.ZipFile(files, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(files))
                files = [os.path.join(os.path.dirname(files), file) for file in zip_ref.namelist()]

        self.traces_loaded = False

        if names is None:
            names = [os.path.basename(file) for file in files]

        # Find the file ExperimentStep.ini and read it
        for file, name in zip(files, names):
            if 'ExperimentStep.ini' in name:
                self.read_experiment_ini(file)

        # Find the file Setting.ini and read it
        for file, name in zip(files, names):
            if 'Setting.ini' in name:
                self.read_settings_ini(file)

        # Find the sensor csvs and read them
        self.read_sensor_data(files, names)

        return None

    def read_experiment_ini(self, file, nr_max_columns=12):
        """
        Read the experiment initialization file.
        
        Parses the ExperimentStep.ini file that contains information about the experimental steps,
        their durations, and types (baseline, association, dissociation, etc.).
        
        Parameters
        ----------
        file : str
            Path to the ExperimentStep.ini file.
        nr_max_columns : int, optional
            Maximum number of columns in the plate, default is 12.
            
        Returns
        -------
        None
            Updates the df_steps attribute with a DataFrame containing step information.
            
        Notes
        -----
        Example format for ExperimentStep.ini:
        
            [Experiment]
            Num=4
            [Experiment1]
            Step1=0,0,60,0
            Step2=1,60,104,1
            Step3=5,104,164,0
            Step4=6,164,464,2
            Step5=7,464,764,3
            Num=5
            ProbeIndex=12
            [ExperimentLog]
            Step1="{"StepType":0,"AssayNumber":1,"PlateIndex":1,"PlateColumnType":12,"RowLocation":0,"Location":0,"Speed":400,"Time":600,"SumTime":0,"ProbeIndex":0,"PickerUsed":1,"AllChannelUsed":[255],"ChannelUsed":0,"bAssayFirstStep":false,"kst":0,"Status":0}"
            Step2="{"StepType":3,"AssayNumber":1,"PlateIndex":1,"PlateColumnType":12,"RowLocation":0,"Location":1,"Speed":1000,"Time":5,"SumTime":0,"ProbeIndex":0,"PickerUsed":1,"AllChannelUsed":[255],"ChannelUsed":0,"bAssayFirstStep":false,"kst":0,"Status":0}"
            Step3="{"StepType":4,"AssayNumber":1,"PlateIndex":1,"PlateColumnType":12,"RowLocation":0,"Location":2,"Speed":1000,"Time":5,"SumTime":0,"ProbeIndex":0,"PickerUsed":1,"AllChannelUsed":[255],"ChannelUsed":0,"bAssayFirstStep":false,"kst":0,"Status":0}"
            Step4="{"StepType":3,"AssayNumber":1,"PlateIndex":1,"PlateColumnType":12,"RowLocation":0,"Location":1,"Speed":1000,"Time":5,"SumTime":0,"ProbeIndex":0,"PickerUsed":1,"AllChannelUsed":[255],"ChannelUsed":0,"bAssayFirstStep":false,"kst":0,"Status":0}"
            Step5="{"StepType":4,"AssayNumber":1,"PlateIndex":1,"PlateColumnType":12,"RowLocation":0,"Location":2,"Speed":1000,"Time":5,"SumTime":0,"ProbeIndex":0,"PickerUsed":1,"AllChannelUsed":[255],"ChannelUsed":0,"bAssayFirstStep":false,"kst":0,"Status":0}"
            Step6="{"StepType":3,"AssayNumber":1,"PlateIndex":1,"PlateColumnType":12,"RowLocation":0,"Location":1,"Speed":1000,"Time":5,"SumTime":0,"ProbeIndex":0,"PickerUsed":1,"AllChannelUsed":[255],"ChannelUsed":0,"bAssayFirstStep":false,"kst":0,"Status":0}"
            Step7="{"StepType":4,"AssayNumber":1,"PlateIndex":1,"PlateColumnType":12,"RowLocation":0,"Location":2,"Speed":1000,"Time":5,"SumTime":0,"ProbeIndex":0,"PickerUsed":1,"AllChannelUsed":[255],"ChannelUsed":0,"bAssayFirstStep":false,"kst":0,"Status":0}"
            Step8="{"StepType":2,"AssayNumber":1,"PlateIndex":0,"PlateColumnType":12,"RowLocation":0,"Location":12,"Speed":1000,"Time":60,"SumTime":0,"ProbeIndex":0,"PickerUsed":1,"AllChannelUsed":[255],"ChannelUsed":0,"bAssayFirstStep":true,"kst":0,"Status":0}"
            Step9="{"StepType":2,"AssayNumber":1,"PlateIndex":0,"PlateColumnType":12,"RowLocation":0,"Location":13,"Speed":400,"Time":44,"SumTime":60,"ProbeIndex":0,"PickerUsed":1,"AllChannelUsed":[255],"ChannelUsed":0,"bAssayFirstStep":false,"kst":1,"Status":1}"

        """

        with open(file, 'r') as f:
            lines = f.read().splitlines()

            # Find the line [ExperimentLog]
            for i, line in enumerate(lines):
                if line.startswith('[ExperimentLog]'):
                    start_index = i + 1
                    break

            # Count the number of items of the next line, when stripping by commas
            nr_items = len(lines[start_index].split(','))

            step_strings = []
            # Find all lines with the same amount of items and store them
            for line in lines[start_index:]:
                if len(line.split(',')) == nr_items:
                    step_strings.append(line)

        # List to hold dictionaries
        step_dicts = []

        # Iterate over the list of strings
        for step_string in step_strings:
            # Extract the JSON part of the string
            json_part = step_string.split('=', 1)[1].strip('"')
            # Convert the JSON string into a dictionary
            step_dict = json.loads(json_part)
            # Append the dictionary to the list
            step_dicts.append(step_dict)

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(step_dicts)

        # Remove extra columns
        df = df[['kst', 'Location', 'Time', 'SumTime', 'AssayNumber', 'Speed']]
        # Add the column with the stepnumber to the beginning
        df.insert(0, 'StepNumber', np.arange(len(df)))

        # Change column name 'Location' to 'Column'
        df.rename(columns={'Location': 'Column'}, inplace=True)

        # Subtract 11 from it - beacuse the plate with the sample usually starts at index 12 to 24
        df['Column'] = df['Column'] - nr_max_columns + 1

        # If kst equals 2, then the step is a association
        # If kst equals 3, then the step is a dissociation

        # Guess if the step was 'ASSOC' or 'DISASSOC',
        # depending on the value of kst
        df['kst'] = df['kst'].replace({2: 'ASSOC', 3: 'DISASSOC', 1: 'LOADING', 0: 'OTHER'})

        # Change column name 'kst' to 'Type'
        df.rename(columns={'kst': 'Type'}, inplace=True)

        # Update the 'Type' column to 'BASELINE' for rows preceding 'ASSOC'
        df.loc[df['Type'].shift(-1) == 'ASSOC', 'Type'] = 'BASELINE'

        self.df_steps = df

        return None

    def read_settings_ini(self, file, nr_max_sensors=8, nr_max_columns=12):
        """
        Read the settings initialization file.
        
        Parses the Setting.ini file that contains information about the experiment settings,
        including sensor configurations, sample concentrations, and plate layout.
        
        Parameters
        ----------
        file : str
            Path to the Setting.ini file.
        nr_max_sensors : int, optional
            Maximum number of sensors per column, default is 8.
        nr_max_columns : int, optional
            Maximum number of columns in the plate, default is 12.
            
        Returns
        -------
        None
            Updates several instance attributes:
            - sensor_names: List of sensor names
            - ligand_conc_df: DataFrame with ligand concentration information
            
        Notes
        -----
        Example format for Setting.ini:
        
            [BasicInformation]
            PreExperiment="{"AssayDescription":"","AssayUser":"Marianna","CreationTime":"03-31-2025 14:01:50",...}"
            ExperimentType=Kinetics
            SaveTime=03/31/2025 15:55:27
            Frequency=10Hz
            DataVersion=2.15.5.1221
            [Regeneration]
            RegenerationEnable=1
            Repeat=3
            RTime=5
            RSpeed=1000
            NTime=5
            NSpeed=1000
            [SampleKinetics]
            Well1=4,-1,-1,-1,,
            Well2=4,-1,-1,-1,,
            ...
        """

        with open(file, 'r') as f:
            lines = f.read().splitlines()

            # Find the molar concentration units
            # below the line MolarConcentrationUnit=NM
            for line in lines:
                if line.startswith('MolarConcentrationUnit='):
                    units = line.split('=')[1].strip().lower()
                    factor = {'nm': 1e-3, 'mm': 1e3, 'pm': 1e-6, 'm': 1e6}.get(units, 1)
                    break

            # Find the line [SampleKinetics]
            for i, line in enumerate(lines):
                if line.startswith('[SampleKinetics]'):
                    start_index = i + 1
                    break

            # Count the number of items of the next line, when stripping by commas
            nr_items = len(lines[start_index].split(','))

            sample_strings = []
            # Find all lines with the same amount of items and store them
            for line in lines[start_index:]:
                if len(line.split(',')) == nr_items:
                    sample_strings.append(line)

            # Leave lines starting with 'Well'
            sample_strings = [line for line in sample_strings if line.startswith('Well')]

            # Now we need to convert wells that go from 1 to 96 into a combination of letters and numbers
            # They are ordered by rows, each column has nr_max_sensors rows
            concs = []

            locations = np.concatenate([np.repeat(x, nr_max_sensors)     for x in range(1, nr_max_columns+1)], axis=0)
            sensors   = np.concatenate([(65 + np.arange(nr_max_sensors)) for _ in range(1, nr_max_columns+1)], axis=0)
            # Apply chr function to sensors
            sensors = [chr(x) for x in sensors]

            labels = []

            for string in sample_strings:
                # We need to extract the concentration : Well49=1,17,170,100,,EGFR
                # We need to split by commas and take the fourth element
                conc = string.split('=')[1].split(',')[3]
                conc = float(conc) if conc != '' else 0

                concs.append(conc)

                # Extract the label, last column
                label = string.split(',')[-1]
                labels.append(label)

            # Create the ligand concentration dataframe
            df_sensor_prev = pd.DataFrame({'Sensor': sensors,
                                           'Location': locations,
                                           'Concentration_micromolar': concs,
                                           'SampleID': labels})

            loading_labels = []
            loading_columns = []
            sample_labels = []
            location = []
            sensor = []
            concs = []

            loading_columns_extended = []

            # Iterate over the associations and find loading column

            for row_index in range(len(self.df_steps)):

                step_type = self.df_steps.iat[row_index, self.df_steps.columns.get_loc('Type')]
                if step_type == 'ASSOC':

                    loading_column = find_loading_column(self.df_steps, row_index)

                    df_temp = df_sensor_prev[df_sensor_prev['Location'].isin([loading_column])]
                    loading_labels += df_temp['SampleID'].to_list()

                    loading_columns.append(loading_column)

                    association_column = self.df_steps.iat[row_index, self.df_steps.columns.get_loc('Column')]

                    location += [association_column for _ in range(nr_max_sensors)]
                    # The expression chr(65 + i) returns the uppercase ASCII letter corresponding to the index i,
                    # where i = 0 gives 'A', i = 1 gives 'B', and so on. For example, chr(65 + 2) returns 'C'.
                    sensor   += [chr(65 + i) for i in range(nr_max_sensors)]

                    loading_columns_extended += [loading_column for _ in range(nr_max_sensors)]

                    df_temp = df_sensor_prev[df_sensor_prev['Location'].isin([association_column])]
                    sample_labels += df_temp['SampleID'].to_list()

                    concs += df_temp['Concentration_micromolar'].to_list()
                else:
                    loading_columns.append('NA')

            # Include the loading column in df_steps
            self.df_steps['Loading_location'] = loading_columns

            sample_id = [x + '-' + y for x, y in zip(sample_labels, loading_labels)]

            replicates = [1 for _ in range(len(sample_id))]

            unq_sample_id = list(set(sample_id))

            for unq in unq_sample_id:

                # Find the index with that sample id
                idx = [i for i, x in enumerate(sample_id) if x == unq]

                replicates[idx[0]] = 1

                idx_cnt = 1
                for i in idx[1:]:
                    if sample_id[i] == sample_id[i - 1]:
                        idx_cnt += 1
                    replicates[i] = (idx_cnt - 1) // nr_max_sensors + 1

            # Create the ligand concentration dataframe
            df_sensor = pd.DataFrame({'Sensor': sensor,
                                      'Analyte_location': location,
                                      'Concentration_micromolar': concs,
                                      'Loading_location': loading_columns_extended,
                                      'SampleID': sample_id,
                                      'Replicate': replicates,
                                      'Experiment': self.name})

            # Multiply concentrations by the factor
            df_sensor['Concentration_micromolar'] = df_sensor['Concentration_micromolar'] * factor

            sensor_names = df_sensor['Sensor'].unique().tolist()
            self.sensor_names = sensor_names

            self.create_unique_sensor_names()

            self.ligand_conc_df = df_sensor

        return None

    def read_sensor_data(self, files, names=None):
        """
        Read the sensor data from CSV files.
        
        Parses the Gator sensor data CSV files containing time series data for each sensor channel.
        The method organizes the data according to experimental steps and sensors.
        
        Parameters
        ----------
        files : list
            List of file paths that may include sensor data CSV files.
        names : list, optional
            List of file names corresponding to the file paths in `files`.
            If None, the file names will be extracted from the file paths.
            
        Returns
        -------
        None
            Updates several instance attributes:
            - xs: List of time arrays for each sensor
            - ys: List of signal arrays for each sensor
            - df_steps: DataFrame with filtered step information containing only steps with data
            - no_steps: Number of steps with data
            - traces_loaded: Boolean indicating if traces were successfully loaded
            
        Notes
        -----
        The sensor data files are expected to be in CSV format with the following characteristics:
        - File names contain 'Channel' and have a '.csv' extension
        - The first column contains signal values
        - The second column contains time values
        """

        fns   = [fn for fn, name in zip(files, names) if '.csv' in name and 'Channel' in name]
        names = [name for name in names if '.csv' in name and 'Channel' in name]

        self.fns = fns

        # Initialize dictionaries with data
        xs = [[] for _ in range(len(self.sensor_names))]
        ys = [[] for _ in range(len(self.sensor_names))]

        # Find steps with data, where the SumTime is zero,
        # or if not zero, it is the same as the previous step

        steps_with_data = []

        for i in range(1, len(self.df_steps)):

            sum_time_i = self.df_steps.iat[i, self.df_steps.columns.get_loc('SumTime')]
            sum_time_prev = self.df_steps.iat[i - 1, self.df_steps.columns.get_loc('SumTime')]

            if sum_time_i == 0 or sum_time_i == sum_time_prev:
                continue
            else:
                steps_with_data.append(i - 1)

        # Create a copy of self.df_steps that will contain all the data and remove steps with no data
        df_steps_all = self.df_steps.copy()

        # Select the steps with data only
        self.df_steps = self.df_steps.iloc[steps_with_data]

        # Remove the stepnumber column
        self.df_steps = self.df_steps.drop(columns=['StepNumber'])

        # Reset the stepnumber column
        self.df_steps['StepNumber'] = np.arange(len(self.df_steps))

        # Reset the index of step_info_with_data
        self.df_steps = self.df_steps.reset_index(drop=True)

        # Rename the column 'Column' to 'Column_location' in df_steps
        self.df_steps.rename(columns={'Column': 'Column_location'}, inplace=True)

        # Move the column StepNumber to the first position
        column_to_move = self.df_steps.pop('StepNumber')
        self.df_steps.insert(0, 'StepNumber', column_to_move)

        # Move the column Loading_location to the fifth position
        column_to_move = self.df_steps.pop('Loading_location')
        self.df_steps.insert(4, 'Loading_location', column_to_move)

        # Convert the columns Loading_location and Column_location to string, for compatibility with get_step_xy
        self.df_steps['Loading_location'] = self.df_steps['Loading_location'].astype(str)
        self.df_steps['Column_location']  = self.df_steps['Column_location'].astype(str)

        self.no_steps = len(self.df_steps)

        # xs will have one element per sensor
        # each element will have as many subelements as steps

        # Extract the assay number and channel number from the file names
        # and sort the files by assay number and channel number
        assay_number = [int(name.split('_')[1].split('Channel')[0]) for name in names]
        channel_number = [int(name.split('Channel')[1].split('.')[0]) for name in names]

        # Find the indices that would sort the arrays by assay number and channel number
        sorted_indices = np.lexsort((channel_number, assay_number))
        # Sort the file names and assay numbers
        fns   = [fns[i] for i in sorted_indices]
        names = [names[i] for i in sorted_indices]

        for fn, name in zip(fns, names):

            try:

                assay_number_i   = int(name.split('_')[1].split('Channel')[0])
                channel_number_i = int(name.split('Channel')[1].split('.')[0])

                df = pd.read_csv(fn, skiprows=1, header=None)

                x = df.iloc[:, 1].to_numpy(dtype=float)
                y = df.iloc[:, 0].to_numpy(dtype=float)

                # Subset df_steps_all according to the assay number
                temp_df = df_steps_all[df_steps_all['AssayNumber'] == assay_number_i]

                # Filter which step numbers are in step_with_data
                temp_df = temp_df[temp_df['StepNumber'].isin(steps_with_data)]

                # Cumulative time for this assay
                temp_df['SumTime2'] = temp_df['Time'].cumsum()

                # Find starting sumtime for the assay
                start_sumtime = temp_df.iloc[0, temp_df.columns.get_loc('SumTime')]

                # For each step, include the data into xs and ys
                for i in range(len(temp_df)):

                    if i == 0:
                        start_time = 0
                    else:
                        start_time = temp_df.iloc[i - 1, temp_df.columns.get_loc('SumTime2')]

                    step_time = temp_df.iloc[i, temp_df.columns.get_loc('Time')]

                    end_time = start_time + step_time

                    sel_idx = np.logical_and(x >= start_time, x < end_time)

                    y_sel = y[sel_idx]
                    x_sel = x[sel_idx]

                    # Add the overall sumtime to x
                    x_sel += start_sumtime

                    # Append the second column to xs
                    xs[channel_number_i - 1].append(x_sel)

                    # Append the first column to ys
                    ys[channel_number_i - 1].append(y_sel)

                self.traces_loaded = True
                self.xs = xs
                self.ys = ys

            except:

                pass

        return None
