import re
import base64
import xml.etree.ElementTree as elementTree

from .utils.processing import etree_to_dict, combine_dicts
from .surface_exp      import *

factor_conc_to_micro = {'nM':1e-3, 'µM':1, 'mM':1e3, 'M':1e6, 'mg/ml':1e3, 'µg/ml':1}

class OctetExperiment(SurfaceBasedExperiment):

    """
    A class used to represent a BLI experiment

    Attributes
    ----------

        name (str):                 name of the experiment
        fns (list):                 list of file names (length n, one per sensor)
        xs (list):                  list of x values (time, length n, one per sensor)
        ys (list):                  list of y values (length n, one per sensor)
        exp_info (list):            list of dictionaries with experimental information
        step_info (list):           list of dictionaries with step information
        no_steps (int):             number of steps
        no_sensors (int):           number of sensors
        sample_column (np.ndarray): array of sample column information (96 elements, one per well)
        sample_row (np.ndarray):    array of sample row information    (96 elements, one per well)
        sample_type (list):         list of sample types   (96 elements, one per well)
        sample_id (list):           list of sample ids     (96 elements, one per well)
        sensor_names_unique (list): list of unique sensor names (length n, one per sensor)
        sensor_names (list):        list of sensor names (length n, one per sensor)
        df_steps    (pd.DataFrame): dataframe with the steps information
        ligand_conc_df (pd.DataFrame): dataframe with the ligand concentration information

        ligand_conc_df.head(2):
        Sensor Analyte_location  Concentration_micromolar SampleID  Replicate Experiment
        A1        5                    0.1300       wt          1          t
        B1        5                    0.0692       wt          1          t

        traces_loaded (bool):       True if traces are loaded
        sample_plate_loaded (bool): True if sample plate information is loaded
        sample_conc (np.array):     array with the sample concentrations (96 elements, one per well)
        sample_conc_labeled (list): list with the sample concentrations labeled (96 elements, one per well)
        steps_performed (pd.DataFrame): dataframe with the steps performed

        steps_performed.head(2):
        #Step         Type              Column   Time
        1            Regeneration      1        25
        2  BaselineNeutralization      2        90
        3         BaselineLoading      3        50
    """

    def __init__(self, name = 'BLI_experiment'):

        """

        Initialize the instance

        """

        super().__init__(name,'BLI_experiment')

    def read_sensor_data(self, files,names=None):

        """

        Read the sensor data from the .frd files

        Results:

            It creates the attributes

                self.traces_loaded
                self.xs
                self.ys
                self.exp_info
                self.step_info
                self.no_steps
                self.no_sensors
                self.sensor_names
                self.df_steps
                self.ligand_conc_df

        """

        if names is None:
            names = files

        if not isinstance(files, list):
            files = [files]
            names = [names]

        fns = [fn for fn,name in zip(files,names) if '.frd' in name]

        if len(fns) < 1:
            self.traces_loaded = False
            return None
        else:
            self.fns = fns

        # Initialize dictionaries with data
        xs, ys, all_expinfo, all_stepinfo, more_info = [], [], [], [], []
        for fn in fns:
            # Load file
            tree = elementTree.parse(fn)
            root = tree.getroot()

            # Extract experimental info
            all_expinfo.append(etree_to_dict(root.find('ExperimentInfo')))

            # Initialize lists for each file
            x_values, y_values, step_info = [], [], []
            more_dict = {'FlowRate': [], 'StepType': [], 'StepName':[], 'StepStatus':[], 'ActualTime':[], 'CycleTime':[]}
            for step in root.find('KineticsData'):
                for step_x in step.findall('AssayXData'):
                    # Convert string to binary
                    data_text = bytes(step_x.text, 'utf-8')
                    # Convert to base64
                    decoded = base64.decodebytes(data_text)
                    # And now convert to float32 array
                    data_values = np.array(np.frombuffer(decoded, dtype=np.float32))
                    x_values.append(data_values)
                for step_y in step.findall('AssayYData'):
                    # Convert string to binary
                    data_text = bytes(step_y.text, 'utf-8')
                    # Convert to base64
                    decoded = base64.decodebytes(data_text)
                    # And now convert to float32 array
                    data_values = np.array(np.frombuffer(decoded, dtype=np.float32))
                    y_values.append(data_values)
                for step_data in step.findall('CommonData'):
                    step_info.append(etree_to_dict(step_data))
                for tag in ['FlowRate', 'StepType', 'StepName', 'StepStatus', 'ActualTime', 'CycleTime']:
                    for step_data in step.findall(tag):
                        more_dict[tag].append(step_data.text)

            xs.append(x_values)
            ys.append(y_values)
            all_stepinfo.append(combine_dicts(step_info))
            more_info.append(more_dict)

        # Merge all_stepinfo and more_info
        for i in range(len(all_stepinfo)):
            all_stepinfo[i] = {**all_stepinfo[i], **more_info[i]}

        # Fill instance
        self.xs   = xs
        self.ys   = ys
        self.exp_info  = all_expinfo
        self.step_info = all_stepinfo
        # Convert text to floats
        self.convert_to_numbers()

        self.no_steps = len(self.step_info[0]['ActualTime'])

        self.no_sensors = len(self.fns)

        self.sensor_names = [self.exp_info[i]['SensorName'] for i in range(self.no_sensors)]

        steps_names = self.step_info[0]['StepName']
        steps_types = self.step_info[0]['StepType']
        steps_start = self.step_info[0]['StartTime'] / 1000 # To seconds
        steps_loc   = self.step_info[0]['SampleLocation']

        self.df_steps = pd.DataFrame({'#Step':np.arange(len(steps_names))+1,
                                      'Name':steps_names,
                                      'Type':steps_types,
                                      'Start':steps_start,
                                      'Column_location':steps_loc})

        # We need to include the loading location in self.df_steps
        loading_location = []
        for row in self.df_steps.iterrows():
            step_type = row[1]['Type']
            if step_type == 'ASSOC':
                # Find the previous loading step
                for i in range(row[0],0,-1):
                    if self.df_steps.iloc[i]['Type'] == 'LOADING':
                        loading_location.append(self.df_steps.iloc[i]['Column_location'])
                        break
            else:
                loading_location.append(np.nan)

        self.df_steps['Loading_location'] = loading_location

        sensor_locs_all = np.concatenate([self.step_info[i]['SampleLocation']       for i in range(self.no_sensors)])
        sensor_type_all = np.concatenate([self.step_info[i]['StepType']             for i in range(self.no_sensors)])

        sensor_molar_conc_all = np.concatenate([self.step_info[i]['MolarConcentration']  for i in range(self.no_sensors)])
        sensor_mass_conc_all  = np.concatenate([self.step_info[i]['Concentration']       for i in range(self.no_sensors)])

        sensor_conc_all = []

        for i in range(len(sensor_molar_conc_all)):

            if sensor_molar_conc_all[i] < 0:

                sensor_conc_all.append(sensor_mass_conc_all[i])

            else:

                sensor_conc_all.append(sensor_molar_conc_all[i])

        sensor_conc_all = np.array(sensor_conc_all)

        sample_id_all   = np.concatenate([self.step_info[i]['SampleID']   for i in range(self.no_sensors)])

        sensor_name_rep = np.concatenate([np.repeat(self.exp_info[i]['SensorName'],len(self.step_info[i]['Concentration'])) for i in range(self.no_sensors)])

        conc_units = np.concatenate([self.step_info[i]['MolarConcUnits'] for i in range(self.no_sensors)])

        df_all = pd.DataFrame({'Sensor':sensor_name_rep,
                               'Analyte_location':sensor_locs_all,
                               'Type':sensor_type_all,
                               'Concentration_micromolar':sensor_conc_all,
                               'ConcUnits':conc_units,
                               'SampleID':sample_id_all})

        # For each association step, find the corresponding loading step
        # and add the loading step to the dataframe as a column next to the association

        # Add empty column to the data frame
        loading_location  = []
        loading_sample_id = []

        for i in range(len(df_all)):

            row = df_all.iloc[i]
            # Find if row is association step
            if row['Type'] == 'ASSOC':

                # Find the previous loading step
                for j in range(i,0,-1):

                    if df_all.iloc[j]['Type'] == 'LOADING':

                        loading_location.append(df_all.iloc[j]['Analyte_location'])
                        loading_sample_id.append(df_all.iloc[j]['SampleID'])
                        break


        # Keep only association or dissociation steps
        df = df_all[df_all['Type'] == 'ASSOC'].copy()

        # ADD column loading_location
        df['Loading_location']  = loading_location

        # Replace None with empty string in loading_sample_id
        loading_sample_id = [x if x is not None else '' for x in loading_sample_id]

        # Include the loading_sample_id, if we have more than one unique value
        unq_loading_ids = np.unique(loading_sample_id)

        if len(unq_loading_ids) > 1:

            # Combine the sample id with the loading id
            df['SampleID'] = df['SampleID'] + ' - ' + loading_sample_id

        # Remove the Type column
        df = df.drop(columns=['Type'])

        # Sort by location and sensor name
        df = df.sort_values(by=['Loading_location','Analyte_location','Sensor'])

        # Add rep column
        sizes = df.groupby(['Loading_location','Analyte_location','Sensor']).size().reset_index(name="Repetitions")

        rep_number = []

        for i in range(len(sizes)):
            rep_number.extend(np.arange(sizes['Repetitions'].iloc[i])+1)

        # Group by sensor and location
        df['Replicate'] = rep_number

        # Sort the dataframe first by sensor, second by Loading location,
        # Third by replicate and finally by location

        df = df.sort_values(by=['Analyte_location','Replicate','Loading_location','Sensor'])

        df['Factor'] = df.apply(lambda x: factor_conc_to_micro[x['ConcUnits']], axis=1)

        df['Concentration_micromolar'] = df['Concentration_micromolar'] * df['Factor']

        # Remove the factor column and conc units
        df = df.drop(columns=['Factor','ConcUnits'])

        # Add the experiment name
        df['Experiment'] = self.name

        self.ligand_conc_df = df

        self.create_unique_sensor_names()

        self.traces_loaded = True

        return None

    def read_sample_plate_info(self,files,names=None):

        """
        Read the sample plate information from the .fmf file

        Results:

            It creates the attributes

                self.sample_column
                self.sample_row
                self.sample_type
                self.sample_id
                self.sample_conc
                self.sample_conc_labeled
                self.sample_plate_loaded
                self.steps_performed

        """

        if names is None:
            names = files

        if not isinstance(files, list):
            files = [files]
            names = [names]

        index = next((i for i, s in enumerate(names) if 'ExpMethod.fmf' in s), None)

        if index is None:

            self.sample_plate_loaded = False
            return None

        file = files[index]

        tree = elementTree.parse(file)
        root = tree.getroot()

        sample_types     = [x.text for x in root.findall(".//SampleType")]
        sample_locations = [x.text for x in root.findall(".//SampleLoc")]
        sample_ids       = [x.text for x in root.findall(".//SampleID")]

        sample_conc_molar    = np.array([float(x.text) for x in root.findall(".//SampleMolarConc")])
        sample_conc_mass     = np.array([float(x.text) for x in root.findall(".//SampleConc")])

        sample_conc          = sample_conc_mass

        sel_ids = ['SAMPLE' in s for s in sample_types]

        counter = 0
        for i in range(len(sample_conc)):

            if sel_ids[i]:

                if sample_conc_molar[counter] > 0:

                    sample_conc[i] = sample_conc_molar[counter]

                counter += 1

        conc_units       = [x.text for x in root.findall(".//ConcUnits")][0]
        molar_conc_units = [x.text for x in root.findall(".//MolarConcUnits")][0]

        factors = [factor_conc_to_micro[molar_conc_units] if 'SAMPLE' in st else factor_conc_to_micro[conc_units] for st in sample_types]

        sample_conc = sample_conc * np.array(factors)
        sample_conc = np.round(sample_conc, 5)

        sample_column = np.array([int(re.sub(r'\D', '', text)) for text in sample_locations])
        sample_row    = np.array([re.sub(r'\d+', '', text)     for text in sample_locations])

        self.sample_column = sample_column
        self.sample_row    = sample_row
        self.sample_type   = sample_types
        self.sample_id     = sample_ids

        self.sample_conc    = sample_conc

        sample_conc_labeled = [f"{x} µM" if t == 'KSAMPLE' and x >= 0 else f"{x} µg/ml" if t != 'KSAMPLE' and x >= 0 else '' for x, t in zip(sample_conc, sample_types)]

        self.sample_conc_labeled = sample_conc_labeled

        data_name     = [x.text for x in root.findall(".//DataName")]
        assay_time    = [x.text for x in root.findall(".//AssayTime")]

        steps_info_df      = pd.DataFrame({'Type':data_name,'Time':assay_time})

        data_name     = [x.text for x in root.findall(".//StepDataName")]
        data_col      = [x.text for x in root.findall(".//SampleCol")]

        steps_performed = pd.DataFrame({'#Step':np.arange(len(data_name))+1,'Type':data_name,'Column':data_col})

        steps_performed = pd.merge(steps_performed, steps_info_df, on='Type', how='left')

        self.steps_performed = steps_performed

        self.sample_plate_loaded = True

        return None

    def convert_to_numbers(self):

        """

        Convert the strings in the step info to numbers

        Results:

            It modifies the attribute self.step_info

        """

        # List of entries in step info
        entries = ['Concentration', 'MolarConcentration', 'MolecularWeight', 'Temperature', 'StartTime',
                   'AssayTime', 'FlowRate', 'ActualTime', 'CycleTime']

        for entry in entries:
            for sensor in range(len(self.fns)):
                # Do sanity check
                try:
                    self.step_info[sensor][entry] = np.array(self.step_info[sensor][entry], dtype=float)
                except:

                    print("Erroneous entry found for %s and sensor %i: %s" % (entry, sensor, self.step_info[sensor][entry]))
                    print("Will set it to -1. Needs to be corrected")
                    # Correct erroneous value
                    for i in range(len(self.step_info[sensor][entry])):
                        try:
                            float(self.step_info[sensor][entry][i])
                        except:
                            self.step_info[sensor][entry][i] = -1
                    self.step_info[sensor][entry] = np.array(self.step_info[sensor][entry], dtype=float)
        return None
