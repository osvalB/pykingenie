import pandas as pd
import numpy  as np

from .surface_exp import SurfaceBasedExperiment

class KinGenieCsv(SurfaceBasedExperiment):

    """
    A class used to represent a KinGenie csv file, which can be exported from the Simulation panel

    Example:

        Time	Signal	Smax	Analyte_concentration_micromolar_constant	Cycle
        0	0	5	0.1	1
        0.5	0.00498512709952641	5	0.1	1
        1	0.00994051854979453	5	0.1	1
        1.5	0.0148661741864963	5	0.1	1
        2	0.0197623830271752	5	0.1	1
        2.5	0.0246293723316269	5	0.1	1
        3	0.0294671469344429	5	0.1	1
        3.5	0.0342760214300913	5	0.1	1
        4	0.0390561847030298	5	0.1	1
        4.5	0.043807638803537	5	0.1	1
        5	0.0485304884515415	5	0.1	1

    Attributes

        name (str):                 name of the experiment
        fn (str):                   file name
        xs (np.array):              list of x values (time, length n, one per sensor)
        ys (np.array):              list of y values (length n, one per sensor)
        no_sensors (int):           number of sensors
        sensor_names (list):        list of sensor names (length n, one per sensor)
        sensor_names_unique (list): list of unique sensor names (length n, one per sensor)
        ligand_conc_df (pd.DataFrame): dataframe with the ligand concentration information

    """

    def __init__(self, name='kingenie_csv'):

        super().__init__(name, 'kingenie_csv')

    def read_csv(self,file):

        """
        Read the KinGenie csv file

        Results:

            It creates the attributes

                self.xs
                self.ys
                self.no_sensors
                self.sensor_names
                self.sensor_names_unique

        """

        df = pd.read_csv(file)

        # Read only the first Smax value
        smax_unq = df['Smax'].unique()
        df       = df[df['Smax'] == smax_unq[0]]

        # Detect if we have the column cycle
        is_single_cycle = 'Cycle' in df.columns and len(df['Cycle'].unique()) > 1

        # Generate one 'fake' sensor per ligand concentration, if it's not a single cycle
        ligand_conc = df['Analyte_concentration_micromolar_constant'].unique()

        # Remove zero concentration
        ligand_conc = ligand_conc[ligand_conc != 0]

        if is_single_cycle:
            self.no_sensors = 1
        else:
            self.no_sensors = len(ligand_conc)

        # Use fake names
        self.sensor_names = ['sim. sensor ' + str(i+1) for i in range(self.no_sensors)]

        # Initiate self.xs and self.ys, one empty list per sensor
        self.xs = [[] for _ in range(self.no_sensors)]
        self.ys = [[] for _ in range(self.no_sensors)]

        steps_start = []

        if not is_single_cycle:

            # Now populate, for each sensor self.xs and self.ys
            for i in range(self.no_sensors):

                # Find the start index of the association phase
                start_idx = df[df['Analyte_concentration_micromolar_constant'] == ligand_conc[i]].index[0]

                if i == self.no_sensors - 1:
                    end_idx = df.shape[0]
                else:
                    end_idx   = df[df['Analyte_concentration_micromolar_constant'] == ligand_conc[i+1]].index[0]

                df_temp = df.iloc[start_idx:end_idx]

                # Extract the association phase and dissociation phase
                df_temp_asso  = df_temp[df_temp['Analyte_concentration_micromolar_constant'] > 0]
                df_temp_disso = df_temp[df_temp['Analyte_concentration_micromolar_constant'] == 0]

                # Extract the time values of the association/dissociation phase
                time_asso = df_temp_asso['Time'].to_numpy()
                time_diss = df_temp_disso['Time'].to_numpy()

                if i == 1:

                    steps_start.append(0)
                    steps_start.append(time_asso[-1])

                # Populate self.xs
                self.xs[i].append(time_asso)
                self.xs[i].append(time_diss)

                # Extract the signal values
                signal_asso = df_temp_asso['Signal'].to_numpy()
                signal_diss = df_temp_disso['Signal'].to_numpy()

                # Populate self.ys
                self.ys[i].append(signal_asso)
                self.ys[i].append(signal_diss)

            # Now generate a fake ligand concentration df
            df_sensor = pd.DataFrame({'Sensor':self.sensor_names,
                                      'Analyte_location':'1',
                                      'Concentration_micromolar':ligand_conc,
                                      'SampleID':'simulation',
                                      'Replicate':1,
                                      'Loading_location': '2',
                                      'Experiment':self.name})

            self.ligand_conc_df = df_sensor

            # Generate a fake steps df
            steps_names = ['Association','Dissociation']
            steps_types = ['ASSOC','DISASSOC']
            steps_loc   = ['1','3']
            steps_load  = ['2', 'NA']

            self.df_steps = pd.DataFrame({'#Step':np.arange(len(steps_names))+1,
                                          'Name':steps_names,
                                          'Type':steps_types,
                                          'Start':steps_start,
                                          'Column_location':steps_loc,
                                          'Loading_location':steps_load})

        else:

            cycles = df['Cycle'].unique()
            cycles.sort()

            n_cycles = len(cycles)

            analyte_location = []
            steps_loc        = []

            for cycle in cycles:

                analyte_location.append(str(cycle+1))
                steps_loc.append(str(cycle+1))
                steps_loc.append(str(999))

                df_temp = df[df['Cycle'] == cycle]

                # Extract the association phase and dissociation phase
                df_temp_asso = df_temp[df_temp['Analyte_concentration_micromolar_constant'] > 0]
                df_temp_disso = df_temp[df_temp['Analyte_concentration_micromolar_constant'] == 0]

                # Extract the time values of the association/dissociation phase
                time_asso = df_temp_asso['Time'].to_numpy()
                time_diss = df_temp_disso['Time'].to_numpy()

                steps_start.append(time_asso[0])
                steps_start.append(time_asso[-1])

                # Populate self.xs
                self.xs[0].append(time_asso)
                self.xs[0].append(time_diss)

                # Extract the signal values
                signal_asso = df_temp_asso['Signal'].to_numpy()
                signal_diss = df_temp_disso['Signal'].to_numpy()

                # Populate self.ys
                self.ys[0].append(signal_asso)
                self.ys[0].append(signal_diss)

            # Now generate a fake ligand concentration df
            df_sensor = pd.DataFrame({'Sensor': self.sensor_names*n_cycles,
                                      'Analyte_location': analyte_location,
                                      'Concentration_micromolar': ligand_conc,
                                      'SampleID': 'simulation',
                                      'Replicate': 1,
                                      'Loading_location': '1',
                                      'Experiment': self.name})

            self.ligand_conc_df = df_sensor

            # Generate a fake steps df
            steps_names = ['Association', 'Dissociation'] * n_cycles
            steps_types = ['ASSOC', 'DISASSOC']           * n_cycles
            steps_load  = ['1', 'NA']                     * n_cycles

            self.df_steps = pd.DataFrame({'#Step': np.arange(len(steps_names)) + 1,
                                          'Name': steps_names,
                                          'Type': steps_types,
                                          'Start': steps_start,
                                          'Column_location': steps_loc,
                                          'Loading_location': steps_load})

        self.create_unique_sensor_names()

        self.traces_loaded = True

        return None
