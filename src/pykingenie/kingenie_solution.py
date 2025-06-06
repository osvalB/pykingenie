import pandas as pd
import numpy  as np

from .solution_exp import SolutionBasedExp

class KinGenieCsvSolution(SolutionBasedExp):

    """
    A class used to represent a KinGenie simulation (solution-based), which can be exported from the Simulation panel

    Attributes
    ----------
        name (str):                 name of the experiment
        xs (np.array):              list of x values (time, length n, one per sensor)
        ys (np.array):              list of y values (length n, one per sensor)
        no_sensors (int):           number of sensors
        sensor_names (list):        list of sensor names (length n, one per sensor)
        sensor_names_unique (list): list of unique sensor names (length n, one per sensor)
        ligand_conc_df (pd.DataFrame): dataframe with the ligand concentration information

    """

    def __init__(self, name):

        super().__init__(name, 'kingenie_csv_solution')

    def read_csv(self, file):
        """
        Read the KinGenie csv file

        Example:

            Time	Signal	Protein_concentration_micromolar Ligand_concentration_micromolar
            0	    0	    5	                                0.1
            0.5	    1	    5	                                0.1

        Results:

            It creates the attributes

                self.xs
                self.ys
                self.no_sensors
                self.sensor_names
                self.sensor_names_unique

        """

        df = pd.read_csv(file)

        # Find the ligand concentrations
        ligand_conc = df['Ligand_concentration_micromolar']

        # Find the protein concentrations
        protein_conc = df['Protein_concentration_micromolar']

        # Combine protein and ligand concentration into one array
        combined_concs_array = [str(x) + ' and ' + str(y) for x, y in zip(protein_conc, ligand_conc)]

        # Add a new column to the dataframe
        df['Combined_concentration'] = combined_concs_array

        # Find the unique combined concentrations
        combined_concs_unq = np.unique(combined_concs_array)

        self.no_traces = len(combined_concs_unq)

        # Use fake names
        self.traces_names = ['sim. trace ' + str(i + 1) for i in range(self.no_traces)]

        # Initiate self.xs and self.ys
        self.xs = []
        self.ys = []

        protein_conc_unqs = []
        ligand_conc_unqs  = []

        # Now populate, for each sensor self.xs and self.ys
        for i, cc in enumerate(combined_concs_unq):
            # Extract the rows with the same combined concentrations
            df_temp = df[df['Combined_concentration'] == cc]

            protein_conc_unqs.append(df_temp['Protein_concentration_micromolar'].to_numpy()[0])
            ligand_conc_unqs.append(df_temp['Ligand_concentration_micromolar'].to_numpy()[0])

            # Extract the time values of the association/dissociation phase
            time_int = df_temp['Time'].to_numpy()

            # Populate self.xs
            self.xs.append(time_int)

            # Extract the signal values
            signal = df_temp['Signal'].to_numpy()

            # Populate self.ys
            self.ys.append(signal)

        # Now generate a fake ligand concentration df
        df_traces = pd.DataFrame({
            'Trace': self.traces_names,
            'Protein_concentration_micromolar': protein_conc_unqs,
            'Ligand_concentration_micromolar': ligand_conc_unqs,
            'SampleID': 'simulation',
            'Experiment': self.name})

        self.conc_df = df_traces

        self.create_unique_traces_names()
        self.traces_loaded = True

        return None