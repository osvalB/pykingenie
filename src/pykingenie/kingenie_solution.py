import pandas as pd
import numpy  as np

from .solution_exp import SolutionBasedExp

class KinGenieCsvSolution(SolutionBasedExp):
    """
    A class used to represent a KinGenie simulation (solution-based).
    
    This class handles KinGenie solution simulation data that can be exported from the Simulation panel.

    Parameters
    ----------
    name : str
        Name of the experiment.

    Attributes
    ----------
    name : str
        Name of the experiment.
    xs : list of numpy.ndarray
        List of x values (time), one array per trace.
    ys : list of numpy.ndarray
        List of y values (signal), one array per trace.
    no_traces : int
        Number of traces.
    traces_names : list of str
        List of trace names, one per trace.
    traces_names_unique : list of str
        List of unique trace names.
    conc_df : pandas.DataFrame
        DataFrame with the concentration information for each trace.
    traces_loaded : bool
        Whether traces have been loaded.
    """

    def __init__(self, name):

        super().__init__(name, 'kingenie_csv_solution')
        
    def read_csv(self, file):
        """
        Read the KinGenie CSV file containing solution-based simulation data.
        
        Parameters
        ----------
        file : str
            Path to the CSV file.
            
        Returns
        -------
        None
            The method populates class attributes with data from the CSV file.
            
        Notes
        -----
        The CSV file should have the following columns:
        - Time: time points of the simulation
        - Signal: signal values at each time point
        - Protein_concentration_micromolar: protein concentration
        - Ligand_concentration_micromolar: ligand concentration
        
        Example CSV format:
        
            Time    Signal  Protein_concentration_micromolar Ligand_concentration_micromolar
            0       0       5                                0.1
            0.5     1       5                                0.1
        
        After reading, this method creates the following attributes:
        - self.xs: list of time values for each trace
        - self.ys: list of signal values for each trace
        - self.no_traces: number of unique traces
        - self.traces_names: list of trace names
        - self.conc_df: DataFrame with concentration information
        """

        df = pd.read_csv(file)

        # Validate the input file
        # Check we have at least 2 columns, time versus signal

        if len(df.columns) < 2:
            raise ValueError("The CSV file must have at least 2 columns.")

        # If we only have two columns, assume that the third is the protein concentration
        if len(df.columns) == 2:
            df['Protein_concentration_micromolar'] = 0

        # If we have only 3 columns, assume that the third is the protein concentration, and the ligand concentration is 0
        if len(df.columns) == 3:
            df['Ligand_concentration_micromolar'] = 0

        # Assume the first 4 columns are Time, Signal, Protein_concentration_micromolar, Ligand_concentration_micromolar
        df = df.iloc[:, 0:4]

        df.columns = ['Time', 'Signal', 'Protein_concentration_micromolar', 'Ligand_concentration_micromolar']

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
        self.traces_names = ['trace ' + str(i + 1) for i in range(self.no_traces)]

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