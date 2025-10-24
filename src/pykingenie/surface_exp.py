import pandas as pd
import numpy  as np

import copy

from pykingenie.utils.processing import if_string_to_list, find_index_of_previous_step_type

class SurfaceBasedExperiment:
    """A class representing surface-based kinetics experiments.
    
    This class handles operations related to surface-based experiments,
    such as biolayer interferometry (BLI) or surface plasmon resonance (SPR).
    It provides functionality for data processing, analysis, and visualization.
    
    Parameters
    ----------
    name : str
        The name of the experiment.
    type : str
        The type of the experiment (e.g., 'Octet', 'SPR').
        
    Attributes
    ----------
    name : str
        The name of the experiment.
    type : str
        The type of the experiment.
    xs : list of list
        Time values for each sensor and step.
    ys : list of list
        Signal values for each sensor and step.
    sensor_names : list
        Names of the sensors.
    ligand_conc_df : pandas.DataFrame
        DataFrame containing ligand concentration information.
    df_steps : pandas.DataFrame
        DataFrame containing step information.
    steps_performed : list
        List of steps performed in the experiment.
    traces_loaded : bool
        Whether traces have been loaded.
    sample_plate_loaded : bool
        Whether the sample plate information has been loaded.
    """

    def __init__(self, name, type):
        """Initialize the SurfaceBasedExperiment instance.
        
        Parameters
        ----------
        name : str
            The name of the experiment.
        type : str
            The type of the experiment (e.g., 'Octet', 'SPR').
        """

        self.name                = name
        self.type                = type

        self.xs                  = None
        self.ys                  = None
        self.sensor_names        = None
        self.ligand_conc_df      = None
        self.df_steps            = None
        self.steps_performed     = None
        self.traces_loaded       = False
        self.sample_plate_loaded = False

        self.fns                 = None
        self.exp_info            = None
        self.step_info           = None
        self.no_steps            = None
        self.no_sensors          = None
        self.sample_column       = None
        self.sample_row          = None
        self.sample_type         = None
        self.sample_id           = None
        self.sensor_names_unique = None
        self.sample_conc         = None
        self.sample_conc_labeled = None

    def create_unique_sensor_names(self):
        """Create unique sensor names by adding the experiment name to each sensor name.
        
        This method updates the `sensor_names_unique` attribute by prefixing
        each sensor name with the experiment name.
        
        Returns
        -------
        None
            The method modifies the `sensor_names_unique` attribute in-place.
        """

        self.sensor_names_unique = [self.name + ' ' + sensor_name for sensor_name in self.sensor_names]

        return None

    def check_sensor_name(self, new_sensor_name):
        """Check if a sensor name already exists and modify it if necessary.
        
        Parameters
        ----------
        new_sensor_name : str
            The proposed name for a new sensor.
            
        Returns
        -------
        str
            A unique sensor name, either the original if it doesn't exist 
            or a modified version with ' rep' appended.
        """
        if new_sensor_name in self.sensor_names:
            new_sensor_name += ' rep'

        return new_sensor_name

    def subtraction_one_to_one(self, sensor_name1, sensor_name2, inplace=True):
        """Subtract the signal of one sensor from another.
        
        Parameters
        ----------
        sensor_name1 : str
            Name of the sensor to subtract from.
        sensor_name2 : str
            Name of the sensor to subtract.
        inplace : bool, optional
            If True, the subtraction is done in place, otherwise a new sensor
            is created, by default True.
            
        Returns
        -------
        None
            The method modifies the instance attributes in-place.
            
        Notes
        -----
        This method modifies the following attributes:
        - self.xs
        - self.ys
        - self.sensor_names
        - self.ligand_conc_df
        
        Raises
        ------
        RuntimeError
            If no traces are loaded or if sensors are incompatible.
        """

        if not self.traces_loaded:
            raise RuntimeError("No traces loaded. Cannot perform subtraction.")

        new_sensor_name = sensor_name1 + ' - ' + sensor_name2

        new_sensor_name = self.check_sensor_name(new_sensor_name)

        sensor1 = self.sensor_names.index(sensor_name1)
        sensor2 = self.sensor_names.index(sensor_name2)


        # Check if sensors are compatible - return an error if not
        if len(self.xs[sensor1]) != len(self.xs[sensor2]):
            raise RuntimeError("Sensors have different number of steps")

        if len(self.xs[sensor1][0]) != len(self.xs[sensor2][0]):
            raise RuntimeError("Sensors have different number of points")

        # Subtract
        if inplace:

            for i in range(len(self.xs[sensor1])):
                self.ys[sensor1][i] -= self.ys[sensor2][i]
                self.sensor_names[sensor1] = new_sensor_name

            self.ligand_conc_df['Sensor'] = self.ligand_conc_df['Sensor'].replace(sensor_name1,new_sensor_name)

        else:
            ys = []
            for i in range(len(self.xs[sensor1])):
                ys.append(self.ys[sensor1][i] - self.ys[sensor2][i])

            # Fill instance
            self.xs.append(self.xs[sensor1])
            self.ys.append(ys)
            self.sensor_names.append(new_sensor_name)

            # Add new sensor name to the ligand conc df
            previous_row        = self.ligand_conc_df[self.ligand_conc_df['Sensor'] == sensor_name1]
            new_row             = previous_row.copy()
            new_row['Sensor']   = new_sensor_name
            new_row['SampleID'] = new_row['SampleID'] + ' bl subtracted'

            self.ligand_conc_df = pd.concat([self.ligand_conc_df,new_row])

        self.create_unique_sensor_names()

        return None

    def subtraction(self, list_of_sensor_names, reference_sensor, inplace=True):
        """Apply subtraction operation to a list of sensors.
        
        Parameters
        ----------
        list_of_sensor_names : list or str
            List of sensor names to subtract from. If a string is provided,
            it will be converted to a list.
        reference_sensor : str
            Name of the sensor to subtract.
        inplace : bool, optional
            If True, the subtraction is done in place, otherwise new sensors
            are created, by default True.
            
        Returns
        -------
        None
            The method modifies the instance attributes in-place.
            
        Notes
        -----
        This method modifies the following attributes:
        - self.xs
        - self.ys
        - self.sensor_names
        - self.ligand_conc_df
        """

        list_of_sensor_names = if_string_to_list(list_of_sensor_names)

        for sensor_name in list_of_sensor_names:
            self.subtraction_one_to_one(sensor_name, reference_sensor, inplace=inplace)

        return  None

    def average(self, list_of_sensor_names, new_sensor_name='Average'):
        """Average the signals of the sensors in the list.
        
        Parameters
        ----------
        list_of_sensor_names : list
            List of sensor names to average.
        new_sensor_name : str, optional
            Name of the new sensor, by default 'Average'.
            
        Returns
        -------
        None
            The method modifies the instance attributes in-place.
            
        Notes
        -----
        This method modifies the following attributes:
        - self.xs
        - self.ys
        - self.sensor_names
        - self.ligand_conc_df
            
        Raises
        ------
        RuntimeError
            If no traces are loaded.
        """

        # Check if sensors are loaded
        if not self.traces_loaded:
            raise RuntimeError("No traces loaded. Cannot perform averaging.")

        new_sensor_name = self.check_sensor_name(new_sensor_name)

        ys = []

        num_sensors = len(list_of_sensor_names)
        sensor1 = self.sensor_names.index(list_of_sensor_names[0])

        for i in range(len(self.xs[sensor1])):

            sensors = [self.sensor_names.index(sensor_name) for sensor_name in list_of_sensor_names]

            sum_ys = sum(self.ys[sensor][i] for sensor in sensors)
            ys.append(sum_ys / num_sensors)

        # Fill instance
        self.xs.append(self.xs[sensor1])
        self.ys.append(ys)
        self.sensor_names.append(new_sensor_name)

        # Add new sensor name to the ligand conc df
        previous_row        = self.ligand_conc_df[self.ligand_conc_df['Sensor'] == list_of_sensor_names[0]]
        new_row             = previous_row.copy()
        new_row['Sensor']   = new_sensor_name
        new_row['SampleID'] = new_row['SampleID'] + ' averaged'

        self.ligand_conc_df = pd.concat([self.ligand_conc_df,new_row])

        self.create_unique_sensor_names()

        return None

    def align_association(self, sensor_names=None, inplace=True, new_names=False, npoints=10):
        """Align BLI traces based on the signal before the association step(s).
        
        Parameters
        ----------
        sensor_names : str or list
            Name of the sensor(s) to align. If a string is provided,
            it will be converted to a list.
            If None, all sensors will be aligned.
        inplace : bool, optional
            If True, the alignment is done in place, otherwise new sensors
            are created, by default True.
        new_names : bool, optional
            If True, new sensor names are generated, otherwise the original
            names are kept, by default False.
        npoints : int, optional
            Number of points to use for averaging at alignment positions,
            by default 10.
            
        Returns
        -------
        None
            The method modifies the instance attributes in-place.
            
        Notes
        -----
        This method modifies the following attributes:
        - self.xs
        - self.ys
        - self.sensor_names
        - self.ligand_conc_df
        
        The alignment is performed by subtracting the average signal of the
        last `npoints` points before each association step.
        """
        if sensor_names is None:
            sensor_names = self.sensor_names
        
        else:
            sensor_names = if_string_to_list(sensor_names)

        # Find the index of the association steps
        association_steps_indices = self.df_steps.index[self.df_steps['Type'] == 'ASSOC'].to_numpy()

        # Find the dissociation steps indexes
        dissociation_steps_indices = self.df_steps.index[self.df_steps['Type'] == 'DISASSOC'].to_numpy()

        # Remove all association steps that come directly after a dissociation step (useful for single cycle kinetics)
        for idx in association_steps_indices:
            if idx-1 in dissociation_steps_indices:
                association_steps_indices = np.delete(association_steps_indices, np.where(association_steps_indices == idx)[0])

        sensor_indices = [self.sensor_names.index(sensor_name) for sensor_name in sensor_names]

        # Determine the usage of new sensor names
        use_new_names = not inplace or (inplace and new_names)

        for sensor in sensor_indices:

            # Start - Determine the new sensor name
            new_sensor_name = self.sensor_names[sensor] + ' aligned' if use_new_names else self.sensor_names[sensor]

            if new_sensor_name in self.sensor_names and not inplace:
                new_sensor_name += ' rep'
            # End of - Determine the new sensor name

            # Create a copy of the list
            ys = copy.deepcopy(self.ys[sensor])

            for i, association_step_index in enumerate(association_steps_indices):

                # Skip if the association step is the first one - in other words, there is no previous baseline step
                # Because there will be no last point to subtract...
                if association_step_index == 0:
                    continue

                #  Subtract the first point of the previous baseline step
                last_point = np.mean(self.ys[sensor][association_step_index-1][-npoints:])

                if i == 0:

                    for step in range(association_step_index-1):

                        value = self.ys[sensor][step] - last_point

                        if inplace:

                            self.ys[sensor][step] = value

                        else:

                            ys[step]    = value

                for step in range(association_step_index-1,self.no_steps):

                    value = self.ys[sensor][step] - last_point

                    if inplace:

                        self.ys[sensor][step] = value

                    else:

                        ys[step] = value

            if inplace:

                #Replace in the ligand conc df the sensor name
                if use_new_names:

                    self.ligand_conc_df['Sensor'] = self.ligand_conc_df['Sensor'].replace(self.sensor_names[sensor],new_sensor_name)
                    self.sensor_names[sensor]     = new_sensor_name

            else:

                self.xs.append(self.xs[sensor])
                self.ys.append(ys)
                self.sensor_names.append(new_sensor_name)

                # Add the new sensor name to the ligand conc df
                previous_row        = self.ligand_conc_df[self.ligand_conc_df['Sensor'] == self.sensor_names[sensor]]
                new_row             = previous_row.copy()
                new_row['Sensor']   = new_sensor_name
                new_row['SampleID'] = new_row['SampleID'] + ' aligned'

                self.ligand_conc_df = pd.concat([self.ligand_conc_df,new_row])

        self.create_unique_sensor_names()

        return None

    def subtract_experiment(self, other_experiment, inplace=True):
        
        """Subtract another SurfaceBasedExperiment from this one on a sensor-by-sensor basis.
        
        Parameters
        ----------
        other_experiment : SurfaceBasedExperiment
            The experiment to subtract from this one.
        inplace : bool, optional
            If True, the subtraction is done in place, otherwise new sensors
            are created, by default True.
        """

        # Verify that both experiments have the same number of sensors
        if len(self.sensor_names) != len(other_experiment.sensor_names):
            raise RuntimeError("Experiments have different number of sensors")
        
        # Verify that both experiments x-data is the same
        if not np.allclose(self.xs[0][0], other_experiment.xs[0][0],rtol=0.01):
            raise RuntimeError("Experiments have different time data")

        # Sort sensor names alphanumerically
        self_sensor_names_sorted = sorted(self.sensor_names)
        other_sensor_names_sorted = sorted(other_experiment.sensor_names)

        for j,sensor_name1 in enumerate(self_sensor_names_sorted):

            sensor_name2 = other_sensor_names_sorted[j]

            new_sensor_name = sensor_name1 + ' - ' + sensor_name2

            new_sensor_name = self.check_sensor_name(new_sensor_name)

            sensor1 = self.sensor_names.index(sensor_name1)
            sensor2 = other_experiment.sensor_names.index(sensor_name2)

            # Subtract
            if inplace:

                for i in range(len(self.xs[sensor1])):
                    self.ys[sensor1][i] -= other_experiment.ys[sensor2][i]
                    self.sensor_names[sensor1] = new_sensor_name

                self.ligand_conc_df['Sensor'] = self.ligand_conc_df['Sensor'].replace(sensor_name1,new_sensor_name)

            else:
                ys = []
                for i in range(len(self.xs[sensor1])):
                    ys.append(self.ys[sensor1][i] - other_experiment.ys[sensor2][i])

                # Fill instance
                self.xs.append(self.xs[sensor1])
                self.ys.append(ys)
                self.sensor_names.append(new_sensor_name)

                # Add new sensor name to the ligand conc df
                previous_row        = self.ligand_conc_df[self.ligand_conc_df['Sensor'] == sensor_name1]
                new_row             = previous_row.copy()
                new_row['Sensor']   = new_sensor_name
                new_row['SampleID'] = new_row['SampleID'] + ' bl subtracted'

                self.ligand_conc_df = pd.concat([self.ligand_conc_df,new_row])

            self.create_unique_sensor_names()

        return None

    def subtraction_by_column(self,sensors_column_one,sensors_column_two,inplace=True):
        
        """Subtract a whole column of sensors from another column of sensors.
        Useful when a screening experiment has been done

        Parameters
        ----------
        sensors_column_one : int
            Column number of the sensors to subtract from.
        sensors_column_two : int
            Column number of the sensors to subtract.
        inplace : bool, optional
            If True, the subtraction is done in place, otherwise new sensors
            are created, by default True.
        Returns
        -------
        A list of strings mentioning which sensors were subtracted from which.
        """

        # Find all sensor names that contain the column number one 
        sensors_one = [sensor_name for sensor_name in self.sensor_names if str(sensors_column_one) in sensor_name]

        # Sort them in alphabetical order
        sensors_one.sort()

        return_strings = []

        # Apply subtraction for each sensor in column one with the corresponding sensor in column two
        for sensor_name in sensors_one:

            # Create the corresponding sensor name in column two
            sensor_name_two = sensor_name.replace(str(sensors_column_one),str(sensors_column_two))

            self.subtraction_one_to_one(sensor_name,sensor_name_two,inplace=inplace)

            return_strings.append(f"Subtracted {sensor_name_two} from {sensor_name}")

        return return_strings

    def align_dissociation(self, sensor_names, inplace=True, new_names=False, npoints=10):
        """Align BLI traces based on the signal before the dissociation step(s).
        
        Parameters
        ----------
        sensor_names : str or list
            Name of the sensor(s) to align. If a string is provided,
            it will be converted to a list.
        inplace : bool, optional
            If True, the alignment is done in place, otherwise new sensors
            are created, by default True.
        new_names : bool, optional
            If True, new sensor names are generated, otherwise the original
            names are kept, by default False.
        npoints : int, optional
            Number of points to use for averaging at alignment positions,
            by default 10.
            
        Returns
        -------
        None
            The method modifies the instance attributes in-place.
            
        Notes
        -----
        This method modifies the following attributes:
        - self.xs
        - self.ys
        - self.sensor_names
        - self.ligand_conc_df
        
        The alignment is performed by smoothing the transition between association
        and dissociation steps.
        """
        sensor_names = if_string_to_list(sensor_names)

        # Find the index of the dissociation steps
        dissociation_steps_indices = self.df_steps.index[self.df_steps['Type'] == 'DISASSOC'].to_numpy()

        sensor_indices = [self.sensor_names.index(sensor_name) for sensor_name in sensor_names]

        use_new_names = not inplace or (inplace and new_names)

        for sensor in sensor_indices:

            # Determine the new sensor name
            new_sensor_name = self.sensor_names[sensor] + ' diss. aligned' if use_new_names else self.sensor_names[sensor]

            if new_sensor_name in self.sensor_names and not inplace:
                new_sensor_name += ' rep'

            ys = self.ys.copy()

            for diss_step_index in dissociation_steps_indices:

                #  Subtract the difference between the steps
                last_point = np.mean(self.ys[sensor][diss_step_index-1][-npoints:])
                next_point = np.mean(self.ys[sensor][diss_step_index][:npoints])

                diff = next_point - last_point

                value = self.ys[sensor][diss_step_index] - diff

                if inplace:

                    self.ys[sensor][diss_step_index] = value

                else:

                    ys[sensor][diss_step_index] = value

            if inplace:

                #Replace in the ligand conc df the sensor name
                self.ligand_conc_df['Sensor'] = self.ligand_conc_df['Sensor'].replace(self.sensor_names[sensor],new_sensor_name)

                self.sensor_names[sensor] = new_sensor_name

            else:

                self.xs.append(self.xs[sensor])
                self.ys.append(ys)
                self.sensor_names.append(self.sensor_names[sensor] + ' diss. aligned')

                # Add the new sensor name to the ligand conc df
                previous_row        = self.ligand_conc_df[self.ligand_conc_df['Sensor'] == self.sensor_names[sensor]]
                new_row             = previous_row.copy()
                new_row['Sensor']   = new_sensor_name
                new_row['SampleID'] = new_row['SampleID'] + ' diss. aligned'

                self.ligand_conc_df = pd.concat([self.ligand_conc_df,new_row])

        self.create_unique_sensor_names()

        return None

    def discard_steps(self, sensor_names, step_types=['KREGENERATION', 'LOADING']):
        """Discard specific step types from the sensors in the list.
        
        Parameters
        ----------
        sensor_names : str or list
            Name of the sensor(s) to process. If a string is provided,
            it will be converted to a list.
        step_types : str or list, optional
            Type(s) of steps to discard, by default ['KREGENERATION', 'LOADING'].
            If a string is provided, it will be converted to a list.
            
        Returns
        -------
        None
            The method modifies the instance attributes in-place.
            
        Notes
        -----
        This method modifies the following attributes:
        - self.ys
        
        The discarded steps are replaced with NaN values.
        """
        sensor_names = if_string_to_list(sensor_names)
        step_types   = if_string_to_list(step_types)

        sensor_indices = [self.sensor_names.index(sensor_name) for sensor_name in sensor_names]

        for step_type in step_types:

            step_indices = self.df_steps.index[self.df_steps['Type'] == step_type].to_numpy()

            for step_index in step_indices:

                for sensor in sensor_indices:

                    self.ys[sensor][step_index] = np.repeat(np.nan,len(self.ys[sensor][step_index]))

        return None

    def get_step_xy(self, sensor_name, location_loading, 
                 location_sample, step_type='ASSOC',
                 replicate=1, type='y'):
        """Return the x or y values of a specific step.
        
        Parameters
        ----------
        sensor_name : str
            Name of the sensor.
        location_loading : int
            Column location of the loading. If zero, assumes only one location.
        location_sample : int
            Column location of the sample. If zero, assumes only one location.
        step_type : str, optional
            Type of step, only 'ASSOC' or 'DISASSOC' are valid, by default 'ASSOC'.
        replicate : int, optional
            Replicate number, by default 1.
        type : str, optional
            Data type to return, either 'x' or 'y', by default 'y'.
            
        Returns
        -------
        numpy.ndarray
            The x (time) or y (signal) values of the specified step.
            
        Raises
        ------
        ValueError
            If location_sample, location_loading, or replicate are not valid integers.
        TypeError
            If any parameter has an incorrect type.
            
        Notes
        -----
        For 'x' type data, time values are processed to start from zero for each step.
        For association steps, time starts at the beginning of the step.
        For dissociation steps, time continues from the association step.
        """

        # Try to convert to integer, the variables location_sample, location_loading and Replicate
        # If it fails, raise an error
        try:
            location_sample  = int(location_sample)
            location_loading = int(location_loading)
            replicate        = int(replicate)
        except ValueError:
            raise ValueError("location_sample, location_loading and replicate must be integers")

        # Verify we have the correct data types
        for var, expected_type, name in [
            (sensor_name, str, "sensor_name"),
            (location_sample, int, "location_sample"),
            (location_loading, int, "location_loading"),
            (step_type, str, "step_type"),
            (replicate, int, "replicate"),
            (type, str, "type"),
        ]:
            if not isinstance(var, expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")

        sensor = self.sensor_names.index(sensor_name)

        cond   = self.df_steps['Type']             == 'ASSOC'

        if location_sample != 0:

            cond = np.logical_and(cond,self.df_steps['Column_location'] == str(location_sample))

        if location_loading != 0:

            cond = np.logical_and(cond,self.df_steps['Loading_location'] == str(location_loading))

        step_index = self.df_steps[cond].index.to_numpy()[replicate-1] + 1*(step_type == 'DISASSOC')

        if type == 'x':

            time = self.xs[sensor][step_index]

            try:

                # Find if we have single-cycle kinetics or multi-cycle kinetics
                previous_type = self.df_steps['Type'][step_index - 2]
                single_cycle  = previous_type == step_type

            except:

                single_cycle = False

            if not single_cycle:

                # If the step_type is an association step, subtract the first data point
                if step_type == 'ASSOC':

                    time = time - time[0]

                # If the step_type is a dissociation step, subtract the first data point of the previous step
                else:

                    time = time - self.xs[sensor][step_index-1][0]

            else:

                # Find the index of the first step_type, from the single cycle
                # Iterate over the previous steps, two at a time, until we find a step that is not a step of the same type
                i = find_index_of_previous_step_type(self.df_steps, step_index, step_type)
                
                # If we did not find a previous step of a different type
                # Assign i to the first step index of the same type
                # This is useful for single-cycle kinetics where we do not have a previous step (e.g., baseline) of a different type
                if i is None:

                    # Find the index where the step_type matches
                    i = np.where(self.df_steps['Type'] == step_type)[0][0] - 2

                # If the step_type is an association step, subtract the first data point
                if step_type == 'ASSOC':

                    time = time - self.xs[sensor][i+2][0]

                # If the step_type is a dissociation step, subtract the first data point of the previous step
                else:

                    time = time - self.xs[sensor][i+1][0]

            return time

        else:

            return self.ys[sensor][step_index]