import pandas as pd
import numpy  as np

import copy

class SurfaceBasedExperiment:

    def __init__(self,name,type):
        """
        Initialize the instance
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

        """
        Create unique sensor names by adding the name of the experiment to the sensor names
        """

        self.sensor_names_unique = [self.name + ' ' + sensor_name for sensor_name in self.sensor_names]

        return None

    def subtraction_one_to_one(self, sensor_name1, sensor_name2,inplace=True):

        """

        Subtract the signal of sensor2 from sensor1

        Args:

            sensor_name1 (str): name of the sensor to subtract from
            sensor_name2 (str): name of the sensor to subtract
            inplace (bool):     if True, the subtraction is done in place, otherwise a new sensor is created

        Results:

            It modifies the attributes self.xs, self.ys, self.sensor_names and self.ligand_conc_df

        """

        new_sensor_name = sensor_name1 + ' - ' + sensor_name2

        if new_sensor_name in self.sensor_names:

            new_sensor_name = new_sensor_name + ' rep'

        sensor1 = self.sensor_names.index(sensor_name1)
        sensor2 = self.sensor_names.index(sensor_name2)

        if not self.traces_loaded:
            print("No traces loaded")
            return None
        # Check if sensors are compatible
        if len(self.xs[sensor1]) != len(self.xs[sensor2]):
            print("Sensors have different number of steps")
            return None

        if len(self.xs[sensor1][0]) != len(self.xs[sensor2][0]):
            print("Sensors have different number of points")
            return None

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

    def subtraction(self,list_of_sensor_names,reference_sensor,inplace=True):

        """
        Apply the subtract operation to a list of sensors

        Args:
            list_of_sensor_names (list): list of sensor names to subtract
            reference_sensor (str): name of the sensor to subtract from
            inplace (bool):     if True, the subtraction is done in place, otherwise a new sensor is created
        Results:
            It modifies the attributes self.xs, self.ys, self.sensor_names and self.ligand_conc_df
        """

        if not isinstance(list_of_sensor_names, list):
            list_of_sensor_names = [list_of_sensor_names]

        for sensor_name in list_of_sensor_names:
            self.subtraction_one_to_one(sensor_name, reference_sensor, inplace=inplace)

        return  None

    def average(self,list_of_sensor_names,new_sensor_name='Average'):

        """

        Average the signals of the sensors in the list

        Args:

            list_of_sensor_names (list): list of sensor names to average
            new_sensor_name (str):       name of the new sensor

        Results:

            It modifies the attributes self.xs, self.ys, self.sensor_names and self.ligand_conc_df

        """

        # Check if sensors are loaded
        if not self.traces_loaded:
            print("No traces loaded")
            return None

        if new_sensor_name in self.sensor_names:

            new_sensor_name = new_sensor_name + ' rep'

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

    def align_association(self,sensor_names,inplace=True,new_names = False, npoints=10):

        """

        Align the BLI traces based on the signal before the association step(s)

        Args:

            sensor_names (str or list): name of the sensor(s) to align
            inplace (bool):             if True, the alignment is done in place, otherwise a new sensor is created
            new_names (bool):          if True, the new sensor name is used, otherwise the old one is used

        Results:

            It modifies the attributes self.xs, self.ys, self.sensor_names and self.ligand_conc_df

        """

        if not isinstance(sensor_names, list):
            sensor_names = [sensor_names]

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

    def align_dissociation(self,sensor_names,inplace=True,new_names = False,npoints=10):

        """
        Align the BLI traces based on the signal before the dissociation step(s)

        Args:

            sensor_names (str or list): name of the sensor(s) to align
            inplace (bool):             if True, the alignment is done in place, otherwise a new sensor is created

        Results:

            It modifies the attributes self.xs, self.ys, self.sensor_names and self.ligand_conc_df

        """

        if not isinstance(sensor_names, list):
            sensor_names = [sensor_names]

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

    def discard_steps(self,sensor_names,step_types=['KREGENERATION','LOADING']):

        """

        Discard the steps of the sensors in the list

        Args:

            sensor_names (str or list): name of the sensor(s) to analyse
            step_types (str or list):    type of the steps to discard

        Results:

            It modifies the attributes self.xs, self.ys, self.sensor_names and self.ligand_conc_df

        """
        if not isinstance(sensor_names, list):
            sensor_names = [sensor_names]

        if not isinstance(step_types, list):
            step_types = [step_types]

        sensor_indices = [self.sensor_names.index(sensor_name) for sensor_name in sensor_names]

        for step_type in step_types:

            step_indices = self.df_steps.index[self.df_steps['Type'] == step_type].to_numpy()

            for step_index in step_indices:

                for sensor in sensor_indices:

                    self.ys[sensor][step_index] = np.repeat(np.nan,len(self.ys[sensor][step_index]))

        return None

    def get_step_xy(self,sensor_name,location_loading,
                    location_sample,step_type='ASSOC',
                    replicate=1,type='y'):

        """
        Return the x or y values of a certain step

        Args:

            sensor_name (str): name of the sensor
            location_sample (int):    column location of the sample. If zero, we assume we only have one location
            location_loading (int):    column location of the loading. If zero, we assume we only have one location
            step_type (str):   type of the step, ASSOC or DISASSOC only
            replicate (int):   replicate number
            type (str):        x or y

        Returns:

            x or y (np.n) values of the step

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

                i = None
                # Find the index of the first association, from the single cycle
                # Iterate over the previous steps, two at a time, until we find a step that is not a step of the same type

                for idx in range(step_index-2,0,-2):
                    if self.df_steps['Type'][idx] != step_type:
                        i = idx
                        break

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