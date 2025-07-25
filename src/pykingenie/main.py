import pandas as pd

from .fitter          import KineticsFitter
from .fitter_solution import KineticsFitterSolution

class KineticsAnalyzer:

    def __init__(self):

        self.experiments      = {}
        self.experiment_names = []
        self.fittings         = {}
        self.fittings_names   = []

    def delete_experiment(self,experiment_names):

        if not isinstance(experiment_names, list):
            experiment_names = [experiment_names]

        for experiment_name in experiment_names:

            if experiment_name in self.experiment_names:

                del self.experiments[experiment_name]
                self.experiment_names.remove(experiment_name)

        return None

    def add_experiment(self,experiment,experiment_name):

        self.delete_experiment(experiment_name)

        self.experiments[experiment_name] = experiment
        self.experiment_names.append(experiment_name)

        return None

    def init_fittings(self):

        self.fittings       = {}
        self.fittings_names = []

        return None

    def add_fitting(self,fitting,fitting_name):

        if fitting_name in self.fittings_names:

                del self.fittings[fitting_name]
                self.fittings_names.remove(fitting_name)

        self.fittings[fitting_name] = fitting
        self.fittings_names.append(fitting_name)

        return None

    def get_experiment_properties(self, variable,fittings=False):

        if fittings:

            return [getattr(self.fittings[fitting_name], variable) for fitting_name in self.fittings_names]

        else:

            return [getattr(self.experiments[exp_name], variable) for exp_name in self.experiment_names]

    def merge_ligand_conc_df(self):

        # Combine ligand concentration data frames from experiments
        dfs = [exp.ligand_conc_df for exp in self.experiments.values()]
        df  = pd.concat(dfs, ignore_index=True)

        # Add a 'Select' column with all values set to True
        df['Select'] = True

        columns_to_convert     = ["Analyte_location", "Loading_location", "Replicate"]
        df[columns_to_convert] = df[columns_to_convert].apply(lambda col: col.astype(int))

        # For each combination of SampleID, Experiment, and Replicate, create a unique ID
        df['Smax_ID'] = (df['SampleID'].astype(str) +
                         df['Experiment'].astype(str) +
                         df['Replicate'].astype(str))
        df['Smax_ID'] = pd.factorize(df['Smax_ID'])[0].astype(int)

        # Move specified columns to the end
        columns_to_move = ['Analyte_location', 'Loading_location', 'Replicate', 'Experiment']
        for col in columns_to_move:
            if col in df.columns:
                df = df[[c for c in df.columns if c != col] + [col]]

        # Rename the column 'Concentration_micromolar' to '[Analyte] (μM)'
        df.rename(columns={'Concentration_micromolar': '[Analyte] (μM)'}, inplace=True)

        # Delete columns if they have only one unique value
        for col in ['Experiment', 'Replicate', 'Analyte_location', 'Loading_location']:
            if col in df.columns and df[col].nunique() == 1:
                df.drop(columns=[col], inplace=True)

        self.combined_ligand_conc_df = df.copy()

        return None

    def merge_conc_df_solution(self):

        # Combine concentration data frames from solution-based experiments
        dfs = [exp.conc_df for exp in self.experiments.values()]
        df  = pd.concat(dfs, ignore_index=True)

        # Add a 'Select' column with all values set to True
        df['Select'] = True

        # Move specified columns to the end
        columns_to_move = ['Experiment']
        for col in columns_to_move:
            if col in df.columns:
                df = df[[c for c in df.columns if c != col] + [col]]

        # Rename the column 'Protein_concentration_micromolar' to '[Protein] (μM)'
        df.rename(columns={'Protein_concentration_micromolar': '[Protein] (μM)'}, inplace=True)
        # Rename the column 'Ligand_concentration_micromolar' to '[Ligand] (μM)'
        df.rename(columns={'Ligand_concentration_micromolar': '[Ligand] (μM)'}, inplace=True)

        self.combined_conc_df = df.copy()

        return None

    def generate_fittings(self,df):

        """
        Given a data frame, extract the kinetics data and create a fitting object for each sample.
        The data frame should contain the following columns:
        - 'Sensor': The sensor ID
        - 'Experiment': The experiment ID
        - 'Replicate': The replicate number
        - 'Analyte_location': The location of the analyte
        - 'Loading_location': The location of the loading
        - 'Concentration_micromolar': The concentration of the analyte in micromolar
        - 'SampleID': The sample ID
        - 'Smax_ID': The Smax ID
        - 'Select': A boolean value indicating whether to select the sample or not

        """

        # List of messages to be printed to the console
        messages          = []

        time_diss_all  =  []
        diss_all       =  []
        time_assoc_all =  []
        assoc_all      =  []

        # Reset dataframe index
        df.reset_index(drop=True, inplace=True)

        # Find if 'Experiment' is in the data frame column names
        df_colnames    = df.columns
        have_exp_column = 'Experiment'        in df_colnames
        have_rep_column = 'Replicate'         in df_colnames
        have_analyte_loc = 'Analyte_location' in df_colnames
        have_loading_loc = 'Loading_location' in df_colnames

        if not have_exp_column:
            exp = self.experiment_names[0]
        if not have_rep_column:
            replicate = 1
        if not have_analyte_loc:
            analyte_loc = 0
        if not have_loading_loc:
            loading_loc = 0

        # Find which column has the analyte concentration, with the word '[Analyte]' and rename it to 'Concentration_micromolar'
        for column in df.columns:
            if '[Analyte]' in column:
                df.rename(columns={column: 'Concentration_micromolar'}, inplace=True)

        for row in range(len(df)):
            if have_exp_column:
                exp = df.loc[row, 'Experiment']
            if have_rep_column:
                replicate = int(df.loc[row, 'Replicate'])
            if have_analyte_loc:
                analyte_loc = int(df.loc[row, 'Analyte_location'])
            if have_loading_loc:
                loading_loc = int(df.loc[row, 'Loading_location'])

            sensor = df.loc[row, 'Sensor']

            diss_all.append(self.experiments[exp].get_step_xy(sensor, loading_loc, analyte_loc, 'DISASSOC', replicate, 'y'))
            assoc_all.append(self.experiments[exp].get_step_xy(sensor, loading_loc, analyte_loc, 'ASSOC', replicate, 'y'))

            time_diss  = self.experiments[exp].get_step_xy(sensor, loading_loc, analyte_loc, 'DISASSOC', replicate, 'x')
            time_assoc = self.experiments[exp].get_step_xy(sensor, loading_loc, analyte_loc, 'ASSOC', replicate, 'x')

            time_diss_all.append(time_diss)
            time_assoc_all.append(time_assoc)

        # Find same samples with different experiment IDs
        unique_sample = df['SampleID'].unique()

        # Group by sample
        for unq in unique_sample:

            assoc_lst = []
            diss_lst  = []

            time_assoc_lst = []
            time_diss_lst  = []

            lig_conc_vec = []
            smax_id_vec  = []

            ids = df.index[df['SampleID'] == unq].tolist()

            # Iterate over the Smax IDs
            unq_smax_id = df.loc[ids, 'Smax_ID'].unique()

            message_for_sample = False

            for smax_id in unq_smax_id:
                ids = df.index[
                    (df['SampleID'] == unq) &
                    (df['Smax_ID'] == smax_id) &
                    (df['Concentration_micromolar'] > 0) &
                    (df['Select'])
                ].tolist()

                if len(ids) == 0:
                    continue

                if not message_for_sample:
                    messages.append("Creating a new fitting object for sample: " + unq)
                    message_for_sample = True

                assoc_sel = [assoc_all[i] for i in ids]
                diss_sel  = [diss_all[i]  for i in ids]

                time_assoc_sel = [time_assoc_all[i] for i in ids]
                time_diss_sel  = [time_diss_all[i]  for i in ids]
                lig_conc       = df.loc[ids, 'Concentration_micromolar'].tolist()
                smax_ids       = df.loc[ids, 'Smax_ID'].tolist()

                # Append to the lists
                assoc_lst.extend(assoc_sel)
                diss_lst.extend(diss_sel)
                time_assoc_lst.extend(time_assoc_sel)
                time_diss_lst.extend(time_diss_sel)
                lig_conc_vec.extend(lig_conc)
                smax_id_vec.extend(smax_ids)

            if len(assoc_lst) == 0:
                continue

            # Find initial times of the association signal
            time_inits = [t[0] for t in time_assoc_lst]

            # Find the index that sorts them
            sorted_indices = [index for index, _ in sorted(enumerate(time_inits), key=lambda x: x[1])]

            # Sort them
            time_assoc_lst = [time_assoc_lst[i] for i in sorted_indices]
            assoc_lst      = [assoc_lst[i]      for i in sorted_indices]
            time_diss_lst  = [time_diss_lst[i]  for i in sorted_indices]
            diss_lst       = [diss_lst[i]       for i in sorted_indices]
            lig_conc_vec   = [lig_conc_vec[i]   for i in sorted_indices]
            smax_id_vec    = [smax_id_vec[i]    for i in sorted_indices]

            fit = KineticsFitter(
                time_assoc_lst=time_assoc_lst,
                association_signal_lst=assoc_lst,
                lig_conc_lst=lig_conc_vec,
                time_diss_lst=time_diss_lst,
                dissociation_signal_lst=diss_lst,
                smax_id=smax_id_vec,
                name_lst=[f"{unq}_id_{smax}" for smax in unq_smax_id],
                is_single_cycle=any([t[0] > 1 for t in time_assoc_lst])
            )

            fit.get_steady_state()

            self.add_fitting(fit, unq)

        return messages

    def generate_fittings_solution(self,df):

        """
        Given a data frame, extract the kinetics data and create a fitting object for each sample.
        The data frame should contain the following columns:
        - 'Trace': The trace ID
        - 'Experiment': The experiment ID
        - '[Protein] (μM)': The protein concentration in micromolar
        - '[Ligand] (μM)': The ligand concentration in micromolar
        - 'SampleID': The sample ID
        - 'Select': A boolean value indicating whether to select the sample or not
        """

        # List of messages to be printed to the console
        messages          = []

        time_assoc_all =  []
        assoc_all      =  []

        # Reset dataframe index
        df.reset_index(drop=True, inplace=True)

        # Find if 'Experiment' is in the data frame column names
        df_colnames     = df.columns
        have_exp_column = 'Experiment'        in df_colnames

        if not have_exp_column:
            exp = self.experiment_names[0]

        # Find which column has the Protein concentration, with the word '[Protein]' and rename it to 'Protein_concentration_micromolar'
        # Find which column has the ligand concentration, with the word '[Ligand]' and rename it to 'Ligand_concentration_micromolar'
        for column in df.columns:
            if '[Protein]' in column:
                df.rename(columns={column: 'Protein_concentration_micromolar'}, inplace=True)
            if '[Ligand]' in column:
                df.rename(columns={column: 'Ligand_concentration_micromolar'}, inplace=True)

        for row in range(len(df)):
            if have_exp_column:
                exp = df.loc[row, 'Experiment']

            trace = df.loc[row, 'Trace']

            x = self.experiments[exp].get_trace_xy(trace, 'x')
            y = self.experiments[exp].get_trace_xy(trace, 'y')

            time_assoc_all.append(x)
            assoc_all.append(y)

        # Find same samples with different experiment IDs
        unique_sample = df['SampleID'].unique()

        # Group by sample
        for unq in unique_sample:

            assoc_lst = []

            time_assoc_lst = []

            lig_conc_vec  = []
            prot_conc_vec = []

            ids = df.index[df['SampleID'] == unq].tolist()

            if len(ids) == 0:
                continue

            messages.append("Creating a new fitting object for sample: " + unq)

            assoc_sel = [assoc_all[i] for i in ids]
            time_assoc_sel = [time_assoc_all[i] for i in ids]
            lig_conc       = df.loc[ids, 'Ligand_concentration_micromolar'].tolist()
            prot_conc      = df.loc[ids, 'Protein_concentration_micromolar'].tolist()

            # Append to the lists
            assoc_lst.extend(assoc_sel)
            time_assoc_lst.extend(time_assoc_sel)
            lig_conc_vec.extend(lig_conc)
            prot_conc_vec.extend(prot_conc)

            fit = KineticsFitterSolution(
                name=unq,
                assoc=assoc_lst,
                lig_conc=lig_conc_vec,
                protein_conc=prot_conc_vec,
                time_assoc=time_assoc_lst

            )

            fit.get_steady_state()

            self.add_fitting(fit, unq)

        return messages

    def submit_steady_state_fitting(self):

        for kf in self.fittings.values():

            if not kf.is_single_cycle:
                kf.fit_steady_state()
                kf.create_fitting_bounds_table()

            else:

                kf.Smax_upper_bound_factor = 1e2 # Normal values for lower than micromolar affinity

        return None

    def submit_fitting_solution(self,fitting_model='single'):

        """
        Fit a variety of models to the kinetics data in the solution-based experiments.
        """

        for kf in self.fittings.values():

            if fitting_model == 'single':
                kf.fit_single_exponentials()

            elif fitting_model == 'double':
                kf.fit_double_exponentials()

            elif fitting_model == 'one_binding_site':
                kf.fit_one_binding_site()

            elif fitting_model == 'one_binding_site_if':
                kf.fit_induced_fit()

            else:
                raise ValueError("Unknown fitting model: " + fitting_model)

        return None

    def submit_kinetics_fitting(self,fitting_model='one_to_one',
                                fitting_region='association_dissociation',
                                linkedSmax=False):

        for kf in self.fittings.values():

            try:

                if fitting_model == 'one_to_one' and fitting_region == 'association_dissociation':
                    kf.fit_one_site_assoc_and_disso(shared_smax=linkedSmax)
                    kf.fit_single_exponentials()

                if fitting_model == 'one_to_one_mtl' and fitting_region == 'association_dissociation':
                    kf.fit_one_site_assoc_and_disso(shared_smax=linkedSmax, fit_ktr=True)

                if fitting_model == 'one_to_one' and fitting_region == 'association':
                    kf.fit_one_site_association(shared_smax=linkedSmax)
                    kf.fit_single_exponentials()

                if fitting_model == 'one_to_one' and fitting_region == 'dissociation':
                    kf.fit_one_site_dissociation()

                if fitting_model == 'one_to_one_if' and fitting_region == 'association_dissociation':
                    kf.fit_one_site_if_assoc_and_disso(shared_smax=linkedSmax)

                kf.create_fitting_bounds_table()

            except:

                pass

        return None

    def calculate_asymmetric_error(self,shared_smax=True,fixed_t0=True,fit_ktr=False):

        for kf in self.fittings.values():

            kf.calculate_ci95(
                shared_smax=shared_smax,
                fixed_t0=fixed_t0,
                fit_ktr=fit_ktr
            )

        return None

    def get_fitting_results(self):
        """
        Get the results of the fitting process into a nice dataframe
        """
        dfs = []

        for name in self.fittings_names:
            fit_params = self.fittings[name].fit_params_kinetics
            if fit_params is not None:
                df = fit_params.copy()
                df['Name'] = name
                dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        self.fit_params_kinetics_all = df.copy()

        return None

