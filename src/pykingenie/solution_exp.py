
class SolutionBasedExp:

    def __init__(self,name,type):

        self.name                = name
        self.type                = type

        self.xs                  = None
        self.ys                  = None
        self.no_traces           = None
        self.traces_names        = None
        self.traces_names_unique = None
        self.conc_df             = None
        self.traces_loaded       = False

    def create_unique_traces_names(self):

        """
        Create unique traces names by appending the name of the experiment to the trace names.
        """

        self.traces_names_unique = [self.name + ' ' + trace_name for trace_name in self.traces_names]

        return None

    def get_trace_xy(self,trace_name,type='y'):

        """
        Return the x or y values of a certain step

        Args:

            trace_name (str): name of the trace
            type (str):        x or y

        Returns:

            x or y (np.n) values of the step

        """

        trace_id = self.traces_names.index(trace_name)

        if type == 'x':

            return self.xs[trace_id]

        else:

            return self.ys[trace_id]

