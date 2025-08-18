
class SolutionBasedExp:
    """
    Base class for solution-based kinetics experiments.
    
    This class serves as a foundation for classes that handle solution-based
    kinetic experiments data.
    
    Parameters
    ----------
    name : str
        Name of the experiment.
    type : str
        Type of the experiment.
        
    Attributes
    ----------
    name : str
        Name of the experiment.
    type : str
        Type of the experiment.
    xs : list of numpy.ndarray or None
        List of x values (time) for each trace.
    ys : list of numpy.ndarray or None
        List of y values (signal) for each trace.
    no_traces : int or None
        Number of traces.
    traces_names : list of str or None
        List of trace names.
    traces_names_unique : list of str or None
        List of unique trace names (includes experiment name).
    conc_df : pandas.DataFrame or None
        DataFrame with concentration information.
    traces_loaded : bool
        Whether traces have been loaded.
    """

    def __init__(self, name, type):
        """Initialize the SolutionBasedExp instance.
        
        Parameters
        ----------
        name : str
            Name of the experiment.
        type : str
            Type of the experiment.
        """

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
        Create unique trace names by appending the experiment name to the trace names.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            Updates the traces_names_unique attribute.
        
        Notes
        -----
        This method creates unique identifiers for each trace by combining
        the experiment name with the individual trace names.
        """

        self.traces_names_unique = [self.name + ' ' + trace_name for trace_name in self.traces_names]

        return None

    def cut_off_time(self,time):

        """
        Remove all data before a specified time.

        Parameters
        ----------
        time : float
            The time to cut off the data.
        """

        for i in range(len(self.xs)):

            self.ys[i] = self.ys[i][self.xs[i] <= time]
            self.xs[i] = self.xs[i][self.xs[i] <= time]

        return None

    def get_trace_xy(self, trace_name, type='y'):
        """
        Return the x or y values of a specific trace.
        
        Parameters
        ----------
        trace_name : str
            Name of the trace to retrieve data from.
        type : {'x', 'y'}, optional
            Type of data to return. Default is 'y'.
            
        Returns
        -------
        numpy.ndarray
            The x or y values of the specified trace.
            
        Notes
        -----
        This method finds the trace by name in the traces_names list
        and returns either its x values (time) or y values (signal).
        """

        trace_id = self.traces_names.index(trace_name)

        if type == 'x':

            return self.xs[trace_id]

        else:

            return self.ys[trace_id]

