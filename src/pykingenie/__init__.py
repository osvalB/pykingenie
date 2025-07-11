from .utils.processing       import *
from .utils.math             import *
from .utils.signal_surface   import *
from .utils.signal_solution  import *
from .utils.fitting_surface  import *
from .utils.fitting_general  import *
from .utils.fitting_solution import *

try:
    from .utils.plotting import *
except ImportError as e:
    # print error but continue without plotting
    print(f"Plotting libraries not available: {e}")
    #please install pyplot, kaleido and matplotlib
    print("Please install plotly, kaleido and matplotlib for plotting capabilities.")

from .octet             import *
from .gator             import *
from .kingenie_surface  import *
from .main              import *
from .kingenie_solution import *
