from .main              import KineticsAnalyzer
from .fitter_solution   import KineticsFitterSolution
from .fitter_surface    import KineticsFitter
from .octet             import OctetExperiment
from .gator             import GatorExperiment
from .kingenie_solution import KinGenieCsvSolution
from .kingenie_surface  import KinGenieCsv

from .utils.processing import (
    guess_experiment_name,
    guess_experiment_type,
    concat_signal_lst,
    get_palette,
    get_plotting_df,
    subset_data,
    sample_type_to_letter,
    get_colors_from_numeric_values
)

from .utils.math import (
    single_exponential,
    double_exponential,
    median_filter,
    rss_p,
    get_rss,
    get_desired_rss
)

from .utils.signal_surface import (
    steady_state_one_site,
    one_site_association_analytical,
    one_site_dissociation_analytical,
    ode_one_site_mass_transport_association,
    ode_one_site_mass_transport_dissociation,
    solve_ode_one_site_mass_transport_association,
    solve_ode_one_site_mass_transport_dissociation,
    differential_matrix_association_induced_fit,
    differential_matrix_association_conformational_selection,
    differential_matrix_dissociation_induced_fit,
    differential_matrix_dissociation_conformational_selection,
    constant_vector_induced_fit,
    constant_vector_conformational_selection,
    solve_all_states_fast,
    solve_steady_state,
    solve_induced_fit_association,
    solve_conformational_selection_association,
    solve_induced_fit_dissociation,
    solve_conformational_selection_dissociation,
    ode_mixture_analyte_association,
    solve_ode_mixture_analyte_association,
    ode_mixture_analyte_dissociation,
    solve_ode_mixture_analyte_dissociation
)

from .utils.signal_solution  import (
    ode_one_site_insolution,
    solve_ode_one_site_insolution,
    signal_ode_one_site_insolution,
    ode_induced_fit_insolution,
    solve_ode_induced_fit_insolution,
    signal_ode_induced_fit_insolution,
    ode_conformational_selection_insolution,
    solve_ode_conformational_selection_insolution,
    signal_ode_conformational_selection_insolution,
    get_initial_concentration_conformational_selection,
    get_kobs_induced_fit,
    get_kobs_conformational_selection
)

from .utils.fitting_surface  import (
    guess_initial_signal,
    fit_steady_state_one_site,
    steady_state_one_site_asymmetric_ci95,
    fit_one_site_association,
    fit_one_site_dissociation,
    fit_one_site_assoc_and_disso,
    fit_induced_fit_sites_assoc_and_disso,
    fit_one_site_assoc_and_disso_ktr,
    one_site_assoc_and_disso_asymmetric_ci95,
    one_site_assoc_and_disso_asymmetric_ci95_koff
)

from .utils.fitting_general  import (
    fit_single_exponential,
    fit_double_exponential
)

from .utils.fitting_solution import (
    fit_one_site_solution,
    fit_induced_fit_solution,
    find_initial_parameters_induced_fit_solution
)

from .utils.plotting import (
    config_fig,
    plot_plate_info,
    plot_traces,
    plot_traces_all,
    plot_steady_state,
    plot_association_dissociation
)

# Package metadata
__version__ = '0.1.0'
__author__ = 'osvalB'
__email__ = 'oburastero@gmail.com'
__description__ = 'A Python package for kinetics data analysis'