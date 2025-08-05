import numpy         as np
import pandas        as pd

from scipy.integrate import solve_ivp
from scipy.linalg    import solve
from scipy.linalg    import expm

__all__ = [
    'steady_state_one_site',
    'one_site_association_analytical',
    'one_site_dissociation_analytical',
    'ode_one_site_mass_transport_association',
    'ode_one_site_mass_transport_dissociation',
    'solve_ode_one_site_mass_transport_association',
    'solve_ode_one_site_mass_transport_dissociation',
    'differential_matrix_association_induced_fit',
    'differential_matrix_association_conformational_selection',
    'differential_matrix_dissociation_induced_fit',
    'differential_matrix_dissociation_conformational_selection',
    'constant_vector_induced_fit',
    'constant_vector_conformational_selection',
    'solve_steady_state',
    'solve_all_states_fast',
    'solve_induced_fit_association',
    'solve_conformational_selection_association',
    'solve_induced_fit_dissociation',
    'solve_conformational_selection_dissociation',
    'ode_mixture_analyte_association',
    'solve_ode_mixture_analyte_association',
    'ode_mixture_analyte_dissociation',
    'solve_ode_mixture_analyte_dissociation'
]

def steady_state_one_site(C, Rmax, Kd):
    """
    Calculate the steady state signal for a given concentration of ligand.
    
    If the concentration is zero, the signal is zero.
    If the concentration is infinite, the signal is Rmax.
    The units of Kd must match the units of C (ligand concentration).
    Rmax depends on the amount of loaded receptor.

    Parameters
    ----------
    C : np.ndarray
        Concentration of the analyte.
    Rmax : float
        Maximum response of the analyte.
    Kd : float
        Equilibrium dissociation constant.
        
    Returns
    -------
    np.ndarray
        Steady state signal, according to the given parameters.
    """
    C    = np.array(C)
    Rmax = np.array(Rmax)
    signal = Rmax*C/(Kd + C)
    return signal

def one_site_association_analytical(t, s0, s_max, k_off, Kd, A, t0=0):
    """
    Analytical solution for the one site association model.

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    s0 : float
        Initial signal.
    s_max : float
        Maximum signal.
    k_off : float
        Dissociation rate constant.
    Kd : float
        Equilibrium dissociation constant.
    A : float
        Concentration of the analyte.
    t0 : float, optional
        Initial time offset, default is 0.

    Returns
    -------
    np.ndarray
        Signal over time.
    """
    t = t - t0
    # Precompute constants
    rate = k_off * (A + Kd) / Kd
    s_eq = (A / (A + Kd)) * s_max

    # Solution for s(t)
    s_t = s_eq + (s0 - s_eq) * np.exp(-rate * t)

    return s_t

def one_site_dissociation_analytical(t, s0, k_off, t0=0):
    """
    Analytical solution for the one site model - dissociation phase.

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    s0 : float
        Initial signal.
    k_off : float
        Dissociation rate constant.
    t0 : float, optional
        Initial time offset, default is 0.

    Returns
    -------
    np.ndarray
        Signal over time.
    """
    t   = t + t0
    s_t = s0 * np.exp(-k_off*t)

    return s_t

def ode_one_site_mass_transport_association(t, y, params, analyte_conc):
    """
    ODE for Mass Transport-limited binding.
    
    See https://www.cell.com/AJHG/fulltext/S0006-3495(07)70982-7

    Parameters
    ----------
    t : float
        Time variable.
    y : list
        List of the state variables [s1,cs].
    params : list
        List of the parameters [K_d, k_off, k_tr, s_max].
    analyte_conc : float
        Concentration of the analyte.

    Returns
    -------
    list
        [ds1_dt, dcs_dt] representing the signal and analyte surface concentration derivatives.
    """
    s1, cs = y
    K_d, k_off, k_tr, s_max = params

    k_on = k_off / K_d

    c0 = analyte_conc - cs

    ds1_dt = k_on * cs * (s_max - s1) - k_off * s1
    dcs_dt = k_tr * (c0 - cs) - ds1_dt

    return [ds1_dt, dcs_dt]

def ode_one_site_mass_transport_dissociation(t, s1, params):
    """
    ODE for Mass Transport-limited binding - dissociation phase.
    
    See Equation 7 from https://pmc.ncbi.nlm.nih.gov/articles/PMC4134667/

    Parameters
    ----------
    t : float
        Time variable.
    s1 : float
        The state variable representing the signal.
    params : list
        List of the parameters [K_d, k_off, k_tr, s_max].

    Returns
    -------
    float
        The derivative of signal over time.
    """
    K_d, k_off, k_tr, s_max = params

    k_on = k_off / K_d

    ds_dt = (-k_off * s1) / (1 + (k_on / k_tr) * (s_max - s1))

    return ds_dt

def solve_ode_one_site_mass_transport_association(t, s1_0, cs_0, analyte_conc, K_d, k_off, k_tr, s_max, t0=0):
    """
    Solves the ODE for the one site mass transport model - association phase.

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    s1_0 : float
        Initial signal.
    cs_0 : float
        Initial analyte surface concentration.
    analyte_conc : float
        Concentration of the analyte.
    K_d : float
        Equilibrium dissociation constant.
    k_off : float
        Dissociation rate constant.
    k_tr : float
        Mass transport rate constant.
    s_max : float
        Maximum signal.
    t0 : float, optional
        Initial time offset, default is 0.
        
    Returns
    -------
    np.ndarray
        Signal over time.
    """
    t = t + t0

    out = solve_ivp(ode_one_site_mass_transport_association,t_span=[np.min(t), np.max(t)],
                    t_eval=t,y0=[s1_0,cs_0],args=([K_d,k_off,k_tr,s_max],analyte_conc),method="LSODA")

    signal = out.y[0]
    return signal

def solve_ode_one_site_mass_transport_dissociation(t, s1_0, K_d, k_off, k_tr, s_max, t0=0):
    """
    Solves the ODE for the one site mass transport model - dissociation phase.

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    s1_0 : float
        Initial signal.
    K_d : float
        Equilibrium dissociation constant.
    k_off : float
        Dissociation rate constant.
    k_tr : float
        Mass transport rate constant.
    s_max : float
        Maximum signal.
    t0 : float, optional
        Initial time offset, default is 0.
        
    Returns
    -------
    np.ndarray
        Signal over time.
    """
    t = t + t0

    out = solve_ivp(ode_one_site_mass_transport_dissociation,t_span=[np.min(t), np.max(t)],
                    t_eval=t,y0=[s1_0],args=([K_d,k_off,k_tr,s_max],),method="LSODA")

    signal = out.y[0]
    return signal

def differential_matrix_association_induced_fit(koff, kon, kc, krev, a):
    """
    Association step set of differential equations for the induced fit model.
    
    Differential equations for -dS/dt and dS1/dt where:
    S    = R([E2S] + [E1S]) ; Total signal
    S1   = R[E2S]           ; Signal produced by E2S
    Smax = R([Et])          ; Maximum signal

    Note: E is here the molecule attached to the surface, S is the substrate.
    The signal produced by E2S and E1S is the same.

    Parameters
    ----------
    koff : float
        Rate constant for E2S -> E2 + S.
    kon : float
        Rate constant for E2 + S -> E2S.
    kc : float
        Rate constant for E1 -> E2.
    krev : float
        Rate constant for E2 -> E1.
    a : float
        Concentration of the analyte (molecule being flown).

    Returns
    -------
    np.ndarray
        Differential matrix.
    """
    row1 = [-koff - kon * a, -koff]
    row2 = [-kc, -kc - krev]

    matrix = np.array([row1, row2])

    return matrix

def differential_matrix_association_conformational_selection(koff, kon, kc, krev, a):
    """
    Conformational selection model for the association step.
    
    E1     <-> E2
    E2 + S <-> E2S
    
    Set of differential equations for (d[E1]*R)/dt and d[E2S]*R/dt.
    The signal is only given by E2S.

    Parameters
    ----------
    koff : float
        Rate constant for E2S -> E2 + S.
    kon : float
        Rate constant for E2 + S -> E2S.
    kc : float
        Rate constant for E1 -> E2.
    krev : float
        Rate constant for E2 -> E1.
    a : float
        Concentration of the analyte (molecule being flown).
        
    Returns
    -------
    np.ndarray
        Differential matrix.
    """
    row1 = [-kc - krev, -krev]
    row2 = [-kon*a,-kon*a - koff]

    matrix = np.array([row1, row2])

    return matrix

def differential_matrix_dissociation_induced_fit(koff, kc, krev):
    """
    Dissociation step set of differential equations for the induced fit model.
    
    Differential equations for -ds/dt and ds1/dt.

    Parameters
    ----------
    koff : float
        Rate constant for E2S -> E2 + S.
    kc : float
        Rate constant for E1 -> E2.
    krev : float
        Rate constant for E2 -> E1.

    Returns
    -------
    np.ndarray
        Differential matrix.
    """
    row1 = [-koff, -koff]
    row2 = [-kc, -kc - krev]

    matrix = np.array([row1, row2])

    return matrix

def differential_matrix_dissociation_conformational_selection(koff, kc, krev):
    """
    Dissociation step set of differential equations for the conformational selection model.
    
    Differential equations for (d[E1]*R)/dt and d[E2S]*R/dt.
    The signal is only proportional to E2S.

    Parameters
    ----------
    koff : float
        Rate constant for E2S -> E2 + S.
    kc : float
        Rate constant for E1 -> E2.
    krev : float
        Rate constant for E2 -> E1.

    Returns
    -------
    np.ndarray
        Differential matrix.
    """
    row1 = [-kc -krev, -krev]
    row2 = [0, -koff]

    matrix = np.array([row1, row2])

    return matrix

def constant_vector_induced_fit(koff, a, kc):
    """
    Calculate the constant vector for the induced fit model.
    
    Parameters
    ----------
    koff : float
        Rate constant for E·S -> E + S.
    a : float
        Concentration of the analyte (molecule being flown).
    kc : float
        Rate constant for E·S -> ES, induced fit step.
        
    Returns
    -------
    np.ndarray
        Constant vector.
    """
    return [koff * a, kc * a]

def constant_vector_conformational_selection(kon, smax, krev, a):
    """
    Calculate the constant vector for the conformational selection model.
    
    Parameters
    ----------
    kon : float
        Rate constant for E2 + S -> E2S.
    smax : float
        Signal proportional to the complex (E2S).
    krev : float
        Rate constant for E2 -> E1.
    a : float
        Concentration of the analyte (molecule being flown).
        
    Returns
    -------
    np.ndarray
        Constant vector.
    """
    return [krev*smax, kon*a*smax]

def solve_steady_state(A, b):
    """
    Calculate the steady state solution of the system of equations.
    
    Parameters
    ----------
    A : np.ndarray
        Differential matrix.
    b : np.ndarray
        Constant vector.
        
    Returns
    -------
    np.ndarray
        Steady state solution.
    """
    return -solve(A, b)

def solve_induced_fit_association(time, a_conc, kon, koff, kc, krev, sP1L=0, sP2L=0, smax=0):
    """
    Obtain the signal for the induced fit model (surface-based).

    We assume that the signal is given by the complex E·S and ES is the same.
    In other words, the complex before and after the conformational change produces the same signal.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    a_conc : float
        Concentration of the analyte (molecule being flown).
    kon : float
        Rate constant for E + S -> E·S.
    koff : float
        Rate constant for E·S -> E + S.
    kc : float
        Rate constant for E·S -> ES.
    krev : float
        Rate constant for ES -> E·S.
    sP1L : float, optional
        Initial concentration of E·S, default is 0.
    sP2L : float, optional
        Initial concentration of ES, default is 0.
    smax : float, optional
        Signal proportional to the complex (E·S or ES), default is 0.

    Returns
    -------
    np.ndarray
        Signal over time.
    """
    time = time - np.min(time) # Start at zero

    b_col = constant_vector_induced_fit(koff, smax, kc)

    # Initial conditions for St - R(E·S) - R(ES), and R(ES)
    # where St is the max signal, R(E·S) is the signal produced by E·S, and R(ES) is the signal produced by ES

    initial_conditions = np.array([smax-sP1L-sP2L, sP2L])

    m = differential_matrix_association_induced_fit(koff, kon, kc, krev, a_conc)

    res_steady_state = solve_steady_state(m, b_col)

    states = solve_all_states_fast(time, m, initial_conditions, res_steady_state)

    # Create a numpy array of three columns
    col1 = smax - states[:, 0]  # Total signal minus the signal produced by E·S and ES
    col2 = smax - states[:, 0] - states[:, 1]  # Signal produced by E·S
    col3 = states[:, 1]  # Signal produced by ES

    arr = np.column_stack((col1, col2, col3))

    return arr

def solve_all_states_fast(time, A, initial_conditions, steady_state_value):
    """
    Solve the system of differential equations for all time points using matrix exponentiation.
    
    Parameters
    ----------
    time : np.ndarray
        Array of time points to solve the system for.
    A : np.ndarray
        Differential matrix of the system.
    initial_conditions : np.ndarray
        Initial conditions for the system.
    steady_state_value : np.ndarray
        Steady state solution of the system.
        
    Returns
    -------
    np.ndarray
        Array of shape (len(time), len(initial_conditions)) containing the state of the system at each time point.
    """

    v = initial_conditions - steady_state_value
    eigvals, eigvecs = np.linalg.eig(A)
    Vinv = np.linalg.inv(eigvecs)

    # Precompute eigvecs @ v once
    alpha = Vinv @ v

    # Preallocate output
    states = np.empty((len(time), len(v)))

    # Loop over time using vectorized exponentials
    for i, t in enumerate(time):
        exp_diag = np.exp(eigvals * t)
        exp_v = eigvecs @ (exp_diag * alpha)
        states[i] = steady_state_value + exp_v

    return states

def solve_conformational_selection_association(time, a_conc, kon, koff, kc, krev, smax=0, sP1=0, sP2L=0):
    """
    Obtain the signal for the conformational selection model (surface-based).

    We assume that the signal is given by the complex E2S only.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    a_conc : float
        Concentration of the analyte (molecule being flown).
    kon : float
        Rate constant for E2 + S -> E2S.
    koff : float
        Rate constant for E2S -> E2 + S.
    kc : float
        Rate constant for E1 -> E2.
    krev : float
        Rate constant for E2 -> E1.
    smax : float, optional
        Maximum signal that the complex (E2S) can produce, default is 0.
    sP1 : float, optional
        Initial concentration of E1, default is 0.
    sP2L : float, optional
        Initial concentration of E2S, default is 0.

    Returns
    -------
    np.ndarray
        Signal over time, with columns for the signal, sP1, and sP2.
    """

    time = time - np.min(time) # Start at zero

    b_col   = constant_vector_conformational_selection(kon, smax, krev, a_conc)

    initial_conditions = np.array([smax / (kc / krev + 1), 0]) if sP2L == 0 else np.array([sP1, sP2L])

    m = differential_matrix_association_conformational_selection(koff, kon, kc, krev, a_conc)

    res_steady_state = solve_steady_state(m, b_col)

    states = solve_all_states_fast(time, m, initial_conditions, res_steady_state)

    signal = states[:, 1]
    sP1 = states[:, 0]
    sP2 = smax - states[:, 1] - states[:, 0]

    arr = np.column_stack((signal, sP1, sP2))

    return arr

def solve_induced_fit_dissociation(time, koff, kc, krev, s0=0, sP2L=0, smax=0):
    """
    Obtain the dissociation signal for the induced fit model (surface-based).

    We assume that the signal given by the complex E·S and ES is the same.
    In other words, the complex before and after the conformational change produces the same signal.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    koff : float
        Rate constant for E·S -> E + S.
    kc : float
        Rate constant for E·S -> ES.
    krev : float
        Rate constant for ES -> E·S.
    s0 : float, optional
        Initial signal, default is 0.
    sP2L : float, optional
        Initial concentration of ES, default is 0.
    smax : float, optional
        Signal proportional to the complex (E·S or ES), default is 0.

    Returns
    -------
    np.ndarray
        Signal over time, with columns for total signal, signal produced by E·S, and signal produced by ES.
    """
    time = time - np.min(time) # Start at zero

    b_col = constant_vector_induced_fit(koff, smax, kc)
    m     = differential_matrix_dissociation_induced_fit(koff, kc, krev)

    res_steady_state = solve_steady_state(m, b_col)

    initial_conditions = np.array([smax-s0, sP2L])

    states = solve_all_states_fast(time, m, initial_conditions, res_steady_state)

    # Create a numpy array of three columns
    col1 = smax - states[:, 0]  # Total signal minus the signal produced by E·S and ES
    col2 = smax - states[:, 0] - states[:, 1]  # Signal produced by E·S
    col3 = states[:, 1]  # Signal produced by ES

    arr = np.column_stack((col1, col2, col3))

    return arr

def solve_conformational_selection_dissociation(time, koff, kc, krev, smax=0, sP1=0, sP2L=0):
    """
    Obtain the signal for the conformational selection model (surface-based) during dissociation.

    We assume that the signal is given by the complex E2S only.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    koff : float
        Rate constant for E2S -> E2 + S.
    kc : float
        Rate constant for E1 -> E2.
    krev : float
        Rate constant for E2 -> E1.
    smax : float, optional
        Signal proportional to the complex (E2S), default is 0.
    sP1 : float, optional
        Initial concentration of E1, default is 0.
    sP2L : float, optional
        Initial concentration of E2S, default is 0.

    Returns
    -------
    np.ndarray
        Array with columns for signal (produced by E2S), sP1 (E1 concentration), and sP2 (E2 concentration).
    """

    time = time - np.min(time) # Start at zero

    b_col   = constant_vector_conformational_selection(0, smax, krev, 0)

    initial_conditions = np.array([sP1, sP2L])

    m = differential_matrix_dissociation_conformational_selection(koff, kc, krev)

    res_steady_state = solve_steady_state(m, b_col)

    states = solve_all_states_fast(time, m, initial_conditions, res_steady_state)

    # Create a numpy array of three columns
    signal = states[:, 1]  # Signal produced by E2S
    sP1 = states[:, 0]  # Signal proportional to the protein in state E1
    sP2 = smax - states[:, 1] - states[:, 0]  # Signal proportional to the protein in state E2S

    arr = np.column_stack((signal, sP1, sP2))

    return arr

def ode_mixture_analyte_association(t, Ris, C_TOT, Fis, Ris_max, koffs, Kds):
    """
    We assume a Langmuir 1:1 interaction between each analyte (Ai) and the ligand (L):

    Ai + L <-> AiL, ∀i in [1,N]

    Based on https://doi.org/10.1038/s41598-022-18450-y

    This model is equivalent to the heterogenous analyte model

    The parameters to fit are the factor weights (Fis), the maximum response of each analyte (Ris_max),
    the dissociation rate constants (koffs), and the equilibrium dissociation constants (Kds)

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    Ris : list
        Response of each analyte.
    C_TOT : float
        Total concentration of the analyte mixture.
    Fis : np.ndarray
        Factor weights.
    Ris_max : np.ndarray
        Maximum response of each analyte.
    koffs : np.ndarray
        Dissociation rate constants.
    Kds : np.ndarray
        Equilibrium dissociation constants.

    Returns
    -------
    np.ndarray
        Rate of change of the response of each analyte.
    """

    kons = koffs / Kds

    choclo = (1 - np.sum(Ris / Ris_max))

    dRis = kons * Fis * C_TOT * Ris_max * choclo - koffs * Ris

    return dRis

def solve_ode_mixture_analyte_association(t, Ris0, C_TOT, Fis, Ris_max, koffs, Kds, t0=0):
    """
    Solves the ODE for the mixture analyte model - association phase

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    Ris0 : list
        Initial response of each analyte.
    C_TOT : float
        Total concentration of the analyte mixture.
    Fis : np.ndarray
        Factor weights.
    Ris_max : np.ndarray
        Maximum response of each analyte.
    koffs : np.ndarray
        Dissociation rate constants.
    Kds : np.ndarray
        Equilibrium dissociation constants.
    t0 : float, optional
        Initial time offset, default is 0.

    Returns
    -------
    np.ndarray
        Response of EACH ANALYTE over time.
    """

    t = t + t0

    out = solve_ivp(ode_mixture_analyte_association, t_span=[np.min(t), np.max(t)],
                    t_eval=t, y0=Ris0, args=(C_TOT, Fis, Ris_max, koffs, Kds), method="LSODA")

    return out.y

def ode_mixture_analyte_dissociation(t, Ris, koffs):
    """
    We assume a Langmuir 1:1 interaction between each analyte (Ai) and the ligand (L):

    Ai + L <-> AiL, ∀i in [1,N]

    Based on https://doi.org/10.1038/s41598-022-18450-y

    This model is equivalent to the heterogenous analyte model

    The parameters to fit are the dissociation rate constants (koffs)

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    Ris : list
        Response of each analyte.
    koffs : np.ndarray
        Dissociation rate constants.

    Returns
    -------
    np.ndarray
        Rate of change of the response of each analyte.
    """

    dRis_dt = - koffs * Ris

    return dRis_dt

def solve_ode_mixture_analyte_dissociation(t, Ris0, koffs, t0=0):
    """
    Solves the ODE for the mixture analyte model - dissociation phase

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    Ris0 : list
        Initial response of each analyte.
    koffs : np.ndarray
        Dissociation rate constants.
    t0 : float, optional
        Initial time offset, default is 0.

    Returns
    -------
    np.ndarray
        Response of EACH ANALYTE over time.
    """

    t = t + t0

    out = solve_ivp(ode_mixture_analyte_dissociation, t_span=[np.min(t), np.max(t)],
                    t_eval=t, y0=Ris0, args=(koffs,), method="LSODA")

    return out.y