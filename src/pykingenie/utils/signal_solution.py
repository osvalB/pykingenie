import  numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    'ode_one_site_insolution',
    'solve_ode_one_site_insolution',
    'signal_ode_one_site_insolution',
    'ode_induced_fit_insolution',
    'solve_ode_induced_fit_insolution',
    'signal_ode_induced_fit_insolution',
    'ode_conformational_selection_insolution',
    'solve_ode_conformational_selection_insolution',
    'signal_ode_conformational_selection_insolution',
    'get_initial_concentration_conformational_selection',
    'get_kobs_induced_fit',
    'get_kobs_conformational_selection'
]

def ode_one_site_insolution(t,complex_conc,koff,Kd,a_total,b_total):

    """

    ODE for the one binding site model in solution

    Args:
        t (float): time
        complex_conc (float): concentration of the complex
        koff (float): off rate constant
        Kd (float): equilibrium dissociation constant
        a_total (float): total concentration of the ligand
        b_total (float): total concentration of the receptor

    Returns:

        dab_dt (float): complex concentration

    """

    kon = koff / Kd

    a = a_total - complex_conc
    b = b_total - complex_conc

    dab_dt = kon * a * b - koff * complex_conc

    return  dab_dt

def solve_ode_one_site_insolution(t,koff,Kd,a_total,b_total,t0=0):

    """

    Solve the ODE for the one binding site model in solution
    Assumes that the initial concentration of the complex is zero

    Args:
        t (np.ndarray): time points to calculate the complex concentration
        koff (float): off rate constant
        Kd (float): equilibrium dissociation constant
        a_total (float): total concentration of the ligand
        b_total (float): total concentration of the receptor
        t0 (float): initial time

    Returns:

        out.y[0] (np.ndarray): complex concentration over time

    """

    t = t + t0

    out = solve_ivp(ode_one_site_insolution,t_span=[np.min(t), np.max(t)],
                    t_eval=t,y0=[0],args=(koff,Kd,a_total,b_total),method="LSODA")

    return out.y[0]

def signal_ode_one_site_insolution(t,koff,Kd,a_total,b_total,t0=0,signal_a=0,signal_b=0,signal_complex=0):

    """

    Solve the ODE for the one binding site model and compute the signal

    Args:
        t (np.ndarray): time
        koff (float): off rate constant
        Kd (float): equilibrium dissociation constant
        a_total (float): total concentration of the ligand
        b_total (float): total concentration of the receptor
        t0 (float): initial time
        signal_a (float): signal for the ligand
        signal_b (float): signal for the receptor
        signal_complex (float): signal for the complex

    Returns:

        signal (np.ndarray): signal over time

    """

    complex_conc = solve_ode_one_site_insolution(t,koff,Kd,a_total,b_total,t0)

    signal = signal_a * (a_total - complex_conc) + signal_b * (b_total - complex_conc) + signal_complex * complex_conc

    return signal

def ode_induced_fit_insolution(t, y, k1, k_minus1, k2, k_minus2):

    """

    Species:

        E: Free enzyme
        S: Free substrate
        E·S: Intermediate complex
        ES: Induced Enzyme-substrate complex

    ODE for the induced fit model:

        E + S <-> E·S (initial binding)
        E·S   <-> ES (induced fit)

    Args:

        t (float): time
        y (list): concentrations of E, S, E·S, and ES
        k1 (float): rate constant for E + S -> E·S
        k_minus1 (float): rate constant for E·S -> E + S
        k2 (float): rate constant for E·S -> ES
        k_minus2 (float): rate constant for ES -> E·S

    Returns:

        [dE, dS, dE_S, dES] (list): concentration of E, S, E·S, and ES

    Example values taken from Fabian Paul ,Thomas R. Weikl, 2016:

        k+ = 100 / (μM*s)  k+ equals k1       in our notation
        k− = 100 / s       k- equals k_minus1 in our notation
        ke = 10 / s        ke equals k2       in our notation
        kr = 1 /s          kr equals k_minus2 in our notation

    """

    E, S, E_S, ES = y

    # Differential equations
    dE = -k1 * E * S + k_minus1 * E_S
    dS = -k1 * E * S + k_minus1 * E_S
    dE_S = k1 * E * S - k_minus1 * E_S - k2 * E_S + k_minus2 * ES
    dES = k2 * E_S - k_minus2 * ES

    return [dE, dS, dE_S, dES]

def ode_induced_fit_insolution(t, y, k1, k_minus1, k2, k_minus2, E_tot, S_tot):
    """
    Reduced ODEs for induced fit model using conservation of E and S.
    y: [E_S, ES]
    Args:
        t (float): time
        y (list): concentrations of E·S and ES
        k1 (float): rate constant for E + S -> E·S
        k_minus1 (float): rate constant for E·S -> E + S
        k2 (float): rate constant for E·S -> ES
        k_minus2 (float): rate constant for ES -> E·S
        E_tot (float): total concentration of the enzyme
        S_tot (float): total concentration of the substrate
    Returns:
        [dE_S, dES] (list): concentration of E·S and ES
    """
    E_S, ES = y
    E = E_tot - E_S - ES
    S = S_tot - E_S - ES

    dE_S = k1 * E * S - k_minus1 * E_S - k2 * E_S + k_minus2 * ES
    dES  = k2 * E_S   - k_minus2 * ES

    return [dE_S, dES]

def solve_ode_induced_fit_insolution(t, y0, k1, k_minus1, k2, k_minus2, E_tot, S_tot, t0=0):
    """
    Solve the reduced ODE for the induced fit model in solution.
    Args:
        t (np.ndarray): time points to calculate the complex concentration
        y0 (list): initial concentrations of E·S and ES
        k1 (float): rate constant for E + S -> E·S
        k_minus1 (float): rate constant for E·S -> E + S
        k2 (float): rate constant for E·S -> ES
        k_minus2 (float): rate constant for ES -> E·S
        E_tot (float): total concentration of the enzyme
        S_tot (float): total concentration of the substrate
        t0 (float): initial time
    Returns:
        out.y (list): solution of the ODE, contains the concentration of E·S and ES
    """
    t = t + t0
    out = solve_ivp(ode_induced_fit_insolution, t_span=[np.min(t), np.max(t)],
                    t_eval=t, y0=y0, args=(k1, k_minus1, k2, k_minus2, E_tot, S_tot), method="LSODA")
    # Include the concentrations of E and S in the output
    E = E_tot - out.y[0] - out.y[1]  # E_S is out.y[0], ES is out.y[1]
    S = S_tot - out.y[0] - out.y[1]  # E_S is out.y[0], ES is out.y[1]
    out.y = np.vstack((E, S, out.y[0], out.y[1]))  # Stack E, S, E_S, ES
    return out.y

def signal_ode_induced_fit_insolution(t, y, k1, k_minus1, k2, k_minus2, E_tot, S_tot,
                                      t0=0, signal_E=0, signal_S=0, signal_ES_int=0, signal_ES=0):
    """
    Solve the reduced ODE for the induced fit model and compute the signal
    Args:
        t (np.ndarray): time
        y (list): concentrations of E·S and ES
        k1 (float): rate constant for E + S -> E·S
        k_minus1 (float): rate constant for E·S -> E + S
        k2 (float): rate constant for E·S -> ES
        k_minus2 (float): rate constant for ES -> E·S
        E_tot (float): total concentration of the enzyme
        S_tot (float): total concentration of the substrate
        t0 (float): initial time
        signal_E (float): signal for E
        signal_S (float): signal for S
        signal_ES_int (float): signal for E·S
        signal_ES (float): signal for ES
    Returns:
       signal (np.ndarray): signal over time
    """
    species = solve_ode_induced_fit_insolution(t, y, k1, k_minus1, k2, k_minus2, E_tot, S_tot, t0)
    signal = signal_E * species[0] + signal_S * species[1] + signal_ES_int * species[2] + signal_ES * species[3]
    return signal

def ode_conformational_selection_insolution(t, y, k1, k_minus1, k2, k_minus2):

    """

    Species:

        E1: Free enzyme in state 1
        E2: Free enzyme in state 2
        S: Free substrate
        E2S:  Complex

    ODE for the induced fit model:

        E1     <-> E2  (conformational change)
        E2 + S <-> E2S (binding)

    Args:

        t (float): time
        y (list): concentrations of E1, E2, S, and E2S
        k1 (float): rate constant for E1 -> E2
        k_minus1 (float): rate constant for E2 -> E1
        k2 (float): rate constant for E2 + S -> E2S
        k_minus2 (float): rate constant for E2S -> E2 + S

    Returns:

        [dE1, dE2, dS, dE2S] (list): concentration of E1, E2, S, E2S

    Example values taken from Fabian Paul ,Thomas R. Weikl, 2016:

        k+ = 100 / (μM*s)  k+ equals k2       in our notation
        k− = 1 / s         k- equals k_minus2 in our notation
        ke = 10 / s        ke equals k1       in our notation
        kr = 100 / s       kr equals k_minus1 in our notation

        Protein concentration = 0.5 μM
        Ligand concentration  = 1.8 μM

    """

    E1, E2, S, E2S = y

    # Differential equations
    dE1 = -k1 * E1 + k_minus1 * E2
    dE2 =  k1 * E1 - k_minus1 * E2 - k2 * E2 * S + k_minus2 * E2S
    dS  = -k2 * E2 * S + k_minus2 * E2S
    dE2S = k2 * E2 * S - k_minus2 * E2S

    return [dE1, dE2, dS, dE2S]

def solve_ode_conformational_selection_insolution(t, y, k1, k_minus1, k2, k_minus2,t0=0):

    """

    Solve ODE for the conformational selection model

    Args:

        t (np.ndarray): time points to calculate the complex concentration
        y (list): initial concentrations of E1, E2, S, and E2S
        k1 (float): rate constant for E1 -> E2
        k_minus1 (float): rate constant for E2 -> E1
        k2 (float): rate constant for E2 + S -> E2S
        k_minus2 (float): rate constant for E2S -> E2 + S
        t0 (float): initial time

    Returns:

        out.y (list): solution of the ODE, contains the concentration of E1, E2, S, E2S

    """

    t = t + t0

    out = solve_ivp(ode_conformational_selection_insolution,t_span=[np.min(t), np.max(t)],
                    t_eval=t,y0=y,args=(k1, k_minus1, k2, k_minus2),method="LSODA")

    return out.y

def signal_ode_conformational_selection_insolution(t, y, k1, k_minus1, k2, k_minus2,t0=0,signal_E1=0,signal_E2=0,signal_S=0,signal_E2S=0):

    """

    Solve the ODE for the conformational selection model and compute the signal

    Args:

        t (np.ndarray): time
        y (list): concentrations of E1, E2, S, and E2S
        k1 (float): rate constant for E1 -> E2              aka kc
        k_minus1 (float): rate constant for E2 -> E1        aka krev
        k2 (float): rate constant for E2 + S -> E2S         aka kon
        k_minus2 (float): rate constant for E2S -> E2 + S   aka koff
        t0 (float): initial time
        signal_E1 (float): signal for E1
        signal_E2 (float): signal for E2
        signal_S (float): signal for S
        signal_E2S (float): signal for E2S

    Returns:

        signal (np.ndarray): signal over time

    """

    species = solve_ode_conformational_selection_insolution(t, y, k1, k_minus1, k2, k_minus2,t0)

    signal = signal_E1 * species[0] + signal_E2 * species[1] + signal_S * species[2] + signal_E2S * species[3]

    return signal

def get_initial_concentration_conformational_selection(total_protein,k1,k_minus1):

    """

    Obtain the equilibrium concentrations of E1 and E2 for the conformational selection model
    assuming absence of ligand

    Args:
        total_protein (float): total concentration of the protein
        k1 (float): rate constant for E1 -> E2
        k_minus1 (float): rate constant for E2 -> E1

    Returns:

        [E1, E2] (list): equilibrium concentrations of E1 and E2

    """

    E1 = total_protein / (1 + k1 / k_minus1)
    E2 = total_protein - E1

    return [E1, E2]

def get_kobs_induced_fit(tot_lig, tot_prot, kr, ke, kon, koff,dominant=True):

    """
    Calculate the observed rate constant (relaxation rate) for the induced fit model in solution

    Args:

        tot_lig (float): total concentration of the ligand
        tot_prot (float): total concentration of the protein
        kr (float): rate constant for E2S -> E1S
        ke (float): rate constant for E1S -> E2S
        kon (float): rate constant for E1 + S -> E1S
        koff (float): rate constant for E1S -> E1 + S
        dominant (bool): if True, calculate the dominant relaxation rate, otherwise calculate the non-dominant relaxation rate

    Returns:

        kobs (float): observed rate constant

    """

    kd_app = koff * ke / (kon * (ke + kr))
    delta  = np.sqrt((tot_lig - tot_prot + kd_app) ** 2 + 4 * tot_prot * kd_app)
    gamma  = -ke - kr + koff + kon * (delta - kd_app)

    kobs = ke + kr + 0.5 * gamma + (-1*dominant * 0.5 * np.sqrt(gamma ** 2 + 4 * koff * kr))

    return kobs  # units are 1/s

def get_kobs_conformational_selection(tot_lig, tot_prot, kr, ke, kon, koff,dominant=True):

    """
    Calculate the observed rate constant (relaxation rate) for the conformational selection model in solution

    Args:

        tot_lig (float): total concentration of the ligand
        tot_prot (float): total concentration of the protein
        kr (float): rate constant for E2 -> E1
        ke (float): rate constant for E1 -> E2
        kon (float): rate constant for E2 + S -> E2S
        koff (float): rate constant for E2S -> E2 + S

    Returns:

        kobs (float): observed rate constant

    """

    kd_app = koff * (ke + kr) / (kon * ke)
    delta = np.sqrt((tot_lig - tot_prot + kd_app)**2 + 4 * tot_prot * kd_app)
    beta = 2 * kr * (2 * ke - koff - koff * (delta - tot_lig + tot_prot) / kd_app)
    alpha = kr - ke + koff * ((2 * ke + kr) * delta + kr * (tot_lig - tot_prot - kd_app)) / (2 * ke * kd_app)


    kobs = ke + 0.5 * alpha + (-1*dominant* 0.5 * np.sqrt(alpha**2 + beta))

    return kobs  # units are 1/s