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

def ode_one_site_insolution(t, complex_conc, koff, Kd, a_total, b_total):
    """
    ODE for the one binding site model in solution.

    Parameters
    ----------
    t : float
        Time.
    complex_conc : float
        Concentration of the complex.
    koff : float
        Off rate constant.
    Kd : float
        Equilibrium dissociation constant.
    a_total : float
        Total concentration of the ligand.
    b_total : float
        Total concentration of the receptor.

    Returns
    -------
    float
        Rate of change of complex concentration.
    """
    kon = koff / Kd

    a = a_total - complex_conc
    b = b_total - complex_conc

    dab_dt = kon * a * b - koff * complex_conc

    return  dab_dt

def solve_ode_one_site_insolution(t, koff, Kd, a_total, b_total, t0=0):
    """
    Solve the ODE for the one binding site model in solution.
    
    Assumes that the initial concentration of the complex is zero.

    Parameters
    ----------
    t : np.ndarray
        Time points to calculate the complex concentration.
    koff : float
        Off rate constant.
    Kd : float
        Equilibrium dissociation constant.
    a_total : float
        Total concentration of the ligand.
    b_total : float
        Total concentration of the receptor.
    t0 : float, optional
        Initial time, default is 0.

    Returns
    -------
    np.ndarray
        Complex concentration over time.
    """
    t = t + t0

    out = solve_ivp(ode_one_site_insolution,t_span=[np.min(t), np.max(t)],
                    t_eval=t,y0=[0],args=(koff,Kd,a_total,b_total),method="LSODA")

    return out.y[0]

def signal_ode_one_site_insolution(t, koff, Kd, a_total, b_total, t0=0, signal_a=0, signal_b=0, signal_complex=0):
    """
    Solve the ODE for the one binding site model and compute the signal.

    Parameters
    ----------
    t : np.ndarray
        Time.
    koff : float
        Off rate constant.
    Kd : float
        Equilibrium dissociation constant.
    a_total : float
        Total concentration of the ligand.
    b_total : float
        Total concentration of the receptor.
    t0 : float, optional
        Initial time, default is 0.
    signal_a : float, optional
        Signal for the ligand, default is 0.
    signal_b : float, optional
        Signal for the receptor, default is 0.
    signal_complex : float, optional
        Signal for the complex, default is 0.

    Returns
    -------
    np.ndarray
        Signal over time.
    """
    complex_conc = solve_ode_one_site_insolution(t, koff, Kd, a_total, b_total, t0)

    signal = signal_a * (a_total - complex_conc) + signal_b * (b_total - complex_conc) + signal_complex * complex_conc

    return signal


def ode_induced_fit_insolution(t, y, k1, k_minus1, k2, k_minus2, E_tot, S_tot):
    """
    Reduced ODEs for induced fit model using conservation of E and S.
    
    Parameters
    ----------
    t : float
        Time.
    y : list
        Concentrations of E·S and ES.
    k1 : float
        Rate constant for E + S -> E·S.
    k_minus1 : float
        Rate constant for E·S -> E + S.
    k2 : float
        Rate constant for E·S -> ES.
    k_minus2 : float
        Rate constant for ES -> E·S.
    E_tot : float
        Total concentration of the enzyme.
    S_tot : float
        Total concentration of the substrate.
        
    Returns
    -------
    list
        [dE_S, dES] - Rate of change for concentrations of E·S and ES.
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
    
    Parameters
    ----------
    t : np.ndarray
        Time points to calculate the complex concentration.
    y0 : list
        Initial concentrations of E·S and ES.
    k1 : float
        Rate constant for E + S -> E·S.
    k_minus1 : float
        Rate constant for E·S -> E + S.
    k2 : float
        Rate constant for E·S -> ES.
    k_minus2 : float
        Rate constant for ES -> E·S.
    E_tot : float
        Total concentration of the enzyme.
    S_tot : float
        Total concentration of the substrate.
    t0 : float, optional
        Initial time, default is 0.
        
    Returns
    -------
    np.ndarray
        Solution of the ODE, contains the concentration of E, S, E·S and ES.
    """
    t = t + t0
    out = solve_ivp(ode_induced_fit_insolution, t_span=[np.min(t), np.max(t)],
                    t_eval=t, y0=y0, args=(k1, k_minus1, k2, k_minus2, E_tot, S_tot), method="LSODA")
    # Include the concentrations of E and S in the output
    E = E_tot - out.y[0] - out.y[1]  # E_S is out.y[0], ES is out.y[1]
    S = S_tot - out.y[0] - out.y[1]  # E_S is out.y[0], ES is out.y[1]
    out = np.vstack((E, S, out.y[0], out.y[1]))  # Stack E, S, E_S, ES

    return out

def signal_ode_induced_fit_insolution(t, y, k1, k_minus1, k2, k_minus2, E_tot, S_tot,
                                      t0=0, signal_E=0, signal_S=0, signal_ES_int=0, signal_ES=0):
    """
    Solve the reduced ODE for the induced fit model and compute the signal.
    
    Parameters
    ----------
    t : np.ndarray
        Time.
    y : list
        Concentrations of E·S and ES.
    k1 : float
        Rate constant for E + S -> E·S.
    k_minus1 : float
        Rate constant for E·S -> E + S.
    k2 : float
        Rate constant for E·S -> ES.
    k_minus2 : float
        Rate constant for ES -> E·S.
    E_tot : float
        Total concentration of the enzyme.
    S_tot : float
        Total concentration of the substrate.
    t0 : float, optional
        Initial time, default is 0.
    signal_E : float, optional
        Signal for E, default is 0.
    signal_S : float, optional
        Signal for S, default is 0.
    signal_ES_int : float, optional
        Signal for E·S, default is 0.
    signal_ES : float, optional
        Signal for ES, default is 0.
        
    Returns
    -------
    np.ndarray
        Signal over time.
    """
    species = solve_ode_induced_fit_insolution(t, y, k1, k_minus1, k2, k_minus2, E_tot, S_tot, t0)
    signal = signal_E * species[0] + signal_S * species[1] + signal_ES_int * species[2] + signal_ES * species[3]
    return signal


def ode_conformational_selection_insolution(t, y, k1, k_minus1, k2, k_minus2, E_tot, S_tot):

    """
    Reduced ODEs for conformational selection using conservation:
      E_tot = E1 + E2 + E2S
      S_tot = S + E2S

    State vector y: [E2, E2S]

    Returns: [dE2_dt, dE2S_dt]
    """

    E2, E2S = y
    E1 = E_tot - E2 - E2S
    S  = S_tot - E2S

    dE2  = k1 * E1 - k_minus1 * E2 - k2 * E2 * S + k_minus2 * E2S
    dE2S = k2 * E2 * S - k_minus2 * E2S

    return [dE2, dE2S]

def solve_ode_conformational_selection_insolution(t, y, k1, k_minus1, k2, k_minus2, E_tot,S_tot,t0=0):
    """
    Solve ODE for the conformational selection model.

    Parameters
    ----------
    t : np.ndarray
        Time points to calculate the complex concentration.
    y : list
        Initial concentrations of E2 and E2S.
    k1 : float
        Rate constant for E1 -> E2.
    k_minus1 : float
        Rate constant for E2 -> E1.
    k2 : float
        Rate constant for E2 + S -> E2S.
    k_minus2 : float
        Rate constant for E2S -> E2 + S.
    E_tot : float
        total protein concentration
    S_tot : float
        total substrate concentration
    t0 : float, optional
        Initial time, default is 0.

    Returns
    -------
    np.ndarray
        Solution of the ODE, contains the concentration of E1, E2, S, E2S.
    """
    t = t + t0

    out = solve_ivp(ode_conformational_selection_insolution,
                    t_span=[np.min(t), np.max(t)],
                    t_eval=t, y0=y, args=(k1, k_minus1, k2, k_minus2,E_tot,S_tot), method="LSODA")

    # out includes [dE2, dE2S], we need to add E1 and S
    # E1 in the first column, S in the third column

    E1 = E_tot - out.y[0] - out.y[1]
    S  = S_tot - out.y[1]

    out = np.vstack((E1, out.y[0], S, out.y[1]))

    return out

def signal_ode_conformational_selection_insolution(
        t, y,
        k1, k_minus1,
        k2, k_minus2,
        E_tot, S_tot,
        t0=0, signal_E1=0, signal_E2=0, signal_S=0, signal_E2S=0):
    """
    Solve the ODE for the conformational selection model and compute the signal.
    
    Parameters
    ----------
    t : np.ndarray
        Time.
    y : list
        Concentrations of E2 and E2S.
    k1 : float
        Rate constant for E1 -> E2 (aka kc).
    k_minus1 : float
        Rate constant for E2 -> E1 (aka krev).
    k2 : float
        Rate constant for E2 + S -> E2S (aka kon).
    k_minus2 : float
        Rate constant for E2S -> E2 + S (aka koff).
    E_tot : float
        total protein concentration
    S_tot : float
        total substrate concentration
    t0 : float, optional
        Initial time, default is 0.
    signal_E1 : float, optional
        Signal for E1, default is 0.
    signal_E2 : float, optional
        Signal for E2, default is 0.
    signal_S : float, optional
        Signal for S, default is 0.
    signal_E2S : float, optional
        Signal for E2S, default is 0.
        
    Returns
    -------
    np.ndarray
        Signal over time.
    """
    species = solve_ode_conformational_selection_insolution(t, y, k1, k_minus1, k2, k_minus2, E_tot, S_tot, t0)

    signal = signal_E1 * species[0] + signal_E2 * species[1] + signal_S * species[2] + signal_E2S * species[3]

    return signal

def get_initial_concentration_conformational_selection(total_protein, k1, k_minus1):
    """
    Obtain the equilibrium concentrations of E1 and E2 for the conformational selection model.
    
    Assumes absence of ligand.
    
    Parameters
    ----------
    total_protein : float
        Total concentration of the protein.
    k1 : float
        Rate constant for E1 -> E2.
    k_minus1 : float
        Rate constant for E2 -> E1.
        
    Returns
    -------
    list
        [E1, E2] - Equilibrium concentrations of E1 and E2.
    """
    E1 = total_protein / (1 + k1 / k_minus1)
    E2 = total_protein - E1

    return [E1, E2]

def get_kobs_induced_fit(tot_lig, tot_prot, kr, ke, kon, koff, dominant=True):
    """
    Calculate the observed rate constant (relaxation rate) for the induced fit model in solution.
    
    Parameters
    ----------
    tot_lig : float
        Total concentration of the ligand.
    tot_prot : float
        Total concentration of the protein.
    kr : float
        Rate constant for E2S -> E1S.
    ke : float
        Rate constant for E1S -> E2S.
    kon : float
        Rate constant for E1 + S -> E1S.
    koff : float
        Rate constant for E1S -> E1 + S.
    dominant : bool, optional
        If True, calculate the dominant relaxation rate, otherwise calculate the non-dominant relaxation rate, default is True.
        
    Returns
    -------
    float
        Observed rate constant (in 1/s).
    """
    kd_app = koff * ke / (kon * (ke + kr))
    delta  = np.sqrt((tot_lig - tot_prot + kd_app) ** 2 + 4 * tot_prot * kd_app)
    gamma  = -ke - kr + koff + kon * (delta - kd_app)

    kobs = ke + kr + 0.5 * gamma + (-1*dominant * 0.5 * np.sqrt(gamma ** 2 + 4 * koff * kr))

    return kobs  # units are 1/s

def get_kobs_conformational_selection(tot_lig, tot_prot, kr, ke, kon, koff, dominant=True):
    """
    Calculate the observed rate constant (relaxation rate) for the conformational selection model in solution.
    
    Parameters
    ----------
    tot_lig : float
        Total concentration of the ligand.
    tot_prot : float
        Total concentration of the protein.
    kr : float
        Rate constant for E2 -> E1.
    ke : float
        Rate constant for E1 -> E2.
    kon : float
        Rate constant for E2 + S -> E2S.
    koff : float
        Rate constant for E2S -> E2 + S.
    dominant : bool, optional
        If True, calculate the dominant relaxation rate, otherwise calculate the non-dominant relaxation rate, default is True.
        
    Returns
    -------
    float
        Observed rate constant (in 1/s).
    """
    kd_app = koff * (ke + kr) / (kon * ke)
    delta = np.sqrt((tot_lig - tot_prot + kd_app)**2 + 4 * tot_prot * kd_app)
    beta = 2 * kr * (2 * ke - koff - koff * (delta - tot_lig + tot_prot) / kd_app)
    alpha = kr - ke + koff * ((2 * ke + kr) * delta + kr * (tot_lig - tot_prot - kd_app)) / (2 * ke * kd_app)


    kobs = ke + 0.5 * alpha + (-1*dominant* 0.5 * np.sqrt(alpha**2 + beta))

    return kobs  # units are 1/s