"""Calculate surface-based kinetic signals and numerical trajectories."""

import numpy         as np
import pandas        as pd

from scipy.integrate import solve_ivp, odeint
from scipy.linalg    import solve
from scipy.linalg    import expm

__all__ = [
    'steady_state_one_site',
    'one_site_association_analytical',
    'one_site_dissociation_analytical',
    'ode_one_site_mass_transport',
    'solve_ode_one_site_mass_transport_association',
    'solve_ode_one_site_mass_transport_dissociation',
    'one_site_mass_transport_analytical',
    'one_site_mass_transport_association_analytical',
    'one_site_mass_transport_dissociation_analytical',
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
    'solve_ode_mixture_analyte_dissociation',
    'steady_state_two_site_heterogeneous_ligand',
    'solve_two_site_heterogeneous_ligand_association',
    'solve_two_site_heterogeneous_ligand_dissociation',
    'steady_state_two_site',
    'steady_state_two_site_cooperative',
    'differential_matrix_association_two_site',
    'differential_matrix_association_two_site_cooperative',
    'constant_vector_two_site',
    'differential_matrix_dissociation_two_site',
    'differential_matrix_dissociation_two_site_cooperative',
    'solve_two_site_association',
    'solve_two_site_cooperative_association',
    'solve_two_site_dissociation',
    'solve_two_site_cooperative_dissociation',
    'ode_bivalent_analyte',
    'solve_bivalent_analyte_association',
    'solve_bivalent_analyte_dissociation'
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

def ode_one_site_mass_transport(t, s1, params, analyte_conc):
    """
    Quasi-steady-state ODE for mass transport-limited one-site binding.

    The thin layer next to the surface is assumed to settle instantly (Svitel et al. 2007,
    Eq. 8, https://doi.org/10.1529/biophysj.106.094615), which gives

        dS/dt = (k_on * c0 * (s_max - S) - k_off * S) / (1 + k_on * (s_max - S) / k_tr)

    Used for both phases: association with the injected concentration, dissociation with c0 = 0.

    ``k_tr`` is a lumped transport coefficient in signal units (signal per µM per second),
    the product of the physical transport rate and the signal produced per unit of bound analyte.
    It is not the physical transport coefficient, and it scales with the signal units.

    Parameters
    ----------
    t : float
        Time variable.
    s1 : float or list
        Signal.
    params : list
        List of the parameters [K_d, k_off, k_tr, s_max].
    analyte_conc : float
        Concentration of the analyte in the bulk, 0 during dissociation.

    Returns
    -------
    float
        The derivative of the signal over time.
    """
    K_d, k_off, k_tr, s_max = params

    k_on = k_off / K_d
    # Free sites cannot be negative. Without this, a solver step past s_max can land where the
    # denominator is negative and the signal grows without bound.
    free = np.maximum(s_max - s1, 0.0)

    return (k_on * analyte_conc * free - k_off * s1) / (1 + k_on * free / k_tr)

def solve_ode_one_site_mass_transport_association(t, s1_0, analyte_conc, K_d, k_off, k_tr, s_max, t0=0):
    """
    Solves the ODE for the one site mass transport model - association phase.

    See ``ode_one_site_mass_transport``.

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    s1_0 : float
        Initial signal.
    analyte_conc : float
        Concentration of the analyte.
    K_d : float
        Equilibrium dissociation constant.
    k_off : float
        Dissociation rate constant.
    k_tr : float
        Lumped mass transport coefficient (signal per µM per second).
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

    out = solve_ivp(ode_one_site_mass_transport,t_span=[np.min(t), np.max(t)],
                    t_eval=t,y0=[s1_0],args=([K_d,k_off,k_tr,s_max],analyte_conc),method="LSODA",
                    rtol=1e-8,atol=1e-10*s_max)

    return out.y[0]

def solve_ode_one_site_mass_transport_dissociation(t, s1_0, K_d, k_off, k_tr, s_max, t0=0):
    """
    Solves the ODE for the one site mass transport model - dissociation phase.

    Same model as the association, with no analyte in the bulk. See ``ode_one_site_mass_transport``.

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
        Lumped mass transport coefficient (signal per µM per second).
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

    out = solve_ivp(ode_one_site_mass_transport,t_span=[np.min(t), np.max(t)],
                    t_eval=t,y0=[s1_0],args=([K_d,k_off,k_tr,s_max],0.0),method="LSODA",
                    rtol=1e-8,atol=1e-10*s_max)

    return out.y[0]

def _lambertw_log_form(L, y0, n_halley=4):
    """
    Solve ``ln|y| + y = L`` on the branch through ``y0``.

    This is the principal branch of the Lambert W function, y = W0(sign(y0) * exp(L)),
    written so that exp(L) is never formed when it would overflow. Forward in time
    (L below its value at y0) y moves from y0 towards 0. Backward in time it moves away
    from 0; for y0 < 0 it stops at -1, the end of the principal branch.

    The equation is solved for u = ln|y| with Halley steps from a series start, in
    real arithmetic (scipy.special.lambertw works in complex numbers and was most of
    the cost of the mass transport fit). For y > 0, u + exp(u) = L is increasing and
    convex for every L. For -1 <= y < 0, u - exp(u) = L starts from the expansion
    around the branch point y = -1 when -exp(L) is near -1/e, otherwise from the
    series of W around 0.

    Parameters
    ----------
    L : np.ndarray
        Right-hand side, ``ln|y0| + y0`` minus the scaled elapsed time.
    y0 : float
        Value at time zero, not 0. Sets the sign of the solution.
    n_halley : int, optional
        Halley steps from the series start, default is 4 (3 already reach machine
        precision against 50-digit references).

    Returns
    -------
    np.ndarray
        The solution y, with the same shape as L.
    """
    # This function is AI generated.
    L = np.asarray(L, dtype=float)
    backward = L > np.log(abs(y0)) + y0

    # Bounds per point: between y0 and 0 forward in time, beyond y0 backward
    if y0 > 0:
        lo = np.where(backward, y0, 0.0)
        hi = np.where(backward, np.inf, y0)
    else:
        lo = np.where(backward, -1.0, y0)
        hi = np.where(backward, y0, 0.0)

    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        if y0 > 0:
            # Start: u = L - exp(L) where y is small, u = ln(L - ln L) where it is large
            big = L > 1.0
            L_big = np.where(big, L, 2.0)
            u = np.where(big, np.log(L_big - np.log(L_big)), L - np.exp(np.minimum(L, 1.0)))
            for _ in range(n_halley):
                e = np.exp(u)
                h, dh = u + e - L, 1.0 + e
                u = u - 2 * h * dh / (2 * dh * dh - h * e)
            y = np.exp(u)
        else:
            # z = -exp(L) >= -1/e, except backward past the branch point, where y is clipped to -1
            z = -np.exp(np.minimum(L, -1.0))
            p = np.sqrt(np.maximum(2.0 * (1.0 + np.e * z), 0.0))
            y_branch = -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p ** 3
            y_zero = z * (1.0 - z * (1.0 - z * (1.5 - z * 8.0 / 3.0)))
            y = np.where(z < -0.25, y_branch, y_zero)
            # Where exp(L) underflows, u = ln(-y) is L itself
            u = np.where(L < -700, L, np.log(-np.clip(y, -1.0, -1e-320)))
            for _ in range(n_halley):
                e = np.exp(u)
                h, dh = u - e - L, 1.0 - e
                step = 2 * h * dh / (2 * dh * dh + h * e)
                u = np.minimum(u - np.where(np.isfinite(step), step, 0.0), 0.0)
            y = -np.exp(u)

        y = np.clip(y, lo, hi)

    return y

def one_site_mass_transport_analytical(t, s1_0, analyte_conc, K_d, k_off, k_tr, s_max, t0=0):
    """
    Closed-form solution of the mass transport-limited one-site model.

    Solves the same equation as ``ode_one_site_mass_transport``

        dS/dt = (k_on * c0 * (s_max - S) - k_off * S) / (1 + k_on * (s_max - S) / k_tr)

    without numerical integration. With free sites F = s_max - S, k = k_on / k_tr,
    k_s = k_on * c0 + k_off, F_eq = k_off * s_max / k_s and B = 1 + k * F_eq, the
    variables separate and y = k * (F - F_eq) / B satisfies

        y * exp(y) = y0 * exp(y0) * exp(-k_s * t / B)

    so y(t) = W0(y0 * exp(y0 - k_s * t / B)), with W0 the principal branch of the
    Lambert W function. The principal branch always applies, because F >= 0 gives y > -1.

    The signal is ``s1_0`` at time ``min(t) + t0``, as in ``one_site_association_analytical``.
    Points before that are the same solution followed backward in time. With t0 = 0 this
    matches ``solve_ode_one_site_mass_transport_association``; in the ODE solvers t0 has no
    effect, because the integration always starts at the first time point.

    Backward in time from above equilibrium the surface can fill (S = s_max). Before that
    point the signal follows dS/dt = -k_off * S, as in the ODE, which clips free sites at 0.

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    s1_0 : float
        Signal at time ``min(t) + t0``. Values above s_max are treated as s_max.
    analyte_conc : float
        Concentration of the analyte in the bulk, 0 during dissociation.
    K_d : float
        Equilibrium dissociation constant.
    k_off : float
        Dissociation rate constant.
    k_tr : float
        Lumped mass transport coefficient (signal per µM per second). np.inf gives the
        1:1 model without transport limitation.
    s_max : float
        Maximum signal.
    t0 : float, optional
        Time after the first time point at which the signal is s1_0, default is 0.

    Returns
    -------
    np.ndarray
        Signal over time.
    """
    # This function is AI generated.
    t = np.asarray(t, dtype=float)
    elapsed = t - np.min(t) - t0

    k_on = k_off / K_d
    k    = k_on / k_tr
    k_s  = k_on * analyte_conc + k_off

    f_eq = k_off * s_max / k_s
    s0   = min(s1_0, s_max)
    u0   = s_max - s0 - f_eq

    if u0 == 0:
        return np.full_like(elapsed, s_max - f_eq)

    if k == 0:
        # No transport limitation: the 1:1 exponential
        s = s0 - u0 * np.expm1(-k_s * elapsed)
        # Time at which the surface is full (F = 0), reached only backward from above equilibrium
        t_full = np.log(-u0 / f_eq) / k_s if u0 < 0 else -np.inf
    else:
        B  = 1 + k * f_eq
        y0 = k * u0 / B

        L0 = np.log(abs(y0)) + y0
        y  = _lambertw_log_form(L0 - k_s * elapsed / B, y0)
        y[elapsed == 0] = y0
        # Written relative to s0 so the first point is exact
        s  = s0 - (B / k) * (y - y0)

        if u0 < 0:
            y_full = -k * f_eq / B
            t_full = (L0 - np.log(-y_full) - y_full) * B / k_s
        else:
            t_full = -np.inf

    # Before the surface fills there are no free sites. Like the ODE, which clips free sites
    # at zero, the signal then follows dS/dt = -k_off * S.
    full = elapsed < t_full
    s[full] = s_max * np.exp(-k_off * (elapsed[full] - t_full))

    return s

def one_site_mass_transport_association_analytical(t, s1_0, analyte_conc, K_d, k_off, k_tr, s_max, t0=0):
    """
    Closed-form mass transport model - association phase.

    Drop-in replacement for ``solve_ode_one_site_mass_transport_association``.
    See ``one_site_mass_transport_analytical``.

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    s1_0 : float
        Initial signal.
    analyte_conc : float
        Concentration of the analyte.
    K_d : float
        Equilibrium dissociation constant.
    k_off : float
        Dissociation rate constant.
    k_tr : float
        Lumped mass transport coefficient (signal per µM per second).
    s_max : float
        Maximum signal.
    t0 : float, optional
        Time after the first time point at which the signal is s1_0, default is 0.

    Returns
    -------
    np.ndarray
        Signal over time.
    """
    return one_site_mass_transport_analytical(t, s1_0, analyte_conc, K_d, k_off, k_tr, s_max, t0)

def one_site_mass_transport_dissociation_analytical(t, s1_0, K_d, k_off, k_tr, s_max, t0=0):
    """
    Closed-form mass transport model - dissociation phase.

    Drop-in replacement for ``solve_ode_one_site_mass_transport_dissociation``.
    See ``one_site_mass_transport_analytical``.

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
        Lumped mass transport coefficient (signal per µM per second).
    s_max : float
        Maximum signal.
    t0 : float, optional
        Time after the first time point at which the signal is s1_0, default is 0.

    Returns
    -------
    np.ndarray
        Signal over time.
    """
    return one_site_mass_transport_analytical(t, s1_0, 0.0, K_d, k_off, k_tr, s_max, t0)

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
        Rate constant for E1S -> E1 + S.
    kon : float
        Rate constant for E1 + S -> E1S.
    kc : float
        Rate constant for E1S -> E2S.
    krev : float
        Rate constant for E2S -> E1S.
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
        Rate constant for E1S -> E1 + S.
    kc : float
        Rate constant for E1S -> E2S.
    krev : float
        Rate constant for E2S -> E1S.

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
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.shape == (2, 2):
        # Closed-form inverse: much faster than a general solver for 2x2 systems,
        # which every surface model here uses (AI generated).
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if det != 0:
            return np.array([-(A[1, 1] * b[0] - A[0, 1] * b[1]) / det,
                             -(A[0, 0] * b[1] - A[1, 0] * b[0]) / det])

    return -solve(A, b)

def solve_induced_fit_association(time, a_conc, kon, koff, kc, krev, sP1L=0, sP2L=0, smax=0, t0=0,
                                  signal_ratio=1.0):
    """
    Obtain the signal for the induced fit model (surface-based).

    By default the complex before (E·S) and after (ES) the conformational
    change produce the same signal. Set ``signal_ratio`` to let the induced
    complex ES produce a different signal: the total signal is then
    R(E·S) + signal_ratio·R(ES), where R is the response of E·S.

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
        Initial signal proportional to the concentration of E·S, default is 0.
    sP2L : float, optional
        Initial signal proportional to the concentration of ES, default is 0.
    smax : float, optional
        Signal of the surface fully occupied by the intermediate complex
        (E·S), default is 0.
    t0 : float, optional
        Time offset; the association starts at time t0 after the first
        time point, default is 0.
    signal_ratio : float, optional
        Signal of the induced complex (ES) relative to the intermediate
        complex (E·S), default is 1 (same signal).

    Returns
    -------
    np.ndarray
        Array with three columns: the total signal, the signal of E·S, and
        the amount of ES expressed in E·S signal units (ES contributes
        signal_ratio times this column to the total signal).
    """
    # Edited with AI assistance and reviewed by a human in PR #16: added t0 so a fitted t0 shifts the association.
    time = time - np.min(time) - t0

    b_col = constant_vector_induced_fit(koff, smax, kc)

    # Initial conditions for St - R(E·S) - R(ES), and R(ES)
    # where St is the max signal, R(E·S) is the signal produced by E·S, and R(ES) is the signal produced by ES

    initial_conditions = np.array([smax-sP1L-sP2L, sP2L])

    m = differential_matrix_association_induced_fit(koff, kon, kc, krev, a_conc)

    res_steady_state = solve_steady_state(m, b_col)

    states = solve_all_states_fast(time, m, initial_conditions, res_steady_state)

    return _induced_fit_signal_columns(states, smax, signal_ratio)

def _induced_fit_signal_columns(states, smax, signal_ratio):
    """
    Convert the induced fit states into signal columns.

    Parameters
    ----------
    states : np.ndarray
        States of the system, with columns smax - R([E·S] + [ES]) and R[ES].
    smax : float
        Signal of the surface fully occupied by E·S.
    signal_ratio : float
        Signal of ES relative to E·S.

    Returns
    -------
    np.ndarray
        Columns: total signal, signal of E·S, and amount of ES in E·S signal units.
    """
    # This function is AI generated (induced complex signal ratio).
    # Edited with AI assistance: filled in place instead of np.column_stack
    columns = np.empty((states.shape[0], 3))
    np.subtract(smax - states[:, 0], states[:, 1], out=columns[:, 1])  # Signal produced by E·S
    columns[:, 2] = states[:, 1]                                       # ES, in E·S signal units
    columns[:, 0] = columns[:, 1] + signal_ratio * columns[:, 2]       # Total signal

    return columns

def solve_all_states_fast(time, A, initial_conditions, steady_state_value):
    """
    Solve the linear system dx/dt = A·x + b for all time points.

    The solution is

        x(t) = x_ss + exp(A·t) · (x₀ − x_ss)

    where x_ss is the steady state. For a 2×2 matrix, the case of every
    surface model here, exp(A·t) is evaluated in closed form (Putzer):

        exp(A·t) = e^(λ1·t)·I + f(t)·(A − λ1·I),
        f(t) = (e^(λ1·t) − e^(λ2·t)) / (λ1 − λ2)

    which also holds when the eigenvalues are equal and A cannot be
    diagonalised (f(t) then tends to t·e^(λ1·t)). Larger matrices use the
    eigendecomposition of A. All time points are evaluated at once.

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
        Array of shape (len(time), len(initial_conditions)) containing the
        state of the system at each time point.
    """
    # Edited with AI assistance and reviewed by a human in PR #16: closed-form exp(A·t) for 2x2 matrices that cannot be diagonalised.
    # Edited with AI assistance: the closed form is used for every 2x2 matrix, about 3x faster than the eigendecomposition.
    # Edited with AI assistance: real eigenvalues (every surface model) use scalar coefficients and
    # write the two columns directly, about 2x faster than building them with np.outer.
    time = np.asarray(time, dtype=float)
    A = np.asarray(A, dtype=float)
    steady_state_value = np.asarray(steady_state_value, dtype=float)
    v = np.asarray(initial_conditions, dtype=float) - steady_state_value

    if A.shape == (2, 2):
        a, b, c, d = (float(entry) for entry in A.ravel())
        half_tr = 0.5 * (a + d)
        # (tr/2)² − det written without cancellation
        disc = (0.5 * (a - d)) ** 2 + b * c

        if disc >= 0:
            root = disc ** 0.5
            # The eigenvalue of larger magnitude, then the other from the determinant
            big = half_tr + root if half_tr >= 0 else half_tr - root
            small = (a * d - b * c) / big if big != 0 else 0.0
            lam1, lam2 = (big, small) if big >= small else (small, big)
            gap = lam1 - lam2

            v0, v1 = float(v[0]), float(v[1])
            w0 = (a - lam1) * v0 + b * v1  # (A − λ1·I)·v
            w1 = c * v0 + (d - lam1) * v1

            # f(t) = (e^(λ1·t) − e^(λ2·t)) / (λ1 − λ2) = e^(λ1·t)·(1 − e^(−x)) / (λ1 − λ2), x = (λ1 − λ2)·t,
            # which tends to t·e^(λ1·t) for small x
            e1 = np.exp(lam1 * time)
            x = gap * time
            not_small = np.abs(x) >= 1e-12
            with np.errstate(invalid='ignore', divide='ignore'):
                f = e1 * np.where(not_small, -np.expm1(-x) / gap, time)

            states = np.empty((time.size, 2))
            states[:, 0] = steady_state_value[0] + v0 * e1 + w0 * f
            states[:, 1] = steady_state_value[1] + v1 * e1 + w1 * f
            return states

        # Complex pair of eigenvalues
        root = np.sqrt(complex(disc))
        # The eigenvalue of larger magnitude, then the other from the determinant
        big = half_tr + root if half_tr >= 0 else half_tr - root
        small = (a * d - b * c) / big if big != 0 else 0.0
        # λ1 has the larger real part, so x = (λ1 − λ2)·t >= 0 for t >= 0
        lam1, lam2 = (big, small) if np.real(big) >= np.real(small) else (small, big)

        x = (lam1 - lam2) * time
        g = np.ones_like(x)
        not_small = np.abs(x) >= 1e-12
        g[not_small] = -np.expm1(-x[not_small]) / x[not_small]  # (1 − e^(−x)) / x
        e1 = np.exp(lam1 * time)
        f = e1 * time * g

        w = (A - lam1 * np.eye(2)) @ v
        states = steady_state_value + np.outer(e1, v) + np.outer(f, w)
        return states.real

    eigvals, eigvecs = np.linalg.eig(A)
    alpha = np.linalg.solve(eigvecs, v)
    # exp_matrix[i, j] = exp(eigvals[j] * time[i])   — shape (N, n)
    exp_matrix = np.exp(np.outer(time, eigvals))
    states = steady_state_value + (exp_matrix * alpha) @ eigvecs.T
    return states.real                           # discard tiny imaginary noise

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
        Initial signal proportional to the concentration of E1, default is 0.
    sP2L : float, optional
        Initial signal proportional to the concentration of E2S, default is 0.

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

def solve_induced_fit_dissociation(time, koff, kc, krev, s0=0, sP2L=0, smax=0, signal_ratio=1.0):
    """
    Obtain the dissociation signal for the induced fit model (surface-based).

    By default the complex before (E·S) and after (ES) the conformational
    change produce the same signal. Set ``signal_ratio`` to let ES produce a
    different signal (see solve_induced_fit_association).

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
        Initial total signal, default is 0.
    sP2L : float, optional
        Initial amount of ES in E·S signal units (third column returned by
        solve_induced_fit_association), default is 0.
    smax : float, optional
        Signal of the surface fully occupied by E·S, default is 0.
    signal_ratio : float, optional
        Signal of ES relative to E·S, default is 1 (same signal).

    Returns
    -------
    np.ndarray
        Signal over time, with columns for total signal, signal produced by
        E·S, and amount of ES in E·S signal units.
    """
    time = time - np.min(time) # Start at zero

    b_col = constant_vector_induced_fit(koff, smax, kc)
    m     = differential_matrix_dissociation_induced_fit(koff, kc, krev)

    res_steady_state = solve_steady_state(m, b_col)

    # The total signal is R[E·S] + signal_ratio·R[ES], so the bound amount is
    # s0 - (signal_ratio - 1)·sP2L
    initial_conditions = np.array([smax - s0 + (signal_ratio - 1) * sP2L, sP2L])

    states = solve_all_states_fast(time, m, initial_conditions, res_steady_state)

    return _induced_fit_signal_columns(states, smax, signal_ratio)

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
        Initial signal proportional to the concentration of E1, default is 0.
    sP2L : float, optional
        Initial signal proportional to the concentration of E2S, default is 0.

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
    Solves the mixture analyte model - association phase.

    The equations of ``ode_mixture_analyte_association`` are linear,

        dR/dt = A·R + b,  A = −diag(koffs) − u·vᵀ,  b = u,

    with u_i = kon_i·F_i·C_TOT·Ris_max_i and v_j = 1/Ris_max_j, so they are
    solved exactly with ``solve_all_states_fast`` instead of numerically.

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    Ris0 : list
        Response of each analyte when the association starts.
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
        Time after the first time point at which the association starts
        (the response is Ris0 then), default is 0.

    Returns
    -------
    np.ndarray
        Response of EACH ANALYTE over time, shape (number of analytes, len(t)).
    """
    # Edited with AI assistance and reviewed by a human in PR #16: tighter solver tolerances.
    # Edited with AI assistance: exact solution of the linear system, and t0 shifts the start
    # (with solve_ivp starting at the first time point, t0 had no effect).
    t = np.asarray(t, dtype=float)
    elapsed = t - np.min(t) - t0

    Ris_max = np.atleast_1d(np.asarray(Ris_max, dtype=float))
    koffs = np.atleast_1d(np.asarray(koffs, dtype=float))
    Fis = np.atleast_1d(np.asarray(Fis, dtype=float))
    kons = koffs / np.atleast_1d(np.asarray(Kds, dtype=float))

    u = kons * Fis * C_TOT * Ris_max
    A = -np.diag(koffs) - np.outer(u, 1.0 / Ris_max)

    steady_state = solve_steady_state(A, u)
    states = solve_all_states_fast(elapsed, A, np.asarray(Ris0, dtype=float), steady_state)

    return states.T

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
    Solves the mixture analyte model - dissociation phase.

    Each analyte dissociates on its own, so the solution is exact:
    R_i(t) = Ris0_i·exp(−koff_i·t).

    Parameters
    ----------
    t : np.ndarray
        Time variable.
    Ris0 : list
        Response of each analyte when the dissociation starts.
    koffs : np.ndarray
        Dissociation rate constants.
    t0 : float, optional
        Time after the first time point at which the dissociation starts
        (the response is Ris0 then), default is 0.

    Returns
    -------
    np.ndarray
        Response of EACH ANALYTE over time, shape (number of analytes, len(t)).
    """
    # Edited with AI assistance and reviewed by a human in PR #16: tighter solver tolerances.
    # Edited with AI assistance: exact solution, and t0 shifts the start (it had no effect).
    t = np.asarray(t, dtype=float)
    elapsed = t - np.min(t) - t0

    Ris0 = np.atleast_1d(np.asarray(Ris0, dtype=float))
    koffs = np.atleast_1d(np.asarray(koffs, dtype=float))

    return Ris0[:, None] * np.exp(-np.outer(koffs, elapsed))

def _validate_fraction_site1(fraction_site1):
    """Validate that a site-1 population fraction lies between zero and one."""
    if fraction_site1 < 0 or fraction_site1 > 1:
        raise ValueError("fraction_site1 must be between 0 and 1.")


def steady_state_two_site_heterogeneous_ligand(C, Rmax, Kd1, Kd2, fraction_site1):
    """
    Steady state signal for a 2:1 heterogeneous ligand model.

    The model is the weighted sum of two independent 1:1 binding signals.
    ``fraction_site1`` is the fraction of the total binding response
    contributed by interaction 1; interaction 2 contributes
    ``1 - fraction_site1``.

    Parameters
    ----------
    C : np.ndarray
        Concentration of the analyte.
    Rmax : float
        Total maximum response from both ligand interactions.
    Kd1 : float
        Equilibrium dissociation constant for interaction 1.
    Kd2 : float
        Equilibrium dissociation constant for interaction 2.
    fraction_site1 : float
        Fractional contribution of interaction 1, from 0 to 1.

    Returns
    -------
    np.ndarray
        Steady state signal.
    """
    _validate_fraction_site1(fraction_site1)

    Rmax1 = Rmax * fraction_site1
    Rmax2 = Rmax * (1 - fraction_site1)
    signal1 = steady_state_one_site(C, Rmax1, Kd1)
    signal2 = steady_state_one_site(C, Rmax2, Kd2)
    return signal1 + signal2


def solve_two_site_heterogeneous_ligand_association(
    time,
    a_conc,
    Kd1,
    koff1,
    Kd2,
    koff2,
    Rmax=0,
    fraction_site1=0.5,
    s1_0=0,
    s2_0=0,
    t0=0,
):
    """
    Association signal for a 2:1 heterogeneous ligand model.

    The association trace is computed as two independent 1:1 association
    curves added together. The total response is split into two component
    responses using ``fraction_site1``.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    a_conc : float
        Analyte concentration.
    Kd1 : float
        Equilibrium dissociation constant for interaction 1.
    koff1 : float
        Dissociation rate constant for interaction 1.
    Kd2 : float
        Equilibrium dissociation constant for interaction 2.
    koff2 : float
        Dissociation rate constant for interaction 2.
    Rmax : float, optional
        Total maximum response from both ligand interactions, default is 0.
    fraction_site1 : float, optional
        Fractional contribution of interaction 1, from 0 to 1, default is 0.5.
    s1_0 : float, optional
        Initial signal from interaction 1, default is 0.
    s2_0 : float, optional
        Initial signal from interaction 2, default is 0.
    t0 : float, optional
        Initial time offset, default is 0.

    Returns
    -------
    np.ndarray
        Array with columns [total_signal, signal_site1, signal_site2].
    """
    _validate_fraction_site1(fraction_site1)

    time = np.array(time)
    Rmax1 = Rmax * fraction_site1
    Rmax2 = Rmax * (1 - fraction_site1)

    signal1 = one_site_association_analytical(time, s1_0, Rmax1, koff1, Kd1, a_conc, t0)
    signal2 = one_site_association_analytical(time, s2_0, Rmax2, koff2, Kd2, a_conc, t0)
    total_signal = signal1 + signal2

    return np.column_stack((total_signal, signal1, signal2))


def solve_two_site_heterogeneous_ligand_dissociation(
    time,
    koff1,
    koff2,
    s0=0,
    fraction_site1=0.5,
    s1_0=None,
    s2_0=None,
    t0=0,
):
    """
    Dissociation signal for a 2:1 heterogeneous ligand model.

    The dissociation trace is computed as two independent 1:1 dissociation
    curves added together. If component initial signals are not provided,
    ``s0`` is split using ``fraction_site1``.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    koff1 : float
        Dissociation rate constant for interaction 1.
    koff2 : float
        Dissociation rate constant for interaction 2.
    s0 : float, optional
        Initial total signal, default is 0.
    fraction_site1 : float, optional
        Fractional contribution of interaction 1, from 0 to 1, default is 0.5.
    s1_0 : float, optional
        Initial signal from interaction 1. If None, use s0 * fraction_site1.
    s2_0 : float, optional
        Initial signal from interaction 2. If None, use s0 * (1 - fraction_site1).
    t0 : float, optional
        Initial time offset, default is 0.

    Returns
    -------
    np.ndarray
        Array with columns [total_signal, signal_site1, signal_site2].
    """
    _validate_fraction_site1(fraction_site1)

    time = np.array(time)

    if s1_0 is None:
        s1_0 = s0 * fraction_site1
    if s2_0 is None:
        s2_0 = s0 * (1 - fraction_site1)

    signal1 = one_site_dissociation_analytical(time, s1_0, koff1, t0)
    signal2 = one_site_dissociation_analytical(time, s2_0, koff2, t0)
    total_signal = signal1 + signal2

    return np.column_stack((total_signal, signal1, signal2))


def steady_state_two_site(C, Rmax_PL, Rmax_LPL, Kd):
    """
    Steady state signal for a ligand with two identical, independent binding sites.

    The immobilized ligand P has two equivalent binding sites for the analyte L.
    PL (= LP) denotes singly-bound ligand, LPL denotes doubly-bound ligand.

    For two identical independent sites with intrinsic dissociation constant Kd,
    the equilibrium fractions are:

    - fPL  = 2 * theta * (1 - theta)
    - fLPL = theta ** 2

    where theta = C / (C + Kd) is the single-site occupancy.

    Parameters
    ----------
    C : np.ndarray
        Concentration of the analyte.
    Rmax_PL : float
        Maximum signal when all ligand is singly bound (PL state).
    Rmax_LPL : float
        Maximum signal when all ligand is doubly bound (LPL state).
    Kd : float
        Intrinsic equilibrium dissociation constant (per site).

    Returns
    -------
    np.ndarray
        Steady state signal.
    """
    C     = np.array(C)
    theta = C / (C + Kd)
    fPL   = 2 * theta * (1 - theta)
    fLPL  = theta ** 2
    signal = Rmax_PL * fPL + Rmax_LPL * fLPL
    return signal


def steady_state_two_site_cooperative(C, Rmax_PL, Rmax_LPL, Kd, sigma):
    """
    Steady state signal for a ligand with two identical binding sites and cooperativity.

    The cooperativity factor sigma modifies the second binding event.
    The intrinsic dissociation constant for the second site becomes Kd / sigma.

    - sigma > 1 : positive cooperativity (second binding is tighter).
    - sigma < 1 : negative cooperativity.
    - sigma = 1 : identical independent sites (non-cooperative).

    The equilibrium fractions are computed from the partition function:

        Z = 1 + 2 * C / Kd + sigma * (C / Kd) ** 2

    - fPL  = (2 * C / Kd) / Z
    - fLPL = sigma * (C / Kd) ** 2 / Z

    Parameters
    ----------
    C : np.ndarray
        Concentration of the analyte.
    Rmax_PL : float
        Maximum signal when all ligand is singly bound (PL state).
    Rmax_LPL : float
        Maximum signal when all ligand is doubly bound (LPL state).
    Kd : float
        Intrinsic equilibrium dissociation constant (per site).
    sigma : float
        Cooperativity factor.

    Returns
    -------
    np.ndarray
        Steady state signal.
    """
    C    = np.array(C)
    r    = C / Kd
    Z    = 1 + 2 * r + sigma * r ** 2
    fPL  = (2 * r) / Z
    fLPL = (sigma * r ** 2) / Z
    signal = Rmax_PL * fPL + Rmax_LPL * fLPL
    return signal


def differential_matrix_association_two_site(kon, koff, a):
    """
    Differential matrix for the two-site model during association (non-cooperative).

    The ligand P (immobilized) has two equivalent, independent binding sites.
    The analyte L flows at fixed concentration ``a``.

    State variables: [fPL, fLPL] where

    - fPL  = fraction of ligand in the singly-bound state (PL + LP)
    - fLPL = fraction of ligand in the doubly-bound state (LPL)
    - fP   = 1 - fPL - fLPL (free ligand, from conservation)

    Reactions with statistical factors::

        P  + L  -> PL  : 2 * kon * a * fP    (2 equivalent empty sites)
        PL      -> P+L : koff * fPL           (1 occupied site)
        PL + L  -> LPL : kon * a * fPL        (1 remaining empty site)
        LPL     -> PL+L: 2 * koff * fLPL      (2 occupied sites)

    The resulting ODE is  d[fPL, fLPL]/dt = A @ [fPL, fLPL] + b
    where b = constant_vector_two_site(kon, a).

    Parameters
    ----------
    kon : float
        Intrinsic association rate constant (per site).
    koff : float
        Intrinsic dissociation rate constant (per site).
    a : float
        Analyte concentration.

    Returns
    -------
    np.ndarray
        2×2 differential matrix.
    """
    row1 = [-(3 * kon * a + koff), 2 * koff - 2 * kon * a]
    row2 = [kon * a,               -2 * koff]
    return np.array([row1, row2])


def differential_matrix_association_two_site_cooperative(kon, koff, sigma, a):
    """
    Differential matrix for the two-site model during association (cooperative).

    Same as :func:`differential_matrix_association_two_site`, but the second
    binding step is modified by the cooperativity factor sigma.

    The kinetic effect is split symmetrically between on-rate and off-rate
    using sqrt(sigma):

    - Second site on-rate  = sqrt(sigma) * kon
    - Second site off-rate = koff / sqrt(sigma)

    This gives an effective Kd for the second site of Kd / sigma.

    Parameters
    ----------
    kon : float
        Intrinsic association rate constant (per site).
    koff : float
        Intrinsic dissociation rate constant (per site).
    sigma : float
        Cooperativity factor. sigma > 1 for positive cooperativity.
    a : float
        Analyte concentration.

    Returns
    -------
    np.ndarray
        2×2 differential matrix.
    """
    s = np.sqrt(sigma)
    row1 = [-(2 * kon * a + koff + s * kon * a), 2 * koff / s - 2 * kon * a]
    row2 = [s * kon * a,                         -2 * koff / s]
    return np.array([row1, row2])


def constant_vector_two_site(kon, a):
    """
    Constant vector for the two-site association ODE.

    This vector is the same for both cooperative and non-cooperative cases.

    Parameters
    ----------
    kon : float
        Intrinsic association rate constant (per site).
    a : float
        Analyte concentration.

    Returns
    -------
    np.ndarray
        Constant vector [2*kon*a, 0].
    """
    return np.array([2 * kon * a, 0.0])


def differential_matrix_dissociation_two_site(koff):
    """
    Differential matrix for the two-site model during dissociation (non-cooperative).

    When analyte concentration is zero, only dissociation occurs::

        PL  -> P + L  : koff * fPL          (1 occupied site)
        LPL -> PL + L : 2 * koff * fLPL     (2 occupied sites, either can leave)

    Parameters
    ----------
    koff : float
        Intrinsic dissociation rate constant (per site).

    Returns
    -------
    np.ndarray
        2×2 differential matrix.
    """
    row1 = [-koff,  2 * koff]
    row2 = [0,     -2 * koff]
    return np.array([row1, row2])


def differential_matrix_dissociation_two_site_cooperative(koff, sigma):
    """
    Differential matrix for the two-site model during dissociation (cooperative).

    The kinetic effect is split symmetrically using sqrt(sigma), so the
    per-site off-rate in the doubly-bound state is koff / sqrt(sigma)::

        PL  -> P + L  : koff * fPL                       (first site, unmodified)
        LPL -> PL + L : 2 * (koff / sqrt(sigma)) * fLPL  (second site, modified)

    Parameters
    ----------
    koff : float
        Intrinsic dissociation rate constant (per site).
    sigma : float
        Cooperativity factor.

    Returns
    -------
    np.ndarray
        2×2 differential matrix.
    """
    s = np.sqrt(sigma)
    row1 = [-koff,  2 * koff / s]
    row2 = [0,     -2 * koff / s]
    return np.array([row1, row2])


def solve_two_site_association(time, a_conc, kon, koff, Rmax_PL=0, Rmax_LPL=0, fPL_0=0, fLPL_0=0, t0=0):
    """
    Solve the two-site binding model (non-cooperative) during association.

    The immobilized ligand P has two equivalent, independent binding sites.
    The analyte L flows at fixed concentration ``a_conc``.
    The observed signal is a weighted sum of the singly-bound and doubly-bound
    fractions, with different response factors ``Rmax_PL`` and ``Rmax_LPL``.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    a_conc : float
        Analyte concentration.
    kon : float
        Intrinsic association rate constant (per site).
    koff : float
        Intrinsic dissociation rate constant (per site).
    Rmax_PL : float, optional
        Signal when all ligand is singly bound (PL), default is 0.
    Rmax_LPL : float, optional
        Signal when all ligand is doubly bound (LPL), default is 0.
    fPL_0 : float, optional
        Initial fraction of singly-bound ligand, default is 0.
    fLPL_0 : float, optional
        Initial fraction of doubly-bound ligand, default is 0.
    t0 : float, optional
        Time offset; the association starts at time t0 after the first
        time point, default is 0.

    Returns
    -------
    np.ndarray
        Array with columns [total_signal, signal_PL, signal_LPL].
    """
    # Edited with AI assistance and reviewed by a human in PR #16: added t0 so a fitted t0 shifts the association.
    time = time - np.min(time) - t0

    A = differential_matrix_association_two_site(kon, koff, a_conc)
    b = constant_vector_two_site(kon, a_conc)

    initial_conditions = np.array([fPL_0, fLPL_0])
    steady_state       = solve_steady_state(A, b)

    states = solve_all_states_fast(time, A, initial_conditions, steady_state)

    signal_PL    = Rmax_PL  * states[:, 0]
    signal_LPL   = Rmax_LPL * states[:, 1]
    total_signal = signal_PL + signal_LPL

    return np.column_stack((total_signal, signal_PL, signal_LPL))


def solve_two_site_cooperative_association(time, a_conc, kon, koff, sigma, Rmax_PL=0, Rmax_LPL=0, fPL_0=0, fLPL_0=0, t0=0):
    """
    Solve the two-site binding model (cooperative) during association.

    Same as :func:`solve_two_site_association` but with a cooperativity
    factor ``sigma`` that modifies the second binding step:

    - Second site on-rate  = sqrt(sigma) * kon
    - Second site off-rate = koff / sqrt(sigma)
    - Effective Kd for the second site = Kd / sigma

    Parameters
    ----------
    time : np.ndarray
        Time array.
    a_conc : float
        Analyte concentration.
    kon : float
        Intrinsic association rate constant (per site).
    koff : float
        Intrinsic dissociation rate constant (per site).
    sigma : float
        Cooperativity factor. sigma > 1 for positive cooperativity.
    Rmax_PL : float, optional
        Signal when all ligand is singly bound (PL), default is 0.
    Rmax_LPL : float, optional
        Signal when all ligand is doubly bound (LPL), default is 0.
    fPL_0 : float, optional
        Initial fraction of singly-bound ligand, default is 0.
    fLPL_0 : float, optional
        Initial fraction of doubly-bound ligand, default is 0.
    t0 : float, optional
        Time offset; the association starts at time t0 after the first
        time point, default is 0.

    Returns
    -------
    np.ndarray
        Array with columns [total_signal, signal_PL, signal_LPL].
    """
    # Edited with AI assistance and reviewed by a human in PR #16: added t0 so a fitted t0 shifts the association.
    time = time - np.min(time) - t0

    A = differential_matrix_association_two_site_cooperative(kon, koff, sigma, a_conc)
    b = constant_vector_two_site(kon, a_conc)

    initial_conditions = np.array([fPL_0, fLPL_0])
    steady_state       = solve_steady_state(A, b)

    states = solve_all_states_fast(time, A, initial_conditions, steady_state)

    signal_PL    = Rmax_PL  * states[:, 0]
    signal_LPL   = Rmax_LPL * states[:, 1]
    total_signal = signal_PL + signal_LPL

    return np.column_stack((total_signal, signal_PL, signal_LPL))


def solve_two_site_dissociation(time, koff, Rmax_PL=0, Rmax_LPL=0, fPL_0=0, fLPL_0=0):
    """
    Solve the two-site binding model (non-cooperative) during dissociation.

    Analyte concentration is zero. Only dissociation of existing complexes occurs.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    koff : float
        Intrinsic dissociation rate constant (per site).
    Rmax_PL : float, optional
        Signal when all ligand is singly bound (PL), default is 0.
    Rmax_LPL : float, optional
        Signal when all ligand is doubly bound (LPL), default is 0.
    fPL_0 : float, optional
        Initial fraction of singly-bound ligand.
    fLPL_0 : float, optional
        Initial fraction of doubly-bound ligand.

    Returns
    -------
    np.ndarray
        Array with columns [total_signal, signal_PL, signal_LPL].
    """
    time = time - np.min(time)

    A = differential_matrix_dissociation_two_site(koff)
    b = np.zeros(2)

    initial_conditions = np.array([fPL_0, fLPL_0])
    steady_state       = solve_steady_state(A, b)

    states = solve_all_states_fast(time, A, initial_conditions, steady_state)

    signal_PL    = Rmax_PL  * states[:, 0]
    signal_LPL   = Rmax_LPL * states[:, 1]
    total_signal = signal_PL + signal_LPL

    return np.column_stack((total_signal, signal_PL, signal_LPL))


def solve_two_site_cooperative_dissociation(time, koff, sigma, Rmax_PL=0, Rmax_LPL=0, fPL_0=0, fLPL_0=0):
    """
    Solve the two-site binding model (cooperative) during dissociation.

    Same as :func:`solve_two_site_dissociation` but with a cooperativity
    factor ``sigma`` that modifies the second site off-rate:

    - First site off-rate  = koff                (PL -> P + L, unmodified)
    - Second site off-rate = koff / sqrt(sigma)   (LPL -> PL + L, modified)

    Parameters
    ----------
    time : np.ndarray
        Time array.
    koff : float
        Intrinsic dissociation rate constant (per site).
    sigma : float
        Cooperativity factor.
    Rmax_PL : float, optional
        Signal when all ligand is singly bound (PL), default is 0.
    Rmax_LPL : float, optional
        Signal when all ligand is doubly bound (LPL), default is 0.
    fPL_0 : float, optional
        Initial fraction of singly-bound ligand.
    fLPL_0 : float, optional
        Initial fraction of doubly-bound ligand.

    Returns
    -------
    np.ndarray
        Array with columns [total_signal, signal_PL, signal_LPL].
    """
    time = time - np.min(time)

    A = differential_matrix_dissociation_two_site_cooperative(koff, sigma)
    b = np.zeros(2)

    initial_conditions = np.array([fPL_0, fLPL_0])
    steady_state       = solve_steady_state(A, b)

    states = solve_all_states_fast(time, A, initial_conditions, steady_state)

    signal_PL    = Rmax_PL  * states[:, 0]
    signal_LPL   = Rmax_LPL * states[:, 1]
    total_signal = signal_PL + signal_LPL

    return np.column_stack((total_signal, signal_PL, signal_LPL))


# Bivalent analyte model. This section is AI generated.
#
# A bivalent analyte A binds the immobilised ligand L with one arm (AL), and the
# bound analyte can bind a second, neighbouring ligand with the other arm (AL2):
#
#     A + L   <=> AL     ka1 [1/(µM·s)], kd1 [1/s]
#     AL + L  <=> AL2    ka2 [1/(signal·s)], kd2 [1/s]
#
# Following the Biacore bivalent analyte model (and DukeCHSI/Bivalent-Analyte-Modeling),
# with x1 = [AL] and x2 = [AL2] in signal units (RU for SPR, nm for BLI) and free
# ligand l = Rmax - x1 - 2*x2:
#
#     dx1/dt = 2*ka1*C*l - kd1*x1 - (ka2*x1*l - 2*kd2*x2)
#     dx2/dt = ka2*x1*l - 2*kd2*x2
#
# The factors 2 count the two arms of the analyte. Rmax is the response with every
# ligand bound by one analyte, and the signal is x1 + x2 (one analyte per complex).

def ode_bivalent_analyte(t, x, ka1, kd1, ka2, kd2, a_conc, Rmax):
    """
    Rates of change of the bivalent analyte complexes.

    Parameters
    ----------
    t : float
        Time (unused; the system is autonomous).
    x : array_like
        [x1, x2]: singly bound (AL) and doubly bound (AL2) analyte, in response units.
    ka1 : float
        Association rate constant of the first arm, 1/(µM·s).
    kd1 : float
        Dissociation rate constant of the first arm, 1/s.
    ka2 : float
        Association rate constant of the second arm, 1/(signal·s).
    kd2 : float
        Dissociation rate constant of the second arm, 1/s.
    a_conc : float
        Analyte concentration, µM (0 during dissociation).
    Rmax : float
        Response with every ligand bound by one analyte.

    Returns
    -------
    list
        [dx1/dt, dx2/dt].
    """
    # Python floats: arithmetic on numpy scalars made the integration ~2x slower
    x1, x2 = x.tolist() if isinstance(x, np.ndarray) else x
    free = Rmax - x1 - 2 * x2
    step1 = 2 * ka1 * a_conc * free - kd1 * x1
    step2 = ka2 * x1 * free - 2 * kd2 * x2
    return [step1 - step2, step2]


def _jacobian_bivalent_analyte(t, x, ka1, kd1, ka2, kd2, a_conc, Rmax):
    """Jacobian of ode_bivalent_analyte with respect to [x1, x2]."""
    x1, x2 = x.tolist() if isinstance(x, np.ndarray) else x
    free = Rmax - x1 - 2 * x2
    d_step2_dx1 = ka2 * (free - x1)
    d_step2_dx2 = -2 * ka2 * x1 - 2 * kd2
    return [[-2 * ka1 * a_conc - kd1 - d_step2_dx1, -4 * ka1 * a_conc - d_step2_dx2],
            [d_step2_dx1, d_step2_dx2]]


def _solve_bivalent_analyte(tau, x0, ka1, kd1, ka2, kd2, a_conc, Rmax, rtol=1e-8):
    """
    Integrate the bivalent analyte model.

    Parameters
    ----------
    tau : np.ndarray
        Times from the start of the step. Points before 0 keep the initial state.
    x0 : list
        Initial [x1, x2].
    rtol : float, optional
        Relative tolerance of the integration, default is 1e-8. The absolute
        tolerance is rtol * 1e-2 * Rmax.

    Returns
    -------
    np.ndarray
        Array with columns [x1 + x2, x1, x2].
    """
    tau = np.asarray(tau, dtype=float)
    states = np.tile(np.asarray(x0, dtype=float), (tau.size, 1))
    after = tau >= 0
    if np.any(after) and np.max(tau) > 0:
        # odeint runs LSODA in compiled code; solve_ivp's Python stepping was ~10x slower
        t_eval = np.concatenate([[0.0], tau[after]])
        args = tuple(float(v) for v in (ka1, kd1, ka2, kd2, a_conc, Rmax))
        out, info = odeint(ode_bivalent_analyte, x0, t_eval, args=args, Dfun=_jacobian_bivalent_analyte,
                           tfirst=True, rtol=rtol, atol=rtol * 1e-2 * max(abs(Rmax), 1e-12),
                           mxstep=100000, full_output=True)
        if info['message'] != 'Integration successful.':
            raise RuntimeError(f"The bivalent analyte model could not be solved: {info['message']}")
        states[after] = out[1:]
    return np.column_stack((states.sum(axis=1), states[:, 0], states[:, 1]))


def solve_bivalent_analyte_association(time, a_conc, Kd1, koff1, ka2, kd2, Rmax,
                                       x1_0=0.0, x2_0=0.0, t0=0, rtol=1e-8):
    """
    Association signal of a bivalent analyte binding an immobilised ligand.

    See the description of the model above ode_bivalent_analyte.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    a_conc : float
        Analyte concentration, µM.
    Kd1 : float
        Equilibrium dissociation constant of the first arm (kd1/ka1), µM.
    koff1 : float
        Dissociation rate constant of the first arm (kd1), 1/s.
    ka2 : float
        Association rate constant of the second arm, 1/(signal·s).
    kd2 : float
        Dissociation rate constant of the second arm, 1/s.
    Rmax : float
        Response with every ligand bound by one analyte.
    x1_0, x2_0 : float, optional
        Singly and doubly bound analyte at the start of the step, default is 0.
    t0 : float, optional
        Time offset; the association starts at time t0 after the first time
        point, default is 0.
    rtol : float, optional
        Relative tolerance of the integration, default is 1e-8.

    Returns
    -------
    np.ndarray
        Array with columns [total_signal, signal_AL, signal_AL2].
    """
    time = np.asarray(time, dtype=float)
    tau = time - np.min(time) - t0
    return _solve_bivalent_analyte(tau, [x1_0, x2_0], koff1 / Kd1, koff1, ka2, kd2, a_conc, Rmax, rtol)


def solve_bivalent_analyte_dissociation(time, koff1, ka2, kd2, Rmax, x1_0=0.0, x2_0=0.0, rtol=1e-8):
    """
    Dissociation signal of a bivalent analyte from an immobilised ligand.

    Without analyte in solution, AL2 can still open one arm and rebind it, and
    AL loses the analyte with rate kd1. See the description above ode_bivalent_analyte.

    Parameters
    ----------
    time : np.ndarray
        Time array; the dissociation starts at the first time point.
    koff1 : float
        Dissociation rate constant of the first arm (kd1), 1/s.
    ka2 : float
        Association rate constant of the second arm, 1/(signal·s).
    kd2 : float
        Dissociation rate constant of the second arm, 1/s.
    Rmax : float
        Response with every ligand bound by one analyte.
    x1_0, x2_0 : float, optional
        Singly and doubly bound analyte at the start of the dissociation.
    rtol : float, optional
        Relative tolerance of the integration, default is 1e-8.

    Returns
    -------
    np.ndarray
        Array with columns [total_signal, signal_AL, signal_AL2].
    """
    time = np.asarray(time, dtype=float)
    return _solve_bivalent_analyte(time - np.min(time), [x1_0, x2_0], 0.0, koff1, ka2, kd2, 0.0, Rmax, rtol)
