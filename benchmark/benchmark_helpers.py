"""
Helpers for the parameter-recovery benchmark notebooks.

This file is AI generated.

The data are simulated from first principles: each notebook writes the rate
equations of its model, and this module integrates them with SciPy through the
injection schedule of the experiment. Nothing here calls pykingenie, so the
simulated data do not depend on the code being benchmarked.

Experiment layouts
------------------
Multi-cycle kinetics (MCK): each concentration is injected on a free surface
(regenerated between cycles), followed by a dissociation in buffer.

Single-cycle kinetics (SCK): the concentrations are injected one after the
other, in increasing order, without regeneration. Each injection is followed by
a short dissociation, and the last one by a long dissociation. Every step starts
from the state at the end of the previous step.

The first dissociation point is the last association point of the same cycle
(the injection ends there), so the data follow the instrument's timeline exactly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def integrate(rhs, y0, t, conc, params):
    """
    Integrate dy/dt = rhs(t, y, conc, params) from y0 at t[0], output at every t.

    Uses the implicit Radau method with tight tolerances, so stiff models (for
    example mass transport with a thin diffusion layer) are solved accurately.
    """
    t = np.asarray(t, dtype=float)
    sol = solve_ivp(rhs, (t[0], t[-1]), np.asarray(y0, dtype=float), t_eval=t,
                    args=(conc, params), method='Radau', rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def _noisy(signal, noise_sd, rng):
    return signal + rng.normal(0.0, noise_sd, signal.shape)


def simulate_mck(rhs, signal, y0, params, concs, assoc_time=300.0, disso_time=600.0,
                 dt=1.0, noise_sd=0.1, seed=0):
    """
    Simulate multi-cycle kinetics.

    Parameters
    ----------
    rhs : callable
        rhs(t, y, conc, params) giving dy/dt for the model states y.
    signal : callable
        signal(y, params) giving the instrument signal for an array of states
        (one row per time point).
    y0 : list
        States of the free surface.
    params : dict
        True model parameters, passed to rhs and signal.
    concs : list
        Analyte concentration of each cycle (µM).
    assoc_time, disso_time, dt : float
        Length of the association and dissociation, and sampling interval (s).
    noise_sd : float
        Standard deviation of the Gaussian noise added to the signal.
    seed : int
        Seed of the random noise.

    Returns
    -------
    dict
        time_assoc, assoc, time_disso, disso (lists, one array per cycle, with
        noise), and assoc_true, disso_true (noise-free).
    """
    rng = np.random.default_rng(seed)
    t_a = np.arange(0.0, assoc_time + dt / 2, dt)
    t_d = np.arange(assoc_time, assoc_time + disso_time + dt / 2, dt)
    out = {k: [] for k in ['time_assoc', 'assoc', 'time_disso', 'disso', 'assoc_true', 'disso_true']}
    for conc in concs:
        y_a = integrate(rhs, y0, t_a, conc, params)
        y_d = integrate(rhs, y_a[-1], t_d, 0.0, params)
        s_a, s_d = signal(y_a, params), signal(y_d, params)
        out['time_assoc'].append(t_a.copy())
        out['time_disso'].append(t_d.copy())
        out['assoc_true'].append(s_a)
        out['disso_true'].append(s_d)
        out['assoc'].append(_noisy(s_a, noise_sd, rng))
        out['disso'].append(_noisy(s_d, noise_sd, rng))
    return out


def simulate_sck(rhs, signal, y0, params, concs, assoc_time=120.0, disso_time=60.0,
                 final_disso_time=900.0, dt=1.0, noise_sd=0.1, seed=0):
    """
    Simulate single-cycle kinetics.

    The concentrations are injected in the given order. Each injection lasts
    assoc_time and is followed by disso_time of buffer; the last injection is
    followed by final_disso_time. Times are absolute (seconds from the first
    injection). See simulate_mck for the other parameters and the output.
    """
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ['time_assoc', 'assoc', 'time_disso', 'disso', 'assoc_true', 'disso_true']}
    y, start = np.asarray(y0, dtype=float), 0.0
    for i, conc in enumerate(concs):
        d_time = final_disso_time if i == len(concs) - 1 else disso_time
        t_a = np.arange(start, start + assoc_time + dt / 2, dt)
        t_d = np.arange(t_a[-1], t_a[-1] + d_time + dt / 2, dt)
        y_a = integrate(rhs, y, t_a, conc, params)
        y_d = integrate(rhs, y_a[-1], t_d, 0.0, params)
        s_a, s_d = signal(y_a, params), signal(y_d, params)
        out['time_assoc'].append(t_a)
        out['time_disso'].append(t_d)
        out['assoc_true'].append(s_a)
        out['disso_true'].append(s_d)
        out['assoc'].append(_noisy(s_a, noise_sd, rng))
        out['disso'].append(_noisy(s_d, noise_sd, rng))
        y, start = y_d[-1], t_d[-1]
    return out


def compare_parameters(true_values, fitted_values, rtol=0.05):
    """
    Table of true and fitted parameters, with the relative error.

    Parameters
    ----------
    true_values, fitted_values : dict
        Parameter name -> value. Only the names in true_values are compared.
    rtol : float
        Largest relative error counted as recovered, default 5%.

    Returns
    -------
    pd.DataFrame
        One row per parameter: true, fitted, relative error (%) and recovered.
    """
    rows = []
    for name, true in true_values.items():
        fitted = float(fitted_values[name])
        rel_error = (fitted - true) / true
        rows.append({'parameter': name, 'true': true, 'fitted': fitted,
                     'rel. error (%)': 100 * rel_error, 'recovered': abs(rel_error) <= rtol})
    return pd.DataFrame(rows).set_index('parameter')


def plot_fit(data, fitter, concs, title):
    """Plot the simulated data, the pykingenie fit and the residuals."""
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(concs)))
    for i, conc in enumerate(concs):
        for t, y, fit in [(data['time_assoc'][i], data['assoc'][i], fitter.signal_assoc_fit[i]),
                          (data['time_disso'][i], data['disso'][i], fitter.signal_disso_fit[i])]:
            ax.plot(t, y, '.', ms=2, color=colors[i], alpha=0.4)
            ax.plot(t, fit, '-', lw=1.2, color='black')
            ax_res.plot(t, y - np.asarray(fit), '.', ms=1, color=colors[i])
        ax.plot([], [], '.', color=colors[i], label=f'{conc:.3g} µM')
    ax.plot([], [], '-', color='black', label='pykingenie fit')
    ax.set_ylabel('Signal (RU)')
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax_res.axhline(0, color='grey', lw=0.8)
    ax_res.set_ylabel('Residual')
    ax_res.set_xlabel('Time (s)')
    fig.tight_layout()
    return fig
