# This code is AI generated.
"""
Closed-form solution of dx/dt = A·x + b for 2×2 systems, used by the
induced-fit, conformational-selection and two-site surface models.

The reference is scipy.linalg.expm on well-separated eigenvalues. Near equal
eigenvalues expm itself loses accuracy (about 1e-5 for the cooperative
two-site dissociation with sigma within 1e-12 of 4, against 1e-16 for the
closed form, checked with 50-digit arithmetic), so those cases are checked
against exact formulas instead (see also test_model_audit_fixes.py).
"""
import itertools

import numpy as np
import pytest
from scipy.linalg import LinAlgError, expm

from pykingenie.utils.signal_surface import (
    constant_vector_conformational_selection,
    constant_vector_induced_fit,
    constant_vector_two_site,
    differential_matrix_association_conformational_selection,
    differential_matrix_association_induced_fit,
    differential_matrix_association_two_site_cooperative,
    differential_matrix_dissociation_induced_fit,
    solve_all_states_fast,
    solve_steady_state,
)

T = np.linspace(0, 600, 121)
RATES = [1e-4, 1e-2, 1.0, 10.0]


def _expm_reference(A, x0, x_ss, t=T):
    return np.array([x_ss + expm(A * ti) @ (x0 - x_ss) for ti in t])


def _model_systems():
    smax = 5.0
    for kon, koff, kc, krev, a in itertools.product([1e-2, 1e2], RATES, RATES, RATES, [0.01, 100]):
        yield (differential_matrix_association_induced_fit(koff, kon, kc, krev, a),
               constant_vector_induced_fit(koff, smax, kc), np.array([smax, 0.0]))
        yield (differential_matrix_dissociation_induced_fit(koff, kc, krev),
               constant_vector_induced_fit(koff, smax, kc), np.array([1.0, 2.0]))
        yield (differential_matrix_association_conformational_selection(koff, kon, kc, krev, a),
               constant_vector_conformational_selection(kon, smax, krev, a),
               np.array([smax / (kc / krev + 1), 0.0]))
    for kon, koff, sigma, a in itertools.product([1e-2, 1e2], RATES, [0.01, 1.0, 100.0], [0.01, 100]):
        yield (differential_matrix_association_two_site_cooperative(kon, koff, sigma, a),
               constant_vector_two_site(kon, a), np.zeros(2))


def test_closed_form_matches_expm_for_the_surface_models():
    """Induced fit, conformational selection and two-site systems, including stiff ones (rates 1e-4 to 1e4)."""
    worst = 0.0
    for A, b, x0 in _model_systems():
        A = np.asarray(A, dtype=float)
        x_ss = solve_steady_state(A, b)
        states = solve_all_states_fast(T, A, x0, x_ss)
        ref = _expm_reference(A, x0, x_ss)
        worst = max(worst, np.max(np.abs(states - ref)) / np.max(np.abs(ref)))
    assert worst < 1e-9


def test_steady_state_2x2_matches_general_solver():
    """Some systems have a condition number up to 2e10, so both solvers carry errors of about 1e-11
    (against 50-digit arithmetic: 7e-12 for the closed form, 9e-12 for LAPACK)."""
    for A, b, _ in _model_systems():
        A = np.asarray(A, dtype=float)
        ref = -np.linalg.solve(A, b)
        np.testing.assert_allclose(solve_steady_state(A, b), ref, rtol=0, atol=1e-10 * np.max(np.abs(ref)))


def test_steady_state_singular_2x2_raises():
    """A singular 2×2 matrix falls back to the general solver, which reports it."""
    with pytest.raises(LinAlgError):
        solve_steady_state(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 1.0]))


def test_complex_eigenvalues():
    """A damped rotation: exp(A·t) = e^(−k·t)·[[cos ωt, −sin ωt], [sin ωt, cos ωt]]."""
    k, w = 0.01, 0.05
    A = np.array([[-k, -w], [w, -k]])
    x0 = np.array([1.0, 0.5])
    states = solve_all_states_fast(T, A, x0, np.zeros(2))
    c, s, e = np.cos(w * T), np.sin(w * T), np.exp(-k * T)
    expected = np.column_stack((e * (c * x0[0] - s * x0[1]), e * (s * x0[0] + c * x0[1])))
    np.testing.assert_allclose(states, expected, rtol=0, atol=1e-13)


def test_zero_matrix_keeps_the_initial_state():
    states = solve_all_states_fast(T, np.zeros((2, 2)), np.array([1.0, 2.0]), np.zeros(2))
    np.testing.assert_array_equal(states, np.tile([1.0, 2.0], (len(T), 1)))


def test_negative_times():
    """Times before the start (a fitted t0) follow the same solution backward."""
    A = np.array([[-0.05, 0.01], [0.02, -0.03]])
    x0, x_ss = np.array([1.0, 2.0]), np.array([0.2, 0.1])
    t = np.linspace(-20, 20, 41)
    np.testing.assert_allclose(solve_all_states_fast(t, A, x0, x_ss), _expm_reference(A, x0, x_ss, t),
                               rtol=1e-12)


def test_larger_systems_use_the_eigendecomposition():
    A = np.array([[-0.3, 0.1, 0.0], [0.1, -0.2, 0.05], [0.0, 0.05, -0.1]])
    b = np.array([0.1, 0.0, 0.2])
    x0 = np.array([1.0, 0.0, 0.5])
    x_ss = solve_steady_state(A, b)
    np.testing.assert_allclose(x_ss, -np.linalg.solve(A, b), rtol=1e-12)
    np.testing.assert_allclose(solve_all_states_fast(T, A, x0, x_ss), _expm_reference(A, x0, x_ss),
                               rtol=1e-10, atol=1e-14)


# Real eigenvalues (every surface model) take a path with scalar coefficients


def _putzer_with_outer(time, A, x0, x_ss):
    """The previous implementation of the 2x2 closed form, built with np.outer."""
    a, b, c, d = A.ravel()
    half_tr, disc = 0.5 * (a + d), (0.5 * (a - d)) ** 2 + b * c
    root = np.sqrt(disc)
    big = half_tr + root if half_tr >= 0 else half_tr - root
    small = (a * d - b * c) / big if big != 0 else 0.0
    lam1, lam2 = max(big, small), min(big, small)
    x = (lam1 - lam2) * time
    g = np.ones_like(x)
    g[np.abs(x) >= 1e-12] = -np.expm1(-x[np.abs(x) >= 1e-12]) / x[np.abs(x) >= 1e-12]
    e1 = np.exp(lam1 * time)
    v = x0 - x_ss
    return x_ss + np.outer(e1, v) + np.outer(e1 * time * g, (A - lam1 * np.eye(2)) @ v)


def test_real_eigenvalues_match_the_previous_implementation():
    t = T
    for A, b, x0 in _model_systems():
        A = np.asarray(A, dtype=float)
        x_ss = solve_steady_state(A, b)
        states = solve_all_states_fast(t, A, x0, x_ss)
        previous = _putzer_with_outer(t, A, x0, x_ss)
        scale = np.max(np.abs(previous), axis=0)
        np.testing.assert_allclose(states / scale, previous / scale, rtol=0, atol=1e-13)


@pytest.mark.parametrize("gap", [0.0, 1e-15, 1e-9])
def test_equal_and_nearly_equal_eigenvalues(gap):
    """[[λ+gap, 1], [0, λ]]: exp(A·t) = [[e^((λ+gap)·t), (e^((λ+gap)·t) − e^(λ·t)) / gap], [0, e^(λ·t)]],
    with (e^((λ+gap)·t) − e^(λ·t)) / gap = t·e^(λ·t) when gap = 0."""
    lam = -0.02
    A = np.array([[lam + gap, 1.0], [0.0, lam]])
    x0 = np.array([0.5, 2.0])
    states = solve_all_states_fast(T, A, x0, np.zeros(2))

    e = np.exp(lam * T)
    coupling = T * e if gap == 0 else e * np.expm1(gap * T) / gap
    expected = np.column_stack((np.exp((lam + gap) * T) * x0[0] + coupling * x0[1], e * x0[1]))
    np.testing.assert_allclose(states, expected, rtol=1e-12, atol=1e-15)


def test_single_time_point():
    A = np.array([[-0.05, 0.01], [0.02, -0.03]])
    x0, x_ss = np.array([1.0, 2.0]), np.array([0.2, 0.1])
    states = solve_all_states_fast(np.array([7.0]), A, x0, x_ss)
    assert states.shape == (1, 2)
    np.testing.assert_allclose(states, _expm_reference(A, x0, x_ss, t=[7.0]), rtol=1e-12)
