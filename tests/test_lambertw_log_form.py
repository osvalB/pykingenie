# This code is AI generated.
"""
Real-valued solver of ln|y| + y = L used by the closed-form mass transport model,
against reference values of the Lambert W function computed with mpmath (50 digits).
"""
import numpy as np
import pytest
from scipy.special import lambertw

from pykingenie.utils.signal_surface import _lambertw_log_form

# (L, W0(exp(L))): y > 0, from y close to 0 to beyond the overflow of exp(L)
POSITIVE = [
    (-700, 9.8596765437597708567e-305),
    (-100, 3.720075976020835963e-44),
    (-30, 9.3576229688392989538e-14),
    (-5, 0.0066930004977309932744),
    (-1, 0.27846454276107379511),
    (-0.1, 0.53169161977901767911),
    (0, 0.567143290409783873),
    (0.5, 0.76624860816175025888),
    (0.9, 0.95063017363368636079),
    (1.0, 1.0),
    (1.1, 1.0506197611984362393),
    (3, 2.2079400315693229986),
    (10, 7.9294200950196973486),
    (50, 46.16771916549208967),
    (700, 693.45830887902549834),
    (710, 703.44401171195456432),
    (1e4, 9990.7905809942519249),
    (1e10, 9999999976.9741490724),
    (1e15, 999999999999965.46122),
]

# (L, W0(-exp(L))): -1 <= y < 0, from y close to 0 to close to the branch point y = -1
NEGATIVE = [
    (-700, -9.8596765437597708567e-305),
    (-100, -3.720075976020835963e-44),
    (-30, -9.357622968841050256e-14),
    (-5, -0.0067838113520969712541),
    (-2, -0.15859433956303936215),
    (-1.5, -0.30170956268433601153),
    (-1.2, -0.49323942377515349179),
    (-1.05, -0.71618945516934408316),
    (-1.01, -0.86516524893316525714),
    (-1.001, -0.9559428077409543079),
    (-1.00001, -0.99553452822565061189),
    (-1.00000001, -0.99985858531078048984),
]


def _solve_forward(table, y0):
    """All points forward in time from y0, so only the bounds y0 and 0 apply."""
    L, expected = (np.array(column, dtype=float) for column in zip(*table))
    return _lambertw_log_form(L, y0), expected


def test_positive_branch_matches_reference():
    y, expected = _solve_forward(POSITIVE, 2e15)
    np.testing.assert_allclose(y, expected, rtol=1e-14, atol=0)


def test_negative_branch_matches_reference():
    """Close to y = -1 the solution is ill-conditioned: an error of eps in L moves y by
    about sqrt(eps), so the last points get a looser tolerance."""
    y, expected = _solve_forward(NEGATIVE, -1.0 + 1e-15)
    np.testing.assert_allclose(y[:-2], expected[:-2], rtol=1e-14, atol=0)
    np.testing.assert_allclose(y[-2:], expected[-2:], rtol=1e-9, atol=0)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_dense_grid_matches_scipy(sign):
    """Away from the branch point, scipy's complex Lambert W is accurate to rounding."""
    L = np.linspace(-40, 40, 2001) if sign > 0 else np.linspace(-40, -1.1, 2001)
    y = _lambertw_log_form(L, 1e3 if sign > 0 else -1.0 + 1e-15)
    np.testing.assert_allclose(y, lambertw(sign * np.exp(L)).real, rtol=1e-13, atol=0)


def test_backward_in_time():
    """Backward in time y moves away from 0: above y0 for y0 > 0, down to -1 for y0 < 0,
    and stays at -1 past the end of the principal branch."""
    y0 = 0.5
    L = np.log(y0) + y0 + np.array([0.0, 1.0, 10.0])
    y = _lambertw_log_form(L, y0)
    np.testing.assert_allclose(np.log(y) + y, L, rtol=1e-14)
    assert np.all(y >= y0)

    y0 = -0.5
    L0 = np.log(-y0) + y0
    y = _lambertw_log_form(np.array([L0 + 0.1, -1.0, -0.5, 3.0]), y0)
    np.testing.assert_allclose(np.log(-y[0]) + y[0], L0 + 0.1, rtol=1e-14)
    assert -1.0 < y[0] < y0
    np.testing.assert_array_equal(y[1:], -1.0)


def test_solution_stays_between_the_start_and_zero():
    for y0 in (3.0, -0.7):
        L0 = np.log(abs(y0)) + y0
        y = _lambertw_log_form(L0 - np.array([0.0, 1e-12, 1.0, 50.0]), y0)
        assert y[0] == pytest.approx(y0, rel=1e-14)
        assert np.all(np.abs(y) <= abs(y0)) and np.all(np.sign(y) == np.sign(y0))
