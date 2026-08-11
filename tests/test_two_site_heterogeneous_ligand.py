import pytest
import numpy as np

from pykingenie.utils.signal_surface import (
    one_site_association_analytical,
    one_site_dissociation_analytical,
    steady_state_one_site,
    steady_state_two_site_heterogeneous_ligand,
    solve_two_site_heterogeneous_ligand_association,
    solve_two_site_heterogeneous_ligand_dissociation,
)


def test_steady_state_is_weighted_sum_of_two_one_site_curves():
    C = np.logspace(-3, 2, 50)
    Rmax = 10.0
    Kd1 = 0.1
    Kd2 = 5.0
    fraction_site1 = 0.35

    response = steady_state_two_site_heterogeneous_ligand(C, Rmax, Kd1, Kd2, fraction_site1)
    expected = (
        steady_state_one_site(C, Rmax * fraction_site1, Kd1)
        + steady_state_one_site(C, Rmax * (1 - fraction_site1), Kd2)
    )

    assert isinstance(response, np.ndarray)
    np.testing.assert_allclose(response, expected, atol=1e-12)


def test_fraction_limits_reduce_to_single_one_site_curve():
    C = np.logspace(-3, 2, 50)
    Rmax = 8.0
    Kd1 = 0.2
    Kd2 = 3.0

    site1_only = steady_state_two_site_heterogeneous_ligand(C, Rmax, Kd1, Kd2, 1.0)
    site2_only = steady_state_two_site_heterogeneous_ligand(C, Rmax, Kd1, Kd2, 0.0)

    np.testing.assert_allclose(site1_only, steady_state_one_site(C, Rmax, Kd1), atol=1e-12)
    np.testing.assert_allclose(site2_only, steady_state_one_site(C, Rmax, Kd2), atol=1e-12)


def test_identical_sites_with_half_fraction_reduce_to_one_site_curve():
    C = np.logspace(-3, 2, 50)
    Rmax = 12.0
    Kd = 0.4

    response = steady_state_two_site_heterogeneous_ligand(C, Rmax, Kd, Kd, 0.5)

    np.testing.assert_allclose(response, steady_state_one_site(C, Rmax, Kd), atol=1e-12)


def test_steady_state_high_and_low_concentration_limits():
    Rmax = 10.0
    Kd1 = 0.1
    Kd2 = 5.0
    fraction_site1 = 0.35

    response = steady_state_two_site_heterogeneous_ligand(
        np.array([0.0, 1e8]), Rmax, Kd1, Kd2, fraction_site1
    )

    assert np.isclose(response[0], 0.0)
    assert np.isclose(response[-1], Rmax, rtol=1e-6)


def test_low_concentration_regime_is_dominated_by_tighter_site():
    C = 1e-4
    Rmax = 10.0
    Kd1 = 0.01
    Kd2 = 100.0
    fraction_site1 = 0.5

    site1 = steady_state_one_site(C, Rmax * fraction_site1, Kd1)
    site2 = steady_state_one_site(C, Rmax * (1 - fraction_site1), Kd2)

    assert site1 > 1000 * site2


def test_association_components_match_one_site_curves():
    time = np.linspace(0, 100, 60)
    a_conc = 2.0
    Rmax = 9.0
    fraction_site1 = 0.4
    Kd1, koff1 = 0.2, 0.05
    Kd2, koff2 = 3.0, 0.2

    result = solve_two_site_heterogeneous_ligand_association(
        time, a_conc, Kd1, koff1, Kd2, koff2, Rmax, fraction_site1
    )

    expected_site1 = one_site_association_analytical(
        time, 0, Rmax * fraction_site1, koff1, Kd1, a_conc
    )
    expected_site2 = one_site_association_analytical(
        time, 0, Rmax * (1 - fraction_site1), koff2, Kd2, a_conc
    )

    assert result.shape == (len(time), 3)
    np.testing.assert_allclose(result[:, 1], expected_site1, atol=1e-12)
    np.testing.assert_allclose(result[:, 2], expected_site2, atol=1e-12)
    np.testing.assert_allclose(result[:, 0], expected_site1 + expected_site2, atol=1e-12)


def test_association_converges_to_steady_state():
    time = np.linspace(0, 5000, 500)
    a_conc = 1.5
    Rmax = 7.0
    fraction_site1 = 0.25
    Kd1, koff1 = 0.2, 0.05
    Kd2, koff2 = 2.0, 0.2

    result = solve_two_site_heterogeneous_ligand_association(
        time, a_conc, Kd1, koff1, Kd2, koff2, Rmax, fraction_site1
    )
    expected = steady_state_two_site_heterogeneous_ligand(
        a_conc, Rmax, Kd1, Kd2, fraction_site1
    )

    assert np.isclose(result[-1, 0], expected, rtol=1e-6)


def test_association_with_initial_components_starts_at_component_sum():
    time = np.linspace(0, 100, 60)
    s1_0 = 1.2
    s2_0 = 0.7

    result = solve_two_site_heterogeneous_ligand_association(
        time,
        a_conc=2.0,
        Kd1=0.2,
        koff1=0.05,
        Kd2=3.0,
        koff2=0.2,
        Rmax=9.0,
        fraction_site1=0.4,
        s1_0=s1_0,
        s2_0=s2_0,
    )

    assert np.isclose(result[0, 0], s1_0 + s2_0)
    assert np.isclose(result[0, 1], s1_0)
    assert np.isclose(result[0, 2], s2_0)


def test_association_components_are_consistent_and_non_negative():
    time = np.linspace(0, 100, 60)

    result = solve_two_site_heterogeneous_ligand_association(
        time,
        a_conc=2.0,
        Kd1=0.2,
        koff1=0.05,
        Kd2=3.0,
        koff2=0.2,
        Rmax=9.0,
        fraction_site1=0.4,
    )

    np.testing.assert_allclose(result[:, 0], result[:, 1] + result[:, 2], atol=1e-12)
    assert np.all(result >= -1e-12)


def test_dissociation_components_match_one_site_curves():
    time = np.linspace(0, 100, 60)
    s0 = 7.5
    fraction_site1 = 0.3
    koff1 = 0.03
    koff2 = 0.2

    result = solve_two_site_heterogeneous_ligand_dissociation(
        time, koff1, koff2, s0=s0, fraction_site1=fraction_site1
    )

    expected_site1 = one_site_dissociation_analytical(time, s0 * fraction_site1, koff1)
    expected_site2 = one_site_dissociation_analytical(time, s0 * (1 - fraction_site1), koff2)

    assert result.shape == (len(time), 3)
    assert np.isclose(result[0, 0], s0)
    np.testing.assert_allclose(result[:, 1], expected_site1, atol=1e-12)
    np.testing.assert_allclose(result[:, 2], expected_site2, atol=1e-12)
    np.testing.assert_allclose(result[:, 0], expected_site1 + expected_site2, atol=1e-12)


def test_dissociation_component_initial_values_override_fraction_split():
    time = np.array([0, 10, 100, 1e6])
    s1_0 = 2.0
    s2_0 = 5.0
    koff1 = 0.03
    koff2 = 0.2

    result = solve_two_site_heterogeneous_ligand_dissociation(
        time, koff1, koff2, fraction_site1=0.9, s1_0=s1_0, s2_0=s2_0
    )

    assert np.isclose(result[0, 0], s1_0 + s2_0)
    assert np.isclose(result[0, 1], s1_0)
    assert np.isclose(result[0, 2], s2_0)
    assert np.isclose(result[-1, 0], 0, atol=1e-6)


def test_dissociation_components_are_consistent_and_non_negative():
    time = np.linspace(0, 100, 60)

    result = solve_two_site_heterogeneous_ligand_dissociation(
        time, koff1=0.03, koff2=0.2, s0=7.5, fraction_site1=0.3
    )

    np.testing.assert_allclose(result[:, 0], result[:, 1] + result[:, 2], atol=1e-12)
    assert np.all(result >= -1e-12)


def test_slower_dissociation_rate_retains_more_component_signal():
    time = np.linspace(0, 100, 60)

    result = solve_two_site_heterogeneous_ligand_dissociation(
        time, koff1=0.01, koff2=0.2, s1_0=3.0, s2_0=3.0
    )

    assert result[-1, 1] > result[-1, 2]


def test_identical_dissociation_rates_reduce_to_single_exponential():
    time = np.linspace(0, 100, 60)
    s0 = 7.5
    koff = 0.04

    result = solve_two_site_heterogeneous_ligand_dissociation(
        time, koff1=koff, koff2=koff, s0=s0, fraction_site1=0.3
    )
    expected = one_site_dissociation_analytical(time, s0, koff)

    np.testing.assert_allclose(result[:, 0], expected, atol=1e-12)


def test_scalar_and_array_steady_state_inputs_work():
    Rmax = 10.0
    Kd1 = 0.1
    Kd2 = 5.0
    fraction_site1 = 0.35

    scalar = steady_state_two_site_heterogeneous_ligand(1.0, Rmax, Kd1, Kd2, fraction_site1)
    array = steady_state_two_site_heterogeneous_ligand(
        np.array([1.0]), Rmax, Kd1, Kd2, fraction_site1
    )

    assert np.isscalar(scalar) or np.asarray(scalar).shape == ()
    np.testing.assert_allclose(array[0], scalar, atol=1e-12)


def test_list_and_array_time_inputs_work():
    time_list = [0, 1, 2, 3]
    time_array = np.array(time_list)

    assoc_list = solve_two_site_heterogeneous_ligand_association(
        time_list, 2.0, 0.2, 0.05, 3.0, 0.2, Rmax=9.0, fraction_site1=0.4
    )
    assoc_array = solve_two_site_heterogeneous_ligand_association(
        time_array, 2.0, 0.2, 0.05, 3.0, 0.2, Rmax=9.0, fraction_site1=0.4
    )
    disso_list = solve_two_site_heterogeneous_ligand_dissociation(
        time_list, 0.03, 0.2, s0=7.5, fraction_site1=0.3
    )
    disso_array = solve_two_site_heterogeneous_ligand_dissociation(
        time_array, 0.03, 0.2, s0=7.5, fraction_site1=0.3
    )

    np.testing.assert_allclose(assoc_list, assoc_array, atol=1e-12)
    np.testing.assert_allclose(disso_list, disso_array, atol=1e-12)


@pytest.mark.parametrize("fraction_site1", [-0.1, 1.1])
def test_invalid_fraction_raises_value_error(fraction_site1):
    with pytest.raises(ValueError, match="fraction_site1 must be between 0 and 1"):
        steady_state_two_site_heterogeneous_ligand(1.0, 10.0, 0.1, 5.0, fraction_site1)

    with pytest.raises(ValueError, match="fraction_site1 must be between 0 and 1"):
        solve_two_site_heterogeneous_ligand_association(
            [0, 1], 2.0, 0.2, 0.05, 3.0, 0.2, Rmax=9.0, fraction_site1=fraction_site1
        )

    with pytest.raises(ValueError, match="fraction_site1 must be between 0 and 1"):
        solve_two_site_heterogeneous_ligand_dissociation(
            [0, 1], 0.03, 0.2, s0=7.5, fraction_site1=fraction_site1
        )
