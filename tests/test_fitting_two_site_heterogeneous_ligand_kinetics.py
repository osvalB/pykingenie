import numpy as np

from pykingenie.utils.fitting_surface import fit_two_site_heterogeneous_ligand_assoc_and_disso
from pykingenie.utils.signal_surface import (
    solve_two_site_heterogeneous_ligand_association,
    solve_two_site_heterogeneous_ligand_dissociation,
)


def _simulate_trace(time_assoc, time_disso, analyte_conc, Kd1, koff1, Kd2, koff2, Rmax, fraction_site1, s1_0=0.0, s2_0=0.0):
    assoc = solve_two_site_heterogeneous_ligand_association(
        time_assoc - time_assoc[0],
        analyte_conc,
        Kd1,
        koff1,
        Kd2,
        koff2,
        Rmax=Rmax,
        fraction_site1=fraction_site1,
        s1_0=s1_0,
        s2_0=s2_0,
    )
    disso = solve_two_site_heterogeneous_ligand_dissociation(
        time_disso - time_disso[0],
        koff1,
        koff2,
        fraction_site1=fraction_site1,
        s1_0=assoc[-1, 1],
        s2_0=assoc[-1, 2],
    )
    return assoc, disso


def test_fit_heterogeneous_ligand_kinetics_multi_concentration_recovery():
    Kd1_true = 0.1
    koff1_true = 0.03
    Kd2_true = 5.0
    koff2_true = 0.2
    fraction_true = 0.35
    Rmax_true = 10.0

    time_assoc = [np.linspace(0, 160, 100) for _ in range(3)]
    time_disso = [np.linspace(0, 160, 100) for _ in range(3)]
    analyte_conc = [0.05, 0.5, 10.0]

    assoc_signal_lst = []
    disso_signal_lst = []
    for t_assoc, t_disso, conc in zip(time_assoc, time_disso, analyte_conc):
        assoc, disso = _simulate_trace(
            t_assoc,
            t_disso,
            conc,
            Kd1_true,
            koff1_true,
            Kd2_true,
            koff2_true,
            Rmax_true,
            fraction_true,
        )
        assoc_signal_lst.append(assoc[:, 0])
        disso_signal_lst.append(disso[:, 0])

    fit, _, fit_assoc, fit_disso = fit_two_site_heterogeneous_ligand_assoc_and_disso(
        assoc_signal_lst=assoc_signal_lst,
        assoc_time_lst=time_assoc,
        analyte_conc_lst=analyte_conc,
        disso_signal_lst=disso_signal_lst,
        disso_time_lst=time_disso,
        initial_parameters=[0.2, 0.05, 3.0, 0.1, 0.5, 8.0],
        low_bounds=[1e-3, 1e-3, 1.0, 1e-3, 0.0, 0.0],
        high_bounds=[0.5, 1.0, 50.0, 1.0, 1.0, 100.0],
        smax_idx=[0, 0, 0],
        shared_smax=True,
        fixed_t0=True,
    )

    assert np.isclose(fit[0], Kd1_true, rtol=0.1)
    assert np.isclose(fit[1], koff1_true, rtol=0.1)
    assert np.isclose(fit[2], Kd2_true, rtol=0.1)
    assert np.isclose(fit[3], koff2_true, rtol=0.1)
    assert np.isclose(fit[4], fraction_true, rtol=0.1)
    assert np.isclose(fit[5], Rmax_true, rtol=0.1)
    for expected, fitted in zip(assoc_signal_lst, fit_assoc):
        np.testing.assert_allclose(fitted, expected, atol=1e-6)
    for expected, fitted in zip(disso_signal_lst, fit_disso):
        np.testing.assert_allclose(fitted, expected, atol=1e-6)


def test_fit_heterogeneous_ligand_kinetics_fixed_shape_recovers_trace_rmax_values():
    Kd1_true = 0.12
    koff1_true = 0.04
    Kd2_true = 4.5
    koff2_true = 0.18
    fraction_true = 0.4
    rmax_values = [8.0, 14.0]
    analyte_conc = [0.3, 6.0]

    time_assoc = [np.linspace(0, 140, 90), np.linspace(0, 140, 90)]
    time_disso = [np.linspace(0, 140, 90), np.linspace(0, 140, 90)]

    assoc_signal_lst = []
    disso_signal_lst = []
    for t_assoc, t_disso, conc, Rmax in zip(time_assoc, time_disso, analyte_conc, rmax_values):
        assoc, disso = _simulate_trace(
            t_assoc,
            t_disso,
            conc,
            Kd1_true,
            koff1_true,
            Kd2_true,
            koff2_true,
            Rmax,
            fraction_true,
        )
        assoc_signal_lst.append(assoc[:, 0])
        disso_signal_lst.append(disso[:, 0])

    fit, _, fit_assoc, fit_disso = fit_two_site_heterogeneous_ligand_assoc_and_disso(
        assoc_signal_lst=assoc_signal_lst,
        assoc_time_lst=time_assoc,
        analyte_conc_lst=analyte_conc,
        disso_signal_lst=disso_signal_lst,
        disso_time_lst=time_disso,
        initial_parameters=[7.0, 12.0],
        low_bounds=[0.0, 0.0],
        high_bounds=[100.0, 100.0],
        smax_idx=[0, 1],
        shared_smax=False,
        fixed_t0=True,
        fixed_Kd1=True,
        Kd1_value=Kd1_true,
        fixed_koff1=True,
        koff1_value=koff1_true,
        fixed_Kd2=True,
        Kd2_value=Kd2_true,
        fixed_koff2=True,
        koff2_value=koff2_true,
        fixed_fraction_site1=True,
        fraction_site1_value=fraction_true,
    )

    assert np.isclose(fit[0], rmax_values[0], rtol=1e-3)
    assert np.isclose(fit[1], rmax_values[1], rtol=1e-3)
    for expected, fitted in zip(assoc_signal_lst, fit_assoc):
        np.testing.assert_allclose(fitted, expected, atol=1e-8)
    for expected, fitted in zip(disso_signal_lst, fit_disso):
        np.testing.assert_allclose(fitted, expected, atol=1e-8)


def test_fit_heterogeneous_ligand_kinetics_continuous_cycle_carries_component_state():
    Kd1_true = 0.1
    koff1_true = 0.03
    Kd2_true = 5.0
    koff2_true = 0.2
    fraction_true = 0.35
    Rmax_true = 10.0

    time_assoc_1 = np.linspace(0, 80, 70)
    time_disso_1 = np.linspace(0, 80, 70)
    time_assoc_2 = np.linspace(81, 161, 70)
    time_disso_2 = np.linspace(0, 80, 70)

    assoc_1, disso_1 = _simulate_trace(
        time_assoc_1,
        time_disso_1,
        0.5,
        Kd1_true,
        koff1_true,
        Kd2_true,
        koff2_true,
        Rmax_true,
        fraction_true,
    )
    assoc_2, disso_2 = _simulate_trace(
        time_assoc_2,
        time_disso_2,
        0.5,
        Kd1_true,
        koff1_true,
        Kd2_true,
        koff2_true,
        Rmax_true,
        fraction_true,
        s1_0=disso_1[-1, 1],
        s2_0=disso_1[-1, 2],
    )

    fit, _, fit_assoc, fit_disso = fit_two_site_heterogeneous_ligand_assoc_and_disso(
        assoc_signal_lst=[assoc_1[:, 0], assoc_2[:, 0]],
        assoc_time_lst=[time_assoc_1, time_assoc_2],
        analyte_conc_lst=[0.5, 0.5],
        disso_signal_lst=[disso_1[:, 0], disso_2[:, 0]],
        disso_time_lst=[time_disso_1, time_disso_2],
        initial_parameters=[8.0],
        low_bounds=[0.0],
        high_bounds=[100.0],
        smax_idx=[0, 0],
        shared_smax=True,
        fixed_t0=True,
        fixed_Kd1=True,
        Kd1_value=Kd1_true,
        fixed_koff1=True,
        koff1_value=koff1_true,
        fixed_Kd2=True,
        Kd2_value=Kd2_true,
        fixed_koff2=True,
        koff2_value=koff2_true,
        fixed_fraction_site1=True,
        fraction_site1_value=fraction_true,
    )

    assert np.isclose(fit[0], Rmax_true, rtol=1e-3)
    np.testing.assert_allclose(fit_assoc[0], assoc_1[:, 0], atol=1e-8)
    np.testing.assert_allclose(fit_disso[0], disso_1[:, 0], atol=1e-8)
    np.testing.assert_allclose(fit_assoc[1], assoc_2[:, 0], atol=1e-8)
    np.testing.assert_allclose(fit_disso[1], disso_2[:, 0], atol=1e-8)


def test_fit_heterogeneous_ligand_kinetics_default_indices_t0_and_noncontinuous_branch():
    Kd1_true = 0.1
    koff1_true = 0.03
    Kd2_true = 5.0
    koff2_true = 0.2
    fraction_true = 0.35
    t0_1 = 0.1
    t0_2 = -0.1
    rmax_1 = 8.0
    rmax_2 = 12.0

    time_assoc_1 = np.linspace(0, 80, 70)
    time_disso_1 = np.linspace(0, 80, 70)
    time_assoc_2 = np.linspace(200, 280, 70)
    time_disso_2 = np.linspace(0, 80, 70)

    assoc_1 = solve_two_site_heterogeneous_ligand_association(
        time_assoc_1 - time_assoc_1[0],
        0.5,
        Kd1_true,
        koff1_true,
        Kd2_true,
        koff2_true,
        Rmax=rmax_1,
        fraction_site1=fraction_true,
        t0=t0_1,
    )
    disso_1 = solve_two_site_heterogeneous_ligand_dissociation(
        time_disso_1 - time_disso_1[0],
        koff1_true,
        koff2_true,
        fraction_site1=fraction_true,
        s1_0=assoc_1[-1, 1],
        s2_0=assoc_1[-1, 2],
    )
    assoc_2 = solve_two_site_heterogeneous_ligand_association(
        time_assoc_2 - time_assoc_2[0],
        0.5,
        Kd1_true,
        koff1_true,
        Kd2_true,
        koff2_true,
        Rmax=rmax_2,
        fraction_site1=fraction_true,
        t0=t0_2,
    )
    disso_2 = solve_two_site_heterogeneous_ligand_dissociation(
        time_disso_2 - time_disso_2[0],
        koff1_true,
        koff2_true,
        fraction_site1=fraction_true,
        s1_0=assoc_2[-1, 1],
        s2_0=assoc_2[-1, 2],
    )

    fit, _, fit_assoc, fit_disso = fit_two_site_heterogeneous_ligand_assoc_and_disso(
        assoc_signal_lst=[assoc_1[:, 0], assoc_2[:, 0]],
        assoc_time_lst=[time_assoc_1, time_assoc_2],
        analyte_conc_lst=[0.5, 0.5],
        disso_signal_lst=[disso_1[:, 0], disso_2[:, 0]],
        disso_time_lst=[time_disso_1, time_disso_2],
        initial_parameters=[0.0, 0.0, 7.0, 11.0],
        low_bounds=[-0.5, -0.5, 0.0, 0.0],
        high_bounds=[0.5, 0.5, 100.0, 100.0],
        shared_smax=False,
        fixed_t0=False,
        fixed_Kd1=True,
        Kd1_value=Kd1_true,
        fixed_koff1=True,
        koff1_value=koff1_true,
        fixed_Kd2=True,
        Kd2_value=Kd2_true,
        fixed_koff2=True,
        koff2_value=koff2_true,
        fixed_fraction_site1=True,
        fraction_site1_value=fraction_true,
    )

    assert np.isclose(fit[0], t0_1, atol=1e-3)
    assert np.isclose(fit[1], t0_2, atol=1e-3)
    assert np.isclose(fit[2], rmax_1, rtol=1e-3)
    assert np.isclose(fit[3], rmax_2, rtol=1e-3)
    np.testing.assert_allclose(fit_assoc[0], assoc_1[:, 0], atol=1e-8)
    np.testing.assert_allclose(fit_disso[0], disso_1[:, 0], atol=1e-8)
    np.testing.assert_allclose(fit_assoc[1], assoc_2[:, 0], atol=1e-8)
    np.testing.assert_allclose(fit_disso[1], disso_2[:, 0], atol=1e-8)
