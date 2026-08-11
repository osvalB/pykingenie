import pytest
import numpy as np

from pykingenie.fitter_surface import KineticsFitter
from pykingenie.utils.signal_surface import (
    solve_two_site_heterogeneous_ligand_association,
    solve_two_site_heterogeneous_ligand_dissociation,
)


def _create_heterogeneous_ligand_kinetic_fitter(
    smax_id=None,
    name_lst=None,
    set_kd_ss=True,
    rmax_values=None,
    t0_values=None,
):
    Kd1_true = 0.1
    koff1_true = 0.03
    Kd2_true = 5.0
    koff2_true = 0.2
    fraction_true = 0.35

    smax_id = np.array([0, 0, 0]) if smax_id is None else np.array(smax_id)
    lig_conc_lst = np.array([0.05, 0.5, 10.0, 25.0])[:len(smax_id)]
    time_assoc_lst = [np.linspace(0, 160, 80) for _ in range(len(smax_id))]
    time_disso_lst = [np.linspace(0, 160, 80) for _ in range(len(smax_id))]

    if rmax_values is None:
        rmax_values = [10.0 for _ in range(len(smax_id))]
    if t0_values is None:
        t0_values = [0.0 for _ in range(len(np.unique(smax_id)))]

    assoc_lst = []
    disso_lst = []
    for i, (t_assoc, t_disso, conc) in enumerate(zip(time_assoc_lst, time_disso_lst, lig_conc_lst)):
        assoc = solve_two_site_heterogeneous_ligand_association(
            t_assoc,
            conc,
            Kd1_true,
            koff1_true,
            Kd2_true,
            koff2_true,
            Rmax=rmax_values[i],
            fraction_site1=fraction_true,
            t0=t0_values[smax_id[i]],
        )
        disso = solve_two_site_heterogeneous_ligand_dissociation(
            t_disso,
            koff1_true,
            koff2_true,
            fraction_site1=fraction_true,
            s1_0=assoc[-1, 1],
            s2_0=assoc[-1, 2],
        )
        assoc_lst.append(assoc[:, 0])
        disso_lst.append(disso[:, 0])

    fitter_surface = KineticsFitter(
        time_assoc_lst=time_assoc_lst,
        association_signal_lst=assoc_lst,
        lig_conc_lst=lig_conc_lst,
        time_diss_lst=time_disso_lst,
        dissociation_signal_lst=disso_lst,
        smax_id=smax_id,
        name_lst=name_lst,
    )
    if set_kd_ss:
        fitter_surface.Kd_ss = 1.0
    fitter_surface.Smax_upper_bound_factor = 100

    truth = {
        "Kd1": Kd1_true,
        "koff1": koff1_true,
        "Kd2": Kd2_true,
        "koff2": koff2_true,
        "fraction_site1": fraction_true,
        "Rmax": rmax_values,
        "t0": t0_values,
    }

    return fitter_surface, truth


class _PreseededInvalidSignalFitter(KineticsFitter):
    def fit_steady_state_one_site(self):
        self.Kd_ss = 1.0

    def get_steady_state(self):
        self.smax_guesses_unq = [10.0]
        self.smax_guesses_shared = [10.0 for _ in self.smax_id]

    def fit_one_site_assoc_and_disso(self, shared_smax=True, fixed_t0=True, fit_ktr=False):
        self.Kd = 1.0
        self.k_off = 0.1
        self.Smax = [10.0]


def _create_invalid_signal_fitter_for_grid_failure():
    time_assoc_lst = [np.linspace(0.0, 1.0, 6)]
    time_disso_lst = [np.linspace(0.0, 1.0, 6)]
    assoc_lst = [np.full(6, np.nan)]
    disso_lst = [np.full(6, np.nan)]

    return _PreseededInvalidSignalFitter(
        time_assoc_lst=time_assoc_lst,
        association_signal_lst=assoc_lst,
        lig_conc_lst=[0.1],
        time_diss_lst=time_disso_lst,
        dissociation_signal_lst=disso_lst,
        smax_id=np.array([0]),
    )


def test_fit_two_site_heterogeneous_ligand_assoc_and_disso_grid_search():
    fitter_surface, truth = _create_heterogeneous_ligand_kinetic_fitter(name_lst=["synthetic"])

    fitter_surface.fit_two_site_heterogeneous_ligand_assoc_and_disso(
        shared_smax=True,
        fixed_t0=True,
        Kd1_values=[0.1, 0.5, 10.0],
        Kd2_values=[0.05, 1.0, 5.0],
    )

    assert fitter_surface.best_Kd1_grid < fitter_surface.best_Kd2_grid
    assert np.all(
        fitter_surface.heterogeneous_ligand_grid_search["Kd1_value"]
        < fitter_surface.heterogeneous_ligand_grid_search["Kd2_value"]
    )

    assert np.isclose(fitter_surface.Kd1, truth["Kd1"], rtol=0.1)
    assert np.isclose(fitter_surface.k_off1, truth["koff1"], rtol=0.1)
    assert np.isclose(fitter_surface.Kd2, truth["Kd2"], rtol=0.1)
    assert np.isclose(fitter_surface.k_off2, truth["koff2"], rtol=0.1)
    assert np.isclose(fitter_surface.fraction_site1, truth["fraction_site1"], rtol=0.1)
    assert np.isclose(fitter_surface.Smax[0], truth["Rmax"][0], rtol=0.1)

    df_fit = fitter_surface.fit_params_kinetics
    for column in [
        "Kd1 [µM]",
        "k_off1 [1/s]",
        "Kd2 [µM]",
        "k_off2 [1/s]",
        "fraction_site1",
        "Rmax",
        "(Derived) k_on1 [1/µM/s]",
        "(Derived) k_on2 [1/µM/s]",
    ]:
        assert column in df_fit.columns

    df = fitter_surface.create_export_df(type="fit")
    assert set(df["Type"]) == {"Association", "Dissociation"}


def test_fit_two_site_heterogeneous_ligand_grid_requires_kd1_lower_than_kd2():
    fitter_surface, _ = _create_heterogeneous_ligand_kinetic_fitter()

    with pytest.raises(ValueError, match="Kd1 < Kd2"):
        fitter_surface.fit_two_site_heterogeneous_ligand_assoc_and_disso(
            Kd1_values=[5.0, 10.0],
            Kd2_values=[0.1, 1.0],
        )


def test_fit_two_site_heterogeneous_ligand_uses_defaults_and_shared_t0():
    fitter_surface, truth = _create_heterogeneous_ligand_kinetic_fitter(
        name_lst=None,
        set_kd_ss=False,
        t0_values=[0.01],
    )

    fitter_surface.fit_two_site_heterogeneous_ligand_assoc_and_disso(
        shared_smax=True,
        fixed_t0=False,
    )

    assert fitter_surface.Kd_ss is not None
    assert len(fitter_surface.heterogeneous_ligand_grid_search) > 1
    assert fitter_surface.fit_params_kinetics["Name"].tolist() == ["group_0"]
    assert "t0" in fitter_surface.fit_params_kinetics.columns
    assert np.isclose(fitter_surface.fit_params_kinetics["t0"].iloc[0], truth["t0"][0], atol=0.02)
    assert len(fitter_surface.p0) == 7


def test_fit_two_site_heterogeneous_ligand_unshared_t0_expands_single_name():
    fitter_surface, truth = _create_heterogeneous_ligand_kinetic_fitter(
        smax_id=[0, 1],
        name_lst=["sample"],
        rmax_values=[8.0, 12.0],
        t0_values=[0.01, 0.02],
    )

    fitter_surface.fit_two_site_heterogeneous_ligand_assoc_and_disso(
        shared_smax=False,
        fixed_t0=False,
        Kd1_values=[0.1],
        Kd2_values=[5.0],
    )

    assert fitter_surface.fit_params_kinetics["Name"].tolist() == ["sample", "sample"]
    assert "t0" in fitter_surface.fit_params_kinetics.columns
    assert np.allclose(fitter_surface.fit_params_kinetics["t0"], truth["t0"], atol=0.03)
    assert fitter_surface.fit_params_kinetics_error["t0"].notna().all()
    assert np.allclose(fitter_surface.Smax, truth["Rmax"], rtol=0.1)


def test_fit_two_site_heterogeneous_ligand_raises_when_grid_fits_all_fail():
    fitter_surface = _create_invalid_signal_fitter_for_grid_failure()

    with pytest.raises(RuntimeError, match="Grid search failed"):
        fitter_surface.fit_two_site_heterogeneous_ligand_assoc_and_disso(
            Kd1_values=[0.1],
            Kd2_values=[5.0],
        )
