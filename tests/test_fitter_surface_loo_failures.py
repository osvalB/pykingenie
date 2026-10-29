import pytest

import pykingenie.fitter_surface as fitter_surface_module
from pykingenie.fitter_surface import KineticsFitter

"""
The objective of these test is just to test the exception clause in the leave-one-out (loo)
fitting methods. Real fittings are tested in other test modules.
"""

def _create_loo_fitter():
    fitter = object.__new__(KineticsFitter)
    fitter.assoc_lst = [[1.0], [2.0]]
    fitter.time_assoc_lst = [[0.0], [1.0]]
    fitter.disso_lst = [[1.0], [2.0]]
    fitter.time_disso_lst = [[0.0], [1.0]]
    fitter.disso_lst_subset = fitter.disso_lst
    fitter.time_disso_lst_subset = fitter.time_disso_lst
    fitter.lig_conc_lst = [0.1, 1.0]
    fitter.smax_id = [0, 0]
    fitter.params = [1.0, 2.0, 3.0, 4.0, 5.0]
    fitter.low_bounds = [0.0] * len(fitter.params)
    fitter.high_bounds = [10.0] * len(fitter.params)
    fitter.shared_smax = True
    fitter.fixed_t0 = True
    fitter.fit_ktr = False
    fitter.fit_sigma = False
    fitter.smax_param_start = 2
    return fitter


@pytest.mark.parametrize(
    ("method_name", "fit_function_name", "fit_values", "output_count", "expected"),
    [
        (
            "one_site_dissociation_loo",
            "fit_one_site_dissociation",
            [0.25, 1.0],
            3,
            {"k_off_loo": [0.25]},
        ),
        (
            "one_site_assoc_and_disso_loo",
            "fit_one_site_assoc_and_disso",
            [4.0, 5.0],
            4,
            {"Kd_loo": [4.0], "k_off_loo": [5.0]},
        ),
        (
            "two_site_assoc_and_disso_loo",
            "fit_two_site_assoc_and_disso",
            [0.3, 0.04],
            4,
            {"Kd_loo": [0.3], "k_off_loo": [0.04]},
        ),
        (
            "two_site_heterogeneous_ligand_assoc_and_disso_loo",
            "fit_two_site_heterogeneous_ligand_assoc_and_disso",
            [0.1, 0.02, 2.0, 0.2, 0.4],
            4,
            {
                "Kd1_loo": [0.1],
                "k_off1_loo": [0.02],
                "Kd2_loo": [2.0],
                "k_off2_loo": [0.2],
                "fraction_site1_loo": [0.4],
            },
        ),
        (
            "one_site_if_assoc_and_disso_loo",
            "fit_induced_fit_sites_assoc_and_disso",
            [0.1, 0.02, 1.0, 10.0],
            4,
            {
                "k_on_loo": [0.1],
                "k_off_loo": [0.02],
                "k_c_loo": [1.0],
                "k_rev_loo": [10.0],
            },
        ),
    ],
)
def test_loo_skips_failed_fit_and_keeps_successful_values(
    monkeypatch, method_name, fit_function_name, fit_values, output_count, expected
):
    fitter = _create_loo_fitter()
    attempts = []

    def fake_fit(*args, **kwargs):
        attempts.append(None)
        if len(attempts) == 1:
            raise RuntimeError("simulated LOO fit failure")
        return (fit_values, *([None] * (output_count - 1)))

    monkeypatch.setattr(fitter_surface_module, fit_function_name, fake_fit)

    getattr(fitter, method_name)()

    assert len(attempts) == 2
    for attribute, values in expected.items():
        assert getattr(fitter, attribute) == values