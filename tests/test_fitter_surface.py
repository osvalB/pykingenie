import pytest
import numpy as np

from pykingenie.main                    import KineticsAnalyzer
from pykingenie.kingenie_surface        import KinGenieCsv
from pykingenie.fitter_surface                  import KineticsFitter
from pykingenie.utils.fitting_general   import re_fit

from pykingenie.utils.fitting_surface   import (
    fit_steady_state_one_site,
    get_smax_upper_bound_factor
)

test_file_1 = "./test_files/simulation_Kd-10_koff-0.01.csv"
test_file_2 = "./test_files/simulation_Kd-0.5_koff-0.01_Ktr-0.005.csv"
test_file_3 = "./test_files/simulation_kon-0.1_koff-0.01_kc-1_krev-10.csv"

### Obtain a fitter instance

def create_fitter(file):

    kingenie = KinGenieCsv()
    kingenie.read_csv(file)

    pyKinetics = KineticsAnalyzer()
    pyKinetics.add_experiment(kingenie, 'test_surface')

    pyKinetics.merge_ligand_conc_df()

    df = pyKinetics.combined_ligand_conc_df

    pyKinetics.generate_fittings(df)

    fitter_surface = list(pyKinetics.fittings.values())[0]

    # Subset the data to include less points for faster - testing

    fitter_surface.assoc_lst = [x[::4] for x in fitter_surface.assoc_lst]
    fitter_surface.disso_lst =  [x[::4] for x in fitter_surface.disso_lst]
    fitter_surface.time_assoc_lst =  [x[::4] for x in fitter_surface.time_assoc_lst]
    fitter_surface.time_disso_lst =  [x[::4] for x in fitter_surface.time_disso_lst]

    return fitter_surface

## End of obtaining a fitter instance

def test_fitter_is_instance():
    """
    Test if the fitter is an instance of KinGenieFitter
    """
    fitter_surface = create_fitter(test_file_1)
    assert isinstance(fitter_surface, KineticsFitter), "The fitter should be an instance of KinGenieCsv."

def test_get_k_off_initial_guess():

    fitter_surface = KineticsFitter(
        time_assoc_lst=[[1,2,3],[1,2,3],[1,2,3]],
        association_signal_lst=[[1,2,3],[1,2,3],[1,2,3]],
        lig_conc_lst=[1,2,3]
    )

    # Verify raise ValueError if Kd_ss is not set
    with pytest.raises(ValueError):
        fitter_surface.get_k_off_initial_guess()

    # Set single cycle to True and run the method again - Kd_ss should be the median of the ligand concentrations
    fitter_surface.is_single_cycle = True

    fitter_surface.get_k_off_initial_guess()

    # Verify that self.Kd_ss is 2
    assert fitter_surface.Kd_ss == 2, f"Expected Kd_ss to be 2, got {fitter_surface.Kd_ss}."

    # Now set single cycle to False and run the method again with a predefined Kd_ss
    fitter_surface.is_single_cycle = False
    fitter_surface.Kd_ss = 5
    p0, low_bounds, high_bounds = fitter_surface.get_k_off_initial_guess()

    # Verify that p0[0] is 5
    assert p0[0] == 5, f"Expected p0[0] to be 5, got {p0[0]}."


def test_fit_steady_state():
    """
    Test the fitting of steady state data -
    """
    # Assuming the fitter has a method to fit steady state data
    fitter_surface = create_fitter(test_file_1)
    fitter_surface.signal_ss = None # force running self.fit_steady_state()
    fitter_surface.fit_steady_state()

    # Check if the fitting was successful - Kd should be between 9 and 11
    assert 9 < fitter_surface.Kd_ss < 11, f"Expected Kd to be between 9 and 11, got {fitter_surface.Kd}"

    fitter_surface.create_fitting_bounds_table()

    # Check if the bounds table is created correctly - the first column values should be higher than the second column values
    assert all(fitter_surface.fitted_params_boundaries.iloc[:, 0] > fitter_surface.fitted_params_boundaries.iloc[:, 1]), \
           "The first column (fitted values) of the bounds table should be higher than the second column (low bounds)."

    # Check if the fitted parameters are within the expected range
    assert all(fitter_surface.fitted_params_boundaries.iloc[:, 0] < fitter_surface.fitted_params_boundaries.iloc[:, 2]), \
           "The first column (fitted values) of the bounds table should be higher than the second column (high bounds)."

def test_re_fit():

    fitter_surface = create_fitter(test_file_1)

    signal_ss = fitter_surface.signal_ss
    lig_conc_lst_per_id = fitter_surface.lig_conc_lst_per_id

    p0 = [10,3]
    low_bounds = [0,0]
    high_bounds = [100,4]

    fit, cov, fit_vals = fit_steady_state_one_site(
        signal_ss, lig_conc_lst_per_id,
        p0, low_bounds, high_bounds)

    assert fit is not None, "The fit should not be None."

    # The Smax parameter should be close to 4 - it is actually 5 but we set wrong bounds
    np.testing.assert_almost_equal(fit[1], 4, decimal=1, err_msg="The Smax parameter should be close to 4.")

    kwargs = {
        'signal_lst': signal_ss,
        'ligand_lst': lig_conc_lst_per_id
    }

    # Now we re-fit the data
    fit, cov, fit_vals, low_bounds, high_bounds = re_fit(
        fit=fit,
        cov=cov,
        fit_vals=fit_vals,
        fit_fx=fit_steady_state_one_site,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        times=1,
        **kwargs
    )

    # Assert that Smax is now 5
    np.testing.assert_almost_equal(fit[1], 5, decimal=1, err_msg="The Smax parameter should be close to 5 after re-fitting.")

    ################ Now we test the re-fit with wrong lower bounds ########

    p0 = [50,3]
    low_bounds = [40,0]
    high_bounds = [100,8]

    fit, cov, fit_vals = fit_steady_state_one_site(
        signal_ss, lig_conc_lst_per_id,
        p0, low_bounds, high_bounds)

    assert fit is not None, "The fit should not be None."

    # The Kd should be close to 40, it is actually 10 but we set wrong bounds
    np.testing.assert_almost_equal(fit[0], 40, decimal=1, err_msg="The Kd parameter should be close to 40.")

    # Now we re-fit the data
    fit, cov, fit_vals, low_bounds, high_bounds = re_fit(
        fit,
        cov,
        fit_vals,
        fit_steady_state_one_site,
        low_bounds,
        high_bounds,
        times=1,
        **kwargs
    )

    # Assert that Kd is between 9 and 11
    assert 9 < fit[0] < 11, f"Expected Kd to be between 9 and 11, got {fit[0]}."

def test_get_smax_upper_bound_factor():

    val = get_smax_upper_bound_factor(0.1)

    assert  val == 50, f"Expected Smax upper bound factor to be 5, got {val}."

    val = get_smax_upper_bound_factor(5)

    assert  val == 1e2, f"Expected Smax upper bound factor to be 100, got {val}."

    val = get_smax_upper_bound_factor(80)

    assert  val == 1e3, f"Expected Smax upper bound factor to be 1e3, got {val}."

def test_fit_one_site_association():

    fitter_surface = create_fitter(test_file_1)

    # catch value error if we did not obtain Kd_ss first
    with pytest.raises(ValueError):

        fitter_surface.fit_one_site_association(shared_smax=True)

    fitter_surface.fit_steady_state()
    fitter_surface.fit_one_site_association(shared_smax=True)
    # Check if the fitting was successful - Kd should be between 9 and 11
    assert 9 < fitter_surface.Kd < 11, f"Expected Kd to be between 9 and 11, got {fitter_surface.Kd}."

    fitter_surface.fit_steady_state()
    fitter_surface.fit_one_site_association(shared_smax=False)

    # Check if the fitting was successful - Kd should be between 9 and 11
    assert 9 < fitter_surface.Kd < 11, f"Expected Kd to be between 9 and 11, got {fitter_surface.Kd}."

def test_fit_one_site_dissociation_with_timelimit():

    fitter_surface = create_fitter(test_file_1)
    fitter_surface.fit_one_site_dissociation(time_limit=60)

    # Check if the fitting was successful - k_off should be between 0.009 and 0.011
    assert 0.009 < fitter_surface.k_off < 0.011, f"Expected k_off to be between 0.009 and 0.011, got {fitter_surface.k_off}."

def test_fit_single_exponentials():

    fitter_surface = create_fitter(test_file_1)
    fitter_surface.fit_single_exponentials()

    expected = [0.010200452389933872, 0.010440995257854515, 0.010970189567279933, 0.012134417048015848,
              0.014695717505634857, 0.020330578512396707, 0.032727272727272744, 0.05999999999999999]

    np.testing.assert_almost_equal(
        fitter_surface.k_obs,
        expected,
        decimal=3,
        err_msg="The k_obs values should be close to the expected values."
    )

def test_fit_one_site_assoc_and_disso_0():

    fitter_surface = create_fitter(test_file_1)
    fitter_surface.fit_steady_state()

    # fit with shared_smax=True, fixed_t0=True and fit_ktr=False
    fitter_surface.fit_one_site_assoc_and_disso(shared_smax=True,fixed_t0=True,fit_ktr=False)

    # Compute the asymmetric interval for Kd
    fitter_surface.calculate_ci95(shared_smax=True,fixed_t0=True,fit_ktr=False)

    ci95 = fitter_surface.fit_params_kinetics_ci95

    assert all(ci95.iloc[:, 1] < ci95.iloc[:, 2]), \
              "The lower bound of the 95% CI should be less than the upper bound."

    # fit with shared_smax=False, fixed_t0=True and fit_ktr=False
    fitter_surface.fit_one_site_assoc_and_disso(shared_smax=False,fixed_t0=True,fit_ktr=False)
    fitter_surface.calculate_ci95(shared_smax=False,fixed_t0=True,fit_ktr=False)

    ci95 = fitter_surface.fit_params_kinetics_ci95

    assert all(ci95.iloc[:, 1] < ci95.iloc[:, 2]), \
              "The lower bound of the 95% CI should be less than the upper bound."

    # fit with shared_smax=False, fixed_t0=False and fit_ktr=False
    fitter_surface.fit_one_site_assoc_and_disso(shared_smax=False,fixed_t0=False,fit_ktr=False)

    # Verify that the number of fitted parameters is correct:
    # we expect smax, Kd, koff plus one t0 per curve
    n_fitted_params = len(fitter_surface.params)

    assert n_fitted_params == 3 + len(fitter_surface.lig_conc_lst), \
           f"Expected 3 + {len(fitter_surface.lig_conc_lst)} t0 parameters, got {n_fitted_params}."

def test_fit_one_site_assoc_and_disso_1():

    fitter_surface = create_fitter(test_file_2)

    # catch value error if we did not obtain Kd_ss first
    with pytest.raises(ValueError):
        fitter_surface.fit_one_site_assoc_and_disso(shared_smax=True,fixed_t0=True,fit_ktr=True)

    fitter_surface.Kd_ss = 0.5
    fitter_surface.Smax_upper_bound_factor = 50

    fitter_surface.fit_one_site_assoc_and_disso(shared_smax=True, fixed_t0=True, fit_ktr=True)

    # Check if ktr is between 0.004 and 0.006
    ktr_fitted = fitter_surface.fit_params_kinetics['Ktr'].iloc[0]

    assert 0.004 < ktr_fitted < 0.006, f"Expected ktr to be between 0.004 and 0.006, got {ktr_fitted}."

def test_fit_one_site_assoc_and_disso_2():

    fitter_surface = create_fitter(test_file_2)
    fitter_surface.Kd_ss = 0.5
    fitter_surface.Smax_upper_bound_factor = 50

    # Now we test the fitting with not shared Smax
    fitter_surface.fit_one_site_assoc_and_disso(shared_smax=False, fixed_t0=True, fit_ktr=True)

    # Check if ktr is between 0.004 and 0.006
    ktr_fitted = fitter_surface.fit_params_kinetics['Ktr'].iloc[0]

    assert 0.004 < ktr_fitted < 0.006, f"Expected ktr to be between 0.004 and 0.006, got {ktr_fitted}."

def test_fit_one_site_assoc_and_disso_3():

    fitter_surface = create_fitter(test_file_2)
    fitter_surface.Kd_ss = 0.5
    fitter_surface.Smax_upper_bound_factor = 50
    # Now we test the fitting with shared_smax=False and fixed_t0=False
    fitter_surface.fit_one_site_assoc_and_disso(shared_smax=True, fixed_t0=False, fit_ktr=True)

    # Check if ktr is between 0.004 and 0.006
    ktr_fitted = fitter_surface.fit_params_kinetics['Ktr'].iloc[0]

    assert 0.004 < ktr_fitted < 0.006, f"Expected ktr to be between 0.004 and 0.006, got {ktr_fitted}."

    # check that the number of fitted parameters is correct: we expect smax, Kd, koff, ktr plus one t0 per curve
    n_fitted_params = len(fitter_surface.params)

    assert n_fitted_params == 4 + len(fitter_surface.lig_conc_lst_per_id), \
           f"Expected 4 + {len(fitter_surface.lig_conc_lst_per_id)} t0 parameters, got {n_fitted_params}."

def test_fit_one_site_assoc_and_disso_if():

    fitter_surface = create_fitter(test_file_3)
    fitter_surface.fit_steady_state()

    fitter_surface.fit_one_site_if_assoc_and_disso(shared_smax=True)

    # Asses if self.k_on, self.k_off, self.k_c, self.k_rev are correctly fitted
    assert 0.08 < fitter_surface.k_on < 0.12, f"Expected k_on to be between 0.08 and 0.12, got {fitter_surface.k_on}."

    assert 0.008 < fitter_surface.k_off < 0.012, f"Expected k_off to be between 0.008 and 0.012, got {fitter_surface.k_off}."

    assert 0.7 < fitter_surface.k_c < 1.3, f"Expected k_c to be between 0.7 and 1.3, got {fitter_surface.k_c}."

    assert 7 < fitter_surface.k_rev < 13, f"Expected k_rev to be between 7 and 13, got {fitter_surface.k_rev}."

    # Now fit with shared_smax=False
