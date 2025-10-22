import pytest

from pykingenie.kingenie_surface import KinGenieCsv
from pykingenie.main import KineticsAnalyzer

from numpy.testing import assert_almost_equal

csv_test_file = "./test_files/single_cycle_kingenie.csv"
kingenie = KinGenieCsv()
kingenie.read_csv(csv_test_file)

pk = KineticsAnalyzer()
pk.add_experiment(kingenie, 'test_kingenie_csv')

def test_fit_single_cycle():

    pk.merge_ligand_conc_df()

    df = pk.combined_ligand_conc_df.copy()

    pk.generate_fittings(df)

    pk.submit_kinetics_fitting()

    Kd = pk.get_experiment_properties('Kd', fittings=True)[0]

    assert_almost_equal(Kd,0.5)

def test_fit_single_cycle_with_deleted_step():

    pk.merge_ligand_conc_df()

    df = pk.combined_ligand_conc_df.copy()

    df.loc[df.index[2], 'Select'] = False

    print(df)

    pk.init_fittings()
    pk.generate_fittings(df)

    pk.submit_kinetics_fitting()

    Kd = pk.get_experiment_properties('Kd', fittings=True)[0]

    assert_almost_equal(Kd,0.5)

