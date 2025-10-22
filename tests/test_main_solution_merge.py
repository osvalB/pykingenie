import pytest

from pykingenie.main import KineticsAnalyzer
from pykingenie.kingenie_solution import KinGenieCsvSolution

pyKinetics = KineticsAnalyzer()

file1 = './test_files/solution_data_1.csv'
file2 = './test_files/solution_data_2.csv'

def test_merge_solution_experiments():

    k1 = KinGenieCsvSolution('test1')
    k2 = KinGenieCsvSolution('test2')

    k1.read_csv(file1)
    k2.read_csv(file2)

    pk = KineticsAnalyzer()

    pk.add_experiment(k1, 'test1')
    pk.add_experiment(k2, 'test2')

    pk.collapse_solution_experiments()

    assert len(pk.experiments) == 1

