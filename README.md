# pyKinGenie

[![Test and build](https://github.com/osvalB/pykingenie/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/osvalB/pykingenie/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/osvalB/pykingenie/graph/badge.svg)](https://codecov.io/gh/osvalB/pykingenie)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://osvalb.github.io/pykingenie/)

Fit binding kinetics from SPR, BLI, and solution traces. pyKinGenie imports sensorgrams, aligns and plots traces, and estimates binding rates and affinities across concentrations.

## Install

```bash
python -m pip install pykingenie
```

To run the examples below, clone this repository and run `python -m pip install -e .` from the checkout with Python 3.11 or newer. For development tools, use `uv sync --extra dev`. [Installation guide](docs/installation.rst)

## Input files

Pass a path or a list of companion files for one experiment to `KineticsAnalyzer.load()`. It chooses the reader from the file contents. Use `detect_file_format(path)` to check what a file is recognized as.

| Input | Files and metadata to supply | Example |
| --- | --- | --- |
| Octet BLI | The `.frd` files for one run. Add the optional `.fmf` file for plate information. | [Sensor file](test_files/230309_001.frd), [plate file](test_files/230309_ExpMethod.fmf), [workflow](notebooks/9_octet_example_one_site_induced_fit_linked_rmax.ipynb) |
| Gator BLI | A run ZIP or `Setting.ini`, `ExperimentStep.ini`, and the `Assay_*_Channel*.csv` files. A ZIP extracts next to itself. | [Run ZIP](test_files/gator_example_folder.zip) |
| Biacore native | One `.blr` result file. The reader uses its recorded cycles, injection events, and concentrations. | [Result file](test_files/SPR_sensograms_with_Streptavidin_1.blr), [workflow](notebooks/11_biacore_spr_import.ipynb) |
| Paired-column SPR text | One TXT, CSV, or TSV file with matching `_X` time and `_Y` response columns for each curve. Add injection timings and concentrations as described below. | [Biacore T200](notebooks/14_biacore_cycle_sample_import.ipynb), [Run/Ch/Cy](notebooks/12_spr_paired_run_import.ipynb) |
| Carterra SPR text | One CSV with a row per curve in `Label, X1, Y1, X2, Y2, …` format. Export the sensorgram worksheet as CSV and add timings and concentrations. | [Carterra notebook](notebooks/15_carterra_csv_import.ipynb) |
| Generic surface steps and traces | Two files. The steps table has a duration in seconds (`Step Time`) and a type (`Step Type Name`) per row. The traces table starts with `X Values` and has one response column per sensor. Supply `Concentration` in the steps table or `concentrations=` in µM for the association steps. | [Steps](test_files/surface_csv_steps.txt), [traces](test_files/surface_csv_traces.csv) |
| KinGenie surface CSV | One table with `Time`, `Signal`, `Smax`, and `Analyte_concentration_micromolar_constant`. Add `Cycle` for single-cycle data. | [Surface CSV](test_files/single_cycle_kingenie.csv) |
| KinGenie solution CSV | One table in column order `Time`, `Signal`, `Protein_concentration_micromolar`, `Ligand_concentration_micromolar`. For a plain time-and-signal table, use `KinGenieCsvSolution.read_csv()` directly. | [Solution CSV](test_files/kingenie_solution.csv) |

Delimited readers detect commas, semicolons, and tabs. The two SPR text readers also accept `.gz` files and detect decimal commas. Use standard CSV quotes around decimal-comma values in comma-separated files. Other readers expect decimal points.

### Timing and concentrations for SPR text

- Pass `association_end` and `association_start` (default 0). Boundaries are in seconds, including with `time_unit="min"`. Set `concentrations` as a list in file order or a mapping by curve label. Values are in µM unless you set `concentration_unit`. Labels with a molar `conc=` value can supply the concentration.
- Pass a TitrationAnalysis Data Info Sheet (`.xlsx`, CSV, or TSV) as `info=` to supply timing, concentrations, and sample IDs. Explicit arguments take precedence. Carterra also accepts `top_concentration` and `dilution_factor`, with `highest_last=False` if the highest concentration is first.
- For independent paired-column curves, timings can be shared or provided per curve. For a titration without regeneration, use `single_cycle=True`, give one start, end, and concentration per injection, and assign each curve its own `sample_id`. Carterra uses one timing window for all rows and requires acquisition order and `cycle_duration` for a single-cycle run. [SPR fit examples](notebooks/SPR_FITTING.md)

## Fit a surface experiment

This example uses the bundled single-cycle simulation. For experimental data, review the selected curves and concentrations and apply any alignment or reference subtraction before generating fits. [Octet preprocessing example](notebooks/9_octet_example_one_site_induced_fit_linked_rmax.ipynb)

```python
from pykingenie import KineticsAnalyzer, plot_association_dissociation

analysis = KineticsAnalyzer()
analysis.load("test_files/single_cycle_kingenie.csv")
analysis.merge_ligand_conc_df()
analysis.generate_fittings(analysis.combined_ligand_conc_df)

fitted, failed = analysis.submit_kinetics_fitting(shared_smax=True)
if failed:
    raise RuntimeError("A surface fit failed.")
print(fitted[0].fit_params_kinetics)
plot_association_dissociation(analysis).show()
```

The default fit is 1:1. Use `fitting_model` to select mass transport, induced fit, two-site binding with optional cooperativity, heterogeneous ligand, or bivalent analyte. Use `trim_time` in `generate_fittings()` to leave the first seconds of each phase out of the fit. [Analyzer API source](src/pykingenie/main.py) · [Model examples](docs/examples.rst) · [Benchmarks](benchmark/README.md)

## Fit a solution experiment

```python
from pykingenie import KineticsAnalyzer

analysis = KineticsAnalyzer()
analysis.load("test_files/kingenie_solution.csv")
analysis.merge_conc_df_solution()
analysis.generate_fittings_solution(analysis.combined_conc_df)
analysis.submit_fitting_solution("one_binding_site")
print(next(iter(analysis.fittings.values())).fit_params_kinetics)
```

Solution fits also support single and double exponentials, induced fit, and conformational selection.

## Results and examples

- Fitters keep results in memory by model and settings. Use `fit_results` to inspect them, `save_fit_result()` for a named snapshot, and `restore_fit_result()` to restore one. Use `compare_saved_fits()` for reduced χ² and BIC on the same observations. [Fitter API source](src/pykingenie/fitter.py)
- Use the top-level Plotly functions or the alternative [Reflex XY plotting notebook](notebooks/10_reflex_xy_plotting.ipynb).
- See the [advanced Carterra fit](notebooks/16_carterra_global_smax.ipynb) for shared surface capacity and leave-one-out sensitivity, and [SPR parser validation](validation/spr_parsers/README.md) for import comparisons.

## Contributing

Questions and contributions are welcome at oburastero@gmail.com. If you use pyKinGenie, please cite [Burastero (2026)](docs/citation.rst).

AI assistants may help with documentation, boilerplate tests, plotting, and file handling. Mark files that contain AI-generated code. A person must write, or review, understand, and validate, changes that affect numerical results before merge. This includes models, fitting, preprocessing, and parameter interpretation. Validate against simulations with known parameters or validated datasets, and inspect AI-assisted preprocessing and fits before reporting estimates.

Released under the [MIT License](LICENSE). See the [third-party notices](THIRD_PARTY_NOTICES.md) for sample data and other included materials.
