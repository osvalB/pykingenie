[![Test and build](https://github.com/osvalB/pykingenie/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/osvalB/pykingenie/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/osvalB/pykingenie/graph/badge.svg)](https://codecov.io/gh/osvalB/pykingenie)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://osvalb.github.io/pykingenie)

Welcome to **pyKinGenie**!, a python package for analysing binding kinetics data 

## Features

- Import Octet files (.frd)
- Import Gator files (Assay_#_Channel#.csv, Setting.ini and ExperimentStep.ini)
- Import Biacore result files (.blr)
- Import paired X/Y SPR text exports, including Biacore T200, and Carterra row-oriented CSV exports
- Import custom CSV files 
- Process the traces (alignment, baseline correction)
- Calculate binding kinetics (*k*<sub>on</sub>, *k*<sub>off</sub>, *K*<sub>D</sub>)
- Global fitting of multiple traces
- One-to-one, one-to-one with mass transport limitation, and one-to-one with induced fit models

## Installation

Install PyKinGenie with pip:

```bash
pip install pykingenie
```

If you use [uv](https://docs.astral.sh/uv/), install it into the current environment with:

```bash
uv pip install pykingenie
```

Or add it to an existing uv-managed project:

```bash
uv add pykingenie
```

For development:

```bash
git clone https://github.com/osvalB/pykingenie
cd pykingenie
uv sync --extra dev
```

## Quickstart

```python
from pykingenie import *

bli = OctetExperiment('test')
# Replace with your own folder containing .frd files !!! The .frd files can be exported from the Octet software
files = ['230309_00'+str(x+1)+'.frd' for x in range(8)]
files = [folder + file for file in files]

files.sort()

bli.read_sensor_data(files)

bli.align_association(bli.sensor_names)
bli.align_dissociation(bli.sensor_names)

# We use the sensor #8 as the reference sensor for baseline correction. Remember than python indexing starts at 0.
bli.subtraction(bli.sensor_names[:7], bli.sensor_names[7])

pyKinetics = KineticsAnalyzer()
pyKinetics.add_experiment(bli, 'test')

labels = pyKinetics.get_experiment_properties('sensor_names')
ids    = pyKinetics.get_experiment_properties('sensor_names_unique')

# Flatten the lists
labels = [item for sublist in labels for item in sublist]
ids    = [item for sublist in ids for item in sublist]

df = get_plotting_df(ids, labels)

figure = plot_traces_all(pyKinetics,df)
figure.write_image('test_bli_traces.png')

pyKinetics.merge_ligand_conc_df()
pyKinetics.init_fittings()

pyKinetics.generate_fittings(pyKinetics.combined_ligand_conc_df)

fig = plot_association_dissociation(pyKinetics,split_by_smax_id=True)
fig.write_image('test_bli.png')

pyKinetics.submit_steady_state_fitting()
pyKinetics.submit_kinetics_fitting()

fig = plot_association_dissociation(pyKinetics)
fig.write_image('test_bli_fitted.png')
```

## SPR sensorgram text exports

`SprTextExperiment` reads text/CSV/TSV files with one named `_X` time column
and `_Y` response column per curve. Verified examples include Biacore T200
`Cycle=..._Fc=...` and `Cycle: ...` headers, and rstoolbox's `Run/Ch/Cy/conc`
export. Delimiters are detected; decimal commas and gzip files are supported.

```python
from pykingenie import SprTextExperiment, CarterraExperiment

spr = SprTextExperiment("t200")
spr.read_file(
    "sensorgrams.txt",
    association_start=0,       # seconds on the exported time axis
    association_end=120,
    concentrations=[25, 50],   # one concentration per measured curve, in file order
    concentration_unit="nM",
    decimal=",",              # use "." for dot-decimal exports
)

carterra = CarterraExperiment("lsa")
carterra.read_csv(
    "sensorgrams.csv",         # Label, X1, Y1, X2, Y2, ...; one curve per row
    association_end=120,
    concentrations=[25, 50],
    concentration_unit="nM",
)
```

Supply the actual injection boundaries for your experiment: association is
`[association_start, association_end)`, and dissociation starts at
`association_end`. These readers represent one association/dissociation pair
per curve; full run logs and single-cycle injection series need their own step
metadata. File times can be seconds or minutes (`time_unit="min"`); boundaries
are always seconds. Independent curve time grids and measured responses are
preserved without interpolation, alignment or reference subtraction. Exported
`fitted curve` columns are excluded; Carterra `NotFitted` rows are retained.
For Carterra XLSX exports, save the sensorgram sheet as CSV first.

Concentrations can also be a mapping from full curve labels to values. Without
explicit concentrations, only `conc=...` labels with a known molar unit are
converted; set `concentration_unit` to interpret unitless `conc=...` labels.
Unresolved concentrations generate a warning and stay `NaN` in
`ligand_conc_df`. Carterra `Conc. N` indices, bare numbers and mass units are
never treated as molar concentrations. Supply these values before fitting.
The common `sample_id="sample"` groups all curves as one binding interaction;
pass a list or curve-name mapping of sample IDs for exports containing several
interactions. The resulting experiments use the standard `KineticsAnalyzer`
workflow shown above. Public sample provenance and licenses are recorded in
[the fixture sources](test_files/spr_exports/SOURCES.md).
The sample data retain their upstream licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md) for attribution and reuse terms.
Independent parser comparisons and the missing-response behavior found in the
full public rstoolbox example are documented in
[SPR parser validation](validation/spr_parsers/README.md).

Executed import examples use full-resolution, representative curves from
`notebooks/data/spr_exports/`, with sensorgram plots and available metadata.
The Duke T200 and Carterra examples additionally use pyKinGenie's existing
one-site fitter and compare its output with the authors' reference reports:

| Input file | Notebook |
| --- | --- |
| `paired_run.csv` | [Run/Ch/Cy paired export](notebooks/12_spr_paired_run_import.ipynb) |
| `biacore_cycle_fc.tsv` | [Biacore Cycle/Fc export](notebooks/13_biacore_cycle_fc_import.ipynb) |
| `biacore_cycle_sample.tsv` | [Biacore Cycle/sample export](notebooks/14_biacore_cycle_sample_import.ipynb) |
| `carterra_rows.csv` | [Carterra CSV export](notebooks/15_carterra_csv_import.ipynb) |

The [teaching-data sources](notebooks/data/spr_exports/SOURCES.md) record curve
selection, full-resolution point counts, licenses, and explicit missing-pair
handling. The small files in `test_files/spr_exports/` remain parser regression
fixtures. The Cycle/Fc example displays full binning cycles plus a display zoom;
it does not represent a single association/dissociation experiment.

These curve exports have no plate-well coordinates. The notebooks display the
available curve labels and draw a plate layout only if plate metadata exists.
The Duke examples use their original companion sample sheets for concentrations
and phase timing. Each calls `align_association(inplace=True, npoints=10)` on
a copy before fitting and keeps the raw import plot.
The rstoolbox and binning notebooks demonstrate import only. See
[SPR fits with published comparisons](notebooks/SPR_FITTING.md) for the API calls,
source reports and remaining differences in fitting setup. Both Duke examples
use the same five concentration curves and nominal protocol windows as the
author reports, verified against their original sensorgram figures.

## Detecting input formats

`detect_file_format` inspects file contents to select an input reader. All three
paired-column SPR variants share `SprTextExperiment`; Carterra's row-oriented
layout is detected separately:

```python
from pathlib import Path
from pykingenie import detect_file_format

for file in sorted(Path("test_files/spr_exports").glob("*")):
    if file.suffix in {".csv", ".tsv"}:
        print(file.name, detect_file_format(file))
# biacore_cycle_fc.tsv spr_text
# biacore_cycle_sample.tsv spr_text
# carterra_rows.csv carterra_csv
# paired_run.csv spr_text
```

| Detected identifier | Reader |
| --- | --- |
| `spr_text` | `SprTextExperiment.read_file` |
| `carterra_csv` | `CarterraExperiment.read_csv` |
| `octet_frd` | `OctetExperiment.read_sensor_data` |
| `biacore_blr` | `BiacoreExperiment.read_blr` |
| `gator_csv`, `gator_steps`, `gator_settings`, `gator_zip` | `GatorExperiment.read_all_gator_data` with the complete file set or archive |
| `kingenie_surface` | `KinGenieCsv.read_csv` |
| `kingenie_solution` | `KinGenieCsvSolution.read_csv` |
| `surface_csv_steps`, `surface_csv_traces` | `SurfaceCsvExperiment.read_files` with both files |

Text detection searches the first 64 KiB of uncompressed content, handles
metadata preambles, BOMs, Windows-1252 text, gzip, and comma/tab/semicolon
delimiters. Native Biacore and Gator archives are checked for their internal
structure. Renaming a file does not change its content-based identifier.
Detection identifies the layout; the reader still validates the measurements
and requires its normal arguments, including decimal conventions, injection
boundaries, concentrations and companion files where applicable.

Ambiguous or unsupported inputs return `unknown`, including generic two-column
`Time,Signal` data and native XLSX files. Choose the reader explicitly for such
inputs. Missing/inaccessible files raise the normal filesystem error.
`guess_experiment_type` remains the compatible surface/solution helper, with its
historical `surface` fallback; use `detect_file_format` for reader selection.

## Keeping results when changing models

Each fitter automatically keeps successful model results in `fit_results`.
This works for direct fitting calls and fits submitted through `KineticsAnalyzer`.
The usual attributes, such as `fit_params_kinetics` and `signal_assoc_fit`,
continue to describe the active result.

```python
fitter = next(iter(pyKinetics.fittings.values()))
fitter.fit_one_site_assoc_and_disso()
one_site = fitter.active_fit_result

fitter.fit_two_site_assoc_and_disso()

# Read the previous model without changing the active result.
previous_parameters = fitter.fit_results[one_site]['fit_params_kinetics']
print(list(fitter.fit_results))

# Restore its parameters, fitted curves, settings and diagnostics for plotting/export.
fitter.restore_fit_result(one_site)
fig = plot_association_dissociation(pyKinetics)
```

Snapshots are independent copies of fitted arrays and tables, including errors,
correlations, bounds and any calculated confidence intervals or leave-one-out
results. Failed fits do not replace saved results.
Internal fits used only to find initial guesses do not replace saved models.

Every model and argument combination is saved separately in `fit_result_history`.
For example, shared and per-trace Smax fits can be restored by their arguments:

```python
fitter.fit_one_site_assoc_and_disso(shared_smax=True, fixed_t0=True)
fitter.save_fit_result('one_site_shared_smax', shared_smax=True, fixed_t0=True)

fitter.fit_one_site_assoc_and_disso(shared_smax=False)

fitter.restore_fit_result(
    'one_site_association_and_dissociation', shared_smax=True, fixed_t0=True,
)
assert fitter.shared_smax is True
print(fitter.fit_settings)  # All arguments used by this fit, including defaults.

# Named snapshots can also be restored directly, or checked against arguments.
fitter.restore_fit_result('one_site_shared_smax', shared_smax=True)
```

`save_fit_result(name, **fit_arguments)` always stores all actual fit arguments.
Arguments supplied to save are checked against the active fit, so a result cannot
be saved with settings different from those used to calculate it. To change an
argument, run the fit with that argument first.

`fit_results[model]` holds the latest saved snapshot at that key. Restore a model
without arguments to use that snapshot, or supply arguments to select a unique
configuration from `fit_result_history[model]`. Ambiguous or unmatched selections
raise an error. Refitting an identical configuration replaces its previous result;
use named snapshots to retain repeated runs with identical arguments.

Surface result keys include `steady_state_one_site`, `one_site_association`,
`one_site_dissociation`, `one_site_association_and_dissociation`, and
`two_site_association_and_dissociation`. Mass transport uses
`one_site_mass_transport`; fitted cooperativity uses `two_site_cooperative` or
`steady_state_two_site_cooperative`. Induced fit uses `one_site_if_assoc_and_disso`
or `one_site_if_variable_signal` when fitting the signal ratio. Heterogeneous
ligand fits use `two_site_heterogeneous_ligand_association_and_dissociation` or
`steady_state_two_site_heterogeneous_ligand`. Solution result keys are
`single_exponential`, `double_exponential`, `one_binding_site`, `induced_fit`,
and `conformational_selection`. Each snapshot records the call options in
`fit_settings`; `active_fit_result` gives the current key.

History belongs to each fitter and is kept in memory. `clear_fittings()` clears
the active fit while keeping saved results available for restoration.

## Validating and comparing saved fits

Every fit records a fingerprint of its input traces, time arrays, concentrations,
Smax grouping, trimming, and single-cycle setting. Restore, save, and diagnostics validate
this fingerprint before using a result. Changing preprocessing or data selection
raises `ValueError` when an earlier result is used. Restore the original inputs
to use that result again, or run a new fit on the changed data. Display names and
equivalent list/array containers do not affect validation.

```python
fitter.validate_fit_result()  # Check the active result; returns True or raises.
fitter.validate_fit_result('one_site_shared_smax', shared_smax=True)

fitter.fit_one_site_assoc_and_disso(shared_smax=True)
fitter.save_fit_result('shared')
fitter.fit_one_site_assoc_and_disso(shared_smax=False)
fitter.save_fit_result('per_trace')

comparison = fitter.compare_saved_fits(['shared', 'per_trace'])
print(comparison)
# Optional ranking, best BIC first:
ranked = fitter.compare_saved_fits(['shared', 'per_trace'], sort_by='BIC')
```

The comparison table has `Model`, `Fit settings`, `Parameter count`, `Reduced χ²`,
and `BIC` columns, indexed by saved result name. The selected order is preserved
by default; `sort_by='BIC'` sorts increasingly, preserving ties. Lower BIC is
preferred. With no names, `compare_saved_fits()` includes each
stored model/argument configuration once, without duplicating named aliases.
The comparison reads saved results without changing the active fit.

All selected results must belong to the current dataset and use exactly the same
fitted observations and time ranges. For example, compare association/dissociation
models together, steady-state models together, or dissociation-only fits with the
same time window. Select compatible saved names when the archive contains mixed
regions. Missing or non-finite predictions and non-positive residual degrees of
freedom raise `ValueError`.

For these unweighted fits, reduced chi squared is `RSS / (n - k)`, assuming unit
observation variance. `n` counts the actual fitted points after trimming or time
limits, and `k` counts every
free parameter, including amplitudes, offsets, time offsets, and parameters fitted
per trace. A single exponential has three parameters per trace; a double
exponential has five. BIC is:

```text
BIC = n * log(RSS / n) + k * log(n)
```

A perfect fit has zero reduced chi squared and BIC of negative infinity.

## XY plotting

An alternative implementation uses [Reflex XY](https://reflex.dev/docs/xy/).
Try the complete sample workflow in
[`notebooks/10_reflex_xy_plotting.ipynb`](notebooks/10_reflex_xy_plotting.ipynb).
XY is installed with pyKinGenie on Python 3.11 or newer:

```bash
pip install pykingenie
# Development checkout:
uv sync --extra dev
```

XY helpers use reusable dataclasses for styling and rendering settings:

```python
from pykingenie.utils.plotting_xy import (
    PlotConfig,
    AxisConfig,
    LayoutConfig,
    LegendConfig,
    TraceConfig,
    config_fig,
    plot_plate_info,
    plot_traces,
    plot_traces_all,
    plot_steady_state,
    plot_association_dissociation,
)

plot_config = PlotConfig(plot_width=1000, plot_height=550, font_size=14)
axis_config = AxisConfig(showgrid_x=True, showgrid_y=False, x_bounds='data')
layout_config = LayoutConfig(ncols=2, gap=16, shared_yaxes=True)
legend_config = LegendConfig(loc='upper right', font_size=12)
trace_config = TraceConfig(marker_size=1, line_width=2, exact_lines=False)

fig = plot_traces_all(
    pyKinetics,
    pyKinetics.get_legends_table(),
    plot_config=plot_config,
    axis_config=axis_config,
    layout_config=layout_config,
    legend_config=legend_config,
    trace_config=trace_config,
)
fig.show()  # Interactive notebook display
fig.to_html('sensor_traces.html')  # Standalone interactive chart
fig.write_image('sensor_traces.png')
fig.write_image('sensor_traces.svg')

fig = plot_association_dissociation(
    pyKinetics,
    plot_config=plot_config,
    layout_config=LayoutConfig(split_by_smax_id=True),
    trace_config=TraceConfig(show_fit=True, fit_color='black', max_points=2000),
)
fig = config_fig(fig, 'kinetics', plot_config=PlotConfig(plot_type='png'), save=True)
```

All dimensions are **pixels**. Unset fields (`None`) retain the helper's defaults,
and configuration objects can be reused without being modified by plotting.
Use `dataclasses.replace(config, ...)` to derive a variant.

| Dataclass | Settings |
| --- | --- |
| `PlotConfig` | Pixel dimensions, font size, title, export format, initial x/y ranges |
| `AxisConfig` | Grid lines, ticks, line widths, labels, scales, reversal, navigation bounds |
| `LayoutConfig` | Columns or stacked panels, spacing, shared domains, subplot titles, Smax grouping |
| `LegendConfig` | Visibility, placement, columns, title, typography, interactions |
| `TraceConfig` | Markers, lines, opacity, fitted overlays, smoothing, sampling, exact lines |

For example, `PlotConfig(x_range=(0, 1000))` sets the initial time range and
`AxisConfig(x_bounds='data')` prevents panning far beyond the experiment.
`plot_steady_state(..., trace_config=TraceConfig(show_fit=True))` adds fitted curves.

Plate layouts and individual sensor plots return `xy.Chart`; subplot helpers
return `xy.facets.FacetGrid` composed from independent XY panels. Configure sizes
before displaying live widgets, and keep the value returned by `config_fig`.
Both types support the display and export methods above. PNG and SVG exports use
XY's native renderer. Kinetics plots retain logarithmic Viridis concentration
colors and use labeled concentration legends in place of the Plotly colorbar.
Concentration legends and sensor legends with both lines and markers are
informational; use the legends table's `Show` column to select sensors.
`PlotConfig.plot_type` sets the download/export format. Saving is explicit through
`write_image` or `config_fig(..., save=True)`.

The existing top-level plotting imports continue to use Plotly.

Raw XY traces use automatic line simplification by default. When zooming into
standalone HTML, lines can skip some displayed markers. Use
`plot_traces_all(..., trace_config=TraceConfig(exact_lines=True))` or the same
`trace_config` in `plot_traces`
to connect every displayed sample. Exact lines increase HTML size and rendering
work; the initial sampling limit remains the same for lines and markers.

## Contributing

Contributions are welcome! Please contact me at oburastero@embl-hamburg.de

## Citation

If you use `pyKinGenie`, please cite it as:

Burastero, O. (2026). `pyKinGenie` (Version 1.0) [Computer software].
GitHub. https://github.com/osvalB/pykingenie

```bibtex
@software{burastero_2026_pykingenie,
  author = {Burastero, Osvaldo},
  title = {pykingenie},
  version = {1.0},
  year = {2026},
  url = {https://github.com/osvalB/pykingenie}
}
```
## Acceptable usage of AI assistants

AI coding assistants may be used for code that does not affect
numerical results, for example:

- plotting and figure styling,
- import and export helpers, file handling and formatting,
- documentation, examples and boilerplate tests.

Files containing AI-generated code start with a note saying so
(e.g. `src/pykingenie/utils/plotting_xy.py`).

Code that affects the results needs human oversight: a person must write it,
or review, understand and validate it before it is merged. This covers:

- mathematical models, rate equations and signal simulations,
- fitting routines, initial guesses, constraints and error estimates,
- data processing that changes signal values (alignment, reference
  subtraction, baseline and inter-step corrections),
- the interpretation of fitted parameters such as kon, koff and KD.

Changes to these parts must be checked against known results, such as
simulated data with known parameters or previously validated datasets,
before they are accepted.

The same applies when an AI assistant runs analyses for you (for example
through the pyKinGenie MCP server): check the processing steps it chose and
inspect the fits yourself before you report any parameters. 

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
