[![Test and build](https://github.com/osvalB/pykingenie/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/osvalB/pykingenie/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/osvalB/pykingenie/graph/badge.svg)](https://codecov.io/gh/osvalB/pykingenie)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://osvalb.github.io/pykingenie)

Welcome to **pyKinGenie**!, a python package for analysing binding kinetics data 

## Features

- Import Octet files (.frd)
- Import Gator files (Assay_#_Channel#.csv, Setting.ini and ExperimentStep.ini)
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

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
