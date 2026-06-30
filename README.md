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

## Contributing

Contributions are welcome! Please contact me at oburastero@embl-hamburg.de

## Citation

If you use `pyKinGenie`, please cite it as:

Burastero, O. (2026). `pyKinGenie` (Version 1.0.0) [Computer software].
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
