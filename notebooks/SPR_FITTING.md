# SPR fits with published comparisons

Only the two Duke examples include fitting. They use a **one-site (1:1 Langmuir)
model with local Rmax**, matching the model in the authors' TitrationAnalysis
example reports. These examples compare selected curves from the same source
datasets; they are not exact reproductions of the complete published fits.

| Notebook | Retained model | Published Kd | Notebook Kd |
| --- | --- | ---: | ---: |
| [14: Biacore T200 Cycle/sample](14_biacore_cycle_sample_import.ipynb) | One site, local Rmax and association-only RI offsets | 65.74 nM | 66.03 nM |
| [15: Carterra](15_carterra_csv_import.ipynb) | One site, local Rmax and measured prebound initial state | 14.29 nM | 17.62 nM |

The T200 comparison uses Chip 1 / Channel 2 / fitting #1. Its kon and koff
estimates differ by approximately 1.8% and 2.2% from the author report.
The Carterra comparison uses Block 1 / A1 / ROI 1 / fitting #1. Its Kd is
approximately 23.4% higher and its koff 2.3% higher than the author report.
The notebooks show all three parameters, units and percentage differences.

[Notebook 12](12_spr_paired_run_import.ipynb) and
[notebook 13](13_biacore_cycle_fc_import.ipynb) demonstrate import, available
metadata and raw sensorgrams only. No comparable published kinetic-parameter
report has been verified for those examples. No fits or model selection are
included for them.

## Reference reports

The reference is [Li et al., TitrationAnalysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10667272/),
DOI [10.12688/gatesopenres.14743.2](https://doi.org/10.12688/gatesopenres.14743.2).
The exact numeric comparisons use the authors' published example CSVs at
pinned revision `cb8112503658d24c4c6576503243fa506bf99e27`:

- **T200:** [parameter values (CSV)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Titration%20Analysis_T200_Report_Example.csv),
  [parameter table (PDF)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Titration%20Analysis_T200_Report_Example.pdf), and
  [original fitted-sensorgram figures (PDF)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Titration%20Analysis_T200_Sensorgram_Example.pdf).
- **Carterra:** [parameter values (CSV)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra/Titration%20Analysis_Carterra_Report_Example.csv),
  [parameter table (PDF)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra/Titration%20Analysis_Carterra_Report_Example.pdf), and
  [original fitted-sensorgram figures (PDF)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra/Titration%20Analysis_Carterra_Sensorgram_Example.pdf).

Unmodified CSV copies are included in `data/spr_exports/metadata/`, with CC BY 4.0
attribution, exact paths and SHA-256 hashes in the [source notes](data/spr_exports/SOURCES.md).
[fit_metadata.json](data/spr_exports/fit_metadata.json) identifies each report
row and its parameters in molar units. The original companion sample sheets
supply concentrations, event times and regeneration/RI settings.

The author reports fit **five curves**. Each notebook retains its original
**three representative curves**. For Carterra these are not the exact same
concentration subset selected for the published report. Alignment, initial-state
estimates and transition masks also differ. Agreement is a parameter comparison,
not independent validation of intrinsic kinetics. The authors' Mathematica
software has not been executed here.

## Preprocessing and fitting

Each fitting notebook makes a deep copy of the imported experiment, then calls
`fit_experiment.align_association(inplace=True, npoints=10)`. This subtracts the
mean of the last 10 native pre-association observations from every phase. The
removed offsets are displayed and recorded. The imported experiment, its time
grids and raw plot remain unchanged. No second baseline subtraction is applied.

The helper [spr_fitting.py](spr_fitting.py) evaluates pyKinGenie's analytical
one-site association and dissociation functions. Association starts from the
physical injection time and propagates to its true end, even if scored points
near transitions are excluded. Dissociation starts from that end state.

- T200: concentrations 0.8868, 1.774, 3.547 µM; association 0–180 s;
  dissociation 180–780 s. Regeneration gives zero initial bound signal after
  alignment. Each curve has its own Rmax and association-only RI offset.
- Carterra: concentrations 0.0313, 0.25, 1.0 µM; association 120–420 s;
  dissociation 420–1140 s. No regeneration. The physical initial bound signal
  is the original median over 90–115 s, reconstructed from aligned data plus
  the removed offset. Model predictions subtract that offset to match the
  aligned response coordinate. An additional instrumental background is
  assumed zero. Rmax represents total physical capacity, including prebound
  material.

Exclude 3 s around transitions and use at most one original point per
one-second interval. Carterra retains its slower native observations. No fitted
observation is averaged, interpolated or smoothed. Use equal response weighting
and bounded SciPy least squares, with eight deterministic starts (seed 2918)
for this single model. The repeats improve optimizer convergence; no alternative
models, model rankings or sensitivity experiments are run.

The notebooks show the fitted sensorgrams, residuals, point estimates and
published comparison. Compact numerical outputs are retained for
[T200](data/spr_exports/fits/biacore_cycle_sample.json) and
[Carterra](data/spr_exports/fits/carterra_rows.json). No exploratory fit reports
are included.
