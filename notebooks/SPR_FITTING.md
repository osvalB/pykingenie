# SPR fits using pyKinGenie's existing API

The Duke T200 and Carterra notebooks call pyKinGenie's standard one-site fitter
and show its actual output beside the published values. There is no separate
notebook fitting implementation, optimizer or forward model.

The fitting sequence in both notebooks is:

```python
fit_experiment = deepcopy(experiment)
fit_experiment.align_association(inplace=True, npoints=10)

fit_analyzer = KineticsAnalyzer()
fit_analyzer.add_experiment(fit_experiment, fit_experiment.name)
fit_analyzer.merge_ligand_conc_df()
fit_analyzer.generate_fittings(
    fit_analyzer.combined_ligand_conc_df.copy(), trim_time=3,
)
fit_analyzer.submit_steady_state_fitting()
fitter = next(iter(fit_analyzer.fittings.values()))
fitter.fit_one_site_assoc_and_disso(shared_smax=False, calc_loo=False)
```

`submit_steady_state_fitting` supplies the initialization required by the library.
Only the final kinetic one-site fit is shown. Local Smax, Kd and koff come from
`fitter.fit_params_kinetics`; the literature comparison uses `fitter.Kd` and
`fitter.k_off`. The comparison table only converts units, derives kon = koff/Kd,
and calculates percentage differences. The fitted plot uses the existing
`pykingenie.plot_association_dissociation` function.

## Results and comparison scope

| Notebook | Published Kd | pyKinGenie Kd | Scope |
| --- | ---: | ---: | --- |
| [14: Biacore T200](14_biacore_cycle_sample_import.ipynb) | 65.74 nM | approximately 70.65 nM | One-site/local Smax; the standard method does not include the author's independent association RI offsets. |
| [15: Carterra](15_carterra_csv_import.ipynb) | 14.29 nM | approximately 51.93 nM | One-site/local Smax on separately aligned exported curves; the full non-regenerative history and absolute prebound signal are not reconstructed. |

The author reports use five curves; the notebooks retain their three
representative imported curves. The standard fitting results replace the
previous custom-helper results. These are comparisons of the same source
experiments with differences in fitting setup, not exact reproductions of the
published fits. Carterra in particular does **not** reproduce the published
kinetics with this standard workflow. No custom correction is added to force
agreement.

[Notebook 12](12_spr_paired_run_import.ipynb) and
[notebook 13](13_biacore_cycle_fc_import.ipynb) remain import-only examples.
No model searches, rankings, sensitivity analyses or leave-one-out fits are run.

## Preprocessing

The fitting copy uses the existing `align_association(inplace=True, npoints=10)`
method to subtract the mean of the last 10 pre-association points from every
phase. The raw imported experiment and its plot remain unchanged.
`generate_fittings(..., trim_time=3)` uses the library's masks to exclude the
first three seconds of each association and dissociation. All other imported
observations retain their native sampling. No custom thinning, interpolation,
smoothing, median-baseline correction, bulk-offset fit or occupied-state
correction is applied.

The library's `get_step_xy` supplies step-relative times and the existing
one-site fitter handles the model evaluation and initial signals. These calls
are used as implemented; the notebooks do not replace the fitter's clock or
state handling. All imported post-association observations are available to the
fitter, including any export tail beyond the source sheet's nominal protocol.
The plot's display point budget does not change the fitted observations.

## Published values and figures

Reference: [Li et al., TitrationAnalysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10667272/),
DOI [10.12688/gatesopenres.14743.2](https://doi.org/10.12688/gatesopenres.14743.2).
The exact numeric comparisons use the author reports at pinned revision
`cb8112503658d24c4c6576503243fa506bf99e27`:

- **T200:** [parameter values (CSV)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Titration%20Analysis_T200_Report_Example.csv),
  [parameter table (PDF)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Titration%20Analysis_T200_Report_Example.pdf),
  [fitted sensorgrams (PDF)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Titration%20Analysis_T200_Sensorgram_Example.pdf).
  Use Chip 1 / Channel 2 / fitting #1, local Rmax.
- **Carterra:** [parameter values (CSV)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra/Titration%20Analysis_Carterra_Report_Example.csv),
  [parameter table (PDF)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra/Titration%20Analysis_Carterra_Report_Example.pdf),
  [fitted sensorgrams (PDF)](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra/Titration%20Analysis_Carterra_Sensorgram_Example.pdf).
  Use Block 1 / A1 / ROI 1 / fitting #1.

The original sample sheets and CSV reports are retained unmodified in
`data/spr_exports/metadata/`, with CC BY 4.0 attribution and hashes in the
[source notes](data/spr_exports/SOURCES.md).
[fit_metadata.json](data/spr_exports/fit_metadata.json) records the verified
concentrations, import phase times, source sheet rows and published parameters.
Original regeneration and bulk-shift flags describe the source protocol; they
do not enable additional options in the standard fitter. The authors'
Mathematica software has not been executed here.
