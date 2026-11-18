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
    fit_analyzer.combined_ligand_conc_df.copy(), trim_time=0,
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
| [14: Biacore T200](14_biacore_cycle_sample_import.ipynb) | 65.74 nM | approximately 71.80 nM | One-site/local Smax; the standard method does not include the author's independent association RI offsets. |
| [15: Carterra](15_carterra_csv_import.ipynb) | 14.29 nM | approximately 47.38 nM | One-site/local Smax on separately aligned exported curves; the full non-regenerative history and absolute prebound signal are not reconstructed. |

Both notebooks now use the **same five concentrations and curve identities**
as page 1 of the author sensorgram reports. T200 uses cycles 18–22 at
**221.7, 443.4, 886.8, 1774, 3547 nM**. Carterra uses worksheet rows 10–14,
Block 1/A1/ROI 1, at **62.5, 125, 250, 500, 1000 nM**. The selections were
verified against the original PDF figures and companion concentration lists.

The data excerpts retain every finite point in the corresponding protocol
windows: T200 baseline −60–0 s, association 0–180 s and dissociation 180–780 s;
Carterra baseline 0–120 s, association 120–420 s and dissociation 420–1140 s.
No curves are selected merely for having a large response. These comparisons
still have differences in fitting setup and are not exact reproductions of the
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
`generate_fittings(..., trim_time=0)` fits all retained association and
dissociation points, matching **Asso. Skip = 0** and **Disso. Skip = 0** in the
source sheets. All imported observations retain their native sampling. No
custom thinning, interpolation, smoothing, median-baseline correction, bulk-offset
fit or occupied-state
correction is applied.

The library's `get_step_xy` supplies step-relative times and the existing
one-site fitter handles the model evaluation and initial signals. These calls
are used as implemented; the notebooks do not replace the fitter's clock or
state handling. Export tails beyond the source sheet's nominal protocol are
excluded when preparing the data excerpts, with omission counts recorded in the
manifest. The plot's display point budget does not change fitted observations.

The T200 source contains missing/non-finite pairs: 179 in cycle 18, 9 in cycle 19
and 8 in cycle 21. Those unavailable observations remain gaps; no interpolation
is applied. Missing-pair and window-omission counts are documented in the
[source notes](data/spr_exports/SOURCES.md) and manifest.

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
