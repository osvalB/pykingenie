# SPR fits using pyKinGenie's existing API

The Duke T200 and Carterra notebooks call pyKinGenie's standard one-site fitter
and show its actual output beside the published values. There is no separate
notebook fitting implementation, optimizer or forward model.

The fitting sequence in the two five-curve literature comparisons is:

```python
fit_experiment = deepcopy(experiment)
fit_experiment.align_association(inplace=True, npoints=10)

fit_analyzer = KineticsAnalyzer()
fit_analyzer.add_experiment(fit_experiment, fit_experiment.name)
fit_analyzer.merge_ligand_conc_df()
selected_curves = fit_analyzer.combined_ligand_conc_df.copy()
if fit_experiment.exp_info.get("single_cycle"):
    selected_curves["Select"] = selected_curves["Analyte_location"].astype(int).isin(
        fit_metadata["source_concentration_entries_one_based"]
    )
fit_analyzer.generate_fittings(selected_curves, trim_time=0)
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
| [15: Carterra](15_carterra_csv_import.ipynb) | 14.29 nM | approximately 16.42 nM | Native single-cycle/local Smax; first selected initial signal estimated by the library, later states carried from preceding dissociations. |

**Carterra timing limitation:** the cycle period of 1140 s reconstructs a
nominal protocol clock from the source sheet. An absolute injection log was
not available. Carterra kinetic results are conditional on that period and
are comparisons with the published values, rather than exact reproductions.

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
published fits. Carterra uses the standard single-cycle workflow described
below, giving a Kd approximately 15% above the reference.

[Notebook 12](12_spr_paired_run_import.ipynb) and
[notebook 13](13_biacore_cycle_fc_import.ipynb) remain import-only examples.
No model searches or rankings are run. The two five-curve literature comparisons
do not run leave-one-out fits. A separate advanced example is described below.

## Advanced Carterra example: all 13 injections and global Smax

[Notebook 16](16_carterra_global_smax.ipynb) retains the same imported sensor,
one-time baseline alignment and reconstructed clock, and selects all 13
injections from 0.244 to 1000 nM. It calls the existing fitter with:

```python
selected_curves["Select"] = True
fit_analyzer.generate_fittings(selected_curves, trim_time=0)
fit_analyzer.submit_steady_state_fitting()
fitter = next(iter(fit_analyzer.fittings.values()))
fitter.fit_one_site_assoc_and_disso(shared_smax=True, calc_loo=True)
```

The global-Smax fit has three parameters and gives approximately Kd 22.36 nM,
koff 8.581e-5 s^-1 and Smax 224.18 RU. It changes both the injection selection
and the capacity assumption from the published five-injection/local-Rmax fit.
[Notebook 15](15_carterra_csv_import.ipynb) remains the primary literature
comparison. Systematic deviations in the global-Smax fit and the unverified
absolute clock limit interpretation of its kinetic constants.

The native leave-one-out method omits each injection's association and
dissociation observations and refits the remaining 12 with a shared Smax.
It retains Kd and koff estimates, their means and their population SDs
(`ddof=0`). The notebook shows the native result table, every omitted injection,
and unit-converted summaries with derived kon. All 13 refits succeed; their
Kd mean is 23.59 nM with SD 2.62 nM. Smax estimates from individual deletion
refits are not retained by the native method.

Deleting an injection creates a gap in the single-cycle history. The fitter
estimates the next retained association's initial response from the first
30 s of that association, falling back to the first observed response if the
estimate fails. It holds this initial response fixed during the kinetic refit,
then resumes state propagation. In particular, omitting injection 1 does not
force injection 2 to start at zero or shift it to the beginning of the clock.
The observed data are unchanged. These SDs measure deletion sensitivity;
they are neither jackknife standard errors nor confidence intervals, and
they are not held-out prediction errors.

## Preprocessing

The fitting copy uses the existing `align_association(inplace=True, npoints=10)`
method on a copy. T200 aligns each independent curve. Carterra aligns only the
first baseline of its single-cycle sequence: the existing method skips later
associations directly following a dissociation. Its raw imported responses
and relative occupied signals are preserved.
`generate_fittings(..., trim_time=0)` fits all retained association and
dissociation points, matching **Asso. Skip = 0** and **Disso. Skip = 0** in the
source sheets. All imported observations retain their native sampling. No
custom thinning, interpolation, smoothing, median-baseline correction or
bulk-offset fitting is applied.

### Carterra single-cycle import

The notebook imports **all 13 injections** on the physical sensor, then selects
source concentration entries 9–13 for the five-curve published comparison:

```python
experiment.read_csv(
    data_file,
    association_start=120,
    association_end=420,
    concentrations=fit_metadata["cycle_concentrations"],
    concentration_unit="uM",
    sample_id=fit_metadata["analyte"],
    single_cycle=True,
    cycle_duration=fit_metadata["cycle_duration_seconds"],
)
```

This produces one sensor and 27 steps, alternating association and dissociation
after the first baseline. Original row labels remain in `Source_curve`, and
concentrations map to unique `Analyte_location` values. The next injection's
baseline is part of the preceding zero-analyte dissociation. This preserves the
clock and continued dissociation up to the following association.

The notebook calls `align_association` on this full sequence and passes a
selection table with only injections 9–13 selected to `generate_fittings`.
It asserts `fitter.is_single_cycle` and that every subsequent selected association
is continuous with the preceding dissociation. The existing one-site fitter
estimates the first selected injection's nonzero initial signal, since earlier
injections are outside the fit, and carries the fitted bound state forward
for later injections. The fitting algorithm is unchanged.

**Timing assumption:** 1140 s is the source sheet's nominal baseline +
association + fitted dissociation duration. It reconstructs a protocol clock;
it is not an original absolute acquisition log. Supply the actual cycle period
when known. Original export tails after each row's 1140 s window are omitted.
Following baselines are included as inter-injection wash observations in the
continuous fit, beyond the author's per-row fitted dissociation interval.
Initial-state estimation and local-capacity handling also differ from the
published implementation, so agreement is not an exact reproduction.

The fitted plot uses pyKinGenie's standard plotting function on this shared
clock. Its display point budget does not change fitted observations.

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
