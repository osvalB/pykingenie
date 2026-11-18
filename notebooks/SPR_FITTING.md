# Fitting the representative SPR imports

These are analyses of the **selected measured curves** in notebooks 12–15,
not model-recovery simulations or reanalyses of every injection in the source
experiments. Each notebook includes model comparisons, fitted overlays,
residuals, conditional parameter intervals, correlations, and refits after
changing the observation window and event timing. Raw imports remain intact.

## Decisions

“Best” below means the lowest conditional BIC **among the candidates tested on
identical observations**. It does not establish the molecular mechanism.

| Notebook / data | Best candidate | Fit RMSE | Decision |
| --- | --- | --- | --- |
| [12: rstoolbox paired Run/Ch/Cy](12_spr_paired_run_import.ipynb) | Two independent ligand populations, local Rmax | 0.645 exported response units | Do not trust quantitative kinetic parameters: concentration units and events are unverified, residuals are strongly structured, and the second affinity is unstable. |
| [13: Biacore Cycle/Fc binning](13_biacore_cycle_fc_import.ipynb) | Empirical double exponential, local amplitudes, early episode only | 16.41 RU | No defensible binding constants or two-component interpretation. The full multi-event run is not a single kinetic experiment. |
| [14: Biacore Cycle/sample T200](14_biacore_cycle_sample_import.ipynb) | Two independent ligand populations, local Rmax and local association RI offsets | 0.126 RU | Good descriptive fit; dominant slow kinetics are provisional. Two molecular site classes and the nominal parameter precision are not established. |
| [15: Carterra](15_carterra_csv_import.ipynb) | One site, local Rmax, occupied initial state in aligned response coordinates | 1.523 RU | Useful approximate one-site description; precise intrinsic constants are not reliable from these three curves. |

The numerical audit reports are [paired Run/Ch/Cy](data/spr_exports/fits/paired_run.json),
[Cycle/Fc](data/spr_exports/fits/biacore_cycle_fc.json),
[Cycle/sample](data/spr_exports/fits/biacore_cycle_sample.json), and
[Carterra](data/spr_exports/fits/carterra_rows.json).

## What determines these decisions

**rstoolbox.** The best two-population fit has affinities about 0.625 and 136
in the **exported concentration-label scale**, not verified µM or nM. Its second
affinity has a nominal interval about 20–920 and almost perfectly correlated
parameters. Halving dissociation changes that affinity by about 81-fold and
drives the population fraction toward its lower bound. Residual lag-1
correlations exceed 0.99 and RMSE is about 34–56 times the baseline noise
estimate. A visually close overlay and a large BIC advantage cannot validate
this model or its physical parameters.

**Cycle/Fc.** Fit only the inferred 30–150 s association and 150–230 s early
post-injection episode. The remaining full run is still plotted. No analyte
concentrations or event log are available. The best empirical model has nearly
equal association rates (~0.0313 s⁻¹), so their separate values and amplitudes
are poorly identifiable. Its residuals are large and smooth, and shifting the
inferred events by one second changes RMSE markedly. Its empirical decay
constants are not established koff values, and there is no kon or Kd estimate.

**T200 Cycle/sample.** Verified concentrations are 0.8868, 1.774, 3.547 µM.
The dominant population (~96% of capacity) has Kd ~49.0 nM and
koff ~8.1×10⁻⁵ s⁻¹; the minor component has Kd ~1.86 µM and
koff ~0.0173 s⁻¹. These are conditional estimates. All three concentrations
are well above the dominant Kd, limiting affinity information. Halving the
dissociation window increases dominant Kd and koff by ~22% and ~23%; a −1 s
event shift changes the minor-population Kd by ~27%. Residual correlations
remain 0.95–0.98 and RMSE is about 6–7 times the baseline noise estimate. The minor
component could absorb systematic effects; its presence is not proof of a
second molecular binding site.

**Carterra.** Verified concentrations are 0.0313, 0.25, 1.0 µM. One-site with
local Rmax gives Kd ~17.6 nM and koff ~7.55×10⁻⁵ s⁻¹. The transport model
adds no useful residual improvement and its transport coefficient approaches
the unconstrained, negligible-transport limit. Only ~5% of signal dissociates
over 720 s. Halving that window increases Kd by ~67% (~29.4 nM) and koff by
~53%, well beyond the nominal local intervals. Local capacities (~192, 202,
221 RU) and fixed measured initial occupancy provide a descriptive fit, not
proof of a constant, homogeneous surface. The selected injections are not a
complete non-regenerative series.

For reliable intrinsic constants, analyze the full concentration/injection
series with verified event logs and appropriate surface/reference controls.
For the slow Duke interactions, longer dissociation and concentrations near
the apparent affinity would help. Experimental checks are needed to distinguish
transport, surface heterogeneity and molecular mechanisms.

## Metadata and licensing

[fit_metadata.json](data/spr_exports/fit_metadata.json) records concentrations,
event times, baseline windows, regeneration/RI settings, provenance links, and
the source sample-sheet rows. The original Duke companion workbooks are
included unmodified in [metadata/](data/spr_exports/metadata/), with SHA-256
checksums and CC BY 4.0 attribution in the [source notes](data/spr_exports/SOURCES.md).

- T200: baseline −60–0 s, association 0–180 s, dissociation 180–780 s,
  regenerated; association-only bulk/RI terms are allowed for every candidate.
  Selected cycles 20–22 map to entries 6–8 of the ascending molar list.
- Carterra: baseline 0–120 s, association 120–420 s, dissociation 420–1140 s,
  no regeneration. Selected worksheet rows 9, 12, 14 map to concentration-list
  entries 8, 11, 13 by source row order, following the author's import layout.
  Literal `Conc. N` strings are not interpreted as molar values.
- rstoolbox and Cycle/Fc timings are inferred from the curves and marked
  **unverified**. No physical concentration unit is invented for either file.

The Carterra surface is treated as occupied before injection. Its original
median signal over 90–115 s is retained as the physical initial bound signal,
with zero additional background assumed. `align_association` changes the
response coordinate by subtracting the last-10-baseline mean; the model
subtracts that same offset from its physical response. Thus the fitted plot
starts near zero without treating the surface as empty. Unknown background
and uncertainty in this initial estimate are limitations. Hidden-state models
would require the complete preceding injection history or additional
initial-state allocation parameters, so only one-site and lumped transport
are compared for this subset.

## Reproducible method

The notebook helper [spr_fitting.py](spr_fitting.py) uses pyKinGenie's forward
binding functions and SciPy bounded least squares. It is separate from the
production fitter API so occupied initial states, true event clocks, RI terms,
and masks are explicit. The analytical mass-transport model is the lumped
quasi-steady model already implemented by pyKinGenie.

Before every fit, the notebook makes a deep copy of the imported experiment
and calls **`fit_experiment.align_association(inplace=True, npoints=10)`**.
For each sensor this subtracts the mean of the last 10 baseline observations
from baseline, association and dissociation. The alignment uses native
observations, so its averaging duration depends on the sampling interval.
Original times and the raw import plot are preserved. The notebook displays
the removed offsets and verifies a zero mean over those aligned baseline
points. The offsets are saved in each audit report under
`metadata.association_alignment`. All sensitivity runs retain this alignment.

The helper does not apply another baseline subtraction to the aligned copy.
Regenerated/unknown-regeneration series assume zero initial bound signal
after alignment. For non-regenerated Carterra, the original physical initial
occupancy is reconstructed by adding the alignment offset back to the median
aligned signal over 90–115 s; predictions then subtract the alignment offset.
Capacities and their optimizer bounds remain in physical bound-response units.

All candidates for a given file share the same observations, alignment offsets,
initial-state estimates, event times, and equal response-error weighting. A
three-second guard excludes transition points. At most one **original** point
per one-second interval is fitted, without averaging the fitted observations,
interpolation or filtering.
Carterra's slower native observations are retained. Discarding oversampled
points does not make the remaining errors independent. Pre-injection points
estimate fixed alignment offsets/initial signal and are not scored. Every fitted
rate, fraction, capacity and RI offset is counted in k. Fixed input estimates
are not counted as fitted parameters, making rankings conditional on them.

Association state is propagated from the actual start to the actual end even
if the first/last scored points are masked. Dissociation starts from that end
state, rather than the last scored association point. Regenerated binding
series compare one-site, transport, two independent ligand populations and
induced fit, each with shared or local Rmax. Induced-fit state signals are
assumed equal. Capacity fractions are shared across curves. These are candidate
approximations, not an exhaustive model search.

Eight starts use seed 2918; positive parameters use log coordinates. Bounds are
recorded in the helper. Rates span 10⁻⁷–1 s⁻¹, affinities span five orders
below/above the concentration range, fractions span 0.01–0.99, and capacities
have broad response-scaled bounds. The optimizer uses up to 700 evaluations
per main start. Sensitivity fits use two starts (including the primary solution)
and up to 3,000 evaluations per start. Unconverged models are reported and
excluded. Successful convergence is not proof of a global minimum.

Use `BIC = n log(RSS/n) + k log(n)`, dropping constants common to candidates.
The unknown common noise-variance parameter would add the same penalty to
each candidate. Reduced χ² is displayed as `RSS/(n-k)` with **unit variance**;
it has response-squared units and is not a calibrated chi-square test.
BIC values across files or different observation windows are not compared.
Correlated and systematic residuals weaken the independent-Gaussian-error
interpretation of the rankings.

Parameter intervals use local Jacobian curvature with RSS/(n−k) variance and
are explicitly labeled **conditional iid intervals**. Singular directions are
not turned into precise intervals with a pseudoinverse. They do not account for
serial correlation, uncertain concentrations, event times, baseline/initial
states, or model misspecification. Correlation matrices and normalized-Jacobian
condition numbers diagnose local identifiability, not scientific validation.
Baseline noise uses the robust MAD of one-second first differences divided by
√2; drift/correlation can bias it, so RMSE/noise is only a supporting diagnostic.

Window checks use 6/10 s transition guards, 5 s observation spacing, half of the
dissociation duration, and ±1 s event shifts. Report parameter ratios to the
primary solution, not cross-window BIC comparisons. These checks are not
cross-validation or calibrated uncertainty estimates. No leave-one-out fits
are run. The case-specific verdicts summarize these diagnostics and metadata
limitations; the helper does not implement a general automated trust classifier.

Scientific context: [Cytiva kinetics/affinity guide](https://cdn.cytivalifesciences.com/api/public/content/digi-33041-pdf)
and [Schuck & Zhao on transport and surface heterogeneity](https://pmc.ncbi.nlm.nih.gov/articles/PMC4134667/)
discuss the physical controls needed to interpret SPR kinetics. The numerical
results and decisions above come from the notebook analyses.
