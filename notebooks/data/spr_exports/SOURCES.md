# Representative SPR teaching curves

These files are full-resolution selections for import notebooks 12–15. They are
separate from the small regression fixtures in
[`test_files/spr_exports/`](../../../test_files/spr_exports/SOURCES.md).
The rstoolbox and binning selections show clear measured responses. The two
Duke selections contain the exact five concentration curves identified on page 1
of the authors' sensorgram reports, within their nominal protocol windows.
These examples do not estimate model recovery or summarize all source
experiments. No curve is simulated, smoothed, fitted, baseline-subtracted, or
resampled.

The [manifest](manifest.json) records the original source hashes, exact selected
columns/rows, retained labels, per-curve counts, missing-pair omissions and output
hashes. Every retained timestamp and response was compared numerically with its
source value after importing the teaching file through the production reader:
all values match exactly.

| Teaching file | Original source at pinned revision | Selection | Retained points per curve | License / credit |
| --- | --- | --- | --- | --- |
| `paired_run.csv` | [rstoolbox, notebook/data/example03/spr_data.txt](https://github.com/lpdi-epfl/rstoolbox/blob/2bfc9be5ec6111603fb78700f201b80331ef0132/notebook/data/example03/spr_data.txt) | Measured cycles 14–16, `conc=15`, `30`, `60`; zero-based columns 16–21 | 5,621 / 5,621 / 5,621 | MIT, copyright (c) 2017 Jaume Bonet; [retained license](../../../test_files/spr_exports/rstoolbox-LICENSE) |
| `biacore_cycle_fc.tsv` | [RKI-ZBS, Biacore T200 binning export](https://github.com/RKI-ZBS/SPR_kinetics_binning/blob/ceaae964586dcbb00ac53bcae301bc097d92f3f4/Binning/Binning_Raw_Inspector_T200/input/2022-05-10_Kurven_Binning_huCapRaw.txt) | Complete cycles 1 and 2, flow-cell pair 4-3; zero-based columns 0–3 | 1,105 / 1,104 | MIT, copyright (c) 2024 SternDan; [retained license](../../../test_files/spr_exports/SPR_kinetics_binning-LICENSE) |
| `biacore_cycle_sample.tsv` | [Duke TitrationAnalysis, Example T200 Chip 1 Channel 2.txt](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Example%20T200%20Chip%201%20Channel%202.txt) | Author report cycles 18–22, 221.7–3547 nM; zero-based columns 6–15; window −60–780 s | 8,221 / 8,391 / 8,400 / 8,392 / 8,400 | CC BY 4.0, Kan Li and S Moses Dennison, Duke University; [retained license](../../../test_files/spr_exports/TitrationAnalysis-LICENSE) |
| `carterra_rows.csv` | [Duke TitrationAnalysis, original Carterra workbook](https://github.com/DukeCHSI/TitrationAnalysis/tree/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra) | Author report worksheet rows 10–14 (one-based), Block 1/A1/ROI 1, 62.5–1000 nM; window 0–1140 s | 519 / 517 / 510 / 510 / 509 | CC BY 4.0, Kan Li and S Moses Dennison, Duke University; [retained license](../../../test_files/spr_exports/TitrationAnalysis-LICENSE) |

## Transformations and missing observations

- Paired text files retain every finite source observation in the selected
  curves and stated windows, in the original order. The rstoolbox and binning
  selections retain their full available time ranges; the Duke examples retain
  their author-specified protocol windows. Entirely
  empty rows outside those curves are omitted. Column labels have outer
  whitespace trimmed. Exports use UTF-8 BOM and LF line endings; decimal commas
  remain unchanged in the Cycle/Fc TSV. Empty trailing TSV fields are omitted
  when a curve ends earlier; the reader pads those missing pair slots. The Duke source text is transcoded from
  Windows-1252 to UTF-8 without changing its decoded labels or numeric values.
- The rstoolbox source has 49 missing/non-finite response pairs in each selected
  curve, **147 pairs total**. Both cells of just those pairs are cleared in the
  teaching export. The strict reader skips the resulting empty pair, preserving
  the corresponding sampling gap. Its behavior on the unmodified source still
  raises an error, as recorded in the original
  [reference comparison](../../../validation/spr_parsers/README.md).
- The Duke Cycle/sample source has **196 missing/non-finite pairs** in the
  five selected curves: **179 in cycle 18, 9 in cycle 19, 0 in cycle 20,
  8 in cycle 21, 0 in cycle 22**. Only these unavailable pairs are cleared.
  Additionally, **16 finite observations outside −60–780 s** are omitted
  (3 / 4 / 4 / 3 / 2 per curve). These are protocol-window exclusions, not
  downsampling; all finite in-window values are retained unchanged.
- The Carterra archive parts are joined, and cached worksheet values from
  `Example Exported Data - Carterra.xlsx` are converted to CSV. The original
  `Label, X1, Y1, ..., X534, Y534` header and selected row labels are retained.
  Spreadsheet cell coordinates are respected when preserving blank pairs.
  For worksheet rows 10–14, all finite observations in 0–1140 s are retained.
  **66 observations after 1140 s** are omitted (13 / 14 / 13 / 13 / 13 per row)
  to match the nominal protocol end. No in-window observation is removed.
- No missing response is interpolated, replaced by zero, or assigned a fitted
  value. Notebook plots break lines across gaps exceeding three median sampling
  intervals. The Cycle/Fc notebook's zoom changes display limits only; the full
  binning cycles remain in its left panel and in the imported data.

## Interpretation and reuse

The Cycle/Fc file is an epitope-binning protocol, so a complete cycle need not
look like a single association followed by exponential dissociation. The
teaching example displays that protocol honestly rather than converting it to
a one-site kinetic curve. Positive initial responses in the Carterra workbook
are also retained without resetting them to zero.

Concentration labels are copied verbatim. Unitless `conc=...` labels, Carterra
`Conc. N` indices, mass units, and truncated unit strings do not establish µM.
No plate-well mapping is inferred. The Duke notebooks now use the original
companion metadata below for molar concentrations and event timing; rstoolbox
and binning timing remains explicitly provisional. Import partitions do not
affect the concatenated raw sensorgram plots.

## Companion metadata used for fitting

Two original workbooks from the same pinned Duke revision are included
**unmodified** in `metadata/`. They retain **CC BY 4.0**, Kan Li and S Moses
Dennison, Duke University, and the [full retained license](../../../test_files/spr_exports/TitrationAnalysis-LICENSE).
Redistribution and adaptation are permitted with attribution and an indication
of changes. The normalized [fit_metadata.json](fit_metadata.json) is an adaptation:
it selects the relevant author rows, converts M to µM, maps selected curve
positions to concentrations, and records import phase times and source settings.
The original workbooks are not required at notebook runtime.

| Unmodified workbook | Original pinned source | SHA-256 |
| --- | --- | --- |
| `metadata/Example T200 Data Info Sheet.xlsx` | [Duke T200 sample sheet](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Example%20T200%20Data%20Info%20Sheet.xlsx) | `469af4090726a5d230bbe119ba769b5d17b3b9a5f9bc8edc1913d060c4a571ba` |
| `metadata/Example Data Info Sheet - Carterra.xlsx` | [Duke Carterra sample sheet](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra/Example%20Data%20Info%20Sheet%20-%20Carterra.xlsx) | `6688322663cba3a0c4f55e83e1a0b4f9a7a75f8efff72494fa2717464049c7e2` |

The T200 selected cycles 18–22 map to concentration-list entries **4–8**.
The Carterra selected worksheet rows **10–14** map by row position to ascending
entries **9–13**, following the original author's importer layout. Literal `Conc. N`
labels are not used to guess molar concentrations. See
[fits with published comparisons](../../SPR_FITTING.md) for the complete
mapping, existing pyKinGenie fitting API, reference reports and comparison scope.
The derived notebook tables and plots retain the corresponding source-data
credits and licenses. No fitted values replace the measured data files.
Before fitting, each Duke notebook calls `align_association` on a separate copy,
subtracting the mean of the last 10 pre-association points. This transformation
is shown explicitly in the notebooks; it does not modify these source excerpts.

## Published reference fitting reports

The following **unmodified** author output reports come from the same pinned
Duke revision as the data and sample sheets. They retain **CC BY 4.0**, Kan Li
and S Moses Dennison, Duke University, with the [retained license](../../../test_files/spr_exports/TitrationAnalysis-LICENSE).
The normalized `literature_reference` fields in `fit_metadata.json` select one
report row and retain its kon, koff and Kd values; they are adaptations of these
reports. Only the corresponding one-site/local-Rmax fits are demonstrated.

| Original report | Pinned source | SHA-256 |
| --- | --- | --- |
| `metadata/Titration Analysis_T200_Report_Example.csv` | [Duke T200 report](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Titration%20Analysis_T200_Report_Example.csv) | `cb144fadd6e8be97f6ebdaf5786bc7221ae54291859a5308a0d12f11340b0e5d` |
| `metadata/Titration Analysis_Carterra_Report_Example.csv` | [Duke Carterra report](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra/Titration%20Analysis_Carterra_Report_Example.csv) | `bd623cfc2f903ff0afbb3b844e624c2591a66a89ffd05dfc91073daca3b99cc3` |

Use T200 Chip 1 / Channel 2 / fitting #1, and Carterra Block 1 / A1 / ROI 1 /
fitting #1. Both notebooks now fit the **same five concentration curves** as
those reports, verified from the legends on **page 1** of the original sensorgram
PDFs and the companion concentration lists. The manifest records the PDF URLs,
SHA-256 hashes, page and selected concentrations. Association/dissociation
windows and zero skipped seconds also match the source sheets. Alignment,
sampling and fitting-state/bulk-shift settings still differ, so these are
comparisons of the same source experiments rather than exact reproductions.
Reference publication: [Li et al., TitrationAnalysis](https://doi.org/10.12688/gatesopenres.14743.2).

The original license audit, including TitrationAnalysis's later CC BY 4.0
license and older conflicting citation metadata, is recorded in the
[original source notes](../../../test_files/spr_exports/SOURCES.md).
These teaching files and their plotted sensorgrams retain the upstream data
licenses. See [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md) for the full
credits and requested Duke citation. Inclusion does not imply endorsement.
