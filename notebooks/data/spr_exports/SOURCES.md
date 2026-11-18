# Representative SPR teaching curves

These files are full-resolution selections for import notebooks 12–15. They are
separate from the small regression fixtures in
[`test_files/spr_exports/`](../../../test_files/spr_exports/SOURCES.md).
Curves were selected to show clear measured responses, not to estimate model
recovery or summarize the original experiments. No curve is simulated, smoothed,
fitted, baseline-subtracted, or resampled.

The [manifest](manifest.json) records the original source hashes, exact selected
columns/rows, retained labels, per-curve counts, missing-pair omissions and output
hashes. Every retained timestamp and response was compared numerically with its
source value after importing the teaching file through the production reader:
all values match exactly.

| Teaching file | Original source at pinned revision | Selection | Retained points per curve | License / credit |
| --- | --- | --- | --- | --- |
| `paired_run.csv` | [rstoolbox, notebook/data/example03/spr_data.txt](https://github.com/lpdi-epfl/rstoolbox/blob/2bfc9be5ec6111603fb78700f201b80331ef0132/notebook/data/example03/spr_data.txt) | Measured cycles 14–16, `conc=15`, `30`, `60`; zero-based columns 16–21 | 5,621 / 5,621 / 5,621 | MIT, copyright (c) 2017 Jaume Bonet; [retained license](../../../test_files/spr_exports/rstoolbox-LICENSE) |
| `biacore_cycle_fc.tsv` | [RKI-ZBS, Biacore T200 binning export](https://github.com/RKI-ZBS/SPR_kinetics_binning/blob/ceaae964586dcbb00ac53bcae301bc097d92f3f4/Binning/Binning_Raw_Inspector_T200/input/2022-05-10_Kurven_Binning_huCapRaw.txt) | Complete cycles 1 and 2, flow-cell pair 4-3; zero-based columns 0–3 | 1,105 / 1,104 | MIT, copyright (c) 2024 SternDan; [retained license](../../../test_files/spr_exports/SPR_kinetics_binning-LICENSE) |
| `biacore_cycle_sample.tsv` | [Duke TitrationAnalysis, Example T200 Chip 1 Channel 2.txt](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Example%20T200%20Chip%201%20Channel%202.txt) | Measured cycles 20–22, verbatim labels `50 µg`, `100 µ`, `200 µ`; zero-based columns 10–15 | 8,404 / 8,395 / 8,402 | CC BY 4.0, Kan Li and S Moses Dennison, Duke University; [retained license](../../../test_files/spr_exports/TitrationAnalysis-LICENSE) |
| `carterra_rows.csv` | [Duke TitrationAnalysis, original Carterra workbook](https://github.com/DukeCHSI/TitrationAnalysis/tree/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra) | Worksheet rows 9, 12, 14 (one-based), Ligand 1 / AE.A244: `Conc. 6`, `Conc. 3`, `53 µg/ml` | 533 / 523 / 522 | CC BY 4.0, Kan Li and S Moses Dennison, Duke University; [retained license](../../../test_files/spr_exports/TitrationAnalysis-LICENSE) |

## Transformations and missing observations

- Paired text files retain every finite source observation in the selected
  curves, in the original order. No finite measurement is removed. Entirely
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
- The Duke Cycle/sample source has **eight missing/non-finite pairs in cycle
  21**; only those pairs are omitted. Cycles 20 and 22 have no such omissions.
- The Carterra archive parts are joined, and cached worksheet values from
  `Example Exported Data - Carterra.xlsx` are converted to CSV. The original
  `Label, X1, Y1, ..., X534, Y534` header and selected row labels are retained.
  Spreadsheet cell coordinates are respected when preserving blank pairs.
  No observation from the three selected rows is removed.
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
No plate-well mapping or injection-event timing is inferred. API phase
boundaries shown in the notebooks are illustrative and do not affect the
concatenated raw sensorgram plots.

The original license audit, including TitrationAnalysis's later CC BY 4.0
license and older conflicting citation metadata, is recorded in the
[original source notes](../../../test_files/spr_exports/SOURCES.md).
These teaching files and their plotted sensorgrams retain the upstream data
licenses. See [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md) for the full
credits and requested Duke citation. Inclusion does not imply endorsement.
