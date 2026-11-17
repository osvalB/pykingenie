# Public SPR export fixtures

These are excerpts of public examples, not newly acquired measurements. Only
file-layout compatibility is tested; injection timing and molar concentrations
are not established by these fixtures. The sparse paired-column excerpts are
unsuitable for estimating kinetic parameters.

| Fixture | Original source (pinned revision) | License |
| --- | --- | --- |
| `paired_run.csv` | [rstoolbox: notebook/data/example03/spr_data.txt](https://github.com/lpdi-epfl/rstoolbox/blob/2bfc9be5ec6111603fb78700f201b80331ef0132/notebook/data/example03/spr_data.txt) | MIT, Jaume Bonet; see `rstoolbox-LICENSE` |
| `biacore_cycle_fc.tsv` | [SPR_kinetics_binning: Binning/Binning_Raw_Inspector_T200/input/2022-05-10_Kurven_Binning_huCapRaw.txt](https://github.com/RKI-ZBS/SPR_kinetics_binning/blob/ceaae964586dcbb00ac53bcae301bc097d92f3f4/Binning/Binning_Raw_Inspector_T200/input/2022-05-10_Kurven_Binning_huCapRaw.txt) | MIT; see `SPR_kinetics_binning-LICENSE` |
| `biacore_cycle_sample.tsv` | [TitrationAnalysis: Example Files - T200/Example T200 Chip 1 Channel 2.txt](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Example%20T200%20Chip%201%20Channel%202.txt) | CC BY 4.0, Kan (Jonathan) Li / Duke CHSI; see `TitrationAnalysis-LICENSE` |
| `carterra_rows.csv` | [TitrationAnalysis: Example Files - Carterra](https://github.com/DukeCHSI/TitrationAnalysis/tree/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra), `Example Exported Data - Carterra.zip.001` and `.002` | CC BY 4.0, Kan (Jonathan) Li / Duke CHSI; see `TitrationAnalysis-LICENSE` |

Changes made to prepare these fixtures:

- Paired exports: retain the original header and values in columns 0–3
  (zero-based). The rstoolbox excerpt also retains columns 22–23, an exported
  fitted-curve pair, to check exclusion. Keep 40 source rows, using integer
  indices from `numpy.linspace(0, n_rows - 1, 40, dtype=int)` after omitting
  completely empty source rows. Omit selected rows with no data in the retained
  columns. No interpolation or changes to numeric values.
  Preserve decimal commas in the Cycle/Fc example. Write a UTF-8 BOM and LF
  line endings; remove trailing whitespace at the end of the header line.
- Carterra: join the two public ZIP parts, read the worksheet's cached cell
  values from `Example Exported Data - Carterra.xlsx`, and save the header and first three sensorgrams
  as UTF-8 CSV. Retain all 534 X/Y pairs. This is a CSV conversion of the public
  workbook, not a claim of native XLSX support.

SHA-256 of the complete downloaded source files before excerpting:

```text
spr_data.txt:
1bf92879abcbcf4d56f21d92d0570c1e9f438dc0807e5ee60aa8da729b3619e5
2022-05-10_Kurven_Binning_huCapRaw.txt:
0bf881f731c4bf9fd93185f42d825b82ff358b0ff37809f22aefde9850971364
Example T200 Chip 1 Channel 2.txt:
439fdd9d96dea646f111473697532fa766fd4d0be1f39ca18bcbdd200043300a
Enclosed Carterra XLSX:
8a940a49b1dacafd7f918e155caa59339214c2d31042c434063788e155df73a7
```

The rstoolbox example establishes its Run/Ch/Cy/conc layout, but does not
establish an instrument identity or concentration unit. The Duke T200 labels
include bare numbers and truncated mass-concentration units. Carterra's
`Conc. N` denotes a concentration index. None of these are assumed to be µM.
Parser code is implemented independently; no upstream parser code is included.
