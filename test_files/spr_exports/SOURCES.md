# Public SPR export fixtures

These are excerpts of public examples, not newly acquired measurements. Only
file-layout compatibility is tested; injection timing and molar concentrations
are not established by these fixtures. The sparse paired-column excerpts are
unsuitable for estimating kinetic parameters.

| Fixture | Original source (pinned revision) | License |
| --- | --- | --- |
| `paired_run.csv` | [rstoolbox: notebook/data/example03/spr_data.txt](https://github.com/lpdi-epfl/rstoolbox/blob/2bfc9be5ec6111603fb78700f201b80331ef0132/notebook/data/example03/spr_data.txt) | MIT, copyright (c) 2017 Jaume Bonet; [retained license](rstoolbox-LICENSE) |
| `biacore_cycle_fc.tsv` | [SPR_kinetics_binning: Binning/Binning_Raw_Inspector_T200/input/2022-05-10_Kurven_Binning_huCapRaw.txt](https://github.com/RKI-ZBS/SPR_kinetics_binning/blob/ceaae964586dcbb00ac53bcae301bc097d92f3f4/Binning/Binning_Raw_Inspector_T200/input/2022-05-10_Kurven_Binning_huCapRaw.txt) | MIT, copyright (c) 2024 SternDan; [retained license](SPR_kinetics_binning-LICENSE) |
| `biacore_cycle_sample.tsv` | [TitrationAnalysis: Example Files - T200/Example T200 Chip 1 Channel 2.txt](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20T200/Example%20T200%20Chip%201%20Channel%202.txt) | CC BY 4.0, Kan Li and S Moses Dennison, Duke University; [retained license](TitrationAnalysis-LICENSE) |
| `carterra_rows.csv` | [TitrationAnalysis: Example Files - Carterra](https://github.com/DukeCHSI/TitrationAnalysis/tree/cb8112503658d24c4c6576503243fa506bf99e27/Example%20Files%20-%20Carterra), `Example Exported Data - Carterra.zip.001` and `.002` | CC BY 4.0, Kan Li and S Moses Dennison, Duke University; [retained license](TitrationAnalysis-LICENSE) |

## License verification — 2026-11-17

The reuse assessment is based on the licenses distributed with these exact
repository revisions, not on the files merely being publicly downloadable.
The root LICENSE, root README, and complete recursive file tree were inspected
for each revision. No separate license, copyright notice file, or restrictive
README was found in the sample directories or their parent directories. The
three retained license files match the pinned upstream LICENSE files byte for
byte.

- rstoolbox: [upstream MIT license](https://github.com/lpdi-epfl/rstoolbox/blob/2bfc9be5ec6111603fb78700f201b80331ef0132/LICENSE).
  The example is distributed in its notebook data directory. We apply the
  repository's MIT license to this example and retain its full notice.
- SPR_kinetics_binning: [upstream MIT license](https://github.com/RKI-ZBS/SPR_kinetics_binning/blob/ceaae964586dcbb00ac53bcae301bc097d92f3f4/LICENSE).
  Its [README](https://github.com/RKI-ZBS/SPR_kinetics_binning/blob/ceaae964586dcbb00ac53bcae301bc097d92f3f4/README.md)
  describes the repository as containing both the measurement data and analysis
  scripts. We apply its repository MIT license and retain its full notice.
- TitrationAnalysis: the pinned [LICENSE](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/LICENSE)
  is CC BY 4.0. The pinned [README](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/README.md)
  explicitly licenses the project under CC BY 4.0 and identifies these files as
  the supplied example inputs. We retain CC BY 4.0 for the derived fixtures,
  credit both cited creators, link the original material and license, and state
  the changes below. These fixtures are not relicensed under pyKinGenie's MIT
  license. See the [top-level notices](../../THIRD_PARTY_NOTICES.md).

### TitrationAnalysis metadata discrepancy

The pinned [CITATION.cff](https://github.com/DukeCHSI/TitrationAnalysis/blob/cb8112503658d24c4c6576503243fa506bf99e27/CITATION.cff)
still says `GPL-3.0`. The original `v1.0.0` tag also has a GPLv3 LICENSE. The
repository subsequently changed its license: [commit 4701a57, June 5, 2023](https://github.com/DukeCHSI/TitrationAnalysis/commit/4701a57f155f2f122a88e409509b3be48d0211b7),
by `jonathanliduke`, installed the CC BY 4.0 text retained here. The current
README agrees with that text. The [Zenodo record](https://zenodo.org/records/7998652)
also declares `cc-by-4.0` in its [API metadata](https://zenodo.org/api/records/7998652),
checked on the review date. The T200 example and both Carterra ZIP parts have
identical Git blob IDs at the original tag and the pinned revision.

The basis for CC BY 4.0 reuse here is the explicit LICENSE and README at the
later pinned revision, supported by the author-made license change. The
conflicting citation field is recorded rather than silently treated as
consistent. The original GPL-tagged license is not the license being applied
to these excerpts.

## Fixture transformations

Changes made to prepare these fixtures:

- Paired exports: retain the original header and values in columns 0–3
  (zero-based). The rstoolbox excerpt also retains columns 22–23, an exported
  fitted-curve pair, to check exclusion. Keep 40 source rows, using integer
  indices from `numpy.linspace(0, n_rows - 1, 40, dtype=int)` after omitting
  completely empty source rows. Omit selected rows with no data in the retained
  columns. No interpolation or changes to numeric values.
  Preserve decimal commas in the Cycle/Fc example. Write a UTF-8 BOM and LF
  line endings; remove trailing whitespace at the end of the header line.
  The Duke T200 source is Windows-1252 text; transcode it to UTF-8 without
  changing decoded labels or measurement values.
- Carterra: join the two public ZIP parts, read the worksheet's cached cell
  values from `Example Exported Data - Carterra.xlsx`, and save the header and first three sensorgrams
  as UTF-8 CSV. Retain all 534 X/Y pairs. This is a CSV conversion of the public
  workbook, not a claim of native XLSX support.

## Source checksums

SHA-256 of the complete upstream source files before text conversion or excerpting:

```text
spr_data.txt:
1bf92879abcbcf4d56f21d92d0570c1e9f438dc0807e5ee60aa8da729b3619e5
2022-05-10_Kurven_Binning_huCapRaw.txt:
0bf881f731c4bf9fd93185f42d825b82ff358b0ff37809f22aefde9850971364
Example T200 Chip 1 Channel 2.txt:
0361356ba6cb7be1a7f38ab9c5c5976407af0b78791bb9fad0a780dbc4751aa0
Enclosed Carterra XLSX:
8a940a49b1dacafd7f918e155caa59339214c2d31042c434063788e155df73a7
```

The complete text sources and both Carterra archive parts were verified against
their Git blob IDs at the pinned revisions. The Duke T200 checksum above is
for the original Windows-1252 bytes, rather than the UTF-8 working copy used
when preparing the excerpt.

The rstoolbox example establishes its Run/Ch/Cy/conc layout, but does not
establish an instrument identity or concentration unit. The Duke T200 labels
include bare numbers and truncated mass-concentration units. Carterra's
`Conc. N` denotes a concentration index. None of these are assumed to be µM.
Parser code is implemented independently; no upstream parser code is included.
