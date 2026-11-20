# Independent SPR parser comparisons

Run on 2026-11-17, with the production readers from commit `1dcf06d`.
The checked-in [results.json](results.json) contains source/input hashes,
versions, counts, numerical errors and the known behavioral difference. No
production reader was changed as a result of these comparisons.

| Comparison | Measured curves | Complete time/response pairs | Maximum absolute time/response difference |
| --- | ---: | ---: | ---: |
| rstoolbox fixture, original CSV | 2 | 40 | 0 / 0 |
| Same fixture, TSV | 2 | 40 | 0 / 0 |
| Same fixture, gzip CSV | 2 | 40 | 0 / 0 |
| Same fixture, reordered curves and X/Y columns | 2 | 40 | 0 / 0 |
| Full rstoolbox example, complete observations only | 11 | 61,851 | 0 / 0 |
| Biacore Cycle/Fc fixture vs original R importer | 2 | 78 | 0 / 0 |
| Full Biacore Cycle/Fc example vs original R importer | 68 | 75,043 | 0 / 0 |
| Carterra CSV fixture vs native workbook via readxl | 3 | 1,602 | 0 / 0 |

Every numeric comparison checks curve identity, curve count, point count and
every timestamp and response, without interpolation, resampling, alignment or
subtraction. An absolute tolerance of `1e-10` is allowed for cross-language
serialization, but the observed difference was zero in all eight comparisons.
Data were concatenated across the three pyKinGenie phases before comparison;
these checks do not validate the caller-supplied injection boundaries.

## Missing-response behavior found in the full rstoolbox source

The full public rstoolbox source has **49 timestamps with `NaN` responses per
measured curve, 539 pairs in total**. The unchanged external parser retains
those missing values. pyKinGenie rejects them as non-finite measurements. Its
rejection of the original file is explicitly checked and recorded as
`intentional_behavior_difference`, separately from the numeric matches.

For the complete-observation comparison, the runner creates a temporary copy
with just those 539 partial pairs removed. Complete observations keep their
original values and order. The result is compared with the external parser's
complete observations from the original source. This filtering is explicit in
the runner and report; it is not a feature of the production reader. Neither
source files nor retained fixtures are edited by the comparison. The full
unmodified rstoolbox file therefore cannot currently be imported directly.

## References actually executed

- **rstoolbox 1.0.1**, installed in a separate Python 3.11.13 environment:
  [public `read_SPR` API](https://jaumebonet.cat/RosettaSilentToolbox/generated/rstoolbox.io.read_SPR.html).
  The installed `experimental.py` matches the complete module at
  [commit 2bfc9be](https://github.com/lpdi-epfl/rstoolbox/blob/2bfc9be5ec6111603fb78700f201b80331ef0132/rstoolbox/io/experimental.py)
  byte for byte. The runner verifies its SHA-256. It imports and calls the
  package normally, with no copied, monkeypatched or rewritten parser code.
  The adapter separates measured from fitted curves using the reference output.
  rstoolbox discards run/channel/cycle metadata, so curves are matched by unique
  concentration label, without assigning any physical concentration unit.
- **RKI's original Biacore R import section**, from
  [commit ceaae96](https://github.com/RKI-ZBS/SPR_kinetics_binning/blob/ceaae964586dcbb00ac53bcae301bc097d92f3f4/Binning/Binning_Raw_Inspector_T200/binDataImport_final.R).
  The runner verifies the complete source file's SHA-256, parses it with R and
  evaluates the unchanged expressions from `inputDFhuCap` through `outputHu`.
  This is a scoped comparison of the original import section, rather than
  execution of the full analysis script. Its workspace cleanup, package
  installer, report-sheet import and later normalization are excluded.
  The harness supplies an isolated working directory and loads dplyr/tidyr.
- **R readxl 1.4.5**, reading the first three curves directly from the original
  50 MB Carterra workbook. This independently checks the CSV conversion and
  all retained labels/times/responses. It is a generic Excel-reader comparison,
  **not execution of an independent Carterra SPR parser**.

R was 4.5.1, with dplyr 1.1.4, tidyr 1.3.1 and jsonlite 2.0.0. TitrationAnalysis
was not run because Mathematica is not installed. The Duke T200 Cycle/sample
layout consequently still has no executed domain-specific reference parser.
Missing concentration metadata and injection boundaries remain covered by the
normal reader tests, rather than by these external references.

## Reproduce

Run from the repository root. Keep the reference environment separate from
the current pyKinGenie environment: rstoolbox's legacy build system fails on
Python 3.12, and its dependencies use older pandas APIs.

```sh
uv venv --python 3.11 /tmp/spr-reference-env
uv pip install --python /tmp/spr-reference-env/bin/python \
  -r validation/spr_parsers/requirements-reference.txt
```

R needs `dplyr`, `tidyr`, `jsonlite` and `readxl`. Download the original R script
without editing it:

```sh
curl --fail --location \
  https://raw.githubusercontent.com/RKI-ZBS/SPR_kinetics_binning/ceaae964586dcbb00ac53bcae301bc097d92f3f4/Binning/Binning_Raw_Inspector_T200/binDataImport_final.R \
  --output /tmp/rki-reference.R
```

Using the current pyKinGenie environment, run all retained paired-column
fixture comparisons, including TSV/gzip/reordering variants:

```sh
PYTHONPATH=src python validation/spr_parsers/compare.py \
  --rstoolbox-python /tmp/spr-reference-env/bin/python \
  --rscript Rscript \
  --rki-source /tmp/rki-reference.R \
  --output /tmp/spr-parser-comparison.json
```

For the full run reported above, also supply:

```text
--full-rstoolbox /path/to/spr_data.txt
--full-biacore /path/to/2022-05-10_Kurven_Binning_huCapRaw.txt
--carterra-xlsx "/path/to/Example Exported Data - Carterra.xlsx"
```

The full data files are optional local inputs, downloaded from the pinned
sources in [SOURCES.md](../../test_files/spr_exports/SOURCES.md). The Carterra
workbook is extracted from the two original ZIP parts; it is not an Excel file
regenerated from our CSV. Paths and transient directories are not stored in the
results. The large original files and external packages are not vendored or
added as normal test/runtime dependencies. The retained data licenses and
attribution remain in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
