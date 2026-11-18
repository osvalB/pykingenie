# Third-party sample data

The SPR sample files listed below retain their upstream licenses. The project's
MIT license does not replace the CC BY 4.0 license on the TitrationAnalysis data.
These are modified excerpts of public examples; their inclusion does not imply
endorsement by their authors or instrument manufacturers.

The rendered sensorgrams and data tables in import notebooks 12–15 are derived
from full-resolution selections of these same sources and retain the respective
data licenses below. Each
notebook credits its source and links the retained license and transformations.
The fitted plots, notebook result tables and metadata tables retain the same
source-data credits. The original Duke companion workbooks and published
reference CSV reports in `notebooks/data/spr_exports/metadata/` and their normalized rows in
`fit_metadata.json` are also CC BY 4.0, Kan Li and S Moses Dennison, Duke
University. The workbooks and reference reports are unmodified; selection, unit conversion and
fitting changes are described in the teaching-data source notes.

The four filenames in the table exist both as small regression fixtures in
`test_files/spr_exports/` and as representative teaching selections in
`notebooks/data/spr_exports/`. The same source credits and license terms apply
to both. The teaching selections, omissions and source checksums are detailed
in their own [source notes](notebooks/data/spr_exports/SOURCES.md).

| Files in both sample directories | Origin and credit | License and retained notice |
| --- | --- | --- |
| `paired_run.csv` | [rstoolbox](https://github.com/lpdi-epfl/rstoolbox), copyright (c) 2017 Jaume Bonet | MIT; [full notice](test_files/spr_exports/rstoolbox-LICENSE) |
| `biacore_cycle_fc.tsv` | [SPR_kinetics_binning](https://github.com/RKI-ZBS/SPR_kinetics_binning), copyright (c) 2024 SternDan | MIT; [full notice](test_files/spr_exports/SPR_kinetics_binning-LICENSE) |
| `biacore_cycle_sample.tsv`, `carterra_rows.csv` | [TitrationAnalysis tool](https://github.com/DukeCHSI/TitrationAnalysis), Kan Li and S Moses Dennison, Duke University | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); [full notice](test_files/spr_exports/TitrationAnalysis-LICENSE) |

MIT permits copying, modification and redistribution with the copyright and
permission notices retained. CC BY 4.0 permits redistribution and adaptation,
including commercial use, with attribution, a license link, and an indication
of changes; it prohibits additional restrictions on the licensed material.
The notices above, retained license texts, and linked provenance record provide
those credits, links and change descriptions for these fixtures.

See [SOURCES.md](test_files/spr_exports/SOURCES.md) for the exact upstream paths,
pinned revisions, source checksums, excerpt/conversion steps, and license review.
No upstream parser code has been copied into the new readers.

Suggested citation for the Duke examples, as requested upstream:

Li, Kan, and Dennison, S Moses. (2023). *TitrationAnalysis tool* (v1.0.0).
Zenodo. [doi:10.5281/zenodo.7998652](https://doi.org/10.5281/zenodo.7998652).
The license assessment uses the later pinned repository revision documented in
SOURCES.md, rather than the license file in the original v1.0.0 tag.
