"""Choose and run the reader for a set of input files.

``KineticsAnalyzer.load`` calls ``load_experiment``. It runs
``detect_file_format`` on every file, checks that the files form one
experiment of one format, creates the matching experiment class and calls its
reader. Settings the files do not contain (SPR timing, concentrations,
``info=``) are passed through as keyword options to that reader.
"""

from pathlib import Path

from .biacore import BiacoreExperiment
from .carterra import CarterraExperiment
from .file_detection import detect_file_format
from .gator import GatorExperiment
from .kingenie_solution import KinGenieCsvSolution
from .kingenie_surface import KinGenieCsv
from .octet import OctetExperiment
from .spr_text import SprTextExperiment
from .surface_csv import SurfaceCsvExperiment
from .utils.processing import guess_experiment_name

__all__ = ["load_experiment"]

_GATOR_PARTS = {"gator_steps", "gator_settings", "gator_csv"}
# Single-file formats: experiment class and the name of its reader method.
_SINGLE = {
    "biacore_blr": (BiacoreExperiment, "read_blr"),
    "kingenie_surface": (KinGenieCsv, "read_csv"),
    "kingenie_solution": (KinGenieCsvSolution, "read_csv"),
    "spr_text": (SprTextExperiment, "read_file"),
    "carterra_csv": (CarterraExperiment, "read_csv"),
}
# Readers that take keyword options; the others read the files alone.
_WITH_OPTIONS = {"spr_text", "carterra_csv", "surface_csv"}


def _format_of(file):
    # Octet plate layouts (.fmf) are not content-detected; they only make sense
    # next to .frd files, which is checked below.
    if file.suffix.casefold() == ".fmf":
        return "octet_fmf"
    return detect_file_format(file)


def _describe(files, formats):
    return "; ".join(f"{file.name}: {kind}" for file, kind in zip(files, formats))


def load_experiment(files, name=None, taken=(), **options):
    """Read ``files`` with the reader their format needs; return the experiment.

    ``files`` is one path or a list of paths that together form one
    experiment: Octet ``.frd`` files (plus optional ``.fmf`` plate files), a
    Biacore ``.blr`` file, a Gator zip or its ``Setting.ini``,
    ``ExperimentStep.ini`` and sensor CSVs, a KinGenie surface or solution CSV,
    a generic steps file plus traces file, a paired-column SPR export, or a
    Carterra CSV. ``options`` go to the reader and are accepted only by the
    readers that take them (SPR text, Carterra, generic steps/traces). The
    experiment is named ``name``, or after the Octet run or the file, with a
    number appended if that automatic name is in ``taken``.
    """
    if isinstance(files, (str, Path)):
        files = [files]
    files = [Path(file) for file in files]
    if not files:
        raise ValueError("No files to load.")
    formats = [_format_of(file) for file in files]
    kinds = set(formats)

    if "unknown" in kinds:
        raise ValueError(
            "Unrecognized input; load it with its experiment class instead. "
            + _describe(files, formats)
        )

    def of(kind):
        return [str(file) for file, k in zip(files, formats) if k == kind]

    if kinds <= {"octet_frd", "octet_fmf"} and "octet_frd" in kinds:
        kind = "octet"
        default = guess_experiment_name(of("octet_frd")[0])
    elif kinds == {"gator_zip"} and len(files) == 1:
        kind = "gator"
        default = files[0].stem
    elif kinds <= _GATOR_PARTS and {"gator_steps", "gator_settings"} <= kinds:
        if len(of("gator_steps")) > 1 or len(of("gator_settings")) > 1 or not of("gator_csv"):
            raise ValueError(
                "A Gator experiment needs one ExperimentStep.ini, one Setting.ini "
                "and its sensor CSV files. " + _describe(files, formats)
            )
        kind = "gator"
        default = files[0].parent.name
    elif kinds == {"surface_csv_steps", "surface_csv_traces"} and len(files) == 2:
        kind = "surface_csv"
        default = Path(of("surface_csv_traces")[0]).stem
    elif len(files) == 1 and formats[0] in _SINGLE:
        kind = formats[0]
        default = files[0].stem
    else:
        raise ValueError(
            "These files do not form one experiment of a single format. "
            + _describe(files, formats)
        )

    if options and kind not in _WITH_OPTIONS:
        raise TypeError(
            f"The {kind} reader takes no options; got " + ", ".join(sorted(options)) + "."
        )
    if name is None:
        name, number = default, 2
        while name in taken:
            name, number = f"{default} ({number})", number + 1

    if kind == "octet":
        experiment = OctetExperiment(name)
        experiment.read_sensor_data(of("octet_frd"))
        if of("octet_fmf"):
            experiment.read_sample_plate_info(of("octet_fmf"))
    elif kind == "gator":
        experiment = GatorExperiment(name)
        # A zip is extracted next to itself by the Gator reader.
        experiment.read_all_gator_data(str(files[0]) if kinds == {"gator_zip"}
                                       else [str(file) for file in files])
    elif kind == "surface_csv":
        experiment = SurfaceCsvExperiment(name)
        experiment.read_files(of("surface_csv_steps")[0], of("surface_csv_traces")[0],
                              **options)
    else:
        cls, method = _SINGLE[kind]
        experiment = cls(name)
        getattr(experiment, method)(str(files[0]), **options)
    return experiment
