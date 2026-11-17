"""Differential checks against external parsers, kept out of normal test dependencies."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import warnings

import numpy as np

from pykingenie import CarterraExperiment, SprTextExperiment

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "test_files" / "spr_exports"
RST_SOURCE_SHA256 = "0d8564b98946071f700fb038cd89dde1763510a05238bd7952aec174362e56cf"
RKI_SOURCE_SHA256 = "a39a2fb8a21b683119b20887b6fe38a61d212dc875d0bcee31a423a1637698ce"


def compare(exp, reference, key):
    expected = {curve["name"]: curve for curve in reference["curves"]}
    actual = {key(name): i for i, name in enumerate(exp.sensor_names)}
    if len(expected) != len(reference["curves"]) or len(actual) != exp.no_sensors:
        raise AssertionError("Curve identity is ambiguous.")
    assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
    points, errors = 0, {"time": 0.0, "response": 0.0}
    for name, i in actual.items():
        for field, arrays in [("time", exp.xs[i]), ("response", exp.ys[i])]:
            values = np.concatenate(arrays)
            target = np.asarray(expected[name][field], dtype=float)
            assert values.shape == target.shape, (name, field, values.shape, target.shape)
            np.testing.assert_allclose(values, target, rtol=0, atol=1e-10)
            errors[field] = max(errors[field], float(np.max(np.abs(values - target))))
        points += len(np.concatenate(exp.xs[i]))
    return {"curves": len(actual), "points": points, "max_absolute_error": errors}


def load(path, *, decimal=".", carterra=False):
    exp = CarterraExperiment() if carterra else SprTextExperiment()
    reader = exp.read_csv if carterra else exp.read_file
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", UserWarning)
        reader(path, association_end=300, decimal=decimal)
    return exp


def complete_observations(source, destination):
    """Explicit comparison policy; never applied inside the production reader."""
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    pairs = {}
    for i, label in enumerate(rows[0]):
        if "fitted curve" not in label:
            name, axis = label.strip().rsplit("_", 1)
            pairs.setdefault(name, {})[axis] = i
    removed = 0

    def missing(value):
        return value.strip().lower() in ("", "nan")

    for row in rows[1:]:
        for columns in pairs.values():
            x, y = columns["X"], columns["Y"]
            if missing(row[x]) != missing(row[y]):
                row[x] = row[y] = ""
                removed += 1
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rstoolbox-python", required=True)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--rki-source", type=Path, required=True)
    parser.add_argument("--full-rstoolbox", type=Path)
    parser.add_argument("--full-biacore", type=Path)
    parser.add_argument("--carterra-xlsx", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.rki_source.read_bytes()).hexdigest() == RKI_SOURCE_SHA256
    results = []
    with tempfile.TemporaryDirectory(prefix="spr-reference-") as temp:
        output = Path(temp) / "reference.json"

        def check(name, file, command, key, *, reference_input=None, **kwargs):
            # Silence only warnings from the legacy reference's dependencies.
            subprocess.run(
                command,
                check=True,
                env={**os.environ, "PYTHONWARNINGS": "ignore::FutureWarning"},
            )
            reference = json.loads(output.read_text())
            if reference.get("source_sha256") is not None:
                assert reference["source_sha256"] == RST_SOURCE_SHA256
            original_sha = hashlib.sha256(file.read_bytes()).hexdigest()
            reference_sha = hashlib.sha256((reference_input or file).read_bytes()).hexdigest()
            normalization = None
            if reference.get("incomplete_pairs"):
                try:
                    load(file, **kwargs)
                except ValueError as error:
                    reason = str(error)
                    assert "Incomplete X/Y" in reason or "Non-finite measurement" in reason
                else:
                    raise AssertionError("Expected strict rejection of the incomplete source.")
                removed = sum(reference["incomplete_pairs"].values())
                results.append(
                    {
                        "case": name,
                        "status": "intentional_behavior_difference",
                        "input_sha256": original_sha,
                        "incomplete_pairs": reference["incomplete_pairs"],
                        "pykingenie": "raises ValueError for missing/non-finite measurements",
                        "error": reason,
                        "rstoolbox": "retains timestamps with NaN responses",
                    }
                )
                print(
                    f"EXPECTED {name}: strict rejection of {removed} incomplete pairs", flush=True
                )
                normalized = Path(temp) / "complete-observations.csv"
                assert complete_observations(file, normalized) == removed
                file = normalized
                name += " / complete observations only"
                normalization = (
                    f"Removed {removed} incomplete pairs; no interpolation or imputation"
                )
            result = compare(load(file, **kwargs), reference, key)
            result.update(
                {
                    "case": name,
                    "status": "match",
                    "input_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "original_input_sha256": original_sha,
                    "reference_input_sha256": reference_sha,
                    "normalization": normalization,
                    "reference": {k: v for k, v in reference.items() if k != "curves"},
                }
            )
            results.append(result)
            print(f"PASS {name}: {result['curves']} curves, {result['points']} points", flush=True)

        rst_cases = [("rstoolbox fixture", FIXTURES / "paired_run.csv")]
        fixture_rows = list(csv.reader((FIXTURES / "paired_run.csv").open(encoding="utf-8-sig")))
        tab_file = Path(temp) / "paired.tsv"
        with tab_file.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream, delimiter="\t").writerows(fixture_rows)
        gzip_file = Path(temp) / "paired.csv.gz"
        gzip_file.write_bytes(gzip.compress((FIXTURES / "paired_run.csv").read_bytes(), mtime=0))
        reordered_file = Path(temp) / "reordered.csv"
        with reordered_file.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(
                [[row[i] for i in [3, 2, 5, 4, 1, 0]] for row in fixture_rows]
            )
        rst_cases.extend(
            [
                ("rstoolbox TSV variant", tab_file),
                ("rstoolbox gzip variant", gzip_file),
                ("rstoolbox reordered X/Y variant", reordered_file),
            ]
        )
        if args.full_rstoolbox:
            rst_cases.append(("rstoolbox full public source", args.full_rstoolbox))
        for name, file in rst_cases:
            check(
                name,
                file,
                [
                    args.rstoolbox_python,
                    str(HERE / "rstoolbox_reference.py"),
                    str(file),
                    str(output),
                ],
                lambda label: label.rsplit("conc=", 1)[1].strip(),
            )

        rki_cases = [("Biacore fixture / original R importer", FIXTURES / "biacore_cycle_fc.tsv")]
        if args.full_biacore:
            rki_cases.append(("Biacore full source / original R importer", args.full_biacore))
        for name, file in rki_cases:
            check(
                name,
                file,
                [
                    args.rscript,
                    str(HERE / "r_reference.R"),
                    "rki",
                    str(file),
                    str(output),
                    str(args.rki_source),
                ],
                lambda label: ":".join(re.fullmatch(r"Cycle=(\d+)_Fc=(.+)", label).groups()),
                decimal=",",
            )

        if args.carterra_xlsx:
            check(
                "Carterra CSV / original workbook via readxl",
                FIXTURES / "carterra_rows.csv",
                [
                    args.rscript,
                    str(HERE / "r_reference.R"),
                    "carterra",
                    str(args.carterra_xlsx),
                    str(output),
                ],
                lambda label: label,
                reference_input=args.carterra_xlsx,
                carterra=True,
            )
    args.output.write_text(
        json.dumps(
            {
                "absolute_tolerance": 1e-10,
                "reference_source_sha256": {
                    "rstoolbox": RST_SOURCE_SHA256,
                    "rki": RKI_SOURCE_SHA256,
                },
                "results": results,
                "limitations": [
                    "Concentration labels are matched without assuming a molar unit.",
                    "rstoolbox drops run/channel/cycle metadata; full labels cannot be compared.",
                    "RKI comparison runs its unchanged raw-import section, not the full script.",
                    "Carterra comparison uses readxl, not the Mathematica SPR parser.",
                    "Injection boundaries and inferred concentrations have no external oracle.",
                    "Duke Cycle/sample export has no executed domain-specific reference parser.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
