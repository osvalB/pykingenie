#!/usr/bin/env python3
"""
Script to build documentation locally using Sphinx.
"""

import sys
import subprocess
import shutil
from pathlib import Path


def build_docs():
    """Build the documentation."""
    repo_root = Path(__file__).resolve().parent
    docs_dir = repo_root / "docs"
    build_dir = docs_dir / "_build" / "html"

    shutil.rmtree(build_dir, ignore_errors=True)

    print("Generating API documentation...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx.ext.apidoc",
            "-o",
            str(docs_dir),
            str(repo_root / "src" / "pykingenie"),
            "--force",
            "--module-first",
        ],
        check=True,
    )

    print("Building HTML documentation...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-E",
            "-a",
            "-b",
            "html",
            str(docs_dir),
            str(build_dir),
        ],
        check=True,
    )

    html_path = build_dir / "index.html"
    print("Documentation built successfully!")
    print(f"Open: {html_path.absolute()}")

    return html_path


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        pass
    else:
        build_docs()
