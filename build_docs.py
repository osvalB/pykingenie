#!/usr/bin/env python3
"""
Script to build documentation locally using Sphinx.
"""

import os
import sys
import subprocess
from pathlib import Path


def build_docs():
    """Build the documentation."""
    docs_dir = Path("docs")

    os.chdir(docs_dir)

    # Remove the _build directory if it exists
    if "_build" in os.listdir():
        print("Removing existing _build directory...")
        subprocess.run(["rm", "-rf", "_build"])

    print("Generating API documentation...")
    subprocess.run([
        sys.executable, "-m", "sphinx.ext.apidoc",
        "-o", ".", "../src/pykingenie", "--force", "--module-first"
    ])

    print("Building HTML documentation...")
    subprocess.run([
        sys.executable, "-m", "sphinx", "-b", "html", ".", "_build/html"
    ])

    print(f"✅ Documentation built successfully!")

    return None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        pass
    else:
        build_docs()