#!/usr/bin/env python3
"""
Script to build documentation locally using Sphinx.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def build_docs():
    """Build the documentation."""
    # Get the current working directory and docs directory
    docs_dir = Path("docs")
    
    # Change to the docs directory
    os.chdir(docs_dir)
    print("Generating API documentation...")
    subprocess.run([
        sys.executable, "-m", "sphinx.ext.apidoc",
        "-o", ".", "../src/pykingenie", "--force", "--module-first"
    ])

    print("Building HTML documentation...")
    subprocess.run([
        sys.executable, "-m", "sphinx", "-b", "html", ".", "_build/html"
    ])

    html_path = docs_dir / "_build" / "html" / "index.html"
    print(f"✅ Documentation built successfully!")
    print(f"📖 Open: {html_path.absolute()}")

    return html_path


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        pass
    else:
        build_docs()