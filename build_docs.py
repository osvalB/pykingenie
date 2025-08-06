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
    cwd = os.getcwd()
    docs_dir = Path("docs")
    
    # Create docs directory if it doesn't exist
    docs_dir.mkdir(exist_ok=True)
    
    # Change to the docs directory
    os.chdir(docs_dir)

    # Remove the _build directory if it exists
    build_dir = Path("_build")
    if build_dir.exists():
        print(f"Removing existing {build_dir} directory...")
        shutil.rmtree(build_dir)
    
    # Create the _build/html directories
    html_dir = build_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    print("Generating API documentation...")
    try:
        subprocess.run([
            sys.executable, "-m", "sphinx.ext.apidoc",
            "-o", ".", "../src/pykingenie", "--force", "--module-first"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating API documentation: {e}")
        # Return to original directory before exiting
        os.chdir(cwd)
        sys.exit(1)

    print("Building HTML documentation...")
    try:
        subprocess.run([
            sys.executable, "-m", "sphinx", "-b", "html", ".", "_build/html"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error building HTML documentation: {e}")
        # Return to original directory before exiting
        os.chdir(cwd)
        sys.exit(1)

    # Print debug information
    html_path = Path("_build/html").absolute()
    print(f"✅ Documentation built successfully!")
    print(f"📖 Documentation directory: {html_path}")
    
    # List files in the build directory to verify content
    if html_path.exists():
        print(f"Files in the build directory: {list(html_path.iterdir())}")
    else:
        print(f"Warning: Build directory {html_path} does not exist!")
    
    # Return to the original directory
    os.chdir(cwd)
    
    return html_path


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        pass
    else:
        build_docs()