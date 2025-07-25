#!/usr/bin/env python3
"""
Script to build documentation locally using Sphinx.
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_docs():
    """Set up Sphinx documentation structure."""
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    # Change to docs directory
    os.chdir(docs_dir)

    # Initialize Sphinx if conf.py doesn't exist
    if not (docs_dir / "conf.py").exists():
        print("Initializing Sphinx documentation...")
        subprocess.run([
            sys.executable, "-m", "sphinx.cmd.quickstart",
            "-q", "-p", "PyKinGenie", "-a", "osvalB", "-v", "0.1.0",
            "--ext-autodoc", "--ext-viewcode", "--makefile", "--no-batchfile", "."
        ])

    # Create enhanced conf.py
    conf_content = '''
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'PyKinGenie'
copyright = '2025, osvalB'
author = 'osvalB'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx_copybutton',
    'sphinx_design',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "github_url": "https://github.com/osvalB/pykingenie",
    "use_edit_page_button": False,  # Disable for local builds
    "show_toc_level": 2,
    "navigation_with_keys": False,
}

html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

numpydoc_show_class_members = False
'''

    with open("conf.py", "w") as f:
        f.write(conf_content)

    # Create comprehensive index.rst
    index_content = '''
PyKinGenie Documentation
========================

Welcome to PyKinGenie, a Python package for analyzing mass photometry data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   modules

Installation
============

Install PyKinGenie using pip:

.. code-block:: bash

   pip install pykingenie

Or for development:

.. code-block:: bash

   git clone https://github.com/osvalB/pykingenie
   cd pykingenie
   pip install -e ".[dev]"


API Reference
=============

.. toctree::
   :maxdepth: 4

   pykingenie

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
'''

    with open("index.rst", "w") as f:
        f.write(index_content)

    print("✅ Sphinx documentation setup complete!")
    return docs_dir


def build_docs():
    """Build the documentation."""
    docs_dir = Path("docs")

    if not docs_dir.exists():
        docs_dir = setup_docs()

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
        setup_docs()
    else:
        build_docs()