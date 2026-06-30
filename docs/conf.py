
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

project = 'PyKinGenie'
copyright = '2025, Osvaldo Burastero'
author = 'Osvaldo Burastero'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'numpydoc',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_design',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
    'exclude-members': '__weakref__',
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = True

numpydoc_show_class_members = False
nbsphinx_execute = 'never'
nbsphinx_allow_errors = False
nbsphinx_codecell_lexer = 'python'


def setup(app):
    app.add_css_file('custom.css')
