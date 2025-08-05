
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
