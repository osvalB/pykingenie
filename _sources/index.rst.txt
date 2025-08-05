PyKinGenie Documentation
========================

Welcome to PyKinGenie, a Python package for analyzing kinetics data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   pykingenie
   pykingenie.utils

Installation
============

Install PyPhotoMol using pip:

.. code-block:: bash

   pip install pyphotomol

Or for development:

.. code-block:: bash

   git clone https://github.com/osvalB/pyPhotoMol.git
   cd pyPhotoMol
   pip install -e ".[dev]"

Quick Start
===========

.. code-block:: python

   from pyphotomol.main import PyPhotoMol

   # Create instance and import data
   model = PyPhotoMol()
   model.import_file('data.h5')

   # Analyze data
   model.count_binding_events()
   model.create_histogram(use_masses=True, window=[0, 1000], bin_width=20)
   model.guess_peaks(min_height=5, min_distance=3)

   # View operation history
   model.print_logbook_summary()

Features
========

* **Data Import**: Import HDF5 and CSV mass photometry data files
* **Histogram Analysis**: Create and analyze histograms of mass and contrast distributions  
* **Peak Detection**: Automated peak finding in histograms
* **Mass-Contrast Conversion**: Convert between mass and contrast with calibration parameters
* **Gaussian Fitting**: Fit single and multiple truncated Gaussian functions to histogram data
* **Comprehensive Logging**: Track all operations with detailed logbooks

API Reference
=============

.. toctree::
   :maxdepth: 4

   pykingenie
   pykingenie.utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
