# PyKinGenie Documentation

This directory contains the documentation for the PyKinGenie package, built with Sphinx.

## Building Documentation

You can build the documentation locally using the provided `build_docs.py` script in the project root:

```bash
# From the project root
python build_docs.py
```

Or use the convenience script to build and view the documentation:

```bash
# From the project root
./view_docs.sh
```

## Documentation Structure

- `conf.py`: Sphinx configuration file
- `index.rst`: Main documentation page
- `modules.rst`: Modules listing
- `pykingenie.rst`: Module API documentation
- `_build/html/`: Generated HTML documentation (after building)

## GitHub Pages

The documentation is automatically published to GitHub Pages when changes are pushed to the main branch via the GitHub Actions workflow.

You can view the latest published documentation at:
https://osvalB.github.io/pykingenie/
