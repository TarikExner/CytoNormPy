# CytoNormPy

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[badge-tests]: https://img.shields.io/github/actions/workflow/status/TarikExner/CytoNormPy/pytest.yml?branch=main
[link-tests]: https://github.com/TarikExner/CytoNormPy/actions/workflows/pytest.yml
[badge-docs]: https://img.shields.io/readthedocs/cytonormpy

A python port for the CytoNorm R library.

# Installation

It is recommended to choose conda as your package manager. Conda can be obtained, e.g., by installing the Miniconda distribution. For detailed instructions, please refer to the respective documentation.

With conda installed, open your terminal and create a new environment by executing the following commands::

    conda create -n cytonormpy python=3.10
    conda activate cytonormpy

## PyPI

Currently, cytonormpy is in beta-phase. There will be a PyPI release once the beta phase is finished.

    pip install cytonormpy


## Development Version

In order to get the latest version, install from [GitHub](https://github.com/TarikExner/CytoNormPy) using
    
    pip install git+https://github.com/TarikExner/CytoNormPy@main

Alternatively, clone the repository to your local hard drive via

    git clone https://github.com/TarikExner/CytoNormPy.git && cd CytoNormpy
    git checkout --track origin/main
    pip install .[docs, test]

## Jupyter notebooks

Jupyter notebooks are highly recommended due to their extensive visualization capabilities. Install jupyter via

    conda install jupyter

and run the notebook by entering `jupyter-notebook` into the terminal.

## Getting Started

Please refer to the [documentation][link-docs]. Examples are found under "vignettes" and currently include:
- [CytoNormPy from FCS files][link-docs-fcs-vignette]
- [CytoNormPy from AnnData objects][link-docs-anndata-vignette]
- [CytoNormPy plotting][link-docs-plotting-vignette]

## Contributing

Your contributions are welcome!

Please submit an issue or pull request via Github! Pull requests with updated documentation and accompanying unit tests are preferred but not obligate!


[link-docs]: https://cytonormpy.readthedocs.io
[link-docs-fcs-vignette]: https://cytonormpy.readthedocs.io/en/latest/vignettes/cnp_fcs_file.html
[link-docs-anndata-vignette]: https://cytonormpy.readthedocs.io/en/latest/vignettes/cnp_anndata.html
[link-docs-plotting-vignette]: https://cytonormpy.readthedocs.io/en/latest/vignettes/cnp_plotting.html
