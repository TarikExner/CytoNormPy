# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

import matplotlib
matplotlib.use("agg")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CytoNormPy'
copyright = '2024, Tarik Exner, Nicolaj Hackert'
author = 'Tarik Exner, Nicolaj Hackert'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath('../../CytoNormPy/'))

extensions = [
    "sphinxcontrib.bibtex",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "nbsphinx",  # for notebook implementation
    "nbsphinx_link",  # necessary to keep vignettes outside of sphinx root directory
    "matplotlib.sphinxext.plot_directive"  # necessary to include inline plots via documentation
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

bibtex_bibfiles = ['references.bib']

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"

# autodoc-typehint
simplify_optional_unions = False

# plot_directives
plot_include_source = True
plot_formats = [("png", 80)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_rcparams = {"savefig.bbox": "tight"}
plot_apply_rcparams = True

# issues_github_path = ""
# autodoc_default_flags = ['members']
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "CytoNormPy"
