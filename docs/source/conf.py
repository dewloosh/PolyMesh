# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# For ideas:

# https://github.com/pradyunsg/furo/blob/main/docs/conf.py
# https://github.com/sphinx-gallery/sphinx-gallery/blob/master/doc/conf.py

import sys
import os
from datetime import date
import warnings

import polymesh

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PolyMesh"
copyright = "2014-%s, Bence Balogh" % date.today().year
author = "Bence Balogh"

# The short X.Y version.
version = polymesh.__version__
# The full version, including alpha/beta/rc tags.
release = polymesh.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # allows to work with markdown files
    "myst_parser",  # pip install myst-parser for this
    # to plot summary about durations of file generations
    "sphinx.ext.duration",
    # to test code snippets in docstrings
    "sphinx.ext.doctest",
    # for automatic exploration of the source files
    "sphinx.ext.autodoc",
    # to enable cross referencing other documents on the internet
    "sphinx.ext.intersphinx",
    # Napoleon is a extension that enables Sphinx to parse both NumPy and Google style docstrings
    "sphinx.ext.napoleon",
    # 'sphinx_gallery.gen_gallery',
    # 'sphinx_gallery.load_style',  # load CSS for gallery (needs SG >= 0.6)
    "nbsphinx",  # to handle jupyter notebooks
    "nbsphinx_link",  # for including notebook files from outside the sphinx source root
    "sphinx_copybutton",  # for "copy to clipboard" buttons
    "sphinx.ext.mathjax",  # for math equations
    #"sphinxcontrib.bibtex",  # for bibliographic references
    "sphinxcontrib.rsvgconverter",  # for SVG->PDF conversion in LaTeX output
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_inline_tabs",
]

autosummary_generate = True

templates_path = ["_templates"]

exclude_patterns = ["_build"]

# The master toctree document.
master_doc = "index"

language = "EN"

# See warnings about bad links
nitpicky = True
nitpick_ignore = [
    ("", "Pygments lexer name 'ipython' is not known"),
    ("", "Pygments lexer name 'ipython3' is not known"),
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
pygments_dark_style = "github-dark"
highlight_language = "python3"

intersphinx_mapping = {
    "python": (r"https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": (r"https://numpy.org/doc/stable/", None),
    "scipy": (r"http://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": (r"https://matplotlib.org/stable", None),
    "sphinx": (r"https://www.sphinx-doc.org/en/master", None),
    "pandas": (r"https://pandas.pydata.org/pandas-docs/stable/", None),
    "awkward": (r"https://awkward-array.readthedocs.io/en/latest/", None),
    "neumann": (r"https://neumann.readthedocs.io/en/latest/", None),
}

# -- MathJax Configuration -------------------------------------------------

mathjax3_config = {
    "tex": {"tags": "ams", "useLabelIds": True},
}

# -- Image scapers configuration -------------------------------------------------

image_scrapers = ("matplotlib",)

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
    },
    "dark_css_variables": {
        # "color-brand-primary": "red",
        # "color-brand-content": "#CC3333",
        "color-brand-primary": "orange",
        "color-brand-content": "orange",
        # "color-admonition-background": "orange",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/dewloosh/PolyMesh",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/dewloosh/PolyMesh/",
    "source_branch": "main",
    "source_directory": "docs/",
}
