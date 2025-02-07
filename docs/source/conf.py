import datetime

project = "Grain Analysis"
author = "Sergey Ilyev"
copyright = f"{datetime.datetime.now().year}, {author}"
release = "0.4"
rst_prolog = f"""
.. |name| replace:: {project}
"""


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # generate documentation from docstrings
    "sphinx.ext.intersphinx",  # link to other projects' documentation
    "sphinx.ext.doctest",  # test snippets in the documentation
    "sphinx.ext.autosummary",  # create summaries for modules
    "sphinx.ext.viewcode",  # add links to source code
    "sphinx_copybutton",  # add copy buttons to code blocks
    "sphinx_autodoc_typehints",  #  move typehints to descriptions
    "sphinx_design",  # add tab elements
    "sphinx_codeautolink",  # add intersphinx links in code blocks
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

copybutton_exclude = ".linenos, .gp"
copybutton_prompt_text = " "  # to remove space in (.venv) $

typehints_use_rtype = False
typehints_defaults = "comma"

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
