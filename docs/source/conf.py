# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CIMAP Algorithm'
copyright = '2022, Gregorio Dotti'
author = 'Gregorio Dotti'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'nbsphinx'
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True
inheritance_graph_attrs = dict(rankdir="TB", size='""')
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'
autodoc_member_order = 'bysource'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)
add_module_names = False
 
def setup(app):
    app.add_css_file('my_theme.css')
    
 
html_static_path = ['_static']

html_favicon = '_static/politologo.png'

# Add logo for project
html_logo = '_static/Logo.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}