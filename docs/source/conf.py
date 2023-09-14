# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'RoboRL Navigator'
copyright = '2021, EST'
author = 'EST'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_theme_options = {
    'logo_only': False,
    'display_version': False,
    'style_nav_header_background': 'white',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 2,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
