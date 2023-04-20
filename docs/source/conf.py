# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gwfast'
copyright = '2022, Francesco Iacovelli and Michele Mancarella'
author = 'Francesco Iacovelli and Michele Mancarella'
release = '1.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

language = 'en'

extensions = ['sphinx.ext.duration',
              'sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'nbsphinx',
              'myst_parser',
              'sphinx.ext.intersphinx',
              'sphinx_copybutton',
              'sphinx_search.extension',
             ]

templates_path = ['_templates']
exclude_patterns = ['_build', '.DS_Store']
suppress_warnings = ["myst.header"]

intersphinx_mapping = {
    'numdifftools': ('https://numdifftools.readthedocs.io/en/latest/', None),
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
    'jax' : ('https://jax.readthedocs.io/en/latest/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    'analytics_anonymize_ip': False,
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'linear-gradient(290deg, rgba(200,200,200,1) 45%, rgba(62,141,123,1) 90%, rgba(47,127,123,1) 100%)',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': False,
    'titles_only': False,
    # File-wide metadata
    'github_url' : 'https://github.com/CosmoStatGW/gwfast',
    'source_repository': 'https://github.com/CosmoStatGW/gwfast',
    'source_branch' : 'master',
    'source_directory' : 'docs/source/',
}

html_logo = 'gwfast_logo.png'

#def setup(app):
#    app.add_css_file('custom.css')
    
html_css_files = ['custom.css']

# -- Options for LaTex output -------------------------------------------------
latex_elements = {
    'preamble': r'''
\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\usepackage{amsmath}
''',

}

# -- Import and definitions -------------------------------------------------

import pathlib
import sys
import os

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

sys.path.insert(0, os.path.abspath('../gwfast/'))

# See https://stackoverflow.com/questions/7250659/how-to-use-python-to-programmatically-generate-part-of-sphinx-documentation/18143318#18143318

from os.path import basename

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from docutils.parsers.rst import Directive
from docutils import nodes, statemachine

class ExecDirective(Directive):
    """Execute the specified python code and insert the output into the document"""
    has_content = True

    def run(self):
        oldStdout, sys.stdout = sys.stdout, StringIO()

        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)

        try:
            exec('\n'.join(self.content))
            text = sys.stdout.getvalue()
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            return [nodes.error(None, nodes.paragraph(text = "Unable to execute python code at %s:%d:" % (basename(source), self.lineno)), nodes.paragraph(text = str(sys.exc_info()[1])))]
        finally:
            sys.stdout = oldStdout

def setup(app):
    app.add_directive('exec', ExecDirective)
