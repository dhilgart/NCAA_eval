from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "ncaa_eval"
copyright = "2026, Dan Hilgart"
author = "Dan Hilgart"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

exclude_patterns = ["_build"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# myst-parser warns on Markdown TOC anchor links (e.g. [section](#section))
# and references to files outside the Sphinx source tree; these are valid
# Markdown constructs that render correctly on GitHub but have no Sphinx equivalent.
# misc.highlighting_failure suppresses "Pygments lexer name 'mermaid' is not known"
# from mermaid code blocks in testing guides (no mermaid extension installed).
suppress_warnings = ["myst.xref_missing", "misc.highlighting_failure"]

html_theme = "furo"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
