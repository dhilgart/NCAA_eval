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

suppress_warnings = ["myst.xref_missing"]

html_theme = "furo"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
