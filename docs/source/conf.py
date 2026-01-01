"""Configuration file for the ByzPy documentation build."""

from __future__ import annotations

import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PY_SRC = os.path.join(ROOT, "python")
if PY_SRC not in sys.path:
    sys.path.insert(0, PY_SRC)

from byzpy import __version__ as _byzpy_version

project = "ByzPy"
author = "ByzPy contributors"
copyright = f"{datetime.now():%Y}, {author}"
release = _byzpy_version
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "furo"
html_static_path = ["_static"]
pygments_dark_style = "native"
html_css_files = ["custom.css"]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

myst_enable_extensions = [
    "colon_fence",
]
