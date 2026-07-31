"""Sphinx configuration for the ttt-strat API reference."""

from __future__ import annotations

import sys
from pathlib import Path

# Safety net so ``ttt_strat`` is importable even without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

project = "ttt-strat"
copyright = "2026, Duarte Martins"
author = "Duarte Martins"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Render dataclass "Attributes" sections as :ivar: field lists rather than
# separate ``.. attribute::`` directives, which would otherwise collide with
# the same attributes discovered by autodoc's own ``:members:`` (Sphinx
# reports these as "duplicate object description" warnings).
napoleon_use_ivar = True
# "Reference" (citation) is a non-standard numpydoc section used in the
# w_prime models. Without this, napoleon leaves it as a raw RST section,
# and repeating that title across bartram.py/skiba.py/caen.py produces
# "duplicate label reference" warnings project-wide.
napoleon_custom_sections = ["Reference"]

autodoc_typehints = "description"
autodoc_member_order = "bysource"

html_theme = "furo"
html_static_path = []

# Note: `@numba.njit`-decorated functions in ``physics.py`` are wrapped in a
# ``CPUDispatcher`` object. Empirically (numba 0.6x), the dispatcher already
# forwards ``__doc__`` and ``inspect.signature()`` to the original Python
# function, so autodoc renders them correctly with no extra configuration.
