"""CSSD — Cubic smoothing splines for discontinuous signals.

A high-performance Rust core (via PyO3) wrapped in a NumPy-friendly Python API.
Mirrors the MATLAB reference implementation by Storath & Weinmann (2023).
"""

from __future__ import annotations

from ._api import cssd, CssdOutput
from .cv import cssd_cv, CssdCvOutput
from .ppform import PiecewisePoly

__all__ = ["cssd", "cssd_cv", "CssdOutput", "CssdCvOutput", "PiecewisePoly"]
__version__ = "0.1.0"
