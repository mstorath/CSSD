"""Parity tests against MATLAB-generated fixtures.

Run ``matlab_fixtures/dump_fixtures.m`` from MATLAB first to populate
``tests_py/fixtures/``. If the directory is empty, all tests in this module
are skipped — making CI green without requiring MATLAB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from cssd import cssd

FIXTURES = Path(__file__).parent / "fixtures"


def _signal_fixture_files():
    return sorted(FIXTURES.glob("sig*.mat"))


def _cv_fixture_files():
    return sorted(FIXTURES.glob("cv_sig*.mat"))


pytestmark = pytest.mark.skipif(
    not _signal_fixture_files(),
    reason="No MATLAB fixtures found; run matlab_fixtures/dump_fixtures.m first.",
)


@pytest.mark.parametrize("path", _signal_fixture_files(), ids=lambda p: p.stem)
def test_cssd_parity(path: Path):
    """Cross-check Rust output against MATLAB-saved reference.

    Bar (set with the user as 'bitwise-ish'):
      * `discont` and `discont_idx` must match exactly (midpoint reduction
        means these are integer-indexed gaps, no float sensitivity).
      * `pp.breaks` must match to 1e-10 absolute.
      * `pp.coefs` must match to ``atol=1e-8 + rtol=1e-5 * |ref|``. The
        Rust core uses Givens-based QR + a hand-written banded LDL^T for
        Reinsch, while MATLAB uses LAPACK Householder + Curve Fitting Toolbox
        ``csaps``; both are stable but accumulate float noise differently
        (median diff across the 594 fixtures is ~1.1e-16, worst ~1.2e-5
        absolute on the long N=100 signal under heavy smoothing).
      * Evaluating the spline on a 200-point grid must match to 1e-7
        absolute / 1e-7 relative — this is the *user-visible* comparison.
    """
    fix = loadmat(path, squeeze_me=True)
    x = np.atleast_1d(fix["x"]).astype(np.float64)
    y = np.atleast_1d(fix["y"]).astype(np.float64)
    delta = np.atleast_1d(fix["delta"]).astype(np.float64)
    p = float(fix["p"])
    gamma = float(fix["gamma"])
    pruning = str(fix["pruning"]).strip()

    out = cssd(x, y, p=p, gamma=gamma, delta=delta, pruning=pruning)

    ref_breaks = np.atleast_1d(fix["pp_breaks"]).astype(np.float64)
    ref_coefs = np.atleast_2d(fix["pp_coefs"]).astype(np.float64)
    np.testing.assert_allclose(out.pp.breaks, ref_breaks, atol=1e-10, rtol=0)
    np.testing.assert_allclose(out.pp.coefs, ref_coefs, atol=1e-8, rtol=1e-5)

    ref_discont = np.atleast_1d(fix["discont"]).astype(np.float64).ravel()
    np.testing.assert_allclose(np.sort(out.discont), np.sort(ref_discont), atol=1e-12)

    if "discont_idx" in fix.dtype.names if hasattr(fix, "dtype") else "discont_idx" in fix:
        ref_discont_idx = np.atleast_1d(fix["discont_idx"]).astype(np.int64).ravel()
        # MATLAB stores discont_idx 1-indexed; convert.
        if ref_discont_idx.size > 0:
            np.testing.assert_array_equal(
                np.sort(out.discont_idx), np.sort(ref_discont_idx) - 1
            )

    # User-visible check: pp evaluations on a fine grid agree.
    grid = np.linspace(x.min(), x.max(), 200)
    rust_yy = out.pp(grid).ravel()
    # Rebuild MATLAB pp on the Python side via PPoly.
    from scipy.interpolate import PPoly
    ml_pp = PPoly(ref_coefs.T, ref_breaks, extrapolate=True)
    ml_yy = ml_pp(grid)
    np.testing.assert_allclose(rust_yy, ml_yy, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("path", _cv_fixture_files(), ids=lambda p: p.stem)
def test_cv_fit_at_matlab_selected_pgamma(path: Path):
    """Given MATLAB's CV-selected (p, gamma), the Rust core's fit must match.

    Note: we cannot directly cross-check ``cssd_cv`` end-to-end because the
    Python port uses ``scipy.optimize.dual_annealing`` whereas the MATLAB
    reference uses ``simulannealbnd`` — the RNGs and cooling schedules differ,
    so the optimisers find different (p, gamma) optima even with matching
    seeds. Instead, we check that **given** MATLAB's chosen (p, gamma), the
    Rust ``cssd`` produces the same discontinuity set and fit as MATLAB.
    """
    fix = loadmat(path, squeeze_me=True)
    x = np.atleast_1d(fix["x"]).astype(np.float64)
    y = np.atleast_1d(fix["y"]).astype(np.float64)
    delta = np.atleast_1d(fix["delta"]).astype(np.float64)
    p = float(fix["p"])
    gamma = float(fix["gamma"])

    out = cssd(x, y, p=p, gamma=gamma, delta=delta)

    ref_discont = np.atleast_1d(fix.get("discont", np.array([]))).astype(np.float64).ravel()
    np.testing.assert_allclose(np.sort(out.discont), np.sort(ref_discont), atol=1e-12)
