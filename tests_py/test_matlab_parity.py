# Live Rust↔MATLAB parity tests for cssd.
#
# Each test:
#   1. Calls the Rust port (cssd Python module backed by _cssd_core).
#   2. Calls the MATLAB cssd() implementation via the host `matlab` shim
#      (`/usr/local/bin/matlab`, installed by the dev container's
#      post-create.sh — proxies to host MATLAB over SSH).
#   3. Asserts agreement on pp.coefs, pp.breaks, and discont_idx.
#
# Skipped (not failed) when:
#   - the matlab shim or HOST_MATLAB env are unavailable.
#
# Tolerance: atol=1e-8, rtol=1e-5 on pp.coefs reflects the empirical gap
# between MATLAB's LAPACK QR + Curve Fitting Toolbox csaps and the Rust
# port's Givens QR + hand-rolled banded LDL^T. Tightening below this is
# empirically not feasible without aligning the linear-algebra kernels.
# pp.breaks are exact integer multiples of x-step in our test cases so
# 1e-10 is comfortable; discont_idx is integer-valued and must match
# exactly.
#
# This test replaces the previous fixture-based test_matlab_parity.py
# (which loaded 594 pre-baked .mat files in tests_py/fixtures/). The live
# approach mirrors Pottslab's tests/test_matlab_parity.py pattern: no
# fixture management, always exercises the actual MATLAB code, surfaces
# regressions in either side immediately.

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from cssd import cssd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]


def _shim_available() -> bool:
    return shutil.which("matlab") is not None and bool(os.environ.get("HOST_MATLAB"))


pytestmark = pytest.mark.skipif(
    not _shim_available(),
    reason="matlab shim not configured (HOST_MATLAB unset or matlab not on PATH)",
)


def _run_matlab_cssd(x, y, p, gamma, pruning="FPVI"):
    """Invoke MATLAB ``cssd(x, y, p, gamma, [], [], 'pruning', PR)`` and
    return a dict with the relevant output fields parsed from CSV.

    ``y`` may be 1-D (treated as a single column) or 2-D (n, dim).
    """
    work_dir = Path(tempfile.mkdtemp(prefix="cssd-parity-", dir=WORKSPACE_ROOT))
    try:
        coefs_path = work_dir / "coefs.csv"
        breaks_path = work_dir / "breaks.csv"
        didx_path = work_dir / "discont_idx.csv"
        script_path = work_dir / "run_parity.m"

        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        x_lit = "; ".join(f"{v:.17e}" for v in x_arr)
        y_rows = []
        for i in range(y_arr.shape[0]):
            y_rows.append(", ".join(f"{v:.17e}" for v in y_arr[i, :]))
        y_lit = "; ".join(y_rows)

        gamma_lit = "Inf" if np.isinf(gamma) else f"{gamma:.17e}"

        script = (
            f"addpath(genpath('{REPO_ROOT}'));\n"
            f"x = [{x_lit}];\n"
            f"y = [{y_lit}];\n"
            f"p = {p:.17e};\n"
            f"gamma = {gamma_lit};\n"
            f"out = cssd(x, y, p, gamma, [], [], 'pruning', '{pruning}');\n"
            f"writematrix(out.pp.coefs, '{coefs_path}');\n"
            f"writematrix(out.pp.breaks(:), '{breaks_path}');\n"
            f"if isempty(out.discont_idx); didx = zeros(0,1); else; didx = out.discont_idx(:); end;\n"
            f"writematrix(didx, '{didx_path}');\n"
        )
        script_path.write_text(script)

        result = subprocess.run(
            ["matlab", "-batch", f"addpath('{work_dir}'); run_parity"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MATLAB cssd failed (rc={result.returncode}):\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        coefs = np.loadtxt(coefs_path, delimiter=",", ndmin=2)
        breaks = np.loadtxt(breaks_path, delimiter=",").ravel()

        # discont_idx may be empty (no discontinuities). loadtxt of an
        # empty CSV raises; handle by checking file size.
        if didx_path.stat().st_size == 0:
            didx = np.array([], dtype=np.int64)
        else:
            didx_raw = np.loadtxt(didx_path, delimiter=",").ravel()
            # MATLAB indices are 1-based; Rust uses 0-based.
            didx = (didx_raw.astype(np.int64) - 1)

        return {"coefs": coefs, "breaks": breaks, "discont_idx": didx}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _assert_cssd_parity(out_rust, out_matlab, *, label=""):
    """Compare Rust vs MATLAB outputs at the standard tolerance."""
    rust_breaks = np.asarray(out_rust.pp.breaks)
    rust_coefs = np.asarray(out_rust.pp.coefs)
    rust_didx = np.asarray(out_rust.discont_idx, dtype=np.int64)

    npt.assert_allclose(
        rust_breaks,
        out_matlab["breaks"],
        atol=1e-10,
        err_msg=f"{label}: pp.breaks mismatch",
    )
    npt.assert_allclose(
        rust_coefs,
        out_matlab["coefs"],
        atol=1e-8,
        rtol=1e-5,
        err_msg=f"{label}: pp.coefs mismatch",
    )
    npt.assert_array_equal(
        rust_didx,
        out_matlab["discont_idx"],
        err_msg=f"{label}: discont_idx mismatch",
    )


# ----------------------------------------------------------------------
# Step signals (one discontinuity)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("pruning", ["FPVI", "PELT"])
def test_step_p09_g1(pruning):
    n = 12
    x = np.arange(1.0, n + 1)
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    out_rust = cssd(x, y, p=0.9, gamma=1.0, pruning=pruning)
    out_matlab = _run_matlab_cssd(x, y, 0.9, 1.0, pruning=pruning)
    _assert_cssd_parity(out_rust, out_matlab, label=f"step_p09_g1_{pruning}")


@pytest.mark.parametrize("pruning", ["FPVI", "PELT"])
def test_step_p05_g05(pruning):
    n = 16
    x = np.arange(1.0, n + 1)
    y = np.concatenate([np.zeros(n // 2), 2 * np.ones(n // 2)])
    out_rust = cssd(x, y, p=0.5, gamma=0.5, pruning=pruning)
    out_matlab = _run_matlab_cssd(x, y, 0.5, 0.5, pruning=pruning)
    _assert_cssd_parity(out_rust, out_matlab, label=f"step_p05_g05_{pruning}")


# ----------------------------------------------------------------------
# Smooth signal: gamma = Inf degenerates to classical smoothing spline
# ----------------------------------------------------------------------

def test_smooth_quadratic_p099_ginf():
    n = 10
    x = np.arange(1.0, n + 1)
    t = np.linspace(0, 1, n)
    y = t * t
    out_rust = cssd(x, y, p=0.99, gamma=np.inf)
    out_matlab = _run_matlab_cssd(x, y, 0.99, np.inf, pruning="FPVI")
    _assert_cssd_parity(out_rust, out_matlab, label="smooth_quadratic_p099_ginf")


# ----------------------------------------------------------------------
# Random signals — robustness across the (p, gamma) plane
# ----------------------------------------------------------------------

@pytest.mark.parametrize("seed,n,p,gamma", [
    (0, 20, 0.9, 1.0),
    (1, 30, 0.5, 0.1),
    (2, 25, 0.99, 10.0),
])
def test_random_signal(seed, n, p, gamma):
    rng = np.random.default_rng(seed)
    x = np.arange(1.0, n + 1)
    y = rng.standard_normal(n)
    out_rust = cssd(x, y, p=p, gamma=gamma)
    out_matlab = _run_matlab_cssd(x, y, p, gamma, pruning="FPVI")
    _assert_cssd_parity(out_rust, out_matlab, label=f"random_seed{seed}_n{n}")
