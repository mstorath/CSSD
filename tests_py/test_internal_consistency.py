"""Algorithm-internal cross-check (no MATLAB needed).

Mirrors the Rust ``parity.rs`` integration test from the Python side, plus
exercises the longer signals from TestCSSD.m. FPVI and PELT must produce
bitwise-equivalent pp coefficients (per ``TestCSSD.m::prunings``).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

from cssd import cssd


SHORT_SIGNALS = [
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 2, 2],
    [0, 1, 0, 1, 0, 1],
]


def _long_signals():
    rng = np.random.default_rng(123)
    funcs = [
        lambda x: jv(1, 20 * x)
        + x * ((0.3 <= x) & (x <= 0.4))
        - x * ((0.6 <= x) & (x <= 1.0)),
        lambda x: 4 * np.sin(4 * np.pi * x) - np.sign(x - 0.3) - np.sign(0.72 - x),
    ]
    out = []
    for f in funcs:
        x = np.sort(rng.random(100))
        y = f(x)
        delta = 0.1 * np.ones_like(x)
        out.append((x, y, delta))
    return out


@pytest.mark.parametrize("sig_idx,values", list(enumerate(SHORT_SIGNALS)))
@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("gamma", [1e-6, 1e-3, 1.0, 100.0])
def test_fpvi_pelt_short(sig_idx, values, p, gamma):
    x = np.arange(1, len(values) + 1, dtype=float)
    y = np.asarray(values, dtype=float)
    out_f = cssd(x, y, p=p, gamma=gamma, pruning="FPVI")
    out_p = cssd(x, y, p=p, gamma=gamma, pruning="PELT")
    np.testing.assert_allclose(out_f.pp.coefs, out_p.pp.coefs, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(out_f.partition, out_p.partition)


@pytest.mark.parametrize("p", [0.5, 0.999])
@pytest.mark.parametrize("gamma", [1e-3, 1.0, 100.0])
def test_fpvi_pelt_long(p, gamma):
    for x, y, delta in _long_signals():
        out_f = cssd(x, y, p=p, gamma=gamma, delta=delta, pruning="FPVI")
        out_p = cssd(x, y, p=p, gamma=gamma, delta=delta, pruning="PELT")
        np.testing.assert_allclose(out_f.pp.coefs, out_p.pp.coefs, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(out_f.partition, out_p.partition)


def test_gamma_inf_no_discontinuities():
    x = np.linspace(0, 1, 20)
    y = np.sin(2 * np.pi * x)
    out = cssd(x, y, p=0.99, gamma=np.inf)
    assert out.discont.size == 0


def test_pp_evaluable():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    out = cssd(x, y, p=0.9, gamma=0.01)
    yy = out.pp(x)
    # Output should match data closely on each side of the discontinuity.
    assert yy[0] == pytest.approx(0.0, abs=0.1)
    assert yy[-1] == pytest.approx(1.0, abs=0.1)


def test_vector_valued():
    x = np.linspace(0, 1, 20)
    y = np.column_stack([np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)])
    # p=1 is exact interpolation; p<1 gives smoothing whose magnitude depends
    # on the data's curvature, not just on |1-p|.
    out = cssd(x, y, p=1.0, gamma=np.inf)
    assert out.pp.dim == 2
    yy = out.pp(x)
    np.testing.assert_allclose(yy, y, atol=1e-10)
