"""Piecewise polynomial wrapper around `scipy.interpolate.PPoly`.

The Rust core returns pp-form data in MATLAB convention: `coefs` of shape
`(pieces * dim, order)` with rows ordered by piece-then-dimension and columns
in *decreasing* powers. SciPy's `PPoly` expects shape `(order, pieces, dim)`
with rows in decreasing powers. We translate once at the boundary.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PPoly


class PiecewisePoly:
    """Callable piecewise polynomial. Wraps `scipy.interpolate.PPoly`.

    Construction:
        PiecewisePoly.from_matlab(breaks, coefs, dim)
    where ``breaks`` is shape ``(pieces+1,)`` and ``coefs`` is
    ``(pieces * dim, order)`` in MATLAB pp-form.
    """

    __slots__ = ("_pp", "_dim", "_order", "_breaks", "_coefs_matlab")

    def __init__(self, pp: PPoly, dim: int, order: int, breaks: np.ndarray, coefs_matlab: np.ndarray):
        self._pp = pp
        self._dim = dim
        self._order = order
        self._breaks = breaks
        self._coefs_matlab = coefs_matlab

    @classmethod
    def from_matlab(cls, breaks: np.ndarray, coefs: np.ndarray, dim: int) -> "PiecewisePoly":
        breaks = np.asarray(breaks, dtype=np.float64)
        coefs = np.asarray(coefs, dtype=np.float64)
        order = coefs.shape[1]
        pieces = breaks.size - 1
        if dim == 1:
            # SciPy PPoly: c shape (order, pieces).
            c = coefs.T  # (order, pieces)
            pp = PPoly(c, breaks, extrapolate=True)
        else:
            # SciPy PPoly: c shape (order, pieces, dim) when y is 1D extra axis.
            # MATLAB row layout: row = piece*dim + d -> reshape to (pieces, dim, order)
            # then transpose to (order, pieces, dim).
            c = coefs.reshape(pieces, dim, order).transpose(2, 0, 1)
            pp = PPoly(c, breaks, extrapolate=True)
        return cls(pp, dim, order, breaks, coefs)

    @property
    def breaks(self) -> np.ndarray:
        return self._breaks

    @property
    def coefs(self) -> np.ndarray:
        """MATLAB-form coefficients, shape ``(pieces * dim, order)``."""
        return self._coefs_matlab

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def order(self) -> int:
        return self._order

    @property
    def pieces(self) -> int:
        return self._breaks.size - 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self._pp(x)

    def derivative(self, n: int = 1) -> "PiecewisePoly":
        d_pp = self._pp.derivative(n)
        # Reconstruct MATLAB-form coefs from SciPy form.
        if self._dim == 1:
            new_coefs = d_pp.c.T
        else:
            new_coefs = d_pp.c.transpose(1, 2, 0).reshape(self.pieces * self._dim, -1)
        return PiecewisePoly(
            d_pp,
            self._dim,
            d_pp.c.shape[0],
            np.asarray(d_pp.x, dtype=np.float64),
            new_coefs,
        )

    def antiderivative(self, n: int = 1) -> "PiecewisePoly":
        a_pp = self._pp.antiderivative(n)
        if self._dim == 1:
            new_coefs = a_pp.c.T
        else:
            new_coefs = a_pp.c.transpose(1, 2, 0).reshape(self.pieces * self._dim, -1)
        return PiecewisePoly(
            a_pp,
            self._dim,
            a_pp.c.shape[0],
            np.asarray(a_pp.x, dtype=np.float64),
            new_coefs,
        )

    def __repr__(self) -> str:
        return f"PiecewisePoly(pieces={self.pieces}, dim={self._dim}, order={self._order})"
