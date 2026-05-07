"""Public ``cssd`` function: thin wrapper around the Rust extension."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np

from . import _cssd_core
from .ppform import PiecewisePoly

ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass
class CssdOutput:
    """Result of :func:`cssd`. Mirrors MATLAB's ``output`` struct."""

    pp: PiecewisePoly
    discont: np.ndarray
    discont_idx: np.ndarray
    interval_cell: List[np.ndarray]
    pp_cell: List[PiecewisePoly]
    x: np.ndarray
    y: np.ndarray
    complexity_counter: int
    F: np.ndarray = field(repr=False)
    partition: np.ndarray = field(repr=False)
    precondition: str = "none"
    tau: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)


def _coerce_y(y: ArrayLike) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"y must be 1-D or 2-D, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def cssd(
    x: Optional[ArrayLike],
    y: ArrayLike,
    p: float,
    gamma: float,
    delta: Optional[ArrayLike] = None,
    *,
    pruning: str = "FPVI",
    precondition: str = "none",
) -> CssdOutput:
    """Cubic smoothing spline with discontinuities.

    Parameters
    ----------
    x
        Data sites, shape ``(N,)``. May be ``None`` (defaults to ``1..N``).
    y
        Data values, shape ``(N,)`` or ``(N, D)`` for vector-valued samples.
    p
        Smoothness parameter in ``[0, 1]``. Larger values ⇒ smoother fit.
    gamma
        Discontinuity penalty in ``[0, +inf]`` (``np.inf`` reduces to a
        classical smoothing spline).
    delta
        Per-point standard deviations, shape ``(N,)``. Defaults to ones.
    pruning
        ``"FPVI"`` (default, fastest typical case) or ``"PELT"``.
    precondition
        Diagonal column-scaling of the Hermite design matrix. ``"none"``
        (default) reproduces the paper's algorithm exactly. ``"local"``
        applies a per-knot τ_i computed from the mesh + (α, β) — improves
        the conditioning of the QR system on highly non-uniform meshes,
        leaves the answer invariant in exact arithmetic, and adds O(N) to
        the setup cost.

    Returns
    -------
    CssdOutput
    """
    y_arr = _coerce_y(y)
    x_arr = None if x is None else np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    delta_arr = (
        None
        if delta is None
        else np.ascontiguousarray(np.asarray(delta, dtype=np.float64))
    )

    res = _cssd_core.cssd(
        x_arr, y_arr, float(p), float(gamma), delta_arr, pruning, precondition
    )

    pp = PiecewisePoly.from_matlab(res["breaks"], res["coefs"], int(res["dim"]))
    pp_cell = [
        PiecewisePoly.from_matlab(d["breaks"], d["coefs"], int(d["dim"]))
        for d in res["pp_cell"]
    ]
    return CssdOutput(
        pp=pp,
        discont=np.asarray(res["discont"]),
        discont_idx=np.asarray(res["discont_idx"], dtype=np.int64),
        interval_cell=[np.asarray(iv, dtype=np.int64) for iv in res["interval_cell"]],
        pp_cell=pp_cell,
        x=np.asarray(res["x"]),
        y=np.asarray(res["y"]),
        complexity_counter=int(res["complexity_counter"]),
        F=np.asarray(res["F"]),
        partition=np.asarray(res["partition"], dtype=np.int64),
        precondition=str(res["precondition"]),
        tau=np.asarray(res["tau"]),
    )
