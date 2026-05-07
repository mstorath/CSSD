"""Cross-validated parameter selection for CSSD.

Mirrors ``cssd_cv.m`` but uses ``scipy.optimize`` (``dual_annealing`` +
``Nelder-Mead``) instead of MATLAB's ``simulannealbnd`` + ``fminsearch``.

The smoothing-parameter pair ``(p, gamma)`` is reparametrised as ``(p, q)``
on the unit square via ``gamma = p * q / (1 - q)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from scipy.optimize import dual_annealing, minimize

from . import _cssd_core
from ._api import CssdOutput, _coerce_y, cssd

ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass
class CssdCvOutput:
    p: float
    gamma: float
    cv_score: float
    fit: CssdOutput
    cv_fun: Callable[[float, float], float]


def _kfold_split(n: int, k: int, rng: np.random.Generator) -> List[List[int]]:
    """Round-robin K-fold split mirroring ``kfoldcv_split.m``."""
    perm = rng.permutation(n)
    folds = [sorted(int(i) for i in perm[start::k]) for start in range(k)]
    return folds


def cssd_cv(
    x: Optional[ArrayLike],
    y: ArrayLike,
    cv_type: str = "random",
    cv_arg: Optional[Union[int, List[Sequence[int]]]] = None,
    delta: Optional[ArrayLike] = None,
    starting_point: Optional[Sequence[float]] = None,
    *,
    verbose: bool = False,
    max_time: Optional[float] = None,
    pruning: str = "FPVI",
    precondition: str = "none",
    random_state: Optional[int] = None,
    seed: Optional[int] = None,
) -> CssdCvOutput:
    """Select ``(p, gamma)`` for CSSD by minimising the K-fold CV score.

    Parameters
    ----------
    cv_type
        ``"random"`` (default), ``"equi"`` or ``"custom"``.
    cv_arg
        For ``"random"`` / ``"equi"`` the number of folds (default 5).
        For ``"custom"`` a list of K index vectors.
    starting_point
        Optional ``(p0, gamma0)`` warm start (default ``(0.5, 1)``).
    random_state, seed
        Either alias accepted; controls fold randomisation and SA.
    """
    y_arr = _coerce_y(y)
    n = y_arr.shape[0]
    if x is None:
        x_arr = np.arange(1, n + 1, dtype=np.float64)
    else:
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    delta_arr = (
        np.ones(n, dtype=np.float64)
        if delta is None
        else np.ascontiguousarray(np.asarray(delta, dtype=np.float64))
    )
    rs = random_state if random_state is not None else seed
    rng = np.random.default_rng(rs)

    if cv_type == "random":
        folds = _kfold_split(n, int(cv_arg if cv_arg is not None else 5), rng)
    elif cv_type == "equi":
        k = int(cv_arg if cv_arg is not None else 5)
        folds = [list(range(start, n, k)) for start in range(k)]
    elif cv_type == "custom":
        if cv_arg is None:
            raise ValueError("cv_type='custom' requires cv_arg=list of folds")
        folds = [list(int(i) for i in fold) for fold in cv_arg]
    else:
        raise ValueError(f"Unknown cv_type {cv_type!r}")

    folds_i64 = [[int(i) for i in fold] for fold in folds]

    def cv_fun(p: float, gamma: float) -> float:
        return float(
            _cssd_core.cssd_cvscore(
                x_arr,
                y_arr,
                float(p),
                float(gamma),
                delta_arr,
                folds_i64,
                pruning,
                precondition,
            )
        )

    def objective(z: np.ndarray) -> float:
        p_val = float(z[0])
        q = float(z[1])
        if not (0.0 <= p_val <= 1.0) or not (0.0 <= q < 1.0):
            return float("inf")
        gamma = p_val * q / (1.0 - q) if q < 1.0 else float("inf")
        if gamma <= 0.0:
            return float("inf")
        return cv_fun(p_val, gamma)

    if starting_point is None:
        z0 = np.array([0.5, 0.5])
    else:
        p0, gamma0 = float(starting_point[0]), float(starting_point[1])
        q0 = gamma0 / (p0 + gamma0) if (p0 + gamma0) > 0 else 0.5
        z0 = np.array([p0, q0])

    bounds = [(0.0, 1.0), (0.0, 0.999_999)]

    da_kwargs = {
        "bounds": bounds,
        "x0": z0,
        "seed": rs,
        "no_local_search": True,
    }
    if max_time is not None:
        da_kwargs["maxiter"] = max(1, int(max_time))
    if verbose:
        da_kwargs["callback"] = lambda x, f, ctx: print(f"SA z={x} f={f:.6g}")
    da_res = dual_annealing(objective, **da_kwargs)

    nm_res = minimize(
        objective,
        da_res.x,
        method="Nelder-Mead",
        options={"disp": verbose, "xatol": 1e-6, "fatol": 1e-9},
    )

    p_star = float(nm_res.x[0])
    q_star = float(nm_res.x[1])
    gamma_star = p_star * q_star / (1.0 - q_star) if q_star < 1.0 else float("inf")
    score = float(nm_res.fun)

    fit = cssd(
        x_arr, y_arr, p_star, gamma_star, delta_arr, pruning=pruning, precondition=precondition
    )

    return CssdCvOutput(
        p=p_star,
        gamma=gamma_star,
        cv_score=score,
        fit=fit,
        cv_fun=cv_fun,
    )
