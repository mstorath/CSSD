"""Synthetic test signal: Bessel + 3 jumps. Port of demos/Ex_Synthetic.m.

Run with ``python -m demos_py.ex_synthetic`` or ``python demos_py/ex_synthetic.py``.
Use ``--smoke`` for a fast (K=20) self-check that exercises the pipeline
without showing plots.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

from cssd import cssd


def true_signal(x: np.ndarray) -> np.ndarray:
    return (
        jv(1, 20 * x)
        + x * ((0.3 <= x) & (x <= 0.4))
        - x * ((0.6 <= x) & (x <= 1.0))
    )


def main(K: int = 1000, show: bool = True) -> None:
    rng = np.random.default_rng(123)
    N = 100
    sigma = 0.1
    delta = sigma * np.ones(N)

    x_all, y_all = [], []
    for _ in range(K):
        x = np.sort(rng.random(N))
        y = true_signal(x) + sigma * rng.standard_normal(N)
        x_all.append(x)
        y_all.append(y)

    p = 0.999
    gammas = [4.0, 8.0, 12.0, np.inf]
    nn = 5000
    xx = np.linspace(0.0, 1.0, nn)
    yy_curves = {g: np.zeros((K, nn)) for g in gammas}
    discont_all = {g: [] for g in gammas}
    for k in range(K):
        for g in gammas:
            out = cssd(x_all[k], y_all[k], p=p, gamma=g, delta=delta)
            yy_curves[g][k] = out.pp(xx).ravel()
            discont_all[g].extend(out.discont.tolist())

    if not show:
        return
    fig, axes = plt.subplots(3, 4, figsize=(14, 7), constrained_layout=True)
    fig.canvas.manager.set_window_title("Synthetic signal")

    axes[0, 0].plot(xx, true_signal(xx), ".", color="#0072BD")
    axes[0, 0].set_title("(a) True signal")
    axes[0, 1].plot(x_all[0], y_all[0], "ok", markersize=3)
    axes[0, 1].set_title("(b) Sample realisation")
    for ax in axes[0, 2:]:
        ax.axis("off")

    for j, g in enumerate(gammas):
        ax = axes[1, j]
        ax.plot(xx, yy_curves[g][0], ".", color="#77AC30", markersize=1)
        ql = np.quantile(yy_curves[g], 0.025, axis=0)
        qh = np.quantile(yy_curves[g], 0.975, axis=0)
        ax.fill_between(xx, ql, qh, alpha=0.3, color="#77AC30")
        title = (
            "Smoothing spline (γ=∞)"
            if not np.isfinite(g)
            else f"CSSD γ={g:g}"
        )
        ax.set_title(title)
        ax.set_ylim(-1.3, 1.1)

        ax_h = axes[2, j]
        if discont_all[g]:
            ax_h.hist(discont_all[g], bins=np.linspace(-0.005, 1.005, 102), color="#77AC30")
            ax_h.set_xlim(0, 1)
            ax_h.set_ylim(0, K)
        else:
            ax_h.axis("off")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run a fast K=20 smoke test without plotting")
    parser.add_argument("--K", type=int, default=1000)
    args = parser.parse_args()
    if args.smoke:
        main(K=20, show=False)
        print("ex_synthetic: smoke OK")
    else:
        main(K=args.K, show=True)
