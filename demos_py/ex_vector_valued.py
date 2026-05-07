"""Vector-valued (2-component) signal demo. Port of demos/Ex_VectorValued.m."""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

from cssd import cssd


def g1(x: np.ndarray) -> np.ndarray:
    return 4 * (
        jv(1, 20 * x)
        + x * ((0.3 <= x) & (x <= 0.4))
        - x * ((0.6 <= x) & (x <= 1.0))
    )


def g2(x: np.ndarray) -> np.ndarray:
    return 4.0 * np.sin(4 * np.pi * x) - np.sign(x - 0.3) - np.sign(0.72 - x)


def main(K: int = 200, show: bool = True) -> None:
    rng = np.random.default_rng(123)
    N = 200
    sigma = 0.6
    delta = sigma * np.ones(N)
    p = 0.9999
    gammas = [13.0, 15.0, 17.0, np.inf]
    nn = 3000
    xx = np.linspace(0.0, 1.0, nn)
    yy_curves = {g: np.zeros((K, nn, 2)) for g in gammas}

    for k in range(K):
        x = np.sort(rng.random(N))
        y = np.column_stack([
            g1(x) + sigma * rng.standard_normal(N),
            g2(x) + sigma * rng.standard_normal(N),
        ])
        for g in gammas:
            out = cssd(x, y, p=p, gamma=g, delta=delta)
            yy_curves[g][k] = out.pp(xx)

    if not show:
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    fig.canvas.manager.set_window_title("Vector-valued")
    for j, g in enumerate(gammas):
        ax = axes[j]
        ax.plot(xx, yy_curves[g][0, :, 0], ".", color="#77AC30", markersize=1)
        ax.plot(xx, yy_curves[g][0, :, 1], ".", color="#4DBEEE", markersize=1)
        ax.set_ylim(-8, 6)
        ax.set_title(f"γ={g:g}" if np.isfinite(g) else "γ=∞")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        main(K=5, show=False)
        print("ex_vector_valued: smoke OK")
    else:
        main(K=200, show=True)
