"""HeaviSine test signal. Port of demos/Ex_HeaviSine.m."""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from cssd import cssd


def heavi_sine(x: np.ndarray) -> np.ndarray:
    return 4.0 * np.sin(4 * np.pi * x) - np.sign(x - 0.3) - np.sign(0.72 - x)


def main(K: int = 1000, show: bool = True) -> None:
    rng = np.random.default_rng(123)
    N = 200
    sigma = 0.4
    delta = sigma * np.ones(N)
    p = 0.9999
    gammas = [10.0, 20.0, 30.0, np.inf]

    nn = 5000
    xx = np.linspace(0.0, 1.0, nn)
    yy_curves = {g: np.zeros((K, nn)) for g in gammas}
    discont_all = {g: [] for g in gammas}

    x_first = y_first = None
    for k in range(K):
        x = np.sort(rng.random(N))
        y = heavi_sine(x) + sigma * rng.standard_normal(N)
        if k == 0:
            x_first, y_first = x, y
        for g in gammas:
            out = cssd(x, y, p=p, gamma=g, delta=delta)
            yy_curves[g][k] = out.pp(xx).ravel()
            discont_all[g].extend(out.discont.tolist())

    if not show:
        return

    fig, axes = plt.subplots(3, 4, figsize=(14, 7), constrained_layout=True)
    fig.canvas.manager.set_window_title("HeaviSine")
    axes[0, 0].plot(xx, heavi_sine(xx), ".", color="#0072BD")
    axes[0, 0].set_title("(a) True signal")
    axes[0, 1].plot(x_first, y_first, "ok", markersize=3)
    axes[0, 1].set_title("(b) Sample realisation")
    for ax in axes[0, 2:]:
        ax.axis("off")
    for j, g in enumerate(gammas):
        ax = axes[1, j]
        ax.plot(xx, yy_curves[g][0], ".", color="#77AC30", markersize=1)
        ql = np.quantile(yy_curves[g], 0.025, axis=0)
        qh = np.quantile(yy_curves[g], 0.975, axis=0)
        ax.fill_between(xx, ql, qh, alpha=0.3, color="#77AC30")
        ax.set_ylim(-8, 6)
        ax.set_title(f"γ={g:g}" if np.isfinite(g) else "γ=∞ (smoothing spline)")
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--K", type=int, default=1000)
    args = parser.parse_args()
    if args.smoke:
        main(K=10, show=False)
        print("ex_heavi_sine: smoke OK")
    else:
        main(K=args.K, show=True)
