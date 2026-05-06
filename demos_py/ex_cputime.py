"""CPU-time scaling for FPVI vs PELT pruning. Port of demos/Ex_CPUTime.m
(without the ruptures baseline; install ``ruptures`` and re-add if desired)."""

from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from cssd import cssd


def heavi_sine(x: np.ndarray) -> np.ndarray:
    return 4.0 * np.sin(4 * np.pi * x) - np.sign(x - 0.3) - np.sign(0.72 - x)


def main(K: int = 3, show: bool = True) -> None:
    rng = np.random.default_rng(0)
    p = 0.9999
    gamma = 20.0
    sigma = 0.4
    lengths = [250, 500, 1000, 2000, 4000, 8000]
    fpvi_dense = np.zeros((len(lengths), K))
    pelt_dense = np.zeros((len(lengths), K))

    for k in range(K):
        for i, N in enumerate(lengths):
            x = np.linspace(0.0, 1.0, N)
            y = heavi_sine(x) + sigma * rng.standard_normal(N)
            delta = sigma * np.ones(N)

            t0 = time.perf_counter()
            cssd(x, y, p=p, gamma=gamma, delta=delta, pruning="FPVI")
            fpvi_dense[i, k] = time.perf_counter() - t0

            t0 = time.perf_counter()
            cssd(x, y, p=p, gamma=gamma, delta=delta, pruning="PELT")
            pelt_dense[i, k] = time.perf_counter() - t0

    fpvi_mean = fpvi_dense.mean(axis=1)
    pelt_mean = pelt_dense.mean(axis=1)
    print("Length     FPVI(s)    PELT(s)")
    for i, N in enumerate(lengths):
        print(f"{N:6d}  {fpvi_mean[i]:9.4f}  {pelt_mean[i]:9.4f}")

    if not show:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.canvas.manager.set_window_title("CSSD CPU time")
    ax.loglog(lengths, fpvi_mean, "-x", label="FPVI", linewidth=2)
    ax.loglog(lengths, pelt_mean, "-x", label="PELT", linewidth=2)
    ax.set_xlabel("Signal length")
    ax.set_ylabel("Runtime [sec]")
    ax.grid(True, which="both")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        main(K=1, show=False)
        print("ex_cputime: smoke OK")
    else:
        main()
