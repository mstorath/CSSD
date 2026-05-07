"""Old Faithful geyser data with CV-selected (p, gamma). Port of demos/Ex_Geyser_CV.m."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cssd import cssd, cssd_cv


def load_faithful(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read demos/faithful.txt into (eruptions, waiting) pair."""
    rows = []
    with path.open() as f:
        for line in f.readlines()[15:]:  # skip the 15-line header
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def main(show: bool = True) -> None:
    here = Path(__file__).parent
    faithful_path = here.parent / "demos" / "faithful.txt"
    if not faithful_path.exists():
        raise FileNotFoundError(faithful_path)
    eruptions, waiting = load_faithful(faithful_path)

    perm = np.argsort(eruptions)
    x = eruptions[perm]
    y = waiting[perm]

    cv = cssd_cv(
        x, y,
        cv_type="random",
        cv_arg=5,
        starting_point=(0.59, 526.7),
        random_state=123,
    )
    fit_cv = cv.fit
    fit_jump = cssd(x, y, p=cv.p, gamma=145.0)
    fit_lin = cssd(x, y, p=0.0, gamma=np.inf)

    if not show:
        print(
            f"CV: p={cv.p:.5f} gamma={cv.gamma:.2f} cv_score={cv.cv_score:.2f}"
        )
        return

    xx = np.linspace(x.min(), x.max(), 1000)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.canvas.manager.set_window_title("Geyser")
    ax.plot(x, y, "ok", markersize=4, label="Data")
    ax.plot(xx, fit_cv.pp(xx).ravel(), "-",
            label=f"CSSD CV (γ={cv.gamma:.1f}, p={cv.p:.4f})", linewidth=2)
    ax.plot(xx, fit_jump.pp(xx).ravel(), "--",
            label=f"CSSD γ=145, p={cv.p:.4f}", linewidth=2)
    ax.plot(xx, fit_lin.pp(xx).ravel(), ":", label="Linear (p=0, γ=∞)", linewidth=2)
    ax.set_xlabel("Duration of eruption (min)")
    ax.set_ylabel("Time to next eruption (min)")
    for d in fit_jump.discont:
        ax.axvline(d, color="#999999", linestyle="--", linewidth=1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(show=not args.smoke)
    if args.smoke:
        print("ex_geyser_cv: smoke OK")
