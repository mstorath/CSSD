//! Dynamic-programming inner loops for CSSD: FPVI and PELT pruning variants.
//!
//! Direct port of the main loops in `cssd.m` for the standard case
//! `0 < p < 1, gamma > 0`. Both variants share the same Bellman recurrence and
//! produce identical `(F, partition)` outputs (per `TestCSSD.m::prunings`),
//! differing only in which `(lb, rb)` pairs are visited.

use crate::eps_lr::{self, State};
use ndarray::{s, Array2, ArrayView1, ArrayView2};

/// Output of a DP run.
#[derive(Debug, Clone)]
pub struct DpResult {
    pub f: Vec<f64>,
    /// `partition[rb]` is the 0-indexed left bound of the optimal segment
    /// ending at `rb`. `partition[rb] == 0` means no preceding discontinuity.
    pub partition: Vec<usize>,
    pub complexity_counter: usize,
}

fn y_row(y: ArrayView2<f64>, idx: usize) -> Array2<f64> {
    let dim = y.ncols();
    let mut out = Array2::<f64>::zeros((1, dim));
    for d in 0..dim {
        out[[0, d]] = y[[idx, d]];
    }
    out
}

fn y_pair_reversed(y: ArrayView2<f64>, top: usize, bottom: usize) -> Array2<f64> {
    let dim = y.ncols();
    let mut out = Array2::<f64>::zeros((2, dim));
    for d in 0..dim {
        out[[0, d]] = y[[top, d]];
        out[[1, d]] = y[[bottom, d]];
    }
    out
}

/// Precompute initial Bellman values: `F[r] = eps_{1,r}` (energy of the spline
/// fit on `[0, r]` with no discontinuities).
fn precompute_initial_f(
    y: ArrayView2<f64>,
    h: ArrayView1<f64>,
    alpha: ArrayView1<f64>,
    beta: f64,
    tau: ArrayView1<f64>,
) -> (Vec<f64>, usize) {
    let n = y.nrows();
    let mut f = vec![0.0_f64; n];
    // First-fed knot is index 0, second-fed is index 1.
    let mut state = eps_lr::start(
        y.slice(s![0..2, ..]),
        h[0],
        [alpha[0], alpha[1]],
        beta,
        tau[0],
        tau[1],
    );
    for r in 2..n {
        // Kept knot from previous step is r-1; new knot is r.
        state = eps_lr::update(
            &state,
            y_row(y, r).view(),
            h[r - 1],
            alpha[r],
            beta,
            tau[r - 1],
            tau[r],
        );
        f[r] = state.eps;
    }
    (f, n)
}

/// FPVI pruning. Iterates left bounds in reverse, breaking out of the inner
/// loop as soon as `eps_lr + gamma >= F[rb]` (then no smaller lb can improve).
pub fn run_fpvi(
    y: ArrayView2<f64>,
    h: ArrayView1<f64>,
    alpha: ArrayView1<f64>,
    beta: f64,
    gamma: f64,
    tau: ArrayView1<f64>,
) -> DpResult {
    let n = y.nrows();
    let (mut f, mut complexity_counter) = precompute_initial_f(y, h, alpha, beta, tau);
    let mut partition = vec![0usize; n];

    if gamma >= f[n - 1] {
        // Full skip: the no-discontinuity solution is already optimal.
        return DpResult {
            f,
            partition,
            complexity_counter,
        };
    }

    for rb in 2..n {
        let mut blb = 0usize;
        let mut state: Option<State> = None;
        for lb in (1..rb).rev() {
            if lb == rb - 1 {
                complexity_counter += 2;
                // Reversed feed: first-fed is rb, second-fed is rb-1.
                state = Some(eps_lr::start(
                    y_pair_reversed(y, rb, rb - 1).view(),
                    h[rb - 1],
                    [alpha[rb], alpha[rb - 1]],
                    beta,
                    tau[rb],
                    tau[rb - 1],
                ));
            } else {
                complexity_counter += 1;
                // Kept knot from previous step is lb+1, new knot is lb.
                state = Some(eps_lr::update(
                    state.as_ref().unwrap(),
                    y_row(y, lb).view(),
                    h[lb],
                    alpha[lb],
                    beta,
                    tau[lb + 1],
                    tau[lb],
                ));
            }
            let eps_lr_val = state.as_ref().unwrap().eps;

            if eps_lr_val + gamma >= f[rb] {
                break;
            }
            let candidate = f[lb - 1] + gamma + eps_lr_val;
            if candidate < f[rb] {
                f[rb] = candidate;
                blb = lb;
            }
        }
        partition[rb] = blb;
    }

    DpResult {
        f,
        partition,
        complexity_counter,
    }
}

/// PELT pruning. Maintains an active list of left bounds; after processing
/// each `rb`, prunes any `lb` whose lower bound `F[lb-1] + state[lb].eps`
/// already exceeds the best `F[rb]`.
pub fn run_pelt(
    y: ArrayView2<f64>,
    h: ArrayView1<f64>,
    alpha: ArrayView1<f64>,
    beta: f64,
    gamma: f64,
    tau: ArrayView1<f64>,
) -> DpResult {
    let n = y.nrows();
    let (mut f, mut complexity_counter) = precompute_initial_f(y, h, alpha, beta, tau);
    let mut partition = vec![0usize; n];

    if gamma >= f[n - 1] {
        return DpResult {
            f,
            partition,
            complexity_counter,
        };
    }

    // Per-lb state: initialised on demand for lb in 2..n-1 with the 2-point
    // reversed pair, then incrementally extended each time `rb` advances.
    let mut state_at: Vec<Option<State>> = (0..n).map(|_| None).collect();
    // MATLAB rb=2:N-1 (1-indexed) → lb_0 = 1..n-2 inclusive (0-indexed).
    for lb in 1..n - 1 {
        // Natural order initialisation (MATLAB uses yi(rb:rb+1,:), not reversed).
        // First-fed knot is `lb`, second-fed is `lb+1`.
        let mut y_pair = Array2::<f64>::zeros((2, y.ncols()));
        for d in 0..y.ncols() {
            y_pair[[0, d]] = y[[lb, d]];
            y_pair[[1, d]] = y[[lb + 1, d]];
        }
        state_at[lb] = Some(eps_lr::start(
            y_pair.view(),
            h[lb],
            [alpha[lb], alpha[lb + 1]],
            beta,
            tau[lb],
            tau[lb + 1],
        ));
    }

    let mut active: Vec<usize> = vec![1];

    for rb in 2..n {
        let mut blb = 0usize;
        // Iterate active list from back to front (matches MATLAB's reverse
        // listIterator).
        let mut idx = active.len();
        while idx > 0 {
            idx -= 1;
            let lb = active[idx];
            // For lb such that rb - lb > 1, advance the state by adding y[rb].
            if rb - lb > 1 {
                let prev = state_at[lb].as_ref().unwrap();
                // Kept knot from previous step is rb-1; new knot is rb.
                let new_state = eps_lr::update(
                    prev,
                    y_row(y, rb).view(),
                    h[rb - 1],
                    alpha[rb],
                    beta,
                    tau[rb - 1],
                    tau[rb],
                );
                state_at[lb] = Some(new_state);
                complexity_counter += 1;
            }
            let eps_lr_val = state_at[lb].as_ref().unwrap().eps;
            let candidate = f[lb - 1] + gamma + eps_lr_val;
            if candidate < f[rb] {
                f[rb] = candidate;
                blb = lb;
            }
        }
        partition[rb] = blb;

        // Add rb to the active list (will be considered as a potential lb
        // when rb' > rb).
        if rb < n - 1 {
            active.push(rb);
        }

        // PELT pruning: drop any lb whose lower bound already exceeds f[rb].
        active.retain(|&lb| {
            let eps = state_at[lb].as_ref().unwrap().eps;
            f[lb - 1] + eps <= f[rb]
        });
    }

    DpResult {
        f,
        partition,
        complexity_counter,
    }
}

/// Piecewise-linear (`p == 0`) variant: design matrix `[1, x] / delta`,
/// least-squares lines on each candidate segment, no pruning.
pub fn run_p0(
    x: ArrayView1<f64>,
    y: ArrayView2<f64>,
    delta: ArrayView1<f64>,
    gamma: f64,
) -> DpResult {
    let n = y.nrows();
    let dim = y.ncols();

    // Augmented matrix A0 = [B | rhs] of shape (N, 2 + D) where B columns are
    // [1/delta, x/delta] and rhs columns are y/delta.
    let mut a = Array2::<f64>::zeros((n, 2 + dim));
    for i in 0..n {
        a[[i, 0]] = 1.0 / delta[i];
        a[[i, 1]] = x[i] / delta[i];
        for d in 0..dim {
            a[[i, 2 + d]] = y[[i, d]] / delta[i];
        }
    }

    // Precompute eps_{1,r}.
    let mut a_pre = a.clone();
    apply_givens_2x2(&mut a_pre, 0, 1, 0);
    let mut f = vec![0.0_f64; n];
    let mut counter = n;
    let mut eps_1r = 0.0;
    for r in 2..n {
        apply_givens_2x2(&mut a_pre, 0, r, 0);
        apply_givens_2x2_from_col(&mut a_pre, 1, r, 1);
        let mut sq = 0.0;
        for d in 0..dim {
            sq += a_pre[[r, 2 + d]].powi(2);
        }
        eps_1r += sq;
        f[r] = eps_1r;
    }

    let mut partition = vec![0usize; n];
    if gamma >= f[n - 1] {
        return DpResult {
            f,
            partition,
            complexity_counter: counter,
        };
    }

    for rb in 2..n {
        let mut blb = 0usize;
        // Working copy of the first rb+1 rows.
        let mut aw = a.slice(s![0..rb + 1, ..]).to_owned();
        let mut eps_lr_val = 0.0;
        for lb in (1..rb).rev() {
            let last = aw.nrows() - 1;
            if lb == rb - 1 {
                apply_givens_2x2(&mut aw, last, last - 1, 0);
            } else {
                counter += 1;
                apply_givens_2x2(&mut aw, last, lb, 0);
                apply_givens_2x2_from_col(&mut aw, last - 1, lb, 1);
                let mut sq = 0.0;
                for d in 0..dim {
                    sq += aw[[lb, 2 + d]].powi(2);
                }
                eps_lr_val += sq;
            }
            let candidate = f[lb - 1] + gamma + eps_lr_val;
            if candidate < f[rb] {
                f[rb] = candidate;
                blb = lb;
            }
        }
        partition[rb] = blb;
    }

    DpResult {
        f,
        partition,
        complexity_counter: counter,
    }
}

/// Apply a Givens rotation that zeros `a[row, col]` using the pivot `a[piv, col]`.
fn apply_givens_2x2(a: &mut Array2<f64>, piv: usize, row: usize, col: usize) {
    apply_givens_2x2_from_col(a, piv, row, col)
}

fn apply_givens_2x2_from_col(a: &mut Array2<f64>, piv: usize, row: usize, start_col: usize) {
    let upper = a[[piv, start_col]];
    let lower = a[[row, start_col]];
    if lower == 0.0 {
        return;
    }
    let r = upper.hypot(lower);
    if r == 0.0 {
        return;
    }
    let c = upper / r;
    let s = lower / r;
    let n_cols = a.ncols();
    for k in start_col..n_cols {
        let u = a[[piv, k]];
        let l = a[[row, k]];
        a[[piv, k]] = c * u + s * l;
        a[[row, k]] = -s * u + c * l;
    }
    a[[row, start_col]] = 0.0;
}
