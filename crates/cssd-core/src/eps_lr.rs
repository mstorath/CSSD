//! Fast incremental spline-energy computation via QR updates.
//!
//! Direct port of `subroutines/startEpsLR.m` and `subroutines/updateEpsLR.m`.
//!
//! The state of a spline fit on the index range `[l, r]` is encoded as a tuple
//! `(eps, R, z)` where `R` is 5×4 upper triangular (with the last row zero, kept
//! for compatibility with the update step) and `z` is 5×D, both produced from
//! the QR factorisation of the design matrix.

use ndarray::{s, Array2, ArrayView2};

/// Incremental energy state for the spline on a contiguous index range.
#[derive(Debug, Clone)]
pub struct State {
    pub eps: f64,
    /// 5×4 upper-triangular factor (with bottom row zero).
    pub r: Array2<f64>,
    /// 5×D right-hand side after Q^T application.
    pub z: Array2<f64>,
}

/// In-place Givens QR. Triangularises `a` (m×n with m >= n) by left-applying
/// plane rotations, applying the same rotations to `z` (m×d).
///
/// Order is top-down (zero column 0 first, then column 1, ...), within a column
/// we zero from the bottom upward — matching the natural variant used in
/// `cssd.m`'s p=0 piecewise-linear branch.
fn givens_qr_inplace(a: &mut Array2<f64>, z: &mut Array2<f64>) {
    let (m, n) = a.dim();
    debug_assert_eq!(z.nrows(), m);
    let zd = z.ncols();
    for col in 0..n.min(m) {
        for row in (col + 1..m).rev() {
            let upper = a[[col, col]];
            let lower = a[[row, col]];
            if lower == 0.0 {
                continue;
            }
            let r = upper.hypot(lower);
            let c = upper / r;
            let s = lower / r;
            for k in col..n {
                let u = a[[col, k]];
                let l = a[[row, k]];
                a[[col, k]] = c * u + s * l;
                a[[row, k]] = -s * u + c * l;
            }
            for k in 0..zd {
                let u = z[[col, k]];
                let l = z[[row, k]];
                z[[col, k]] = c * u + s * l;
                z[[row, k]] = -s * u + c * l;
            }
            a[[row, col]] = 0.0;
        }
    }
}

/// Build the spline-kernel rows shared by `start` and `update`.
///
/// Two rows of the design matrix, with optional column scaling:
/// ```text
/// [ 2β√3 d^(-3/2)   τ_l·β√3 d^(-1/2)  -2β√3 d^(-3/2)   τ_r·β√3 d^(-1/2) ]
/// [      0          τ_l·β   d^(-1/2)        0          -τ_r·β  d^(-1/2) ]
/// ```
/// where `tau_left` and `tau_right` are the slope-scaling factors for the
/// two coupled knots (cols 2 and 4 respectively). Pass `(1.0, 1.0)` to
/// disable preconditioning.
fn spline_rows(d: f64, beta: f64, tau_left: f64, tau_right: f64) -> [[f64; 4]; 2] {
    let sqrt3 = 3.0_f64.sqrt();
    let d_m32 = d.powf(-1.5);
    let d_m12 = d.powf(-0.5);
    [
        [
            2.0 * beta * sqrt3 * d_m32,
            tau_left * beta * sqrt3 * d_m12,
            -2.0 * beta * sqrt3 * d_m32,
            tau_right * beta * sqrt3 * d_m12,
        ],
        [0.0, tau_left * beta * d_m12, 0.0, -tau_right * beta * d_m12],
    ]
}

/// Initialises the QR state on the two-point interval [l, l+1].
///
/// Mirrors `startEpsLR.m`. `y` is the 2×D block of data values, `d` is
/// `x[l+1] - x[l]`, `alpha = [α_l, α_{l+1}]`. `tau_left, tau_right` are
/// the slope-column scalings for the first-fed and second-fed knot
/// respectively (1.0 when preconditioning is off).
pub fn start(
    y: ArrayView2<f64>,
    d: f64,
    alpha: [f64; 2],
    beta: f64,
    tau_left: f64,
    tau_right: f64,
) -> State {
    debug_assert_eq!(y.nrows(), 2);
    let dim = y.ncols();

    let kernel = spline_rows(d, beta, tau_left, tau_right);

    let mut a = Array2::<f64>::zeros((4, 4));
    a[[0, 0]] = alpha[0];
    for j in 0..4 {
        a[[1, j]] = kernel[0][j];
        a[[2, j]] = kernel[1][j];
    }
    a[[3, 2]] = alpha[1];

    let mut z = Array2::<f64>::zeros((4, dim));
    for j in 0..dim {
        z[[0, j]] = alpha[0] * y[[0, j]];
        z[[3, j]] = alpha[1] * y[[1, j]];
    }

    givens_qr_inplace(&mut a, &mut z);

    // Pad to 5×4 / 5×D to match the shape expected by `update`.
    let mut r5 = Array2::<f64>::zeros((5, 4));
    r5.slice_mut(s![..4, ..]).assign(&a);
    let mut z5 = Array2::<f64>::zeros((5, dim));
    z5.slice_mut(s![..4, ..]).assign(&z);

    State {
        eps: 0.0,
        r: r5,
        z: z5,
    }
}

/// Extends the QR state by one data point, returning the new state.
///
/// Mirrors `updateEpsLR.m`. `y_rp1` is the 1×D row of data at the new index,
/// `d_r = x[r+1] - x[r]`, `alpha` = α at the new index. `tau_left` is the
/// scaling for the previously-fed knot (whose column was kept as the first
/// 2 cols of the new R), `tau_right` is for the just-added knot.
pub fn update(
    prev: &State,
    y_rp1: ArrayView2<f64>,
    d_r: f64,
    alpha: f64,
    beta: f64,
    tau_left: f64,
    tau_right: f64,
) -> State {
    debug_assert_eq!(y_rp1.nrows(), 1);
    let dim = y_rp1.ncols();
    debug_assert_eq!(prev.z.ncols(), dim);

    let kernel = spline_rows(d_r, beta, tau_left, tau_right);

    let mut r_new = Array2::<f64>::zeros((5, 4));
    // Top-left 2×2 from previous R rows 2..4, cols 2..4 (0-indexed).
    r_new[[0, 0]] = prev.r[[2, 2]];
    r_new[[0, 1]] = prev.r[[2, 3]];
    r_new[[1, 1]] = prev.r[[3, 3]];
    // Spline-kernel rows.
    for j in 0..4 {
        r_new[[2, j]] = kernel[0][j];
        r_new[[3, j]] = kernel[1][j];
    }
    // Data row.
    r_new[[4, 2]] = alpha;

    let mut z_new = Array2::<f64>::zeros((5, dim));
    for j in 0..dim {
        z_new[[0, j]] = prev.z[[2, j]];
        z_new[[1, j]] = prev.z[[3, j]];
        z_new[[4, j]] = alpha * y_rp1[[0, j]];
    }

    givens_qr_inplace(&mut r_new, &mut z_new);

    // Energy contribution from the residual row (row 4).
    let mut delta_eps = 0.0;
    for j in 0..dim {
        delta_eps += z_new[[4, j]].powi(2);
    }

    State {
        eps: prev.eps + delta_eps,
        r: r_new,
        z: z_new,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Two-point spline interpolation has zero energy.
    #[test]
    fn two_point_zero_energy() {
        let y = array![[0.0], [1.0]];
        let st = start(y.view(), 1.0, [10.0, 10.0], 1.0, 1.0, 1.0);
        assert_abs_diff_eq!(st.eps, 0.0);
    }

    /// Energy is monotonically non-decreasing as we extend the interval.
    #[test]
    fn energy_monotone() {
        let y = array![[0.0], [1.0], [0.0], [2.0], [-1.0]];
        let alpha = [3.0; 5];
        let beta = 0.5;
        let mut st = start(
            y.slice(s![0..2, ..]),
            1.0,
            [alpha[0], alpha[1]],
            beta,
            1.0,
            1.0,
        );
        let mut prev_eps = st.eps;
        for r in 2..5 {
            st = update(
                &st,
                y.slice(s![r..r + 1, ..]),
                1.0,
                alpha[r],
                beta,
                1.0,
                1.0,
            );
            assert!(
                st.eps >= prev_eps - 1e-15,
                "eps should be non-decreasing: prev={} now={}",
                prev_eps,
                st.eps
            );
            prev_eps = st.eps;
        }
    }

    /// Energy of a perfectly linear signal is zero (a line has zero second derivative).
    #[test]
    fn linear_signal_zero_energy() {
        // y_i = 2*x_i + 1 with uniform x.
        let y = array![[1.0], [3.0], [5.0], [7.0], [9.0]];
        let alpha = [1.0; 5];
        let beta = 1.0; // give smoothness term a real weight
        let mut st = start(
            y.slice(s![0..2, ..]),
            1.0,
            [alpha[0], alpha[1]],
            beta,
            1.0,
            1.0,
        );
        for r in 2..5 {
            st = update(
                &st,
                y.slice(s![r..r + 1, ..]),
                1.0,
                alpha[r],
                beta,
                1.0,
                1.0,
            );
        }
        // For a perfect line through points with infinite data weight, both
        // data residual and smoothness residual are zero.
        assert_abs_diff_eq!(st.eps, 0.0, epsilon = 1e-10);
    }

    /// Energy is rotationally / reversal invariant: starting from the reversed
    /// pair and adding the rest in reverse should yield the same energy as
    /// natural order.
    #[test]
    fn energy_order_invariant() {
        let y = array![[0.5], [1.0], [-0.3], [2.0], [0.8]];
        let alpha = [1.0; 5];
        let beta = 0.5;

        let mut nat = start(
            y.slice(s![0..2, ..]),
            1.0,
            [alpha[0], alpha[1]],
            beta,
            1.0,
            1.0,
        );
        for r in 2..5 {
            nat = update(
                &nat,
                y.slice(s![r..r + 1, ..]),
                1.0,
                alpha[r],
                beta,
                1.0,
                1.0,
            );
        }
        let e_nat = nat.eps;

        let y_pair = array![[y[[4, 0]]], [y[[3, 0]]]];
        let mut rev = start(y_pair.view(), 1.0, [alpha[4], alpha[3]], beta, 1.0, 1.0);
        for r in (0..3).rev() {
            let y_row = array![[y[[r, 0]]]];
            rev = update(&rev, y_row.view(), 1.0, alpha[r], beta, 1.0, 1.0);
        }
        assert_abs_diff_eq!(e_nat, rev.eps, epsilon = 1e-10);
    }

    /// Different alpha (per-point weights) → different energies.
    #[test]
    fn alpha_affects_energy() {
        let y = array![[0.0], [10.0], [0.0]];
        let beta = 0.5;
        let mut st_low = start(y.slice(s![0..2, ..]), 1.0, [1.0, 1.0], beta, 1.0, 1.0);
        st_low = update(&st_low, y.slice(s![2..3, ..]), 1.0, 1.0, beta, 1.0, 1.0);

        let mut st_high = start(y.slice(s![0..2, ..]), 1.0, [10.0, 10.0], beta, 1.0, 1.0);
        st_high = update(&st_high, y.slice(s![2..3, ..]), 1.0, 10.0, beta, 1.0, 1.0);

        // Higher alpha → tighter data fit → lower data residual contribution
        // is offset by stronger smoothness penalty mismatch. The two values
        // should at least differ.
        assert!((st_low.eps - st_high.eps).abs() > 1e-6);
    }

    /// Vector-valued y: total energy equals the sum of per-component energies.
    #[test]
    fn vector_valued_decomposes() {
        let y_full = array![[1.0, 2.0], [0.5, -1.0], [2.0, 0.0], [-0.5, 3.0]];
        let alpha = [1.0; 4];
        let beta = 0.3;

        let mut st = start(
            y_full.slice(s![0..2, ..]),
            1.0,
            [alpha[0], alpha[1]],
            beta,
            1.0,
            1.0,
        );
        for r in 2..4 {
            st = update(
                &st,
                y_full.slice(s![r..r + 1, ..]),
                1.0,
                alpha[r],
                beta,
                1.0,
                1.0,
            );
        }
        let total = st.eps;

        // Same computation, one component at a time.
        let mut sum_components = 0.0;
        for c in 0..2 {
            let y = y_full.slice(s![.., c..c + 1]).to_owned();
            let mut st = start(
                y.slice(s![0..2, ..]),
                1.0,
                [alpha[0], alpha[1]],
                beta,
                1.0,
                1.0,
            );
            for r in 2..4 {
                st = update(
                    &st,
                    y.slice(s![r..r + 1, ..]),
                    1.0,
                    alpha[r],
                    beta,
                    1.0,
                    1.0,
                );
            }
            sum_components += st.eps;
        }

        assert_abs_diff_eq!(total, sum_components, epsilon = 1e-12);
    }

    /// Preconditioning leaves the energy invariant (eps is column-scale-invariant).
    #[test]
    fn tau_preserves_energy() {
        let y = array![[0.5], [1.0], [-0.3], [2.0], [0.8]];
        let alpha = [1.0; 5];
        let beta = 0.7;

        // No preconditioning.
        let mut st = start(
            y.slice(s![0..2, ..]),
            1.0,
            [alpha[0], alpha[1]],
            beta,
            1.0,
            1.0,
        );
        for r in 2..5 {
            st = update(
                &st,
                y.slice(s![r..r + 1, ..]),
                1.0,
                alpha[r],
                beta,
                1.0,
                1.0,
            );
        }
        let e_off = st.eps;

        // Wildly non-uniform tau values; the energy should be identical.
        let taus = [0.3, 7.5, 0.1, 100.0, 2.0];
        let mut st = start(
            y.slice(s![0..2, ..]),
            1.0,
            [alpha[0], alpha[1]],
            beta,
            taus[0],
            taus[1],
        );
        for r in 2..5 {
            st = update(
                &st,
                y.slice(s![r..r + 1, ..]),
                1.0,
                alpha[r],
                beta,
                taus[r - 1], // kept knot
                taus[r],     // new knot
            );
        }
        let e_on = st.eps;
        assert_abs_diff_eq!(e_off, e_on, epsilon = 1e-10);
    }
}
