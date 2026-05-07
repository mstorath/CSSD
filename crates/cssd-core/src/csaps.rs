//! Weighted cubic smoothing spline. MATLAB `csaps` analogue.
//!
//! Implements Reinsch's algorithm: minimise
//! `p Σ w_i (y_i - f(x_i))^2 + (1-p) ∫ f''(t)^2 dt`
//! over natural cubic splines `f` with knots at the data sites `x`. The system
//! reduces to a symmetric pentadiagonal equation in the interior second
//! derivatives, solved by banded LDL^T.
//!
//! The output pp-form matches MATLAB's: cubic pieces `[c0, c1, c2, c3]` per
//! interval in decreasing-power order, evaluated locally as
//! `c0·t^3 + c1·t^2 + c2·t + c3` with `t = x - breaks[i]`.

use crate::ppform::PiecewisePolynomial;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Weighted cubic smoothing spline.
///
/// `x` must be strictly increasing (handle duplicates upstream via [`crate::chk`]).
/// `y` is `(N, D)`, where `D` is the vector dimension. `w` are the per-point
/// weights (`w_i = 1 / delta_i^2` in MATLAB convention). `p ∈ [0, 1]` is the
/// smoothing parameter (1 ⇒ interpolation, 0 ⇒ straight line fit).
pub fn weighted_smoothing_spline(
    x: ArrayView1<f64>,
    y: ArrayView2<f64>,
    p: f64,
    w: ArrayView1<f64>,
) -> PiecewisePolynomial {
    let n = x.len();
    let dim = y.ncols();
    assert_eq!(y.nrows(), n, "y rows must equal x length");
    assert_eq!(w.len(), n, "w length must equal x length");
    assert!(n >= 2, "csaps requires N >= 2");

    if n == 2 {
        return linear_through_two(x, y);
    }

    if p == 0.0 {
        // Pure smoothness: optimum is the weighted least-squares line.
        return weighted_ls_line(x, y, w);
    }

    let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();

    // Solve for second derivatives at all knots (c_0 = c_{n-1} = 0 for natural spline,
    // interior c_1..c_{n-2} are the unknowns u of size M = n - 2).
    let m = n - 2;
    let mut c_full = Array2::<f64>::zeros((n, dim));
    if m == 0 {
        // No interior knots: f is linear through both endpoints (at p < 1).
        // For p = 1 this still degenerates to linear through 2 points which we
        // handled above; n == 2 was caught earlier so m >= 1 normally.
        return linear_through_two(x, y);
    }

    // Build the symmetric pentadiagonal LHS: A = p R + (1-p) T W^{-1} T^T
    // R: tridiagonal, R[k,k] = (h_k + h_{k+1}) / 3, R[k,k+1] = h_{k+1} / 6
    // T (M x N): T[k, k] = 1/h_k, T[k, k+1] = -(1/h_k + 1/h_{k+1}), T[k, k+2] = 1/h_{k+1}
    // T W^{-1} T^T entries (let s_i = 1/w_i):
    //   diag[k] = T[k, k]^2 s_k + T[k, k+1]^2 s_{k+1} + T[k, k+2]^2 s_{k+2}
    //   sup1[k] = T[k, k+1] T[k+1, k+1] s_{k+1} + T[k, k+2] T[k+1, k+2] s_{k+2}
    //   sup2[k] = T[k, k+2] T[k+2, k+2] s_{k+2}

    let mut diag = vec![0.0_f64; m];
    let mut sup1 = vec![0.0_f64; m.saturating_sub(1)];
    let mut sup2 = vec![0.0_f64; m.saturating_sub(2)];

    for k in 0..m {
        let inv_h_k = 1.0 / h[k];
        let inv_h_kp = 1.0 / h[k + 1];
        let t_kk = inv_h_k;
        let t_kkp = -(inv_h_k + inv_h_kp);
        let t_kkpp = inv_h_kp;
        let s_k = 1.0 / w[k];
        let s_kp = 1.0 / w[k + 1];
        let s_kpp = 1.0 / w[k + 2];
        let twt_diag = t_kk * t_kk * s_k + t_kkp * t_kkp * s_kp + t_kkpp * t_kkpp * s_kpp;
        let r_diag = (h[k] + h[k + 1]) / 3.0;
        diag[k] = p * r_diag + (1.0 - p) * twt_diag;

        if k + 1 < m {
            let inv_h_kpp = 1.0 / h[k + 2];
            // sup1[k]: T[k, k+1] T[k+1, k+1] s_{k+1} + T[k, k+2] T[k+1, k+2] s_{k+2}
            // T[k+1, k+1] = 1/h_{k+1}, T[k+1, k+2] = -(1/h_{k+1} + 1/h_{k+2})
            let twt_sup1 = t_kkp * inv_h_kp * s_kp + t_kkpp * (-(inv_h_kp + inv_h_kpp)) * s_kpp;
            let r_sup1 = h[k + 1] / 6.0;
            sup1[k] = p * r_sup1 + (1.0 - p) * twt_sup1;
        }
        if k + 2 < m {
            let inv_h_kpp = 1.0 / h[k + 2];
            // sup2[k]: T[k, k+2] T[k+2, k+2] s_{k+2} = (1/h_{k+1}) * (1/h_{k+2}) * s_{k+2}
            let twt_sup2 = t_kkpp * inv_h_kpp * s_kpp;
            sup2[k] = (1.0 - p) * twt_sup2;
        }
    }

    // Build RHS: B = p T y, shape (M, D).
    let mut rhs = Array2::<f64>::zeros((m, dim));
    for k in 0..m {
        let inv_h_k = 1.0 / h[k];
        let inv_h_kp = 1.0 / h[k + 1];
        for d in 0..dim {
            rhs[[k, d]] = p
                * (inv_h_k * y[[k, d]] - (inv_h_k + inv_h_kp) * y[[k + 1, d]]
                    + inv_h_kp * y[[k + 2, d]]);
        }
    }

    // Solve A u = rhs by banded LDL^T.
    let u = banded_ldlt_solve(&diag, &sup1, &sup2, &rhs);

    // Fill the full c vector with c_0 = c_{n-1} = 0, c_1..c_{n-2} = u.
    for k in 0..m {
        for d in 0..dim {
            c_full[[k + 1, d]] = u[[k, d]];
        }
    }

    // Recover f = y - (1-p)/p * W^{-1} * T^T * u.
    // T^T * u (size N): for i = 0..n-1,
    //   (T^T u)[i] = T[i-2, i] u_{i-2} + T[i-1, i] u_{i-1} + T[i, i] u_i (with bounds)
    // where T[k, k] = 1/h_k, T[k, k+1] = -(1/h_k + 1/h_{k+1}), T[k, k+2] = 1/h_{k+1}.
    // i.e. T^T[i, k] is non-zero for k ∈ {i-2, i-1, i}, with the same values
    // permuted: T^T[i, i-2] = T[i-2, i] = 1/h_{i-1};
    //           T^T[i, i-1] = T[i-1, i] = -(1/h_{i-1} + 1/h_i);
    //           T^T[i, i]   = T[i, i]   = 1/h_i.
    let mut f_vals = y.to_owned();
    if (1.0 - p).abs() > 0.0 {
        let scale = (1.0 - p) / p;
        for i in 0..n {
            for d in 0..dim {
                let mut tt_u = 0.0;
                // u is indexed by interior position k = 0..m, corresponding to knot i = k+1.
                // T[k, k] u_k contributes to T^T u at i = k.
                // T[k, k+1] u_k contributes to T^T u at i = k+1.
                // T[k, k+2] u_k contributes to T^T u at i = k+2.
                if i >= 2 && i - 2 < m {
                    let k = i - 2;
                    tt_u += (1.0 / h[k + 1]) * u[[k, d]];
                }
                if i >= 1 && i - 1 < m {
                    let k = i - 1;
                    tt_u += -(1.0 / h[k] + 1.0 / h[k + 1]) * u[[k, d]];
                }
                if i < m {
                    let k = i;
                    tt_u += (1.0 / h[k]) * u[[k, d]];
                }
                f_vals[[i, d]] -= scale * tt_u / w[i];
            }
        }
    }

    // Build pp-form: per interval i = 0..n-2 with t = x - x_i:
    //   piece_i(t) = c0 t^3 + c1 t^2 + c2 t + c3
    // where c3 = f_i, c2 = b_i, c1 = c_i / 2, c0 = (c_{i+1} - c_i) / (6 h_i)
    // and b_i = (f_{i+1} - f_i) / h_i - (h_i / 6) (2 c_i + c_{i+1}).
    let pieces = n - 1;
    let mut coefs = Array2::<f64>::zeros((pieces * dim, 4));
    for i in 0..pieces {
        let hi = h[i];
        for d in 0..dim {
            let row = i * dim + d;
            let ci = c_full[[i, d]];
            let cip = c_full[[i + 1, d]];
            let fi = f_vals[[i, d]];
            let fip = f_vals[[i + 1, d]];
            let bi = (fip - fi) / hi - (hi / 6.0) * (2.0 * ci + cip);
            coefs[[row, 0]] = (cip - ci) / (6.0 * hi);
            coefs[[row, 1]] = ci / 2.0;
            coefs[[row, 2]] = bi;
            coefs[[row, 3]] = fi;
        }
    }

    let breaks = x.to_owned();
    PiecewisePolynomial::new(breaks, coefs, dim)
}

/// Solve the symmetric pentadiagonal system `A u = rhs` via banded LDL^T.
/// `diag`, `sup1`, `sup2` are the main diagonal, first super-diagonal, and
/// second super-diagonal of the symmetric `A`.
fn banded_ldlt_solve(diag: &[f64], sup1: &[f64], sup2: &[f64], rhs: &Array2<f64>) -> Array2<f64> {
    let m = diag.len();
    let dim = rhs.ncols();
    let mut d = vec![0.0_f64; m];
    let mut l1 = vec![0.0_f64; m]; // L[i, i-1]
    let mut l2 = vec![0.0_f64; m]; // L[i, i-2]

    for i in 0..m {
        let b_i = if i >= 2 { sup2[i - 2] } else { 0.0 }; // A[i, i-2]
        let a_i = if i >= 1 { sup1[i - 1] } else { 0.0 }; // A[i, i-1]
        let dd_i = diag[i];

        if i >= 2 {
            l2[i] = b_i / d[i - 2];
        }
        if i >= 1 {
            let cross = if i >= 2 {
                l2[i] * l1[i - 1] * d[i - 2]
            } else {
                0.0
            };
            l1[i] = (a_i - cross) / d[i - 1];
        }
        let mut sub = 0.0;
        if i >= 1 {
            sub += l1[i] * l1[i] * d[i - 1];
        }
        if i >= 2 {
            sub += l2[i] * l2[i] * d[i - 2];
        }
        d[i] = dd_i - sub;
    }

    // Forward solve L y = rhs.
    let mut y = rhs.to_owned();
    for i in 0..m {
        for j in 0..dim {
            let mut v = y[[i, j]];
            if i >= 1 {
                v -= l1[i] * y[[i - 1, j]];
            }
            if i >= 2 {
                v -= l2[i] * y[[i - 2, j]];
            }
            y[[i, j]] = v;
        }
    }
    // Diagonal solve D z = y.
    for i in 0..m {
        for j in 0..dim {
            y[[i, j]] /= d[i];
        }
    }
    // Backward solve L^T u = z.
    for i in (0..m).rev() {
        for j in 0..dim {
            let mut v = y[[i, j]];
            if i + 1 < m {
                v -= l1[i + 1] * y[[i + 1, j]];
            }
            if i + 2 < m {
                v -= l2[i + 2] * y[[i + 2, j]];
            }
            y[[i, j]] = v;
        }
    }
    y
}

/// Weighted least-squares straight line through `(x, y)`. One linear piece per
/// data interval (replicating the slope/intercept across all pieces produces
/// the same callable; we use one piece spanning `[x_0, x_{n-1}]` for compactness
/// and let downstream `linext` handle further extension).
fn weighted_ls_line(
    x: ArrayView1<f64>,
    y: ArrayView2<f64>,
    w: ArrayView1<f64>,
) -> PiecewisePolynomial {
    let n = x.len();
    let dim = y.ncols();
    let mut sw = 0.0_f64;
    let mut sx = 0.0_f64;
    let mut sxx = 0.0_f64;
    for i in 0..n {
        sw += w[i];
        sx += w[i] * x[i];
        sxx += w[i] * x[i] * x[i];
    }
    let denom = sw * sxx - sx * sx;

    // MATLAB pp pieces are per data interval; we mirror that for compatibility
    // with merge / eval, so `pp.pieces() == n - 1`.
    let pieces = n - 1;
    let mut coefs = Array2::<f64>::zeros((pieces * dim, 4));
    for d in 0..dim {
        let mut sy = 0.0_f64;
        let mut sxy = 0.0_f64;
        for i in 0..n {
            sy += w[i] * y[[i, d]];
            sxy += w[i] * x[i] * y[[i, d]];
        }
        let (slope, intercept_at_zero) = if denom == 0.0 {
            // Degenerate (shouldn't happen for distinct x_i with positive weights).
            (0.0, sy / sw)
        } else {
            ((sw * sxy - sx * sy) / denom, (sxx * sy - sx * sxy) / denom)
        };
        for i in 0..pieces {
            let row = i * dim + d;
            coefs[[row, 2]] = slope;
            coefs[[row, 3]] = intercept_at_zero + slope * x[i];
        }
    }
    PiecewisePolynomial::new(x.to_owned(), coefs, dim)
}

fn linear_through_two(x: ArrayView1<f64>, y: ArrayView2<f64>) -> PiecewisePolynomial {
    let dim = y.ncols();
    let h = x[1] - x[0];
    let mut coefs = Array2::<f64>::zeros((dim, 4));
    for d in 0..dim {
        let slope = (y[[1, d]] - y[[0, d]]) / h;
        coefs[[d, 2]] = slope;
        coefs[[d, 3]] = y[[0, d]];
    }
    PiecewisePolynomial::new(Array1::from_vec(vec![x[0], x[1]]), coefs, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    /// Interpolation (p=1) reproduces the data values at the knots.
    #[test]
    fn p1_interpolates() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![[0.0], [1.0], [0.0], [-1.0], [0.0]];
        let w = Array1::from_elem(5, 1.0);
        let pp = weighted_smoothing_spline(x.view(), y.view(), 1.0, w.view());
        let yy = pp.eval(x.view());
        for i in 0..5 {
            assert_abs_diff_eq!(yy[[i, 0]], y[[i, 0]], epsilon = 1e-10);
        }
    }

    /// Smoothing of a constant signal returns the constant.
    #[test]
    fn smooth_constant() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![[3.0], [3.0], [3.0], [3.0], [3.0]];
        let w = Array1::from_elem(5, 1.0);
        let pp = weighted_smoothing_spline(x.view(), y.view(), 0.5, w.view());
        let xx = array![0.5, 1.5, 2.5, 3.5];
        let yy = pp.eval(xx.view());
        for i in 0..xx.len() {
            assert_abs_diff_eq!(yy[[i, 0]], 3.0, epsilon = 1e-10);
        }
    }

    /// Smoothing of a linear signal returns the line.
    #[test]
    fn smooth_linear() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![[1.0], [3.0], [5.0], [7.0], [9.0]];
        let w = Array1::from_elem(5, 1.0);
        let pp = weighted_smoothing_spline(x.view(), y.view(), 0.7, w.view());
        let xx = array![0.5, 1.5, 2.5, 3.5];
        let yy = pp.eval(xx.view());
        for i in 0..xx.len() {
            let expected = 1.0 + 2.0 * xx[i];
            assert_abs_diff_eq!(yy[[i, 0]], expected, epsilon = 1e-10);
        }
    }

    /// Vector-valued: components are independent.
    #[test]
    fn vector_valued() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![[0.0, 0.0], [1.0, 2.0], [0.0, 4.0], [-1.0, 6.0]];
        let w = Array1::from_elem(4, 1.0);
        let pp = weighted_smoothing_spline(x.view(), y.view(), 1.0, w.view());
        let yy = pp.eval(x.view());
        for i in 0..4 {
            assert_abs_diff_eq!(yy[[i, 0]], y[[i, 0]], epsilon = 1e-10);
            assert_abs_diff_eq!(yy[[i, 1]], y[[i, 1]], epsilon = 1e-10);
        }
    }

    /// Weights: a heavily-weighted point is pulled exactly through.
    #[test]
    fn heavy_weight_pulls_through() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![[0.0], [0.0], [10.0], [0.0], [0.0]];
        let mut w = Array1::from_elem(5, 1.0);
        w[2] = 1e10;
        let pp = weighted_smoothing_spline(x.view(), y.view(), 0.5, w.view());
        let yy = pp.eval(x.view());
        assert_abs_diff_eq!(yy[[2, 0]], 10.0, epsilon = 1e-3);
    }

    /// p=0 returns the weighted least-squares straight line.
    #[test]
    fn p0_is_weighted_ls_line() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![[0.0], [2.0], [4.0], [6.0], [8.0]]; // exactly y = 2x
        let w = Array1::from_elem(5, 1.0);
        let pp = weighted_smoothing_spline(x.view(), y.view(), 0.0, w.view());
        let yy = pp.eval(x.view());
        for i in 0..5 {
            assert_abs_diff_eq!(yy[[i, 0]], 2.0 * x[i], epsilon = 1e-10);
        }
    }

    /// p=0 with non-collinear data: returns the LS line, not interpolation.
    #[test]
    fn p0_smooths_outliers() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![[0.0], [0.0], [10.0], [0.0], [0.0]]; // outlier at i=2
        let w = Array1::from_elem(5, 1.0);
        let pp = weighted_smoothing_spline(x.view(), y.view(), 0.0, w.view());
        let yy = pp.eval(x.view());
        // The LS line through (0,0),(1,0),(2,10),(3,0),(4,0) has mean y=2,
        // and goes through (0, 2-0.5*slope*0... ) — easier just to assert it's smooth.
        for i in 0..5 {
            assert!(
                yy[[i, 0]].abs() <= 5.0,
                "p=0 should not pass through the outlier"
            );
        }
        // And the second derivative is zero on a line.
        let p0 = yy[[0, 0]];
        let p4 = yy[[4, 0]];
        let p2 = yy[[2, 0]];
        let predicted = p0 + (p4 - p0) * 0.5;
        assert_abs_diff_eq!(p2, predicted, epsilon = 1e-10);
    }

    /// Two-point case: must produce a straight line for any p.
    #[test]
    fn two_points_is_line() {
        let x = array![0.0, 2.0];
        let y = array![[0.0], [4.0]];
        let w = Array1::from_elem(2, 1.0);
        for p in [0.0, 0.3, 0.7, 1.0] {
            let pp = weighted_smoothing_spline(x.view(), y.view(), p, w.view());
            let xx = array![0.5, 1.0, 1.5];
            let yy = pp.eval(xx.view());
            for (i, &xi) in xx.iter().enumerate() {
                assert_abs_diff_eq!(yy[[i, 0]], 2.0 * xi, epsilon = 1e-10);
            }
        }
    }

    /// Non-uniform spacing: interpolation still works.
    #[test]
    fn non_uniform_x() {
        let x = array![0.0, 0.1, 0.5, 1.5, 5.0];
        let y = array![[0.0], [0.01], [0.25], [2.25], [25.0]]; // y = x^2
        let w = Array1::from_elem(5, 1.0);
        let pp = weighted_smoothing_spline(x.view(), y.view(), 1.0, w.view());
        let yy = pp.eval(x.view());
        for i in 0..5 {
            assert_abs_diff_eq!(yy[[i, 0]], y[[i, 0]], epsilon = 1e-10);
        }
    }

    /// Per-point weights modulate where the spline passes.
    #[test]
    fn weights_modulate_fit() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![[0.0], [1.0], [0.0], [1.0], [0.0]];
        let w_uniform = Array1::from_elem(5, 1.0);
        let w_heavy_mid = Array1::from_vec(vec![1.0, 1.0, 1e6, 1.0, 1.0]);
        let pp_u = weighted_smoothing_spline(x.view(), y.view(), 0.3, w_uniform.view());
        let pp_h = weighted_smoothing_spline(x.view(), y.view(), 0.3, w_heavy_mid.view());
        let mid_u = pp_u.eval(array![2.0].view())[[0, 0]];
        let mid_h = pp_h.eval(array![2.0].view())[[0, 0]];
        // High-weight middle point pulls the fit toward y[2]=0; uniform weights
        // smooth the alternation toward 0.5.
        assert!(
            mid_h.abs() < mid_u.abs(),
            "heavy mid weight should pull spline closer to y[2]=0 (got {mid_h} vs {mid_u})"
        );
    }
}
