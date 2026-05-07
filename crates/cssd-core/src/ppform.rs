//! Piecewise polynomial in pp-form, mirroring MATLAB's structure.
//!
//! Layout matches MATLAB's pp-form so that interop and parity are easy:
//! - `breaks`: length `pieces+1`, strictly increasing
//! - `coefs`: shape `(pieces * dim, order)`, where row `(piece*dim + d)` holds
//!   the coefficients in *decreasing* powers (MATLAB convention) for piece
//!   `piece`, dimension `d`. Evaluation at `x` in `[breaks[i], breaks[i+1])`:
//!   `sum_{k=0..order} coefs[i*dim + d, k] * (x - breaks[i])^(order-1-k)`
//! - `dim`: vector dimension D
//! - `order`: number of coefficients per piece (4 for cubic)

use ndarray::{s, Array1, Array2, ArrayView1};

/// Piecewise polynomial in MATLAB pp-form.
#[derive(Debug, Clone)]
pub struct PiecewisePolynomial {
    pub breaks: Array1<f64>,
    pub coefs: Array2<f64>,
    pub dim: usize,
    pub order: usize,
}

impl PiecewisePolynomial {
    pub fn pieces(&self) -> usize {
        self.breaks.len().saturating_sub(1)
    }

    /// Construct from breaks and coefs (MATLAB `ppmak` analogue).
    pub fn new(breaks: Array1<f64>, coefs: Array2<f64>, dim: usize) -> Self {
        let pieces = breaks.len() - 1;
        let order = coefs.ncols();
        debug_assert_eq!(coefs.nrows(), pieces * dim);
        Self {
            breaks,
            coefs,
            dim,
            order,
        }
    }

    /// Evaluate at the given points. Returns an `(N, dim)` array.
    pub fn eval(&self, xx: ArrayView1<f64>) -> Array2<f64> {
        let n = xx.len();
        let mut out = Array2::<f64>::zeros((n, self.dim));
        let pieces = self.pieces();
        if pieces == 0 {
            return out;
        }
        for (i, &x) in xx.iter().enumerate() {
            let piece = locate_piece(&self.breaks, x);
            let t = x - self.breaks[piece];
            for d in 0..self.dim {
                let row = piece * self.dim + d;
                // Horner in MATLAB convention (decreasing powers).
                let mut v = self.coefs[[row, 0]];
                for k in 1..self.order {
                    v = v * t + self.coefs[[row, k]];
                }
                out[[i, d]] = v;
            }
        }
        out
    }

    /// Evaluate a *scalar-output* spline at one point. Returns an `(dim,)` row.
    pub fn eval_at(&self, x: f64) -> Array1<f64> {
        let pieces = self.pieces();
        let piece = if pieces == 0 {
            0
        } else {
            locate_piece(&self.breaks, x)
        };
        let t = x - self.breaks.get(piece).copied().unwrap_or(0.0);
        let mut out = Array1::<f64>::zeros(self.dim);
        for d in 0..self.dim {
            let row = piece * self.dim + d;
            let mut v = self.coefs[[row, 0]];
            for k in 1..self.order {
                v = v * t + self.coefs[[row, k]];
            }
            out[d] = v;
        }
        out
    }

    /// In-place pad coefficients to cubic order (4) by prepending zero columns.
    /// Mirrors `embed_pptocubic.m`.
    pub fn embed_to_cubic(&mut self) {
        if self.order >= 4 {
            return;
        }
        let pad = 4 - self.order;
        let m = self.coefs.nrows();
        let mut new_coefs = Array2::<f64>::zeros((m, 4));
        new_coefs.slice_mut(s![.., pad..]).assign(&self.coefs);
        self.coefs = new_coefs;
        self.order = 4;
    }

    /// Linear extension to a wider domain. Mirrors `linext_pp.m`.
    /// Adds two extra pieces: `[l, breaks[0]]` and `[breaks[end], r]`, both
    /// linear segments tangent to the spline at the corresponding boundary.
    pub fn linext(&mut self, l: f64, r: f64) {
        assert!(l <= self.breaks[0] && *self.breaks.last().unwrap() <= r);
        self.embed_to_cubic();

        let first = self.breaks[0];
        let last = *self.breaks.last().unwrap();

        // First derivative coefficients, evaluated at the endpoints.
        // For cubic [a, b, c, d] (decreasing powers), derivative is [3a, 2b, c]
        // in decreasing powers; evaluated at t=0 it equals c.
        let base_first = self.eval_at(first);
        let slope_first = self.deriv_eval_at(first);
        let base_last = self.eval_at(last);
        let slope_last = self.deriv_eval_at(last);

        let dim = self.dim;
        let pieces = self.pieces();
        let mut new_breaks = Array1::<f64>::zeros(pieces + 3);
        new_breaks[0] = l;
        for (i, &b) in self.breaks.iter().enumerate() {
            new_breaks[i + 1] = b;
        }
        new_breaks[pieces + 2] = r;

        // base_l: spline value extended linearly from first to l.
        let mut base_l = Array1::<f64>::zeros(dim);
        for d in 0..dim {
            base_l[d] = base_first[d] + slope_first[d] * (l - first);
        }

        let mut new_coefs = Array2::<f64>::zeros(((pieces + 2) * dim, 4));
        // Left linear piece: coefs in decreasing powers [0, 0, slope, base_l]
        for d in 0..dim {
            new_coefs[[d, 2]] = slope_first[d];
            new_coefs[[d, 3]] = base_l[d];
        }
        // Original pieces.
        for i in 0..pieces {
            for d in 0..dim {
                let src_row = i * dim + d;
                let dst_row = (i + 1) * dim + d;
                for k in 0..4 {
                    new_coefs[[dst_row, k]] = self.coefs[[src_row, k]];
                }
            }
        }
        // Right linear piece: starts at `last`, value base_last, slope slope_last.
        for d in 0..dim {
            let row = (pieces + 1) * dim + d;
            new_coefs[[row, 2]] = slope_last[d];
            new_coefs[[row, 3]] = base_last[d];
        }

        self.breaks = new_breaks;
        self.coefs = new_coefs;
    }

    /// Evaluate first derivative at a single point. Used by `linext`.
    fn deriv_eval_at(&self, x: f64) -> Array1<f64> {
        let pieces = self.pieces();
        let piece = if pieces == 0 {
            0
        } else {
            locate_piece(&self.breaks, x)
        };
        let t = x - self.breaks.get(piece).copied().unwrap_or(0.0);
        let mut out = Array1::<f64>::zeros(self.dim);
        // Derivative of order-`o` polynomial (decreasing powers) is
        // coeffs * [(o-1), (o-2), ..., 1, 0] applied positionally.
        let o = self.order;
        for d in 0..self.dim {
            let row = piece * self.dim + d;
            // Build derivative coefs (length o-1).
            // Original: c[0]*t^(o-1) + c[1]*t^(o-2) + ... + c[o-1]
            // Derivative: (o-1)*c[0]*t^(o-2) + (o-2)*c[1]*t^(o-3) + ... + c[o-2]
            let mut v = (o - 1) as f64 * self.coefs[[row, 0]];
            for k in 1..o - 1 {
                v = v * t + (o - 1 - k) as f64 * self.coefs[[row, k]];
            }
            out[d] = v;
        }
        out
    }

    /// Concatenate piecewise polynomials with matching endpoints.
    /// Mirrors `merge_ppcell.m`.
    pub fn merge(parts: Vec<PiecewisePolynomial>) -> PiecewisePolynomial {
        assert!(!parts.is_empty());
        let dim = parts[0].dim;
        let order = parts.iter().map(|p| p.order).max().unwrap();
        // Embed all to common order if needed (cssd reconstruction always uses order=4).
        let parts: Vec<PiecewisePolynomial> = parts
            .into_iter()
            .map(|mut p| {
                if p.order < order {
                    let pad = order - p.order;
                    let m = p.coefs.nrows();
                    let mut new_coefs = Array2::<f64>::zeros((m, order));
                    new_coefs.slice_mut(s![.., pad..]).assign(&p.coefs);
                    p.coefs = new_coefs;
                    p.order = order;
                }
                p
            })
            .collect();

        // Concatenate breaks (drop overlapping endpoints) and coefs.
        let total_pieces: usize = parts.iter().map(|p| p.pieces()).sum();
        let mut breaks = Vec::<f64>::with_capacity(total_pieces + 1);
        let mut coefs = Array2::<f64>::zeros((total_pieces * dim, order));
        breaks.extend(parts[0].breaks.iter().copied());
        let mut row = 0;
        for d in 0..parts[0].coefs.nrows() {
            for k in 0..order {
                coefs[[row, k]] = parts[0].coefs[[d, k]];
            }
            row += 1;
        }
        for p in parts.iter().skip(1) {
            // Drop the last current break (it equals the next's first break).
            // Append new breaks excluding the first.
            for (i, &b) in p.breaks.iter().enumerate() {
                if i == 0 {
                    continue;
                }
                breaks.push(b);
            }
            for d in 0..p.coefs.nrows() {
                for k in 0..order {
                    coefs[[row, k]] = p.coefs[[d, k]];
                }
                row += 1;
            }
        }

        PiecewisePolynomial {
            breaks: Array1::from_vec(breaks),
            coefs,
            dim,
            order,
        }
    }
}

/// Locate the piece index for `x`: the largest `i` with `breaks[i] <= x`,
/// clamped to `[0, pieces-1]` so out-of-range queries extrapolate the boundary
/// piece (matching MATLAB's `ppval` behaviour for cubics).
fn locate_piece(breaks: &Array1<f64>, x: f64) -> usize {
    let pieces = breaks.len() - 1;
    if x <= breaks[0] {
        return 0;
    }
    if x >= breaks[pieces] {
        return pieces - 1;
    }
    // Binary search for the upper bound.
    let mut lo = 0usize;
    let mut hi = pieces;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if breaks[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Inner energy of a cubic spline: `∫ (pp''(x))^2 dx` with linear extension.
/// Mirrors `spline_innerenergy.m`. For a cubic the second derivative is
/// piecewise linear, so the integral on each interval is `h * (l0^2 + l0*lh + lh^2) / 3`.
pub fn spline_inner_energy(pp: &PiecewisePolynomial) -> Array1<f64> {
    // Linearly extend so endpoints contribute zero. We don't actually need to
    // mutate; the integral over the linear extensions is zero (second derivative
    // of a linear segment is zero), so we can directly integrate over the
    // original pieces.
    assert_eq!(pp.order, 4, "spline_inner_energy expects cubic pp");
    let pieces = pp.pieces();
    let dim = pp.dim;
    let mut energy = Array1::<f64>::zeros(dim);
    for i in 0..pieces {
        let h = pp.breaks[i + 1] - pp.breaks[i];
        for d in 0..dim {
            let row = i * dim + d;
            // Cubic in decreasing powers: c0*t^3 + c1*t^2 + c2*t + c3
            // Second derivative: 6*c0*t + 2*c1
            let l0 = 2.0 * pp.coefs[[row, 1]];
            let lh = 6.0 * pp.coefs[[row, 0]] * h + 2.0 * pp.coefs[[row, 1]];
            energy[d] += h * (l0 * l0 + l0 * lh + lh * lh) / 3.0;
        }
    }
    energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    fn linear_pp(breaks: Vec<f64>, slope: f64, intercept: f64) -> PiecewisePolynomial {
        let n = breaks.len() - 1;
        let mut coefs = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            // y = slope * (x - breaks[i]) + (slope * breaks[i] + intercept)
            coefs[[i, 2]] = slope;
            coefs[[i, 3]] = slope * breaks[i] + intercept;
        }
        PiecewisePolynomial::new(Array1::from_vec(breaks), coefs, 1)
    }

    #[test]
    fn eval_linear() {
        let pp = linear_pp(vec![0.0, 1.0, 2.0], 2.0, 1.0);
        let xx = array![0.0, 0.5, 1.0, 1.5, 2.0];
        let yy = pp.eval(xx.view());
        for (i, &x) in xx.iter().enumerate() {
            assert_abs_diff_eq!(yy[[i, 0]], 2.0 * x + 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn embed_to_cubic_pads_left() {
        let mut pp = PiecewisePolynomial::new(
            Array1::from_vec(vec![0.0, 1.0]),
            Array2::from_shape_vec((1, 2), vec![3.0, 5.0]).unwrap(),
            1,
        );
        pp.embed_to_cubic();
        assert_eq!(pp.order, 4);
        // [3, 5] -> [0, 0, 3, 5]
        assert_eq!(pp.coefs[[0, 0]], 0.0);
        assert_eq!(pp.coefs[[0, 1]], 0.0);
        assert_eq!(pp.coefs[[0, 2]], 3.0);
        assert_eq!(pp.coefs[[0, 3]], 5.0);
    }

    #[test]
    fn linext_extends_linearly() {
        let mut pp = linear_pp(vec![1.0, 2.0], 3.0, 0.0); // y = 3x
        pp.linext(0.0, 3.0);
        assert_eq!(pp.pieces(), 3);
        // At x = 0.5, value should be 1.5 (from linear extension on the left).
        let v = pp.eval(array![0.5].view());
        assert_abs_diff_eq!(v[[0, 0]], 1.5, epsilon = 1e-12);
        // At x = 2.5, value should be 7.5.
        let v = pp.eval(array![2.5].view());
        assert_abs_diff_eq!(v[[0, 0]], 7.5, epsilon = 1e-12);
    }

    #[test]
    fn merge_two_pieces() {
        let p1 = linear_pp(vec![0.0, 1.0], 1.0, 0.0); // y = x on [0,1]
        let p2 = linear_pp(vec![1.0, 2.0], -1.0, 2.0); // y = 2 - x on [1,2]
        let m = PiecewisePolynomial::merge(vec![p1, p2]);
        assert_eq!(m.pieces(), 2);
        let v = m.eval(array![0.5, 1.5].view());
        assert_abs_diff_eq!(v[[0, 0]], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(v[[1, 0]], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn inner_energy_of_linear_is_zero() {
        let pp = linear_pp(vec![0.0, 1.0, 2.0], 3.0, 1.0);
        let e = spline_inner_energy(&pp);
        assert_abs_diff_eq!(e[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn eval_clamps_below_first_break() {
        // With our extrapolation policy, x < breaks[0] uses the first piece's
        // polynomial extrapolated from t=x-breaks[0] (negative t).
        let pp = linear_pp(vec![0.0, 1.0], 2.0, 1.0);
        let yy = pp.eval(array![-0.5].view());
        // At piece 0 with t=-0.5: 2*(-0.5)+1 = 0. Confirms standard
        // extrapolation rather than clamp-to-edge.
        assert_abs_diff_eq!(yy[[0, 0]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn eval_clamps_above_last_break() {
        let pp = linear_pp(vec![0.0, 1.0, 2.0], 2.0, 1.0);
        let yy = pp.eval(array![3.0].view());
        // Last piece [1,2] with t=3-1=2: 2*2+(2*1+1) = 4+3 = 7.
        assert_abs_diff_eq!(yy[[0, 0]], 7.0, epsilon = 1e-12);
    }

    #[test]
    fn eval_at_break_picks_right_piece() {
        // Two pieces with a discontinuity at x=1.
        let mut coefs = Array2::<f64>::zeros((2, 4));
        coefs[[0, 3]] = 5.0; // piece 0 constant 5 on [0,1)
        coefs[[1, 3]] = 9.0; // piece 1 constant 9 on [1,2)
        let pp = PiecewisePolynomial::new(Array1::from_vec(vec![0.0, 1.0, 2.0]), coefs, 1);
        // At x=1.0 exactly, we pick the piece with breaks[i] <= 1.0, that's i=1.
        let yy = pp.eval(array![1.0].view());
        assert_abs_diff_eq!(yy[[0, 0]], 9.0, epsilon = 1e-12);
    }

    #[test]
    fn embed_idempotent() {
        let mut pp = linear_pp(vec![0.0, 1.0], 2.0, 0.0);
        let coefs_before = pp.coefs.clone();
        pp.embed_to_cubic();
        pp.embed_to_cubic(); // double embed should be a no-op
        assert_eq!(pp.coefs, coefs_before);
    }

    #[test]
    fn linext_idempotent_when_already_extended() {
        let mut pp = linear_pp(vec![0.0, 1.0], 1.0, 0.0);
        pp.linext(-1.0, 2.0);
        let pieces_after_first = pp.pieces();
        pp.linext(-1.0, 2.0);
        // Second call adds another two pieces (it doesn't check whether
        // current bounds already match) — make this expectation explicit.
        assert_eq!(pp.pieces(), pieces_after_first + 2);
    }

    #[test]
    fn merge_preserves_dim() {
        let mut p1 = linear_pp(vec![0.0, 1.0], 1.0, 0.0);
        let mut p2 = linear_pp(vec![1.0, 2.0], 1.0, 0.0);
        // Promote both to dim=1, leave alone.
        p1.embed_to_cubic();
        p2.embed_to_cubic();
        let m = PiecewisePolynomial::merge(vec![p1, p2]);
        assert_eq!(m.dim, 1);
        assert_eq!(m.order, 4);
    }

    #[test]
    fn merge_single_piece() {
        let pp = linear_pp(vec![0.0, 1.0, 2.0], 2.0, 1.0);
        let m = PiecewisePolynomial::merge(vec![pp.clone()]);
        assert_eq!(m.pieces(), pp.pieces());
        assert_eq!(m.coefs, pp.coefs);
    }

    #[test]
    fn linext_vector_valued() {
        let mut coefs = Array2::<f64>::zeros((2, 4));
        // Piece 0, dim 0: y = 2x + 1
        coefs[[0, 2]] = 2.0;
        coefs[[0, 3]] = 1.0;
        // Piece 0, dim 1: y = -x + 5
        coefs[[1, 2]] = -1.0;
        coefs[[1, 3]] = 5.0;
        let pp = PiecewisePolynomial::new(Array1::from_vec(vec![0.0, 1.0]), coefs, 2);

        let mut pp_ext = pp.clone();
        pp_ext.linext(-2.0, 3.0);
        // At x=-1 (within left ext): values should be linear extension.
        let v = pp_ext.eval(array![-1.0].view());
        let xq = -1.0_f64;
        assert_abs_diff_eq!(v[[0, 0]], 2.0 * xq + 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[[0, 1]], -xq + 5.0, epsilon = 1e-12);
        // At x=2 (within right ext): same linear formulas extended.
        let v = pp_ext.eval(array![2.0].view());
        let xq = 2.0_f64;
        assert_abs_diff_eq!(v[[0, 0]], 2.0 * xq + 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[[0, 1]], -xq + 5.0, epsilon = 1e-12);
    }

    #[test]
    fn inner_energy_of_constant_is_zero() {
        // pp_const: f(x) = 7 on [0, 1].
        let mut coefs = Array2::<f64>::zeros((1, 4));
        coefs[[0, 3]] = 7.0;
        let pp = PiecewisePolynomial::new(Array1::from_vec(vec![0.0, 1.0]), coefs, 1);
        let e = spline_inner_energy(&pp);
        assert_abs_diff_eq!(e[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn inner_energy_of_quadratic_known() {
        // f(x) = x^2 on [0, 1]. f''(x) = 2.
        // ∫_0^1 (2)^2 dx = 4.
        let mut coefs = Array2::<f64>::zeros((1, 4));
        coefs[[0, 0]] = 0.0; // c0 (t^3)
        coefs[[0, 1]] = 1.0; // c1 (t^2)
        coefs[[0, 2]] = 0.0; // c2 (t)
        coefs[[0, 3]] = 0.0; // c3
        let pp = PiecewisePolynomial::new(Array1::from_vec(vec![0.0, 1.0]), coefs, 1);
        let e = spline_inner_energy(&pp);
        assert_abs_diff_eq!(e[0], 4.0, epsilon = 1e-12);
    }
}
