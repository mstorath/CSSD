//! Input validation and canonicalisation, mirroring `chkxydelta.m` +
//! MATLAB's `chckxywp` for the cssd use case.
//!
//! Behaviour:
//! - `y` is a vector of length `N` (scalar samples) or a matrix of shape
//!   `(N, D)` (vector-valued samples).
//! - `x`, `delta` (optional) must have length `N` or be empty.
//! - Non-finite samples are dropped.
//! - Duplicate sites are aggregated by weighted mean (weights `1 / delta^2`).
//! - Output is sorted ascending by `x`.

use crate::{CssdError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[derive(Debug, Clone)]
pub struct Canonical {
    pub x: Array1<f64>,
    /// Shape `(N, D)`.
    pub y: Array2<f64>,
    /// Effective weights `w = 1 / delta^2`.
    pub w: Array1<f64>,
    /// Effective deltas `1 / sqrt(w)`.
    pub delta: Array1<f64>,
}

pub fn chk_xy_delta(
    x: Option<ArrayView1<f64>>,
    y: ArrayView2<f64>,
    delta: Option<ArrayView1<f64>>,
) -> Result<Canonical> {
    let n = y.nrows();
    let d = y.ncols();
    if n < 2 {
        return Err(CssdError::NotEnoughData(n));
    }

    let x_in: Array1<f64> = match x {
        Some(v) if v.len() != n => return Err(CssdError::MismatchedXY(v.len(), n)),
        Some(v) => v.to_owned(),
        None => Array1::from_iter((1..=n).map(|i| i as f64)),
    };

    let delta_in: Array1<f64> = match delta {
        Some(v) if v.len() != n => return Err(CssdError::MismatchedDelta(n, v.len())),
        Some(v) => v.to_owned(),
        None => Array1::from_elem(n, 1.0),
    };

    // Drop non-finite y rows.
    let mask: Vec<bool> = (0..n)
        .map(|i| (0..d).all(|j| y[[i, j]].is_finite()))
        .collect();
    let n_finite: usize = mask.iter().filter(|&&b| b).count();
    if n_finite < 2 {
        return Err(CssdError::NotEnoughData(n_finite));
    }
    let mut xs = Vec::with_capacity(n_finite);
    let mut ys = Vec::with_capacity(n_finite);
    let mut ws = Vec::with_capacity(n_finite);
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        xs.push(x_in[i]);
        for j in 0..d {
            ys.push(y[[i, j]]);
        }
        ws.push(1.0 / (delta_in[i] * delta_in[i]));
    }

    // Sort by x while keeping rows of y aligned. Aggregate exact duplicates by
    // weighted average (matches the documented behaviour of `csaps`).
    let mut order: Vec<usize> = (0..n_finite).collect();
    order.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).expect("NaN in x"));

    let mut out_x = Vec::<f64>::with_capacity(n_finite);
    let mut out_y = Vec::<f64>::with_capacity(n_finite * d);
    let mut out_w = Vec::<f64>::with_capacity(n_finite);

    let mut i = 0;
    while i < n_finite {
        let xi = xs[order[i]];
        let mut wsum = ws[order[i]];
        let mut ysum: Vec<f64> = (0..d)
            .map(|j| ws[order[i]] * ys[order[i] * d + j])
            .collect();
        let mut k = i + 1;
        while k < n_finite && xs[order[k]] == xi {
            wsum += ws[order[k]];
            for j in 0..d {
                ysum[j] += ws[order[k]] * ys[order[k] * d + j];
            }
            k += 1;
        }
        out_x.push(xi);
        for j in 0..d {
            out_y.push(ysum[j] / wsum);
        }
        out_w.push(wsum);
        i = k;
    }

    let n_out = out_x.len();
    if n_out < 2 {
        return Err(CssdError::NotEnoughData(n_out));
    }

    let x = Array1::from_vec(out_x);
    let y = Array2::from_shape_vec((n_out, d), out_y).expect("shape");
    let w = Array1::from_vec(out_w);
    let delta = w.mapv(|wi| (1.0 / wi).sqrt());

    Ok(Canonical { x, y, w, delta })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn passthrough_no_changes() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![[1.0], [4.0], [9.0]];
        let c = chk_xy_delta(Some(x.view()), y.view(), None).unwrap();
        assert_eq!(c.x.len(), 3);
        assert_eq!(c.y.shape(), &[3, 1]);
    }

    #[test]
    fn sorts_by_x() {
        let x = array![3.0, 1.0, 2.0];
        let y = array![[9.0], [1.0], [4.0]];
        let c = chk_xy_delta(Some(x.view()), y.view(), None).unwrap();
        assert_eq!(c.x.to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(c.y.column(0).to_vec(), vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn aggregates_duplicates() {
        let x = array![1.0, 1.0, 2.0];
        let y = array![[0.0], [2.0], [3.0]];
        let delta = array![1.0, 1.0, 1.0];
        let c = chk_xy_delta(Some(x.view()), y.view(), Some(delta.view())).unwrap();
        assert_eq!(c.x.len(), 2);
        assert_abs_diff_eq!(c.y[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c.w[0], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn drops_nan() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![[1.0], [f64::NAN], [9.0]];
        let c = chk_xy_delta(Some(x.view()), y.view(), None).unwrap();
        assert_eq!(c.x.len(), 2);
    }

    #[test]
    fn drops_inf() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![[1.0], [f64::INFINITY], [3.0], [f64::NEG_INFINITY]];
        let c = chk_xy_delta(Some(x.view()), y.view(), None).unwrap();
        assert_eq!(c.x.len(), 2);
        assert_eq!(c.x.to_vec(), vec![1.0, 3.0]);
    }

    #[test]
    fn errors_on_too_few_points() {
        let x = array![1.0];
        let y = array![[1.0]];
        let err = chk_xy_delta(Some(x.view()), y.view(), None).unwrap_err();
        matches!(err, CssdError::NotEnoughData(_));
    }

    #[test]
    fn errors_on_mostly_nan() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![[1.0], [f64::NAN], [f64::NAN]];
        let err = chk_xy_delta(Some(x.view()), y.view(), None).unwrap_err();
        matches!(err, CssdError::NotEnoughData(_));
    }

    #[test]
    fn errors_on_mismatched_x() {
        let x = array![1.0, 2.0];
        let y = array![[1.0], [4.0], [9.0]];
        let err = chk_xy_delta(Some(x.view()), y.view(), None).unwrap_err();
        matches!(err, CssdError::MismatchedXY(_, _));
    }

    #[test]
    fn errors_on_mismatched_delta() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![[1.0], [4.0], [9.0]];
        let delta = array![1.0, 1.0];
        let err = chk_xy_delta(Some(x.view()), y.view(), Some(delta.view())).unwrap_err();
        matches!(err, CssdError::MismatchedDelta(_, _));
    }

    #[test]
    fn default_x_is_one_indexed_range() {
        let y = array![[1.0], [4.0], [9.0]];
        let c = chk_xy_delta(None, y.view(), None).unwrap();
        assert_eq!(c.x.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn aggregates_three_duplicates() {
        let x = array![1.0, 1.0, 1.0, 2.0];
        let y = array![[0.0], [3.0], [6.0], [10.0]];
        let delta = array![1.0, 1.0, 1.0, 1.0];
        let c = chk_xy_delta(Some(x.view()), y.view(), Some(delta.view())).unwrap();
        assert_eq!(c.x.len(), 2);
        assert_abs_diff_eq!(c.y[[0, 0]], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c.w[0], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn duplicates_are_weighted() {
        // The first duplicate has delta=0.1 (weight 100), value 0; the second
        // has delta=1 (weight 1), value 10. Weighted mean = (100*0+1*10)/101.
        let x = array![1.0, 1.0, 2.0];
        let y = array![[0.0], [10.0], [3.0]];
        let delta = array![0.1, 1.0, 1.0];
        let c = chk_xy_delta(Some(x.view()), y.view(), Some(delta.view())).unwrap();
        assert_abs_diff_eq!(c.y[[0, 0]], 10.0 / 101.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c.w[0], 101.0, epsilon = 1e-12);
    }

    #[test]
    fn vector_valued_passthrough() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let c = chk_xy_delta(Some(x.view()), y.view(), None).unwrap();
        assert_eq!(c.y.shape(), &[3, 2]);
    }

    #[test]
    fn vector_valued_drops_partial_nan() {
        // Drop a row if ANY component is non-finite.
        let x = array![1.0, 2.0, 3.0];
        let y = array![[1.0, 2.0], [3.0, f64::NAN], [5.0, 6.0]];
        let c = chk_xy_delta(Some(x.view()), y.view(), None).unwrap();
        assert_eq!(c.x.len(), 2);
    }
}
