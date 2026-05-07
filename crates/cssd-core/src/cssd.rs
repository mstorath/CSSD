//! Public CSSD entry point: input check, DP, reconstruction.

use crate::chk::{chk_xy_delta, Canonical};
use crate::csaps::weighted_smoothing_spline;
use crate::dp::{run_fpvi, run_p0, run_pelt, DpResult};
use crate::ppform::PiecewisePolynomial;
use crate::precond::{local_tau, unit_tau, Preconditioning};
use crate::{CssdError, Result};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pruning {
    Fpvi,
    Pelt,
}

#[derive(Debug, Clone)]
pub struct CssdOutput {
    pub pp: PiecewisePolynomial,
    pub discont: Array1<f64>,
    pub interval_cell: Vec<Vec<usize>>,
    pub pp_cell: Vec<PiecewisePolynomial>,
    pub discont_idx: Array1<usize>,
    pub x: Array1<f64>,
    pub y: Array2<f64>,
    pub complexity_counter: usize,
    /// Bellman values F[r] returned by the DP (before reconstruction).
    /// Useful for parity testing.
    pub f: Vec<f64>,
    /// Best left-bound for each rb. Useful for parity testing.
    pub partition: Vec<usize>,
    /// Which preconditioning mode was used.
    pub precondition: Preconditioning,
    /// The slope-scaling vector that was applied (only meaningful when
    /// `precondition == Local`; in `None` mode this is all-ones).
    pub tau: Array1<f64>,
}

/// Compute a cubic smoothing spline with discontinuities.
///
/// Mirrors `cssd.m`: solves
/// `min_{f, J}  p Σ ((y_i - f(x_i))/δ_i)^2  +  (1-p) ∫_{[x_1, x_N] \ J} f''(t)^2 dt  +  γ |J|`.
///
/// `gamma == f64::INFINITY` (or `p == 1`) reduces to a classical smoothing
/// spline; `p == 0` reduces to penalised piecewise-linear regression.
pub fn cssd(
    x: Option<ArrayView1<f64>>,
    y: ArrayView2<f64>,
    p: f64,
    gamma: f64,
    delta: Option<ArrayView1<f64>>,
    pruning: Pruning,
    precondition: Preconditioning,
) -> Result<CssdOutput> {
    if !(0.0..=1.0).contains(&p) {
        return Err(CssdError::InvalidP(p));
    }
    if gamma < 0.0 {
        return Err(CssdError::InvalidGamma(gamma));
    }

    let canon = chk_xy_delta(x, y, delta)?;
    let Canonical { x, y, w, delta } = canon;
    let n = x.len();
    let dim = y.ncols();

    if gamma.is_infinite() || p == 1.0 {
        // Classical smoothing spline / interpolation. After the audit fix
        // (PORTING_NOTES.md §N8), `output.pp` is linext-extended to match
        // the convention of the DP branch — `pp.breaks` always extends one
        // unit beyond `[x_1, x_N]`, and `pp_cell[0] == output.pp`.
        let mut pp = weighted_smoothing_spline(x.view(), y.view(), p, w.view());
        pp.linext(x[0] - 1.0, *x.last().unwrap() + 1.0);
        pp.embed_to_cubic();
        let pp_cell = vec![pp.clone()];
        let interval_cell = vec![(0..n).collect::<Vec<_>>()];
        return Ok(CssdOutput {
            pp,
            discont: Array1::from_vec(vec![]),
            interval_cell,
            pp_cell,
            discont_idx: Array1::from_vec(vec![]),
            x,
            y,
            complexity_counter: n,
            f: vec![0.0; n],
            partition: vec![0; n],
            precondition,
            tau: unit_tau(n),
        });
    }

    // Compute the slope-scaling vector once. For `Local`, uses the mesh +
    // (alpha, beta) to balance per-knot column norms; for `None`, all-ones.
    let alpha: Array1<f64> = delta.iter().map(|d| p.sqrt() / d).collect();
    let beta = (1.0 - p).sqrt();
    let h: Array1<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
    let tau = match precondition {
        Preconditioning::None => unit_tau(n),
        Preconditioning::Local => local_tau(h.view(), alpha.view(), beta),
    };

    // Run DP.
    let dp = if p == 0.0 {
        // The p=0 piecewise-linear branch uses a different design matrix
        // (B = [1, x] / delta) where preconditioning isn't relevant.
        run_p0(x.view(), y.view(), delta.view(), gamma)
    } else {
        match pruning {
            Pruning::Fpvi => run_fpvi(y.view(), h.view(), alpha.view(), beta, gamma, tau.view()),
            Pruning::Pelt => run_pelt(y.view(), h.view(), alpha.view(), beta, gamma, tau.view()),
        }
    };

    // Reconstruction (right-to-left).
    let DpResult {
        f,
        partition,
        complexity_counter,
    } = dp;

    let mut pp_cell_rev: Vec<PiecewisePolynomial> = Vec::new();
    let mut interval_cell_rev: Vec<Vec<usize>> = Vec::new();
    let mut discont_locations_rev: Vec<f64> = Vec::new();
    let mut upper_discont = *x.last().unwrap() + 1.0;

    let mut rb_opt: Option<usize> = Some(n - 1);
    while let Some(rb) = rb_opt {
        let lb = partition[rb];
        let lower_discont = if lb == 0 {
            x[0] - 1.0
        } else {
            (x[lb] + x[lb - 1]) / 2.0
        };
        let interval: Vec<usize> = (lb..=rb).collect();
        interval_cell_rev.push(interval.clone());

        let pp = if interval.len() == 1 {
            // Constant piece: pp value = y[lb], extended over [lower_discont, upper_discont].
            let mut coefs = Array2::<f64>::zeros((dim, 4));
            for d in 0..dim {
                coefs[[d, 3]] = y[[lb, d]];
            }
            PiecewisePolynomial::new(
                Array1::from_vec(vec![lower_discont, upper_discont]),
                coefs,
                dim,
            )
        } else {
            let xi = x.slice(s![lb..=rb]).to_owned();
            let yi = y.slice(s![lb..=rb, ..]).to_owned();
            let wi = w.slice(s![lb..=rb]).to_owned();
            let mut pp = weighted_smoothing_spline(xi.view(), yi.view(), p, wi.view());
            pp.linext(lower_discont, upper_discont);
            pp.embed_to_cubic();
            pp
        };
        pp_cell_rev.push(pp);

        if lb == 0 {
            break;
        }
        discont_locations_rev.push(lower_discont);
        rb_opt = Some(lb - 1);
        upper_discont = lower_discont;
    }

    pp_cell_rev.reverse();
    interval_cell_rev.reverse();
    discont_locations_rev.reverse();

    let pp_cell = pp_cell_rev;
    let interval_cell = interval_cell_rev;
    let discont = Array1::from_vec(discont_locations_rev);
    let pp = PiecewisePolynomial::merge(pp_cell.clone());

    // discont_idx: end indices of each interval except the last.
    let discont_idx: Array1<usize> = if interval_cell.len() <= 1 {
        Array1::from_vec(vec![])
    } else {
        let mut v = Vec::with_capacity(interval_cell.len() - 1);
        for i in 0..interval_cell.len() - 1 {
            v.push(*interval_cell[i].last().unwrap());
        }
        Array1::from_vec(v)
    };

    Ok(CssdOutput {
        pp,
        discont,
        interval_cell,
        pp_cell,
        discont_idx,
        x,
        y,
        complexity_counter,
        f,
        partition,
        precondition,
        tau,
    })
}
