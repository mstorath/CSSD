//! Paper-compliance tests — verifies the implementation against the specific
//! equations and lemmas in:
#![allow(clippy::needless_range_loop, clippy::manual_div_ceil)]
//!
//!   M. Storath, A. Weinmann, "Smoothing splines for discontinuous signals",
//!   Journal of Computational and Graphical Statistics, 2023,
//!   arXiv:2211.12785v2
//!
//! Each test names the paper artefact it is checking.

use approx::assert_abs_diff_eq;
use cssd_core::ppform::spline_inner_energy;
use cssd_core::{cssd, eps_lr, Preconditioning, Pruning};
use ndarray::{array, s, Array1, Array2};

// ---------- Eq. (8) Hermite parametrisation -------------------------------
//
// Per-interval cubic
//   p_i(x) = c0 + c1·t + c2·t² + c3·t³,  t = x - x_i
// with
//   c0 = f_i,  c1 = f'_i,
//   c2 = -(f'_{i+1} + 2 f'_i)/d_i + 3(f_{i+1} - f_i)/d_i²,
//   c3 = (f'_{i+1} + f'_i)/d_i² + 2(f_i - f_{i+1})/d_i³.
//
// We use this as a sanity check on csaps reconstruction: when csaps fits an
// interpolation through (x_i, y_i), the per-piece coefficients (in MATLAB
// decreasing-power convention) must satisfy these formulas with the Hermite
// derivatives that csaps computes.

#[test]
fn eq8_hermite_parametrisation_for_interpolation() {
    // Interpolating cubic through (0, 0), (1, 1), (2, 0) with natural BCs.
    // Natural cubic: f''(0) = f''(2) = 0. Solve for f'_0, f'_1, f'_2.
    let x = array![0.0, 1.0, 2.0];
    let y = array![[0.0], [1.0], [0.0]];
    let w = Array1::from_elem(3, 1.0);
    let pp = cssd_core::csaps::weighted_smoothing_spline(x.view(), y.view(), 1.0, w.view());

    // For each interval, recover Hermite values f_i, f_{i+1}, f'_i, f'_{i+1}
    // from the polynomial and re-derive the c2, c3 coefficients.
    for i in 0..pp.pieces() {
        let d_i = pp.breaks[i + 1] - pp.breaks[i];
        // pp.coefs row i in MATLAB convention is [c3, c2, c1, c0] (decreasing powers).
        let c3 = pp.coefs[[i, 0]];
        let c2 = pp.coefs[[i, 1]];
        let c1 = pp.coefs[[i, 2]]; // = f'_i
        let c0 = pp.coefs[[i, 3]]; // = f_i

        // f_i, f_{i+1}: evaluate at t=0 and t=d_i.
        let f_i = c0;
        let f_ip1 = c3 * d_i.powi(3) + c2 * d_i.powi(2) + c1 * d_i + c0;
        // f'_i = c1; f'_{i+1} = derivative at t = d_i.
        let fp_i = c1;
        let fp_ip1 = 3.0 * c3 * d_i.powi(2) + 2.0 * c2 * d_i + c1;

        // Re-derive from eq. (8).
        let c2_expected = -(fp_ip1 + 2.0 * fp_i) / d_i + 3.0 * (f_ip1 - f_i) / d_i.powi(2);
        let c3_expected = (fp_ip1 + fp_i) / d_i.powi(2) + 2.0 * (f_i - f_ip1) / d_i.powi(3);

        assert_abs_diff_eq!(c2, c2_expected, epsilon = 1e-12);
        assert_abs_diff_eq!(c3, c3_expected, epsilon = 1e-12);
    }
}

// ---------- Eq. (9) U_i smoothness factor matrix --------------------------
//
//   U_i = [ 2√3 / d_i^(3/2)   √3 / √d_i  -2√3 / d_i^(3/2)   √3 / √d_i ]
//         [       0             1 / √d_i        0             -1 / √d_i ]
//
// β = √(1-p) is multiplied externally. The factorisation B_i = U_i^T U_i
// gives the integral ∫ (p''_i)² dx as v_i^T B_i v_i with
// v_i = [f_i, f'_i, f_{i+1}, f'_{i+1}].

#[test]
fn eq9_ui_factorisation_recovers_inner_energy() {
    // For a cubic with known second-derivative integral, verify that
    // β² · ‖U_i v_i‖² equals (1-p) · ∫ (p''(t))² dt with β² = (1-p).
    //
    // Take a simple cubic on [0, 1]: p(t) = t² (so p''=2 const, ∫p''² = 4).
    // v = [p(0), p'(0), p(1), p'(1)] = [0, 0, 1, 2]. d = 1. β² · ‖U v‖² = ?
    let d = 1.0_f64;
    let v = array![0.0, 0.0, 1.0, 2.0];
    let sqrt3 = 3.0_f64.sqrt();
    let u_row1 = array![
        2.0 * sqrt3 * d.powf(-1.5),
        sqrt3 * d.powf(-0.5),
        -2.0 * sqrt3 * d.powf(-1.5),
        sqrt3 * d.powf(-0.5),
    ];
    let u_row2 = array![0.0, d.powf(-0.5), 0.0, -d.powf(-0.5)];
    let row1: f64 = u_row1.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let row2: f64 = u_row2.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let u_norm_sq = row1 * row1 + row2 * row2;
    // Expected = ∫_0^1 (p''(t))² dt = ∫ 4 dt = 4.
    assert_abs_diff_eq!(u_norm_sq, 4.0, epsilon = 1e-12);
}

#[test]
fn eq9_ui_zero_on_linear() {
    // For a linear p(t) = a + b·t, p'' = 0, so v^T B v = 0.
    // v = [a, b, a+b, b]. ‖U v‖² = 0.
    let d = 1.0_f64;
    let v = array![0.5, 1.0, 1.5, 1.0];
    let sqrt3 = 3.0_f64.sqrt();
    let u_row1 = array![
        2.0 * sqrt3 * d.powf(-1.5),
        sqrt3 * d.powf(-0.5),
        -2.0 * sqrt3 * d.powf(-1.5),
        sqrt3 * d.powf(-0.5),
    ];
    let u_row2 = array![0.0, d.powf(-0.5), 0.0, -d.powf(-0.5)];
    let row1: f64 = u_row1.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let row2: f64 = u_row2.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    assert_abs_diff_eq!(row1 * row1 + row2 * row2, 0.0, epsilon = 1e-12);
}

// ---------- Eq. (15) E{1:r+1} = E{1:r} + (z'_5)² --------------------------
//
// The QR-update step adds exactly one residual contribution per data point.
// For a perfectly-linear y, the increment is zero (linear data has zero
// smoothness energy and can be fit exactly).

#[test]
fn eq15_zero_increment_on_linear_signal() {
    let y = array![[1.0], [3.0], [5.0], [7.0], [9.0]]; // y = 2x + 1
    let alpha = [1.0; 5];
    let beta = 1.0;

    let mut st = eps_lr::start(
        y.slice(s![0..2, ..]),
        1.0,
        [alpha[0], alpha[1]],
        beta,
        1.0,
        1.0,
    );
    let mut prev = st.eps;
    for r in 2..5 {
        st = eps_lr::update(
            &st,
            y.slice(s![r..r + 1, ..]),
            1.0,
            alpha[r],
            beta,
            1.0,
            1.0,
        );
        let increment = st.eps - prev;
        assert!(
            increment >= -1e-15,
            "increment must be non-negative (eq. 15)"
        );
        assert_abs_diff_eq!(increment, 0.0, epsilon = 1e-10);
        prev = st.eps;
    }
}

// ---------- Eq. (7) Bellman recurrence -----------------------------------
//
//   F*_r = min_{l=1..r} { E{l:r} + γ + F*_{l-1} },  F*_0 = -γ.
//
// We use the equivalent split form (l=1 is the no-discontinuity case
// represented by F*_r = E{1:r} initially, then candidates with l ≥ 2 are
// considered). The reconstructed segments must satisfy: the sum of their
// individual energies plus γ·|J| equals the minimum F*_N.

#[test]
fn eq7_bellman_value_equals_segment_energy_plus_gamma_cost() {
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let y = array![
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [3.0],
        [3.0],
        [3.0],
        [3.0],
        [3.0]
    ];
    let delta = Array1::from_elem(10, 1.0);
    let gamma = 0.01;
    let p = 0.99;
    let out = cssd(
        Some(x.view()),
        y.view(),
        p,
        gamma,
        Some(delta.view()),
        Pruning::Fpvi,
        Preconditioning::None,
    )
    .unwrap();

    // F*_N (last entry of f) should equal sum of per-segment energies + γ·|J|.
    let f_n = out.f[out.f.len() - 1];
    let n_jumps = out.discont.len() as f64;

    // Recompute per-segment energies via the Hermite/QR machinery.
    let alpha: Array1<f64> = delta.iter().map(|d| p.sqrt() / d).collect();
    let beta = (1.0 - p).sqrt();
    let h: Array1<f64> = (0..x.len() - 1).map(|i| x[i + 1] - x[i]).collect();
    let mut total_energy = 0.0;
    for interval in &out.interval_cell {
        if interval.len() < 2 {
            continue; // E{l:l} = 0
        }
        let lo = interval[0];
        let hi = *interval.last().unwrap();
        let mut state = eps_lr::start(
            y.slice(s![lo..lo + 2, ..]),
            h[lo],
            [alpha[lo], alpha[lo + 1]],
            beta,
            1.0,
            1.0,
        );
        for r in lo + 2..=hi {
            let y_row = y.slice(s![r..r + 1, ..]);
            state = eps_lr::update(&state, y_row, h[r - 1], alpha[r], beta, 1.0, 1.0);
        }
        total_energy += state.eps;
    }
    assert_abs_diff_eq!(f_n, total_energy + gamma * n_jumps, epsilon = 1e-9);
}

// ---------- Lemma 1.1: ≤1 discontinuity between adjacent data sites -------
//
// Restriction to midpoints (sec. 2.2) enforces this — discont_idx is a
// strictly-ascending list of integer indices.

#[test]
fn lemma_1_1_at_most_one_discont_between_data_sites() {
    let x = Array1::from_iter((0..30).map(|i| i as f64));
    let mut y = Array2::<f64>::zeros((30, 1));
    for i in 0..30 {
        y[[i, 0]] = if i < 10 {
            0.0
        } else if i < 20 {
            1.0
        } else {
            -1.0
        };
    }
    let out = cssd(
        Some(x.view()),
        y.view(),
        0.99,
        0.05,
        None,
        Pruning::Fpvi,
        Preconditioning::None,
    )
    .unwrap();

    let mut prev: i64 = -1;
    for &idx in out.discont_idx.iter() {
        let i = idx as i64;
        assert!(
            i > prev,
            "discont_idx must be strictly ascending: {:?}",
            out.discont_idx
        );
        prev = i;
    }
}

// ---------- Lemma 1.2: |J| ≤ ⌈N/2⌉ - 1 -----------------------------------

#[test]
fn lemma_1_2_max_discontinuity_count() {
    // Adversarial: pure-noise signal with very small γ should attempt many jumps.
    let n = 21;
    let x = Array1::from_iter((0..n).map(|i| i as f64));
    let mut y = Array2::<f64>::zeros((n, 1));
    let mut state = 1u64;
    for i in 0..n {
        // Simple xorshift to avoid `rand` dep; deterministic.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        y[[i, 0]] = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
    }

    let out = cssd(
        Some(x.view()),
        y.view(),
        0.5,
        1e-12,
        None,
        Pruning::Fpvi,
        Preconditioning::None,
    )
    .unwrap();

    let max_jumps = (n + 1) / 2 - 1; // ⌈N/2⌉ - 1
    assert!(
        out.discont.len() <= max_jumps,
        "Lemma 1.2 violated: {} discontinuities for N={} (max ⌈N/2⌉-1={})",
        out.discont.len(),
        n,
        max_jumps
    );
}

// ---------- Sec. 2.2: midpoint property ----------------------------------
//
// Each detected discontinuity in `discont` lies exactly at a midpoint of
// adjacent data sites.

#[test]
fn discontinuities_are_at_midpoints_of_adjacent_data_sites() {
    let x = array![0.0, 1.5, 2.7, 4.1, 5.0, 6.3, 7.0, 8.0, 9.0, 10.0];
    let y = array![
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [5.0],
        [5.0],
        [5.0],
        [5.0],
        [5.0]
    ];
    let out = cssd(
        Some(x.view()),
        y.view(),
        0.95,
        0.01,
        None,
        Pruning::Fpvi,
        Preconditioning::None,
    )
    .unwrap();

    for &d in out.discont.iter() {
        let mut found = false;
        for i in 0..x.len() - 1 {
            let mid = 0.5 * (x[i] + x[i + 1]);
            if (mid - d).abs() < 1e-12 {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "Discontinuity {} is not at any midpoint {:?}",
            d,
            x.windows(2)
                .into_iter()
                .map(|w| 0.5 * (w[0] + w[1]))
                .collect::<Vec<_>>()
        );
    }
}

// ---------- Theorem 4 / runtime sanity: γ=∞ ⇒ exactly one segment ----------

#[test]
fn gamma_inf_yields_single_segment() {
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = array![[0.0], [0.0], [0.0], [10.0], [10.0], [10.0], [10.0]];
    let out = cssd(
        Some(x.view()),
        y.view(),
        0.99,
        f64::INFINITY,
        None,
        Pruning::Fpvi,
        Preconditioning::None,
    )
    .unwrap();
    assert_eq!(out.discont.len(), 0);
    assert_eq!(out.interval_cell.len(), 1);
}

// ---------- Sec. 1: smoothness energy ∫ f''² is captured exactly ----------
//
// The fast-update energy at l=1, r=N must equal the inner energy of the
// reconstructed natural cubic spline (when γ=∞ and p<1, only the smoothness
// term contributes the residual).

#[test]
fn fast_update_energy_matches_reinsch_smoothness_for_smooth_data() {
    let x = array![0.0, 0.25, 0.5, 0.75, 1.0];
    let y = array![[0.0], [0.5], [0.8], [1.0], [1.2]]; // smooth, no jumps
    let p = 0.5;

    let out = cssd(
        Some(x.view()),
        y.view(),
        p,
        f64::INFINITY,
        None,
        Pruning::Fpvi,
        Preconditioning::None,
    )
    .unwrap();

    // Reconstruct via Reinsch (csaps) and compute its smoothness energy.
    let w: Array1<f64> = Array1::from_elem(x.len(), 1.0);
    let pp_recon = cssd_core::csaps::weighted_smoothing_spline(x.view(), y.view(), p, w.view());
    let smooth_e = spline_inner_energy(&pp_recon);

    // Data residual term:
    let yy = pp_recon.eval(x.view());
    let mut data_e = 0.0;
    for i in 0..x.len() {
        let r = yy[[i, 0]] - y[[i, 0]];
        data_e += r * r;
    }
    let total = p * data_e + (1.0 - p) * smooth_e[0];

    // The fast-update reports F[N-1] = E{1:N}. With γ=∞ and any p, the no-jump
    // segment is optimal; the Bellman value at the last position should
    // recover the same energy, modulo Reinsch ↔ fast-QR numerics.
    // (Note: the gamma=∞ short-circuit doesn't run the DP, so we can't read
    // F directly. Instead we run with gamma=large-but-finite.)
    let out_finite = cssd(
        Some(x.view()),
        y.view(),
        p,
        1e10,
        None,
        Pruning::Fpvi,
        Preconditioning::None,
    )
    .unwrap();
    let f_last = out_finite.f[x.len() - 1];
    assert_abs_diff_eq!(f_last, total, epsilon = 1e-8);
    let _ = out;
}

// ---------- Remark 2: vector-valued energy decomposes additively ----------

#[test]
fn remark2_vector_energy_is_sum_of_components() {
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
    let y_full = array![[1.0, 2.0], [0.5, -1.0], [2.0, 0.0], [-0.5, 3.0], [1.0, 1.0]];
    let p = 0.5;
    let gamma = 1e8; // effectively no jumps
    let out_full = cssd(
        Some(x.view()),
        y_full.view(),
        p,
        gamma,
        None,
        Pruning::Fpvi,
        Preconditioning::None,
    )
    .unwrap();

    let mut sum_components = 0.0_f64;
    for c in 0..2 {
        let y_c = y_full.slice(s![.., c..c + 1]).to_owned();
        let out_c = cssd(
            Some(x.view()),
            y_c.view(),
            p,
            gamma,
            None,
            Pruning::Fpvi,
            Preconditioning::None,
        )
        .unwrap();
        sum_components += out_c.f[out_c.f.len() - 1];
    }
    let f_total = out_full.f[out_full.f.len() - 1];
    assert_abs_diff_eq!(f_total, sum_components, epsilon = 1e-10);
}

// ---------- Sec. 3.4: CV score formula -----------------------------------
//
// CV(p, γ) = (1/N) Σ_k Σ_{i ∈ Fold_k} ((f^{-k}(x_i) - y_i) / δ_i)²

#[test]
fn cv_score_matches_definition() {
    let x = Array1::from_iter((0..20).map(|i| i as f64 / 19.0));
    let y = Array1::from_iter((0..20).map(|i| {
        let xi = i as f64 / 19.0;
        (2.0 * std::f64::consts::PI * xi).sin()
    }))
    .insert_axis(ndarray::Axis(1));
    let delta = Array1::from_elem(20, 1.0);

    // Two folds, alternating indices.
    let fold0: Vec<usize> = (0..20).step_by(2).collect();
    let fold1: Vec<usize> = (1..20).step_by(2).collect();
    let folds = vec![fold0.clone(), fold1.clone()];

    let p = 0.5;
    let gamma = 1.0;
    let cv = cssd_core::cssd_cvscore(
        x.view(),
        y.view(),
        p,
        gamma,
        delta.view(),
        &folds,
        Pruning::Fpvi,
        Preconditioning::None,
    );

    // Manual recomputation.
    let mut total = 0.0_f64;
    for fold in &folds {
        let train_idx: Vec<usize> = (0..x.len()).filter(|i| !fold.contains(i)).collect();
        let x_train: Array1<f64> = train_idx.iter().map(|&i| x[i]).collect();
        let y_train: Array2<f64> = {
            let mut a = Array2::zeros((train_idx.len(), 1));
            for (j, &i) in train_idx.iter().enumerate() {
                a[[j, 0]] = y[[i, 0]];
            }
            a
        };
        let delta_train: Array1<f64> = train_idx.iter().map(|&i| delta[i]).collect();
        let out = cssd(
            Some(x_train.view()),
            y_train.view(),
            p,
            gamma,
            Some(delta_train.view()),
            Pruning::Fpvi,
            Preconditioning::None,
        )
        .unwrap();
        for &i in fold {
            let pred = out.pp.eval(array![x[i]].view())[[0, 0]];
            let r = (pred - y[[i, 0]]) / delta[i];
            total += r * r;
        }
    }
    let expected = total / x.len() as f64;
    assert_abs_diff_eq!(cv, expected, epsilon = 1e-12);
}
