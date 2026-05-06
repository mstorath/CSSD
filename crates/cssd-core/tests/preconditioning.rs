//! Tests for the locally-adaptive preconditioning option.
//!
//! Diagonal column scaling leaves the LSQ minimum invariant, so:
//! - `partition` (and `discont`) must be byte-equal between modes.
//! - `F` values must agree to within float noise (much better than the
//!   typical numerical tolerance — for these inputs the median diff is
//!   ~ machine epsilon).
//! - `pp.coefs` from reconstruction must agree (Reinsch is unaffected by
//!   the Hermite-form preconditioning).

use approx::assert_abs_diff_eq;
use cssd_core::{cssd, Preconditioning, Pruning};
use ndarray::{Array1, Array2};

fn run(
    x: &Array1<f64>,
    y: &Array1<f64>,
    p: f64,
    gamma: f64,
    mode: Preconditioning,
) -> cssd_core::CssdOutput {
    let y2 = Array2::from_shape_vec((y.len(), 1), y.to_vec()).unwrap();
    let delta = Array1::from_elem(x.len(), 1.0);
    cssd(
        Some(x.view()),
        y2.view(),
        p,
        gamma,
        Some(delta.view()),
        Pruning::Fpvi,
        mode,
    )
    .unwrap()
}

#[test]
fn partition_invariance_uniform_mesh() {
    let x = Array1::from_iter((0..30).map(|i| i as f64));
    let y = Array1::from_iter((0..30).map(|i| (0.3 * i as f64).sin()));
    for &gamma in &[1e-3, 0.1, 1.0, 100.0] {
        let off = run(&x, &y, 0.99, gamma, Preconditioning::None);
        let on = run(&x, &y, 0.99, gamma, Preconditioning::Local);
        assert_eq!(off.partition, on.partition, "gamma={gamma}");
        assert_eq!(off.discont.len(), on.discont.len());
    }
}

#[test]
fn partition_invariance_nonuniform_mesh() {
    // Mesh ratio ~ 100.
    let mut xs: Vec<f64> = (0..50).map(|i| 0.001 * i as f64).collect();
    xs.extend((1..50).map(|i| 0.05 + 0.02 * i as f64));
    let x = Array1::from_vec(xs);
    let y: Array1<f64> = x.iter().map(|&xi| (10.0 * xi).sin()).collect();

    for &gamma in &[1e-4, 0.01, 1.0, 1e6] {
        let off = run(&x, &y, 0.99, gamma, Preconditioning::None);
        let on = run(&x, &y, 0.99, gamma, Preconditioning::Local);
        assert_eq!(off.partition, on.partition, "gamma={gamma}");
    }
}

#[test]
fn f_values_agree_to_float_precision() {
    // Plain smoothing-spline regime (gamma = 1e10) with non-uniform mesh.
    let mut xs: Vec<f64> = (0..40).map(|i| 0.005 * i as f64).collect();
    xs.extend((1..40).map(|i| 0.2 + 0.02 * i as f64));
    let x = Array1::from_vec(xs);
    let y: Array1<f64> = x.iter().map(|&xi| (10.0 * xi).sin()).collect();

    let off = run(&x, &y, 0.99, 1e10, Preconditioning::None);
    let on = run(&x, &y, 0.99, 1e10, Preconditioning::Local);

    for i in 0..off.f.len() {
        if off.f[i].abs() < 1e-12 {
            assert!(on.f[i].abs() < 1e-9);
        } else {
            let rel = (off.f[i] - on.f[i]).abs() / off.f[i].abs();
            assert!(
                rel < 1e-9,
                "F[{i}]: off={} on={} rel={rel}",
                off.f[i],
                on.f[i]
            );
        }
    }
}

#[test]
fn pp_coefs_agree_after_reconstruction() {
    let x = Array1::from_iter((0..20).map(|i| (i as f64) * 0.3));
    let y: Array1<f64> = x.iter().map(|&xi| xi * xi - xi).collect();
    let off = run(&x, &y, 0.7, 0.5, Preconditioning::None);
    let on = run(&x, &y, 0.7, 0.5, Preconditioning::Local);
    assert_eq!(off.pp.coefs.dim(), on.pp.coefs.dim());
    for (a, b) in off.pp.coefs.iter().zip(on.pp.coefs.iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-10);
    }
}

#[test]
fn local_tau_recovers_global_for_uniform_mesh() {
    // For uniform spacing and the smoothness-dominated regime (small alpha),
    // the locally-adaptive tau should be approximately the global value
    // sqrt(3) / h at every interior knot.
    let n = 10;
    let h_val = 0.5;
    let h = Array1::from_elem(n - 1, h_val);
    // alpha small relative to beta to suppress the data-fidelity contribution.
    let alpha = Array1::from_elem(n, 1e-6);
    let beta = 1.0;
    let tau = cssd_core::precond::local_tau(h.view(), alpha.view(), beta);
    let expected = (3.0_f64).sqrt() / h_val;
    for i in 0..n {
        let rel = (tau[i] - expected).abs() / expected;
        assert!(
            rel < 1e-6,
            "tau[{i}] = {} differs from expected {} (rel diff {})",
            tau[i],
            expected,
            rel
        );
    }
}

#[test]
fn fpvi_pelt_agree_with_local_preconditioning() {
    // The FPVI/PELT cross-check (existing parity test) should still hold
    // when both are run with Local preconditioning.
    let x = Array1::from_iter((0..15).map(|i| 0.7_f64.powi(i))); // geometric mesh, very non-uniform
    let y: Array1<f64> = x.iter().map(|&xi| xi.cos()).collect();

    for &gamma in &[1e-4, 0.1, 10.0] {
        let y2 = Array2::from_shape_vec((y.len(), 1), y.to_vec()).unwrap();
        let delta = Array1::from_elem(x.len(), 1.0);
        let f = cssd(
            Some(x.view()),
            y2.view(),
            0.5,
            gamma,
            Some(delta.view()),
            Pruning::Fpvi,
            Preconditioning::Local,
        )
        .unwrap();
        let pe = cssd(
            Some(x.view()),
            y2.view(),
            0.5,
            gamma,
            Some(delta.view()),
            Pruning::Pelt,
            Preconditioning::Local,
        )
        .unwrap();
        assert_eq!(f.partition, pe.partition);
        for (a, b) in f.pp.coefs.iter().zip(pe.pp.coefs.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-10);
        }
    }
}
