//! Algorithm-internal cross-check: FPVI and PELT must produce identical
//! outputs over a small signal × parameter grid. Mirrors
//! `tests/TestCSSD.m::prunings` (without MATLAB fixtures).

use approx::assert_abs_diff_eq;
use cssd_core::{cssd, Preconditioning, Pruning};
use ndarray::{Array1, Array2};

fn signal(values: &[f64]) -> (Array1<f64>, Array2<f64>) {
    let n = values.len();
    let x = Array1::from_iter((0..n).map(|i| (i + 1) as f64));
    let y = Array2::from_shape_vec((n, 1), values.to_vec()).unwrap();
    (x, y)
}

fn signals() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ]
}

#[test]
fn fpvi_pelt_agree_on_short_signals() {
    let p_grid = (0..30).map(|i| i as f64 / 29.0).collect::<Vec<_>>();
    let gamma_grid = (-10..=4).map(|k| 10f64.powi(k)).collect::<Vec<_>>();

    for (sidx, sig) in signals().iter().enumerate() {
        let (x, y) = signal(sig);
        let delta = Array1::from_elem(x.len(), 1.0);

        for &p in &p_grid {
            for &gamma in &gamma_grid {
                let out_fpvi = cssd(
                    Some(x.view()),
                    y.view(),
                    p,
                    gamma,
                    Some(delta.view()),
                    Pruning::Fpvi,
                    Preconditioning::None,
                )
                .expect("fpvi");
                let out_pelt = cssd(
                    Some(x.view()),
                    y.view(),
                    p,
                    gamma,
                    Some(delta.view()),
                    Pruning::Pelt,
                    Preconditioning::None,
                )
                .expect("pelt");

                let msg = format!("signal {} p={p} gamma={gamma}: pp coefs differ", sidx);
                assert_eq!(
                    out_fpvi.pp.coefs.dim(),
                    out_pelt.pp.coefs.dim(),
                    "{msg} (shape)"
                );
                for (a, b) in out_fpvi.pp.coefs.iter().zip(out_pelt.pp.coefs.iter()) {
                    assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
                }
                assert_eq!(
                    out_fpvi.partition, out_pelt.partition,
                    "signal {sidx} p={p} gamma={gamma}: partition differs"
                );
            }
        }
    }
}
