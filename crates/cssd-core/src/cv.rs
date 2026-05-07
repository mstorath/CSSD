//! K-fold cross-validation score for CSSD. Mirrors `cssd_cvscore.m`.

use crate::cssd::{cssd, Pruning};
use crate::precond::Preconditioning;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Compute the K-fold cross-validation score for a given (p, gamma).
///
/// `folds` is a list of K index vectors, each containing the test indices for
/// that fold (0-indexed into the original `(x, y)` arrays).
pub fn cssd_cvscore(
    x: ArrayView1<f64>,
    y: ArrayView2<f64>,
    p: f64,
    gamma: f64,
    delta: ArrayView1<f64>,
    folds: &[Vec<usize>],
    pruning: Pruning,
    precondition: Preconditioning,
) -> f64 {
    if !(0.0..=1.0).contains(&p) || gamma <= 0.0 {
        return f64::INFINITY;
    }
    let n = x.len();
    let dim = y.ncols();
    let k = folds.len();

    let mut total = 0.0_f64;
    for fold in folds {
        let test_set: std::collections::HashSet<usize> = fold.iter().copied().collect();
        let train_idx: Vec<usize> = (0..n).filter(|i| !test_set.contains(i)).collect();
        let test_idx = fold;

        let mut x_train = Array1::<f64>::zeros(train_idx.len());
        let mut y_train = Array2::<f64>::zeros((train_idx.len(), dim));
        let mut delta_train = Array1::<f64>::zeros(train_idx.len());
        for (j, &i) in train_idx.iter().enumerate() {
            x_train[j] = x[i];
            for d in 0..dim {
                y_train[[j, d]] = y[[i, d]];
            }
            delta_train[j] = delta[i];
        }

        let out = match cssd(
            Some(x_train.view()),
            y_train.view(),
            p,
            gamma,
            Some(delta_train.view()),
            pruning,
            precondition,
        ) {
            Ok(o) => o,
            Err(_) => return f64::INFINITY,
        };

        let n_test = test_idx.len();
        let mut x_test = Array1::<f64>::zeros(n_test);
        for (j, &i) in test_idx.iter().enumerate() {
            x_test[j] = x[i];
        }
        let pred = out.pp.eval(x_test.view());
        for (j, &i) in test_idx.iter().enumerate() {
            for d in 0..dim {
                let r = (pred[[j, d]] - y[[i, d]]) / delta[i];
                total += r * r;
            }
        }
        let _ = k;
    }
    total / n as f64
}
