//! Cubic smoothing splines for discontinuous signals (CSSD).
//!
//! Rust port of the MATLAB reference implementation by Storath & Weinmann
//! (Journal of Computational and Graphical Statistics, 2023).

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod chk;
pub mod csaps;
pub mod cssd;
pub mod cv;
pub mod dp;
pub mod eps_lr;
pub mod ppform;
pub mod precond;
pub mod qr;

pub use cssd::{cssd, CssdOutput, Pruning};
pub use cv::cssd_cvscore;
pub use ppform::PiecewisePolynomial;
pub use precond::Preconditioning;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CssdError {
    #[error("p must satisfy 0 <= p <= 1, got {0}")]
    InvalidP(f64),
    #[error("gamma must satisfy 0 <= gamma, got {0}")]
    InvalidGamma(f64),
    #[error("at least two finite data points are required, got {0}")]
    NotEnoughData(usize),
    #[error("x and y must have matching length: x has {0}, y has {1}")]
    MismatchedXY(usize, usize),
    #[error("delta must have the same length as x ({0}), got {1}")]
    MismatchedDelta(usize, usize),
    #[error("x must be sorted ascending; duplicates are aggregated, but not unsorted input")]
    UnsortedX,
}

pub type Result<T> = std::result::Result<T, CssdError>;
