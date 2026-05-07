//! Givens rotations and small QR primitives.
//!
//! `planerot` matches MATLAB's behaviour: returns G such that `G * [a; b] = [r; 0]`
//! with `r = hypot(a, b)`. The convention is the standard one used by LAPACK's
//! `drotg` with sign chosen so that the resulting `r` is non-negative when `a > 0`.

use nalgebra::Matrix2;

/// Givens rotation matrix that zeros the second component of `[a, b]`.
///
/// Mirrors MATLAB's `planerot([a; b])`: `G * [a; b] = [r; 0]`. Edge case `a==b==0`
/// returns the identity.
#[inline]
pub fn planerot(a: f64, b: f64) -> Matrix2<f64> {
    if b == 0.0 {
        Matrix2::identity()
    } else {
        let r = a.hypot(b);
        let c = a / r;
        let s = b / r;
        // [c  s]
        // [-s c]
        Matrix2::new(c, s, -s, c)
    }
}
