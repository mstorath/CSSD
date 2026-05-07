//! Locally-adaptive Jacobi preconditioning of the Hermite design matrix.
//!
//! The design matrix $A^{(r)}$ in the paper has columns alternating between
//! "value" $f_i$ and "slope" $f'_i$ unknowns, with the smoothness rows
//! contributing entries of order $d^{-3/2}$ to value columns and $d^{-1/2}$ to
//! slope columns. For non-uniform meshes (large mesh ratio) this gives
//! column norms that differ by orders of magnitude, blowing up the
//! condition number $\kappa(A^{(r)})$.
//!
//! A diagonal column-scaling preconditioner $D = \mathrm{diag}(1, \tau_1, 1,
//! \tau_2, \ldots)$ — i.e. multiplying every slope column by a per-knot
//! factor — leaves the LSQ minimum invariant but can rebalance column
//! norms. The QR-update structure (paper eq. 14) is preserved as long as
//! the existing $\tau_1, \dots, \tau_r$ are *frozen* once set, so the
//! preconditioner extends block-diagonally as new knots are added.
//!
//! ## The local choice
//!
//! For interior knot $i$, the column-norm balance gives
//! $$\tau_i^2 \;=\; \frac{\alpha_i^2 + 12\beta^2(h_{i-1}^{-3} + h_i^{-3})}{4\beta^2(h_{i-1}^{-1} + h_i^{-1})}$$
//! where $\alpha_i = \sqrt{p}/\delta_i$, $\beta = \sqrt{1-p}$, and
//! $h_j = x_{j+1} - x_j$. Boundaries use a one-sided version.

use ndarray::{Array1, ArrayView1};

/// Preconditioning mode for the Hermite QR system in the DP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preconditioning {
    /// $\tau_i \equiv 1$ — original paper algorithm, no scaling.
    #[default]
    None,
    /// Locally-adaptive $\tau_i$ from the algebraically-balanced formula.
    Local,
}

impl Preconditioning {
    /// Parse the API string `"none"` / `"local"` (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "local" => Some(Self::Local),
            _ => None,
        }
    }
}

/// Compute the locally-adaptive slope-scaling vector $\tau \in \mathbb{R}^N$.
///
/// Boundaries use a one-sided neighbour ($h_0$ for knot 0, $h_{N-2}$ for
/// knot $N-1$); interiors use both adjacent intervals.
///
/// `h.len() == n - 1` and `alpha.len() == n` are required.
pub fn local_tau(h: ArrayView1<f64>, alpha: ArrayView1<f64>, beta: f64) -> Array1<f64> {
    let n = alpha.len();
    debug_assert_eq!(h.len() + 1, n);
    let mut tau = Array1::<f64>::ones(n);
    let beta2 = beta * beta;

    // β must be > 0 for the formula to make sense; β = 0 ⇒ p = 1 ⇒ pure
    // interpolation, which never enters the DP path anyway. Defend by
    // returning all-ones if it does.
    if beta2 == 0.0 {
        return tau;
    }

    let knot_tau = |a: f64, h_inv_cube_sum: f64, h_inv_sum: f64| -> f64 {
        // τ² = (α² + 12 β² · Σ h⁻³) / (4 β² · Σ h⁻¹)
        let num = a * a + 12.0 * beta2 * h_inv_cube_sum;
        let den = 4.0 * beta2 * h_inv_sum;
        (num / den).sqrt()
    };

    // First knot: only h[0].
    {
        let h0 = h[0];
        tau[0] = knot_tau(alpha[0], h0.powi(-3), 1.0 / h0);
    }
    // Last knot: only h[N-2].
    {
        let hl = h[n - 2];
        tau[n - 1] = knot_tau(alpha[n - 1], hl.powi(-3), 1.0 / hl);
    }
    // Interior.
    for i in 1..n - 1 {
        let hl = h[i - 1];
        let hr = h[i];
        tau[i] = knot_tau(alpha[i], hl.powi(-3) + hr.powi(-3), 1.0 / hl + 1.0 / hr);
    }

    tau
}

/// All-ones τ vector — a convenience for the `None` preconditioning mode.
pub fn unit_tau(n: usize) -> Array1<f64> {
    Array1::ones(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn parse_strings() {
        assert_eq!(Preconditioning::parse("none"), Some(Preconditioning::None));
        assert_eq!(Preconditioning::parse("None"), Some(Preconditioning::None));
        assert_eq!(
            Preconditioning::parse("LOCAL"),
            Some(Preconditioning::Local)
        );
        assert_eq!(Preconditioning::parse("global"), None);
    }

    #[test]
    fn unit_tau_is_ones() {
        let t = unit_tau(5);
        for &v in t.iter() {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn uniform_mesh_recovers_global_tau() {
        // For uniform spacing h and α = 0 (smoothness-dominated),
        // τ_i = √3 / h for interior knots.
        let h = Array1::from_elem(4, 0.25); // 5 knots, uniform spacing 0.25
        let alpha = Array1::from_elem(5, 0.0);
        let tau = local_tau(h.view(), alpha.view(), 1.0);
        let expected_interior = (3.0_f64).sqrt() / 0.25;
        for i in 1..4 {
            assert_abs_diff_eq!(tau[i], expected_interior, epsilon = 1e-12);
        }
        // Boundaries use one-sided formula: τ = √3 / h (same value here).
        assert_abs_diff_eq!(tau[0], expected_interior, epsilon = 1e-12);
        assert_abs_diff_eq!(tau[4], expected_interior, epsilon = 1e-12);
    }

    #[test]
    fn larger_h_gives_smaller_tau() {
        let h = array![0.1, 1.0, 0.1, 1.0];
        let alpha = Array1::zeros(5);
        let tau = local_tau(h.view(), alpha.view(), 1.0);
        // Knot 0 (only h[0]=0.1): τ ~ √3/0.1
        // Knot 1 (h[0]=0.1, h[1]=1.0): smaller h dominates, τ between √3/0.1 and √3/1.
        // Knot 2 (h[1]=1.0, h[2]=0.1): same as knot 1 by symmetry.
        assert!(tau[0] > tau[1]); // pure h=0.1 vs mixed
        assert_abs_diff_eq!(tau[1], tau[2], epsilon = 1e-12); // symmetry
    }

    #[test]
    fn data_term_increases_tau() {
        // Larger α (heavier data weight) should increase τ.
        let h = Array1::from_elem(4, 0.5);
        let alpha_low = Array1::from_elem(5, 0.0);
        let alpha_high = Array1::from_elem(5, 100.0);
        let tau_low = local_tau(h.view(), alpha_low.view(), 1.0);
        let tau_high = local_tau(h.view(), alpha_high.view(), 1.0);
        for i in 0..5 {
            assert!(tau_high[i] > tau_low[i], "τ should grow with α at knot {i}");
        }
    }
}
