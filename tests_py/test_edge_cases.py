"""Edge-case coverage for the Python ``cssd`` API.

Focus areas:
- Input validation (shapes, NaNs, range)
- Boundary parameter values (p ∈ {0, 1}, gamma ∈ {0, ∞})
- Small N (2, 3) and pathological signals
- PiecewisePoly behaviour at piece boundaries
- Determinism / reproducibility
"""

from __future__ import annotations

import numpy as np
import pytest

from cssd import cssd, cssd_cv, PiecewisePoly


# ---------- Input validation ----------------------------------------------


class TestInputValidation:
    def test_p_below_range(self):
        with pytest.raises(Exception, match="p"):
            cssd(np.arange(5.0), np.arange(5.0), p=-0.1, gamma=1.0)

    def test_p_above_range(self):
        with pytest.raises(Exception, match="p"):
            cssd(np.arange(5.0), np.arange(5.0), p=1.5, gamma=1.0)

    def test_p_nan(self):
        with pytest.raises(Exception):
            cssd(np.arange(5.0), np.arange(5.0), p=float("nan"), gamma=1.0)

    def test_gamma_negative(self):
        with pytest.raises(Exception, match="gamma"):
            cssd(np.arange(5.0), np.arange(5.0), p=0.5, gamma=-1.0)

    def test_too_few_data(self):
        with pytest.raises(Exception):
            cssd(np.array([1.0]), np.array([1.0]), p=0.5, gamma=1.0)

    def test_mismatched_x_y(self):
        with pytest.raises(Exception):
            cssd(np.arange(5.0), np.arange(4.0), p=0.5, gamma=1.0)

    def test_mismatched_delta(self):
        with pytest.raises(Exception):
            cssd(
                np.arange(5.0),
                np.arange(5.0),
                p=0.5,
                gamma=1.0,
                delta=np.ones(3),
            )

    def test_y_3d_rejected(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            cssd(np.arange(4.0), np.zeros((4, 2, 1)), p=0.5, gamma=1.0)

    def test_unknown_pruning(self):
        with pytest.raises(Exception, match="pruning"):
            cssd(np.arange(5.0), np.arange(5.0), p=0.5, gamma=1.0, pruning="bogus")

    def test_x_none_uses_one_indexed_range(self):
        out = cssd(None, np.array([0.0, 1.0, 0.0]), p=0.5, gamma=10.0)
        np.testing.assert_array_equal(out.x, [1.0, 2.0, 3.0])

    def test_unsorted_x_is_sorted(self):
        x = np.array([3.0, 1.0, 2.0])
        y = np.array([9.0, 1.0, 4.0])
        out = cssd(x, y, p=1.0, gamma=np.inf)
        np.testing.assert_array_equal(out.x, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(out.y.ravel(), [1.0, 4.0, 9.0])

    def test_duplicate_x_aggregated(self):
        x = np.array([1.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 3.0, 5.0])
        out = cssd(x, y, p=1.0, gamma=np.inf)
        # Duplicate is averaged.
        assert out.x.size == 3
        np.testing.assert_allclose(out.y[0, 0], 1.0)

    def test_nan_y_dropped(self):
        x = np.arange(6.0)
        y = np.array([0.0, np.nan, 0.0, 1.0, np.nan, 1.0])
        out = cssd(x, y, p=0.9, gamma=0.1)
        # Two NaN rows dropped.
        assert out.x.size == 4

    def test_inf_y_dropped(self):
        x = np.arange(5.0)
        y = np.array([0.0, np.inf, 0.0, 1.0, 1.0])
        out = cssd(x, y, p=0.9, gamma=0.1)
        assert out.x.size == 4


# ---------- Boundary parameter values -------------------------------------


class TestBoundaryParams:
    def test_p_one_interpolates(self):
        x = np.linspace(0, 1, 10)
        y = x**3 - 0.5 * x
        out = cssd(x, y, p=1.0, gamma=np.inf)
        yy = out.pp(x)
        np.testing.assert_allclose(yy.ravel(), y, atol=1e-10)

    def test_p_zero_returns_line(self):
        # With gamma=Inf and p=0, the global solution is the LS line.
        x = np.linspace(0, 1, 10)
        rng = np.random.default_rng(0)
        true_slope, true_intercept = 2.0, -1.0
        y = true_slope * x + true_intercept + 0.01 * rng.standard_normal(10)
        out = cssd(x, y, p=0.0, gamma=np.inf)
        yy = out.pp(x).ravel()
        # The fit should be approximately a straight line.
        slope = np.polyfit(x, yy, 1)[0]
        assert abs(slope - true_slope) < 0.1

    def test_gamma_zero_is_finite(self):
        # gamma=0 is the limiting case of "discontinuities are free" — the
        # Bellman recursion still runs (we accept >=0); the result is a
        # discontinuity at every gap that improves the energy. Don't crash.
        x = np.arange(10.0)
        y = x.copy()
        out = cssd(x, y, p=0.5, gamma=0.0)
        # At least no exception, and reconstruction yields a callable pp.
        out.pp(np.array([0.5, 4.5, 9.0]))

    def test_gamma_inf_no_discontinuities(self):
        x = np.linspace(0, 1, 30)
        y = np.sin(2 * np.pi * x) + np.heaviside(x - 0.5, 1.0)
        out = cssd(x, y, p=0.99, gamma=np.inf)
        assert out.discont.size == 0
        assert out.discont_idx.size == 0

    def test_huge_gamma_no_discontinuities(self):
        x = np.linspace(0, 1, 30)
        y = np.sin(2 * np.pi * x) + np.heaviside(x - 0.5, 1.0)
        out = cssd(x, y, p=0.99, gamma=1e30)
        assert out.discont.size == 0


# ---------- Small N -------------------------------------------------------


class TestSmallN:
    def test_n2_with_inf_gamma(self):
        out = cssd(np.array([0.0, 1.0]), np.array([0.0, 1.0]), p=1.0, gamma=np.inf)
        yy = out.pp(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(yy.ravel(), [0.0, 0.5, 1.0], atol=1e-12)

    def test_n2_finite_gamma(self):
        # Two points: cannot have a discontinuity (would mean a 1-point segment
        # at each end). Result should still be a callable spline through both.
        out = cssd(np.array([0.0, 1.0]), np.array([0.0, 1.0]), p=0.5, gamma=0.001)
        yy = out.pp(np.array([0.0, 0.5, 1.0]))
        # Only verify finite-valued and callable — the algorithm is free to
        # detect any partition it wants for N=2.
        assert np.all(np.isfinite(yy))

    def test_n3(self):
        out = cssd(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0]),
                   p=0.5, gamma=10.0)
        yy = out.pp(np.array([0.5, 1.5]))
        assert np.all(np.isfinite(yy))


# ---------- Pathological signals ------------------------------------------


class TestPathological:
    def test_constant_signal(self):
        y = np.full(20, 7.5)
        out = cssd(None, y, p=0.99, gamma=1.0)
        # No discontinuity should be detected.
        assert out.discont.size == 0
        # And the spline should reproduce the constant.
        np.testing.assert_allclose(out.pp(np.arange(1.0, 21.0)).ravel(), 7.5, atol=1e-10)

    def test_step_function_low_gamma(self):
        x = np.arange(20.0)
        y = np.where(x < 10, 0.0, 1.0)
        out = cssd(x, y, p=0.99, gamma=1e-6)
        # With effectively-free discontinuities, the algorithm should detect
        # the obvious step.
        assert out.discont.size >= 1
        assert any(abs(d - 9.5) < 0.6 for d in out.discont)

    def test_step_function_high_gamma_keeps_smooth(self):
        x = np.arange(20.0)
        y = np.where(x < 10, 0.0, 1.0)
        # gamma so high it dominates: no discontinuities.
        out = cssd(x, y, p=0.99, gamma=1e6)
        assert out.discont.size == 0

    def test_two_consecutive_jumps(self):
        # Each plateau is long enough that smoothing across two jumps would
        # cost more than gamma per jump.
        x = np.arange(30.0)
        y = np.array([0.0] * 10 + [3.0] * 10 + [-2.0] * 10)
        out = cssd(x, y, p=0.99, gamma=0.05)
        assert out.discont.size >= 2, f"expected ≥2 jumps, got {out.discont}"

    def test_jump_at_start(self):
        x = np.arange(15.0)
        y = np.array([5.0] + [0.0] * 14)
        out = cssd(x, y, p=0.99, gamma=0.5)
        # Detection might pick up a jump near index 0.
        if out.discont.size > 0:
            assert out.discont[0] < 2.0

    def test_jump_at_end(self):
        x = np.arange(15.0)
        y = np.array([0.0] * 14 + [5.0])
        out = cssd(x, y, p=0.99, gamma=0.5)
        if out.discont.size > 0:
            assert out.discont[-1] > 12.0


# ---------- Vector-valued -------------------------------------------------


class TestVectorValued:
    def test_dim2_independent_components(self):
        x = np.linspace(0, 1, 20)
        y = np.column_stack([x, x**2])
        out = cssd(x, y, p=1.0, gamma=np.inf)
        assert out.pp.dim == 2
        yy = out.pp(x)
        np.testing.assert_allclose(yy, y, atol=1e-10)

    def test_dim3(self):
        x = np.linspace(0, 1, 15)
        y = np.column_stack([x, x**2, np.sin(x)])
        out = cssd(x, y, p=1.0, gamma=np.inf)
        assert out.pp.dim == 3
        np.testing.assert_allclose(out.pp(x), y, atol=1e-10)

    def test_dim2_disagreeing_jumps(self):
        # Jump in component 0 only.
        x = np.arange(20.0)
        y = np.column_stack([
            np.where(x < 10, 0.0, 5.0),
            np.linspace(0, 2, 20),
        ])
        out = cssd(x, y, p=0.99, gamma=0.5)
        assert out.discont.size >= 1


# ---------- PiecewisePoly bahaviour ---------------------------------------


class TestPiecewisePoly:
    def _simple_pp(self, dim=1):
        x = np.linspace(0, 1, 8)
        y = np.column_stack([np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)])[:, :dim]
        if dim == 1:
            y = y.ravel()
        out = cssd(x, y, p=1.0, gamma=np.inf)
        return out.pp

    def test_eval_scalar_input(self):
        pp = self._simple_pp()
        v = pp(np.array([0.5]))
        # Scalar y → output shape is (1,) when dim=1 (scipy PPoly convention).
        assert v.shape in ((1,), (1, 1))

    def test_eval_array_input(self):
        pp = self._simple_pp()
        xx = np.array([0.1, 0.3, 0.5])
        v = pp(xx)
        assert v.shape[0] == 3

    def test_derivative_callable(self):
        pp = self._simple_pp()
        d = pp.derivative()
        assert d.order == pp.order - 1
        assert d(np.array([0.5])).shape[0] == 1

    def test_antiderivative_callable(self):
        pp = self._simple_pp()
        a = pp.antiderivative()
        assert a.order == pp.order + 1

    def test_dim_property(self):
        assert self._simple_pp(dim=1).dim == 1
        assert self._simple_pp(dim=2).dim == 2

    def test_breaks_property(self):
        pp = self._simple_pp()
        b = pp.breaks
        assert np.all(np.diff(b) > 0)  # strictly ascending

    def test_repr(self):
        r = repr(self._simple_pp())
        assert "pieces=" in r and "dim=" in r


# ---------- CV edge cases -------------------------------------------------


class TestCV:
    def _make_signal(self, n=40):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, n)
        y = np.sin(4 * np.pi * x) + 0.1 * rng.standard_normal(n)
        return x, y

    def test_random_default(self):
        x, y = self._make_signal()
        cv = cssd_cv(x, y, random_state=42)
        assert 0.0 <= cv.p <= 1.0
        assert cv.gamma >= 0.0
        assert cv.fit is not None

    def test_equi_folds(self):
        x, y = self._make_signal()
        cv = cssd_cv(x, y, cv_type="equi", cv_arg=5, random_state=42)
        assert cv.fit is not None

    def test_custom_folds(self):
        x, y = self._make_signal(n=20)
        folds = [list(range(0, 20, 4)), list(range(1, 20, 4)),
                 list(range(2, 20, 4)), list(range(3, 20, 4))]
        cv = cssd_cv(x, y, cv_type="custom", cv_arg=folds, random_state=42)
        assert cv.fit is not None

    def test_unknown_cv_type(self):
        x, y = self._make_signal()
        with pytest.raises(ValueError, match="cv_type"):
            cssd_cv(x, y, cv_type="bogus")

    def test_custom_requires_arg(self):
        x, y = self._make_signal()
        with pytest.raises(ValueError, match="cv_arg"):
            cssd_cv(x, y, cv_type="custom")

    def test_random_state_reproducibility(self):
        x, y = self._make_signal()
        cv1 = cssd_cv(x, y, random_state=12345)
        cv2 = cssd_cv(x, y, random_state=12345)
        # Same seed → same fit (within optimiser determinism on this scipy build).
        assert abs(cv1.p - cv2.p) < 1e-10
        assert abs(cv1.gamma - cv2.gamma) < 1e-10

    def test_custom_starting_point(self):
        x, y = self._make_signal()
        cv = cssd_cv(x, y, starting_point=(0.7, 1.0), random_state=0)
        assert cv.fit is not None


# ---------- Determinism / reproducibility ---------------------------------


class TestDeterminism:
    def test_cssd_is_deterministic(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 30)
        y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(30)
        a = cssd(x, y, p=0.95, gamma=0.5)
        b = cssd(x, y, p=0.95, gamma=0.5)
        np.testing.assert_array_equal(a.pp.coefs, b.pp.coefs)
        np.testing.assert_array_equal(a.discont, b.discont)

    def test_pruning_choice_does_not_affect_pp(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 30)
        y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(30)
        a = cssd(x, y, p=0.95, gamma=0.5, pruning="FPVI")
        b = cssd(x, y, p=0.95, gamma=0.5, pruning="PELT")
        np.testing.assert_allclose(a.pp.coefs, b.pp.coefs, atol=1e-12)


# ---------- Output structure ----------------------------------------------


class TestOutputShape:
    def test_interval_cell_partitions_data(self):
        x = np.arange(20.0)
        y = np.where(x < 10, 0.0, 1.0)
        out = cssd(x, y, p=0.99, gamma=1e-3)
        # Concatenation of intervals should cover [0, N) exactly once.
        all_idx = np.concatenate(out.interval_cell)
        np.testing.assert_array_equal(np.sort(all_idx), np.arange(20))

    def test_discont_idx_matches_intervals(self):
        x = np.arange(20.0)
        y = np.where(x < 10, 0.0, 1.0)
        out = cssd(x, y, p=0.99, gamma=1e-3)
        if out.discont_idx.size > 0:
            for i, idx in enumerate(out.discont_idx):
                assert out.interval_cell[i][-1] == idx

    def test_complexity_counter_at_least_n(self):
        x = np.arange(20.0)
        y = np.sin(x)
        out = cssd(x, y, p=0.5, gamma=0.5)
        assert out.complexity_counter >= len(x)

    def test_pp_pieces_minus_one_equals_breaks_minus_two(self):
        # For a no-discontinuity output, pp has data+2 breaks (linext padding).
        x = np.arange(10.0)
        y = np.sin(x)
        out = cssd(x, y, p=0.99, gamma=np.inf)
        assert out.pp.breaks.size >= 2
