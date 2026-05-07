"""Tests for the optional `precondition='local'` argument."""

from __future__ import annotations

import numpy as np
import pytest

from cssd import cssd, cssd_cv


def _nonuniform_mesh(seed: int = 0, mesh_ratio: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a mesh with controllable mesh ratio + a smooth target signal."""
    rng = np.random.default_rng(seed)
    n_dense, n_sparse = 40, 40
    dense = rng.uniform(0.0, 1.0 / mesh_ratio, size=n_dense)
    sparse = rng.uniform(1.0 / mesh_ratio, 1.0, size=n_sparse)
    x = np.sort(np.concatenate([dense, sparse]))
    y = np.sin(8 * np.pi * x)
    return x, y


def test_default_is_none():
    x = np.linspace(0, 1, 10)
    out = cssd(x, np.sin(x), p=0.99, gamma=1e10)
    assert out.precondition == "none"


def test_local_string_accepted():
    x = np.linspace(0, 1, 10)
    out = cssd(x, np.sin(x), p=0.99, gamma=1e10, precondition="local")
    assert out.precondition == "local"
    assert out.tau.shape == (10,)
    assert np.all(out.tau > 0)


def test_unknown_precondition_errors():
    x = np.linspace(0, 1, 10)
    with pytest.raises(Exception, match="precondition"):
        cssd(x, np.sin(x), p=0.99, gamma=1e10, precondition="bogus")


def test_partition_invariance_uniform():
    x = np.linspace(0, 1, 30)
    y = np.where(x < 0.5, 0.0, 1.0)
    for gamma in [1e-3, 0.1, 1.0, 100.0]:
        off = cssd(x, y, p=0.99, gamma=gamma, precondition="none")
        on = cssd(x, y, p=0.99, gamma=gamma, precondition="local")
        np.testing.assert_array_equal(off.partition, on.partition)


def test_partition_invariance_nonuniform():
    x, y = _nonuniform_mesh(mesh_ratio=200.0)
    for gamma in [1e-4, 0.1, 100.0]:
        off = cssd(x, y, p=0.99, gamma=gamma, precondition="none")
        on = cssd(x, y, p=0.99, gamma=gamma, precondition="local")
        np.testing.assert_array_equal(off.partition, on.partition)


def test_F_values_agree():
    """In exact arithmetic F is preconditioning-invariant; here we expect
    agreement to ~1e-10 relative on moderately conditioned input."""
    x, y = _nonuniform_mesh(mesh_ratio=10.0)
    off = cssd(x, y, p=0.99, gamma=1e10, precondition="none")
    on = cssd(x, y, p=0.99, gamma=1e10, precondition="local")
    np.testing.assert_allclose(off.F, on.F, rtol=1e-10, atol=1e-12)


def test_pp_coefs_agree():
    x, y = _nonuniform_mesh(mesh_ratio=20.0)
    off = cssd(x, y, p=0.7, gamma=1.0, precondition="none")
    on = cssd(x, y, p=0.7, gamma=1.0, precondition="local")
    np.testing.assert_allclose(off.pp.coefs, on.pp.coefs, rtol=1e-9, atol=1e-10)


def test_tau_is_ones_when_none():
    x = np.linspace(0, 1, 10)
    out = cssd(x, np.sin(x), p=0.99, gamma=1e10, precondition="none")
    np.testing.assert_array_equal(out.tau, np.ones(10))


def test_local_tau_grows_with_smaller_h():
    x = np.array([0.0, 0.001, 0.002, 0.5, 1.0])
    y = np.sin(x)
    out = cssd(x, y, p=0.99, gamma=1e10, precondition="local")
    # Knots at the dense end should have larger tau (since h^{-1} ~ 1000).
    assert out.tau[1] > out.tau[3]
    assert out.tau[1] > out.tau[4]


def test_cssd_cv_accepts_precondition():
    x, y = _nonuniform_mesh(seed=1, mesh_ratio=50.0)
    cv = cssd_cv(x, y, cv_type="random", cv_arg=3, precondition="local", random_state=0)
    assert np.isfinite(cv.cv_score)
    assert cv.fit.precondition == "local"


def test_p_zero_branch_unaffected():
    """The p=0 piecewise-linear branch doesn't use the Hermite QR — the
    preconditioning option should be silently ignored there."""
    x, y = _nonuniform_mesh(mesh_ratio=10.0)
    off = cssd(x, y, p=0.0, gamma=0.1, precondition="none")
    on = cssd(x, y, p=0.0, gamma=0.1, precondition="local")
    np.testing.assert_array_equal(off.partition, on.partition)
