"""Tests for Neumann-Zagier matrix construction."""
import numpy as np
import pytest


def test_symplectic_and_inverse(nz_m004):
    nz = nz_m004
    assert nz.is_symplectic()
    product = nz.g_NZ @ nz.g_NZ_inv()
    np.testing.assert_array_almost_equal(product, np.eye(2 * nz.n), decimal=9)


def test_symplectic_nonunit_smith(nz_v0901):
    """v0901 has Smith invariant factors > 1 — g_NZ must still be symplectic."""
    nz = nz_v0901
    assert nz.is_symplectic()
    det = np.linalg.det(nz.g_NZ)
    assert abs(abs(det) - 1) < 1e-6
    product = nz.g_NZ @ nz.g_NZ_inv()
    np.testing.assert_array_almost_equal(product, np.eye(2 * nz.n), decimal=9)


def test_affine_shift_dimensions(nz_m004):
    nz = nz_m004
    assert nz.nu_x.shape == (nz.n,)
    assert nz.nu_p.shape == (nz.n,)
    assert nz.g_NZ.shape == (2 * nz.n, 2 * nz.n)


def test_inv_scaled_integrality(nz_m004):
    S, scaled = nz_m004.g_NZ_inv_scaled()
    assert isinstance(S, int) and S >= 1
    assert scaled.dtype == np.int64


def test_num_hard_easy(nz_m004):
    nz = nz_m004
    assert nz.num_hard + nz.num_easy == nz.n - nz.r


def test_cusp_basis_change(nz_m004):
    from manifold_index.core.neumann_zagier import apply_cusp_basis_change
    nz2 = apply_cusp_basis_change(nz_m004, 0, 1, 0)
    assert nz2.is_symplectic()


def test_cusp_basis_change_even_p_raises(nz_m004):
    from manifold_index.core.neumann_zagier import apply_cusp_basis_change
    with pytest.raises(ValueError, match="even"):
        apply_cusp_basis_change(nz_m004, 0, 2, 1)
