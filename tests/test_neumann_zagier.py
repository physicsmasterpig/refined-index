"""Tests for Neumann-Zagier matrix."""

import numpy as np


def test_symplectic_and_inverse(nz_m004):
    nz = nz_m004
    assert nz.is_symplectic()
    product = nz.g_NZ @ nz.g_NZ_inv()
    np.testing.assert_array_almost_equal(product, np.eye(2 * nz.n), decimal=9)


def test_symplectic_nonunit_smith(nz_v0901):
    """v0901 has Smith invariant factors > 1 -- verify g_NZ is still symplectic."""
    nz = nz_v0901
    assert nz.is_symplectic(), "g_NZ for v0901 should be symplectic"
    # det must be +/-1
    det = np.linalg.det(nz.g_NZ)
    assert abs(abs(det) - 1) < 1e-6, f"det(g_NZ) = {det}, expected +/-1"
    # Inverse check
    product = nz.g_NZ @ nz.g_NZ_inv()
    np.testing.assert_array_almost_equal(product, np.eye(2 * nz.n), decimal=9)
