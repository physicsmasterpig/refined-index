"""Tests for Neumann-Zagier matrix."""

import numpy as np


def test_symplectic_and_inverse(nz_m004):
    nz = nz_m004
    assert nz.is_symplectic()
    product = nz.g_NZ @ nz.g_NZ_inv()
    np.testing.assert_array_almost_equal(product, np.eye(2 * nz.n), decimal=9)
