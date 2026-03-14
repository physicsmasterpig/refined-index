"""
tests/test_manifold.py — Tests for manifold loading (Step 2).
"""

import pytest
import numpy as np
from manifold_index.core.manifold import load_manifold, ManifoldData


def test_load_manifold_known():
    """SnaPy should successfully load a well-known manifold."""
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    assert isinstance(data, ManifoldData)
    assert data.num_cusps >= 1
    assert data.num_tetrahedra >= 1
    assert data.gluing_matrix is not None


def test_load_manifold_unknown():
    """Loading an unknown manifold name should raise ValueError."""
    pytest.importorskip("snappy")
    with pytest.raises(ValueError, match="could not load"):
        load_manifold("not_a_real_manifold_xyz")


@pytest.mark.parametrize("name,expected_n,expected_r", [
    ("m004",   2, 1),
    ("s776",   6, 3),
    ("t12047", 8, 4),
])
def test_gluing_matrix_shape(name, expected_n, expected_r):
    """
    Gluing matrix must be (n + 2r) x (3n) for all manifolds.
    See SPEC.md §Step 2.
    """
    pytest.importorskip("snappy")
    data = load_manifold(name)
    n, r = data.num_tetrahedra, data.num_cusps
    assert n == expected_n
    assert r == expected_r
    assert data.gluing_matrix.shape == (n + 2 * r, 3 * n)
    assert data.gluing_matrix.dtype == np.dtype("int")


@pytest.mark.parametrize("name,expected_n,expected_r", [
    ("m004",   2, 1),
    ("s776",   6, 3),
])
def test_equation_slice_properties(name, expected_n, expected_r):
    """Edge / meridian / longitude slices have the right shapes."""
    pytest.importorskip("snappy")
    data = load_manifold(name)
    n, r = data.num_tetrahedra, data.num_cusps
    assert data.edge_equations.shape      == (n, 3 * n)
    assert data.meridian_equations.shape  == (r, 3 * n)
    assert data.longitude_equations.shape == (r, 3 * n)


@pytest.mark.parametrize("name,expected_n,expected_r", [
    ("m004",   2, 1),
    ("s776",   6, 3),
    ("t12047", 8, 4),
])
def test_cusp_interleaved_ordering(name, expected_n, expected_r):
    """
    Cusp rows are interleaved: row n+2k = meridian k, row n+2k+1 = longitude k.
    Verify that cusp_equations(k) matches the correct rows of the raw matrix.
    See SPEC.md §Step 2.
    """
    pytest.importorskip("snappy")
    data = load_manifold(name)
    n, r = data.num_tetrahedra, data.num_cusps
    for k in range(r):
        mu, lam = data.cusp_equations(k)
        assert np.array_equal(mu,  data.gluing_matrix[n + 2 * k])
        assert np.array_equal(lam, data.gluing_matrix[n + 2 * k + 1])
