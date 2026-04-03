"""Tests for reduced gluing equations."""
import pytest
import numpy as np
from manifold_index.core.manifold import load_manifold
from manifold_index.core.gluing_equations import reduce_gluing_equations


def test_shapes_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    rd = reduce_gluing_equations(data)
    n, r = 2, 1
    assert rd.edge_coeffs.shape == (n, 2 * n)
    assert rd.edge_consts.shape == (n,)
    assert rd.cusp_coeffs.shape == (2 * r, 2 * n)
    assert rd.cusp_consts.shape == (2 * r,)


def test_edge_rank_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    rd = reduce_gluing_equations(data)
    assert np.linalg.matrix_rank(rd.edge_coeffs) == data.num_tetrahedra - data.num_cusps


def test_independent_edge_count():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    rd = reduce_gluing_equations(data)
    assert len(rd.independent_edge_indices) == data.num_tetrahedra - data.num_cusps


def test_commutator_meridian_longitude():
    """[μ, λ] = 2 for every cusp."""
    pytest.importorskip("snappy")
    for name in ["m004", "m003"]:
        data = load_manifold(name)
        rd = reduce_gluing_equations(data)
        for k in range(data.num_cusps):
            mu = rd.meridian_coeffs(k)
            lam = rd.longitude_coeffs(k)
            assert rd.commutator(mu, lam) == 2, f"{name} cusp {k}: [μ,λ] ≠ 2"


def test_symplectic_matrix_structure():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    rd = reduce_gluing_equations(data)
    omega = rd.symplectic_matrix
    assert np.array_equal(omega, -omega.T)
    for i in range(data.num_tetrahedra):
        assert omega[2 * i, 2 * i + 1] == 1
        assert omega[2 * i + 1, 2 * i] == -1


def test_independent_edge_coeffs_shape():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    rd = reduce_gluing_equations(data)
    n_int = data.num_tetrahedra - data.num_cusps
    assert rd.independent_edge_coeffs.shape == (n_int, 2 * data.num_tetrahedra)
