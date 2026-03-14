"""
tests/test_gluing_equations.py — Tests for reduced gluing equations (Step 2 post-processing).
"""

import pytest
import numpy as np
from manifold_index.core.manifold import load_manifold
from manifold_index.core.gluing_equations import reduce_gluing_equations, ReducedGluingData


@pytest.fixture(params=["m004", "s776", "t12047", "m125", "s000"])
def reduced(request):
    pytest.importorskip("snappy")
    data = load_manifold(request.param)
    return data, reduce_gluing_equations(data)


def test_reduced_shapes(reduced):
    """Coefficient matrices have the right shapes after Z_i' substitution."""
    data, rd = reduced
    n, r = data.num_tetrahedra, data.num_cusps
    assert rd.edge_coeffs.shape  == (n, 2 * n)
    assert rd.edge_consts.shape  == (n,)
    assert rd.cusp_coeffs.shape  == (2 * r, 2 * n)
    assert rd.cusp_consts.shape  == (2 * r,)


def test_edge_rank_equals_n_minus_r(reduced):
    """
    The rank of the reduced edge coefficient matrix must equal n - r.
    See SPEC.md §Step 2 / Gluing equations structure.
    """
    data, rd = reduced
    n, r = data.num_tetrahedra, data.num_cusps
    rank = np.linalg.matrix_rank(rd.edge_coeffs)
    assert rank == n - r


def test_independent_edge_basis_count(reduced):
    """independent_edge_indices has exactly n - r entries."""
    data, rd = reduced
    n, r = data.num_tetrahedra, data.num_cusps
    assert len(rd.independent_edge_indices) == n - r


def test_independent_edge_basis_is_independent(reduced):
    """The selected rows are indeed linearly independent."""
    data, rd = reduced
    n, r = data.num_tetrahedra, data.num_cusps
    basis = rd.independent_edge_coeffs
    assert np.linalg.matrix_rank(basis) == n - r


def test_symplectic_matrix_structure(reduced):
    """
    Ω[2i, 2i+1] = +1, Ω[2i+1, 2i] = -1, all others 0.
    Encodes [Z_i, Z_i''] = 1.
    """
    data, rd = reduced
    n = data.num_tetrahedra
    omega = rd.symplectic_matrix
    assert omega.shape == (2 * n, 2 * n)
    for i in range(n):
        assert omega[2*i,   2*i+1] == +1
        assert omega[2*i+1, 2*i  ] == -1
        # off-diagonal blocks between different tets are zero
        for j in range(n):
            if i != j:
                assert omega[2*i,   2*j  ] == 0
                assert omega[2*i,   2*j+1] == 0
                assert omega[2*i+1, 2*j  ] == 0
                assert omega[2*i+1, 2*j+1] == 0


def test_meridian_longitude_commutator(reduced):
    """
    [M_k, L_k] = 2  for each cusp k.
    Edge equations must commute with M_k and L_k: [E_j, M_k] = [E_j, L_k] = 0.
    See SPEC.md §Step 2 commutation relations.
    """
    data, rd = reduced
    n, r = data.num_tetrahedra, data.num_cusps

    for k in range(r):
        mu  = rd.meridian_coeffs(k)
        lam = rd.longitude_coeffs(k)

        # [M, L] = 2
        assert rd.commutator(mu, lam) == 2, \
            f"[M_{k}, L_{k}] = {rd.commutator(mu, lam)}, expected 2"

        # Each independent edge commutes with M and L
        for j, edge in enumerate(rd.independent_edge_coeffs):
            assert rd.commutator(edge, mu)  == 0, \
                f"[edge_{j}, M_{k}] != 0"
            assert rd.commutator(edge, lam) == 0, \
                f"[edge_{j}, L_{k}] != 0"

    # Edges also commute with each other
    for i, ei in enumerate(rd.independent_edge_coeffs):
        for j, ej in enumerate(rd.independent_edge_coeffs):
            assert rd.commutator(ei, ej) == 0, \
                f"[edge_{i}, edge_{j}] != 0"
