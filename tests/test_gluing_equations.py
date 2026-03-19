"""tests/test_gluing_equations.py — Smoke tests for reduced gluing equations."""

import pytest
import numpy as np
from manifold_index.core.manifold import load_manifold
from manifold_index.core.gluing_equations import reduce_gluing_equations


@pytest.fixture(scope="module")
def reduced_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    return data, reduce_gluing_equations(data)


def test_edge_rank_equals_n_minus_r(reduced_m004):
    data, rd = reduced_m004
    n, r = data.num_tetrahedra, data.num_cusps
    rank = np.linalg.matrix_rank(rd.edge_coeffs)
    assert rank == n - r


def test_meridian_longitude_commutator(reduced_m004):
    data, rd = reduced_m004
    n, r = data.num_tetrahedra, data.num_cusps
    for k in range(r):
        mu = rd.meridian_coeffs(k)
        lam = rd.longitude_coeffs(k)
        assert rd.commutator(mu, lam) == 2
        for edge in rd.independent_edge_coeffs:
            assert rd.commutator(edge, mu) == 0
            assert rd.commutator(edge, lam) == 0
