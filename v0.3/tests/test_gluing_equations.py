"""Tests for reduced gluing equations."""

import pytest
import numpy as np
from manifold_index.core.manifold import load_manifold
from manifold_index.core.gluing_equations import reduce_gluing_equations


def test_edge_rank_and_commutators():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    rd = reduce_gluing_equations(data)
    n, r = data.num_tetrahedra, data.num_cusps
    assert np.linalg.matrix_rank(rd.edge_coeffs) == n - r
    mu = rd.meridian_coeffs(0)
    lam = rd.longitude_coeffs(0)
    assert rd.commutator(mu, lam) == 2
