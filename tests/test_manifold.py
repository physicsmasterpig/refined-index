"""tests/test_manifold.py — Smoke tests for manifold loading."""

import pytest
import numpy as np
from manifold_index.core.manifold import load_manifold, ManifoldData


def test_load_m004():
    """Load m004 and check basic structure."""
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    assert isinstance(data, ManifoldData)
    assert data.num_tetrahedra == 2
    assert data.num_cusps == 1
    n, r = data.num_tetrahedra, data.num_cusps
    assert data.gluing_matrix.shape == (n + 2 * r, 3 * n)
    assert data.edge_equations.shape == (n, 3 * n)
    assert data.meridian_equations.shape == (r, 3 * n)
    assert data.longitude_equations.shape == (r, 3 * n)


def test_load_unknown_raises():
    """Loading unknown manifold raises ValueError."""
    pytest.importorskip("snappy")
    with pytest.raises(ValueError, match="could not load"):
        load_manifold("not_a_real_manifold_xyz")
