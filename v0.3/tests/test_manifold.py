"""Tests for manifold loading."""

import pytest
from manifold_index.core.manifold import load_manifold, ManifoldData


def test_load_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    assert isinstance(data, ManifoldData)
    assert data.num_tetrahedra == 2
    assert data.num_cusps == 1


def test_load_unknown_raises():
    pytest.importorskip("snappy")
    with pytest.raises(ValueError):
        load_manifold("not_a_real_manifold_xyz")
