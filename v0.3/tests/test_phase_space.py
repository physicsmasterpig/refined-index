"""Tests for easy-edge detection."""

import numpy as np
import pytest
from manifold_index.core.phase_space import _is_easy, find_easy_edges, EasyEdgeResult
from manifold_index.core.manifold import load_manifold


def test_is_easy():
    assert _is_easy(np.array([2, 0, 0, 0, 3, 0]), n=2)
    assert not _is_easy(np.array([2, 1, 0, 0, 0, 0]), n=2)


def test_find_easy_edges_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    result = find_easy_edges(data)
    assert isinstance(result, EasyEdgeResult)
    assert len(result.basis_edges) == data.num_tetrahedra - data.num_cusps
