"""tests/test_phase_space.py — Smoke tests for easy-edge detection."""

import numpy as np
import pytest
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import _is_easy, find_easy_edges, EasyEdgeResult


def test_single_nonzero_per_tet_is_easy():
    assert _is_easy(np.array([2, 0, 0, 0, 3, 0]), n=2)


def test_two_nonzero_in_same_tet_is_hard():
    assert not _is_easy(np.array([2, 1, 0, 0, 0, 0]), n=2)


def test_find_easy_edges_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    result = find_easy_edges(data)
    assert isinstance(result, EasyEdgeResult)
    n, r = data.num_tetrahedra, data.num_cusps
    assert len(result.basis_edges) == n - r
    for edge in result.all_easy:
        assert _is_easy(edge, n)
