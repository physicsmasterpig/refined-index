"""Tests for easy-edge detection."""
import numpy as np
import pytest
from manifold_index.core.phase_space import _is_easy, find_easy_edges, EasyEdgeResult
from manifold_index.core.manifold import load_manifold


def test_is_easy_true():
    assert _is_easy(np.array([2, 0, 0, 0, 3, 0]), n=2)


def test_is_easy_false():
    assert not _is_easy(np.array([2, 1, 0, 0, 0, 0]), n=2)


def test_find_easy_edges_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    result = find_easy_edges(data)
    assert isinstance(result, EasyEdgeResult)
    assert len(result.basis_edges) == data.num_tetrahedra - data.num_cusps


def test_find_easy_edges_m003():
    # m003's single internal edge [2,0,1,2,0,1] has two nonzero entries per tet
    # so it is NOT easy; it ends up in hard_padding.
    pytest.importorskip("snappy")
    data = load_manifold("m003")
    result = find_easy_edges(data)
    assert len(result.basis_edges) == data.num_tetrahedra - data.num_cusps
    assert len(result.hard_padding) == 1
    assert len(result.all_easy) == 0


def test_easy_edges_are_easy():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    result = find_easy_edges(data)
    for edge in result.all_easy:
        assert _is_easy(edge, data.num_tetrahedra)


def test_basis_edges_independent():
    """Basis edges must be linearly independent in the reduced space."""
    pytest.importorskip("snappy")
    from manifold_index.core.gluing_equations import _reduce_row
    data = load_manifold("m004")
    result = find_easy_edges(data)
    n = data.num_tetrahedra
    reduced_vecs = []
    for edge in result.basis_edges:
        _, rv = _reduce_row(edge, n)
        reduced_vecs.append(rv)
    if reduced_vecs:
        mat = np.array(reduced_vecs, dtype=float)
        assert np.linalg.matrix_rank(mat) == len(reduced_vecs)
