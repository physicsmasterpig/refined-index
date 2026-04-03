"""Tests for manifold loading."""
import pytest
from manifold_index.core.manifold import load_manifold, ManifoldData


def test_load_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    assert isinstance(data, ManifoldData)
    assert data.num_tetrahedra == 2
    assert data.num_cusps == 1
    assert data.gluing_matrix.shape == (4, 6)


def test_load_m003():
    pytest.importorskip("snappy")
    data = load_manifold("m003")
    assert data.num_tetrahedra == 2
    assert data.num_cusps == 1


def test_edge_equations_shape():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    assert data.edge_equations.shape == (2, 6)


def test_cusp_equations_shape():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    mu, lam = data.cusp_equations(0)
    assert mu.shape == (6,)
    assert lam.shape == (6,)


def test_meridian_longitude_shapes():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    assert data.meridian_equations.shape == (1, 6)
    assert data.longitude_equations.shape == (1, 6)


def test_load_unknown_raises():
    pytest.importorskip("snappy")
    with pytest.raises(ValueError):
        load_manifold("not_a_real_manifold_xyz")
