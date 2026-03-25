"""Shared fixtures for the manifold-index test suite."""

import pytest


@pytest.fixture(scope="session")
def nz_m004():
    """NZ data for m004 (2 tet, 1 cusp, 1 hard edge)."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("m004")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)


@pytest.fixture(scope="session")
def nz_m003():
    """NZ data for m003 (2 tet, 1 cusp)."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("m003")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)
