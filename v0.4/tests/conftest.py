"""Shared fixtures for the manifold-index test suite."""
import pytest


@pytest.fixture(scope="session")
def nz_m004():
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("m004")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)


@pytest.fixture(scope="session")
def nz_m003():
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("m003")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)


@pytest.fixture(scope="session")
def nz_v0901():
    """v0901 (7 tet, 1 cusp) — has non-unit Smith invariant factors."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("v0901")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)
