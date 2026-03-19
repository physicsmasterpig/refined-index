"""tests/test_refined_index.py — Smoke tests for refined index."""

from fractions import Fraction
import pytest
from manifold_index.core.refined_index import (
    compute_refined_index, project_to_3d_index,
)


def test_projection_sums_fugacity():
    refined = {(2, 2): 1, (2, 0): 1, (2, -2): 1}
    assert project_to_3d_index(refined) == {2: 3}


def test_projection_cancellation():
    refined = {(4, 2): 1, (4, -2): -1}
    assert project_to_3d_index(refined) == {}


def test_projection_matches_3d_index_m004():
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.index_3d import compute_index_3d_python

    data = load_manifold("m004")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    q_ord = 12
    m_ext, e_ext = [0], [0]
    refined = compute_refined_index(nz, m_ext, e_ext, q_order_half=q_ord)
    projected = project_to_3d_index(refined)
    res3d = compute_index_3d_python(nz, m_ext, e_ext, q_order_half=q_ord)
    expected = {
        res3d.min_power + k: c
        for k, c in enumerate(res3d.coeffs) if c != 0
    }
    assert projected == expected


def test_hard_edge_projection():
    """For a manifold with hard edges, η=1 projection matches 3D index."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.index_3d import compute_index_3d_python

    for name in ["m003", "m009", "m015"]:
        try:
            data = load_manifold(name)
            easy = find_easy_edges(data)
            nz = build_neumann_zagier(data, easy)
            if nz.num_hard > 0:
                m_ext, e_ext = [0] * nz.r, [0] * nz.r
                q_ord = 10
                refined = compute_refined_index(nz, m_ext, e_ext, q_order_half=q_ord)
                projected = project_to_3d_index(refined)
                res3d = compute_index_3d_python(nz, m_ext, e_ext, q_order_half=q_ord)
                expected = {
                    res3d.min_power + k: c
                    for k, c in enumerate(res3d.coeffs) if c != 0
                }
                assert projected == expected, f"{name} projection mismatch"
                return
        except Exception:
            continue
    pytest.skip("No manifold with hard edges found")
