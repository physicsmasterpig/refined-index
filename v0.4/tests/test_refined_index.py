"""Tests for refined index."""
from manifold_index.core.refined_index import (
    compute_refined_index, project_to_3d_index,
)


def test_projection_arithmetic():
    assert project_to_3d_index({(2, 2): 1, (2, 0): 1, (2, -2): 1}) == {2: 3}
    assert project_to_3d_index({(4, 2): 1, (4, -2): -1}) == {}


def test_projection_matches_3d_index(nz_m004):
    from manifold_index.core.index_3d import compute_index_3d_python
    q_ord = 12
    refined = compute_refined_index(nz_m004, [0], [0], q_order_half=q_ord)
    projected = project_to_3d_index(refined)
    res3d = compute_index_3d_python(nz_m004, [0], [0], q_order_half=q_ord)
    expected = {res3d.min_power + k: c for k, c in enumerate(res3d.coeffs) if c != 0}
    assert projected == expected
