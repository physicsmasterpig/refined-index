"""tests/test_weyl_check.py — Smoke tests for Weyl symmetry checks."""

from fractions import Fraction
import pytest

from manifold_index.core.weyl_check import (
    ABVectors, check_adjoint_character, check_weyl_symmetry,
    compute_ab_vectors, compute_ab_vectors_for_cusp,
    extract_leading_eta_exponents, run_weyl_checks,
)
from manifold_index.core.refined_index import RefinedIndexResult


def test_extract_leading_single_term():
    result = {(4, 2): 3}
    eta = extract_leading_eta_exponents(result, 1)
    assert eta == [Fraction(1)]


def test_compute_ab_combined():
    """b=-1/2, a=-1/2 from meridian+longitude pairs."""
    def _e(m, e, res):
        return ([m], [Fraction(e)], res)

    entries = [
        _e(+2, 0, {(0, 2): 1}),
        _e(-2, 0, {(0, -2): 1}),
        _e(0, +1, {(0, 1): 1}),
        _e(0, -1, {(0, -1): 1}),
    ]
    ab = compute_ab_vectors(entries, 1)
    assert ab is not None
    assert ab.a == [Fraction(-1, 2)]
    assert ab.b == [Fraction(-1, 2)]
    # a = -1/2 is not integer → not compatible
    assert ab.is_valid is False


def test_weyl_symmetry_adjoint():
    """η^{-1} + 1 + η at q^0 is Weyl-symmetric."""
    result = {(0, -2): 1, (0, 0): 1, (0, 2): 1}
    ab = ABVectors(a=[Fraction(0)], b=[Fraction(0)], num_hard=1)
    entries = [([0], [Fraction(0)], result)]
    res = check_weyl_symmetry(entries, 1, ab)
    assert res[((0,), (Fraction(0),))] is True


def test_adjoint_character_exact():
    result = {(2, -2): 1, (2, 0): 1, (2, 2): 1}
    leading = [Fraction(0)]
    assert check_adjoint_character(result, leading, 1, 0) is True


def test_m004_ab_vectors():
    """m004 integration: (a, b) should be valid."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.refined_index import compute_refined_index

    data = load_manifold("m004")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    num_hard = nz.num_hard
    assert num_hard >= 1

    entries = []
    for m in [-2, -1, 0, 1, 2]:
        for e in [-1, 0, 1]:
            result = compute_refined_index(nz, [m], [Fraction(e)], q_order_half=20)
            entries.append(([m], [Fraction(e)], result))

    ab = compute_ab_vectors(entries, num_hard)
    assert ab is not None
    # m004: a = 1/2 (not integer), b = -1/2 (half-integer)
    assert ab.b_is_half_integer == [True] * num_hard
    assert ab.a_is_integer == [False] * num_hard
    # Not compatible: a ∉ ℤ
    assert ab.is_valid is False


# ── compute_ab_vectors_for_cusp tests ─────────────────────────────────


def test_ab_for_cusp_m004():
    """m004 (1-cusp): per-cusp extraction should match global extraction."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier

    data = load_manifold("m004")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    num_hard = nz.num_hard
    assert num_hard >= 1

    ab = compute_ab_vectors_for_cusp(nz, cusp_idx=0, q_order_half=20)
    assert ab is not None
    assert ab.num_hard == num_hard
    # m004: a = 1/2, b = -1/2 → not compatible (a ∉ ℤ)
    assert ab.a == [Fraction(1, 2)]
    assert ab.b == [Fraction(-1, 2)]
    assert ab.is_valid is False
    assert ab.edge_compatible == [False]


def test_ab_for_cusp_matches_global_1cusp():
    """For 1-cusp, per-cusp extraction should agree with global compute_ab_vectors."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.refined_index import compute_refined_index

    data = load_manifold("m004")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    num_hard = nz.num_hard

    # Global extraction (old-style, summing over all cusps — same for 1 cusp)
    entries = []
    for m in [-2, -1, 0, 1, 2]:
        for e_half in [-2, -1, 0, 1, 2]:
            e = Fraction(e_half, 2)
            result = compute_refined_index(nz, [m], [e], q_order_half=20)
            entries.append(([m], [e], result))
    ab_global = compute_ab_vectors(entries, num_hard)

    # Per-cusp extraction (new function)
    ab_cusp = compute_ab_vectors_for_cusp(nz, cusp_idx=0, q_order_half=20)

    assert ab_global is not None
    assert ab_cusp is not None
    assert ab_global.a == ab_cusp.a
    assert ab_global.b == ab_cusp.b


def test_ab_for_cusp_after_basis_change():
    """Weyl vectors can be extracted after NC basis change."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import (
        build_neumann_zagier, apply_general_cusp_basis_change,
    )

    data = load_manifold("m004")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)

    # Apply an SL(2,Z) basis change: [[2,1],[1,1]] (det = 1)
    nz_nc = apply_general_cusp_basis_change(nz, cusp_idx=0, a=2, b=1, c=1, d=1)

    ab_nc = compute_ab_vectors_for_cusp(nz_nc, cusp_idx=0, q_order_half=20)
    assert ab_nc is not None
    assert ab_nc.num_hard == nz.num_hard
    # Note: is_valid (a ∈ Z, b ∈ Z/2) is basis-dependent and may not hold
    # for arbitrary SL(2,Z) transforms.  The extraction itself should succeed.


def test_ab_for_cusp_zero_hard():
    """With num_hard=0, should return empty vectors."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_phase_space_basis
    from manifold_index.core.neumann_zagier import build_neumann_zagier

    data = load_manifold("m003")
    ps = find_phase_space_basis(data)
    nz = build_neumann_zagier(data, ps)

    if nz.num_hard == 0:
        ab = compute_ab_vectors_for_cusp(nz, cusp_idx=0, q_order_half=10)
        assert ab is not None
        assert ab.a == []
        assert ab.b == []


def test_edge_compatible_m004():
    """m004: a_stored=1 (odd) → edge 0 incompatible, zeroed by make_filling_compatible."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier

    data = load_manifold("m004")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)

    ab = compute_ab_vectors_for_cusp(nz, cusp_idx=0, q_order_half=20)
    assert ab is not None
    # a = 1/2 ∉ ℤ → incompatible
    assert ab.edge_compatible == [False]

    ab_c = ab.make_filling_compatible()
    assert ab_c.a == [Fraction(0)]
    assert ab_c.b == [Fraction(0)]


def test_edge_compatible_m003():
    """m003 with easy-edge basis has a=1 (integer) → compatible."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier

    data = load_manifold("m003")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    if nz.num_hard == 0:
        pytest.skip("m003 has no hard edges in this basis")

    ab = compute_ab_vectors_for_cusp(nz, cusp_idx=0, q_order_half=15)
    assert ab is not None
    assert ab.edge_compatible == [True]

    ab_c = ab.make_filling_compatible()
    assert ab_c.a == ab.a  # unchanged
    assert ab_c.b == ab.b
