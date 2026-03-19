"""tests/test_basis_selection.py — Smoke tests for basis selection."""

from fractions import Fraction
import pytest
from manifold_index.core.basis_selection import (
    BasisSelection, CycleChoice, make_basis_selection,
)


def test_zero_zero_raises():
    with pytest.raises(ValueError, match="not a valid cycle"):
        CycleChoice(cusp_idx=0, P=0, Q=0)


def test_make_basis_selection_default_M():
    class _Stub:
        r = 2
    bs = make_basis_selection(_Stub(), [], [None, None], default="M")
    assert bs.m_ext == [1, 1]
    assert bs.e_ext == [Fraction(0), Fraction(0)]


def test_make_basis_selection_explicit():
    class _Stub:
        r = 1
    bs = make_basis_selection(_Stub(), [], [(2, 3)])
    assert bs.m_ext == [2]
    assert bs.e_ext == [Fraction(3, 2)]


def test_snapy_integration():
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("m004")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    bs = make_basis_selection(nz, [], [(1, 2)])
    assert bs.m_ext == [1]
    assert bs.e_ext == [Fraction(1)]
