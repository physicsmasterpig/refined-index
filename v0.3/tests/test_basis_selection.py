"""Tests for basis selection."""

from fractions import Fraction
import pytest
from manifold_index.core.basis_selection import CycleChoice, make_basis_selection


def test_zero_zero_raises():
    with pytest.raises(ValueError):
        CycleChoice(cusp_idx=0, P=0, Q=0)


def test_make_basis_selection():
    class _Stub:
        r = 1
    bs = make_basis_selection(_Stub(), [], [(2, 3)])
    assert bs.m_ext == [2]
    assert bs.e_ext == [Fraction(3, 2)]
