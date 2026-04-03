"""Tests for basis selection."""
from fractions import Fraction
import pytest
from manifold_index.core.basis_selection import (
    CycleChoice, BasisSelection, make_basis_selection,
    default_meridian_choice, default_longitude_choice,
)


def test_zero_zero_raises():
    with pytest.raises(ValueError):
        CycleChoice(cusp_idx=0, P=0, Q=0)


def test_non_primitive_raises():
    with pytest.raises(ValueError):
        CycleChoice(cusp_idx=0, P=2, Q=4)


def test_cycle_choice_properties():
    cc = CycleChoice(cusp_idx=0, P=3, Q=2)
    assert cc.m == 3
    assert cc.e == Fraction(1)  # 2/2
    assert cc.slope_str == "3/2"


def test_make_basis_selection_with_stub():
    class _Stub:
        r = 1
    bs = make_basis_selection(_Stub(), [], [(2, 3)])
    assert bs.m_ext == [2]
    assert bs.e_ext == [Fraction(3, 2)]


def test_default_choices():
    m_ch = default_meridian_choice(0)
    assert m_ch.m == 1 and m_ch.e == Fraction(0) and m_ch.is_default
    l_ch = default_longitude_choice(0)
    assert l_ch.m == 0 and l_ch.e == Fraction(1, 2) and l_ch.is_default


def test_basis_selection_ordering():
    """choices must have cusp_idx == i."""
    with pytest.raises(ValueError):
        BasisSelection(choices=[CycleChoice(cusp_idx=1, P=1, Q=0)])


def test_basis_selection_m_ext_e_ext():
    bs = BasisSelection(choices=[
        CycleChoice(0, P=1, Q=2),
        CycleChoice(1, P=3, Q=2),
    ])
    assert bs.m_ext == [1, 3]
    assert bs.e_ext == [Fraction(1), Fraction(1)]
    assert bs.r == 2
