"""Tests for Dehn filling."""

from fractions import Fraction
from manifold_index.core.dehn_filling import (
    KernelTerm, _apply_kernel, _particular_solution, find_rs,
)


def test_find_rs():
    R, S = find_rs(3, 2)
    assert R * 2 - 3 * S == 1


def test_particular_solution():
    m0, e0 = _particular_solution(3, 2, 2)
    assert 3 * m0 + 2 * 2 * e0 == 2


def test_apply_kernel_identity():
    term = KernelTerm(m=0, e=Fraction(0), c=0, phase=0)
    s = {0: Fraction(1), 2: Fraction(3)}
    assert _apply_kernel(term, s) == s
