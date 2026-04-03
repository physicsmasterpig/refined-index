"""Tests for Dehn filling."""
from fractions import Fraction
import pytest
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
    """c=0, phase=0 → K = ½·1·(q⁰+q⁰) = 1, so result = input series."""
    term = KernelTerm(m=0, e=Fraction(0), c=0, phase=0)
    s = {0: Fraction(1), 2: Fraction(3)}
    assert _apply_kernel(term, s) == s


def test_find_rs_various():
    for P, Q in [(1, 2), (5, 3), (7, 4), (1, 0)]:
        if Q == 0:
            R, S = find_rs(P, Q)
            assert R * 0 - P * S == 1  # R·0 - P·S = -P·S = 1 means P=1,S=-1
        else:
            R, S = find_rs(P, Q)
            assert R * Q - P * S == 1, f"RQ-PS≠1 for P={P},Q={Q}"
