"""Tests for refined Dehn filling."""

from fractions import Fraction
from manifold_index.core.refined_dehn_filling import (
    _is_kernel_frac, hj_continued_fraction, compute_filled_refined_index,
)


def test_hj_cf_recovery():
    def recover(ks):
        if len(ks) == 1:
            return Fraction(ks[0])
        x = Fraction(ks[-1])
        for k in reversed(ks[:-1]):
            x = k - Fraction(1, x)
        return x
    for P, Q in [(1, 2), (5, 2), (3, 4), (7, 5)]:
        assert recover(hj_continued_fraction(P, Q)) == Fraction(P, Q)


def test_is_kernel_identity():
    result = _is_kernel_frac(0, Fraction(0), 0, Fraction(0), qq_order=20, eta_order=10)
    assert sum(c for (qq, _), c in result.items() if qq == 0) == 1


def test_ell1_matches_unrefined(nz_m003):
    from manifold_index.core.dehn_filling import compute_filled_index
    unrefined = compute_filled_index(nz_m003, 0, 5, 1, q_order_half=10)
    refined = compute_filled_refined_index(nz_m003, 0, 5, 1, q_order_half=10, eta_order=0)
    eta1 = refined.eta1_series()
    for qq_p, c in unrefined.series.items():
        assert eta1.get(qq_p, Fraction(0)) == c
