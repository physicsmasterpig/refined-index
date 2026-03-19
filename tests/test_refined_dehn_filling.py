"""tests/test_refined_dehn_filling.py — Smoke tests for refined Dehn filling."""

from fractions import Fraction
import pytest

from manifold_index.core.refined_dehn_filling import (
    FilledRefinedResult, QEtaSeries,
    _is_kernel, hj_continued_fraction,
)


def test_hj_cf_recovery():
    """HJ continued fraction reconstructs the original fraction."""
    def recover(ks):
        if len(ks) == 1:
            return Fraction(ks[0])
        x = Fraction(ks[-1])
        for k in reversed(ks[:-1]):
            x = k - Fraction(1, x)
        return x

    for P, Q in [(1, 2), (5, 2), (3, 4), (7, 5), (1, 3), (2, 3)]:
        ks = hj_continued_fraction(P, Q)
        assert recover(ks) == Fraction(P, Q), f"CF for {P}/{Q} doesn't recover"


def test_is_kernel_eta1_identity():
    """IS(0,0,0,0;η=1) has qq^0 = 1."""
    result = _is_kernel(0, Fraction(0), 0, Fraction(0), qq_order=20, eta_order=10)
    qq0_sum = sum(c for (qq_p, _), c in result.items() if qq_p == 0)
    assert qq0_sum == 1


def test_is_kernel_stable_region():
    """IS(0,0,0,0;η=1): q>0 terms cancel in stable region."""
    qq_order, eta_order = 20, 5
    result = _is_kernel(0, Fraction(0), 0, Fraction(0),
                        qq_order=qq_order, eta_order=eta_order)
    stable_cutoff = qq_order - 2 * eta_order
    eta1 = {}
    for (qq_p, _), c in result.items():
        if qq_p <= stable_cutoff:
            eta1[qq_p] = eta1.get(qq_p, Fraction(0)) + c
    nonzero = {k: v for k, v in eta1.items() if v != 0}
    assert set(nonzero.keys()) <= {0}
    assert nonzero.get(0, Fraction(0)) == 1


@pytest.fixture
def m003_nz():
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("m003")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)


def test_ell1_matches_unrefined(m003_nz):
    from manifold_index.core.dehn_filling import compute_filled_index
    from manifold_index.core.refined_dehn_filling import compute_filled_refined_index
    P, Q = 5, 1
    unrefined = compute_filled_index(m003_nz, 0, P, Q, q_order_half=10)
    refined = compute_filled_refined_index(m003_nz, 0, P, Q, q_order_half=10, eta_order=0)
    assert not refined.has_cusp_eta
    eta1 = refined.eta1_series()
    for qq_p, c_unref in unrefined.series.items():
        assert eta1.get(qq_p, Fraction(0)) == c_unref, f"Mismatch at qq^{qq_p}"


@pytest.mark.xfail(
    reason="IS kernel truncation artefacts prevent exact η=1 recovery for ℓ=2",
    strict=False,
)
def test_ell2_eta1_matches_unrefined(m003_nz):
    from manifold_index.core.dehn_filling import compute_filled_index
    from manifold_index.core.refined_dehn_filling import compute_filled_refined_index
    P, Q = 1, 2
    q_order = 8
    unrefined = compute_filled_index(m003_nz, 0, P, Q, q_order_half=q_order)
    refined = compute_filled_refined_index(m003_nz, 0, P, Q, q_order_half=q_order, eta_order=6)
    eta1 = refined.eta1_series()
    stable_cutoff = q_order - 4
    for qq_p in range(0, stable_cutoff + 1):
        c_ref = eta1.get(qq_p, Fraction(0))
        c_unref = unrefined.series.get(qq_p, Fraction(0))
        assert c_ref == c_unref, f"At qq^{qq_p}: {c_ref} vs {c_unref}"


def test_hj_ks_stored(m003_nz):
    from manifold_index.core.refined_dehn_filling import compute_filled_refined_index
    for P, Q, expected in [(1, 2, [1, 2]), (3, 2, [2, 2]), (1, 1, [1])]:
        result = compute_filled_refined_index(m003_nz, 0, P, Q, q_order_half=4, eta_order=2)
        assert result.hj_ks == expected, f"Slope {P}/{Q}: {result.hj_ks} != {expected}"
