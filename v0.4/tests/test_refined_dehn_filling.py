"""Tests for refined_dehn_filling — Phase 10.

Tests cover:
  T10.1 — HJ continued fraction
  T10.2 — ẽI_S integrality
  T10.3 — I_S ×2 scaling
  T10.4 — ℓ=1 filling (m003, P=5, Q=2)
  T10.5 — ℓ≥2 filling (m003, P=1, Q=2)
  T10.6 — LCD consistency
  T10.7 — cache content-key safety
"""
from fractions import Fraction

import pytest


# ---------------------------------------------------------------------------
# T10.1 — Hirzebruch-Jung continued fraction
# ---------------------------------------------------------------------------

def test_hj_cf_basic_cases():
    from manifold_index.core.refined_dehn_filling import hj_continued_fraction

    assert hj_continued_fraction(1, 3) == [0, -3]
    assert hj_continued_fraction(5, 2) == [2, -2]   # shortest form: 5/2 = 2 - 1/(-2)
    assert hj_continued_fraction(1, 1) == [1]
    assert hj_continued_fraction(1, 0) == [0, 0]    # Q=0 special
    assert hj_continued_fraction(-1, 0) == [0, 0]
    assert hj_continued_fraction(4, 3) == [1, -3]


def _eval_hj_cf(ks: list) -> Fraction:
    """Evaluate a HJ-CF [k1, k2, ..., kl] as a Fraction."""
    if len(ks) == 1:
        return Fraction(ks[0])
    # P/Q = k1 - 1/(rest)
    tail = _eval_hj_cf(ks[1:])
    return Fraction(ks[0]) - Fraction(1, 1) / tail


def test_hj_cf_roundtrip():
    from manifold_index.core.refined_dehn_filling import hj_continued_fraction

    cases = [
        (1, 3),
        (5, 2),
        (1, 1),
        (4, 3),
    ]
    for P, Q in cases:
        ks = hj_continued_fraction(P, Q)
        recovered = _eval_hj_cf(ks)
        assert recovered == Fraction(P, Q), (
            f"Roundtrip failed for ({P},{Q}): CF={ks}, got {recovered}"
        )


# ---------------------------------------------------------------------------
# T10.2 — ẽI_S integrality
# ---------------------------------------------------------------------------

def test_etilde_is_integrality():
    from manifold_index.core.refined_dehn_filling import _etilde_is

    result = _etilde_is(0, Fraction(0), 0, Fraction(0), qq_order=10, eta_order=10)
    assert len(result) > 0, "_etilde_is returned empty for (0,0,0,0)"
    for key, val in result.items():
        assert isinstance(val, int), f"Non-integer value {val!r} at key {key}"


# ---------------------------------------------------------------------------
# T10.3 — I_S ×2 scaling
# ---------------------------------------------------------------------------

def test_is_kernel_x2_integral():
    from manifold_index.core.refined_dehn_filling import _is_kernel

    # Test a few (m1, e1, m2, e2) pairs
    test_cases = [
        (0, Fraction(0), 0, Fraction(0)),
        (1, Fraction(0), 0, Fraction(0)),
        (0, Fraction(1), 0, Fraction(0)),
        (2, Fraction(1), 1, Fraction(0)),
    ]
    for m1, e1, m2, e2 in test_cases:
        raw = _is_kernel(m1, e1, m2, e2, qq_order=8, eta_order=8)
        for key, val in raw.items():
            assert isinstance(val, int), (
                f"_is_kernel returned non-int {val!r} for ({m1},{e1},{m2},{e2}) key={key}"
            )


# ---------------------------------------------------------------------------
# Fixtures for manifold-dependent tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nz_m003_bc():
    """m003 with NC cusp basis change applied."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier, apply_cusp_basis_change
    data = load_manifold("m003")
    easy = find_easy_edges(data)
    nz = build_neumann_zagier(data, easy)
    return apply_cusp_basis_change(nz, 0, 1, 0)


# ---------------------------------------------------------------------------
# T10.4 — ℓ=1 filling (m003, P=3, Q=1 → HJ-CF=[3], ℓ=1)
# ---------------------------------------------------------------------------

def test_l1_filling_m003(nz_m003_bc):
    from manifold_index.core.refined_dehn_filling import compute_filled_refined_index, hj_continued_fraction

    # P=3, Q=1 → HJ-CF=[3], length 1 → no IS chain, no cusp η
    assert hj_continued_fraction(3, 1) == [3], "Precondition: (3,1) must be ℓ=1"
    result = compute_filled_refined_index(nz_m003_bc, 0, 3, 1, q_order_half=8)
    assert not result.is_zero, "ℓ=1 filling (3/1) returned zero"
    assert result.has_cusp_eta is False, (
        f"ℓ=1 filling should have no cusp η, got has_cusp_eta={result.has_cusp_eta}"
    )


# ---------------------------------------------------------------------------
# T10.5 — ℓ≥2 filling (m003, P=1, Q=2)
# ---------------------------------------------------------------------------

def test_l2_filling_m003(nz_m003_bc):
    from manifold_index.core.refined_dehn_filling import compute_filled_refined_index

    result = compute_filled_refined_index(nz_m003_bc, 0, 1, 2, q_order_half=8)
    assert result.has_cusp_eta is True, "ℓ≥2 filling should have cusp η"
    assert result.num_cusp_eta == 1, (
        f"Expected num_cusp_eta=1, got {result.num_cusp_eta}"
    )
    # Diamond truncation: all keys satisfy k[0] + |k[-1]| ≤ 8
    for k in result.series:
        assert k[0] + abs(k[-1]) <= 8, (
            f"Diamond truncation violated: key {k}, qq+|eta|={k[0]+abs(k[-1])} > 8"
        )


# ---------------------------------------------------------------------------
# T10.6 — LCD consistency
# ---------------------------------------------------------------------------

def test_lcd_consistency(nz_m003_bc):
    """For ℓ≥2, all final Fraction values have denominator dividing 2^ℓ."""
    from manifold_index.core.refined_dehn_filling import (
        compute_filled_refined_index,
        hj_continued_fraction,
    )

    P, Q = 1, 2
    qq_order = 8
    hj_ks = hj_continued_fraction(P, Q)
    ell = len(hj_ks)
    lcd = 1 << ell  # 2^ℓ

    result = compute_filled_refined_index(nz_m003_bc, 0, P, Q, q_order_half=qq_order)
    for key, val in result.series.items():
        frac = Fraction(val)
        # Numerator should be integer after multiplying by lcd
        assert (frac * lcd).denominator == 1, (
            f"LCD inconsistency at key {key}: {frac} * {lcd} is not integer"
        )


# ---------------------------------------------------------------------------
# T10.7 — Cache content-key safety
# ---------------------------------------------------------------------------

def test_cache_content_key_safety():
    """Two NZ objects with different content must use separate cache slots."""
    pytest.importorskip("snappy")
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier, apply_cusp_basis_change
    from manifold_index.core.refined_dehn_filling import (
        _cached_compute_refined_index, clear_filling_caches, _iref_cache,
    )

    clear_filling_caches()
    assert len(_iref_cache) == 0, "Cache should be empty after clear"

    data_m003 = load_manifold("m003")
    easy_m003 = find_easy_edges(data_m003)
    nz_m003 = build_neumann_zagier(data_m003, easy_m003)

    data_m004 = load_manifold("m004")
    easy_m004 = find_easy_edges(data_m004)
    nz_m004 = build_neumann_zagier(data_m004, easy_m004)

    # Compute for m003 — should add one cache key
    _cached_compute_refined_index(nz_m003, [0], [Fraction(0)], q_order_half=6)
    n_after_m003 = len(_iref_cache)

    # Compute for m004 — should add a DIFFERENT cache key (not hit m003's entry)
    _cached_compute_refined_index(nz_m004, [0], [Fraction(0)], q_order_half=6)
    n_after_m004 = len(_iref_cache)

    assert n_after_m004 > n_after_m003, (
        "Computing m004 did not add a new cache entry — possible cache key collision with m003"
    )
