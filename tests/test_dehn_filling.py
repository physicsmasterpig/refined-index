"""tests/test_dehn_filling.py — Smoke tests for Dehn filling."""

import subprocess
from fractions import Fraction
from math import gcd

import pytest

from manifold_index.core.dehn_filling import (
    FilledIndexResult, KernelTerm, QSeries,
    _apply_kernel, _particular_solution, compute_filled_index,
    find_non_closable_cycles, find_rs,
)


def _has_mathematica():
    for cmd in (
        "wolframscript",
        "/usr/local/bin/wolframscript",
        "/Applications/Wolfram.app/Contents/MacOS/wolframscript",
        "/Applications/Mathematica.app/Contents/MacOS/wolframscript",
        "/Applications/Wolfram Engine.app/Contents/MacOS/wolframscript",
    ):
        try:
            r = subprocess.run([cmd, "-version"], capture_output=True, timeout=10)
            if r.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _has_snapy():
    try:
        import snappy
        return True
    except ImportError:
        return False

skip_no_snapy = pytest.mark.skipif(
    not _has_snapy(), reason="SnaPy not installed"
)
skip_no_math = pytest.mark.skipif(
    not _has_mathematica(), reason="Mathematica not available"
)


@pytest.fixture(scope="module")
def nz_m004():
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("m004")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)


def test_find_rs():
    R, S = find_rs(3, 2)
    assert R * 2 - 3 * S == 1


def test_particular_solution():
    m0, e0 = _particular_solution(3, 2, 2)
    assert 3 * m0 + 2 * 2 * e0 == 2


def test_apply_kernel_identity():
    """c=0, phase=0 → identity kernel."""
    term = KernelTerm(m=0, e=Fraction(0), c=0, phase=0)
    s = {0: Fraction(1), 2: Fraction(3)}
    assert _apply_kernel(term, s) == s


@skip_no_math
def test_compute_filled_index_smoke(nz_m004):
    result = compute_filled_index(nz_m004, cusp_idx=0, P=2, Q=1, q_order_half=8)
    assert isinstance(result, FilledIndexResult)
    for v in result.series.values():
        assert isinstance(v, Fraction)


# ---------- v1683 regression ----------

@pytest.fixture(scope="module")
def nz_v1683():
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    data = load_manifold("v1683")
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)


@skip_no_math
def test_v1683_meridian_only_q8(nz_v1683):
    """Regression: only (1,0) should be non-closable for v1683."""
    result = find_non_closable_cycles(
        nz_v1683, cusp_idx=0,
        p_range=range(-2, 3), q_range=range(0, 3),
        q_order_half=8,
    )
    cycle_set = {(nc.P, nc.Q) for nc in result.cycles}
    assert cycle_set == {(1, 0)}


@skip_no_math
def test_v1683_meridian_only_q4(nz_v1683):
    """Regression: buffer-clamp bug at small q_order_half."""
    result = find_non_closable_cycles(
        nz_v1683, cusp_idx=0,
        p_range=range(-2, 3), q_range=range(0, 3),
        q_order_half=4,
    )
    cycle_set = {(nc.P, nc.Q) for nc in result.cycles}
    assert cycle_set == {(1, 0)}
