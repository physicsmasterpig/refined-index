"""
tests/test_basis_selection.py — Tests for Step 6: basis selection.

Test classes
------------
TestCycleChoice
    Pure unit tests for CycleChoice construction, validation, and properties.
    No SnaPy required.

TestBasisSelection
    Pure unit tests for BasisSelection construction and properties.
    No SnaPy required.

TestDefaultChoices
    Unit tests for default_meridian_choice and default_longitude_choice.
    No SnaPy required.

TestMakeBasisSelection
    Unit tests for make_basis_selection.  Uses mock NonClosableCycleResult
    objects to avoid needing SnaPy.

TestMakeBasisSelectionWithSnapy
    Integration tests for make_basis_selection with real manifolds.
    Requires SnaPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import pytest

from manifold_index.core.basis_selection import (
    BasisSelection,
    CycleChoice,
    default_longitude_choice,
    default_meridian_choice,
    make_basis_selection,
)


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

def _has_snapy() -> bool:
    try:
        import snappy  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_snapy = pytest.mark.skipif(
    not _has_snapy(), reason="SnaPy not installed"
)


# ---------------------------------------------------------------------------
# Minimal stubs for NonClosableCycle / NonClosableCycleResult
# (avoid importing dehn_filling to keep tests independent)
# ---------------------------------------------------------------------------

@dataclass
class _Cycle:
    cusp_idx: int
    P: int
    Q: int


@dataclass
class _CycleResult:
    cusp_idx: int
    cycles: list[_Cycle] = field(default_factory=list)
    slopes_tested: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_nz(name: str):
    """Build NeumannZagierData from a named SnaPy manifold."""
    from manifold_index.core.manifold import load_manifold
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.phase_space import find_easy_edges

    data = load_manifold(name)
    easy = find_easy_edges(data)
    return build_neumann_zagier(data, easy)


# ===========================================================================
# TestCycleChoice — pure unit tests
# ===========================================================================

class TestCycleChoice:
    """CycleChoice construction, properties, and validation."""

    # ---- construction ----

    def test_meridian(self):
        cc = CycleChoice(cusp_idx=0, P=1, Q=0)
        assert cc.m == 1
        assert cc.e == Fraction(0)

    def test_longitude(self):
        cc = CycleChoice(cusp_idx=0, P=0, Q=1)
        assert cc.m == 0
        assert cc.e == Fraction(1, 2)

    def test_general_slope(self):
        cc = CycleChoice(cusp_idx=2, P=3, Q=-2)
        assert cc.m == 3
        assert cc.e == Fraction(-2, 2)   # = -1

    def test_half_integer_e(self):
        cc = CycleChoice(cusp_idx=0, P=2, Q=3)
        assert cc.e == Fraction(3, 2)

    def test_negative_P(self):
        cc = CycleChoice(cusp_idx=1, P=-1, Q=0)
        assert cc.m == -1
        assert cc.e == Fraction(0)

    # ---- slope_str ----

    def test_slope_str_meridian(self):
        cc = CycleChoice(cusp_idx=0, P=1, Q=0)
        assert cc.slope_str == "1/0"

    def test_slope_str_negative(self):
        cc = CycleChoice(cusp_idx=0, P=-2, Q=3)
        assert cc.slope_str == "-2/3"

    # ---- auto label ----

    def test_auto_label_meridian(self):
        cc = CycleChoice(cusp_idx=0, P=1, Q=0)
        assert "M" in cc.label or "1/0" in cc.label

    def test_auto_label_longitude(self):
        cc = CycleChoice(cusp_idx=0, P=0, Q=1)
        assert "L" in cc.label or "0/1" in cc.label

    def test_auto_label_general(self):
        cc = CycleChoice(cusp_idx=0, P=3, Q=-2)
        assert "3" in cc.label and "-2" in cc.label or "3/-2" in cc.label

    def test_custom_label(self):
        cc = CycleChoice(cusp_idx=0, P=1, Q=2, label="my label")
        assert cc.label == "my label"

    # ---- validation ----

    def test_zero_zero_raises(self):
        with pytest.raises(ValueError, match="not a valid cycle"):
            CycleChoice(cusp_idx=0, P=0, Q=0)

    def test_non_primitive_raises(self):
        with pytest.raises(ValueError, match="not primitive"):
            CycleChoice(cusp_idx=0, P=2, Q=4)

    def test_non_primitive_negative_raises(self):
        with pytest.raises(ValueError, match="not primitive"):
            CycleChoice(cusp_idx=0, P=-3, Q=6)

    def test_primitive_works(self):
        # (2, 3) is primitive
        cc = CycleChoice(cusp_idx=0, P=2, Q=3)
        assert cc.P == 2 and cc.Q == 3

    # ---- str ----

    def test_str_contains_cusp(self):
        cc = CycleChoice(cusp_idx=5, P=1, Q=0)
        assert "5" in str(cc)

    def test_str_contains_m_e(self):
        cc = CycleChoice(cusp_idx=0, P=2, Q=3)
        s = str(cc)
        assert "m=2" in s
        assert "e=" in s


# ===========================================================================
# TestBasisSelection — pure unit tests
# ===========================================================================

class TestBasisSelection:
    """BasisSelection construction, properties, and validation."""

    def _make(self, slopes: list[tuple[int, int]]) -> BasisSelection:
        choices = [CycleChoice(cusp_idx=i, P=P, Q=Q) for i, (P, Q) in enumerate(slopes)]
        return BasisSelection(choices=choices)

    # ---- m_ext / e_ext ----

    def test_single_cusp_meridian(self):
        bs = self._make([(1, 0)])
        assert bs.m_ext == [1]
        assert bs.e_ext == [Fraction(0)]

    def test_single_cusp_longitude(self):
        bs = self._make([(0, 1)])
        assert bs.m_ext == [0]
        assert bs.e_ext == [Fraction(1, 2)]

    def test_two_cusps(self):
        bs = self._make([(1, 0), (2, 3)])
        assert bs.m_ext == [1, 2]
        assert bs.e_ext == [Fraction(0), Fraction(3, 2)]

    def test_three_cusps(self):
        bs = self._make([(1, 0), (0, 1), (-1, 2)])
        assert bs.m_ext == [1, 0, -1]
        assert bs.e_ext == [Fraction(0), Fraction(1, 2), Fraction(1)]

    def test_r_property(self):
        bs = self._make([(1, 0), (0, 1)])
        assert bs.r == 2

    # ---- validation ----

    def test_empty_choices_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            BasisSelection(choices=[])

    def test_wrong_cusp_order_raises(self):
        choices = [
            CycleChoice(cusp_idx=1, P=1, Q=0),
            CycleChoice(cusp_idx=0, P=0, Q=1),
        ]
        with pytest.raises(ValueError, match="cusp-index order"):
            BasisSelection(choices=choices)

    def test_gap_in_cusp_indices_raises(self):
        choices = [
            CycleChoice(cusp_idx=0, P=1, Q=0),
            CycleChoice(cusp_idx=2, P=0, Q=1),   # gap: no cusp 1
        ]
        with pytest.raises(ValueError, match="cusp-index order"):
            BasisSelection(choices=choices)

    # ---- summary / str ----

    def test_summary_contains_r(self):
        bs = self._make([(1, 0), (0, 1)])
        s = bs.summary()
        assert "2 cusps" in s

    def test_str_contains_m_ext(self):
        bs = self._make([(2, 3)])
        s = str(bs)
        assert "m_ext" in s

    def test_str_contains_e_ext(self):
        bs = self._make([(2, 3)])
        s = str(bs)
        assert "e_ext" in s


# ===========================================================================
# TestDefaultChoices — pure unit tests
# ===========================================================================

class TestDefaultChoices:
    """default_meridian_choice and default_longitude_choice."""

    def test_meridian_m_e(self):
        cc = default_meridian_choice(cusp_idx=3)
        assert cc.cusp_idx == 3
        assert cc.m == 1
        assert cc.e == Fraction(0)
        assert cc.is_default is True

    def test_longitude_m_e(self):
        cc = default_longitude_choice(cusp_idx=1)
        assert cc.cusp_idx == 1
        assert cc.m == 0
        assert cc.e == Fraction(1, 2)
        assert cc.is_default is True

    def test_meridian_label(self):
        cc = default_meridian_choice(0)
        assert "M" in cc.label or "1/0" in cc.label

    def test_longitude_label(self):
        cc = default_longitude_choice(0)
        assert "L" in cc.label or "0/1" in cc.label


# ===========================================================================
# TestMakeBasisSelection — unit tests with mock data
# ===========================================================================

class TestMakeBasisSelection:
    """make_basis_selection with stub NeumannZagierData and mock cycle results."""

    # ------------------------------------------------------------------
    # Minimal NeumannZagierData stub
    # ------------------------------------------------------------------

    @staticmethod
    def _nz_stub(r: int):
        """Return a minimal object with .r = r."""

        class _Stub:
            pass

        s = _Stub()
        s.r = r
        return s

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_all_none_default_M(self):
        nz = self._nz_stub(2)
        bs = make_basis_selection(nz, [], [None, None], default="M")
        assert bs.m_ext == [1, 1]
        assert bs.e_ext == [Fraction(0), Fraction(0)]
        assert all(c.is_default for c in bs.choices)

    def test_all_none_default_L(self):
        nz = self._nz_stub(2)
        bs = make_basis_selection(nz, [], [None, None], default="L")
        assert bs.m_ext == [0, 0]
        assert bs.e_ext == [Fraction(1, 2), Fraction(1, 2)]
        assert all(c.is_default for c in bs.choices)

    def test_explicit_slope(self):
        nz = self._nz_stub(1)
        bs = make_basis_selection(nz, [], [(2, 3)])
        assert bs.m_ext == [2]
        assert bs.e_ext == [Fraction(3, 2)]

    def test_mixed_choices(self):
        nz = self._nz_stub(3)
        bs = make_basis_selection(nz, [], [(2, 3), None, (-1, 0)], default="M")
        assert bs.m_ext == [2, 1, -1]
        assert bs.e_ext == [Fraction(3, 2), Fraction(0), Fraction(0)]

    def test_non_closable_label(self):
        """A slope found in cycle_results gets a 'non-closable' label."""
        nz = self._nz_stub(1)
        res = _CycleResult(cusp_idx=0, cycles=[_Cycle(0, 2, 3)])
        bs = make_basis_selection(nz, [res], [(2, 3)])
        assert "non-closable" in bs.choices[0].label

    def test_non_found_slope_label(self):
        """A slope NOT in cycle_results gets a generic 'slope' label (non-strict)."""
        nz = self._nz_stub(1)
        bs = make_basis_selection(nz, [], [(5, 7)])
        assert "non-closable" not in bs.choices[0].label
        assert "5" in bs.choices[0].label or "slope" in bs.choices[0].label

    def test_wrong_choices_length_raises(self):
        nz = self._nz_stub(2)
        with pytest.raises(ValueError, match=r"len\(choices\)"):
            make_basis_selection(nz, [], [None])  # only 1, need 2

    def test_zero_zero_choice_raises(self):
        nz = self._nz_stub(1)
        with pytest.raises(ValueError, match="not a valid cycle"):
            make_basis_selection(nz, [], [(0, 0)])

    def test_non_primitive_choice_raises(self):
        nz = self._nz_stub(1)
        with pytest.raises(ValueError, match="not primitive"):
            make_basis_selection(nz, [], [(2, 4)])

    def test_invalid_default_raises(self):
        nz = self._nz_stub(1)
        with pytest.raises(ValueError, match="default="):
            make_basis_selection(nz, [], [None], default="X")

    # ---- strict mode ----

    def test_strict_known_slope_ok(self):
        nz = self._nz_stub(1)
        res = _CycleResult(cusp_idx=0, cycles=[_Cycle(0, 2, 3)])
        bs = make_basis_selection(nz, [res], [(2, 3)], strict=True)
        assert bs.choices[0].P == 2

    def test_strict_unknown_slope_raises(self):
        nz = self._nz_stub(1)
        res = _CycleResult(cusp_idx=0, cycles=[_Cycle(0, 2, 3)])
        with pytest.raises(ValueError, match="not found to be non-closable"):
            make_basis_selection(nz, [res], [(5, 7)], strict=True)

    def test_strict_none_choice_is_ok(self):
        """None (default) is always accepted even in strict mode."""
        nz = self._nz_stub(1)
        bs = make_basis_selection(nz, [], [None], default="M", strict=True)
        assert bs.choices[0].is_default is True

    # ---- single cusp ----

    def test_single_cusp_meridian(self):
        nz = self._nz_stub(1)
        bs = make_basis_selection(nz, [], [(1, 0)])
        assert bs.m_ext == [1]
        assert bs.e_ext == [Fraction(0)]
        assert bs.choices[0].is_default is True   # M is marked as default

    def test_single_cusp_longitude(self):
        nz = self._nz_stub(1)
        bs = make_basis_selection(nz, [], [(0, 1)])
        assert bs.m_ext == [0]
        assert bs.e_ext == [Fraction(1, 2)]
        assert bs.choices[0].is_default is True   # L is marked as default


# ===========================================================================
# TestMakeBasisSelectionWithSnapy — integration tests, SnaPy required
# ===========================================================================

class TestMakeBasisSelectionWithSnapy:
    """Integration tests that build nz_data from a real SnaPy manifold."""

    @skip_no_snapy
    def test_4_1_single_cusp_default(self):
        """4_1 has one cusp; default M gives m_ext=[1], e_ext=[0]."""
        nz = _load_nz("4_1")
        bs = make_basis_selection(nz, [], [None], default="M")
        assert bs.r == 1
        assert bs.m_ext == [1]
        assert bs.e_ext == [Fraction(0)]

    @skip_no_snapy
    def test_4_1_explicit_slope(self):
        """4_1: explicit slope (-2, 3) → m=-2, e=3/2."""
        nz = _load_nz("4_1")
        bs = make_basis_selection(nz, [], [(-2, 3)])
        assert bs.m_ext == [-2]
        assert bs.e_ext == [Fraction(3, 2)]

    @skip_no_snapy
    def test_m004_single_cusp(self):
        """m004 has r=1 cusp; basic sanity check."""
        nz = _load_nz("m004")
        assert nz.r == 1
        bs = make_basis_selection(nz, [], [(1, 2)])
        assert bs.m_ext == [1]
        assert bs.e_ext == [Fraction(1)]

    @skip_no_snapy
    def test_m003_single_cusp(self):
        """m003 has r=1 cusp; basic single-cusp check."""
        nz = _load_nz("m003")
        assert nz.r == 1
        bs = make_basis_selection(nz, [], [(1, 2)])
        assert bs.r == 1
        assert bs.m_ext == [1]
        assert bs.e_ext == [Fraction(1)]

    @skip_no_snapy
    def test_m003_wrong_length_raises(self):
        """Providing wrong number of choices raises ValueError."""
        nz = _load_nz("m003")
        with pytest.raises(ValueError, match=r"len\(choices\)"):
            make_basis_selection(nz, [], [None, None])  # need 1, gave 2

    @skip_no_snapy
    def test_m003_all_default_M(self):
        """m003 one-None with default='M' → meridian."""
        nz = _load_nz("m003")
        bs = make_basis_selection(nz, [], [None], default="M")
        assert bs.m_ext == [1]
        assert bs.e_ext == [Fraction(0)]

    @skip_no_snapy
    def test_m003_all_default_L(self):
        """m003 one-None with default='L' → longitude."""
        nz = _load_nz("m003")
        bs = make_basis_selection(nz, [], [None], default="L")
        assert bs.m_ext == [0]
        assert bs.e_ext == [Fraction(1, 2)]

    @skip_no_snapy
    def test_basis_selection_feeds_compute_refined_index(self):
        """BasisSelection.m_ext / e_ext can be fed directly to compute_refined_index."""
        from manifold_index.core.refined_index import compute_refined_index

        nz = _load_nz("4_1")
        bs = make_basis_selection(nz, [], [(1, 0)], default="M")

        # Should not raise; result is a dict
        result = compute_refined_index(nz, bs.m_ext, bs.e_ext, q_order_half=10)
        assert isinstance(result, dict)

    @skip_no_snapy
    def test_basis_selection_generic_zero_zero(self):
        """Slope (0,0) is invalid; use (1,0) or (0,1) for standard evaluation."""
        nz = _load_nz("4_1")
        with pytest.raises(ValueError, match="not a valid cycle"):
            make_basis_selection(nz, [], [(0, 0)])
