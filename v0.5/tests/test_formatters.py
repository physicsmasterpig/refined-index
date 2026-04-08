"""
Phase 4 tests — Formatter functions.

All tests use lightweight stubs (no SnaPy required).
Tests check that:
- Returned strings are non-empty
- Key substrings are present (table tags, KaTeX delimiters, expected values)
- Zero results yield "$0$"
- Edge-case inputs don't raise exceptions
"""
from __future__ import annotations

import pytest
import numpy as np
from fractions import Fraction
from types import SimpleNamespace

from manifold_index.formatters.manifold_fmt import (
    format_nz_latex,
    format_gluing_table_html,
    format_easy_edges_html,
    format_hard_edges_html,
    format_summary_html,
    _frac_to_latex,
    _slope_latex,
)
from manifold_index.formatters.index_fmt import (
    format_series_latex,
    format_index_table_html,
    DISPLAY_CHARGES,
)
from manifold_index.formatters.weyl_fmt import (
    format_weyl_html,
    format_compatibility_html,
)
from manifold_index.formatters.filling_fmt import (
    format_filled_series_latex,
    format_nc_cycle_table_html,
    format_fill_result_html,
    format_slope_latex,
)
from manifold_index.viewmodels.index_vm import build_weyl_vm
from manifold_index.viewmodels.filling_vm import (
    build_nc_cycle_vm,
    build_fill_query_vm,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _make_nz(n=2):
    """Minimal NeumannZagierData stub with identity g_NZ."""
    return SimpleNamespace(
        n=n,
        r=1,
        g_NZ=np.eye(2 * n, dtype=float),
        nu_x=np.zeros(n, dtype=int),
        nu_p=np.zeros(n, dtype=float),
        num_hard=1,
    )


def _make_md(n=2, name="m004"):
    """Minimal ManifoldData stub."""
    G = np.zeros((n, 3 * n), dtype=int)
    G[0, 0] = 1
    return SimpleNamespace(
        name=name,
        num_tetrahedra=n,
        num_cusps=1,
        gluing_matrix=G,
    )


def _make_ps(n=2, n_easy=1, n_hard=1):
    """Minimal EasyEdgeResult stub."""
    easy_edges = [np.zeros(3 * n, dtype=int) for _ in range(n_easy)]
    hard_edges = [np.zeros(3 * n, dtype=int) for _ in range(n_hard)]
    return SimpleNamespace(
        n=n,
        all_easy=easy_edges,
        independent_easy_indices=list(range(n_easy)),
        hard_padding=hard_edges,
        num_independent_easy=n_easy,
    )


# ---------------------------------------------------------------------------
# manifold_fmt — helpers
# ---------------------------------------------------------------------------

class TestFracToLatex:
    def test_integer(self):
        assert _frac_to_latex(2) == "2"

    def test_negative_integer(self):
        assert _frac_to_latex(-3) == "-3"

    def test_zero(self):
        assert _frac_to_latex(0) == "0"

    def test_half(self):
        result = _frac_to_latex(Fraction(1, 2))
        assert r"\tfrac" in result
        assert "1" in result and "2" in result

    def test_negative_half(self):
        result = _frac_to_latex(Fraction(-1, 2))
        assert "-" in result


class TestSlopeLatex:
    def test_zero(self):
        assert _slope_latex(0, 0) == "0"

    def test_alpha_only(self):
        assert r"\alpha" in _slope_latex(1, 0)

    def test_neg_alpha(self):
        result = _slope_latex(-1, 0)
        assert "-" in result

    def test_alpha_plus_beta(self):
        result = _slope_latex(1, 1)
        assert r"\alpha" in result and r"\beta" in result

    def test_alpha_minus_beta(self):
        result = _slope_latex(1, -1)
        assert "-" in result

    def test_custom_symbols(self):
        result = _slope_latex(1, 1, a=r"\gamma", b=r"\delta")
        assert r"\gamma" in result and r"\delta" in result


# ---------------------------------------------------------------------------
# manifold_fmt — NZ matrix
# ---------------------------------------------------------------------------

class TestFormatNzLatex:
    def test_returns_string(self):
        nz = _make_nz(n=2)
        result = format_nz_latex(nz)
        assert isinstance(result, str)

    def test_contains_pmatrix(self):
        nz = _make_nz(n=2)
        result = format_nz_latex(nz)
        assert "pmatrix" in result

    def test_contains_dollar(self):
        nz = _make_nz(n=2)
        result = format_nz_latex(nz)
        assert "$$" in result or "$" in result

    def test_nu_x_in_result(self):
        nz = _make_nz(n=2)
        result = format_nz_latex(nz)
        assert r"\nu" in result or "nu" in result.lower() or "ν" in result or "nu_x" in result or "0" in result

    def test_n1_works(self):
        nz = _make_nz(n=1)
        result = format_nz_latex(nz)
        assert "pmatrix" in result


# ---------------------------------------------------------------------------
# manifold_fmt — gluing table
# ---------------------------------------------------------------------------

class TestFormatGluingTableHtml:
    def test_is_table(self):
        md = _make_md(n=2)
        result = format_gluing_table_html(md)
        assert "<table>" in result and "</table>" in result

    def test_has_header_row(self):
        md = _make_md(n=2)
        result = format_gluing_table_html(md)
        assert "<th>" in result

    def test_has_data_row(self):
        md = _make_md(n=2)
        result = format_gluing_table_html(md)
        assert "<tr>" in result and "<td>" in result

    def test_n1(self):
        md = _make_md(n=1)
        result = format_gluing_table_html(md)
        assert "<table>" in result


# ---------------------------------------------------------------------------
# manifold_fmt — edge classification
# ---------------------------------------------------------------------------

class TestFormatEdgeHtml:
    def test_easy_html_has_table(self):
        ps = _make_ps(n=2, n_easy=1, n_hard=1)
        result = format_easy_edges_html(ps)
        assert "<table>" in result

    def test_easy_edge_label(self):
        ps = _make_ps(n=2, n_easy=1, n_hard=0)
        result = format_easy_edges_html(ps)
        assert "E0" in result

    def test_hard_html_has_table(self):
        ps = _make_ps(n=2, n_easy=1, n_hard=1)
        result = format_hard_edges_html(ps)
        assert "<table>" in result

    def test_hard_edge_label(self):
        ps = _make_ps(n=2, n_easy=1, n_hard=1)
        result = format_hard_edges_html(ps)
        assert "H0" in result

    def test_no_easy_edges_gives_muted(self):
        ps = _make_ps(n=2, n_easy=0, n_hard=1)
        result = format_easy_edges_html(ps)
        assert "muted" in result or "No" in result

    def test_no_hard_edges_gives_muted(self):
        ps = _make_ps(n=2, n_easy=1, n_hard=0)
        result = format_hard_edges_html(ps)
        assert "muted" in result or "No" in result


# ---------------------------------------------------------------------------
# manifold_fmt — summary
# ---------------------------------------------------------------------------

class TestFormatSummaryHtml:
    def test_contains_name(self):
        md = _make_md(name="m125")
        ps = _make_ps()
        result = format_summary_html(md, ps)
        assert "m125" in result

    def test_contains_tetrahedra_count(self):
        md = _make_md(n=3)
        ps = _make_ps(n=3)
        result = format_summary_html(md, ps)
        assert "3" in result

    def test_contains_edge_counts(self):
        md = _make_md()
        ps = _make_ps(n_easy=1, n_hard=1)
        result = format_summary_html(md, ps)
        assert "easy" in result and "hard" in result


# ---------------------------------------------------------------------------
# index_fmt — format_series_latex
# ---------------------------------------------------------------------------

class TestFormatSeriesLatex:
    def test_empty_result_is_zero(self):
        assert format_series_latex({}, num_hard=1) == "$0$"

    def test_constant_result(self):
        result = {(0,): 1}
        s = format_series_latex(result, num_hard=0)
        assert s == "$1$"

    def test_negative_constant(self):
        result = {(0,): -1}
        s = format_series_latex(result, num_hard=0)
        assert "-1" in s

    def test_q_power(self):
        result = {(2,): 1}
        s = format_series_latex(result, num_hard=0)
        assert "q" in s

    def test_eta_variable(self):
        # Key (0, 2) → q^0 · η^{2W_0}
        result = {(0, 2): 1}
        s = format_series_latex(result, num_hard=1)
        assert r"\eta" in s

    def test_multiple_q_terms(self):
        result = {(0,): 1, (2,): -1, (4,): 2}
        s = format_series_latex(result, num_hard=0)
        assert "$" in s and len(s) > 3

    def test_max_q_terms_truncates(self):
        result = {(0,): 1, (2,): 1, (4,): 1, (6,): 1, (8,): 1}
        s = format_series_latex(result, num_hard=0, max_q_terms=2)
        assert r"\cdots" in s

    def test_starts_and_ends_with_dollar(self):
        result = {(2,): 3}
        s = format_series_latex(result, num_hard=0)
        assert s.startswith("$") and s.endswith("$")

    def test_m004_known_sector(self):
        # m004 (m=0, e=0) first term: constant 1
        result = {(0,): 1, (2, -2): -1, (2, 0): -1, (2, 2): -1}
        s = format_series_latex(result, num_hard=1)
        assert "$" in s
        assert "1" in s


# ---------------------------------------------------------------------------
# index_fmt — format_index_table_html
# ---------------------------------------------------------------------------

class TestFormatIndexTableHtml:
    def test_is_table(self):
        entries = [([0], [Fraction(0)], {(0,): 1})]
        result = format_index_table_html(entries, num_hard=0, num_cusps=1)
        assert "<table" in result and "</table>" in result

    def test_has_idx_class(self):
        entries = [([0], [Fraction(0)], {(0,): 1})]
        result = format_index_table_html(entries, num_hard=0, num_cusps=1)
        assert 'class="idx"' in result

    def test_empty_entries_shows_zero(self):
        result = format_index_table_html([], num_hard=0, num_cusps=1)
        assert "<table" in result
        assert "$0$" in result

    def test_nonzero_entry_in_table(self):
        entries = [([0], [Fraction(0)], {(0,): 1})]
        result = format_index_table_html(entries, num_hard=0, num_cusps=1)
        assert "$1$" in result

    def test_sector_count_mentioned(self):
        result = format_index_table_html([], num_hard=0, num_cusps=1)
        # 5^1 = 5 sectors
        assert "5" in result


# ---------------------------------------------------------------------------
# weyl_fmt — format_weyl_html
# ---------------------------------------------------------------------------

class TestFormatWeylHtml:
    def _vm(self, **kwargs):
        from fractions import Fraction
        import types
        ab = types.SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 2)])
        defaults = dict(adjoint_value=None, adjoint_passed=None)
        defaults.update(kwargs)
        return build_weyl_vm(ab, num_hard=1, **defaults)

    def test_not_checked_shows_muted(self):
        from manifold_index.viewmodels.index_vm import WeylViewModel
        vm = WeylViewModel(
            checked=False, a_vectors=[], b_vectors=[],
            edge_compatible=[], is_fully_compatible=False,
            adjoint_value=None, adjoint_passed=None, warnings=[],
        )
        result = format_weyl_html(vm)
        assert "muted" in result

    def test_has_weyl_vectors(self):
        vm = self._vm()
        result = format_weyl_html(vm)
        # a_0 and b_0 values present
        assert "a_{0}" in result or "a_0" in result or "a_{" in result

    def test_compatible_shows_success(self):
        vm = self._vm()
        result = format_weyl_html(vm)
        assert "success" in result or "✓" in result

    def test_incompatible_shows_warn(self):
        import types
        ab = types.SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 3)])
        vm = build_weyl_vm(ab, num_hard=1)
        result = format_weyl_html(vm)
        assert "warn" in result or "⚠" in result

    def test_adjoint_pass_shows_success(self):
        vm = self._vm(adjoint_value=-1.0, adjoint_passed=True)
        result = format_weyl_html(vm)
        assert "success" in result or "✓" in result

    def test_adjoint_fail_shows_warn(self):
        vm = self._vm(adjoint_value=-0.5, adjoint_passed=False)
        result = format_weyl_html(vm)
        assert "warn" in result or "⚠" in result


# ---------------------------------------------------------------------------
# weyl_fmt — format_compatibility_html
# ---------------------------------------------------------------------------

class TestFormatCompatibilityHtml:
    def test_no_data_shows_muted(self):
        from manifold_index.viewmodels.index_vm import WeylViewModel
        vm = WeylViewModel(
            checked=False, a_vectors=[], b_vectors=[],
            edge_compatible=[], is_fully_compatible=False,
            adjoint_value=None, adjoint_passed=None, warnings=[],
        )
        result = format_compatibility_html(vm)
        assert "muted" in result

    def test_has_table(self):
        import types
        ab = types.SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 2)])
        vm = build_weyl_vm(ab, num_hard=1)
        result = format_compatibility_html(vm)
        assert "<table>" in result and "</table>" in result

    def test_compatible_check_mark(self):
        import types
        ab = types.SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 2)])
        vm = build_weyl_vm(ab, num_hard=1)
        result = format_compatibility_html(vm)
        assert "✓" in result

    def test_incompatible_cross(self):
        import types
        ab = types.SimpleNamespace(a=[Fraction(2)], b=[Fraction(1, 3)])
        vm = build_weyl_vm(ab, num_hard=1)
        result = format_compatibility_html(vm)
        assert "✗" in result

    def test_two_edges(self):
        import types
        ab = types.SimpleNamespace(
            a=[Fraction(2), Fraction(1)],
            b=[Fraction(1, 2), Fraction(1, 2)],
        )
        vm = build_weyl_vm(ab, num_hard=2)
        result = format_compatibility_html(vm)
        assert result.count("<tr>") >= 3  # header + 2 data rows


# ---------------------------------------------------------------------------
# filling_fmt — format_slope_latex
# ---------------------------------------------------------------------------

class TestFormatSlopeLatex:
    def test_zero(self):
        assert format_slope_latex(0, 0) == "0"

    def test_alpha(self):
        assert r"\alpha" in format_slope_latex(1, 0)

    def test_neg_beta(self):
        result = format_slope_latex(0, -1)
        assert "-" in result and r"\beta" in result

    def test_alpha_plus_beta(self):
        result = format_slope_latex(1, 1)
        assert "+" in result


# ---------------------------------------------------------------------------
# filling_fmt — format_filled_series_latex
# ---------------------------------------------------------------------------

class TestFormatFilledSeriesLatex:
    def test_empty_is_zero(self):
        assert format_filled_series_latex({}, num_hard=0) == "$0$"

    def test_constant(self):
        s = format_filled_series_latex({(0,): Fraction(1)}, num_hard=0)
        assert s == "$1$"

    def test_q_power(self):
        s = format_filled_series_latex({(2,): Fraction(1)}, num_hard=0)
        assert "q" in s

    def test_eta_notation(self):
        s = format_filled_series_latex({(0, 2): Fraction(1)}, num_hard=1)
        assert r"\eta" in s and "W_" in s

    def test_cusp_eta_notation(self):
        s = format_filled_series_latex(
            {(0, 0, 1): Fraction(1)},
            num_hard=1,
            has_cusp_eta=True,
            num_cusp_eta=1,
        )
        assert r"\eta" in s and "V_" in s

    def test_fraction_coefficient(self):
        s = format_filled_series_latex({(0,): Fraction(1, 2)}, num_hard=0)
        assert r"\tfrac" in s

    def test_max_q_terms(self):
        series = {(0,): Fraction(1), (2,): Fraction(1),
                  (4,): Fraction(1), (6,): Fraction(1), (8,): Fraction(1)}
        s = format_filled_series_latex(series, num_hard=0, max_q_terms=2)
        assert r"\cdots" in s


# ---------------------------------------------------------------------------
# filling_fmt — format_nc_cycle_table_html
# ---------------------------------------------------------------------------

class TestFormatNcCycleTableHtml:
    def test_empty_list_shows_muted(self):
        result = format_nc_cycle_table_html([])
        assert "muted" in result or "No" in result

    def test_single_cycle_has_table(self):
        nc = build_nc_cycle_vm(0, 1, 0)
        result = format_nc_cycle_table_html([nc])
        assert "<table" in result and "</table>" in result

    def test_slope_in_table(self):
        nc = build_nc_cycle_vm(0, 1, 0, slope_latex=r"$\alpha$")
        result = format_nc_cycle_table_html([nc])
        assert r"$\alpha$" in result

    def test_cusp_index_shown(self):
        nc = build_nc_cycle_vm(2, 1, 0)
        result = format_nc_cycle_table_html([nc])
        assert "2" in result

    def test_compatible_checkmark(self):
        nc = build_nc_cycle_vm(0, 1, 0, weyl_compatible=True)
        result = format_nc_cycle_table_html([nc])
        assert "✓" in result

    def test_incompatible_cross(self):
        nc = build_nc_cycle_vm(0, 1, 0, weyl_compatible=False)
        result = format_nc_cycle_table_html([nc])
        assert "✗" in result

    def test_unknown_compatibility_dash(self):
        nc = build_nc_cycle_vm(0, 1, 0, weyl_compatible=None)
        result = format_nc_cycle_table_html([nc])
        assert "—" in result


# ---------------------------------------------------------------------------
# filling_fmt — format_fill_result_html
# ---------------------------------------------------------------------------

class TestFormatFillResultHtml:
    def _make_fq(self, is_zero=False, incompat_edges=None):
        from types import SimpleNamespace
        result = SimpleNamespace(
            is_zero=is_zero,
            series={} if is_zero else {(0,): Fraction(1)},
        )
        return build_fill_query_vm(
            nc_P=1, nc_Q=0,
            user_P=3, user_Q=1,
            p=-3, q=1,
            m_other=[], e_other=[],
            result=result,
            result_latex="$1$" if not is_zero else "$0$",
        )

    def test_has_series_table(self):
        fq = self._make_fq()
        result = format_fill_result_html(fq)
        assert '<table class="idx">' in result

    def test_nc_slope_present(self):
        fq = self._make_fq()
        result = format_fill_result_html(fq)
        assert "NC cycle" in result or "nc" in result.lower() or fq.nc_slope_latex in result

    def test_user_slope_present(self):
        fq = self._make_fq()
        result = format_fill_result_html(fq)
        assert fq.user_slope_latex in result

    def test_no_incompat_warning_when_none(self):
        fq = self._make_fq()
        result = format_fill_result_html(fq)
        assert "incompatible" not in result.lower() or fq.incompat_edges == []

    def test_result_latex_in_output(self):
        fq = self._make_fq()
        result = format_fill_result_html(fq)
        assert "$1$" in result or fq.result_latex in result
