"""Tests for the export formatting functions in utils/exporters.py."""

from __future__ import annotations

import json
from fractions import Fraction
from unittest.mock import MagicMock

import numpy as np
import pytest

from manifold_index.utils.exporters import (
    clipboard_latex,
    clipboard_plain_text,
    to_latex_filled_series,
    to_latex_series,
    to_mathematica_filled_series,
    to_mathematica_series,
    write_full_report,
    write_json,
    write_latex,
    write_mathematica,
    write_plain_text,
    write_report,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_nz_mock(n=2, r=1, num_hard=1, num_easy=1):
    """Minimal mock of NeumannZagierData."""
    nz = MagicMock()
    nz.n = n
    nz.r = r
    nz.num_hard = num_hard
    nz.num_easy = num_easy
    nz.g_NZ = np.eye(2 * n, dtype=float)
    nz.nu_x = np.zeros(n, dtype=int)
    nz.nu_p = np.zeros(n, dtype=float)
    nz.is_symplectic.return_value = True
    nz.g_NZ_inv_scaled.return_value = (1, np.eye(2 * n, dtype=np.int64))
    nz.g_NZ_inv.return_value = np.eye(2 * n, dtype=float)
    nz.inv_denom = 1
    return nz


def _make_easy_mock(n=2, num_easy=1, num_hard=1):
    """Minimal mock of EasyEdgeResult with real arrays."""
    ps = MagicMock()
    # Each easy edge is a 3n-vector.  Fake pattern: (0,1,-1, 0,0,0)
    easy_vec = np.zeros(3 * n, dtype=int)
    easy_vec[1] = 1
    easy_vec[2] = -1
    ps.all_easy = [easy_vec] * num_easy
    ps.num_independent_easy = num_easy
    ps.independent_easy_indices = set(range(num_easy))
    # Each hard edge is a 3n-vector.  Fake pattern: (1,0,-1, 0,1,-1)
    hard_vec = np.zeros(3 * n, dtype=int)
    hard_vec[0] = 1
    hard_vec[2] = -1
    hard_vec[4] = 1
    hard_vec[5] = -1
    ps.hard_padding = [hard_vec] * num_hard
    ps.n = n
    ps.r = 1
    return ps


def _make_md_mock(name="m003", n=2, r=1):
    """Minimal mock of ManifoldData."""
    md = MagicMock()
    md.name = name
    md.num_tetrahedra = n
    md.num_cusps = r
    md.gluing_matrix = None  # skip raw SnaPy sections
    return md


def _make_weyl_mock(num_hard=1, all_symmetric=True):
    """Minimal mock of WeylCheckResult."""
    w = MagicMock()
    w.ab = MagicMock()
    w.ab.a = [Fraction(1)] * num_hard
    w.ab.b = [Fraction(1, 2)] * num_hard
    w.ab.edge_compatible = [True] * num_hard
    w.ab.a_is_integer = True
    w.ab.b_is_half_integer = True
    w.ab.is_valid = True
    w.all_weyl_symmetric = all_symmetric
    w.weyl_symmetric = {(0, 0): True}
    w.adjoint = MagicMock()
    w.adjoint.projected_value = -1
    w.adjoint.is_pass = True
    w.adjoint.missing_e = []
    w.adjoint.c_e = {Fraction(0): 1}
    return w


# ── § 1  Series formatters ───────────────────────────────────────────

def test_latex_zero():
    assert to_latex_series({}, 1) == "0"


def test_latex_constant():
    result = {(0,): 5}
    assert to_latex_series(result, 0) == "5"


def test_latex_single_q():
    # q^1 means qq_pow = 2
    result = {(2, 0): 3}
    assert to_latex_series(result, 1) == r"3 \, q"


def test_latex_q_half():
    # q^{1/2} means qq_pow = 1
    result = {(1, 0): 1}
    assert to_latex_series(result, 1) == "q^{1/2}"


def test_latex_eta_subscript():
    # key = (2, 2) means q * eta_0
    result = {(2, 2): 1}
    tex = to_latex_series(result, 1)
    assert r"\eta_0" in tex
    assert "q" in tex


def test_latex_eta_negative():
    result = {(0, -2): 1}
    tex = to_latex_series(result, 1)
    assert r"\eta_0^{-1}" in tex


def test_latex_eta_higher_power():
    result = {(0, 6): 1}
    tex = to_latex_series(result, 1)
    assert r"\eta_0^{3}" in tex


def test_latex_multiple_terms():
    result = {(0, 0): 1, (2, 0): -1, (4, 2): 2}
    tex = to_latex_series(result, 1)
    assert "+" in tex or "-" in tex


def test_latex_negative_coeff():
    result = {(2, 0): -1}
    tex = to_latex_series(result, 1)
    assert tex.startswith("-")


def test_latex_two_hard_edges():
    result = {(0, 2, -2): 1}
    tex = to_latex_series(result, 2)
    assert r"\eta_0" in tex
    assert r"\eta_1^{-1}" in tex


# ── Filled series ────────────────────────────────────────────────────

def test_filled_zero():
    assert to_latex_filled_series({}, 1) == "0"


def test_filled_hard_only():
    series = {(2, 2): 1}
    tex = to_latex_filled_series(series, 1, 0)
    assert r"\eta_0" in tex


def test_filled_with_cusp():
    series = {(2, 0, 1): 1}  # q * eta^{2V_0}
    tex = to_latex_filled_series(series, 1, 1)
    assert r"\eta^{2V_0}" in tex


def test_filled_cusp_higher_power():
    series = {(0, 0, 3): 1}  # eta^{6V_0}
    tex = to_latex_filled_series(series, 1, 1)
    assert r"\eta^{6V_0}" in tex


def test_filled_max_q_truncation():
    series = {(0, 0): 1, (2, 0): 2, (4, 0): 3, (6, 0): 4}
    tex = to_latex_filled_series(series, 0, 0, max_q_terms=1)
    # max_q_terms=1 means max_qq=2, only keys with qq_pow <= 2
    assert "4" not in tex  # q^3 term excluded
    assert r"\cdots" in tex


# ── Mathematica ──────────────────────────────────────────────────────

def test_mathematica_zero():
    assert to_mathematica_series({}, 1) == "0"


def test_mathematica_q_power():
    result = {(2, 0): 3}
    assert to_mathematica_series(result, 1) == "3 q"


def test_mathematica_eta():
    result = {(0, 2): 1}
    expr = to_mathematica_series(result, 1)
    assert "eta[0]" in expr


def test_mathematica_eta_power():
    result = {(0, 6): 1}
    expr = to_mathematica_series(result, 1)
    assert "eta[0]^3" in expr


def test_mathematica_cusp():
    series = {(2, 0, 2): 1}
    expr = to_mathematica_filled_series(series, 1, 1)
    assert "etaCusp[0]^2" in expr


def test_mathematica_cusp_single():
    series = {(0, 0, 1): 1}
    expr = to_mathematica_filled_series(series, 1, 1)
    assert "etaCusp[0]" in expr


# ── § 2  File writers ─────────────────────────────────────────────────

def test_write_latex(tmp_path):
    nz = _make_nz_mock()
    entries = [([0], [Fraction(0)], {(0, 0): 1, (2, 2): -1})]
    weyl = _make_weyl_mock()
    p = tmp_path / "test.tex"
    write_latex(p, "m003", nz, entries, weyl)
    text = p.read_text()
    assert r"\documentclass" in text
    assert r"\begin{document}" in text
    assert r"\end{document}" in text
    assert "m003" in text
    assert r"\eta_0" in text


def test_write_latex_no_weyl(tmp_path):
    nz = _make_nz_mock()
    entries = [([0], [Fraction(0)], {(0, 0): 1})]
    p = tmp_path / "test.tex"
    write_latex(p, "m003", nz, entries, weyl=None)
    text = p.read_text()
    assert r"\documentclass" in text


def test_write_report_legacy(tmp_path):
    """Legacy write_report wrapper delegates to write_full_report."""
    nz = _make_nz_mock()
    easy = _make_easy_mock()
    entries = [([0], [Fraction(0)], {(0, 0): 1})]
    weyl = _make_weyl_mock()
    p = tmp_path / "report.tex"
    write_report(p, "m003", nz, easy, entries, weyl, q_order_half=10)
    text = p.read_text()
    assert r"\tableofcontents" in text
    assert "Neumann" in text
    assert "Weyl" in text


def test_write_full_report(tmp_path):
    """Direct write_full_report with all mock objects."""
    md = _make_md_mock()
    nz = _make_nz_mock()
    easy = _make_easy_mock()
    entries = [([0], [Fraction(0)], {(0, 0): 1})]
    weyl = _make_weyl_mock()
    p = tmp_path / "report.tex"
    write_full_report(p, md, easy, nz, entries, weyl, q_order_half=10)
    text = p.read_text()
    assert r"\tableofcontents" in text
    assert "Phase Space" in text or "Edge Classification" in text
    assert "Neumann" in text
    assert "Weyl" in text
    assert r"\end{document}" in text


def test_write_full_report_no_weyl(tmp_path):
    """Full report without Weyl data."""
    md = _make_md_mock()
    nz = _make_nz_mock()
    easy = _make_easy_mock()
    entries = [([0], [Fraction(0)], {(0, 0): 1})]
    p = tmp_path / "report.tex"
    write_full_report(p, md, easy, nz, entries)
    text = p.read_text()
    assert r"\tableofcontents" in text
    assert r"\end{document}" in text


def test_write_mathematica(tmp_path):
    md = _make_md_mock()
    nz = _make_nz_mock()
    entries = [([0], [Fraction(0)], {(0, 0): 1, (2, 2): -1})]
    p = tmp_path / "test.m"
    write_mathematica(p, md, nz, entries)
    text = p.read_text()
    assert 'Iref["m003"' in text
    assert "eta[0]" in text


def test_write_plain_text(tmp_path):
    nz = _make_nz_mock()
    entries = [([0], [Fraction(0)], {(0, 0): 1})]
    p = tmp_path / "test.txt"
    write_plain_text(p, "m003", nz, entries)
    text = p.read_text()
    assert "m003" in text
    assert "I(" in text


def test_write_json(tmp_path):
    md = _make_md_mock()
    nz = _make_nz_mock()
    easy = _make_easy_mock()
    entries = [([0], [Fraction(0)], {(0, 0): 1})]
    p = tmp_path / "test.json"
    write_json(p, md, easy, nz, entries)
    data = json.loads(p.read_text())
    assert data["manifold"] == "m003"
    assert len(data["sectors"]) == 1
    assert "coefficients" in data["sectors"][0]


def test_write_json_with_weyl(tmp_path):
    md = _make_md_mock()
    nz = _make_nz_mock()
    easy = _make_easy_mock()
    entries = [([0], [Fraction(0)], {(0, 0): 1})]
    weyl = _make_weyl_mock()
    p = tmp_path / "test.json"
    write_json(p, md, easy, nz, entries, weyl)
    data = json.loads(p.read_text())
    assert "weyl" in data
    assert data["weyl"]["all_symmetric"] is True


# ── § 3  Clipboard helpers ───────────────────────────────────────────

def test_clipboard_latex():
    entries = [([0], [Fraction(0)], {(0, 0): 1, (2, 2): -1})]
    text = clipboard_latex("m003", entries, 1)
    assert r"\begin{align*}" in text
    assert r"\eta_0" in text


def test_clipboard_plain_text():
    entries = [([0], [Fraction(0)], {(0, 0): 1})]
    text = clipboard_plain_text("m003", entries, 1)
    assert "I(" in text
