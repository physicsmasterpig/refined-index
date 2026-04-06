"""
utils/exporters.py — Export formatters and file writers (v2).

Two output types:
  1. **Full LaTeX report** — comprehensive, compilable .tex document
     with every piece of computed data.
  2. **Data files** — JSON and Mathematica for downstream analysis.

Public API
----------
Series formatters (low-level, shared)
    to_latex_series, to_latex_filled_series
    to_mathematica_series, to_mathematica_filled_series

File writers
    write_full_report      — comprehensive LaTeX .tex
    write_json             — structured JSON data
    write_mathematica      — Mathematica .m rules

Clipboard helpers
    clipboard_latex, clipboard_plain_text
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# ======================================================================
# § 1  Low-level monomial formatters
# ======================================================================

# ── LaTeX ─────────────────────────────────────────────────────────────

def _latex_q_factor(qq_pow: int) -> str:
    """LaTeX string for the q factor.  *qq_pow* is the doubled q-power."""
    if qq_pow == 0:
        return ""
    if qq_pow == 2:
        return "q"
    if qq_pow % 2 == 0:
        n = qq_pow // 2
        return f"q^{{{n}}}"
    return f"q^{{{qq_pow}/2}}"


def _latex_eta_factors_hard(key: tuple[int, ...], num_hard: int) -> str:
    r"""LaTeX factors for hard-edge η's.

    ``key[1+a]`` is the **doubled** exponent of η_a: true power = key[1+a]/2.
    """
    parts: list[str] = []
    for a in range(num_hard):
        exp2 = key[1 + a]
        if exp2 == 0:
            continue
        if exp2 == 2:
            parts.append(rf"\eta_{a}")
        elif exp2 == -2:
            parts.append(rf"\eta_{a}^{{-1}}")
        elif exp2 % 2 == 0:
            p = exp2 // 2
            parts.append(rf"\eta_{a}^{{{p}}}")
        else:
            parts.append(rf"\eta_{a}^{{{exp2}/2}}")
    return r" \, ".join(parts)


def _latex_eta_factors_cusp(
    key: tuple[int, ...], num_hard: int, num_cusp_eta: int
) -> str:
    r"""LaTeX factors for cusp η's (appear after hard η's in the key)."""
    parts: list[str] = []
    for ci in range(num_cusp_eta):
        pos = 1 + num_hard + ci
        if pos >= len(key):
            break
        cusp_exp = key[pos]
        if cusp_exp == 0:
            continue
        if num_cusp_eta == 1:
            label = "V_0"
        else:
            label = f"V_{ci}"
        coeff = 2 * cusp_exp
        if coeff == 1:
            parts.append(rf"\eta^{{{label}}}")
        elif coeff == -1:
            parts.append(rf"\eta^{{-{label}}}")
        else:
            parts.append(rf"\eta^{{{coeff}{label}}}")
    return r" \, ".join(parts)


def _latex_monomial(
    key: tuple[int, ...],
    coeff: int,
    num_hard: int,
    num_cusp_eta: int = 0,
) -> str:
    """Full LaTeX monomial string (with coefficient, q, all η's)."""
    q_str = _latex_q_factor(key[0])
    hard_str = _latex_eta_factors_hard(key, num_hard)
    cusp_str = (
        _latex_eta_factors_cusp(key, num_hard, num_cusp_eta)
        if num_cusp_eta > 0
        else ""
    )
    parts = [s for s in (q_str, hard_str, cusp_str) if s]
    body = r" \, ".join(parts)

    if not body:
        return str(coeff)
    if coeff == 1:
        return body
    if coeff == -1:
        return f"-{body}"
    return rf"{coeff} \, {body}"


# ── Mathematica ───────────────────────────────────────────────────────

def _math_q_factor(qq_pow: int) -> str:
    if qq_pow == 0:
        return ""
    if qq_pow == 2:
        return "q"
    if qq_pow % 2 == 0:
        return f"q^{qq_pow // 2}"
    return f"q^({qq_pow}/2)"


def _math_eta_hard(key: tuple[int, ...], num_hard: int) -> str:
    parts: list[str] = []
    for a in range(num_hard):
        exp2 = key[1 + a]
        if exp2 == 0:
            continue
        p = exp2 // 2 if exp2 % 2 == 0 else f"({exp2}/2)"
        if p == 1:
            parts.append(f"eta[{a}]")
        elif p == -1:
            parts.append(f"eta[{a}]^(-1)")
        else:
            parts.append(f"eta[{a}]^{p}")
    return " ".join(parts)


def _math_eta_cusp(
    key: tuple[int, ...], num_hard: int, num_cusp_eta: int
) -> str:
    parts: list[str] = []
    for ci in range(num_cusp_eta):
        pos = 1 + num_hard + ci
        if pos >= len(key):
            break
        cusp_exp = key[pos]
        if cusp_exp == 0:
            continue
        if cusp_exp == 1:
            parts.append(f"etaCusp[{ci}]")
        elif cusp_exp == -1:
            parts.append(f"etaCusp[{ci}]^(-1)")
        else:
            parts.append(f"etaCusp[{ci}]^{cusp_exp}")
    return " ".join(parts)


def _math_monomial(
    key: tuple[int, ...],
    coeff: int,
    num_hard: int,
    num_cusp_eta: int = 0,
) -> str:
    q_str = _math_q_factor(key[0])
    hard_str = _math_eta_hard(key, num_hard)
    cusp_str = (
        _math_eta_cusp(key, num_hard, num_cusp_eta)
        if num_cusp_eta > 0
        else ""
    )
    parts = [s for s in (q_str, hard_str, cusp_str) if s]
    body = " ".join(parts)

    if not body:
        return str(coeff)
    if coeff == 1:
        return body
    if coeff == -1:
        return f"-{body}"
    return f"{coeff} {body}"


# ======================================================================
# § 2  Series formatters
# ======================================================================

def to_latex_series(result: dict, num_hard: int) -> str:
    r"""Format a refined index (unfilled) as a LaTeX sum."""
    if not result:
        return "0"
    terms: list[str] = []
    for key in sorted(result.keys()):
        c = result[key]
        if c == 0:
            continue
        m = _latex_monomial(key, c, num_hard)
        if terms and not m.startswith("-"):
            terms.append(f"+ {m}")
        else:
            terms.append(m)
    return " ".join(terms) if terms else "0"


def to_latex_filled_series(
    series: dict,
    num_hard: int,
    num_cusp_eta: int = 0,
    max_q_terms: int | None = None,
) -> str:
    r"""Format a filled refined index as LaTeX."""
    if not series:
        return "0"

    if max_q_terms is not None:
        max_qq = 2 * max_q_terms
        filtered = {k: v for k, v in series.items() if k[0] <= max_qq}
        truncated = len(series) > len(filtered)
        series = filtered
    else:
        truncated = False

    terms: list[str] = []
    for key in sorted(series.keys()):
        c = series[key]
        if c == 0:
            continue
        m = _latex_monomial(key, c, num_hard, num_cusp_eta)
        if terms and not m.startswith("-"):
            terms.append(f"+ {m}")
        else:
            terms.append(m)
    text = " ".join(terms) if terms else "0"
    if truncated:
        text += r" + \cdots"
    return text


def to_mathematica_series(result: dict, num_hard: int) -> str:
    """Format a refined index (unfilled) as a Mathematica expression."""
    if not result:
        return "0"
    terms: list[str] = []
    for key in sorted(result.keys()):
        c = result[key]
        if c == 0:
            continue
        m = _math_monomial(key, c, num_hard)
        if terms and not m.startswith("-"):
            terms.append(f"+ {m}")
        else:
            terms.append(m)
    return " ".join(terms) if terms else "0"


def to_mathematica_filled_series(
    series: dict, num_hard: int, num_cusp_eta: int = 0
) -> str:
    """Format a filled refined index as a Mathematica expression."""
    if not series:
        return "0"
    terms: list[str] = []
    for key in sorted(series.keys()):
        c = series[key]
        if c == 0:
            continue
        m = _math_monomial(key, c, num_hard, num_cusp_eta)
        if terms and not m.startswith("-"):
            terms.append(f"+ {m}")
        else:
            terms.append(m)
    return " ".join(terms) if terms else "0"


# ======================================================================
# § 3  Charge label helper
# ======================================================================

def _charge_label(m_ext: Sequence, e_ext: Sequence, latex: bool = False) -> str:
    """Compact label like ``m=(0,1), e=(1/2,0)``."""
    def _fmt(seq: Sequence) -> str:
        return ", ".join(str(x) for x in seq)
    m_str = _fmt(m_ext)
    e_str = _fmt(e_ext)
    if latex:
        return rf"\mathbf{{m}}=({m_str}),\;\mathbf{{e}}=({e_str})"
    return f"m=({m_str}), e=({e_str})"


# ======================================================================
# § 4  LaTeX helpers
# ======================================================================

def _frac_tex(v) -> str:
    """Format a number as LaTeX: 0, 1, -2, 1/2, -3/2."""
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(int(f))
    sign = "-" if f < 0 else ""
    return rf"{sign}\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def _int_or_frac_tex(v) -> str:
    """Format float that might be half-integer."""
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(int(f))
    sign = "-" if f < 0 else ""
    return rf"{sign}\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def _matrix_tex(mat: np.ndarray, env: str = "pmatrix") -> str:
    """Render a numpy matrix as LaTeX pmatrix."""
    rows = []
    for i in range(mat.shape[0]):
        entries = []
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if isinstance(v, (float, np.floating)):
                entries.append(_int_or_frac_tex(v))
            elif isinstance(v, Fraction):
                entries.append(_frac_tex(v))
            else:
                entries.append(str(int(v)))
        rows.append(" & ".join(entries))
    body = " \\\\\n".join(rows)
    return rf"\begin{{{env}}}" + "\n" + body + "\n" + rf"\end{{{env}}}"


def _row_label(i: int, n: int, r: int, num_hard: int) -> str:
    """Human-readable row label for g_NZ."""
    if i < r:
        return rf"\mu_{{{i}}}"
    elif i < r + num_hard:
        return rf"\text{{H}}_{{{i - r}}}"
    elif i < n:
        return rf"\text{{E}}_{{{i - r - num_hard}}}"
    elif i < n + r:
        return rf"\lambda_{{{i - n}}}/2"
    else:
        return rf"\Gamma_{{{i - n - r}}}"


def _fmt_linear_combination(terms: list) -> str:
    """Format a list of (Fraction_coeff, var_str) as a LaTeX linear combination."""
    from fractions import Fraction
    if not terms:
        return "0"
    parts = []
    for idx_t, (coeff, var) in enumerate(terms):
        f = Fraction(coeff).limit_denominator(1000)
        if f == 1:
            c_str = ""
        elif f == -1:
            c_str = "-"
        elif f.denominator == 1:
            c_str = str(int(f))
        else:
            sign = "-" if f < 0 else ""
            c_str = rf"{sign}\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"
        if idx_t == 0:
            if c_str == "":
                parts.append(var)
            elif c_str == "-":
                parts.append(f"-{var}")
            else:
                parts.append(f"{c_str}{var}")
        else:
            if f > 0:
                if c_str == "":
                    parts.append(f"+ {var}")
                else:
                    parts.append(f"+ {c_str}{var}")
            else:
                if c_str == "-":
                    parts.append(f"- {var}")
                else:
                    # remove leading minus
                    parts.append(f"- {c_str[1:]}{var}")
    return " ".join(parts)


# ======================================================================
# § 5  Full LaTeX Report
# ======================================================================

_REPORT_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{hyperref}
\usepackage{xcolor}

\definecolor{pass}{HTML}{2ea043}
\definecolor{fail}{HTML}{cf222e}
\definecolor{warn}{HTML}{d4880a}

\newcommand{\Iref}{I^{\mathrm{ref}}}
\newcommand{\Ifill}{I^{\mathrm{ref,filled}}}
\newcommand{\IDelta}{I_{\Delta}}
\let\oldcheckmark\checkmark
\renewcommand{\checkmark}{\textcolor{pass}{\bfseries\oldcheckmark}}
\newcommand{\crossmark}{\textcolor{fail}{\bfseries\sffamily X}}
\newcommand{\passmark}{\textcolor{pass}{PASS}}
\newcommand{\failmark}{\textcolor{fail}{FAIL}}

\begin{document}
"""

_REPORT_POSTAMBLE = r"""
\end{document}
"""


def write_full_report(
    path: Path | str,
    manifold_data: Any,
    easy_result: Any,
    nz_data: Any,
    entries: list,
    weyl_result: Any = None,
    dehn_results: list | None = None,
    q_order_half: int | None = None,
) -> None:
    """Write a comprehensive, compilable LaTeX report.

    Parameters
    ----------
    path : Path or str
    manifold_data : ManifoldData
    easy_result : EasyEdgeResult
    nz_data : NeumannZagierData
    entries : list of (m_ext, e_ext, result) triples
    weyl_result : WeylCheckResult or None
    dehn_results : list of TransformedFillResult / MultiCuspFillResult or None
    q_order_half : int or None
    """
    path = Path(path)


    md = manifold_data
    nz = nz_data
    ps = easy_result
    n = nz.n
    r = nz.r

    L: list[str] = []  # accumulate lines
    L.append(_REPORT_PREAMBLE)

    # ── Title ──
    L.append(rf"\title{{Refined 3D Index — Full Report \\ \large {_tex_escape(md.name)}}}")
    L.append(r"\author{Generated by \texttt{manifold-index}}")
    L.append(r"\date{\today}")
    L.append(r"\maketitle")
    L.append(r"\tableofcontents")
    L.append(r"\newpage")
    L.append("")

    # ────────────────────────────────────────────────────────
    # Section 1: Manifold Overview
    # ────────────────────────────────────────────────────────
    L.append(r"\section{Manifold Overview}")
    L.append(rf"Manifold name: \textbf{{{_tex_escape(md.name)}}}")
    L.append("")
    L.append(r"\begin{itemize}")
    L.append(rf"  \item Number of ideal tetrahedra: $n = {n}$")
    L.append(rf"  \item Number of cusps: $r = {r}$")
    L.append(rf"  \item Internal edges: $n - r = {n - r}$ "
             rf"(easy: {nz.num_easy}, hard: {nz.num_hard})")
    if q_order_half is not None:
        L.append(rf"  \item Computation order: $q^{{{q_order_half}/2}}$")
    # Boundary curve range from entries
    if entries:
        all_m = [v for m_ext, _, _ in entries for v in m_ext]
        all_e = [v for _, e_ext, _ in entries for v in e_ext]
        m_vals = sorted(set(Fraction(v).limit_denominator(1000) for v in all_m))
        e_vals = sorted(set(Fraction(v).limit_denominator(1000) for v in all_e))
        m_range_str = ", ".join(_int_or_frac_tex(v) for v in m_vals)
        e_range_str = ", ".join(_int_or_frac_tex(v) for v in e_vals)
        L.append(rf"  \item Boundary curve range: "
                 rf"$m \in \{{{m_range_str}\}}$, "
                 rf"$e \in \{{{e_range_str}\}}$")
    L.append(r"\end{itemize}")
    L.append("")


    # ────────────────────────────────────────────────────────
    # Section 2: Ideal Triangulation (SnaPy Raw Data)
    # ────────────────────────────────────────────────────────
    L.append(r"\section{Ideal Triangulation (SnaPy Data)}")
    L.append(r"Gluing equation matrix from SnaPy, shape "
             rf"$({n + 2*r} \times {3*n})$.")
    L.append("")

    if md.gluing_matrix is not None:
        G = md.gluing_matrix
        _append_gluing_table(L, G[:n], n, "Edge", range(n))
        L.append("")

        cusp_labels = []
        _alpha_beta = (r"\alpha", r"\beta")
        for k in range(r):
            if r == 1:
                cusp_labels.append(rf"${_alpha_beta[0]}$")
                cusp_labels.append(rf"${_alpha_beta[1]}$")
            else:
                cusp_labels.append(rf"${_alpha_beta[0]}_{{{k}}}$")
                cusp_labels.append(rf"${_alpha_beta[1]}_{{{k}}}$")
        _append_gluing_table(L, G[n:n+2*r], n, "Cusp", range(2*r), cusp_labels)
        L.append("")

    # ────────────────────────────────────────────────────────
    # Section 3: Edge Classification
    # ────────────────────────────────────────────────────────
    L.append(r"\section{Edge Classification}")

    if ps is not None:
        L.append(rf"Easy edges: {len(ps.all_easy)} total, "
                 rf"{ps.num_independent_easy} independent. "
                 rf"Hard edges: {len(ps.hard_padding)}.")
        L.append("")

        if ps.hard_padding:
            L.append(r"\begin{longtable}{c l l}")
            L.append(r"\toprule")
            L.append(r"\multicolumn{3}{l}{\textbf{Hard Edges}} \\ \midrule")
            L.append(r"Index & Symbolic Equation & Triplets $(f_i, g_i, h_i)$ \\ \midrule")
            L.append(r"\endhead")
            for idx, hedge in enumerate(ps.hard_padding):
                triplets = _edge_triplets_tex(hedge, n)
                sym_eq = _edge_equation_tex(hedge, n)
                L.append(rf"H$_{{{idx}}}$ & ${sym_eq} = 2$ & ${triplets}$ \\")
            L.append(r"\bottomrule")
            L.append(r"\end{longtable}")
            L.append("")

        if ps.all_easy:
            L.append(r"\begin{longtable}{c c l l}")
            L.append(r"\toprule")
            L.append(r"\multicolumn{4}{l}{\textbf{Easy Edges}} \\ \midrule")
            L.append(r"Index & Independent? & Symbolic Equation & Triplets $(f_i, g_i, h_i)$ \\ \midrule")
            L.append(r"\endhead")
            for idx, edge in enumerate(ps.all_easy):
                is_indep = idx in ps.independent_easy_indices
                mark = r"\checkmark" if is_indep else ""
                triplets = _edge_triplets_tex(edge, n)
                sym_eq = _edge_equation_tex(edge, n)
                L.append(rf"E$_{{{idx}}}$ & {mark} & ${sym_eq} = 2$ & ${triplets}$ \\")
            L.append(r"\bottomrule")
            L.append(r"\end{longtable}")
            L.append("")

    # ────────────────────────────────────────────────────────
    # Section 4: Neumann-Zagier Data
    # ────────────────────────────────────────────────────────
    L.append(r"\section{Neumann--Zagier Data}")

    # g_NZ matrix with row labels
    L.append(r"$g_{\text{NZ}} \in \mathrm{Sp}(2n, \mathbb{Q})$, "
             rf"${2*n} \times {2*n}$, column ordering "
             r"$(Z_1, \ldots, Z_n, Z_1'', \ldots, Z_n'')$.")
    L.append(r"{\small")
    L.append(r"\[")
    L.append(r"g_{\text{NZ}} = \left(\begin{array}{" + "r" * (2*n) + "}")
    for i in range(2*n):
        row_cells = []
        for j in range(2*n):
            row_cells.append(_int_or_frac_tex(nz.g_NZ[i, j]))
        label = _row_label(i, n, r, nz.num_hard)
        row_str = " & ".join(row_cells)
        L.append(rf"  {row_str} \\  %\; {label}")
    L.append(r"\end{array}\right)")
    L.append(r"\]")
    L.append(r"}")
    L.append("")

    # Row labels legend
    L.append(r"Row labels (right margin): ")
    labels = []
    for i in range(2*n):
        labels.append(f"${_row_label(i, n, r, nz.num_hard)}$")
    L.append(", ".join(labels) + ".")
    L.append("")

    # Affine shifts
    nu_x_str = ", ".join(str(int(v)) for v in nz.nu_x)
    nu_p_str = ", ".join(_int_or_frac_tex(v) for v in nz.nu_p)
    L.append(rf"Affine shifts: $\nu_x = ({nu_x_str})$, "
             rf"$\nu_p = ({nu_p_str})$.")
    L.append("")

    # ────────────────────────────────────────────────────────
    # Section 5: 3D Index Formula
    # ────────────────────────────────────────────────────────
    L.append(r"\section{3D Index Formula}")
    L.append(r"The 3D index is computed via:")
    L.append(r"\[")
    L.append(r"  I(\mathbf{m}_{\text{ext}}, \mathbf{e}_{\text{ext}}) = "
             r"\sum_{\mathbf{e}_{\text{int}} \in (\tfrac{1}{2}\mathbb{Z})^{n-r}} "
             r"(-q^{1/2})^{\mathbf{m} \cdot \nu_p \;-\; \mathbf{e} \cdot \nu_x} "
             r"\;\prod_{a=1}^{n} \IDelta(m_a, e_a)")
    L.append(r"\]")
    L.append(r"where $\kappa = (\mathbf{m}_{\text{ext}},\, \mathbf{0}^{n-r},\, "
             r"\mathbf{e}_{\text{ext}},\, \mathbf{e}_{\text{int}})$ "
             r"and the local charges are:")
    L.append(r"\[")
    L.append(r"  m_a = (g_{\text{NZ}}^{-1}\kappa)_a, \qquad "
             r"  e_a = (g_{\text{NZ}}^{-1}\kappa)_{n+a}")
    L.append(r"\]")
    L.append(r"Only terms with integer $(m_a, e_a)$ contribute.")
    L.append("")

    # Explicit local charges
    L.append(r"\noindent\textbf{Explicit Local Charges.}")
    g_inv = nz.g_NZ_inv()  # (2n, 2n) Fraction array
    for a in range(n):
        tet_label = rf"\Delta_{{{a+1}}}"
        m_terms = []
        e_terms = []
        # External m contributions (columns 0..r-1)
        for k in range(r):
            c = g_inv[a, k]
            if c != 0:
                m_terms.append((c, rf"m_{{{k}}}"))
            c2 = g_inv[n + a, k]
            if c2 != 0:
                e_terms.append((c2, rf"m_{{{k}}}"))
        # Columns r..n-1 are zero (internal m forced to 0), skip
        # External e contributions (columns n..n+r-1)
        for k in range(r):
            c = g_inv[a, n + k]
            if c != 0:
                m_terms.append((c, rf"e_{{{k}}}"))
            c2 = g_inv[n + a, n + k]
            if c2 != 0:
                e_terms.append((c2, rf"e_{{{k}}}"))
        # Internal e contributions (columns n+r..2n-1)
        for k in range(n - r):
            c = g_inv[a, n + r + k]
            if c != 0:
                m_terms.append((c, rf"e_{{\text{{int}},{k}}}"))
            c2 = g_inv[n + a, n + r + k]
            if c2 != 0:
                e_terms.append((c2, rf"e_{{\text{{int}},{k}}}"))

        m_str = _fmt_linear_combination(m_terms)
        e_str = _fmt_linear_combination(e_terms)
        L.append(rf"${tet_label}$: $\;m_{{{a+1}}} = {m_str}$, "
                 rf"$\;e_{{{a+1}}} = {e_str}$")
        L.append("")

    # Tetrahedron index formula
    L.append(r"The tetrahedron index $\IDelta(m, e)$ is given by")
    L.append(r"\[")
    L.append(r"  I_t(m, e) = \sum_{k=\max(0,-e)}^{\infty} "
             r"\frac{(-1)^k \, q^{k(k+1)/2 - (2k+e)m/2}}"
             r"{\prod_{j=1}^{k}(1-q^j) \,\prod_{j=1}^{k+e}(1-q^j)}")
    L.append(r"\]")
    L.append(r"with $\IDelta(m,e) = (-q^{1/2})^m \cdot I_t(-m-e, m)$ "
             r"when $m + e \ge 0$, else $\IDelta(m,e) = I_t(m, e)$.")
    L.append("")

    # ────────────────────────────────────────────────────────
    # Section 6: Refined Index Results
    # ────────────────────────────────────────────────────────
    L.append(r"\section{Refined Index Results}")
    if q_order_half is not None:
        L.append(rf"Series truncated at $q^{{{q_order_half}/2}}$ "
                 rf"(i.e.\ $N_{{\max}} = {q_order_half // 2}$).")
    L.append(rf"Number of computed sectors: {len(entries)}.")
    n_nonzero = sum(1 for _, _, res in entries if res)
    L.append(rf"Non-zero sectors: {n_nonzero}.")
    L.append("")

    # Group by m_ext for readability
    if entries:
        L.append(r"\begin{align*}")
        for idx, (m_ext, e_ext, result) in enumerate(entries):
            label = _charge_label(m_ext, e_ext, latex=True)
            tex = to_latex_series(result, nz.num_hard)
            if not result:
                tex = "0"
            end = r" \\" if idx < len(entries) - 1 else ""
            L.append(rf"  \Iref({label}) &= {tex}{end}")
        L.append(r"\end{align*}")
        L.append("")

    # ────────────────────────────────────────────────────────
    # Section 7: Weyl Symmetry
    # ────────────────────────────────────────────────────────
    L.append(r"\section{Weyl Symmetry}")

    if weyl_result is None:
        L.append(r"No Weyl symmetry data available.")
    else:
        L.append(r"Define the Weyl-shifted index:")
        L.append(r"\[")
        L.append(r"  f(\eta_j; m, e) = \eta_j^{\sum_I (a_{j,I} \cdot e_I + b_{j,I} \cdot m_I)} "
                 r"\cdot \Iref(m, e)")
        L.append(r"\]")
        L.append(r"Weyl symmetry requires $f(m, e) = f(-m, -e)$ for each sector pair, "
                 r"where $f$ denotes the Weyl-shifted index above.")
        L.append("")

        ab = weyl_result.ab
        if ab is not None:
            L.append(r"\begin{center}")
            L.append(r"\begin{tabular}{c c c c c}")
            L.append(r"\toprule")
            L.append(r"Hard edge $j$ & $a_j$ & $b_j$ & $a_j \in \mathbb{Z}$? "
                     r"& $2b_j \in \mathbb{Z}$? \\ \midrule")
            for j in range(ab.num_hard):
                if ab.cusp_columns is not None:
                    a_str = "(" + ", ".join(
                        _frac_tex(col.a[j]) for col in ab.cusp_columns
                    ) + ")"
                    b_str = "(" + ", ".join(
                        _frac_tex(col.b[j]) for col in ab.cusp_columns
                    ) + ")"
                    a_int_ok = all(col.a[j].denominator == 1 for col in ab.cusp_columns)
                    b_hint_ok = all((2 * col.b[j]).denominator == 1 for col in ab.cusp_columns)
                else:
                    a_str = _frac_tex(ab.a[j])
                    b_str = _frac_tex(ab.b[j])
                    a_int_ok = ab.a[j].denominator == 1
                    b_hint_ok = (2 * ab.b[j]).denominator == 1
                a_int = r"\checkmark" if a_int_ok else r"\crossmark"
                b_hint = r"\checkmark" if b_hint_ok else r"\crossmark"
                L.append(rf"  {j} & ${a_str}$ & ${b_str}$ & {a_int} & {b_hint} \\")
            L.append(r"\bottomrule")
            L.append(r"\end{tabular}")
            L.append(r"\end{center}")
            L.append("")

            if ab.is_valid:
                L.append(r"{\color{pass} $a \in \mathbb{Z}^k$, "
                         r"$b \in (\tfrac{1}{2}\mathbb{Z})^k$ — "
                         r"Dehn filling compatible.}")
            else:
                L.append(r"{\color{fail} $a \notin \mathbb{Z}^k$ or "
                         r"$2b \notin \mathbb{Z}^k$ — "
                         r"Dehn filling \textbf{not} compatible.}")
            L.append("")

            # Edge compatibility detail
            if hasattr(ab, 'edge_compatible'):
                L.append(r"Edge compatibility: "
                         r"each hard edge $j$ must have "
                         r"$a_j \in \mathbb{Z}$; incompatible edges are collapsed ($\eta_j = 1$).")
                parts = []
                for j in range(ab.num_hard):
                    compat = ab.a_is_integer[j]
                    status = r"\passmark" if compat else r"\failmark"
                    parts.append(rf"H$_{j}$: {status}")
                L.append(" ".join(parts))
                L.append("")

        # Weyl symmetry check per sector
        if weyl_result.weyl_symmetric:
            n_sym = sum(weyl_result.weyl_symmetric.values())
            n_total = len(weyl_result.weyl_symmetric)
            n_fail = n_total - n_sym
            status = r"\passmark" if weyl_result.all_weyl_symmetric else r"\failmark"
            L.append(rf"Weyl symmetry $f(m,e)=f(-m,-e)$: {n_sym}/{n_total} sectors pass {status}")
            if n_fail > 0 and not weyl_result.all_weyl_symmetric:
                L.append(rf" ({n_fail} failures may be due to missing partner sectors "
                         r"or finite truncation at large $|m|$, $|e|$).")
            L.append("")

        # Adjoint projection
        adj = weyl_result.adjoint if hasattr(weyl_result, 'adjoint') else None
        if adj is not None:
            L.append(r"The adjoint $\mathrm{su}(2)$ projection (eq.~2.59--2.61) "
                     r"integrates out $\eta$ and cusp "
                     r"fugacities against the Haar measure "
                     r"weighted by the adjoint character; "
                     r"the result must equal $-1$ per filled cusp.")
            L.append("")

            if adj.c_e:
                L.append(r"Intermediate $c_e$ values ($(q^1, \eta^0)$ coefficient "
                         r"of $\Iref(m=0, e)$):")
                L.append(r"\begin{center}")
                L.append(r"\begin{tabular}{c c}")
                L.append(r"\toprule")
                L.append(r"$e$ & $c_e$ \\ \midrule")
                for e_val in sorted(adj.c_e.keys()):
                    L.append(rf"  ${_frac_tex(e_val)}$ & ${adj.c_e[e_val]}$ \\")
                L.append(r"\bottomrule")
                L.append(r"\end{tabular}")
                L.append(r"\end{center}")
                L.append("")

            if adj.missing_e:
                missing_str = ", ".join(_frac_tex(e) for e in adj.missing_e)
                L.append(rf"Missing $e$ values: ${missing_str}$.")
                L.append("")

            if adj.projected_value is not None:
                status = r"\passmark" if adj.is_pass else r"\failmark"
                L.append(rf"Projected value: ${adj.projected_value}$ "
                         rf"(expected $-1$): {status}")
            else:
                L.append(r"Projected value: could not compute (missing entries).")
            L.append("")

    # ────────────────────────────────────────────────────────
    # Section 8: Dehn Filling
    # ────────────────────────────────────────────────────────
    if dehn_results:
        from manifold_index.app.workers import (
            TransformedFillResult, MultiCuspFillResult,
        )
        from manifold_index.core.neumann_zagier import apply_cusp_basis_change
        from manifold_index.core.refined_dehn_filling import hj_continued_fraction

        L.append(r"\section{Dehn Filling}")
        L.append("")

        fill_idx = 0
        for res in dehn_results:
            fill_idx += 1
            if isinstance(res, TransformedFillResult):
                _append_single_cusp_filling(
                    L, res, nz, n, r, fill_idx,
                )
            elif isinstance(res, MultiCuspFillResult):
                _append_multi_cusp_filling(
                    L, res, nz, n, r, fill_idx,
                )

    # ── End ──
    L.append(_REPORT_POSTAMBLE)

    path.write_text("\n".join(L), encoding="utf-8")


# ── Dehn filling sub-sections ─────────────────────────────────────────

def _append_single_cusp_filling(
    L: list[str],
    res: Any,
    nz: Any,
    n: int,
    r: int,
    fill_idx: int,
) -> None:
    """Append a single-cusp Dehn filling section."""
    from manifold_index.core.neumann_zagier import apply_cusp_basis_change
    from manifold_index.core.refined_dehn_filling import hj_continued_fraction

    L.append(rf"\subsection{{Filling {fill_idx}: Cusp {res.cusp_idx}, "
             rf"slope $({res.P_user}, {res.Q_user})$}}")
    L.append("")

    # Slope info
    L.append(r"\subsubsection{Slope Transformation}")
    L.append(r"\begin{itemize}")
    L.append(rf"  \item User slope (physical): "
             rf"$P = {res.P_user},\; Q = {res.Q_user}$ "
             rf"(cycle ${res.P_user} \cdot M + {res.Q_user} \cdot L$)")
    L.append(rf"  \item Non-closable cycle: "
             rf"$\gamma = {res.P_nc} \cdot M + {res.Q_nc} \cdot L$ "
             rf"(slope ${res.P_nc}/{res.Q_nc}$)")
    L.append(rf"  \item SL$(2,\mathbb{{Z}})$ complement: "
             rf"$\delta = {res.R} \cdot M + {res.S} \cdot L$")
    L.append(rf"  \item Verification: $\det \begin{{pmatrix}} "
             rf"{res.P_nc} & {res.R} \\ {res.Q_nc} & {res.S} "
             rf"\end{{pmatrix}} = "
             rf"{res.P_nc * res.S - res.Q_nc * res.R}$ (should be $+1$)")
    L.append(rf"  \item User slope in $(\gamma, \delta)$ basis: "
             rf"$(p, q) = ({res.p}, {res.q})$")
    L.append(r"\end{itemize}")
    L.append("")

    # HJ continued fraction
    try:
        hj = hj_continued_fraction(res.p, res.q)
        hj_str = ", ".join(str(k) for k in hj)
        L.append(r"\subsubsection{Hirzebruch--Jung Continued Fraction}")
        L.append(rf"$p/q = {res.p}/{res.q}$: $[{hj_str}]$, "
                 rf"$\ell = {len(hj)}$.")
        L.append("")
    except Exception:
        pass

    # NZ matrix after basis change
    L.append(r"\subsubsection{$g_{\text{NZ}}$ After Basis Change}")
    try:
        nz_changed = apply_cusp_basis_change(nz, res.cusp_idx, res.P_nc, res.Q_nc)
        L.append(r"After replacing cusp basis to the NC cycle "
                 rf"$({res.P_nc}, {res.Q_nc})$:")
        L.append(r"{\small")
        L.append(r"\[")
        L.append(r"g_{\text{NZ}}' = " + _matrix_tex(nz_changed.g_NZ))
        L.append(r"\]")
        L.append(r"}")
        # Affine shifts
        nu_x_str = ", ".join(str(int(v)) for v in nz_changed.nu_x)
        nu_p_str = ", ".join(_int_or_frac_tex(v) for v in nz_changed.nu_p)
        L.append(rf"$\nu_x' = ({nu_x_str})$, $\nu_p' = ({nu_p_str})$.")
        L.append("")
    except Exception as exc:
        L.append(rf"Could not compute basis change: {_tex_escape(str(exc))}")
        L.append("")

    # Physical Weyl vectors
    if res.weyl_a_phys is not None:
        L.append(r"\subsubsection{Physical Weyl Vectors}")
        a_str = ", ".join(_frac_tex(v) for v in res.weyl_a_phys)
        b_str = ", ".join(_frac_tex(v) for v in res.weyl_b_phys) if res.weyl_b_phys else "N/A"
        L.append(rf"$\mathbf{{a}}_{{\text{{phys}}}} = ({a_str})$")
        L.append("")
        L.append(rf"$\mathbf{{b}}_{{\text{{phys}}}} = ({b_str})$")
        L.append("")

        # Check edge compatibility
        incompatible = []
        for j, a_val in enumerate(res.weyl_a_phys):
            if Fraction(a_val).denominator != 1:
                incompatible.append(j)
        if incompatible:
            idx_str = ", ".join(f"H$_{j}$" for j in incompatible)
            L.append(rf"Incompatible edges (will be collapsed): {idx_str}.")
        else:
            L.append(r"All hard edges are compatible ($a_j \in \mathbb{Z}$).")
        L.append("")

    # Filled results
    L.append(r"\subsubsection{Filled Refined Index}")
    for m_other, e_other, fr in res.fill_results:
        label = _charge_label(m_other, e_other, latex=True) if m_other else ""
        tex = to_latex_filled_series(fr.series, fr.num_hard, fr.num_cusp_eta)

        if label:
            L.append(rf"\paragraph{{{label}}}")

        hj_str = ", ".join(str(k) for k in fr.hj_ks)
        L.append(rf"HJ: $[{hj_str}]$, $\ell = {len(fr.hj_ks)}$, "
                 rf"kernel terms: {fr.n_kernel_terms}.")
        L.append(r"\[")
        L.append(rf"  \Ifill_{{{res.P_user}/{res.Q_user}}} = {tex}")
        L.append(r"\]")
        L.append("")


def _append_multi_cusp_filling(
    L: list[str],
    res: Any,
    nz: Any,
    n: int,
    r: int,
    fill_idx: int,
) -> None:
    """Append a multi-cusp Dehn filling section."""
    from manifold_index.core.neumann_zagier import apply_cusp_basis_change
    from manifold_index.core.refined_dehn_filling import hj_continued_fraction

    cusps_str = ", ".join(str(ci.cusp_idx) for ci in res.cusp_info)
    L.append(rf"\subsection{{Filling {fill_idx}: Multi-cusp ({cusps_str})}}")
    L.append("")

    for ci in res.cusp_info:
        L.append(rf"\subsubsection{{Cusp {ci.cusp_idx}: "
                 rf"slope $({ci.P_user}, {ci.Q_user})$}}")
        L.append(r"\begin{itemize}")
        L.append(rf"  \item NC cycle: $({ci.P_nc}, {ci.Q_nc})$")
        L.append(rf"  \item Complement: $({ci.R}, {ci.S})$")
        L.append(rf"  \item $(p, q) = ({ci.p}, {ci.q})$")
        L.append(r"\end{itemize}")
        L.append("")

        # Physical Weyl
        if ci.weyl_a_phys is not None:
            a_str = ", ".join(_frac_tex(v) for v in ci.weyl_a_phys)
            b_str = ", ".join(_frac_tex(v) for v in ci.weyl_b_phys) if ci.weyl_b_phys else "N/A"
            L.append(rf"$\mathbf{{a}}_{{\text{{phys}}}} = ({a_str})$, "
                     rf"$\mathbf{{b}}_{{\text{{phys}}}} = ({b_str})$.")
            L.append("")

        # NZ after basis change
        try:
            nz_changed = apply_cusp_basis_change(nz, ci.cusp_idx, ci.P_nc, ci.Q_nc)
            L.append(r"{\small")
            L.append(r"\[")
            L.append(rf"g_{{\text{{NZ}}}}'_{{\text{{cusp {ci.cusp_idx}}}}} = "
                     + _matrix_tex(nz_changed.g_NZ))
            L.append(r"\]")
            L.append(r"}")
            L.append("")
            nz = nz_changed  # chain for next cusp
        except Exception:
            pass

    # Combined result
    fr_obj = res.fill_result
    if hasattr(fr_obj, "series"):
        L.append(r"\subsubsection{Combined Filled Result}")
        tex = to_latex_filled_series(
            fr_obj.series,
            getattr(fr_obj, "num_hard", 0),
            getattr(fr_obj, "num_cusp_eta", 0),
        )
        L.append(r"\[")
        L.append(rf"  \Ifill = {tex}")
        L.append(r"\]")
    L.append("")


# ── Gluing table helper ───────────────────────────────────────────────

def _append_gluing_table(
    L: list[str],
    rows: np.ndarray,
    n: int,
    prefix: str,
    row_range,
    labels: list[str] | None = None,
) -> None:
    """Append a LaTeX longtable of gluing equation triplets."""
    L.append(r"\begin{longtable}{c" + " c" * n + "}")
    L.append(r"\toprule")
    header = prefix
    for j in range(n):
        header += rf" & $(Z_{{{j+1}}}, Z_{{{j+1}}}', Z_{{{j+1}}}'')$"
    header += r" \\ \midrule"
    L.append(header)
    L.append(r"\endhead")

    for idx, i in enumerate(row_range):
        if labels is not None:
            label = labels[idx]
        else:
            label = str(i)
        entries = [label]
        for j in range(n):
            a = int(rows[idx, 3 * j])
            b = int(rows[idx, 3 * j + 1])
            c = int(rows[idx, 3 * j + 2])
            entries.append(f"$({a}, {b}, {c})$")
        L.append(" & ".join(entries) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{longtable}")


def _edge_triplets_tex(edge: np.ndarray, n: int) -> str:
    """Format an edge vector (3n,) as a string of LaTeX triplets."""
    parts = []
    for j in range(n):
        a = int(edge[3 * j])
        b = int(edge[3 * j + 1])
        c = int(edge[3 * j + 2])
        parts.append(rf"({a},{b},{c})")
    return r"\;" .join(parts)


def _edge_equation_tex(edge: np.ndarray, n: int) -> str:
    r"""Convert a (3n,) gluing vector to a symbolic LaTeX equation.

    For tetrahedron j the triplet (a_j, b_j, c_j) contributes
    ``a_j Z_j + b_j Z_j' + c_j Z_j''``.  The equation is ``= 2``
    for an edge equation.
    """
    terms: list[tuple[int, str]] = []
    for j in range(n):
        a = int(edge[3 * j])
        b = int(edge[3 * j + 1])
        c = int(edge[3 * j + 2])
        if a != 0:
            terms.append((a, rf"Z_{{{j+1}}}"))
        if b != 0:
            terms.append((b, rf"Z_{{{j+1}}}'"))
        if c != 0:
            terms.append((c, rf"Z_{{{j+1}}}''"))

    if not terms:
        return "0"
    parts: list[str] = []
    for idx_t, (coeff, var) in enumerate(terms):
        if idx_t == 0:
            if coeff == 1:
                parts.append(var)
            elif coeff == -1:
                parts.append(f"-{var}")
            else:
                parts.append(f"{coeff}{var}")
        else:
            if coeff == 1:
                parts.append(f"+ {var}")
            elif coeff == -1:
                parts.append(f"- {var}")
            elif coeff > 0:
                parts.append(f"+ {coeff}{var}")
            else:
                parts.append(f"- {-coeff}{var}")
    return " ".join(parts)


def _tex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    for ch in ('_', '&', '%', '#', '{', '}'):
        s = s.replace(ch, '\\' + ch)
    return s


# ======================================================================
# § 6  JSON Data Writer
# ======================================================================

def write_json(
    path: Path | str,
    manifold_data: Any,
    easy_result: Any,
    nz_data: Any,
    entries: list,
    weyl_result: Any = None,
    dehn_results: list | None = None,
    q_order_half: int | None = None,
) -> None:
    """Write comprehensive JSON data file."""
    path = Path(path)
    md = manifold_data
    nz = nz_data
    ps = easy_result
    n = nz.n
    r = nz.r

    # ── Manifold info ──
    data: dict[str, Any] = {
        "manifold": md.name,
        "n": n,
        "r": r,
        "num_hard": nz.num_hard,
        "num_easy": nz.num_easy,
        "q_order_half": q_order_half,
    }

    # ── Gluing matrix ──
    if md.gluing_matrix is not None:
        data["gluing_matrix"] = md.gluing_matrix.tolist()

    # ── NZ matrix ──
    data["g_NZ"] = nz.g_NZ.tolist()
    data["nu_x"] = [int(v) for v in nz.nu_x]
    data["nu_p"] = [float(v) for v in nz.nu_p]
    S, g_inv_int = nz.g_NZ_inv_scaled()
    data["g_NZ_inv_scale"] = S
    data["g_NZ_inv_scaled"] = g_inv_int.tolist()
    data["is_symplectic"] = nz.is_symplectic()

    # ── Edge classification ──
    if ps is not None:
        data["easy_edges"] = {
            "count": len(ps.all_easy),
            "independent_count": ps.num_independent_easy,
            "independent_indices": list(ps.independent_easy_indices),
            "all_easy_vectors": [e.tolist() for e in ps.all_easy],
            "hard_vectors": [h.tolist() for h in ps.hard_padding],
        }

    # ── Refined index sectors ──
    sectors: list[dict] = []
    for m_ext, e_ext, result in entries:
        coefficients: dict[str, int] = {}
        for key, val in sorted(result.items()):
            coefficients[str(key)] = int(val) if isinstance(val, (int, Fraction)) else val
        sectors.append({
            "m": [int(x) if isinstance(x, (int, Fraction)) else x for x in m_ext],
            "e": [str(x) for x in e_ext],
            "coefficients": coefficients,
        })
    data["sectors"] = sectors

    # ── Weyl symmetry ──
    if weyl_result is not None:
        weyl_data: dict[str, Any] = {}
        ab = weyl_result.ab
        if ab is not None:
            weyl_data["a"] = [str(v) for v in ab.a]
            weyl_data["b"] = [str(v) for v in ab.b]
            weyl_data["a_is_integer"] = ab.a_is_integer
            weyl_data["b_is_half_integer"] = ab.b_is_half_integer
            weyl_data["is_valid"] = ab.is_valid
        if weyl_result.weyl_symmetric:
            weyl_data["symmetric_sectors"] = {
                str(k): v for k, v in weyl_result.weyl_symmetric.items()
            }
            weyl_data["all_symmetric"] = weyl_result.all_weyl_symmetric
        adj = getattr(weyl_result, 'adjoint', None)
        if adj is not None:
            weyl_data["adjoint"] = {
                "projected_value": adj.projected_value,
                "is_pass": adj.is_pass,
                "c_e": {str(k): v for k, v in adj.c_e.items()},
                "missing_e": [str(e) for e in adj.missing_e],
            }
        data["weyl"] = weyl_data

    # ── Dehn filling ──
    if dehn_results:
        data["dehn_filling"] = _json_dehn(dehn_results)

    class _FractionEncoder(json.JSONEncoder):
        def default(self, obj):
            from fractions import Fraction as _Fraction
            if isinstance(obj, _Fraction):
                return {"__fraction__": True, "n": obj.numerator, "d": obj.denominator}
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    path.write_text(json.dumps(data, indent=2, cls=_FractionEncoder), encoding="utf-8")


def _json_dehn(dehn_results: list) -> list:
    """Serialise Dehn filling results for JSON."""
    from manifold_index.app.workers import TransformedFillResult, MultiCuspFillResult

    out: list[dict] = []
    for res in dehn_results:
        if isinstance(res, TransformedFillResult):
            fills = []
            for m_other, e_other, fr in res.fill_results:
                fills.append({
                    "m_other": [str(x) for x in m_other],
                    "e_other": [str(x) for x in e_other],
                    "P": fr.P, "Q": fr.Q,
                    "hj_ks": fr.hj_ks,
                    "n_kernel_terms": fr.n_kernel_terms,
                    "num_hard": fr.num_hard,
                    "has_cusp_eta": fr.has_cusp_eta,
                    "num_cusp_eta": fr.num_cusp_eta,
                    "series": {str(k): int(v) for k, v in fr.series.items()},
                })
            entry = {
                "type": "single",
                "cusp_idx": res.cusp_idx,
                "P_user": res.P_user, "Q_user": res.Q_user,
                "P_nc": res.P_nc, "Q_nc": res.Q_nc,
                "R": res.R, "S": res.S,
                "p": res.p, "q": res.q,
                "fill_results": fills,
            }
            if res.weyl_a_phys is not None:
                entry["weyl_a_phys"] = [str(v) for v in res.weyl_a_phys]
            if res.weyl_b_phys is not None:
                entry["weyl_b_phys"] = [str(v) for v in res.weyl_b_phys]
            out.append(entry)
        elif isinstance(res, MultiCuspFillResult):
            cusp_entries = []
            for ci in res.cusp_info:
                ce: dict[str, Any] = {
                    "cusp_idx": ci.cusp_idx,
                    "P_user": ci.P_user, "Q_user": ci.Q_user,
                    "P_nc": ci.P_nc, "Q_nc": ci.Q_nc,
                    "R": ci.R, "S": ci.S,
                    "p": ci.p, "q": ci.q,
                }
                if ci.weyl_a_phys is not None:
                    ce["weyl_a_phys"] = [str(v) for v in ci.weyl_a_phys]
                if ci.weyl_b_phys is not None:
                    ce["weyl_b_phys"] = [str(v) for v in ci.weyl_b_phys]
                cusp_entries.append(ce)

            entry2: dict[str, Any] = {
                "type": "multi",
                "cusps": cusp_entries,
            }
            fr_obj = res.fill_result
            if hasattr(fr_obj, "series"):
                entry2["fill_result"] = {
                    "series": {str(k): int(v) for k, v in fr_obj.series.items()},
                    "num_hard": getattr(fr_obj, "num_hard", 0),
                    "num_cusp_eta": getattr(fr_obj, "num_cusp_eta", 0),
                    "hj_ks": getattr(fr_obj, "hj_ks", []),
                }
            out.append(entry2)
    return out


# ======================================================================
# § 7  Mathematica Data Writer
# ======================================================================

def write_mathematica(
    path: Path | str,
    manifold_data: Any,
    nz_data: Any,
    entries: list,
    weyl_result: Any = None,
    dehn_results: list | None = None,
    q_order_half: int | None = None,
) -> None:
    """Write comprehensive Mathematica .m data file."""
    path = Path(path)
    md = manifold_data
    nz = nz_data
    n = nz.n
    r = nz.r
    name = md.name

    L: list[str] = []
    L.append(f'(* ═══════════════════════════════════════════════════════ *)')
    L.append(f'(* Refined 3D Index: {name}                               *)')
    L.append(f'(* Generated by manifold-index                            *)')
    L.append(f'(* ═══════════════════════════════════════════════════════ *)')
    L.append("")

    # ── Manifold parameters ──
    L.append(f'(* Manifold parameters *)')
    L.append(f'manifoldName = "{name}";')
    L.append(f'nTet = {n};  (* number of tetrahedra *)')
    L.append(f'nCusp = {r};  (* number of cusps *)')
    L.append(f'nHard = {nz.num_hard};')
    L.append(f'nEasy = {nz.num_easy};')
    if q_order_half is not None:
        L.append(f'qOrderHalf = {q_order_half};')
    L.append("")

    # ── Gluing matrix ──
    if md.gluing_matrix is not None:
        L.append("(* SnaPy gluing matrix *)")
        mat_str = _np_to_mathematica(md.gluing_matrix)
        L.append(f'gluingMatrix = {mat_str};')
        L.append("")

    # ── NZ matrix ──
    L.append("(* Neumann-Zagier matrix *)")
    L.append(f'gNZ = {_np_to_mathematica(nz.g_NZ)};')
    L.append(f'nuX = {{{", ".join(str(int(v)) for v in nz.nu_x)}}};')
    L.append(f'nuP = {{{", ".join(_math_frac(v) for v in nz.nu_p)}}};')
    L.append("")

    # ── g_NZ^{-1} ──
    S, g_inv_int = nz.g_NZ_inv_scaled()
    L.append(f'(* g_NZ^{{-1}} = gNZInvScaled / {S} *)')
    L.append(f'gNZInvScale = {S};')
    L.append(f'gNZInvScaled = {_np_to_mathematica(g_inv_int)};')
    L.append(f'gNZInv = gNZInvScaled / gNZInvScale;')
    L.append("")

    # ── Refined index sectors ──
    L.append("(* ─── Refined index sectors ─── *)")
    for m_ext, e_ext, result in entries:
        m_str = "{" + ", ".join(str(x) for x in m_ext) + "}"
        e_str = "{" + ", ".join(str(x) for x in e_ext) + "}"
        expr = to_mathematica_series(result, nz.num_hard)
        L.append(f'Iref["{name}", {m_str}, {e_str}] = {expr};')
    L.append("")

    # ── Weyl data ──
    if weyl_result is not None and weyl_result.ab is not None:
        ab = weyl_result.ab
        L.append("(* Weyl symmetry vectors *)")
        L.append(f'weylA = {{{", ".join(_math_frac(v) for v in ab.a)}}};')
        L.append(f'weylB = {{{", ".join(_math_frac(v) for v in ab.b)}}};')
        adj = getattr(weyl_result, 'adjoint', None)
        if adj is not None and adj.projected_value is not None:
            L.append(f'adjointProjection = {adj.projected_value};  '
                     f'(* should be -1, pass={adj.is_pass} *)')
        L.append("")

    # ── Dehn filling ──
    if dehn_results:
        from manifold_index.app.workers import TransformedFillResult, MultiCuspFillResult

        L.append("(* ─── Dehn filling results ─── *)")
        for res in dehn_results:
            if isinstance(res, TransformedFillResult):
                L.append(f'(* Cusp {res.cusp_idx}: slope ({res.P_user},{res.Q_user}), '
                         f'NC ({res.P_nc},{res.Q_nc}), (p,q)=({res.p},{res.q}) *)')
                for m_other, e_other, fr in res.fill_results:
                    m_str = "{" + ", ".join(str(x) for x in m_other) + "}" if m_other else "{}"
                    e_str = "{" + ", ".join(str(x) for x in e_other) + "}" if e_other else "{}"
                    expr = to_mathematica_filled_series(
                        fr.series, fr.num_hard, fr.num_cusp_eta
                    )
                    hj_str = "{" + ", ".join(str(k) for k in fr.hj_ks) + "}"
                    L.append(
                        f'IrefFilled["{name}", {{{res.P_user}, {res.Q_user}}}, '
                        f'{m_str}, {e_str}] = {expr};'
                    )
                    L.append(f'hjCF["{name}", {{{res.P_user}, {res.Q_user}}}] = {hj_str};')
                if res.weyl_a_phys is not None:
                    L.append(f'weylAPhys["{name}", {{{res.P_user}, {res.Q_user}}}] = '
                             f'{{{", ".join(_math_frac(v) for v in res.weyl_a_phys)}}};')
                if res.weyl_b_phys is not None:
                    L.append(f'weylBPhys["{name}", {{{res.P_user}, {res.Q_user}}}] = '
                             f'{{{", ".join(_math_frac(v) for v in res.weyl_b_phys)}}};')
            elif isinstance(res, MultiCuspFillResult):
                cusps = ", ".join(str(ci.cusp_idx) for ci in res.cusp_info)
                L.append(f'(* Multi-cusp: cusps {cusps} *)')
                fr_obj = res.fill_result
                if hasattr(fr_obj, "series"):
                    expr = to_mathematica_filled_series(
                        fr_obj.series,
                        getattr(fr_obj, "num_hard", 0),
                        getattr(fr_obj, "num_cusp_eta", 0),
                    )
                    L.append(f'IrefMulti["{name}", {{{cusps}}}] = {expr};')
        L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


def _np_to_mathematica(arr: np.ndarray) -> str:
    """Convert a numpy 2D array to Mathematica matrix notation."""
    if arr.ndim == 1:
        return "{" + ", ".join(_math_frac(v) for v in arr) + "}"
    rows = []
    for i in range(arr.shape[0]):
        entries = ", ".join(_math_frac(arr[i, j]) for j in range(arr.shape[1]))
        rows.append("{" + entries + "}")
    return "{" + ", ".join(rows) + "}"


def _math_frac(v) -> str:
    """Format a number for Mathematica (exact fraction if not integer)."""
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(int(f))
    return f"{f.numerator}/{f.denominator}"


# ======================================================================
# § 8  Clipboard helpers
# ======================================================================

def clipboard_latex(
    manifold_name: str,
    entries: list,
    num_hard: int,
    dehn_results: list | None = None,
    include_dehn: bool = False,
) -> str:
    """Build a LaTeX snippet suitable for pasting into a document."""
    lines: list[str] = []
    lines.append(r"\begin{align*}")
    for i, (m_ext, e_ext, result) in enumerate(entries):
        tex = to_latex_series(result, num_hard)
        label = _charge_label(m_ext, e_ext, latex=True)
        end = r" \\" if i < len(entries) - 1 else ""
        lines.append(rf"  \Iref({label}) &= {tex}{end}")
    lines.append(r"\end{align*}")
    if include_dehn and dehn_results:
        lines.append("")
        lines.append("% Dehn filling results")
        from manifold_index.app.workers import TransformedFillResult, MultiCuspFillResult
        for res in dehn_results:
            if isinstance(res, TransformedFillResult):
                for m_other, e_other, fr in res.fill_results:
                    tex = to_latex_filled_series(
                        fr.series, fr.num_hard, fr.num_cusp_eta
                    )
                    lines.append(rf"\Iref_{{{res.P_user}/{res.Q_user}}} = {tex}")
    return "\n".join(lines)


def _plain_series(series: dict, num_hard: int, num_cusp_eta: int = 0) -> str:
    """Simple plain-text series representation."""
    if not series:
        return "0"
    parts: list[str] = []
    for key in sorted(series.keys()):
        c = series[key]
        if c == 0:
            continue
        qq_pow = key[0]
        q_half = qq_pow / 2
        q_str = "" if qq_pow == 0 else f"q^{q_half:g}"
        eta_parts: list[str] = []
        for a in range(num_hard):
            if 1 + a < len(key):
                exp2 = key[1 + a]
                if exp2 != 0:
                    p = exp2 / 2
                    eta_parts.append(f"eta_{a}^{p:g}" if p != 1 else f"eta_{a}")
        for ci in range(num_cusp_eta):
            pos = 1 + num_hard + ci
            if pos < len(key) and key[pos] != 0:
                eta_parts.append(f"etaV_{ci}^{key[pos]}" if key[pos] != 1 else f"etaV_{ci}")
        h = "*".join(eta_parts)
        body_parts = [s for s in (q_str, h) if s]
        body = "*".join(body_parts) if body_parts else "1"
        if c == 1:
            parts.append(body)
        elif c == -1:
            parts.append(f"-{body}")
        else:
            parts.append(f"{c}*{body}")
    return " + ".join(parts).replace("+ -", "- ") if parts else "0"


def clipboard_plain_text(
    manifold_name: str,
    entries: list,
    num_hard: int,
    dehn_results: list | None = None,
    include_dehn: bool = False,
) -> str:
    """Build a plain-text snippet suitable for pasting."""
    parts: list[str] = []
    parts.append(f"Refined 3D Index: {manifold_name}")
    for m_ext, e_ext, result in entries:
        label = _charge_label(m_ext, e_ext)
        parts.append(f"I({','.join(str(x) for x in m_ext)}; {','.join(str(x) for x in e_ext)}) = "
                     f"{_plain_series(result, num_hard)}")
    if include_dehn and dehn_results:
        from manifold_index.app.workers import TransformedFillResult, MultiCuspFillResult
        parts.append("")
        parts.append("--- Dehn Filling ---")
        for res in dehn_results:
            if isinstance(res, TransformedFillResult):
                parts.append(f"Cusp {res.cusp_idx}: ({res.P_user},{res.Q_user}) "
                             f"via NC ({res.P_nc},{res.Q_nc}), (p,q)=({res.p},{res.q})")
                for m_other, e_other, fr in res.fill_results:
                    label = _charge_label(m_other, e_other) if m_other else ""
                    if label:
                        parts.append(f"  {label}")
                    parts.append(f"  I_filled = {_plain_series(fr.series, fr.num_hard, fr.num_cusp_eta)}")
    return "\n".join(parts)


# ======================================================================
# § 9  Legacy compatibility wrappers
# ======================================================================
# These maintain backward compatibility with the old API signatures
# used by export_panel.py.  They delegate to the new implementations.

def write_latex(
    path: Path | str,
    manifold_name: str,
    nz: Any,
    entries: list,
    weyl: Any = None,
    dehn_results: list | None = None,
    include_dehn: bool = False,
) -> None:
    """Legacy wrapper — writes a minimal LaTeX document."""
    path = Path(path)
    preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage[margin=2.5cm]{geometry}
\usepackage{hyperref}
\newcommand{\Iref}{I^{\mathrm{ref}}}
\begin{document}
"""
    lines: list[str] = [preamble]
    lines.append(rf"\title{{Refined 3D Index: {manifold_name}}}")
    lines.append(r"\maketitle")
    lines.append(r"\section{Refined Index}")
    for m_ext, e_ext, result in entries:
        label = _charge_label(m_ext, e_ext, latex=True)
        lines.append(rf"\subsection{{{label}}}")
        tex = to_latex_series(result, nz.num_hard)
        lines.append(rf"\[ \Iref = {tex} \]")
        lines.append("")
    if include_dehn and dehn_results:
        from manifold_index.app.workers import TransformedFillResult, MultiCuspFillResult
        lines.append(r"\section{Dehn Filling}")
        for res in dehn_results:
            if isinstance(res, TransformedFillResult):
                for m_other, e_other, fr in res.fill_results:
                    tex = to_latex_filled_series(fr.series, fr.num_hard, fr.num_cusp_eta)
                    lines.append(rf"\[ \Iref_{{{res.P_user}/{res.Q_user}}} = {tex} \]")
    lines.append(r"\end{document}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    path: Path | str,
    manifold_name: str,
    nz: Any,
    easy_result: Any,
    entries: list,
    weyl: Any = None,
    dehn_results: list | None = None,
    include_dehn: bool = False,
    q_order_half: int | None = None,
) -> None:
    """Legacy wrapper — delegates to write_full_report.

    Reconstructs a minimal ManifoldData-like object if needed.
    """
    # Build a minimal ManifoldData stub with gluing_matrix = None
    # The full report will skip gluing-matrix sections gracefully.
    class _Stub:
        def __init__(self, name):
            self.name = name
            self.gluing_matrix = None
            self.num_tetrahedra = nz.n
            self.num_cusps = nz.r

    md = _Stub(manifold_name)
    write_full_report(
        path, md, easy_result, nz, entries, weyl,
        dehn_results if include_dehn else None,
        q_order_half,
    )


def write_plain_text(
    path: Path | str,
    manifold_name: str,
    nz: Any,
    entries: list,
    weyl: Any = None,
    dehn_results: list | None = None,
    include_dehn: bool = False,
) -> None:
    """Legacy wrapper — writes plain text."""
    text = clipboard_plain_text(
        manifold_name, entries, nz.num_hard,
        dehn_results if include_dehn else None,
        include_dehn,
    )
    Path(path).write_text(text, encoding="utf-8")
