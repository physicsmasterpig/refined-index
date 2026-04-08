"""
formatters.manifold_fmt
=======================
HTML / LaTeX formatters for the Manifold card (Card ①).

Public API
----------
format_nz_latex(nz)            → $$...$$ LaTeX string for the NZ matrix
format_gluing_table_html(md)   → <table> HTML (SnaPy gluing equations)
format_easy_edges_html(ps)     → <table> HTML for independent easy edges
format_hard_edges_html(ps)     → <table> HTML for hard-padding edges
format_summary_html(md, ps)    → <p> blurb: name / tetrahedra / cusps / edges

BLUEPRINT reference: §4 (formatters split), §10 (ManifoldCard display)
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers  (also re-exported for sibling formatters)
# ---------------------------------------------------------------------------

def _frac_to_latex(v: "Fraction | float | int") -> str:
    """Format a number as a compact LaTeX string: 0, 1, -2, ½, -3/2 …"""
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(int(f))
    sign = "-" if f < 0 else ""
    return rf"{sign}\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def _coeff_to_latex(c: "Fraction | float | int") -> str:
    """Format a matrix coefficient: integers or fractions."""
    return _frac_to_latex(c)


def _slope_latex(P: int, Q: int,
                 a: str = r"\alpha", b: str = r"\beta") -> str:
    r"""Format $P\,\alpha + Q\,\beta$ with correct sign handling.

    Examples: ``\alpha``, ``-\alpha + 2\,\beta``, ``\alpha - \beta``, ``0``.
    """
    if P == 0 and Q == 0:
        return "0"
    parts: list[str] = []
    if P != 0:
        if P == 1:
            parts.append(a)
        elif P == -1:
            parts.append(f"-{a}")
        else:
            parts.append(f"{P}\\,{a}")
    if Q != 0:
        if Q > 0:
            q_part = f"{Q}\\,{b}" if Q != 1 else b
            if parts:
                parts.append(f" + {q_part}")
            else:
                parts.append(q_part)
        else:
            q_part = f"{-Q}\\,{b}" if Q != -1 else b
            if parts:
                parts.append(f" - {q_part}")
            else:
                parts.append(f"-{q_part}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# NZ matrix
# ---------------------------------------------------------------------------

def format_nz_latex(nz) -> str:
    """Return a $$...$$ display-mode LaTeX string for the NZ matrix.

    Parameters
    ----------
    nz : NeumannZagierData
        Requires ``nz.n``, ``nz.g_NZ`` (2n×2n array), ``nz.nu_x``,
        ``nz.nu_p``.
    """
    n = nz.n
    size = 2 * n

    mat_rows: list[str] = []
    for i in range(size):
        entries = [_coeff_to_latex(nz.g_NZ[i, j]) for j in range(size)]
        mat_rows.append("  " + " & ".join(entries))
    mat_body = " \\\\\n".join(mat_rows)

    nu_x_parts = ", ".join(str(int(v)) for v in nz.nu_x)
    nu_p_parts = ", ".join(
        _frac_to_latex(Fraction(float(v)).limit_denominator(1000))
        for v in nz.nu_p
    )

    return (
        rf"$$g_{{\text{{NZ}}}} = \begin{{pmatrix}}"
        "\n"
        f"{mat_body}"
        "\n"
        rf"\end{{pmatrix}}$$"
        "\n"
        rf"<p>$\nu_x = ({nu_x_parts})$, "
        rf"$\nu_p = ({nu_p_parts})$</p>"
    )


# ---------------------------------------------------------------------------
# Gluing equations table
# ---------------------------------------------------------------------------

def format_gluing_table_html(md) -> str:
    """Return an HTML ``<table>`` of gluing equations.

    Parameters
    ----------
    md : ManifoldData
        Requires ``md.num_tetrahedra``, ``md.gluing_matrix``.
    """
    n = md.num_tetrahedra
    G = md.gluing_matrix

    header = "<tr><th>Edge</th>"
    for j in range(n):
        header += f"<th>$(Z_{{{j+1}}}, Z_{{{j+1}}}', Z_{{{j+1}}}'')$</th>"
    header += "</tr>\n"

    rows = ""
    for i in range(n):
        rows += f"<tr><td><b>{i}</b></td>"
        for j in range(n):
            a = int(G[i, 3 * j])
            b = int(G[i, 3 * j + 1])
            c = int(G[i, 3 * j + 2])
            rows += f"<td>$({a}, {b}, {c})$</td>"
        rows += "</tr>\n"

    return f"<table>\n{header}{rows}</table>\n"


# ---------------------------------------------------------------------------
# Edge classification tables
# ---------------------------------------------------------------------------

def format_easy_edges_html(ps) -> str:
    """Return an HTML ``<table>`` for independent easy edges.

    Parameters
    ----------
    ps : EasyEdgeResult
        Requires ``ps.all_easy``, ``ps.independent_easy_indices``, ``ps.n``.
    """
    n = ps.n
    lines = ""
    for idx, ei in enumerate(ps.independent_easy_indices):
        edge = ps.all_easy[ei]
        triplets = _edge_triplets_latex(edge, n)
        lines += (
            f'<tr><td><b>E{idx}</b></td>'
            f"<td>${triplets}$</td>"
            f"<td>basis row {idx + 1}</td></tr>\n"
        )
    if not lines:
        return '<p class="muted">No easy edges.</p>\n'
    return (
        "<table>\n"
        "<tr><th>Edge</th><th>Triplets</th><th>Role</th></tr>\n"
        f"{lines}"
        "</table>\n"
    )


def format_hard_edges_html(ps) -> str:
    """Return an HTML ``<table>`` for hard-padding edges.

    Parameters
    ----------
    ps : EasyEdgeResult
        Requires ``ps.hard_padding``, ``ps.n``,
        ``ps.num_independent_easy``.
    """
    n = ps.n
    n_easy = ps.num_independent_easy
    lines = ""
    for idx, hedge in enumerate(ps.hard_padding):
        triplets = _edge_triplets_latex(hedge, n)
        lines += (
            f'<tr><td><b>H{idx}</b></td>'
            f"<td>${triplets}$</td>"
            f"<td>basis row {n_easy + idx + 1}</td></tr>\n"
        )
    if not lines:
        return '<p class="muted">No hard edges.</p>\n'
    return (
        "<table>\n"
        "<tr><th>Edge</th><th>Triplets</th><th>Role</th></tr>\n"
        f"{lines}"
        "</table>\n"
    )


def _edge_triplets_latex(edge: "np.ndarray", n: int) -> str:
    """Format triplets (a,b,c) for each tetrahedron as a LaTeX string."""
    parts = []
    for j in range(n):
        a, b, c = int(edge[3 * j]), int(edge[3 * j + 1]), int(edge[3 * j + 2])
        parts.append(rf"({a},{b},{c})\;")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Summary blurb
# ---------------------------------------------------------------------------

def format_summary_html(md, ps) -> str:
    """Return a short HTML blurb: name, tetrahedra, cusps, edge counts.

    Parameters
    ----------
    md : ManifoldData
    ps : EasyEdgeResult
    """
    n_easy = ps.num_independent_easy
    n_hard = len(ps.hard_padding)
    return (
        f"<p><b>{md.name}</b></p>\n"
        f"<p>Tetrahedra: <b>{md.num_tetrahedra}</b>"
        f" &nbsp;&bull;&nbsp; Cusps: <b>{md.num_cusps}</b></p>\n"
        f"<p>Edges: <b>{n_easy + n_hard}</b>"
        f" &nbsp;(easy: <b>{n_easy}</b>, hard: <b>{n_hard}</b>)</p>\n"
    )
