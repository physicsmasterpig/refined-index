"""
formatters.manifold_fmt
=======================
HTML / LaTeX formatters for the Manifold card (Card ①).

Public API
----------
format_nz_latex(nz)               → $$...$$ LaTeX string for the NZ matrix
format_gluing_table_html(md)      → <table> HTML (SnaPy gluing equations)
format_easy_edges_html(ps, md)    → <table> HTML for independent easy edges with compositions
format_hard_edges_html(ps, md)    → <table> HTML for hard-padding edges with compositions
format_summary_html(md, ps)       → <p> blurb: name / tetrahedra / cusps / edges

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

    header = "<tr><th>$\\textrm{{Edge}}$</th>"
    for j in range(n):
        header += f"<th>$(Z_{{{j+1}}}, Z_{{{j+1}}}', Z_{{{j+1}}}'')$</th>"
    header += "</tr>\n"

    rows = ""
    for i in range(n):
        rows += f"<tr><td>${i}$</td>"
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

def format_easy_edges_html(ps, md=None) -> str:
    """Return an HTML ``<table>`` for independent easy edges.

    Parameters
    ----------
    ps : EasyEdgeResult
        Requires ``ps.all_easy``, ``ps.independent_easy_indices``, ``ps.n``.
    md : ManifoldData, optional
        If provided, shows how each edge is composed from gluing equations.
    """
    n = ps.n
    lines = ""
    for idx, ei in enumerate(ps.independent_easy_indices):
        edge = ps.all_easy[ei]
        triplets = _edge_triplets_latex(edge, n)
        composition = ""
        if md is not None:
            comp = _compute_edge_factorization(edge, md)
            if comp:
                composition = f"<td>${comp}$</td>"
            else:
                composition = "<td></td>"
        else:
            composition = "<td></td>"
        lines += (
            f'<tr><td><b>$E{idx}$</b></td>'
            f"<td>${triplets}$</td>"
            f"{composition}"
            f"<td>$\\textrm{{basis row }}{idx + 1}$</td></tr>\n"
        )
    if not lines:
        return '<p class="muted">$\\textrm{{No easy edges.}}$</p>\n'
    header = (
        "<tr><th>$\\textrm{{Edge}}$</th><th>$\\textrm{{Triplets}}$</th>"
        f"{'<th>$\\textrm{{Composition}}$</th>' if md is not None else ''}"
        "<th>$\\textrm{{Role}}$</th></tr>\n"
    )
    return (
        "<table>\n"
        f"{header}"
        f"{lines}"
        "</table>\n"
    )


def format_hard_edges_html(ps, md=None) -> str:
    """Return an HTML ``<table>`` for hard-padding edges.

    Parameters
    ----------
    ps : EasyEdgeResult
        Requires ``ps.hard_padding``, ``ps.n``,
        ``ps.num_independent_easy``.
    md : ManifoldData, optional
        If provided, shows how each edge is composed from gluing equations.
    """
    n = ps.n
    n_easy = ps.num_independent_easy
    lines = ""
    for idx, hedge in enumerate(ps.hard_padding):
        triplets = _edge_triplets_latex(hedge, n)
        composition = ""
        if md is not None:
            comp = _compute_edge_factorization(hedge, md)
            if comp:
                composition = f"<td>${comp}$</td>"
            else:
                composition = "<td></td>"
        else:
            composition = "<td></td>"
        lines += (
            f'<tr><td><b>$H{idx}$</b></td>'
            f"<td>${triplets}$</td>"
            f"{composition}"
            f"<td>$\\textrm{{basis row }}{n_easy + idx + 1}$</td></tr>\n"
        )
    if not lines:
        return '<p class="muted">No hard edges.</p>\n'
    header = (
        "<tr><th>$\\textrm{{Edge}}$</th><th>$\\textrm{{Triplets}}$</th>"
        f"{'<th>$\\textrm{{Composition}}$</th>' if md is not None else ''}"
        "<th>$\\textrm{{Role}}$</th></tr>\n"
    )
    return (
        "<table>\n"
        f"{header}"
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


def _compute_edge_factorization(edge: "np.ndarray", md) -> str:
    """Compute how an edge is composed: E = a @ edge_equations + b @ T_matrix.

    Returns a LaTeX expression like: "2·row₀ + 3·row₁ + T₀"
    """
    try:
        n = md.num_tetrahedra
        edge_eqs = md.edge_equations.astype(float)  # shape (n, 3n)

        # Build T_matrix where each row sums the three coordinates for one tet
        # T_matrix[i] = (0,...,0, 1, 1, 1, 0,...,0) with ones at positions 3i, 3i+1, 3i+2
        T_matrix = np.zeros((n, 3 * n), dtype=float)
        for i in range(n):
            T_matrix[i, 3 * i : 3 * i + 3] = 1.0

        # Build augmented system: vstack([edge_eqs, T_matrix]) with shape (2n, 3n)
        aug_matrix = np.vstack([edge_eqs, T_matrix])

        # Solve: coeffs @ aug_matrix = edge
        # This is an overdetermined or underdetermined system, use lstsq
        from scipy.linalg import lstsq

        coeffs, residual, rank, _ = lstsq(aug_matrix.T, edge.astype(float))

        # Check if solution is valid (low residual)
        if residual.size > 0 and float(residual[0]) > 0.5:
            return ""

        # Round to integers
        coeffs_int = np.round(coeffs).astype(int)

        # Verify the solution
        reconstructed = coeffs_int @ aug_matrix
        if np.linalg.norm(reconstructed - edge.astype(float)) > 0.5:
            return ""

        # Format output
        parts = []

        # Column coefficients (gluing equation rows)
        for i in range(n):
            c = coeffs_int[i]
            if c == 0:
                continue
            if c == 1:
                parts.append(f"c_{{{i}}}")
            elif c == -1:
                parts.append(f"-c_{{{i}}}")
            else:
                parts.append(f"{c}\\,c_{{{i}}}")

        # T coefficients
        for i in range(n):
            c = coeffs_int[n + i]
            if c == 0:
                continue
            if c == 1:
                parts.append(f"T_{{{i}}}")
            elif c == -1:
                parts.append(f"-T_{{{i}}}")
            else:
                parts.append(f"{c}\\,T_{{{i}}}")

        if not parts:
            return "0"

        # Join with proper spacing
        result = parts[0]
        for part in parts[1:]:
            if part[0] == '-':
                result += f" {part}"
            else:
                result += f" + {part}"

        return result
    except Exception:
        return ""


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
