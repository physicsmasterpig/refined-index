"""
app/formatters.py — Generate HTML fragments for KaTeX rendering.

Each function takes core data structures and returns an HTML string fragment
(no <html>/<head>/<body> — those are added by katex.build_katex_html).
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product as itertools_product
from typing import Sequence

import numpy as np

from manifold_index.core.manifold import ManifoldData
from manifold_index.core.phase_space import EasyEdgeResult
from manifold_index.core.neumann_zagier import NeumannZagierData
from manifold_index.core.refined_index import RefinedIndexResult
from manifold_index.core.weyl_check import ABVectors, WeylCheckResult


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _frac_to_latex(v: Fraction | float) -> str:
    """Format a number as a LaTeX string: 0, 1, -2, 1/2, -3/2 etc."""
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(int(f))
    sign = "-" if f < 0 else ""
    return rf"{sign}\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def _coeff_to_latex(c: int | float | Fraction) -> str:
    """Format a matrix coefficient in LaTeX."""
    f = Fraction(c).limit_denominator(1000)
    if f.denominator == 1:
        return str(int(f))
    sign = "-" if f < 0 else ""
    return rf"{sign}\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def _slope_latex(P: int, Q: int, a: str = r"\alpha", b: str = r"\beta") -> str:
    r"""Format $P\,\alpha + Q\,\beta$ with proper sign handling.

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


def _series_to_katex(
    result: RefinedIndexResult,
    num_hard: int,
    max_q_terms: int = 4,
) -> str:
    """Convert a RefinedIndexResult to a compact KaTeX string.

    Groups monomials by q-power and displays up to *max_q_terms* distinct
    q-powers.  Uses η^{±2W_j} notation for hard-edge fugacity variables.
    """
    if not result:
        return "$0$"

    # Group by q-half-power
    by_q: dict[int, dict[tuple[int, ...], int]] = {}
    for key, coeff in result.items():
        if coeff == 0:
            continue
        q_half = key[0]
        eta_key = key[1:]
        if q_half not in by_q:
            by_q[q_half] = {}
        by_q[q_half][eta_key] = by_q[q_half].get(eta_key, 0) + coeff

    if not by_q:
        return "$0$"

    sorted_q = sorted(by_q.keys())

    def _eta_part(eta_pows: tuple[int, ...]) -> str:
        """Format the η part of a monomial."""
        parts = []
        for a, exp_x2 in enumerate(eta_pows):
            if exp_x2 == 0:
                continue
            if exp_x2 == 2:
                parts.append(rf"\eta^{{2W_{a}}}")
            elif exp_x2 == -2:
                parts.append(rf"\eta^{{-2W_{a}}}")
            elif exp_x2 == 1:
                parts.append(rf"\eta^{{W_{a}}}")
            elif exp_x2 == -1:
                parts.append(rf"\eta^{{-W_{a}}}")
            else:
                parts.append(rf"\eta^{{{exp_x2}W_{a}}}")
        return "".join(parts)

    def _q_factor(q_half: int) -> str:
        """Format q^{q_half/2}."""
        if q_half == 0:
            return ""
        if q_half == 2:
            return "q"
        if q_half % 2 == 0:
            n = q_half // 2
            return f"q^{{{n}}}" if n != 1 else "q"
        return rf"q^{{{q_half}/2}}"

    # Build term strings grouped by q-power
    terms: list[str] = []
    q_count = 0
    for q_half in sorted_q:
        eta_dict = by_q[q_half]
        if q_count >= max_q_terms:
            terms.append(r"\cdots")
            break

        # Sort eta terms: pure constant first, then by eta exponents
        sorted_eta = sorted(eta_dict.keys())
        q_str = _q_factor(q_half)

        # Collect coefficient+η parts (WITHOUT q factor — q is applied
        # once at the group level so it is not duplicated).
        sub_parts: list[str] = []
        for eta_key in sorted_eta:
            c = eta_dict[eta_key]
            if c == 0:
                continue
            eta_str = _eta_part(eta_key)

            if not eta_str:
                # Pure coefficient (possibly ×q, added below)
                sub_parts.append(str(c))
            else:
                if c == 1:
                    sub_parts.append(eta_str)
                elif c == -1:
                    sub_parts.append(f"-{eta_str}")
                else:
                    sub_parts.append(f"{c}{eta_str}")

        if not sub_parts:
            continue

        # Combine sub_parts and attach the q factor once
        if len(sub_parts) == 1:
            part = sub_parts[0]
            if q_str:
                # Single coefficient × q^k
                if part == "1":
                    terms.append(q_str)
                elif part == "-1":
                    terms.append(f"-{q_str}")
                else:
                    terms.append(f"{part}{q_str}")
            else:
                terms.append(part)
        else:
            # Multiple η terms at same q-power: (c₁η₁ + c₂η₂ + …)q^k
            inner = sub_parts[0]
            for sp in sub_parts[1:]:
                if sp.startswith("-"):
                    inner += sp
                else:
                    inner += "+" + sp
            if q_str:
                terms.append(f"({inner}){q_str}")
            else:
                terms.append(inner)

        q_count += 1

    if not terms:
        return "$0$"

    # Join with + signs
    result_str = terms[0]
    for t in terms[1:]:
        if t.startswith("-") or t.startswith(r"\cdots"):
            if t.startswith(r"\cdots"):
                result_str += " + " + t
            else:
                result_str += " " + t
        else:
            result_str += " + " + t

    return f"${result_str}$"


# ═══════════════════════════════════════════════════════════════════════════
# Panel 1: Manifold Analysis
# ═══════════════════════════════════════════════════════════════════════════

def format_manifold_info(md: ManifoldData, ps: EasyEdgeResult) -> str:
    """Manifold summary: name, tetrahedra, cusps, edge counts."""
    n_easy = ps.num_independent_easy
    n_hard = len(ps.hard_padding)
    return f"""
<h3>Manifold</h3>
<p><b>{md.name}</b></p>
<p>Tetrahedra: <b>{md.num_tetrahedra}</b> &nbsp;&bull;&nbsp; Cusps: <b>{md.num_cusps}</b></p>
<p>Internal edges: <b>{n_easy + n_hard}</b> &nbsp;(easy: <b>{n_easy}</b>, hard: <b>{n_hard}</b>)</p>
"""


def format_gluing_equations(md: ManifoldData) -> str:
    """Gluing equations table from SnaPy data."""
    n = md.num_tetrahedra
    G = md.gluing_matrix

    # Header row: Edge | (Z_1, Z_1', Z_1'') | ... | (Z_n, Z_n', Z_n'')
    header = "<tr><th>Edge</th>"
    for j in range(n):
        header += f"<th>$(Z_{j+1}, Z_{j+1}', Z_{j+1}'')$</th>"
    header += "</tr>\n"

    rows = ""
    for i in range(n):  # only edge equations
        rows += f"<tr><td><b>{i}</b></td>"
        for j in range(n):
            a = int(G[i, 3 * j])
            b = int(G[i, 3 * j + 1])
            c = int(G[i, 3 * j + 2])
            rows += f"<td>$({a}, {b}, {c})$</td>"
        rows += "</tr>\n"

    return f"""
<h3>Gluing Equations (SnaPy)</h3>
<table>
{header}{rows}</table>
"""


def format_edge_classification(ps: EasyEdgeResult) -> str:
    """Edge classification table showing easy/hard edges with triplets."""
    n = ps.n
    lines = ""

    # Easy edges (independent ones)
    for idx, ei in enumerate(ps.independent_easy_indices):
        edge = ps.all_easy[ei]
        triplets = ""
        for j in range(n):
            a, b, c = int(edge[3 * j]), int(edge[3 * j + 1]), int(edge[3 * j + 2])
            triplets += rf"({a},{b},{c})\;"
        # Find which SnaPy edge this corresponds to (if known)
        lines += (
            f'<tr><td><b>E{idx}</b> <span class="muted">(easy)</span></td>'
            f"<td>${triplets}$</td>"
            f"<td>basis row {idx + 1}</td></tr>\n"
        )

    # Hard edges
    for idx, hedge in enumerate(ps.hard_padding):
        triplets = ""
        for j in range(n):
            a, b, c = int(hedge[3 * j]), int(hedge[3 * j + 1]), int(hedge[3 * j + 2])
            triplets += rf"({a},{b},{c})\;"
        lines += (
            f'<tr><td><b>H{idx}</b> <span class="muted">(hard)</span></td>'
            f"<td>${triplets}$</td>"
            f"<td>basis row {ps.num_independent_easy + idx + 1}</td></tr>\n"
        )

    return f"""
<h3>Edge Classification</h3>
<table>
<tr><th></th><th>Triplets</th><th>Role</th></tr>
{lines}</table>
"""


def format_nz_matrix(nz: NeumannZagierData) -> str:
    """Neumann–Zagier matrix display with KaTeX."""
    n = nz.n
    size = 2 * n

    # Build matrix rows
    mat_rows: list[str] = []
    for i in range(size):
        entries = []
        for j in range(size):
            entries.append(_coeff_to_latex(nz.g_NZ[i, j]))
        mat_rows.append("  " + " & ".join(entries))

    mat_body = " \\\\\\\\\n".join(mat_rows)

    # Affine shifts
    nu_x_parts = ", ".join(str(int(v)) for v in nz.nu_x)
    nu_p_parts = ", ".join(_frac_to_latex(Fraction(v).limit_denominator(1000)) for v in nz.nu_p)

    return f"""
<h3>Neumann–Zagier Data</h3>
<p>$g_{{\\text{{NZ}}}} \\in \\mathrm{{Sp}}(2r,\\,\\mathbb{{Q}})$, &nbsp;
$r = {n}$ tetrahedra &nbsp;→&nbsp; ${size} \\times {size}$ matrix</p>
$$g_{{\\text{{NZ}}}} = \\begin{{pmatrix}}
{mat_body}
\\end{{pmatrix}}$$
<p>Affine shifts: &nbsp;
$\\nu_x = ({nu_x_parts})$, &nbsp;
$\\nu_p = ({nu_p_parts})$</p>
"""


def format_weyl_check(weyl: WeylCheckResult | None, nz: NeumannZagierData) -> str:
    """Weyl symmetry section: a, b vectors, compatibility, and adjoint projection."""
    if weyl is None:
        return """
<h3>Weyl Symmetry</h3>
<p class="muted">Computing…</p>
"""

    lines = """
<h3>Weyl Symmetry</h3>
<p>Convention: $f(m,e) = \\eta^{\\sum_I(a_I \\cdot e_I + b_I \\cdot m_I)} \\cdot I^{\\text{ref}}(m,e)$, &nbsp;
Weyl: $f(m,e) = f(-m,-e)$</p>
"""

    if weyl.ab is not None:
        ab = weyl.ab
        if ab.cusp_columns is not None:
            # Multi-cusp: show per-cusp matrix
            lines += "<p>Per-cusp Weyl vectors:</p>\n"
            for j in range(ab.num_hard):
                a_parts = ", ".join(
                    _frac_to_latex(col.a[j]) for col in ab.cusp_columns
                )
                b_parts = ", ".join(
                    _frac_to_latex(col.b[j]) for col in ab.cusp_columns
                )
                lines += (
                    f"<p>$a_{j} = ({a_parts}), \\quad "
                    f"b_{j} = ({b_parts})$</p>\n"
                )
        else:
            for j in range(ab.num_hard):
                a_str = _frac_to_latex(ab.a[j])
                b_str = _frac_to_latex(ab.b[j])
                lines += f"<p>$a_{j} = {a_str}, \\quad b_{j} = {b_str}$</p>\n"

        if ab.is_valid:
            lines += '<p class="success">✓ &nbsp; $a \\in \\mathbb{Z}$, $b \\in \\mathbb{Z}/2$ — Dehn filling compatible</p>\n'
        else:
            lines += '<p class="warn">⚠ &nbsp; $a \\notin \\mathbb{Z}$ or $2b \\notin \\mathbb{Z}$ — Dehn filling <b>not</b> compatible</p>\n'
    else:
        lines += '<p class="warn">⚠ &nbsp; Could not determine Weyl vectors (insufficient data)</p>\n'

    # Weyl symmetry check
    if weyl.weyl_symmetric:
        n_sym = sum(weyl.weyl_symmetric.values())
        n_total = len(weyl.weyl_symmetric)
        if weyl.all_weyl_symmetric:
            lines += f'<p class="success">✓ &nbsp; Weyl symmetry: {n_sym}/{n_total} sectors symmetric</p>\n'
        else:
            lines += f'<p class="warn">⚠ &nbsp; Weyl symmetry: {n_sym}/{n_total} sectors symmetric</p>\n'

    # Adjoint projection check (eq 2.59–2.61)
    adj = weyl.adjoint
    if adj is not None:
        if adj.missing_e:
            missing_str = ", ".join(str(e) for e in adj.missing_e)
            lines += (
                f'<p class="warn">⚠ &nbsp; Adjoint $q^1$ projection: '
                f'incomplete (missing $e = {missing_str}$ entries)</p>\n'
            )
        elif adj.projected_value is not None:
            if adj.is_pass:
                lines += (
                    '<p class="success">✓ &nbsp; Adjoint $q^1$ projection: '
                    '$\\mathcal{J}_{q^1}|_{\\mathrm{adj}\\,su(2)} = -1$ &nbsp; ✓</p>\n'
                )
            else:
                lines += (
                    f'<p class="warn">⚠ &nbsp; Adjoint $q^1$ projection: '
                    f'got {adj.projected_value}, expected $-1$</p>\n'
                )
        else:
            lines += '<p class="warn">⚠ &nbsp; Adjoint $q^1$ projection: non-integer result</p>\n'
    else:
        lines += '<p class="muted">Adjoint $q^1$ projection: could not compute</p>\n'

    return lines


# ─────────────────────────────────────────────────────────────────────────
# Refined index table — the big one
# ─────────────────────────────────────────────────────────────────────────

# Display charges per cusp: (alpha_coeff, beta_coeff) pairs
# Convention: e·α − (m/2)·β  →  alpha_coeff = e, beta_coeff = −m/2
DISPLAY_CHARGES = [
    (0, 0),                        # m=0, e=0
    (0, Fraction(-1, 2)),          # m=1, e=0
    (Fraction(1, 2), 0),           # m=0, e=1/2
    (0, -1),                       # m=2, e=0
    (1, 0),                        # m=0, e=1
]


def _charge_to_me(alpha: Fraction, beta: Fraction) -> tuple[int, Fraction]:
    """Convert (alpha_coeff, beta_coeff) → (m, e) for compute_refined_index."""
    # Convention: e·α − (m/2)·β  →  alpha = e, beta = −m/2
    e = Fraction(alpha)
    m = int(-beta * 2)
    return m, e


def _alpha_latex(coeff: Fraction, cusp: int) -> str:
    """Format a single alpha term like '0\\,\\alpha_1' or '-\\tfrac{1}{2}\\,\\alpha_2'."""
    c = Fraction(coeff)
    if c == 0:
        return rf"0\,\alpha_{cusp + 1}"
    return rf"{_frac_to_latex(c)}\,\alpha_{cusp + 1}"


def _beta_latex(coeff: Fraction, cusp: int) -> str:
    """Format a single beta term like '0\\,\\beta_1' or '1\\,\\beta_2'."""
    c = Fraction(coeff)
    if c == 0:
        return rf"0\,\beta_{cusp + 1}"
    return rf"{_frac_to_latex(c)}\,\beta_{cusp + 1}"


def format_refined_index_table(
    entries: list[tuple[list[int], list, RefinedIndexResult]],
    nz: NeumannZagierData,
    max_q_terms: int = 4,
) -> str:
    """Build the refined index table HTML from computed entries.

    Parameters
    ----------
    entries : list of (m_ext, e_ext, result) triples
        All computed sectors (typically 25^r for Weyl + display).
    nz : NeumannZagierData
    max_q_terms : int
        How many q-power terms to show per row.
    """
    r = nz.r

    # Build lookup: (tuple(m), tuple(e)) → result
    lookup: dict[tuple, RefinedIndexResult] = {}
    for m_ext, e_ext, res in entries:
        key = (tuple(m_ext), tuple(e_ext))
        lookup[key] = res

    # Generate the 5^r display rows
    display_combos = list(itertools_product(DISPLAY_CHARGES, repeat=r))
    n_display = len(display_combos)

    # Count non-zero
    n_nonzero = 0
    for combo in display_combos:
        m_ext = []
        e_ext = []
        for alpha, beta in combo:
            m, e = _charge_to_me(alpha, beta)
            m_ext.append(m)
            e_ext.append(e)
        key = (tuple(m_ext), tuple(e_ext))
        res = lookup.get(key)
        if res:
            n_nonzero += 1

    html = f"""
<h3>Refined Index</h3>
<p class="muted">Charges per cusp: $0,\\, \\pm\\tfrac{{1}}{{2}},\\, \\pm 1$
&nbsp;→&nbsp; $5^{r} = {n_display}$ sectors ({n_nonzero} non-zero).
Label: $I(m_i, e_i) \\to I(e_i\\,\\alpha_i - \\tfrac{{m_i}}{{2}}\\,\\beta_i)$.</p>

<table class="idx">
"""

    for combo in display_combos:
        m_ext = []
        e_ext = []
        alphas = []
        betas = []
        for alpha_coeff, beta_coeff in combo:
            m, e = _charge_to_me(alpha_coeff, beta_coeff)
            m_ext.append(m)
            e_ext.append(e)
            alphas.append(alpha_coeff)
            betas.append(beta_coeff)

        key = (tuple(m_ext), tuple(e_ext))
        res = lookup.get(key)

        # Build the alpha column: "-½α₁ +; 0α₂"
        alpha_parts = []
        for i, a in enumerate(alphas):
            alpha_parts.append(_alpha_latex(a, i))
        alpha_col = r" +\; ".join(alpha_parts)
        # Fix leading sign: if first term is negative, the + should be -
        # Actually, the join always uses +; so we need to handle negative first terms
        # Let's do it properly
        alpha_col = _alpha_latex(alphas[0], 0)
        for i in range(1, r):
            a = Fraction(alphas[i])
            if a < 0:
                alpha_col += rf" -\; {_frac_to_latex(-a)}\,\alpha_{i + 1}"
            else:
                alpha_col += rf" +\; {_alpha_latex(a, i)}"

        # Build the beta column: "+; 0β₁ +; ½β₂"
        beta_parts = []
        for i, b in enumerate(betas):
            if i == 0:
                if Fraction(b) < 0:
                    beta_parts.append(rf"-\; {_frac_to_latex(-Fraction(b))}\,\beta_{i + 1}")
                else:
                    beta_parts.append(rf"+\; {_beta_latex(b, i)}")
            else:
                if Fraction(b) < 0:
                    beta_parts.append(rf"-\; {_frac_to_latex(-Fraction(b))}\,\beta_{i + 1}")
                else:
                    beta_parts.append(rf"+\; {_beta_latex(b, i)}")
        beta_col = " ".join(beta_parts)

        # Series
        if res:
            series_str = _series_to_katex(res, nz.num_hard, max_q_terms=max_q_terms)
        else:
            series_str = "$0$"

        html += f"""<tr>
  <td class="i">$I($</td>
  <td class="al">${alpha_col}$</td>
  <td class="bl">${beta_col}$</td>
  <td class="cp">$)$</td>
  <td class="eq">$=$</td>
  <td class="sr">{series_str}</td>
</tr>
"""

    html += "</table>\n"
    return html


def format_panel1_html(
    md: ManifoldData,
    ps: EasyEdgeResult,
    nz: NeumannZagierData,
    entries: list[tuple[list[int], list, RefinedIndexResult]] | None = None,
    weyl_result: WeylCheckResult | None = None,
    max_q_terms: int = 4,
) -> str:
    """Assemble the full Panel 1 HTML body from all components.

    If *entries* is None (still computing), omits the refined index table.
    """
    parts = [
        format_manifold_info(md, ps),
        format_gluing_equations(md),
        format_edge_classification(ps),
        format_nz_matrix(nz),
    ]
    if entries is not None:
        parts.append(format_refined_index_table(entries, nz, max_q_terms))
    if entries is not None:
        # Show Weyl check after the index table
        parts.append(format_weyl_check(weyl_result, nz))
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 2: Dehn Filling
# ═══════════════════════════════════════════════════════════════════════════

def format_nc_cycles(
    nc_results: list,  # list[NonClosableCycleResult]
    nz: NeumannZagierData,
) -> str:
    """Intermediate display: just report how many NC cycles were found."""
    if not nc_results:
        return ""

    html = '<h3>Non-closable Cycles</h3>\n'
    for nc in nc_results:
        cusp = nc.cusp_idx
        n_nc = len(nc.cycles)
        if n_nc == 0:
            html += f'<p class="warn">No non-closable cycles found at cusp {cusp}.</p>\n'
        else:
            cycles_str = ", ".join(
                f"$({c.P},\\,{c.Q})$" for c in nc.cycles
            )
            html += (
                f"<p>Cusp {cusp}: <b>{n_nc}</b> NC cycle(s) — {cycles_str}"
                f" &nbsp; <span class=\"muted\">(computing filled index…)</span></p>\n"
            )
    return html


def format_transformed_fill_results(
    transformed_results: list,  # list[TransformedFillResult]
    nz=None,    # NeumannZagierData | None — needed for multi-cusp charge labels
    max_q_terms: int = 4,
) -> str:
    """Format the unified NC cycle table + filled refined index results.

    Top section: numbered table with γ_i, δ_i, transformed slope, k-vector.
    Bottom section: per-label index results in the same I(α,β) = series
    format used by Panel 1.
    """
    if not transformed_results:
        return ""

    html = ""

    # Group by cusp
    by_cusp: dict[int, list] = {}
    for tr in transformed_results:
        by_cusp.setdefault(tr.cusp_idx, []).append(tr)

    for cusp_idx in sorted(by_cusp):
        trs = by_cusp[cusp_idx]
        P_user = trs[0].P_user
        Q_user = trs[0].Q_user

        html += '<h3>Dehn Filling — Cusp ' + str(cusp_idx) + '</h3>\n'
        html += (
            '<p>User slope: '
            f'${_slope_latex(P_user, Q_user, "\\alpha", "\\beta")}$'
            '</p>\n'
        )

        # ── NC cycle + transform table ────────────────────────
        html += '<table class="nc">\n'
        html += (
            '<tr>'
            '<th>#</th>'
            '<th>$\\gamma_i$</th>'
            '<th>$\\delta_i$</th>'
            '<th>Transformed slope</th>'
            '<th>$\\mathbf{k}$</th>'
            '<th>$a$</th>'
            '<th>$b$</th>'
            '</tr>\n'
        )

        for i, tr in enumerate(trs, 1):
            gamma_str = f"${_slope_latex(tr.P_nc, tr.Q_nc)}$"
            delta_str = f"${_slope_latex(tr.R, tr.S)}$"

            # Transformed slope in (γ_i, δ_i) notation
            g_sym = f"\\gamma_{{{i}}}"
            d_sym = f"\\delta_{{{i}}}"
            slope_str = f"${_slope_latex(tr.p, tr.q, g_sym, d_sym)}$"

            # HJ continued fraction from the first fill result
            if tr.fill_results:
                hj_ks = tr.fill_results[0][2].hj_ks
                k_str = "$[" + ",\\,".join(str(k) for k in hj_ks) + "]$"
            else:
                k_str = "—"

            # Weyl vectors (physical values): a_phys, b_phys per hard edge
            if tr.weyl_a_phys is not None:
                a_entries = [_frac_to_latex(v) for v in tr.weyl_a_phys]
                a_str = "$(" + ",\\;".join(a_entries) + ")$"
            else:
                a_str = "—"
            if tr.weyl_b_phys is not None:
                b_entries = [_frac_to_latex(v) for v in tr.weyl_b_phys]
                b_str = "$(" + ",\\;".join(b_entries) + ")$"
            else:
                b_str = "—"

            html += (
                f'<tr>'
                f'<td><b>{i}</b></td>'
                f'<td>{gamma_str}</td>'
                f'<td>{delta_str}</td>'
                f'<td>{slope_str}</td>'
                f'<td>{k_str}</td>'
                f'<td>{a_str}</td>'
                f'<td>{b_str}</td>'
                f'</tr>\n'
            )

        html += '</table>\n'

        # ── Filled refined index results (Panel-1 style) ─────
        html += '<h3>Filled Refined Index</h3>\n'

        # Determine unfilled cusps (needed for charge display)
        r = nz.r if nz is not None else 1
        unfilled_cusps = [k for k in range(r) if k != cusp_idx]
        has_ext = len(unfilled_cusps) > 0

        for i, tr in enumerate(trs, 1):
            fill_results = tr.fill_results
            if not fill_results:
                html += f'<p class="warn">#{i}: No results computed.</p>\n'
                continue

            html += f'<p style="margin:10px 0 2px 0;"><b>#{i}</b></p>\n'

            if has_ext:
                # ── Multi-cusp: show I( α + β ) = series per charge ──
                html += '<table class="idx">\n'

                for m_o, e_o, fr in fill_results:
                    series_str = _filled_series_to_katex(fr, max_q_terms=max_q_terms)

                    # Convert (m, e) back to (alpha, beta) notation:
                    # Convention: e·α − (m/2)·β  →  alpha = e, beta = −m/2
                    alphas: list[Fraction] = []
                    betas: list[Fraction] = []
                    for idx, k in enumerate(unfilled_cusps):
                        m_val = m_o[idx] if idx < len(m_o) else 0
                        e_val = e_o[idx] if idx < len(e_o) else 0
                        alphas.append(Fraction(e_val))
                        betas.append(Fraction(-m_val, 2))

                    # Build alpha column (like Panel 1)
                    alpha_col = _alpha_latex(alphas[0], unfilled_cusps[0])
                    for j in range(1, len(unfilled_cusps)):
                        a = alphas[j]
                        k = unfilled_cusps[j]
                        if a < 0:
                            alpha_col += rf" -\; {_frac_to_latex(-a)}\,\alpha_{k + 1}"
                        else:
                            alpha_col += rf" +\; {_alpha_latex(a, k)}"

                    # Build beta column (like Panel 1)
                    beta_parts = []
                    for j, k in enumerate(unfilled_cusps):
                        b = betas[j]
                        if b < 0:
                            beta_parts.append(rf"-\; {_frac_to_latex(-b)}\,\beta_{k + 1}")
                        else:
                            beta_parts.append(rf"+\; {_beta_latex(b, k)}")
                    beta_col = " ".join(beta_parts)

                    html += (
                        f'<tr>'
                        f'<td class="i">$I($</td>'
                        f'<td class="al">${alpha_col}$</td>'
                        f'<td class="bl">${beta_col}$</td>'
                        f'<td class="cp">$)$</td>'
                        f'<td class="eq">$=$</td>'
                        f'<td class="sr">{series_str}</td>'
                        f'</tr>\n'
                    )

                html += '</table>\n'

            else:
                # ── Single-cusp: no external charges, just show result ──
                if fill_results:
                    _m_o, _e_o, fr = fill_results[0]
                    series_str = _filled_series_to_katex(fr, max_q_terms=max_q_terms)
                    html += '<table class="idx">\n'
                    html += (
                        f'<tr>'
                        f'<td class="eq">$=$</td>'
                        f'<td class="sr">{series_str}</td>'
                        f'</tr>\n'
                    )
                    html += '</table>\n'

    return html


def _filled_series_to_katex(
    fr,  # FilledRefinedResult
    max_q_terms: int = 4,
) -> str:
    """Convert a FilledRefinedResult's MultiEtaSeries to compact KaTeX.

    Displays hard-edge η^{±2W_a} and cusp η^{2V_i} in the same format
    as Panel 1's ``_series_to_katex``, extended with cusp-η dimensions.
    Supports multiple cusp η's from sequential multi-cusp filling.
    Coefficients are Fraction-valued.
    """
    if fr.is_zero:
        return "$0$"

    series = fr.series
    num_hard = fr.num_hard
    has_cusp_eta = fr.has_cusp_eta
    num_cusp_eta = getattr(fr, "num_cusp_eta", 1 if has_cusp_eta else 0)

    # ── Group by qq-power ──
    by_q: dict[int, dict[tuple[int, ...], Fraction]] = {}
    for key, c in series.items():
        if c == 0:
            continue
        qq = key[0]
        eta_key = key[1:]
        bq = by_q.setdefault(qq, {})
        bq[eta_key] = bq.get(eta_key, Fraction(0)) + c

    if not by_q:
        return "$0$"

    sorted_q = sorted(by_q.keys())

    # ── η formatting ──
    def _eta_part(eta_pows: tuple[int, ...]) -> str:
        parts: list[str] = []
        # Hard-edge η's (stored as 2×exponent)
        for a in range(min(num_hard, len(eta_pows))):
            exp2 = eta_pows[a]
            if exp2 == 0:
                continue
            if exp2 == 2:
                parts.append(rf"\eta^{{2W_{a}}}")
            elif exp2 == -2:
                parts.append(rf"\eta^{{-2W_{a}}}")
            elif exp2 == 1:
                parts.append(rf"\eta^{{W_{a}}}")
            elif exp2 == -1:
                parts.append(rf"\eta^{{-W_{a}}}")
            else:
                parts.append(rf"\eta^{{{exp2}W_{a}}}")
        # Cusp η's (integer exponents, after hard-edge dimensions)
        # Rendered as η^{2V_i}, η^{-2V_i}, η^{2k·V_i}, etc.
        for ci in range(num_cusp_eta):
            pos = num_hard + ci
            if pos >= len(eta_pows):
                break
            ce = eta_pows[pos]
            if ce == 0:
                continue
            coeff = 2 * ce
            if coeff == 2:
                parts.append(rf"\eta^{{2V_{ci}}}")
            elif coeff == -2:
                parts.append(rf"\eta^{{-2V_{ci}}}")
            else:
                parts.append(rf"\eta^{{{coeff}V_{ci}}}")
        return "".join(parts)

    def _q_factor(q_half: int) -> str:
        if q_half == 0:
            return ""
        if q_half == 2:
            return "q"
        if q_half % 2 == 0:
            n = q_half // 2
            return f"q^{{{n}}}" if n != 1 else "q"
        return rf"q^{{{q_half}/2}}"

    def _coeff_str(c: Fraction) -> str:
        """Fraction → short LaTeX (no enclosing $)."""
        if c.denominator == 1:
            return str(int(c))
        sign = "-" if c < 0 else ""
        return rf"{sign}\tfrac{{{abs(c.numerator)}}}{{{c.denominator}}}"

    # ── Build term strings ──
    terms: list[str] = []
    q_count = 0
    for qq in sorted_q:
        eta_dict = by_q[qq]
        if q_count >= max_q_terms:
            terms.append(r"\cdots")
            break

        sorted_eta = sorted(eta_dict.keys())
        q_str = _q_factor(qq)

        # Build sub-parts WITHOUT q factor (q is applied as group outer).
        sub_parts: list[str] = []
        for ek in sorted_eta:
            c = eta_dict[ek]
            if c == 0:
                continue
            eta_str = _eta_part(ek)

            if not eta_str:
                # Pure coefficient (no η)
                sub_parts.append(_coeff_str(c))
            else:
                if c == 1:
                    sub_parts.append(eta_str)
                elif c == -1:
                    sub_parts.append(f"-{eta_str}")
                else:
                    sub_parts.append(f"{_coeff_str(c)}{eta_str}")

        if not sub_parts:
            continue

        # Combine sub-parts, then attach q factor
        if len(sub_parts) == 1:
            inner = sub_parts[0]
        else:
            inner = sub_parts[0]
            for sp in sub_parts[1:]:
                if sp.startswith("-"):
                    inner += sp
                else:
                    inner += "+" + sp

        if q_str:
            if len(sub_parts) == 1:
                # Single monomial: coeff·η · q^k
                if inner == "1":
                    terms.append(q_str)
                elif inner == "-1":
                    terms.append(f"-{q_str}")
                else:
                    terms.append(f"{inner}{q_str}")
            else:
                terms.append(f"({inner}){q_str}")
        else:
            # q^0 — just the coefficient·η part
            terms.append(inner)

        q_count += 1

    if not terms:
        return "$0$"

    result_str = terms[0]
    for t in terms[1:]:
        if t.startswith("-") or t.startswith(r"\cdots"):
            if t.startswith(r"\cdots"):
                result_str += " + " + t
            else:
                result_str += " " + t
        else:
            result_str += " + " + t

    return f"${result_str}$"


def format_multi_cusp_fill_results(
    multi_results: list,  # list[MultiCuspFillResult]
    nz=None,              # NeumannZagierData | None
    max_q_terms: int = 6,
) -> str:
    """Format multi-cusp sequential filling results.

    Displays one section per NC cycle combination, with a per-cusp
    table showing γ_i, δ_i, transformed slope, Weyl vectors,
    followed by the single combined filled refined index.
    """
    if not multi_results:
        return ""

    html = '<h3>Multi-Cusp Dehn Filling (Sequential)</h3>\n'

    # All combinations share the same user slopes — extract from first
    first = multi_results[0]
    slopes_parts = []
    for ci in first.cusp_info:
        slopes_parts.append(
            f'Cusp {ci.cusp_idx}: '
            f'${_slope_latex(ci.P_user, ci.Q_user, "\\\\alpha", "\\\\beta")}$'
        )
    html += '<p>User slopes: ' + ' &nbsp;|&nbsp; '.join(slopes_parts) + '</p>\n'
    html += f'<p class="muted">{len(multi_results)} NC cycle combination(s)</p>\n'

    for combo_idx, mr in enumerate(multi_results, 1):
        cusp_infos = mr.cusp_info
        fill_result = mr.fill_result

        html += f'<h4 style="margin-top:16px;">Combination #{combo_idx}</h4>\n'

        # ── Per-cusp NC cycle table ──────────────────────────
        html += '<table class="nc">\n'
        html += (
            '<tr>'
            '<th>Cusp</th>'
            '<th>$\\gamma$</th>'
            '<th>$\\delta$</th>'
            '<th>Transformed slope</th>'
            '<th>$a$</th>'
            '<th>$b$</th>'
            '</tr>\n'
        )

        for ci in cusp_infos:
            gamma_str = f"${_slope_latex(ci.P_nc, ci.Q_nc)}$"
            delta_str = f"${_slope_latex(ci.R, ci.S)}$"

            g_sym = f"\\gamma_{{{ci.cusp_idx}}}"
            d_sym = f"\\delta_{{{ci.cusp_idx}}}"
            slope_str = f"${_slope_latex(ci.p, ci.q, g_sym, d_sym)}$"

            if ci.weyl_a_phys is not None:
                a_entries = [_frac_to_latex(v) for v in ci.weyl_a_phys]
                a_str = "$(" + ",\\;".join(a_entries) + ")$"
            else:
                a_str = "—"
            if ci.weyl_b_phys is not None:
                b_entries = [_frac_to_latex(v) for v in ci.weyl_b_phys]
                b_str = "$(" + ",\\;".join(b_entries) + ")$"
            else:
                b_str = "—"

            html += (
                f'<tr>'
                f'<td><b>{ci.cusp_idx}</b></td>'
                f'<td>{gamma_str}</td>'
                f'<td>{delta_str}</td>'
                f'<td>{slope_str}</td>'
                f'<td>{a_str}</td>'
                f'<td>{b_str}</td>'
                f'</tr>\n'
            )

        html += '</table>\n'

        # ── Combined filled refined index ────────────────────
        if fill_result is not None:
            series_str = _filled_series_to_katex(fill_result, max_q_terms=max_q_terms)
            html += '<table class="idx">\n'
            html += (
                f'<tr>'
                f'<td class="eq">$=$</td>'
                f'<td class="sr">{series_str}</td>'
                f'</tr>\n'
            )
            html += '</table>\n'
        else:
            html += '<p class="warn">No result computed.</p>\n'

    return html


def format_dehn_compatibility(weyl: WeylCheckResult | None) -> str:
    """Format Dehn filling compatibility and refinement choice summary.

    Shows per-edge compatibility of the Weyl vectors with half-integer
    filling, and which refinement variables are turned off.
    """
    if weyl is None or weyl.ab is None:
        return ""

    ab = weyl.ab
    n = ab.num_hard
    compat = ab.edge_compatible

    lines: list[str] = []
    lines.append('<h3>Refinement &amp; Compatibility</h3>')

    # ── Per-edge table ────────────────────────────────────────────────
    lines.append('<table class="compat" style="border-collapse:collapse; '
                 'margin:4px 0 8px 0; font-size:0.9em;">')
    lines.append('<tr style="border-bottom:1px solid #555;">'
                 '<th style="padding:2px 8px;">Edge</th>'
                 '<th style="padding:2px 8px;">$a_j$</th>'
                 '<th style="padding:2px 8px;">$b_j$</th>'
                 '<th style="padding:2px 8px;">$a_j \\in \\mathbb{Z}$?</th>'
                 '<th style="padding:2px 8px;">$2b_j \\in \\mathbb{Z}$?</th>'
                 '<th style="padding:2px 8px;">Status</th></tr>')
    for j in range(n):
        a_ok = ab.a_is_integer[j]
        b_ok = ab.b_is_half_integer[j]
        ok = compat[j]
        if ab.cusp_columns is not None:
            # Multi-cusp: show per-cusp vector
            a_str = "(" + ", ".join(
                _frac_to_latex(col.a[j]) for col in ab.cusp_columns
            ) + ")"
            b_str = "(" + ", ".join(
                _frac_to_latex(col.b[j]) for col in ab.cusp_columns
            ) + ")"
        else:
            a_str = _frac_to_latex(ab.a[j])
            b_str = _frac_to_latex(ab.b[j])
        a_icon = "\u2713" if a_ok else "\u2717"
        b_icon = "\u2713" if b_ok else "\u2717"
        status = ('<span style="color:#2ea043;">\u2713 compatible</span>' if ok
                  else '<span style="color:#cf222e;">\u2717 incompatible</span>')
        lines.append(
            f'<tr>'
            f'<td style="padding:2px 8px; text-align:center;">{j}</td>'
            f'<td style="padding:2px 8px; text-align:center;">$ {a_str} $</td>'
            f'<td style="padding:2px 8px; text-align:center;">$ {b_str} $</td>'
            f'<td style="padding:2px 8px; text-align:center;">{a_icon}</td>'
            f'<td style="padding:2px 8px; text-align:center;">{b_icon}</td>'
            f'<td style="padding:2px 8px;">{status}</td>'
            f'</tr>'
        )
    lines.append('</table>')

    # ── Refinement choice summary ─────────────────────────────────────
    incomp = [j for j in range(n) if not compat[j]]
    if not incomp:
        lines.append(
            '<p class="success">All edges compatible \u2014 '
            'full refinement ($\\eta_j$ active for all $j$).</p>'
        )
    else:
        if n == 1:
            lines.append(
                '<p class="warn">Edge 0 incompatible \u2014 '
                'refinement turned off ($\\eta_0 = 1$).</p>'
            )
        else:
            edge_list = ", ".join(str(j) for j in incomp)
            eta_list = ", ".join(f"\\eta_{j}" for j in incomp)
            lines.append(
                f'<p class="warn">Edge(s) {{{edge_list}}} incompatible \u2192 '
                f'set ${eta_list} = 1$ &nbsp;($W_j = 0$) for filling.</p>'
            )
        # Show the effective (filling-compatible) vectors
        ab_eff = ab.make_filling_compatible()
        eff_parts = []
        for j in range(n):
            if ab_eff.cusp_columns is not None:
                a_s = "(" + ", ".join(
                    _frac_to_latex(col.a[j]) for col in ab_eff.cusp_columns
                ) + ")"
                b_s = "(" + ", ".join(
                    _frac_to_latex(col.b[j]) for col in ab_eff.cusp_columns
                ) + ")"
            else:
                a_s = _frac_to_latex(ab_eff.a[j])
                b_s = _frac_to_latex(ab_eff.b[j])
            eff_parts.append(f"a_{j}={a_s},\\; b_{j}={b_s}")
        lines.append(
            f'<p style="font-size:0.85em;">Effective vectors after zeroing: '
            f'${"; \\quad ".join(eff_parts)}$</p>'
        )

    return "\n".join(lines)


def format_panel2_html(
    nc_results: list | None = None,
    transformed_results: list | None = None,
    multi_cusp_results: list | None = None,
    nz: NeumannZagierData | None = None,
    weyl: WeylCheckResult | None = None,
    max_q_terms: int = 4,
) -> str:
    """Assemble the full Panel 2 HTML body.

    Workflow display:
      1. NC cycles found (from search)
      2. User's slope transformed into each NC basis → filled refined index

    For multi-cusp filling (all cusps filled simultaneously), pass
    *multi_cusp_results* instead of *transformed_results*.
    """
    parts = []

    # Always show compatibility summary if Weyl data is available
    compat_html = format_dehn_compatibility(weyl)
    if compat_html:
        parts.append(compat_html)

    if (nc_results is None and transformed_results is None
            and multi_cusp_results is None):
        parts.append("""
<h3>Dehn Filling</h3>
<p class="muted">Set slopes for each cusp, configure the NC search range,
then click <b>Dehn Fill ▶</b>.</p>
<p class="muted" style="margin-top:6px; font-size:0.85em;">
The pipeline will:<br>
&nbsp;① Search for non-closable (NC) cycles in the given range
(deduplicate $\\gamma \\leftrightarrow -\\gamma$ pairs).<br>
&nbsp;② Transform your slope into each NC cycle's $(\\gamma, \\delta)$ basis.<br>
&nbsp;③ Compute the filled refined index at the transformed slope for each NC cycle.
</p>
""")
        return "\n".join(parts)

    if multi_cusp_results is not None and len(multi_cusp_results) > 0:
        # ── Multi-cusp sequential filling ─────────────────────
        parts.append(format_multi_cusp_fill_results(
            multi_cusp_results, nz=nz, max_q_terms=max_q_terms,
        ))
    elif transformed_results is not None and len(transformed_results) > 0:
        # ── Single-cusp filling ───────────────────────────────
        parts.append(format_transformed_fill_results(
            transformed_results, nz=nz, max_q_terms=max_q_terms,
        ))
    elif nc_results is not None:
        # ── Intermediate display: NC search in progress ───────
        parts.append(format_nc_cycles(nc_results, nz))

    return "\n".join(parts)
