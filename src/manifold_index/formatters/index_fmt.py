"""
formatters.index_fmt
====================
HTML / LaTeX formatters for the Index card (Card ②).

Public API
----------
format_series_latex(result, num_hard, max_q_terms)
    → compact ``$...$`` KaTeX series string

format_index_table_html(queries, num_hard, max_q_terms)
    → ``<table class="idx">`` HTML showing I^ref rows

BLUEPRINT reference: §4 (formatters split), §11 (IndexCard display)
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product as itertools_product
from typing import Any


# ---------------------------------------------------------------------------
# Series → KaTeX
# ---------------------------------------------------------------------------

def format_series_latex(
    result: dict,
    num_hard: int,
    max_q_terms: int = 4,
) -> str:
    """Convert an ``I^ref`` result dict to a compact KaTeX ``$...$`` string.

    Parameters
    ----------
    result : dict
        Keys are tuples ``(q_half_power, eta_1_x2, eta_2_x2, …)``.
        Values are integer or Fraction coefficients.
    num_hard : int
        Number of hard edges (η variables).
    max_q_terms : int
        Maximum distinct q-power groups to show (rest replaced by ``…``).

    Returns
    -------
    str
        A string like ``"$1 + q + (-2 + \\eta^{2W_0})q^2 + \\cdots$"``.
    """
    if not result:
        return "$0$"

    # ── Group by q-half power ─────────────────────────────────────────────
    by_q: dict[int, dict[tuple, Any]] = {}
    for key, coeff in result.items():
        if coeff == 0:
            continue
        q_half = key[0]
        eta_key = key[1:]
        by_q.setdefault(q_half, {})[eta_key] = (
            by_q.get(q_half, {}).get(eta_key, 0) + coeff
        )

    if not by_q:
        return "$0$"

    sorted_q = sorted(by_q.keys())

    # ── Inner helpers ─────────────────────────────────────────────────────

    def _eta_part(eta_pows: tuple) -> str:
        parts: list[str] = []
        for a, exp2 in enumerate(eta_pows):
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
        return "".join(parts)

    def _q_factor(q_half: int) -> str:
        if q_half == 0:
            return ""
        if q_half == 2:
            return "q"
        if q_half % 2 == 0:
            n = q_half // 2
            return f"q^{{{n}}}"
        return rf"q^{{{q_half}/2}}"

    # ── Build term strings ────────────────────────────────────────────────
    terms: list[str] = []
    q_count = 0
    for q_half in sorted_q:
        eta_dict = by_q[q_half]
        if q_count >= max_q_terms:
            terms.append(r"\cdots")
            break

        q_str = _q_factor(q_half)
        sorted_eta = sorted(eta_dict.keys())
        sub_parts: list[str] = []

        for ek in sorted_eta:
            c = eta_dict[ek]
            if c == 0:
                continue
            eta_str = _eta_part(ek)
            if not eta_str:
                sub_parts.append(str(int(c)) if int(c) == c else str(c))
            elif c == 1:
                sub_parts.append(eta_str)
            elif c == -1:
                sub_parts.append(f"-{eta_str}")
            else:
                sub_parts.append(f"{int(c) if int(c) == c else c}{eta_str}")

        if not sub_parts:
            continue

        if len(sub_parts) == 1:
            part = sub_parts[0]
            if q_str:
                if part == "1":
                    terms.append(q_str)
                elif part == "-1":
                    terms.append(f"-{q_str}")
                else:
                    terms.append(f"{part}{q_str}")
            else:
                terms.append(part)
        else:
            inner = sub_parts[0]
            for sp in sub_parts[1:]:
                inner += sp if sp.startswith("-") else "+" + sp
            if q_str:
                terms.append(f"({inner}){q_str}")
            else:
                terms.append(inner)

        q_count += 1

    if not terms:
        return "$0$"

    # ── Join terms with sign-aware "+" ─────────────────────────────────────
    result_str = terms[0]
    for t in terms[1:]:
        if t.startswith("-") or t.startswith(r"\cdots"):
            result_str += (" + " if t.startswith(r"\cdots") else " ") + t
        else:
            result_str += " + " + t

    return f"${result_str}$"


# ---------------------------------------------------------------------------
# Index table HTML
# ---------------------------------------------------------------------------

# Canonical display charges: (alpha_coeff, beta_coeff) per cusp
# Convention: e·α − (m/2)·β  →  alpha = e, beta = −m/2
DISPLAY_CHARGES = [
    (0, 0),
    (0, Fraction(-1, 2)),
    (Fraction(1, 2), 0),
    (0, -1),
    (1, 0),
]


def _frac_to_latex(v) -> str:
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(int(f))
    sign = "-" if f < 0 else ""
    return rf"{sign}\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def _charge_to_me(alpha: Fraction, beta: Fraction) -> tuple:
    """(alpha_coeff, beta_coeff) → (m, e) for compute_refined_index."""
    e = Fraction(alpha)
    m = int(-beta * 2)
    return m, e


def _alpha_latex(coeff, cusp: int) -> str:
    c = Fraction(coeff)
    if c == 0:
        return rf"0\,\alpha_{{{cusp + 1}}}"
    return rf"{_frac_to_latex(c)}\,\alpha_{{{cusp + 1}}}"


def _beta_latex(coeff, cusp: int) -> str:
    c = Fraction(coeff)
    if c == 0:
        return rf"0\,\beta_{{{cusp + 1}}}"
    return rf"{_frac_to_latex(c)}\,\beta_{{{cusp + 1}}}"


def format_index_table_html(
    entries: list,
    num_hard: int,
    num_cusps: int = 1,
    max_q_terms: int = 4,
) -> str:
    """Build the refined-index table HTML.

    Parameters
    ----------
    entries : list of (m_ext, e_ext, result_dict)
        Computed sectors — each ``result_dict`` maps ``(q_half, *etas)``
        to int/Fraction coefficients.
    num_hard : int
        Number of hard edges (η variables).
    num_cusps : int
        Number of cusps (= r in NZDATA).  Determines the charge grid.
    max_q_terms : int
        Truncation for the series display.

    Returns
    -------
    str
        HTML ``<table class="idx">`` block.
    """
    r = num_cusps

    # Build lookup: (tuple(m), tuple(e)) → result_dict
    lookup: dict[tuple, dict] = {}
    for m_ext, e_ext, res in entries:
        lookup[(tuple(m_ext), tuple(e_ext))] = res

    display_combos = list(itertools_product(DISPLAY_CHARGES, repeat=r))
    n_display = len(display_combos)

    n_nonzero = sum(
        1 for combo in display_combos
        if _combo_lookup(combo, r, lookup)
    )

    html = (
        f'<p class="muted">$5^{{{r}}} = {n_display}$ sectors '
        f"({n_nonzero} non-zero)</p>\n"
        '<table class="idx">\n'
    )

    for combo in display_combos:
        m_ext, e_ext, alphas, betas = _expand_combo(combo, r)
        key = (tuple(m_ext), tuple(e_ext))
        res = lookup.get(key)
        series_str = format_series_latex(res, num_hard, max_q_terms) if res else "$0$"

        alpha_col = _build_alpha_col(alphas, r)
        beta_col = _build_beta_col(betas, r)

        html += (
            f"<tr>"
            f'<td class="i">$I($</td>'
            f'<td class="al">${alpha_col}$</td>'
            f'<td class="bl">${beta_col}$</td>'
            f'<td class="cp">$)$</td>'
            f'<td class="eq">$=$</td>'
            f'<td class="sr">{series_str}</td>'
            f"</tr>\n"
        )

    html += "</table>\n"
    return html


# ── format_index_table_html helpers ──────────────────────────────────────

def _combo_lookup(combo, r, lookup):
    m_ext, e_ext, _, _ = _expand_combo(combo, r)
    return lookup.get((tuple(m_ext), tuple(e_ext)))


def _expand_combo(combo, r):
    m_ext, e_ext, alphas, betas = [], [], [], []
    for alpha, beta in combo:
        m, e = _charge_to_me(Fraction(alpha), Fraction(beta))
        m_ext.append(m)
        e_ext.append(e)
        alphas.append(Fraction(alpha))
        betas.append(Fraction(beta))
    return m_ext, e_ext, alphas, betas


def _build_alpha_col(alphas, r) -> str:
    col = _alpha_latex(alphas[0], 0)
    for i in range(1, r):
        a = alphas[i]
        if a < 0:
            col += rf" -\; {_frac_to_latex(-a)}\,\alpha_{{{i + 1}}}"
        else:
            col += rf" +\; {_alpha_latex(a, i)}"
    return col


def _build_beta_col(betas, r) -> str:
    parts = []
    for i in range(r):
        b = betas[i]
        if b < 0:
            parts.append(rf"-\; {_frac_to_latex(-b)}\,\beta_{{{i + 1}}}")
        else:
            parts.append(rf"+\; {_beta_latex(b, i)}")
    return " ".join(parts)
