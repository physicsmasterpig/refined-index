"""
formatters.filling_fmt_v2
==========================
Improved HTML/LaTeX formatters for Dehn Filling (Card ③) with v0.4-style design.

Features:
- Proper mathematical notation (α, β, γ, δ with subscripts)
- Sophisticated table layouts with proper alignment
- HJ continued fraction display
- Transformed slope in γᵢ/δᵢ basis
- Per-cycle Weyl vectors
- Bézout complement (δᵢ) computation for NC cycles
"""
from __future__ import annotations

from fractions import Fraction
from manifold_index.viewmodels.filling_vm import NCCycleViewModel


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def frac_to_latex(v) -> str:
    """Format a number as LaTeX: 0, 1, -2, 1/2, -3/2 etc."""
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(int(f))
    sign = "-" if f < 0 else ""
    return rf"{sign}\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"


# Alias for backwards compatibility
_frac_to_latex = frac_to_latex


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm: returns (gcd, x, y) where a*x + b*y = gcd.

    Works correctly even if a or b are negative.
    """
    if b == 0:
        if a >= 0:
            return a, 1, 0
        else:
            return -a, -1, 0
    gcd, x1, y1 = _extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return gcd, x, y


def _bezout_complement(P: int, Q: int) -> tuple[int, int]:
    """Compute Bézout complement (R, S) such that P·S - Q·R = 1.

    Given a slope (P, Q) in the (α, β) basis, compute its dual (R, S)
    such that P·S - Q·R = 1 (unimodular transformation).

    This handles signs correctly: for any (P, Q) with gcd(P,Q)=1,
    there exist unique integers (R, S) satisfying the equation.
    """
    if P == 0 and Q == 0:
        return 0, 0

    # Find (x, y) such that P·x + Q·y = gcd(P, Q)
    gcd, x, y = _extended_gcd(P, Q)

    if abs(gcd) != 1:
        # Not coprime - shouldn't happen for valid NC cycles
        # Return a default that won't cause division by zero
        return 0, 1

    # From P·x + Q·y = gcd, we want P·S - Q·R = 1
    # If gcd = 1: set S = x, R = -y  →  P·x - Q·(-y) = 1  ✓
    # If gcd = -1: set S = -x, R = y  →  P·(-x) - Q·y = -(P·x + Q·y) = -(-1) = 1  ✓
    if gcd == 1:
        return (-y, x)
    else:
        return (y, -x)


def format_slope_latex(P: int, Q: int, a: str = r"\gamma", b: str = r"\delta") -> str:
    r"""Format $P\,\gamma + Q\,\delta$ with proper sign handling.

    Examples: γ, -γ + 2δ, α - β, 0
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
            parts.append(f" + {q_part}" if parts else q_part)
        else:
            q_part = f"{-Q}\\,{b}" if Q != -1 else b
            parts.append(f" - {q_part}" if parts else f"-{q_part}")
    return "".join(parts)


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


# ─────────────────────────────────────────────────────────────────────────
# NC Cycle Table (v0.4 style)
# ─────────────────────────────────────────────────────────────────────────

def format_nc_cycle_table_html(nc_cycles: list[NCCycleViewModel]) -> str:
    """Return HTML table matching v0.4 style with γᵢ, δᵢ, and Weyl vectors.

    Columns (v0.4-style):
    - # : cycle number
    - γᵢ : NC cycle (P, Q)
    - δᵢ : Bézout complement (R, S)
    - Weyl a : Weyl vector a
    - Weyl b : Weyl vector b
    - Source : computed or cached
    """
    if not nc_cycles:
        return '<p class="muted">No non-closable cycles found.</p>\n'

    html = (
        '<table class="nc" style="font-size: 0.95em;">\n'
        '<tr>\n'
        '  <th>#</th>\n'
        '  <th>$\\gamma_i$</th>\n'
        '  <th>$\\delta_i$</th>\n'
        '  <th>$a_i$</th>\n'
        '  <th>$b_i$</th>\n'
        '  <th>Source</th>\n'
        '</tr>\n'
    )

    for i, nc in enumerate(nc_cycles, 1):
        # γᵢ = (P, Q) in the (α, β) basis
        gamma_str = format_slope_latex(nc.P, nc.Q, r"\alpha", r"\beta")

        # δᵢ = (R, S) — Bézout complement
        R, S = _bezout_complement(nc.P, nc.Q)
        delta_str = format_slope_latex(R, S, r"\alpha", r"\beta")

        # Weyl vectors
        if nc.weyl_a and nc.weyl_b:
            a_str = ", ".join(_frac_to_latex(a) for a in nc.weyl_a)
            b_str = ", ".join(_frac_to_latex(b) for b in nc.weyl_b)
            weyl_a = f"$({a_str})$"
            weyl_b = f"$({b_str})$"
        else:
            weyl_a = "—"
            weyl_b = "—"

        html += (
            f'<tr>\n'
            f'  <td style="text-align: center;"><b>{i}</b></td>\n'
            f'  <td style="text-align: center;">${gamma_str}$</td>\n'
            f'  <td style="text-align: center;">${delta_str}$</td>\n'
            f'  <td style="text-align: center;">{weyl_a}</td>\n'
            f'  <td style="text-align: center;">{weyl_b}</td>\n'
            f'  <td style="text-align: center;"><small>{nc.source}</small></td>\n'
            f'</tr>\n'
        )

    html += '</table>\n'
    return html


# ─────────────────────────────────────────────────────────────────────────
# Fill Result Display (v0.4 style)
# ─────────────────────────────────────────────────────────────────────────

def format_fill_result_as_index_row(
    user_P: int,
    user_Q: int,
    series_latex: str,
    cusp_idx: int = 0,
) -> str:
    """Format a single fill result as v0.4-style I(Aα + Bβ) = series table row.

    Returns HTML for a single <tr> with columns: I( | α | β | ) | = | series

    This matches the v0.4 sophisticated table design with proper LaTeX formatting
    and visual alignment for easy reading.
    """
    # Convert user slope to α/β notation (single cusp)
    # For single-cusp: user_P is A in Aα + Bβ, user_Q is B
    A = Fraction(user_P)
    B = Fraction(user_Q)

    # Format α column (just one term for single cusp)
    if A == 0:
        alpha_col = r"0\,\alpha"
    elif A == 1:
        alpha_col = r"\alpha"
    elif A == -1:
        alpha_col = r"-\alpha"
    else:
        alpha_col = rf"{_frac_to_latex(A)}\,\alpha"

    # Format β column
    beta_parts = []
    if B < 0:
        beta_parts.append(rf"-\; {_frac_to_latex(-B)}\,\beta")
    elif B == 0:
        beta_parts.append(r"+\; 0\,\beta")
    elif B == 1:
        beta_parts.append(r"+\; \beta")
    else:
        beta_parts.append(rf"+\; {_frac_to_latex(B)}\,\beta")
    beta_col = " ".join(beta_parts)

    # Build the table row with v0.4-style columns
    html = (
        f'<tr>\n'
        f'  <td class="i">$I($</td>\n'
        f'  <td class="al">${alpha_col}$</td>\n'
        f'  <td class="bl">${beta_col}$</td>\n'
        f'  <td class="cp">$)$</td>\n'
        f'  <td class="eq">$=$</td>\n'
        f'  <td class="sr">{series_latex}</td>\n'
        f'</tr>\n'
    )
    return html


def format_fill_result_detailed(
    nc_P: int,
    nc_Q: int,
    user_P: int,
    user_Q: int,
    p: int,
    q: int,
    weyl_a: list | None = None,
    weyl_b: list | None = None,
    hj_ks: list | None = None,
    series_latex: str = "—",
) -> str:
    """Format a filled index result with v0.4-style layout.

    Shows:
    - NC cycle as γᵢ, δᵢ
    - User slope in original basis
    - Transformed slope in new basis
    - HJ continued fraction
    - Weyl vectors
    - Result series
    """
    parts = []

    # NC cycle info
    gamma_str = format_slope_latex(nc_P, nc_Q)
    parts.append(f'<p><b>NC basis:</b> $\\gamma = {gamma_str}$</p>')

    # User slope
    user_slope = format_slope_latex(user_P, user_Q, r"\alpha", r"\beta")
    parts.append(f'<p><b>User slope:</b> ${user_slope}$</p>')

    # Transformed slope in new basis
    trans_slope = format_slope_latex(p, q, r"\gamma", r"\delta")
    parts.append(f'<p><b>Transformed:</b> ${trans_slope}$</p>')

    # HJ continued fraction (if available)
    if hj_ks:
        hj_str = "[" + ", ".join(str(k) for k in hj_ks) + "]"
        parts.append(f'<p><b>HJ:</b> $[{hj_str}]$</p>')

    # Weyl vectors (if available)
    if weyl_a and weyl_b:
        a_str = ", ".join(_frac_to_latex(a) for a in weyl_a)
        b_str = ", ".join(_frac_to_latex(b) for b in weyl_b)
        parts.append(f'<p><b>Weyl:</b> $a = ({a_str})$, $b = ({b_str})$</p>')

    # Result series
    parts.append(f'<p><b>$\\mathcal{{I}}$ result:</b> {series_latex}</p>')

    return '\n'.join(parts)


# ─────────────────────────────────────────────────────────────────────────
# Refined Series Display (from filling_fmt.py)
# ─────────────────────────────────────────────────────────────────────────

def format_filled_series_latex(
    series: dict,
    num_hard: int,
    has_cusp_eta: bool = False,
    num_cusp_eta: int = 0,
    max_q_terms: int = 4,
) -> str:
    """Convert a filled I^ref series dict to a compact KaTeX ``$...$`` string.

    Handles hard-edge η^{±2W_a} and cusp-η η^{2V_i} notation.

    Parameters
    ----------
    series : dict
        Keys ``(qq, eta_hard_0_x2, …, cusp_eta_0, …)`` → Fraction coefficients.
    num_hard : int
        Number of hard edges (η^{W_a} variables).
    has_cusp_eta : bool
        Whether cusp η variables are present.
    num_cusp_eta : int
        Number of cusp η dimensions (0 if has_cusp_eta is False).
    max_q_terms : int
        Maximum q-power groups shown before "…".
    """
    if not series:
        return "$0$"

    # ── Group by qq power ─────────────────────────────────────────────────
    by_q: dict[int, dict[tuple, Fraction]] = {}
    for key, c in series.items():
        if c == 0:
            continue
        qq = key[0]
        ek = key[1:]
        bq = by_q.setdefault(qq, {})
        bq[ek] = bq.get(ek, Fraction(0)) + Fraction(c)

    if not by_q:
        return "$0$"

    sorted_q = sorted(by_q.keys())

    # ── η formatter ───────────────────────────────────────────────────────
    def _eta_part(eta_pows: tuple) -> str:
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
        # Cusp η's (integer exponent, rendered as η^{2kV_i})
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

    def _q_factor(qq: int) -> str:
        if qq == 0:
            return ""
        if qq == 2:
            return "q"
        if qq % 2 == 0:
            return f"q^{{{qq // 2}}}"
        return rf"q^{{{qq}/2}}"

    def _coeff_str(c: Fraction) -> str:
        if c.denominator == 1:
            return str(int(c))
        sign = "-" if c < 0 else ""
        return rf"{sign}\tfrac{{{abs(c.numerator)}}}{{{c.denominator}}}"

    # ── Build terms ───────────────────────────────────────────────────────
    terms: list[str] = []
    q_count = 0
    for qq in sorted_q:
        eta_dict = by_q[qq]
        if q_count >= max_q_terms:
            terms.append(r"\cdots")
            break

        q_str = _q_factor(qq)
        sorted_eta = sorted(eta_dict.keys())
        sub_parts: list[str] = []

        for ek in sorted_eta:
            c = eta_dict[ek]
            if c == 0:
                continue
            eta_str = _eta_part(ek)
            if not eta_str:
                sub_parts.append(_coeff_str(c))
            elif c == 1:
                sub_parts.append(eta_str)
            elif c == -1:
                sub_parts.append(f"-{eta_str}")
            else:
                sub_parts.append(f"{_coeff_str(c)}{eta_str}")

        if not sub_parts:
            continue

        if len(sub_parts) == 1:
            inner = sub_parts[0]
        else:
            inner = sub_parts[0]
            for sp in sub_parts[1:]:
                inner += sp if sp.startswith("-") else "+" + sp

        if q_str:
            if len(sub_parts) == 1:
                if inner == "1":
                    terms.append(q_str)
                elif inner == "-1":
                    terms.append(f"-{q_str}")
                else:
                    terms.append(f"{inner}{q_str}")
            else:
                terms.append(f"({inner}){q_str}")
        else:
            terms.append(inner)

        q_count += 1

    if not terms:
        return "$0$"

    result_str = terms[0]
    for t in terms[1:]:
        if t.startswith("-") or t.startswith(r"\cdots"):
            result_str += (" + " if t.startswith(r"\cdots") else " ") + t
        else:
            result_str += " + " + t

    return f"${result_str}$"


def format_filled_index_table_html(
    fill_queries: list,  # list[FillQuery]
    nc_info: str = "",
) -> str:
    """Format filled index results table in v0.4-style with I(Aα + Bβ) = series rows.

    Parameters
    ----------
    fill_queries : list[FillQuery]
        List of fill query results with user slopes and series
    nc_info : str
        Optional header describing the NC cycle (e.g., "γ = 3α + 5β")

    Returns
    -------
    str
        HTML table with v0.4-style I(α + β) = series rows
    """
    if not fill_queries:
        return '<p class="muted">No fill results</p>\n'

    html = '<table class="idx" style="width: 100%;">\n'

    for fq in fill_queries:
        # Get the result series
        if hasattr(fq, 'result') and fq.result is not None:
            # For FilledRefinedResult
            if hasattr(fq.result, 'series'):
                series_latex = f"${fq.result.series}$" if isinstance(fq.result.series, str) else "$0$"
            else:
                series_latex = "$0$"
        elif hasattr(fq, 'series_latex'):
            series_latex = fq.series_latex
        else:
            series_latex = "—"

        # Format as v0.4-style row: I(Aα + Bβ) = series
        row = format_fill_result_as_index_row(
            user_P=fq.user_P,
            user_Q=fq.user_Q,
            series_latex=series_latex,
            cusp_idx=fq.cusp_idx,
        )
        html += row

    html += '</table>\n'
    return html


def format_unrefined_series_latex(
    series: dict,
    max_q_terms: int = 4,
) -> str:
    """Convert an unrefined (plain 3D) Dehn filled series to a KaTeX ``$...$`` string.

    Unrefined series has keys that are just q_half_power (integer) → Fraction coefficients.
    No η variables.

    Parameters
    ----------
    series : dict
        Keys: q_half_power (int) → Fraction coefficients
    max_q_terms : int
        Maximum q-power terms shown before "…".

    Returns
    -------
    str
        A string like ``"$-q^{1/2} + 3q + 2q^{3/2}$"``.
    """
    if not series:
        return "$0$"

    sorted_q = sorted(series.keys())
    terms: list[str] = []

    def _q_factor(qq: int) -> str:
        if qq == 0:
            return ""
        if qq == 2:
            return "q"
        if qq % 2 == 0:
            return f"q^{{{qq // 2}}}"
        return rf"q^{{{qq}/2}}"

    def _coeff_str(c: Fraction | int) -> str:
        c = Fraction(c).limit_denominator(1000)
        if c.denominator == 1:
            return str(int(c))
        sign = "-" if c < 0 else ""
        return rf"{sign}\tfrac{{{abs(c.numerator)}}}{{{c.denominator}}}"

    q_count = 0
    for qq in sorted_q:
        if q_count >= max_q_terms:
            terms.append(r"\cdots")
            break

        c = series[qq]
        if c == 0:
            continue

        q_str = _q_factor(qq)
        c_str = _coeff_str(c)

        if not q_str:
            terms.append(c_str)
        elif c == 1:
            terms.append(q_str)
        elif c == -1:
            terms.append(f"-{q_str}")
        else:
            terms.append(f"{c_str}{q_str}")

        q_count += 1

    if not terms:
        return "$0$"

    result_str = terms[0]
    for t in terms[1:]:
        if t.startswith("-") or t.startswith(r"\cdots"):
            result_str += (" + " if t.startswith(r"\cdots") else " ") + t
        else:
            result_str += " + " + t

    return f"${result_str}$"
