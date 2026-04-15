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
    """Return (c, d) = (−R, −S) matching the actual basis-change longitude.

    The fill computation uses ``find_rs(P, Q)`` → (R, S) with R·Q − P·S = 1
    and applies the SL(2,ℤ) basis change with matrix [[P, Q], [−R, −S]].
    The new longitude is therefore δ = (−R)·α + (−S)·β.

    This function returns exactly that (−R, −S) so the displayed δ matches
    what the computation actually uses.
    """
    if P == 0 and Q == 0:
        return 0, 0
    try:
        from manifold_index.core.dehn_filling import find_rs as _find_rs
        R, S = _find_rs(P, Q)
        return -R, -S
    except Exception:
        return 0, 1


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
    """Format a single alpha term like '0\\alpha_1' or '-\\tfrac{1}{2}\\alpha_2'."""
    c = Fraction(coeff)
    if c == 0:
        return rf"0\alpha_{cusp + 1}"
    return rf"{_frac_to_latex(c)}\alpha_{cusp + 1}"


def _beta_latex(coeff: Fraction, cusp: int) -> str:
    """Format a single beta term like '0\\beta_1' or '1\\beta_2'."""
    c = Fraction(coeff)
    if c == 0:
        return rf"0\beta_{cusp + 1}"
    return rf"{_frac_to_latex(c)}\beta_{cusp + 1}"


# ─────────────────────────────────────────────────────────────────────────
# NC Cycle Table (v0.4 style)
# ─────────────────────────────────────────────────────────────────────────

def format_nc_cycle_table_html(nc_cycles: list[NCCycleViewModel]) -> str:
    """Return HTML table with γᵢ, δᵢ, Weyl vectors, and strongly-NC status.

    Columns:
    - #         : cycle number
    - γᵢ        : NC cycle (P, Q)
    - δᵢ        : Bézout complement (R, S)
    - aᵢ        : Weyl a vector
    - bᵢ        : Weyl b vector
    - Unref. q^1: projection value; "Marginal" label when proj ≥ 0
    - Source    : computed or cached
    """
    if not nc_cycles:
        return '<p class="muted">No non-closable cycles found.</p>\n'

    html = (
        '<table class="nc" style="font-size: 0.95em;">\n'
        '<tr>\n'
        '  <th>#</th>\n'
        '  <th>$\\gamma_i$</th>\n'
        '  <th>$\\delta_i$</th>\n'
        '  <th>$\\left.\\textrm{Coeff}_{q^1}(\\mathcal{I}^{\\vec{\gamma},\\vec{\delta}}(\\vec{m}=0,\\vec{u}))\\right|_{\\textrm{adj}\\,\\mathfrak{su}(2)_i}$</th>\n'
        '  <th>Source</th>\n'
        '</tr>\n'
    )

    for i, nc in enumerate(nc_cycles, 1):
        # γᵢ = (P, Q) in the (α, β) basis
        gamma_str = format_slope_latex(nc.P, nc.Q, r"\alpha", r"\beta")

        # δᵢ = (R, S) — Bézout complement
        R, S = _bezout_complement(nc.P, nc.Q)
        delta_str = format_slope_latex(R, S, r"\alpha", r"\beta")

        # Unrefined q^1 projection column.
        # Notation: \left.\textrm{Coeff}_{q^1}\,
        #               \mathcal{I}^{(\vec{\gamma},\vec{\delta})}(\vec{m}=0,\vec{u})
        #           \right|_{(\textrm{adj}\,\mathfrak{su}(2)_i)}
        # "Marginal" label shown when proj ≥ 0.
        is_marginal = getattr(nc, 'is_marginal', None)
        snc_val     = getattr(nc, 'unrefined_q1_proj', None)
        
        if is_marginal is None:
            snc_cell = (
                f'<span style="white-space:nowrap;color:#888">'
                f'&nbsp;—'
                f'</span>'
            )
        else:
            # Green when proj ≤ -1 (non-marginal), red + "Marginal" when proj ≥ 0
            color     = "#f85149" if is_marginal else "#3fb950"
            val_latex = _frac_to_latex(snc_val) if snc_val is not None else r"\cdot"
            marginal_tag = '&nbsp;<span style="color:#f85149"><b>Marginal</b></span>' if is_marginal else ''
            snc_cell = (
                f'<span style="white-space:nowrap">'
                f'$\\color{{{color}}}{{{val_latex}}}$'
                f'{marginal_tag}'
                f'</span>'
            )

        html += (
            f'<tr>\n'
            f'  <td style="text-align: center;"><b>{i}</b></td>\n'
            f'  <td style="text-align: center;">${gamma_str}$</td>\n'
            f'  <td style="text-align: center;">${delta_str}$</td>\n'
            f'  <td style="text-align: center;">{snc_cell}</td>\n'
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
    max_q_terms: int = 9999,
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
            if ce == 1:
                parts.append(rf"\eta^{{V_{ci}}}")
            elif ce == -1:
                parts.append(rf"\eta^{{-V_{ci}}}")
            else:
                parts.append(rf"\eta^{{{ce}V_{ci}}}")
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


def format_refined_index_notation(m_ext: list, e_ext: list) -> tuple[str, str]:
    """Convert (m, e) charge tuples to I(Aα + Bβ) notation.

    Convention: e·α − (m/2)·β  →  alpha_coeff = e, beta_coeff = −m/2

    For single-cusp (r=1):
        m_ext = [m], e_ext = [e]
        Returns: (HTML cells for I(eα - (m/2)β), "=") in proper LaTeX

    For multi-cusp (r>1):
        m_ext = [m₁, m₂, ...], e_ext = [e₁, e₂, ...]
        Returns: (HTML cells for I(e₁α₁ - (m₁/2)β₁ + ...), "=") in proper LaTeX

    Returns
    -------
    tuple[str, str]
        (index_notation_html, equals_html)
        where index_notation_html contains <td> elements for I, α, β, )
        and equals_html contains <td class="eq">$=$</td>
    """
    if not m_ext or not e_ext:
        index_html = '<td class="i">$\\mathcal{I}($</td><td class="al">$0$</td><td class="bl"></td><td class="cp">$)$</td>'
        equals_html = '<td class="eq">$=$</td>'
        return index_html, equals_html

    r = len(m_ext)
    alphas = [Fraction(e_ext[i]) if i < len(e_ext) else Fraction(0) for i in range(r)]
    betas = [Fraction(-m_ext[i], 2) if i < len(m_ext) else Fraction(0) for i in range(r)]

    # Build α column
    if r == 1:
        # Single cusp: just show α, not α₁
        if alphas[0] == 0:
            alpha_col = r"0\alpha"
        elif alphas[0] == 1:
            alpha_col = r"\alpha"
        elif alphas[0] == -1:
            alpha_col = r"-\alpha"
        else:
            alpha_col = rf"{_frac_to_latex(alphas[0])}\alpha"
    else:
        # Multi-cusp: show α₁, α₂, etc.
        alpha_col = _alpha_latex(alphas[0], 0)
        for i in range(1, r):
            a = alphas[i]
            if a < 0:
                alpha_col += rf" -\; {_frac_to_latex(-a)}\alpha_{i + 1}"
            else:
                alpha_col += rf" +\; {_alpha_latex(a, i)}"

    # Build β column with proper sign handling
    beta_parts = []
    for i in range(r):
        b = betas[i]
        cusp_idx = i if r > 1 else 0

        if r == 1:
            # Single cusp
            if b < 0:
                beta_parts.append(rf"-\; {_frac_to_latex(-b)}\beta")
            elif b == 0:
                beta_parts.append(r"+\; 0\beta")
            elif b == 1:
                beta_parts.append(r"+\; \beta")
            else:
                beta_parts.append(rf"+\; {_frac_to_latex(b)}\beta")
        else:
            # Multi-cusp
            if i == 0:
                if b < 0:
                    beta_parts.append(rf"-\; {_frac_to_latex(-b)}\beta_{cusp_idx + 1}")
                elif b == 0:
                    beta_parts.append(rf"+\; 0\beta_{cusp_idx + 1}")
                else:
                    beta_parts.append(rf"+\; {_frac_to_latex(b)}\beta_{cusp_idx + 1}")
            else:
                if b < 0:
                    beta_parts.append(rf"-\; {_frac_to_latex(-b)}\beta_{cusp_idx + 1}")
                else:
                    beta_parts.append(rf"+\; {_frac_to_latex(b)}\beta_{cusp_idx + 1}")

    beta_col = " ".join(beta_parts)

    # Combine as v0.4-style I(α + β) notation
    index_html = (
        f'<td class="i">$\\mathcal{{I}}($</td>'
        f'<td class="al">${alpha_col}$</td>'
        f'<td class="bl">${beta_col}$</td>'
        f'<td class="cp">$)$</td>'
    )
    equals_html = '<td class="eq">$=$</td>'
    return index_html, equals_html


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
    max_q_terms: int = 9999,
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


# ─────────────────────────────────────────────────────────────────────────
# Fill info panel  (NC cycle · transformed slope · HJ k-vector)
# ─────────────────────────────────────────────────────────────────────────

def format_fill_info_html(
    cusp_specs_aug: list,
    result,
    adjoint_per_cusp: "list[dict] | None" = None,
) -> str:
    r"""HTML for the fill-info panel shown above the result table.

    One row per filled cusp with columns:
      C_i | γ = P·α+Q·β | δ = R·α+S·β | Slope: p·γ+q·δ | k=[…], ℓ
      | \left.\mathcal{I}_{q^1}(\eta,\vec{u};\mathbf{a},\mathbf{b})
             \right|_{\mathrm{adj}\,\mathfrak{su}(2)_{n+I}} = value ✓/✗

    The adjoint column uses I = 1-based index among the filled cusps.
    """
    if not cusp_specs_aug:
        return ""

    hj_ks: list = []
    if result is not None and hasattr(result, "hj_ks"):
        hj_ks = list(result.hj_ks)

    # Build cusp_idx → adjoint entry map for O(1) lookup per row
    adj_map: dict = {}
    if adjoint_per_cusp:
        for entry in adjoint_per_cusp:
            adj_map[entry.get("cusp_idx")] = entry

    rows: list = []
    for fill_idx, spec in enumerate(cusp_specs_aug, 1):   # I = 1, 2, …, d
        nc_P = int(spec.get("nc_P", 1))
        nc_Q = int(spec.get("nc_Q", 0))
        p    = int(spec.get("p", 0))
        q    = int(spec.get("q", 1))
        ci   = spec.get("cusp_idx", 0)

        R, S = _bezout_complement(nc_P, nc_Q)
        gamma_str = format_slope_latex(nc_P, nc_Q, a=r"\alpha", b=r"\beta")
        delta_str = format_slope_latex(R, S,    a=r"\alpha", b=r"\beta")
        slope_str = format_slope_latex(p, q,    a=r"\gamma", b=r"\delta")

        k_cell = ""
        if hj_ks:
            k_inner = ",\\,".join(str(k) for k in hj_ks)
            k_cell = (
                f"<td style='padding:2px 12px 2px 0;white-space:nowrap'>"
                f"$\\mathbf{{k}}=[{k_inner}]$,&nbsp;$\\ell={len(hj_ks)}$"
                f"</td>"
            )

        # ── Weyl a / b columns (for this filled cusp) ──────────────────
        weyl_a = spec.get("weyl_a")
        weyl_b = spec.get("weyl_b")
        if weyl_a is not None:
            a_str = ",\\,".join(_frac_to_latex(v) for v in weyl_a)
            a_cell = (
                f"<td style='padding:2px 12px 2px 0;white-space:nowrap'>"
                f"$a=({a_str})$</td>"
            )
        else:
            a_cell = "<td style='padding:2px 12px 2px 0;color:#888'>—</td>"
        if weyl_b is not None:
            b_str = ",\\,".join(_frac_to_latex(v) for v in weyl_b)
            b_cell = (
                f"<td style='padding:2px 12px 2px 0;white-space:nowrap'>"
                f"$b=({b_str})$</td>"
            )
        else:
            b_cell = "<td style='padding:2px 12px 2px 0;color:#888'>—</td>"

        # ── Adjoint column ──────────────────────────────────────────────
        # Notation: \left.\mathcal{I}_{q^1}(\eta,\vec{u};\mathbf{a},\mathbf{b})
        #             \right|_{\mathrm{adj}\,\mathfrak{su}(2)_{n+I}}
        # where I = fill_idx (1-based among filled cusps).
        subscript = f"n+{fill_idx}" if fill_idx > 1 else "n+1"
        lhs = (
            r"\left.\mathcal{I}_{q^1}"
            r"(\eta,\vec{u};\mathbf{a},\mathbf{b})"
            rf"\right|_{{(\textrm{{adj}}\,\mathfrak{{su}}(2)_{{{subscript}}})}}"
        )

        entry = adj_map.get(ci)
        if entry is None:
            # No adjoint data yet (async worker not done)
            adj_cell = (
                f"<td style='padding:2px 0 2px 12px;white-space:nowrap;color:#888'>"
                f"${lhs}$&nbsp;—"
                f"</td>"
            )
        else:
            val = entry.get("value")
            ok  = entry.get("is_pass")
            if val is None:
                adj_cell = (
                    f"<td style='padding:2px 0 2px 12px;white-space:nowrap;color:#888'>"
                    f"${lhs}$&nbsp;—"
                    f"</td>"
                )
            else:
                color  = "#3fb950" if ok else "#f85149"
                symbol = "✓" if ok else "✗"
                adj_cell = (
                    f"<td style='padding:2px 0 2px 12px;white-space:nowrap'>"
                    f"${lhs}="
                    f"\\color{{{color}}}{{{_frac_to_latex(val)}}}$"
                    f"&nbsp;<span style='color:{color}'>{symbol}</span>"
                    f"</td>"
                )

        rows.append(
            f"<tr>"
            f"<td style='padding:2px 10px 2px 0;white-space:nowrap;font-weight:bold'>C{ci}</td>"
            f"<td style='padding:2px 12px 2px 0;white-space:nowrap'>$\\gamma={gamma_str}$</td>"
            f"<td style='padding:2px 12px 2px 0;white-space:nowrap'>$\\delta={delta_str}$</td>"
            f"{a_cell}"
            f"{b_cell}"
            f"<td style='padding:2px 12px 2px 0;white-space:nowrap'>Slope:&nbsp;${slope_str}$</td>"
            f"{k_cell}"
            f"{adj_cell}"
            f"</tr>"
        )

    css = "<style>body{font-size:11px;margin:4px 0}table{border-collapse:collapse}td{vertical-align:baseline}</style>"
    return css + "<table>" + "".join(rows) + "</table>"


def format_charge_as_alphabeta(m, e, cusp_idx=None) -> str:
    """Format a single (m, e) charge as e·α − (m/2)·β in LaTeX (no $ delimiters)."""
    A = Fraction(e).limit_denominator(1000)
    B = Fraction(-int(round(float(m))), 2)

    suffix = f"_{{{cusp_idx + 1}}}" if cusp_idx is not None else ""
    return format_slope_latex(
        int(A) if A.denominator == 1 else A,
        int(B) if B.denominator == 1 else B,
        a=rf"\alpha{suffix}",
        b=rf"\beta{suffix}",
    )


def format_multi_fill_row_label(cusp_specs_aug: list, unfilled_charges: list, all_cusp_count: int) -> str:
    """Build the 'm' column metadata for a multi-fill result row.

    Filled cusps: NC cycle γ = P·α + Q·β and user slope.
    Unfilled cusps: I(Aα + Bβ) charge notation.
    """
    filled_idxs = {s["cusp_idx"] for s in cusp_specs_aug}
    unfilled_idxs = [i for i in range(all_cusp_count) if i not in filled_idxs]

    parts: list = []
    for spec in cusp_specs_aug:
        nc_P, nc_Q = int(spec["nc_P"]), int(spec["nc_Q"])
        user_P, user_Q = int(spec["user_P"]), int(spec["user_Q"])
        p_val = spec.get("p")
        q_val = spec.get("q")
        ci = spec["cusp_idx"]
        gamma = format_slope_latex(nc_P, nc_Q, a=r"\alpha", b=r"\beta")
        slope = format_slope_latex(user_P, user_Q, a=r"\alpha", b=r"\beta")
        line = f"C{ci}: $\\gamma={gamma}$, $={slope}$"
        if p_val is not None and q_val is not None:
            trans = format_slope_latex(int(p_val), int(q_val), a=r"\gamma", b=r"\delta")
            line += f" → ${trans}$"
        parts.append(line)

    if unfilled_charges and unfilled_idxs:
        charge_strs: list = []
        for (m, e), uidx in zip(unfilled_charges, unfilled_idxs):
            ab = format_charge_as_alphabeta(m, e, cusp_idx=uidx)
            charge_strs.append(ab)
        inner = ",\\,".join(charge_strs)
        parts.append(f"$\\mathcal{{I}}({inner})$")

    return "<br>".join(parts)


# ─────────────────────────────────────────────────────────────────────────
# SeriesTable row-cell builders  (html-notation mode)
# ─────────────────────────────────────────────────────────────────────────

def _alpha_chrg(A: Fraction) -> str:
    """Format A·α for the right-aligned alpha cell."""
    if A == 0:  return r"0\,\alpha"
    if A == 1:  return r"\alpha"
    if A == -1: return r"-\alpha"
    return rf"{frac_to_latex(A)}\,\alpha"


def _beta_chrg(B: Fraction) -> str:
    """Format ±B·β with leading sign for the left-aligned beta cell."""
    if B == 0:
        return ""
    absB = abs(B)
    sign = "+" if B > 0 else "-"
    if absB == 1:
        return rf"{sign}\,\beta"
    return rf"{sign}{frac_to_latex(absB)}\,\beta"


def _charge_ab_strs(m_other: list, e_other: list) -> tuple[str, str]:
    """Convert first unfilled-cusp (m, e) to (alpha_str, beta_str)."""
    m_val = m_other[0] if m_other else 0
    e_val = e_other[0] if e_other else Fraction(0)
    A = Fraction(e_val).limit_denominator(1000)
    B = Fraction(-int(round(float(m_val))), 2)   # −m/2
    if A == 0 and B == 0:
        return "0", ""
    return _alpha_chrg(A), _beta_chrg(B)


def _has_charge(m_other: list, e_other: list) -> bool:
    """Return True when there is at least one non-zero unfilled-cusp charge."""
    return bool(m_other) or bool(e_other)


def _charge_cells_html(alpha_str: str, beta_str: str) -> str:
    """Build the three middle <td> cells: (  Aα  +Bβ  )."""
    beta_td = f"<td class='bl'>${beta_str}$</td>" if beta_str else "<td class='bl'></td>"
    return (
        f"<td class='al'>${alpha_str}$</td>"
        f"{beta_td}"
        f"<td class='cp'>$)$</td>"
    )


def build_fill_row_cells(
    p: int, q: int,
    m_other: list, e_other: list,
    slope_a: str = r"\gamma",
    slope_b: str = r"\delta",
) -> tuple[str, str]:
    r"""Build (m_td_cells, eq_td_cell) for SeriesTable html-notation.

    * No unfilled cusps (m_other/e_other empty): 4 cells showing
      ``\mathcal{I}^{pγ+qδ}`` with the three argument cells left blank.
    * With unfilled cusps: 4 cells showing
      ``\mathcal{I}^{pγ+qδ}(  Aα  +Bβ  )``.

    Pass ``slope_a=r"\alpha", slope_b=r"\beta"`` for unrefined rows.
    """
    sup = format_slope_latex(p, q, a=slope_a, b=slope_b)
    eq_cell = "<td class='eq'>$=$</td>"

    if not _has_charge(m_other, e_other):
        # No unfilled cusps — omit the (...) argument entirely
        return (
            f"<td class='i'>$\\mathcal{{I}}^{{{sup}}}$</td>"
            "<td class='al'></td>"
            "<td class='bl'></td>"
            "<td class='cp'></td>"
        ), eq_cell

    prefix = rf"\mathcal{{I}}^{{{sup}}}("
    alpha_str, beta_str = _charge_ab_strs(m_other, e_other)
    return (
        f"<td class='i'>${prefix}$</td>"
        + _charge_cells_html(alpha_str, beta_str)
    ), eq_cell


def build_fill_placeholder_cells(m_other: list, e_other: list) -> tuple[str, str]:
    r"""Placeholder row cells while the transformed slope is not yet known.

    No-charge case: ``\mathcal{I}^{\ldots}``.
    Charge case:    ``\mathcal{I}^{\ldots}(Aα + Bβ)``.
    Updated via ``update_row_metadata`` once the worker payload arrives.
    """
    eq_cell = "<td class='eq'>$=$</td>"

    if not _has_charge(m_other, e_other):
        return (
            r"<td class='i'>$\mathcal{I}^{\ldots}$</td>"
            "<td class='al'></td>"
            "<td class='bl'></td>"
            "<td class='cp'></td>"
        ), eq_cell

    alpha_str, beta_str = _charge_ab_strs(m_other, e_other)
    return (
        r"<td class='i'>$\mathcal{I}^{\ldots}($</td>"
        + _charge_cells_html(alpha_str, beta_str)
    ), eq_cell


# ─────────────────────────────────────────────────────────────────────────
# Multi-fill row-cell builders  (cusp-subscripted γᵢ/δᵢ and αᵢ/βᵢ)
# ─────────────────────────────────────────────────────────────────────────

def _build_filled_superscript(cusp_specs: list) -> str:
    """Build P'₁γ₁+Q'₁δ₁+P'₂γ₂+Q'₂δ₂ from filled cusp specs.

    Each spec must have keys ``cusp_idx``, ``p``, ``q``.
    """
    sup = ""
    for spec in cusp_specs:
        ci = spec["cusp_idx"]
        p  = spec.get("p", 0)
        q  = spec.get("q", 0)
        if p == 0 and q == 0:
            continue
        term = format_slope_latex(p, q, a=rf"\gamma_{{{ci}}}", b=rf"\delta_{{{ci}}}")
        if sup and not term.startswith("-"):
            sup += "+"
        sup += term
    return sup or "0"


def _build_unfilled_charge_cells_html(unfilled_cusp_charges: list) -> str:
    r"""Build 4x+(x-1) ``<td>`` cells for x unfilled cusps.

    Each cusp contributes 4 cells: coeff_α | sym_α | coeff_β | sym_β
    Separator between cusp groups: ``[+]_eq``  (x-1 of these)

    Columns align vertically across rows:
      [A₀]  [α₀]  [±B₀]  [β₀]   +   [A₁]  [α₁]  [±B₁]  [β₁]
       al    sym    al     sym   eq    al    sym    al     sym
    """
    cells = ""
    for idx, (ci, m, e) in enumerate(unfilled_cusp_charges):
        A = Fraction(e).limit_denominator(1000)
        B = Fraction(-int(round(float(m))), 2)

        # Inter-cusp separator
        if idx > 0:
            cells += "<td class='eq'>$+$</td>"

        # α coefficient (right-aligned, no forced leading sign)
        a_coeff = frac_to_latex(A)
        cells += f"<td class='al'>${a_coeff}$</td>"

        # α symbol (left-aligned)
        cells += f"<td class='sym'>$\\alpha_{{{ci}}}$</td>"

        # β coefficient (right-aligned, always with explicit sign)
        if B >= 0:
            b_coeff = f"+{frac_to_latex(B)}"
        else:
            b_coeff = frac_to_latex(B)   # frac_to_latex already adds −
        cells += f"<td class='al'>${b_coeff}$</td>"

        # β symbol (left-aligned)
        cells += f"<td class='sym'>$\\beta_{{{ci}}}$</td>"

    return cells


def build_multi_fill_row_cells(
    cusp_specs: list,
    unfilled_cusp_charges: list,   # [(cusp_idx, m, e), …]
) -> tuple[str, str]:
    r"""Build final row cells for a multi-cusp fill result.

    m_val always has the structure:
      [i: I^{…}(] + [3x-1 charge cells] + [cp: )]
    = 3x+1 total cells, where x = len(unfilled_cusp_charges).

    For x=0: [i: I^{…}] [al:] [bl:] [cp:] (4 cells, no argument).
    """
    sup     = _build_filled_superscript(cusp_specs)
    eq_cell = "<td class='eq'>$=$</td>"

    if not unfilled_cusp_charges:
        return (
            f"<td class='i'>$\\mathcal{{I}}^{{{sup}}}$</td>"
            "<td class='al'></td>"
            "<td class='bl'></td>"
            "<td class='cp'></td>"
        ), eq_cell

    prefix       = rf"\mathcal{{I}}^{{{sup}}}("
    charge_cells = _build_unfilled_charge_cells_html(unfilled_cusp_charges)
    return (
        f"<td class='i'>${prefix}$</td>"
        + charge_cells
        + "<td class='cp'>$)$</td>"
    ), eq_cell


def build_multi_fill_placeholder_cells(
    cusp_specs: list,              # one dict per filled cusp (has user_P, user_Q, cusp_idx)
    unfilled_cusp_charges: list,   # [(cusp_idx, m, e), …]
) -> tuple[str, str]:
    r"""Placeholder cells for multi-fill while p′/q′ are not yet known.

    Uses user slopes (in α/β basis) as the superscript so that no ``\ldots``
    ever appears; ``update_row_metadata`` will replace these with the true
    γ/δ-transformed slopes once the worker finishes.
    """
    # Build superscript from user slopes (α/β basis, subscripted)
    sup = ""
    for spec in cusp_specs:
        ci = spec.get("cusp_idx", 0)
        uP = spec.get("user_P", 0)
        uQ = spec.get("user_Q", 0)
        if uP == 0 and uQ == 0:
            continue
        term = format_slope_latex(uP, uQ,
                                  a=rf"\alpha_{{{ci}}}", b=rf"\beta_{{{ci}}}")
        if sup and not term.startswith("-"):
            sup += "+"
        sup += term
    if not sup:
        sup = "?"

    eq_cell = "<td class='eq'>$=$</td>"

    if not unfilled_cusp_charges:
        return (
            f"<td class='i'>$\\mathcal{{I}}^{{{sup}}}$</td>"
            "<td class='al'></td>"
            "<td class='bl'></td>"
            "<td class='cp'></td>"
        ), eq_cell

    charge_cells = _build_unfilled_charge_cells_html(unfilled_cusp_charges)
    prefix = rf"\mathcal{{I}}^{{{sup}}}("
    return (
        f"<td class='i'>${prefix}$</td>"
        + charge_cells
        + "<td class='cp'>$)$</td>"
    ), eq_cell
