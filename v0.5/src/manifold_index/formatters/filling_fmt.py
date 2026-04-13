"""
formatters.filling_fmt
======================
HTML / LaTeX formatters for the Dehn Filling card (Card ③).

Public API
----------
format_filled_series_latex(series, num_hard, has_cusp_eta, max_q_terms)
    → compact ``$...$`` KaTeX string for a filled index series

format_nc_cycle_table_html(nc_cycles)
    → ``<table class="nc">`` HTML listing NC cycle slopes

format_fill_result_html(fq)
    → HTML fragment for one FillQueryViewModel result row

format_slope_latex(P, Q)
    → ``$P\\alpha + Q\\beta$`` KaTeX (public convenience re-export)

BLUEPRINT reference: §4 (formatters split), §11 (FillingCard display)
"""
from __future__ import annotations

from fractions import Fraction

from manifold_index.viewmodels.filling_vm import (
    FillQueryViewModel,
    NCCycleViewModel,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frac_to_latex(v) -> str:
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(int(f))
    sign = "-" if f < 0 else ""
    return rf"{sign}\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def format_slope_latex(P: int, Q: int,
                       a: str = r"\gamma", b: str = r"\delta") -> str:
    r"""Format $P\,\gamma + Q\,\delta$ with correct sign handling.

    The default basis is (γ, δ) = (meridian, longitude) for Dehn filling.
    Returns a bare LaTeX string (no surrounding ``$``).
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


# ---------------------------------------------------------------------------
# Filled series → KaTeX
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NC cycle table
# ---------------------------------------------------------------------------

def format_nc_cycle_table_html(
    nc_cycles: list[NCCycleViewModel],
    weyl_a: list | None = None,
    weyl_b: list | None = None,
    adjoint_proj_pass: bool | None = None,
) -> str:
    """Return ``<table class="nc">`` with per-cycle Weyl vectors in expanded view.

    Each NC cycle shows its own Weyl vectors (from basis change in that cycle).

    Parameters
    ----------
    nc_cycles : list[NCCycleViewModel]
        Each VM contains its own weyl_a, weyl_b, adjoint_proj_pass (per-cycle).
    weyl_a, weyl_b, adjoint_proj_pass
        (Legacy parameters, ignored if per-cycle data present)
    """
    if not nc_cycles:
        return '<p class="muted">No non-closable cycles found.</p>\n'

    header = (
        '<table class="nc">\n'
        "<tr>"
        "<th>#</th>"
        "<th>Slope $\\gamma$</th>"
        "<th>Weyl Vectors (a, b)</th>"
        "<th>q¹ Projection</th>"
        "<th>Source</th>"
        "</tr>\n"
    )
    rows = ""
    for i, nc in enumerate(nc_cycles, 1):
        # Format Weyl vectors for this cycle
        if nc.weyl_a and nc.weyl_b:
            a_str = ", ".join(str(a) for a in nc.weyl_a[:min(2, len(nc.weyl_a))])
            b_str = ", ".join(str(b) for b in nc.weyl_b[:min(2, len(nc.weyl_b))])
            a_ellipsis = "…" if len(nc.weyl_a) > 2 else ""
            b_ellipsis = "…" if len(nc.weyl_b) > 2 else ""
            weyl_str = f"a=[{a_str}{a_ellipsis}] b=[{b_str}{b_ellipsis}]"
        else:
            weyl_str = "—"

        # Format q¹ projection result (show actual value alongside pass/fail)
        val = getattr(nc, "adjoint_proj_value", None)
        val_str = f" = {val}" if val is not None else ""
        if nc.adjoint_proj_pass is True:
            q1_str = f"✓ Pass{val_str}"
        elif nc.adjoint_proj_pass is False:
            q1_str = f"✗ Fail{val_str}"
        else:
            q1_str = "—"

        # Overall compatibility
        w = nc.weyl_compatible
        a = nc.adjoint_proj_pass
        if w is False or a is False:
            compat_class = 'style="color: #d00;"'
        elif w is True and a is True:
            compat_class = 'style="color: #0a0;"'
        else:
            compat_class = ''

        rows += (
            f"<tr {compat_class}>"
            f"<td>{i}</td>"
            f"<td>${nc.slope_latex}$</td>"
            f"<td><small>{weyl_str}</small></td>"
            f"<td>{q1_str}</td>"
            f"<td>{nc.source}</td>"
            f"</tr>\n"
        )

    return header + rows + "</table>\n"


# ---------------------------------------------------------------------------
# Fill-query result HTML
# ---------------------------------------------------------------------------

def format_fill_result_html(fq: FillQueryViewModel) -> str:
    """Return an HTML fragment for one ``FillQueryViewModel``.

    Shows the NC slope, user slope, transformed (p,q) coordinates,
    and the filled series.

    Parameters
    ----------
    fq : FillQueryViewModel
    """
    parts: list[str] = []

    # Slope header
    parts.append(
        f"<p>NC cycle: {fq.nc_slope_latex} &nbsp;|&nbsp; "
        f"User slope: {fq.user_slope_latex}</p>"
    )

    # Transformed coordinates
    transformed = format_slope_latex(fq.p, fq.q,
                                     a=r"\gamma", b=r"\delta")
    parts.append(
        f"<p>Transformed slope: ${transformed}$"
        f" &nbsp; $(p,q) = ({fq.p},\\,{fq.q})$</p>"
    )

    # Weyl vectors (if available)
    if fq.weyl_a_latex is not None:
        parts.append(
            f"<p>Weyl: $a = {fq.weyl_a_latex}$, "
            f"$b = {fq.weyl_b_latex}$</p>"
        )

    # Series result
    result_block = (
        '<table class="idx">\n'
        f'<tr><td class="eq">$=$</td>'
        f'<td class="sr">{fq.result_latex}</td></tr>\n'
        "</table>\n"
    )
    parts.append(result_block)

    # Incompatible edges warning
    if fq.incompat_edges:
        edge_list = ", ".join(str(e) for e in fq.incompat_edges)
        parts.append(
            f'<p class="warn">⚠ Edges {{{edge_list}}} incompatible with '
            "this filling slope — η zeroed.</p>"
        )

    return "\n".join(parts) + "\n"
