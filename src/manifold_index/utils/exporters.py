"""
utils/exporters.py — Export formatters and file writers.

Provides LaTeX, Mathematica, plain-text and JSON output for the refined
3D-index results (unfilled + Dehn filled).

Public API
----------
Series formatters
    to_latex_series, to_latex_filled_series
    to_mathematica_series, to_mathematica_filled_series

File writers
    write_latex, write_report, write_mathematica, write_plain_text, write_json

Clipboard helpers
    clipboard_latex, clipboard_plain_text
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

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
    Display convention:  η_a^{p} where p = key[1+a]//2.
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
    r"""LaTeX factors for cusp η's (appear after hard η's in the key).

    Cusp exponents are stored directly (NOT doubled) in the key.
    Display convention: \eta^{2V_c} for exponent 1, \eta^{4V_c} for exponent 2, etc.
    """
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

    if not body:                          # constant term
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
    r"""Format a refined index (unfilled) as a LaTeX sum.

    Parameters
    ----------
    result : dict
        ``{(qq_pow, 2W_0, …): coeff, …}``  (keys have 1+num_hard entries).
    num_hard : int

    Returns
    -------
    str
        LaTeX math-mode expression, e.g.  ``q^{1/2} \, \eta_0 - 3 \, q``.
    """
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
    r"""Format a filled refined index as LaTeX.

    Parameters
    ----------
    series : dict
        ``{(qq_pow, 2W_0, …, cusp0, …): coeff, …}``.
    num_hard : int
    num_cusp_eta : int
    max_q_terms : int or None
        If set, truncate at this many distinct q-powers and append ``+ \cdots``.
    """
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
# § 4  File writers
# ======================================================================

_LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{hyperref}

\newcommand{\Iref}{I^{\mathrm{ref}}}

\begin{document}
"""

_LATEX_POSTAMBLE = r"""
\end{document}
"""


# ── Write helpers ─────────────────────────────────────────────────────

def _write_latex_dehn_section(
    lines: list[str],
    dehn_results: list,
) -> None:
    """Append a Dehn-filling section to a list of LaTeX lines."""
    from manifold_index.app.workers import TransformedFillResult, MultiCuspFillResult

    lines.append(r"\section{Dehn Filling Results}")
    for res in dehn_results:
        if isinstance(res, TransformedFillResult):
            lines.append(
                rf"\subsection{{Cusp {res.cusp_idx}: "
                rf"slope $({res.P_user}, {res.Q_user})$, "
                rf"NC cycle $({res.P_nc}, {res.Q_nc})$}}"
            )
            lines.append(
                rf"Transformed coordinates: $(p, q) = ({res.p}, {res.q})$."
            )
            lines.append("")
            for m_other, e_other, fr in res.fill_results:
                num_hard = fr.num_hard
                num_cusp_eta = fr.num_cusp_eta
                label = _charge_label(m_other, e_other, latex=True) if m_other else ""
                if label:
                    lines.append(rf"\paragraph{{{label}}}")
                tex_series = to_latex_filled_series(
                    fr.series, num_hard, num_cusp_eta
                )
                hj_str = ", ".join(str(k) for k in fr.hj_ks)
                lines.append(
                    rf"HJ = $[{hj_str}]$, "
                    rf"$\ell = {len(fr.hj_ks)}$"
                )
                lines.append(rf"\[")
                lines.append(rf"  \Iref_{{{res.P_user}/{res.Q_user}}} = {tex_series}")
                lines.append(rf"\]")
                lines.append("")
        elif isinstance(res, MultiCuspFillResult):
            cusps = ", ".join(str(ci.cusp_idx) for ci in res.cusp_info)
            lines.append(rf"\subsection{{Multi-cusp filling: cusps {cusps}}}")
            for ci in res.cusp_info:
                lines.append(
                    rf"Cusp {ci.cusp_idx}: "
                    rf"slope $({ci.P_user}, {ci.Q_user})$, "
                    rf"NC $({ci.P_nc}, {ci.Q_nc})$, "
                    rf"$(p,q) = ({ci.p}, {ci.q})$."
                )
            fr_obj = res.fill_result
            if hasattr(fr_obj, "series"):
                tex_series = to_latex_filled_series(
                    fr_obj.series,
                    getattr(fr_obj, "num_hard", 0),
                    getattr(fr_obj, "num_cusp_eta", 0),
                )
                lines.append(rf"\[")
                lines.append(rf"  \Iref = {tex_series}")
                lines.append(rf"\]")
            lines.append("")


def _write_nz_matrix_section(lines: list[str], nz: Any) -> None:
    """Append Neumann-Zagier matrix display to LaTeX lines."""
    lines.append(r"\section{Neumann--Zagier Data}")
    lines.append(rf"$n = {nz.n}$ tetrahedra, $r = {nz.r}$ cusps, "
                 rf"{nz.num_hard} hard, {nz.num_easy} easy edges.")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"g_{\mathrm{NZ}} = \begin{pmatrix}")
    g = nz.g_NZ
    rows = []
    for i in range(g.shape[0]):
        row = " & ".join(str(int(g[i, j])) for j in range(g.shape[1]))
        rows.append(row)
    lines.append(r" \\".join(rows))
    lines.append(r"\end{pmatrix}")
    lines.append(r"\]")
    lines.append("")


def _write_weyl_detail(lines: list[str], weyl: Any, nz: Any) -> None:
    """Append Weyl-symmetry detail to LaTeX lines."""
    lines.append(r"\section{Weyl Symmetry}")
    if weyl is None:
        lines.append("No Weyl data available.")
        return
    ab = weyl.ab
    a_str = ", ".join(str(x) for x in ab.a)
    b_str = ", ".join(str(x) for x in ab.b)
    lines.append(rf"$\mathbf{{a}} = ({a_str})$, $\mathbf{{b}} = ({b_str})$.")
    lines.append("")
    if hasattr(weyl, "adjoint") and weyl.adjoint is not None:
        adj = weyl.adjoint
        status = r"\checkmark" if adj.is_pass else r"\times"
        lines.append(
            rf"Adjoint projection: ${status}$ "
            rf"(projected value $= {adj.projected_value}$)."
        )
    lines.append("")


# ── Public writers ────────────────────────────────────────────────────

def write_latex(
    path: Path | str,
    manifold_name: str,
    nz: Any,
    entries: list,
    weyl: Any = None,
    dehn_results: list | None = None,
    include_dehn: bool = False,
) -> None:
    """Write a compilable LaTeX document with refined index results."""
    path = Path(path)
    lines: list[str] = [_LATEX_PREAMBLE]
    lines.append(rf"\title{{Refined 3D Index: {manifold_name}}}")
    lines.append(r"\maketitle")
    lines.append("")
    lines.append(r"\section{Refined Index}")
    lines.append(
        rf"$n = {nz.n}$ tetrahedra, $r = {nz.r}$ cusps, "
        rf"{nz.num_hard} hard edge(s)."
    )
    lines.append("")
    for m_ext, e_ext, result in entries:
        label = _charge_label(m_ext, e_ext, latex=True)
        lines.append(rf"\subsection{{{label}}}")
        tex = to_latex_series(result, nz.num_hard)
        lines.append(rf"\[")
        lines.append(rf"  \Iref = {tex}")
        lines.append(rf"\]")
        lines.append("")
    if weyl is not None:
        _write_weyl_detail(lines, weyl, nz)
    if include_dehn and dehn_results:
        _write_latex_dehn_section(lines, dehn_results)
    lines.append(_LATEX_POSTAMBLE)
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
    """Write a full report LaTeX document (NZ matrix, phase space, Weyl, etc.)."""
    path = Path(path)
    lines: list[str] = [_LATEX_PREAMBLE]
    lines.append(rf"\title{{Full Report: {manifold_name}}}")
    lines.append(r"\maketitle")
    lines.append(r"\tableofcontents")
    lines.append(r"\newpage")
    lines.append("")
    # NZ data section
    _write_nz_matrix_section(lines, nz)
    # Phase space section
    lines.append(r"\section{Phase Space}")
    if easy_result is not None:
        lines.append(
            rf"{nz.num_easy} easy edge(s), "
            rf"{nz.num_hard} hard edge(s)."
        )
        if hasattr(easy_result, "num_independent_easy"):
            lines.append(
                rf"Independent easy = {easy_result.num_independent_easy}."
            )
    else:
        lines.append("No easy-edge data available.")
    lines.append("")
    # Refined index section
    lines.append(r"\section{Refined Index}")
    if q_order_half is not None:
        lines.append(rf"Computed to order $q^{{{q_order_half}/2}}$.")
    lines.append("")
    for m_ext, e_ext, result in entries:
        label = _charge_label(m_ext, e_ext, latex=True)
        lines.append(rf"\subsection{{{label}}}")
        tex = to_latex_series(result, nz.num_hard)
        lines.append(rf"\[")
        lines.append(rf"  \Iref = {tex}")
        lines.append(rf"\]")
        lines.append("")
    # Weyl section
    _write_weyl_detail(lines, weyl, nz)
    # Dehn section
    if include_dehn and dehn_results:
        _write_latex_dehn_section(lines, dehn_results)
    lines.append(_LATEX_POSTAMBLE)
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Mathematica ───────────────────────────────────────────────────────

def _write_mathematica_dehn(
    lines: list[str],
    name: str,
    dehn_results: list,
) -> None:
    """Append Dehn filling results as Mathematica rules."""
    from manifold_index.app.workers import TransformedFillResult, MultiCuspFillResult

    lines.append("(* Dehn filling results *)")
    for res in dehn_results:
        if isinstance(res, TransformedFillResult):
            for m_other, e_other, fr in res.fill_results:
                m_str = "{" + ", ".join(str(x) for x in m_other) + "}" if m_other else "{}"
                e_str = "{" + ", ".join(str(x) for x in e_other) + "}" if e_other else "{}"
                expr = to_mathematica_filled_series(
                    fr.series, fr.num_hard, fr.num_cusp_eta
                )
                lines.append(
                    f'IrefFilled["{name}", {{{res.P_user}, {res.Q_user}}}, '
                    f'{m_str}, {e_str}] = {expr};'
                )
        elif isinstance(res, MultiCuspFillResult):
            fr_obj = res.fill_result
            if hasattr(fr_obj, "series"):
                expr = to_mathematica_filled_series(
                    fr_obj.series,
                    getattr(fr_obj, "num_hard", 0),
                    getattr(fr_obj, "num_cusp_eta", 0),
                )
                cusps = ", ".join(str(ci.cusp_idx) for ci in res.cusp_info)
                lines.append(f'IrefMulti["{name}", {{{cusps}}}] = {expr};')
    lines.append("")


def write_mathematica(
    path: Path | str,
    manifold_name: str,
    nz: Any,
    entries: list,
    weyl: Any = None,
    dehn_results: list | None = None,
    include_dehn: bool = False,
) -> None:
    """Write a Mathematica .m file with refined index rules."""
    path = Path(path)
    lines: list[str] = []
    lines.append(f'(* Refined 3D Index for {manifold_name} *)')
    lines.append(f'(* n = {nz.n}, r = {nz.r}, hard = {nz.num_hard} *)')
    lines.append("")
    for m_ext, e_ext, result in entries:
        m_str = "{" + ", ".join(str(x) for x in m_ext) + "}"
        e_str = "{" + ", ".join(str(x) for x in e_ext) + "}"
        expr = to_mathematica_series(result, nz.num_hard)
        lines.append(f'Iref["{manifold_name}", {m_str}, {e_str}] = {expr};')
    lines.append("")
    if include_dehn and dehn_results:
        _write_mathematica_dehn(lines, manifold_name, dehn_results)
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Plain text ────────────────────────────────────────────────────────

def _write_plain_text_dehn(f, dehn_results: list, num_hard: int) -> None:
    """Append Dehn filling results to a plain-text file handle."""
    from manifold_index.app.workers import TransformedFillResult, MultiCuspFillResult

    f.write("\n--- Dehn Filling Results ---\n\n")
    for res in dehn_results:
        if isinstance(res, TransformedFillResult):
            f.write(
                f"Cusp {res.cusp_idx}: slope ({res.P_user}, {res.Q_user}), "
                f"NC cycle ({res.P_nc}, {res.Q_nc})\n"
            )
            f.write(f"Transformed: (p, q) = ({res.p}, {res.q})\n")
            for m_other, e_other, fr in res.fill_results:
                label = _charge_label(m_other, e_other) if m_other else ""
                if label:
                    f.write(f"  {label}\n")
                hj_str = ", ".join(str(k) for k in fr.hj_ks)
                f.write(f"  HJ = [{hj_str}], ell = {len(fr.hj_ks)}\n")
                f.write(f"  I_filled = {_plain_series(fr.series, fr.num_hard, fr.num_cusp_eta)}\n\n")
        elif isinstance(res, MultiCuspFillResult):
            cusps = ", ".join(str(ci.cusp_idx) for ci in res.cusp_info)
            f.write(f"Multi-cusp: cusps {cusps}\n")
            for ci in res.cusp_info:
                f.write(
                    f"  Cusp {ci.cusp_idx}: slope ({ci.P_user}, {ci.Q_user}), "
                    f"NC ({ci.P_nc}, {ci.Q_nc}), (p,q) = ({ci.p}, {ci.q})\n"
                )
            fr_obj = res.fill_result
            if hasattr(fr_obj, "series"):
                f.write(
                    f"  I_filled = {_plain_series(fr_obj.series, getattr(fr_obj, 'num_hard', 0), getattr(fr_obj, 'num_cusp_eta', 0))}\n"
                )
            f.write("\n")


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


def write_plain_text(
    path: Path | str,
    manifold_name: str,
    nz: Any,
    entries: list,
    weyl: Any = None,
    dehn_results: list | None = None,
    include_dehn: bool = False,
) -> None:
    """Write results as plain text."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Refined 3D Index: {manifold_name}\n")
        f.write(f"n = {nz.n}, r = {nz.r}, hard = {nz.num_hard}, easy = {nz.num_easy}\n\n")
        for m_ext, e_ext, result in entries:
            label = _charge_label(m_ext, e_ext)
            f.write(f"{label}\n")
            f.write(f"  I({','.join(str(x) for x in m_ext)}; {','.join(str(x) for x in e_ext)}) = "
                    f"{_plain_series(result, nz.num_hard)}\n\n")
        if include_dehn and dehn_results:
            _write_plain_text_dehn(f, dehn_results, nz.num_hard)


# ── JSON ──────────────────────────────────────────────────────────────

def write_json(
    path: Path | str,
    manifold_name: str,
    nz: Any,
    entries: list,
    weyl: Any = None,
    dehn_results: list | None = None,
    include_dehn: bool = False,
) -> None:
    """Write results as JSON."""
    path = Path(path)
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
    data: dict[str, Any] = {
        "manifold": manifold_name,
        "n": nz.n,
        "r": nz.r,
        "num_hard": nz.num_hard,
        "num_easy": nz.num_easy,
        "sectors": sectors,
    }
    if include_dehn and dehn_results:
        data["dehn_filling"] = _json_dehn(dehn_results)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


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
                    "series": {str(k): int(v) for k, v in fr.series.items()},
                })
            out.append({
                "type": "single",
                "cusp_idx": res.cusp_idx,
                "P_user": res.P_user, "Q_user": res.Q_user,
                "P_nc": res.P_nc, "Q_nc": res.Q_nc,
                "p": res.p, "q": res.q,
                "fill_results": fills,
            })
        elif isinstance(res, MultiCuspFillResult):
            out.append({
                "type": "multi",
                "cusps": [
                    {"cusp_idx": ci.cusp_idx, "P_user": ci.P_user, "Q_user": ci.Q_user}
                    for ci in res.cusp_info
                ],
            })
    return out


# ======================================================================
# § 5  Clipboard helpers
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
    return "\n".join(parts)
