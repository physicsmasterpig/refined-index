"""
app/formatters.py — Export formatting helpers.

All text/LaTeX/JSON/Mathematica formatting extracted from the old gui.py
so the UI code stays clean.
"""

from __future__ import annotations

import datetime
import json
import re
from fractions import Fraction
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_frac(v: Fraction) -> str:
    """Format a Fraction as '3' or '1/2' (never '3/1')."""
    return str(int(v)) if v.denominator == 1 else str(v)


def _fmt_charge(m_ext: list, e_ext: list) -> str:
    """Format external charges, e.g. '+2, 0' or '0, 1/2'."""
    parts: list[str] = []
    for v in m_ext:
        parts.append(str(v) if v == 0 else f"{v:+d}")
    for v in e_ext:
        f = Fraction(v).limit_denominator(1000)
        parts.append("0" if f == 0 else str(f))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Weyl-manifest plain-text formatting
# ---------------------------------------------------------------------------

def fmt_centre_text(centre: list[Fraction], eta_vars: list[str]) -> str:
    """Format the Weyl monomial η^{centre} as a plain-text prefix."""
    parts: list[str] = []
    for j, c in enumerate(centre):
        if c == 0:
            continue
        exp_x2 = int(c * 2)
        if exp_x2 == 1:
            parts.append(f"η^(v_{j})")
        elif exp_x2 == -1:
            parts.append(f"η^(-v_{j})")
        else:
            parts.append(f"η^({exp_x2}·v_{j})")
    return " · ".join(parts) if parts else "1"


def centre_to_latex(centre: list[Fraction], num_hard: int) -> str:
    """Format the Weyl monomial η^{centre} as a LaTeX string."""
    parts: list[str] = []
    for j, c in enumerate(centre):
        if c == 0:
            continue
        exp_x2 = int(c * 2)
        if exp_x2 == 1:
            parts.append(rf"\eta^{{v_{{{j}}}}}")
        elif exp_x2 == -1:
            parts.append(rf"\eta^{{-v_{{{j}}}}}")
        else:
            parts.append(rf"\eta^{{{exp_x2} \, v_{{{j}}}}}")
    return " ".join(parts) if parts else "1"


def format_weyl_manifest_text(
    entries: list,
    num_hard: int,
    ab: object,
    eta_vars: list[str],
    q_var: str = "q",
) -> str:
    """Format refined index entries in Weyl-manifest form for plain-text display."""
    from manifold_index.core.weyl_check import strip_weyl_monomial
    from manifold_index.core.refined_index import format_refined_index

    lines: list[str] = []
    for m_ext, e_ext, result in entries:
        charge = _fmt_charge(m_ext, e_ext)
        if not result:
            lines.append(f"I({charge})  =  0")
            continue
        centre, stripped = strip_weyl_monomial(result, m_ext, e_ext, ab, num_hard)
        monomial = fmt_centre_text(centre, eta_vars)
        series = format_refined_index(stripped, num_hard, q_var=q_var, eta_vars=eta_vars)
        if monomial == "1":
            lines.append(f"I({charge})  =  {series}")
        else:
            lines.append(f"I({charge})  =  {monomial}  ·  ( {series} )")
    return "\n\n".join(lines) if lines else "0"


# ---------------------------------------------------------------------------
# LaTeX series formatter
# ---------------------------------------------------------------------------

def series_to_latex(result: dict, num_hard: int) -> str:
    """Convert a RefinedIndexResult to a LaTeX math string."""

    def _monomial(key: tuple[int, ...], coeff: int) -> str | None:
        if coeff == 0:
            return None
        q_pow = key[0]
        eta_pows_x2 = key[1:]
        factors: list[str] = []

        if q_pow != 0:
            if q_pow % 2 == 0:
                n = q_pow // 2
                factors.append(f"q^{{{n}}}" if n != 1 else "q")
            else:
                factors.append(f"q^{{{q_pow}/2}}")

        for a, exp_x2 in enumerate(eta_pows_x2):
            if exp_x2 == 0:
                continue
            if exp_x2 == 1:
                factors.append(rf"\eta^{{v_{{{a}}}}}")
            elif exp_x2 == -1:
                factors.append(rf"\eta^{{-v_{{{a}}}}}")
            else:
                factors.append(rf"\eta^{{{exp_x2} \, v_{{{a}}}}}")

        body = " ".join(factors) if factors else ""
        if not body:
            return ("-" if coeff < 0 else "") + str(abs(coeff))
        abs_coeff = abs(coeff)
        coeff_str = "" if abs_coeff == 1 else f"{abs_coeff} "
        return ("-" if coeff < 0 else "") + coeff_str + body

    sorted_items = sorted(result.items(), key=lambda kv: kv[0])
    parts: list[str] = []
    for k, v in sorted_items:
        m = _monomial(k, v)
        if m is None:
            continue
        if parts and not m.startswith("-"):
            parts.append(f"+ {m}")
        elif parts and m.startswith("-"):
            parts.append(f"- {m[1:]}")
        else:
            parts.append(m)
    return " ".join(parts) if parts else "0"


# ---------------------------------------------------------------------------
# Mathematica expression formatter
# ---------------------------------------------------------------------------

def series_to_mathematica(result: dict, num_hard: int) -> str:
    """Convert a RefinedIndexResult to a Mathematica expression string."""
    if not result:
        return "0"
    terms: list[str] = []
    for key, coeff in sorted(result.items()):
        q_half_pow = key[0]
        eta_pows_x2 = key[1:]
        factors: list[str] = []

        if q_half_pow != 0:
            if q_half_pow % 2 == 0:
                n = q_half_pow // 2
                factors.append(f"q^{n}" if n != 1 else "q")
            else:
                factors.append(f"q^({q_half_pow}/2)")

        for a, exp_x2 in enumerate(eta_pows_x2):
            if exp_x2 == 0:
                continue
            if exp_x2 == 1:
                factors.append(f"\\[Eta]^(v[{a}])")
            elif exp_x2 == -1:
                factors.append(f"\\[Eta]^(-v[{a}])")
            else:
                factors.append(f"\\[Eta]^({exp_x2} v[{a}])")

        body = " ".join(factors) if factors else "1"
        if coeff == 1:
            terms.append(body)
        elif coeff == -1:
            terms.append(f"-{body}")
        else:
            terms.append(f"{coeff} {body}" if body != "1" else str(coeff))

    expr = " + ".join(terms)
    expr = expr.replace("+ -", "- ")
    return expr


# ---------------------------------------------------------------------------
# Mathematica notebook (.nb) builder
# ---------------------------------------------------------------------------

def _nb_escape(s: str) -> str:
    """Escape a Python string for embedding inside BoxData in a .nb file."""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\t", "    ")
    s = s.replace("\n", "\\n")
    return s


def build_nb_content(
    entries: list,
    name: str,
    q_ord: int,
    num_hard: int,
    basis_summary: str,
    timestamp: str = "",
) -> str:
    """Build a proper Mathematica Notebook (.nb) file."""
    safe = re.sub(r"[^a-zA-Z0-9]", "", name)
    table_name = f"i{safe}RefTable"
    func_name = f"i{safe}ref"

    nonzero = [(m, e, r) for m, e, r in entries if r]
    n_pairs = len(entries)
    n_nonzero = len(nonzero)

    if entries:
        all_m = [mv for m_ext, _e, _r in entries for mv in m_ext]
        all_e_frac = [
            Fraction(ev).limit_denominator(1000)
            for _m, e_ext, _r in entries for ev in e_ext
        ]
        m_min, m_max_v = min(all_m), max(all_m)
        e_nums = [float(f) for f in all_e_frac]
        e_min, e_max_v = min(e_nums), max(e_nums)

        def _fmt_e(v: float) -> str:
            f = Fraction(v).limit_denominator(1000)
            return str(f.numerator) if f.denominator == 1 else str(f)

        subtitle = (
            f"m \u2208 [{m_min}, {m_max_v}]  |  "
            f"e \u2208 [{_fmt_e(e_min)}, {_fmt_e(e_max_v)}]  |  "
            f"{n_nonzero} nonzero / {n_pairs} pairs  |  Nmax = {q_ord}"
            + (f"  |  Generated: {timestamp}" if timestamp else "")
        )
    else:
        subtitle = (
            f"Nmax = {q_ord}  (no entries)"
            + (f"  |  Generated: {timestamp}" if timestamp else "")
        )

    def text_cell(content: str, style: str) -> str:
        return f'Cell["{content.replace(chr(34), chr(92)+chr(34))}", "{style}"]'

    def input_cell(code: str) -> str:
        return f'Cell[BoxData["{_nb_escape(code)}"], "Input"]'

    clear_code = f"ClearAll[q, \\[Eta], v, {func_name}, {table_name}];"

    items: list[str] = []
    for m_ext, e_ext, result in entries:
        args = (
            [str(m) for m in m_ext]
            + [str(Fraction(ev).limit_denominator(1000)) for ev in e_ext]
        )
        key = "{" + ", ".join(args) + "}"
        series = series_to_mathematica(result, num_hard)
        items.append(f"  {key} -> ({series})")

    table_code = "\n".join([
        f"(* Precomputed series table: {n_nonzero} nonzero of {n_pairs} entries, Nmax = {q_ord} *)",
        f'(* Zero entries are stored explicitly as 0 (distinct from Missing["NotComputed"]) *)',
        f"{table_name} = <|",
        ",\n".join(items),
        "|>;",
    ])

    q_series_order = (q_ord + 1) // 2

    # Determine number of cusps from entries
    r = len(entries[0][0]) if entries else 1

    if r == 1:
        # Single-cusp: funcName[m_Integer, e_]
        func_params = "m_Integer, e_"
        key_args = "{m, e}"
        usage_code = (
            f"(* Usage: evaluate the refined index at a given (m, e) *)\n"
            f"(* e can be integer or half-integer (Rational), e.g. 1/2 *)\n"
            f"{func_name}[0, 0]        (* integer e *)\n"
            f"{func_name}[0, 1/2]      (* half-integer e *)\n"
            f'(* 0 = computed and exactly zero;  Missing["NotComputed"] = outside grid *)\n'
            f"Keys[{table_name}]   (* list all precomputed (m,e) pairs *)"
        )
        func_label = f"Lookup function   {func_name}[m, e]"
    else:
        # Multi-cusp: funcName[m0_Integer, m1_Integer, ..., e0_, e1_, ...]
        m_params = ", ".join(f"m{i}_Integer" for i in range(r))
        e_params = ", ".join(f"e{i}_" for i in range(r))
        func_params = f"{m_params}, {e_params}"
        m_key = ", ".join(f"m{i}" for i in range(r))
        e_key = ", ".join(f"e{i}" for i in range(r))
        key_args = "{" + f"{m_key}, {e_key}" + "}"
        m_zeros = ", ".join("0" for _ in range(r))
        e_zeros = ", ".join("0" for _ in range(r))
        e_halves = ", ".join("1/2" for _ in range(r))
        m_labels = ", ".join(f"m{i}" for i in range(r))
        e_labels = ", ".join(f"e{i}" for i in range(r))
        usage_code = (
            f"(* Usage: evaluate the refined index at given (m0,...,m{r-1}, e0,...,e{r-1}) *)\n"
            f"(* Each e can be integer or half-integer (Rational) *)\n"
            f"{func_name}[{m_zeros}, {e_zeros}]\n"
            f"{func_name}[{m_zeros}, {e_halves}]\n"
            f'(* 0 = computed and exactly zero;  Missing["NotComputed"] = outside grid *)\n'
            f"Keys[{table_name}]   (* list all precomputed ({m_labels},{e_labels}) tuples *)"
        )
        func_label = f"Lookup function   {func_name}[{m_labels}, {e_labels}]"

    lookup_code = (
        f"qOrder = {q_series_order};\n"
        f"{func_name}[{func_params}] :=\n"
        f'  With[{{raw = Lookup[{table_name}, Key[{key_args}], Missing["NotComputed"]]}},\n'
        f"    If[MissingQ[raw], raw,\n"
        f"      If[raw === 0, 0,\n"
        f"        Series[raw, {{q, 0, qOrder}}] // ExpandAll]]];"
    )

    cells = [
        text_cell(f"Refined Index Table: {name}", "Title"),
        text_cell(subtitle, "Subtitle"),
        text_cell("Setup", "Subsection"),
        input_cell(clear_code),
        text_cell("Precomputed refined index table", "Subsection"),
        input_cell(table_code),
        text_cell(func_label, "Subsection"),
        input_cell(lookup_code),
        text_cell("Usage", "Subsection"),
        input_cell(usage_code),
    ]

    header = "\n".join([
        "(* Content-type: application/vnd.wolfram.mathematica *)",
        "",
        "(*** Wolfram Notebook File ***)",
        "(* http://www.wolfram.com/nb *)",
        "",
        "(* CreatedBy='Mathematica 14.0' *)",
        "",
    ])

    return header + "Notebook[{\n" + ",\n".join(cells) + "\n}]"


# ---------------------------------------------------------------------------
# Full export builders
# ---------------------------------------------------------------------------

def build_plain_text(
    entries: list,
    name: str,
    q_ord: int,
    num_hard: int,
    basis_summary: str,
    weyl_result=None,
) -> str:
    """Build the full plain-text export."""
    from manifold_index.core.refined_index import format_multi_point_index

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ab_valid = weyl_result is not None and weyl_result.ab_valid
    eta_vars = [f"η^(2·v_{a})" for a in range(num_hard)]

    if ab_valid:
        body = format_weyl_manifest_text(
            entries, num_hard, weyl_result.ab, eta_vars, q_var="q"
        )
        weyl_note = (
            "# Output: Weyl-manifest form  f(m,e) = η^{b·m+a·e} · I(m,e),  f(η)=f(η⁻¹)"
        )
    else:
        body = format_multi_point_index(
            entries, num_hard, q_var="q", eta_vars=eta_vars, show_zero=False,
        )
        weyl_note = ""

    lines = [
        f"# Refined Index — {name}",
        f"# Generated: {ts}",
        f"# q_order_half = {q_ord}",
        f"# num_hard = {num_hard}",
        f"# Basis: {basis_summary}",
    ]
    if weyl_note:
        lines.append(weyl_note)
    lines += ["", body]
    return "\n".join(lines)


def build_latex(
    entries: list,
    name: str,
    q_ord: int,
    num_hard: int,
    basis_summary: str,
    weyl_result=None,
) -> str:
    """Build the LaTeX export."""
    from manifold_index.core.weyl_check import strip_weyl_monomial

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ab_valid = weyl_result is not None and weyl_result.ab_valid

    basis_lines = basis_summary.splitlines()
    commented_basis = "\n".join(f"% {l}" for l in basis_lines)

    lines = [
        r"\documentclass{article}",
        r"\usepackage{amsmath, amssymb}",
        r"\begin{document}",
        "",
        rf"% Refined Index of {name}, $q$-order $= {q_ord}/2$",
        f"% Generated: {ts}",
        commented_basis,
    ]
    if ab_valid:
        lines.append(
            r"% Weyl-manifest form: $f(m,e) = \eta^{b \cdot m + a \cdot e}"
            r" \cdot I(m,e)$,  $f(\eta) = f(\eta^{-1})$"
        )
    lines += ["", r"\begin{align*}"]

    for m_ext, e_ext, result in entries:
        if not result:
            continue
        if ab_valid:
            centre, stripped = strip_weyl_monomial(
                result, m_ext, e_ext, weyl_result.ab, num_hard
            )
            prefix = centre_to_latex(centre, num_hard)
            body_latex = series_to_latex(stripped, num_hard)
            charge = _fmt_charge(m_ext, e_ext)
            if prefix == "1":
                lines.append(rf"I({charge}) &= {body_latex} \\")
            else:
                lines.append(rf"I({charge}) &= {prefix} \left( {body_latex} \right) \\")
        else:
            series = series_to_latex(result, num_hard)
            charge = _fmt_charge(m_ext, e_ext)
            lines.append(rf"I({charge}) &= {series} \\")

    lines += [r"\end{align*}", "", r"\end{document}"]
    return "\n".join(lines)


def build_json(
    entries: list,
    name: str,
    q_ord: int,
    num_hard: int,
    basis_summary: str,
    weyl_result=None,
) -> str:
    """Build the JSON export."""
    from manifold_index.core.weyl_check import strip_weyl_monomial

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ab_valid = weyl_result is not None and weyl_result.ab_valid

    eval_list = []
    for m_ext, e_ext, result in entries:
        entry: dict = {
            "m_ext": m_ext,
            "e_ext": [float(Fraction(e).limit_denominator(1000)) for e in e_ext],
            "refined_index": {
                str(list(k)): v for k, v in result.items()
            },
        }
        if ab_valid and result:
            centre, stripped = strip_weyl_monomial(
                result, m_ext, e_ext, weyl_result.ab, num_hard
            )
            entry["weyl_monomial_exponents"] = [str(c) for c in centre]
            entry["weyl_manifest_series"] = {
                str(list(k)): v for k, v in stripped.items()
            }
        eval_list.append(entry)

    data = {
        "manifold": name,
        "generated": ts,
        "q_order_half": q_ord,
        "num_hard": num_hard,
        "basis": basis_summary,
        "evaluations": eval_list,
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Full Report LaTeX builder
# ---------------------------------------------------------------------------

def _latex_frac(v) -> str:
    """Format a Fraction (or float) as LaTeX.  E.g. Fraction(1,2) → \\frac{1}{2}."""
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(f.numerator)
    sign = "-" if f < 0 else ""
    return rf"{sign}\frac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def _latex_matrix(arr, label: str = "", env: str = "pmatrix", max_cols: int = 14) -> str:
    """Format a 2-D array as a LaTeX matrix.  Truncate if too wide."""
    import numpy as np
    rows, cols = arr.shape
    truncated = cols > max_cols
    show_cols = min(cols, max_cols)
    total_cols = show_cols + (1 if truncated else 0)
    lines: list[str] = []
    if label:
        lines.append(rf"{label} = ")

    # Use array inside \left(\right) for adjustbox compatibility
    col_spec = "c" * total_cols
    lines.append(rf"\left(\begin{{array}}{{{col_spec}}}")
    for i in range(rows):
        entries = []
        for j in range(show_cols):
            v = arr[i, j]
            if isinstance(v, (Fraction,)):
                entries.append(_latex_frac(v))
            elif isinstance(v, float) and v != int(v):
                entries.append(_latex_frac(Fraction(v).limit_denominator(1000)))
            else:
                entries.append(str(int(v)))
        if truncated:
            entries.append(r"\cdots")
        lines.append(" & ".join(entries) + r" \\")
    lines.append(r"\end{array}\right)")
    return "\n".join(lines)


def _edge_3n_to_latex(edge_3n, n: int) -> str:
    """Format a 3n edge vector as a LaTeX sum of Z_i, Z_i', Z_i'' terms."""
    parts: list[str] = []
    for i in range(n):
        triplet = edge_3n[3*i : 3*i + 3]
        names = [rf"Z_{{{i+1}}}", rf"Z'_{{{i+1}}}", rf"Z''_{{{i+1}}}"]
        for slot in range(3):
            c = int(triplet[slot])
            if c == 0:
                continue
            if c == 1 and not parts:
                parts.append(names[slot])
            elif c == -1 and not parts:
                parts.append(f"-{names[slot]}")
            elif c == 1:
                parts.append(f"+ {names[slot]}")
            elif c == -1:
                parts.append(f"- {names[slot]}")
            elif c > 0 and not parts:
                parts.append(f"{c}\\,{names[slot]}")
            elif c > 0:
                parts.append(f"+ {c}\\,{names[slot]}")
            else:
                parts.append(f"- {abs(c)}\\,{names[slot]}")
    return " ".join(parts) if parts else "0"


def _filled_series_to_latex(series: dict) -> str:
    """Format a QSeries dict (key k → coefficient of q^{k/2}) as LaTeX."""
    if not series:
        return "0"
    parts: list[str] = []
    for k in sorted(series):
        c = series[k]
        if c == 0:
            continue
        # Format the coefficient
        if k == 0:
            parts.append(str(c))
        elif k == 2:
            # q^1
            if c == 1:
                parts.append("q" if not parts else "+ q")
            elif c == -1:
                parts.append("-q" if not parts else "- q")
            elif c > 0:
                parts.append(f"{c}\\,q" if not parts else f"+ {c}\\,q")
            else:
                parts.append(f"- {abs(c)}\\,q")
        elif k % 2 == 0:
            pw = k // 2
            label = rf"q^{{{pw}}}"
            if c == 1:
                parts.append(label if not parts else f"+ {label}")
            elif c == -1:
                parts.append(f"-{label}" if not parts else f"- {label}")
            elif c > 0:
                parts.append(f"{c}\\,{label}" if not parts else f"+ {c}\\,{label}")
            else:
                parts.append(f"- {abs(c)}\\,{label}")
        else:
            label = rf"q^{{{k}/2}}"
            if c == 1:
                parts.append(label if not parts else f"+ {label}")
            elif c == -1:
                parts.append(f"-{label}" if not parts else f"- {label}")
            elif c > 0:
                parts.append(f"{c}\\,{label}" if not parts else f"+ {c}\\,{label}")
            else:
                parts.append(f"- {abs(c)}\\,{label}")
    return " ".join(parts) if parts else "0"


def build_full_report_latex(
    entries: list,
    name: str,
    q_ord: int,
    num_hard: int,
    basis_summary: str,
    weyl_result=None,
    pipeline_result=None,
    basis_selection=None,
    nz_changed=None,
    filled_refined_result=None,
) -> str:
    """Build a comprehensive LaTeX report with all pipeline data.

    Sections:
      1. Manifold Summary
      2. SnaPy Gluing Equations
      3. Edge Classification (easy / hard)
      4. Neumann-Zagier Matrix & Affine Shifts
      5. Local Charges Formula
      6. Non-Closable Cycles & Basis Selection
      7. Dehn Filled Unrefined Indices
      8. Weyl Symmetry & Dehn Filling Compatibility
      9. Refined Index Series
      10. Refined Dehn Filling Result
    """
    import numpy as np
    from manifold_index.core.weyl_check import strip_weyl_monomial

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ab_valid = weyl_result is not None and weyl_result.ab_valid

    # Collect data objects
    nz = None
    manifold_data = None
    easy_result = None
    cycle_results = []
    if pipeline_result is not None:
        nz = nz_changed if nz_changed is not None else pipeline_result.nz_data
        manifold_data = getattr(pipeline_result, "manifold_data", None)
        easy_result = getattr(pipeline_result, "easy_result", None)
        cycle_results = pipeline_result.cycle_results
    elif nz_changed is not None:
        nz = nz_changed

    n = nz.n if nz else 0
    r = nz.r if nz else 0

    lines: list[str] = []

    # Format the manifold name for LaTeX:
    # If the name contains _ or ^ (math notation like 5_1^2), use math mode.
    # Otherwise use \texttt{} for monospaced text.
    if "_" in name or "^" in name:
        tex_name = f"${name}$"
    else:
        tex_name = rf"\texttt{{{name}}}"

    # ── Preamble ──────────────────────────────────────────────────
    lines.append(r"\documentclass[11pt,a4paper]{article}")
    lines.append(r"\usepackage[margin=2cm]{geometry}")
    lines.append(r"\usepackage{amsmath, amssymb, amsfonts}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{adjustbox}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\allowdisplaybreaks")
    lines.append(r"\newcommand{\half}{\tfrac{1}{2}}")
    lines.append("")
    lines.append(rf"\title{{Full Report: Refined 3D Index of {tex_name}}}")
    lines.append(rf"\date{{{ts}}}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append(r"\tableofcontents")
    lines.append(r"\newpage")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # §1  Manifold Summary
    # ══════════════════════════════════════════════════════════════
    lines.append(r"\section{Manifold Summary}")
    lines.append(r"\begin{itemize}")
    lines.append(rf"  \item Manifold: {tex_name}")
    lines.append(rf"  \item Number of tetrahedra: $n = {n}$")
    lines.append(rf"  \item Number of cusps: $r = {r}$")
    if nz:
        lines.append(rf"  \item Independent easy edges: ${nz.num_easy}$")
        lines.append(rf"  \item Hard edges (fugacity variables): ${nz.num_hard}$")
    lines.append(rf"  \item $q$-order (half): ${q_ord}$ \quad ($q$-order: ${q_ord}/2$)")
    lines.append(r"\end{itemize}")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # §2  SnaPy Gluing Equations
    # ══════════════════════════════════════════════════════════════
    if manifold_data is not None:
        lines.append(r"\section{SnaPy Gluing Equations}")
        lines.append(r"The gluing equation matrix has rows:")
        lines.append(r"\begin{itemize}")
        lines.append(rf"  \item Rows $0\ldots{n-1}$: edge equations")
        lines.append(rf"  \item Rows ${n},{n+2},\ldots$: meridian equations (per cusp)")
        lines.append(rf"  \item Rows ${n+1},{n+3},\ldots$: longitude equations (per cusp)")
        lines.append(r"\end{itemize}")
        lines.append(r"Columns are ordered as $Z_1, Z'_1, Z''_1, Z_2, Z'_2, Z''_2, \ldots, "
                     rf"Z_{n}, Z'_{n}, Z''_{n}$.")
        lines.append("")

        G = manifold_data.gluing_matrix
        if 3 * n <= 18:
            # Small enough to display the full matrix
            mat_str = _latex_matrix(G, label=r"G_{\text{glue}}")
            lines.append(r"\begin{adjustbox}{max width=\linewidth,center}")
            lines.append(rf"$\displaystyle {mat_str}$")
            lines.append(r"\end{adjustbox}")
        else:
            lines.append(r"(Matrix too large to display in full; $3n = "
                         rf"{3*n}$ columns.)")

        # Show individual cusp equations
        lines.append(r"\subsection*{Cusp Equations}")
        for k in range(r):
            merid = manifold_data.gluing_matrix[n + 2*k]
            longi = manifold_data.gluing_matrix[n + 2*k + 1]
            lines.append(rf"\paragraph{{Cusp {k}:}}")
            lines.append(r"\begin{align*}")
            lines.append(rf"  \mu_{k} &= {_edge_3n_to_latex(merid, n)} \\")
            lines.append(rf"  \lambda_{k} &= {_edge_3n_to_latex(longi, n)}")
            lines.append(r"\end{align*}")
        lines.append("")

    # ══════════════════════════════════════════════════════════════
    # §3  Edge Classification
    # ══════════════════════════════════════════════════════════════
    if easy_result is not None:
        lines.append(r"\section{Edge Classification}")
        lines.append(rf"Total SnaPy edge equations: ${n}$.")
        lines.append(rf"Independent edge basis rank: $n - r = {n - r}$.")
        lines.append(rf"Of these, ${easy_result.num_independent_easy}$ are "
                     rf"\textbf{{easy}} and ${len(easy_result.hard_padding)}$ are "
                     r"\textbf{hard}.")
        lines.append("")

        # Hard edges
        if easy_result.hard_padding:
            lines.append(r"\subsection{Hard Edges}")
            lines.append(r"Hard edges contribute fugacity variables "
                         r"$\eta_0, \eta_1, \ldots$ to the refined index.")
            lines.append(r"\begin{enumerate}")
            for idx, edge in enumerate(easy_result.hard_padding):
                expr = _edge_3n_to_latex(edge, n)
                lines.append(rf"  \item $C_{{\text{{hard}},{idx}}} = {expr}$")
            lines.append(r"\end{enumerate}")
            lines.append("")

        # Easy edges
        indep_easy = [easy_result.all_easy[i]
                      for i in easy_result.independent_easy_indices]
        if indep_easy:
            lines.append(r"\subsection{Easy Edges (independent basis)}")
            lines.append(r"Easy edges are summed over without fugacity "
                         r"(internal charges only).")
            lines.append(r"\begin{enumerate}")
            for idx, edge in enumerate(indep_easy):
                expr = _edge_3n_to_latex(edge, n)
                lines.append(rf"  \item $C_{{\text{{easy}},{idx}}} = {expr}$")
            lines.append(r"\end{enumerate}")
            lines.append("")

        lines.append(rf"Basis ordering in $g_{{NZ}}$ (position block rows): "
                     r"meridians ($r$ rows), hard edges, easy edges.")
        lines.append("")

    # ══════════════════════════════════════════════════════════════
    # §4  Neumann-Zagier Matrix & Affine Shifts
    # ══════════════════════════════════════════════════════════════
    if nz is not None:
        lines.append(r"\section{Neumann--Zagier Data}")

        lines.append(r"\subsection{Symplectic Matrix $g_{NZ}$}")
        lines.append(r"Row ordering: meridians $(\mu_0,\ldots,\mu_{r-1})$, "
                     r"hard edges, easy edges $\mid$ "
                     r"longitudes$/2$ $(\lambda_0/2,\ldots)$, "
                     r"$\Gamma$ vectors.")
        lines.append(r"Column ordering: "
                     r"$Z_1,\ldots,Z_n, Z''_1,\ldots,Z''_n$ (block form).")
        lines.append("")
        if 2*n <= 14:
            # Convert to Fraction for exact display
            g_frac = np.array(
                [[Fraction(v).limit_denominator(1000) for v in row]
                 for row in nz.g_NZ], dtype=object
            )
            mat_str = _latex_matrix(g_frac, label=r"g_{NZ}")
            lines.append(r"\begin{adjustbox}{max width=\linewidth,center}")
            lines.append(rf"$\displaystyle {mat_str}$")
            lines.append(r"\end{adjustbox}")
        else:
            lines.append(rf"(Matrix size ${2*n}\times{2*n}$ — too large to display.)")
        lines.append("")

        lines.append(r"\subsection{Inverse $g_{NZ}^{-1}$}")
        g_inv = nz.g_NZ_inv()
        if 2*n <= 14:
            mat_str = _latex_matrix(g_inv, label=r"g_{NZ}^{-1}")
            lines.append(r"\begin{adjustbox}{max width=\linewidth,center}")
            lines.append(rf"$\displaystyle {mat_str}$")
            lines.append(r"\end{adjustbox}")
        else:
            lines.append(rf"(Matrix size ${2*n}\times{2*n}$ — too large to display.)")
        lines.append("")

        lines.append(r"\subsection{Affine Shifts}")
        nu_x_str = ", ".join(str(int(v)) for v in nz.nu_x)
        lines.append(rf"$$\nu_x = ({nu_x_str})$$")
        nu_p_parts = []
        for v in nz.nu_p:
            nu_p_parts.append(_latex_frac(Fraction(v).limit_denominator(1000)))
        nu_p_str = ", ".join(nu_p_parts)
        lines.append(rf"$$\nu_p = ({nu_p_str})$$")
        lines.append("")

    # ══════════════════════════════════════════════════════════════
    # §5  Local Charges Formula
    # ══════════════════════════════════════════════════════════════
    lines.append(r"\section{Local Charges Formula}")
    lines.append(r"The refined index is computed by the formula:")
    lines.append(r"\begin{equation}")
    lines.append(r"  I^{\text{ref}}(q;\, \eta_0, \ldots, \eta_{k-1})")
    lines.append(r"  = \sum_{e_{\text{int}} \in (\frac{1}{2}\mathbb{Z})^{n-r}}")
    lines.append(r"    \biggl[\prod_{a=0}^{k-1} \eta_a^{e_{r+a}}\biggr]")
    lines.append(r"    \cdot (-q^{1/2})^{\varphi}")
    lines.append(r"    \cdot \prod_{j=1}^{n}")
    lines.append(r"      I_\Delta\bigl((g_{NZ}^{-1}\kappa)_j,\;")
    lines.append(r"                     (g_{NZ}^{-1}\kappa)_{n+j}\bigr)")
    lines.append(r"\end{equation}")
    lines.append(r"where:")
    lines.append(r"\begin{itemize}")
    lines.append(r"  \item $\kappa = (m_{\text{ext}},\, \underbrace{0,\ldots,0}_{n-r},"
                 r"\; e_{\text{ext}},\, e_{\text{int}})$ is the $2n$-vector of "
                 r"external + internal charges.")
    lines.append(r"  \item The \textbf{local charges} (tetrahedron arguments) are "
                 r"$(m_j, e_j) = (g_{NZ}^{-1}\kappa)_j$ and "
                 r"$(g_{NZ}^{-1}\kappa)_{n+j}$.")
    lines.append(r"  \item The \textbf{phase exponent} is $\varphi "
                 r"= m_{\text{full}} \cdot \nu_p - e_{\text{full}} \cdot \nu_x$.")
    lines.append(r"  \item $I_\Delta(m, e)$ is the tetrahedron index.")
    lines.append(r"  \item $k = \text{num\_hard}$ is the number of hard "
                 r"edges carrying fugacity variables.")
    lines.append(r"\end{itemize}")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # §6  Non-Closable Cycles & Basis Selection
    # ══════════════════════════════════════════════════════════════
    lines.append(r"\section{Non-Closable Cycles \& Basis Selection}")

    if cycle_results:
        for cr in cycle_results:
            k = cr.cusp_idx
            if cr.cycles:
                slope_list = ", ".join(
                    f"({c.P}, {c.Q})" for c in cr.cycles
                )
                lines.append(
                    rf"Cusp~{k}: non-closable cycles found: ${slope_list}$."
                )
            else:
                lines.append(rf"Cusp~{k}: no non-closable cycles found in search range.")
        lines.append("")
    else:
        lines.append(r"(No cycle search results available.)")
        lines.append("")

    if basis_selection is not None:
        lines.append(r"\subsection*{Selected Basis}")
        lines.append(r"\begin{itemize}")
        for ch in basis_selection.choices:
            default_note = r" \; [\text{default}]" if ch.is_default else ""
            lines.append(
                rf"  \item Cusp~{ch.cusp_idx}: slope $({ch.P},\, {ch.Q})$ "
                rf"$\;\Rightarrow\; m = {ch.m},\; e = {_latex_frac(ch.e)}"
                rf"{default_note}$"
            )
        lines.append(r"\end{itemize}")
        lines.append("")

    # ══════════════════════════════════════════════════════════════
    # §7  Dehn Filled Unrefined Indices
    # ══════════════════════════════════════════════════════════════
    _has_filled = any(
        getattr(cr, "filled_indices", None)
        for cr in cycle_results
    )
    if _has_filled:
        lines.append(r"\section{Dehn Filled Unrefined Indices}")
        lines.append(r"The unrefined Dehn-filled index $I_{P/Q}(q)$ is "
                     r"computed for each slope tested in the non-closable "
                     r"cycle search.  A slope is \emph{non-closable} when "
                     r"$I_{P/Q}(q)$ is stably zero.")
        lines.append("")

        for cr in cycle_results:
            filled_dict = getattr(cr, "filled_indices", {})
            if not filled_dict:
                continue
            k = cr.cusp_idx
            lines.append(rf"\subsection*{{Cusp {k}}}")

            # Separate non-zero and zero slopes
            nonzero_slopes = []
            zero_slopes = []
            for (P, Q) in sorted(filled_dict.keys()):
                fi = filled_dict[(P, Q)]
                if fi.is_stably_zero():
                    zero_slopes.append((P, Q))
                else:
                    nonzero_slopes.append((P, Q, fi))

            if zero_slopes:
                slope_list = ", ".join(f"$({P},{Q})$" for P, Q in zero_slopes)
                lines.append(
                    rf"Stably zero (non-closable): {slope_list}."
                )
                lines.append("")

            if nonzero_slopes:
                lines.append(r"\begin{longtable}{ccl}")
                lines.append(r"\toprule")
                lines.append(r"$P$ & $Q$ & $I_{P/Q}(q)$ \\")
                lines.append(r"\midrule")
                lines.append(r"\endhead")
                for P, Q, fi in nonzero_slopes:
                    poly = _filled_series_to_latex(fi.series)
                    lines.append(rf"${P}$ & ${Q}$ & ${poly}$ \\")
                lines.append(r"\bottomrule")
                lines.append(r"\end{longtable}")
                lines.append("")
            elif not zero_slopes:
                lines.append(r"(No slopes tested at this cusp.)")
                lines.append("")

    # ══════════════════════════════════════════════════════════════
    # §8  Weyl Symmetry & Dehn Filling Compatibility
    # ══════════════════════════════════════════════════════════════
    lines.append(r"\section{Weyl Symmetry \& Dehn Filling Compatibility}")

    if weyl_result is not None and weyl_result.ab is not None:
        ab = weyl_result.ab
        a_new = [Fraction(1, 2) * av for av in ab.a]
        b_new = list(ab.b)

        lines.append(r"Convention: $f(\eta) = \eta^{b\cdot m + a\cdot e} "
                     r"\cdot I(m,e)$, \quad $f(\eta) = f(\eta^{-1})$.")
        lines.append("")

        a_strs = ", ".join(_latex_frac(v) for v in a_new)
        b_strs = ", ".join(_latex_frac(v) for v in b_new)
        lines.append(rf"$$a = ({a_strs}), \qquad b = ({b_strs})$$")
        lines.append("")

        # Integrality checks
        a_int = all(v.denominator == 1 for v in a_new)
        b2_int = all((2*v).denominator == 1 for v in b_new)
        lines.append(r"\begin{itemize}")
        sym = r"$\checkmark$" if a_int else r"$\times$"
        lines.append(rf"  \item $a \in \mathbb{{Z}}$: {sym}")
        sym = r"$\checkmark$" if b2_int else r"$\times$"
        lines.append(rf"  \item $2b \in \mathbb{{Z}}$: {sym}")
        lines.append(r"\end{itemize}")

        if a_int and b2_int:
            lines.append(r"Dehn filling prerequisites satisfied.  "
                         r"Compatibility depends on the choice of slope $(P, Q)$: "
                         r"$\mu = b P + a (Q/2)$ must be an integer.")
        elif not a_int:
            lines.append(r"\textcolor{red}{Dehn filling not directly compatible: "
                         r"$a \notin \mathbb{Z}$.}")
        else:
            lines.append(r"\textcolor{red}{Dehn filling not directly compatible: "
                         r"$2b \notin \mathbb{Z}$.}")

        lines.append("")

        # Weyl symmetry verification
        n_sym = sum(weyl_result.weyl_symmetric.values())
        n_tot = len(weyl_result.weyl_symmetric)
        sym = r"$\checkmark$" if weyl_result.all_weyl_symmetric else r"$\times$"
        lines.append(rf"Weyl symmetry check: {sym} ({n_sym}/{n_tot} entries pass).")
    else:
        lines.append(r"Weyl symmetry: $(a, b)$ could not be determined "
                     r"(insufficient data or no hard edges).")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # §9  Refined Index Series
    # ══════════════════════════════════════════════════════════════
    lines.append(r"\section{Refined Index}")
    q_max = q_ord // 2
    q_frac = f"{q_ord}/2" if q_ord % 2 else str(q_max)
    lines.append(rf"Evaluated at $q$-order (half) $= {q_ord}$, "
                 rf"giving series up to $q^{{{q_frac}}}$.")
    lines.append("")

    if ab_valid:
        lines.append(r"Output in \textbf{Weyl-manifest form}: "
                     r"$f(m,e) = \eta^{b\cdot m + a\cdot e}\cdot I(m,e)$, "
                     r"where the displayed $I(m,e)$ satisfies "
                     r"$I(\eta) = I(\eta^{-1})$.")
    lines.append("")
    lines.append(r"\begin{align*}")

    for m_ext, e_ext, result in entries:
        if not result:
            continue
        charge = _fmt_charge(m_ext, e_ext)
        if ab_valid:
            centre, stripped = strip_weyl_monomial(
                result, m_ext, e_ext, weyl_result.ab, num_hard
            )
            prefix = centre_to_latex(centre, num_hard)
            body_latex = series_to_latex(stripped, num_hard)
            if prefix == "1":
                lines.append(rf"I({charge}) &= {body_latex} \\")
            else:
                lines.append(rf"I({charge}) &= {prefix} \left( {body_latex} \right) \\")
        else:
            series = series_to_latex(result, num_hard)
            lines.append(rf"I({charge}) &= {series} \\")

    lines.append(r"\end{align*}")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # §10  Refined Dehn Filling Result
    # ══════════════════════════════════════════════════════════════
    if filled_refined_result is not None:
        lines.append(r"\section{Refined Dehn Filling}")
        P_rf = filled_refined_result.P
        Q_rf = filled_refined_result.Q
        lines.append(
            rf"Refined Dehn-filled index at slope $({P_rf},\, {Q_rf})$:"
        )
        lines.append("")

        if filled_refined_result.is_zero:
            lines.append(
                rf"$$I^{{\text{{ref}}}}_{{{P_rf}/{Q_rf}}}(\eta) = 0$$"
            )
        else:
            text = filled_refined_result.as_q_eta_string(
                q_var="q", eta_var=r"\eta", half_pow=True
            )
            # Escape LaTeX-unfriendly characters and wrap
            lines.append(r"\begin{equation}")
            lines.append(
                rf"  I^{{\text{{ref}}}}_{{{P_rf}/{Q_rf}}}(\eta) = {text}"
            )
            lines.append(r"\end{equation}")

        # Show η₁-specialisation if available
        try:
            eta1 = filled_refined_result.eta1_series()
            if eta1:
                lines.append("")
                lines.append(r"\subsection*{Specialisation $\eta = 1$}")
                poly = _filled_series_to_latex(eta1)
                lines.append(
                    rf"$$I^{{\text{{ref}}}}_{{{P_rf}/{Q_rf}}}(\eta\!=\!1) = {poly}$$"
                )
        except Exception:
            pass

        lines.append("")

    lines.append(r"\end{document}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Results directory & auto-save
# ---------------------------------------------------------------------------

def results_dir() -> Path:
    """Return (and create) the results directory.

    - **Development** (pyproject.toml found): ``<project>/results/``
    - **Frozen app** or fallback: ``~/Documents/ManifoldIndex/results/``

    The latter ensures the auto-save always targets a user-writable
    directory, even when ``cwd`` is ``/`` (as happens when a user
    double-clicks a ``.app`` bundle on macOS).
    """
    import sys

    # Frozen bundle → always use the user's Documents folder
    if getattr(sys, "frozen", False):
        d = Path.home() / "Documents" / "ManifoldIndex" / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # Development: walk up to the project root
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            d = parent / "results"
            d.mkdir(exist_ok=True)
            return d

    # Ultimate fallback
    d = Path.home() / "Documents" / "ManifoldIndex" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def auto_save_nb(
    entries: list, name: str, q_ord: int, num_hard: int, basis_summary: str
) -> Path:
    """Write the .nb file to results/{name}_index.nb and return the path."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    content = build_nb_content(entries, name, q_ord, num_hard, basis_summary, ts)
    out_path = results_dir() / f"{name}_index.nb"
    out_path.write_text(content, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# KaTeX HTML formatters (for in-app math display)
# ---------------------------------------------------------------------------

def _html_esc(s: str) -> str:
    """Escape text for safe embedding in HTML."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_series_katex_html(
    entries: list,
    num_hard: int,
    weyl_result=None,
) -> str:
    """Build HTML with embedded KaTeX $$…$$ blocks for the refined index.

    Parameters
    ----------
    entries : list of (m_ext, e_ext, RefinedIndexResult)
    num_hard : int
    weyl_result : WeylCheckResult | None

    Returns
    -------
    str — HTML body with KaTeX-delimited math.
    """
    from manifold_index.core.weyl_check import strip_weyl_monomial

    ab_valid = weyl_result is not None and weyl_result.ab_valid
    html_parts: list[str] = []

    for m_ext, e_ext, result in entries:
        charge = _fmt_charge(m_ext, e_ext)
        label = f"<div class=\"sector-label\">({charge})</div>"

        if not result:
            html_parts.append(
                f'<div class="sector">{label}'
                f'$$I({_latex_esc_charge(m_ext, e_ext)}) = 0$$'
                f'</div>'
            )
            continue

        lhs = f"I({_latex_esc_charge(m_ext, e_ext)})"

        if ab_valid:
            centre, stripped = strip_weyl_monomial(
                result, m_ext, e_ext, weyl_result.ab, num_hard
            )
            prefix = centre_to_latex(centre, num_hard)
            body = series_to_latex(stripped, num_hard)
            if prefix == "1":
                math = f"{lhs} = {body}"
            else:
                math = f"{lhs} = {prefix} \\left( {body} \\right)"
        else:
            body = series_to_latex(result, num_hard)
            math = f"{lhs} = {body}"

        html_parts.append(
            f'<div class="sector">{label}$${math}$$</div>'
        )

    return "\n".join(html_parts) if html_parts else "<p>No sectors computed.</p>"


def _latex_esc_charge(m_ext: list, e_ext: list) -> str:
    """Format external charges for LaTeX."""
    parts: list[str] = []
    for v in m_ext:
        parts.append(str(v) if v == 0 else f"{v:+d}")
    for v in e_ext:
        f = Fraction(v).limit_denominator(1000)
        if f == 0:
            parts.append("0")
        elif f.denominator == 1:
            parts.append(str(f.numerator))
        else:
            parts.append(rf"\tfrac{{{f.numerator}}}{{{f.denominator}}}")
    return ",\\, ".join(parts)


def format_manifold_info_html(
    name: str,
    nz_data,
    manifold_data=None,
    easy_result=None,
) -> str:
    """Build HTML for the manifold/edge info panel (no KaTeX needed)."""
    n = nz_data.n
    r = nz_data.r
    num_hard = nz_data.num_hard
    num_easy = nz_data.num_easy

    lines: list[str] = []
    lines.append(f"<h3>Manifold: {_html_esc(name)}</h3>")
    lines.append(
        f"<p>Tetrahedra: <b>{n}</b> &nbsp;&bull;&nbsp; "
        f"Cusps: <b>{r}</b> &nbsp;&bull;&nbsp; "
        f"Internal edges: <b>{n - r}</b> "
        f"(hard: <b>{num_hard}</b>, easy: <b>{num_easy}</b>)</p>"
    )

    # SnaPy edge equations
    if manifold_data is not None:
        lines.append("<h3>SnaPy Edge Equations</h3>")
        lines.append('<table class="edge-table"><tr><th>Edge</th><th>Triplets (Z, Z′, Z″) per tet</th></tr>')
        edge_eqs = manifold_data.edge_equations
        for i in range(n):
            row = edge_eqs[i]
            parts = []
            for t in range(n):
                triplet = row[3 * t: 3 * t + 3]
                parts.append(f"({triplet[0]},{triplet[1]},{triplet[2]})")
            lines.append(
                f"<tr><td><b>E{i}</b></td><td>{_html_esc(' '.join(parts))}</td></tr>"
            )
        lines.append("</table>")

    # Easy / hard edge classification
    if easy_result is not None:
        lines.append("<h3>Easy Edges</h3>")
        lines.append('<table class="edge-table"><tr><th>#</th><th>Triplets</th><th>Role</th></tr>')
        for idx, edge_vec in enumerate(easy_result.all_easy):
            marker = ""
            if idx in easy_result.independent_easy_indices:
                basis_pos = r + num_hard + easy_result.independent_easy_indices.index(idx)
                marker = f'<span class="success">basis row {basis_pos}</span>'
            parts = []
            for t in range(n):
                triplet = edge_vec[3 * t: 3 * t + 3]
                parts.append(f"({triplet[0]},{triplet[1]},{triplet[2]})")
            lines.append(
                f"<tr><td><b>E{idx}</b></td>"
                f"<td>{_html_esc(' '.join(parts))}</td>"
                f"<td>{marker}</td></tr>"
            )
        lines.append("</table>")
        lines.append(
            f'<p>Independent easy: <b>{len(easy_result.independent_easy_indices)}</b> '
            f'&mdash; Indices: {easy_result.independent_easy_indices}</p>'
        )

        if easy_result.hard_padding:
            lines.append("<h3>Hard Edges (SnaPy padding)</h3>")
            lines.append('<table class="edge-table"><tr><th>#</th><th>Triplets</th><th>Basis row</th></tr>')
            for h_idx, hard_vec in enumerate(easy_result.hard_padding):
                basis_pos = r + h_idx
                parts = []
                for t in range(n):
                    triplet = hard_vec[3 * t: 3 * t + 3]
                    parts.append(f"({triplet[0]},{triplet[1]},{triplet[2]})")
                lines.append(
                    f"<tr><td><b>H{h_idx}</b></td>"
                    f"<td>{_html_esc(' '.join(parts))}</td>"
                    f"<td>row {basis_pos}</td></tr>"
                )
            lines.append("</table>")
        else:
            lines.append('<p class="muted">No hard edges — all internal edges are easy.</p>')

    return "\n".join(lines)
