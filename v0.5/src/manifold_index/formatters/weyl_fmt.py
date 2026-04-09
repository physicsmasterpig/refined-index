"""
formatters.weyl_fmt
===================
HTML formatters for Weyl symmetry data (Card ②).

Public API
----------
format_weyl_html(vm)           → HTML section: a/b vectors + check summary
format_compatibility_html(vm)  → <table> HTML: per-edge compatibility

Both functions accept a ``WeylViewModel`` from
``manifold_index.viewmodels.index_vm``.

BLUEPRINT reference: §4 (formatters split), §11 (IndexCard display)
"""
from __future__ import annotations

from fractions import Fraction

from manifold_index.viewmodels.index_vm import WeylViewModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frac_to_latex(v) -> str:
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(int(f))
    sign = "-" if f < 0 else ""
    return rf"{sign}\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"


# ---------------------------------------------------------------------------
# Weyl section HTML
# ---------------------------------------------------------------------------

def format_weyl_html(vm: WeylViewModel) -> str:
    """Return an HTML fragment for the Weyl check summary.

    Covers:
    - a_j, b_j Weyl vectors per hard edge
    - Dehn-filling compatibility flag per edge
    - Adjoint q^1 projection result

    Parameters
    ----------
    vm : WeylViewModel
        Pre-built viewmodel from ``build_weyl_vm``.
    """
    if not vm.checked:
        return '<p class="muted">Weyl check not yet run.</p>\n'

    lines: list[str] = []
    lines.append(
        "<p>Convention: "
        r"$f(m,e) = \eta^{\sum_I(a_I \cdot e_I + b_I \cdot m_I)} "
        r"\cdot I^{\text{ref}}(m,e)$, "
        r"&nbsp; Weyl: $f(m,e) = f(-m,-e)$</p>"
    )

    if vm.cusp_a_matrix is not None and vm.cusp_b_matrix is not None:
        # Multi-cusp: a_j and b_j are vectors with one component per cusp.
        lines.append(
            "<p>Per-cusp Weyl vectors "
            r"($a_j = (a_j^{(0)}, a_j^{(1)}, \ldots)$, same for $b_j$):</p>"
        )
        for j, (a_row, b_row) in enumerate(zip(vm.cusp_a_matrix, vm.cusp_b_matrix)):
            a_parts = ", ".join(_frac_to_latex(v) for v in a_row)
            b_parts = ", ".join(_frac_to_latex(v) for v in b_row)
            lines.append(
                f"<p>$a_{{{j}}} = ({a_parts}), \\quad b_{{{j}}} = ({b_parts})$</p>"
            )
        if vm.is_fully_compatible:
            lines.append(
                '<p class="success">'
                r"✓ &nbsp; $a \in \mathbb{Z}^r$, $b \in (\mathbb{Z}/2)^r$"
                " — Dehn filling compatible</p>"
            )
        else:
            lines.append(
                '<p class="warn">'
                r"⚠ &nbsp; Some edges not half-integer compatible"
                " — filling may need η zeroing</p>"
            )
    elif vm.a_vectors and vm.b_vectors:
        for j, (a, b) in enumerate(zip(vm.a_vectors, vm.b_vectors)):
            a_str = _frac_to_latex(Fraction(a).limit_denominator(1000))
            b_str = _frac_to_latex(Fraction(b).limit_denominator(1000))
            lines.append(f"<p>$a_{{{j}}} = {a_str}, \\quad b_{{{j}}} = {b_str}$</p>")

        if vm.is_fully_compatible:
            lines.append(
                '<p class="success">'
                r"✓ &nbsp; $a \in \mathbb{Z}$, $b \in \mathbb{Z}/2$"
                " — Dehn filling compatible</p>"
            )
        else:
            lines.append(
                '<p class="warn">'
                r"⚠ &nbsp; Some edges not half-integer compatible"
                " — filling may need η zeroing</p>"
            )
    else:
        lines.append(
            '<p class="warn">'
            "⚠ &nbsp; Could not determine Weyl vectors</p>"
        )

    # Adjoint check
    if vm.adjoint_passed is True:
        lines.append(
            '<p class="success">'
            r"✓ &nbsp; Adjoint $q^1$ projection: "
            r"$\mathcal{J}_{q^1}|_{\mathrm{adj}\,su(2)} = -1$ &nbsp; ✓</p>"
        )
    elif vm.adjoint_passed is False:
        val_str = (
            f"{vm.adjoint_value:.4g}"
            if vm.adjoint_value is not None
            else "non-integer"
        )
        lines.append(
            f'<p class="warn">'
            f"⚠ &nbsp; Adjoint $q^1$ projection: got {val_str}, "
            f"expected $-1$</p>"
        )
    else:
        lines.append(
            '<p class="muted">Adjoint $q^1$ projection: not computed</p>'
        )

    # Any misc warnings from the check
    for w in vm.warnings:
        lines.append(f'<p class="warn">⚠ &nbsp; {w}</p>')

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Compatibility table HTML
# ---------------------------------------------------------------------------

def format_compatibility_html(vm: WeylViewModel) -> str:
    """Return an HTML ``<table>`` for per-edge Weyl compatibility.

    Each row shows edge index, a_j, b_j, and a ✓/✗ compatibility badge.

    Parameters
    ----------
    vm : WeylViewModel
    """
    if not vm.checked or not vm.a_vectors:
        return '<p class="muted">No Weyl data available.</p>\n'

    header = (
        "<table>\n"
        "<tr>"
        "<th>Edge</th>"
        r"<th>$a_j$</th>"
        r"<th>$b_j$</th>"
        r"<th>$a_j \in \mathbb{Z}$</th>"
        r"<th>$2b_j \in \mathbb{Z}$</th>"
        "<th>Status</th>"
        "</tr>\n"
    )

    rows = ""
    for j, (a, b, ok) in enumerate(
        zip(vm.a_vectors, vm.b_vectors, vm.edge_compatible)
    ):
        fa = Fraction(a).limit_denominator(1000)
        fb = Fraction(b).limit_denominator(1000)
        a_ok = fa.denominator == 1
        b_ok = (fb * 2).denominator == 1
        a_icon = "✓" if a_ok else "✗"
        b_icon = "✓" if b_ok else "✗"
        status = (
            '<span style="color:#2ea043;">✓ compatible</span>'
            if ok
            else '<span style="color:#cf222e;">✗ incompatible</span>'
        )
        rows += (
            f"<tr>"
            f"<td>{j}</td>"
            f"<td>${_frac_to_latex(fa)}$</td>"
            f"<td>${_frac_to_latex(fb)}$</td>"
            f"<td>{a_icon}</td>"
            f"<td>{b_icon}</td>"
            f"<td>{status}</td>"
            f"</tr>\n"
        )

    return header + rows + "</table>\n"
