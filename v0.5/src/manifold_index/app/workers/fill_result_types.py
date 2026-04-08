"""fill_result_types.py — Lightweight tagged result types for Dehn filling export.

These classes serve purely as type-tags for ``isinstance`` checks inside
``utils/exporters.py``.  All attributes are set dynamically (via
``__new__`` + attribute assignment in ``export_service.py``) so the
classes carry no ``__init__`` arguments.

Attributes populated by the shim builder
-----------------------------------------
TransformedFillResult
    cusp_idx, P_user, Q_user, P_nc, Q_nc, R, S, p, q,
    weyl_a_phys, weyl_b_phys, fill_results

UnrefinedFillResult
    cusp_idx, P_user, Q_user, fill_results

MultiCuspFillResult
    cusp_info  (list of _CuspInfo), fill_result
"""

from __future__ import annotations

__all__ = [
    "TransformedFillResult",
    "UnrefinedFillResult",
    "MultiCuspFillResult",
]


class TransformedFillResult:
    """Single-cusp Dehn filling that used a non-closable cycle."""


class UnrefinedFillResult:
    """Single-cusp Dehn filling that fell back to the unrefined formula
    (no non-closable cycles were found in the search range)."""


class MultiCuspFillResult:
    """Simultaneous Dehn filling of multiple cusps."""
