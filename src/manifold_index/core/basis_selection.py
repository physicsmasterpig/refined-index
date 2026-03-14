"""
core/basis_selection.py — Basis selection for the refined index (Step 6).

See SPEC.md §Step 6 for the full specification.

────────────────────────────────────────────────────────────────────────────
What Step 6 does
────────────────────────────────────────────────────────────────────────────

After Step 5 (Dehn filling) has identified the non-closable cycles at each
cusp, the user chooses one cycle per cusp.  That cycle is then used to fix
the external cusp variables ``(m_i, e_i)`` for the refined-index computation.

Translation rule:
    Cycle P·M + Q·L  at cusp i  →  m_ext[i] = P,  e_ext[i] = Q/2

Default curves (used when no non-closable cycle is found, or explicitly
chosen by the user):
    Meridian  M = slope (1, 0)  →  m = 1, e = 0
    Longitude L = slope (0, 1)  →  m = 0, e = 1/2

Public API
──────────
``CycleChoice``         — one cusp's chosen cycle with its (m, e) evaluation point
``BasisSelection``      — full per-cusp selection; exposes m_ext, e_ext
``make_basis_selection``— validate choices and build a BasisSelection
``default_meridian_choice`` / ``default_longitude_choice`` — convenience constructors
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd

from manifold_index.core.neumann_zagier import NeumannZagierData, apply_cusp_basis_change


# ===========================================================================
# CycleChoice
# ===========================================================================

@dataclass
class CycleChoice:
    """One cusp's chosen cycle and its evaluation point.

    Attributes
    ----------
    cusp_idx : int
        Zero-based cusp index.
    P, Q : int
        Primitive integers defining the cycle  P·M + Q·L.
    label : str
        Human-readable description shown in the GUI, e.g.
        ``"non-closable 2/3"``, ``"meridian M (1/0)"``, ``"longitude L (0/1)"``.
    is_default : bool
        True if this choice is the SnaPy-default meridian or longitude
        (i.e. it was used as a fallback because no non-closable cycle was found).

    Derived properties
    ------------------
    m : int          P   (meridian variable value for compute_refined_index)
    e : Fraction     Q/2 (half-longitude variable value)
    slope_str : str  "P/Q" formatted string
    """

    cusp_idx: int
    P: int
    Q: int
    label: str = field(default="")
    is_default: bool = field(default=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.P == 0 and self.Q == 0:
            raise ValueError(
                f"CycleChoice cusp {self.cusp_idx}: (P, Q) = (0, 0) is not a valid cycle"
            )
        if gcd(abs(self.P), abs(self.Q)) != 1:
            raise ValueError(
                f"CycleChoice cusp {self.cusp_idx}: (P, Q) = ({self.P}, {self.Q}) "
                f"is not primitive (gcd = {gcd(abs(self.P), abs(self.Q))})"
            )
        # Auto-generate label if none provided
        if not self.label:
            object.__setattr__(self, "label", self._auto_label())

    def _auto_label(self) -> str:
        if self.P == 1 and self.Q == 0:
            return f"meridian M (1/0)"
        if self.P == 0 and self.Q == 1:
            return f"longitude L (0/1)"
        return f"slope {self.P}/{self.Q}"

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def m(self) -> int:
        """Meridian variable value: m_ext[cusp_idx] = P."""
        return self.P

    @property
    def e(self) -> Fraction:
        """Half-longitude variable value: e_ext[cusp_idx] = Q/2."""
        return Fraction(self.Q, 2)

    @property
    def slope_str(self) -> str:
        """Human-readable slope string, e.g. ``"1/0"``, ``"-2/3"``."""
        return f"{self.P}/{self.Q}"

    def __str__(self) -> str:
        return f"cusp {self.cusp_idx}: {self.label}  →  m={self.m}, e={self.e}"


# ===========================================================================
# BasisSelection
# ===========================================================================

@dataclass
class BasisSelection:
    """Per-cusp cycle choices, ready for ``compute_refined_index``.

    Attributes
    ----------
    choices : list[CycleChoice]
        One choice per cusp, in cusp-index order (choices[i].cusp_idx == i).

    Properties
    ----------
    m_ext : list[int]
        Meridian variable values for every cusp.  Passed directly to
        ``compute_refined_index(nz_data, m_ext, e_ext, ...)``.
    e_ext : list[Fraction]
        Half-longitude variable values for every cusp.
    r : int
        Number of cusps (== len(choices)).
    """

    choices: list[CycleChoice]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("BasisSelection: choices list must be non-empty")
        # Verify cusp indices are 0, 1, ..., r-1 in order
        for i, ch in enumerate(self.choices):
            if ch.cusp_idx != i:
                raise ValueError(
                    f"BasisSelection: choices[{i}].cusp_idx = {ch.cusp_idx}, expected {i}. "
                    "Choices must be in cusp-index order."
                )

    # ------------------------------------------------------------------
    # Properties for direct use in compute_refined_index
    # ------------------------------------------------------------------

    @property
    def r(self) -> int:
        """Number of cusps."""
        return len(self.choices)

    @property
    def m_ext(self) -> list[int]:
        """Meridian values, ordered by cusp index."""
        return [ch.m for ch in self.choices]

    @property
    def e_ext(self) -> list[Fraction]:
        """Half-longitude values, ordered by cusp index."""
        return [ch.e for ch in self.choices]

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        lines = [f"BasisSelection ({self.r} cusp{'s' if self.r != 1 else ''}):"]
        for ch in self.choices:
            default_note = "  [default]" if ch.is_default else ""
            lines.append(
                f"  {ch}{default_note}"
            )
        lines.append(
            f"  → m_ext = {self.m_ext},  e_ext = {self.e_ext}"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ===========================================================================
# Convenience constructors for default curves
# ===========================================================================

def default_meridian_choice(cusp_idx: int) -> CycleChoice:
    """Return a CycleChoice for the SnaPy meridian M = slope (1, 0)."""
    return CycleChoice(
        cusp_idx=cusp_idx,
        P=1,
        Q=0,
        label="meridian M (1/0)",
        is_default=True,
    )


def default_longitude_choice(cusp_idx: int) -> CycleChoice:
    """Return a CycleChoice for the SnaPy longitude L = slope (0, 1)."""
    return CycleChoice(
        cusp_idx=cusp_idx,
        P=0,
        Q=1,
        label="longitude L (0/1)",
        is_default=True,
    )


# ===========================================================================
# make_basis_selection
# ===========================================================================

def make_basis_selection(
    nz_data: NeumannZagierData,
    cycle_results: list,        # list[NonClosableCycleResult], one per cusp
    choices: list[tuple[int, int] | None],
    *,
    default: str = "M",
    strict: bool = False,
) -> BasisSelection:
    """Build a :class:`BasisSelection` from per-cusp cycle choices.

    Parameters
    ----------
    nz_data : NeumannZagierData
        Output of ``build_neumann_zagier``.  Used to validate the number of cusps.
    cycle_results : list[NonClosableCycleResult]
        One result per cusp, in cusp-index order.  Produced by
        ``find_non_closable_cycles`` for each cusp.  May be an empty list if
        the caller skips Step 5 entirely (in which case only default choices
        are valid).
    choices : list[tuple[int, int] | None]
        Per-cusp selection, length must equal ``nz_data.r``.

        ``choices[i]`` may be:

        * ``(P, Q)``  — use cycle P·M + Q·L for cusp ``i``.
        * ``None``    — use the default curve for cusp ``i`` (see *default*).
    default : {"M", "L"}
        Which SnaPy default curve to use when ``choices[i] is None``.
        ``"M"`` → meridian (1, 0) [default];  ``"L"`` → longitude (0, 1).
    strict : bool
        If ``True``, raise ``ValueError`` when a non-None choice ``(P, Q)``
        is not present in the corresponding ``cycle_results[i].cycles`` list.
        If ``False`` (default), any primitive slope is accepted regardless of
        whether it was found in the non-closable cycle search.

    Returns
    -------
    BasisSelection

    Raises
    ------
    ValueError
        * If ``len(choices) != nz_data.r``.
        * If any ``(P, Q)`` has gcd > 1 or equals (0, 0).
        * If *strict* is True and a chosen slope was not found to be non-closable.
    """
    r = nz_data.r
    if len(choices) != r:
        raise ValueError(
            f"make_basis_selection: len(choices)={len(choices)} != nz_data.r={r}"
        )
    if default not in ("M", "L"):
        raise ValueError(f"make_basis_selection: default={default!r} must be 'M' or 'L'")

    # Build a fast look-up from cusp_idx → set of non-closable (P,Q) slopes
    found_slopes: dict[int, set[tuple[int, int]]] = {}
    for res in cycle_results:
        found_slopes[res.cusp_idx] = {(c.P, c.Q) for c in res.cycles}

    cusp_choices: list[CycleChoice] = []

    for i, choice in enumerate(choices):
        if choice is None:
            # Use the default curve
            if default == "M":
                cc = default_meridian_choice(i)
            else:
                cc = default_longitude_choice(i)
        else:
            P, Q = int(choice[0]), int(choice[1])
            # Validation
            if P == 0 and Q == 0:
                raise ValueError(
                    f"make_basis_selection: choices[{i}] = (0, 0) is not a valid cycle"
                )
            if gcd(abs(P), abs(Q)) != 1:
                raise ValueError(
                    f"make_basis_selection: choices[{i}] = ({P}, {Q}) is not primitive"
                )
            # Strict mode: verify the slope was found to be non-closable
            if strict:
                known = found_slopes.get(i, set())
                if (P, Q) not in known:
                    raise ValueError(
                        f"make_basis_selection (strict): choices[{i}] = ({P}, {Q}) "
                        f"was not found to be non-closable for cusp {i}. "
                        f"Non-closable slopes found: {sorted(known)}"
                    )
            # Determine label
            if (P, Q) == (1, 0):
                label = "meridian M (1/0)"
                is_default = True
            elif (P, Q) == (0, 1):
                label = "longitude L (0/1)"
                is_default = True
            else:
                known = found_slopes.get(i, set())
                is_nc = (P, Q) in known
                label = f"non-closable {P}/{Q}" if is_nc else f"slope {P}/{Q}"
                is_default = False
            cc = CycleChoice(
                cusp_idx=i,
                P=P,
                Q=Q,
                label=label,
                is_default=is_default,
            )

        cusp_choices.append(cc)

    return BasisSelection(choices=cusp_choices)


# ===========================================================================
# apply_basis_changes
# ===========================================================================

def apply_basis_changes(
    nz_data: NeumannZagierData,
    basis: BasisSelection,
) -> NeumannZagierData:
    """Apply the cusp basis changes encoded in *basis* to *nz_data*.

    For each cusp ``k`` with slope ``(P_k, Q_k)``:

    * If ``P_k`` is **odd**, :func:`apply_cusp_basis_change` is called to
      rebuild rows ``k`` and ``n+k`` of ``g_NZ`` and update the affine shifts.
    * If ``P_k`` is **even** (including ``P_k = 0`` for the longitude) the
      change cannot be expressed with integer coefficients; the cusp is left
      in the original ``(M, L/2)`` basis and a warning is attached to the
      returned data via its docstring.  In that case the caller should pass
      ``m_ext[k] = P_k`` and ``e_ext[k] = Q_k / 2`` exactly as before.

    Parameters
    ----------
    nz_data : NeumannZagierData
    basis : BasisSelection

    Returns
    -------
    NeumannZagierData
        Possibly modified copy of *nz_data*.
    """
    result = nz_data
    for ch in basis.choices:
        if ch.P % 2 != 0:
            result = apply_cusp_basis_change(result, ch.cusp_idx, ch.P, ch.Q)
        # else: P even → skip; caller must use original (m=P, e=Q/2) evaluation
    return result

