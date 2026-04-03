"""Basis selection: cycle choice per cusp and cusp basis change application."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd


@dataclass
class CycleChoice:
    cusp_idx: int
    P: int
    Q: int
    label: str = ""
    is_default: bool = False

    def __post_init__(self):
        if self.P == 0 and self.Q == 0:
            raise ValueError("CycleChoice: (P, Q) = (0, 0) is not a valid slope")
        if gcd(abs(self.P), abs(self.Q)) != 1:
            raise ValueError(
                f"CycleChoice: (P={self.P}, Q={self.Q}) is not primitive; "
                f"gcd = {gcd(abs(self.P), abs(self.Q))}"
            )
        if not self.label:
            if self.P == 1 and self.Q == 0:
                self.label = "meridian M (1/0)"
            elif self.P == 0 and self.Q == 1:
                self.label = "longitude L (0/1)"
            else:
                self.label = f"slope {self.P}/{self.Q}"

    @property
    def m(self) -> int:
        return self.P

    @property
    def e(self) -> Fraction:
        return Fraction(self.Q, 2)

    @property
    def slope_str(self) -> str:
        return f"{self.P}/{self.Q}"


@dataclass
class BasisSelection:
    choices: list[CycleChoice]

    def __post_init__(self):
        if not self.choices:
            raise ValueError("BasisSelection: choices must be non-empty")
        for i, cc in enumerate(self.choices):
            if cc.cusp_idx != i:
                raise ValueError(
                    f"BasisSelection: choices[{i}].cusp_idx = {cc.cusp_idx} ≠ {i}"
                )

    @property
    def r(self) -> int:
        return len(self.choices)

    @property
    def m_ext(self) -> list[int]:
        return [cc.m for cc in self.choices]

    @property
    def e_ext(self) -> list[Fraction]:
        return [cc.e for cc in self.choices]

    def summary(self) -> str:
        lines = []
        for cc in self.choices:
            lines.append(f"  cusp {cc.cusp_idx}: {cc.label}  (m={cc.m}, e={cc.e})")
        return "\n".join(lines)


def default_meridian_choice(cusp_idx: int) -> CycleChoice:
    return CycleChoice(cusp_idx, P=1, Q=0, label="meridian M (1/0)", is_default=True)


def default_longitude_choice(cusp_idx: int) -> CycleChoice:
    return CycleChoice(cusp_idx, P=0, Q=1, label="longitude L (0/1)", is_default=True)


def make_basis_selection(
    nz_data,
    cycle_results: list,
    choices: list,
    *,
    default: str = "M",
    strict: bool = False,
) -> BasisSelection:
    """Build a BasisSelection from a list of (P,Q) tuples or None per cusp.

    Parameters
    ----------
    nz_data : provides .r (number of cusps)
    cycle_results : list[NonClosableCycleResult] (from Phase 9), may be empty
    choices : list[tuple[int,int] | None], length r
    default : "M" or "L" — which default curve for None entries
    strict : if True, raise ValueError if chosen slope not in cycle_results
    """
    r = nz_data.r
    found_slopes: dict[int, set[tuple[int, int]]] = {}
    for ncr in cycle_results:
        idx = ncr.cusp_idx
        found_slopes.setdefault(idx, set())
        for nc in ncr.cycles:
            found_slopes[idx].add((nc.P, nc.Q))

    cusp_choices: list[CycleChoice] = []
    for i in range(r):
        ch = choices[i] if i < len(choices) else None
        if ch is None:
            if default == "L":
                cc = default_longitude_choice(i)
            else:
                cc = default_meridian_choice(i)
        else:
            P, Q = int(ch[0]), int(ch[1])
            if gcd(abs(P), abs(Q)) != 1:
                raise ValueError(f"make_basis_selection: ({P},{Q}) not primitive")
            if strict:
                slopes = found_slopes.get(i, set())
                if (P, Q) not in slopes:
                    raise ValueError(
                        f"make_basis_selection: slope ({P},{Q}) not found in "
                        f"non-closable cycles for cusp {i}"
                    )
            is_def = (P == 1 and Q == 0) or (P == 0 and Q == 1)
            cc = CycleChoice(cusp_idx=i, P=P, Q=Q, is_default=is_def)
        cusp_choices.append(cc)

    return BasisSelection(choices=cusp_choices)


def apply_basis_changes(nz_data, basis: BasisSelection):
    """Apply cusp basis changes for all odd-P cusps in the selection.

    Even-P cusps (including P=0 for longitude) are skipped — the caller
    evaluates at (m=P, e=Q/2) using the unchanged basis for that cusp.

    Returns a NeumannZagierData with only odd-P cusps basis-changed,
    applied sequentially.
    """
    from manifold_index.core.neumann_zagier import apply_cusp_basis_change
    nz = nz_data
    for cc in basis.choices:
        if cc.P % 2 != 0:
            nz = apply_cusp_basis_change(nz, cc.cusp_idx, cc.P, cc.Q)
    return nz
