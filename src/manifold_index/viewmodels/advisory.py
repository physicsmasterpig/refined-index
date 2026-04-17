"""
viewmodels.advisory
===================
Advisory system — level enum, card-status enum, action dataclass,
and the Advisory dataclass itself.

No Qt dependency.  No core/ imports.  Pure data.

BLUEPRINT references
--------------------
§2.5  Advisory Banners (visual layout)
§12   Edge Case & Advisory System (all category A–D IDs)
§13   Phase 3 implementation notes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AdvisoryLevel(Enum):
    """Severity / presentation level of an advisory banner."""
    INFO    = "info"      # neutral information, blue tint
    WARNING = "warning"   # caution, amber tint
    ERROR   = "error"     # hard failure, red tint
    ACTION  = "action"    # user decision required, purple tint


class CardStatus(Enum):
    """State of a CollapsibleCard in the pipeline."""
    LOCKED  = "locked"
    READY   = "ready"
    RUNNING = "running"
    DONE    = "done"
    WARNING = "warning"
    ERROR   = "error"
    STALE   = "stale"


# ---------------------------------------------------------------------------
# AdvisoryAction
# ---------------------------------------------------------------------------

@dataclass
class AdvisoryAction:
    """One clickable button inside an advisory banner.

    Parameters
    ----------
    label : str
        Text displayed on the button.
    callback : Callable[[], None] | None
        Called when the button is clicked.  ``None`` means no-op (the
        action is cosmetic / handled externally by the card widget).
    """
    label: str
    callback: Callable[[], None] | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Advisory
# ---------------------------------------------------------------------------

@dataclass
class Advisory:
    """An advisory banner attached to a pipeline card.

    Parameters
    ----------
    advisory_id : str
        Stable identifier such as ``"A1"``, ``"B3"``, ``"C1"``.
    level : AdvisoryLevel
        Controls visual styling and which banners are shown when collapsed.
    title : str
        Short title (≤ 60 chars) shown in bold.
    body : str
        Longer explanation shown in the expanded banner.
    actions : list[AdvisoryAction]
        Inline action buttons (may be empty for INFO / WARNING advisories).

    Notes
    -----
    Advisories are **stateless** — they are re-generated each time a
    ViewModel is rebuilt.  No mutable state is stored here.
    """
    advisory_id: str
    level: AdvisoryLevel
    title: str
    body: str
    actions: list[AdvisoryAction] = field(default_factory=list)

    # ── Convenience constructors ──────────────────────────────────────

    @classmethod
    def info(cls, advisory_id: str, title: str, body: str,
             actions: list[AdvisoryAction] | None = None) -> "Advisory":
        return cls(advisory_id, AdvisoryLevel.INFO, title, body, actions or [])

    @classmethod
    def warning(cls, advisory_id: str, title: str, body: str,
                actions: list[AdvisoryAction] | None = None) -> "Advisory":
        return cls(advisory_id, AdvisoryLevel.WARNING, title, body, actions or [])

    @classmethod
    def error(cls, advisory_id: str, title: str, body: str,
              actions: list[AdvisoryAction] | None = None) -> "Advisory":
        return cls(advisory_id, AdvisoryLevel.ERROR, title, body, actions or [])

    @classmethod
    def action(cls, advisory_id: str, title: str, body: str,
               actions: list[AdvisoryAction] | None = None) -> "Advisory":
        return cls(advisory_id, AdvisoryLevel.ACTION, title, body, actions or [])

    # ── Helpers ───────────────────────────────────────────────────────

    @property
    def is_blocking(self) -> bool:
        """True for ERROR and ACTION advisories — user must respond."""
        return self.level in (AdvisoryLevel.ERROR, AdvisoryLevel.ACTION)

    @property
    def show_when_collapsed(self) -> bool:
        """ERROR advisories always show even when the card is collapsed."""
        return self.level == AdvisoryLevel.ERROR


# ---------------------------------------------------------------------------
# Pre-built advisory factories for each BLUEPRINT category
# ---------------------------------------------------------------------------

class Advisories:
    """Namespace of factory methods for every advisory defined in BLUEPRINT §12."""

    # ── Category A — Manifold properties ─────────────────────────────

    @staticmethod
    def A1() -> Advisory:
        """All edges easy: refined index == 3D index."""
        return Advisory.info(
            "A1",
            "All edges easy",
            "Refined index equals the ordinary 3D index. No \u03b7 variables.",
        )

    @staticmethod
    def A2(name: str) -> Advisory:
        """SnaPy cannot find the manifold."""
        return Advisory.error(
            "A2",
            "Manifold not found",
            f"SnaPy cannot find \u2018{name}\u2019. Check spelling.",
            actions=[AdvisoryAction("Suggest examples")],
        )

    @staticmethod
    def A3() -> Advisory:
        """Degenerate triangulation shapes."""
        return Advisory.warning(
            "A3",
            "Degenerate triangulation",
            "Shape parameters may be unreliable. Consider retriangulating.",
            actions=[AdvisoryAction("Proceed anyway")],
        )

    # ── Category B — Refined index ───────────────────────────────────

    @staticmethod
    def B1(qq: int) -> Advisory:
        """Result is identically zero."""
        return Advisory.info(
            "B1",
            "Sector vanishes",
            f"I\u02beref(m,e) = 0 at qq={qq}. May be exact or truncation artifact.",
        )

    @staticmethod
    def B2(cache_qq: int, req_qq: int) -> Advisory:
        """Cache qq < requested qq."""
        return Advisory.action(
            "B2",
            "Cache qq mismatch",
            f"Cache at qq={cache_qq}, requested qq={req_qq}.",
            actions=[
                AdvisoryAction(f"Use cache (qq={cache_qq})"),
                AdvisoryAction("Compute fresh"),
            ],
        )

    @staticmethod
    def B3() -> Advisory:
        """Weyl extraction failed."""
        return Advisory.warning(
            "B3",
            "Weyl extraction failed",
            "Cannot extract (a,b) vectors. Filling proceeds without Weyl correction.",
        )

    @staticmethod
    def B4(edges: list[int]) -> Advisory:
        """Partial Weyl incompatibility."""
        edge_str = ", ".join(str(e) for e in edges)
        return Advisory.warning(
            "B4",
            "Partial Weyl incompatibility",
            f"Edges {edge_str} incompatible with half-integer e. "
            "Will be projected out for filling.",
        )

    @staticmethod
    def B5(val: float) -> Advisory:
        """Adjoint su(2) check failed."""
        return Advisory.warning(
            "B5",
            "Adjoint check failed",
            f"\u00bd(c\u208b\u2081+c\u208a\u2081\u2212c\u208b\u2082\u2212c\u208a\u2082)"
            f" = {val:.4g} \u2260 \u22121. Refined filling may be inconsistent.",
            actions=[AdvisoryAction("Proceed anyway"), AdvisoryAction("Increase qq")],
        )

    # ── Category C — Dehn Filling ─────────────────────────────────────

    @staticmethod
    def C1() -> Advisory:
        """No non-closable cycles found."""
        return Advisory.action(
            "C1",
            "No non-closable cycles found",
            "Every slope in the search range is closable. "
            "Refined filling requires an NC cycle as basis.",
            actions=[
                AdvisoryAction("Widen search range"),
                AdvisoryAction("Unrefined filling"),
                AdvisoryAction("Skip filling"),
            ],
        )

    @staticmethod
    def C2() -> Advisory:
        """Trivial slope in NC basis (p=0)."""
        return Advisory.warning(
            "C2",
            "Trivial slope in NC basis",
            "The user\u2019s slope maps to the NC cycle itself (p=0). "
            "Filling along it gives zero by definition.",
        )

    @staticmethod
    def C3(P: int, Q: int, length: int, est_seconds: float) -> Advisory:
        """Kernel not cached for long HJ chain."""
        return Advisory.action(
            "C3",
            "Kernel not cached",
            f"Slope ({P},{Q}) \u2192 HJ chain of length {length}. "
            f"Estimated build time: ~{est_seconds:.0f}s.",
            actions=[
                AdvisoryAction("Build & cache kernel"),
                AdvisoryAction("Use unrefined kernel"),
            ],
        )

    @staticmethod
    def C4(r: int, n: int, total: int) -> Advisory:
        """Multi-cusp NC combination explosion."""
        return Advisory.action(
            "C4",
            "Many NC combinations",
            f"{r} cusps \u00d7 {n} NC cycles each = {total} combinations. "
            "Each requires a filling computation.",
            actions=[
                AdvisoryAction(f"Compute all {total}"),
                AdvisoryAction("Pick specific combination"),
            ],
        )

    @staticmethod
    def C5(P: int, Q: int) -> Advisory:
        """Non-primitive slope."""
        return Advisory.error(
            "C5",
            "Non-primitive slope",
            f"(P,Q) = ({P},{Q}) is not coprime. Dehn surgery requires a primitive cycle.",
        )

    @staticmethod
    def C6() -> Advisory:
        """Kernel computation cancelled."""
        return Advisory.info(
            "C6",
            "Build cancelled",
            "Kernel build was cancelled. Previous cache state is unaffected.",
            actions=[AdvisoryAction("Retry"), AdvisoryAction("Use unrefined")],
        )

    # ── Category D — Cross-card state ────────────────────────────────

    @staticmethod
    def D1(old_name: str) -> Advisory:
        """Manifold changed after downstream results exist."""
        return Advisory.action(
            "D1",
            "Manifold changed",
            f"All results for \u2018{old_name}\u2019 will be cleared.",
            actions=[AdvisoryAction("Confirm"), AdvisoryAction("Cancel")],
        )

    @staticmethod
    def D2(old_qq: int, new_qq: int) -> Advisory:
        """qq order changed after filling results computed."""
        return Advisory.warning(
            "D2",
            "qq order changed",
            f"Filling results were at qq={old_qq}. May be insufficient at qq={new_qq}.",
            actions=[AdvisoryAction("Keep results"), AdvisoryAction("Recompute")],
        )

    @staticmethod
    def D3(saved_qq: int, now_qq: int) -> Advisory:
        """Session restored with lower qq."""
        return Advisory.warning(
            "D3",
            "Restored session",
            f"Loaded session computed at qq={saved_qq}. Current setting is qq={now_qq}.",
            actions=[AdvisoryAction("Keep saved"), AdvisoryAction("Recompute")],
        )
