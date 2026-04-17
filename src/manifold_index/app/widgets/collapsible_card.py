"""app/widgets/collapsible_card.py — Expandable/collapsible pipeline card.

Core UI element.  All four pipeline cards use this widget.
See BLUEPRINT §9.1 and §2.6 for the state machine and visual layout.

State machine::

    LOCKED  → (prerequisite met) → READY
    READY   → (run)              → RUNNING → DONE | WARNING | ERROR
    DONE    ← expand/collapse    → DONE
    DONE | WARNING → (upstream)  → STALE
    STALE   → (re-run)           → RUNNING → DONE | WARNING | ERROR
    ERROR   → (re-run)           → RUNNING → DONE | WARNING | ERROR

Visual layout (expanded)::

    ┌─ ① Title ────────────────────── [status badge] [▴ collapse] ─┐
    │  advisory banner(s)                                            │
    │  body widget                                                   │
    └────────────────────────────────────────────────────────────────┘

Visual layout (collapsed)::

    ┌─ ① Title ── [summary] ─────────── [status badge] [▾ expand] ─┐
    └────────────────────────────────────────────────────────────────┘

Visual layout (locked)::

    ┌─ ④ Title ──────────────────────────────────── Locked ─────────┐
    │  (optional lock blurb set by caller via set_summary)           │
    └────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from manifold_index.viewmodels.advisory import Advisory, CardStatus
from manifold_index.app.widgets.advisory_banner import AdvisoryBanner
from manifold_index.app.theme import icons


# ---------------------------------------------------------------------------
# Badge text/class helpers
# ---------------------------------------------------------------------------

_STATUS_BADGE_TEXT: dict[CardStatus, str] = {
    CardStatus.LOCKED:  "Locked",
    CardStatus.READY:   "Ready",
    CardStatus.RUNNING: f"{icons.RUNNING} Running",
    CardStatus.DONE:    f"{icons.DONE} Done",
    CardStatus.WARNING: f"{icons.WARNING} Warning",
    CardStatus.ERROR:   f"{icons.ERROR} Error",
    CardStatus.STALE:   f"{icons.STALE} Stale",
}

_STATUS_BADGE_CLASS: dict[CardStatus, str] = {
    CardStatus.LOCKED:  "badge-locked",
    CardStatus.READY:   "badge-ready",
    CardStatus.RUNNING: "badge-running",
    CardStatus.DONE:    "badge-done",
    CardStatus.WARNING: "badge-warning",
    CardStatus.ERROR:   "badge-error",
    CardStatus.STALE:   "badge-stale",
}

# Numeric circle characters ①–⑨
_CIRCLE_DIGIT = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]


def _circle(n: int) -> str:
    if 1 <= n <= len(_CIRCLE_DIGIT):
        return _CIRCLE_DIGIT[n - 1]
    return str(n)


# ---------------------------------------------------------------------------
# CollapsibleCard
# ---------------------------------------------------------------------------

class CollapsibleCard(QFrame):
    """Expandable/collapsible pipeline section card.

    Parameters
    ----------
    card_index : int
        1-based index shown in the header circle (①, ②, …).
    title : str
        Card title shown in the header.

    Signals
    -------
    expand_requested(int)   — user clicked to expand (card_index).
    collapse_requested(int) — user clicked to collapse (card_index).
    """

    expand_requested   = Signal(int)
    collapse_requested = Signal(int)

    def __init__(
        self,
        card_index: int,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._card_index = card_index
        self._title_text = title
        self._status = CardStatus.LOCKED
        self._expanded = False
        self._body_widget: QWidget | None = None

        self.setProperty("class", "pipeline-card")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # --- Root layout ---
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)

        # --- Header ---
        self._header_frame = QFrame()
        self._header_frame.setProperty("class", "card-header")
        self._header_frame.setFrameShape(QFrame.Shape.NoFrame)
        self._header_layout = QHBoxLayout(self._header_frame)
        self._header_layout.setContentsMargins(12, 10, 12, 10)
        self._header_layout.setSpacing(8)

        # Circle number
        self._index_label = QLabel(_circle(card_index))
        self._index_label.setProperty("class", "card-index")
        self._index_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Title
        self._title_label = QLabel(title)
        self._title_label.setProperty("class", "card-title")

        # Summary (collapsed inline view)
        self._summary_label = QLabel()
        self._summary_label.setProperty("class", "card-summary")
        self._summary_label.setVisible(False)

        # Stretch
        self._header_stretch = QWidget()
        self._header_stretch.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        # Status badge
        self._badge_label = QLabel()
        self._badge_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Toggle button
        self._toggle_btn = QPushButton(icons.EXPAND)
        self._toggle_btn.setProperty("class", "card-toggle")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        self._toggle_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._header_layout.addWidget(self._index_label)
        self._header_layout.addWidget(self._title_label)
        self._header_layout.addWidget(self._summary_label, 1)
        self._header_layout.addWidget(self._header_stretch, 1)
        self._header_layout.addWidget(self._badge_label)
        self._header_layout.addWidget(self._toggle_btn)

        self._root.addWidget(self._header_frame)

        # --- Body area (hidden when collapsed / locked) ---
        self._body_frame = QFrame()
        self._body_frame.setProperty("class", "card-body")
        self._body_frame.setFrameShape(QFrame.Shape.NoFrame)
        self._body_layout = QVBoxLayout(self._body_frame)
        self._body_layout.setContentsMargins(12, 4, 12, 12)
        self._body_layout.setSpacing(6)

        # Advisory banners container
        self._advisories_widget = QWidget()
        self._advisories_layout = QVBoxLayout(self._advisories_widget)
        self._advisories_layout.setContentsMargins(0, 0, 0, 0)
        self._advisories_layout.setSpacing(4)
        self._body_layout.addWidget(self._advisories_widget)

        # Placeholder for the body widget
        self._body_placeholder = QWidget()
        self._body_placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._body_layout.addWidget(self._body_placeholder)

        self._body_frame.setVisible(False)
        self._root.addWidget(self._body_frame)

        # Apply initial locked state
        self._apply_status(CardStatus.LOCKED)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_status(self, status: CardStatus) -> None:
        """Update the card status and refresh visual state."""
        self._status = status
        self._apply_status(status)

    def get_status(self) -> CardStatus:
        """Return the current card status."""
        return self._status

    def set_summary(self, text: str) -> None:
        """Set the one-liner summary shown when the card is collapsed.

        Also used for locked blurb (visible in locked body area).
        """
        self._summary_label.setText(text)

    def set_body(self, widget: QWidget) -> None:
        """Replace the expanded body widget."""
        if self._body_widget is not None:
            self._body_layout.removeWidget(self._body_widget)
            self._body_widget.setParent(None)  # type: ignore[arg-type]
        self._body_widget = widget
        self._body_layout.removeWidget(self._body_placeholder)
        self._body_placeholder.setParent(None)  # type: ignore[arg-type]
        self._body_layout.addWidget(widget)

    def set_advisories(self, advisories: list[Advisory]) -> None:
        """Replace the advisory banner list inside the card body."""
        # Clear existing banners
        while self._advisories_layout.count():
            item = self._advisories_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for adv in advisories:
            banner = AdvisoryBanner(adv, parent=self)
            self._advisories_layout.addWidget(banner)

        self._advisories_widget.setVisible(bool(advisories))

    def is_expanded(self) -> bool:
        """Return True if the card is currently expanded."""
        return self._expanded

    def expand(self) -> None:
        """Programmatically expand the card."""
        if self._status == CardStatus.LOCKED:
            return
        self._set_expanded(True)

    def collapse(self) -> None:
        """Programmatically collapse the card."""
        self._set_expanded(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_toggle_clicked(self) -> None:
        if self._status == CardStatus.LOCKED:
            return
        if self._expanded:
            self.collapse_requested.emit(self._card_index)
            self._set_expanded(False)
        else:
            self.expand_requested.emit(self._card_index)
            self._set_expanded(True)

    def _set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._body_frame.setVisible(expanded)

        # Summary visible only in collapsed (non-locked) state
        is_locked = self._status == CardStatus.LOCKED
        self._summary_label.setVisible(not expanded and not is_locked)
        self._header_stretch.setVisible(expanded or is_locked)

        self._toggle_btn.setText(icons.COLLAPSE if expanded else icons.EXPAND)

    def _apply_status(self, status: CardStatus) -> None:
        badge_text  = _STATUS_BADGE_TEXT.get(status, status.value)
        badge_class = _STATUS_BADGE_CLASS.get(status, "badge-ready")

        self._badge_label.setText(badge_text)
        self._badge_label.setProperty("class", badge_class)
        # Force QSS re-evaluation after property change
        self._badge_label.style().unpolish(self._badge_label)
        self._badge_label.style().polish(self._badge_label)

        is_locked = status == CardStatus.LOCKED
        self._toggle_btn.setVisible(not is_locked)
        self._header_stretch.setVisible(is_locked or self._expanded)
        self._summary_label.setVisible(
            not self._expanded and not is_locked
        )

        if is_locked:
            self._body_frame.setVisible(False)
            self._expanded = False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QScrollArea, QVBoxLayout, QWidget
    from manifold_index.viewmodels.advisory import AdvisoryAction, AdvisoryLevel

    app = QApplication(sys.argv)
    from manifold_index.app.theme.style import build_stylesheet
    app.setStyleSheet(build_stylesheet())

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    container = QWidget()
    vbox = QVBoxLayout(container)
    vbox.setSpacing(8)
    vbox.setContentsMargins(16, 16, 16, 16)

    # Card 1 — DONE, expanded, with advisory
    c1 = CollapsibleCard(1, "Manifold")
    c1.set_status(CardStatus.DONE)
    c1.set_summary("m004 · 2 tet · 1 cusp · 1 hard edge")
    body1 = QLabel("<pre>m004 — 2 tetrahedra, volume 2.029…</pre>")
    c1.set_body(body1)
    c1.set_advisories([
        Advisory(
            advisory_id="A1",
            level=AdvisoryLevel.INFO,
            title="Kernel loaded from cache",
            body="Using local kernel cache for m004.",
        )
    ])
    c1.expand()
    vbox.addWidget(c1)

    # Card 2 — RUNNING
    c2 = CollapsibleCard(2, "Refined Index")
    c2.set_status(CardStatus.RUNNING)
    c2.set_summary("Computing (0, 0)…")
    body2 = QLabel("…computing…")
    c2.set_body(body2)
    c2.expand()
    vbox.addWidget(c2)

    # Card 3 — WARNING, collapsed
    c3 = CollapsibleCard(3, "Dehn Filling")
    c3.set_status(CardStatus.WARNING)
    c3.set_summary("1 NC cycle · slope (2, 1)")
    body3 = QLabel("results here")
    c3.set_body(body3)
    vbox.addWidget(c3)

    # Card 4 — LOCKED
    c4 = CollapsibleCard(4, "Export")
    c4.set_status(CardStatus.LOCKED)
    c4.set_summary("Complete steps ①–③ first")
    vbox.addWidget(c4)

    vbox.addStretch(1)
    scroll.setWidget(container)
    scroll.setWindowTitle("CollapsibleCard smoke test")
    scroll.resize(780, 700)
    scroll.show()
    sys.exit(app.exec())
