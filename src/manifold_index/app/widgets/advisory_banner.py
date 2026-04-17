"""app/widgets/advisory_banner.py — Inline advisory/warning banner.

Renders one ``Advisory`` dataclass as a horizontal stripe inside a
``CollapsibleCard``.  See BLUEPRINT §2.5 and §9.3.

Layout::

    ┌────────────────────────────────────────────────────────────────────┐
    │  Left border (4 px, colour matches level)                          │
    │  [level tag]   TITLE                                               │
    │  Body text                                 [Action A] [Action B]   │
    └────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from manifold_index.viewmodels.advisory import Advisory, AdvisoryLevel


# Mapping level → QFrame property class name (matched in QSS)
_LEVEL_CLASS: dict[AdvisoryLevel, str] = {
    AdvisoryLevel.INFO:    "advisory-info",
    AdvisoryLevel.WARNING: "advisory-warning",
    AdvisoryLevel.ERROR:   "advisory-error",
    AdvisoryLevel.ACTION:  "advisory-action",
}

# Small-caps tag text shown to the left of the title
_LEVEL_TAG: dict[AdvisoryLevel, str] = {
    AdvisoryLevel.INFO:    "info",
    AdvisoryLevel.WARNING: "warning",
    AdvisoryLevel.ERROR:   "error",
    AdvisoryLevel.ACTION:  "action required",
}


class AdvisoryBanner(QFrame):
    """Renders one Advisory as a styled inline banner.

    Signals:
        dismissed — emitted when an action button is pressed (if the
                    action's callback returns ``"dismiss"`` — caller
                    decides whether to hide the banner).
    """

    dismissed = Signal()

    def __init__(self, advisory: Advisory, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        level_class = _LEVEL_CLASS.get(advisory.level, "advisory-info")
        self.setProperty("class", level_class)
        self.setFrameShape(QFrame.Shape.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(4)

        # --- Top row: level tag + title ---
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        tag_text = _LEVEL_TAG.get(advisory.level, "info")
        tag_label = QLabel(tag_text.upper())
        tag_label.setProperty("class", "advisory-level-tag")
        tag_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        title_label = QLabel(advisory.title)
        title_label.setProperty("class", "advisory-title")
        title_label.setWordWrap(True)

        top_row.addWidget(tag_label)
        top_row.addWidget(title_label, 1)
        outer.addLayout(top_row)

        # --- Body text ---
        if advisory.body:
            body_label = QLabel(advisory.body)
            body_label.setProperty("class", "advisory-body")
            body_label.setWordWrap(True)
            outer.addWidget(body_label)

        # --- Action buttons (right-aligned) ---
        if advisory.actions:
            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(0, 2, 0, 0)
            btn_row.setSpacing(6)
            btn_row.addStretch(1)
            for action in advisory.actions:
                btn = QPushButton(action.label)
                btn.setProperty("class", "secondary")
                # callback=None means the card handles it externally; still
                # emit dismissed so the banner clears, but don't call None().
                cb = action.callback
                if cb is not None:
                    btn.clicked.connect(
                        lambda checked=False, _cb=cb: _cb()
                    )
                btn.clicked.connect(self.dismissed)
                btn_row.addWidget(btn)
            outer.addLayout(btn_row)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QScrollArea
    from manifold_index.viewmodels.advisory import AdvisoryAction

    app = QApplication(sys.argv)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setSpacing(8)
    layout.setContentsMargins(16, 16, 16, 16)

    banners = [
        Advisory(
            level=AdvisoryLevel.INFO,
            title="Kernel loaded from cache",
            body="The kernel was loaded from the local I^ref cache.",
            actions=[],
        ),
        Advisory(
            level=AdvisoryLevel.WARNING,
            title="No non-closable cycles found",
            body="The manifold has no NC cycles for this slope range.  "
                 "Dehn filling will use a fallback.",
            actions=[
                AdvisoryAction("Expand range", lambda: print("expand")),
            ],
        ),
        Advisory(
            level=AdvisoryLevel.ERROR,
            title="Weyl symmetry check failed",
            body="The refined index does not satisfy expected Weyl symmetry.",
            actions=[
                AdvisoryAction("Retry with η=0", lambda: print("retry")),
                AdvisoryAction("Dismiss", lambda: print("dismiss")),
            ],
        ),
        Advisory(
            level=AdvisoryLevel.ACTION,
            title="Multiple NC cycles available",
            body="Three NC cycles were found.  Select one to proceed with "
                 "Dehn filling.",
            actions=[
                AdvisoryAction("Use cycle 1", lambda: print("cycle 1")),
                AdvisoryAction("Use cycle 2", lambda: print("cycle 2")),
            ],
        ),
    ]

    for adv in banners:
        layout.addWidget(AdvisoryBanner(adv))

    layout.addStretch(1)
    scroll.setWidget(container)
    scroll.setWindowTitle("AdvisoryBanner smoke test")
    scroll.resize(700, 500)
    scroll.show()
    sys.exit(app.exec())
