"""app/pipeline/pipeline_view.py — Scrollable column of 4 pipeline cards.

BLUEPRINT §10.1.

PipelineView holds the shared Session object and wires all four card signals
together.  Cards themselves never hold session state — they always read from
the single Session passed in.

Layout (top → bottom, inside QScrollArea)
  StepperBar  ["Manifold", "Index", "Fill", "Export"]
  ManifoldCard  (Card ①)
  IndexCard     (Card ②, initially locked)
  FillingCard   (Card ③, initially locked)
  ExportCard    (Card ④, initially locked)

Signal flow
-----------
Card emits session_updated(Session)
  → PipelineView._on_session_updated(session)
  → updates stepper statuses
  → unlocks/locks downstream cards

Card ① load_failed
  → locks Cards ②-④
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from manifold_index.services.session import PipelineStage, Session
from manifold_index.viewmodels.advisory import CardStatus
from manifold_index.app.widgets.stepper import StepperBar
from manifold_index.app.pipeline.manifold_card import ManifoldCard
from manifold_index.app.pipeline.index_card import IndexCard
from manifold_index.app.pipeline.filling_card import FillingCard
from manifold_index.app.pipeline.export_card import ExportCard


class PipelineView(QWidget):
    """Main pipeline panel: scrollable column of four cards + stepper.

    Signals
    -------
    session_changed(Session)  — forwarded from any card update.
    """

    session_changed = Signal(object)

    def __init__(
        self,
        session: Session | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session or Session()

        # ── Layout ────────────────────────────────────────────────────
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Stepper sits above the scroll area, pinned at top
        self._stepper = StepperBar(["Manifold", "Index", "Fill", "Export"])
        self._stepper.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._stepper.step_clicked.connect(self._on_step_clicked)
        root_layout.addWidget(self._stepper)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)  # type: ignore[attr-defined]
        root_layout.addWidget(scroll)

        # Container inside scroll
        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        cl = QVBoxLayout(container)
        cl.setContentsMargins(16, 16, 16, 16)
        cl.setSpacing(12)

        # ── Cards ─────────────────────────────────────────────────────
        self._manifold_card = ManifoldCard(self._session, parent=container)
        self._index_card    = IndexCard(self._session, parent=container)
        self._filling_card  = FillingCard(self._session, parent=container)
        self._export_card   = ExportCard(self._session, parent=container)

        cl.addWidget(self._manifold_card)
        cl.addWidget(self._index_card)
        cl.addWidget(self._filling_card)
        cl.addWidget(self._export_card)
        cl.addStretch(1)

        scroll.setWidget(container)

        # ── Wire signals ──────────────────────────────────────────────
        self._manifold_card.session_updated.connect(self._on_manifold_updated)
        self._manifold_card.load_failed.connect(self._on_load_failed)
        self._index_card.session_updated.connect(self._on_index_updated)
        self._filling_card.session_updated.connect(self._on_filling_updated)
        self._export_card.session_updated.connect(self._on_export_updated)

        # Initial stepper state
        self._sync_stepper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def session(self) -> Session:
        return self._session

    def restore_session(self, session: Session) -> None:
        """Replace the shared session (e.g. after loading from JSON).

        Re-renders all cards from the restored data.
        """
        self._session = session
        self._manifold_card.refresh(session)
        if session.stage >= PipelineStage.LOADED:
            self._index_card.unlock(session)
            self._index_card.refresh(session)
        if session.stage >= PipelineStage.INDEXED:
            self._filling_card.unlock(session)
            self._filling_card.refresh(session)
        if session.has_any_results():
            self._export_card.unlock(session)
            self._export_card.refresh(session)
        self._sync_stepper()

    # ------------------------------------------------------------------
    # Slot: Card ① finished
    # ------------------------------------------------------------------

    def _on_manifold_updated(self, session: Session) -> None:
        self._session = session
        # Unlock cards downstream
        self._index_card.unlock(session)
        # Lock cards further down in case this is a reload
        self._filling_card.lock()
        self._export_card.lock()
        # Export unlocks if there are any results (at min stage=LOADED)
        if session.has_any_results():
            self._export_card.unlock(session)
        self._sync_stepper()
        self.session_changed.emit(session)

    def _on_load_failed(self) -> None:
        self._index_card.lock()
        self._filling_card.lock()
        self._export_card.lock()
        self._sync_stepper()

    # ------------------------------------------------------------------
    # Slot: Card ② finished
    # ------------------------------------------------------------------

    def _on_index_updated(self, session: Session) -> None:
        self._session = session
        # Unlock filling once we have any index query
        if session.index_queries:
            self._filling_card.unlock(session)
        if session.has_any_results():
            self._export_card.unlock(session)
        self._sync_stepper()
        self.session_changed.emit(session)

    # ------------------------------------------------------------------
    # Slot: Card ③ finished
    # ------------------------------------------------------------------

    def _on_filling_updated(self, session: Session) -> None:
        self._session = session
        if session.has_any_results():
            self._export_card.unlock(session)
        self._sync_stepper()
        self.session_changed.emit(session)

    # ------------------------------------------------------------------
    # Slot: Card ④ exported
    # ------------------------------------------------------------------

    def _on_export_updated(self, session: Session) -> None:
        self._session = session
        self._sync_stepper()
        self.session_changed.emit(session)

    # ------------------------------------------------------------------
    # Stepper
    # ------------------------------------------------------------------

    def _sync_stepper(self) -> None:
        s = self._session
        # Card ① — Manifold
        if s.stage >= PipelineStage.LOADED:
            self._stepper.set_step_status(1, CardStatus.DONE)
        elif s.manifold_name:
            self._stepper.set_step_status(1, CardStatus.RUNNING)
        else:
            self._stepper.set_step_status(1, CardStatus.READY)

        # Card ② — Index
        if s.stage >= PipelineStage.INDEXED:
            self._stepper.set_step_status(2, CardStatus.DONE)
        elif s.stage >= PipelineStage.LOADED:
            self._stepper.set_step_status(2, CardStatus.READY)
        else:
            self._stepper.set_step_status(2, CardStatus.LOCKED)

        # Card ③ — Fill
        if s.stage >= PipelineStage.FILLED:
            self._stepper.set_step_status(3, CardStatus.DONE)
        elif s.index_queries:
            self._stepper.set_step_status(3, CardStatus.READY)
        else:
            self._stepper.set_step_status(3, CardStatus.LOCKED)

        # Card ④ — Export
        if s.export_path:
            self._stepper.set_step_status(4, CardStatus.DONE)
        elif s.has_any_results():
            self._stepper.set_step_status(4, CardStatus.READY)
        else:
            self._stepper.set_step_status(4, CardStatus.LOCKED)

    def _on_step_clicked(self, card_index: int) -> None:
        """Scroll to / expand the clicked card."""
        cards = [
            self._manifold_card,
            self._index_card,
            self._filling_card,
            self._export_card,
        ]
        if 1 <= card_index <= len(cards):
            target = cards[card_index - 1]
            target.show()
            target.raise_()

