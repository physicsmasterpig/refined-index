"""app/window.py — MainWindow: top bar + QStackedWidget.

BLUEPRINT §13.9.

≤ 80 lines core logic.  Only responsibilities:
  1. Create QStackedWidget with PipelineView and DataHubView.
  2. Create top bar with mode-toggle buttons.
  3. Pass a shared Session to PipelineView.
  4. Apply stylesheet.
  5. Keyboard shortcuts: Cmd+1–4 (expand card), Cmd+Enter (compute).
  6. Session save/restore (Cmd+S / Cmd+O / Cmd+Shift+S).
  7. Manifold name autocomplete (SnaPy census list).
  8. closeEvent: stop all workers; wait up to 2 s each.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QStackedWidget, QStatusBar, QVBoxLayout, QWidget,
)

from manifold_index.services.session import Session
from manifold_index.app.pipeline.pipeline_view import PipelineView
from manifold_index.app.datahub.datahub_view import DataHubView
from manifold_index.app.theme.style import build_stylesheet

from manifold_index import __version__ as _VERSION
_APP_TITLE = f"Refined Index Calculator  v{_VERSION}"


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_APP_TITLE)
        self.resize(960, 760)
        self.setStyleSheet(build_stylesheet())

        self._session = Session()

        # ── Central widget ────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────────────
        top_bar = QWidget()
        top_bar.setProperty("class", "top-bar")
        top_bar.setFixedHeight(44)
        tb = QHBoxLayout(top_bar)
        tb.setContentsMargins(16, 0, 16, 0)
        tb.setSpacing(8)

        title_lbl = QLabel(_APP_TITLE)
        title_lbl.setProperty("class", "app-title")
        tb.addWidget(title_lbl)
        tb.addStretch(1)

        self._calc_btn = QPushButton("Calculator")
        self._calc_btn.setProperty("class", "mode-active")
        self._calc_btn.setCheckable(True)
        self._calc_btn.setChecked(True)
        self._calc_btn.clicked.connect(lambda: self._switch_mode(0))

        self._hub_btn = QPushButton("Data Hub")
        self._hub_btn.setProperty("class", "mode-inactive")
        self._hub_btn.setCheckable(True)
        self._hub_btn.clicked.connect(lambda: self._switch_mode(1))

        tb.addWidget(self._calc_btn)
        tb.addWidget(self._hub_btn)
        root.addWidget(top_bar)

        # ── Stack ─────────────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._pipeline_view = PipelineView(self._session, parent=self._stack)
        self._datahub_view  = DataHubView(parent=self._stack)
        self._stack.addWidget(self._pipeline_view)
        self._stack.addWidget(self._datahub_view)
        root.addWidget(self._stack, 1)

        # ── Status bar ────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._pipeline_view.session_changed.connect(self._on_session_changed)
        self._datahub_view.status_message.connect(
            lambda m: self._status_bar.showMessage(m, 5000)
        )

        # ── Keyboard shortcuts ────────────────────────────────────────
        self._setup_shortcuts()

        # ── Autocomplete (async — don't block startup) ────────────────
        self._setup_autocomplete()

        # ── Restore last session ──────────────────────────────────────
        self._try_restore_last_session()

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _switch_mode(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        self._calc_btn.setChecked(idx == 0)
        self._hub_btn.setChecked(idx == 1)
        self._calc_btn.setProperty("class", "mode-active" if idx == 0 else "mode-inactive")
        self._hub_btn.setProperty("class", "mode-active" if idx == 1 else "mode-inactive")
        for btn in (self._calc_btn, self._hub_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    def _on_session_changed(self, session: Session) -> None:
        self._session = session
        n = session.index_query_count()
        self._status_bar.showMessage(
            f"{session.manifold_name or '—'}  ·  stage: {session.stage.name}  ·  "
            f"{n} quer{'ies' if n != 1 else 'y'}",
            5000,
        )

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _setup_shortcuts(self) -> None:
        mod = "Ctrl"  # on macOS PySide6 maps Ctrl→Cmd in QKeySequence

        # Cmd+1–4: switch to / expand pipeline card
        for i in range(1, 5):
            sc = QShortcut(QKeySequence(f"{mod}+{i}"), self)
            sc.activated.connect(lambda idx=i: self._focus_card(idx))

        # Cmd+Enter: trigger compute in whichever card is active
        sc_enter = QShortcut(QKeySequence(f"{mod}+Return"), self)
        sc_enter.activated.connect(self._trigger_compute)

        # Cmd+S: save session
        sc_save = QShortcut(QKeySequence(f"{mod}+S"), self)
        sc_save.activated.connect(self._save_session)

        # Cmd+O: open / restore session
        sc_open = QShortcut(QKeySequence(f"{mod}+O"), self)
        sc_open.activated.connect(self._open_session)

    def _focus_card(self, card_index: int) -> None:
        """Switch to Calculator mode and expand the given card."""
        self._switch_mode(0)
        self._pipeline_view._on_step_clicked(card_index)  # type: ignore[attr-defined]

    def _trigger_compute(self) -> None:
        """Click the primary action button in the currently visible card."""
        self._switch_mode(0)
        view = self._pipeline_view
        # Try each card in order; fire the first enabled primary button found
        for card in (
            view._manifold_card,   # type: ignore[attr-defined]
            view._index_card,      # type: ignore[attr-defined]
            view._filling_card,    # type: ignore[attr-defined]
        ):
            for child in card.findChildren(QPushButton):
                if child.property("class") == "primary" and child.isEnabled():
                    child.click()
                    return

    # ------------------------------------------------------------------
    # Session save / restore
    # ------------------------------------------------------------------

    def _save_session(self) -> None:
        try:
            from manifold_index.services.session_store import save_session
            path = save_session(self._session)
            self._status_bar.showMessage(f"Session saved → {path.name}", 4000)
        except Exception as exc:
            self._status_bar.showMessage(f"Save failed: {exc}", 5000)

    def _open_session(self) -> None:
        from manifold_index.services.session_store import load_session, _SESSION_DIR
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", str(_SESSION_DIR), "JSON (*.json)"
        )
        if not path:
            return
        try:
            session = load_session(path)
            self._session = session
            self._pipeline_view.restore_session(session)
            self._switch_mode(0)
            self._status_bar.showMessage(
                f"Session restored: {session.manifold_name or '(empty)'}", 4000
            )
        except Exception as exc:
            self._status_bar.showMessage(f"Restore failed: {exc}", 5000)

    def _try_restore_last_session(self) -> None:
        """Silently restore the last saved session (ignore errors)."""
        try:
            from manifold_index.services.session_store import load_last_session
            session = load_last_session()
            if session and session.manifold_name:
                self._session = session
                self._pipeline_view.restore_session(session)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Autocomplete
    # ------------------------------------------------------------------

    def _setup_autocomplete(self) -> None:
        """Attach SnaPy census autocomplete to the manifold name field."""
        try:
            import snappy
            names: list[str] = []
            for attr in ("OrientableCuspedCensus", "OrientableClosedCensus",
                         "NonorientableCuspedCensus"):
                try:
                    col = getattr(snappy, attr, None)
                    if col is not None and hasattr(col, "keys"):
                        names.extend(col.keys())
                except Exception:
                    pass
        except Exception:
            return
        try:
            from PySide6.QtWidgets import QCompleter
            from PySide6.QtCore import QStringListModel
            view = self._pipeline_view
            # ManifoldCard stores the name QLineEdit as _name_edit
            name_edit = getattr(
                getattr(view, "_manifold_card", None), "_name_edit", None
            )
            if name_edit is None:
                return
            completer = QCompleter(names, self)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            name_edit.setCompleter(completer)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # closeEvent
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        # Auto-save on close if there is anything to save
        if self._session.manifold_name:
            try:
                from manifold_index.services.session_store import save_session
                save_session(self._session)
            except Exception:
                pass

        # Stop all running worker threads
        for worker in self.findChildren(QThread):
            if worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                worker.quit()
                worker.wait(2000)

        super().closeEvent(event)


