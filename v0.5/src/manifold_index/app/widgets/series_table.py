"""app/widgets/series_table.py — q-series result table with hover row actions.

See BLUEPRINT §9.5 and §2.8.

Columns::

    m  |  e  |  Series (q^½ terms)  |  Source

Row hover actions appear on the right as tertiary buttons:
  [Copy LaTeX]  [Remove]

The table is otherwise uncluttered (BLUEPRINT §2.8).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QWidget,
)

from manifold_index.app.theme import colors as C


_COL_M      = 0
_COL_E      = 1
_COL_SERIES = 2
_COL_SOURCE = 3
_COL_ACTIONS = 4   # hidden column used by overlay trick; we use a separate widget

_HEADERS = ["m", "e", "Series", "Source", ""]  # last col for action widgets


class SeriesTable(QTableWidget):
    """QTableWidget showing accumulated query results.

    Signals
    -------
    row_removed(int)            — row index that was removed.
    copy_latex_requested(int)   — row index whose LaTeX was requested.
    """

    row_removed          = Signal(int)
    copy_latex_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 5, parent)

        self.setHorizontalHeaderLabels(_HEADERS)
        self.horizontalHeader().setSectionResizeMode(
            _COL_SERIES, QHeaderView.ResizeMode.Stretch
        )
        self.horizontalHeader().setSectionResizeMode(
            _COL_SOURCE, QHeaderView.ResizeMode.ResizeToContents
        )
        self.horizontalHeader().setSectionResizeMode(
            _COL_M, QHeaderView.ResizeMode.ResizeToContents
        )
        self.horizontalHeader().setSectionResizeMode(
            _COL_E, QHeaderView.ResizeMode.ResizeToContents
        )
        # Actions column: just wide enough for two small buttons
        self.horizontalHeader().setSectionResizeMode(
            _COL_ACTIONS, QHeaderView.ResizeMode.Fixed
        )
        self.setColumnWidth(_COL_ACTIONS, 160)

        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Hide the empty header of the actions column
        self.horizontalHeader().setSectionResizeMode(
            _COL_ACTIONS, QHeaderView.ResizeMode.Fixed
        )
        self.setColumnWidth(_COL_ACTIONS, 168)

        # Track which row the mouse is hovering over for action visibility
        self._hover_row: int = -1
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().installEventFilter(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_row(
        self,
        m: int | str,
        e: int | str,
        series_latex: str,
        source: str,
    ) -> int:
        """Append a row and return its row index."""
        row = self.rowCount()
        self.insertRow(row)
        self._set_row_data(row, m, e, series_latex, source)
        self._install_row_actions(row)
        return row

    def set_row_computing(self, row: int) -> None:
        """Mark a row as currently computing (shows placeholder)."""
        item = self.item(row, _COL_SERIES)
        if item:
            item.setText("[computing…]")
            item.setForeground(__import__("PySide6.QtGui", fromlist=["QColor"]).QColor(C.TEXT_MUTED))
        src = self.item(row, _COL_SOURCE)
        if src:
            src.setText("—")

    def set_row_result(self, row: int, series_latex: str, source: str) -> None:
        """Update a row with the computed result."""
        item = self.item(row, _COL_SERIES)
        if item:
            from PySide6.QtGui import QColor
            item.setText(series_latex)
            item.setForeground(QColor(C.TEXT_PRIMARY))
        src = self.item(row, _COL_SOURCE)
        if src:
            src.setText(source)

    def clear_rows(self) -> None:
        """Remove all rows."""
        self.setRowCount(0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_row_data(
        self,
        row: int,
        m: int | str,
        e: int | str,
        series_latex: str,
        source: str,
    ) -> None:
        self.setItem(row, _COL_M,      QTableWidgetItem(str(m)))
        self.setItem(row, _COL_E,      QTableWidgetItem(str(e)))
        self.setItem(row, _COL_SERIES, QTableWidgetItem(series_latex))
        self.setItem(row, _COL_SOURCE, QTableWidgetItem(source))
        # Centre-align m and e columns
        for col in (_COL_M, _COL_E):
            item = self.item(row, col)
            if item:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                )

    def _install_row_actions(self, row: int) -> None:
        """Install hover action widget for *row*."""
        action_widget = _RowActionWidget(row, self)
        action_widget.copy_latex_clicked.connect(self.copy_latex_requested)
        action_widget.remove_clicked.connect(self._remove_row)
        action_widget.setVisible(False)
        self.setCellWidget(row, _COL_ACTIONS, action_widget)

    def _remove_row(self, row: int) -> None:
        self.row_removed.emit(row)
        self.removeRow(row)
        # Re-index action widgets below removed row
        for r in range(row, self.rowCount()):
            w = self.cellWidget(r, _COL_ACTIONS)
            if isinstance(w, _RowActionWidget):
                w.update_row(r)

    def _show_actions_for_row(self, row: int) -> None:
        if self._hover_row == row:
            return
        # Hide previous
        if self._hover_row >= 0:
            w = self.cellWidget(self._hover_row, _COL_ACTIONS)
            if w:
                w.setVisible(False)
        self._hover_row = row
        w = self.cellWidget(row, _COL_ACTIONS)
        if w:
            w.setVisible(True)

    def _hide_actions(self) -> None:
        if self._hover_row >= 0:
            w = self.cellWidget(self._hover_row, _COL_ACTIONS)
            if w:
                w.setVisible(False)
        self._hover_row = -1

    # ------------------------------------------------------------------
    # Event filter — track hover row on viewport
    # ------------------------------------------------------------------

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        if obj is self.viewport():
            if event.type() == QEvent.Type.MouseMove:
                pos = event.pos()  # type: ignore[attr-defined]
                row = self.rowAt(pos.y())
                if row >= 0:
                    self._show_actions_for_row(row)
                else:
                    self._hide_actions()
            elif event.type() in (QEvent.Type.Leave, QEvent.Type.MouseButtonRelease):
                self._hide_actions()
        return super().eventFilter(obj, event)


class _RowActionWidget(QWidget):
    """Widget placed in the actions column of a row."""

    copy_latex_clicked = Signal(int)
    remove_clicked     = Signal(int)

    def __init__(self, row: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._row = row
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._copy_btn = QPushButton("Copy LaTeX")
        self._copy_btn.setProperty("class", "tertiary")
        self._copy_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._copy_btn.clicked.connect(lambda: self.copy_latex_clicked.emit(self._row))

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setProperty("class", "tertiary")
        self._remove_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self._row))

        layout.addWidget(self._copy_btn)
        layout.addWidget(self._remove_btn)

    def update_row(self, row: int) -> None:
        self._row = row


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

    app = QApplication(sys.argv)
    from manifold_index.app.theme.style import build_stylesheet
    app.setStyleSheet(build_stylesheet())

    win = QWidget()
    win.setWindowTitle("SeriesTable smoke test")
    layout = QVBoxLayout(win)
    layout.setContentsMargins(16, 16, 16, 16)

    tbl = SeriesTable()
    tbl.add_row(0, 0, r"1 - q^{1/2} + q - q^{3/2} + \cdots", "computed")
    tbl.add_row(1, 0, r"q^{1/2} - 2q + 3q^{3/2} - \cdots", "computed")
    row = tbl.add_row(0, "½", "", "—")
    tbl.set_row_computing(row)

    tbl.copy_latex_requested.connect(lambda r: print(f"Copy LaTeX row {r}"))
    tbl.row_removed.connect(lambda r: print(f"Removed row {r}"))

    layout.addWidget(tbl)
    win.resize(800, 300)
    win.show()
    sys.exit(app.exec())
