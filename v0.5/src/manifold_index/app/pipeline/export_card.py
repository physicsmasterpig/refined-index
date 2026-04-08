"""app/pipeline/export_card.py — Card ④: Export session results.

BLUEPRINT §10.5.

Unlocks as soon as session.has_any_results() is True (stage ≥ LOADED).

Sections
--------
  A  Data checklist  — manifold, index, Weyl, NC cycles, filling
  B  Format selection — LaTeX / Mathematica / Full Report / JSON
  C  Output path + Export button
  D  "Copy LaTeX" / "Copy Mathematica" tertiary buttons (1.5 s "Copied." feedback)
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QVBoxLayout, QWidget,
)

from manifold_index.services.session import Session
from manifold_index.services.export_service import ExportService
from manifold_index.viewmodels.advisory import CardStatus
from manifold_index.viewmodels.export_vm import (
    ExportFormatSelection, build_export_vm,
)
from manifold_index.app.widgets.collapsible_card import CollapsibleCard


class ExportCard(QWidget):
    """Card ④: export session data to files or clipboard.

    Signals
    -------
    session_updated(Session)
    """

    session_updated = Signal(object)

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session

        self._card = CollapsibleCard(4, "Export", parent=self)
        self._card.set_status(CardStatus.LOCKED)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._card)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)

        # ── A: Data checklist ─────────────────────────────────────────
        data_box = QGroupBox("Include")
        data_layout = QVBoxLayout(data_box)
        self._chk_manifold = QCheckBox("Manifold (NZ data, gluing equations)")
        self._chk_index    = QCheckBox("Index queries (I^ref)")
        self._chk_weyl     = QCheckBox("Weyl check (a, b vectors)")
        self._chk_nc       = QCheckBox("NC cycles")
        self._chk_filling  = QCheckBox("Filled index queries")
        for chk in (
            self._chk_manifold, self._chk_index,
            self._chk_weyl, self._chk_nc, self._chk_filling,
        ):
            chk.setChecked(True)
            data_layout.addWidget(chk)
        bl.addWidget(data_box)

        # ── B: Format selection ───────────────────────────────────────
        fmt_box = QGroupBox("Format")
        fmt_layout = QHBoxLayout(fmt_box)
        self._fmt_latex  = QCheckBox("LaTeX")
        self._fmt_math   = QCheckBox("Mathematica")
        self._fmt_report = QCheckBox("Full Report")
        self._fmt_json   = QCheckBox("JSON")
        self._fmt_latex.setChecked(True)
        self._fmt_math.setChecked(True)
        for chk in (self._fmt_latex, self._fmt_math, self._fmt_report, self._fmt_json):
            fmt_layout.addWidget(chk)
        fmt_layout.addStretch(1)
        bl.addWidget(fmt_box)

        # ── C: Output path ────────────────────────────────────────────
        path_box = QGroupBox("Output")
        path_layout = QVBoxLayout(path_box)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Output directory…")
        self._path_edit.setText(session.export_path or "")
        path_row.addWidget(self._path_edit)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setProperty("class", "secondary")
        self._browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(self._browse_btn)
        path_layout.addLayout(path_row)

        export_row = QHBoxLayout()
        self._export_btn = QPushButton("Export")
        self._export_btn.setProperty("class", "primary")
        self._export_btn.clicked.connect(self._on_export)
        export_row.addWidget(self._export_btn)

        # Copy buttons
        self._copy_latex_btn = QPushButton("Copy LaTeX")
        self._copy_latex_btn.setProperty("class", "tertiary")
        self._copy_latex_btn.clicked.connect(self._on_copy_latex)
        export_row.addWidget(self._copy_latex_btn)

        self._copy_math_btn = QPushButton("Copy Mathematica")
        self._copy_math_btn.setProperty("class", "tertiary")
        self._copy_math_btn.clicked.connect(self._on_copy_math)
        export_row.addWidget(self._copy_math_btn)

        export_row.addStretch(1)
        path_layout.addLayout(export_row)

        self._feedback_label = QLabel()
        self._feedback_label.setProperty("class", "muted")
        self._feedback_label.setVisible(False)
        path_layout.addWidget(self._feedback_label)

        self._last_path_label = QLabel()
        self._last_path_label.setProperty("class", "muted")
        self._last_path_label.setVisible(False)
        path_layout.addWidget(self._last_path_label)

        bl.addWidget(path_box)

        self._card.set_body(body)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def unlock(self, session: Session) -> None:
        self._session = session
        self._card.set_status(CardStatus.READY)
        self._card.expand()
        self._refresh_data_checkboxes()

    def lock(self) -> None:
        self._card.set_status(CardStatus.LOCKED)

    def refresh(self, session: Session) -> None:
        self._session = session
        self._refresh_data_checkboxes()
        if session.export_path:
            self._path_edit.setText(session.export_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_data_checkboxes(self) -> None:
        s = self._session
        available = ExportService.available_data(s)
        self._chk_manifold.setEnabled(bool(available.get("manifold", False)))
        self._chk_manifold.setChecked(bool(available.get("manifold", False)))
        n_iq = int(available.get("index_queries", 0))
        self._chk_index.setEnabled(n_iq > 0)
        self._chk_index.setChecked(n_iq > 0)
        self._chk_weyl.setEnabled(bool(available.get("weyl", False)))
        self._chk_weyl.setChecked(bool(available.get("weyl", False)))
        n_nc = int(available.get("nc_cycles", 0))
        self._chk_nc.setEnabled(n_nc > 0)
        self._chk_nc.setChecked(n_nc > 0)
        n_fq = int(available.get("fill_queries", 0))
        self._chk_filling.setEnabled(n_fq > 0)
        self._chk_filling.setChecked(n_fq > 0)

    def _on_browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose output directory",
            self._path_edit.text() or str(Path.home()),
        )
        if directory:
            self._path_edit.setText(directory)

    def _output_dir(self) -> Path:
        txt = self._path_edit.text().strip()
        return Path(txt) if txt else Path.home() / "manifold_export"

    def _on_export(self) -> None:
        s = self._session
        out_dir = self._output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        s.export_path = str(out_dir)

        include_filling = self._chk_filling.isChecked()
        exported: list[str] = []

        try:
            if self._fmt_latex.isChecked():
                out_path = out_dir / f"{s.manifold_name or 'session'}.tex"
                ExportService.write_latex(s, str(out_path), include_filling)
                exported.append(out_path.name)

            if self._fmt_math.isChecked():
                out_path = out_dir / f"{s.manifold_name or 'session'}.m"
                ExportService.write_mathematica(s, str(out_path), include_filling)
                exported.append(out_path.name)

            if self._fmt_report.isChecked():
                out_path = out_dir / f"{s.manifold_name or 'session'}_report.tex"
                ExportService.write_full_report(s, str(out_path))
                exported.append(out_path.name)

            if self._fmt_json.isChecked():
                out_path = out_dir / f"{s.manifold_name or 'session'}.json"
                ExportService.write_json(s, str(out_path))
                exported.append(out_path.name)

            if exported:
                names = ", ".join(exported)
                self._last_path_label.setText(f"Exported: {names} → {out_dir}")
                self._last_path_label.setVisible(True)
                self._card.set_status(CardStatus.DONE)
                self._card.set_summary(f"Exported {len(exported)} file(s) → {out_dir.name}/")
                self.session_updated.emit(s)

        except Exception as exc:
            self._last_path_label.setText(f"Export error: {exc}")
            self._last_path_label.setVisible(True)
            self._card.set_status(CardStatus.ERROR)

    def _on_copy_latex(self) -> None:
        self._copy_to_clipboard("latex")

    def _on_copy_math(self) -> None:
        self._copy_to_clipboard("mathematica")

    def _copy_to_clipboard(self, fmt: str) -> None:
        import tempfile
        s = self._session
        tmp: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".tex" if fmt == "latex" else ".m",
                delete=False, mode="w"
            ) as f:
                tmp = f.name
            if fmt == "latex":
                ExportService.write_latex(s, tmp, True)
            else:
                ExportService.write_mathematica(s, tmp, True)
            text = Path(tmp).read_text(encoding="utf-8")

            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(text)
            self._show_copied_feedback()
        except Exception as exc:
            self._show_feedback(f"Copy failed: {exc}", ms=3000)
        finally:
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)

    def _show_copied_feedback(self) -> None:
        self._show_feedback("Copied.", ms=1500)

    def _show_feedback(self, text: str, ms: int = 1500) -> None:
        self._feedback_label.setText(text)
        self._feedback_label.setVisible(True)
        QTimer.singleShot(ms, lambda: self._feedback_label.setVisible(False))

