"""app/pipeline/export_card.py — Card ④: Export session results.

Unlocks as soon as session.has_any_results() is True (stage ≥ LOADED).

Always writes three files to the chosen output directory:
  {name}.tex   — full LaTeX report (all info, calculation steps, results)
  {name}.m     — Mathematica-readable data file (series data, etc.)
  {name}.json  — structured JSON of all session data

If a LaTeX toolchain (latexmk / pdflatex) is on PATH, a "Compile PDF"
button appears after export and produces {name}.pdf in the same folder.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from manifold_index.services.session import Session
from manifold_index.services.export_service import ExportService
from manifold_index.viewmodels.advisory import CardStatus
from manifold_index.app.widgets.collapsible_card import CollapsibleCard


_PDF_TIMEOUT_MS = 90_000  # 90 s hard cap on LaTeX compile


def _detect_latex_compiler() -> tuple[str, list[str]] | None:
    """Return (program, base_args) for the first available LaTeX compiler.

    Preference: latexmk > tectonic > pdflatex. Returns None if none found.
    """
    if shutil.which("latexmk"):
        return ("latexmk", ["-pdf", "-interaction=nonstopmode", "-halt-on-error"])
    if shutil.which("tectonic"):
        return ("tectonic", ["--keep-logs", "--synctex=none"])
    if shutil.which("pdflatex"):
        return ("pdflatex", ["-interaction=nonstopmode", "-halt-on-error"])
    return None


class ExportCard(QWidget):
    """Card ④: export session data to files.

    Signals
    -------
    session_updated(Session)
    """

    session_updated = Signal(object)

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._last_tex_path: Path | None = None
        self._pdf_process: QProcess | None = None
        self._pdf_timeout: QTimer | None = None

        self._card = CollapsibleCard(4, "Export", parent=self)
        self._card.set_status(CardStatus.LOCKED)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._card)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)

        info = QLabel(
            "Writes three files: <b>.tex</b> full report, <b>.m</b> "
            "Mathematica data, <b>.json</b> structured data."
        )
        info.setProperty("class", "muted")
        info.setWordWrap(True)
        bl.addWidget(info)

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

        self._pdf_btn = QPushButton("Compile PDF")
        self._pdf_btn.setProperty("class", "secondary")
        self._pdf_btn.clicked.connect(self._on_compile_pdf)
        self._pdf_btn.setEnabled(False)
        compiler = _detect_latex_compiler()
        if compiler is None:
            self._pdf_btn.setToolTip(
                "No LaTeX toolchain found on PATH. "
                "Install MacTeX / TeX Live or Tectonic to enable."
            )
        else:
            self._pdf_btn.setToolTip(
                f"Compile the exported .tex into a PDF using {compiler[0]}."
            )
        export_row.addWidget(self._pdf_btn)

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

    def lock(self) -> None:
        self._card.set_status(CardStatus.LOCKED)

    def refresh(self, session: Session) -> None:
        self._session = session
        if session.export_path:
            self._path_edit.setText(session.export_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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

        stem = s.manifold_name or "session"
        exported: list[str] = []

        try:
            tex_path = out_dir / f"{stem}.tex"
            ExportService.write_full_report(s, str(tex_path), include_filling=True)
            exported.append(tex_path.name)

            m_path = out_dir / f"{stem}.m"
            ExportService.write_mathematica(s, str(m_path), include_filling=True)
            exported.append(m_path.name)

            json_path = out_dir / f"{stem}.json"
            ExportService.write_json(s, str(json_path), include_filling=True)
            exported.append(json_path.name)

            names = ", ".join(exported)
            self._last_path_label.setText(f"Exported: {names} → {out_dir}")
            self._last_path_label.setVisible(True)
            self._card.set_status(CardStatus.DONE)
            self._card.set_summary(f"Exported {len(exported)} files → {out_dir.name}/")

            self._last_tex_path = tex_path
            self._pdf_btn.setEnabled(_detect_latex_compiler() is not None)

            self.session_updated.emit(s)

        except Exception as exc:
            self._last_path_label.setText(f"Export error: {exc}")
            self._last_path_label.setVisible(True)
            self._card.set_status(CardStatus.ERROR)

    # ------------------------------------------------------------------
    # PDF compilation
    # ------------------------------------------------------------------

    def _on_compile_pdf(self) -> None:
        if self._last_tex_path is None or not self._last_tex_path.exists():
            self._show_feedback("Export a .tex file first.", ms=3000)
            return
        if self._pdf_process is not None:
            return  # already running

        compiler = _detect_latex_compiler()
        if compiler is None:
            self._show_feedback(
                "No LaTeX toolchain found on PATH.", ms=4000
            )
            return

        program, base_args = compiler
        tex_path = self._last_tex_path
        args = base_args + [tex_path.name]

        self._pdf_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._feedback_label.setText(f"Compiling PDF with {program}…")
        self._feedback_label.setVisible(True)

        proc = QProcess(self)
        proc.setWorkingDirectory(str(tex_path.parent))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.finished.connect(self._on_pdf_finished)
        proc.errorOccurred.connect(self._on_pdf_error)
        self._pdf_process = proc

        self._pdf_timeout = QTimer(self)
        self._pdf_timeout.setSingleShot(True)
        self._pdf_timeout.timeout.connect(self._on_pdf_timeout)
        self._pdf_timeout.start(_PDF_TIMEOUT_MS)

        proc.start(program, args)

    def _on_pdf_finished(self, exit_code: int, _status) -> None:
        proc = self._pdf_process
        if proc is None:
            return
        if self._pdf_timeout is not None:
            self._pdf_timeout.stop()
            self._pdf_timeout = None

        output = bytes(proc.readAll()).decode("utf-8", errors="replace")
        self._pdf_process = None
        proc.deleteLater()

        self._export_btn.setEnabled(True)
        self._pdf_btn.setEnabled(True)

        if self._last_tex_path is None:
            return
        pdf_path = self._last_tex_path.with_suffix(".pdf")

        if exit_code == 0 and pdf_path.exists():
            self._last_path_label.setText(f"PDF compiled → {pdf_path}")
            self._last_path_label.setVisible(True)
            self._feedback_label.setVisible(False)
        else:
            tail = "\n".join(output.splitlines()[-8:]) or f"exit code {exit_code}"
            self._last_path_label.setText(f"PDF compile failed:\n{tail}")
            self._last_path_label.setVisible(True)
            self._feedback_label.setVisible(False)

    def _on_pdf_error(self, _err) -> None:
        if self._pdf_process is None:
            return
        err_str = self._pdf_process.errorString()
        if self._pdf_timeout is not None:
            self._pdf_timeout.stop()
            self._pdf_timeout = None
        self._pdf_process.deleteLater()
        self._pdf_process = None
        self._export_btn.setEnabled(True)
        self._pdf_btn.setEnabled(True)
        self._feedback_label.setVisible(False)
        self._last_path_label.setText(f"PDF compile error: {err_str}")
        self._last_path_label.setVisible(True)

    def _on_pdf_timeout(self) -> None:
        if self._pdf_process is None:
            return
        self._pdf_process.kill()
        self._pdf_process.waitForFinished(2000)
        self._pdf_process.deleteLater()
        self._pdf_process = None
        self._pdf_timeout = None
        self._export_btn.setEnabled(True)
        self._pdf_btn.setEnabled(True)
        self._feedback_label.setVisible(False)
        self._last_path_label.setText(
            f"PDF compile timed out after {_PDF_TIMEOUT_MS // 1000} s."
        )
        self._last_path_label.setVisible(True)

    def _show_feedback(self, text: str, ms: int = 1500) -> None:
        self._feedback_label.setText(text)
        self._feedback_label.setVisible(True)
        QTimer.singleShot(ms, lambda: self._feedback_label.setVisible(False))
