"""
pages/export_page.py — Page 4: Batch export to multiple formats.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from manifold_index.app.formatters import (
    build_plain_text,
    build_latex,
    build_json,
    build_nb_content,
    build_full_report_latex,
    results_dir,
)


class ExportPage(QWidget):
    """Page 4: batch export with format selection and options."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results_page = None   # set by MainWindow
        self._build_ui()

    def set_results_page(self, page) -> None:
        """Accept an OverviewPage (or any object with the required properties)."""
        self._results_page = page

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(20)
        root.setContentsMargins(40, 32, 40, 32)

        title = QLabel("Export")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        root.addWidget(title)

        subtitle = QLabel("Save computation results in one or more formats.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: palette(mid); font-size: 12px; margin-bottom: 8px;")
        root.addWidget(subtitle)

        # ── Format selection ──────────────────────────────────────
        fmt_group = QGroupBox("Formats")
        fmt_vbox = QVBoxLayout(fmt_group)
        fmt_vbox.setSpacing(8)

        self._chk_txt = QCheckBox("Plain text  (.txt)")
        self._chk_txt.setChecked(True)
        fmt_vbox.addWidget(self._chk_txt)

        self._chk_latex = QCheckBox("LaTeX  (.tex)")
        self._chk_latex.setChecked(True)
        fmt_vbox.addWidget(self._chk_latex)

        self._chk_json = QCheckBox("JSON  (.json)")
        self._chk_json.setChecked(True)
        fmt_vbox.addWidget(self._chk_json)

        self._chk_nb = QCheckBox("Mathematica Notebook  (.nb)")
        self._chk_nb.setChecked(True)
        fmt_vbox.addWidget(self._chk_nb)

        self._chk_report = QCheckBox("Full Report  (.tex) — SnaPy data, NZ matrix, edges, Weyl, indices")
        self._chk_report.setChecked(False)
        fmt_vbox.addWidget(self._chk_report)

        root.addWidget(fmt_group)

        # ── Options ───────────────────────────────────────────────
        opt_group = QGroupBox("Options")
        opt_vbox = QVBoxLayout(opt_group)
        opt_vbox.setSpacing(8)

        self._chk_weyl = QCheckBox("Include Weyl-manifest form (when valid)")
        self._chk_weyl.setChecked(True)
        opt_vbox.addWidget(self._chk_weyl)

        self._chk_header = QCheckBox("Include basis summary header")
        self._chk_header.setChecked(True)
        opt_vbox.addWidget(self._chk_header)

        root.addWidget(opt_group)

        # ── Output path ──────────────────────────────────────────
        path_group = QGroupBox("Output")
        path_form = QVBoxLayout(path_group)
        path_form.setSpacing(8)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Directory:"))
        self._dir_edit = QLineEdit()
        self._dir_edit.setText(str(results_dir()))
        self._dir_edit.setReadOnly(False)
        dir_row.addWidget(self._dir_edit, 1)
        browse_btn = QPushButton("Browse …")
        browse_btn.setObjectName("secondary")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._on_browse)
        dir_row.addWidget(browse_btn)
        path_form.addLayout(dir_row)

        prefix_row = QHBoxLayout()
        prefix_row.addWidget(QLabel("Filename prefix:"))
        self._prefix_edit = QLineEdit()
        self._prefix_edit.setPlaceholderText("e.g. m003_index")
        prefix_row.addWidget(self._prefix_edit, 1)
        path_form.addLayout(prefix_row)

        root.addWidget(path_group)

        # ── Action buttons ────────────────────────────────────────
        btn_row = QHBoxLayout()

        copy_btn = QPushButton("⎘  Copy Series to Clipboard")
        copy_btn.setObjectName("secondary")
        copy_btn.setFixedHeight(36)
        copy_btn.clicked.connect(self._on_copy_clipboard)
        btn_row.addWidget(copy_btn)

        btn_row.addStretch()

        self._export_btn = QPushButton("Export All Selected  ▶")
        self._export_btn.setObjectName("primary")
        self._export_btn.setFixedHeight(40)
        self._export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self._export_btn)

        root.addLayout(btn_row)

        # ── Status ────────────────────────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 11px;")
        root.addWidget(self._status_label)

        root.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, manifold_name: str) -> None:
        """Update the prefix when we navigate to this page."""
        self._prefix_edit.setText(f"{manifold_name}_index")
        self._status_label.setText("")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_data(self):
        """Pull everything needed for export from the overview page."""
        rp = self._results_page
        if rp is None:
            return None
        if rp.refined_results is None:
            return None
        nz = rp.nz_data
        if nz is None:
            return None
        return {
            "entries": rp.refined_results,
            "name": self._prefix_edit.text().replace("_index", "") or "unknown",
            "q_ord": getattr(nz, "_q_order_half", 10),
            "num_hard": nz.num_hard,
            "basis_summary": "",
            "weyl_result": rp.weyl_result if self._chk_weyl.isChecked() else None,
        }

    def _get_report_data(self):
        """Pull extended data needed for the full report."""
        rp = self._results_page
        if rp is None:
            return None
        if rp.refined_results is None:
            return None
        nz = rp.nz_data
        if nz is None:
            return None
        return {
            "entries": rp.refined_results,
            "name": self._prefix_edit.text().replace("_index", "") or "unknown",
            "q_ord": getattr(nz, "_q_order_half", 10),
            "num_hard": nz.num_hard,
            "basis_summary": "",
            "weyl_result": rp.weyl_result,
            "pipeline_result": None,
            "basis_selection": None,
            "nz_changed": nz,
            "filled_refined_result": None,
        }

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self._dir_edit.text()
        )
        if d:
            self._dir_edit.setText(d)

    def _on_copy_clipboard(self) -> None:
        data = self._get_data()
        if data is None:
            QMessageBox.warning(self, "No data", "No results to copy yet.")
            return
        txt = build_plain_text(**data)
        QApplication.clipboard().setText(txt)
        self._status_label.setText("✓  Copied to clipboard.")
        self._status_label.setStyleSheet("color: #2ea043; font-size: 11px;")

    @Slot()
    def _on_export(self) -> None:
        data = self._get_data()
        if data is None:
            QMessageBox.warning(self, "No data", "No results to export yet.")
            return

        out_dir = Path(self._dir_edit.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._prefix_edit.text().strip() or "refined_index"

        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        exported: list[str] = []

        try:
            if self._chk_txt.isChecked():
                content = build_plain_text(**data)
                p = out_dir / f"{prefix}.txt"
                p.write_text(content, encoding="utf-8")
                exported.append(p.name)

            if self._chk_latex.isChecked():
                content = build_latex(**data)
                p = out_dir / f"{prefix}.tex"
                p.write_text(content, encoding="utf-8")
                exported.append(p.name)

            if self._chk_json.isChecked():
                content = build_json(**data)
                p = out_dir / f"{prefix}.json"
                p.write_text(content, encoding="utf-8")
                exported.append(p.name)

            if self._chk_nb.isChecked():
                content = build_nb_content(
                    data["entries"], data["name"], data["q_ord"],
                    data["num_hard"], data["basis_summary"], ts,
                )
                p = out_dir / f"{prefix}.nb"
                p.write_text(content, encoding="utf-8")
                exported.append(p.name)

            if self._chk_report.isChecked():
                report_data = self._get_report_data()
                if report_data is not None:
                    content = build_full_report_latex(**report_data)
                    p = out_dir / f"{prefix}_report.tex"
                    p.write_text(content, encoding="utf-8")
                    exported.append(p.name)

            if exported:
                self._status_label.setText(
                    f"✓  Exported {len(exported)} file(s) to {out_dir}:  "
                    + ", ".join(exported)
                )
                self._status_label.setStyleSheet("color: #2ea043; font-size: 11px;")
            else:
                self._status_label.setText("⚠  No formats selected.")
                self._status_label.setStyleSheet("color: #d4880a; font-size: 11px;")
        except Exception as exc:
            self._status_label.setText(f"✗  Export failed: {exc}")
            self._status_label.setStyleSheet("color: #d1242f; font-size: 11px;")
