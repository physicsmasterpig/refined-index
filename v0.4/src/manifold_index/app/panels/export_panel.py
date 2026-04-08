"""
app/panels/export_panel.py — Panel 3: Export.

Save results in various formats: LaTeX, Mathematica, plain text, JSON.
Receives data from Panel 1 (refined index) and Panel 2 (Dehn filling).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionTitle")
    return lbl


class ExportPanel(QFrame):
    """Panel 3: export results to various formats."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._data: dict | None = None
        self._dehn_results: list | None = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)

        t = QLabel("③ Export")
        t.setObjectName("panelTitle")
        outer.addWidget(t)

        sub = QLabel("Save results to various formats.")
        sub.setObjectName("panelSubtitle")
        sub.setWordWrap(True)
        outer.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep)

        # ── Format section ──
        outer.addWidget(_section_label("Formats"))

        self._chk_tex = QCheckBox("LaTeX  (.tex)")
        self._chk_tex.setChecked(True)
        outer.addWidget(self._chk_tex)

        self._chk_report = QCheckBox("Full Report  (.tex)")
        self._chk_report.setChecked(False)
        outer.addWidget(self._chk_report)

        self._chk_nb = QCheckBox("Mathematica  (.m)")
        self._chk_nb.setChecked(True)
        outer.addWidget(self._chk_nb)

        self._chk_txt = QCheckBox("Plain text  (.txt)")
        self._chk_txt.setChecked(False)
        outer.addWidget(self._chk_txt)

        self._chk_json = QCheckBox("JSON  (.json)")
        self._chk_json.setChecked(False)
        outer.addWidget(self._chk_json)

        outer.addSpacing(8)

        # ── Options ──
        outer.addWidget(_section_label("Options"))

        self._chk_dehn = QCheckBox("Include Dehn filling results")
        self._chk_dehn.setChecked(True)
        outer.addWidget(self._chk_dehn)

        outer.addSpacing(8)

        # ── Output path ──
        outer.addWidget(_section_label("Output"))

        dir_row = QWidget()
        dir_h = QHBoxLayout(dir_row)
        dir_h.setContentsMargins(0, 0, 0, 0)
        dir_h.setSpacing(4)
        self._dir_edit = QLineEdit()
        self._dir_edit.setText("~/results/")
        self._dir_edit.setFixedHeight(28)
        dir_h.addWidget(self._dir_edit, 1)
        browse = QPushButton("…")
        browse.setObjectName("secondary")
        browse.setFixedWidth(28)
        browse.setFixedHeight(28)
        browse.clicked.connect(self._browse_dir)
        dir_h.addWidget(browse)
        outer.addWidget(dir_row)

        prefix_row = QWidget()
        prefix_h = QHBoxLayout(prefix_row)
        prefix_h.setContentsMargins(0, 0, 0, 0)
        prefix_h.setSpacing(4)
        prefix_h.addWidget(QLabel("Prefix:"))
        self._prefix_edit = QLineEdit()
        self._prefix_edit.setText("index")
        self._prefix_edit.setFixedHeight(28)
        prefix_h.addWidget(self._prefix_edit, 1)
        outer.addWidget(prefix_row)

        outer.addSpacing(12)

        # ── Export buttons ──
        self._export_btn = QPushButton("Export All  ▶")
        self._export_btn.setObjectName("primary")
        self._export_btn.setFixedHeight(36)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        outer.addWidget(self._export_btn)

        outer.addSpacing(4)

        self._copy_tex_btn = QPushButton("⎘  Copy LaTeX to Clipboard")
        self._copy_tex_btn.setObjectName("secondary")
        self._copy_tex_btn.setFixedHeight(30)
        self._copy_tex_btn.setEnabled(False)
        self._copy_tex_btn.clicked.connect(self._copy_latex)
        outer.addWidget(self._copy_tex_btn)

        self._copy_txt_btn = QPushButton("⎘  Copy Plain Text")
        self._copy_txt_btn.setObjectName("secondary")
        self._copy_txt_btn.setFixedHeight(30)
        self._copy_txt_btn.setEnabled(False)
        self._copy_txt_btn.clicked.connect(self._copy_text)
        outer.addWidget(self._copy_txt_btn)

        outer.addSpacing(12)

        # ── Data Pack Export section ──
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep2)

        outer.addWidget(_section_label("Data Pack Export"))

        dp_sub = QLabel(
            "Export the I<sup>ref</sup> cache for the current manifold "
            "to Mathematica or JSON."
        )
        dp_sub.setWordWrap(True)
        dp_sub.setStyleSheet("font-size: 11px; color: #8b949e;")
        outer.addWidget(dp_sub)

        outer.addSpacing(4)

        self._export_iref_m_btn = QPushButton("📦  Export Iʳᵉᶠ  →  .m.zip")
        self._export_iref_m_btn.setObjectName("secondary")
        self._export_iref_m_btn.setFixedHeight(30)
        self._export_iref_m_btn.setEnabled(False)
        self._export_iref_m_btn.clicked.connect(self._export_iref_mathematica)
        outer.addWidget(self._export_iref_m_btn)

        self._export_iref_json_btn = QPushButton("📦  Export Iʳᵉᶠ  →  .json.zip")
        self._export_iref_json_btn.setObjectName("secondary")
        self._export_iref_json_btn.setFixedHeight(30)
        self._export_iref_json_btn.setEnabled(False)
        self._export_iref_json_btn.clicked.connect(self._export_iref_json)
        outer.addWidget(self._export_iref_json_btn)

        self._export_kernel_btn = QPushButton("📦  Export Kernels  →  .zip")
        self._export_kernel_btn.setObjectName("secondary")
        self._export_kernel_btn.setFixedHeight(30)
        self._export_kernel_btn.setEnabled(True)  # kernels don't need manifold data
        self._export_kernel_btn.clicked.connect(self._export_kernels)
        outer.addWidget(self._export_kernel_btn)

        outer.addSpacing(8)

        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 11px;")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        outer.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, data: dict) -> None:
        """Store computation results from Panel 1 for export.

        *data* should contain:
            manifold_data, easy_result, nz_data, entries, weyl_result, q_order_half
        """
        self._data = data
        self._dehn_results = None  # reset when new manifold loaded
        name = data.get("manifold_data")
        if name:
            self._prefix_edit.setText(f"{name.name}_index")
        self._export_btn.setEnabled(True)
        self._copy_tex_btn.setEnabled(True)
        self._copy_txt_btn.setEnabled(True)
        self._export_iref_m_btn.setEnabled(True)
        self._export_iref_json_btn.setEnabled(True)

    def set_dehn_data(self, results: list) -> None:
        """Store Dehn filling results from Panel 2 for export.

        *results* is a list of ``TransformedFillResult`` or
        ``MultiCuspFillResult`` objects.
        """
        self._dehn_results = results
        self._status.setText("✓  Dehn filling data received for export")
        self._status.setStyleSheet("color: #539bf5; font-size: 11px;")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _manifold_name(self) -> str:
        md = self._data.get("manifold_data") if self._data else None
        return md.name if md else "unknown"

    def _include_dehn(self) -> bool:
        return self._chk_dehn.isChecked() and self._dehn_results is not None

    def _dehn_data(self) -> list | None:
        return self._dehn_results if self._include_dehn() else None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot()
    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self._dir_edit.setText(d)

    @Slot()
    def _on_export(self) -> None:
        if not self._data:
            return

        from manifold_index.utils.exporters import (
            write_full_report,
            write_json,
            write_latex,
            write_mathematica,
            write_plain_text,
        )

        out_dir = Path(self._dir_edit.text()).expanduser()
        prefix = self._prefix_edit.text().strip() or "index"

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Export error", f"Cannot create directory:\n{exc}")
            return

        name = self._manifold_name()
        md = self._data.get("manifold_data")
        nz = self._data["nz_data"]
        entries = self._data.get("entries", [])
        weyl = self._data.get("weyl_result")
        easy = self._data.get("easy_result")
        q_order_half = self._data.get("q_order_half")
        dehn = self._dehn_data()
        inc_dehn = self._include_dehn()

        exported: list[str] = []

        try:
            if self._chk_tex.isChecked():
                p = out_dir / f"{prefix}.tex"
                write_latex(p, name, nz, entries, weyl, dehn, inc_dehn)
                exported.append(p.name)

            if self._chk_report.isChecked():
                p = out_dir / f"{prefix}_report.tex"
                write_full_report(p, md, easy, nz, entries, weyl,
                                  dehn if inc_dehn else None, q_order_half)
                exported.append(p.name)

            if self._chk_nb.isChecked():
                p = out_dir / f"{prefix}.m"
                write_mathematica(p, md, nz, entries, weyl, dehn if inc_dehn else None, q_order_half)
                exported.append(p.name)

            if self._chk_txt.isChecked():
                p = out_dir / f"{prefix}.txt"
                write_plain_text(p, name, nz, entries, weyl, dehn, inc_dehn)
                exported.append(p.name)

            if self._chk_json.isChecked():
                p = out_dir / f"{prefix}.json"
                write_json(p, md, easy, nz, entries, weyl, dehn if inc_dehn else None, q_order_half)
                exported.append(p.name)

        except Exception as exc:
            QMessageBox.critical(self, "Export error", f"Failed:\n{exc}")
            return

        if exported:
            self._status.setText(f"✓  Exported: {', '.join(exported)}")
            self._status.setStyleSheet("color: #2ea043; font-size: 11px;")
        else:
            self._status.setText("No formats selected.")
            self._status.setStyleSheet("color: #d4880a; font-size: 11px;")

    @Slot()
    def _copy_latex(self) -> None:
        if not self._data:
            return
        from manifold_index.utils.exporters import clipboard_latex

        nz = self._data["nz_data"]
        entries = self._data.get("entries", [])
        name = self._manifold_name()
        md = self._data.get("manifold_data")
        dehn = self._dehn_data()

        text = clipboard_latex(name, entries, nz.num_hard, dehn, self._include_dehn())
        QApplication.clipboard().setText(text)
        self._status.setText("✓  LaTeX copied to clipboard")
        self._status.setStyleSheet("color: #2ea043; font-size: 11px;")

    @Slot()
    def _copy_text(self) -> None:
        if not self._data:
            return
        from manifold_index.utils.exporters import clipboard_plain_text

        nz = self._data["nz_data"]
        entries = self._data.get("entries", [])
        name = self._manifold_name()
        md = self._data.get("manifold_data")
        dehn = self._dehn_data()

        text = clipboard_plain_text(name, entries, nz.num_hard, dehn, self._include_dehn())
        QApplication.clipboard().setText(text)
        self._status.setText("✓  Plain text copied to clipboard")
        self._status.setStyleSheet("color: #2ea043; font-size: 11px;")

    # ------------------------------------------------------------------
    # Data-pack export slots
    # ------------------------------------------------------------------

    def _find_iref_cache_path(self) -> Path | None:
        """Locate the iref cache .pkl.gz for the current manifold."""
        from manifold_index.utils.cache_export import find_iref_file

        name = self._manifold_name()
        if name == "unknown":
            return None
        return find_iref_file(name)

    @Slot()
    def _export_iref_mathematica(self) -> None:
        if not self._data:
            return
        from manifold_index.utils.cache_export import (
            export_iref_zip,
            load_iref_file,
        )

        pkl = self._find_iref_cache_path()
        if pkl is None:
            self._status.setText("✗  No I^ref cache found for this manifold")
            self._status.setStyleSheet("color: #f85149; font-size: 11px;")
            return

        name = self._manifold_name()
        default_name = f"iref_{name.replace('/', '_')}.zip"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save I^ref Mathematica archive",
            str(Path.home() / "Desktop" / default_name),
            "Zip Archive (*.zip);;All Files (*)",
        )
        if not path:
            return

        try:
            data = load_iref_file(pkl)
            n = export_iref_zip(data, Path(path), fmt="mathematica")
            self._status.setText(
                f"✓  Exported {n} entries → {Path(path).name}"
            )
            self._status.setStyleSheet("color: #2ea043; font-size: 11px;")
        except Exception as exc:
            self._status.setText(f"✗  Export failed: {exc}")
            self._status.setStyleSheet("color: #f85149; font-size: 11px;")

    @Slot()
    def _export_iref_json(self) -> None:
        if not self._data:
            return
        from manifold_index.utils.cache_export import (
            export_iref_zip,
            load_iref_file,
        )

        pkl = self._find_iref_cache_path()
        if pkl is None:
            self._status.setText("✗  No I^ref cache found for this manifold")
            self._status.setStyleSheet("color: #f85149; font-size: 11px;")
            return

        name = self._manifold_name()
        default_name = f"iref_{name.replace('/', '_')}_json.zip"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save I^ref JSON archive",
            str(Path.home() / "Desktop" / default_name),
            "Zip Archive (*.zip);;All Files (*)",
        )
        if not path:
            return

        try:
            data = load_iref_file(pkl)
            n = export_iref_zip(data, Path(path), fmt="json")
            self._status.setText(
                f"✓  Exported {n} entries → {Path(path).name}"
            )
            self._status.setStyleSheet("color: #2ea043; font-size: 11px;")
        except Exception as exc:
            self._status.setText(f"✗  Export failed: {exc}")
            self._status.setStyleSheet("color: #f85149; font-size: 11px;")

    @Slot()
    def _export_kernels(self) -> None:
        from manifold_index.utils.cache_export import (
            export_kernels_zip,
            list_kernel_files,
        )

        files = list_kernel_files()
        if not files:
            self._status.setText("✗  No kernel cache files found")
            self._status.setStyleSheet("color: #f85149; font-size: 11px;")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save kernel archive",
            str(Path.home() / "Desktop" / "kernels.zip"),
            "Zip Archive (*.zip);;All Files (*)",
        )
        if not path:
            return

        try:
            exported, errors = export_kernels_zip(files, Path(path))
            msg = f"✓  Exported {exported} kernel(s) → {Path(path).name}"
            if errors:
                msg += f"  ({errors} failed)"
            self._status.setText(msg)
            self._status.setStyleSheet("color: #2ea043; font-size: 11px;")
        except Exception as exc:
            self._status.setText(f"✗  Export failed: {exc}")
            self._status.setStyleSheet("color: #f85149; font-size: 11px;")
