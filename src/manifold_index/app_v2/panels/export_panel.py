"""
app_v2/panels/export_panel.py — Panel 3: Export.

Save results in various formats: LaTeX, Mathematica, plain text, JSON.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
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

        # ── Format section ────────────────────────────────────
        outer.addWidget(_section_label("Formats"))

        self._chk_tex = QCheckBox("LaTeX  (.tex)")
        self._chk_tex.setChecked(True)
        outer.addWidget(self._chk_tex)

        self._chk_report = QCheckBox("Full Report  (.tex)")
        self._chk_report.setChecked(True)
        outer.addWidget(self._chk_report)

        self._chk_nb = QCheckBox("Mathematica  (.nb)")
        self._chk_nb.setChecked(True)
        outer.addWidget(self._chk_nb)

        self._chk_txt = QCheckBox("Plain text  (.txt)")
        self._chk_txt.setChecked(False)
        outer.addWidget(self._chk_txt)

        self._chk_json = QCheckBox("JSON  (.json)")
        self._chk_json.setChecked(False)
        outer.addWidget(self._chk_json)

        outer.addSpacing(8)

        # ── Options ───────────────────────────────────────────
        outer.addWidget(_section_label("Options"))

        self._chk_weyl = QCheckBox("Weyl-manifest form")
        self._chk_weyl.setChecked(True)
        outer.addWidget(self._chk_weyl)

        self._chk_dehn = QCheckBox("Include Dehn filling results")
        self._chk_dehn.setChecked(True)
        outer.addWidget(self._chk_dehn)

        outer.addSpacing(8)

        # ── Output path ───────────────────────────────────────
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

        # ── Export buttons ────────────────────────────────────
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
        """Store computation results for export.

        *data* should contain:
            manifold_data, easy_result, nz_data, entries, weyl_result, q_order_half
        """
        self._data = data
        name = data.get("manifold_data")
        if name:
            self._prefix_edit.setText(f"{name.name}_index")
        self._export_btn.setEnabled(True)
        self._copy_tex_btn.setEnabled(True)
        self._copy_txt_btn.setEnabled(True)

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

        out_dir = Path(self._dir_edit.text()).expanduser()
        prefix = self._prefix_edit.text().strip() or "index"

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Export error", f"Cannot create directory:\n{exc}")
            return

        exported = []
        nz = self._data["nz_data"]
        entries = self._data.get("entries", [])
        weyl = self._data.get("weyl_result")

        # Plain text export
        if self._chk_txt.isChecked():
            txt_path = out_dir / f"{prefix}.txt"
            self._write_plain_text(txt_path, nz, entries, weyl)
            exported.append(txt_path.name)

        # JSON export
        if self._chk_json.isChecked():
            json_path = out_dir / f"{prefix}.json"
            self._write_json(json_path, nz, entries)
            exported.append(json_path.name)

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
        from manifold_index.core.refined_index import format_refined_index

        nz = self._data["nz_data"]
        entries = self._data.get("entries", [])

        lines = [f"% Refined index for {self._data['manifold_data'].name}"]
        lines.append(f"% Computed {datetime.datetime.now().isoformat()}")
        lines.append("")

        for m_ext, e_ext, result in entries:
            if not result:
                continue
            fmt = format_refined_index(result, nz.num_hard)
            lines.append(f"% I({m_ext}, {e_ext})")
            lines.append(f"  {fmt}")
            lines.append("")

        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText("\n".join(lines))
        self._status.setText("✓  LaTeX copied to clipboard")
        self._status.setStyleSheet("color: #2ea043; font-size: 11px;")

    @Slot()
    def _copy_text(self) -> None:
        if not self._data:
            return
        from manifold_index.core.refined_index import format_refined_index

        nz = self._data["nz_data"]
        entries = self._data.get("entries", [])

        lines = [f"Refined index for {self._data['manifold_data'].name}"]
        for m_ext, e_ext, result in entries:
            if not result:
                continue
            fmt = format_refined_index(result, nz.num_hard)
            lines.append(f"I({m_ext}, {e_ext}) = {fmt}")

        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText("\n".join(lines))
        self._status.setText("✓  Plain text copied to clipboard")
        self._status.setStyleSheet("color: #2ea043; font-size: 11px;")

    # ------------------------------------------------------------------
    # File writers
    # ------------------------------------------------------------------

    def _write_plain_text(self, path: Path, nz, entries, weyl) -> None:
        from manifold_index.core.refined_index import format_refined_index

        with open(path, "w") as f:
            f.write(f"Refined 3D Index — {self._data['manifold_data'].name}\n")
            f.write(f"Computed: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Tetrahedra: {nz.n}  Cusps: {nz.r}\n")
            f.write(f"Hard edges: {nz.num_hard}  Easy edges: {nz.num_easy}\n\n")

            for m_ext, e_ext, result in entries:
                if not result:
                    continue
                fmt = format_refined_index(result, nz.num_hard)
                f.write(f"I({m_ext}, {e_ext}) = {fmt}\n")

    def _write_json(self, path: Path, nz, entries) -> None:
        import json

        data = {
            "manifold": self._data["manifold_data"].name,
            "n_tet": nz.n,
            "n_cusps": nz.r,
            "num_hard": nz.num_hard,
            "sectors": [],
        }
        for m_ext, e_ext, result in entries:
            if not result:
                continue
            sector = {
                "m_ext": [int(m) for m in m_ext],
                "e_ext": [str(e) for e in e_ext],
                "coefficients": {
                    str(k): int(v) for k, v in result.items() if v != 0
                },
            }
            data["sectors"].append(sector)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
