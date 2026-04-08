"""app/datahub/export_tab.py — Export & Share Tab for Data Hub.

BLUEPRINT §11.4.

Three sections:
  A  Cache Browser — filterable list of local cache files with checkboxes
  B  Export Selected — format selection (Mathematica / JSON) + output dir
  C  Publish as Data Pack — pack metadata + "Create archive" button
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from manifold_index.services.datahub_service import DataHubService


class ExportTab(QWidget):
    """Tab ③: browse cache, export to formats, create data packs.

    Signals
    -------
    archive_created(str)   — path of the .tar.gz that was created
    """

    archive_created = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._all_files: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── A: Cache Browser ─────────────────────────────────────────
        browser_box = QGroupBox("Cache Browser")
        bbl = QVBoxLayout(browser_box)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "Kernels", "I^ref", "NC"])
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._filter_combo)
        filter_row.addWidget(QLabel("Search:"))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("filename…")
        self._search_edit.textChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._search_edit, 1)
        bbl.addLayout(filter_row)

        self._files_table = QTableWidget(0, 5)
        self._files_table.setHorizontalHeaderLabels(
            ["☑", "Filename", "Type", "qq", "Size"]
        )
        self._files_table.setColumnWidth(0, 30)
        self._files_table.setColumnWidth(1, 280)
        self._files_table.setColumnWidth(2, 70)
        self._files_table.setColumnWidth(3, 50)
        self._files_table.setColumnWidth(4, 70)
        self._files_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._files_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._files_table.horizontalHeader().setStretchLastSection(False)
        self._files_table.setMinimumHeight(160)
        bbl.addWidget(self._files_table)

        sel_row = QHBoxLayout()
        sel_row.addStretch(1)
        self._sel_all_btn = QPushButton("Select all")
        self._sel_all_btn.setProperty("class", "tertiary")
        self._sel_all_btn.clicked.connect(self._on_select_all)
        sel_row.addWidget(self._sel_all_btn)
        self._clear_sel_btn = QPushButton("Clear")
        self._clear_sel_btn.setProperty("class", "tertiary")
        self._clear_sel_btn.clicked.connect(self._on_clear_sel)
        sel_row.addWidget(self._clear_sel_btn)
        self._refresh_files_btn = QPushButton("Refresh")
        self._refresh_files_btn.setProperty("class", "secondary")
        self._refresh_files_btn.clicked.connect(self._load_files)
        sel_row.addWidget(self._refresh_files_btn)
        bbl.addLayout(sel_row)

        layout.addWidget(browser_box, 2)

        # ── B: Export Selected ────────────────────────────────────────
        export_box = QGroupBox("Export Selected")
        ebl = QVBoxLayout(export_box)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self._fmt_math_chk = QCheckBox("Mathematica (.m)")
        self._fmt_math_chk.setChecked(True)
        self._fmt_json_chk = QCheckBox("JSON (.json)")
        fmt_row.addWidget(self._fmt_math_chk)
        fmt_row.addWidget(self._fmt_json_chk)
        fmt_row.addStretch(1)
        ebl.addLayout(fmt_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._export_path = QLineEdit()
        self._export_path.setPlaceholderText(str(Path.home() / "manifold_export"))
        out_row.addWidget(self._export_path, 1)
        self._export_browse_btn = QPushButton("Browse")
        self._export_browse_btn.setProperty("class", "secondary")
        self._export_browse_btn.clicked.connect(self._on_export_browse)
        out_row.addWidget(self._export_browse_btn)
        ebl.addLayout(out_row)

        export_btn_row = QHBoxLayout()
        export_btn_row.addStretch(1)
        self._export_btn = QPushButton("Export selected")
        self._export_btn.setProperty("class", "primary")
        self._export_btn.clicked.connect(self._on_export_selected)
        export_btn_row.addWidget(self._export_btn)
        ebl.addLayout(export_btn_row)

        self._export_status = QLabel()
        self._export_status.setProperty("class", "muted")
        self._export_status.setVisible(False)
        ebl.addWidget(self._export_status)

        layout.addWidget(export_box)

        # ── C: Publish as Data Pack ───────────────────────────────────
        pub_box = QGroupBox("Publish as Data Pack")
        pbl = QVBoxLayout(pub_box)

        def _labeled_edit(label: str, placeholder: str = "") -> tuple[QLabel, QLineEdit]:
            lbl = QLabel(label)
            lbl.setFixedWidth(100)
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            return lbl, edit

        for attr, lbl_text, ph in [
            ("_pub_id",   "Pack ID:",      "kernels_qq50_custom"),
            ("_pub_name", "Pack name:",    "Filling Kernels qq=50 (custom)"),
            ("_pub_desc", "Description:",  "Pre-computed kernels at qq=50"),
            ("_pub_tag",  "Release tag:",  "data-v2"),
        ]:
            row = QHBoxLayout()
            lbl, edit = _labeled_edit(lbl_text, ph)
            row.addWidget(lbl)
            row.addWidget(edit, 1)
            pbl.addLayout(row)
            setattr(self, attr, edit)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target dir:"))
        self._target_kernels = QRadioButton("kernel_cache")
        self._target_iref    = QRadioButton("iref_cache")
        self._target_nc      = QRadioButton("nc_cache")
        self._target_kernels.setChecked(True)
        tg = QButtonGroup(pub_box)
        for rb in (self._target_kernels, self._target_iref, self._target_nc):
            tg.addButton(rb)
            target_row.addWidget(rb)
        target_row.addStretch(1)
        pbl.addLayout(target_row)

        pub_out_row = QHBoxLayout()
        pub_out_row.addWidget(QLabel("Dist dir:"))
        self._pub_out_path = QLineEdit()
        self._pub_out_path.setPlaceholderText("dist/")
        pub_out_row.addWidget(self._pub_out_path, 1)
        self._pub_browse_btn = QPushButton("Browse")
        self._pub_browse_btn.setProperty("class", "secondary")
        self._pub_browse_btn.clicked.connect(self._on_pub_browse)
        pub_out_row.addWidget(self._pub_browse_btn)
        pbl.addLayout(pub_out_row)

        create_row = QHBoxLayout()
        self._create_btn = QPushButton("Create archive")
        self._create_btn.setProperty("class", "primary")
        self._create_btn.clicked.connect(self._on_create_archive)
        create_row.addWidget(self._create_btn)
        create_row.addStretch(1)
        pbl.addLayout(create_row)

        self._pub_result = QLabel()
        self._pub_result.setProperty("class", "muted")
        self._pub_result.setVisible(False)
        self._pub_result.setWordWrap(True)
        pbl.addWidget(self._pub_result)

        layout.addWidget(pub_box)

        # Initial load
        self._load_files()

    # ------------------------------------------------------------------
    # Cache file listing
    # ------------------------------------------------------------------

    def _load_files(self) -> None:
        try:
            self._all_files = DataHubService.list_cache_files()
        except Exception as exc:
            self._export_status.setText(f"List error: {exc}")
            self._export_status.setVisible(True)
            self._all_files = []
        self._rebuild_table()

    def _on_filter_changed(self) -> None:
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        type_map = {"All": None, "Kernels": "kernels", "I^ref": "iref", "NC": "nc"}
        type_filter = type_map.get(self._filter_combo.currentText())
        search_text = self._search_edit.text().strip().lower()

        self._files_table.setRowCount(0)
        for entry in self._all_files:
            if type_filter and entry.get("type") != type_filter:
                continue
            path = entry.get("path", "")
            filename = Path(path).name if path else _entry_filename(entry)
            if search_text and search_text not in filename.lower():
                continue

            row = self._files_table.rowCount()
            self._files_table.insertRow(row)

            chk = QTableWidgetItem()
            chk.setCheckState(Qt.CheckState.Unchecked)
            chk.setFlags(chk.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setData(Qt.ItemDataRole.UserRole, entry)
            self._files_table.setItem(row, 0, chk)
            self._files_table.setItem(row, 1, QTableWidgetItem(filename))
            self._files_table.setItem(row, 2, QTableWidgetItem(entry.get("type", "?")))
            self._files_table.setItem(row, 3, QTableWidgetItem(str(entry.get("qq_order", "?"))))
            size_b = entry.get("size_bytes", 0)
            size_str = f"{size_b / 1e6:.1f} MB" if size_b else "—"
            self._files_table.setItem(row, 4, QTableWidgetItem(size_str))

    def _selected_entries(self) -> list[dict]:
        entries = []
        for row in range(self._files_table.rowCount()):
            item = self._files_table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                entry = item.data(Qt.ItemDataRole.UserRole)
                if entry:
                    entries.append(entry)
        return entries

    def _on_select_all(self) -> None:
        for row in range(self._files_table.rowCount()):
            item = self._files_table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked)

    def _on_clear_sel(self) -> None:
        for row in range(self._files_table.rowCount()):
            item = self._files_table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)

    # ------------------------------------------------------------------
    # Export selected
    # ------------------------------------------------------------------

    def _on_export_browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose output directory",
            self._export_path.text() or str(Path.home()),
        )
        if directory:
            self._export_path.setText(directory)

    def _on_export_selected(self) -> None:
        entries = self._selected_entries()
        if not entries:
            self._show_export_status("No files selected.")
            return

        formats = []
        if self._fmt_math_chk.isChecked():
            formats.append("mathematica")
        if self._fmt_json_chk.isChecked():
            formats.append("json")
        if not formats:
            self._show_export_status("Select at least one format.")
            return

        out_dir = self._export_path.text().strip() or str(Path.home() / "manifold_export")
        paths   = [e.get("path", "") for e in entries if e.get("path")]

        try:
            written = DataHubService.export_cache_files(paths, formats, out_dir)
            self._show_export_status(f"Exported {len(written)} file(s) → {out_dir}")
        except Exception as exc:
            self._show_export_status(f"Export error: {exc}")

    def _show_export_status(self, msg: str) -> None:
        self._export_status.setText(msg)
        self._export_status.setVisible(True)
        QTimer.singleShot(4000, lambda: self._export_status.setVisible(False))

    # ------------------------------------------------------------------
    # Publish as Data Pack
    # ------------------------------------------------------------------

    def _on_pub_browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose dist directory",
            self._pub_out_path.text() or "dist",
        )
        if directory:
            self._pub_out_path.setText(directory)

    def _on_create_archive(self) -> None:
        entries = self._selected_entries()
        if not entries:
            self._pub_result.setText("No files selected in browser.")
            self._pub_result.setVisible(True)
            return

        pack_name   = self._pub_id.text().strip() or "custom_pack"  # type: ignore[attr-defined]
        release_tag = self._pub_tag.text().strip() or "data-v1"     # type: ignore[attr-defined]
        out_dir     = self._pub_out_path.text().strip() or "dist"

        paths = [e.get("path", "") for e in entries if e.get("path")]

        try:
            self._create_btn.setEnabled(False)
            result = DataHubService.create_tarball(
                file_paths     = paths,
                pack_name      = pack_name,
                release_tag    = release_tag,
                output_dir     = out_dir,
                update_registry= True,
            )
            archive = str(result["path"])
            sha     = result["sha256"][:16]
            size_mb = result["size_bytes"] / 1e6
            n       = result["n_files"]
            self._pub_result.setText(
                f"→ {archive}  ({size_mb:.1f} MB · {n} files · SHA-256: {sha}…)\n"
                "data_packs.json updated.\n"
                "Next: upload to GitHub Releases as a release asset."
            )
            self._pub_result.setVisible(True)
            self.archive_created.emit(archive)
        except Exception as exc:
            self._pub_result.setText(f"Archive error: {exc}")
            self._pub_result.setVisible(True)
        finally:
            self._create_btn.setEnabled(True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry_filename(entry: dict) -> str:
    """Build a display filename from a cache entry dict."""
    t = entry.get("type", "?")
    if t == "kernels":
        return f"kernel_{entry.get('P',0)}_{entry.get('Q',0)}_qq{entry.get('qq_order','?')}.pkl.gz"
    if t == "iref":
        return f"iref_{entry.get('manifold_name','?')}_qq{entry.get('qq_order','?')}.pkl.gz"
    if t == "nc":
        return f"nc_{entry.get('manifold_name','?')}_qq{entry.get('qq_order','?')}.pkl.gz"
    return "unknown"

