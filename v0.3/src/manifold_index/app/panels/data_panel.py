"""
app/panels/data_panel.py — Data Packs tab.

Lets users browse, download, and manage optional pre-computed data packs
(filling kernels, refined indices for census manifolds).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from manifold_index.core.data_packs import (
    PackInfo,
    PackRegistry,
    check_installed,
    download_and_install,
    load_registry,
    uninstall_pack,
)


# ── Column indices ────────────────────────────────────────────────────
COL_NAME = 0
COL_DESC = 1
COL_SIZE = 2
COL_STATUS = 3
COL_ACTION = 4


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionTitle")
    return lbl


# ── Download worker thread ────────────────────────────────────────────

class _DownloadWorker(QThread):
    """Background thread for downloading + extracting a single pack."""

    progress = Signal(int, int)    # (received_bytes, total_bytes)
    status = Signal(str)
    finished = Signal(str, int)    # (pack_id, n_files)
    error = Signal(str, str)       # (pack_id, error_message)

    def __init__(
        self,
        registry: PackRegistry,
        pack: PackInfo,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._pack = pack

    def run(self) -> None:
        try:
            n = download_and_install(
                self._registry,
                self._pack,
                progress_fn=self.progress.emit,
                status_fn=self.status.emit,
            )
            self.finished.emit(self._pack.id, n)
        except Exception as exc:
            self.error.emit(self._pack.id, str(exc))


# ── Panel ─────────────────────────────────────────────────────────────

class DataPanel(QWidget):
    """Full-page tab for browsing and downloading data packs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._registry: PackRegistry | None = None
        self._worker: _DownloadWorker | None = None
        self._setup_ui()
        QTimer.singleShot(0, self._load_registry)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)

        # Title
        t = QLabel("Data Packs")
        t.setObjectName("panelTitle")
        root.addWidget(t)

        sub = QLabel(
            "Download pre-computed data to skip expensive computations.  "
            "Packs are extracted into your local cache directory and used "
            "automatically by the app."
        )
        sub.setObjectName("panelSubtitle")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # Cache directory label
        from manifold_index.core.kernel_cache import _user_cache_dir
        cache_str = str(_user_cache_dir())
        loc = QLabel(f"Cache: <code>{cache_str}</code>")
        loc.setTextFormat(Qt.TextFormat.RichText)
        loc.setStyleSheet("color: #8b949e; font-size: 11px;")
        root.addWidget(loc)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)

        self._refresh_btn = QPushButton("🔄  Check for Updates")
        self._refresh_btn.setFixedWidth(180)
        self._refresh_btn.clicked.connect(self._refresh_remote)
        toolbar.addWidget(self._refresh_btn)

        toolbar.addStretch()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        toolbar.addWidget(self._status_label)

        root.addLayout(toolbar)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "Pack", "Description", "Size", "Status", "Action",
        ])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_DESC, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_SIZE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_ACTION, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(COL_ACTION, 140)

        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self._table)

        # Progress bar (shared)
        prog_row = QHBoxLayout()
        prog_row.setContentsMargins(0, 0, 0, 0)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(18)
        prog_row.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        self._progress_label.setFixedWidth(120)
        prog_row.addWidget(self._progress_label)

        root.addLayout(prog_row)

        root.addStretch()

    # ------------------------------------------------------------------
    # Registry loading
    # ------------------------------------------------------------------

    def _load_registry(self) -> None:
        """Load the bundled registry and populate the table."""
        self._registry = load_registry(use_remote=False)
        check_installed(self._registry)
        self._populate_table()
        self._status_label.setText(
            f"{len(self._registry.packs)} packs available "
            f"(registry v{self._registry.version})"
        )

    def _refresh_remote(self) -> None:
        """Re-fetch registry from GitHub and refresh."""
        self._refresh_btn.setEnabled(False)
        self._status_label.setText("Checking for updates…")
        try:
            self._registry = load_registry(use_remote=True)
            check_installed(self._registry)
            self._populate_table()
            self._status_label.setText(
                f"✓ {len(self._registry.packs)} packs "
                f"(registry v{self._registry.version}, updated)"
            )
        except Exception as exc:
            self._status_label.setText(f"Update failed: {exc}")
        finally:
            self._refresh_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _populate_table(self) -> None:
        """Fill the table from the current registry."""
        if not self._registry:
            return

        packs = self._registry.packs
        self._table.setRowCount(len(packs))

        for row, pack in enumerate(packs):
            # Name
            name_item = QTableWidgetItem(pack.name)
            name_item.setToolTip(pack.id)
            font = name_item.font()
            font.setBold(True)
            name_item.setFont(font)
            self._table.setItem(row, COL_NAME, name_item)

            # Description
            desc_item = QTableWidgetItem(pack.description)
            self._table.setItem(row, COL_DESC, desc_item)

            # Size
            size_item = QTableWidgetItem(pack.size_human)
            size_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, COL_SIZE, size_item)

            # Status
            if pack.installed:
                status_text = f"✅ {pack.installed_files} files"
            else:
                status_text = "Not installed"
            status_item = QTableWidgetItem(status_text)
            self._table.setItem(row, COL_STATUS, status_item)

            # Action button
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(4, 2, 4, 2)
            btn_layout.setSpacing(4)

            if pack.installed:
                remove_btn = QPushButton("🗑  Remove")
                remove_btn.setFixedWidth(90)
                remove_btn.clicked.connect(
                    lambda checked, pid=pack.id: self._remove_pack(pid)
                )
                btn_layout.addWidget(remove_btn)
            else:
                dl_btn = QPushButton("⬇  Download")
                dl_btn.setFixedWidth(110)
                dl_btn.clicked.connect(
                    lambda checked, pid=pack.id: self._download_pack(pid)
                )
                btn_layout.addWidget(dl_btn)

            self._table.setCellWidget(row, COL_ACTION, btn_widget)

        self._table.resizeRowsToContents()

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_pack(self, pack_id: str) -> None:
        """Start downloading a pack in a background thread."""
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self, "Download in progress",
                "Please wait for the current download to finish.",
            )
            return

        reg = self._registry
        if not reg:
            return
        pack = reg.get(pack_id)
        if not pack:
            return

        # Disable all buttons during download
        self._set_buttons_enabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress_label.setText("Starting…")

        worker = _DownloadWorker(reg, pack, parent=self)
        worker.progress.connect(self._on_dl_progress)
        worker.status.connect(self._on_dl_status)
        worker.finished.connect(self._on_dl_finished)
        worker.error.connect(self._on_dl_error)

        self._worker = worker
        worker.start()

    @Slot(int, int)
    def _on_dl_progress(self, received: int, total: int) -> None:
        if total > 0:
            pct = int(100 * received / total)
            self._progress.setValue(pct)
            recv_mb = received / (1024 ** 2)
            total_mb = total / (1024 ** 2)
            self._progress_label.setText(f"{recv_mb:.1f} / {total_mb:.1f} MB")
        else:
            recv_mb = received / (1024 ** 2)
            self._progress_label.setText(f"{recv_mb:.1f} MB")

    @Slot(str)
    def _on_dl_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    @Slot(str, int)
    def _on_dl_finished(self, pack_id: str, n_files: int) -> None:
        self._progress.setVisible(False)
        self._progress_label.setText("")
        self._status_label.setText(f"✓ {pack_id}: {n_files} files installed")
        self._set_buttons_enabled(True)
        # Refresh table to show updated status
        check_installed(self._registry)
        self._populate_table()

    @Slot(str, str)
    def _on_dl_error(self, pack_id: str, error_msg: str) -> None:
        self._progress.setVisible(False)
        self._progress_label.setText("")
        self._status_label.setText(f"✗ Download failed")
        self._set_buttons_enabled(True)
        QMessageBox.critical(
            self, "Download Error",
            f"Failed to download {pack_id}:\n{error_msg}",
        )

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def _remove_pack(self, pack_id: str) -> None:
        """Remove an installed pack."""
        reg = self._registry
        if not reg:
            return
        pack = reg.get(pack_id)
        if not pack:
            return

        reply = QMessageBox.question(
            self,
            "Remove Data Pack",
            f'Remove all cached files for \u201c{pack.name}\u201d?\n\n'
            f"This will delete files from the {pack.target_subdir} cache.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        n = uninstall_pack(pack)
        self._status_label.setText(f"Removed {n} files from {pack.target_subdir}")
        check_installed(self._registry)
        self._populate_table()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_buttons_enabled(self, enabled: bool) -> None:
        """Enable/disable all action buttons in the table."""
        for row in range(self._table.rowCount()):
            w = self._table.cellWidget(row, COL_ACTION)
            if w:
                for child in w.findChildren(QPushButton):
                    child.setEnabled(enabled)
        self._refresh_btn.setEnabled(enabled)
