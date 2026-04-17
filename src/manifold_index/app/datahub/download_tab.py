"""app/datahub/download_tab.py — Download Tab for Data Hub.

BLUEPRINT §11.2.

Shows available remote data packs grouped by category (kernels / iref / nc).
Displays installed pack details (coverage, qq order).
"Check for Updates" refreshes the registry from GitHub.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QCoreApplication, Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QScrollArea, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from manifold_index.services.datahub_service import DataHubService
from manifold_index.app.workers.download_worker import DownloadWorker


class DownloadTab(QWidget):
    """Tab ①: Browse and download remote data packs.

    Signals
    -------
    download_finished(str)   — pack_name of a successfully installed pack
    """

    download_finished = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._registry = None
        self._worker: DownloadWorker | None = None
        self._pending_packs: list = []   # queue of packs left to download
        self._installed_names: list[str] = []
        self._total_packs: int = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Top controls ──────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        self._status_label = QLabel("Ready.")
        self._status_label.setProperty("class", "muted")
        ctrl_row.addWidget(self._status_label, 1)

        self._refresh_btn = QPushButton("Check for Updates")
        self._refresh_btn.setProperty("class", "secondary")
        self._refresh_btn.clicked.connect(self._on_refresh)
        ctrl_row.addWidget(self._refresh_btn)
        layout.addLayout(ctrl_row)

        # ── Pack tree ─────────────────────────────────────────────────
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Pack", "Status", "Size", "qq", "Coverage"])
        self._tree.setColumnWidth(0, 260)
        self._tree.setColumnWidth(1, 90)
        self._tree.setColumnWidth(2, 70)
        self._tree.setColumnWidth(3, 50)
        self._tree.setColumnWidth(4, 130)
        self._tree.setRootIsDecorated(True)
        self._tree.setAlternatingRowColors(True)
        layout.addWidget(self._tree, 1)

        # ── Download controls ─────────────────────────────────────────
        dl_box = QGroupBox("Download")
        dl_layout = QVBoxLayout(dl_box)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setVisible(False)
        dl_layout.addWidget(self._progress)

        btn_row = QHBoxLayout()
        self._download_btn = QPushButton("Download Selected")
        self._download_btn.setProperty("class", "primary")
        self._download_btn.setEnabled(False)
        self._download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(self._download_btn)

        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setProperty("class", "danger")
        self._remove_btn.setEnabled(False)
        self._remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(self._remove_btn)
        btn_row.addStretch(1)
        dl_layout.addLayout(btn_row)

        self._dl_status = QLabel()
        self._dl_status.setProperty("class", "muted")
        self._dl_status.setVisible(False)
        dl_layout.addWidget(self._dl_status)

        layout.addWidget(dl_box)

        # Wire selection
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)

        # Load bundled registry on startup
        self._load_registry(use_remote=False)

    # ------------------------------------------------------------------
    # Registry / tree population
    # ------------------------------------------------------------------

    def _load_registry(self, use_remote: bool = False) -> None:
        self._status_label.setText("Loading registry…")
        try:
            self._registry = DataHubService.load_registry(use_remote=use_remote)
            self._populate_tree()
            self._status_label.setText(
                "Remote registry loaded." if use_remote else "Bundled registry loaded."
            )
        except Exception as exc:
            self._status_label.setText(f"Registry error: {exc}")

    def _on_refresh(self) -> None:
        self._refresh_btn.setEnabled(False)
        self._load_registry(use_remote=True)
        self._refresh_btn.setEnabled(True)

    def _populate_tree(self) -> None:
        self._tree.clear()
        if self._registry is None:
            return

        packs = getattr(self._registry, "packs", [])
        categories: dict[str, list] = {}
        for pack in packs:
            cat = getattr(pack, "category", "other")
            categories.setdefault(cat, []).append(pack)

        cat_order = ["kernels", "iref", "nc", "other"]
        cat_labels = {
            "kernels": "Filling Kernels",
            "iref":    "I^ref Cache",
            "nc":      "NC Cycle Cache",
            "other":   "Other",
        }

        pack_count = 0
        for cat in cat_order:
            packs_in_cat = categories.get(cat, [])
            if not packs_in_cat:
                continue
            header = QTreeWidgetItem([cat_labels.get(cat, cat)])
            header.setFlags(header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            font = header.font(0)
            font.setBold(True)
            header.setFont(0, font)
            self._tree.addTopLevelItem(header)

            for pack in packs_in_cat:
                installed = getattr(pack, "installed", False)
                status = "✅ installed" if installed else "⬇ available"
                size_mb = getattr(pack, "size_mb", None)
                size_str = f"{size_mb:.0f} MB" if size_mb else "?"
                qq = str(getattr(pack, "qq_order", "?"))
                coverage = getattr(pack, "coverage", "")
                name = getattr(pack, "name", str(pack))
                row = QTreeWidgetItem([name, status, size_str, qq, coverage])
                row.setData(0, Qt.ItemDataRole.UserRole, pack)
                header.addChild(row)
                pack_count += 1
                # Yield to event loop every 50 packs to keep UI responsive
                # when building tree with many data packs.
                if pack_count % 50 == 0:
                    QCoreApplication.processEvents()

            header.setExpanded(True)

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        has_avail = False
        has_installed = False
        for item in items:
            pack = item.data(0, Qt.ItemDataRole.UserRole)
            if pack is None:
                continue
            if getattr(pack, "installed", False):
                has_installed = True
            else:
                has_avail = True
        self._download_btn.setEnabled(has_avail)
        self._remove_btn.setEnabled(has_installed)

    # ------------------------------------------------------------------
    # Download / Remove
    # ------------------------------------------------------------------

    def _on_download(self) -> None:
        if self._registry is None:
            return
        items = self._tree.selectedItems()
        packs = [
            item.data(0, Qt.ItemDataRole.UserRole) for item in items
            if item.data(0, Qt.ItemDataRole.UserRole) is not None
            and not getattr(item.data(0, Qt.ItemDataRole.UserRole), "installed", False)
        ]
        if not packs:
            return

        self._download_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._dl_status.setVisible(True)

        self._pending_packs = list(packs)
        self._installed_names = []
        self._total_packs = len(packs)
        self._start_next_download()

    def _start_next_download(self) -> None:
        if not self._pending_packs:
            self._finalize_download_batch()
            return
        pack = self._pending_packs.pop(0)
        idx = self._total_packs - len(self._pending_packs)
        name = getattr(pack, "name", "")
        self._dl_status.setText(f"Downloading {idx}/{self._total_packs}: {name}…")
        self._progress.setValue(0)
        self._worker = DownloadWorker(
            registry    = self._registry,
            pack        = pack,
            parent      = self,
        )
        self._worker.progress.connect(lambda r, t: self._progress.setValue(
            int(100 * r / t) if t > 0 else 0
        ))
        self._worker.status.connect(self._dl_status.setText)
        self._worker.finished.connect(self._on_download_finished)
        self._worker.error.connect(self._on_download_error)
        self._worker.start()

    def _on_download_finished(self, payload: dict) -> None:
        pack_name = payload.get("pack_name", "")
        if pack_name:
            self._installed_names.append(pack_name)
            self.download_finished.emit(pack_name)
        self._start_next_download()

    def _on_download_error(self, msg: str) -> None:
        # Abort the remaining queue on first error.
        self._pending_packs = []
        self._download_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._dl_status.setText(f"Error: {msg}")
        self._load_registry(use_remote=False)

    def _finalize_download_batch(self) -> None:
        self._download_btn.setEnabled(True)
        self._progress.setVisible(False)
        n = len(self._installed_names)
        if n == 1:
            self._dl_status.setText(f"✅ {self._installed_names[0]} installed")
        elif n > 1:
            self._dl_status.setText(f"✅ {n} packs installed")
        self._load_registry(use_remote=False)

    def _on_remove(self) -> None:
        items = self._tree.selectedItems()
        removed = 0
        for item in items:
            pack = item.data(0, Qt.ItemDataRole.UserRole)
            if pack and getattr(pack, "installed", False):
                try:
                    DataHubService.remove_pack(pack)
                    removed += 1
                except Exception as exc:
                    self._dl_status.setText(f"Remove error: {exc}")
                    self._dl_status.setVisible(True)
        if removed:
            self._dl_status.setText(f"Removed {removed} pack(s).")
            self._dl_status.setVisible(True)
            self._load_registry(use_remote=False)

