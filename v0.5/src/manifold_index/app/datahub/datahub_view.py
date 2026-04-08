"""app/datahub/datahub_view.py — Data Hub Mode: QTabWidget with 3 sub-tabs.

BLUEPRINT §11.1.

Tabs:  Download | Generate | Export & Share
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from manifold_index.app.datahub.download_tab import DownloadTab
from manifold_index.app.datahub.generate_tab import GenerateTab
from manifold_index.app.datahub.export_tab import ExportTab


class DataHubView(QWidget):
    """Data Hub panel (three sub-tabs).

    Signals
    -------
    status_message(str)   — forwarded from any tab for the status bar
    """

    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._download_tab = DownloadTab(parent=self)
        self._generate_tab = GenerateTab(parent=self)
        self._export_tab   = ExportTab(parent=self)

        self._tabs.addTab(self._download_tab, "Download")
        self._tabs.addTab(self._generate_tab, "Generate")
        self._tabs.addTab(self._export_tab,   "Export & Share")

        # Forward status messages upward
        self._download_tab.download_finished.connect(
            lambda name: self.status_message.emit(f"Pack installed: {name}")
        )
        self._generate_tab.task_finished.connect(
            lambda msg: self.status_message.emit(msg)
        )
        self._export_tab.archive_created.connect(
            lambda path: self.status_message.emit(f"Archive created: {path}")
        )

