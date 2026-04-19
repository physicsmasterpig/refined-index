"""app/dev_mode.py — runtime-toggleable developer-mode flag.

Seeded from the ``MANIFOLD_INDEX_DEV`` env var at import, and from QSettings
if a prior session set it.  UI widgets listen to ``dev_mode.changed`` to
show/hide developer-only controls live.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, QSettings, Signal


class _DevModeFlag(QObject):
    changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        env_on = os.environ.get("MANIFOLD_INDEX_DEV", "0") == "1"
        try:
            stored = QSettings("manifold_index", "app").value(
                "dev_mode", False, type=bool
            )
        except Exception:
            stored = False
        self._on: bool = bool(env_on or stored)

    def is_on(self) -> bool:
        return self._on

    def set(self, on: bool) -> None:
        on = bool(on)
        if on == self._on:
            return
        self._on = on
        try:
            QSettings("manifold_index", "app").setValue("dev_mode", on)
        except Exception:
            pass
        self.changed.emit(on)


dev_mode = _DevModeFlag()
