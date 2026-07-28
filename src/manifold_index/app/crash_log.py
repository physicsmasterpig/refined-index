"""app/crash_log.py — persistent crash + diagnostic logging.

The Windows build runs with ``console=False``: ``sys.stdout`` and
``sys.stderr`` are ``None``, so an uncaught exception (or a hard crash
in a C extension) leaves no trace at all — the app just disappears.
This module gives every failure a durable file trail:

    macOS:   ~/Library/Caches/manifold-index/logs/
    Linux:   $XDG_CACHE_HOME/manifold-index/logs/
    Windows: %LOCALAPPDATA%/manifold-index/logs/

Files
-----
``crash.log``        appended by ``sys.excepthook`` / ``threading.excepthook``
                     with a timestamped traceback; also collects Qt
                     warning/critical messages.
``faulthandler.log`` overwritten each launch; receives the low-level
                     traceback if a C extension segfaults.

Deliberately imports nothing heavy (no numpy, no snappy) — it must be
safe to install before anything else can fail.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

_LOG_DIR: Path | None = None
_FAULT_FH = None  # keep the file object alive for faulthandler
_INSTALLED = False


def log_dir() -> Path | None:
    """Directory that receives the log files (``None`` before install)."""
    return _LOG_DIR


def _resolve_log_dir() -> Path:
    env_override = os.environ.get("MANIFOLD_INDEX_CACHE_DIR")
    if env_override:
        base = Path(env_override)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "manifold-index"
    elif sys.platform == "win32":
        base = Path(
            os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        ) / "manifold-index"
    else:
        base = Path(
            os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        ) / "manifold-index"
    return base / "logs"


def _append(text: str) -> None:
    if _LOG_DIR is None:
        return
    try:
        with open(_LOG_DIR / "crash.log", "a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass  # never let logging itself take the app down


def _log_exception(prefix: str, exc_type, exc, tb) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    lines = "".join(traceback.format_exception(exc_type, exc, tb))
    _append(f"\n[{stamp}] {prefix}\n{lines}")


def install(app_version: str = "unknown") -> None:
    """Install excepthooks + faulthandler.  Idempotent."""
    global _LOG_DIR, _FAULT_FH, _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    try:
        _LOG_DIR = _resolve_log_dir()
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        _LOG_DIR = None
        return  # read-only environment — run without logging

    stamp = datetime.now().isoformat(timespec="seconds")
    _append(
        f"\n===== launch {stamp} — v{app_version} "
        f"python {sys.version.split()[0]} {sys.platform} "
        f"frozen={bool(getattr(sys, 'frozen', False))} =====\n"
    )

    try:
        _FAULT_FH = open(_LOG_DIR / "faulthandler.log", "w", encoding="utf-8")
        faulthandler.enable(file=_FAULT_FH, all_threads=True)
    except OSError:
        pass

    prev_hook = sys.excepthook

    def _excepthook(exc_type, exc, tb):
        _log_exception("UNCAUGHT (main thread)", exc_type, exc, tb)
        _show_dialog(exc_type, exc)
        if prev_hook is not None:
            prev_hook(exc_type, exc, tb)

    sys.excepthook = _excepthook

    prev_thread_hook = threading.excepthook

    def _thread_hook(args):
        _log_exception(
            f"UNCAUGHT (thread {args.thread.name if args.thread else '?'})",
            args.exc_type, args.exc_value, args.exc_traceback,
        )
        if prev_thread_hook is not None:
            prev_thread_hook(args)

    threading.excepthook = _thread_hook

    _install_qt_handler()


def _install_qt_handler() -> None:
    """Mirror Qt warning/critical/fatal messages into crash.log."""
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except ImportError:
        return

    _WANTED = {
        QtMsgType.QtWarningMsg: "QT-WARN",
        QtMsgType.QtCriticalMsg: "QT-CRIT",
        QtMsgType.QtFatalMsg: "QT-FATAL",
    }

    def _qt_handler(msg_type, context, message):
        label = _WANTED.get(msg_type)
        if label is not None:
            stamp = datetime.now().isoformat(timespec="seconds")
            _append(f"[{stamp}] {label}: {message}\n")

    qInstallMessageHandler(_qt_handler)


def arm_watchdog(timeout_s: float = 240.0) -> None:
    """Self-diagnose hangs: after *timeout_s* dump every thread's stack
    to faulthandler.log and hard-exit.  Used by the CI smoke mode — a
    GUI-thread deadlock would otherwise time out with zero output."""
    if _FAULT_FH is not None:
        faulthandler.dump_traceback_later(timeout_s, exit=True, file=_FAULT_FH)


def _show_dialog(exc_type, exc) -> None:
    """Best-effort modal dialog so a startup crash is not invisible."""
    if os.environ.get("MANIFOLDINDEX_SMOKE") or \
            os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return  # headless/CI — a modal dialog would hang forever
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is None:
            QApplication([])
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("ManifoldIndex — unexpected error")
        location = str(_LOG_DIR / "crash.log") if _LOG_DIR else "(unavailable)"
        box.setText(
            f"The application hit an unexpected error:\n\n"
            f"{exc_type.__name__}: {exc}\n\n"
            f"Details were written to:\n{location}"
        )
        box.exec()
    except Exception:
        pass
