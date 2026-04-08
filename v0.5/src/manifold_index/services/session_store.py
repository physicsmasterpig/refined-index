"""services/session_store.py — Session save / restore to disk.

Phase 10.  Sessions are stored as JSON files in::

    ~/.manifold_index_sessions/
        <manifold_name>_<timestamp>.json
        last.json            ← symlink / copy of the most recent file

Usage::

    from manifold_index.services.session_store import save_session, load_last_session

    save_session(session)
    session = load_last_session()   # None if no session exists
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from manifold_index.services.session import Session, session_from_dict, session_to_dict

_SESSION_DIR = Path.home() / ".manifold_index_sessions"


def _session_dir() -> Path:
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSION_DIR


def save_session(session: Session) -> Path:
    """Serialise *session* and write it to disk.

    Returns
    -------
    Path
        The path of the file written.
    """
    d = _session_dir()
    ts = int(time.time())
    name = session.manifold_name or "untitled"
    # Sanitise: remove characters unsafe for filenames
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    filename = f"{safe}_{ts}.json"
    path = d / filename

    data = session_to_dict(session)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Keep a "last.json" pointer (overwrite unconditionally)
    last = d / "last.json"
    try:
        with open(last, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass

    return path


def load_session(path: str | Path) -> Session:
    """Load a session from *path*.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as a valid session.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        return session_from_dict(data)
    except Exception as exc:
        raise ValueError(f"Cannot restore session from {path}: {exc}") from exc


def load_last_session() -> Session | None:
    """Return the most recently saved session, or ``None``."""
    last = _session_dir() / "last.json"
    if not last.exists():
        return None
    try:
        return load_session(last)
    except Exception:
        return None


def list_saved_sessions() -> list[dict]:
    """Return metadata for all saved sessions, newest first.

    Each entry: ``{"path": Path, "manifold_name": str, "timestamp": float}``.
    """
    d = _session_dir()
    entries: list[dict] = []
    for path in sorted(d.glob("*.json"), reverse=True):
        if path.name == "last.json":
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            entries.append({
                "path": path,
                "manifold_name": data.get("manifold_name", "?"),
                "timestamp": path.stat().st_mtime,
                "stage": data.get("stage", 0),
            })
        except Exception:
            continue
    return entries
