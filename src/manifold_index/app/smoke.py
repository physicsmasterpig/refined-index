"""app/smoke.py — self-test mode for CI.

When the environment variable ``MANIFOLDINDEX_SMOKE`` is set,
``launch_gui()`` schedules :func:`run` shortly after the main window is
shown.  It exercises the failure classes that have historically broken
the frozen Windows build — census database access, the process-pool
spawn path, WebEngine availability — writes a JSON report, and exits
the app with 0 (all passed) or 1.

Report path: ``$MANIFOLDINDEX_SMOKE_OUT`` (default ``./smoke.json``).

Run headless with::

    QT_QPA_PLATFORM=offscreen MANIFOLDINDEX_SMOKE=1 ./ManifoldIndex.exe
"""

from __future__ import annotations

import json
import os
import sys


def _check_snappy() -> dict:
    """Census SQLite databases + SnapPy C extension."""
    import snappy

    m = snappy.Manifold("m003")
    return {"ok": True, "num_tetrahedra": int(m.num_tetrahedra())}


def _check_manifold_pipeline() -> dict:
    """The app's own manifold-analysis layer on a small census manifold."""
    from manifold_index.core.manifold import load_manifold

    data = load_manifold("m003")
    return {"ok": True, "n_tetrahedra": int(data.num_tetrahedra)}


def _check_spawn_pool() -> dict:
    """Process-pool roundtrip via the same context the kernel code uses.

    In the frozen exe this re-executes the bundle for the child process —
    exactly the path that parallel kernel precomputation depends on.
    """
    from concurrent.futures import ProcessPoolExecutor

    from manifold_index.core.kernel_cache import _pool_context

    with ProcessPoolExecutor(max_workers=1, mp_context=_pool_context()) as pool:
        result = pool.submit(pow, 2, 8).result(timeout=180)
    if result != 256:
        raise RuntimeError(f"pool returned {result!r}")
    return {"ok": True, "method": _pool_context().get_start_method()}


def _check_webengine() -> dict:
    from manifold_index.app.widgets.math_view import _HAS_WEBENGINE

    return {"ok": bool(_HAS_WEBENGINE)}


def run(app) -> None:
    """Execute all checks, write the report, and exit the application."""
    from manifold_index import __version__

    report: dict = {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "frozen": bool(getattr(sys, "frozen", False)),
        "checks": {},
    }
    checks = {
        "snappy_census": _check_snappy,
        "manifold_pipeline": _check_manifold_pipeline,
        "spawn_pool": _check_spawn_pool,
        "webengine": _check_webengine,
    }
    all_ok = True
    for name, fn in checks.items():
        try:
            report["checks"][name] = fn()
        except Exception as exc:  # noqa: BLE001 — report, don't die
            import traceback

            report["checks"][name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        all_ok = all_ok and bool(report["checks"][name].get("ok"))

    report["ok"] = all_ok
    out = os.environ.get("MANIFOLDINDEX_SMOKE_OUT", "smoke.json")
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except OSError:
        all_ok = False

    app.exit(0 if all_ok else 1)


def schedule(app, delay_ms: int = 3000) -> None:
    """Arrange for :func:`run` once the event loop has settled."""
    from PySide6.QtCore import QTimer

    QTimer.singleShot(delay_ms, lambda: run(app))
