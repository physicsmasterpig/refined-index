"""
app/main.py — GUI entry point for the Refined Index Calculator.

Launches the three-screen PySide6 GUI:
  Screen 1 — Manifold input + parameters
  Screen 2 — Basis selection (automatic pipeline + interactive cusp selection)
  Screen 3 — Refined index output + export

Run with:
    python -m manifold_index.app.main
or via the 'refined-index' console_script entry point.
"""

from manifold_index.app.gui import launch_gui


def main() -> None:
    import sys
    if "--smoke-test" in sys.argv:
        _smoke_test()
        return
    launch_gui()


def _smoke_test() -> None:
    """Quick headless check that snappy loads manifolds correctly."""
    import traceback

    tests = [
        ("import snappy", lambda: __import__("snappy")),
        ("Manifold('m004')", lambda: __import__("snappy").Manifold("m004")),
        ("load_manifold('m004')",
         lambda: __import__("manifold_index.core.manifold",
                            fromlist=["load_manifold"]).load_manifold("m004")),
    ]
    ok = True
    for label, fn in tests:
        try:
            result = fn()
            print(f"  PASS: {label} -> {result}")
        except Exception as exc:
            print(f"  FAIL: {label} -> {exc}")
            traceback.print_exc()
            ok = False
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
