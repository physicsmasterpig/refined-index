"""Smoke test for the bundled app — run from terminal to test snappy."""
import sys
import os


def main():
    # If running from the bundle, sys._MEIPASS will be set
    meipass = getattr(sys, "_MEIPASS", None)
    print(f"frozen: {getattr(sys, 'frozen', False)}")
    print(f"_MEIPASS: {meipass}")
    print()

    # Test 1: Import snappy
    print("=== Test 1: Import snappy ===")
    try:
        import snappy
        print(f"  OK: snappy {snappy.version()} loaded from {snappy.__file__}")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Create a manifold
    print()
    print("=== Test 2: snappy.Manifold('m004') ===")
    try:
        M = snappy.Manifold("m004")
        print(f"  OK: {M}, volume={M.volume():.6f}")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 3: Get gluing equations
    print()
    print("=== Test 3: Gluing equations ===")
    try:
        eqs = M.gluing_equations("rect")
        print(f"  OK: {len(eqs)} equations")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 4: Test our manifold loader
    print()
    print("=== Test 4: load_manifold('m004') ===")
    try:
        from manifold_index.core.manifold import load_manifold
        data = load_manifold("m004")
        print(f"  OK: {data.name}, N={data.N} tets, {data.r} cusp(s)")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("All tests passed!")


if __name__ == "__main__":
    main()
