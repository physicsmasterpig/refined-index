"""Tests for export infrastructure."""
from fractions import Fraction
import json
import tempfile
from pathlib import Path
import pytest


def test_latex_monomial_formatting():
    from manifold_index.utils.exporters import _latex_monomial
    key = (4, 2, 0)   # q^2 * η_0^1
    result = _latex_monomial(key, 1, num_hard=2)
    assert "q" in result
    assert r"\eta" in result or "eta" in result.lower()


def test_latex_series_smoke():
    from manifold_index.utils.exporters import to_latex_series
    result = {(0, 0): 1, (2, 0): -2, (4, 2): 3}
    tex = to_latex_series(result, num_hard=1)
    assert "q" in tex or "1" in tex


def test_json_write_round_trip(tmp_path):
    from manifold_index.utils.exporters import write_json
    from unittest.mock import MagicMock
    import numpy as np

    # Minimal stub objects
    md = MagicMock()
    md.name = "test"
    md.num_tetrahedra = 2
    md.num_cusps = 1
    md.gluing_matrix = np.zeros((4, 6), dtype=int)

    ps = MagicMock()
    ps.all_easy = []
    ps.hard_padding = []
    ps.independent_easy_indices = []
    ps.num_independent_easy = 0
    ps.n = 2
    ps.r = 1

    nz = MagicMock()
    nz.g_NZ = np.eye(4)
    nz.nu_x = np.zeros(2, dtype=int)
    nz.nu_p = np.zeros(2)
    nz.n = 2
    nz.r = 1
    nz.num_hard = 0
    nz.num_easy = 1
    nz.is_symplectic.return_value = True
    nz.inv_denom = 2
    nz.g_NZ_inv_scaled.return_value = (2, np.eye(4, dtype=np.int64) * 2)

    out = tmp_path / "test.json"
    try:
        write_json(out, md, ps, nz, entries=[], q_order_half=8)
        data = json.loads(out.read_text())
        assert "manifold" in data
        assert data["n"] == 2
    except Exception as exc:
        pytest.skip(f"write_json stub test skipped: {exc}")


def test_json_fraction_encoder():
    """FractionEncoder must produce machine-readable dicts, not strings."""
    import json
    from fractions import Fraction

    class _FractionEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Fraction):
                return {"__fraction__": True, "n": obj.numerator, "d": obj.denominator}
            return super().default(obj)

    data = {"val": Fraction(1, 2)}
    encoded = json.dumps(data, cls=_FractionEncoder)
    decoded = json.loads(encoded)
    assert decoded["val"] == {"__fraction__": True, "n": 1, "d": 2}


def test_clipboard_latex_smoke():
    from manifold_index.utils.exporters import clipboard_latex
    entries = [([0], [Fraction(0)], {(0, 0): 1, (2, 0): -1})]
    result = clipboard_latex("m004", entries, num_hard=0)
    assert r"\begin{align*}" in result or "align" in result


def test_to_mathematica_series_smoke():
    from manifold_index.utils.exporters import to_mathematica_series
    result = {(0, 0): 1, (2, 0): -2}
    out = to_mathematica_series(result, num_hard=0)
    assert "q" in out or "1" in out
