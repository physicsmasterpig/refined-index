"""Tests for GUI workers and formatters (no display required)."""
from fractions import Fraction
import pytest


def test_build_eval_grid_1cusp():
    from manifold_index.app.workers import build_eval_grid
    grid = build_eval_grid(1)
    assert len(grid) == 45   # 5 m × 9 e values


def test_build_eval_grid_2cusps():
    from manifold_index.app.workers import build_eval_grid
    grid = build_eval_grid(2)
    assert len(grid) == 45 ** 2


def test_series_to_katex():
    from manifold_index.app.formatters import _series_to_katex
    result = {(2, 2): 1, (4, 0): -1}
    tex = _series_to_katex(result, num_hard=1)
    assert r"\eta" in tex or "eta" in tex.lower()
    assert "q" in tex


def test_katex_html_generation():
    from manifold_index.app.katex import build_katex_html
    html = build_katex_html("<p>$x^2$</p>")
    assert "katex" in html.lower()
    assert "<p>$x^2$</p>" in html


def test_transformed_fill_result_dataclass():
    from manifold_index.app.workers import TransformedFillResult
    tfr = TransformedFillResult(
        cusp_idx=0, P_nc=1, Q_nc=0, R=0, S=1,
        p=1, q=3, P_user=1, Q_user=3,
        fill_results=[],
    )
    assert tfr.cusp_idx == 0
    assert tfr.P_nc == 1
    assert tfr.fill_results == []
