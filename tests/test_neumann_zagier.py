"""tests/test_neumann_zagier.py — Smoke tests for Neumann-Zagier matrix."""

import numpy as np
import pytest
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier, NeumannZagierData


@pytest.fixture(scope="module")
def nz_m004():
    pytest.importorskip("snappy")
    data = load_manifold("m004")
    easy = find_easy_edges(data)
    return data, easy, build_neumann_zagier(data, easy)


def _symplectic_pairing(u, v, n):
    return int(u[:n] @ v[n:] - u[n:] @ v[:n])


def test_is_symplectic(nz_m004):
    _, _, nz = nz_m004
    assert nz.is_symplectic()


def test_symplectic_inverse(nz_m004):
    _, _, nz = nz_m004
    product = nz.g_NZ @ nz.g_NZ_inv()
    np.testing.assert_array_almost_equal(product, np.eye(2 * nz.n), decimal=9)


def test_commutators(nz_m004):
    _, _, nz = nz_m004
    n = nz.n
    g = nz.g_NZ
    pos, mom = g[:n], g[n:]
    for i in range(n):
        for j in range(n):
            expected = 1 if i == j else 0
            assert _symplectic_pairing(pos[i], mom[j], n) == expected
    for i in range(n):
        for j in range(n):
            assert _symplectic_pairing(pos[i], pos[j], n) == 0
            assert _symplectic_pairing(mom[i], mom[j], n) == 0


def test_meridian_longitude_pairing(nz_m004):
    _, _, nz = nz_m004
    n, r = nz.n, nz.r
    g = nz.g_NZ
    for i in range(r):
        for j in range(r):
            expected = 1 if i == j else 0
            assert _symplectic_pairing(g[i], g[n + j], n) == expected
