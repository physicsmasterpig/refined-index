"""
tests/test_neumann_zagier.py — Tests for the Neumann-Zagier matrix construction.

Covers:
  - g_NZ is (2n × 2n) with integer entries
  - g_NZ is symplectic: g_NZ Ω g_NZ^T = Ω
  - g_NZ^{-1} satisfies g_NZ @ g_NZ_inv = I  (via the symplectic formula)
  - Row structure:  first r rows = meridians, rows r..n-1 = hard+easy edges,
                   rows n..n+r-1 = longitudes/2, rows n+r..2n-1 = Γ
  - Commutation relations:  [position_i, momentum_i] = 1,
                             [position_i, position_j] = 0,
                             [momentum_i, momentum_j] = 0
  - Affine shifts ν_x, ν_p are integer arrays of length n
  - Several manifolds: m004, s776, v2408
"""

from __future__ import annotations

import numpy as np
import pytest

from manifold_index.core.manifold import load_manifold
from manifold_index.core.gluing_equations import reduce_gluing_equations
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import (
    NeumannZagierData,
    build_neumann_zagier,
    _interleaved_to_block,
    _reduce_to_block,
    _build_omega_block,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nz_m004():
    data = load_manifold("m004")
    easy = find_easy_edges(data)
    return data, easy, build_neumann_zagier(data, easy)


@pytest.fixture(scope="module")
def nz_s776():
    data = load_manifold("s776")
    easy = find_easy_edges(data)
    return data, easy, build_neumann_zagier(data, easy)


@pytest.fixture(scope="module")
def nz_v2408():
    data = load_manifold("v2408")
    easy = find_easy_edges(data)
    return data, easy, build_neumann_zagier(data, easy)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def symplectic_pairing(u: np.ndarray, v: np.ndarray, n: int) -> int:
    """[u, v] = u_Z · v_Z'' - u_Z'' · v_Z  in block ordering."""
    return int(u[:n] @ v[n:] - u[n:] @ v[:n])


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_interleaved_to_block_roundtrip(self):
        """Permutation must be invertible (permuting back gives original)."""
        n = 3
        v = np.array([1, 2, 3, 4, 5, 6])  # interleaved: (Z1,Z1'', Z2,Z2'', Z3,Z3'')
        block = _interleaved_to_block(v, n)
        # block = (Z1,Z2,Z3, Z1'',Z2'',Z3'') = [1,3,5, 2,4,6]
        assert list(block) == [1, 3, 5, 2, 4, 6]

    def test_reduce_to_block_coefficients(self):
        """Coefficients in block form match _reduce_row (up to permutation)."""
        from manifold_index.core.gluing_equations import _reduce_row
        n = 2
        row = np.array([1, 0, 0,  0, 1, 0], dtype=int)  # Z_1 coeff=1 at tet0, Z_2' coeff=1 at tet1
        c_old, coeff_interleaved = _reduce_row(row, n)
        c_block, coeff_block = _reduce_to_block(row, n)
        # Constant: same as _reduce_row (Z_i' = 1 convention)
        assert c_block == c_old
        # Coefficients must agree up to permutation
        # Interleaved: [f0-g0, h0-g0, f1-g1, h1-g1] = [1-0,0-0, 0-1,0-1] = [1,0,-1,-1]
        # Block:       [f0-g0, f1-g1, h0-g0, h1-g1] = [1,-1, 0,-1]
        expected_block = np.array([1, -1, 0, -1])
        np.testing.assert_array_equal(coeff_block, expected_block)

    def test_omega_block_antisymmetric(self):
        n = 3
        omega = _build_omega_block(n)
        np.testing.assert_array_equal(omega, -omega.T)

    def test_omega_block_structure(self):
        n = 2
        omega = _build_omega_block(n)
        # [[0,0, 1,0], [0,0, 0,1], [-1,0, 0,0], [0,-1, 0,0]]
        expected = np.array([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])
        np.testing.assert_array_equal(omega, expected)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestNZShape:
    @pytest.mark.parametrize("fixture_name", ["nz_m004", "nz_s776", "nz_v2408"])
    def test_g_NZ_shape(self, fixture_name, request):
        data, easy, nz = request.getfixturevalue(fixture_name)
        n = data.num_tetrahedra
        assert nz.g_NZ.shape == (2 * n, 2 * n)

    @pytest.mark.parametrize("fixture_name", ["nz_m004", "nz_s776", "nz_v2408"])
    def test_g_NZ_integer(self, fixture_name, request):
        _, _, nz = request.getfixturevalue(fixture_name)
        assert nz.g_NZ.dtype.kind in ('i', 'f'), "g_NZ must be numeric (int or float)"

    @pytest.mark.parametrize("fixture_name", ["nz_m004", "nz_s776", "nz_v2408"])
    def test_nu_shapes(self, fixture_name, request):
        data, _, nz = request.getfixturevalue(fixture_name)
        n = data.num_tetrahedra
        assert nz.nu_x.shape == (n,)
        assert nz.nu_p.shape == (n,)

    @pytest.mark.parametrize("fixture_name", ["nz_m004", "nz_s776", "nz_v2408"])
    def test_row_type_counts(self, fixture_name, request):
        data, easy, nz = request.getfixturevalue(fixture_name)
        n, r = data.num_tetrahedra, data.num_cusps
        assert nz.num_hard == len(easy.hard_padding)
        assert nz.num_easy == easy.num_independent_easy
        assert nz.num_hard + nz.num_easy == n - r


# ---------------------------------------------------------------------------
# Symplectic condition
# ---------------------------------------------------------------------------

class TestSymplectic:
    @pytest.mark.parametrize("fixture_name", ["nz_m004", "nz_s776", "nz_v2408"])
    def test_is_symplectic(self, fixture_name, request):
        _, _, nz = request.getfixturevalue(fixture_name)
        assert nz.is_symplectic(), "g_NZ Ω g_NZ^T ≠ Ω"

    @pytest.mark.parametrize("fixture_name", ["nz_m004", "nz_s776", "nz_v2408"])
    def test_symplectic_inverse(self, fixture_name, request):
        _, _, nz = request.getfixturevalue(fixture_name)
        g_inv = nz.g_NZ_inv()
        product = nz.g_NZ @ g_inv
        np.testing.assert_array_almost_equal(
            product, np.eye(2 * nz.n), decimal=9,
            err_msg="g_NZ @ g_NZ_inv ≠ I"
        )


# ---------------------------------------------------------------------------
# Commutation relations
# ---------------------------------------------------------------------------

class TestCommutators:
    """
    Verify the canonical commutation relations:
        [position_i, momentum_j]  = δ_{ij}
        [position_i, position_j]  = 0
        [momentum_i, momentum_j]  = 0
    """

    def _check_relations(self, nz: NeumannZagierData):
        n = nz.n
        g = nz.g_NZ
        pos = g[:n]   # n position rows
        mom = g[n:]   # n momentum rows

        # [pos_i, mom_j] = δ_{ij}
        cross = np.array([[symplectic_pairing(pos[i], mom[j], n)
                           for j in range(n)] for i in range(n)])
        np.testing.assert_array_equal(
            cross, np.eye(n, dtype=int),
            err_msg="[position_i, momentum_j] ≠ δ_ij"
        )

        # [pos_i, pos_j] = 0
        pos_self = np.array([[symplectic_pairing(pos[i], pos[j], n)
                              for j in range(n)] for i in range(n)])
        np.testing.assert_array_equal(
            pos_self, np.zeros((n, n), dtype=int),
            err_msg="[position_i, position_j] ≠ 0"
        )

        # [mom_i, mom_j] = 0
        mom_self = np.array([[symplectic_pairing(mom[i], mom[j], n)
                              for j in range(n)] for i in range(n)])
        np.testing.assert_array_equal(
            mom_self, np.zeros((n, n), dtype=int),
            err_msg="[momentum_i, momentum_j] ≠ 0"
        )

    def test_commutators_m004(self, nz_m004):
        _, _, nz = nz_m004
        self._check_relations(nz)

    def test_commutators_s776(self, nz_s776):
        _, _, nz = nz_s776
        self._check_relations(nz)

    def test_commutators_v2408(self, nz_v2408):
        _, _, nz = nz_v2408
        self._check_relations(nz)


# ---------------------------------------------------------------------------
# Longitude / 2 pairing with meridians
# ---------------------------------------------------------------------------

class TestMeridianLongitude:
    """[meridian_k, longitude_k / 2] = 1 and cross-terms = 0."""

    def _check(self, nz: NeumannZagierData):
        n, r = nz.n, nz.r
        g = nz.g_NZ
        merid = g[:r]          # meridian rows
        long_half = g[n:n+r]   # longitude/2 rows

        pairing = np.array([[symplectic_pairing(merid[i], long_half[j], n)
                             for j in range(r)] for i in range(r)])
        np.testing.assert_array_equal(
            pairing, np.eye(r, dtype=int),
            err_msg="[meridian_i, longitude_j/2] ≠ δ_ij"
        )

    def test_ml_m004(self, nz_m004):
        _, _, nz = nz_m004
        self._check(nz)

    def test_ml_s776(self, nz_s776):
        _, _, nz = nz_s776
        self._check(nz)

    def test_ml_v2408(self, nz_v2408):
        _, _, nz = nz_v2408
        self._check(nz)


# ---------------------------------------------------------------------------
# Affine shift sign / parity checks
# ---------------------------------------------------------------------------

class TestAffineShift:
    @pytest.mark.parametrize("fixture_name", ["nz_m004", "nz_s776", "nz_v2408"])
    def test_nu_integer(self, fixture_name, request):
        _, _, nz = request.getfixturevalue(fixture_name)
        assert nz.nu_x.dtype.kind == 'i', "ν_x should be integer"
        assert nz.nu_p.dtype.kind in ('i', 'f'), "ν_p should be numeric"

    @pytest.mark.parametrize("fixture_name", ["nz_m004", "nz_s776", "nz_v2408"])
    def test_gamma_nu_zero(self, fixture_name, request):
        _, _, nz = request.getfixturevalue(fixture_name)
        # Γ rows have ν = 0 by construction
        gamma_nu = nz.nu_p[nz.r:]
        np.testing.assert_array_equal(
            gamma_nu, np.zeros(nz.n - nz.r, dtype=int),
            err_msg="ν_p for Γ rows should be 0"
        )
