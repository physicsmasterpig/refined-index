"""Tests for kernel cache (smoke tests — full precomputation is slow)."""
import pytest
from manifold_index.core.kernel_cache import (
    KernelTable, load_kernel_table, list_cached_kernels, clear_kernel_cache,
)


def test_kernel_table_construction():
    """KernelTable can be constructed with empty table."""
    kt = KernelTable(P=1, Q=2, qq_order=8, qq_internal=14, eta_order=4,
                     hj_ks=[0, -2], table={}, m_scan=10, e_scan=10)
    assert kt.P == 1 and kt.Q == 2
    assert kt.qq_order == 8


def test_load_nonexistent_kernel():
    """Loading a kernel that doesn't exist returns None."""
    result = load_kernel_table(999, 999, 8)
    assert result is None


def test_list_cached_kernels_returns_list():
    kernels = list_cached_kernels()
    assert isinstance(kernels, list)


def test_clear_kernel_cache():
    n = clear_kernel_cache()
    assert isinstance(n, int) and n >= 0


def test_bundled_kernels_loadable():
    """Bundled kernels in data/kernel_cache/ must be loadable (if any exist)."""
    kernels = list_cached_kernels()
    for P, Q, qq in kernels:
        kt = load_kernel_table(P, Q, qq)
        assert kt is not None, f"Failed to load bundled kernel P={P},Q={Q},qq={qq}"
