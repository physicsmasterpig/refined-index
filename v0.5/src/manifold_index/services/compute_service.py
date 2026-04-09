"""
services.compute_service
========================
Thin orchestration layer over core/: load manifold, probe cache,
compute and project the refined index, run Weyl checks.

Rule: no PySide6, no app/ imports.  All arguments and return values are
plain Python / NumPy / core dataclass objects.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from manifold_index.core import (
    manifold as _manifold_mod,
    neumann_zagier as _nz_mod,
    phase_space as _ps_mod,
    refined_index as _ri_mod,
    kernel_cache as _kc_mod,
    weyl_check as _wc_mod,
)


class ComputeService:
    """Orchestrates the load → NZ → index pipeline.

    Every method is a ``@staticmethod`` so the service can be used without
    instantiation.  Workers call one method per dispatch.
    """

    # ── Loading ───────────────────────────────────────────────────────

    @staticmethod
    def load_manifold(name: str) -> tuple[Any, Any, Any]:
        """Load manifold, find phase space basis, build NZ matrix.

        Returns
        -------
        (ManifoldData, EasyEdgeResult, NeumannZagierData)

        Raises
        ------
        ValueError
            If SnaPy cannot find the manifold or if the gluing matrix
            has unexpected dimensions.
        ImportError
            If snappy is not installed.
        """
        manifold_data = _manifold_mod.load_manifold(name)
        easy_result = _ps_mod.find_easy_edges(manifold_data)
        nz_data = _nz_mod.build_neumann_zagier(manifold_data, easy_result)
        return manifold_data, easy_result, nz_data

    # ── Cache probing ─────────────────────────────────────────────────

    @staticmethod
    def probe_cache(name: str, nz_data: Any) -> dict:
        """Check which pre-computed data are available for *name*.

        Returns
        -------
        dict with keys ``"iref"``, ``"nc"``, ``"kernels"``:

        .. code-block:: python

            {
                "iref":    {"available": bool, "qq_order": int | None,
                            "m_range": int | None},
                "nc":      {"available": bool, "qq_order": int | None,
                            "p_range": int | None},
                "kernels": {"available": bool, "count": int,
                            "qq_orders": list[int]},
            }
        """
        # ── I^ref cache ───
        iref_info: dict[str, Any] = {"available": False, "qq_order": None, "m_range": None}
        try:
            iref_files = _kc_mod.list_iref_caches()
            for rec in iref_files:
                if rec.get("manifold_name") == name:
                    iref_info["available"] = True
                    iref_info["qq_order"] = rec.get("qq_order")
                    iref_info["m_range"] = rec.get("m_max")
                    break
        except Exception:
            pass

        # ── NC cycle cache ───
        nc_info: dict[str, Any] = {"available": False, "qq_order": None, "p_range": None}
        try:
            nc_files = _kc_mod.list_nc_cycle_caches()
            for rec in nc_files:
                if rec.get("manifold_name") == name:
                    nc_info["available"] = True
                    nc_info["qq_order"] = rec.get("qq_order")
                    nc_info["p_range"] = rec.get("p_range")
                    break
        except Exception:
            pass

        # ── Kernel cache ───
        kernels_info: dict[str, Any] = {"available": False, "count": 0, "qq_orders": []}
        try:
            cached = _kc_mod.list_cached_kernels()
            if cached:
                kernels_info["available"] = True
                kernels_info["count"] = len(cached)
                kernels_info["qq_orders"] = sorted({qq for _, _, qq in cached})
        except Exception:
            pass

        return {"iref": iref_info, "nc": nc_info, "kernels": kernels_info}

    # ── Index computation ─────────────────────────────────────────────

    @staticmethod
    def compute_refined_index(
        nz_data: Any,
        m_ext: list[int],
        e_ext: list[Fraction],
        q_order_half: int,
    ) -> Any:
        """Compute I^ref(m_ext, e_ext) at truncation *q_order_half*.

        Returns
        -------
        RefinedIndexResult — ``dict[tuple[int, ...], int]``.
        """
        return _ri_mod.compute_refined_index(
            nz_data, m_ext, e_ext, q_order_half=q_order_half,
        )

    @staticmethod
    def load_refined_index_from_cache(
        name: str,
        nz_data: Any,
        m_ext: list[int],
        e_ext: list[Fraction],
        q_order_half: int,
    ) -> Any | None:
        """Try to load I^ref(m,e) from the disk iref cache.

        Returns the RefinedIndexResult on a cache hit, or ``None`` on miss.
        """
        try:
            loaded = _kc_mod.load_iref_cache(
                nz_data, manifold_name=name, qq_filter=q_order_half,
            )
            if loaded == 0:
                return None
            # After loading, query the in-memory cache directly via compute
            # (it will now be a cache hit with ~zero cost).
            return _ri_mod.compute_refined_index(
                nz_data, m_ext, e_ext, q_order_half=q_order_half,
            )
        except Exception:
            return None

    @staticmethod
    def enumerate_iref_cache(
        name: str,
        nz_data: Any,
        q_order_half: int,
    ) -> list[tuple[list, list, Any]]:
        """Return every cached I^ref entry for *name* as (m_ext, e_ext, result).

        Reads the cache file directly without touching the in-memory cache.
        Returns an empty list if no cache file is found or on any error.
        """
        try:
            return _kc_mod.enumerate_iref_entries(
                nz_data, manifold_name=name, qq_filter=q_order_half,
            )
        except Exception:
            return []

    @staticmethod
    def project_refined_index(
        result: Any,
        active_edges: list[bool],
    ) -> Any:
        """Project *result* by setting inactive η variables to 1.

        For each inactive edge *j* (``active_edges[j] == False``),
        sum over all doubled-exponent values of η_j, effectively setting
        η_j = 1.

        Returns a new dict — input is never mutated.
        Passing all-True active_edges returns a shallow copy of *result*.
        """
        if all(active_edges):
            return dict(result)

        num_hard = len(active_edges)
        projected: dict[tuple[int, ...], int] = {}

        for key, coeff in result.items():
            # Build a new key with inactive η exponents collapsed to 0
            new_key_list = [key[0]]
            for j in range(num_hard):
                new_key_list.append(key[1 + j] if active_edges[j] else 0)
            new_key = tuple(new_key_list)
            projected[new_key] = projected.get(new_key, 0) + coeff

        # Drop zero-coefficient terms
        return {k: v for k, v in projected.items() if v != 0}

    # ── Weyl check ────────────────────────────────────────────────────

    @staticmethod
    def run_weyl_check(
        entries: list[tuple[list[int], list[Fraction], Any]],
        num_hard: int,
        q_order_half: int,
        cusp_idx: int = 0,
    ) -> Any | None:
        """Run the full Weyl-symmetry check on a collection of index entries.

        Parameters
        ----------
        entries : list of ``(m_ext, e_ext, RefinedIndexResult)`` triples.
        num_hard : int — number of hard edges (η variables).
        q_order_half : int — truncation order used when computing the entries.
        cusp_idx : int — which cusp to check for adjoint projection (default 0).

        Returns
        -------
        ABVectors if the Weyl vectors can be computed, else ``None``.
        """
        result = _wc_mod.run_weyl_checks(
            entries, num_hard, cusp_idx=cusp_idx, q_order_half=q_order_half,
        )
        ab = result.ab if (result.ab is not None and result.ab_valid) else None
        adjoint_pass: "bool | None" = (
            result.adjoint.is_pass if result.adjoint is not None else None
        )
        adjoint_value: "int | None" = (
            result.adjoint.projected_value if result.adjoint is not None else None
        )
        return ab, adjoint_pass, adjoint_value