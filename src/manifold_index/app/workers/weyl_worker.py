"""app/workers/weyl_worker.py — QThread workers for Weyl-symmetry checks.

``WeylWorker``
    Runs ``ComputeService.run_weyl_check`` on pre-computed entries.

``NcCompatWorker``
    Combined worker for per-NC-cycle compatibility checks: applies the
    basis change, recomputes index entries, then runs the Weyl check —
    all off the main thread so the UI stays responsive.

``MultiCuspNcCompatWorker``
    Joint adjoint-projection check for d simultaneously filled cusps.
    Applies basis changes for ALL filled cusps, computes index entries at
    all 4^d combinations of (e_1,…,e_d) ∈ {−2,−1,+1,+2}^d with unfilled
    cusps at e=0, then runs the multi-cusp adjoint integration.

Finished payload (all workers)::

    {
        "ab_vectors": ABVectors | None,
        "adjoint_is_pass": bool | None,
        "adjoint_value": float | None,
    }
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from PySide6.QtCore import QThread, Signal

from manifold_index.services.compute_service import ComputeService


class WeylWorker(QThread):
    """Run the Weyl-symmetry check on a collection of index entries."""

    status   = Signal(str)
    progress = Signal(int, int)
    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        entries: list[tuple[list[int], list[Fraction], Any]],
        num_hard: int,
        q_order_half: int,
        cusp_idx: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._entries      = entries
        self._num_hard     = num_hard
        self._q_order_half = q_order_half
        self._cusp_idx     = cusp_idx

    def run(self) -> None:
        try:
            self.status.emit("Running Weyl symmetry check…")
            ab_vectors, adjoint_is_pass, adjoint_value = ComputeService.run_weyl_check(
                self._entries,
                self._num_hard,
                self._q_order_half,
                self._cusp_idx,
            )
            self.finished.emit({
                "ab_vectors": ab_vectors,
                "adjoint_is_pass": adjoint_is_pass,
                "adjoint_value": adjoint_value,
            })
        except Exception as exc:
            self.error.emit(str(exc))


class NcCompatWorker(QThread):
    """Basis-change + index recomputation + Weyl check for one NC cycle.

    Runs entirely off the main thread so the UI stays responsive during
    per-cycle compatibility checks after NC search.
    """

    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        nz_data: Any,
        P: int,
        Q: int,
        cusp_idx: int,
        index_queries: list,
        num_hard: int,
        q_order_half: int,
        parent=None,
        *,
        # v1.1: pass-through to enable hard-edge basis optimisation.
        # All three (manifold_name, manifold_data, easy_result) must be
        # provided for optimisation to run; otherwise the worker falls
        # back to the default basis unchanged.  ``manifold_data`` MUST
        # be a thread-safe copy with ``raw=None`` (SnaPy's SQLite session
        # is thread-bound — calling load_manifold() here would silently
        # raise and disable the optimiser).
        manifold_name: "str | None" = None,
        manifold_data: Any = None,
        easy_result: Any = None,
        optimise_basis: bool = True,
        optimise_coeff_range: int = 1,
    ) -> None:
        super().__init__(parent)
        self._nz_data       = nz_data
        self._P             = P
        self._Q             = Q
        self._cusp_idx      = cusp_idx
        self._index_queries = index_queries
        self._num_hard      = num_hard
        self._q_order_half  = q_order_half
        self._manifold_name = manifold_name
        self._manifold_data = manifold_data
        self._easy_result   = easy_result
        self._optimise_basis = optimise_basis
        self._optimise_coeff_range = optimise_coeff_range

    def run(self) -> None:
        try:
            from manifold_index.core import (          # noqa: PLC0415
                dehn_filling as _df,
                neumann_zagier as _nz,
            )

            # ── v1.1: hard-edge basis optimisation (optional) ───────────
            #
            # When the caller supplied ``manifold_name`` + ``easy_result``,
            # search the unimodular basis space for an optimised hard
            # basis that yields max refinement subject to integer adjoint
            # projection.  If an improvement is found, swap nz_data for
            # the rebuilt one and continue normally — every subsequent
            # step (basis change, probe grid, Weyl check, marginal check)
            # then operates on the optimised basis transparently.
            basis_optimised = False
            basis_G: "list[list[int]] | None" = None
            default_refinement: "int | None" = None
            optimised_refinement: "int | None" = None
            optimised_easy_result: Any = None  # for FillWorker to rebuild nz
            opt_diag: dict = {}

            if (self._optimise_basis and self._manifold_data is not None
                    and self._easy_result is not None):
                try:
                    from manifold_index.core.optimal_basis import (  # noqa: PLC0415
                        find_optimal_hard_basis,
                    )
                    md = self._manifold_data  # thread-safe copy (raw=None)
                    opt = find_optimal_hard_basis(
                        md, self._easy_result, self._cusp_idx,
                        self._P, self._Q,
                        q_order_half=self._q_order_half,
                        coeff_range=self._optimise_coeff_range,
                    )
                    if opt is not None:
                        # Rebuild nz_data with the optimised basis
                        new_nz = _nz.build_neumann_zagier(md, opt.new_easy_result)
                        self._nz_data = new_nz
                        basis_optimised = True
                        basis_G = opt.G
                        default_refinement = opt.default_refinement
                        optimised_refinement = opt.refinement
                        # Pass the EasyEdgeResult back so the fill workers
                        # can rebuild the same nz_data for the actual
                        # filled-index computation.  Without this the
                        # FillWorker uses ``s.nz_data`` (default basis)
                        # while the (a, b) come from the optimised basis,
                        # causing an η-index mismatch and producing the
                        # wrong refinement count in the output.
                        optimised_easy_result = opt.new_easy_result
                        opt_diag = {
                            "n_searched": opt.n_candidates_searched,
                            "n_verified": opt.n_candidates_verified,
                        }
                except Exception as _exc:
                    # Surface failures to stderr so users can report what
                    # went wrong, but never block the user — fall back to
                    # the default basis.
                    import traceback as _tb  # noqa: PLC0415
                    print(f"[WEYL-NC] basis-opt skipped: {_exc}", flush=True)
                    _tb.print_exc()

            R, S = _df.find_rs(self._P, self._Q)
            nz_nc = _nz.apply_general_cusp_basis_change(
                self._nz_data, self._cusp_idx,
                a=self._P, b=self._Q, c=-R, d=-S,
            )
            from fractions import Fraction as _Frac  # noqa: PLC0415

            # ── User-grid entries (basis-changed) ────────────────────────
            seen: set[tuple] = set()
            entries_nc: list = []
            for q in self._index_queries:
                if q.result is not None:
                    result_nc = ComputeService.compute_refined_index(
                        nz_nc, q.m_ext, q.e_ext, self._q_order_half
                    )
                    if result_nc is not None:
                        key = (tuple(q.m_ext), tuple(q.e_ext))
                        seen.add(key)
                        entries_nc.append((q.m_ext, q.e_ext, result_nc))

            # ── Dedicated (a, b) probe grid for cusp 0 in NC basis ──────
            # v0.4 used compute_ab_vectors_for_cusp() which always computed
            # a fresh 5×5 probe grid.  We replicate that here so b-extraction
            # works even when the user's index_queries have no e=0 entries.
            #
            # For a:  m=0, e ∈ {−2,−1,−½,+½,+1,+2}
            # For b:  e=0, m ∈ {−2,−1,+1,+2}
            n_cusps = len(self._nz_data.e_ext_size) if hasattr(self._nz_data, 'e_ext_size') else 1
            try:
                n_cusps = self._nz_data.r
            except AttributeError:
                n_cusps = 1
            # a-probe: m=0, e varies (integers + half-integers) at target cusp
            for e_val in (_Frac(-2), _Frac(-1), _Frac(-1, 2),
                          _Frac(1, 2), _Frac(1), _Frac(2)):
                m_ext = [0] * n_cusps
                e_ext = [_Frac(0)] * n_cusps
                e_ext[self._cusp_idx] = e_val
                key = (tuple(m_ext), tuple(e_ext))
                if key not in seen:
                    result_nc = ComputeService.compute_refined_index(
                        nz_nc, m_ext, e_ext, self._q_order_half
                    )
                    if result_nc is not None:
                        seen.add(key)
                        entries_nc.append((m_ext, e_ext, result_nc))
            # b-probe: e=0, m varies at target cusp
            for m_val in (-2, -1, 1, 2):
                m_ext = [0] * n_cusps
                e_ext = [_Frac(0)] * n_cusps
                m_ext[self._cusp_idx] = m_val
                key = (tuple(m_ext), tuple(e_ext))
                if key not in seen:
                    result_nc = ComputeService.compute_refined_index(
                        nz_nc, m_ext, e_ext, self._q_order_half
                    )
                    if result_nc is not None:
                        seen.add(key)
                        entries_nc.append((m_ext, e_ext, result_nc))
            # Cross-cusp a-probe (multi-cusp only): m=0 everywhere,
            # e≠0 at each OTHER cusp.  Needed for the u-independence
            # condition I(m=0, e) = 0 for ALL e ≠ 0.
            if n_cusps > 1:
                for ci in range(n_cusps):
                    if ci == self._cusp_idx:
                        continue
                    for e_val in (_Frac(-1), _Frac(1)):
                        m_ext = [0] * n_cusps
                        e_ext = [_Frac(0)] * n_cusps
                        e_ext[ci] = e_val
                        key = (tuple(m_ext), tuple(e_ext))
                        if key not in seen:
                            result_nc = ComputeService.compute_refined_index(
                                nz_nc, m_ext, e_ext, self._q_order_half
                            )
                            if result_nc is not None:
                                seen.add(key)
                                entries_nc.append((m_ext, e_ext, result_nc))

            if not entries_nc:
                self.finished.emit({
                    "ab_vectors": None,
                    "adjoint_is_pass": None,
                    "adjoint_value": None,
                    "is_marginal": None,
                    "unrefined_q1_proj": None,
                    "basis_optimised": basis_optimised,
                    "basis_G": basis_G,
                    "default_refinement": default_refinement,
                    "optimised_refinement": optimised_refinement,
                    "optimised_easy_result": optimised_easy_result,
                    "basis_opt_diag": opt_diag,
                })
                return

            ab_vectors, adjoint_is_pass, adjoint_value = ComputeService.run_weyl_check(
                entries_nc, self._num_hard, self._q_order_half, self._cusp_idx,
            )

            # ── u-independence override for adjoint check ────────────────
            # If I(m=0, e) = 0 for all e ≠ 0 (at ANY cusp), the index is
            # u-independent and the adjoint check passes automatically.
            # For multi-cusp: e≠0 means any component of the e-vector is nonzero.
            try:
                a_probe_all_zero = True
                for m_ext_p, e_ext_p, res_p in entries_nc:
                    # a-probe entries: m=0 at all cusps, e≠0 at any cusp
                    if all(mv == 0 for mv in m_ext_p):
                        if any(ev != 0 for ev in e_ext_p) and res_p is not None:
                            if hasattr(res_p, 'coeffs'):
                                if any(c != 0 for c in res_p.coeffs):
                                    a_probe_all_zero = False
                                    break
                            elif res_p:  # non-zero result
                                a_probe_all_zero = False
                                break
                if a_probe_all_zero and not adjoint_is_pass:
                    adjoint_is_pass = True
            except Exception:
                pass

            # For multi-cusp manifolds, ab.a/b default to cusp 0's column.
            # Extract the correct cusp's column so callers get the right values.
            if (ab_vectors is not None
                    and ab_vectors.cusp_columns is not None
                    and self._cusp_idx < len(ab_vectors.cusp_columns)):
                col = ab_vectors.cusp_columns[self._cusp_idx]
                ab_vectors = type(ab_vectors)(
                    a=list(col.a),
                    b=list(col.b),
                    num_hard=ab_vectors.num_hard,
                    num_cusps=ab_vectors.num_cusps,
                    cusp_columns=ab_vectors.cusp_columns,
                    warnings=ab_vectors.warnings,
                )

            # ── Unrefined q^1 projection (marginal check) ───────────────
            # Compute I^{3D}(m=0, e) = I^ref(m,e; η=1) directly.
            # An NC cycle is *marginal* iff the adjoint SU(2) projection
            # of the unrefined q^1 coefficient is ≥ 0.  When marginal the
            # unrefined filling kernel K(P,Q) is used instead of K^ref.
            from manifold_index.core.index_3d import (  # noqa: PLC0415
                compute_index_3d_python,
            )
            from manifold_index.core.weyl_check import (  # noqa: PLC0415
                AdjointProjectionResult,
            )
            try:
                n_cusps = nz_nc.r
                c_e_snc: dict[_Frac, int] = {}
                needed = [_Frac(-2), _Frac(-1), _Frac(1), _Frac(2)]
                # Track whether the full series is zero for each nonzero e
                all_nonzero_e_vanish = True
                for e_val in needed:
                    m_ext_s = [0] * n_cusps
                    e_ext_s = [_Frac(0)] * n_cusps
                    e_ext_s[self._cusp_idx] = e_val
                    r3d = compute_index_3d_python(
                        nz_nc, m_ext_s, e_ext_s, self._q_order_half
                    )
                    # Check if the entire series is zero for this e≠0
                    if any(c != 0 for c in r3d.coeffs):
                        all_nonzero_e_vanish = False
                    # q^1 = qq^2; account for min_power offset
                    idx = 2 - r3d.min_power
                    q1_coeff = (
                        r3d.coeffs[idx]
                        if 0 <= idx < len(r3d.coeffs)
                        else 0
                    )
                    c_e_snc[e_val] = q1_coeff

                # Multi-cusp: also check cross-cusp e directions
                # I(m=0, e) must be 0 for e≠0 at ANY cusp, not just target.
                if n_cusps > 1 and all_nonzero_e_vanish:
                    for ci in range(n_cusps):
                        if ci == self._cusp_idx:
                            continue
                        for e_val in (_Frac(-1), _Frac(1)):
                            m_ext_s = [0] * n_cusps
                            e_ext_s = [_Frac(0)] * n_cusps
                            e_ext_s[ci] = e_val
                            r3d = compute_index_3d_python(
                                nz_nc, m_ext_s, e_ext_s, self._q_order_half
                            )
                            if any(c != 0 for c in r3d.coeffs):
                                all_nonzero_e_vanish = False
                                break
                        if not all_nonzero_e_vanish:
                            break

                missing = [e for e in needed if e not in c_e_snc]
                if missing:
                    is_marginal      = None
                    unrefined_q1_proj = None
                else:
                    num = (c_e_snc[_Frac(-1)] + c_e_snc[_Frac(1)]
                           - c_e_snc[_Frac(-2)] - c_e_snc[_Frac(2)])
                    if num % 2 != 0:
                        is_marginal      = None
                        unrefined_q1_proj = None
                    else:
                        unrefined_q1_proj = num // 2
                        # If I(m=0, e) = 0 for all sampled e≠0, the index
                        # is u-independent → NOT marginal.  Torus knots
                        # (3_1, 5_1, 7_1, …) fall into this category.
                        is_marginal = False if all_nonzero_e_vanish else (unrefined_q1_proj >= 0)
            except Exception:
                is_marginal      = None
                unrefined_q1_proj = None

            self.finished.emit({
                "ab_vectors":       ab_vectors,
                "adjoint_is_pass":  adjoint_is_pass,
                "adjoint_value":    adjoint_value,
                "is_marginal":      is_marginal,
                "unrefined_q1_proj": unrefined_q1_proj,
                "basis_optimised":   basis_optimised,
                "basis_G":           basis_G,
                "default_refinement":   default_refinement,
                "optimised_refinement": optimised_refinement,
                "optimised_easy_result": optimised_easy_result,
                "basis_opt_diag":    opt_diag,
            })
        except Exception as exc:
            self.error.emit(str(exc))


class MultiCuspNcCompatWorker(QThread):
    """Joint adjoint projection check for d simultaneously filled cusps.

    For d cusps being filled, the q¹ adjoint projection must integrate over
    ALL d filled-cusp fugacities jointly — not one at a time.  This worker:

    1. Applies basis changes for every cusp in *cusp_specs* (each cusp's NC
       cycle becomes the new meridian).
    2. Computes index entries at m=0 for all 4^d combinations of
       (e_{n+1},…,e_{n+d}) ∈ {−2,−1,+1,+2}^d with all unfilled cusp charges
       set to zero.
    3. Recomputes any existing user-grid entries in the new (NC) basis so
       that ``compute_ab_vectors`` can extract Weyl (a, b) vectors.
    4. Runs ``run_weyl_check`` with *filled_cusp_indices* so that the
       adjoint check uses ``check_adjoint_projection_multi_cusp``.

    Parameters
    ----------
    nz_data
        NeumannZagierData for the manifold (original basis).
    cusp_specs : list[dict]
        One dict per filled cusp; each must contain
        ``{"cusp_idx": int, "nc_P": int, "nc_Q": int}``.
    index_queries : list
        Pre-computed index queries used to extract Weyl (a, b) vectors.
    num_hard : int
    q_order_half : int
    """

    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        nz_data: Any,
        cusp_specs: list,
        index_queries: list,
        num_hard: int,
        q_order_half: int,
        parent=None,
        *,
        # v1.1: pass-through for hard-edge basis optimisation.
        # ``manifold_data`` MUST be a thread-safe copy with raw=None
        # (see NcCompatWorker for rationale).
        manifold_name: "str | None" = None,
        manifold_data: Any = None,
        easy_result: Any = None,
        optimise_basis: bool = True,
        optimise_coeff_range: int = 1,
    ) -> None:
        super().__init__(parent)
        self._nz_data       = nz_data
        self._cusp_specs    = cusp_specs
        self._index_queries = index_queries
        self._num_hard      = num_hard
        self._q_order_half  = q_order_half
        self._manifold_name = manifold_name
        self._manifold_data = manifold_data
        self._easy_result   = easy_result
        self._optimise_basis = optimise_basis
        self._optimise_coeff_range = optimise_coeff_range

    def run(self) -> None:
        try:
            from itertools import product as _product              # noqa: PLC0415
            from fractions import Fraction as _Frac                # noqa: PLC0415
            from manifold_index.core import (                      # noqa: PLC0415
                dehn_filling as _df,
                neumann_zagier as _nz,
            )
            from manifold_index.core.weyl_check import run_weyl_checks  # noqa: PLC0415

            # ── v1.1: multi-cusp hard-edge basis optimisation ───────────
            basis_optimised = False
            basis_G: "list[list[int]] | None" = None
            default_refinement: "int | None" = None
            optimised_refinement: "int | None" = None
            optimised_easy_result: Any = None
            opt_diag: dict = {}

            if (self._optimise_basis and self._manifold_data is not None
                    and self._easy_result is not None):
                try:
                    from manifold_index.core.optimal_basis import (  # noqa: PLC0415
                        find_optimal_hard_basis_multi,
                    )
                    md = self._manifold_data  # thread-safe copy (raw=None)
                    opt = find_optimal_hard_basis_multi(
                        md, self._easy_result, self._cusp_specs,
                        q_order_half=self._q_order_half,
                        coeff_range=self._optimise_coeff_range,
                    )
                    if opt is not None:
                        new_nz = _nz.build_neumann_zagier(md, opt.new_easy_result)
                        self._nz_data = new_nz
                        basis_optimised = True
                        basis_G = opt.G
                        default_refinement = opt.default_refinement
                        optimised_refinement = opt.refinement
                        optimised_easy_result = opt.new_easy_result
                        opt_diag = {
                            "n_searched": opt.n_candidates_searched,
                            "n_verified": opt.n_candidates_verified,
                        }
                except Exception as _exc:
                    import traceback as _tb  # noqa: PLC0415
                    print(f"[MULTI-NC] basis-opt skipped: {_exc}", flush=True)
                    _tb.print_exc()

            # ── Apply basis changes for all filled cusps ─────────────
            nz_nc = self._nz_data
            for spec in self._cusp_specs:
                P, Q     = spec["nc_P"], spec["nc_Q"]
                cusp_idx = spec["cusp_idx"]
                R, S     = _df.find_rs(P, Q)
                nz_nc = _nz.apply_general_cusp_basis_change(
                    nz_nc, cusp_idx,
                    a=P, b=Q, c=-R, d=-S,
                )

            n_cusps = nz_nc.r
            filled_cusp_indices = [spec["cusp_idx"] for spec in self._cusp_specs]
            d = len(filled_cusp_indices)

            # ── Recompute existing user-grid entries in the NC basis ──
            # (needed so compute_ab_vectors can extract Weyl vectors)
            seen: set[tuple] = set()
            entries_nc: list = []
            for q in self._index_queries:
                if q.result is not None:
                    result_nc = ComputeService.compute_refined_index(
                        nz_nc, q.m_ext, q.e_ext, self._q_order_half
                    )
                    if result_nc is not None:
                        key = (tuple(q.m_ext), tuple(q.e_ext))
                        seen.add(key)
                        entries_nc.append((q.m_ext, q.e_ext, result_nc))

            # ── Dedicated (a, b) probe grid for each filled cusp ────────
            # For each filled cusp I, add:
            #   a-probe: only cusp I varies in e (integers + half-integers),
            #            all other cusps at (m=0, e=0)
            #   b-probe: only cusp I varies in m, e=0 everywhere
            # This mirrors v0.4's compute_ab_vectors_for_cusp() approach and
            # ensures b can be extracted even when the user's grid lacks e=0.
            _a_probe_e = [_Frac(-2), _Frac(-1), _Frac(-1, 2),
                          _Frac(1, 2), _Frac(1), _Frac(2)]
            _b_probe_m = [-2, -1, 1, 2]

            for ci in filled_cusp_indices:
                # a-probe
                for e_val in _a_probe_e:
                    m_ext = [0] * n_cusps
                    e_ext = [_Frac(0)] * n_cusps
                    e_ext[ci] = e_val
                    key = (tuple(m_ext), tuple(e_ext))
                    if key not in seen:
                        result_nc = ComputeService.compute_refined_index(
                            nz_nc, m_ext, e_ext, self._q_order_half
                        )
                        if result_nc is not None:
                            seen.add(key)
                            entries_nc.append((m_ext, e_ext, result_nc))
                # b-probe
                for m_val in _b_probe_m:
                    m_ext = [0] * n_cusps
                    e_ext = [_Frac(0)] * n_cusps
                    m_ext[ci] = m_val
                    key = (tuple(m_ext), tuple(e_ext))
                    if key not in seen:
                        result_nc = ComputeService.compute_refined_index(
                            nz_nc, m_ext, e_ext, self._q_order_half
                        )
                        if result_nc is not None:
                            seen.add(key)
                            entries_nc.append((m_ext, e_ext, result_nc))

            # ── Required adjoint-check points ────────────────────────
            # For each target cusp I:
            #   e_I ∈ {−2,−1,+1,+2}  (K_target nonzero)
            #   e_J ∈ {−1, 0, +1}    (K_other nonzero) for all J ≠ I
            # We compute all combinations from the union over all targets.
            e_target = [_Frac(-2), _Frac(-1), _Frac(1), _Frac(2)]
            e_other  = [_Frac(-1), _Frac(0),  _Frac(1)]

            for target_idx in range(d):
                for e_t in e_target:
                    for e_o_combo in _product(e_other, repeat=d - 1):
                        m_ext = [0] * n_cusps
                        e_ext = [_Frac(0)] * n_cusps
                        o = 0
                        for j, ci in enumerate(filled_cusp_indices):
                            if j == target_idx:
                                e_ext[ci] = e_t
                            else:
                                e_ext[ci] = e_o_combo[o]
                                o += 1
                        key = (tuple(m_ext), tuple(e_ext))
                        if key not in seen:
                            result_nc = ComputeService.compute_refined_index(
                                nz_nc, m_ext, e_ext, self._q_order_half
                            )
                            if result_nc is not None:
                                seen.add(key)
                                entries_nc.append((m_ext, e_ext, result_nc))

            if not entries_nc:
                self.finished.emit({
                    "ab_vectors": None,
                    "adjoint_is_pass": None,
                    "adjoint_value": None,
                    "per_cusp_adjoint": [],
                    "basis_optimised": basis_optimised,
                    "basis_G": basis_G,
                    "default_refinement": default_refinement,
                    "optimised_refinement": optimised_refinement,
                    "optimised_easy_result": optimised_easy_result,
                    "basis_opt_diag": opt_diag,
                })
                return

            wc = run_weyl_checks(
                entries_nc,
                self._num_hard,
                filled_cusp_indices=filled_cusp_indices,
            )

            ab = wc.ab  # Return ab even if some edges are incompatible;
            # callers use ab.edge_compatible / make_filling_compatible().

            # Build per-cusp adjoint summary for the UI
            per_cusp: list[dict] = []
            if wc.multi_cusp_adjoint is not None:
                for idx, r in enumerate(wc.multi_cusp_adjoint.results):
                    ci = filled_cusp_indices[idx] if idx < len(filled_cusp_indices) else idx
                    per_cusp.append({
                        "cusp_idx": ci,
                        "is_pass": r.is_pass,
                        "value": r.projected_value,
                    })

            self.finished.emit({
                "ab_vectors": ab,
                "adjoint_is_pass": wc.multi_cusp_adjoint.all_pass if wc.multi_cusp_adjoint else None,
                "adjoint_value": None,   # not meaningful as a single number for d>1
                "per_cusp_adjoint": per_cusp,
                "basis_optimised": basis_optimised,
                "basis_G": basis_G,
                "default_refinement": default_refinement,
                "optimised_refinement": optimised_refinement,
                "optimised_easy_result": optimised_easy_result,
                "basis_opt_diag": opt_diag,
            })
        except Exception as exc:
            self.error.emit(str(exc))
