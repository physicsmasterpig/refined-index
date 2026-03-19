#!/usr/bin/env python3
"""Debug: Compare refined|_{η=1} vs unrefined for slope 1/2 on m003."""
from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.dehn_filling import compute_filled_index
from manifold_index.core.refined_dehn_filling import (
    compute_filled_refined_index,
    hj_continued_fraction,
)

# Build the same NeumannZagierData used in tests
data = load_manifold("m003")
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)
P, Q = 1, 2
q_order = 10
eta_order = 6

print(f"Slope P/Q = {P}/{Q}")
print(f"HJ-CF = {hj_continued_fraction(P, Q)}")
print(f"q_order_half = {q_order}, eta_order = {eta_order}")
print()

# Unrefined
unrefined = compute_filled_index(nz, 0, P, Q, q_order_half=q_order, verbose=True)
print(f"\nUnrefined series: {dict(sorted(unrefined.series.items()))}")
print()

# Refined with verbose
# Re-run ℓ≥2 pipeline partially to capture intermediate state
from manifold_index.core.refined_dehn_filling import (
    compute_filled_refined_index, hj_continued_fraction, _apply_is_step,
    _enumerate_slope1_all, _refined_to_multi, _multi_add
)

hj = hj_continued_fraction(P, Q)
ell = len(hj)
qq_order = q_order
qq_internal = qq_order + 2 * eta_order

# Recreate the grid-scan/state exactly as in compute_filled_refined_index
print("Rebuilding grid scan to inspect intermediate state...")
state = {}
m_scan = 2 * qq_internal
e_scan = qq_internal
for m_i in range(-m_scan, m_scan + 1):
    for e_half in range(-2 * e_scan, 2 * e_scan + 1):
        e_i = Fraction(e_half, 2)
        m_ext, e_ext = None, None
        # build m_ext/e_ext
        other_m_iter = iter([0] * (nz.r - 1))
        other_e_iter = iter([0] * (nz.r - 1))
        m_ext = []
        e_ext = []
        for k_idx in range(nz.r):
            if k_idx == 0:
                m_ext.append(m_i)
                e_ext.append(e_i)
            else:
                m_ext.append(next(other_m_iter))
                e_ext.append(next(other_e_iter))

        refined_idx = None
        from manifold_index.core.refined_index import compute_refined_index
        refined_idx = compute_refined_index(nz, m_ext, e_ext, q_order_half=qq_internal)
        if not refined_idx:
            continue
        multi = _refined_to_multi(refined_idx, append_cusp_eta=True)
        existing = state.get((m_i, e_i))
        state[(m_i, e_i)] = _multi_add(existing, multi) if existing else multi

print(f"Inspected state entries: {len(state)}")

# Compute total qq^0 sum in state when evaluating η=1 (before final K)
def multi_eta1_series(multi):
    res = {}
    for key, coeff in multi.items():
        qq_p = key[0]
        res[qq_p] = res.get(qq_p, Fraction(0)) + coeff
    return res

total_before = Fraction(0)
for (m1, e1), src in state.items():
    s = multi_eta1_series(src)
    total_before += s.get(0, Fraction(0))
print(f"Total qq^0 in state before final K (sum over all src series at η=1): {total_before}")

# Now apply final K as compute_filled_refined_index does and inspect
final_terms = _enumerate_slope1_all(hj[-1], 2 * q_order)
final_term_info = {}
seen = set()
for m1, e1, c_final, phase_final in final_terms:
    key = (m1, e1)
    if key in seen:
        continue
    seen.add(key)
    final_term_info[key] = (c_final, phase_final, 1)

matched = 0
unmatched = 0
total_after_apply = Fraction(0)
from manifold_index.core.refined_dehn_filling import _apply_k1_factor_multi, _multi_add
for (m1, e1), src_series in state.items():
    info = final_term_info.get((m1, e1))
    if info is None:
        unmatched += 1
        continue
    matched += 1
    c_final, phase_final, mult_final = info
    contribution = _apply_k1_factor_multi(src_series, c_final, phase_final, mult_final, qq_internal)
    # evaluate contribution at η=1
    s = multi_eta1_series(contribution)
    total_after_apply += s.get(0, Fraction(0))

print(f"Final K matched entries: {matched}, unmatched: {unmatched}")
print(f"Total qq^0 after applying final K: {total_after_apply}")

# Now get the final FilledRefinedResult and its eta1 series for comparison
refined = compute_filled_refined_index(nz, 0, P, Q, q_order_half=q_order, eta_order=eta_order, verbose=False)
eta1 = refined.eta1_series()
print("Refined|_η=1:")
for k, v in sorted(eta1.items()):
    print(f"  qq^{k:2d}: {v}")
print("Unrefined:")
for k, v in sorted(unrefined.series.items()):
    print(f"  qq^{k:2d}: {v}")
print()

# Detailed diagnostic for qq^0
qq = 0
c_ref = eta1.get(qq, Fraction(0))
c_unref = unrefined.series.get(qq, Fraction(0))
print(f"Diagnostic qq^0: refined|_η=1 = {c_ref}, unrefined = {c_unref}")

if c_ref != c_unref:
    print("Mismatch at qq^0. Inspecting contributions:")
    # Show which kernel/unrefined terms contributed to qq^0 for clarity
    print("  (Already printed unrefined series from compute_filled_index above.)")
    print("  For refined, list truncated final_terms keys count and sample: (first 10 keys)")
    # Recompute parts to extract detailed contributions (reuse previous variables?)
    # As a quick heuristic, print number of multi-eta entries that have qq^0.
    ref_q0_keys = [k for k in refined.series.keys() if k[0] == 0]
    print(f"    refined multi-η entries with qq^0: {len(ref_q0_keys)}")
    for k in ref_q0_keys[:10]:
        print(f"      key={k}, coeff={refined.series[k]}")
