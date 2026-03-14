"""
Two checks:
1. Verify I_{3D}(m,e) == I_{3D}(-m,-e) exactly for all c=0 kernel terms
   used in the (1,2) computation on m003.
2. Cross-validate: manually sum both +t and -t directions without
   multiplicity, compare result to the optimized computation.
"""
from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.dehn_filling import (
    KernelTerm,
    find_rs, enumerate_kernel_terms,
    _apply_kernel, _qseries_from_result, _qseries_add, _qseries_truncate,
    _qseries_scale,
)
from manifold_index.core.index_3d import compute_index_3d_python

data = load_manifold("m003")
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)

P, Q, q_ord = 1, 2, 20
R, S = find_rs(P, Q)
terms = enumerate_kernel_terms(P, Q, R, S, nz, 0, [], [], q_ord)

c0_terms = [t for t in terms if t.c == 0]
print(f"c=0 terms: {len(c0_terms)} (multiplicities: {set(t.multiplicity for t in c0_terms)})")

# CHECK 1: symmetry I(m,e) == I(-m,-e) for each c=0 term with phase > 0
print("\nCHECK 1: I_{3D}(m,e) == I_{3D}(-m,-e) for all c=0 terms with t>0")
all_symmetric = True
for kt in c0_terms:
    if kt.phase <= 0:
        continue
    idx_q = q_ord + abs(kt.phase)
    res_pos = compute_index_3d_python(nz, m_ext=[kt.m], e_ext=[kt.e], q_order_half=idx_q)
    res_neg = compute_index_3d_python(nz, m_ext=[-kt.m], e_ext=[-kt.e], q_order_half=idx_q)
    s_pos = {k: v for k, v in zip(
        range(res_pos.min_power, res_pos.min_power + len(res_pos.coeffs)),
        res_pos.coeffs) if v != 0}
    s_neg = {k: v for k, v in zip(
        range(res_neg.min_power, res_neg.min_power + len(res_neg.coeffs)),
        res_neg.coeffs) if v != 0}
    if s_pos != s_neg:
        print(f"  MISMATCH at phase={kt.phase}: m={kt.m}, e={kt.e}")
        print(f"    I(+m,+e) = {s_pos}")
        print(f"    I(-m,-e) = {s_neg}")
        all_symmetric = False
positive_t_count = sum(1 for t in c0_terms if t.phase > 0)
if all_symmetric:
    print(f"  OK: all {positive_t_count} positive-t terms satisfy I(m,e) == I(-m,-e) exactly")

# CHECK 2: brute-force re-sum both ±t directions without multiplicity, compare
print("\nCHECK 2: brute-force sum (both +t and -t, no multiplicity) vs optimized")

# Optimized result (what compute_filled_index does with our new code)
total_optimized: dict = {}
for kt in terms:
    idx_q = q_ord + (abs(kt.phase) if kt.c == 0 else 0)
    res = compute_index_3d_python(nz, m_ext=[kt.m], e_ext=[kt.e], q_order_half=idx_q)
    s = _qseries_from_result(res)
    contrib = _apply_kernel(kt, s)
    if kt.multiplicity != 1:
        contrib = _qseries_scale(contrib, Fraction(kt.multiplicity))
    total_optimized = _qseries_add(total_optimized, contrib)
total_optimized = _qseries_truncate(total_optimized, q_ord)

# Brute-force result (include both +t and -t, no multiplicity shortcut)
total_brute: dict = {}
for kt in terms:
    if kt.c != 0:
        idx_q = q_ord
        res = compute_index_3d_python(nz, m_ext=[kt.m], e_ext=[kt.e], q_order_half=idx_q)
        s = _qseries_from_result(res)
        contrib = _apply_kernel(kt, s)
        total_brute = _qseries_add(total_brute, contrib)
    else:
        # Include both (m,e,phase) AND (-m,-e,-phase) each with weight 1
        for sign in ([1] if kt.phase == 0 else [1, -1]):
            m_i = sign * kt.m
            e_i = Fraction(sign) * kt.e
            phase_i = sign * kt.phase
            idx_q = q_ord + abs(phase_i)
            res = compute_index_3d_python(nz, m_ext=[m_i], e_ext=[e_i], q_order_half=idx_q)
            s = _qseries_from_result(res)
            kt_i = KernelTerm(m=m_i, e=e_i, c=0, phase=phase_i)
            contrib = _apply_kernel(kt_i, s)
            total_brute = _qseries_add(total_brute, contrib)
total_brute = _qseries_truncate(total_brute, q_ord)

if total_optimized == total_brute:
    print("  OK: optimized == brute-force exactly (bit-for-bit identical Fraction values)")
else:
    print("  MISMATCH!")
    print(f"  optimized: {dict(sorted(total_optimized.items()))}")
    print(f"  brute:     {dict(sorted(total_brute.items()))}")
    diff = _qseries_add(total_optimized, {k: -v for k, v in total_brute.items()})
    nonzero_diff = {k: v for k, v in diff.items() if v != 0}
    print(f"  diff (nonzero): {dict(sorted(nonzero_diff.items()))}")

print("\nDone.")
