"""Manual Weyl symmetry check for m125 with per-cusp vectors."""
import sys; sys.path.insert(0, 'src')
from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.refined_index import compute_refined_index

name = 'm125'
md = load_manifold(name)
easy = find_easy_edges(md)
nz = build_neumann_zagier(md, easy)
qq = 20

# CORRECT per-cusp Weyl vectors derived from raw shift analysis:
#   a_cusp = [1, 0], b_cusp = [0, 0]
# Weyl monomial: eta^{1*e0 + 0*e1 + 0*m0 + 0*m1} = eta^{e0}
a_cusp = [Fraction(1), Fraction(0)]
b_cusp = [Fraction(0), Fraction(0)]

def apply_weyl_percusp(result, m_ext, e_ext, a_cusp, b_cusp):
    shift = sum(a_cusp[I] * e_ext[I] + b_cusp[I] * m_ext[I]
                for I in range(len(m_ext)))
    shift_x2 = 2 * shift
    if shift_x2.denominator != 1:
        return None  # fractional shift
    shift_x2 = int(shift_x2)
    shifted = {}
    for key, coeff in result.items():
        if coeff == 0:
            continue
        new_key = (key[0], key[1] + shift_x2)
        shifted[new_key] = shifted.get(new_key, 0) + coeff
    return {k: v for k, v in shifted.items() if v != 0}

n_pass = 0
n_fail = 0
n_zero = 0
n_frac = 0
fail_examples = []

for m0 in [-2, 0, 2]:
  for m1 in range(-3, 4):
    for e0_x2 in range(-3, 4):
      for e1_x2 in range(-4, 5):
        e0 = Fraction(e0_x2, 2)
        e1 = Fraction(e1_x2, 2)
        m_ext = [m0, m1]
        e_ext = [e0, e1]
        neg_m = [-x for x in m_ext]
        neg_e = [-x for x in e_ext]

        resA = compute_refined_index(nz, m_ext, e_ext, q_order_half=qq)
        resB = compute_refined_index(nz, neg_m, neg_e, q_order_half=qq)

        if not resA and not resB:
            n_zero += 1
            continue
        if not resA or not resB:
            n_fail += 1
            continue

        fA = apply_weyl_percusp(resA, m_ext, e_ext, a_cusp, b_cusp)
        fB = apply_weyl_percusp(resB, neg_m, neg_e, a_cusp, b_cusp)

        if fA is None or fB is None:
            n_frac += 1
            continue

        if fA == fB:
            n_pass += 1
        else:
            n_fail += 1
            if len(fail_examples) < 5:
                all_k = set(fA.keys()) | set(fB.keys())
                diffs = [(k, fA.get(k,0), fB.get(k,0))
                         for k in sorted(all_k)
                         if fA.get(k,0) != fB.get(k,0)]
                fail_examples.append((m_ext, e_ext, diffs[:3]))

print(f"Results with per-cusp a=[1,0], b=[0,0]:")
print(f"  PASS:  {n_pass}")
print(f"  FAIL:  {n_fail}")
print(f"  ZERO:  {n_zero}")
print(f"  FRAC:  {n_frac}")
print(f"  Non-zero pairs: {n_pass + n_fail + n_frac}")
print()

if fail_examples:
    print("First few failures:")
    for m, e, diffs in fail_examples:
        print(f"  m={m}, e={[str(x) for x in e]}")
        for k, va, vb in diffs:
            print(f"    f+[{k}]={va}, f-[{k}]={vb}")
else:
    print("ALL non-zero, integer-shift sectors PASS!")
