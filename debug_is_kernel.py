#!/usr/bin/env python3
"""Trace IS kernel computation for the simplest case: HJ-CF [0,0]."""
from fractions import Fraction
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.refined_index import compute_refined_index
from manifold_index.core.refined_dehn_filling import (
    _is_kernel,
    _enumerate_slope1_all,
    _apply_k1_factor_multi,
    _refined_to_multi,
    _multi_convolve_is,
    _multi_add,
)

data = load_manifold("m003")
easy = find_easy_edges(data)
nz = build_neumann_zagier(data, easy)

qq_order = 6
eta_order = 5

# For HJ-CF [0,0], ℓ=2:
# Step 3: Grid scan — collect all non-zero I^ref(m, e)
# Step 4: One IS step with k_current=0, k_next=0
# Step 5: Final K(0, 1) factor

# Let's trace the IS kernel for a few (m, e) → (m1, e1) pairs
# IS kernel arguments: I_S(m, -e - k_current/2·m, m1, e1)
# With k_current=0: I_S(m, -e, m1, e1)

# First, let's see what I^ref values we have
print("Non-zero I^ref(m, e) at cusp (m_other=0, e_other=0):")
print("=" * 60)

state = {}
for m_i in range(-4, 5):
    for e_half in range(-8, 9):
        e_i = Fraction(e_half, 2)
        m_ext = [m_i]
        e_ext = [e_i]
        refined = compute_refined_index(nz, m_ext, e_ext, q_order_half=qq_order)
        if refined:
            multi = _refined_to_multi(refined, append_cusp_eta=True)
            state[(m_i, e_i)] = multi
            # Show first few terms
            terms = sorted(refined.items())[:3]
            print(f"  (m={m_i}, e={e_i}): {len(refined)} terms, first: {terms}")

print(f"\n  Total non-zero (m,e) pairs: {len(state)}")

# Now trace IS kernel for specific cases
print("\n" + "=" * 60)
print("IS kernel samples: I_S(m, -e, m1, e1)")
print("=" * 60)

# For I^ref_{1/0} = 0, we need Σ over all (m,e),(m1,e1):
# Σ K(0,1;m1,e1) · Σ_{(m,e)} I_S(m, -e, m1, e1) · I^ref(m, e)
# = Σ K(0,1;m1,e1) · state_new[(m1,e1)]

# Let's manually compute state_new for a few targets
targets = [(0, Fraction(0)), (1, Fraction(0)), (0, Fraction(1)), (0, Fraction(-1))]

for m1, e1 in targets:
    print(f"\n  Target (m1={m1}, e1={e1}):")
    total_is_at_eta1 = Fraction(0)
    for (m, e), src in state.items():
        e_in = -e  # k_current = 0
        is_val = _is_kernel(m, e_in, m1, e1, qq_order, eta_order)
        if is_val:
            # Sum IS at η=1 (sum all η components)
            is_eta1 = {}
            for (qq_p, eta_exp), c in is_val.items():
                is_eta1[qq_p] = is_eta1.get(qq_p, Fraction(0)) + c
            
            # Show first few IS values
            is_terms = sorted(is_eta1.items())[:5]
            print(f"    I_S(m={m}, e_in={e_in}, {m1}, {e1}) η=1: {is_terms[:3]}...")

# Let's also check: what does the IS kernel at η=1 give for 
# I_S(0, 0, 0, 0)?
print("\n" + "=" * 60)
print("IS kernel I_S(0, 0, 0, 0) full:")
print("=" * 60)
is_00 = _is_kernel(0, Fraction(0), 0, Fraction(0), qq_order, eta_order)
for (qq_p, eta_exp), c in sorted(is_00.items()):
    print(f"  qq^{qq_p} · η^{eta_exp}: {c}")
print(f"  Sum at η=1: qq-series:")
is_eta1_00 = {}
for (qq_p, eta_exp), c in is_00.items():
    is_eta1_00[qq_p] = is_eta1_00.get(qq_p, Fraction(0)) + c
for qq_p in sorted(is_eta1_00.keys()):
    if is_eta1_00[qq_p] != 0:
        print(f"    qq^{qq_p}: {is_eta1_00[qq_p]}")

print("\n" + "=" * 60)
print("IS kernel I_S(1, 0, 0, 0) full:")
print("=" * 60)
is_10 = _is_kernel(1, Fraction(0), 0, Fraction(0), qq_order, eta_order)
for (qq_p, eta_exp), c in sorted(is_10.items()):
    print(f"  qq^{qq_p} · η^{eta_exp}: {c}")
