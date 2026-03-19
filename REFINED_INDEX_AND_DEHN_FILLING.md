# Refined 3D Index and Dehn Filling — Detailed Reference

> **Purpose.** This document is a self-contained mathematical reference
> for the refined 3D index and the Dehn filling operation on it,
> written to clarify the author's (the implementor's) understanding
> and for verification by the physicist collaborator.
>
> References:
> - **[DGG]** Dimofte–Gaiotto–Gukov, *3-Manifolds and 3d Indices*
> - **[GK]** Garoufalidis–Kim, *The 3D index of an ideal triangulation…*
> - **[CGK]** Chung–Gang–Kim, *Refined 3D index* (arXiv, Appendix A)

---

## Table of Contents

1. [Setup: Ideal Triangulation and Variables](#1-setup)
2. [The 3D Index I(m, e)](#2-the-3d-index)
3. [Dehn Filling of the 3D Index (Unrefined)](#3-dehn-filling-unrefined)
4. [Non-Closable Cycles](#4-non-closable-cycles)
5. [Easy and Hard Edges](#5-easy-and-hard-edges)
6. [The Refined 3D Index](#6-the-refined-3d-index)
7. [Weyl Symmetry and (a, b) Vectors](#7-weyl-symmetry)
8. [Refined Dehn Filling (The IS Chain)](#8-refined-dehn-filling)
9. [Multi-Cusp Manifolds](#9-multi-cusp-manifolds)
10. [What the Code Currently Does vs What It Should Do](#10-code-status)

---

## 1. Setup

### 1.1 The manifold

We work with a cusped hyperbolic 3-manifold **M** with an ideal
triangulation into **n** tetrahedra and **r** cusps.

### 1.2 Variables

Each tetrahedron `i` (i = 1, …, n) has three shape parameters
`Z_i, Z_i', Z_i''` satisfying `Z_i + Z_i' + Z_i'' = 1`
(in our normalization with `iπ` divided out).

After eliminating `Z_i'`, we have **2n** reduced variables:

```
v = (Z₁, …, Zₙ, Z₁'', …, Zₙ'')       [block ordering]
```

### 1.3 Symplectic structure

The commutation relation is `[Zᵢ, Zᵢ''] = 1`. In the 2n-dimensional
space this gives the symplectic form:

```
Ω = [[0ₙ,  Iₙ],
     [−Iₙ, 0ₙ]]
```

### 1.4 Neumann-Zagier matrix g_NZ

The matrix `g_NZ ∈ Sp(2n, ℤ)` is a symplectic basis change. Its rows are
organized as:

| Rows | Content | Count |
|------|---------|-------|
| 0 … r−1 | Meridians Mₖ (one per cusp) | r |
| r … r+d_hard−1 | Hard internal edges | d_hard |
| r+d_hard … n−1 | Easy internal edges | d_easy |
| n … n+r−1 | Half-longitudes Λₖ = Lₖ/2 | r |
| n+r … 2n−1 | Γ vectors (momentum conjugates of internal edges) | n−r |

where `d_hard + d_easy = n − r`.

The top n rows form the **position block** P; the bottom n rows form
the **momentum block** Q.

### 1.5 Affine shifts ν_x, ν_p

The variables satisfy:

```
g_NZ · v + ν = RHS
```

where the affine shift `ν = (ν_x, ν_p)` accounts for the constant terms
in the gluing equations.

---

## 2. The 3D Index

### 2.1 External variables

The 3D index is a function of **cusp charges** only:

```
I(m₁, e₁, m₂, e₂, …, mᵣ, eᵣ; q)
```

where:
- `mₖ ∈ ℤ` is the **meridian charge** at cusp k
- `eₖ ∈ (½)ℤ` is the **half-longitude charge** at cusp k
  (i.e., `eₖ` is conjugate to `Λₖ = Lₖ/2`, so `2eₖ` is the
  longitude charge)

We write this compactly as `I(m⃗, e⃗)` where `m⃗, e⃗` have length `r`.

### 2.2 Formula

```
I(m⃗, e⃗) = Σ_{e_int ∈ (½)ℤ^{n−r}}
    (−q^{½})^{m_full · ν_p − e_full · ν_x}
    · ∏_{a=0}^{n−1} I_Δ(tet_m_a, tet_e_a)
```

where:
- `m_full = (m⃗, 0^{n−r})` — internal edge meridians are 0
- `e_full = (e⃗, e_int)` — internal edge charges are summed over
- `κ = (m_full, e_full)` — the full 2n-vector
- `(tet_m_a, tet_e_a) = (g_NZ⁻¹ κ)_a, (g_NZ⁻¹ κ)_{n+a}` — the
  arguments passed to each tetrahedron index

### 2.3 Key point: I is a function of ALL cusp charges

For a manifold with r = 2 cusps, `I = I(m₁, e₁, m₂, e₂)` depends
on **four** integer/half-integer variables (plus q). It is NOT a
single number — it is a function.

---

## 3. Dehn Filling of the 3D Index (Unrefined)

### 3.1 What Dehn filling does

**Dehn filling cusp i with slope P/Q** means we impose the constraint
that the cycle `P·Mᵢ + Q·Lᵢ` is contractible. Algebraically, this
is implemented by **summing over** the charges `(mᵢ, eᵢ)` with the
Dehn filling kernel.

### 3.2 The kernel

```
K(P, Q; m, e) = ½ (−1)^{Rm+2Se} ·
    [ δ_{Pm+2Qe, 0} · (q^{(Rm+2Se)/2} + q^{−(Rm+2Se)/2})
      − δ_{Pm+2Qe, −2}
      − δ_{Pm+2Qe, 2} ]
```

where `R, S ∈ ℤ` satisfy `RQ − PS = 1`.

### 3.3 The filled index

```
I_{P/Q}^{(i)}(m_other, e_other) =
    Σ_{mᵢ ∈ ℤ, eᵢ ∈ (½)ℤ}  K(P, Q; mᵢ, eᵢ) · I(m₁, e₁, …, mᵣ, eᵣ)
```

where `(mⱼ, eⱼ)` for `j ≠ i` are held **fixed** at the values
`(m_other, e_other)`.

### 3.4 ⚠️ CRITICAL: The filled index is still a function

**When r ≥ 2 and we fill only cusp i, the result is still a function
of the unfilled cusps' charges:**

```
I_{P/Q}^{(i)} : (mⱼ, eⱼ)_{j≠i} → q-series
```

For a 2-cusp manifold, filling cusp 0 gives:

```
I_{P/Q}^{(0)}(m₁, e₁) = Σ_{m₀, e₀}  K(P,Q; m₀,e₀) · I(m₀, e₀, m₁, e₁)
```

This is a **q-series-valued function** of `(m₁, e₁)`.

### 3.5 What "m_other = 0, e_other = 0" means

In the code, `compute_filled_index(nz_data, cusp_idx, P, Q, m_other, e_other)`
accepts specific numerical values for the other cusps. The default is
`m_other = [0, …, 0]` and `e_other = [0, …, 0]`.

This means **the code evaluates the filled index at one specific point**
in the (m_other, e_other) space. It does NOT compute the filled index
as a function of the remaining variables. This is a deliberate choice:
for the non-closable cycle search, we only need to check vanishing at
one point (typically m_other = e_other = 0) as a necessary condition.

> **However**, for a complete check of non-closability, one should
> verify vanishing for ALL values of (m_other, e_other), not just zero.
> In practice, checking at zero is often sufficient, but it is an
> approximation.

---

## 4. Non-Closable Cycles

### 4.1 Definition

A cycle `P·M + Q·L` at cusp `i` is **non-closable** if:

```
I_{P/Q}^{(i)}(m_other, e_other) = 0
    for ALL values of (m_other, e_other)
```

This is the mathematical definition. A non-closable cycle is one where
the Dehn filling produces the zero q-series **regardless** of what the
other cusp charges are.

### 4.2 Practical test

In our code, we only test with `m_other = e_other = 0`. If
`I_{P/Q}^{(i)}(0, 0) = 0`, we declare the cycle non-closable.

For a 1-cusp manifold (r = 1), there are no other cusps, so this is
exact.

For a multi-cusp manifold (r ≥ 2), this is a **necessary but possibly
not sufficient** condition. A cycle could give zero at m_other = 0 but
be non-zero at other values. We accept this practical simplification.

### 4.3 Why non-closable cycles matter

Non-closable cycles are the candidates for the **basis** of the
refined index. The physical interpretation is that a non-closable cycle
corresponds to a boundary condition that can be "refined" — where the
Dehn filling operation is well-defined in the refined (η-dependent)
setting.

---

## 5. Easy and Hard Edges

### 5.1 Internal edges

The `n − r` internal edges of the triangulation are classified into:

- **Easy edges**: edges where at most one shape parameter is active per
  tetrahedron. These can be found algorithmically (pattern search over
  4ⁿ patterns).
- **Hard edges**: the remaining `d_hard = (n − r) − d_easy` edges that
  require padding from SnaPy edge rows.

### 5.2 Why the distinction matters

- Easy edge charges are **summed over** in the refined index, exactly as
  in the unrefined case. They contribute nothing new.
- Hard edge charges get a **fugacity variable** `ηₐ` each. Instead of
  summing over `e_{r+a}`, each term is weighted by `ηₐ^{e_{r+a}}`. The
  refined index becomes a multi-variable Laurent polynomial in q and
  {η₀, …, η_{d_hard−1}}.

### 5.3 Basis ordering in g_NZ

In the NZ matrix:

```
Position rows:
  [0, r)         : cusps (meridians)
  [r, r+d_hard)  : hard edges
  [r+d_hard, n)  : easy edges

Momentum rows:
  [n, n+r)       : cusps (half-longitudes)
  [n+r, 2n)      : Γ vectors (conjugate to internal edges)
```

The hard edges occupy the first `d_hard` internal-edge slots. Their
charges in `e_int` are `e_int[0], …, e_int[d_hard−1]`.

---

## 6. The Refined 3D Index

### 6.1 Formula

```
I^ref(m⃗, e⃗; q, η₀, …, η_{d_hard−1}) =
    Σ_{e_int ∈ (½)ℤ^{n−r}}
        [ ∏_{a=0}^{d_hard−1}  ηₐ^{e_int[a]} ]
        · (−q^{½})^{m_full · ν_p − e_full · ν_x}
        · ∏_{j=0}^{n−1} I_Δ(tet_m_j, tet_e_j)
```

### 6.2 Key properties

1. **Setting all ηₐ = 1 recovers the unrefined 3D index exactly.**
   This is because the η^{e_int[a]} factor becomes 1 and the remaining
   sum is identical to the unrefined formula.

2. **I^ref is still a function of ALL cusp charges (m⃗, e⃗).**
   Just like the unrefined index, the refined index depends on the
   meridian and half-longitude charges at every cusp.

3. **When d_hard = 0 (no hard edges), I^ref = I identically.**

### 6.3 Output format

The result is stored as:

```python
dict[tuple[int, ...], int]
# key   = (qq_power, 2·η₀_exp, 2·η₁_exp, …)
# value = integer coefficient
```

The factor of 2 in the η exponents is because half-integer exponents are
possible; storing `2 × exponent` keeps all keys integer.

---

## 7. Weyl Symmetry and (a, b) Vectors

### 7.1 Motivation

Before performing refined Dehn filling at a cusp, three conditions must
be verified:

1. **Non-closability** (the cycle P·M + Q·L gives zero unrefined index)
2. **Weyl symmetry** (the refined index, after multiplication by a
   suitable monomial, is invariant under ηₐ ↔ ηₐ⁻¹)
3. **Adjoint character** (the q¹ coefficient of the Weyl-manifest form
   equals η + 1 + η⁻¹ for each hard edge)

### 7.2 The (a, b) vectors

For each hard edge a, there exist:
- `bₐ ∈ (½)ℤ` (half-integer)
- `aₐ ∈ ℤ` (integer, but stored as `2·aₐ` in the code)

such that the **Weyl-manifest form**:

```
f(m⃗, e⃗) = ηₐ^{bₐ · m_sum + aₐ · e_sum} · I^ref(m⃗, e⃗)
```

is symmetric under `ηₐ ↔ ηₐ⁻¹`.

Here `m_sum = Σₖ mₖ` and `e_sum = Σₖ eₖ` (sums over all cusps).

### 7.3 Storage convention

In the code (`ABVectors`):
- `a[j]` stores `2 · aₐ` (always integer)
- `b[j]` stores `bₐ` directly (half-integer, as `Fraction`)

---

## 8. Refined Dehn Filling (The IS Chain)

### 8.1 When is the IS chain needed?

The **unrefined** Dehn filling kernel K(P, Q; m, e) suffices when:

- `|Q| = 1` (integer surgery), OR
- We are computing the **unrefined** filled index

The **IS chain** (refined kernel) is needed when:

- `|Q| ≥ 2` (genuinely non-integer surgery), AND
- We want the **refined** filled index (with η dependence)

### 8.2 Hirzebruch-Jung continued fraction

Every rational slope P/Q admits a **Hirzebruch-Jung (HJ) continued
fraction** expansion:

```
P/Q = k₁ − 1/(k₂ − 1/(… − 1/kₗ))
```

with all `kᵢ ≥ 2`. The length ℓ determines the number of IS kernel
steps needed.

Special cases:
- `|Q| = 1` → ℓ = 1, k = [P/Q]. Only the unrefined kernel K(k₁, 1) is
  needed. No IS steps.
- `|Q| ≥ 2` → ℓ ≥ 2. Need ℓ−1 IS convolution steps plus a final
  K(kₗ, 1) application.

### 8.3 The IS kernel

The IS kernel `I_S(m₁, e₁, m₂, e₂; η)` is defined via the
**ẽI_S inner function**:

```
ẽI_S(m₁, e₁, m₂, e₂; η) =
    Σ_{e,t ∈ ℤ}  η^e
    · I_Δ(−e₁ − m₂/2,   −e/2 + e₁ + m₁/2 + t)
    · I_Δ( e₁ + m₂/2,   −e/2 + e₂ − m₂/2 + t)
    · I_Δ(−e₂ − m₁/2,    e₂ + m₁/2 + t)
    · I_Δ( e₂ + m₁/2,    e₁ − m₂/2 + t)
    · (−q^{½})^{−e + e₁ + e₂ + m₁/2 − m₂/2 + 2t}
```

The full IS kernel combines three evaluations:

```
I_S(m₁, e₁, m₂, e₂; η) =
    ½·(−1)^{m₁} · [qq^{m₁} + qq^{−m₁}] · ẽI_S(m₁, e₁, m₂, e₂; η)
  − ½·(−1)^{m₁} · ẽI_S(m₁, e₁−1, m₂, e₂; η)
  − ½·(−1)^{m₁} · ẽI_S(m₁, e₁+1, m₂, e₂; η)
```

Note: **all three terms** carry the factor `½·(−1)^{m₁}`.

### 8.4 The kernel chain (eq. A.7)

For HJ-CF `k = [k₁, …, kₗ]`, the refined Dehn filling kernel is:

```
K^ref(P, Q; m, e; η_cusp) =
    Σ_{m₁,e₁} … Σ_{mₗ₋₁,eₗ₋₁}
        I_S(m,  −e − k₁/2·m,   m₁, e₁; η_cusp)
      · I_S(m₁, −e₁ − k₂/2·m₁, m₂, e₂; η_cusp)
      · …
      · K(kₗ, 1; mₗ₋₁, eₗ₋₁)
```

The chain has **ℓ−1 IS convolution steps** followed by **one final
unrefined kernel** K(kₗ, 1; ·).

### 8.5 The filled refined index

```
I^ref_{P/Q}^{(i)}(m_other, e_other; q, η_hard, η_cusp) =
    Σ_{mᵢ, eᵢ}
        K^ref(P, Q; mᵢ, eᵢ; η_cusp)
        · I^ref(m₁, e₁, …, mᵣ, eᵣ; q, η_hard)
```

where again `(mⱼ, eⱼ)` for `j ≠ i` are held fixed.

### 8.6 The result has BOTH η_hard AND η_cusp

The output of refined Dehn filling is a multi-variable series in:
- `q^{1/2}` (the q-series variable)
- `η₀, …, η_{d_hard−1}` (hard-edge fugacities, from I^ref)
- `η_cusp` (cusp fugacity, from the IS chain)

For ℓ = 1 (integer surgery), there is no IS chain, so no η_cusp appears.

### 8.7 Truncation and stability

The IS kernel involves an infinite sum over (e, t). In practice we
truncate:
- `qq_order`: maximum power of q^{1/2}
- `eta_order`: maximum |η_cusp exponent|

The stable region of the IS kernel at finite truncation is approximately:

```
qq ≤ qq_order − 2·eta_order
```

The code inflates `qq_order` internally by `2·eta_order` to ensure the
user-requested output range is fully within the stable region.

---

## 9. Multi-Cusp Manifolds

### 9.1 The central point of this section

**When you fill one cusp of a multi-cusp manifold, the result is still
a function of the unfilled cusps' charges. It is NOT a single q-series.**

For a 2-cusp manifold with cusps 0 and 1:

```
Filling cusp 0 with slope P/Q:
  I_{P/Q}^{(0)}(m₁, e₁) = Σ_{m₀, e₀}  K(P,Q; m₀,e₀) · I(m₀,e₀,m₁,e₁)
```

This is a q-series **for each choice of** `(m₁, e₁)`.

Similarly for the refined version:

```
I^ref_{P/Q}^{(0)}(m₁, e₁; η_hard, η_cusp) =
    Σ_{m₀, e₀}  K^ref(…; η_cusp) · I^ref(m₀,e₀,m₁,e₁; η_hard)
```

### 9.2 What the current code does

The current code's `compute_filled_index` and `compute_filled_refined_index`
accept `m_other` and `e_other` as **fixed numerical values**. They compute
the filled index **at that one point**.

This means for a 2-cusp manifold, if you call:

```python
result = compute_filled_index(nz, cusp_idx=0, P=3, Q=1,
                              m_other=[0], e_other=[0])
```

you get `I_{3/1}^{(0)}(m₁=0, e₁=0)` — a single q-series.

If you want the result at `m₁=1, e₁=0`, you must call again:

```python
result = compute_filled_index(nz, cusp_idx=0, P=3, Q=1,
                              m_other=[1], e_other=[0])
```

### 9.3 Filling multiple cusps simultaneously

Filling cusps 0 and 1 simultaneously (with possibly different slopes)
requires summing over BOTH sets of cusp charges:

```
I_{P₀/Q₀, P₁/Q₁} =
    Σ_{m₀,e₀} Σ_{m₁,e₁}
        K(P₀,Q₀; m₀,e₀) · K(P₁,Q₁; m₁,e₁) · I(m₀,e₀,m₁,e₁)
```

This is currently NOT implemented. The app explicitly rejects
multi-cusp simultaneous filling with the message:
*"Multi-cusp simultaneous filling is not yet supported."*

### 9.4 What should the app show for multi-cusp manifolds?

When a user selects one cusp to fill and there are unfilled cusps,
the app should **either**:

(a) **Show the result as a function** of the remaining cusp charges
    (e.g., a table of q-series for various (m_other, e_other) values), OR

(b) **Let the user specify** the remaining cusp charges and show the
    result at that point, OR

(c) **Default to m_other = 0, e_other = 0** with a clear label stating
    "Evaluated at (m₁, e₁) = (0, 0)".

Currently the code does (c) implicitly — the filled index is always
evaluated at m_other = e_other = 0 — but this is not communicated
clearly to the user in the app.

---

## 10. What the Code Currently Does vs What It Should Do

### 10.1 Current behavior

| Aspect | Current behavior |
|--------|-----------------|
| 3D index I(m, e) | Computed for specific (m_ext, e_ext) per call |
| Refined index I^ref | Computed for specific (m_ext, e_ext) per call, with η fugacities for hard edges |
| Unrefined Dehn filling | Sums over cusp i charges at fixed m_other, e_other (default 0) |
| Refined Dehn filling | Same: sums over cusp i at fixed m_other, e_other (default 0) |
| Non-closable cycle search | Tests vanishing at m_other = e_other = 0 only |
| Multi-cusp filling | Not supported |
| App display | Shows single q-series result, no indication of remaining cusp dependence |

### 10.2 Correctness assessment

- **For 1-cusp manifolds (r=1):** Everything is correct. There are no
  other cusps, so m_other/e_other are empty. The filled index IS a single
  q-series.

- **For multi-cusp manifolds (r ≥ 2):**
  - The **computation** is correct for each call — it correctly evaluates
    the filled index at the specified (m_other, e_other) values.
  - The **non-closable cycle search** tests only at (0, 0), which is a
    necessary but not sufficient condition for non-closability. This could
    miss cycles that are non-closable at (0,0) but closable at other
    values, or vice versa.
  - The **app** does not expose the (m_other, e_other) dependence to the
    user, which could be misleading.

### 10.3 Recommended improvements

1. **App UI**: Add input fields for unfilled cusp charges (m_other, e_other)
   on the Dehn filling page, so the user can explore the dependence.

2. **Non-closable search**: Optionally test at multiple (m_other, e_other)
   values to increase confidence in the non-closability determination.

3. **Display**: When showing a filled index result for a multi-cusp
   manifold, clearly label it as "evaluated at (mⱼ, eⱼ) = (…, …) for
   unfilled cusps j ∈ {…}".

4. **Multi-cusp filling**: Implement simultaneous filling of multiple cusps
   (future feature).

---

## Appendix A: Summary of Notation

| Symbol | Meaning |
|--------|---------|
| n | Number of tetrahedra |
| r | Number of cusps |
| d_hard | Number of hard internal edges |
| d_easy | Number of easy internal edges (d_hard + d_easy = n − r) |
| Mₖ | Meridian at cusp k |
| Lₖ | Longitude at cusp k |
| Λₖ = Lₖ/2 | Half-longitude (momentum variable at cusp k) |
| mₖ | Meridian charge at cusp k (integer) |
| eₖ | Half-longitude charge at cusp k (half-integer) |
| m⃗, e⃗ | Vectors of length r: all cusp charges |
| m_full | (m⃗, 0^{n−r}): full charge vector with internal m = 0 |
| e_full | (e⃗, e_int): full charge vector with internal e summed |
| κ | (m_full, e_full): the 2n-vector fed to g_NZ⁻¹ |
| I_Δ(m, e) | Tetrahedron index |
| I(m⃗, e⃗) | 3D index (q-series) |
| I^ref(m⃗, e⃗; η) | Refined 3D index (q-η-series) |
| K(P, Q; m, e) | Dehn filling kernel |
| I_{P/Q}^{(i)}(m_other, e_other) | Filled index: function of unfilled cusp charges |
| ηₐ | Fugacity for hard edge a |
| η_cusp | Cusp fugacity from IS chain |
| HJ-CF | Hirzebruch-Jung continued fraction |
| I_S | IS kernel (refined Dehn filling intermediate kernel) |
| ẽI_S | IS inner function (before combining with δ-function terms) |
| qq | = q^{1/2}, the series variable |
| (a, b) | Weyl symmetry vectors |

## Appendix B: Dehn Filling Kernel Families

For slope P/Q with auxiliary integers R, S satisfying RQ − PS = 1:

| Family | Constraint Pm + 2Qe = c | Particular solution | General solution | Phase |
|--------|--------------------------|---------------------|------------------|-------|
| c = 0 | Pm + 2Qe = 0 | m₀ = 0, e₀ = 0 | m = Qt, e = −Pt/2 | t |
| c = +2 | Pm + 2Qe = 2 | m₊ = 2S, e₊ = (1−PS)/Q | m = m₊ + Qt, e = e₊ − Pt/2 | Rm₊ + 2Se₊ + t |
| c = −2 | Pm + 2Qe = −2 | m₋ = −2S, e₋ = (−1+PS)/Q | m = m₋ + Qt, e = e₋ − Pt/2 | Rm₋ + 2Se₋ + t |

The kernel factor for each family:
- c = 0: `½·(−1)^{phase} · [q^{phase/2} + q^{−phase/2}]`
- c = ±2: `−½·(−1)^{phase}` (constant, no q-shift)

Multiplicity doubling:
- c = ±2: multiplicity 2 (because c = −2 and c = +2 give identical contributions)
- c = 0, t > 0: multiplicity 2 (because ±t give the same contribution via I(−m,−e) = I(m,e))
- c = 0, t = 0: multiplicity 1
