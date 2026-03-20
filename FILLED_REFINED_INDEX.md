# Filled Refined Index — Step-by-Step Manual

> **Purpose:** This document describes every step of computing the **filled refined
> index** $I^{\text{ref}}_{P/Q}$ in extreme detail, so that each intermediate
> quantity can be verified by hand.

---

## Table of Contents

1. [Notation and Conventions](#1-notation-and-conventions)
2. [Workflow Overview (GUI / Worker)](#2-workflow-overview)
3. [Step 0 — Input Data](#step-0--input-data)
4. [Step 1 — Non-Closable Cycle Search](#step-1--non-closable-cycle-search)
5. [Step 2 — SL(2,ℤ) Basis Change per NC Cycle](#step-2--sl2ℤ-basis-change-per-nc-cycle)
6. [Step 2b — Weyl Vector Transform per NC Cycle](#step-2b--weyl-vector-transform-per-nc-cycle)
7. [Step 3 — Slope Transform](#step-3--slope-transform)
8. [Step 4 — Hirzebruch-Jung Continued Fraction](#step-4--hirzebruch-jung-continued-fraction)
9. [Step 5 — Case ℓ = 1: Unrefined Kernel Only](#step-5--case-ℓ--1-unrefined-kernel-only)
10. [Step 6 — Case ℓ ≥ 2: Full IS Convolution Chain](#step-6--case-ℓ--2-full-is-convolution-chain)
11. [Step 7 — Final Assembly and Truncation](#step-7--final-assembly-and-truncation)
12. [Appendix A — Tetrahedron Index $I_\Delta(m,e)$](#appendix-a--tetrahedron-index)
13. [Appendix B — Refined Index $I^{\text{ref}}(m,e;\eta)$](#appendix-b--refined-index)
14. [Appendix C — Worked Example (m003, slope 3/2)](#appendix-c--worked-example)
15. [Appendix D — Weyl Vectors: Calculation and Application](#appendix-d--weyl-vectors-calculation-and-application)

---

## 1. Notation and Conventions

| Symbol | Meaning |
|--------|---------|
| $n$ | Number of tetrahedra |
| $r$ | Number of cusps |
| $g_{\text{NZ}}$ | Neumann-Zagier matrix, $\in \text{Sp}(2n, \mathbb{Q})$, size $2n \times 2n$ |
| $\nu_x$ | Affine shift for position rows, length $n$ (integer) |
| $\nu_p$ | Affine shift for momentum rows, length $n$ (may be half-integer) |
| $M_k$ | Row $k$ of $g_{\text{NZ}}$ (meridian of cusp $k$, $k < r$) |
| $L_k/2$ | Row $n+k$ of $g_{\text{NZ}}$ (half-longitude of cusp $k$) |
| $(m, e)$ | Cusp charge variables: $m \in \mathbb{Z}$, $e \in \frac{1}{2}\mathbb{Z}$ |
| $\alpha_k, \beta_k$ | Canonical peripheral curves at cusp $k$: $\alpha_k = M_k$, $\beta_k = L_k$ |
| $qq$ | $= q^{1/2}$ (the expansion variable) |
| $I_\Delta(m, e)$ | Tetrahedron index (see [Appendix A](#appendix-a--tetrahedron-index)) |
| $\eta_a$ | Fugacity for hard edge $a$ (refined index) |
| $\eta_c$ | Cusp $\eta$ variable from IS convolution chain |

### Charge convention

The physical cycle corresponding to charge $(m_k, e_k)$ at cusp $k$ is:

$$
\text{cycle}(m_k, e_k) = -e_k \cdot \alpha_k + \frac{m_k}{2} \cdot \beta_k
$$

where $\alpha_k$ = meridian, $\beta_k$ = longitude.

### Slope convention

The Dehn filling slope $(P, Q)$ fills the cycle $P \cdot M_k + Q \cdot L_k = P \cdot \alpha + Q \cdot \beta$.
In the NZ formalism with $(M, L/2)$ rows, the filling condition is:

$$
P \cdot m + 2Q \cdot e = c, \qquad c \in \{-2, 0, 2\}
$$

### Symplectic pairing

$$
\{M_k, L_k/2\} = 1, \qquad \{M_k, M_j\} = \{L_k/2, L_j/2\} = 0 \quad (k \neq j)
$$

---

## 2. Workflow Overview

The GUI worker (`workers.py → DehnFillingWorker`) orchestrates the full pipeline:

```
Input: nz_data, cusp_configs[{cusp_idx, P_user, Q_user}], q_order_half, weyl_a, weyl_b

For each cusp to fill:
  │
  ├─ Step 1: Search NC cycles at this cusp
  │   └── compute_filled_index(nz, cusp_idx, P, Q, m_other=0, e_other=0)
  │       for each candidate (P,Q) in search range
  │       → keep those where result.is_stably_zero()
  │       → deduplicate: keep one from {γ, −γ}
  │
  For each NC cycle γ = (P_nc, Q_nc):
  │
  ├─ Step 2: SL(2,ℤ) basis change
  │   └── nz_nc = apply_general_cusp_basis_change(nz, cusp_idx, P_nc, Q_nc, R, S)
  │
  ├─ Step 2b: Transform Weyl vectors into new basis
  │   └── (weyl_a_nc, weyl_b_nc) = transform_weyl_vectors(weyl_a, weyl_b, P_nc, Q_nc, R, S)
  │
  ├─ Step 3: Slope transform
  │   └── (p, q) in new basis
  │
  └─ Step 4–7: compute_filled_refined_index(nz_nc, cusp_idx, p, q, weyl_a_nc, weyl_b_nc, ...)
      └── HJ-CF → ℓ=1 or ℓ≥2 path → FilledRefinedResult
```

---

## Step 0 — Input Data

### What you need before starting

1. **`NeumannZagierData`** (from `build_neumann_zagier`):
   - $g_{\text{NZ}}$: $2n \times 2n$ rational symplectic matrix
   - $\nu_x$: integer array of length $n$
   - $\nu_p$: rational array of length $n$ (entries may be half-integer)
   - $n, r, \text{num\_hard}, \text{num\_easy}$

2. **User's slope**: $(P_{\text{user}}, Q_{\text{user}})$ — coprime integers

3. **Cusp to fill**: `cusp_idx` (0-based)

4. **Truncation**: `q_order_half` = maximum power of $qq = q^{1/2}$ to keep

5. **Other-cusp charges**: $(m_{\text{other}}, e_{\text{other}})$, default all zeros

---

## Step 1 — Non-Closable Cycle Search

### Goal
Find all cycles $\gamma = P_{nc}\alpha + Q_{nc}\beta$ at the given cusp such that
the **unrefined** filled 3D index $I_{\gamma}$ vanishes identically.

### Procedure

For each candidate slope $(P, Q)$ in the search range (e.g. $P \in [-2,2], Q \in [-2,2]$):

1. **Find auxiliary integers** $R, S$ with $R \cdot Q - P \cdot S = 1$ (via `find_rs`):
   - Use extended Euclidean algorithm: `_ext_gcd(|Q|, |P|)` → $(g, x, y)$
   - Correct signs: $R = x \cdot \text{sign}(Q)$, $S = -y \cdot \text{sign}(P)$
   - Verify: $R \cdot Q - P \cdot S = 1$ ✓

2. **Enumerate kernel terms** for the (unrefined) Dehn filling kernel $K(P,Q;m,e)$:

   The kernel is non-zero only when $Pm + 2Qe = c$ for $c \in \{-2, 0, 2\}$.

   For each $c$, find a particular solution $(m_c, e_c)$:
   - Solve $P \cdot m_c + Q \cdot (2e_c) = c$ via extended GCD
   - General family: $m_t = m_c + Qt$, $e_t = e_c - Pt/2$
   - Phase: $\text{phase}_t = R \cdot m_t + 2S \cdot e_t = (Rm_c + 2Se_c) + t$

   **Kernel factor:**
   - $c = 0$: $\;\frac{1}{2}(-1)^{\text{phase}} \big(q^{\text{phase}/2} + q^{-\text{phase}/2}\big)$
   - $c = \pm 2$: $\;-\frac{1}{2}(-1)^{\text{phase}}$

3. **Compute filled index** for each kernel term:
   $$
   I_{P/Q} = \sum_{(m,e) \in \text{kernel}} K(P,Q;m,e) \cdot I_{3D}(m_{\text{ext}}, e_{\text{ext}})
   $$
   where $m_{\text{ext}}$ inserts $m$ at position `cusp_idx` and zeros elsewhere.

4. **Check stable zero**: The cycle is non-closable if $I_{P/Q} = 0$ (after ignoring
   boundary artifacts in the top `buffer` powers of the truncated series).

5. **Deduplicate**: Keep one canonical representative from each $\{(P,Q), (-P,-Q)\}$ pair.
   Convention: $Q > 0$, or ($Q = 0$ and $P > 0$).

### Output
A list of non-closable cycles: $(P_{nc}, Q_{nc})$ for this cusp.

---

## Step 2 — SL(2,ℤ) Basis Change per NC Cycle

### Goal
For each NC cycle $\gamma = P_{nc}\alpha + Q_{nc}\beta$, change the peripheral
basis at the filled cusp so that $\gamma$ becomes the new meridian. This requires
rebuilding the Neumann-Zagier data.

### Why this is necessary
The refined Dehn filling formula assumes the **non-closable cycle is the meridian**
($M$-direction). If $\gamma \neq \alpha$, we must transform the NZ data so that
$\gamma$ occupies the meridian row. Different NC cycles yield genuinely different
NZ matrices, and the refined index must be recomputed from scratch for each.

### Procedure

1. **Find the SL(2,ℤ) complement** $\delta = R\alpha + S\beta$:
   - Start from `find_rs(P_nc, Q_nc)` → $(R_0, S_0)$ with $R_0 Q_{nc} - P_{nc} S_0 = 1$
   - Set $R = -R_0$, $S = -S_0$ so that:
   $$
   \det\begin{pmatrix} P_{nc} & R \\ Q_{nc} & S \end{pmatrix} = P_{nc} S - R Q_{nc} = +1
   $$
   This is the SL(2,ℤ) convention: $\begin{pmatrix} \gamma \\ \delta \end{pmatrix} = \begin{pmatrix} P_{nc} & Q_{nc} \\ R & S \end{pmatrix} \begin{pmatrix} \alpha \\ \beta \end{pmatrix}$

2. **Apply `apply_general_cusp_basis_change(nz, cusp_idx, a=P_nc, b=Q_nc, c=R, d=S)`**:

   The SL(2,ℤ) matrix $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ acts on $(\mu, \lambda)$ as:
   $$
   \text{new } \mu = a\mu + b\lambda, \qquad \text{new } \lambda = c\mu + d\lambda
   $$

   In the NZ convention $(M, L/2)$ where $M \leftrightarrow \mu$ and $L \leftrightarrow \lambda$:

   $$
   \boxed{
   \begin{aligned}
   \text{new } M &= a \cdot M + 2b \cdot (L/2) \\
   \text{new } L/2 &= \frac{c}{2} \cdot M + d \cdot (L/2)
   \end{aligned}
   }
   $$

   **Concretely for row $k$ = `cusp_idx` of $g_{\text{NZ}}$:**
   ```
   g_NZ_new[k]     = a · g_NZ[k]     + 2b · g_NZ[n+k]       (new M row)
   g_NZ_new[n+k]   = (c/2) · g_NZ[k] + d  · g_NZ[n+k]       (new L/2 row)
   ```

   **Affine shifts:**
   ```
   nu_x_new[k] = a · nu_x[k] + 2b · nu_p[k]       (always integer)
   nu_p_new[k] = (c/2) · nu_x[k] + d · nu_p[k]     (may be half-integer)
   ```

   **Symplectic pairing preserved:**
   $$
   \{\text{new } M, \text{new } L/2\} = ad - bc = 1 \quad \checkmark
   $$

3. **Note:** Unlike the older `apply_cusp_basis_change`, this does **not** require
   $a$ to be odd. The new $L/2$ row may acquire half-integer entries, which the
   NZ data format already supports.

### Output
`nz_nc` — a new `NeumannZagierData` with the cusp rows and shifts updated.

### Manual check
- Verify $\det \begin{pmatrix} P_{nc} & R \\ Q_{nc} & S \end{pmatrix} = +1$
- Verify new $g_{\text{NZ}}$ is symplectic: $g_{\text{NZ,new}} \cdot \Omega \cdot g_{\text{NZ,new}}^T = \Omega$
- Verify $\nu_{x,\text{new}}[k]$ is integer

---

## Step 2b — Weyl Vector Transform per NC Cycle

### Goal
Transform the Weyl-symmetry vectors $(a, b)$ from the original NZ basis into
the new NZ basis established in Step 2. The Weyl vectors are **basis-dependent**:
they describe how the η-centre of the refined index depends on the cusp charges
$(m, e)$, and these charges have a different physical meaning in each NC basis.

### Background: what are Weyl vectors?

The refined index $I^{\text{ref}}(m, e; \eta)$ is **not** generically Weyl-symmetric
(i.e., not invariant under $\eta_j \to \eta_j^{-1}$). However, for suitable manifolds
there exist vectors $a \in \mathbb{Z}^{\text{num\_hard}}$ and
$b \in (\mathbb{Z}/2)^{\text{num\_hard}}$ such that:

$$
f(m, e; \eta) \;=\; \eta^{b \cdot m + a \cdot e} \cdot I^{\text{ref}}(m, e; \eta)
$$

**is** Weyl-symmetric: $f(m, e; \eta) = f(m, e; \eta^{-1})$.

Here the dot products are $b \cdot m = \sum_j b_j \cdot \Sigma m_k$ and
$a \cdot e = \sum_j a_j \cdot \Sigma e_k$, summing the cusp charges.

### How Weyl vectors are initially computed

The vectors $(a, b)$ are extracted from the 25ʳ evaluation grid of the refined
index (computed in Panel 1). The procedure is in `compute_ab_vectors`:

1. **Compute η-centres** — For each entry $(m_{\text{ext}}, e_{\text{ext}}, I^{\text{ref}})$
   in the evaluation grid, compute the coefficient-weighted centre of the η-exponents
   at the leading (lowest) q-order:
   $$
   \text{centre}_j(m, e) = \frac{\sum_k \eta_{j,k}^{\text{exp}} \cdot c_k}{\sum_k c_k}
   $$
   where the sum is over all terms at the minimum q-half-power.

2. **The centre satisfies the linear relation:**
   $$
   \text{centre}_j(m, e) = b_j \cdot \Sigma m + \frac{a_j}{2} \cdot \Sigma e
   $$

3. **Extract $b$ from meridian pairs** — Find entries with all $e_k = 0$:
   $$
   b_j = -\frac{\text{centre}_j(+m, 0) - \text{centre}_j(-m, 0)}{2m}
   $$

4. **Extract $a$ from longitude pairs** — Find entries with all $m_k = 0$:
   $$
   a_j = -\frac{\text{centre}_j(0, +e) - \text{centre}_j(0, -e)}{|e|}
   $$

5. **Consensus check** — Multiple pairs are used; consistency is verified.

6. **Integrality validation** — $a_j$ must be integer, $b_j$ must be half-integer.
   If either fails, the Weyl vectors are invalid and no shift is applied.

### Storage convention

| Symbol | Attribute | Stores | Type |
|--------|-----------|--------|------|
| $a_j$ | `ABVectors.a[j]` | $2 \cdot a_{\text{actual}}$ | integer (Fraction) |
| $b_j$ | `ABVectors.b[j]` | $b_{\text{actual}}$ | half-integer (Fraction) |

The `_apply_weyl_shift` function computes the doubled shift:
```
shift_x2[j] = 2 · b_stored[j] · Σm + a_stored[j] · Σe
```
which represents $2 \times (b_j \cdot m + a_j \cdot e)$, matching the doubled η-exponent
encoding in the keys.

### Why the transform is necessary

When Step 2 changes the peripheral-curve basis at the filled cusp by the
SL(2,ℤ) matrix $\begin{pmatrix} P_{\text{nc}} & Q_{\text{nc}} \\ R & S \end{pmatrix}$,
the charge coordinates $(m', e')$ in the **new** basis correspond to **different physical
cycles** than $(m, e)$ in the old basis. Since the Weyl vectors describe the
η–charge relationship, they must be updated to the new coordinates.

### The algebraic transform

The η-centre at the filled cusp in the old coordinates:

$$
\text{centre}_j = b_j \cdot m + a_j \cdot e
$$

Under the NZ-convention basis change:
$$
m = \alpha \cdot m' + 2\beta \cdot e', \qquad e = \frac{\gamma}{2} \cdot m' + \delta \cdot e'
$$
where $(\alpha, \beta, \gamma, \delta) = (P_{\text{nc}}, Q_{\text{nc}}, R, S)$.

Substituting and matching coefficients against
$\text{centre}_j = b_j^{\text{nc}} \cdot m' + a_j^{\text{nc}} \cdot e'$:

$$
\begin{pmatrix} b_j \\ a_j \end{pmatrix}
= \begin{pmatrix} \alpha & \gamma/2 \\ 2\beta & \delta \end{pmatrix}
\begin{pmatrix} b_j^{\text{nc}} \\ a_j^{\text{nc}} \end{pmatrix}
$$

The matrix determinant is $\alpha\delta - \gamma\beta = 1$ (SL(2,ℤ)), so:

$$
\boxed{
\begin{pmatrix} b_j^{\text{nc}} \\ a_j^{\text{nc}} \end{pmatrix}
= \begin{pmatrix} S & -R/2 \\ -2Q_{\text{nc}} & P_{\text{nc}} \end{pmatrix}
\begin{pmatrix} b_j \\ a_j \end{pmatrix}
}
$$

i.e.:
$$
b_j^{\text{nc}} = S \cdot b_j - \frac{R}{2} \cdot a_j, \qquad
a_j^{\text{nc}} = -2Q_{\text{nc}} \cdot b_j + P_{\text{nc}} \cdot a_j
$$

### In stored convention

$$
\begin{aligned}
a_{\text{stored}}^{\text{nc}}[j] &= P_{\text{nc}} \cdot a_{\text{stored}}[j] - 4\, Q_{\text{nc}} \cdot b_{\text{stored}}[j] \\[4pt]
b_{\text{stored}}^{\text{nc}}[j] &= S \cdot b_{\text{stored}}[j] - \frac{R}{4} \cdot a_{\text{stored}}[j]
\end{aligned}
$$

### Integrality preservation

The transform preserves the required integrality:
- $a_{\text{stored}}^{\text{nc}} \in \mathbb{Z}$: since $P_{\text{nc}} \cdot (\text{even}) \in \mathbb{Z}$ and $4 Q_{\text{nc}} \cdot (\text{half-int}) \in \mathbb{Z}$.
- $2 \cdot b_{\text{stored}}^{\text{nc}} \in \mathbb{Z}$: since $2S \cdot (\text{half-int}) \in \mathbb{Z}$ and $R \cdot a_{\text{stored}}/2 \in \mathbb{Z}$ (because $a_{\text{stored}}$ is even).

### Implementation

```python
from manifold_index.core.weyl_check import transform_weyl_vectors

weyl_a_nc, weyl_b_nc = transform_weyl_vectors(
    weyl_a, weyl_b, P_nc=P_nc, Q_nc=Q_nc, R=R, S=S,
)
```

### Manual check
- Identity transform $(P_{\text{nc}}, Q_{\text{nc}}, R, S) = (1, 0, 0, 1)$: vectors unchanged
- Roundtrip: apply transform then its inverse → recover original vectors
- Verify: $a_{\text{stored}}^{\text{nc}} \in \mathbb{Z}$ and $2 \cdot b_{\text{stored}}^{\text{nc}} \in \mathbb{Z}$

### Output
$(a^{\text{nc}}, b^{\text{nc}})$ — the Weyl vectors in the new cusp basis, passed to
`compute_filled_refined_index(nz_nc, ..., weyl_a=weyl_a_nc, weyl_b=weyl_b_nc)`.

---

## Step 3 — Slope Transform

### Goal
Express the user's original slope $(P_{\text{user}}, Q_{\text{user}})$ in the new
$(\gamma, \delta)$ basis.

### Procedure

The basis change matrix is:
$$
\begin{pmatrix} P_{nc} & R \\ Q_{nc} & S \end{pmatrix} \begin{pmatrix} p \\ q \end{pmatrix} = \begin{pmatrix} P_{\text{user}} \\ Q_{\text{user}} \end{pmatrix}
$$

Since $\det = 1$, the inverse is:
$$
\begin{pmatrix} p \\ q \end{pmatrix} = \begin{pmatrix} S & -R \\ -Q_{nc} & P_{nc} \end{pmatrix} \begin{pmatrix} P_{\text{user}} \\ Q_{\text{user}} \end{pmatrix}
$$

So:
$$
\boxed{
\begin{aligned}
p &= S \cdot P_{\text{user}} - R \cdot Q_{\text{user}} \\
q &= -Q_{nc} \cdot P_{\text{user}} + P_{nc} \cdot Q_{\text{user}}
\end{aligned}
}
$$

### Manual check
- Verify reconstruction: $P_{nc} \cdot p + R \cdot q = P_{\text{user}}$ and $Q_{nc} \cdot p + S \cdot q = Q_{\text{user}}$
- The filling cycle in the new basis is $p\gamma + q\delta$
- **Key property:** Since $\gamma$ is non-closable, $q \neq 0$ generically. When $|q| = 1$, we get $\ell = 1$ (efficient). When $|q| > 1$, we get $\ell \geq 2$ (IS convolution chain).

### Output
$(p, q)$ — the slope in the NC cycle's basis. Pass to `compute_filled_refined_index(nz_nc, cusp_idx, P=p, Q=q, ...)`.

---

## Step 4 — Hirzebruch-Jung Continued Fraction

### Goal
Decompose $p/q$ into the HJ continued fraction:
$$
\frac{p}{q} = k_1 - \cfrac{1}{k_2 - \cfrac{1}{\cdots - \cfrac{1}{k_\ell}}}
$$

### Algorithm (in `hj_continued_fraction`)

```python
def hj_continued_fraction(P, Q):
    if Q == 0:  return [0, 0]          # special case: longitude/meridian
    if Q < 0:   P, Q = -P, -Q         # normalise Q > 0
    x = Fraction(P, Q)
    ks = []
    while True:
        k = ceil(x)
        ks.append(k)
        remainder = k - x
        if remainder == 0:  break
        x = 1 / remainder
    return ks
```

### Special cases
| Input | HJ-CF | $\ell$ |
|-------|-------|--------|
| $Q = 0, P = \pm 1$ | $[0, 0]$ | 2 |
| $\|Q\| = 1$ | $[p/q]$ | **1** (unrefined kernel suffices) |
| $P/Q = 1/2$ | $[1, 2]$ | 2 |
| $P/Q = 5/2$ | $[3, 2]$ | 2 |
| $P/Q = 3/2$ | $[2, 2]$ | 2 |

### The key distinction
- **$\ell = 1$**: Only the unrefined kernel $K(k_1, 1; m, e)$ is needed. No IS convolution, no cusp $\eta$.
- **$\ell \geq 2$**: The full IS convolution chain is needed. The result includes a cusp $\eta$ variable.

---

## Step 5 — Case $\ell = 1$: Unrefined Kernel Only

### When this applies
$|Q| = 1$ (equivalently, the HJ-CF has length 1). The slope is an integer surgery: $p/q = k_1$.

### Mathematical formula

$$
I^{\text{ref}}_{k_1} = \sum_{(m,e) \in \text{supp}(K)} K(k_1, 1; m, e) \cdot I^{\text{ref}}(m_{\text{ext}}, e_{\text{ext}}; \eta_{\text{hard}})
$$

where $K(k_1, 1; m, e)$ is the unrefined Dehn filling kernel at slope $k_1/1$.

### Detailed sub-steps

#### 5a. Enumerate kernel support

For slope $(k_1, 1)$, the auxiliary $(R, S) = (1, 0)$ (since $R \cdot 1 - k_1 \cdot 0 = 1$).

The filling condition $k_1 m + 2e = c$ gives:
- **$c = 0$**: $m_t = t$, $e_t = -k_1 t/2$, phase $= m_t = t$
- **$c = 2$**: particular solution $(m_c, e_c)$ with $k_1 m_c + 2e_c = 2$, family: $m_t = m_c + t$, $e_t = e_c - k_1 t/2$, phase $= m_t$

Scan $|t| \leq$ `m1_range`.

#### 5b. For each kernel term $(m_t, e_t, c, \text{phase})$

1. **Build full charges**: `_make_ext(m_t, e_t)` inserts $(m_t, e_t)$ at position `cusp_idx` in the charge vector, filling other cusps with `m_other`, `e_other`.

2. **Compute refined index**:
   ```
   refined = compute_refined_index(nz_nc, m_ext, e_ext, q_order_half + extra_q)
   ```
   where `extra_q = |phase|` for $c=0$ (to accommodate the $q^{\pm\text{phase}/2}$ shift).

   If `refined` is empty → skip.

3. **Apply Weyl shift** (using the NC-transformed Weyl vectors from Step 2b):
   $$
   I^{\text{ref}} \mapsto \eta^{b^{\text{nc}} \cdot m + a^{\text{nc}} \cdot e} \cdot I^{\text{ref}}
   $$
   This pre-multiplies by the Weyl monomial (in the new basis) before the kernel is applied.

4. **Convert to MultiEtaSeries** (no cusp $\eta$ appended since $\ell = 1$).

5. **Determine multiplicity**:
   - $c = 2$: mult $= 2$ (antipodal symmetry with $c = -2$)
   - $c = 0, t \neq 0$: mult $= 2$ ($\pm t$ contribute identically)
   - $c = 0, t = 0$: mult $= 1$ (no partner)

6. **Apply kernel factor** $K(k_1, 1; m, e)$:

   For $c = 0$:
   $$
   \text{contribution} = \text{mult} \cdot \frac{1}{2}(-1)^{\text{phase}} \big(qq^{\text{phase}} + qq^{-\text{phase}}\big) \cdot \text{series}
   $$
   (shifts the $qq$-powers by $\pm$phase)

   For $c = \pm 2$:
   $$
   \text{contribution} = \text{mult} \cdot \big(-\frac{1}{2}\big)(-1)^{\text{phase}} \cdot \text{series}
   $$
   (constant factor, no $qq$-shift)

7. **Accumulate**: `total_series += contribution`

#### 5c. Output

`FilledRefinedResult` with `has_cusp_eta = False`. Key format:
```
(qq_power, 2·η_0_exp, 2·η_1_exp, ..., 2·η_{k-1}_exp)  →  Fraction coefficient
```

---

## Step 6 — Case $\ell \geq 2$: Full IS Convolution Chain

### When this applies
$|Q| \geq 2$ (the HJ-CF has length $\geq 2$). This is the most general case.

### Mathematical formula (eq. A.7 of DFK)

$$
K^{\text{ref}}(P,Q; m,e; \eta_c) = \sum_{m_1,e_1} \cdots \sum_{m_{\ell-1},e_{\ell-1}}
I_S(m, -e - \tfrac{k_1}{2}m, m_1, e_1) \cdot
I_S(m_1, -e_1 - \tfrac{k_2}{2}m_1, m_2, e_2) \cdots
K(k_\ell, 1; m_{\ell-1}, e_{\ell-1})
$$

### Detailed sub-steps

#### 6a. Internal truncation buffer

The IS kernel introduces boundary artifacts near the truncation edge. To prevent
these from contaminating the user-visible result, we inflate the internal
$qq$-order:

$$
\text{qq\_internal} = \text{qq\_order} + \underbrace{\lfloor\text{qq\_order}/2\rfloor + 4}_{\text{IS buffer}}
$$

All intermediate computations use `qq_internal`. The final result is truncated
back to `qq_order` in [Step 7](#step-7--final-assembly-and-truncation).

#### 6b. Grid scan — initialise state

Scan all $(m_i, e_i)$ at the filled cusp:
- $m_i \in [-2 \cdot \text{qq\_internal}, +2 \cdot \text{qq\_internal}]$
- $e_i \in \{j/2 : j \in [-2 \cdot \text{qq\_internal}, +2 \cdot \text{qq\_internal}]\}$

For each $(m_i, e_i)$:

1. Build `m_ext, e_ext` via `_make_ext`
2. Compute `refined = compute_refined_index(nz_nc, m_ext, e_ext, qq_internal)`
3. If non-zero:
   - Apply Weyl shift (using NC-transformed vectors from Step 2b)
   - Convert to `MultiEtaSeries` with `cusp_eta = 0` appended as last key dimension
   - Store in `state[(m_i, e_i)] = multi_series`

**Key format at this stage:**
```
(qq_power, 2·η_0_exp, ..., 2·η_{k-1}_exp, cusp_eta_exp)  →  Fraction
```
Initially `cusp_eta_exp = 0` for all entries.

#### 6c. IS convolution steps (repeat $\ell - 1$ times)

For step $i = 1, \ldots, \ell-1$:

**Input:** `state[(m, e)]` — current MultiEtaSeries indexed by $(m, e)$

**Output:** `new_state[(m_1, e_1)]` — new MultiEtaSeries indexed by $(m_1, e_1)$

For each $(m, e)$ in current state with non-empty series:

1. **E-transform**: $e_{\text{in}} = -e - \frac{k_i}{2} m$

2. **Enumerate target $(m_1, e_1)$** from `_enumerate_slope1_all(k_{i+1}, m1_range)`:
   - All $c \in \{-2, 0, 2\}$, all $t \in [-\text{m1\_range}, +\text{m1\_range}]$
   - No symmetry shortcuts (IS kernel breaks $(m,e) \to (-m,-e)$ symmetry)

3. **Compute IS kernel**: `I_S(m, e_in, m_1, e_1; η_c)` → `QEtaSeries`

4. **Convolve**: $\text{product} = I_S \otimes \text{state}[(m,e)]$
   - $qq$-powers add
   - Hard $\eta$ dimensions pass through unchanged
   - Cusp $\eta$ exponents add (IS kernel's $\eta$ contributes to last dimension)

5. **Accumulate**: `new_state[(m_1, e_1)] += product`

Replace `state = new_state`.

#### 6d. The IS kernel $I_S$ — full formula

$$
I_S(m_1, e_1, m_2, e_2; \eta) = \frac{1}{2}(-1)^{m_1}\big(qq^{m_1} + qq^{-m_1}\big) \cdot \tilde{e}I_S(m_1, e_1, m_2, e_2) - \frac{1}{2}(-1)^{m_1} \cdot \tilde{e}I_S(m_1, e_1-1, m_2, e_2) - \frac{1}{2}(-1)^{m_1} \cdot \tilde{e}I_S(m_1, e_1+1, m_2, e_2)
$$

#### 6e. The inner kernel $\tilde{e}I_S$ (expr8 in DFK.nb)

$$
\tilde{e}I_S(m_1, e_1, m_2, e_2; \eta) = \sum_{e \in \mathbb{Z}} \sum_{t \in \mathbb{Z}} \eta^e \cdot (-qq)^{-e + e_1 + e_2 + m_1/2 - m_2/2 + 2t} \cdot \prod_{j=1}^{4} I_\Delta(a_j, b_j)
$$

where the four tetrahedron-index arguments are:

| Index | First arg ($a_j$) | Second arg ($b_j$) |
|-------|-------------------|---------------------|
| $I_\Delta^{(1)}$ | $-e_1 - m_2/2$ | $-e/2 + e_1 + m_1/2 + t$ |
| $I_\Delta^{(2)}$ | $e_1 + m_2/2$ | $-e/2 + e_2 - m_2/2 + t$ |
| $I_\Delta^{(3)}$ | $-e_2 - m_1/2$ | $e_2 + m_1/2 + t$ |
| $I_\Delta^{(4)}$ | $e_2 + m_1/2$ | $e_1 - m_2/2 + t$ |

**Integrality filters:**
- $a_1 = -e_1 - m_2/2$ must be integer
- $a_3 = -e_2 - m_1/2$ must be integer
- The $\eta$-sum variable $e$ must have parity $(m_1 + m_2) \bmod 2$

**Implementation detail:** $I_\Delta^{(3)}$ and $I_\Delta^{(4)}$ are independent of the $\eta$-sum
variable $e$, so they are computed once per $t$ and their product $s_{34}$ is reused across all
$e$ values (= different `n_eta` values).

#### 6f. Apply final unrefined kernel $K(k_\ell, 1)$

After $\ell - 1$ IS convolution steps, apply the unrefined kernel $K(k_\ell, 1; m_{\ell-1}, e_{\ell-1})$:

1. Enumerate ALL $(m, e)$ from `_enumerate_slope1_all(k_\ell, m1_range)`
2. For each $(m, e)$ that exists in `state`:
   - Look up $(c, \text{phase}, \text{mult}=1)$ from the enumeration
   - Apply kernel factor (same formula as Step 5, but with `mult = 1`)
   - Accumulate into `total_series`

---

## Step 7 — Final Assembly and Truncation

### Truncation (ℓ ≥ 2 only)

Discard all entries with $qq$-power $>$ `qq_order`:
```python
truncated = {k: v for k, v in total_series.items() if k[0] <= qq_order}
```

This removes the boundary artifacts introduced by the IS buffer inflation.

### Output: `FilledRefinedResult`

| Field | Value |
|-------|-------|
| `P, Q` | Slope in the NC basis (not the user's original slope) |
| `cusp_idx` | Which cusp was filled |
| `series` | `MultiEtaSeries` — the filled refined index |
| `qq_order` | User's requested truncation |
| `eta_order` | Max cusp $\eta$ exponent (0 for $\ell=1$) |
| `hj_ks` | $[k_1, \ldots, k_\ell]$ |
| `n_kernel_terms` | Number of non-zero $(m,e)$ pairs evaluated |
| `num_hard` | Number of hard-edge $\eta$ dimensions |
| `has_cusp_eta` | `True` for $\ell \geq 2$, `False` for $\ell = 1$ |

### Key format

- **$\ell = 1$**: `(qq_power, 2·η_0, ..., 2·η_{k-1})` — hard-edge $\eta$'s only
- **$\ell \geq 2$**: `(qq_power, 2·η_0, ..., 2·η_{k-1}, cusp_eta)` — hard-edge $\eta$'s + cusp $\eta$

### Setting all $\eta = 1$

To get the plain $q$-series (for cross-checking different NC cycles):
```python
result.eta1_series()  # sums over all η dimensions → dict[qq_power → Fraction]
```

All NC cycles should give the **same** $\eta = 1$ series.

---

## Appendix A — Tetrahedron Index

### Definition: $I_\Delta(m, e)$

The tetrahedron index is a $q^{1/2}$-series. It uses the MIt convention from
Garoufalidis–Kim:

- If $m + e \geq 0$: $\;I_\Delta(m, e) = (-qq)^m \cdot I_t(-m-e, m)$
- If $m + e < 0$: $\;I_\Delta(m, e) = I_t(m, e)$

where $I_t(m, e)$ is the raw tetrahedron index:

$$
I_t(m, e) = \sum_{n = \max(0, -e)}^{\infty} \frac{(-1)^n \cdot qq^{n(n+1) - (2n+e)m}}{\prod_{k=1}^{n}(1 - qq^{2k}) \cdot \prod_{k=1}^{n+e}(1 - qq^{2k})}
$$

### Key properties
- $I_\Delta(m, e) = 0$ if $m$ or $e$ is non-integer
- $I_\Delta(0, 0) = 1$
- Leading $qq$-power: $\delta(m,e) = \frac{1}{2}(m_+ (m+e)_+ + (-m)_+ e_+ + (-e)_+(-e-m)_+) + \max(0, m, -e)$

### Caching
Results are memoized in a module-level cache keyed by $(m, e, qq\_\text{order})$.

---

## Appendix B — Refined Index

### Definition: $I^{\text{ref}}(m_{\text{ext}}, e_{\text{ext}}; \eta_0, \ldots, \eta_{k-1})$

This is the 3D index with one formal fugacity $\eta_a$ per hard internal edge:

$$
I^{\text{ref}} = \sum_{e_{\text{int}} \in (\frac{1}{2}\mathbb{Z})^{n-r}} \left[\prod_{a=0}^{k-1} \eta_a^{e_{r+a}}\right] \cdot (-q^{1/2})^{m \cdot \nu_p - e \cdot \nu_x} \cdot \prod_{j=0}^{n-1} I_\Delta\big((g_{\text{NZ}}^{-1}\kappa)_j, (g_{\text{NZ}}^{-1}\kappa)_{n+j}\big)
$$

where:
- $\kappa = (m_{\text{full}}, e_{\text{full}})$ with $m_{\text{full}} = (m_{\text{ext}}, 0^{n-r})$, $e_{\text{full}} = (e_{\text{ext}}, e_{\text{int}})$
- The sum is over all $e_{\text{int}}$ such that $(g_{\text{NZ}}^{-1}\kappa)$ has integer entries
- $k = \text{num\_hard}$ is the number of hard edges

### Output key convention
```
key = (qq_power, 2·η_0_exp, 2·η_1_exp, ..., 2·η_{k-1}_exp)
```
The factor of 2 is because fugacity exponents are half-integers; $\times 2$ makes them integers.

### Recovering the ordinary 3D index
Set all $\eta_a = 1$:
$$
I_{3D}(m, e) = \sum_{\text{all η-keys}} I^{\text{ref}}(m, e; \eta_0, \ldots) \bigg|_{\eta_a = 1}
$$

---

## Appendix C — Worked Example (m003, slope 3/2)

### Setup
- Manifold: `m003` ($n = 2$ tetrahedra, $r = 1$ cusp)
- User slope: $(P_{\text{user}}, Q_{\text{user}}) = (3, 2)$ at cusp 0
- `q_order_half = 10`

### Step 1: NC cycle search

Suppose the search finds NC cycle $\gamma = (1, 0) = \alpha$ (the meridian itself).

### Step 2: SL(2,ℤ) basis change

For $\gamma = (P_{nc}, Q_{nc}) = (1, 0)$:
- `find_rs(1, 0)` → $(R_0, S_0)$; since $R_0 \cdot 0 - 1 \cdot S_0 = 1 \Rightarrow S_0 = -1$, pick $R_0 = 1, S_0 = -1$
- $R = -R_0 = -1$, $S = -S_0 = 1$
- Check: $\det\begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix} = 1 \cdot 1 - (-1) \cdot 0 = 1$ ✓
- SL(2,ℤ) matrix: $\begin{pmatrix} 1 & 0 \\ -1 & 1 \end{pmatrix}$

Since this is the identity on $\mu$ (new $\mu = 1 \cdot \mu + 0 \cdot \lambda = \mu$), the NZ data is unchanged: `nz_nc = nz`.

### Step 3: Slope transform

$$
p = S \cdot P_{\text{user}} - R \cdot Q_{\text{user}} = 1 \cdot 3 - (-1) \cdot 2 = 5
$$
$$
q = -Q_{nc} \cdot P_{\text{user}} + P_{nc} \cdot Q_{\text{user}} = 0 \cdot 3 + 1 \cdot 2 = 2
$$

Verify: $P_{nc} \cdot p + R \cdot q = 1 \cdot 5 + (-1) \cdot 2 = 3 = P_{\text{user}}$ ✓  
Verify: $Q_{nc} \cdot p + S \cdot q = 0 \cdot 5 + 1 \cdot 2 = 2 = Q_{\text{user}}$ ✓

### Step 4: HJ-CF

$p/q = 5/2$: $\lceil 5/2 \rceil = 3$, remainder $= 3 - 5/2 = 1/2$, $1/(1/2) = 2$, $\lceil 2 \rceil = 2$, remainder $= 0$.

**Result:** $[k_1, k_2] = [3, 2]$, $\ell = 2$.

### Steps 5–7: ℓ = 2 path

Since $\ell = 2$, we enter the IS convolution chain:

1. **Grid scan**: Compute $I^{\text{ref}}(m, e)$ for all $(m, e)$ in scan range. Initialize `state[(m,e)]` with cusp $\eta = 0$.

2. **One IS step** ($i = 1$):
   - $k_{\text{current}} = k_1 = 3$
   - $k_{\text{next}} = k_2 = 2$
   - For each $(m, e)$ in state: compute $e_{\text{in}} = -e - \frac{3}{2}m$
   - For each target $(m_1, e_1)$ from $K(2, 1)$ support:
     - Compute $I_S(m, e_{\text{in}}, m_1, e_1; \eta_c)$
     - Convolve with `state[(m,e)]`
     - Accumulate into `new_state[(m_1, e_1)]`

3. **Final kernel** $K(k_2 = 2, 1)$:
   - For each $(m_1, e_1)$ in state: apply $K(2, 1; m_1, e_1)$
   - Accumulate into `total_series`

4. **Truncate** to `qq_order = 10`.

### Verify

All NC cycles should give the same result when $\eta \to 1$:
```python
result.eta1_series()  # should match across all NC cycles
```

---

## Summary of Function Call Chain

```
DehnFillingWorker._run()
│
├── compute_filled_index(nz, cusp_idx, P, Q, ...)     ← Step 1 (NC search)
│   ├── find_rs(P, Q)
│   ├── enumerate_kernel_terms(...)
│   │   └── _particular_solution(P, Q, c)
│   │   └── enumerate_summation_terms(nz, m_ext, e_ext, ...)
│   └── compute_index_3d_python(nz, ...)
│       └── _tet_index_series(m, e, qq_order)
│
├── find_rs(P_nc, Q_nc)                                ← Step 2 (complement)
├── apply_general_cusp_basis_change(nz, cusp_idx, ...) ← Step 2 (basis change)
├── transform_weyl_vectors(weyl_a, weyl_b, ...)        ← Step 2b (Weyl transform)
│
├── (p, q) = slope transform                           ← Step 3
│
└── compute_filled_refined_index(nz_nc, cusp_idx, p, q, weyl_a_nc, weyl_b_nc, ...)
    │                                                       ← Steps 4–7
    ├── hj_continued_fraction(p, q)                     ← Step 4
    │
    ├── [if ℓ = 1]:
    │   ├── _enumerate_slope1_terms(k1, m1_range)       ← Step 5a
    │   ├── compute_refined_index(nz_nc, m_ext, e_ext, ...)  ← Step 5b
    │   │   └── enumerate_summation_terms(...)
    │   │   └── _tet_index_series(...)
    │   ├── _apply_weyl_shift(weyl_a_nc, weyl_b_nc)     ← Step 5b (Weyl shift)
    │   └── _apply_k1_factor_multi(...)                  ← Step 5b
    │
    └── [if ℓ ≥ 2]:
        ├── Grid scan → compute_refined_index(...)       ← Step 6b
        ├── _apply_weyl_shift(weyl_a_nc, weyl_b_nc)     ← Step 6b (Weyl shift)
        ├── _apply_is_step(...) × (ℓ−1)                 ← Step 6c
        │   └── _is_kernel(m, e_in, m1, e1, ...)        ← Step 6d
        │       └── _etilde_is(m1, e1, m2, e2, ...)     ← Step 6e
        │           └── _tet_index_series(...)  × 4
        ├── _enumerate_slope1_all(k_ℓ, ...)              ← Step 6f
        └── _apply_k1_factor_multi(...)                  ← Step 6f
```

---

## Appendix D — Weyl Vectors: Calculation and Application

### D.1 Mathematical definition

For a manifold with `num_hard` hard edges, the refined index
$I^{\text{ref}}(m, e; \eta_0, \ldots, \eta_{k-1})$ carries one fugacity $\eta_j$
per hard edge. In general $I^{\text{ref}}$ is **not** invariant under
$\eta_j \to \eta_j^{-1}$ (Weyl reflection).

The Weyl vectors $a \in \mathbb{Z}^k$ and $b \in (\mathbb{Z}/2)^k$ define a
**Weyl monomial** $\eta^{b \cdot m + a \cdot e}$ such that the shifted series

$$
f(m, e; \eta) = \prod_j \eta_j^{b_j \Sigma m + a_j \Sigma e} \cdot I^{\text{ref}}(m, e; \eta)
$$

is Weyl-symmetric: $f(\eta) = f(\eta^{-1})$.

Equivalently, the refined index has the form

$$
I^{\text{ref}}(m, e; \eta) = \eta^{-(b \cdot m + a \cdot e)} \cdot (\text{Weyl-symmetric series in } \eta)
$$

### D.2 Extraction algorithm (`compute_ab_vectors`)

**Input:** A table of refined index evaluations $(m_{\text{ext}}, e_{\text{ext}}, I^{\text{ref}})$
at the 25ʳ-point grid (5 values each for $m$ and $e$ per cusp).

**Step 1:** For each evaluation, compute the coefficient-weighted η-centre at
the leading q-order:
$$
\text{centre}_j = \frac{\sum_{\text{terms at min q}} \eta_j^{\text{exp}} \cdot \text{coeff}}{\sum_{\text{terms at min q}} \text{coeff}}
$$

**Step 2:** The centre satisfies
$$
\text{centre}_j(m, e) = b_j \Sigma m + \frac{a_j}{2} \Sigma e + \text{const}
$$

**Step 3:** Extract $b_j$ from pairs with $e = 0$:
$$
b_j = -\frac{\text{centre}_j(+m, 0) - \text{centre}_j(-m, 0)}{2 \Sigma|m|}
$$

**Step 4:** Extract $a_j$ from pairs with $m = 0$:
$$
a_j = -\frac{\text{centre}_j(0, +e) - \text{centre}_j(0, -e)}{\Sigma|e|}
$$

**Step 5:** Validate: $a_j \in \mathbb{Z}$, $b_j \in \mathbb{Z}/2$.

### D.3 Application (`_apply_weyl_shift`)

Before the Dehn filling kernel acts on $I^{\text{ref}}(m_{\text{ext}}, e_{\text{ext}})$,
we multiply by the Weyl monomial.

In the doubled-exponent key convention (keys are
`(qq_power, 2·η_0_exp, ..., 2·η_{k-1}_exp)`):

```
shift_x2[j] = 2 · b_stored[j] · Σm  +  a_stored[j] · Σe
new_key = (qq_power, 2η_0 + shift_x2[0], ..., 2η_{k-1} + shift_x2[k-1])
```

This is applied at two points in the code:
1. **ℓ = 1 path** (Step 5b): after computing each `refined` and before `_apply_k1_factor_multi`
2. **ℓ ≥ 2 path** (Step 6b): after computing each `refined` in the grid scan, before IS convolution

### D.4 Per-NC-cycle transform (`transform_weyl_vectors`)

When the cusp basis changes by
$\begin{pmatrix} P_{\text{nc}} & Q_{\text{nc}} \\ R & S \end{pmatrix}$,
the Weyl vectors transform as:

| Quantity | Formula (stored convention) |
|----------|----------------------------|
| $a^{\text{nc}}_{\text{stored}}[j]$ | $P_{\text{nc}} \cdot a_{\text{stored}}[j] - 4 Q_{\text{nc}} \cdot b_{\text{stored}}[j]$ |
| $b^{\text{nc}}_{\text{stored}}[j]$ | $S \cdot b_{\text{stored}}[j] - \frac{R}{4} \cdot a_{\text{stored}}[j]$ |

**Key properties:**
- Zero additional computation cost (pure algebra, no re-evaluation of $I^{\text{ref}}$)
- Exact — avoids numerical centre estimation in the new basis
- Identity transform: $(P_{\text{nc}}, Q_{\text{nc}}, R, S) = (1, 0, 0, 1)$ leaves $(a, b)$ unchanged
- Roundtrip: applying a transform then its SL(2,ℤ) inverse recovers the original vectors
- Integrality preserved: $a^{\text{nc}} \in \mathbb{Z}$, $b^{\text{nc}} \in \mathbb{Z}/2$