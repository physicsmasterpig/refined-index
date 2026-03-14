"""
app/interface.py — User-facing interaction layer (CLI).

Handles:
  - Accepting manifold name and optional parameter overrides from the user
  - Displaying intermediate results (non-closable cycles, etc.)
  - Collecting user selections (basis choice per cusp)
  - Printing final output
"""

from manifold_index.core.manifold import load_manifold
from manifold_index.core.index_3d import compute_3d_index
from manifold_index.core.dehn_filling import find_non_closable_cycles
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.refined_index import compute_refined_index


def run_interactive() -> None:
    """Main interactive CLI loop."""
    # --- Step 1: Manifold input ---
    name = input("Enter manifold name: ").strip()

    # --- Step 2: SnaPy extraction ---
    manifold_data = load_manifold(name)
    print(f"Cusps: {manifold_data.num_cusps}, Tetrahedra: {manifold_data.num_tetrahedra}")

    # --- Step 3: 3D index ---
    # TODO: accept optional range overrides from user
    index_3d = compute_3d_index(manifold_data)

    # --- Step 4: Dehn filling / non-closable cycles ---
    # TODO: accept optional slope range from user
    non_closable = find_non_closable_cycles(manifold_data, index_3d)

    # --- Step 5: Basis selection ---
    basis = _select_basis(non_closable, manifold_data.num_cusps)

    # --- Step 6: Easy edges / phase space basis ---
    phase_basis = find_easy_edges(manifold_data)

    # --- Step 7: Refined index ---
    result = compute_refined_index(index_3d, basis, phase_basis)
    print("Refined index:", result)


def _select_basis(non_closable: dict, num_cusps: int) -> dict:
    """
    Display non-closable cycles per cusp and let the user pick a basis cycle.
    Falls back to the default curve if no non-closable cycles exist.
    """
    basis = {}
    for cusp_idx in range(num_cusps):
        cycles = non_closable.get(cusp_idx, [])
        if not cycles:
            print(f"Cusp {cusp_idx}: no non-closable cycles found — using default curve.")
            basis[cusp_idx] = "default"
        else:
            print(f"Cusp {cusp_idx} non-closable cycles:")
            for i, cycle in enumerate(cycles):
                print(f"  [{i}] {cycle}")
            choice = int(input(f"  Select basis for cusp {cusp_idx}: "))
            basis[cusp_idx] = cycles[choice]
    return basis
