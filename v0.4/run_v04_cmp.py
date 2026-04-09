import sys
sys.path.insert(0, '/Users/pmp/Documents/Research/ultimate/v0.4/src')

from manifold_index.core.manifold import load_manifold
from manifold_index.core import neumann_zagier as nz_mod
from manifold_index.core import phase_space as ps_mod
from manifold_index.core import dehn_filling as df_mod
from manifold_index.core import refined_dehn_filling as rdf_mod

md = load_manifold('m006')
easy = ps_mod.find_easy_edges(md)
nz = nz_mod.build_neumann_zagier(md, easy)

nc_r = df_mod.find_non_closable_cycles(nz, cusp_idx=0,
    p_range=range(-5, 6), q_range=range(-5, 6), q_order_half=20)
nc = list(nc_r.cycles)[0]
nc_P, nc_Q = nc.P, nc.Q

user_P, user_Q = -4, 1
R, S = df_mod.find_rs(nc_P, nc_Q)
p = R * user_Q - S * user_P
q = nc_P * user_Q - nc_Q * user_P
nz_nc = nz_mod.apply_general_cusp_basis_change(nz, 0, a=nc_P, b=nc_Q, c=-R, d=-S)
# qq=10 to match v0.4 app's Nmax=10
result_qq10 = rdf_mod.compute_filled_refined_index(nz_nc, cusp_idx=0, P=p, Q=q,
    m_other=[], e_other=[], q_order_half=10, weyl_a=None, weyl_b=None)
result_qq20 = rdf_mod.compute_filled_refined_index(nz_nc, cusp_idx=0, P=p, Q=q,
    m_other=[], e_other=[], q_order_half=20, weyl_a=None, weyl_b=None)

import json
for tag, result in [('qq10', result_qq10), ('qq20', result_qq20)]:
    out = {str(k): [v.numerator, v.denominator] for k, v in sorted(result.series.items())}
    with open(f'/Users/pmp/Documents/Research/ultimate/v04_m006_{tag}.json', 'w') as f:
        json.dump(out, f)
    print(f"v0.4 {tag}: NC=({nc_P},{nc_Q})  p={p}, q={q}  {len(out)} entries", flush=True)
