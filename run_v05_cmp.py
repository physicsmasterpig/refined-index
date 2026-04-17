import sys
sys.path.insert(0, '/Users/pmp/Documents/Research/ultimate/v0.5/src')

from manifold_index.services.compute_service import ComputeService
from manifold_index.services.filling_service import FillingService

_, _, nz = ComputeService.load_manifold('m006')
nc = list(FillingService.find_nc_cycles(
    nz, cusp_idx=0, p_range=(-5,5), q_range=(-5,5), q_order_half=20
).cycles)[0]

_, _, r_qq10 = FillingService.compute_filled_index(
    nz_data=nz, cusp_idx=0, nc_P=nc.P, nc_Q=nc.Q,
    user_P=-4, user_Q=1, m_other=[], e_other=[],
    q_order_half=10, weyl_a=None, weyl_b=None, manifold_name='m006',
)
_, _, r_qq20 = FillingService.compute_filled_index(
    nz_data=nz, cusp_idx=0, nc_P=nc.P, nc_Q=nc.Q,
    user_P=-4, user_Q=1, m_other=[], e_other=[],
    q_order_half=20, weyl_a=None, weyl_b=None, manifold_name='m006',
)

import json
for tag, r in [('qq10', r_qq10), ('qq20', r_qq20)]:
    out = {str(k): [v.numerator, v.denominator] for k, v in sorted(r.series.items())}
    with open(f'/Users/pmp/Documents/Research/ultimate/v05_m006_{tag}.json', 'w') as f:
        json.dump(out, f)
    print(f"v0.5 {tag}: {len(out)} entries written", flush=True)
