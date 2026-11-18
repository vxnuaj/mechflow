"""L1 (prereg §2.4) on the circuit-validation split: per document run base, the full neighbour-feedback cut ('all'),
the frozen head set H*, and 4 of the 20 layer-matched random head sets (rotated by task_id: sets (4·tid + r) mod 20).
Same measurement code as jobs_cutb (batched headcut, MR, dep/coh/drop at j = 2, 3)."""
import json, os

import jobs_cutb
from jobs_cut import cuts as _all_cuts
import jobs_cut

ROT = 4


def l1_task(task, ctx):
    allc = _all_cuts()
    keep = ["all", "Hstar"] + [f"R{(4 * task['task_id'] + r) % 20:02d}" for r in range(ROT)]
    saved = dict(allc)
    try:
        jobs_cut._CUTS = {k: saved[k] for k in keep}
        out = jobs_cutb.cut_task(task, ctx)
    finally:
        jobs_cut._CUTS = saved
    out["random_sets"] = keep[2:]
    return out
