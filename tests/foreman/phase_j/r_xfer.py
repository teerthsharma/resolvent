# -*- coding: utf-8 -*-
"""Row R-XFER: rank 32,704 sites by (f)'s a_t; d_FoX = NLL_a - NLL_FoX.
Within each NLL_a quintile, top-minus-bottom a_t decile of d_FoX.
Control: same statistic for d_f = NLL_a - NLL_f (should reproduce R-STRAT's
+0.030/+0.112/+0.233/+0.300/+0.732). GPU eval only, no training.
Reuses build_arm/eval_batches/per_token_nll/row_mass from abstention_deciles.py
and arms_j.build_for_eval for the FoX arm, exactly as hour one trained it.
"""
import io
import json
import os
import sys
import time

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
REPO = r"C:\Users\seal\Desktop\New folder (32)"
PHASE_J = os.path.join(SCRATCH, "phase_j")
sys.path.insert(0, PHASE_J)
sys.path.insert(0, SCRATCH)
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import design4x5 as D  # noqa: E402
from abstention_deciles import build_arm, eval_batches, per_token_nll, row_mass  # noqa: E402
from ceq.hf.modeling_ceq import CEQAttention  # noqa: E402
import arms_j  # noqa: E402

OUT_JSON = os.path.join(PHASE_J, "r_xfer.json")
RECORD = os.path.join(PHASE_J, "record.jsonl")
CKPT_FOX = os.path.join(PHASE_J, "ckpt_a2F_ss0", "model.pt")

BAR = [0.015, 0.056, 0.117, 0.150, 0.366]
CONTROL_EXPECT = [0.030, 0.112, 0.233, 0.300, 0.732]

started = time.strftime("%H:%M:%S")
t0 = time.time()

D.poll_until_free(timeout_s=900)

m_a, blob_a = build_arm("a")
m_f, blob_f = build_arm("f")

m_fox, fox_fwd = arms_j.build_for_eval("a2F", split_seed=0)
fox_blob = torch.load(CKPT_FOX, map_location="cpu", weights_only=False)
m_fox.load_state_dict(fox_blob["state_dict"], strict=True)
m_fox.to(D.DEVICE).eval()

batches = eval_batches()

d_fox_l, d_f_l, a_t_l, nll_a_l = [], [], [], []
for x in batches:
    nll_a = per_token_nll(m_a, x, softmax_twin=True)
    nll_f = per_token_nll(m_f, x, softmax_twin=False)

    ctx = CEQAttention.forward
    CEQAttention.forward = fox_fwd
    try:
        nll_fox = per_token_nll(m_fox, x, softmax_twin=False)
    finally:
        CEQAttention.forward = ctx

    a_t = row_mass(m_f, x)[:, :-1]
    a_t = 1.0 - a_t

    d_fox_l.append((nll_a - nll_fox).reshape(-1))
    d_f_l.append((nll_a - nll_f).reshape(-1))
    a_t_l.append(a_t.reshape(-1))
    nll_a_l.append(nll_a.reshape(-1))

d_fox = torch.cat(d_fox_l).cpu()
d_f = torch.cat(d_f_l).cpu()
a_t = torch.cat(a_t_l).cpu()
nll_a = torch.cat(nll_a_l).cpu()
n = d_fox.numel()


def top_minus_bottom_by_quintile(d, a, nll):
    order = torch.argsort(nll)
    qe = [round(n * i / 5) for i in range(6)]
    out = []
    for i in range(5):
        idx = order[qe[i]:qe[i + 1]]
        a_s, d_s = a[idx], d[idx]
        m = a_s.numel()
        a_order = torch.argsort(a_s)
        dec = max(1, round(m / 10))
        bottom = d_s[a_order[:dec]].mean()
        top = d_s[a_order[-dec:]].mean()
        out.append(float(top - bottom))
    return out

fox_diffs = top_minus_bottom_by_quintile(d_fox, a_t, nll_a)
f_diffs = top_minus_bottom_by_quintile(d_f, a_t, nll_a)

passes = [fd >= b for fd, b in zip(fox_diffs, BAR)]
verdict = "PASS" if all(passes) else "FAIL"

secs = time.time() - t0
finished = time.strftime("%H:%M:%S")

res = dict(
    n_tokens=n,
    fox_diffs=fox_diffs,
    bar=BAR,
    passes=passes,
    verdict=verdict,
    control_f_diffs=f_diffs,
    control_expect=CONTROL_EXPECT,
    seconds=secs,
)
io.open(OUT_JSON, "w", encoding="utf-8").write(json.dumps(res, indent=1))
print(json.dumps(res, indent=1), flush=True)

row = dict(
    row="R-XFER",
    arms=["a", "f", "a2F"],
    seeds=[0],
    split_seed=0,
    machine="RTX 4060",
    seconds_per_cell=secs,
    tokens_seen=n,
    clock_ratio=None,
    read="exact",
    queue_pos=1,
    started=started,
    finished=finished,
    bar=("top-minus-bottom a_t decile of d_FoX >= "
         "+0.015/+0.056/+0.117/+0.150/+0.366 in all 5 NLL_a quintiles -> PASS "
         "(abstention marks sites any forget gate wins, retire abstention rows); "
         "else FAIL (abstention stays (f)'s own)"),
    measured=dict(d_fox_top_minus_bottom=fox_diffs),
    verdict=verdict,
    control=dict(d_f_top_minus_bottom=f_diffs, expect=CONTROL_EXPECT),
    producer="python phase_j/r_xfer.py",
    output="phase_j/r_xfer.json",
    repro_class="SDPA tolerance",
    killed="ranking the same 32,704 sites by (f)'s a_t and reading d_FoX = NLL_a - NLL_FoX "
           "within each NLL_a quintile, against the abstention-marks-a-forget-gate-win bar",
)
with io.open(RECORD, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
