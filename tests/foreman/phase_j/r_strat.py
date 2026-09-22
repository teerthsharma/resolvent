# -*- coding: utf-8 -*-
"""Row R-STRAT: re-stratify the abstention/difficulty test on the saved pair.
GPU eval only, no training. Reuses build_arm/eval_batches/per_token_nll/row_mass
from abstention_deciles.py rather than reimplementing them.
"""
import io
import json
import os
import sys
import time

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
REPO = r"C:\Users\seal\Desktop\New folder (32)"
sys.path.insert(0, SCRATCH)
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import design4x5 as D  # noqa: E402
from abstention_deciles import build_arm, eval_batches, per_token_nll, row_mass  # noqa: E402
from ceq.hf.modeling_ceq import CEQAttention  # noqa: E402

OUT_JSON = os.path.join(SCRATCH, "phase_j", "r_strat.json")
RECORD = os.path.join(SCRATCH, "phase_j", "record.jsonl")

t0 = time.time()

m_a, blob_a = build_arm("a")
m_f, blob_f = build_arm("f")

batches = eval_batches()

ds, as_, nlls, ents = [], [], [], []
for x in batches:
    nll_a = per_token_nll(m_a, x, softmax_twin=True)
    nll_f = per_token_nll(m_f, x, softmax_twin=False)
    mass = row_mass(m_f, x)[:, :-1]

    # softmax twin's own next-token entropy at each prediction site
    ctx = CEQAttention.forward
    CEQAttention.forward = D.softmax_forward
    try:
        with torch.no_grad():
            logits_a = m_a(input_ids=x).logits[:, :-1].double()
    finally:
        CEQAttention.forward = ctx
    logp = F.log_softmax(logits_a, dim=-1)
    p = logp.exp()
    H = -(p * logp).sum(-1)

    ds.append((nll_a - nll_f).reshape(-1))
    as_.append((1.0 - mass).reshape(-1))
    nlls.append(nll_a.reshape(-1))
    ents.append(H.reshape(-1))

d = torch.cat(ds).cpu()
a_t = torch.cat(as_).cpu()
nll_a = torch.cat(nlls).cpu()
H_t = torch.cat(ents).cpu()
n = d.numel()

# (1) quintiles by NLL_a: mean d_t, mean d_t/NLL_a
order = torch.argsort(nll_a)
qe = [round(n * i / 5) for i in range(6)]
q_mean_d, q_mean_dratio = [], []
for i in range(5):
    idx = order[qe[i]:qe[i + 1]]
    dq, nq = d[idx], nll_a[idx]
    q_mean_d.append(float(dq.mean()))
    q_mean_dratio.append(float((dq / nq.clamp_min(1e-12)).mean()))
ratio_span = max(q_mean_dratio) / min(q_mean_dratio) if min(q_mean_dratio) > 0 else float("inf")

# (2), (3) Spearman correlations
rho_a_H, p_a_H = spearmanr(a_t.numpy(), H_t.numpy())
rho_a_nll, p_a_nll = spearmanr(a_t.numpy(), nll_a.numpy())

# (4) within each NLL_a quintile, mean d_t in top vs bottom a_t decile of that stratum
within = []
for i in range(5):
    idx = order[qe[i]:qe[i + 1]]
    a_s, d_s = a_t[idx], d[idx]
    m = a_s.numel()
    a_order = torch.argsort(a_s)
    dec = max(1, round(m / 10))
    bottom = d_s[a_order[:dec]].mean()
    top = d_s[a_order[-dec:]].mean()
    within.append(dict(top_mean_d=float(top), bottom_mean_d=float(bottom),
                        diff=float(top - bottom)))

headroom = ratio_span < 1.5
all_positive = all(w["diff"] > 0 for w in within)
verdict = "HEADROOM" if headroom else (
    "ABSTENTION SURVIVES STRATIFICATION" if all_positive else "DIFFICULTY")

res = dict(
    n_tokens=n,
    nll_a_quintile_mean_d=q_mean_d,
    nll_a_quintile_mean_d_over_nll=q_mean_dratio,
    d_over_nll_ratio_span=ratio_span,
    spearman_a_vs_entropy=dict(rho=float(rho_a_H), p=float(p_a_H)),
    spearman_a_vs_nll_a=dict(rho=float(rho_a_nll), p=float(p_a_nll)),
    within_quintile_top_minus_bottom_decile=within,
    verdict=verdict,
    n_params_a=blob_a["n_params"], n_params_f=blob_f["n_params"],
    seconds=time.time() - t0,
)
io.open(OUT_JSON, "w", encoding="utf-8").write(json.dumps(res, indent=1))
print(json.dumps(res, indent=1), flush=True)

row = dict(
    row="R-STRAT",
    arms=["a", "f"],
    seeds=[0],
    split_seed=0,
    machine="RTX 4060",
    seconds_per_cell=res["seconds"],
    tokens_seen=n,
    params_numel=dict(a=blob_a["n_params"], f=blob_f["n_params"]),
    bar="d/NLL_a max/min < 1.5 -> HEADROOM; else all-quintile top>bottom -> ABSTENTION SURVIVES; else DIFFICULTY",
    measured=dict(
        d_over_nll_ratio_span=ratio_span,
        spearman_a_vs_entropy=rho_a_H,
        spearman_a_vs_nll_a=rho_a_nll,
        within_quintile_diffs=[w["diff"] for w in within],
    ),
    verdict=verdict,
    control=dict(nll_a_quintile_mean_d=q_mean_d),
    quintile_profile=dict(abs=q_mean_d, rel=q_mean_dratio),
    producer="python phase_j/r_strat.py",
    output="phase_j/r_strat.json",
    repro_class="SDPA tolerance",
    killed="re-stratifying by NLL_a quintile and by within-quintile abstention decile, "
           "on the same 32,704 sites as abstention_deciles.json",
)
with io.open(RECORD, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
