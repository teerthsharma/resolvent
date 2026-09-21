# -*- coding: utf-8 -*-
"""The abstention decile test, written before the pair it reads exists.

The hypothesis under test: the gated arm's advantage is carried by the tokens
where its rows abstain -- where row mass falls furthest below one -- rather than
being spread evenly across the corpus.

Pre-registered, and this file is committed before pair_with_weights.py finishes
so the thresholds cannot be chosen after seeing the answer:

  d_t  = per-token NLL of the softmax twin (a) minus that of the gated arm (f),
         on the SAME held-out tokens. Positive means the gated arm is better
         at that token.
  a_t  = 1 - mean over heads and over layers 1..3 of |sum_j W_tj|, read off the
         gated arm at query position t. Layer 0 is excluded: its exact-zero
         fraction was measured at 100% float32 underflow of the path product,
         which is a different mechanism from gate closure.

  PREDICTION passes iff BOTH hold:
      (i)  mean d_t in the top abstention decile > mean d_t in the bottom decile
      (ii) the top three deciles carry more than 50% of the summed gap
  KILL fires iff mean d_t in the top decile <= mean d_t in the bottom decile.
  CONTROL: stratify the same tokens by the softmax twin's OWN NLL into
      quintiles. The sign of mean d_t must not flip in any stratum. A flip
      means the deciles were reading difficulty, not abstention.

Row mass is computed by the same arithmetic as tests/chase/rowmass/
rowmass_opus_check.py -- |sum_j num_ij| / Z_i at the operator's own consumption
site -- rather than a second implementation of it.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\seal\Desktop\New folder (32)"
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import design4x5 as D  # noqa: E402
import r3_eval as RE  # noqa: E402
import r1_gate as R  # noqa: E402
from ceq import arm_smprime as A  # noqa: E402
from ceq.hf.modeling_ceq import CEQAttention  # noqa: E402

SPLIT_SEED = 0
SEED = 0
LAYERS_FOR_MASS = (1, 2, 3)     # layer 0 excluded: underflow, not closure
OUT = os.path.join(HERE, "abstention_deciles.json")
BOARD = os.path.join(REPO, "house-events.jsonl")


def board(**kw):
    kw.setdefault("agent", "Chase")
    try:
        with io.open(BOARD, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(kw) + "\n")
    except Exception:
        pass


def ckpt_path(arm):
    return os.path.join(D.OUT_DIR_TMPL.format(arm=arm, ss=SPLIT_SEED) + "_pair",
                        "model.pt")


def build_arm(arm):
    """Rebuild exactly the arm design4x5.train_arm trained, then load its weights."""
    common = dict(hidden_size=D.HIDDEN, n_layers=D.LAYERS, n_heads=D.HEADS,
                  seq=D.SEQ, vocab_size=D.VOCAB)
    if arm == "a":
        model = RE.build(operator="sgate", **common)
    elif arm == "f":
        A.magnitude = R.FORMS["hard_concrete"]
        model = R.build_repaired(operator="smprime", **common)
    else:
        raise ValueError(arm)
    blob = torch.load(ckpt_path(arm), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(D.DEVICE).eval()
    return model, blob


def eval_batches():
    """The SAME held-out batches the two arms were scored on: same corpus, same
    by-document split at this split_seed, same fixed eval generator seed."""
    data = RE.DocByteBatches(RE._corpus_text(None, 64 * 1024 * 1024), D.VOCAB,
                             val_frac=0.1, split_seed=SPLIT_SEED)
    eg = torch.Generator().manual_seed(20260921 + SPLIT_SEED)
    return [data.batch("val", D.BATCH, D.SEQ, eg, D.DEVICE)[0]
            for _ in range(D.EVAL_BATCHES)]


def per_token_nll(model, x, softmax_twin):
    """[B, S-1] NLL at each prediction site, standard next-token shift."""
    ctx = CEQAttention.forward
    if softmax_twin:
        CEQAttention.forward = D.softmax_forward
    try:
        with torch.no_grad():
            logits = model(input_ids=x).logits
    finally:
        CEQAttention.forward = ctx
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                           x[:, 1:].reshape(-1), reduction="none"
                           ).view(x.shape[0], -1).double()


def row_mass(model, x):
    """[B, S] mean over heads and LAYERS_FOR_MASS of |sum_j W_tj|.

    Read at the operator's consumption site, the same arithmetic as
    rowmass_opus_check.py: |sum_j num_ij| / Z_i, with Z_i = sum_j mod_ij.
    """
    layers = model.model.layers if hasattr(model, "model") else model.layers
    caps = {}

    def mk(i):
        def hook(mod, inputs):
            caps[i] = inputs[0].detach()
        return hook

    hs = [blk.self_attn.register_forward_pre_hook(mk(i))
          for i, blk in enumerate(layers)]
    try:
        with torch.no_grad():
            model(input_ids=x)
    finally:
        for h in hs:
            h.remove()

    acc = None
    for i in LAYERS_FOR_MASS:
        at, h = layers[i].self_attn, caps[i]
        b, s, _ = h.shape
        q, k, _ = at.qkv(h).chunk(3, dim=-1)
        shp = lambda t: t.view(b, s, at.n_heads, at.d_head).transpose(1, 2)
        q, k = shp(q), shp(k)
        u = at.m_head(h).squeeze(-1).unsqueeze(-2)
        th = at.theta_head(h).squeeze(-1).unsqueeze(-2)
        per_b = []
        for bi in range(b):
            with torch.no_grad():
                num, mod = A.numerator(q[bi:bi + 1], k[bi:bi + 1],
                                       u[bi:bi + 1], th[bi:bi + 1],
                                       qk=at.qk, g=at.g, route="product",
                                       phase_route="gate")
                mi = (num.sum(-1).abs().double()
                      / mod.sum(-1).double().clamp_min(1e-300))   # [1,H,S]
            per_b.append(mi.mean(1))                              # [1,S]
            del num, mod
        m = torch.cat(per_b, 0)                                   # [B,S]
        acc = m if acc is None else acc + m
    return acc / len(LAYERS_FOR_MASS)


def summarize(d, a, nll_a):
    """Deciles on abstention, plus the difficulty control."""
    order = torch.argsort(a)
    n = d.numel()
    edges = [round(n * i / 10) for i in range(11)]
    dec = [d[order[edges[i]:edges[i + 1]]] for i in range(10)]
    means = [float(x.mean()) for x in dec]
    sums = [float(x.sum()) for x in dec]
    total = float(d.sum())

    top3 = sum(sums[7:]) / total if total != 0 else float("nan")
    passes_i = means[-1] > means[0]
    passes_ii = top3 > 0.5
    kill = means[-1] <= means[0]

    q_order = torch.argsort(nll_a)
    qe = [round(n * i / 5) for i in range(6)]
    strata = [float(d[q_order[qe[i]:qe[i + 1]]].mean()) for i in range(5)]
    signs = {s > 0 for s in strata}
    control_ok = len(signs) == 1

    return dict(
        n_tokens=int(n),
        decile_mean_d=means, decile_sum_d=sums, total_d=total,
        top_decile_mean=means[-1], bottom_decile_mean=means[0],
        top3_share_of_gap=top3,
        prediction_i_top_gt_bottom=bool(passes_i),
        prediction_ii_top3_over_half=bool(passes_ii),
        PREDICTION=bool(passes_i and passes_ii),
        KILL=bool(kill),
        control_quintile_mean_d=strata,
        control_sign_stable=bool(control_ok),
        abstention_mean=float(a.mean()), abstention_max=float(a.max()),
    )


def main():
    for arm in ("a", "f"):
        if not os.path.exists(ckpt_path(arm)):
            print("[ABST] missing checkpoint for arm ({}): {}".format(
                arm, ckpt_path(arm)), flush=True)
            print("[ABST] run pair_with_weights.py first -- NOT rebuilding a "
                  "model that was never trained", flush=True)
            return 2

    m_a, blob_a = build_arm("a")
    m_f, blob_f = build_arm("f")
    print("[ABST] a n_params={}  f n_params={}".format(
        blob_a["n_params"], blob_f["n_params"]), flush=True)

    ds, as_, nlls = [], [], []
    for x in eval_batches():
        nll_a = per_token_nll(m_a, x, softmax_twin=True)
        nll_f = per_token_nll(m_f, x, softmax_twin=False)
        mass = row_mass(m_f, x)[:, :-1]          # align to prediction sites
        ds.append((nll_a - nll_f).reshape(-1))
        as_.append((1.0 - mass).reshape(-1))
        nlls.append(nll_a.reshape(-1))

    res = summarize(torch.cat(ds).cpu(), torch.cat(as_).cpu(), torch.cat(nlls).cpu())
    res.update(split_seed=SPLIT_SEED, seed=SEED,
               layers_for_mass=list(LAYERS_FOR_MASS),
               n_params_a=blob_a["n_params"], n_params_f=blob_f["n_params"],
               mean_d=float(torch.cat(ds).mean()))
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)

    verdict = ("PREDICTION" if res["PREDICTION"] else
               "KILL" if res["KILL"] else "NEITHER")
    board(t="test", state=("GREEN" if res["PREDICTION"] else "RED"),
          name="abstention_deciles",
          detail="{} top={:.4f} bottom={:.4f} top3_share={:.3f} control_stable={}"
                 .format(verdict, res["top_decile_mean"], res["bottom_decile_mean"],
                         res["top3_share_of_gap"], res["control_sign_stable"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
