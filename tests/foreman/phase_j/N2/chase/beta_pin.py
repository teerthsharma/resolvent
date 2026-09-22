"""Price of the beta = 1 reroute: held-out NLL of the trained (f) with every layer's
beta pinned to 1 (evaluation only), and gamma* of that pinned operator. CPU only."""
import json
import os

import torch
import torch.nn.functional as F

import rdiag_prime as RP          # sets CPU, paths, imports AD and A
from rdiag_prime import AD, A, _capture, _heads

OUT = os.path.join(RP.HERE, "beta_pin.json")


def _nll_and_gstar(model):
    s, n, gmin = 0.0, 0, float("inf")
    for x in AD.eval_batches():
        logits, ins, _ = _capture(model, x)
        s += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   x[:, 1:].reshape(-1), reduction="sum"))
        n += x[:, 1:].numel()
        with torch.no_grad():
            for li, blk in enumerate(model.model.layers):
                at = blk.self_attn
                q, k, v, u, th = _heads(at, ins[li])
                d = A.operator(q, k, u, th, beta=at.beta, qk=at.qk, g=at.g
                               ).diagonal(dim1=-2, dim2=-1).real
                # run 1 took 1/min over pairs of max_i W_ii (the LARGEST gamma*): 0.934 vs
                # rdiag_prime's 0.4836 exposed it. gamma*_min = 1 / max over everything.
                gmin = min(gmin, float(1.0 / d.max().double()))
    return s / n, gmin


def results():
    model, _ = AD.build_arm("f")
    nll_t, g_t = _nll_and_gstar(model)
    with torch.no_grad():
        for blk in model.model.layers:
            blk.self_attn.beta.fill_(1.0)
    nll_1, g_1 = _nll_and_gstar(model)
    out = dict(nll_trained=nll_t, trained_gstar_min=g_t, nll_beta1=nll_1,
               beta1_gstar_min=g_1, dnll=nll_1 - nll_t)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out), flush=True)
    return out
