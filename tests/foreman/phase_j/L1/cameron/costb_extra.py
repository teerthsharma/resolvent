"""Parity at S=4096 (fused fp32 vs contract-order eager f64) and the forward ratio
across S, same B1 H8 D64 fp32 causal shape family. Writes costb_extra.json."""
import json, os, torch
import torch.nn.functional as F
import test_costb_fused as T, costb_fused as cf, costb_compile as cc

res = {}
with torch.no_grad():
    x, w, q, k, v = T.inputs(1, 2, 4096, 64, seed=5)
    ref = cc.bprime_eager(*(t.double() for t in (x, w, q, k, v)))
    res["S4096_H2_max_abs_err_fused_fp32_vs_eager_f64"] = (cf.mode_bprime_fused(x, w, q, k, v).double() - ref).abs().max().item()
    del ref
    torch.cuda.empty_cache()
    for S in (1024, 2048, 4096, 8192):
        x, w, q, k, v = T.inputs(1, 8, S, 64)
        s = T.cuda_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        m = T.cuda_ms(lambda: cf.mode_bprime_fused(x, w, q, k, v))
        res["S%d" % S] = dict(ms_sdpa=s, ms_fused=m, ratio=m / s)
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(T.HERE, "costb_extra.json"), "w"), indent=1)
