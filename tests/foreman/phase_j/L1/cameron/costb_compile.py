"""No-hand-written-kernel route: contract-order eager Mode B' (head-shared scan,
Hillis-Steele with later*earlier), run under torch.compile. Forward ratio only.
IMPL=compile in test_costb_fused.py is not wired; this file times itself."""
import json, os, torch
import torch.nn.functional as F
import test_costb_fused as T


def scan_contract(q):
    S = q.shape[-2]
    acc, off = q, 1
    one = torch.tensor([1.0, 0, 0, 0], device=q.device, dtype=q.dtype)
    while off < S:
        shifted = torch.cat([one.expand(*q.shape[:-2], off, 4), acc[..., :-off, :]], -2)
        acc = T.qmul(acc, shifted)  # later * earlier
        off *= 2
    return acc


def rot(t, P, conj):
    B, H, S, D = t.shape
    if conj:
        P = P * torch.tensor([1.0, -1, -1, -1], device=P.device, dtype=P.dtype)
    return T.qmul(P[:, None, :, None, :].expand(B, H, S, D // 4, 4), t.reshape(B, H, S, D // 4, 4)).reshape(B, H, S, D)


def bprime_eager(x, w, q, k, v):
    P = scan_contract(F.normalize(x @ w.T, dim=-1))
    return rot(F.scaled_dot_product_attention(q, k, rot(v, P, True), is_causal=True), P, False)


if __name__ == "__main__":
    comp = torch.compile(bprime_eager)
    a = T.inputs(1, 2, 64, 64, seed=1)
    fold_err = (comp(*a).double().cpu() - T.contract_ref(*a)).abs().max().item()
    x, w, q, k, v = T.inputs(1, 8, 4096, 64)
    with torch.no_grad():
        ts, te, tc = [], [], []
        for _ in range(3):
            ts.append(T.cuda_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)))
            te.append(T.cuda_ms(lambda: bprime_eager(x, w, q, k, v)))
            tc.append(T.cuda_ms(lambda: comp(x, w, q, k, v)))
    s, e, c = (sorted(t)[1] for t in (ts, te, tc))
    res = dict(fold_err_vs_contract=fold_err, ms_sdpa=s, ms_eager_contract=e, ms_compiled=c,
               ratio_eager=e / s, ratio_compiled=c / s)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(T.HERE, "costb_compile.json"), "w"), indent=1)
