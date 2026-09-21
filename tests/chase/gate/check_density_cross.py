"""Opus check of the cross-density row: measure, at the blend() call site,
the init zero density / gate level / gate-gradient coverage / backward reach /
span-containment curve for the THREE CONFIGURATIONS THE REFERENCE TABLE
ACTUALLY RAN (m_head.bias = 1 - GATE_INIT_OFF = 0.999) alongside the two cross
cells.  Forward-only, no training, nothing on ceq/ is edited.

Command: python check_density_cross.py
"""
import io
import json
import os
import sys
import time

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
OUT = os.path.join(SCRATCH, "check_density_cross.json")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests", "chase", "gate"))

import torch  # noqa: E402
import r1_gate as G  # noqa: E402
import ceq.arm_smprime as arm_smprime  # noqa: E402
import ceq.hf.train as T  # noqa: E402
from ceq.arm_smprime import GATE_INIT_OFF  # noqa: E402

FORMS = {"clamp": G._ORIG_MAGNITUDE,
         "straight_through": G.magnitude_ste,
         "hard_concrete": G.magnitude_hardconcrete}

# name -> (form, m_head.bias).  The first three are the reference table's own
# rows (bias 0.999, set by r1_gate.build_repaired); the next two are the cross
# cells; the last is the pre-repair init, for scale.
CONFIGS = [
    ("table_row1_clamp_b0.999", "clamp", 1.0 - GATE_INIT_OFF),
    ("table_row2_ste_b0.999", "straight_through", 1.0 - GATE_INIT_OFF),
    ("table_row3_hardconcrete_b0.999", "hard_concrete", 1.0 - GATE_INIT_OFF),
    ("cell_a_clamp_b+3.0", "clamp", 3.0),
    ("cell_b_hardconcrete_b-2.4", "hard_concrete", -2.4),
    ("prerepair_clamp_b0.0", "clamp", 0.0),
]


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Opus-checker",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", json.dumps(row)[:400])


def build_with_bias(m_bias):
    def _build(*, hidden_size, n_layers, n_heads, seq, vocab_size=256, **ov):
        model = G._ORIG_BUILD(hidden_size=hidden_size, n_layers=n_layers,
                              n_heads=n_heads, seq=seq, vocab_size=vocab_size, **ov)
        with torch.no_grad():
            for layer in model.model.layers:
                layer.self_attn.m_head.bias.fill_(m_bias)
                layer.self_attn.theta_head.bias.fill_(GATE_INIT_OFF)
        return model
    return _build


def span_curve(m0, max_L=32):
    """P(a length-L window contains at least one exact zero), off the real
    layer-0 gate values.  flag+cumsum, same ground truth as
    r1_gate.spurious_zero_count, restricted to fixed-length windows."""
    B, S = m0.shape
    flag = (m0 == 0.0).to(torch.float64)
    cum = torch.cat([torch.zeros(B, 1, dtype=torch.float64, device=flag.device),
                     torch.cumsum(flag, dim=1)], dim=1)
    out = {}
    for L in range(1, min(max_L, S) + 1):
        st = torch.arange(0, S - L + 1, device=flag.device)
        out[L] = float(((cum[:, st + L] - cum[:, st]) > 0).to(torch.float64).mean())
    return out


def measure(name, form, m_bias, x_eval):
    arm_smprime.magnitude = FORMS[form]
    try:
        torch.manual_seed(G.SEED)
        model = build_with_bias(m_bias)(hidden_size=G.HIDDEN, n_layers=G.LAYERS,
                                        n_heads=G.HEADS, seq=G.SEQ, vocab_size=256,
                                        operator="smprime").to(G.DEVICE)
        m_layers, u_layers = G.gate_values(model, x_eval)
        ez = G.exact_zero_frac(m_layers)
        reach = G.backward_reach_stats(m_layers)
        m0 = m_layers[0]
        u = u_layers[0].detach().clone().requires_grad_(True)
        mm = FORMS[form](u)
        grad, = torch.autograd.grad(mm.sum(), u)
        frac_grad = float((grad != 0).to(torch.float64).mean())
        rec = dict(
            name=name, form=form, m_bias=float(m_bias),
            exact_zero_init=ez,
            frac_grad_nonzero_init=frac_grad,
            m_mean=float(m0.to(torch.float64).mean()),
            m_frac_eq_one=float((m0 == 1.0).to(torch.float64).mean()),
            m_frac_strictly_interior=float(((m0 > 0.0) & (m0 < 1.0)).to(torch.float64).mean()),
            u_mean=float(u_layers[0].to(torch.float64).mean()),
            u_std=float(u_layers[0].to(torch.float64).std()),
            backward_reach_init=reach,
            span_curve=span_curve(m0.cpu()),
            device=G.DEVICE, seed=G.SEED,
            shape=dict(hidden=G.HIDDEN, layers=G.LAYERS, heads=G.HEADS,
                       seq=G.SEQ, batch=G.BATCH),
            command="python check_density_cross.py",
        )
        del model
        if G.DEVICE == "cuda":
            torch.cuda.empty_cache()
        return rec
    finally:
        arm_smprime.magnitude = G._ORIG_MAGNITUDE


def main():
    data = T.ByteBatches(T._corpus_text(None, 64 * 1024 * 1024), vocab_size=256)
    gen = torch.Generator().manual_seed(12345)
    x_eval, _ = data.batch("val", G.BATCH, G.SEQ, gen, G.DEVICE)

    recs = {}
    for name, form, bias in CONFIGS:
        r = measure(name, form, bias, x_eval)
        recs[name] = r
        c = r["span_curve"]
        board("checker_init_density", name=name, form=form, m_bias=r["m_bias"],
              exact_zero_init=r["exact_zero_init"],
              frac_grad_nonzero_init=r["frac_grad_nonzero_init"],
              m_mean=r["m_mean"], m_frac_eq_one=r["m_frac_eq_one"],
              backward_reach_median=r["backward_reach_init"]["median"],
              backward_reach_mean=r["backward_reach_init"]["mean"],
              span_L2=c[2], span_L8=c[8], span_L16=c[16], span_L32=c[32])
        print(name, json.dumps({k: v for k, v in r.items() if k != "span_curve"}, indent=1)[:900])

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(recs, fh, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
