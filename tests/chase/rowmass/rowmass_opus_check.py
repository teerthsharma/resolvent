"""ROW MASS, read where the operator consumes G. Opus checker pass.

The q2 arm (f) checkpoints carry NO weights (run_record.json only; r3_eval
never calls torch.save). But a TRAINED smprime + hard_concrete model DOES
survive on disk: q3_ckpt_seed{0..4}_hard_concrete/, written by
tests/foreman/q2/q3_gatesweep.py via ceq.hf.train.train(save_every=100),
same operator and same gate form as arm (f), different shape (512/4 vs
128/3) and different row.

Row mass is computed from arm_smprime.numerator()'s OWN return values --
the complex numerator and the modulus row the operator divides by -- so the
quantity is read at the consumption site and is not reconstructed from
theta_head.

CPU only. No training. No GPU work queued.
Command: python rowmass_opus_check.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests", "chase", "gate"))

import torch  # noqa: E402

import ceq.arm_smprime as A  # noqa: E402
import ceq.hf.train as T  # noqa: E402
from ceq.hf.modeling_ceq import CEQForCausalLM  # noqa: E402
import r1_gate as R  # noqa: E402  -- owns FORMS["hard_concrete"]

DEVICE = "cpu"
CKPT = os.path.join(SCRATCH, "q3_ckpt_seed0_hard_concrete")
BATCH, SEQ = 8, 512
EVAL_SEED = 12345  # r1_gate.run_variant's own eval generator seed


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Chase",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    print("[board]", json.dumps(row, default=str)[:400], flush=True)


def lag1(x: torch.Tensor) -> float:
    """Lag-1 autocorrelation of a 1-D series, mean-removed."""
    x = x.to(torch.float64)
    x = x - x.mean()
    d = float((x * x).sum())
    if d == 0.0:
        return float("nan")
    return float((x[:-1] * x[1:]).sum() / d)


def hist(v: torch.Tensor, edges) -> list:
    return [int(((v >= edges[i]) & (v < edges[i + 1])).sum()) for i in range(len(edges) - 1)]


def main():
    torch.set_grad_enabled(False)

    # ---- L-REFLECTOR: the fixed structure, printed BEFORE any scored number
    A.magnitude = R.FORMS["hard_concrete"]   # the gate form this ckpt trained under
    model = CEQForCausalLM.from_pretrained(CKPT).to(DEVICE).eval()
    cfg = model.config
    rec = json.load(io.open(os.path.join(CKPT, "run_record.json"), encoding="utf-8"))
    layers = list(model.model.layers)

    fixed = dict(
        law="L-REFLECTOR",
        checkpoint=CKPT,
        checkpoint_is="TRAINED smprime + hard_concrete, from q3_gatesweep.py; "
                      "NOT the q2 arm (f) checkpoint, which carries no weights",
        operator=cfg.operator,
        gate_form="hard_concrete (r1_gate.magnitude_hardconcrete, deterministic)",
        hidden=cfg.hidden_size, n_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_attention_heads,
        d_head=cfg.hidden_size // cfg.num_attention_heads,
        seq=SEQ, batch=BATCH, device=DEVICE, dtype=str(cfg.dtype),
        steps=rec["steps"], n_params=rec["n_params"],
        final_train_loss=rec["losses"][-1],
        smp_start=dict(beta=cfg.smp_beta, qk=cfg.smp_qk, g=cfg.smp_g),
        eval_batch="ceq.hf.train.ByteBatches(_corpus_text(None, 64MiB)).batch("
                   "'val', 8, 512, Generator(12345)) -- r1_gate's own held-out batch",
        route="product", phase_route="gate",
    )
    print(json.dumps(fixed, indent=2), flush=True)
    board("rowmass_fixed_structure", **fixed)

    # ---- (2) trained beta / qk / g, per layer, this seed and all five
    switches = []
    for i, blk in enumerate(layers):
        at = blk.self_attn
        switches.append(dict(layer=i, beta=float(at.beta), qk=float(at.qk),
                             g=float(at.g),
                             beta_minus_1=float(at.beta) - 1.0))
    all_seeds = {}
    for s in range(5):
        p = os.path.join(SCRATCH, "q3_ckpt_seed{}_hard_concrete".format(s),
                         "run_record.json")
        if os.path.exists(p):
            r = json.load(io.open(p, encoding="utf-8"))
            all_seeds[s] = dict(beta_final=r["beta"][-1]["beta"],
                                beta_census=r["beta_census"])
    print("\n== (2) TRAINED SWITCHES, seed 0, read off the loaded tensors")
    for row in switches:
        print("  ", row)
    print("== beta, all five seeds (from each run_record's beta column)")
    for s, v in all_seeds.items():
        print("   seed", s, [round(b, 6) for b in v["beta_final"]],
              "census", [round(c, 3) for c in v["beta_census"]])

    # ---- the held-out batch, byte-identical to the producer's
    data = T.ByteBatches(T._corpus_text(None, 64 * 1024 * 1024), vocab_size=256)
    gen = torch.Generator().manual_seed(EVAL_SEED)
    x_eval, _ = data.batch("val", BATCH, SEQ, gen, DEVICE)

    # ---- capture hidden states at each attention, then recompute through the
    #      arm's OWN numerator() -- the consumption site.
    caps = {}

    def mk(i):
        def hook(mod, inputs):
            caps[i] = inputs[0].detach()
        return hook

    hs = [blk.self_attn.register_forward_pre_hook(mk(i)) for i, blk in enumerate(layers)]
    model(input_ids=x_eval)
    for h in hs:
        h.remove()

    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999,
             0.9999, 1.0000001]
    out_layers = []
    for i, blk in enumerate(layers):
        at = blk.self_attn
        x = caps[i]
        b, s, d = x.shape
        q, k, _ = at.qkv(x).chunk(3, dim=-1)

        def shape(t):
            return t.view(b, s, at.n_heads, at.d_head).transpose(1, 2)

        q, k = shape(q), shape(k)
        u = at.m_head(x).squeeze(-1).unsqueeze(-2)     # [B,1,S]
        th = at.theta_head(x).squeeze(-1).unsqueeze(-2)
        m_real, th_real = A.blend(u, th, at.g)          # the REALISED gate/phase

        masses, zeros_W, n_W = [], 0, 0
        for bi in range(b):
            num, mod = A.numerator(q[bi:bi + 1], k[bi:bi + 1],
                                   u[bi:bi + 1], th[bi:bi + 1],
                                   qk=at.qk, g=at.g, route="product",
                                   phase_route="gate")
            den = mod.sum(-1)                            # [1,H,S]  == Z_i
            numsum = num.sum(-1)                         # [1,H,S]  complex
            mi = numsum.abs().to(torch.float64) / den.to(torch.float64).clamp_min(1e-300)
            masses.append(mi.reshape(-1))
            # exact zeros of W among the CAUSAL entries (j<=i)
            zb = den.unsqueeze(-1) ** at.beta
            W = torch.complex(num.real / zb, num.imag / zb)
            tri = torch.ones(s, s, dtype=torch.bool).tril()
            Wc = W[..., tri]
            zeros_W += int((Wc == 0).sum())
            n_W += int(Wc.numel())
            del num, mod, W, zb
        mv = torch.cat(masses)

        mrow = m_real.reshape(b, s).to(torch.float64)    # gate sequence, [B,S]
        raw = [lag1(mrow[j]) for j in range(b)]
        dif = [lag1(mrow[j].diff()) for j in range(b)]
        throw = th_real.reshape(b, s).to(torch.float64)
        cum = throw.cumsum(-1)

        rec_i = dict(
            layer=i,
            beta=float(at.beta), qk=float(at.qk), g=float(at.g),
            row_mass_mean=float(mv.mean()), row_mass_min=float(mv.min()),
            row_mass_max=float(mv.max()), row_mass_median=float(mv.median()),
            row_mass_p01=float(mv.quantile(0.01)), row_mass_p05=float(mv.quantile(0.05)),
            row_mass_frac_below_0999=float((mv < 0.999).to(torch.float64).mean()),
            row_mass_frac_below_09=float((mv < 0.9).to(torch.float64).mean()),
            row_mass_frac_below_05=float((mv < 0.5).to(torch.float64).mean()),
            n_rows=int(mv.numel()),
            histogram_edges=edges, histogram=hist(mv, edges),
            gate_m_mean=float(mrow.mean()), gate_m_min=float(mrow.min()),
            gate_m_exact_zero_frac=float((mrow == 0).to(torch.float64).mean()),
            W_exact_zero_frac=zeros_W / n_W, W_causal_entries=n_W,
            theta_realised_std=float(throw.std()),
            theta_realised_absmean=float(throw.abs().mean()),
            cum_phase_span_mean=float((cum.max(-1).values - cum.min(-1).values).mean()),
            w5_lag1_raw_mean=sum(raw) / len(raw),
            w5_lag1_delta_mean=sum(dif) / len(dif),
            w5_lag1_raw=raw, w5_lag1_delta=dif,
        )
        out_layers.append(rec_i)
        print("\n== layer", i, json.dumps({k2: v2 for k2, v2 in rec_i.items()
                                           if not isinstance(v2, list)}, indent=2))
        print("   histogram", rec_i["histogram"])
        del caps[i]

    allm = dict(
        fixed=fixed, switches=switches, beta_all_seeds=all_seeds,
        layers=out_layers,
    )
    with io.open(os.path.join(SCRATCH, "rowmass_opus_check.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(allm, fh, indent=2)
    board("rowmass_measured",
          checkpoint=os.path.basename(CKPT),
          beta_per_layer=[r["beta"] for r in switches],
          row_mass_mean=[r["row_mass_mean"] for r in out_layers],
          row_mass_min=[r["row_mass_min"] for r in out_layers],
          frac_below_0999=[r["row_mass_frac_below_0999"] for r in out_layers],
          W_exact_zero_frac=[r["W_exact_zero_frac"] for r in out_layers],
          w5_lag1_raw=[r["w5_lag1_raw_mean"] for r in out_layers],
          w5_lag1_delta=[r["w5_lag1_delta_mean"] for r in out_layers])

    # one runnable check: the identity the row turns on
    assert all(r["row_mass_max"] <= 1.0 + 1e-6 for r in out_layers), \
        "row mass above 1 is impossible: |sum z| <= sum|z|"
    print("\nOK: |sum G e| <= sum |G| e held on every measured row.")


if __name__ == "__main__":
    main()
