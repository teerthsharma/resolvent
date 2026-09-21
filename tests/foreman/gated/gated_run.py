"""Train ONE gated (operator=smprime) checkpoint through the real wiring
(ceq.hf.train.build + train), save it to the SCRATCHPAD, verify the saved
keys, and read the gate (m_bar, decay length, distribution, refusal channel,
init-vs-trained) on real token batches.

L-REPRO: seed=0, dtype=float32, shape h512/L4/H8/d_head64/seq512/batch8,
device=cuda if available else cpu. Exact command at the bottom of this file's
run (see gated_run.md).
"""
import io
import json
import math
import os
import sys
import time

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
sys.path.insert(0, REPO)

import torch
from ceq.hf import train as T
from ceq.hf.modeling_ceq import CEQForCausalLM
from ceq.arm_smprime import blend

OUT_DIR = os.path.join(SCRATCH, "gated_ckpt")
SUMMARY_PATH = os.path.join(SCRATCH, "gated_run_summary.json")

HIDDEN, LAYERS, HEADS, SEQ, BATCH = 512, 4, 8, 512, 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Foreman",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", row)


GATE_KEYS = ("m_head.weight", "m_head.bias", "theta_head.weight",
             "theta_head.bias", "beta", "qk", "g")


def gate_key_report(state_dict, n_layers):
    """Per-layer presence of the 7 gate-bearing names. Returns (ok, rows)."""
    rows = []
    ok = True
    for i in range(n_layers):
        pfx = "model.layers.{}.self_attn.".format(i)
        present = {k: (pfx + k) in state_dict for k in GATE_KEYS}
        rows.append(present)
        ok = ok and all(present.values())
    return ok, rows


@torch.no_grad()
def gate_values(model, x_ids):
    """Per-layer [B,S] tensors of the ACTUAL m realised by blend(m_head(x),
    theta_head(x), g) -- the same computation arm_smprime.hop() performs, not
    the raw m_head output."""
    model.eval()
    captured = []

    def make_hook():
        def hook(mod, inputs):
            x = inputs[0]
            u_raw = mod.m_head(x).squeeze(-1)
            th_raw = mod.theta_head(x).squeeze(-1)
            m, th = blend(u_raw, th_raw, mod.g)
            captured.append(m.detach().float().cpu())
        return hook

    handles = [layer.self_attn.register_forward_pre_hook(make_hook())
               for layer in model.model.layers]
    model(input_ids=x_ids)
    for h in handles:
        h.remove()
    return captured  # list length n_layers, each [B, S]


def gate_stats(m_layers):
    """m_bar (geometric mean), quantiles, exact-zero fraction, per layer and
    pooled across layers."""
    all_m = torch.cat([m.reshape(-1) for m in m_layers])
    n = all_m.numel()
    n_zero = int((all_m == 0.0).sum())
    nz = all_m[all_m > 0]
    log_m_bar = float(torch.log(nz).mean()) if nz.numel() else float("-inf")
    m_bar = math.exp(log_m_bar) if nz.numel() else 0.0
    # geometric mean over ALL entries (zeros included) is 0 the instant any
    # entry is exactly 0 -- that is the refusal channel doing its job, not a
    # bug in the estimator.
    m_bar_incl_zero = 0.0 if n_zero > 0 else m_bar
    qs = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    quant = torch.quantile(all_m, qs).tolist()
    return dict(
        n=n, n_zero_exact=n_zero, frac_zero_exact=n_zero / n,
        m_bar_geomean_over_nonzero=m_bar,
        m_bar_geomean_all=m_bar_incl_zero,
        min=float(all_m.min()), max=float(all_m.max()),
        mean_arith=float(all_m.mean()),
        quantiles=dict(zip(["p0", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p100"], quant)),
    )


def decay_length(m_bar):
    if m_bar <= 0.0:
        return 0.0
    if m_bar >= 1.0:
        return float("inf")
    return 1.0 / math.log(1.0 / m_bar)


def main():
    t0 = time.time()
    board("start", shape=dict(hidden=HIDDEN, layers=LAYERS, heads=HEADS, seq=SEQ,
                               batch=BATCH), device=DEVICE, seed=SEED)

    # ---- fixed eval batch (val split, held out from training) ----
    data = T.ByteBatches(T._corpus_text(None, 64 * 1024 * 1024), vocab_size=256)
    eval_gen = torch.Generator().manual_seed(12345)
    x_eval, _ = data.batch("val", BATCH, SEQ, eval_gen, DEVICE)

    # ---- capture m at INITIALISATION, bitwise the same init train() will build ----
    torch.manual_seed(SEED)
    model0 = T.build(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS, seq=SEQ,
                      vocab_size=256, operator="smprime").to(DEVICE)
    m_init_layers = gate_values(model0, x_eval)
    stats_init = gate_stats(m_init_layers)
    del model0
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    board("gate_at_init", stats=stats_init)

    # ---- size the run to the wall clock ----
    # Quoted throughput: batch 8 @ h512/L4/H8/seq512 fp32 -> 17,893.3 tok/s on an
    # unshared RTX 4060 Laptop. Another lane is on this GPU, so budget at HALF
    # that (8,946 tok/s) for sizing, and let the loop itself report the real
    # measured rate once step 1 lands.
    TOK_PER_STEP = BATCH * SEQ
    ASSUMED_TOK_S = 17893.3 / 2.0
    STEPS = 800
    SAVE_EVERY = 100
    est_s = STEPS * TOK_PER_STEP / ASSUMED_TOK_S
    board("plan", steps=STEPS, save_every=SAVE_EVERY,
          est_seconds_at_half_quoted_throughput=est_s)

    # ---- train ----
    t_train0 = time.time()
    record = T.train(out_dir=OUT_DIR, steps=STEPS, batch=BATCH, seq=SEQ,
                      hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                      device=DEVICE, vocab_size=256, data_path=None,
                      max_bytes=64 * 1024 * 1024, lr=3e-4, seed=SEED,
                      grad_checkpoint=False, log_every=20, save_every=SAVE_EVERY,
                      operator="smprime")
    train_s = time.time() - t_train0
    measured_tok_s = STEPS * TOK_PER_STEP / train_s
    board("train_done", steps=STEPS, seconds=train_s, measured_tok_s=measured_tok_s,
          loss_first=record["losses"][0], loss_last=record["losses"][-1],
          out_dir=OUT_DIR)

    # ---- verify the artefact: load it back, print/confirm key names ----
    sd_path_candidates = [os.path.join(OUT_DIR, "model.safetensors")]
    from safetensors.torch import load_file
    sd = None
    for p in sd_path_candidates:
        if os.path.exists(p):
            sd = load_file(p)
            break
    if sd is None:
        raise FileNotFoundError("no model.safetensors under {}".format(OUT_DIR))
    key_ok, key_rows = gate_key_report(sd, LAYERS)
    board("verify_artifact", all_gate_keys_present=key_ok, per_layer=key_rows,
          n_keys_total=len(sd))

    # ---- read the gate post-training, same eval batch ----
    model1 = CEQForCausalLM.from_pretrained(OUT_DIR).to(DEVICE)
    m_trained_layers = gate_values(model1, x_eval)
    stats_trained = gate_stats(m_trained_layers)
    board("gate_after_training", stats=stats_trained)

    L_init = decay_length(stats_init["m_bar_geomean_over_nonzero"])
    L_trained = decay_length(stats_trained["m_bar_geomean_over_nonzero"])
    fires_256 = L_trained < SEQ / 2
    fires_64 = L_trained < SEQ / 8

    summary = dict(
        device=DEVICE, seed=SEED, dtype="float32", shape=dict(
            hidden=HIDDEN, n_layers=LAYERS, n_heads=HEADS, d_head=HIDDEN // HEADS,
            seq=SEQ, batch=BATCH, vocab_size=256),
        steps_trained=STEPS, save_every=SAVE_EVERY,
        out_dir=OUT_DIR,
        train_seconds=train_s, measured_tok_s=measured_tok_s,
        loss_curve=record["losses"], loss_first=record["losses"][0],
        loss_last=record["losses"][-1],
        n_params=record["n_params"],
        gate_keys_present_every_layer=key_ok, gate_keys_per_layer=key_rows,
        gate_at_init=stats_init, gate_after_training=stats_trained,
        decay_length_init=L_init, decay_length_trained=L_trained,
        seq_over_2=SEQ / 2, seq_over_8=SEQ / 8,
        fires_L_lt_S_over_2=fires_256, fires_L_lt_S_over_8=fires_64,
        wall_clock_total_s=time.time() - t0,
    )
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    board("done", summary_path=SUMMARY_PATH, wall_clock_total_s=summary["wall_clock_total_s"],
          L_init=L_init, L_trained=L_trained,
          fires_S_over_2=fires_256, fires_S_over_8=fires_64)
    print("SUMMARY_PATH", SUMMARY_PATH)


if __name__ == "__main__":
    main()
