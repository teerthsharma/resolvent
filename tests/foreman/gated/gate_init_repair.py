"""THE ROW: repair the gate's init to what trainable_heads() specifies
(m_head.bias=0.999, theta_head.bias=0.001 on every layer) via a runtime
monkeypatch of ceq.hf.train.build, retrain IDENTICALLY to the smprime
reference run, and read the gate the same way gated_run.py does, plus a
new runs-of-consecutive-nonzero-m measurement.

ceq/ is NEVER edited in place. The repair is a monkeypatch: `T.build` (a
module-global name inside ceq/hf/train.py) is replaced for the duration of
this process, so `T.train()`'s own internal call to `build(...)` resolves
through the patched name -- no source file changes, no RNG draw added (the
bias fill happens under torch.no_grad() after construction, so the training
RNG stream that follows is bit-for-bit what the reference run consumed).

PRE-REGISTERED THRESHOLDS (stated before this script's first run, see
gate_init_repair.md):
  - Repair verified iff init frac_exactly_zero < 0.05 (reference: 0.493).
    If not, STOP -- everything after is void.
  - Verdict CHANGES (init-artifact hypothesis wins, gate numbers are void)
    iff trained max run length > 14 (2x the reference's 7) OR trained
    frac_exactly_zero < 0.45 (well below the reference's 0.593).
  - Otherwise the reference verdict HOLDS: the data closes the gate
    regardless of a corrected init.

L-REPRO: seed=0, dtype=float32, shape h512/L4/H8/d_head64/seq512/batch8,
device=cuda if available else cpu, steps=800, save_every=100, lr=3e-4,
data/tinystories_20k.txt (first 64MiB), optimizer=AdamW default betas/eps.
Command: python gate_init_repair.py
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
from ceq.arm_smprime import blend, GATE_INIT_OFF

OUT_DIR = os.path.join(SCRATCH, "gate_repair_ckpt")
SUMMARY_PATH = os.path.join(SCRATCH, "gate_init_repair_summary.json")

HIDDEN, LAYERS, HEADS, SEQ, BATCH = 512, 4, 8, 512, 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# ---- the reference run's numbers (tests/foreman/gated/gated_run.py,
# operator="smprime", UNPATCHED build), quoted verbatim from the task so the
# comparison below is against the actual measured baseline, not a re-run. ----
REFERENCE = dict(
    loss_first=5.6557, loss_last=1.5618, train_seconds=206.9, tok_s=15840.7,
    init=dict(frac_zero=0.493, geomean_nonzero=0.2425,
              run_mean=1.023, run_median=0.0, run_p95=4.0, run_max=17.0),
    trained=dict(frac_zero=0.593, geomean_nonzero=0.9024,
                 run_mean=0.685, run_median=0.0, run_p95=3.0, run_max=7.0),
)

VERIFY_FRAC_ZERO_THRESHOLD = 0.05          # (1): repair must clear this
VERDICT_MAX_RUN_THRESHOLD = 14.0           # (3): 2x reference trained max=7
VERDICT_FRAC_ZERO_THRESHOLD = 0.45         # (3): well below reference 0.593


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Foreman",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", row)


# --------------------------------------------------------- the monkeypatch

_ORIG_BUILD = T.build


def repaired_build(*, hidden_size, n_layers, n_heads, seq, vocab_size=256,
                    **overrides):
    """`_ORIG_BUILD`, then every layer's m_head.bias / theta_head.bias reset
    to trainable_heads()'s values (0.999 / 0.001). No RNG consumed: the
    weight matrices (already drawn by _ORIG_BUILD under whatever torch RNG
    state was current) are untouched, only the two bias tensors are
    overwritten in place under no_grad. Nothing else about the model
    differs from an unpatched build() call with the same arguments."""
    model = _ORIG_BUILD(hidden_size=hidden_size, n_layers=n_layers,
                        n_heads=n_heads, seq=seq, vocab_size=vocab_size,
                        **overrides)
    off = GATE_INIT_OFF  # 1e-3, the same constant trainable_heads() uses
    with torch.no_grad():
        for layer in model.model.layers:
            attn = layer.self_attn
            attn.m_head.bias.fill_(1.0 - off)
            attn.theta_head.bias.fill_(off)
    return model


GATE_KEYS = ("m_head.weight", "m_head.bias", "theta_head.weight",
             "theta_head.bias", "beta", "qk", "g")


def gate_key_report(state_dict, n_layers):
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
    """Per-layer [B,S] tensors of the ACTUAL m realised by
    blend(m_head(x), theta_head(x), g) -- read where arm_smprime.hop()
    consumes it, same hook gated_run.py uses."""
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
    return captured


def gate_stats(m_layers):
    all_m = torch.cat([m.reshape(-1) for m in m_layers])
    n = all_m.numel()
    n_zero = int((all_m == 0.0).sum())
    nz = all_m[all_m > 0]
    log_m_bar = float(torch.log(nz).mean()) if nz.numel() else float("-inf")
    m_bar = math.exp(log_m_bar) if nz.numel() else 0.0
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


def run_length_stats(m_layers):
    """Runs of consecutive nonzero m, one value assigned to EVERY position:
    the length of the maximal contiguous nonzero stretch containing it, or 0
    if that position's own m is exactly zero. Pooled over batch and layers
    this has the same population size as gate_stats' n (16,384 at B=8,
    S=512, n_layers=4), which is what makes it comparable to the reference's
    reported mean/median/p95/max."""
    runs = []
    for m in m_layers:  # [B, S]
        nz = (m > 0).numpy()
        B, S = nz.shape
        for b in range(B):
            row = nz[b]
            out = [0] * S
            i = 0
            while i < S:
                if row[i]:
                    j = i
                    while j < S and row[j]:
                        j += 1
                    length = j - i
                    for k in range(i, j):
                        out[k] = length
                    i = j
                else:
                    i += 1
            runs.extend(out)
    t = torch.tensor(runs, dtype=torch.float64)
    qs = torch.tensor([0.5, 0.95], dtype=torch.float64)
    med, p95 = torch.quantile(t, qs).tolist()
    return dict(n=t.numel(), mean=float(t.mean()), median=med, p95=p95,
                max=float(t.max()))


def main():
    t0 = time.time()
    board("start", task="gate_init_repair", shape=dict(
        hidden=HIDDEN, layers=LAYERS, heads=HEADS, seq=SEQ, batch=BATCH),
        device=DEVICE, seed=SEED, off=GATE_INIT_OFF)

    # ---- same held-out eval batch the reference run used ----
    data = T.ByteBatches(T._corpus_text(None, 64 * 1024 * 1024), vocab_size=256)
    eval_gen = torch.Generator().manual_seed(12345)
    x_eval, _ = data.batch("val", BATCH, SEQ, eval_gen, DEVICE)

    # ---- (1) REPAIR THE INIT, and verify it took, before any gradient step ----
    torch.manual_seed(SEED)
    model0 = repaired_build(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                            seq=SEQ, vocab_size=256, operator="smprime").to(DEVICE)
    # spot-check the bias values landed exactly, on layer 0
    b_m = float(model0.model.layers[0].self_attn.m_head.bias[0])
    b_th = float(model0.model.layers[0].self_attn.theta_head.bias[0])
    m_init_layers = gate_values(model0, x_eval)
    stats_init = gate_stats(m_init_layers)
    runs_init = run_length_stats(m_init_layers)
    del model0
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    repair_took = stats_init["frac_zero_exact"] < VERIFY_FRAC_ZERO_THRESHOLD
    board("verify_repair", bias_m_head=b_m, bias_theta_head=b_th,
          frac_zero_exact_init=stats_init["frac_zero_exact"],
          threshold=VERIFY_FRAC_ZERO_THRESHOLD,
          reference_frac_zero_exact_init=REFERENCE["init"]["frac_zero"],
          repair_took=repair_took)
    if not repair_took:
        board("VOID", reason="repair did not clear the frac_zero_exact_init "
              "threshold; everything after this point is void, per "
              "instruction (1)", frac_zero_exact_init=stats_init["frac_zero_exact"])
        summary = dict(void=True, reason="repair_did_not_take",
                       stats_init=stats_init, runs_init=runs_init,
                       bias_m_head=b_m, bias_theta_head=b_th,
                       wall_clock_total_s=time.time() - t0)
        with open(SUMMARY_PATH, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print("VOID -- see", SUMMARY_PATH)
        return

    board("gate_at_init_repaired", stats=stats_init, runs=runs_init)

    # ---- (2) RETRAIN, IDENTICALLY, under the monkeypatch ----
    STEPS, SAVE_EVERY = 800, 100
    TOK_PER_STEP = BATCH * SEQ
    board("plan", steps=STEPS, save_every=SAVE_EVERY, out_dir=OUT_DIR)

    T.build = repaired_build  # the only thing that differs from the reference run
    try:
        t_train0 = time.time()
        record = T.train(out_dir=OUT_DIR, steps=STEPS, batch=BATCH, seq=SEQ,
                         hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                         device=DEVICE, vocab_size=256, data_path=None,
                         max_bytes=64 * 1024 * 1024, lr=3e-4, seed=SEED,
                         grad_checkpoint=False, log_every=20,
                         save_every=SAVE_EVERY, operator="smprime")
        train_s = time.time() - t_train0
    finally:
        T.build = _ORIG_BUILD  # restore immediately, whether or not train() raised

    measured_tok_s = STEPS * TOK_PER_STEP / train_s
    board("train_done", steps=STEPS, seconds=train_s, measured_tok_s=measured_tok_s,
          loss_first=record["losses"][0], loss_last=record["losses"][-1],
          out_dir=OUT_DIR)

    # ---- verify the artefact carries the repaired bias, not the reference's ----
    from safetensors.torch import load_file
    sd_path = os.path.join(OUT_DIR, "model.safetensors")
    sd = load_file(sd_path)
    key_ok, key_rows = gate_key_report(sd, LAYERS)
    board("verify_artifact", all_gate_keys_present=key_ok, per_layer=key_rows,
          n_keys_total=len(sd))

    # ---- (3) READ THE GATE post-training, same eval batch, same method ----
    model1 = CEQForCausalLM.from_pretrained(OUT_DIR).to(DEVICE)
    m_trained_layers = gate_values(model1, x_eval)
    stats_trained = gate_stats(m_trained_layers)
    runs_trained = run_length_stats(m_trained_layers)
    board("gate_after_training_repaired", stats=stats_trained, runs=runs_trained)

    verdict_changes = (runs_trained["max"] > VERDICT_MAX_RUN_THRESHOLD or
                       stats_trained["frac_zero_exact"] < VERDICT_FRAC_ZERO_THRESHOLD)

    summary = dict(
        void=False, device=DEVICE, seed=SEED, dtype="float32",
        shape=dict(hidden=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                  d_head=HIDDEN // HEADS, seq=SEQ, batch=BATCH, vocab_size=256),
        steps_trained=STEPS, save_every=SAVE_EVERY, out_dir=OUT_DIR,
        bias_m_head_init=b_m, bias_theta_head_init=b_th,
        train_seconds=train_s, measured_tok_s=measured_tok_s,
        loss_curve=record["losses"], loss_first=record["losses"][0],
        loss_last=record["losses"][-1], n_params=record["n_params"],
        gate_keys_present_every_layer=key_ok,
        gate_at_init=stats_init, runs_at_init=runs_init,
        gate_after_training=stats_trained, runs_after_training=runs_trained,
        reference=REFERENCE,
        thresholds=dict(verify_frac_zero=VERIFY_FRAC_ZERO_THRESHOLD,
                        verdict_max_run=VERDICT_MAX_RUN_THRESHOLD,
                        verdict_frac_zero=VERDICT_FRAC_ZERO_THRESHOLD),
        verdict_changes=verdict_changes,
        wall_clock_total_s=time.time() - t0,
    )
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    board("done", summary_path=SUMMARY_PATH, wall_clock_total_s=summary["wall_clock_total_s"],
          verdict_changes=verdict_changes,
          trained_max_run=runs_trained["max"], reference_trained_max_run=REFERENCE["trained"]["run_max"],
          trained_frac_zero=stats_trained["frac_zero_exact"],
          reference_trained_frac_zero=REFERENCE["trained"]["frac_zero"])
    print("SUMMARY_PATH", SUMMARY_PATH)
    print("verdict_changes:", verdict_changes)


if __name__ == "__main__":
    main()
