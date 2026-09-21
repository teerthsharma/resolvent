"""CHASE. COGS control repair: reproduce the recorded control, then find out if
it is undertrained.

    python cogs_control.py

Imports ceq.harness / ceq.lm from the real repo (not a reimplementation) so
every number below walks the exact code path that produced
results/capability.json's cogs row. Writes nothing into ceq/, results/, or
docs/canon/ -- checkpoints and the curve go into this scratchpad only.

L-REPRO on every number below: seed=0, dtype=fp32 (model default), device=cuda,
COGS seq=192, bs=32, lr=3e-4, arm="softmax". Exact command for each stage is
printed with its result.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

REPO = pathlib.Path(r"C:\Users\seal\Desktop\New folder (32)")
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from ceq import harness, lm

HERE = pathlib.Path(__file__).resolve().parent
CURVE_PATH = HERE / "cogs_curve.jsonl"
CKPT_DIR = HERE / "cogs_ckpts"
CKPT_DIR.mkdir(exist_ok=True)

SEED = 0
BS, LR = 32, 3e-4
GATE_EVAL, GEN_EVAL, MAX_NEW = 256, 512, 192
CHECKPOINTS = [3000, 6000, 9000, 12000]

# The recorded row, results/capability.json -> cogs.
RECORDED_IN_DIST = 0.92578125     # in_distribution.softmax at steps=3000
RECORDED_GEN = 0.029296875        # arms.softmax.exact_match at steps=3000

# --- reproduction tolerance, STATED BEFORE RUNNING -------------------------
# Both splits are scored on a FIXED subsample (256 / 512 items, deterministic
# given n_test -- see harness.eval_indices), so the quantisation floor alone is
# 1/256 = 0.0039 and 1/512 = 0.00195. Training is seeded (model init, batch
# sampler) but runs on cuda, where a handful of kernels (scatter-add in
# backward, cuDNN algorithm choice) are not bit-deterministic across runs, and
# that noise compounds over 3000 gradient steps rather than cancelling.
#
# TOLERANCE: absolute exact-match difference <= 0.03 on EACH split.
# MISS: |measured - recorded| > 0.03 on either split. Concretely, a miss is
# in_distribution outside [0.896, 0.956] or generalization outside
# [0.000, 0.059]. This is checked in code below, not asserted after the fact --
# the run prints PASS or MISS and the tolerance band it was measured against.
TOL = 0.03


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def board(event: dict):
    """One JSON line, agent Chase, appended -- never rewritten."""
    event = {"agent": "Chase", "ts": now(), **event}
    path = pathlib.Path(r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def train_steps(model, opt, g, train, vocab, device, seq, n_steps: int) -> float:
    """`n_steps` more optimizer steps on the SAME model/opt/generator state --
    the continuation half of harness.train_model, split out so 3000+3000+...
    is one continuous 12000-step run and not four independent 3000-step ones.
    """
    model.train()
    loss = torch.zeros((), device=device)
    for _ in range(n_steps):
        idx = torch.randint(len(train), (BS,), generator=g).tolist()
        x, y = harness._batch(train, vocab, idx, device, seq)
        loss = F.cross_entropy(model(x).reshape(-1, len(vocab)), y.reshape(-1),
                               ignore_index=vocab[harness.PAD])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.train_loss = float(loss.detach())
    return model.train_loss


def eval_checkpoint(model, device, step: int) -> dict:
    """Gate (in-distribution, `test`) + generalization (`gen`), identical
    protocol to ceq.capability.run_split: gate_eval=256/max_new=192 on
    `test`, max_eval=512/max_new=192 on `gen`, seed=0 for both."""
    model.eval()
    t0 = time.time()
    gate = harness.score(model, "cogs", "test", device=device,
                         max_eval=GATE_EVAL, seed=SEED, max_new=MAX_NEW)
    gen = harness.score(model, "cogs", "gen", device=device,
                        max_eval=GEN_EVAL, seed=SEED, max_new=MAX_NEW)
    dt = time.time() - t0
    row = dict(step=step, seed=SEED, device=str(device), dtype="fp32",
              bs=BS, lr=LR, seq=model.seq, n_params=model.n_params(),
              gate_eval=GATE_EVAL, gen_eval=GEN_EVAL, max_new=MAX_NEW,
              in_distribution=gate["exact_match"],
              generalization=gen["exact_match"],
              train_loss=model.train_loss, eval_seconds=round(dt, 1),
              command=(f"python cogs_control.py  # step={step} seed={SEED} "
                       f"bs={BS} lr={LR} seq={model.seq} device={device}"))
    model.train()
    return row


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"tolerance stated before running: |measured - recorded| <= {TOL} "
          f"on each split -> miss if in_distribution outside "
          f"[{RECORDED_IN_DIST-TOL:.3f}, {RECORDED_IN_DIST+TOL:.3f}] or "
          f"generalization outside [{max(0,RECORDED_GEN-TOL):.3f}, "
          f"{RECORDED_GEN+TOL:.3f}]", flush=True)
    board(dict(event="start", row="cogs_control", device=str(device),
              seed=SEED, checkpoints=CHECKPOINTS, tol=TOL,
              recorded_in_dist=RECORDED_IN_DIST, recorded_gen=RECORDED_GEN))

    vocab = harness.build_vocab("cogs")
    seq = harness.split_seq("cogs")
    train = harness.load_pairs("cogs", "train")

    model = lm.TinyLM("softmax", seq=seq, vocab=len(vocab), seed=SEED).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED + 1)

    CURVE_PATH.write_text("", encoding="utf-8")  # fresh curve file, this run only
    prev_step = 0
    for step in CHECKPOINTS:
        t0 = time.time()
        train_steps(model, opt, g, train, vocab, device, seq, step - prev_step)
        train_s = time.time() - t0
        row = eval_checkpoint(model, device, step)
        row["train_seconds_this_stage"] = round(train_s, 1)
        prev_step = step

        with open(CURVE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        ckpt_path = CKPT_DIR / f"softmax_cogs_seed{SEED}_step{step}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[step={step}] in_distribution={row['in_distribution']:.5f} "
              f"generalization={row['generalization']:.5f} "
              f"train_loss={row['train_loss']:.4f} "
              f"train={train_s:.0f}s eval={row['eval_seconds']:.0f}s "
              f"ckpt={ckpt_path.name}", flush=True)
        board(dict(event="checkpoint", row="cogs_control", **row))

        if step == 3000:
            miss_in = abs(row["in_distribution"] - RECORDED_IN_DIST) > TOL
            miss_gen = abs(row["generalization"] - RECORDED_GEN) > TOL
            verdict = "MISS" if (miss_in or miss_gen) else "PASS"
            reproduce_row = dict(
                event="reproduction_check", row="cogs_control", verdict=verdict,
                tol=TOL, measured_in_distribution=row["in_distribution"],
                recorded_in_distribution=RECORDED_IN_DIST,
                measured_generalization=row["generalization"],
                recorded_generalization=RECORDED_GEN,
                delta_in_distribution=round(row["in_distribution"] - RECORDED_IN_DIST, 5),
                delta_generalization=round(row["generalization"] - RECORDED_GEN, 5))
            print(f"REPRODUCTION CHECK: {verdict}  "
                  f"in_dist {row['in_distribution']:.5f} vs recorded "
                  f"{RECORDED_IN_DIST:.5f} (delta {reproduce_row['delta_in_distribution']:+.5f})  "
                  f"gen {row['generalization']:.5f} vs recorded {RECORDED_GEN:.5f} "
                  f"(delta {reproduce_row['delta_generalization']:+.5f})", flush=True)
            board(reproduce_row)
            if verdict == "MISS":
                print("STOP: control does not reproduce its own recorded "
                      "numbers within the stated tolerance. Halting before "
                      "the 12,000-step curve -- a bed that will not return "
                      "its own numbers cannot score a longer run either.",
                      flush=True)
                board(dict(event="halt", row="cogs_control",
                          reason="reproduction_miss"))
                return 1

    print("done: curve written to", CURVE_PATH, flush=True)
    board(dict(event="done", row="cogs_control", curve_path=str(CURVE_PATH)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
