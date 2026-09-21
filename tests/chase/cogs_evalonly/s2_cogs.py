"""CHASE S2 -- COGS row, R1's configuration: hard-concrete gate DURING
TRAINING, frozen to bare clamp(u,0,1) AT EVAL. R1 (tests/chase/gate/r1_gate.py,
opus-checked) killed both live-gradient parameterisations on the pre-registered
bias-move must-fire and set "exactness is eval-only: train with hard-concrete,
freeze to clamp at eval" -- that is the configuration this row builds and
scores on COGS.

REUSED, NOT REIMPLEMENTED (see the task brief):
  tests/chase/cogs/lm_patched.py   -- TinyLM w/ kind="smprime" admitted past
                                       ceq/lm.py:66 (scratch copy, ceq/ untouched)
  tests/chase/gate/r1_gate.py      -- magnitude_hardconcrete, spurious_zero_count,
                                       backward_reach_stats, exact_zero_frac
  tests/chase/cogs/cogs_control.py -- train_steps/eval_checkpoint shape, the
                                       already-measured control curve (cited,
                                       not rerun -- see CONTROL_CURVE below)
The magnitude swap is the SAME monkeypatch technique r1_gate.py uses on
ceq.arm_smprime.magnitude: a module-global the arm's own blend() looks up at
call time, so replacing the name changes every caller with no source edit.

Command: python s2_cogs.py [seeds, comma-separated, default "0"]
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import time

REPO = pathlib.Path(r"C:\Users\seal\Desktop\New folder (32)")
SCRATCH = pathlib.Path(r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad")
COGS_DIR = REPO / "tests" / "chase" / "cogs"
GATE_DIR = REPO / "tests" / "chase" / "gate"
BOARD = REPO / "house-events.jsonl"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COGS_DIR))
sys.path.insert(0, str(GATE_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import lm_patched  # noqa: E402  -- scratch copy, kind="smprime" admitted
from ceq import harness  # noqa: E402
harness.lm = lm_patched  # THIS PROCESS ONLY; ceq/lm.py on disk untouched

import ceq.arm_smprime as arm_smprime  # noqa: E402
import r1_gate  # noqa: E402  -- reused instruments, not reimplemented

_CLAMP = arm_smprime.magnitude          # the original bare clamp, bound before
                                         # anything monkeypatches the name
_HARDCONCRETE = r1_gate.magnitude_hardconcrete

BS, LR = 32, 3e-4
GATE_EVAL, GEN_EVAL, MAX_NEW = 256, 512, 192
CHECKPOINTS = [3000, 6000, 9000, 12000]
RESOLUTION_FLOOR = 0.20  # ceq/capability.py:39-40 admission gate

# softmax control curve, tests/chase/cogs/cogs_curve.jsonl -- CITED, NOT RERUN.
CONTROL_CURVE = {3000: 0.021484375, 6000: 0.13671875,
                 9000: 0.146484375, 12000: 0.09765625}
CONTROL_IN_DIST = {3000: 0.93359375, 6000: 0.9453125,
                   9000: 0.9453125, 12000: 0.94921875}
CONTROL_N_PARAMS = 3652096
CONTROL_TRAIN_LOSS = {3000: 0.27306613326072693, 6000: 0.288677841424942,
                      9000: 0.246354877948761, 12000: 0.23310652375221252}
# broken arm (bare clamp, live gradient, same shape), for the underfit question
BROKEN_TRAIN_LOSS_RANGE = (0.407, 0.485)
BROKEN_IN_DIST_3000 = 0.16797
BROKEN_FRAC_ANNIHILATED_INIT = 0.9799
BROKEN_LIVE_KEYS_INIT = 1.93


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def board(event: str, **fields):
    row = dict(agent="Chase", ts=now(), event=event, **fields)
    with io.open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print("[board]", row, flush=True)


def save(name: str, obj):
    (SCRATCH / name).write_text(json.dumps(obj, indent=1, default=str),
                                encoding="utf-8")


# ------------------------------------------------ training / eval (hard-concrete / clamp)

def train_steps(model, opt, g, train, vocab, device, seq, n_steps: int) -> float:
    """`n_steps` optimizer steps, TRAINING FORM = hard-concrete, continuing
    the same model/opt/generator state (cogs_control.py's own pattern)."""
    arm_smprime.magnitude = _HARDCONCRETE
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


def freeze_and_verify(model, device) -> dict:
    """Switch arm_smprime.magnitude to the bare clamp (EVAL form) and verify
    the freeze TOOK: capture the operator's actual realized magnitude off a
    real eval-mode forward pass (forward-pre-hook on block 0's attn, same
    technique r1_gate.py::gate_values uses) and check it is bitwise
    torch.clamp(u,0,1) and NOT hard-concrete(u) on the same u. A freeze that
    did not take makes every number after it a hard-concrete number wearing
    a clamp label."""
    arm_smprime.magnitude = _CLAMP
    identity_ok = arm_smprime.magnitude is _CLAMP

    captured = {}

    def hook(mod, inputs):
        x = inputs[0]
        u_raw = mod.m_head(x).squeeze(-1)
        th_raw = mod.theta_head(x).squeeze(-1)
        m, _ = arm_smprime.blend(u_raw, th_raw, mod.g)
        # blend() applies magnitude to lerp(1, u, g), NOT to u -- and g is a
        # trained AdamW parameter, so after step 1 it is no longer 1.0 and
        # clamp(u) != clamp(lerp(1,u,g)). Compare on the argument the gate
        # actually sees, or the check fails on g's drift, not on the freeze.
        captured["u"] = torch.lerp(torch.ones_like(u_raw), u_raw,
                                   mod.g.detach()).detach().clone()
        captured["u_raw"] = u_raw.detach().clone()
        captured["g"] = float(mod.g.detach())
        captured["m"] = m.detach().clone()

    h = model.blocks[0].attn.register_forward_pre_hook(hook)
    model.eval()
    g = torch.Generator().manual_seed(999)
    x = torch.randint(0, model.vocab, (4, model.seq), generator=g).to(device)
    with torch.no_grad():
        model(x)
    h.remove()

    u, m = captured["u"], captured["m"]
    m_clamp = torch.clamp(u, 0.0, 1.0)
    m_hc = _HARDCONCRETE(u)
    bitwise_eq_clamp = torch.equal(m, m_clamp)
    differs_from_hc = not torch.equal(m, m_hc)
    out = dict(magnitude_is_orig_clamp=identity_ok,
              bitwise_equal_to_clamp_forward=bitwise_eq_clamp,
              differs_from_hardconcrete_forward=differs_from_hc,
              n=int(u.numel()), g_at_freeze=captured["g"],
              max_abs_diff_vs_clamp=float((m - m_clamp).abs().max()),
              max_abs_diff_vs_hardconcrete=float((m - m_hc).abs().max()))
    out["freeze_took"] = identity_ok and bitwise_eq_clamp and differs_from_hc
    return out


def eval_checkpoint(model, device, step: int) -> dict:
    """Gate (in-distribution) + generalization, EVAL FORM = frozen clamp,
    identical protocol to cogs_control.py::eval_checkpoint."""
    arm_smprime.magnitude = _CLAMP
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
              control_in_distribution=CONTROL_IN_DIST.get(step),
              control_generalization=CONTROL_CURVE.get(step),
              control_train_loss=CONTROL_TRAIN_LOSS.get(step),
              command=(f"python s2_cogs.py  # step={step} seed={SEED} "
                       f"bs={BS} lr={LR} seq={model.seq} device={device} "
                       f"train_form=hard_concrete eval_form=clamp"))
    model.train()
    arm_smprime.magnitude = _HARDCONCRETE  # explicit: back to training form
    return row


# ------------------------------------------------------- COGS-shape gate diagnostics

def gate_shape_measure(model, device, form_label: str) -> dict:
    """Fraction of causal operator entries annihilated, mean live keys per
    query out of `seq`, median BACKWARD reach -- same instruments
    (r1_gate.spurious_zero_count / backward_reach_stats / exact_zero_frac),
    same read style as the broken-arm finding (house-events.jsonl,
    'opus-check:why-it-loses-and-the-replacement-route'): a real batch of
    COGS train inputs driven through the model, m captured at block 0 via a
    forward-pre-hook. `arm_smprime.magnitude` is READ, not set, here --
    caller decides which form is live when this is called."""
    vocab = harness.build_vocab("cogs")
    seq = harness.split_seq("cogs")
    train = harness.load_pairs("cogs", "train")
    idx = list(range(min(BS, len(train))))
    x, _ = harness._batch(train, vocab, idx, device, seq)

    m_layers, u_layers = [], []

    def make_hook():
        def hook(mod, inputs):
            xin = inputs[0]
            u_raw = mod.m_head(xin).squeeze(-1)
            th_raw = mod.theta_head(xin).squeeze(-1)
            m, _ = arm_smprime.blend(u_raw, th_raw, mod.g)
            m_layers.append(m.detach().float().cpu())
            u_layers.append(u_raw.detach().float().cpu())
        return hook

    handles = [blk.attn.register_forward_pre_hook(make_hook())
               for blk in model.blocks]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    if was_training:
        model.train()

    spur0 = r1_gate.spurious_zero_count(m_layers[0])  # layer 0, whole batch
    reach = r1_gate.backward_reach_stats(m_layers)
    ez = r1_gate.exact_zero_frac(m_layers)
    B = m_layers[0].shape[0]
    n_pairs, n_float_zero = spur0["n_pairs"], spur0["n_float_zero"]
    return dict(form=form_label, magnitude_fn=arm_smprime.magnitude.__name__
                if hasattr(arm_smprime.magnitude, "__name__") else str(arm_smprime.magnitude),
                seq=seq, batch=B,
                frac_annihilated_layer0=n_float_zero / n_pairs,
                mean_live_keys_per_query_layer0=(n_pairs - n_float_zero) / (B * seq),
                exact_zero_frac_all_layers=ez,
                backward_reach_all_layers=reach,
                spurious_layer0=spur0)


# --------------------------------------------------------------------- main

SEED = 0


def run_one_seed(seed: int, device) -> dict:
    global SEED
    SEED = seed
    result = dict(seed=seed)

    vocab = harness.build_vocab("cogs")
    seq = harness.split_seq("cogs")
    train = harness.load_pairs("cogs", "train")

    arm_smprime.magnitude = _HARDCONCRETE
    model = lm_patched.TinyLM("smprime", seq=seq, vocab=len(vocab), seed=seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 1)

    board("start_seed", row="s2_cogs", seed=seed, checkpoints=CHECKPOINTS,
          resolution_floor=RESOLUTION_FLOOR, train_form="hard_concrete",
          eval_form="clamp", note="R1's eval-only configuration, COGS row")

    # gate shape AT INIT, under the form actually used at that time (training
    # form, hard-concrete, is what's live before the first optimizer step)
    gate_init = gate_shape_measure(model, device, "hard_concrete_at_init")
    board("gate_shape", row="s2_cogs", seed=seed, stage="init", **{
        k: v for k, v in gate_init.items()
        if k not in ("backward_reach_all_layers", "spurious_layer0")},
        backward_reach=gate_init["backward_reach_all_layers"])
    result["gate_init"] = gate_init

    # ================= (1)+(2) FIRST, WRITTEN TO DISK BEFORE ANYTHING ELSE =================
    prev_step = 0
    t0 = time.time()
    train_steps(model, opt, g, train, vocab, device, seq, CHECKPOINTS[0] - prev_step)
    prev_step = CHECKPOINTS[0]
    train_s = time.time() - t0

    freeze = freeze_and_verify(model, device)
    board("freeze_verify", row="s2_cogs", seed=seed, step=CHECKPOINTS[0], **freeze)
    result["freeze_verify"] = freeze
    if not freeze["freeze_took"]:
        result["halted"] = "freeze_did_not_take"
        board("halt", row="s2_cogs", seed=seed, reason="freeze_did_not_take",
              detail=freeze)
        save(f"s2_cogs_seed{seed}_FIRST.json", result)
        return result

    arm_smprime.magnitude = _CLAMP
    model.eval()
    gate0 = harness.score(model, "cogs", "test", device=device,
                          max_eval=GATE_EVAL, seed=seed, max_new=MAX_NEW)
    admission_pass = gate0["exact_match"] >= RESOLUTION_FLOOR
    admission = dict(step=CHECKPOINTS[0], in_distribution=gate0["exact_match"],
                     n_solved=gate0["n_solved"], n_eval=gate0["n_eval"],
                     resolution_floor=RESOLUTION_FLOOR, clears_floor=admission_pass,
                     broken_arm_in_distribution_3000=BROKEN_IN_DIST_3000,
                     train_seconds=round(train_s, 1))
    result["admission_gate"] = admission
    board("admission_gate", row="s2_cogs", seed=seed, **admission)
    save(f"s2_cogs_seed{seed}_FIRST.json", result)  # (1)+(2) on disk NOW

    if not admission_pass:
        result["halted"] = "admission_gate_fails"
        # The row stops here, but three numbers that cost ~20s cost a whole
        # 4.4-minute retrain if they are not taken NOW, and each separates a
        # different cause of a 0.0:
        #   (5) train loss at the halt -- underfit, or trains and still fails
        #   (4) the TRAINED gate shape under both forms -- did the frozen
        #       clamp annihilate an operator the training form kept open
        #   (B') in-distribution under the TRAINING form -- separates "the arm
        #       cannot do COGS" from "the freeze is what destroys it"
        result["train_loss_at_halt"] = model.train_loss
        result["control_train_loss_3000"] = CONTROL_TRAIN_LOSS[CHECKPOINTS[0]]
        result["broken_arm_train_loss_range"] = list(BROKEN_TRAIN_LOSS_RANGE)
        arm_smprime.magnitude = _CLAMP
        gs_clamp = gate_shape_measure(model, device, "clamp_frozen_at_3000")
        arm_smprime.magnitude = _HARDCONCRETE
        gs_hc = gate_shape_measure(model, device, "hard_concrete_at_3000")
        model.eval()
        hc_gate = harness.score(model, "cogs", "test", device=device,
                                max_eval=GATE_EVAL, seed=seed, max_new=MAX_NEW)
        result["trained_gate_shape"] = dict(clamp=gs_clamp, hard_concrete=gs_hc)
        result["in_distribution_under_training_form"] = hc_gate["exact_match"]
        for lbl, gs in (("clamp_frozen_at_3000", gs_clamp),
                        ("hard_concrete_at_3000", gs_hc)):
            board("gate_shape", row="s2_cogs", seed=seed, stage="trained_3000",
                  **{k: v for k, v in gs.items()
                     if k not in ("backward_reach_all_layers", "spurious_layer0")},
                  backward_reach=gs["backward_reach_all_layers"])
        board("halt", row="s2_cogs", seed=seed, reason="admission_gate_fails",
              in_distribution=gate0["exact_match"], floor=RESOLUTION_FLOOR,
              train_loss_at_halt=model.train_loss,
              control_train_loss_3000=CONTROL_TRAIN_LOSS[CHECKPOINTS[0]],
              broken_arm_train_loss_range=list(BROKEN_TRAIN_LOSS_RANGE),
              in_distribution_under_training_form=hc_gate["exact_match"],
              note="generalization number below the admission gate means "
                   "nothing; stopping here per the row's own instruction")
        arm_smprime.magnitude = _HARDCONCRETE
        save(f"s2_cogs_seed{seed}.json", result)
        return result

    model.train()
    arm_smprime.magnitude = _HARDCONCRETE
    # ================================ end (1)+(2) ================================

    # (3) the curve, plus (5) train-loss-vs-control at every checkpoint
    curve = []
    row0 = eval_checkpoint(model, device, CHECKPOINTS[0])
    row0["train_seconds_this_stage"] = round(train_s, 1)
    curve.append(row0)
    with io.open(SCRATCH / f"s2_cogs_curve_seed{seed}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row0) + "\n")
    board("checkpoint", row="s2_cogs", **row0)

    for step in CHECKPOINTS[1:]:
        t0 = time.time()
        train_steps(model, opt, g, train, vocab, device, seq, step - prev_step)
        train_s = time.time() - t0
        prev_step = step
        row = eval_checkpoint(model, device, step)
        row["train_seconds_this_stage"] = round(train_s, 1)
        curve.append(row)
        with io.open(SCRATCH / f"s2_cogs_curve_seed{seed}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        board("checkpoint", row="s2_cogs", **row)

    result["curve"] = curve

    # (4) gate shape at the COGS shape, AFTER TRAINING -- both forms, so the
    # "as trained" and "as eval" reads are both on record and comparable to
    # the broken arm's clamp-only number.
    arm_smprime.magnitude = _HARDCONCRETE
    gate_trained_hc = gate_shape_measure(model, device, "hard_concrete_after_training")
    arm_smprime.magnitude = _CLAMP
    gate_trained_clamp = gate_shape_measure(model, device, "clamp_frozen_after_training")
    arm_smprime.magnitude = _HARDCONCRETE
    for label, g_ in (("trained_hard_concrete", gate_trained_hc),
                      ("trained_clamp_frozen", gate_trained_clamp)):
        board("gate_shape", row="s2_cogs", seed=seed, stage=label, **{
            k: v for k, v in g_.items()
            if k not in ("backward_reach_all_layers", "spurious_layer0")},
            backward_reach=g_["backward_reach_all_layers"])
    result["gate_trained_hard_concrete"] = gate_trained_hc
    result["gate_trained_clamp_frozen"] = gate_trained_clamp

    # (5) underfit check
    final_train_loss = curve[-1]["train_loss"]
    control_loss_lo, control_loss_hi = min(CONTROL_TRAIN_LOSS.values()), max(CONTROL_TRAIN_LOSS.values())
    underfit = dict(
        final_train_loss=final_train_loss,
        control_train_loss_range=[control_loss_lo, control_loss_hi],
        broken_arm_train_loss_range=list(BROKEN_TRAIN_LOSS_RANGE),
        trains_to_control_range=(control_loss_lo - 0.02) <= final_train_loss <= (control_loss_hi + 0.02),
        still_underfit_vs_broken_range=(final_train_loss >= BROKEN_TRAIN_LOSS_RANGE[0]))
    result["underfit"] = underfit
    board("underfit_check", row="s2_cogs", seed=seed, **underfit)

    save(f"s2_cogs_seed{seed}.json", result)
    board("seed_done", row="s2_cogs", seed=seed,
          final_generalization=curve[-1]["generalization"],
          final_in_distribution=curve[-1]["in_distribution"])
    return result


def main():
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0").split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board("start", row="s2_cogs", seeds=seeds, device=str(device),
          checkpoints=CHECKPOINTS, resolution_floor=RESOLUTION_FLOOR,
          note="S2: eval-only ArmSMPrime on COGS -- hard-concrete train, "
               "clamp-frozen eval, R1's configuration")
    all_results = {}
    t_start = time.time()
    for seed in seeds:
        res = run_one_seed(seed, device)
        all_results[seed] = res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save("s2_cogs_all.json", dict(seeds=seeds, results=all_results,
                                   wall_clock_s=round(time.time() - t_start, 1)))
    board("done", row="s2_cogs", n_seeds=len(seeds),
          wall_clock_s=round(time.time() - t_start, 1))


if __name__ == "__main__":
    raise SystemExit(main())
