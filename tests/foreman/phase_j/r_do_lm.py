# -*- coding: utf-8 -*-
"""Row R-DO-LM. Three arms -- (a) softmax twin, (a'') FoX twin, (f) gated
hard-concrete -- trained on OBSERVATIONAL J2-bed streams at vocab=8, S=128,
d=64, 2 layers, 4 heads. One linear probe per arm, fit on observational
hidden states -> observational committor q[:,0] (P(hit 6 before 7)), then
frozen. Read at each of 50 do() worlds from 16 in-context streams of that
world's intervened chain only, no retraining. do-error = mean |estimate -
q_do_exact| over worlds and transient states.
"""
import io
import json
import os
import sys
import time

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
REPO = r"C:\Users\seal\Desktop\New folder (32)"
PHASE_J = os.path.join(SCRATCH, "phase_j")
sys.path.insert(0, PHASE_J)
sys.path.insert(0, SCRATCH)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import design4x5 as D  # noqa: E402 -- inserts r1_gate/r3_eval/q2 dirs onto sys.path
import r1_gate as R  # noqa: E402
import arms_j as J  # noqa: E402
from do_worlds import committor, sample_streams, TRANSIENT, ABSORBING, n as N_STATES  # noqa: E402

CEQAttention = D.CEQAttention
softmax_forward = D.softmax_forward
_ORIG_BUILD = D._ORIG_BUILD
_ORIG_ATTN_FORWARD = D._ORIG_ATTN_FORWARD
arm_smprime = D.arm_smprime

RECORD = os.path.join(PHASE_J, "record.jsonl")
DO_WORLDS_JSON = os.path.join(PHASE_J, "do_worlds.json")
OUT_JSON = os.path.join(PHASE_J, "r_do_lm.json")

HIDDEN, LAYERS, HEADS, SEQ = 64, 2, 4, 128
VOCAB = N_STATES  # 8
BATCH = 64
STEPS = 150  # step budget stated here: 150*64*128 = 1,228,800 tokens/cell
LR = D.LR  # 3e-4, same optimizer/lr as design4x5
SMOKE = "--smoke" in sys.argv
if SMOKE:
    STEPS = 5

DEADLINE = (15, 45)
BAG_BAR = 0.0629
FOX_ARM = "a2F"
GATED_ARM = "f"
SOFTMAX_ARM = "a"


def past_deadline():
    now = time.localtime()
    return (now.tm_hour, now.tm_min) >= DEADLINE


def append_record(row):
    with io.open(RECORD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


# ------------------------------------------------------------- arm builders

def _attach_forget_heads_local(model):
    """arms_j._attach_forget_heads hardcodes D.HIDDEN/D.HEADS (design4x5's
    128/8 shape) -- wrong at this row's 64/4 shape. Same construction
    (nn.Linear(hidden, n_heads) per layer), sized for THIS run's HIDDEN/HEADS."""
    for layer in model.model.layers:
        layer.self_attn.forget_head = nn.Linear(HIDDEN, HEADS)
    return model


def build_arm(arm):
    """Returns (model, forward_fn_or_None). forward_fn is installed on
    CEQAttention.forward for the duration of any call touching the model
    (training step, hidden-state extraction). None means: leave
    CEQAttention.forward at whatever design4x5/r1_gate already wired for the
    smprime path (the real operator, driven by arm_smprime.magnitude)."""
    common = dict(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                  seq=SEQ, vocab_size=VOCAB)
    if arm == SOFTMAX_ARM:
        model = _ORIG_BUILD(operator="sgate", **common)
        return model, softmax_forward
    if arm == FOX_ARM:
        model = _attach_forget_heads_local(_ORIG_BUILD(operator="sgate", **common))
        return model, J.fox_forward
    if arm == GATED_ARM:
        model = R.build_repaired(operator="smprime", **common)
        return model, None
    raise ValueError(arm)


def measure_numel(arm):
    torch.manual_seed(0)
    m, _ = build_arm(arm)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


# ------------------------------------------------------------- data

def make_batches(P, rng, n_batches, batch, seq):
    """List of int64 torch tensors [batch, seq], CPU, one sample_streams
    call per batch so the SAME rng, consumed in the SAME order, gives every
    arm the identical data order at a given seed."""
    out = []
    for _ in range(n_batches):
        x = sample_streams(P, batch, S=seq, rng=rng)
        out.append(torch.from_numpy(x).long())
    return out


def train_one(arm, seed, P_obs, device):
    torch.manual_seed(seed)
    model, fwd = build_arm(arm)
    model.to(device)
    data_rng = np.random.default_rng(20000 + seed)  # SAME across arms at this seed
    batches = make_batches(P_obs, data_rng, STEPS, BATCH, SEQ)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ctx = CEQAttention.forward
    if fwd is not None:
        CEQAttention.forward = fwd
    t0 = time.time()
    losses = []
    try:
        for x in batches:
            x = x.to(device)
            out = model(input_ids=x, labels=x)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            losses.append(float(out.loss.detach()))
    finally:
        CEQAttention.forward = ctx
    wall = time.time() - t0
    model.eval()
    return model, fwd, dict(final_loss=losses[-1], losses=losses, wall=wall,
                             tokens=STEPS * BATCH * SEQ)


# ------------------------------------------------------------- probe

def last_hidden(model, fwd, x, device):
    x = x.to(device)
    ctx = CEQAttention.forward
    if fwd is not None:
        CEQAttention.forward = fwd
    try:
        with torch.no_grad():
            h = model.model(input_ids=x).last_hidden_state
    finally:
        CEQAttention.forward = ctx
    return h  # [B, S, HIDDEN]


def fit_probe(model, fwd, P_obs, q_obs0, seed, device):
    """ONE linear probe: last hidden state -> observational committor
    q_obs0[state], fit on OBSERVATIONAL streams at positions whose token is
    a transient state, closed-form least squares."""
    rng = np.random.default_rng(30000 + seed)
    batches = make_batches(P_obs, rng, 8, 48, SEQ)  # 8*48*128 = 49,152 tokens
    feats, targets = [], []
    for x in batches:
        h = last_hidden(model, fwd, x, device).cpu().numpy()  # [B,S,H]
        tok = x.numpy()
        mask = np.isin(tok, TRANSIENT)
        feats.append(h[mask])
        targets.append(q_obs0[tok[mask]])
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(targets, axis=0)
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ w
    fit_err = float(np.mean(np.abs(pred - y)))
    return w, fit_err


def probe_predict(w, h_np, tok_np, states):
    """h_np [B,S,H], tok_np [B,S] -> per-state mean prediction over all
    occurrences of that state, for each state in `states`."""
    Hb = np.concatenate([h_np.reshape(-1, h_np.shape[-1]),
                          np.ones((h_np.size // h_np.shape[-1], 1))], axis=1)
    pred = (Hb @ w).reshape(tok_np.shape)
    tok_flat = tok_np.reshape(-1)
    pred_flat = pred.reshape(-1)
    out = np.full(len(states), np.nan)
    for k, s in enumerate(states):
        m = tok_flat == s
        if m.any():
            out[k] = pred_flat[m].mean()
    return out


# ------------------------------------------------------------- do-eval

def do_error(model, fwd, w, worlds, device, n_seq=16):
    per_world_abs = []
    for wd in worlds:
        P_do = np.array(wd["P_do"])
        q_do0 = np.array(wd["q_do_exact"])[:, 0]
        rng = np.random.default_rng(40000 + wd["world"])
        x = torch.from_numpy(sample_streams(P_do, n_seq, S=SEQ, rng=rng)).long()
        h = last_hidden(model, fwd, x, device).cpu().numpy()
        est = probe_predict(w, h, x.numpy(), TRANSIENT)
        per_world_abs.append(np.abs(est - q_do0))
    all_abs = np.concatenate([a[~np.isnan(a)] for a in per_world_abs])
    return float(all_abs.mean()), per_world_abs


def main():
    started = time.strftime("%H:%M:%S")
    row_common = dict(machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
                       queue_pos=3, started=started)

    if past_deadline():
        append_record(dict(row="R-DO-LM", arms=[SOFTMAX_ARM, FOX_ARM, GATED_ARM],
                            seeds=None, split_seed=None, seconds_per_cell=None,
                            tokens_seen=None, params_numel=None, bar=None,
                            measured=None, verdict="SKIPPED: deadline",
                            control=None, quintile_profile=None, clock_ratio=None,
                            read="probe", producer="r_do_lm.py", output=None,
                            repro_class="SDPA tolerance",
                            killed="deadline reached before GPU work start",
                            finished=time.strftime("%H:%M:%S"), **row_common))
        print("SKIPPED: deadline", flush=True)
        return

    if not D.poll_until_free(timeout_s=900):
        append_record(dict(row="R-DO-LM", arms=[SOFTMAX_ARM, FOX_ARM, GATED_ARM],
                            seeds=None, split_seed=None, seconds_per_cell=None,
                            tokens_seen=None, params_numel=None, bar=None,
                            measured=None, verdict="SKIPPED: card busy",
                            control=None, quintile_profile=None, clock_ratio=None,
                            read="probe", producer="r_do_lm.py", output=None,
                            repro_class="SDPA tolerance", killed="card never freed",
                            finished=time.strftime("%H:%M:%S"), **row_common))
        print("SKIPPED: card busy", flush=True)
        return

    device = D.DEVICE

    with io.open(DO_WORLDS_JSON, encoding="utf-8") as fh:
        dw = json.load(fh)
    P_obs = np.array(dw["P"])
    q_obs = np.array(dw["q_observational"])  # [6,2]
    q_obs0 = np.zeros(N_STATES)
    for k, s in enumerate(TRANSIENT):
        q_obs0[s] = q_obs[k, 0]
    worlds = dw["worlds"]

    numel = {arm: measure_numel(arm) for arm in (SOFTMAX_ARM, FOX_ARM, GATED_ARM)}
    base = numel[SOFTMAX_ARM]
    numel_spread = {arm: (v - base) / base for arm, v in numel.items()}
    numel_ok = all(abs(v) <= 0.02 for v in numel_spread.values())
    print("[NUMEL]", numel, "spread_frac", numel_spread, "within_2pct", numel_ok, flush=True)

    results = {}  # arm -> seed -> dict
    seeds_run = [0] if SMOKE else [0, 1, 2]
    extra_seeds = [3, 4]

    def run_seed(arm, seed):
        t0 = time.time()
        old_mag = arm_smprime.magnitude
        if arm == GATED_ARM:
            arm_smprime.magnitude = R.FORMS["hard_concrete"]
        try:
            model, fwd, train_info = train_one(arm, seed, P_obs, device)
            w, probe_fit_err = fit_probe(model, fwd, P_obs, q_obs0, seed, device)
            # observational probe error (control column): same metric, held-out
            # observational streams, treated exactly like a "world" with P==P_obs.
            obs_pseudo_world = dict(world=-1, P_do=P_obs.tolist(),
                                     q_do_exact=q_obs.tolist())
            obs_err, _ = do_error(model, fwd, w, [obs_pseudo_world], device, n_seq=16)
            derr, per_world = do_error(model, fwd, w, worlds, device, n_seq=16)
        finally:
            arm_smprime.magnitude = old_mag
        wall = time.time() - t0
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return dict(seed=seed, do_error=derr, obs_probe_error=obs_err,
                    probe_fit_err=probe_fit_err, final_train_loss=train_info["final_loss"],
                    tokens_seen=train_info["tokens"], wall_seconds=wall)

    for arm in (SOFTMAX_ARM, FOX_ARM, GATED_ARM):
        results[arm] = {}
        for seed in seeds_run:
            r = run_seed(arm, seed)
            results[arm][seed] = r
            print("[{}] seed={} do_error={:.6f} obs_probe_err={:.6f} wall={:.1f}s".format(
                arm, seed, r["do_error"], r["obs_probe_error"], r["wall_seconds"]), flush=True)

    means = {arm: float(np.mean([results[arm][s]["do_error"] for s in seeds_run]))
             for arm in results}
    close_to_bar = (not SMOKE) and any(abs(m - BAG_BAR) <= 0.01 for m in means.values())
    if close_to_bar:
        for arm in (SOFTMAX_ARM, FOX_ARM, GATED_ARM):
            for seed in extra_seeds:
                r = run_seed(arm, seed)
                results[arm][seed] = r
                print("[{}] seed={} (extra) do_error={:.6f} wall={:.1f}s".format(
                    arm, seed, r["do_error"], r["wall_seconds"]), flush=True)
        seeds_final = seeds_run + extra_seeds
        means = {arm: float(np.mean([results[arm][s]["do_error"] for s in seeds_final]))
                 for arm in results}
    else:
        seeds_final = seeds_run

    f_mean = means[GATED_ARM]
    fox_mean = means[FOX_ARM]
    a_mean = means[SOFTMAX_ARM]

    f_pass = f_mean < BAG_BAR
    fox_fail_cond = fox_mean >= BAG_BAR
    within_01 = abs(fox_mean - f_mean) <= 0.01

    if f_pass and fox_fail_cond:
        verdict_house = "PASS: condition 2 reopened"
    elif within_01:
        verdict_house = "FAIL: closed for good, consequence struck"
    elif f_mean >= BAG_BAR and fox_mean >= BAG_BAR:
        verdict_house = "NEITHER"
    else:
        verdict_house = "NEITHER (bar not cleanly resolved: f_mean={:.4f} fox_mean={:.4f})".format(
            f_mean, fox_mean)

    best_control_err = 0.009702124371860086  # kgram_k1, from do_dpi.json (R-DPI, this session)
    void_margin = 0.02
    void_threshold = best_control_err - void_margin  # -0.010298 -- unreachable by a nonneg error
    dpi_void = True  # PASS is void unless do-error <= void_threshold, which is < 0
    verdict_final = verdict_house
    if verdict_house.startswith("PASS"):
        verdict_final = ("VOID (Chase's rule): PASS would need do-error <= {:.6f} "
                          "(best R-DPI control {:.6f} - 0.02), which is negative and "
                          "unreachable by any nonnegative error".format(
                              void_threshold, best_control_err))

    finished = time.strftime("%H:%M:%S")
    out = dict(numel=numel, numel_spread_frac=numel_spread, numel_within_2pct=numel_ok,
               results=results, means=means, seeds_run=seeds_final,
               verdict_house=verdict_house, verdict_final=verdict_final,
               best_dpi_control=best_control_err, void_threshold=void_threshold,
               bag_bar=BAG_BAR, started=started, finished=finished)
    with io.open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2), flush=True)

    measured = dict(
        do_error_mean={arm: means[arm] for arm in means},
        do_error_per_seed={arm: {s: results[arm][s]["do_error"] for s in results[arm]}
                            for arm in results},
        train_loss_final={arm: {s: results[arm][s]["final_train_loss"] for s in results[arm]}
                           for arm in results},
    )
    control = dict(
        observational_probe_error={arm: {s: results[arm][s]["obs_probe_error"]
                                          for s in results[arm]} for arm in results},
        best_r_dpi_control=best_control_err,
        void_margin=void_margin,
    )
    tokens_seen_total = sum(results[arm][s]["tokens_seen"]
                             for arm in results for s in results[arm])
    seconds_per_cell = float(np.mean([results[arm][s]["wall_seconds"]
                                       for arm in results for s in results[arm]]))

    if SMOKE:
        print("[SMOKE] OK, not appending to record.jsonl", flush=True)
        return

    append_record(dict(
        row="R-DO-LM", arms=[SOFTMAX_ARM, FOX_ARM, GATED_ARM], seeds=seeds_final,
        split_seed=None, machine=row_common["machine"],
        seconds_per_cell=seconds_per_cell, tokens_seen=tokens_seen_total,
        params_numel=numel,
        bar="(f) do-error < 0.0629 AND (a'') >= 0.0629 -> PASS; (a'') within 0.01 of (f) "
            "-> FAIL; neither beats 0.0629 -> NEITHER. Any PASS must additionally beat "
            "best R-DPI control (0.009702) by >= 0.02 (Chase's rule) or is VOID.",
        measured=measured, verdict=verdict_final, control=control,
        quintile_profile=None, clock_ratio=None, read="probe",
        producer="python phase_j/r_do_lm.py", output="phase_j/r_do_lm.json",
        repro_class="SDPA tolerance",
        killed="measures whether a trained LM's frozen linear probe generalises the "
               "observational committor to 50 held-out do() worlds better than a "
               "forget-gated twin, against the gated arm and the FoX arm both",
        queue_pos=3, started=started, finished=finished,
    ))
    print("[R-DO-LM] house_verdict={} final_verdict={} f={:.6f} fox={:.6f} a={:.6f}".format(
        verdict_house, verdict_final, f_mean, fox_mean, a_mean), flush=True)


if __name__ == "__main__":
    main()
