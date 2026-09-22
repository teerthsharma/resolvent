# -*- coding: utf-8 -*-
"""Row R-DO-IC. Three arms -- (a) softmax twin, (a'') FoX twin, (f) gated
hard-concrete -- same builders as R-DO-LM (d=64, 2 layers, 4 heads,
vocab=8, S=128), COPIED from r_do_lm.py and modified (not edited in place).

Difference from R-DO-LM: every training stream comes from its OWN fresh
chain drawn from 512 training chains (do_ic_bed.py's generator, copied here
keeping P as well as its committor -- do_ic_bed only kept q), cycling
through the 512 chains, fresh sample_streams() call per stream. Probe
target is THAT chain's own committor, not one shared observational P. Read
follows the EVAL PROTOCOL fixed for this row (shared with do_ic_bed.py's
floors): each of the 50 do() worlds' 16 streams is read ONE STREAM AT A
TIME, probe output averaged over the positions where each transient state
occurs in that one stream; a state never visited in that stream falls back
to this seed's own 512-training-chain mean committor. error = grand mean
over worlds, streams, states of |q_hat - q_do_exact[:,0]|.

Step budget: STEPS=150, BATCH=64, SEQ=128 -> 1,228,800 tokens/cell,
identical for every arm and seed (data order pinned by data_rng(20000+seed)
and chain_cursor restarted at 0, both reused byte-identically per arm).
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
arm_smprime = D.arm_smprime

RECORD = os.path.join(PHASE_J, "record.jsonl")
DO_WORLDS_JSON = os.path.join(PHASE_J, "do_worlds.json")
FLOORS_JSON = os.path.join(PHASE_J, "do_ic_floors.json")
OUT_JSON = os.path.join(PHASE_J, "r_do_ic.json")

HIDDEN, LAYERS, HEADS, SEQ = 64, 2, 4, 128
VOCAB = N_STATES  # 8
BATCH = 64
STEPS = 150  # step budget, identical for every arm/seed: 150*64*128 = 1,228,800 tokens/cell
LR = D.LR
SMOKE = "--smoke" in sys.argv
if SMOKE:
    STEPS = 5

DEADLINE = (15, 58)  # GPU work may not START after this local time
N_TRAIN_CHAINS = 512
DIRICHLET_ALPHA = 0.5
FOX_ARM = "a2F"
GATED_ARM = "f"
SOFTMAX_ARM = "a"
SEEDS = [0, 1, 2, 3, 4]


def past_deadline():
    now = time.localtime()
    return (now.tm_hour, now.tm_min) >= DEADLINE


def append_record(row):
    with io.open(RECORD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


# ---------------------------------------------------- training-chain bed

def train_chains_full(seed):
    """SAME rng sequence as do_ic_bed.train_chains(seed) -- default_rng(9000+seed),
    one dirichlet(0.5, size=8) draw per chain in order -- but keeps P, not
    just its committor, so streams can be sampled from each chain. qs[:, k]
    is chain c's own committor for transient state TRANSIENT[k] == k."""
    rng = np.random.default_rng(9000 + seed)
    Ps = np.empty((N_TRAIN_CHAINS, N_STATES, N_STATES))
    qs = np.empty((N_TRAIN_CHAINS, len(TRANSIENT)))
    for c in range(N_TRAIN_CHAINS):
        P = rng.dirichlet(np.full(N_STATES, DIRICHLET_ALPHA), size=N_STATES)
        q, *_ = committor(P)
        Ps[c] = P
        qs[c] = q[:, 0]
    return Ps, qs


# ------------------------------------------------------------- arm builders (unchanged from r_do_lm.py)

def _attach_forget_heads_local(model):
    for layer in model.model.layers:
        layer.self_attn.forget_head = nn.Linear(HIDDEN, HEADS)
    return model


def build_arm(arm):
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


# ------------------------------------------------------------- data: one fresh chain per stream

def make_do_ic_batches(Ps, data_rng, n_batches, batch, seq, chain_cursor):
    """n_batches batches of `batch` streams; each stream drawn from its own
    fresh chain, cycling through the 512 training chains starting at
    chain_cursor[0] (mutated in place -- caller resets it to [0] once per
    arm-seed cell so arms sharing a data_rng seed also share the identical
    chain-cycle position). Returns [(x [batch,seq] int64 tensor, chain_ids
    [batch] int array), ...]."""
    out = []
    for _ in range(n_batches):
        chain_ids = np.array([(chain_cursor[0] + i) % N_TRAIN_CHAINS for i in range(batch)])
        chain_cursor[0] += batch
        rows = np.empty((batch, seq), dtype=np.int64)
        for i, cid in enumerate(chain_ids):
            rows[i] = sample_streams(Ps[cid], 1, S=seq, rng=data_rng)[0]
        out.append((torch.from_numpy(rows).long(), chain_ids))
    return out


def train_one(arm, seed, Ps, device, chain_cursor):
    torch.manual_seed(seed)
    model, fwd = build_arm(arm)
    model.to(device)
    data_rng = np.random.default_rng(20000 + seed)  # SAME across arms at this seed
    batches = make_do_ic_batches(Ps, data_rng, STEPS, BATCH, SEQ, chain_cursor)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ctx = CEQAttention.forward
    if fwd is not None:
        CEQAttention.forward = fwd
    t0 = time.time()
    losses = []
    try:
        for x, _cids in batches:
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


def fit_probe(model, fwd, Ps, qs, seed, device, chain_cursor):
    """ONE linear probe: last hidden state -> the generating stream's OWN
    chain committor q_chain[token], fit across streams drawn from the 512
    training chains (closed-form least squares), chain cursor continuing
    from wherever training left it (fresh streams, not reused batches)."""
    rng = np.random.default_rng(30000 + seed)
    batches = make_do_ic_batches(Ps, rng, 8, 48, SEQ, chain_cursor)  # 8*48*128 = 49,152 tokens
    feats, targets = [], []
    for x, cids in batches:
        h = last_hidden(model, fwd, x, device).cpu().numpy()  # [B,S,H]
        tok = x.numpy()
        mask = np.isin(tok, TRANSIENT)
        feats.append(h[mask])
        b_idx, s_idx = np.nonzero(mask)
        targets.append(qs[cids[b_idx], tok[b_idx, s_idx]])  # each position's OWN chain's committor
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(targets, axis=0)
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ w
    fit_err = float(np.mean(np.abs(pred - y)))
    return w, fit_err


def probe_predict(w, h_np, tok_np, states, fallback):
    """h_np [B,S,H], tok_np [B,S] -> per-state mean prediction over all
    occurrences of that state within THIS h_np/tok_np set (call with a
    single stream, B=1, per the eval protocol). A state with zero
    occurrences falls back to `fallback[k]`."""
    Hb = np.concatenate([h_np.reshape(-1, h_np.shape[-1]),
                          np.ones((h_np.size // h_np.shape[-1], 1))], axis=1)
    pred = (Hb @ w).reshape(tok_np.shape)
    tok_flat = tok_np.reshape(-1)
    pred_flat = pred.reshape(-1)
    out = np.array(fallback, dtype=np.float64).copy()
    for k, s in enumerate(states):
        m = tok_flat == s
        if m.any():
            out[k] = pred_flat[m].mean()
    return out


# ------------------------------------------------------------- do-eval (per-stream, per protocol)

def do_error(model, fwd, w, worlds, device, fallback, n_seq=16):
    """Per EVAL PROTOCOL: each of the 50 worlds' n_seq streams is read ONE
    AT A TIME (own fallback per stream), error = grand mean of |q_hat -
    q_do_exact[:,0]| over worlds, streams, transient states."""
    per_row_abs = []
    for wd in worlds:
        P_do = np.array(wd["P_do"])
        q_do0 = np.array(wd["q_do_exact"])[:, 0]
        rng = np.random.default_rng(40000 + wd["world"])
        x = torch.from_numpy(sample_streams(P_do, n_seq, S=SEQ, rng=rng)).long()
        h = last_hidden(model, fwd, x, device).cpu().numpy()
        tok = x.numpy()
        for s_i in range(n_seq):
            est = probe_predict(w, h[s_i:s_i + 1], tok[s_i:s_i + 1], TRANSIENT, fallback)
            per_row_abs.append(np.abs(est - q_do0))
    all_abs = np.concatenate([r[None, :] for r in per_row_abs], axis=0)
    return float(all_abs.mean()), per_row_abs


def training_chain_control_error(model, fwd, w, Ps, qs, seed, device, fallback, chain_cursor, n_seq=16):
    """Control column: same per-stream read protocol, but reading fresh
    streams off the training chains themselves against EACH stream's own
    chain committor (in-distribution error, no do() intervention)."""
    rng = np.random.default_rng(50000 + seed)
    chain_ids = np.array([(chain_cursor[0] + i) % N_TRAIN_CHAINS for i in range(n_seq)])
    chain_cursor[0] += n_seq
    rows = np.empty((n_seq, SEQ), dtype=np.int64)
    for i, cid in enumerate(chain_ids):
        rows[i] = sample_streams(Ps[cid], 1, S=SEQ, rng=rng)[0]
    x = torch.from_numpy(rows).long()
    h = last_hidden(model, fwd, x, device).cpu().numpy()
    tok = x.numpy()
    per_row_abs = []
    for i in range(n_seq):
        q_true = qs[chain_ids[i]]  # (6,), index k == state id (TRANSIENT contiguous 0..5)
        est = probe_predict(w, h[i:i + 1], tok[i:i + 1], TRANSIENT, fallback)
        per_row_abs.append(np.abs(est - q_true))
    return float(np.mean(np.concatenate([r[None, :] for r in per_row_abs], axis=0)))


def main():
    started = time.strftime("%H:%M:%S")
    row_common = dict(machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
                       queue_pos=1, started=started)

    if past_deadline():
        append_record(dict(row="R-DO-IC", arms=[SOFTMAX_ARM, FOX_ARM, GATED_ARM],
                            seeds=None, split_seed=None, seconds_per_cell=None,
                            tokens_seen=None, params_numel=None, bar=None,
                            measured=None, verdict="SKIPPED: deadline",
                            control=None, quintile_profile=None, clock_ratio=None,
                            read="probe", producer="r_do_ic.py", output=None,
                            repro_class="SDPA tolerance",
                            killed="deadline reached before GPU work start",
                            finished=time.strftime("%H:%M:%S"), **row_common))
        print("SKIPPED: deadline", flush=True)
        return

    if not D.poll_until_free(timeout_s=600):
        append_record(dict(row="R-DO-IC", arms=[SOFTMAX_ARM, FOX_ARM, GATED_ARM],
                            seeds=None, split_seed=None, seconds_per_cell=None,
                            tokens_seen=None, params_numel=None, bar=None,
                            measured=None, verdict="SKIPPED: card busy",
                            control=None, quintile_profile=None, clock_ratio=None,
                            read="probe", producer="r_do_ic.py", output=None,
                            repro_class="SDPA tolerance", killed="card never freed",
                            finished=time.strftime("%H:%M:%S"), **row_common))
        print("SKIPPED: card busy", flush=True)
        return

    device = D.DEVICE

    with io.open(DO_WORLDS_JSON, encoding="utf-8") as fh:
        dw = json.load(fh)
    worlds = dw["worlds"]

    with io.open(FLOORS_JSON, encoding="utf-8") as fh:
        floors = json.load(fh)
    floor_iii = floors["errors"]["dirichlet_posterior_mean"]  # strongest input-only reader
    bar_threshold = 0.8 * floor_iii

    numel = {arm: measure_numel(arm) for arm in (SOFTMAX_ARM, FOX_ARM, GATED_ARM)}
    base = numel[SOFTMAX_ARM]
    numel_spread = {arm: (v - base) / base for arm, v in numel.items()}
    numel_ok = all(abs(v) <= 0.02 for v in numel_spread.values())
    print("[NUMEL]", numel, "spread_frac", numel_spread, "within_2pct", numel_ok, flush=True)

    results = {}  # arm -> seed -> dict

    def run_seed(arm, seed):
        t0 = time.time()
        old_mag = arm_smprime.magnitude
        if arm == GATED_ARM:
            arm_smprime.magnitude = R.FORMS["hard_concrete"]
        try:
            Ps, qs = train_chains_full(seed)
            fallback = qs.mean(axis=0)  # this seed's own 512-training-chain mean committor
            chain_cursor = [0]  # restarted per arm-seed cell: identical data order across arms
            model, fwd, train_info = train_one(arm, seed, Ps, device, chain_cursor)
            w, probe_fit_err = fit_probe(model, fwd, Ps, qs, seed, device, chain_cursor)
            ctrl_err = training_chain_control_error(model, fwd, w, Ps, qs, seed, device,
                                                      fallback, chain_cursor, n_seq=16)
            derr, _per_world = do_error(model, fwd, w, worlds, device, fallback, n_seq=16)
        finally:
            arm_smprime.magnitude = old_mag
        wall = time.time() - t0
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return dict(seed=seed, do_error=derr, training_chain_probe_error=ctrl_err,
                    probe_fit_err=probe_fit_err, final_train_loss=train_info["final_loss"],
                    tokens_seen=train_info["tokens"], wall_seconds=wall)

    a_wall = None
    for arm in (SOFTMAX_ARM, FOX_ARM, GATED_ARM):
        results[arm] = {}
        for seed in SEEDS:
            r = run_seed(arm, seed)
            results[arm][seed] = r
            if arm == SOFTMAX_ARM and a_wall is None:
                a_wall = r["wall_seconds"]
            print("[{}] seed={} do_error={:.6f} train_chain_probe_err={:.6f} wall={:.1f}s".format(
                arm, seed, r["do_error"], r["training_chain_probe_error"], r["wall_seconds"]), flush=True)

    means = {arm: float(np.mean([results[arm][s]["do_error"] for s in SEEDS])) for arm in results}
    f_errs = [results[GATED_ARM][s]["do_error"] for s in SEEDS]
    fox_mean = means[FOX_ARM]
    a_mean = means[SOFTMAX_ARM]
    f_mean = means[GATED_ARM]

    f_pass_5of5 = all(e <= bar_threshold for e in f_errs)
    fox_clears_floor = fox_mean > floor_iii
    fox_fails_too = fox_mean <= bar_threshold
    any_clears_floor = any(means[arm] < floor_iii for arm in results)

    if f_pass_5of5 and fox_clears_floor:
        verdict = "PASS: condition 2 reopened"
    elif f_pass_5of5 and fox_fails_too:
        verdict = "FAIL: closed for good"
    elif not any_clears_floor:
        verdict = "DO RETIRES: a three-line estimator beats every arm"
    else:
        verdict = ("NEITHER: bar not cleanly resolved (f_mean={:.6f} 5/5<=thr={} "
                   "fox_mean={:.6f} vs floor={:.6f})".format(f_mean, f_pass_5of5, fox_mean, floor_iii))

    finished = time.strftime("%H:%M:%S")
    clock_ratio = {arm: (float(np.mean([results[arm][s]["wall_seconds"] for s in SEEDS])) / a_wall)
                   for arm in results}

    out = dict(row="R-DO-IC", numel=numel, numel_spread_frac=numel_spread,
               numel_within_2pct=numel_ok, results=results, means=means,
               seeds=SEEDS, floor_iii_dirichlet_posterior_mean=floor_iii,
               bar_threshold_0p8x_floor=bar_threshold, verdict=verdict,
               clock_ratio=clock_ratio, started=started, finished=finished)
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
        training_chain_probe_error={arm: {s: results[arm][s]["training_chain_probe_error"]
                                           for s in results[arm]} for arm in results},
    )
    tokens_seen_total = sum(results[arm][s]["tokens_seen"] for arm in results for s in results[arm])
    seconds_per_cell = float(np.mean([results[arm][s]["wall_seconds"]
                                       for arm in results for s in results[arm]]))

    if SMOKE:
        print("[SMOKE] OK, not appending to record.jsonl", flush=True)
        return

    append_record(dict(
        row="R-DO-IC", arms=[SOFTMAX_ARM, FOX_ARM, GATED_ARM], seeds=SEEDS,
        split_seed=None, machine=row_common["machine"],
        seconds_per_cell=seconds_per_cell, tokens_seen=tokens_seen_total,
        params_numel=numel,
        bar="(f) do-error <= 0.8x floor(iii)={:.6f} on 5/5 seeds AND (a'') > floor(iii) "
            "-> PASS 'condition 2 reopened'; (a'') <= 0.8x floor(iii) too -> FAIL 'closed "
            "for good'; no arm clears floor(iii) -> 'DO RETIRES'".format(floor_iii),
        measured=measured, verdict=verdict, control=control,
        quintile_profile=None, clock_ratio=clock_ratio, read="probe",
        producer="python phase_j/r_do_ic.py", output="phase_j/r_do_ic.json",
        repro_class="SDPA tolerance",
        killed="measures whether a frozen linear probe on a per-chain-trained LM "
               "generalises to 50 held-out do() worlds better than floor(iii), the "
               "Dirichlet(0.5) posterior-mean input-only reader",
        queue_pos=1, started=started, finished=finished,
    ))
    print("[R-DO-IC] verdict={} f={:.6f} fox={:.6f} a={:.6f} floor_iii={:.6f} thr={:.6f}".format(
        verdict, f_mean, fox_mean, a_mean, floor_iii, bar_threshold), flush=True)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        sys.argv.append("--smoke")
        main()
        print("demo OK")
    else:
        main()
