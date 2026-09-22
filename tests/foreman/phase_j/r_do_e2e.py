# -*- coding: utf-8 -*-
"""Row R-DO-E2E. COPIED from r_do_ic.py and modified (not edited in place).

Only the READ changes: instead of a frozen linear probe fit after training,
a committor head (nn.Linear(d, 1) + sigmoid on the last hidden state) is
attached and trained END-TO-END, jointly with next-token loss, on the same
512-chain training streams: loss = next-token CE + lambda * MSE(head(h_t),
q_own_chain[state_t]) at transient positions, lambda = 1.0. Same arms (a),
(a2F), (f), same numel matching (head params included, equal across arms
since the head architecture is identical for every arm), same step budget
and data order as r_do_ic.py (STEPS=150, BATCH=64, SEQ=128, data_rng(20000+
seed), chain_cursor restarted at 0 per arm-seed cell), seeds 0..4.

Eval EXACTLY as r_do_ic.py's do_error: default_rng(40000+world), 16 streams
x S=128 per world, per-seed fallback (this seed's own 512-training-chain
mean committor) -- but reading the trained head instead of a frozen probe:
average sigmoid(head(h_t)) over the positions where each transient state
occurs in that one stream -> q_hat.

Floors on these exact streams (dispatcher-verified): Dirichlet plug-in
0.07412, 1-gram 0.09032, no-update 0.1024-0.1063. Bar (Dr House): (f) <=
0.8 x 0.07412 = 0.059296 on 5/5 seeds AND (a2F) > 0.07412 -> PASS
"condition 2 reopened". (a2F) <= 0.059296 too -> FAIL "closed for good,
consequence struck". No arm reaches (beats) 0.07412 -> "DO DEAD AT EVERY
READ; Phase J closes as a kernel phase". Arms below 0.07412 but neither
rule satisfied -> NEITHER, report the ordering.
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
import torch.nn.functional as F  # noqa: E402

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
OUT_JSON = os.path.join(PHASE_J, "r_do_e2e.json")

HIDDEN, LAYERS, HEADS, SEQ = 64, 2, 4, 128
VOCAB = N_STATES  # 8
BATCH = 64
STEPS = 150  # step budget, identical for every arm/seed: 150*64*128 = 1,228,800 tokens/cell
LR = D.LR
LAMBDA = 1.0  # weight on the committor-head MSE term, stated per the ruling
SMOKE = "--smoke" in sys.argv
if SMOKE:
    STEPS = 5

DEADLINE = (16, 3)  # GPU work may not START after this local time (dispatcher rule)
N_TRAIN_CHAINS = 512
DIRICHLET_ALPHA = 0.5
FOX_ARM = "a2F"
GATED_ARM = "f"
SOFTMAX_ARM = "a"
SEEDS = [0, 1, 2, 3, 4]

# floors on the arms' exact streams (dispatcher-verified, not recomputed here)
FLOOR_PLUGIN = 0.07412
FLOOR_ONEGRAM = 0.09032
FLOOR_NOUPDATE_LO, FLOOR_NOUPDATE_HI = 0.1024, 0.1063
BAR_THRESHOLD = 0.8 * FLOOR_PLUGIN


def past_deadline():
    now = time.localtime()
    return (now.tm_hour, now.tm_min) >= DEADLINE


def append_record(row):
    with io.open(RECORD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


# ---------------------------------------------------- training-chain bed (unchanged from r_do_ic.py)

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


# ------------------------------------------------------------- arm builders (unchanged from r_do_ic.py)

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


def build_head():
    """The committor head: nn.Linear(d, 1); sigmoid applied at call sites.
    Same architecture for every arm, so its params add an identical
    constant to each arm's numel."""
    return nn.Linear(HIDDEN, 1)


def measure_numel(arm):
    torch.manual_seed(0)
    m, _ = build_arm(arm)
    h = build_head()
    n = sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in h.parameters())
    del m, h
    return n


# ------------------------------------------------------------- data: one fresh chain per stream (unchanged)

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


# ------------------------------------------------------------- last hidden state (unchanged)

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


def last_hidden_grad(model, fwd, x, device):
    """Same as last_hidden but WITHOUT no_grad -- used inside the training
    loop so the committor-head loss backprops into the model too."""
    x = x.to(device)
    ctx = CEQAttention.forward
    if fwd is not None:
        CEQAttention.forward = fwd
    try:
        h = model.model(input_ids=x).last_hidden_state
    finally:
        CEQAttention.forward = ctx
    return h  # [B, S, HIDDEN]


# ------------------------------------------------------------- train end-to-end (replaces train_one + fit_probe)

def train_one_e2e(arm, seed, Ps, qs, device, chain_cursor, head):
    torch.manual_seed(seed)
    model, fwd = build_arm(arm)
    model.to(device)
    head.to(device)
    data_rng = np.random.default_rng(20000 + seed)  # SAME across arms at this seed
    batches = make_do_ic_batches(Ps, data_rng, STEPS, BATCH, SEQ, chain_cursor)

    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR)
    ctx = CEQAttention.forward
    if fwd is not None:
        CEQAttention.forward = fwd
    t0 = time.time()
    losses, mse_losses = [], []
    try:
        for x, cids in batches:
            tok_np = x.numpy()
            xg = x.to(device)
            out = model(input_ids=xg, labels=xg)
            ce_loss = out.loss

            h = model.model(input_ids=xg).last_hidden_state  # [B,S,H], grad-enabled
            mask_np = np.isin(tok_np, TRANSIENT)
            b_idx, s_idx = np.nonzero(mask_np)
            if b_idx.size:
                target = torch.from_numpy(qs[cids[b_idx], tok_np[b_idx, s_idx]]).float().to(device)
                h_masked = h[torch.from_numpy(mask_np).to(device)]
                pred = torch.sigmoid(head(h_masked)).squeeze(-1)
                mse = F.mse_loss(pred, target)
            else:
                mse = torch.zeros((), device=device)

            total = ce_loss + LAMBDA * mse
            opt.zero_grad()
            total.backward()
            opt.step()
            losses.append(float(ce_loss.detach()))
            mse_losses.append(float(mse.detach()))
    finally:
        CEQAttention.forward = ctx
    wall = time.time() - t0
    model.eval()
    head.eval()
    return model, fwd, dict(final_loss=losses[-1], final_mse=mse_losses[-1],
                             losses=losses, wall=wall, tokens=STEPS * BATCH * SEQ)


# ------------------------------------------------------------- head read (replaces probe_predict)

def head_predict(head, h_np, tok_np, states, fallback, device):
    """h_np [B,S,H], tok_np [B,S] -> per-state mean sigmoid(head(h)) over all
    occurrences of that state within THIS h_np/tok_np set (call with a
    single stream, B=1, per the eval protocol). A state with zero
    occurrences falls back to `fallback[k]`."""
    with torch.no_grad():
        h_t = torch.from_numpy(h_np).float().to(device)
        pred = torch.sigmoid(head(h_t)).squeeze(-1).cpu().numpy()  # [B,S]
    tok_flat = tok_np.reshape(-1)
    pred_flat = pred.reshape(-1)
    out = np.array(fallback, dtype=np.float64).copy()
    for k, s in enumerate(states):
        m = tok_flat == s
        if m.any():
            out[k] = pred_flat[m].mean()
    return out


# ------------------------------------------------------------- do-eval (per-stream, per protocol -- EXACTLY r_do_ic.py's do_error, head swapped in for w)

def do_error(model, fwd, head, worlds, device, fallback, n_seq=16):
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
            est = head_predict(head, h[s_i:s_i + 1], tok[s_i:s_i + 1], TRANSIENT, fallback, device)
            per_row_abs.append(np.abs(est - q_do0))
    all_abs = np.concatenate([r[None, :] for r in per_row_abs], axis=0)
    return float(all_abs.mean()), per_row_abs


def training_chain_control_error(model, fwd, head, Ps, qs, seed, device, fallback, chain_cursor, n_seq=16):
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
        est = head_predict(head, h[i:i + 1], tok[i:i + 1], TRANSIENT, fallback, device)
        per_row_abs.append(np.abs(est - q_true))
    return float(np.mean(np.concatenate([r[None, :] for r in per_row_abs], axis=0)))


def main():
    started = time.strftime("%H:%M:%S")
    row_common = dict(machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
                       queue_pos=1, started=started)

    if past_deadline():
        append_record(dict(row="R-DO-E2E", arms=[SOFTMAX_ARM, FOX_ARM, GATED_ARM],
                            seeds=None, split_seed=None, seconds_per_cell=None,
                            tokens_seen=None, params_numel=None, bar=None,
                            measured=None, verdict="SKIPPED: deadline",
                            control=None, quintile_profile=None, clock_ratio=None,
                            read="e2e head", producer="r_do_e2e.py", output=None,
                            repro_class="SDPA tolerance",
                            killed="deadline reached before GPU work start",
                            finished=time.strftime("%H:%M:%S"), **row_common))
        print("SKIPPED: deadline", flush=True)
        return

    if not D.poll_until_free(timeout_s=300):
        append_record(dict(row="R-DO-E2E", arms=[SOFTMAX_ARM, FOX_ARM, GATED_ARM],
                            seeds=None, split_seed=None, seconds_per_cell=None,
                            tokens_seen=None, params_numel=None, bar=None,
                            measured=None, verdict="SKIPPED: card busy",
                            control=None, quintile_profile=None, clock_ratio=None,
                            read="e2e head", producer="r_do_e2e.py", output=None,
                            repro_class="SDPA tolerance", killed="card never freed",
                            finished=time.strftime("%H:%M:%S"), **row_common))
        print("SKIPPED: card busy", flush=True)
        return

    device = D.DEVICE

    with io.open(DO_WORLDS_JSON, encoding="utf-8") as fh:
        dw = json.load(fh)
    worlds = dw["worlds"]

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
            torch.manual_seed(seed)
            head = build_head()
            model, fwd, train_info = train_one_e2e(arm, seed, Ps, qs, device, chain_cursor, head)
            ctrl_err = training_chain_control_error(model, fwd, head, Ps, qs, seed, device,
                                                      fallback, chain_cursor, n_seq=16)
            derr, _per_world = do_error(model, fwd, head, worlds, device, fallback, n_seq=16)
        finally:
            arm_smprime.magnitude = old_mag
        wall = time.time() - t0
        del model, head
        if device == "cuda":
            torch.cuda.empty_cache()
        return dict(seed=seed, do_error=derr, training_chain_probe_error=ctrl_err,
                    final_mse=train_info["final_mse"], final_train_loss=train_info["final_loss"],
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

    f_pass_5of5 = all(e <= BAR_THRESHOLD for e in f_errs)
    fox_above_floor = fox_mean > FLOOR_PLUGIN
    fox_at_or_below_thr = fox_mean <= BAR_THRESHOLD
    any_clears_floor = any(means[arm] < FLOOR_PLUGIN for arm in results)

    if f_pass_5of5 and fox_above_floor:
        verdict = "PASS: condition 2 reopened"
    elif f_pass_5of5 and fox_at_or_below_thr:
        verdict = "FAIL: closed for good, consequence struck"
    elif not any_clears_floor:
        verdict = "DO DEAD AT EVERY READ: Phase J closes as a kernel phase"
    else:
        order = sorted(means, key=lambda arm: means[arm])
        verdict = ("NEITHER: bar not cleanly resolved, ordering {} ({:.6f} < {:.6f} < {:.6f})".format(
            "<".join(order), means[order[0]], means[order[1]], means[order[2]]))

    finished = time.strftime("%H:%M:%S")
    clock_ratio = {arm: (float(np.mean([results[arm][s]["wall_seconds"] for s in SEEDS])) / a_wall)
                   for arm in results}

    out = dict(row="R-DO-E2E", numel=numel, numel_spread_frac=numel_spread,
               numel_within_2pct=numel_ok, results=results, means=means,
               seeds=SEEDS, floor_plugin=FLOOR_PLUGIN, floor_onegram=FLOOR_ONEGRAM,
               floor_noupdate_range=[FLOOR_NOUPDATE_LO, FLOOR_NOUPDATE_HI],
               bar_threshold_0p8x_floor=BAR_THRESHOLD, verdict=verdict,
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
        row="R-DO-E2E", arms=[SOFTMAX_ARM, FOX_ARM, GATED_ARM], seeds=SEEDS,
        split_seed=None, machine=row_common["machine"],
        seconds_per_cell=seconds_per_cell, tokens_seen=tokens_seen_total,
        params_numel=numel,
        bar="(f) do-error <= 0.8x floor_plugin={:.6f} on 5/5 seeds AND (a2F) > floor_plugin "
            "-> PASS 'condition 2 reopened'; (a2F) <= 0.8x floor_plugin too -> FAIL 'closed "
            "for good, consequence struck'; no arm beats floor_plugin -> 'DO DEAD AT EVERY "
            "READ, Phase J closes as a kernel phase'; else NEITHER".format(FLOOR_PLUGIN),
        measured=measured, verdict=verdict, control=control,
        quintile_profile=None, clock_ratio=clock_ratio, read="e2e head",
        producer="python phase_j/r_do_e2e.py", output="phase_j/r_do_e2e.json",
        repro_class="SDPA tolerance",
        killed="measures whether a committor head trained END-TO-END with next-token "
               "loss on a per-chain-trained LM generalises to 50 held-out do() worlds "
               "better than the Dirichlet(0.5) plug-in floor on these exact streams "
               "(0.07412), replacing R-DO-IC's frozen post-hoc probe read",
        queue_pos=1, started=started, finished=finished,
    ))
    print("[R-DO-E2E] verdict={} f={:.6f} fox={:.6f} a={:.6f} floor_plugin={:.6f} thr={:.6f}".format(
        verdict, f_mean, fox_mean, a_mean, FLOOR_PLUGIN, BAR_THRESHOLD), flush=True)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        sys.argv.append("--smoke")
        main()
        print("demo OK")
    else:
        main()
