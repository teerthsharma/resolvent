"""
Opus checker probe for the H3v2 matched four-twin table. Reads, does not edit,
h1_twins_ab_matched.py / phaseh_H3v2_twins_cd.py; imports both by file path
(main() is __main__-guarded in each, so import runs no training).

Two questions the matched table on disk cannot answer by itself:

E1 -- BUDGET RESIDUE. The matched (a)/(b) rows carry final_train_loss
      1.69-1.83 against an untrained 2.09 = ln(8), while (c)/(d) carry
      0.032-0.050. Steps were matched at 400 full-batch; CONVERGENCE was not.
      So "matched budget" may still be an unmatched comparison. Re-run (a) and
      (b) at 4000 steps (10x) and see whether train loss keeps falling and eval
      accuracy keeps rising. If both are flat, 400 steps was enough and the low
      score is the primitive; if they rise, the budget confound survives.

E2 -- HYPOTHESIS-CLASS CONTROL. (c)/(d)'s forward pass IS the bed's generative
      recursion, state = op[verb] @ (state + emb[patient]). Give a VECTOR arm
      the same structural advantage -- the identical recurrence, same depth,
      same readout -- with the per-verb matrix replaced by a per-verb DIAGONAL
      gate, state = g[verb] * (state + emb[patient]). Diagonal gates commute,
      so this arm has the generator's recurrence but cannot represent order.
      Whatever it scores separates "the recurrence is the advantage" from
      "the non-commutative operator is the advantage".

Same bed, same offsets (TWINS_OFFSET/ANCHOR_OFFSET/WRONG_OFFSET), same 5
seeds, same discrete-label metric, same AdamW(lr, wd), same full-batch shape,
float32, CPU. Board: one JSON line per event, appended.
Run: python h1_checker_probe.py
"""
import io, json, os, platform, time, importlib.util
import numpy as np, torch, torch.nn as nn

SCRATCH = os.path.dirname(os.path.abspath(__file__))
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
OUT = os.path.join(SCRATCH, "h1_checker_probe_results.json")
COMMAND = "python h1_checker_probe.py"
PRODUCER = "Foreman/torch-%s/numpy-%s/py-%s" % (torch.__version__, np.__version__, platform.python_version())


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = _imp("h1_ab_matched_ro", os.path.join(SCRATCH, "h1_twins_ab_matched.py"))
CD = _imp("h3v2_twins_cd_ro", os.path.join(SCRATCH, "phaseh_H3v2_twins_cd.py"))

M, D, K, LINE_LEN, C, SEEDS = AB.M, AB.D, AB.K, AB.LINE_LEN, AB.C, AB.SEEDS
N_TRAIN, N_EVAL, LR, WD = AB.N_TRAIN, AB.N_EVAL, AB.LR, AB.WEIGHT_DECAY
assert (AB.TWINS_OFFSET, AB.ANCHOR_OFFSET) == (CD.TWINS_OFFSET, CD.ANCHOR_OFFSET), "split mismatch"
assert (AB.N_TRAIN, AB.N_EVAL, AB.LR, AB.WEIGHT_DECAY) == (CD.N_TRAIN, CD.N_EVAL, CD.LR, CD.WEIGHT_DECAY)


def log(e):
    e = dict(e)
    e.setdefault("agent", "Foreman")
    with io.open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps(e) + "\n")


def T(name, ok, detail, dur=0.0):
    log({"t": "test", "state": "GREEN" if ok else "RED", "name": name, "detail": detail, "dur": round(dur, 4)})
    print(("GREEN " if ok else "RED   ") + name + " :: " + detail)
    return ok


def F(item, text):
    log({"t": "finding", "item": item, "text": text})
    print("FINDING [" + item + "] " + text)


class DiagRecurrentVectorArm(nn.Module):
    """E2: (c)'s recurrence with the matrix replaced by a per-verb diagonal
    gate. Commutative by construction -- elementwise products commute -- so it
    holds the generator's SHAPE with no order-sensitive composition."""

    def __init__(self, d):
        super().__init__()
        self.ent_emb = nn.Embedding(M, d)
        self.gate = nn.Parameter(torch.zeros(K, d))  # exp(0)=1 -> identity at init
        self.readout = nn.Linear(d, C)

    def forward(self, agent0, verbs, patients):
        state = self.ent_emb(agent0)
        for t in range(LINE_LEN):
            g = torch.exp(self.gate)[verbs[:, t]]
            state = g * (state + self.ent_emb(patients[:, t]))
        return self.readout(state)


def numel(m):
    return int(sum(p.numel() for p in m.parameters()))


PROBE_STEPS = (400, 1000, 2000, 4000)


def train_eval(model, tr, ev, steps=4000):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lf = nn.CrossEntropyLoss()
    t0 = time.time()
    traj = {}
    for s in range(1, steps + 1):
        opt.zero_grad()
        loss = lf(model(*tr[:3]), tr[3])
        loss.backward()
        opt.step()
        if s in PROBE_STEPS:
            with torch.no_grad():
                lg = model(*ev[:3])
                traj[s] = {"train_loss": float(loss),
                           "eval_acc": float((lg.argmax(-1) == ev[3]).float().mean())}
    return traj, time.time() - t0


def main():
    t_all = time.time()
    F("checker_probe_start",
      "Opus checker probe on the matched four-twin table. E1: (a)/(b) at 4000 steps vs the "
      "table's 400, to test whether 400 full-batch steps converged them (their rows show "
      "final_train_loss 1.69-1.83 vs (c)/(d)'s 0.032-0.050). E2: vector arm given (c)'s exact "
      "recurrence with a per-verb DIAGONAL (commutative) gate instead of a matrix. Same bed, "
      "offsets TWINS_OFFSET=%d ANCHOR_OFFSET=%d, seeds=%s, n_train=%d n_eval=%d, "
      "AdamW(lr=%.3g, wd=%.1e), full-batch, float32, CPU, producer=%s, command=%s" %
      (AB.TWINS_OFFSET, AB.ANCHOR_OFFSET, list(SEEDS), N_TRAIN, N_EVAL, LR, WD, PRODUCER, COMMAND))

    # d for the diagonal arm: live numel() search against (c)=552, not a formula.
    cand = [(d, numel(DiagRecurrentVectorArm(d))) for d in range(4, 33)]
    d_diag, n_diag = min(cand, key=lambda r: abs(r[1] - 552))
    F("e2_param_match",
      "DiagRecurrentVectorArm live numel() sweep d in [4,32]: chose d=%d -> numel=%d "
      "(vs (c)=552: %+d, vs (a)/(b)=620: %+d). Read off a real nn.Module, not a config." %
      (d_diag, n_diag, n_diag - 552, n_diag - 620))

    rows = {"a": [], "b": [], "diag": []}
    for seed in SEEDS:
        ents, vops, inst = AB.build_population_n(seed, AB.TWINS_OFFSET, N_TRAIN + N_EVAL)
        anchors = AB.anchors_for(seed, AB.ANCHOR_OFFSET)
        tr_i, ev_i = inst[:N_TRAIN], inst[N_TRAIN:]

        def pack(ii):
            a, v, p, ts = AB.to_tensors(ii)
            return a, v, p, torch.tensor(AB.label_of(anchors, ts), dtype=torch.long)

        tr, ev = pack(tr_i), pack(ev_i)

        for arm, use_gate in (("a", False), ("b", True)):
            torch.manual_seed(seed)  # identical init draw to the matched run
            m = AB.GatedArmMatched(use_gate=use_gate, d_model=10, hidden_readout=False)
            assert numel(m) == 620, numel(m)
            traj, dur = train_eval(m, tr, ev)
            rows[arm].append({"seed": seed, "numel": 620, "traj": traj, "seconds": dur})
            log({"t": "test", "agent": "Foreman", "state": "GREEN",
                 "name": "e1_arm_%s_seed%d_to_4000_steps" % (arm, seed),
                 "detail": "steps->(train_loss, eval_acc): " + ", ".join(
                     "%d->(%.4f, %.4f)" % (s, v["train_loss"], v["eval_acc"]) for s, v in traj.items()),
                 "dur": round(dur, 3)})
            print("E1 arm %s seed%d %s" % (arm, seed, traj))

        torch.manual_seed(seed)
        md = DiagRecurrentVectorArm(d_diag)
        traj, dur = train_eval(md, tr, ev)
        fl = AB.eval_floors_on_split(ents, vops, ev_i, anchors, seed)
        rows["diag"].append({"seed": seed, "numel": n_diag, "d": d_diag, "traj": traj,
                             "seconds": dur, "floors": fl})
        log({"t": "test", "agent": "Foreman", "state": "GREEN",
             "name": "e2_diag_recurrent_seed%d" % seed,
             "detail": "numel=%d d=%d steps->(train_loss, eval_acc): %s | last_two_ops floor=%.4f" % (
                 n_diag, d_diag, ", ".join("%d->(%.4f, %.4f)" % (s, v["train_loss"], v["eval_acc"])
                                           for s, v in traj.items()), fl["last_two_ops"]),
             "dur": round(dur, 3)})
        print("E2 diag seed%d %s floor_lt=%.4f" % (seed, traj, fl["last_two_ops"]))

    def agg(key, step):
        v = [r["traj"][step]["eval_acc"] for r in rows[key]]
        l = [r["traj"][step]["train_loss"] for r in rows[key]]
        return {"acc_mean": float(np.mean(v)), "acc_std": float(np.std(v)), "per_seed": v,
                "train_loss_mean": float(np.mean(l))}

    floor_lt = [r["floors"]["last_two_ops"] for r in rows["diag"]]
    floor_mean = float(np.mean(floor_lt))
    A = {s: agg("a", s) for s in PROBE_STEPS}
    B = {s: agg("b", s) for s in PROBE_STEPS}
    Dg = {s: agg("diag", s) for s in PROBE_STEPS}

    a_gain = A[4000]["acc_mean"] - A[400]["acc_mean"]
    b_gain = B[4000]["acc_mean"] - B[400]["acc_mean"]
    e1_converged = abs(a_gain) < 0.02 and abs(b_gain) < 0.02
    T("e1_400_steps_was_convergence_for_a_and_b", e1_converged,
      "10x steps (400->4000), 5 seeds, same init/data/offsets: (a) acc %.4f->%.4f (%+.4f), "
      "train_loss %.4f->%.4f | (b) acc %.4f->%.4f (%+.4f), train_loss %.4f->%.4f | tightest "
      "floor last_two_ops on this eval split = %.4f. GREEN means 400 steps had already "
      "converged them, so the matched-table score is the primitive, not the budget." %
      (A[400]["acc_mean"], A[4000]["acc_mean"], a_gain, A[400]["train_loss_mean"],
       A[4000]["train_loss_mean"], B[400]["acc_mean"], B[4000]["acc_mean"], b_gain,
       B[400]["train_loss_mean"], B[4000]["train_loss_mean"], floor_mean))
    best4k = max(A[4000]["acc_mean"], B[4000]["acc_mean"])
    T("e1_a_or_b_clears_tightest_floor_at_10x_steps", best4k > floor_mean,
      "best vector arm at 4000 steps = %.4f vs last_two_ops floor %.4f (headroom %+.4f). RED "
      "means even 10x the matched budget leaves the vector side below a closed-form floor." %
      (best4k, floor_mean, best4k - floor_mean))

    e2_closes = Dg[4000]["acc_mean"] >= 0.90
    T("e2_recurrence_alone_closes_the_gap", e2_closes,
      "Diagonal (commutative) per-verb gate inside (c)'s exact recurrence, numel=%d d=%d: "
      "acc@400=%.4f (std %.4f) acc@4000=%.4f (std %.4f), train_loss@4000=%.4f, vs floor %.4f "
      "and vs (c)=0.9570. GREEN would mean the 0.957 is the RECURRENCE, not the operator; RED "
      "means the non-commutative operator is doing the work." %
      (n_diag, d_diag, Dg[400]["acc_mean"], Dg[400]["acc_std"], Dg[4000]["acc_mean"],
       Dg[4000]["acc_std"], Dg[4000]["train_loss_mean"], floor_mean))

    res = {"producer": PRODUCER, "command": COMMAND, "dtype": "float32",
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "seeds": list(SEEDS), "n_train": N_TRAIN, "n_eval": N_EVAL, "lr": LR,
           "weight_decay": WD, "optimizer": "AdamW", "full_batch": True,
           "twins_offset": AB.TWINS_OFFSET, "anchor_offset": AB.ANCHOR_OFFSET,
           "floor_last_two_ops_per_seed": floor_lt, "floor_last_two_ops_mean": floor_mean,
           "e1_a": A, "e1_b": B, "e2_diag": Dg, "e2_diag_numel": n_diag, "e2_diag_d": d_diag,
           "rows": rows, "wall_seconds": time.time() - t_all}
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    reds = int(not e1_converged) + int(best4k <= floor_mean) + int(not e2_closes)
    log({"t": "done", "agent": "Foreman", "checks": 3, "reds": reds,
         "text": "checker probe: (a)@400=%.4f (a)@4000=%.4f (b)@400=%.4f (b)@4000=%.4f "
                 "diag_recurrent@400=%.4f @4000=%.4f (numel=%d) floor=%.4f (c)=0.9570. results at %s" %
                 (A[400]["acc_mean"], A[4000]["acc_mean"], B[400]["acc_mean"], B[4000]["acc_mean"],
                  Dg[400]["acc_mean"], Dg[4000]["acc_mean"], n_diag, floor_mean, OUT)})
    print(json.dumps({k: res[k] for k in ("e1_a", "e1_b", "e2_diag", "e2_diag_numel",
                                          "floor_last_two_ops_mean", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
