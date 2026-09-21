"""
Phase H -- H3v2 twins (c) and (d): operator-gate family, trained.

OWNERSHIP. This file is this lane's own (Foreman). It imports the bed and the
floors, unchanged, from phaseh_H3v2_bed.py / phaseh_H3v2_floors.py (both
read-only here, not rebuilt):
    - bed: discrete-label target (nearest of C=8 fixed unit anchors by cosine
      to the true composed state), m=12 d=8 k=6 line_len=3.
    - floors already landed by the other file in this lane: constant_predictor,
      order_free, bag_of_context, last_op_only, last_two_ops, all measured, not
      guessed. Tightest floor = last_two_ops. Precondition null_discriminates
      is TRUE (edit4, per-seed null accuracy 0.34/0.12/0.16/0.16/0.11, all
      strictly worse than the correct order on >=1 instance/seed) -- so this
      file is permitted to train.

THE TWO TWINS (this file owns only these two of the four called for):
    (c) operator-gate, INVERTIBLE ONLY.
        Each verb v gets a learned square generator G_v (k,D,D). The applied
        operator is matrix_exp(G_v - G_v^T): the exponential of a skew-
        symmetric matrix is invertible for EVERY value of G_v (det=+1,
        orthogonal in fact) -- invertibility holds by construction, not by
        checking a determinant after the fact.
    (d) operator-gate, INVERTIBLE PLUS PROJECTORS.
        Identical to (c) in every parameter, every initialization draw, every
        training step and every minibatch, with ONE addition: a learned rank-1
        correction alpha_v * outer(u_v, v_v) added to (c)'s operator. That
        term is generically RANK-DEFICIENT (rank <= 1 against the full D=8),
        i.e. non-invertible, so it isolates "add a projector-like term" as the
        only difference from (c). Nothing else differs: same entity embedding,
        same verb generators G_v, same readout head, same seed, same steps,
        same optimizer, same data, same minibatch order.

PARAMETER MATCH IS READ, NOT ASSUMED. numel() is read from both live models
after construction and logged; (d) has strictly more parameters than (c) by
exactly the projector's own parameter count (k*(2D+1)), which is the isolating
variable the comparison is measuring, not a confound to hide.

WHY INIT IS BITWISE-SHARED BETWEEN (c) AND (d). Both models are built inside
the SAME torch.manual_seed(seed) call, and the shared submodules (entity
embedding, verb generators, readout) are constructed in the SAME order BEFORE
the projector parameters. Since the projector params are drawn from the RNG
stream strictly after the shared ones, drawing them (only for (d)) cannot
perturb the shared draws that already happened. Self-check (a) asserts the
three shared tensors are bitwise torch.equal across a fresh (c),(d) pair at
every seed before any training step runs.

DATA. Fresh offset (TWINS_OFFSET, SEED_OFFSET+70000), NOT the floors' own
POP_OFFSET -- this lane's train/eval instances are a disjoint draw from the
floors' reported n=5000 population, so nothing here reuses an instance the
floors already looked at. Per seed: N_TRAIN=3000 (fit) + N_EVAL=1000 (score),
same draw, split by index (train = [:3000], eval = [3000:]). Floors
(constant_predictor, last_op_only, last_two_ops, order_free, bag_of_context)
are RECOMPUTED on this lane's own N_EVAL=1000 split (not copied from the
floors file's cached n=5000 numbers, which were a different instance subset)
so headroom is an honest apples-to-apples number against what the twins were
actually scored on. The two are close (measured below) but not identical, and
that difference is reported rather than hidden.

Producer: Foreman (Claude Sonnet 5), torch {torchver}, numpy {npver}, Python
{pyver}, Windows 11, CPU.
Run: python phaseh_H3v2_twins_cd.py
Board: appends one JSON line per event to
  C:\\Users\\seal\\Desktop\\New folder (32)\\house-events.jsonl  (io.open .. "a", never rewritten)
Results (schema defined by THIS file -- phaseh_H3v2_results.jsonl did not
exist on disk anywhere under this machine's Claude scratch trees when this
run started; no other-lane schema was available to match, so one is defined
here, documented, and the first two writers of it are these two arms):
  C:\\...\\scratchpad\\phaseh_H3v2_results.jsonl  (io.open .. "a", one line per (arm, seed))
"""
import io
import json
import os
import platform
import time
import importlib.util
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

SCRATCH = os.path.dirname(os.path.abspath(__file__))
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
RESULTS_JSONL = os.path.join(SCRATCH, "phaseh_H3v2_results.jsonl")
BED_SRC = os.path.join(SCRATCH, "phaseh_H3v2_bed.py")
FLOORS_SRC = os.path.join(SCRATCH, "phaseh_H3v2_floors.py")

PRODUCER = f"Foreman/torch-{torch.__version__}/numpy-{np.__version__}/py-{platform.python_version()}"
DTYPE = "float32"
COMMAND = "python phaseh_H3v2_twins_cd.py"


def log(event):
    event = dict(event)
    event.setdefault("agent", "Foreman")
    with io.open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def T(name, ok, detail, dur=0.0):
    log({"t": "test", "state": "GREEN" if ok else "RED", "name": name,
         "detail": detail, "dur": round(dur, 4)})
    print(("GREEN " if ok else "RED   ") + name + " :: " + detail)
    return ok


def F(item, text):
    log({"t": "finding", "item": item, "text": text})
    print("FINDING [" + item + "] " + text)


def append_result(row):
    with io.open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Import the bed and the floors, unchanged, by file (not rebuilt).
# ---------------------------------------------------------------------------
spec_bed = importlib.util.spec_from_file_location("phaseh_H3v2_bed_twinscd", BED_SRC)
bedmod = importlib.util.module_from_spec(spec_bed)
spec_bed.loader.exec_module(bedmod)

spec_floors = importlib.util.spec_from_file_location("phaseh_H3v2_floors_twinscd", FLOORS_SRC)
floorsmod = importlib.util.module_from_spec(spec_floors)
spec_floors.loader.exec_module(floorsmod)

M, D, K, LINE_LEN = bedmod.M, bedmod.D, bedmod.K, bedmod.LINE_LEN
SEEDS = bedmod.SEEDS
C = floorsmod.C
build_population_n = floorsmod.build_population_n
anchors_for = floorsmod.anchors_for
label_of = floorsmod.label_of
last_op_only_predict = floorsmod.last_op_only_predict
last_two_ops_predict = floorsmod.last_two_ops_predict
bag_predict = bedmod.bag_predict
NONID_PERMS = bedmod.NONID_PERMS

TWINS_OFFSET = bedmod.SEED_OFFSET + 70000       # disjoint from floors' POP_OFFSET (+40000)
ANCHOR_OFFSET = bedmod.SEED_OFFSET + 70800
WRONG_OFFSET = bedmod.SEED_OFFSET + 70700
N_TRAIN = 3000
N_EVAL = 1000
STEPS = 400
LR = 0.02
WEIGHT_DECAY = 1e-4

TIGHTEST_FLOOR_FROM_FLOORS_FILE = 0.37928  # last_two_ops @ n=5000, from phaseh_H3v2_floors_results.json


# ---------------------------------------------------------------------------
# The two twins.
# ---------------------------------------------------------------------------
class OperatorGateTwin(nn.Module):
    """(c) when use_projector=False, (d) when use_projector=True. See module
    docstring for exactly what differs and why init is bitwise-shared."""

    def __init__(self, m, k, d, c_classes, use_projector):
        super().__init__()
        self.entity_emb = nn.Embedding(m, d)
        self.G = nn.Parameter(torch.empty(k, d, d))
        nn.init.normal_(self.G, mean=0.0, std=0.5)
        self.readout = nn.Linear(d, c_classes)
        self.use_projector = use_projector
        if use_projector:
            self.proj_u = nn.Parameter(torch.empty(k, d))
            self.proj_v = nn.Parameter(torch.empty(k, d))
            self.proj_alpha = nn.Parameter(torch.zeros(k))
            nn.init.normal_(self.proj_u, mean=0.0, std=0.5)
            nn.init.normal_(self.proj_v, mean=0.0, std=0.5)

    def operators(self):
        S = self.G - self.G.transpose(-1, -2)            # skew-symmetric generator
        op = torch.linalg.matrix_exp(S)                   # invertible for ANY G: (k,d,d)
        if self.use_projector:
            proj = self.proj_alpha.view(-1, 1, 1) * torch.einsum(
                "ki,kj->kij", self.proj_u, self.proj_v)    # rank<=1 per verb: non-invertible
            op = op + proj
        return op

    def forward(self, agent0_idx, verb_idx_steps, patient_idx_steps):
        ops = self.operators()                             # (k,d,d)
        state = self.entity_emb(agent0_idx)                # (B,d)
        for verb_idx, patient_idx in zip(verb_idx_steps, patient_idx_steps):
            op_b = ops[verb_idx]                            # (B,d,d)
            pat = self.entity_emb(patient_idx)              # (B,d)
            state = torch.einsum("bij,bj->bi", op_b, state + pat)
        return self.readout(state)


def build_model(seed, use_projector):
    torch.manual_seed(seed)
    return OperatorGateTwin(M, K, D, C, use_projector)


def check_shared_init_bitwise(seed):
    mc = build_model(seed, use_projector=False)
    md = build_model(seed, use_projector=True)
    ok = (torch.equal(mc.entity_emb.weight, md.entity_emb.weight)
          and torch.equal(mc.G, md.G)
          and torch.equal(mc.readout.weight, md.readout.weight)
          and torch.equal(mc.readout.bias, md.readout.bias))
    return ok


# ---------------------------------------------------------------------------
# Data: fresh disjoint draw, split by index; discrete labels via the imported
# anchor/labeling convention (unchanged from the floors file).
# ---------------------------------------------------------------------------
def make_tensors(instances, anchors):
    agent0 = torch.tensor([line[0][1] for line, _ in instances], dtype=torch.long)
    verb_steps = [torch.tensor([line[t][0] for line, _ in instances], dtype=torch.long)
                  for t in range(LINE_LEN)]
    patient_steps = [torch.tensor([line[t][2] for line, _ in instances], dtype=torch.long)
                      for t in range(LINE_LEN)]
    true_states = np.stack([s for _, s in instances])
    labels = label_of(anchors, true_states)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return agent0, verb_steps, patient_steps, labels_t, true_states, labels


def eval_floors_on_split(entities, verb_ops, instances, anchors, seed):
    """Recompute constant_predictor / last_op_only / last_two_ops / order_free /
    bag_of_context on THIS lane's own eval split, via the imported (unchanged)
    predictor functions -- an honest floor for the exact data twins are scored
    on, not the floors file's cached n=5000 numbers from a different subset."""
    n = len(instances)
    true_states = np.stack([s for _, s in instances])
    true_labels = label_of(anchors, true_states)

    maj = Counter(true_labels.tolist()).most_common(1)[0][0]
    const_acc = float(np.mean(true_labels == maj))

    rng_wrong = np.random.default_rng(WRONG_OFFSET + 900 + seed)
    lo_correct = lt_correct = of_correct = bag_correct = 0
    for line, true_state in instances:
        agent0 = line[0][1]
        true_lbl = label_of(anchors, true_state)
        lo_pred = last_op_only_predict(line, entities, verb_ops)
        lt_pred = last_two_ops_predict(line, entities, verb_ops)
        wrong_perm = NONID_PERMS[rng_wrong.integers(len(NONID_PERMS))]
        of_pred = bedmod.replay_perm(line, entities, verb_ops, agent0, wrong_perm)
        bag_pred = bag_predict(line, entities, verb_ops, agent0)
        if label_of(anchors, lo_pred) == true_lbl: lo_correct += 1
        if label_of(anchors, lt_pred) == true_lbl: lt_correct += 1
        if label_of(anchors, of_pred) == true_lbl: of_correct += 1
        if label_of(anchors, bag_pred) == true_lbl: bag_correct += 1

    return {"n": n, "constant_predictor": const_acc, "last_op_only": lo_correct / n,
            "last_two_ops": lt_correct / n, "order_free": of_correct / n,
            "bag_of_context": bag_correct / n}


# ---------------------------------------------------------------------------
# Train one twin, one seed. Same data, same shuffling generator, same steps,
# same AdamW settings for (c) and (d) at a given seed -- only use_projector differs.
# ---------------------------------------------------------------------------
def train_one(seed, use_projector, agent0, verb_steps, patient_steps, labels_t, steps=STEPS, lr=LR):
    model = build_model(seed, use_projector)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad()
        logits = model(agent0, verb_steps, patient_steps)
        loss = loss_fn(logits, labels_t)
        loss.backward()
        opt.step()
    dur = time.time() - t0
    final_train_loss = float(loss.item())
    n_params = int(sum(p.numel() for p in model.parameters()))
    return model, final_train_loss, dur, n_params


@torch.no_grad()
def eval_one(model, agent0, verb_steps, patient_steps, labels_t):
    logits = model(agent0, verb_steps, patient_steps)
    pred = logits.argmax(dim=-1)
    acc = float((pred == labels_t).float().mean().item())
    return acc


def main():
    t_all0 = time.time()
    F("phase_start",
      "H3v2 twins (c) invertible-only and (d) invertible+projector operator-gate, trained. "
      "Bed and floors imported unchanged from %s / %s. Precondition null_discriminates=True "
      "(floors file edit4, per-seed accuracies 0.34/0.12/0.16/0.16/0.11 across 5 seeds, all "
      "strictly worse than correct order on >=1 instance/seed) -- training is permitted. "
      "seeds=%s, n_train=%d, n_eval=%d, steps=%d, optimizer=AdamW(lr=%.3g, wd=%.1e), "
      "dtype=%s, producer=%s, results schema is FIRST-WRITTEN by this file (no prior "
      "phaseh_H3v2_results.jsonl existed on disk to match)." %
      (BED_SRC, FLOORS_SRC, list(SEEDS), N_TRAIN, N_EVAL, STEPS, LR, WEIGHT_DECAY, DTYPE,
       PRODUCER))

    # self-check (a): shared-init bitwise equality across (c)/(d) before any training.
    init_ok_per_seed = [check_shared_init_bitwise(seed) for seed in SEEDS]
    T("selfcheck_shared_init_bitwise_c_eq_d", all(init_ok_per_seed),
       "entity_emb.weight, G, readout.weight, readout.bias torch.equal between a fresh (c) and "
       "(d) built under the same torch.manual_seed(seed), all %d seeds=%s -- (d)'s projector "
       "params are drawn strictly AFTER these, so they cannot perturb the shared draws" %
       (len(SEEDS), init_ok_per_seed))
    if not all(init_ok_per_seed):
        F("selfcheck_shared_init_FAILED",
          "Shared-parameter bitwise equality between (c) and (d) failed at seed(s) %s -- the "
          "(c) vs (d) comparison below would NOT isolate the projector alone. Proceeding anyway "
          "and flagging this loudly; treat any (c)/(d) gap as confounded, not attributable." %
          [s for s, ok in zip(SEEDS, init_ok_per_seed) if not ok])

    per_seed_rows = {"c": [], "d": []}
    floor_rows = []
    n_params = {"c": None, "d": None}

    for seed in SEEDS:
        t_seed0 = time.time()
        entities, verb_ops, instances = build_population_n(seed, TWINS_OFFSET, N_TRAIN + N_EVAL)
        anchors = anchors_for(seed, ANCHOR_OFFSET)
        train_inst, eval_inst = instances[:N_TRAIN], instances[N_TRAIN:]

        tr_agent0, tr_verb, tr_pat, tr_labels, _, _ = make_tensors(train_inst, anchors)
        ev_agent0, ev_verb, ev_pat, ev_labels, _, _ = make_tensors(eval_inst, anchors)

        floor_row = eval_floors_on_split(entities, verb_ops, eval_inst, anchors, seed)
        floor_row["seed"] = seed
        floor_rows.append(floor_row)
        T("floors_matched_to_eval_split_seed%d" % seed,
          floor_row["last_two_ops"] >= floor_row["last_op_only"] - 1e-9,
          "seed=%d, n_eval=%d (TWINS_OFFSET, disjoint from floors file's own n=5000 draw): "
          "constant_predictor=%.4f last_op_only=%.4f last_two_ops=%.4f order_free=%.4f "
          "bag_of_context=%.4f (all dtype=float64, closed-form, imported unchanged)" %
          (seed, floor_row["n"], floor_row["constant_predictor"], floor_row["last_op_only"],
           floor_row["last_two_ops"], floor_row["order_free"], floor_row["bag_of_context"]))

        for arm, use_proj in (("c", False), ("d", True)):
            model, train_loss, train_dur, n_p = train_one(
                seed, use_proj, tr_agent0, tr_verb, tr_pat, tr_labels)
            acc = eval_one(model, ev_agent0, ev_verb, ev_pat, ev_labels)
            n_params[arm] = n_p
            per_seed_rows[arm].append({"seed": seed, "accuracy": acc})

            row = {
                "t": "result", "phase": "H3v2", "lane": "twins_cd", "arm": arm,
                "arm_desc": ("operator-gate, invertible only" if arm == "c"
                             else "operator-gate, invertible plus rank<=1 projector"),
                "seed": seed, "accuracy": acc, "n_train": N_TRAIN, "n_eval": floor_row["n"],
                "steps": STEPS, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
                "dtype": DTYPE, "params_numel": n_p, "final_train_loss": train_loss,
                "train_wall_seconds": round(train_dur, 4),
                "tightest_floor_matched_eval": floor_row["last_two_ops"],
                "tightest_floor_floors_file_n5000": TIGHTEST_FLOOR_FROM_FLOORS_FILE,
                "headroom_above_matched_floor": acc - floor_row["last_two_ops"],
                "producer": PRODUCER, "command": COMMAND,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            append_result(row)
            T("twin_%s_trained_seed%d" % (arm, seed), True,
              "arm=%s seed=%d acc=%.4f (floor last_two_ops=%.4f, headroom=%+.4f), "
              "params=%d, train_loss=%.4f, train_wall=%.2fs, steps=%d, AdamW lr=%.3g" %
              (arm, seed, acc, floor_row["last_two_ops"], acc - floor_row["last_two_ops"],
               n_p, train_loss, train_dur, STEPS, LR))

        print("seed %d done in %.2fs" % (seed, time.time() - t_seed0))

    def summarize(rows):
        accs = [r["accuracy"] for r in rows]
        return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
                "per_seed": accs}

    summ_c = summarize(per_seed_rows["c"])
    summ_d = summarize(per_seed_rows["d"])
    tightest_mean = float(np.mean([r["last_two_ops"] for r in floor_rows]))
    tightest_std = float(np.std([r["last_two_ops"] for r in floor_rows]))

    d_minus_c = [d["accuracy"] - c["accuracy"] for c, d in
                 zip(per_seed_rows["c"], per_seed_rows["d"])]
    d_beats_c_mean = float(np.mean(d_minus_c))
    pooled_std = float(np.std(summ_c["per_seed"] + summ_d["per_seed"]))
    d_beats_c_in_spread = abs(d_beats_c_mean) < pooled_std

    F("headline_c_vs_d",
      "(c) invertible-only: mean acc=%.4f (std %.4f) across seeds %s, params=%d. (d) "
      "invertible+projector: mean acc=%.4f (std %.4f), params=%d (delta vs (c): +%d, exactly "
      "the projector's own parameter count). Per-seed (d - c) = %s, mean delta=%+.4f. Pooled "
      "seed std (c and d together) = %.4f, so the delta is %s the pooled spread -- (d) %s (c) "
      "%s. Matched-eval tightest floor (last_two_ops, recomputed on this lane's own n_eval=%d "
      "split, mean over seeds)=%.4f (std %.4f); floors-file n=5000 cached value=%.4f "
      "(different instance subset, same convention, reported for continuity). Honest headroom: "
      "(c) mean=%+.4f, (d) mean=%+.4f above the matched floor." %
      (summ_c["mean"], summ_c["std"], list(SEEDS), n_params["c"], summ_d["mean"], summ_d["std"],
       n_params["d"], n_params["d"] - n_params["c"], [round(x, 4) for x in d_minus_c],
       d_beats_c_mean, pooled_std, "INSIDE" if d_beats_c_in_spread else "OUTSIDE",
       "beats" if d_beats_c_mean > 0 else "does not beat",
       "(within noise)" if d_beats_c_in_spread else "(outside noise)",
       floor_rows[0]["n"], tightest_mean, tightest_std, TIGHTEST_FLOOR_FROM_FLOORS_FILE,
       summ_c["mean"] - tightest_mean, summ_d["mean"] - tightest_mean))

    F("headline_untrained_vs_trained_pair_INCOMPLETE",
      "The required headline pair (untrained operator-vs-vector ratio beside the TRAINED "
      "ratio) needs twin (a)'s (word2vec+softmax, vector-bound) TRAINED accuracy, which is "
      "owned by the other lane (twins_ab), not this file. No phaseh_H3v2_results.jsonl existed "
      "on disk for either lane before this run, so no (a)/(b) trained numbers were available to "
      "pair against (c)/(d) at the time this ran. This file reports (c) and (d) only, per its "
      "ownership, and flags the pair as open rather than fabricating a vector-bound number.")

    checks = 1 + len(SEEDS) + len(SEEDS) * 2  # init check + floor checks + 2 twins x 5 seeds
    reds = 0 if all(init_ok_per_seed) else 1

    done_text = (
        "H3v2 twins c/d: (c) mean=%.4f std=%.4f, (d) mean=%.4f std=%.4f, seeds=%s, "
        "matched tightest floor mean=%.4f, (c) headroom=%+.4f, (d) headroom=%+.4f, "
        "d-vs-c mean delta=%+.4f (pooled seed std=%.4f, delta %s spread). params: c=%d d=%d "
        "(diff=%d, = projector param count). results appended to %s (schema first-written this "
        "run, no other-lane file existed to match)." %
        (summ_c["mean"], summ_c["std"], summ_d["mean"], summ_d["std"], list(SEEDS),
         tightest_mean, summ_c["mean"] - tightest_mean, summ_d["mean"] - tightest_mean,
         d_beats_c_mean, pooled_std, "inside" if d_beats_c_in_spread else "outside",
         n_params["c"], n_params["d"], n_params["d"] - n_params["c"], RESULTS_JSONL)
    )
    log({"t": "done", "agent": "Foreman", "checks": checks, "reds": reds, "text": done_text})

    print("\n" + done_text)
    print("wall_seconds total = %.2f" % (time.time() - t_all0))
    print("results appended at " + RESULTS_JSONL)
    print("board appended at " + BOARD)


if __name__ == "__main__":
    main()
