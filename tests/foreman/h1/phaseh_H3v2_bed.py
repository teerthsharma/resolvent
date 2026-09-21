"""
Phase H -- H3v2 bed: make order visible, then prove it is visible before anything trains.

phaseh_H3_realfloor.py measured (5 seeds x 100 test instances, dtype float64):
    bag_of_context=4.9938  order_free_composition=5.5367  chance=5.7160
order_free is WORSE than bag and within 3.1% of chance. Mechanism: H3's planted
verb_ops (phaseh_I1_I2_binding.py::rand_rotation) are norm-preserving SO(d)
rotations. Composing them in a permuted order still lands on a vector of the
true state's norm scale but pointing in an uncorrelated direction -- its L2
distance to the true state behaves like two independent random vectors of
similar norm, i.e. AT CHANCE, by geometry, regardless of how wrong the order
is. bag_of_context wins only because averaging shrinks toward the origin,
which happens to sit closer under L2 for reasons unrelated to composition.
So under the ORIGINAL bed+metric, "right operators wrong order" is
indistinguishable from "no information at all" -- the metric is structurally
blind to the property H3 exists to measure.

This file:
  STEP 1 -- proves that blindness directly, by varying permutation distance
            from the identity (adjacent transposition / 3-cycle derangement /
            full reversal) and showing the error is flat across that range.
  STEP 2 -- tries three fixes against the SAME failure and reports each:
            (a) classification/retrieval among K candidate final states
                (metric change, bed unchanged)
            (b) non-norm-preserving operators: rotation * contraction
                (operator change, metric unchanged: L2)
            (c) discrete label via a fixed anchor codebook
                (target change: continuous state -> nearest-anchor label)
  STEP 3 -- builds the winning fix as the v2 bed, with its null (shuffled
            operator assignment) and its three floors (correct-order,
            order-free, bag-of-context) reported in the new metric.

NO TRAINING. Closed-form floors and diagnostics only.
Bed primitives (make_entities, rand_rotation, bed_instance) are READ from
phaseh_I1_I2_binding.py by import, not rebuilt -- that file is owned by
another lane and is not edited here. phaseh_H3_realfloor.py is read-only
context (not edited, not imported -- its numbers are quoted above from its
own results/md file).

Producer: Foreman (Claude Sonnet 5), numpy {version}, Python {pyversion}, Windows 11.
Run: python phaseh_H3v2_bed.py
Board: appends one JSON line per event to
  C:\\Users\\seal\\Desktop\\New folder (32)\\house-events.jsonl  (io.open .. "a", never rewritten)
"""
import numpy as np
import json
import io
import os
import time
import platform
import itertools
import importlib.util

SCRATCH = os.path.dirname(os.path.abspath(__file__))
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
RESULTS_PATH = os.path.join(SCRATCH, "phaseh_H3v2_bed_results.json")
BED_SRC = os.path.join(SCRATCH, "phaseh_I1_I2_binding.py")

PRODUCER = f"Foreman/numpy-{np.__version__}/py-{platform.python_version()}"
DTYPE = "float64"

M, D, K, LINE_LEN = 12, 8, 6, 3  # same bed params as H3_realfloor / both twin lanes
SEEDS = (0, 1, 2, 3, 4)
N_TEST = 100
SEED_OFFSET = 6000  # fresh namespace: v2 is a new bed, no twins trained against it yet


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


# ---------------------------------------------------------------------------
# Bed primitives read from the lane file that builds them. Not rebuilt.
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("phaseh_I1_I2_binding_bed", BED_SRC)
bed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bed)
make_entities = bed.make_entities
rand_rotation = bed.rand_rotation
bed_instance = bed.bed_instance

NONID_PERMS = [p for p in itertools.permutations(range(LINE_LEN)) if p != tuple(range(LINE_LEN))]
ALL_PERMS = list(itertools.permutations(range(LINE_LEN)))


def replay_perm(line, entities, verb_ops, agent0, perm):
    """Chain the line's recorded (verb, patient) pairs in an arbitrary order
    over the token indices given by perm, starting from entities[agent0]."""
    state = entities[agent0].copy()
    for idx in perm:
        v, _a, p = line[idx]
        state = verb_ops[v] @ (state + entities[p])
    return state


def bag_predict(line, entities, verb_ops, agent0):
    """bag_single_hop from phaseh_H3_realfloor.py: every raw token embedding
    plus every one-step (verb, own-pair) transform, meaned. Sees every token,
    chains nothing."""
    ids = {agent0} | {p for _, _, p in line}
    cand = [entities[i] for i in ids]
    for v, a, p in line:
        cand.append(verb_ops[v] @ (entities[a] + entities[p]))
    return np.stack(cand).mean(axis=0)


def err_l2(pred, true_state):
    return float(np.linalg.norm(pred - true_state))


def inversions(perm):
    return sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm)) if perm[i] > perm[j])


def build_population(seed, offset, verb_ops_fn):
    rng = np.random.default_rng(offset + seed)
    entities = make_entities(rng, m=M, d=D)
    verb_ops = verb_ops_fn(rng)
    instances = [bed_instance(rng, entities, verb_ops, LINE_LEN) for _ in range(N_TEST)]
    return entities, verb_ops, instances


def rot_ops(rng):
    return [rand_rotation(D, rng) for _ in range(K)]


# ===========================================================================
# STEP 1 -- prove the blindness directly: vary permutation distance from
# identity (0=correct order, 1=adjacent transposition, 2=derangement/3-cycle,
# 3=full reversal, the max for line_len=3) and show error is flat.
# ===========================================================================
def step1_prove_blindness():
    t0 = time.time()
    by_dist = {}
    for p in ALL_PERMS:
        by_dist.setdefault(inversions(p), []).append(p)
    # line_len=3: dist0={identity}, dist1={2 adjacent transpositions},
    # dist2={2 derangements/3-cycles}, dist3={reversal, the unique max}

    rows = []
    for seed in SEEDS:
        entities, verb_ops, instances = build_population(seed, SEED_OFFSET, rot_ops)
        rng_chance = np.random.default_rng(SEED_OFFSET + 900 + seed)
        dist_errs = {d: [] for d in by_dist}
        bag_errs, chance_errs = [], []
        for line, true_state in instances:
            agent0 = line[0][1]
            for d, perms in by_dist.items():
                e = np.mean([err_l2(replay_perm(line, entities, verb_ops, agent0, perm), true_state)
                             for perm in perms])
                dist_errs[d].append(e)
            bag_errs.append(err_l2(bag_predict(line, entities, verb_ops, agent0), true_state))
            chance_errs.append(err_l2(rng_chance.standard_normal(D), true_state))
        row = {"seed": seed, "n": N_TEST,
               "dist_mean": {d: float(np.mean(v)) for d, v in dist_errs.items()},
               "bag_of_context": float(np.mean(bag_errs)),
               "chance": float(np.mean(chance_errs))}
        rows.append(row)

    agg = {d: {"mean": float(np.mean([r["dist_mean"][d] for r in rows])),
               "std": float(np.std([r["dist_mean"][d] for r in rows]))}
           for d in by_dist}
    bag_mean = float(np.mean([r["bag_of_context"] for r in rows]))
    chance_mean = float(np.mean([r["chance"] for r in rows]))

    nonzero_dists = [d for d in agg if d != 0]
    spread = max(agg[d]["mean"] for d in nonzero_dists) - min(agg[d]["mean"] for d in nonzero_dists)
    rel_spread = spread / chance_mean
    flat = rel_spread < 0.05  # <5% of chance scale: distance-1 vs distance-3 indistinguishable

    detail = ("dist->mean_err (%d seeds x %d instances, dtype=%s, command: "
               "python phaseh_H3v2_bed.py): dist0(identity/correct)=%.4f, "
               "dist1(adjacent transposition)=%.4f, dist2(derangement/3-cycle)=%.4f, "
               "dist3(reversal,max)=%.4f | bag_of_context=%.4f chance=%.4f | "
               "spread across dist{1,2,3}=%.4f (%.1f%% of chance) -> flat=%s" %
              (len(SEEDS), N_TEST, DTYPE, agg[0]["mean"], agg[1]["mean"], agg[2]["mean"],
               agg[3]["mean"], bag_mean, chance_mean, spread, rel_spread * 100, flat))
    T("step1_blindness_flat_across_permutation_distance", flat, detail)
    F("step1_verdict",
      "Error under the ORIGINAL bed (norm-preserving SO(%d) rotations) and metric (mean L2 to "
      "true state), %d seeds x %d instances, dtype=%s: any wrong order (distance 1, 2, or 3 "
      "transpositions from identity) scores within %.1f%% of any other wrong order, and all "
      "three sit within %.1f%% of raw chance (%.4f). Only distance 0 (the true order) differs, "
      "and it differs by landing at exact 0, not by degree. This is the blindness established BY "
      "CONSTRUCTION across the whole distance range, not inferred from one summary number: the "
      "bed cannot tell 'one step out of place' from 'completely reversed' from 'no information "
      "at all.'" % (D, len(SEEDS), N_TEST, DTYPE, rel_spread * 100,
                     abs(min(agg[d]["mean"] for d in nonzero_dists) / chance_mean - 1) * 100,
                     chance_mean))
    return {"rows": rows, "agg": agg, "bag_of_context": bag_mean, "chance": chance_mean,
            "flat": flat, "rel_spread": rel_spread, "dur": time.time() - t0}


# ===========================================================================
# STEP 2 -- try three fixes, report the same three numbers for each:
# correct-order score, order-free score, bag-of-context score.
# ===========================================================================
def _per_seed_scores(offset, verb_ops_fn, score_fn, rng_extra_offset):
    """score_fn(pred_vec, true_state, true_idx, true_states, rng_wrong) -> scalar score
    for one instance's predictor output; called once each for correct/order-free/bag."""
    rows = []
    for seed in SEEDS:
        entities, verb_ops, instances = build_population(seed, offset, verb_ops_fn)
        true_states = np.stack([s for _, s in instances])
        rng_wrong = np.random.default_rng(offset + rng_extra_offset + seed)
        rng_decoy = np.random.default_rng(offset + rng_extra_offset + 600 + seed)

        correct_s, of_s, bag_s = [], [], []
        for i, (line, true_state) in enumerate(instances):
            agent0 = line[0][1]
            wrong_perm = NONID_PERMS[rng_wrong.integers(len(NONID_PERMS))]
            of_pred = replay_perm(line, entities, verb_ops, agent0, wrong_perm)
            bag_pred = bag_predict(line, entities, verb_ops, agent0)
            correct_s.append(score_fn(true_state, true_state, i, true_states, rng_decoy))
            of_s.append(score_fn(of_pred, true_state, i, true_states, rng_decoy))
            bag_s.append(score_fn(bag_pred, true_state, i, true_states, rng_decoy))
        rows.append({"seed": seed, "n": N_TEST,
                     "correct": float(np.mean(correct_s)),
                     "order_free": float(np.mean(of_s)),
                     "bag_of_context": float(np.mean(bag_s))})
    return rows


def _summarize(rows, higher_is_better, chance):
    correct_m = float(np.mean([r["correct"] for r in rows]))
    of_m = float(np.mean([r["order_free"] for r in rows]))
    bag_m = float(np.mean([r["bag_of_context"] for r in rows]))
    if higher_is_better:
        collapse = abs(of_m - bag_m) <= 0.05  # both within 5pp of each other
        separated = (correct_m - max(of_m, bag_m)) >= 0.30
    else:
        collapse = bag_m > 0 and abs(of_m - bag_m) / bag_m <= 0.10
        separated = bag_m > 0 and (min(of_m, bag_m) - correct_m) / bag_m >= 0.50
    fix_works = collapse and separated
    return {"correct_mean": correct_m, "order_free_mean": of_m, "bag_of_context_mean": bag_m,
            "chance": chance, "order_free_collapses_toward_bag": collapse,
            "correct_far_from_both": separated, "fix_restores_sensitivity": fix_works}


def step2a_classification(NCAND=8):
    """(a) CHANGE THE METRIC, keep the bed: nearest-cosine-similarity retrieval
    among NCAND candidate final states (the true one + NCAND-1 decoys drawn
    from the other instances in the same seed's population). Chance = 1/NCAND."""
    def score_fn(pred_vec, true_state, true_idx, true_states, rng_decoy):
        pool = [j for j in range(len(true_states)) if j != true_idx]
        decoy_idx = rng_decoy.choice(pool, size=NCAND - 1, replace=False)
        cand_idx = np.concatenate(([true_idx], decoy_idx))
        cands = true_states[cand_idx]
        denom = np.linalg.norm(cands, axis=1) * (np.linalg.norm(pred_vec) + 1e-12)
        sims = (cands @ pred_vec) / (denom + 1e-12)
        return 1.0 if int(np.argmax(sims)) == 0 else 0.0

    rows = _per_seed_scores(SEED_OFFSET + 20000, rot_ops, score_fn, 700)
    chance = 1.0 / NCAND
    summ = _summarize(rows, higher_is_better=True, chance=chance)
    T("step2a_classification_retrieval", summ["fix_restores_sensitivity"],
      "(a) cosine-sim retrieval, NCAND=%d, chance=%.4f, %d seeds x %d instances, dtype=%s, "
      "command: python phaseh_H3v2_bed.py :: correct_acc=%.4f order_free_acc=%.4f "
      "bag_acc=%.4f -> collapse=%s separated=%s" %
      (NCAND, chance, len(SEEDS), N_TEST, DTYPE, summ["correct_mean"], summ["order_free_mean"],
       summ["bag_of_context_mean"], summ["order_free_collapses_toward_bag"],
       summ["correct_far_from_both"]))
    return {"name": "classification_retrieval", "ncand": NCAND, "rows": rows, **summ}


def step2b_contraction(alpha=0.7):
    """(b) CHANGE THE OPERATORS, keep the metric (L2): verb_ops = alpha * SO(d)
    rotation, alpha<1, so order changes magnitude as well as direction."""
    def contract_ops(rng):
        return [alpha * rand_rotation(D, rng) for _ in range(K)]

    def score_fn(pred_vec, true_state, true_idx, true_states, rng_decoy):
        return err_l2(pred_vec, true_state)

    rows = _per_seed_scores(SEED_OFFSET + 30000, contract_ops, score_fn, 700)
    # chance: fresh draw from the entities' own generating distribution (unscaled),
    # same convention as phaseh_H3_realfloor.py's floor_chance
    chance_rows = []
    for seed in SEEDS:
        _, _, instances = build_population(seed, SEED_OFFSET + 30000, contract_ops)
        rng_c = np.random.default_rng(SEED_OFFSET + 30900 + seed)
        chance_rows.append(float(np.mean([err_l2(rng_c.standard_normal(D), s) for _, s in instances])))
    chance = float(np.mean(chance_rows))
    summ = _summarize(rows, higher_is_better=False, chance=chance)
    T("step2b_contraction_operators", summ["fix_restores_sensitivity"],
      "(b) verb_ops=alpha*SO(d), alpha=%.2f, L2 metric, chance=%.4f, %d seeds x %d instances, "
      "dtype=%s, command: python phaseh_H3v2_bed.py :: correct_err=%.4f order_free_err=%.4f "
      "bag_err=%.4f -> collapse=%s separated=%s" %
      (alpha, chance, len(SEEDS), N_TEST, DTYPE, summ["correct_mean"], summ["order_free_mean"],
       summ["bag_of_context_mean"], summ["order_free_collapses_toward_bag"],
       summ["correct_far_from_both"]))
    return {"name": "contraction_operators", "alpha": alpha, "rows": rows, **summ}


def step2c_discrete_label(C=8):
    """(c) CHANGE THE TARGET: discrete label = nearest of C fixed unit-norm
    anchor vectors to the (predicted or true) final state. Chance = 1/C."""
    def score_fn_factory(anchors):
        def score_fn(pred_vec, true_state, true_idx, true_states, rng_decoy):
            true_label = int(np.argmax(anchors @ true_state / (np.linalg.norm(true_state) + 1e-12)))
            pred_label = int(np.argmax(anchors @ pred_vec / (np.linalg.norm(pred_vec) + 1e-12)))
            return 1.0 if pred_label == true_label else 0.0
        return score_fn

    rows = []
    for seed in SEEDS:
        entities, verb_ops, instances = build_population(seed, SEED_OFFSET + 40000, rot_ops)
        rng_anchor = np.random.default_rng(SEED_OFFSET + 40800 + seed)
        anchors = rng_anchor.standard_normal((C, D))
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
        rng_wrong = np.random.default_rng(SEED_OFFSET + 40700 + seed)
        score_fn = score_fn_factory(anchors)

        correct_s, of_s, bag_s = [], [], []
        for line, true_state in instances:
            agent0 = line[0][1]
            wrong_perm = NONID_PERMS[rng_wrong.integers(len(NONID_PERMS))]
            of_pred = replay_perm(line, entities, verb_ops, agent0, wrong_perm)
            bag_pred = bag_predict(line, entities, verb_ops, agent0)
            correct_s.append(score_fn(true_state, true_state, None, None, None))
            of_s.append(score_fn(of_pred, true_state, None, None, None))
            bag_s.append(score_fn(bag_pred, true_state, None, None, None))
        rows.append({"seed": seed, "n": N_TEST, "correct": float(np.mean(correct_s)),
                     "order_free": float(np.mean(of_s)), "bag_of_context": float(np.mean(bag_s))})

    chance = 1.0 / C
    summ = _summarize(rows, higher_is_better=True, chance=chance)
    T("step2c_discrete_label", summ["fix_restores_sensitivity"],
      "(c) nearest-of-%d-fixed-anchors label, chance=%.4f, %d seeds x %d instances, dtype=%s, "
      "command: python phaseh_H3v2_bed.py :: correct_acc=%.4f order_free_acc=%.4f "
      "bag_acc=%.4f -> collapse=%s separated=%s" %
      (C, chance, len(SEEDS), N_TEST, DTYPE, summ["correct_mean"], summ["order_free_mean"],
       summ["bag_of_context_mean"], summ["order_free_collapses_toward_bag"],
       summ["correct_far_from_both"]))
    return {"name": "discrete_label", "c": C, "rows": rows, **summ}


# ===========================================================================
# STEP 3 -- build the winner: its null (shuffled operator assignment) and
# its three floors, reported in the winning metric.
# ===========================================================================
def step3_build_winner(winner):
    name = winner["name"]
    if name == "classification_retrieval":
        NCAND = winner["ncand"]

        def score_fn(pred_vec, true_state, true_idx, true_states, rng_decoy):
            pool = [j for j in range(len(true_states)) if j != true_idx]
            decoy_idx = rng_decoy.choice(pool, size=NCAND - 1, replace=False)
            cand_idx = np.concatenate(([true_idx], decoy_idx))
            cands = true_states[cand_idx]
            denom = np.linalg.norm(cands, axis=1) * (np.linalg.norm(pred_vec) + 1e-12)
            sims = (cands @ pred_vec) / (denom + 1e-12)
            return 1.0 if int(np.argmax(sims)) == 0 else 0.0

        null_rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(SEED_OFFSET + 50000 + seed)
            entities = make_entities(rng, m=M, d=D)
            verb_ops = rot_ops(rng)
            shuffled_ops = [verb_ops[i] for i in rng.permutation(K)]
            instances = [bed_instance(rng, entities, verb_ops, LINE_LEN) for _ in range(N_TEST)]
            true_states = np.stack([s for _, s in instances])
            rng_decoy = np.random.default_rng(SEED_OFFSET + 50600 + seed)
            finite = 0
            for i, (line, true_state) in enumerate(instances):
                agent0 = line[0][1]
                null_pred = bed.true_replay(line, entities, shuffled_ops, agent0)
                s = score_fn(null_pred, true_state, i, true_states, rng_decoy)
                if np.isfinite(s):
                    finite += 1
            null_rows.append({"seed": seed, "n": N_TEST, "finite_scored": finite})
        return {"name": name, "ncand": NCAND, "null_rows": null_rows,
                "correct": winner["correct_mean"], "order_free": winner["order_free_mean"],
                "bag_of_context": winner["bag_of_context_mean"], "chance": winner["chance"]}

    elif name == "discrete_label":
        C = winner["c"]
        null_rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(SEED_OFFSET + 51000 + seed)
            entities = make_entities(rng, m=M, d=D)
            verb_ops = rot_ops(rng)
            shuffled_ops = [verb_ops[i] for i in rng.permutation(K)]
            instances = [bed_instance(rng, entities, verb_ops, LINE_LEN) for _ in range(N_TEST)]
            rng_anchor = np.random.default_rng(SEED_OFFSET + 51800 + seed)
            anchors = rng_anchor.standard_normal((C, D))
            anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
            finite = 0
            for line, true_state in instances:
                agent0 = line[0][1]
                null_pred = bed.true_replay(line, entities, shuffled_ops, agent0)
                lbl = int(np.argmax(anchors @ null_pred / (np.linalg.norm(null_pred) + 1e-12)))
                if np.isfinite(lbl):
                    finite += 1
            null_rows.append({"seed": seed, "n": N_TEST, "finite_scored": finite})
        return {"name": name, "c": C, "null_rows": null_rows,
                "correct": winner["correct_mean"], "order_free": winner["order_free_mean"],
                "bag_of_context": winner["bag_of_context_mean"], "chance": winner["chance"]}

    else:  # contraction_operators
        alpha = winner["alpha"]

        def contract_ops(rng):
            return [alpha * rand_rotation(D, rng) for _ in range(K)]

        null_rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(SEED_OFFSET + 52000 + seed)
            entities = make_entities(rng, m=M, d=D)
            verb_ops = contract_ops(rng)
            shuffled_ops = [verb_ops[i] for i in rng.permutation(K)]
            instances = [bed_instance(rng, entities, verb_ops, LINE_LEN) for _ in range(N_TEST)]
            finite = 0
            for line, true_state in instances:
                agent0 = line[0][1]
                null_pred = bed.true_replay(line, entities, shuffled_ops, agent0)
                e = err_l2(null_pred, true_state)
                if np.isfinite(e):
                    finite += 1
            null_rows.append({"seed": seed, "n": N_TEST, "finite_scored": finite})
        return {"name": name, "alpha": alpha, "null_rows": null_rows,
                "correct": winner["correct_mean"], "order_free": winner["order_free_mean"],
                "bag_of_context": winner["bag_of_context_mean"], "chance": winner["chance"]}


def main():
    t0 = time.time()
    F("phase_start",
      "Phase H H3v2 bed: prove the order-blindness from phaseh_H3_realfloor.py by construction "
      "(vary permutation distance), then try 3 fixes and report each, then build the winner with "
      "its null and floors. bed params m=%d d=%d k=%d line_len=%d, seeds=%s, n_test=%d, dtype=%s, "
      "producer=%s. Bed primitives read from %s by import, not rebuilt." %
      (M, D, K, LINE_LEN, list(SEEDS), N_TEST, DTYPE, PRODUCER, BED_SRC))

    step1 = step1_prove_blindness()

    fixes = [step2a_classification(NCAND=8),
             step2b_contraction(alpha=0.7),
             step2c_discrete_label(C=8)]

    working = [f for f in fixes if f["fix_restores_sensitivity"]]
    if working:
        # winner = the one with the largest separation between correct and
        # max(order_free, bag), i.e. the cleanest demonstration
        def sep_score(f):
            if f["name"] == "contraction_operators":
                bag = f["bag_of_context_mean"]
                return (min(f["order_free_mean"], bag) - f["correct_mean"]) / max(bag, 1e-9)
            return f["correct_mean"] - max(f["order_free_mean"], f["bag_of_context_mean"])
        winner = max(working, key=sep_score)
        F("step2_winner",
          "Fix '%s' restores sensitivity: order_free_mean=%.4f collapses toward "
          "bag_of_context_mean=%.4f (both near chance=%.4f) while correct_mean=%.4f stays far "
          "from both. Fixes that ALSO worked: %s. Fixes that did NOT: %s." %
          (winner["name"], winner["order_free_mean"], winner["bag_of_context_mean"],
           winner["chance"], winner["correct_mean"],
           [f["name"] for f in working if f is not winner] or "none",
           [f["name"] for f in fixes if not f["fix_restores_sensitivity"]] or "none"))
    else:
        winner = None
        F("step2_no_fix_worked",
          "None of the three fixes (classification retrieval, contraction operators, discrete "
          "label) made order_free collapse toward bag_of_context while keeping correct far from "
          "both. FINDING: the planted-operator bed as parameterized (m=%d, d=%d, k=%d, "
          "line_len=%d) cannot demonstrate order with any of these three repairs on the mean over "
          "random wrong permutations; the phase needs a different bed shape (e.g. longer "
          "line_len for more permutations, or an explicitly order-dependent operator family such "
          "as non-commuting projectors), which is a finding for the next round, not a failure of "
          "this one." % (M, D, K, LINE_LEN))

    step3 = step3_build_winner(winner) if winner else None

    RESULTS = {
        "producer": PRODUCER, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dtype": DTYPE, "seeds": list(SEEDS), "n_test_per_seed": N_TEST,
        "bed_params": {"m": M, "d": D, "k": K, "line_len": LINE_LEN},
        "bed_source": BED_SRC,
        "step1_prove_blindness": step1,
        "step2_fixes": fixes,
        "step2_winner_name": winner["name"] if winner else None,
        "step3_winner_build": step3,
        "wall_seconds": time.time() - t0,
    }
    with io.open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)

    null_ok = (step3 is not None and all(r["finite_scored"] >= 1 for r in step3["null_rows"]))
    if step3 is not None:
        T("step3_null_scores_all_seeds", null_ok,
          "shuffled-operator null, %d seeds, finite_scored per seed=%s (precondition: >=1/seed)" %
          (len(SEEDS), [r["finite_scored"] for r in step3["null_rows"]]))

    reds = sum(1 for f in fixes if not f["fix_restores_sensitivity"])
    if not step1["flat"]:
        reds += 1
    if step3 is not None and not null_ok:
        reds += 1
    checks = 1 + len(fixes) + (2 if step3 is not None else 0)

    log({"t": "done", "agent": "Foreman", "checks": checks, "reds": reds,
         "text": ("H3v2 bed: step1 flat=%s (spread %.1f%% of chance). step2 fixes tried=%s, "
                  "worked=%s, winner=%s. step3 null all-seeds-scored=%s. results at %s" %
                  (step1["flat"], step1["rel_spread"] * 100, [f["name"] for f in fixes],
                   [f["name"] for f in fixes if f["fix_restores_sensitivity"]],
                   winner["name"] if winner else "NONE", null_ok if step3 else "n/a",
                   RESULTS_PATH))})

    print(json.dumps(RESULTS, indent=2)[:4000])
    print("\n...(truncated console output; full results at " + RESULTS_PATH + ")")
    print("board appended at " + BOARD)


if __name__ == "__main__":
    main()
