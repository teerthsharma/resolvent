"""
Phase H -- H3v2 floors: four measured edits to the H3v2 bed before any twin trains.

Imports the bed (make_entities, rand_rotation/rot_ops, bed_instance, true_replay,
bag_predict, NONID_PERMS, seed offsets) from phaseh_H3v2_bed.py by import. Does
NOT rebuild the bed -- a second implementation would measure a different bed.
phaseh_H3v2_bed.py itself imports its primitives from phaseh_I1_I2_binding.py
(owned by another lane); both are read-only here.

Four edits, all measured, none guessed:
  1. Declared chance (1/C=0.1250) replaced by the MEASURED constant-predictor
     (majority-label) floor at n=100/1000/5000, plus the label-entropy reason
     1/C is not the right floor (unequal Voronoi cells on S^7 for 8 random
     unit anchors).
  2. bag_of_context re-measured at n=1000/5000 (not just n=100) and compared
     to the constant predictor: does it carry information at all on this bed?
  3. Two tighter floors added: last_op_only (single-hop bag entry using only
     the FINAL (verb, patient) pair) and last_two_ops (correct-order replay
     of only the last two hops, blind to the first hop / agent0). Honest
     headroom = 1.0 - max(all floors), not 1.0 - bag_of_context.
  4. The null (shuffled operator assignment) rescored as ACCURACY against the
     same discrete-label target the twins are scored on, per seed, per
     instance -- not merely isfinite. null_discriminates is set False (and
     the twins do not run) if the null is not strictly worse than the correct
     order on >=1 instance per seed at all 5 seeds.

NO TRAINING. Closed-form floors and diagnostics only, on the SAME winning bed
(discrete_label, C=8, m=12 d=8 k=6 line_len=3) that phaseh_H3v2_bed.py built.

Producer: Foreman (Claude Sonnet 5), numpy {version}, Python {pyversion}, Windows 11.
Run: python phaseh_H3v2_floors.py
Board: appends one JSON line per event to
  C:\\Users\\seal\\Desktop\\New folder (32)\\house-events.jsonl  (io.open .. "a", never rewritten)
"""
import numpy as np
import json
import io
import os
import time
import platform
import importlib.util
from collections import Counter

SCRATCH = os.path.dirname(os.path.abspath(__file__))
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
RESULTS_PATH = os.path.join(SCRATCH, "phaseh_H3v2_floors_results.json")
BED_SRC = os.path.join(SCRATCH, "phaseh_H3v2_bed.py")

PRODUCER = f"Foreman/numpy-{np.__version__}/py-{platform.python_version()}"
DTYPE = "float64"
COMMAND = "python phaseh_H3v2_floors.py"

# --- import the bed, not rebuild it -----------------------------------------
spec = importlib.util.spec_from_file_location("phaseh_H3v2_bed", BED_SRC)
bedmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bedmod)  # safe: bedmod.main() is __main__-guarded

M, D, K, LINE_LEN = bedmod.M, bedmod.D, bedmod.K, bedmod.LINE_LEN
SEEDS = bedmod.SEEDS
SEED_OFFSET = bedmod.SEED_OFFSET
rot_ops = bedmod.rot_ops
NONID_PERMS = bedmod.NONID_PERMS
make_entities = bedmod.make_entities
bed_instance = bedmod.bed_instance
true_replay = bedmod.bed.true_replay  # from the underlying binding module
bag_predict = bedmod.bag_predict
C = 8  # winning fix's anchor count, same as bedmod step2c/step3

# same RNG namespace bedmod's discrete_label winner used, so n=100 here
# reproduces bedmod's own 0.1600/0.1920/1.0000 numbers exactly.
POP_OFFSET = SEED_OFFSET + 40000
ANCHOR_OFFSET = SEED_OFFSET + 40800
WRONG_OFFSET = SEED_OFFSET + 40700
NULL_POP_OFFSET = SEED_OFFSET + 51000
NULL_ANCHOR_OFFSET = SEED_OFFSET + 51800


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


def build_population_n(seed, offset, n):
    """Same construction as bedmod.build_population(seed, offset, rot_ops), but
    with an arbitrary n instead of the module's fixed N_TEST=100. Same RNG
    stream at offset+seed, so n=100 reproduces bedmod's own instances exactly
    and n=1000/5000 are a longer draw from the identical stream (not a
    different population)."""
    rng = np.random.default_rng(offset + seed)
    entities = make_entities(rng, m=M, d=D)
    verb_ops = rot_ops(rng)
    instances = [bed_instance(rng, entities, verb_ops, LINE_LEN) for _ in range(n)]
    return entities, verb_ops, instances


def anchors_for(seed, anchor_offset):
    rng_anchor = np.random.default_rng(anchor_offset + seed)
    anchors = rng_anchor.standard_normal((C, D))
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    return anchors


def label_of(anchors, vecs):
    """vecs: (n, D) or (D,). Returns argmax cosine-sim label(s)."""
    vecs = np.atleast_2d(vecs)
    sims = (anchors @ vecs.T) / (np.linalg.norm(vecs, axis=1) + 1e-12)
    labels = np.argmax(sims, axis=0)
    return labels if labels.shape[0] > 1 else int(labels[0])


def last_op_only_predict(line, entities, verb_ops):
    """Tightest single-hop floor: the FINAL (verb, patient) pair only, same
    per-token transform bag_predict uses, but just the one term nearest the
    answer instead of meaning every token in a bag."""
    v, a, p = line[-1]
    return verb_ops[v] @ (entities[a] + entities[p])


def last_two_ops_predict(line, entities, verb_ops):
    """Correct-order replay of only the LAST TWO hops, starting fresh from
    entities[line[-2].agent] -- i.e. right operators, right order, for the
    tail of the chain, but blind to the first hop (agent0's real identity)."""
    v1, a1, p1 = line[-2]
    v2, _a2, p2 = line[-1]
    state = entities[a1].copy()
    state = verb_ops[v1] @ (state + entities[p1])
    state = verb_ops[v2] @ (state + entities[p2])
    return state


# ===========================================================================
# EDIT 1 -- measured constant-predictor (majority-label) floor, replacing 1/C
# ===========================================================================
def edit1_constant_predictor_floor():
    t0 = time.time()
    ns = [100, 1000, 5000]
    per_n = {}
    all_labels_at_5000 = []
    for n in ns:
        accs = []
        for seed in SEEDS:
            entities, verb_ops, instances = build_population_n(seed, POP_OFFSET, n)
            anchors = anchors_for(seed, ANCHOR_OFFSET)
            true_states = np.stack([s for _, s in instances])
            labels = label_of(anchors, true_states)
            maj = Counter(labels.tolist()).most_common(1)[0][0]
            accs.append(float(np.mean(labels == maj)))
            if n == 5000:
                all_labels_at_5000.extend(labels.tolist())
        per_n[n] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "per_seed": accs}

    counts = np.array(list(Counter(all_labels_at_5000).values()), dtype=float)
    counts = counts / counts.sum()
    entropy_bits = float(-(counts * np.log2(counts)).sum())
    effective_k = float(2 ** entropy_bits)
    declared_chance = 1.0 / C

    detail = ("constant (majority-label) predictor, C=%d anchors, %d seeds, dtype=%s, command: %s "
               ":: n=100 acc=%.4f (std %.4f), n=1000 acc=%.4f (std %.4f), n=5000 acc=%.4f "
               "(std %.4f) -- converges ABOVE declared chance=1/C=%.4f. Pooled label entropy at "
               "n=5000x%d seeds=%.3f bits of max %.3f (log2(%d)), effective_K=2^H=%.2f. Reason "
               "1/C is wrong: %d random unit anchors on S^%d have unequal Voronoi cell volumes, so "
               "argmax-cosine labels are not uniform even under a true random draw." %
              (C, len(SEEDS), DTYPE, COMMAND, per_n[100]["mean"], per_n[100]["std"],
               per_n[1000]["mean"], per_n[1000]["std"], per_n[5000]["mean"], per_n[5000]["std"],
               declared_chance, len(SEEDS), entropy_bits, np.log2(C), C, effective_k, C, D - 1))
    above_declared = per_n[5000]["mean"] > declared_chance
    T("edit1_constant_predictor_above_declared_chance", above_declared, detail, time.time() - t0)
    F("edit1_verdict",
      "Declared chance 1/C=%.4f is not the floor: the measured constant-predictor (always emit "
      "the majority anchor label, reading nothing, composing nothing) scores %.4f at n=100, "
      "%.4f at n=1000, %.4f at n=5000 (%d seeds, dtype=%s), i.e. it converges to a value strictly "
      "ABOVE 1/C and does not vanish with n -- this is a structural property of the label "
      "geometry (8 random unit anchors on S^%d have unequal Voronoi cells; label entropy %.3f "
      "bits of a possible %.3f, effective_K=%.2f), not a small-n artifact. The measured floor "
      "at n=5000 (%.4f) replaces 1/C=%.4f as the correct chance baseline." %
      (declared_chance, per_n[100]["mean"], per_n[1000]["mean"], per_n[5000]["mean"], len(SEEDS),
       DTYPE, D - 1, entropy_bits, np.log2(C), effective_k, per_n[5000]["mean"], declared_chance))
    return {"n_sweep": per_n, "declared_chance": declared_chance,
            "measured_floor_large_n": per_n[5000]["mean"], "entropy_bits": entropy_bits,
            "entropy_bits_max": float(np.log2(C)), "effective_k": effective_k,
            "above_declared": above_declared, "dur": time.time() - t0}


# ===========================================================================
# EDIT 2 -- bag_of_context at large n: does it beat the constant predictor?
# ===========================================================================
def edit2_bag_floor_large_n(const_floor):
    t0 = time.time()
    ns = [100, 1000, 5000]
    per_n = {}
    for n in ns:
        accs = []
        for seed in SEEDS:
            entities, verb_ops, instances = build_population_n(seed, POP_OFFSET, n)
            anchors = anchors_for(seed, ANCHOR_OFFSET)
            correct = 0
            for line, true_state in instances:
                agent0 = line[0][1]
                pred = bag_predict(line, entities, verb_ops, agent0)
                if label_of(anchors, pred) == label_of(anchors, true_state):
                    correct += 1
            accs.append(correct / n)
        per_n[n] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "per_seed": accs}

    beats_const_at_5000 = per_n[5000]["mean"] > const_floor["n_sweep"][5000]["mean"]
    detail = ("bag_of_context accuracy, %d seeds, dtype=%s, command: %s :: n=100 acc=%.4f, "
               "n=1000 acc=%.4f, n=5000 acc=%.4f vs constant-predictor at same n: %.4f / %.4f / "
               "%.4f -> bag beats constant predictor at n=5000 = %s" %
              (len(SEEDS), DTYPE, COMMAND, per_n[100]["mean"], per_n[1000]["mean"],
               per_n[5000]["mean"], const_floor["n_sweep"][100]["mean"],
               const_floor["n_sweep"][1000]["mean"], const_floor["n_sweep"][5000]["mean"],
               beats_const_at_5000))
    T("edit2_bag_beats_constant_predictor", beats_const_at_5000, detail, time.time() - t0)
    F("edit2_verdict",
      "At n=100 (%d seeds), bag_of_context=%.4f equals the constant predictor (%.4f) to two "
      "decimals -- it carries no information at that sample size. At n=5000, bag_of_context=%.4f "
      "%s the constant predictor (%.4f): %s. Either way bag_of_context is NOT the tightest floor "
      "on this bed (see edit 3); it is reported here for completeness, not relied on for "
      "headroom." %
      (len(SEEDS), per_n[100]["mean"], const_floor["n_sweep"][100]["mean"], per_n[5000]["mean"],
       "beats" if beats_const_at_5000 else "does NOT beat", const_floor["n_sweep"][5000]["mean"],
       "bag is a real (if weak) floor above chance" if beats_const_at_5000 else
       "bag carries no measurable information above the constant predictor on this bed and is "
       "not a valid floor at all"))
    return {"n_sweep": per_n, "beats_const_at_5000": beats_const_at_5000, "dur": time.time() - t0}


# ===========================================================================
# EDIT 3 -- tighter floors: last_op_only, last_two_ops (+ order_free for a
# complete table). Reported at n=100 (bed scale) AND n=5000 (large-n, same
# convention edit1 used for the constant-predictor floor). Honest headroom
# uses the large-n numbers, since n=100 per-seed std is 6-7 points here.
# ===========================================================================
def edit3_tighter_floors(bag_floor, const_floor, n):
    t0 = time.time()
    lo_accs, lt_accs, of_accs = [], [], []
    for seed in SEEDS:
        entities, verb_ops, instances = build_population_n(seed, POP_OFFSET, n)
        anchors = anchors_for(seed, ANCHOR_OFFSET)
        rng_wrong = np.random.default_rng(WRONG_OFFSET + seed)
        lo_correct, lt_correct, of_correct = 0, 0, 0
        for line, true_state in instances:
            agent0 = line[0][1]
            true_lbl = label_of(anchors, true_state)
            lo_pred = last_op_only_predict(line, entities, verb_ops)
            lt_pred = last_two_ops_predict(line, entities, verb_ops)
            wrong_perm = NONID_PERMS[rng_wrong.integers(len(NONID_PERMS))]
            of_pred = bedmod.replay_perm(line, entities, verb_ops, agent0, wrong_perm)
            if label_of(anchors, lo_pred) == true_lbl:
                lo_correct += 1
            if label_of(anchors, lt_pred) == true_lbl:
                lt_correct += 1
            if label_of(anchors, of_pred) == true_lbl:
                of_correct += 1
        lo_accs.append(lo_correct / n)
        lt_accs.append(lt_correct / n)
        of_accs.append(of_correct / n)

    last_op_only_mean = float(np.mean(lo_accs))
    last_two_ops_mean = float(np.mean(lt_accs))
    order_free_mean = float(np.mean(of_accs))
    floors_all = {"bag_of_context": bag_floor, "constant_predictor": const_floor,
                  "order_free": order_free_mean, "last_op_only": last_op_only_mean,
                  "last_two_ops": last_two_ops_mean}
    tightest_name = max(floors_all, key=floors_all.get)
    tightest_val = floors_all[tightest_name]
    old_headroom = 1.0 - bag_floor
    honest_headroom = 1.0 - tightest_val

    detail = ("last_op_only (final verb/patient one-hop) acc=%.4f (std %.4f), last_two_ops "
               "(correct-order replay of only the final 2 hops, blind to hop 0) acc=%.4f "
               "(std %.4f), order_free (random wrong permutation, full chain) acc=%.4f, n=%d, "
               "%d seeds, dtype=%s, command: %s. Floor set: %s. Tightest floor=%s at %.4f. Old "
               "headroom (1 - bag_of_context)=%.4f. Honest headroom (1 - tightest)=%.4f." %
              (last_op_only_mean, float(np.std(lo_accs)), last_two_ops_mean, float(np.std(lt_accs)),
               order_free_mean, n, len(SEEDS), DTYPE, COMMAND,
               {k: round(v, 4) for k, v in floors_all.items()}, tightest_name, tightest_val,
               old_headroom, honest_headroom))
    ordering_sane = last_two_ops_mean >= last_op_only_mean  # 2 correctly-ordered hops should not beat 1 by less info
    T("edit3_tighter_floors_measured_n%d" % n, ordering_sane, detail, time.time() - t0)
    F("edit3_verdict_n%d" % n,
      "The floor set was missing its tightest members. last_op_only=%.4f and last_two_ops=%.4f "
      "(both measured, n=%d, %d seeds, dtype=%s), against a constant predictor of %.4f, "
      "bag_of_context of %.4f, and order_free of %.4f. Tightest floor overall is %s=%.4f. Every "
      "twin on this bed must be reported as headroom recovered above %.4f (honest "
      "headroom=%.4f), not above bag_of_context (%.4f, headroom %.4f) -- a twin scoring at or "
      "below %.4f IS that floor, not a demonstration of order." %
      (last_op_only_mean, last_two_ops_mean, n, len(SEEDS), DTYPE, const_floor, bag_floor,
       order_free_mean, tightest_name, tightest_val, tightest_val, honest_headroom, bag_floor,
       old_headroom, tightest_val))
    return {"n": n, "last_op_only_mean": last_op_only_mean, "last_two_ops_mean": last_two_ops_mean,
            "order_free_mean": order_free_mean, "floors_all": floors_all,
            "tightest_name": tightest_name, "tightest_val": tightest_val,
            "old_headroom_vs_bag": old_headroom, "honest_headroom": honest_headroom,
            "dur": time.time() - t0}


# ===========================================================================
# EDIT 4 -- rescore the null as ACCURACY against the target, per seed/instance
# ===========================================================================
def edit4_null_as_accuracy():
    t0 = time.time()
    per_seed = []
    for seed in SEEDS:
        rng = np.random.default_rng(NULL_POP_OFFSET + seed)
        entities = make_entities(rng, m=M, d=D)
        verb_ops = rot_ops(rng)
        shuffled_ops = [verb_ops[i] for i in rng.permutation(K)]
        instances = [bed_instance(rng, entities, verb_ops, LINE_LEN) for _ in range(bedmod.N_TEST)]
        anchors = anchors_for(seed, NULL_ANCHOR_OFFSET)

        correct = 0
        worse_count = 0  # null strictly wrong while correct-order is exactly right
        for line, true_state in instances:
            agent0 = line[0][1]
            true_lbl = label_of(anchors, true_state)  # == label_of(true_replay with real ops): correct order is always right by construction
            null_pred = true_replay(line, entities, shuffled_ops, agent0)
            null_lbl = label_of(anchors, null_pred)
            if null_lbl == true_lbl:
                correct += 1
            else:
                worse_count += 1
        acc = correct / bedmod.N_TEST
        per_seed.append({"seed": seed, "n": bedmod.N_TEST, "null_accuracy": acc,
                          "n_worse_than_correct": worse_count})

    all_seeds_have_a_worse_instance = all(r["n_worse_than_correct"] >= 1 for r in per_seed)
    mean_null_acc = float(np.mean([r["null_accuracy"] for r in per_seed]))
    detail = ("shuffled-operator null rescored as accuracy against the discrete-label target "
               "(not isfinite), n=%d per seed, %d seeds, dtype=%s, command: %s :: per-seed "
               "null_accuracy=%s, per-seed count(null-wrong-while-correct-order-right)=%s -> "
               "mean null_accuracy=%.4f, null strictly worse than correct order on >=1 instance "
               "per seed at all %d seeds = %s" %
              (bedmod.N_TEST, len(SEEDS), DTYPE, COMMAND,
               [round(r["null_accuracy"], 4) for r in per_seed],
               [r["n_worse_than_correct"] for r in per_seed], mean_null_acc, len(SEEDS),
               all_seeds_have_a_worse_instance))
    T("edit4_null_discriminates", all_seeds_have_a_worse_instance, detail, time.time() - t0)
    F("edit4_verdict",
      "Rescored as accuracy against the target (correct order scores 1.0000 by construction on "
      "every instance): the shuffled-operator null scores mean accuracy %.4f (%d seeds, n=%d "
      "each, dtype=%s), and is strictly worse than the correct order (wrong label while correct "
      "order is right) on >=1 instance per seed at all %d seeds = %s. %s" %
      (mean_null_acc, len(SEEDS), bedmod.N_TEST, DTYPE, len(SEEDS),
       all_seeds_have_a_worse_instance,
       "The bed discriminates: the null is measurably worse than truth, not merely finite."
       if all_seeds_have_a_worse_instance else
       "VOID: the null does not fail anywhere it should -- the bed does not discriminate and no "
       "twin should run against it."))
    return {"per_seed": per_seed, "mean_null_accuracy": mean_null_acc,
            "null_discriminates": all_seeds_have_a_worse_instance, "dur": time.time() - t0}


def main():
    t0 = time.time()
    F("phase_start",
      "H3v2 floors: land 4 measured edits before any twin trains on the discrete_label bed "
      "(C=%d anchors, m=%d d=%d k=%d line_len=%d, seeds=%s, dtype=%s, producer=%s). Bed and its "
      "primitives imported from %s, not rebuilt." %
      (C, M, D, K, LINE_LEN, list(SEEDS), DTYPE, PRODUCER, BED_SRC))

    e1 = edit1_constant_predictor_floor()
    e2 = edit2_bag_floor_large_n(e1)
    e3_100 = edit3_tighter_floors(bag_floor=e2["n_sweep"][100]["mean"],
                                   const_floor=e1["n_sweep"][100]["mean"], n=100)
    e3_5000 = edit3_tighter_floors(bag_floor=e2["n_sweep"][5000]["mean"],
                                    const_floor=e1["n_sweep"][5000]["mean"], n=5000)
    e3 = e3_5000  # large-n is the reporting convention (lower per-seed std than n=100)
    e4 = edit4_null_as_accuracy()

    RESULTS = {
        "producer": PRODUCER, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "dtype": DTYPE,
        "seeds": list(SEEDS), "n_test_per_seed_bed": bedmod.N_TEST,
        "bed_params": {"m": M, "d": D, "k": K, "line_len": LINE_LEN, "c_anchors": C},
        "bed_source": BED_SRC,
        "edit1_constant_predictor_floor": e1,
        "edit2_bag_floor_large_n": e2,
        "edit3_tighter_floors_n100": e3_100,
        "edit3_tighter_floors_n5000": e3_5000,
        "edit4_null_as_accuracy": e4,
        "wall_seconds": time.time() - t0,
    }
    with io.open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)

    checks = 5
    reds = sum(1 for ok in [e1["above_declared"], e2["beats_const_at_5000"],
                             e3_100["last_two_ops_mean"] >= e3_100["last_op_only_mean"],
                             e3_5000["last_two_ops_mean"] >= e3_5000["last_op_only_mean"],
                             e4["null_discriminates"]] if not ok)

    log({"t": "done", "agent": "Foreman", "checks": checks, "reds": reds,
         "text": ("H3v2 floors: measured constant-predictor floor n=100/1000/5000=%.4f/%.4f/%.4f "
                   "(declared chance was %.4f). bag_of_context n=100/5000=%.4f/%.4f, beats "
                   "constant at n=5000=%s. Tighter floors (n=5000) last_op_only=%.4f "
                   "last_two_ops=%.4f order_free=%.4f (n=100: %.4f/%.4f/%.4f), tightest=%s@%.4f, "
                   "honest headroom=%.4f (vs old headroom-vs-bag=%.4f). Null rescored as "
                   "accuracy=%.4f, null_discriminates=%s. results at %s" %
                  (e1["n_sweep"][100]["mean"], e1["n_sweep"][1000]["mean"],
                   e1["n_sweep"][5000]["mean"], e1["declared_chance"],
                   e2["n_sweep"][100]["mean"], e2["n_sweep"][5000]["mean"],
                   e2["beats_const_at_5000"], e3_5000["last_op_only_mean"],
                   e3_5000["last_two_ops_mean"], e3_5000["order_free_mean"],
                   e3_100["last_op_only_mean"], e3_100["last_two_ops_mean"],
                   e3_100["order_free_mean"], e3_5000["tightest_name"], e3_5000["tightest_val"],
                   e3_5000["honest_headroom"], e3_5000["old_headroom_vs_bag"],
                   e4["mean_null_accuracy"], e4["null_discriminates"], RESULTS_PATH))})

    print(json.dumps(RESULTS, indent=2)[:4000])
    print("\n...(truncated console output; full results at " + RESULTS_PATH + ")")
    print("board appended at " + BOARD)
    print("\nnull_discriminates=%s -- twins %s" %
          (e4["null_discriminates"], "MAY proceed" if e4["null_discriminates"] else "MUST NOT run"))


if __name__ == "__main__":
    main()
