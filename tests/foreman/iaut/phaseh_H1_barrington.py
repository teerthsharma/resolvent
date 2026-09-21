"""
Phase H, Bed H1 -- the Barrington bed.

Task: given a word over a fixed 2-generator alphabet that generates S5,
decide the resulting permutation (the S5 word problem). Oracle is exact
integer permutation composition -- no scoring ambiguity at any length.

Convention (stated once, used everywhere below):
  - A permutation p is a tuple of length 5 with p[i] = image of i.
  - Product p*q means "apply p first, then q": compose(p, q)[i] = q[p[i]].
  - identity = (0,1,2,3,4).
  - A word w = (g1, ..., gL) has oracle value g1*g2*...*gL, built by folding
    compose() left to right: prefix[0] = identity, prefix[k] = compose(prefix[k-1], gk).
  - Matrix form: M_p[i, p[i]] = 1 (row-vector convention e_i @ M_p = e_{p[i]}).
    Then M_p @ M_q = M_{compose(p,q)} exactly (permutation matrices are 0/1,
    the matmul that builds each output entry is a sum of a single 1 and four
    0s, so there is nothing for float64/float32/bf16 rounding to bite into).

Run: python phaseh_H1_barrington.py
Producer: Cameron, Phase H round. Python 3.11.9, numpy 2.4.6, torch 2.14.0+cpu.
"""
import io
import json
import itertools
import os
import random
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "phaseh_H1_barrington_results.json")
BOARD_PATH = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
AGENT = "Cameron"

IDENTITY = (0, 1, 2, 3, 4)


def board(line: dict):
    with io.open(BOARD_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def board_test(name, state, detail, dur=0.0):
    board({"t": "test", "agent": AGENT, "state": state, "name": name, "detail": detail, "dur": round(dur, 4)})


def board_finding(item, text):
    board({"t": "finding", "agent": AGENT, "item": item, "text": text})


# ---------------------------------------------------------------------------
# Group machinery (exact integers, no floats anywhere in this section)
# ---------------------------------------------------------------------------

def compose(p, q):
    """p then q, right-action convention: compose(p,q)[i] = q[p[i]]."""
    return tuple(q[p[i]] for i in range(5))


def inverse(p):
    inv = [0] * 5
    for i, pi in enumerate(p):
        inv[pi] = i
    return tuple(inv)


ALL_S5 = sorted(itertools.permutations(range(5)))
assert len(ALL_S5) == 120
INDEX_OF = {p: i for i, p in enumerate(ALL_S5)}

# Two-generator alphabet: a 5-cycle and a transposition.
# Choice and justification (deliverable requirement): we use the classical
# {n-cycle, adjacent transposition} generating pair, not all 120 elements as
# symbols. Using the full group as the alphabet would make each symbol
# already carry a full state transition, so "scan" degenerates into a lookup
# table walk with no real composition depth -- closer to reading answers off
# a multiplication table than to a Barrington-style branching-program word.
# The 2-generator alphabet keeps the instance a genuine word-over-generators
# problem: state changes on every letter, and reaching an arbitrary target
# permutation requires real composition depth (the Cayley graph diameter of
# S5 under {C, T} is small, but nonzero, at every prefix).
C_GEN = (1, 2, 3, 4, 0)   # 5-cycle: i -> i+1 mod 5
T_GEN = (1, 0, 2, 3, 4)   # transposition swapping 0 and 1
ALPHABET = {"c": C_GEN, "t": T_GEN}


def generates_s5(gens):
    seen = {IDENTITY}
    frontier = [IDENTITY]
    while frontier:
        nxt = []
        for p in frontier:
            for g in gens:
                q = compose(p, g)
                if q not in seen:
                    seen.add(q)
                    nxt.append(q)
        frontier = nxt
    return len(seen) == 120


# ---------------------------------------------------------------------------
# Matrix form
# ---------------------------------------------------------------------------

def perm_to_matrix(p, dtype=np.float64):
    m = np.zeros((5, 5), dtype=dtype)
    for i, pi in enumerate(p):
        m[i, pi] = 1
    return m


def matrix_to_perm(m):
    """Recover a permutation tuple from a (near-)permutation matrix by row argmax."""
    return tuple(int(np.argmax(m[i])) for i in range(5))


GEN_MATRIX_F64 = {k: perm_to_matrix(v, np.float64) for k, v in ALPHABET.items()}
GEN_MATRIX_F32 = {k: perm_to_matrix(v, np.float32) for k, v in ALPHABET.items()}
GEN_MATRIX_BF16 = {k: torch.tensor(perm_to_matrix(v, np.float32)).to(torch.bfloat16) for k, v in ALPHABET.items()}


# ---------------------------------------------------------------------------
# (1) T1: scan of 5x5 permutation operators at length 10,000, bitwise exact
# ---------------------------------------------------------------------------

def run_t1(length=10000, seed=0):
    rng = random.Random(seed)
    letters = [rng.choice(list(ALPHABET.keys())) for _ in range(length)]

    int_prefix = IDENTITY
    m64 = np.eye(5, dtype=np.float64)
    m32 = np.eye(5, dtype=np.float32)
    mbf = torch.eye(5, dtype=torch.bfloat16)

    first_divergence = {"float64": None, "float32": None, "bf16": None}
    mismatches = {"float64": 0, "float32": 0, "bf16": 0}

    t0 = time.time()
    for k, letter in enumerate(letters, start=1):
        int_prefix = compose(int_prefix, ALPHABET[letter])
        m64 = m64 @ GEN_MATRIX_F64[letter]
        m32 = m32 @ GEN_MATRIX_F32[letter]
        mbf = mbf @ GEN_MATRIX_BF16[letter]

        p64 = matrix_to_perm(m64)
        p32 = matrix_to_perm(m32)
        pbf = matrix_to_perm(mbf.to(torch.float32).numpy())

        if p64 != int_prefix:
            mismatches["float64"] += 1
            if first_divergence["float64"] is None:
                first_divergence["float64"] = k
        if p32 != int_prefix:
            mismatches["float32"] += 1
            if first_divergence["float32"] is None:
                first_divergence["float32"] = k
        if pbf != int_prefix:
            mismatches["bf16"] += 1
            if first_divergence["bf16"] is None:
                first_divergence["bf16"] = k
    dur = time.time() - t0

    # Also check exact bit-pattern equality (not just argmax-recovered perm):
    # every surviving entry of a permutation-matrix product must be exactly
    # 0.0 or 1.0 in each dtype tested, with no drift.
    exact_bits = {
        "float64": bool(np.all((m64 == 0) | (m64 == 1))),
        "float32": bool(np.all((m32 == 0) | (m32 == 1))),
        "bf16": bool(torch.all((mbf == 0) | (mbf == 1)).item()),
    }

    result = {
        "length": length,
        "seed": seed,
        "duration_s": round(dur, 4),
        "final_int": list(int_prefix),
        "final_float64": list(matrix_to_perm(m64)),
        "final_float32": list(matrix_to_perm(m32)),
        "final_bf16": list(matrix_to_perm(mbf.to(torch.float32).numpy())),
        "mismatch_counts": mismatches,
        "first_divergence_step": first_divergence,
        "exact_bit_pattern_0_or_1": exact_bits,
        "bitwise_exact_all_dtypes": all(v == 0 for v in mismatches.values()),
    }

    for dtype in ("float64", "float32", "bf16"):
        state = "GREEN" if mismatches[dtype] == 0 else "RED"
        board_test(
            f"T1_scan_{dtype}_len{length}",
            state,
            f"mismatches={mismatches[dtype]} first_divergence={first_divergence[dtype]}",
            dur=dur / 3,
        )

    # Associativity cross-check: sequential fold vs. a divide-and-conquer
    # binary-tree scan must land on the identical permutation, at a smaller
    # length (1024) so the recursion depth stays sane.
    L2 = 1024
    letters2 = letters[:L2]
    mats2 = [ALPHABET[l] for l in letters2]

    def tree_reduce(seq):
        if len(seq) == 1:
            return seq[0]
        mid = len(seq) // 2
        return compose(tree_reduce(seq[:mid]), tree_reduce(seq[mid:]))

    seq_result = IDENTITY
    for m in mats2:
        seq_result = compose(seq_result, m)
    tree_result = tree_reduce(mats2)
    assoc_ok = seq_result == tree_result
    board_test(
        "T1_associative_scan_vs_sequential_fold",
        "GREEN" if assoc_ok else "RED",
        f"len={L2} sequential={seq_result} tree={tree_result}",
    )
    result["associativity_check_len1024"] = {"sequential": list(seq_result), "tree": list(tree_result), "match": assoc_ok}

    board_finding(
        "T1_reproduction",
        f"Scan of 5x5 permutation operators at length {length} (seed {seed}): "
        f"mismatches vs exact integer product = {mismatches} across {{float64,float32,bf16}}. "
        f"{'BITWISE EXACT at all three dtypes, matching the reference instance.' if result['bitwise_exact_all_dtypes'] else 'DIVERGED -- see first_divergence_step.'} "
        f"Reason it must be exact: every surviving entry of a permutation-matrix product is 0 or 1 exactly, "
        f"so there is no rounding for any of the tested dtypes to accumulate."
    )
    return result


# ---------------------------------------------------------------------------
# (2) Non-commuting pair count over all 120 elements
# ---------------------------------------------------------------------------

def run_noncommute():
    n = len(ALL_S5)
    total_unordered_pairs = n * (n - 1) // 2
    noncommuting = 0
    for i in range(n):
        for j in range(i + 1, n):
            g, h = ALL_S5[i], ALL_S5[j]
            if compose(g, h) != compose(h, g):
                noncommuting += 1
    matches_reference = (total_unordered_pairs == 7140) and (noncommuting == 6780)
    board_test(
        "noncommuting_pair_count",
        "GREEN" if matches_reference else "RED",
        f"pairs={total_unordered_pairs} noncommuting={noncommuting} (reference: 7140 / 6780)",
    )
    board_finding(
        "noncommuting_pairs",
        f"Counted over unordered pairs {{g,h}}, i != j, of all C({n},2)={total_unordered_pairs} "
        f"element pairs of S5 (each pair counted once, not as two ordered pairs): "
        f"{noncommuting} of {total_unordered_pairs} pairs have compose(g,h) != compose(h,g). "
        f"{'Matches the reference 6,780 of 7,140 exactly.' if matches_reference else 'DOES NOT match reference 6,780 of 7,140 -- reported flatly, not adjusted toward it.'}"
    )
    return {
        "n_elements": n,
        "counting_convention": "unordered pairs {g,h}, i<j index into a fixed sorted enumeration of ALL_S5, each pair counted once",
        "total_unordered_pairs": total_unordered_pairs,
        "noncommuting": noncommuting,
        "commuting": total_unordered_pairs - noncommuting,
        "reference_pairs": 7140,
        "reference_noncommuting": 6780,
        "matches_reference": matches_reference,
    }


# ---------------------------------------------------------------------------
# Instance generation shared by floors / null / length-leak sections
# ---------------------------------------------------------------------------

LENGTHS = [64, 256, 1024, 4096]
N_INSTANCES = 500


def gen_instances(length, seed):
    """N_INSTANCES random words of `length` letters over ALPHABET, deterministic per (length, seed)."""
    rng = random.Random(f"H1|{length}|{seed}")
    keys = list(ALPHABET.keys())
    instances = []
    for _ in range(N_INSTANCES):
        word = [rng.choice(keys) for _ in range(length)]
        prefix = IDENTITY
        prefix_before_last = IDENTITY
        for k, letter in enumerate(word):
            prefix_before_last = prefix
            prefix = compose(prefix, ALPHABET[letter])
        instances.append({"word": word, "final": prefix, "final_before_last": prefix_before_last})
    return instances


# ---------------------------------------------------------------------------
# (4) Floors, on every length cell: always-identity, last-state, class-frequency
# ---------------------------------------------------------------------------

def run_floors():
    per_length = {}
    for L in LENGTHS:
        inst = gen_instances(L, seed=0)
        finals = [x["final"] for x in inst]
        n = len(inst)

        identity_hits = sum(1 for f in finals if f == IDENTITY)
        identity_acc = identity_hits / n

        laststate_hits = sum(1 for x in inst if x["final"] == x["final_before_last"])
        laststate_acc = laststate_hits / n

        counts = {}
        for f in finals:
            counts[f] = counts.get(f, 0) + 1
        mode_perm, mode_count = max(counts.items(), key=lambda kv: kv[1])
        classfreq_acc = mode_count / n

        # Bed-validity floor: the exact scan itself must match the oracle on
        # every instance (this is the "no scoring ambiguity" sanity check,
        # separate from the null in section 3).
        scan_correct = 0
        for x in inst:
            m = np.eye(5, dtype=np.float64)
            for letter in x["word"]:
                m = m @ GEN_MATRIX_F64[letter]
            if matrix_to_perm(m) == x["final"]:
                scan_correct += 1
        scan_acc = scan_correct / n

        per_length[L] = {
            "n_instances": n,
            "always_identity_acc": identity_acc,
            "last_state_acc": laststate_acc,
            "class_frequency_acc": classfreq_acc,
            "class_frequency_mode_count": mode_count,
            "n_distinct_classes_hit": len(counts),
            "scan_matches_oracle_acc": scan_acc,
        }

        board_test(
            f"floor_last_state_is_zero_L{L}",
            "GREEN" if laststate_acc == 0.0 else "RED",
            f"last-state floor acc={laststate_acc} (expected 0.0: no generator is the identity, so the final letter always moves the state)",
        )
        board_test(
            f"floor_scan_matches_oracle_L{L}",
            "GREEN" if scan_acc == 1.0 else "RED",
            f"scan-vs-oracle acc={scan_acc} over {n} instances",
        )
        board_test(
            f"floors_recorded_L{L}",
            "GREEN",
            f"always-identity={identity_acc:.4f} last-state={laststate_acc:.4f} class-frequency={classfreq_acc:.4f} ({mode_count}/{n})",
        )

    board_finding(
        "floors",
        "Floors computed on every length cell {64,256,1024,4096}, 500 instances/cell, seed 0: "
        + "; ".join(
            f"L={L}: identity={per_length[L]['always_identity_acc']:.4f}, "
            f"last-state={per_length[L]['last_state_acc']:.4f}, "
            f"class-freq={per_length[L]['class_frequency_acc']:.4f} "
            f"({per_length[L]['class_frequency_mode_count']}/{per_length[L]['n_instances']}, "
            f"{per_length[L]['n_distinct_classes_hit']}/120 classes hit)"
            for L in LENGTHS
        )
        + ". last-state is exactly 0.0 at every length because both generators are non-identity, "
        "so composing with the final letter always changes the running permutation (group cancellation)."
    )
    return per_length


# ---------------------------------------------------------------------------
# (3) THE NULL: shuffled operator assignment
# ---------------------------------------------------------------------------

def run_null():
    per_seed = {}
    for seed in range(5):
        seed_total = 0
        seed_wrong = 0
        per_length_wrong = {}
        for L in LENGTHS:
            inst = gen_instances(L, seed=0)  # same fixed instances as the floors section
            rng = random.Random(f"null|{L}|{seed}")
            wrong = 0
            for x in inst:
                shuffled = list(x["word"])
                rng.shuffle(shuffled)  # shuffled operator ASSIGNMENT: same multiset, reordered
                p = IDENTITY
                for letter in shuffled:
                    p = compose(p, ALPHABET[letter])
                if p != x["final"]:
                    wrong += 1
            per_length_wrong[L] = {"wrong": wrong, "total": len(inst)}
            seed_total += len(inst)
            seed_wrong += wrong

        per_seed[seed] = {
            "total_instances": seed_total,
            "wrong": seed_wrong,
            "wrong_frac": seed_wrong / seed_total,
            "by_length": per_length_wrong,
            "void_condition_met": seed_wrong >= 1,
        }
        board_test(
            f"null_shuffled_assignment_seed{seed}",
            "GREEN" if seed_wrong >= 1 else "RED",
            f"wrong={seed_wrong}/{seed_total} (bed requires >=1 wrong per seed; RED here would mean the bed is VOID)",
        )

    all_seeds_ok = all(v["void_condition_met"] for v in per_seed.values())
    board_finding(
        "null_shuffled_operator_assignment",
        "Null = same word, operator order shuffled (composition is non-commutative, so a random reorder "
        "should almost never reproduce the true product). Per-seed wrong-count out of "
        f"{sum(len(gen_instances(L, 0)) for L in LENGTHS)} instances (all 4 lengths pooled): "
        + "; ".join(f"seed{s}: {v['wrong']}/{v['total_instances']} wrong" for s, v in per_seed.items())
        + f". {'Bed is SOUND: every seed scores at least 1 wrong instance.' if all_seeds_ok else 'BED IS VOID: at least one seed produced zero wrong instances under the shuffled null -- no learned row may run on this bed.'}"
    )
    return {"per_seed": per_seed, "bed_sound": all_seeds_ok}


# ---------------------------------------------------------------------------
# (5) Length-leak check: C17 tests at 4L, no test length in any training split
# ---------------------------------------------------------------------------

def run_length_leak_check():
    train_lengths = [64, 256, 1024]
    test_lengths = [4096]  # 4 * max(train_lengths)
    ratio_ok = test_lengths[0] == 4 * max(train_lengths)

    disjoint = set(train_lengths).isdisjoint(set(test_lengths))

    # Word-level check: no generated training word is byte-identical to any
    # generated test word. Lengths already differ so this is automatically
    # true, but it is verified rather than assumed, by hashing every word.
    train_words = set()
    for L in train_lengths:
        for x in gen_instances(L, seed=0):
            train_words.add(tuple(x["word"]))
    test_words = set()
    for L in test_lengths:
        for x in gen_instances(L, seed=0):
            test_words.add(tuple(x["word"]))
    overlap = train_words & test_words

    ok = ratio_ok and disjoint and len(overlap) == 0
    board_test(
        "length_leak_check",
        "GREEN" if ok else "RED",
        f"train_lengths={train_lengths} test_lengths={test_lengths} "
        f"disjoint_lengths={disjoint} word_overlap={len(overlap)} ratio_4L_ok={ratio_ok}",
    )
    board_finding(
        "length_leak_check",
        f"Train lengths {train_lengths}, test length {test_lengths} = 4x the max train length (C17-style 4L test). "
        f"Enforced by construction: length sets are disjoint python sets ({disjoint}), and every generated "
        f"word was additionally hashed by its full letter sequence to confirm zero overlap between the "
        f"{len(train_words)} training words and {len(test_words)} test words ({len(overlap)} shared). "
        "No held-out length is ever a member of a training split for this bed."
    )
    return {
        "train_lengths": train_lengths,
        "test_lengths": test_lengths,
        "ratio_4L_ok": ratio_ok,
        "lengths_disjoint": disjoint,
        "n_train_words": len(train_words),
        "n_test_words": len(test_words),
        "word_overlap": len(overlap),
        "ok": ok,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    assert generates_s5(ALPHABET.values()), "alphabet does not generate S5"
    board_test("alphabet_generates_s5", "GREEN", f"generators={list(ALPHABET.items())} closure_size=120")

    results = {}
    results["t1"] = run_t1()
    results["noncommuting"] = run_noncommute()
    results["floors"] = run_floors()
    results["null"] = run_null()
    results["length_leak_check"] = run_length_leak_check()

    with io.open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    checks = 0
    reds = 0
    # recount from what we already know rather than re-parsing the board file
    checks_list = [
        results["t1"]["mismatch_counts"]["float64"] == 0,
        results["t1"]["mismatch_counts"]["float32"] == 0,
        results["t1"]["mismatch_counts"]["bf16"] == 0,
        results["t1"]["associativity_check_len1024"]["match"],
        results["noncommuting"]["matches_reference"],
        all(v["last_state_acc"] == 0.0 for v in results["floors"].values()),
        all(v["scan_matches_oracle_acc"] == 1.0 for v in results["floors"].values()),
        results["null"]["bed_sound"],
        results["length_leak_check"]["ok"],
    ]
    checks = len(checks_list)
    reds = sum(1 for c in checks_list if not c)

    board({
        "t": "done", "agent": AGENT, "checks": checks, "reds": reds,
        "text": (
            f"Bed H1 (Barrington bed) built and self-checked. T1 scan bitwise-exact at length 10000 "
            f"across float64/float32/bf16: {results['t1']['bitwise_exact_all_dtypes']}. "
            f"Non-commuting pairs {results['noncommuting']['noncommuting']}/{results['noncommuting']['total_unordered_pairs']} "
            f"(reference 6780/7140, match={results['noncommuting']['matches_reference']}). "
            f"Null (shuffled operator order) is sound at all 5 seeds: {results['null']['bed_sound']}. "
            f"Length-leak check ok: {results['length_leak_check']['ok']}. "
            f"{reds} of {checks} self-checks RED."
        )
    })

    print(json.dumps(results, indent=2))
    print(f"\n{checks - reds}/{checks} checks GREEN, results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
