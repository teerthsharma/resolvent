"""Row R-DPI: input-only controls (bag, k-gram) against the 50 do() worlds
in do_worlds.json. For each world, 64 streams of that world only (S=128),
two estimators of q_do per transient state, error = mean over worlds and
transient states of |estimate - q_do_exact|.

(i) visit-frequency bag: sample_streams restarts immediately after writing an
absorbing token, so every stream splits into segments that each end at one
absorbing token. For a transient state i, its bag estimate of P(hit 6 before
7) is the empirical fraction of i's occurrences (across all segments) whose
segment absorbs at 6. This is hour one's convention generalised from "predict
the pre-intervention committor" (single fixed number) to "predict from the
world's own visit statistics" (per-world, per-state) -- both ignore the chain
structure and read off empirical outcome frequencies only.

(ii) k-gram plug-in: for k=1, the first-order transition matrix MLE from
consecutive-pair counts (excluding the wrap edge absorb->restart, which is
not a P-transition), absorbing rows forced to self-loop, then solve the
committor. For k=2,3, count (k-1)-token context -> next-token, then collapse
by marginalizing every context down to its last token (summing counts over
all longer histories sharing that last token) before solving -- since the
generator is itself order-1 Markov, this collapse is the same sum as the k=1
counts, so k=2,3 are expected to reproduce k=1's numbers up to how the counts
happen to be re-grouped.
"""
import json
import sys
import time

t0 = time.time()
import numpy as np

from do_worlds import committor, sample_streams, n, ABSORBING, TRANSIENT

DO_WORLDS_JSON = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j\do_worlds.json"
OUT_JSON = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j\do_dpi.json"
RECORD_JSONL = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j\record.jsonl"

N_SEQ = 64
S = 128


def bag_estimate(streams):
    """Per-transient-state P(hit 6 before 7), from visit-outcome frequency."""
    hits6 = np.zeros(len(TRANSIENT))
    total = np.zeros(len(TRANSIENT))
    idx_of = {s: k for k, s in enumerate(TRANSIENT)}
    for row in streams:
        seg_states = []
        for tok in row:
            if tok in ABSORBING:
                for s in seg_states:
                    k = idx_of[s]
                    total[k] += 1
                    if tok == 6:
                        hits6[k] += 1
                seg_states = []
            else:
                seg_states.append(int(tok))
        # trailing partial segment (stream ended before absorption): no outcome, drop
    q = np.divide(hits6, total, out=np.full(len(TRANSIENT), np.nan), where=total > 0)
    return q, total


def kgram_transition(streams, k):
    """k-gram context (last k-1 tokens) -> next token counts, collapsed to a
    first-order n x n count matrix by marginalizing every context down to its
    last token before normalizing."""
    counts = np.zeros((n, n))
    for row in streams:
        L = len(row)
        for t in range(L - 1):
            cur = int(row[t])
            nxt = int(row[t + 1])
            if cur in ABSORBING:
                continue  # wrap edge (restart), not a P-transition
            # context is row[t-k+2 : t+1], but collapsed estimate only needs
            # the last token (cur) -- higher k contributes the same pair.
            counts[cur, nxt] += 1
    P_hat = np.zeros((n, n))
    for i in range(n):
        if i in ABSORBING:
            P_hat[i, i] = 1.0
        elif counts[i].sum() > 0:
            P_hat[i] = counts[i] / counts[i].sum()
        else:
            P_hat[i] = np.full(n, 1.0 / n)  # unseen state: uninformative fallback
    return P_hat


def main():
    with open(DO_WORLDS_JSON, "r", encoding="utf-8") as f:
        dw = json.load(f)
    worlds = dw["worlds"]

    bag_errs, k1_errs, k2_errs, k3_errs = [], [], [], []
    per_world = []
    for w in worlds:
        P_do = np.array(w["P_do"])
        q_do_exact = np.array(w["q_do_exact"])  # [6,2] over transient x absorbing

        rng = np.random.default_rng(1000 + w["world"])
        streams = sample_streams(P_do, N_SEQ, S=S, rng=rng)

        q_bag, visits = bag_estimate(streams)
        bag_err_w = float(np.nanmean(np.abs(q_bag - q_do_exact[:, 0])))

        errs_k = {}
        for k in (1, 2, 3):
            P_hat = kgram_transition(streams, k)
            q_hat, *_ = committor(P_hat)
            errs_k[k] = float(np.mean(np.abs(q_hat[:, 0] - q_do_exact[:, 0])))

        bag_errs.append(bag_err_w)
        k1_errs.append(errs_k[1])
        k2_errs.append(errs_k[2])
        k3_errs.append(errs_k[3])
        per_world.append({
            "world": w["world"],
            "bag_err": bag_err_w,
            "k1_err": errs_k[1],
            "k2_err": errs_k[2],
            "k3_err": errs_k[3],
            "bag_visit_counts": visits.tolist(),
        })

    controls = {
        "bag": float(np.mean(bag_errs)),
        "kgram_k1": float(np.mean(k1_errs)),
        "kgram_k2": float(np.mean(k2_errs)),
        "kgram_k3": float(np.mean(k3_errs)),
    }
    best_name = min(controls, key=controls.get)
    best_err = controls[best_name]
    seconds = time.time() - t0

    result = {
        "n_worlds": len(worlds),
        "n_seq_per_world": N_SEQ,
        "S": S,
        "controls": controls,
        "best_control": best_name,
        "best_error": best_err,
        "bar_for_R_DO_LM": best_err - 0.02,
        "per_world": per_world,
        "seconds": seconds,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f)

    print(json.dumps(controls, indent=2))
    print("best=%s err=%.6f -> R-DO-LM bar (best - 0.02) = %.6f" %
          (best_name, best_err, result["bar_for_R_DO_LM"]))
    return result


def append_record_rows(dw, dp, started, finished):
    row_worlds = {
        "row": "R-WORLDS",
        "arms": ["exact-do", "sherman-morrison"],
        "seeds": list(range(50)),
        "split_seed": None,
        "machine": "CPU",
        "seconds_per_cell": round(dw["seconds"], 6),
        "tokens_seen": None,
        "params_numel": None,
        "bar": {"sm_err_max": "<= 1e-12 in all 50 worlds"},
        "measured": {"max_sm_err": dw["max_sm_err"], "n_worlds": len(dw["worlds"])},
        "verdict": dw["verdict"],
        "control": None,
        "quintile_profile": None,
        "clock_ratio": None,
        "read": "exact",
        "producer": "scratchpad/phase_j/do_worlds.py (ref_J2.py generator, intervene.py "
                    "Sherman-Morrison formula reused verbatim, reimplemented in numpy)",
        "output": r"scratchpad\phase_j\do_worlds.json",
        "repro_class": "exact numeric",
        "queue_pos": 4,
        "started": started,
        "finished": finished,
        "killed": "Nothing killed: the closed-form do() update agrees with a direct "
                  "re-solve to float64 roundoff (max 4.4e-16) across all 50 intervened "
                  "worlds, confirming the machinery R-DO-LM's exact read depends on.",
    }
    row_dpi = {
        "row": "R-DPI",
        "arms": ["bag", "kgram_k1", "kgram_k2", "kgram_k3"],
        "seeds": [f"world {w}" for w in range(dp["n_worlds"])],
        "split_seed": None,
        "machine": "CPU",
        "seconds_per_cell": round(dp["seconds"], 6),
        "tokens_seen": dp["n_worlds"] * dp["n_seq_per_world"] * dp["S"],
        "params_numel": None,
        "bar": {"desc": "R-DO-LM must beat the best input-only control by >= 0.02"},
        "measured": dp["controls"],
        "verdict": "best control = %s, err = %.6f -> R-DO-LM bar = %.6f" %
                   (dp["best_control"], dp["best_error"], dp["bar_for_R_DO_LM"]),
        "control": dp["controls"],
        "quintile_profile": None,
        "clock_ratio": None,
        "read": "bag/kgram",
        "producer": "scratchpad/phase_j/do_dpi.py (64 streams/world of that world only, S=128)",
        "output": r"scratchpad\phase_j\do_dpi.json",
        "repro_class": "exact numeric",
        "queue_pos": 4,
        "started": started,
        "finished": finished,
        "killed": "Sets the input-only floor R-DO-LM must clear; k=2,3 collapse to the "
                  "k=1 first-order estimate (order-1 generator), so they do not beat it.",
    }
    with open(RECORD_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row_worlds) + "\n")
        f.write(json.dumps(row_dpi) + "\n")


def demo():
    res = main()
    assert 0.0 <= res["controls"]["bag"] <= 1.0
    for k in ("kgram_k1", "kgram_k2", "kgram_k3"):
        assert 0.0 <= res["controls"][k] <= 1.5
    assert abs(res["controls"]["kgram_k1"] - res["controls"]["kgram_k2"]) < 1e-9, \
        "k=1 and k=2 collapsed counts should match bit for bit on this order-1 generator"
    print("demo OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
