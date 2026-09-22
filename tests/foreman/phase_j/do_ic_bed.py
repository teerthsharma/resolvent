# -*- coding: utf-8 -*-
"""Row R-DO-IC-FLOORS. Input-only floors for the do-intervention committor
read, ahead of any arm training. Reuses do_worlds.py (J2 generator, 50 do()
worlds, exact q_do, sample_streams) unmodified -- copied logic, not edited.

train_chains(seed): 512 independent 8x8 chains (Dirichlet(0.5) rows, states
0..7, absorbing 6/7 structurally known and never used by committor()), from
default_rng(9000+seed). Their exact committors give the no-update fallback.

Eval protocol (shared with the arms row): 50 worlds x 16 streams x S=128,
stream k of world w from default_rng(5000 + 16*w + k) over that world's
P_do. Each estimator reads ONE stream and outputs q_hat per transient state
0..5 (P(hit absorbing state 6 first) = q_do_exact[:, 0]); error is the grand
mean of |q_hat - q_do_exact[:, 0]| over worlds, streams, states.
"""
import json
import os
import sys
import time

t0 = time.time()

PHASE_J = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j"
sys.path.insert(0, PHASE_J)
sys.path.insert(0, r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad")
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")

import numpy as np  # noqa: E402

from do_worlds import committor, sample_streams, TRANSIENT, ABSORBING, n as N_STATES  # noqa: E402

N_TRAIN_CHAINS = 512
S = 128
N_WORLDS = 50
N_STREAMS_PER_WORLD = 16
DIRICHLET_ALPHA = 0.5


def train_chains(seed):
    """512 fresh chains from default_rng(9000+seed); returns their exact
    committor P(hit state 6 first) per transient state, shape (512, 6)."""
    rng = np.random.default_rng(9000 + seed)
    qs = np.empty((N_TRAIN_CHAINS, len(TRANSIENT)))
    for c in range(N_TRAIN_CHAINS):
        P = rng.dirichlet(np.full(N_STATES, DIRICHLET_ALPHA), size=N_STATES)
        q, *_ = committor(P)
        qs[c] = q[:, 0]
    return qs


def solve_from_counts(counts, fallback, prior=0.0):
    """Row-normalize `counts` (with optional +prior pseudocount per cell,
    over all 8 targets, for transient rows only) and solve the committor.
    A transient row with zero total count is unseen: its row is filled
    uniformly (to keep the linear system well posed) and its final answer
    is overridden with `fallback` after solving, per the floor's rule."""
    P_hat = np.zeros((N_STATES, N_STATES))
    P_hat[ABSORBING[0], ABSORBING[0]] = 1.0
    P_hat[ABSORBING[1], ABSORBING[1]] = 1.0
    unseen = []
    for i in TRANSIENT:
        row = counts[i].astype(np.float64)
        if prior:
            row = row + prior
        total = row.sum()
        if total <= 0:
            P_hat[i] = 1.0 / N_STATES
            unseen.append(i)
        else:
            P_hat[i] = row / total
    q, *_ = committor(P_hat)
    q_hat = q[:, 0].copy()
    for i in unseen:
        q_hat[TRANSIENT.index(i)] = fallback[TRANSIENT.index(i)]
    return q_hat


def estimate_stream(stream, fallback):
    """Returns (q_no_update, q_1gram, q_dirichlet), each shape (6,)."""
    counts = np.zeros((N_STATES, N_STATES))
    for t in range(len(stream) - 1):
        counts[stream[t], stream[t + 1]] += 1

    q_no_update = fallback.copy()
    q_1gram = solve_from_counts(counts, fallback, prior=0.0)
    q_dirichlet = solve_from_counts(counts, fallback, prior=DIRICHLET_ALPHA)
    return q_no_update, q_1gram, q_dirichlet


def main():
    fallback = train_chains(0).mean(axis=0)  # (6,)

    with open(os.path.join(PHASE_J, "do_worlds.json"), encoding="utf-8") as f:
        worlds = json.load(f)["worlds"]
    assert len(worlds) == N_WORLDS

    errs = {"no_update": [], "onegram": [], "dirichlet": []}
    for w in worlds:
        P_do = np.array(w["P_do"])
        q_do_exact = np.array(w["q_do_exact"])[:, 0]  # (6,)
        wi = w["world"]
        for k in range(N_STREAMS_PER_WORLD):
            rng = np.random.default_rng(5000 + 16 * wi + k)
            stream = sample_streams(P_do, 1, S=S, rng=rng)[0]
            q_nu, q_1g, q_dir = estimate_stream(stream, fallback)
            errs["no_update"].append(np.abs(q_nu - q_do_exact))
            errs["onegram"].append(np.abs(q_1g - q_do_exact))
            errs["dirichlet"].append(np.abs(q_dir - q_do_exact))

    result = {
        "row": "R-DO-IC-FLOORS",
        "n_train_chains": N_TRAIN_CHAINS,
        "n_worlds": N_WORLDS,
        "n_streams_per_world": N_STREAMS_PER_WORLD,
        "S": S,
        "fallback_mean_committor": fallback.tolist(),
        "errors": {
            "no_update": float(np.mean(errs["no_update"])),
            "onegram_mle": float(np.mean(errs["onegram"])),
            "dirichlet_posterior_mean": float(np.mean(errs["dirichlet"])),
        },
    }
    result["strongest"] = min(result["errors"], key=result["errors"].get)
    result["verdict"] = "FLOORS SET"
    result["seconds"] = time.time() - t0

    with open(os.path.join(PHASE_J, "do_ic_floors.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    print("errors=%s strongest=%s seconds=%.2f" %
          (result["errors"], result["strongest"], result["seconds"]))
    return result


def demo():
    res = main()
    assert res["verdict"] == "FLOORS SET"
    assert set(res["errors"]) == {"no_update", "onegram_mle", "dirichlet_posterior_mean"}
    for v in res["errors"].values():
        assert 0.0 <= v <= 1.0
    assert res["strongest"] in res["errors"]
    print("demo OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
