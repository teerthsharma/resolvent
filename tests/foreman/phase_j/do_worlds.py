"""Row R-WORLDS: the J2 bed (n=8, Dirichlet(0.5) rows, default_rng(0), basins
6/7 -- identical generator to ref_J2.py) plus 50 do() intervention worlds.

World w (default_rng(100+w)) picks a transient state i uniformly from
{0..5} and replaces P[i] with a fresh Dirichlet(0.5) row over all 8 states.
q_do is computed two ways -- direct solve on the intervened matrix, and the
Sherman-Morrison rank-1 update whose closed form is intervene.py's docstring
(same formula ref_J2.py already reimplements in plain numpy; reused verbatim
here, not re-derived) -- and the two must agree to <= 1e-12.

Also defines sample_streams(P, n_seq, S, rng): n_seq trajectories of length S
each, token = state id 0..7, starting from a uniformly random transient state,
restarting from a fresh random transient state immediately after any
absorbing token is written.
"""
import json
import sys
import time

t0 = time.time()
import numpy as np

n = 8
ABSORBING = [6, 7]
TRANSIENT = [i for i in range(n) if i not in ABSORBING]  # 0..5


def committor(P, absorbing=ABSORBING):
    T = [i for i in range(n) if i not in absorbing]
    Q = P[np.ix_(T, T)]
    R = P[np.ix_(T, absorbing)]
    M = np.eye(len(T)) - Q
    q = np.linalg.solve(M, R)
    return q, Q, R, M, T


def committor_do_sm(q0, Q, R, M, T, i, new_row_full):
    """Sherman-Morrison update for do(i -> new_row_full). Formula reused
    verbatim from ceqjepa/intervene.py's committor_do docstring / ref_J2.py."""
    t_idx = T.index(i)
    r_T = new_row_full[T]
    r_A = new_row_full[ABSORBING]
    d = r_T - Q[t_idx]
    dR = r_A - R[t_idx]
    e_t = np.zeros(len(T)); e_t[t_idx] = 1.0
    c = np.linalg.solve(M, e_t)
    w = np.linalg.solve(M.T, d)
    den = 1.0 - w[t_idx]
    q_sm = q0 + np.outer(c, (R.T @ w + dR)) / den
    return q_sm, den


def sample_streams(P, n_seq, S=128, rng=None):
    """n_seq token streams of length S. Token = state id 0..7. Trajectories
    start from a uniformly random transient state; on writing an absorbing
    token the next token restarts from a fresh uniformly random transient
    state (same stream, concatenated) until length S is reached."""
    if rng is None:
        rng = np.random.default_rng()
    nT = len(TRANSIENT)
    out = np.empty((n_seq, S), dtype=np.int64)
    for s in range(n_seq):
        cur = TRANSIENT[rng.integers(nT)]
        for t in range(S):
            out[s, t] = cur
            if cur in ABSORBING:
                cur = TRANSIENT[rng.integers(nT)]
            else:
                cur = rng.choice(n, p=P[cur])
    return out


def main():
    rng0 = np.random.default_rng(0)
    P = rng0.dirichlet(np.full(n, 0.5), size=n)
    q0, Q, R, M, T = committor(P)
    assert T == TRANSIENT

    worlds = []
    max_sm_err = 0.0
    for w in range(50):
        rngw = np.random.default_rng(100 + w)
        i = int(TRANSIENT[rngw.integers(len(TRANSIENT))])
        new_row = rngw.dirichlet(np.full(n, 0.5))
        P_do = P.copy()
        P_do[i, :] = new_row

        q_do_exact, *_ = committor(P_do)
        q_do_sm, den = committor_do_sm(q0, Q, R, M, T, i, new_row)
        sm_err = float(np.abs(q_do_sm - q_do_exact).max())
        max_sm_err = max(max_sm_err, sm_err)

        worlds.append({
            "world": w,
            "intervened_state": i,
            "new_row": new_row.tolist(),
            "P_do": P_do.tolist(),
            "q_do_exact": q_do_exact.tolist(),
            "sm_den": float(den),
            "sm_err": sm_err,
        })

    seconds = time.time() - t0
    result = {
        "n": n,
        "absorbing": ABSORBING,
        "transient": TRANSIENT,
        "P": P.tolist(),
        "q_observational": q0.tolist(),
        "worlds": worlds,
        "max_sm_err": max_sm_err,
        "verdict": "PASS" if max_sm_err <= 1e-12 else "FAIL",
        "seconds": seconds,
    }

    out_json = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j\do_worlds.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("max_sm_err=%.3e verdict=%s worlds=%d seconds=%.3f" %
          (max_sm_err, result["verdict"], len(worlds), seconds))
    return result


def demo():
    res = main()
    assert res["verdict"] == "PASS", res["max_sm_err"]
    assert len(res["worlds"]) == 50
    q = np.array(res["q_observational"])
    assert np.allclose(q.sum(axis=1), 1.0)
    # quick sample_streams smoke check
    P = np.array(res["P"])
    rng = np.random.default_rng(1)
    s = sample_streams(P, 4, S=32, rng=rng)
    assert s.shape == (4, 32)
    assert s.min() >= 0 and s.max() < n
    print("demo OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
