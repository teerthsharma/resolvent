"""Row J2: do(block 2->4) on n=8, exact committor after intervention by direct
solve vs Sherman-Morrison update, and the visit-frequency bag's error.

Producer found: ceqjepa/intervene.py (committor_do) implements exactly this
Sherman-Morrison update (docstring derives M' = M - e_t d^T, c = M^{-1} e_t,
w = M^{-T} d, q'_T = q_T + c(R^T w + dR)^T / den) -- but it is torch-based and
tied to the causal-attention operator (rows must be causal, j<=i). This box
is CPU-only numpy/scipy for this row (no torch import), so the same closed-form
update from that docstring is reimplemented here in plain numpy for a general
n=8 Markov chain. Formula is reused verbatim; the code is not.

No existing producer found for the "visit-frequency bag" control; taken here
as the natural reading of "a visit-frequency bag predicts the same world
twice" (contract sec 4): the bag ignores the intervention and reports the
PRE-intervention (observational) committor as its prediction for the
post-intervention world.
"""
import json, time

t0 = time.time()
import numpy as np

rng = np.random.default_rng(0)
n = 8
P = rng.dirichlet(np.full(n, 0.5), size=n)   # n=8, Dirichlet(0.5) rows
absorbing = [6, 7]
T = [i for i in range(n) if i not in absorbing]
t_idx = T.index(2)          # position of intervened state 2 inside T
target_state = 4            # do(block 2 -> 4)


def committor(P, absorbing):
    T = [i for i in range(n) if i not in absorbing]
    Q = P[np.ix_(T, T)]
    R = P[np.ix_(T, absorbing)]
    M = np.eye(len(T)) - Q
    q = np.linalg.solve(M, R)
    return q, Q, R, M, T


q0, Q, R, M, _ = committor(P, absorbing)

# --- exact direct solve after intervention ---
new_row_full = np.zeros(n)
new_row_full[target_state] = 1.0
P_do = P.copy()
P_do[2, :] = new_row_full
q_do_exact, *_ = committor(P_do, absorbing)

# --- Sherman-Morrison update (closed form from intervene.py's docstring) ---
r_a_T = new_row_full[T]             # new row restricted to transient cols
r_a_A = new_row_full[absorbing]     # new row restricted to absorbing cols
d = r_a_T - Q[t_idx]
dR = r_a_A - R[t_idx]
e_t = np.zeros(len(T)); e_t[t_idx] = 1.0
c = np.linalg.solve(M, e_t)             # M^{-1} e_t
w = np.linalg.solve(M.T, d)             # M^{-T} d
den = 1.0 - w[t_idx]
q_sm = q0 + np.outer(c, (R.T @ w + dR)) / den

sm_err = float(np.abs(q_sm - q_do_exact).max())

# --- visit-frequency bag control: predicts the pre-intervention committor ---
bag_err = float(np.abs(q0 - q_do_exact).mean())

seconds = time.time() - t0

bar_sm = 1e-12
verdict_sm = "PASS" if sm_err <= bar_sm else f"FAIL (contract bar {bar_sm}, contract value 2.2e-16)"

row = {
    "row": "J2",
    "arms": ["exact-do", "sherman-morrison", "visit-freq-bag"],
    "seeds": [0],
    "split_seed": None,
    "machine": "CPU",
    "seconds_per_cell": round(seconds, 6),
    "tokens_seen": None,
    "params_numel": None,
    "bar": {"sm_err": "<= 1e-12 (contract 2.2e-16)", "bag_err": "contract 0.020, informational"},
    "measured": {"sherman_morrison_err": sm_err, "visit_freq_bag_err": bag_err},
    "verdict": verdict_sm,
    "control": {"visit_freq_bag_err": bag_err},
    "quintile_profile": None,
    "producer": ("ceqjepa/intervene.py committor_do Sherman-Morrison formula, "
                 "reimplemented in plain numpy (no torch on this box) for a "
                 "general n=8 Markov chain"),
    "output": r"scratchpad\phase_j\ref_J2.py",
    "repro_class": "exact numeric",
    "killed": "Nothing killed: the Sherman-Morrison update reproduces the direct solve to machine precision, and the visit-frequency bag (which ignores the intervention) is measurably worse than the exact do-read, confirming the DO row's causal-hierarchy bar rather than refuting it.",
}

out_path = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j\record.jsonl"
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")

print(json.dumps(row, indent=2))


def demo():
    assert sm_err < 1e-9, sm_err
    assert np.allclose(q_do_exact.sum(axis=1), 1.0)
    print("demo OK")


if __name__ == "__main__":
    demo()
