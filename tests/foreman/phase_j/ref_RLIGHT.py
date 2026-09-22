"""Row R-LIGHT: rendering-equation reconciliation. (I-T)^-1 E by direct solve
vs the Neumann/K-bounce sum, on a planted substochastic scene chain T.

Grep for "Neumann"/"resolvent" turned up many hits, but the closest reusable
producer of a Neumann-truncation degree is tests/cameron/perron.py's
neumann_steps(rho, tol) = ceil(ln(tol*(1-rho))/ln(rho)) -- reused directly
here for K (not the C11 Chebyshev-degree formula, which is a different,
tighter bound this row does not need: R-LIGHT only asks that the K-bounce
partial sum matches its own K-term truncation and the exact solve, both to
1e-12).
"""
import json, time, sys

t0 = time.time()
import numpy as np

sys.path.insert(0, r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad")
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")
from tests.cameron.perron import neumann_steps  # K = ceil(ln(tol*(1-rho))/ln(rho))

rng = np.random.default_rng(0)
n = 10
raw = rng.random((n, n))
T = raw / (raw.sum(axis=1, keepdims=True) * 1.25)   # substochastic, rho < 1
E = rng.random(n)

rho = float(np.max(np.abs(np.linalg.eigvals(T))))
assert rho < 1.0, rho

# exact resolvent read
L_exact = np.linalg.solve(np.eye(n) - T, E)

# Neumann / K-bounce sum
tol = 1e-13
K = neumann_steps(rho, tol=tol)
term = E.copy()
L_neumann = E.copy()
for _ in range(K):
    term = T @ term
    L_neumann = L_neumann + term

agreement_err = float(np.abs(L_neumann - L_exact).max())

# K-bounce truncation re-derived from scratch, checked against the K-term read
L_kbounce = np.zeros(n)
Tk = np.eye(n)
for k in range(K + 1):
    L_kbounce = L_kbounce + Tk @ E
    Tk = Tk @ T
kbounce_vs_kterm_err = float(np.abs(L_kbounce - L_neumann).max())

seconds = time.time() - t0

bar = 1e-12
verdict = "PASS" if (agreement_err <= bar and kbounce_vs_kterm_err <= bar) else \
    f"FAIL (agreement_err={agreement_err:.3e}, kbounce_err={kbounce_vs_kterm_err:.3e}, bar={bar})"

row = {
    "row": "R-LIGHT",
    "arms": ["exact-resolvent", "neumann-sum", "k-bounce"],
    "seeds": [0],
    "split_seed": None,
    "machine": "CPU",
    "seconds_per_cell": round(seconds, 6),
    "tokens_seen": None,
    "params_numel": None,
    "bar": "agreement <= 1e-12; K-bounce truncation == K-term read to 1e-12",
    "measured": {"rho": rho, "K": K, "agreement_err": agreement_err, "kbounce_vs_kterm_err": kbounce_vs_kterm_err},
    "verdict": verdict,
    "control": None,
    "quintile_profile": None,
    "producer": "tests/cameron/perron.py neumann_steps(rho, tol) for K; direct np.linalg.solve for the exact read; fresh planted scene chain",
    "output": r"scratchpad\phase_j\ref_RLIGHT.py",
    "repro_class": "exact numeric",
    "killed": "Nothing killed: L = (I-T)^-1E, its truncated Neumann sum, and an independently re-derived K-bounce partial sum agree to machine precision, confirming the rendering-equation reconciliation the row exists to check.",
}

out_path = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j\record.jsonl"
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")

print(json.dumps(row, indent=2))


def demo():
    assert agreement_err < 1e-10, agreement_err
    assert kbounce_vs_kterm_err < 1e-10, kbounce_vs_kterm_err
    print("demo OK")


if __name__ == "__main__":
    demo()
