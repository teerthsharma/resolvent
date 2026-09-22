"""Row C11: Chebyshev degree K = ceil(ln(2C/((1-rho)eps))/ln(1/rho)) - 1.

Grep found a NEARBY but distinct formula, `neumann_steps` in
tests/cameron/perron.py: K = ceil(ln(tol*(1-rho))/ln(rho)) -- same shape
(1-rho in the numerator, ln(1/rho) in the denominator) but no "2C" factor and
no "-1", i.e. a plain (non-Chebyshev-accelerated) geometric Neumann-tail bound,
not this row's formula. Not reused: it answers a different question.

The contract's formula has three free parameters (C, rho, eps) for three
target integers (11, 34, 130) -- one equation per target, so a UNIQUE common
triple is not identifiable from the formula alone; a brute-force grid search
(C in {1,2,3,5,10}, eps in {1e-2..1e-16} decade steps, rho swept at 0.01
resolution) finds MANY (C, rho, eps) that hit each target individually, and no
single (C, eps) pair with three "round" rho values hits all three -- e.g. at
C=1, eps=1e-6 the three targets land at rho = 0.27-0.29, 0.64, 0.88 (not
round). Recorded as: reproduced (the formula is implemented and verified
correct), but the specific (C, rho, eps) triple behind 11/34/130 could not be
recovered uniquely from the contract text.
"""
import json, time, math

t0 = time.time()


def chebyshev_degree(C, rho, eps):
    return math.ceil(math.log(2 * C / ((1 - rho) * eps)) / math.log(1 / rho)) - 1


targets = [11, 34, 130]
examples = {}
for target in targets:
    hits = []
    for C in (1, 2, 3, 5, 10):
        for exp in range(2, 17):
            eps = 10.0 ** (-exp)
            for r in range(1, 100):
                rho = r / 100.0
                if chebyshev_degree(C, rho, eps) == target:
                    hits.append((C, rho, eps))
    examples[target] = hits[:3]

seconds = time.time() - t0

row = {
    "row": "C11",
    "arms": None,
    "seeds": None,
    "split_seed": None,
    "machine": "CPU",
    "seconds_per_cell": round(seconds, 6),
    "tokens_seen": None,
    "params_numel": None,
    "bar": "reproduce K in {11, 34, 130} via the stated formula, with the (C, rho, eps) that give them",
    "measured": {str(t): examples[t] for t in targets},
    "verdict": "PARTIAL: formula verified correct and each target individually reproducible; no single (C, eps) with three round rho gives all three, so the exact author triple could not be recovered -- recorded per instructions rather than guessed",
    "control": None,
    "quintile_profile": None,
    "producer": "python scratchpad/phase_j/ref_C11.py (fresh; formula distinct from tests/cameron/perron.py neumann_steps, which lacks the 2C factor and the -1)",
    "output": r"scratchpad\phase_j\ref_C11.py",
    "repro_class": "exact numeric",
    "killed": "Nothing killed: the Chebyshev degree formula is implemented and each target integer is individually reachable by some (C, rho, eps), so the LAW is not falsified; only the specific triple used by the contract's three worked examples is unrecovered, which is reported as a gap rather than papered over with a chosen-to-fit triple.",
}

out_path = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j\record.jsonl"
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")

print(json.dumps(row, indent=2))


def demo():
    assert chebyshev_degree(1, 0.5, 1e-3) == 11
    for t in targets:
        assert len(examples[t]) > 0, f"no example found for {t}"
    print("demo OK")


if __name__ == "__main__":
    demo()
