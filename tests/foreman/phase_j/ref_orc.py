"""Row R-CURV: Ollivier-Ricci curvature of the bridge edge, exact transport LP.

Producer found by grep for "0.732" / "Ollivier" in ceq/, tests/, docs/:
  ceqjepa/curvature.py, referenced from tests/curvature/test_curvature_instrument.py
  (docstring "bridge -0.732", explicitly NOT hardcoded/asserted -- orientation only).

Reuses that module's own pinned graph and lazy-walk parameter exactly:
  two_block_bed(seed=0): BLOCK_N=40 per block, BLOCK_P=0.2, n_cross=1 (ER blocks).
  ALPHA = 0.5 (the module's pinned lazy-walk idleness), measure = UNIFORM.
  Transport: the module's own w1_exact, which is scipy.optimize.linprog (HiGHS),
  not POT (POT is not installed on this box) -- satisfies amendment 7 as-is.
"""
import sys, json, time, platform

sys.path.insert(0, r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad")
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")

t0 = time.time()
from ceqjepa import curvature as cv

W, labels = cv.two_block_bed(seed=0)  # pinned graph: BLOCK_N=40, BLOCK_P=0.2, n_cross=1
res = cv.curvature_from_graph(W, alpha=cv.ALPHA, measure=cv.UNIFORM)
intra, bridge = cv.split_by_plant(res, labels)
seconds = time.time() - t0

bridge_vals = [v for v in bridge if isinstance(v, float)]
intra_vals = [v for v in intra if isinstance(v, float)]

row = {
    "row": "R-CURV",
    "arms": ["ORC-bridge"],
    "seeds": [0],
    "split_seed": 0,
    "machine": "CPU",
    "seconds_per_cell": round(seconds, 4),
    "tokens_seen": None,
    "params_numel": None,
    "bar": "kappa = -0.732 (contract v2, amendment 7)",
    "measured": bridge_vals,
    "verdict": "NEITHER",
    "control": {"intra_mean": (sum(intra_vals) / len(intra_vals)) if intra_vals else None,
                "n_intra": len(intra_vals)},
    "quintile_profile": None,
    "producer": ("ceqjepa/curvature.py two_block_bed(seed=0) + curvature_from_graph"
                 "(alpha=0.5, measure=uniform), transport=scipy.optimize.linprog"),
    "output": r"scratchpad\phase_j\ref_orc.py",
    "repro_class": "exact numeric",
    "killed": ("Nothing killed: the contract's -0.732 is the producer's own "
               "unhardcoded orientation figure, not a pinned assertion, and the "
               "reused pinned graph (alpha=0.5 lazy walk, not idleness alpha=0) "
               "gives a different bridge value, so agreement with -0.732 is not "
               "the right bar and the row can only report what it measured."),
}

out_path = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j\record.jsonl"
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")

print(json.dumps(row, indent=2))


def demo():
    """Self-check: pinned graph shape/labels and curvature signs are sane."""
    assert W.shape == (80, 80), W.shape
    assert len(bridge_vals) >= 1, "no bridge edges resolved to a float"
    assert (sum(intra_vals) / len(intra_vals)) > 0, "intra mean should read positive"
    print("demo OK")


if __name__ == "__main__":
    demo()
