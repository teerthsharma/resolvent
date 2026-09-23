# Foreman K1 candidate (d): does bed_k leak the root without following the chain? Bars: test_leak.py.
import json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/K0/cameron")
import bed_k  # noqa: E402

HERE = Path(__file__).parent
N = 4096
CELLS = 16 * 16 * 8 * 2 * 2 * 16


def planted(rng):
    """bed_k structure, but every id is congruent to its lane (rank of its root) mod 16: a leak by construction."""
    parent, depth, root = bed_k.make_bed(N, 16, rng)
    roots = np.flatnonzero(parent < 0)
    lane = np.searchsorted(roots, root)
    ids = np.empty(N, int)
    for c in range(16):
        sel = np.flatnonzero(lane == c)
        ids[sel] = rng.permutation(bed_k.V // 16)[:sel.size] * 16 + c
    return {"ids": ids, "pids": np.where(parent < 0, bed_k.NULL, ids[np.maximum(parent, 0)]), "parent": parent,
            "depth": depth, "root": root}


def cells(b):
    p, ids, pids = b["parent"], b["ids"], b["pids"]
    t = np.arange(N)
    roots = np.flatnonzero(p < 0)
    pb = np.minimum(t // 256, 15)
    gb = np.minimum(np.where(p >= 0, t - p, 0), 63) // 8
    il, pl, rl = ids & 15, pids & 15, ids[roots] & 15
    k = np.arange(roots.size)
    f = ((((k[None, :] * 16 + pb[:, None]) * 8 + gb[:, None]) * 2 + (il[:, None] == rl[None, :])) * 2
         + (pl[:, None] == rl[None, :])) * 16 + rl[None, :]
    return f, roots, (roots[None, :] == b["root"][:, None])


def probe(fit, ev):
    pos, tot = np.zeros(CELLS), np.zeros(CELLS)
    for b in fit:
        f, _, y = cells(b)
        np.add.at(tot, f.ravel(), 1)
        np.add.at(pos, f.ravel(), y.ravel())
    P = (pos + 1) / (tot + 16)
    ok, d = [], []
    for b in ev:
        f, roots, _ = cells(b)
        ok.append(roots[P[f].argmax(1)] == b["root"])
        d.append(b["depth"])
    return np.concatenate(ok), np.concatenate(d)


def main():
    fit = [bed_k.make_test(np.random.default_rng([71, k]), N) for k in range(40)]
    ev = [bed_k.make_test(np.random.default_rng([72, k]), N) for k in range(8)]
    ok, d = probe(fit, ev)
    pok, pd = probe([planted(np.random.default_rng([73, k])) for k in range(40)],
                    [planted(np.random.default_rng([74, k])) for k in range(8)])
    out = {"real": {"acc_deep16": float(ok[d > 16].mean()), "n_deep16": int((d > 16).sum()),
                    "acc_deep128": float(ok[d > 128].mean()), "n_deep128": int((d > 128).sum()),
                    "acc_all": float(ok.mean())},
           "planted": {"acc_deep16": float(pok[pd > 16].mean()), "n_deep16": int((pd > 16).sum())},
           "ids_distinct": bool(all(len(np.unique(b["ids"])) == N for b in fit + ev))}
    (HERE / "leak.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
