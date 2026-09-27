"""Daedalus beds (exact truth) and V2 automatic floors.

Every bed returns (X[N,T] int64, Y[N,T] int64, band[T] bool). Truth is computed
here, in the verifier's process; a candidate only ever sees X (and train Y).
Test draws use seeds derived from sealed/secret.json, so a hard-coded output
table cannot match a fresh draw.

Floors are Bayes plug-in predictors fitted on the train draw and scored on the
eval draw over the scored band: majority, position-only, window-w (w=1..4),
commutative (multiset of the prefix), random encoder (frozen random GRU +
trained linear readout). Bed-specific constructions (recency on X3, the Phase K
window library on pointer chase) live beside them.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DAEDALUS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(DAEDALUS))
PHASE_J = os.path.join(REPO, "tests", "foreman", "phase_j")
N1_CHASE = os.path.join(PHASE_J, "N1", "chase")
SEALED = os.path.join(DAEDALUS, "sealed")


def secret_seed(tag: str) -> int:
    """Seed for sealed draws: sha256(salt || tag). The salt never leaves sealed/."""
    with open(os.path.join(SEALED, "secret.json")) as f:
        salt = json.load(f)["salt"]
    return int.from_bytes(hashlib.sha256((salt + "|" + tag).encode()).digest()[:4], "little")


# ------------------------------------------------------------------ A5 word problem
def _a5():
    sys.path.insert(0, PHASE_J)
    import a5_bed  # reused unedited: build_a5, make_generators (tests/foreman/phase_j/a5_bed.py:68-84)
    elems, index, ident = a5_bed.build_a5()
    gens = a5_bed.make_generators()
    comp = a5_bed._perm_compose
    # table[g, e] = index(g * elems[e])
    table = np.array([[index[comp(g, e)] for e in elems] for g in gens], dtype=np.int64)
    return table, index[ident]


_A5 = None


def a5_word(n: int, T: int, seed: int, band_from: int = 8):
    """A5 word problem: 4 generators, label = index of the running product.
    Seeds are explicit ints (a5_bed.gen_split used hash((name, seed)), which is
    salted per process -- RECORD_L.md F4)."""
    global _A5
    if _A5 is None:
        _A5 = _a5()
    table, ident = _A5
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, T))
    Y = np.empty((n, T), dtype=np.int64)
    cur = np.full(n, ident)
    for t in range(T):
        cur = table[X[:, t], cur]
        Y[:, t] = cur
    band = np.arange(T) >= band_from
    return X.astype(np.int64), Y, band


BEDS = {
    # name: (generator, vocab, n_classes)
    "a5_word_T16": (lambda n, seed: a5_word(n, 16, seed), 4, 60),
}


# ------------------------------------------------------------------ X3 reset bed (Addendum N)
def x3_recency(n: int = 4096, density: float = 0.05, seed: int = 7, slopes=(0.05, 0.2, 1.0)):
    """The X3 reset bed and its recency construction, imported unedited from the
    N1 Chase rebuild (tests/foreman/phase_j/N1/chase/n_impl.py:485-541)."""
    sys.path.insert(0, N1_CHASE)
    import n_impl as NI
    rng = np.random.default_rng(seed)
    grp = NI.group2I()
    tok = NI.reset_seq(n, density, rng)
    tq = NI.truth(tok, NI.GENS, grp)
    return {"resets": int((tok == 0).sum()), "n": n,
            "recency_acc": {str(s): float(NI.recency_acc(tok, s, grp, tq)) for s in slopes}}


# ------------------------------------------------------------------ generic floors
def _plugin(keys_tr, y_tr, keys_ev, y_ev, fallback):
    table = defaultdict(Counter)
    for k, y in zip(keys_tr, y_tr):
        table[k][y] += 1
    hit = 0
    for k, y in zip(keys_ev, y_ev):
        pred = table[k].most_common(1)[0][0] if k in table else fallback
        hit += pred == y
    return hit / max(1, len(y_ev))


def floors(Xtr, Ytr, Xev, Yev, band, windows=(1, 2, 3, 4), random_encoder=True):
    """All automatic floors, scored on the band. Returns {name: acc}."""
    pos = np.flatnonzero(band)
    ytr_b, yev_b = Ytr[:, pos], Yev[:, pos]
    maj = Counter(ytr_b.ravel().tolist()).most_common(1)[0][0]
    out = {"majority": float((yev_b == maj).mean())}

    def keyed(fn):
        ktr, ytr, kev, yev = [], [], [], []
        for t in pos:
            ktr += [fn(Xtr[i], t) for i in range(len(Xtr))]
            ytr += Ytr[:, t].tolist()
            kev += [fn(Xev[i], t) for i in range(len(Xev))]
            yev += Yev[:, t].tolist()
        return _plugin(ktr, ytr, kev, yev, maj)

    out["position_only"] = keyed(lambda x, t: t)
    for w in windows:
        out[f"window_{w}"] = keyed(lambda x, t, w=w: (t, tuple(x[max(0, t - w + 1): t + 1])))
    out["commutative"] = keyed(lambda x, t: (t, tuple(np.bincount(x[: t + 1], minlength=int(Xtr.max()) + 1))))
    if random_encoder:
        out["random_encoder"] = _random_encoder(Xtr, Ytr, Xev, Yev, band)
    return {k: float(v) for k, v in out.items()}


def _random_encoder(Xtr, Ytr, Xev, Yev, band, hidden=64, steps=300, seed=0):
    """Frozen random embedding + frozen random GRU, trained linear readout only."""
    import torch
    g = torch.Generator().manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V, C = int(max(Xtr.max(), Xev.max())) + 1, int(max(Ytr.max(), Yev.max())) + 1
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(V, hidden).to(dev).requires_grad_(False)
    gru = torch.nn.GRU(hidden, hidden, batch_first=True).to(dev).requires_grad_(False)
    head = torch.nn.Linear(hidden, C).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    pos = torch.tensor(np.flatnonzero(band), device=dev)
    xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(Ytr, device=dev)
    with torch.no_grad():
        htr = gru(emb(xt))[0][:, pos]
        hev = gru(emb(torch.tensor(Xev, device=dev)))[0][:, pos]
    ytb = yt[:, pos]
    for _ in range(steps):
        idx = torch.randint(0, len(htr), (512,), generator=g).to(dev)
        loss = torch.nn.functional.cross_entropy(head(htr[idx]).reshape(-1, C), ytb[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(hev).argmax(-1).cpu().numpy()
    return float((pred == Yev[:, np.flatnonzero(band)]).mean())


if __name__ == "__main__":
    # self-check: A5 labels are a valid running product; window floors near chance on the far band;
    # an abelian bed (sum mod 7) is solved exactly by the commutative floor.
    Xtr, Ytr, band = a5_word(4000, 16, 1)
    Xev, Yev, _ = a5_word(1000, 16, 2)
    assert Ytr.max() < 60 and band.sum() == 8
    f = floors(Xtr, Ytr, Xev, Yev, band, random_encoder=False)
    assert max(f.values()) < 0.5, f
    rng = np.random.default_rng(0)
    Xa = rng.integers(0, 4, (3000, 12)); Ya = np.cumsum(Xa, 1) % 7
    fa = floors(Xa[:2000], Ya[:2000], Xa[2000:], Ya[2000:], np.arange(12) >= 6, random_encoder=False)
    assert fa["commutative"] > 0.95, fa  # < 1 only from eval keys unseen in train
    print("beds self-check OK", {k: round(v, 3) for k, v in f.items()})
