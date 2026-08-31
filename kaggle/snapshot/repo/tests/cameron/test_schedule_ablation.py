"""What is the cheapest schedule that beats the 0D-salience control?

THEORY.md section 3 calls the hierarchical near-field/far-field block partition
"the single genuinely new thing", with the merged 0D-persistence salience
builder as "the control to beat".

This measures all of them on REAL attention from a pretrained model, at matched
block budget, against the tda-tdd oracle: the same-budget top-k selection made
with the dense scores in hand. Position on the random->oracle axis is the number
that decides whether the topology did any work.

Schedules compared, all causal, all at exactly B key-blocks per query block:
  random          - null model
  window+sinks    - newest B-1 blocks plus block 0 (StreamingLLM shape)
  salience-0D     - top-B by 0D-persistence of key-block centroids (the control)
  hierarchical    - near-field dense + dyadically dilated far-field (H-matrix)
  oracle top-k    - top-B by true attention mass (upper bound, unimplementable)
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

MODEL = os.environ.get("CAMERON_MODEL", "HuggingFaceTB/SmolLM2-135M")
BLK = 32
SEQ = 1024          # 32 key-blocks
BUDGET = 6          # 6 of up to 32 blocks == 18.75% of the causal budget
LAYERS = [2, 7, 14, 21, 28]
BUDGETS = [4, 6, 10]


# ------------------------------------------------------------- 0D persistence

def persistence_0d(points: np.ndarray) -> np.ndarray:
    """0D persistent homology of a point cloud, single-linkage + elder rule.

    Every point is born at 0; when two components merge the younger one dies at
    the merge distance. Returns each point's death radius (inf for the root).
    """
    n = len(points)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    pairs = sorted(((d[i, j], i, j) for i in range(n) for j in range(i + 1, n)))
    parent = list(range(n))
    birth_rank = list(range(n))  # lower index == elder

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    death = np.full(n, np.inf)
    for dist, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        elder, younger = (ri, rj) if birth_rank[ri] <= birth_rank[rj] else (rj, ri)
        death[younger] = dist
        parent[younger] = elder
    return death


def test_persistence_0d_obeys_the_elder_rule():
    """tda-tdd algebraic contract: n points give exactly n-1 finite deaths."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(12, 4))
    death = persistence_0d(pts)
    assert np.isfinite(death).sum() == len(pts) - 1


def test_persistence_0d_separates_known_clusters():
    """Known-ground-truth control: 3 well-separated clusters leave 3 long bars."""
    rng = np.random.default_rng(1)
    pts = np.concatenate([rng.normal(c, 0.05, size=(6, 2))
                          for c in ([0, 0], [10, 0], [0, 10])])
    death = persistence_0d(pts)
    longest = np.sort(death[np.isfinite(death)])[-2:]
    assert (longest > 5).all(), f"cluster gaps not detected: {longest}"


# ------------------------------------------------------------- real attention

@pytest.fixture(scope="module")
def attention_mass():
    """Block-level attention mass from a real model on a retrieval prompt."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, attn_implementation="eager",
            dtype=torch.float32)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{MODEL} unavailable offline: {exc}")
    model.eval()

    filler = ("The archive stores routine maintenance logs and shipping "
              "manifests for the northern depot, updated each quarter by the "
              "duty clerk on rotation. ")
    prompt = (filler * 4
              + "The access code for the vault is 74195. "
              + filler * 60
              + "Question: what is the access code for the vault? Answer:")
    ids = tok(prompt, return_tensors="pt").input_ids[:, :SEQ]
    assert ids.shape[1] == SEQ, f"prompt only {ids.shape[1]} tokens, need {SEQ}"
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)

    nb = SEQ // BLK
    att = {L: out.attentions[L][0].mean(0).double().numpy()  # heads averaged
           for L in LAYERS}
    # hidden state entering each layer: what its block centroids summarise
    hidden = {L: out.hidden_states[L][0].double().numpy() for L in LAYERS}
    return dict(nb=nb, hidden=hidden, attentions=att,
                n_layers=len(out.attentions))


# ------------------------------------------------------------------ schedules

def _pad(sel, budget):
    sel = list(dict.fromkeys(sel))[:budget]
    return sel


def schedule_random(qb, nb, budget, rng, **_):
    allowed = list(range(qb + 1))
    rng.shuffle(allowed)
    return _pad(allowed, budget)


def schedule_window_sinks(qb, nb, budget, **_):
    return _pad([0] + list(range(qb, max(-1, qb - budget), -1)), budget)


def schedule_salience(qb, nb, budget, persistence=None, **_):
    allowed = np.arange(qb + 1)
    p = np.where(np.isfinite(persistence[allowed]), persistence[allowed],
                 np.nanmax(persistence[np.isfinite(persistence)]) * 2)
    order = allowed[np.argsort(-p)]
    return _pad([qb] + list(order), budget)


def schedule_hierarchical(qb, nb, budget, **_):
    """Near-field dense, far-field dyadically dilated: the H-matrix shape."""
    near = [qb - i for i in range(budget // 2) if qb - i >= 0]
    far, step = [], max(1, budget // 2)
    while qb - step >= 0:
        far.append(qb - step)
        step *= 2
    return _pad(near + far + [0], budget)


def schedule_oracle(qb, nb, budget, mass=None, **_):
    allowed = np.arange(qb + 1)
    return _pad(list(allowed[np.argsort(-mass[qb, allowed])]), budget)


SCHEDULES = {
    "random": schedule_random,
    "window+sinks": schedule_window_sinks,
    "salience-0D": schedule_salience,
    "hierarchical": schedule_hierarchical,
    "oracle": schedule_oracle,
}


def recall(builder, mass, nb, budget, persistence, seed=0):
    """Fraction of true attention mass captured, averaged over query blocks."""
    rng = np.random.default_rng(seed)
    got, tot = 0.0, 0.0
    for qb in range(1, nb):
        sel = builder(qb, nb, budget, rng=rng, mass=mass, persistence=persistence)
        assert all(0 <= s <= qb for s in sel), "schedule broke causality"
        assert len(sel) <= budget, "schedule exceeded budget"
        got += mass[qb, sel].sum()
        tot += mass[qb, : qb + 1].sum()
    return got / tot


@pytest.fixture(scope="module")
def grid(attention_mass):
    """Recall for every schedule over a layer x budget grid, so no single
    (layer, budget) cell can be cherry-picked."""
    nb, hidden, att = attention_mass["nb"], attention_mass["hidden"], \
        attention_mass["attentions"]
    cells = {}
    for L in LAYERS:
        mass = att[L].reshape(nb, BLK, nb, BLK).sum(axis=(1, 3)) / BLK
        persistence = persistence_0d(hidden[L].reshape(nb, BLK, -1).mean(1))
        for B in BUDGETS:
            cells[(L, B)] = {n: recall(b, mass, nb, B, persistence)
                             for n, b in SCHEDULES.items()}
    return cells


def normalized(cell, name):
    """Position on the random -> oracle axis: 0 = null model, 1 = oracle."""
    lo, hi = cell["random"], cell["oracle"]
    return (cell[name] - lo) / (hi - lo + 1e-12)


def win_rate(grid, a, b):
    return sum(c[a] > c[b] for c in grid.values()) / len(grid)


def test_report_the_grid(grid):
    print()
    for (L, B), c in grid.items():
        print(f"layer{L:3d} budget{B:3d} " + " ".join(
            f"{k}={c[k]:.3f}({normalized(c, k):+.2f})" for k in SCHEDULES))


def test_no_schedule_exceeds_the_oracle(grid):
    """The oracle is an upper bound by construction; violating it means the
    budget or causality accounting is wrong."""
    for (L, B), c in grid.items():
        for k, v in c.items():
            assert v <= c["oracle"] + 1e-9, f"layer{L} budget{B} {k}={v}"


def test_hierarchical_beats_the_0d_salience_control(grid):
    """THEORY.md section 3's only new part must beat the merged control."""
    r = win_rate(grid, "hierarchical", "salience-0D")
    assert r > 0.5, (
        f"hierarchical beats the 0D-salience control in only {r:.0%} of "
        f"{len(grid)} layer x budget cells"
    )


def test_hierarchical_beats_plain_window_plus_sinks(grid):
    """If the cheap native pattern ties it, the new part buys nothing."""
    r = win_rate(grid, "hierarchical", "window+sinks")
    assert r > 0.5, (
        f"hierarchical beats window+sinks in only {r:.0%} of {len(grid)} cells; "
        "window+sinks is one mask_mod line and is already shipped as StreamingLLM"
    )


def test_0d_salience_control_beats_random(grid):
    """If the merged control sits at the random null, there is no signal."""
    r = win_rate(grid, "salience-0D", "random")
    assert r > 0.5, f"0D salience beats random in only {r:.0%} of cells"


def test_window_plus_sinks_leaves_headroom_for_a_smarter_schedule(grid):
    """The precondition for doing ANY schedule research.

    If the two-line baseline already sits on the oracle, no schedule builder --
    hierarchical, topological or learned -- has room to win, and section 3 is
    optimising a solved problem.
    """
    med = float(np.median([normalized(c, "window+sinks") for c in grid.values()]))
    assert med < 0.90, (
        f"window+sinks already captures {med:.1%} of the oracle's achievable "
        f"gain (median over {len(grid)} cells); there is almost nothing left to win"
    )
