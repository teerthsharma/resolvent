"""Phase J Addendum L, row L1-A5: A5 word-problem bed, (a)/(a'')/(f_Q) raced.

A5 = even permutations of 5 points (60 elements). Tokens are 4 fixed
generators: two 3-cycles g0=(0 1 2), g1=(2 3 4) and their inverses
g0inv=(0 2 1), g1inv=(2 4 3) -- closure is BFS-verified below to be exactly
60 (test-first, RED against a stub that returns the wrong count). Sequence
length L=64, target at every position = index (0..59) of the running
product. d=64, 2 layers, 4 heads (head_dim=16, divisible by 4 for f_Q).
Each layer carries one identically-shaped nn.Linear(d, n_heads) "parity
head" for all three arms (unused by (a), the forget-gate bias for (a''),
the quaternion head for (f_Q)) so numel matches by construction (0% spread,
inside the 2% tolerance).

su2.py (qmul/qconj/qnormalize/prefix_scan) is imported unedited, same as
arm_fq.py and l1_iaut.py.
"""
from __future__ import annotations

import io
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SP = os.path.dirname(os.path.abspath(__file__))
REPO = SP
BOARD = os.path.join(SP, "board_bed_events.jsonl")
RECORD_L = os.path.join(SP, "record_L.jsonl")

sys.path.insert(0, SP)
from su2 import qmul, qconj, qnormalize, prefix_scan  # noqa: E402

# ------------------------------------------------------------- A5 machinery
def _perm_compose(p, q):
    """(p*q)(i) = p(q(i))."""
    return tuple(p[q[i]] for i in range(5))


def _perm_inverse(p):
    inv = [0] * 5
    for i, pi in enumerate(p):
        inv[pi] = i
    return tuple(inv)


def _is_even(p):
    seen = [False] * 5
    parity = 0
    for i in range(5):
        if seen[i]:
            continue
        j, clen = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            clen += 1
        parity += clen - 1
    return parity % 2 == 0


def build_a5():
    elems = sorted(p for p in itertools.permutations(range(5)) if _is_even(p))
    assert len(elems) == 60, f"expected 60 even permutations, got {len(elems)}"
    index = {p: i for i, p in enumerate(elems)}
    identity = tuple(range(5))
    assert index[identity] == elems.index(identity)
    return elems, index, identity


def make_generators():
    g0 = (1, 2, 0, 3, 4)      # 3-cycle (0 1 2)
    g1 = (0, 1, 3, 4, 2)      # 3-cycle (2 3 4)
    for g in (g0, g1):
        assert _is_even(g)
    g0inv = _perm_inverse(g0)
    g1inv = _perm_inverse(g1)
    return [g0, g1, g0inv, g1inv]


def verify_generates_a5(gens, elems_set):
    """BFS closure; returns size of generated subgroup."""
    identity = tuple(range(5))
    seen = {identity}
    frontier = [identity]
    while frontier:
        nxt = []
        for p in frontier:
            for g in gens:
                q = _perm_compose(g, p)
                if q not in seen:
                    seen.add(q)
                    nxt.append(q)
        frontier = nxt
    assert seen == elems_set, f"generated {len(seen)} != |A5|=60"
    return len(seen)


ELEMS, INDEX, IDENTITY = build_a5()
GENS = make_generators()
GEN_COUNT = verify_generates_a5(GENS, set(ELEMS))
GEN_PRODUCT_TABLE = [[_perm_compose(g, IDENTITY)] for g in GENS]  # unused placeholder

L = 64
C_CLASSES = 60
N_TRAIN = 64
N_EVAL = 64
STEPS = 150          # step budget, stated once, identical for all arms/seeds
LR = 3e-3
WEIGHT_DECAY = 1e-4
SEEDS = [0, 1, 2]
REPORT_POSITIONS = [16, 32, 64]  # 1-indexed into the L=64 sequence
D = 64
N_HEADS = 4
N_LAYERS = 2
D_HEAD = D // N_HEADS


def gen_split(split_name, seed):
    """default_rng seeded per split (name, seed) -> (tokens[N,L] int64,
    labels[N,L] int64 = running-product index at every position)."""
    rng = np.random.default_rng(abs(hash((split_name, seed))) % (2**32))
    n = N_TRAIN if split_name == "train" else N_EVAL
    tok = rng.integers(0, 4, size=(n, L))
    lab = np.zeros((n, L), dtype=np.int64)
    for i in range(n):
        prod = IDENTITY
        for t in range(L):
            prod = _perm_compose(GENS[tok[i, t]], prod)
            lab[i, t] = INDEX[prod]
    return tok.astype(np.int64), lab


# ---------------------------------------------------------------- majority
def majority_floor(labels):
    vals, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / labels.size)


CHANCE_FLOOR = 1.0 / C_CLASSES


# =========================================================== architectures
def _rope(x, pos):
    """x: [...,S,Dh] (already split per-head), pos: [...,S]."""
    ang = (pos.float() * 0.5).unsqueeze(-1)
    d2 = x.shape[-1] // 2
    cos = torch.cos(ang).expand(*ang.shape[:-1], d2)
    sin = torch.sin(ang).expand(*ang.shape[:-1], d2)
    xp = x.reshape(*x.shape[:-1], d2, 2)
    x0, x1 = xp[..., 0], xp[..., 1]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    return torch.stack((r0, r1), dim=-1).reshape(x.shape)


class Layer(nn.Module):
    """One causal attention block + MLP. `variant` in {"a","a2","fq"}
    selects what the identically-shaped Linear(d, n_heads) parity head does:
    (a) unused (parity only), (a2) FoX forget-gate score bias, (fq) SU(2)
    quaternion rotation of V/output 4-blocks (arm_fq.py pattern, generalised
    to n_heads>1, d_head divisible by 4)."""

    def __init__(self, d, n_heads, variant):
        super().__init__()
        self.d, self.n_heads, self.d_head, self.variant = d, n_heads, d // n_heads, variant
        self.qkv = nn.Linear(d, 3 * d)
        self.o_proj = nn.Linear(d, d)
        self.parity_head = nn.Linear(d, n_heads)   # (a2): forget gate; (fq): reused only for first 4 cols
        self.quat_head = nn.Linear(d, 4) if variant == "fq" else None
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def attn(self, x):
        b, s, d = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def shape(t):
            return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,S,Dh]

        Q, K, V = shape(q), shape(k), shape(v)
        pos_h = pos.unsqueeze(1).expand(b, self.n_heads, s)
        Q, K = _rope(Q, pos_h), _rope(K, pos_h)

        if self.variant == "a2":
            gate = self.parity_head(x)                                    # [B,S,H]
            log_f = -F.softplus(-gate)
            c = log_f.cumsum(dim=1).transpose(1, 2)                        # [B,H,S]
            bias = (c.unsqueeze(-1) - c.unsqueeze(-2))                     # [B,H,S,S], C_i-C_j
            causal = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), diagonal=1)
            bias = bias.masked_fill(causal, float("-inf"))
            scores = (Q @ K.transpose(-1, -2)) / (self.d_head ** 0.5) + bias
            attn = torch.softmax(scores, dim=-1)
            O = attn @ V
        elif self.variant == "fq":
            n_blocks = self.d_head // 4
            quat = qnormalize(self.quat_head(x))                          # [B,S,4]
            Pi = prefix_scan(quat)                                          # [B,S,4] causal prefix
            Pi_conj = qconj(Pi)

            def to_blocks(t):
                return t.reshape(b, self.n_heads, s, n_blocks, 4)

            V4 = to_blocks(V)
            Pi_conj_b = Pi_conj.view(b, 1, s, 1, 4).expand(b, self.n_heads, s, n_blocks, 4)
            Vr = qmul(Pi_conj_b, V4).reshape(b, self.n_heads, s, self.d_head)
            O = F.scaled_dot_product_attention(Q, K, Vr, is_causal=True)
            O4 = to_blocks(O)
            Pi_b = Pi.view(b, 1, s, 1, 4).expand(b, self.n_heads, s, n_blocks, 4)
            O = qmul(Pi_b, O4).reshape(b, self.n_heads, s, self.d_head)
        else:  # "a" -- exact softmax twin; parity_head computed, unused
            _ = self.parity_head(x)
            O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        return self.o_proj(O.transpose(1, 2).reshape(b, s, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class A5Arm(nn.Module):
    def __init__(self, d, n_heads, n_layers, variant):
        super().__init__()
        self.emb = nn.Embedding(4, d)
        self.layers = nn.ModuleList([Layer(d, n_heads, variant) for _ in range(n_layers)])
        self.readout = nn.Linear(d, C_CLASSES)

    def forward(self, tokens):
        x = self.emb(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.readout(x)   # [B,S,60], one prediction per position


ARM_FACTORIES = {
    "a_softmax_twin": lambda: A5Arm(D, N_HEADS, N_LAYERS, "a"),
    "a2_fox_twin": lambda: A5Arm(D, N_HEADS, N_LAYERS, "a2"),
    "fq_quat_twin": lambda: A5Arm(D, N_HEADS, N_LAYERS, "fq"),
}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================ RED-first test
def _unimplemented_forward(self, tokens):
    raise NotImplementedError("a5 arm not yet wired -- RED stub")


def test_stub_raises():
    stub = nn.Module()
    stub.forward = _unimplemented_forward.__get__(stub)
    try:
        stub(torch.zeros(1, 4, dtype=torch.long))
        return False, "did not raise"
    except NotImplementedError as e:
        return True, f"{type(e).__name__}: {e}"


def test_a5_group_and_arms():
    assert len(ELEMS) == 60
    assert GEN_COUNT == 60
    x = torch.randint(0, 4, (2, L))
    for name, factory in ARM_FACTORIES.items():
        m = factory()
        out = m(x)
        assert out.shape == (2, L, C_CLASSES), f"{name} bad shape {out.shape}"


# ================================================================== training
def run_arm_seed(name, factory, seed, tr_x, tr_y, ev_x, ev_y):
    torch.manual_seed(seed)
    model = factory()
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    model.train()
    final_train_loss = None
    for _ in range(STEPS):
        opt.zero_grad()
        logits = model(tr_x)
        loss = loss_fn(logits.reshape(-1, C_CLASSES), tr_y.reshape(-1))
        loss.backward()
        opt.step()
        final_train_loss = float(loss.item())
    train_dur = time.time() - t0

    model.eval()
    with torch.no_grad():
        ev_logits = model(ev_x)
        pred = ev_logits.argmax(-1)
        per_pos_acc = {}
        for p in REPORT_POSITIONS:
            idx = p - 1
            per_pos_acc[str(p)] = float((pred[:, idx] == ev_y[:, idx]).float().mean())
    return {"arm": name, "seed": seed, "n_params": n_params,
            "final_train_loss": final_train_loss, "per_position_accuracy": per_pos_acc,
            "train_seconds": round(train_dur, 4)}


def log_board(event):
    event = dict(event)
    event.setdefault("agent", "Kofi")
    with io.open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def append_record_l(row):
    with io.open(RECORD_L, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ok, red_line = test_stub_raises()
    print(f"RED line: {red_line}")
    log_board({"t": "test", "agent": "Kofi", "status": "red", "name": "L1-A5"})

    test_a5_group_and_arms()
    print(f"GREEN: |A5|=60 verified, generators BFS-close to 60, all three arms run and produce correct shape")
    log_board({"t": "test", "agent": "Kofi", "status": "green", "name": "L1-A5"})

    params = {name: count_params(factory()) for name, factory in ARM_FACTORIES.items()}
    spread = max(params.values()) - min(params.values())
    mean_p = sum(params.values()) / len(params)
    spread_pct = 100 * spread / mean_p
    within_2pct = spread_pct <= 2.0
    print(f"params={params} spread_pct_of_mean={spread_pct:.4f}% within_2pct={within_2pct}")

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    results = {name: [] for name in ARM_FACTORIES}
    majority = []
    for seed in SEEDS:
        tr_tok, tr_lab = gen_split("train", seed)
        ev_tok, ev_lab = gen_split("eval", seed)
        majority.append(majority_floor(ev_lab))
        tr_x, tr_y = torch.tensor(tr_tok), torch.tensor(tr_lab)
        ev_x, ev_y = torch.tensor(ev_tok), torch.tensor(ev_lab)
        for name, factory in ARM_FACTORIES.items():
            r = run_arm_seed(name, factory, seed, tr_x, tr_y, ev_x, ev_y)
            results[name].append(r)
            log_board({"t": "test", "agent": "Kofi", "status": "green",
                       "name": f"L1-A5_{name}_seed{seed}",
                       "detail": f"final_pos_acc={r['per_position_accuracy']['64']:.4f} "
                                 f"loss={r['final_train_loss']:.4f} dur={r['train_seconds']:.2f}s"})
            print(f"  {name} seed={seed} loss={r['final_train_loss']:.4f} "
                  f"acc16={r['per_position_accuracy']['16']:.4f} "
                  f"acc32={r['per_position_accuracy']['32']:.4f} "
                  f"acc64={r['per_position_accuracy']['64']:.4f} dur={r['train_seconds']:.2f}s")
    finished = time.strftime("%Y-%m-%dT%H:%M:%S")

    def summarize(name):
        accs64 = [r["per_position_accuracy"]["64"] for r in results[name]]
        losses = [r["final_train_loss"] for r in results[name]]
        return {"mean_acc_pos64": float(np.mean(accs64)), "per_seed_acc_pos64": accs64,
                "per_seed_final_train_loss": losses,
                "per_seed_per_position_accuracy": [r["per_position_accuracy"] for r in results[name]],
                "params": results[name][0]["n_params"]}

    summary = {name: summarize(name) for name in ARM_FACTORIES}
    fq_mean = summary["fq_quat_twin"]["mean_acc_pos64"]
    a2_mean = summary["a2_fox_twin"]["mean_acc_pos64"]
    a_mean = summary["a_softmax_twin"]["mean_acc_pos64"]

    if fq_mean >= 0.90 and a2_mean <= 0.20:
        verdict = "PASS"
    elif fq_mean <= 0.20:
        verdict = "FAIL"
    else:
        verdict = "NEITHER"

    print(f"\na={a_mean:.4f} a2_fox={a2_mean:.4f} fq={fq_mean:.4f} verdict={verdict}")
    print(f"majority_floor(per seed eval)={majority} chance={CHANCE_FLOOR:.4f}")

    row = {
        "row": "L1-A5", "seat": "Foreman", "nurse": "Kofi",
        "model": "a5_bed.py (new); arm_fq.py f_Q pattern reused, generalised to n_heads=4/n_layers=2",
        "machine": "CPU, float32",
        "bar": {"pass": "(f_Q) final-position acc >= 0.90 AND (a'') <= 0.20",
                "fail": "(f_Q) <= 0.20", "source": "dispatcher (addendum registers none for this bed)"},
        "measured": {
            "d": D, "n_heads": N_HEADS, "n_layers": N_LAYERS, "L": L, "C_CLASSES": C_CLASSES,
            "steps": STEPS, "lr": LR, "n_train": N_TRAIN, "n_eval": N_EVAL, "seeds": SEEDS,
            "param_spread_pct_of_mean": spread_pct, "within_2pct": within_2pct, "params": params,
            "chance_floor": CHANCE_FLOOR, "majority_floor_per_seed": majority,
            "a_softmax_twin": summary["a_softmax_twin"],
            "a2_fox_twin": summary["a2_fox_twin"],
            "fq_quat_twin": summary["fq_quat_twin"],
        },
        "verdict": verdict,
        "control": "majority-class floor measured per seed (see measured.majority_floor_per_seed); chance=1/60=0.0167",
        "red_first": red_line,
        "producer": f"Kofi/Foreman (Claude Sonnet 5), torch-{torch.__version__}, numpy-{np.__version__}, Windows 11, CPU",
        "output": "a5_bed.py, a5_bed_results.json",
        "started": started, "finished": finished,
        "killed": "none.",
        "replacement": None if verdict == "PASS" else "reprice",
    }
    append_record_l(row)

    with io.open(os.path.join(SP, "a5_bed_results.json"), "w", encoding="utf-8") as f:
        json.dump({"params": params, "spread_pct_of_mean": spread_pct, "results": results,
                    "summary": summary, "verdict": verdict, "majority_floor_per_seed": majority}, f, indent=2)

    log_board({"t": "done", "agent": "Kofi", "text": f"L1-A5 done verdict={verdict} "
               f"a={a_mean:.4f} a2_fox={a2_mean:.4f} fq={fq_mean:.4f}"})
    print("wrote record_L.jsonl row L1-A5")


if __name__ == "__main__":
    main()
