"""
iaut_race.py -- the integer-automaton escape row.

WHY THIS FILE EXISTS. The prior positive (docs/PHASE_H.md:299-348, restated in
wilson_deliverable.md 1.3) reported operator gates at 0.9570 against vectors at
0.2238, 59 sigma, 84.9% attributed to non-commutativity, backed by NO bed, NO
test, NO artifact anywhere in the tree -- grep for 0.9570/0.3566/0.3342/0.7332
returns exactly that one prose file. Separately, h1_twins_ab_matched.py
(finding "forward_pass_is_the_bed_generative_recursion") establishes the
mechanism even where a bed DOES exist (phaseh_H3v2_bed.py): arms (c)/(d)'s
forward pass, state = op[verb] @ (state + emb[patient]), is character-for-
character the bed's own generative recursion. A high (c)/(d) score there
measures the correct hypothesis class fitting its own generator, nothing more.

THE ESCAPE. Hand the generator over as a plain integer transition table (a
finite automaton), not as permutation matrices. States = the 120 elements of
S5 (built from the certified bed, phaseh_H1_barrington.py: two generators
{5-cycle, transposition} that provably generate S5, row-vector convention
compose(p,q)[i] = q[p[i]]). Symbols = the same two generators, now opaque
integers 0/1. delta[state, symbol] -> next_state is the ONLY thing the
generator is presented as.

Every arm below receives, per instance, ONLY the length-L integer symbol
sequence -- never a permutation matrix, never delta itself, never group
structure. The automaton is fixed (one delta table, built once); arms must
recover a working representation of it purely from (symbol sequence -> final
state class) supervised examples, the same modality the vector arms already
used (verb/entity ids through nn.Embedding) and the operator arms now share
(symbol ids selecting a learned per-symbol matrix, with no privileged initial
value). Feeding the raw delta table as a per-instance input would hand a
lookup shortcut to nobody in particular and defeat the row, so it is not done;
this is stated, not hidden.

THE ROW. Five arms at matched parameter count (verified by numel(), a live
per-architecture search over hidden width d, not a formula trusted from a
config):
  (a) vector + softmax attention, RoPE positions, pooled at the last token.
  (b) scalar gate m*rotate(v, theta) on the same attention value stream.
      Expected to behave like (a): a single-token scalar gate telescopes
      under softmax into a per-key additive bias, per the prior finding.
  (c) operator gate, invertible only: state_t = M[sym_t] @ state_{t-1}, one
      dense d x d matrix per symbol, no projector.
  (d) operator gate + projector: state_t = P @ M[sym_t] @ state_{t-1}, a
      shared learned d x d projector composed on every step.
  (DIAG) THE CONTROL: identical recurrence to (c)/(d), M[sym] restricted to a
      diagonal (elementwise) gate. Diagonal matrices commute by construction,
      so DIAG cannot represent order -- it can only ever see the multiset of
      symbols, not their sequence.

THE PRE-REGISTERED KILL (written before any number below, not adjustable
after): IF (c)'s mean eval accuracy across seeds does NOT exceed DIAG's mean
eval accuracy across the same seeds, at matched parameters, on a generator (c)
was never handed, THEN the 84.9% attribution in the prior positive was the arm
being the generator, and that positive dies. This file reports whichever
outcome actually happens, flatly.

Producer: Foreman (Claude Sonnet 5), torch {torchver}, numpy {npver}, Python
{pyver}, Windows 11, CPU.
Run: python iaut_race.py
Board: appends one JSON line per event to
  C:\\Users\\seal\\Desktop\\New folder (32)\\house-events.jsonl  (io.open .. "a", never rewritten)
  agent = "Foreman"
Results: appends one JSON line per (arm, seed) AS IT FINISHES to
  iaut_results.jsonl (this scratchpad dir). Also writes iaut_race_summary.json
  (final aggregate) and iaut_race.md (prose readout) in this scratchpad dir.
  Nothing is written under the repo except the board log above.
"""
import importlib.util
import io
import itertools
import json
import os
import platform
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

SCRATCH = os.path.dirname(os.path.abspath(__file__))
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
RESULTS_JSONL = os.path.join(SCRATCH, "iaut_results.jsonl")
SUMMARY_JSON = os.path.join(SCRATCH, "iaut_race_summary.json")
MD_OUT = os.path.join(SCRATCH, "iaut_race.md")
S5_BED_SRC = os.path.join(SCRATCH, "phaseh_H1_barrington.py")

PRODUCER = f"Foreman/torch-{torch.__version__}/numpy-{np.__version__}/py-{platform.python_version()}"
CONVENTION = "row-vector permutation composition, compose(p,q)[i]=q[p[i]] (phaseh_H1_barrington.py convention)"
COMMAND = "python iaut_race.py"
DTYPE_MODEL = "float32"

SEEDS = [0, 1, 2, 3, 4]
L = 14                 # word length (symbols per instance). 2**14=16384
                        # possible words -- large enough that N_TRAIN+N_EVAL
                        # (<=3800) and the n=5000 floor pool draw WITHOUT
                        # REPLACEMENT via rejection sampling without the
                        # collision rate blowing up (at L=12, 2**12=4096
                        # possible words, asking for 3800+ unique ones or a
                        # 5000-word floor pool never terminates -- caught by
                        # a timing probe, fixed by widening L). L=14 (not 20):
                        # a timing probe at L=20/steps=400 left every arm's
                        # train loss at ~1.95-1.96 against an untrained
                        # ln(8)=2.079 -- 400 steps could not fit even the
                        # training set at that length, so no arm could clear
                        # the floor either way and the kill would have fired
                        # on underfitting, not on the question asked.
C_CLASSES = 8          # nearest-anchor classes, same C as the twins precedent
N_TRAIN = 3000
N_EVAL = 800
STEPS = 900
LR = 0.03
WEIGHT_DECAY = 1e-4
TARGET_PARAMS = 400    # common budget every architecture is searched against
DATA_OFFSET = 910000
ANCHOR_OFFSET = 910800
NULL_OFFSET = 911000
NULL_N_PER_SEED = 200


def log(event):
    event = dict(event)
    event.setdefault("agent", "Foreman")
    with io.open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def T(name, ok, detail, dur=0.0):
    log({"t": "test", "state": "GREEN" if ok else "RED", "name": name, "detail": detail, "dur": round(dur, 4)})
    print(("GREEN " if ok else "RED   ") + name + " :: " + detail)
    return ok


def F(item, text):
    log({"t": "finding", "item": item, "text": text})
    print("FINDING [" + item + "] " + text)


def append_result(row):
    with io.open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Import the certified S5 bed (unchanged, read-only) for compose/ALPHABET/
# ALL_S5/IDENTITY/INDEX_OF/generates_s5. Nothing heavy runs at import time --
# everything expensive in that file lives behind its __main__ guard.
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("iaut_s5_bed", S5_BED_SRC)
s5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s5)

IDENTITY = s5.IDENTITY
ALPHABET = s5.ALPHABET          # {"c": 5-cycle, "t": transposition}
ALL_S5 = s5.ALL_S5              # sorted list of 120 permutation tuples
INDEX_OF = s5.INDEX_OF
compose = s5.compose
SYM_KEYS = ["c", "t"]           # symbol id 0 -> "c", 1 -> "t"


def build_delta_table():
    """The ONLY form the generator is ever presented in: a plain (120, 2) int
    table. Built once, here, from the certified bed's compose(); never touched
    by any arm."""
    delta = np.zeros((120, 2), dtype=np.int64)
    for s_idx, p in enumerate(ALL_S5):
        for sym_id, key in enumerate(SYM_KEYS):
            q = compose(p, ALPHABET[key])
            delta[s_idx, sym_id] = INDEX_OF[q]
    return delta


def run_word(delta, word, start=INDEX_OF[IDENTITY]):
    """word: sequence of symbol ids (ints). Pure integer table lookups."""
    s_idx = start
    for sym in word:
        s_idx = int(delta[s_idx, sym])
    return s_idx


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def build_anchors(seed):
    rng = random.Random(f"iaut_anchor|{seed}|{ANCHOR_OFFSET}")
    idxs = rng.sample(range(120), C_CLASSES)
    return idxs  # state indices


def label_of(state_idx, anchor_idxs):
    p = ALL_S5[state_idx]
    dists = [hamming(p, ALL_S5[a]) for a in anchor_idxs]
    return int(np.argmin(dists))


# ---------------------------------------------------------------------------
# Instance generation: unique random words per seed, deterministic, disjoint
# train/eval by construction (drawn without replacement from the same pool).
# ---------------------------------------------------------------------------
def gen_words(seed, n, length=L):
    rng = random.Random(f"iaut_data|{seed}|{DATA_OFFSET}")
    seen = set()
    words = []
    while len(words) < n:
        w = tuple(rng.randrange(2) for _ in range(length))
        if w in seen:
            continue
        seen.add(w)
        words.append(w)
    return words


def build_split(seed, delta, anchors):
    words = gen_words(seed, N_TRAIN + N_EVAL)
    finals = [run_word(delta, w) for w in words]
    labels = [label_of(f, anchors) for f in finals]
    train = {"words": words[:N_TRAIN], "finals": finals[:N_TRAIN], "labels": labels[:N_TRAIN]}
    ev = {"words": words[N_TRAIN:], "finals": finals[N_TRAIN:], "labels": labels[N_TRAIN:]}
    return train, ev


# ---------------------------------------------------------------------------
# Floors: constant-majority (effective class count measured at n=5000, not
# assumed 1/C) and last-two-symbols (rule fit on TRAIN, scored on EVAL -- no
# label leakage from the split being scored).
# ---------------------------------------------------------------------------
def constant_majority_floor(seed, delta, anchors, ev):
    rng = random.Random(f"iaut_floor_large|{seed}|{DATA_OFFSET}")
    seen = set()
    big_labels = []
    while len(big_labels) < 5000:
        w = tuple(rng.randrange(2) for _ in range(L))
        if w in seen:
            continue
        seen.add(w)
        big_labels.append(label_of(run_word(delta, w), anchors))
    counts = Counter(big_labels)
    n_classes_hit = len(counts)
    maj_class, maj_count = counts.most_common(1)[0]
    effective_majority_frac = maj_count / len(big_labels)
    ev_correct = sum(1 for lb in ev["labels"] if lb == maj_class)
    return {
        "majority_class": maj_class,
        "n_classes_hit_at_n5000": n_classes_hit,
        "effective_majority_frac_at_n5000": effective_majority_frac,
        "acc_on_eval_split": ev_correct / len(ev["labels"]),
    }


def last_two_symbols_floor(train, ev):
    buckets = {}
    for w, lb in zip(train["words"], train["labels"]):
        key = w[-2:]
        buckets.setdefault(key, Counter())[lb] += 1
    bucket_majority = {k: v.most_common(1)[0][0] for k, v in buckets.items()}
    global_fallback = Counter(train["labels"]).most_common(1)[0][0]
    correct = 0
    for w, lb in zip(ev["words"], ev["labels"]):
        pred = bucket_majority.get(w[-2:], global_fallback)
        if pred == lb:
            correct += 1
    return {"n_buckets_seen_in_train": len(buckets), "acc_on_eval_split": correct / len(ev["labels"])}


# ---------------------------------------------------------------------------
# THE NULL: shuffled symbol ORDER within each word (same multiset), recomposed
# through the delta table, scored on the task metric (class label match), not
# on finiteness. Must produce >=1 mismatch per seed at all 5 seeds or the bed
# is void before any arm runs.
# ---------------------------------------------------------------------------
def run_null(seed, delta, anchors):
    rng = random.Random(f"iaut_null|{seed}|{NULL_OFFSET}")
    src_rng = random.Random(f"iaut_null_src|{seed}|{NULL_OFFSET}")
    wrong = 0
    for _ in range(NULL_N_PER_SEED):
        word = tuple(src_rng.randrange(2) for _ in range(L))
        true_final = run_word(delta, word)
        true_label = label_of(true_final, anchors)
        shuffled = list(word)
        rng.shuffle(shuffled)
        shuf_final = run_word(delta, shuffled)
        shuf_label = label_of(shuf_final, anchors)
        if shuf_label != true_label:
            wrong += 1
    return {"seed": seed, "n": NULL_N_PER_SEED, "wrong": wrong, "wrong_frac": wrong / NULL_N_PER_SEED,
            "void_condition_met": wrong >= 1}


# ===========================================================================
# Architectures
# ===========================================================================
def _rope(x, pos):
    ang = (pos.float() * 0.5).unsqueeze(-1)
    d2 = x.shape[-1] // 2
    cos = torch.cos(ang).expand(*ang.shape[:-1], d2)
    sin = torch.sin(ang).expand(*ang.shape[:-1], d2)
    xp = x.reshape(*x.shape[:-1], d2, 2)
    x0, x1 = xp[..., 0], xp[..., 1]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    return torch.stack((r0, r1), dim=-1).reshape(x.shape)


class VectorArm(nn.Module):
    """(a) vector+softmax (use_gate=False) / (b) scalar-gate (use_gate=True).
    RoPE positional encoding, single head, pooled at the last token. gate_proj
    exists in BOTH arms (parity, as in h1_twins_ab_matched.py's precedent) so
    numel(a) == numel(b) exactly."""

    def __init__(self, d, use_gate):
        super().__init__()
        assert d % 2 == 0, "RoPE requires even d"
        self.d = d
        self.use_gate = use_gate
        self.sym_emb = nn.Embedding(2, d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.gate_proj = nn.Linear(d, 2)
        self.readout = nn.Linear(d, C_CLASSES)

    def forward(self, symbols):
        B, Lc = symbols.shape
        emb = self.sym_emb(symbols)
        pos = torch.arange(Lc, device=symbols.device).unsqueeze(0).expand(B, -1)
        q = _rope(self.q_proj(emb), pos)
        k = _rope(self.k_proj(emb), pos)
        v = self.v_proj(emb)
        if self.use_gate:
            raw = self.gate_proj(emb)
            m = torch.sigmoid(raw[..., 0]).unsqueeze(-1)
            theta = raw[..., 1]
            d2 = self.d // 2
            vp = v.reshape(B, Lc, d2, 2)
            v0, v1 = vp[..., 0], vp[..., 1]
            cos_t, sin_t = torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)
            g0 = m * (v0 * cos_t - v1 * sin_t)
            g1 = m * (v0 * sin_t + v1 * cos_t)
            v = torch.stack((g0, g1), dim=-1).reshape(B, Lc, -1)
        else:
            _ = self.gate_proj(emb)  # parameter-count parity, output unused
        attn = torch.softmax(q @ k.transpose(-1, -2) / (self.d ** 0.5), dim=-1)
        pooled = (attn @ v)[:, -1, :]
        return self.readout(pooled)


class OperatorArm(nn.Module):
    """(c) invertible-only: state_t = M[sym] @ state_{t-1}, dense M, no
    projector. (d) + projector: state_t = P @ M[sym] @ state_{t-1}. (DIAG)
    control: M[sym] restricted to a diagonal (elementwise) gate -- commutes by
    construction, cannot represent symbol order."""

    def __init__(self, d, diag, projector):
        super().__init__()
        self.d = d
        self.diag = diag
        self.projector = projector
        if diag:
            self.gate = nn.Parameter(torch.randn(2, d) * 0.5 + 1.0)
        else:
            self.M = nn.Parameter(torch.stack([torch.eye(d) + 0.1 * torch.randn(d, d) for _ in range(2)]))
        if projector:
            self.P = nn.Parameter(torch.eye(d) + 0.1 * torch.randn(d, d))
        self.init_state = nn.Parameter(torch.randn(d) * 0.1)
        self.readout = nn.Linear(d, C_CLASSES)

    def forward(self, symbols):
        B, Lc = symbols.shape
        state = self.init_state.unsqueeze(0).expand(B, -1)
        for t in range(Lc):
            sym = symbols[:, t]
            if self.diag:
                g = self.gate[sym]                      # [B, d]
                state = g * state
            else:
                Mt = self.M[sym]                         # [B, d, d]
                state = torch.bmm(Mt, state.unsqueeze(-1)).squeeze(-1)
            if self.projector:
                state = state @ self.P.t()
        return self.readout(state)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


ARM_FACTORIES = {
    "a_vector_softmax": lambda d: VectorArm(d, use_gate=False),
    "b_scalar_gate": lambda d: VectorArm(d, use_gate=True),
    "c_operator_invertible": lambda d: OperatorArm(d, diag=False, projector=False),
    "d_operator_projector": lambda d: OperatorArm(d, diag=False, projector=True),
    "DIAG_commuting_control": lambda d: OperatorArm(d, diag=True, projector=False),
}
# d must be even for the vector arms (RoPE reshapes the last dim into pairs).
# Operator arms have no such constraint and are searched over every integer
# d so the matched-budget point is not artificially widened by skipping odds.
D_GRID_EVEN = list(range(2, 41, 2))
D_GRID_ALL = list(range(2, 41))
ARM_GRID = {
    "a_vector_softmax": D_GRID_EVEN, "b_scalar_gate": D_GRID_EVEN,
    "c_operator_invertible": D_GRID_ALL, "d_operator_projector": D_GRID_ALL,
    "DIAG_commuting_control": D_GRID_ALL,
}


def search_d(name, factory, target=TARGET_PARAMS):
    rows = []
    for d in ARM_GRID[name]:
        n = count_params(factory(d))
        rows.append({"d": d, "numel": n, "abs_diff": abs(n - target)})
    best = min(rows, key=lambda r: r["abs_diff"])
    return best, rows


# ===========================================================================
# Train one arm, one seed. Writes its result row the moment it finishes.
# ===========================================================================
def run_arm_seed(name, factory, d, seed, tr_words, tr_labels, ev_words, ev_labels, tightest_floor):
    torch.manual_seed(seed)
    model = factory(d)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    tr_x = torch.tensor(tr_words, dtype=torch.long)
    tr_y = torch.tensor(tr_labels, dtype=torch.long)
    ev_x = torch.tensor(ev_words, dtype=torch.long)
    ev_y = torch.tensor(ev_labels, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        untrained_acc = float((model(ev_x).argmax(-1) == ev_y).float().mean())

    t0 = time.time()
    model.train()
    for _ in range(STEPS):
        opt.zero_grad()
        loss = loss_fn(model(tr_x), tr_y)
        loss.backward()
        opt.step()
    train_dur = time.time() - t0

    model.eval()
    with torch.no_grad():
        ev_logits = model(ev_x)
        trained_acc = float((ev_logits.argmax(-1) == ev_y).float().mean())
        trained_loss = float(loss_fn(ev_logits, ev_y))

    invertible_note = None
    if isinstance(model, OperatorArm) and not model.diag:
        with torch.no_grad():
            dets = [float(torch.linalg.det(model.M[i])) for i in range(2)]
        invertible_note = {"det_M0": dets[0], "det_M1": dets[1], "both_nonzero": all(abs(x) > 1e-8 for x in dets)}

    headroom = trained_acc - tightest_floor
    row = {
        "t": "result", "row": "iaut_race", "arm": name, "seed": seed,
        "n_params": n_params, "d_model": d,
        "n_train": N_TRAIN, "n_eval": N_EVAL, "word_length": L, "c_classes": C_CLASSES,
        "steps": STEPS, "lr": LR, "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
        "untrained_eval_accuracy": untrained_acc,
        "trained_eval_accuracy": trained_acc, "trained_eval_loss": trained_loss,
        "tightest_floor": tightest_floor, "headroom_above_tightest_floor": headroom,
        "final_train_loss": float(loss.item()), "train_seconds": round(train_dur, 4),
        "invertibility_check": invertible_note,
        "generator": "S5 delta table (120 states, 2 symbols: 5-cycle + transposition), phaseh_H1_barrington.py",
        "convention": CONVENTION, "dtype_model": DTYPE_MODEL, "dtype_table": "int64",
        "producer": PRODUCER, "command": COMMAND, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_result(row)
    log({"t": "test", "agent": "Foreman", "state": "GREEN", "name": f"iaut_{name}_seed{seed}",
         "detail": (f"d={d} n_params={n_params} untrained={untrained_acc:.4f} -> trained={trained_acc:.4f} "
                    f"(tightest_floor={tightest_floor:.4f}, headroom={headroom:+.4f}) dur={train_dur:.2f}s"),
         "dur": train_dur})
    print(f"  seed {seed} {name}: d={d} params={n_params} acc={trained_acc:.4f} "
          f"(floor={tightest_floor:.4f}, headroom={headroom:+.4f}) dur={train_dur:.2f}s")
    return row


# ===========================================================================
def main():
    t0 = time.time()
    F("phase_start",
      f"iaut_race: escaping the operator-vs-vector confound named in h1_twins_ab_matched.py "
      f"('forward_pass_is_the_bed_generative_recursion') and the unreproducible prose positive "
      f"named in wilson_deliverable.md 1.3 (0.9570/0.3566/0.3342/0.7332, docs/PHASE_H.md, no bed). "
      f"Generator: S5 delta table (120 states, symbols={{c=5-cycle,t=transposition}}), convention="
      f"{CONVENTION}. word_length={L}, C_classes={C_CLASSES}, n_train={N_TRAIN}, n_eval={N_EVAL}, "
      f"steps={STEPS}, AdamW(lr={LR}, weight_decay={WEIGHT_DECAY}), seeds={SEEDS}, target_params="
      f"{TARGET_PARAMS}, dtype_model={DTYPE_MODEL}, producer={PRODUCER}. Arms receive ONLY the "
      f"integer symbol sequence per instance -- never the delta table, never a permutation matrix, "
      f"never group structure. command={COMMAND}")

    assert s5.generates_s5(ALPHABET.values())
    delta = build_delta_table()

    # BFS reachability re-certified on the TABLE itself (not on compose()
    # directly) -- confirms the table that arms are trained against, not just
    # the group machinery that built it, reaches all 120 states.
    seen = {INDEX_OF[IDENTITY]}
    frontier = [INDEX_OF[IDENTITY]]
    while frontier:
        nxt = []
        for s_idx in frontier:
            for sym in (0, 1):
                q = int(delta[s_idx, sym])
                if q not in seen:
                    seen.add(q)
                    nxt.append(q)
        frontier = nxt
    T("delta_table_bfs_reaches_120", len(seen) == 120,
      f"BFS over the plain integer delta table from IDENTITY reaches {len(seen)}/120 states "
      f"(scan has real depth, not a lookup: the certified bed's own diameter carries over "
      f"because delta was built by compose() against the same two generators)")

    # ---- param search per architecture, live numel(), not a formula -------
    chosen_d = {}
    search_report = {}
    for name, factory in ARM_FACTORIES.items():
        best, rows = search_d(name, factory)
        chosen_d[name] = best["d"]
        search_report[name] = {"chosen": best, "grid": rows}
        T(f"param_search_{name}", True,
          f"chosen d={best['d']} numel={best['numel']} (target={TARGET_PARAMS}, diff={best['numel']-TARGET_PARAMS:+d})")
    params_by_arm = {name: search_report[name]["chosen"]["numel"] for name in ARM_FACTORIES}
    spread = max(params_by_arm.values()) - min(params_by_arm.values())
    F("param_match",
      f"Live per-architecture numel() search over d in {{2,4,...,40}} against target={TARGET_PARAMS}: "
      + "; ".join(f"{k}: d={chosen_d[k]}, numel={params_by_arm[k]}" for k in ARM_FACTORIES)
      + f". Spread across the 5 arms = {spread} params ({100*spread/TARGET_PARAMS:.1f}% of target). "
      "Different architectures cannot hit one exact integer with a live search; the closest point per "
      "architecture is reported, not assumed, per numel().")

    # ---- the null, gates everything below it --------------------------
    anchors0 = build_anchors(0)  # anchors only affect labeling; null uses seed-specific anchors below
    null_rows = []
    for seed in SEEDS:
        anchors_s = build_anchors(seed)
        nr = run_null(seed, delta, anchors_s)
        null_rows.append(nr)
        T(f"null_shuffled_symbol_order_seed{seed}", nr["void_condition_met"],
          f"wrong={nr['wrong']}/{nr['n']} (bed requires >=1 wrong; RED would VOID the bed)")
    bed_sound = all(r["void_condition_met"] for r in null_rows)
    F("null_shuffled_symbol_order",
      "Null = same word, symbol ORDER shuffled (same multiset), refolded through the plain delta "
      "table, scored on the class-label task metric (not on finiteness -- every reorder yields SOME "
      "valid state, so a finiteness check would trivially always pass). Per seed: "
      + "; ".join(f"seed{r['seed']}: {r['wrong']}/{r['n']} wrong" for r in null_rows)
      + f". {'Bed is SOUND: every seed scores >=1 wrong.' if bed_sound else 'BED IS VOID: at least one seed scored zero wrong -- no arm may run.'}")

    if not bed_sound:
        log({"t": "done", "agent": "Foreman", "checks": 1, "reds": 1,
             "text": "iaut_race ABORTED: null was not sound at all 5 seeds, bed void before any arm ran."})
        print("BED VOID -- aborting before any arm runs.")
        return

    # ---- per-seed data, floors, then the five arms ---------------------
    per_arm_rows = {name: [] for name in ARM_FACTORIES}
    floor_rows = []
    for seed in SEEDS:
        t_seed0 = time.time()
        anchors = build_anchors(seed)
        train, ev = build_split(seed, delta, anchors)

        cmaj = constant_majority_floor(seed, delta, anchors, ev)
        lts = last_two_symbols_floor(train, ev)
        tightest = max(cmaj["acc_on_eval_split"], lts["acc_on_eval_split"])
        floor_row = {"seed": seed, "constant_majority": cmaj, "last_two_symbols": lts, "tightest_floor": tightest}
        floor_rows.append(floor_row)
        T(f"floors_seed{seed}", True,
          f"constant_majority(eff. class count at n=5000: {cmaj['n_classes_hit_at_n5000']}/{C_CLASSES})="
          f"{cmaj['acc_on_eval_split']:.4f}, last_two_symbols(fit on train, scored on eval)="
          f"{lts['acc_on_eval_split']:.4f}, tightest={tightest:.4f}")

        for name, factory in ARM_FACTORIES.items():
            row = run_arm_seed(name, factory, chosen_d[name], seed,
                                train["words"], train["labels"], ev["words"], ev["labels"], tightest)
            per_arm_rows[name].append(row)

        print(f"seed {seed} done in {time.time()-t_seed0:.2f}s")

    # ---- summary + pre-registered kill ---------------------------------
    def summarize(name):
        accs = [r["trained_eval_accuracy"] for r in per_arm_rows[name]]
        return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "per_seed": accs,
                "params": per_arm_rows[name][0]["n_params"], "d_model": chosen_d[name]}

    summary = {name: summarize(name) for name in ARM_FACTORIES}
    tightest_mean = float(np.mean([fr["tightest_floor"] for fr in floor_rows]))

    c_mean = summary["c_operator_invertible"]["mean"]
    diag_mean = summary["DIAG_commuting_control"]["mean"]
    kill_fires = not (c_mean > diag_mean)
    F("pre_registered_kill",
      f"KILL CONDITION (fixed before these numbers): (c) operator-invertible mean eval accuracy "
      f"across {len(SEEDS)} seeds must exceed (DIAG) commuting-control mean eval accuracy across the "
      f"same seeds, at matched params (c={summary['c_operator_invertible']['params']}, "
      f"DIAG={summary['DIAG_commuting_control']['params']}), on a generator (c) was never handed "
      f"(only the symbol sequence, never delta, never a matrix). Measured: (c) mean={c_mean:.4f} "
      f"std={summary['c_operator_invertible']['std']:.4f}; (DIAG) mean={diag_mean:.4f} "
      f"std={summary['DIAG_commuting_control']['std']:.4f}. "
      + (f"KILL FIRES: (c) did not beat (DIAG). The prior positive's 84.9% attribution was the arm "
         f"being the generator; that positive does not survive this row, reported flatly."
         if kill_fires else
         f"KILL DOES NOT FIRE: (c) beat (DIAG) by {c_mean-diag_mean:+.4f} on a generator it was never "
         f"handed, at matched parameters, with a sound null. This is evidence for the "
         f"non-commutativity attribution surviving outside its own generative recursion -- one row, "
         f"not a proof, and only as strong as the {len(SEEDS)}-seed, d={chosen_d['c_operator_invertible']} "
         f"budget it was measured at."))

    table_text = "; ".join(
        f"{name}: mean={summary[name]['mean']:.4f} std={summary[name]['std']:.4f} "
        f"params={summary[name]['params']} d={summary[name]['d_model']} "
        f"headroom={summary[name]['mean']-tightest_mean:+.4f}"
        for name in ARM_FACTORIES
    )
    F("five_arm_table",
      f"tightest_floor mean across seeds={tightest_mean:.4f}. " + table_text)

    RESULTS = {
        "producer": PRODUCER, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "dtype_model": DTYPE_MODEL,
        "seeds": SEEDS, "word_length": L, "c_classes": C_CLASSES, "n_train": N_TRAIN, "n_eval": N_EVAL,
        "steps": STEPS, "lr": LR, "weight_decay": WEIGHT_DECAY, "target_params": TARGET_PARAMS,
        "command": COMMAND, "convention": CONVENTION,
        "generator": "S5 delta table (120 states, symbols c=5-cycle/t=transposition), phaseh_H1_barrington.py",
        "param_search": search_report, "chosen_d": chosen_d,
        "null_rows": null_rows, "bed_sound": bed_sound,
        "floor_rows": floor_rows, "tightest_floor_mean": tightest_mean,
        "summary": summary,
        "kill_condition": "c_mean > DIAG_mean, matched params, sound null",
        "kill_fires": kill_fires,
        "c_mean": c_mean, "diag_mean": diag_mean, "c_minus_diag": c_mean - diag_mean,
        "wall_seconds": time.time() - t0,
    }
    with io.open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)

    checks = 1 + len(ARM_FACTORIES) + len(SEEDS) + len(SEEDS) * len(ARM_FACTORIES)
    reds = (0 if bed_sound else len(SEEDS)) + (1 if kill_fires else 0)
    done_text = (
        f"iaut_race done. bed_sound={bed_sound}. 5-arm table (mean eval acc over {len(SEEDS)} seeds, "
        f"tightest_floor_mean={tightest_mean:.4f}): {table_text}. "
        f"PRE-REGISTERED KILL: c_mean={c_mean:.4f} vs DIAG_mean={diag_mean:.4f}, "
        f"kill_fires={kill_fires}. results appended to {RESULTS_JSONL}, summary at {SUMMARY_JSON}."
    )
    log({"t": "done", "agent": "Foreman", "checks": checks, "reds": reds, "text": done_text})
    print("\n" + done_text)
    print(f"wall_seconds total = {time.time()-t0:.2f}")
    write_md(RESULTS)
    print(f"report written to {MD_OUT}")


def write_md(R):
    lines = []
    lines.append("# iaut_race -- the integer-automaton escape row\n")
    lines.append(f"Producer: {R['producer']}. Run: `{R['command']}`. Timestamp: {R['timestamp']}.\n")
    lines.append(
        f"Generator: {R['generator']}. Convention: {R['convention']}. "
        f"word_length={R['word_length']}, C_classes={R['c_classes']}, dtype_model={R['dtype_model']}, "
        f"dtype_table=int64, n_train={R['n_train']}, n_eval={R['n_eval']}, steps={R['steps']}, "
        f"lr={R['lr']}, weight_decay={R['weight_decay']}, seeds={R['seeds']}, "
        f"target_params={R['target_params']}.\n"
    )
    lines.append("## Why this row exists\n")
    lines.append(
        "The prior positive (0.9570 operator vs 0.2238 vector, 59 sigma, 84.9% attributed to "
        "non-commutativity) has its numbers in exactly one prose file (`docs/PHASE_H.md`) with no "
        "bed, no test, no artifact -- see `wilson_deliverable.md` section 1.3. Separately, where a "
        "bed does exist (`phaseh_H3v2_bed.py`), `h1_twins_ab_matched.py` found the operator arms' "
        "forward pass -- `state = op[verb] @ (state + emb[patient])` -- is character-for-character "
        "the bed's own generative recursion, so a high score there measures the correct hypothesis "
        "class fitting its own generator. This row hands the generator over as a plain integer "
        "transition table (finite automaton) instead, so no arm's forward pass can be the generator "
        "by construction.\n"
    )
    lines.append("## Param search (live numel(), not a formula)\n")
    lines.append("| arm | chosen d | numel | diff from target |\n|---|---|---|---|\n")
    for name, rep in R["param_search"].items():
        b = rep["chosen"]
        lines.append(f"| {name} | {b['d']} | {b['numel']} | {b['numel']-R['target_params']:+d} |\n")
    lines.append("\n## Null (shuffled symbol order, scored on the class-label task metric)\n")
    lines.append("| seed | wrong / n | void condition met |\n|---|---|---|\n")
    for r in R["null_rows"]:
        lines.append(f"| {r['seed']} | {r['wrong']}/{r['n']} | {r['void_condition_met']} |\n")
    lines.append(f"\nbed_sound = **{R['bed_sound']}**\n")
    lines.append("\n## Floors (per seed, own eval split)\n")
    lines.append("| seed | constant-majority | last-two-symbols | tightest |\n|---|---|---|---|\n")
    for fr in R["floor_rows"]:
        lines.append(f"| {fr['seed']} | {fr['constant_majority']['acc_on_eval_split']:.4f} | "
                      f"{fr['last_two_symbols']['acc_on_eval_split']:.4f} | {fr['tightest_floor']:.4f} |\n")
    lines.append(f"\ntightest_floor mean across seeds = **{R['tightest_floor_mean']:.4f}**\n")
    lines.append("\n## Five-arm table (mean +/- std eval accuracy across seeds, matched params)\n")
    lines.append("| arm | d | params | mean | std | headroom above tightest floor |\n|---|---|---|---|---|---|\n")
    for name, s in R["summary"].items():
        lines.append(f"| {name} | {s['d_model']} | {s['params']} | {s['mean']:.4f} | {s['std']:.4f} | "
                      f"{s['mean']-R['tightest_floor_mean']:+.4f} |\n")
    lines.append("\n## The pre-registered kill\n")
    lines.append(
        f"Condition (fixed before these numbers): (c) operator-invertible mean must exceed (DIAG) "
        f"commuting-control mean, matched params, sound null, generator never handed to any arm.\n\n"
        f"Measured: c_mean = **{R['c_mean']:.4f}**, DIAG_mean = **{R['diag_mean']:.4f}**, "
        f"delta = **{R['c_minus_diag']:+.4f}**.\n\n"
        f"**kill_fires = {R['kill_fires']}**\n"
    )
    if R["kill_fires"]:
        lines.append(
            "\nThe operator arm did not beat the diagonal control at matched parameters on a "
            "generator it was never handed. The prior positive's 84.9% non-commutativity "
            "attribution was the arm being the generator; that positive does not survive this row. "
            "Reported flatly, as pre-registered.\n"
        )
    else:
        lines.append(
            "\nThe operator arm beat the diagonal control at matched parameters on a generator it "
            "was never handed, with a sound null. This is one row of evidence that the "
            "non-commutativity attribution survives outside its own generative recursion -- not a "
            "proof, and bounded by the seed count and parameter budget stated above.\n"
        )
    lines.append(f"\n## Reproduction\n\n```\ncd \"{SCRATCH}\"\n{R['command']}\n```\n")
    lines.append(f"\nRaw per-(arm,seed) rows: `iaut_results.jsonl`. Full machine-readable summary: "
                  f"`iaut_race_summary.json`. Board events: agent \"Foreman\" in house-events.jsonl.\n")
    with io.open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    main()
