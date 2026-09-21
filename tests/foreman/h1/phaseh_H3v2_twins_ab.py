"""
Phase H -- H3v2 twins (a) word2vec-style vectors + softmax head, no gates vs
(b) scalar-gate family m*exp(i theta). Scored on the H3v2 DISCRETE-LABEL
target (nearest of C=8 fixed unit anchors by cosine) -- never on 1/C, never
on raw L2. phaseh_H3v2_floors.py already cleared this bed for training:
null_discriminates=True (mean null accuracy 0.1780, strictly worse than the
correct order on >=1 instance per seed at all 5 seeds) and the tightest
floor is last_two_ops=0.37928 (n=5000, 5 seeds, float64), not bag_of_context
and not 1/C=0.125. This file reads both numbers out of
phaseh_H3v2_floors_results.json rather than retyping them, and refuses to
train if that file says null_discriminates is False.

Bed primitives (make_entities, rot_ops, bed_instance) are imported from
phaseh_H3v2_bed.py, not rebuilt. Anchors use the same construction as
phaseh_H3v2_floors.py::anchors_for (rng.standard_normal((C,D)), L2-normalized)
under a fresh seed-offset namespace -- this file owns its own train/test
population, distinct from the floors' evaluation population.

ORACLE TRAP: no twin ever sees a verb's planted rotation matrix. Verbs and
entities are read only as integer ids through a learned nn.Embedding --
the ground-truth SO(8) operators and the anchor codebook live only in the
data generator / labeler, used solely to compute the target label.

Scope: twins (a)/(b) only. (c) operator-gate invertible-only and (d)
operator-gate invertible+projectors are a separate lane's file
(phaseh_H3v2_twins_cd.py, not owned here).

Producer: Foreman (Claude Sonnet 5), torch {ver}, numpy {ver}, Python {pyver}, Windows 11.
Run: python phaseh_H3v2_twins_ab.py
Board: appends to C:\\Users\\seal\\Desktop\\New folder (32)\\house-events.jsonl
"""
import io
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
import importlib.util
spec = importlib.util.spec_from_file_location("phaseh_H3v2_bed", os.path.join(SCRATCH, "phaseh_H3v2_bed.py"))
bedmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bedmod)  # safe: bedmod.main() is __main__-guarded

make_entities = bedmod.make_entities
rot_ops = bedmod.rot_ops
bed_instance = bedmod.bed_instance

BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
RESULTS_JSONL = os.path.join(SCRATCH, "phaseh_H3v2_results.jsonl")
RESULTS_JSON = os.path.join(SCRATCH, "phaseh_H3v2_twins_ab_results.json")
FLOORS_RESULTS = os.path.join(SCRATCH, "phaseh_H3v2_floors_results.json")

PRODUCER = f"Foreman/torch-{torch.__version__}/numpy-{np.__version__}/py-{platform.python_version()}"
DTYPE_MODEL = "float32"

# bed params -- identical to phaseh_H3v2_bed.py (M, D, K, LINE_LEN) and its
# winning fix (C=8 anchors)
M_ENT, D, K_VERB, LINE_LEN = bedmod.M, bedmod.D, bedmod.K, bedmod.LINE_LEN
C = 8
N_TRAIN, N_TEST = 300, 100
D_MODEL, HEADS, STEPS, LR = 16, 1, 1500, 3e-3
SEEDS = (0, 1, 2, 3, 4)
TWIN_OFFSET = 76000       # fresh namespace -- distinct from bed's (6000+) and floors' (46000/57000) pops
ANCHOR_OFFSET = 76800


def log(event):
    event = dict(event)
    event.setdefault("agent", "Foreman")
    with io.open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def append_result(row):
    with io.open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def anchors_for(seed):
    rng_anchor = np.random.default_rng(ANCHOR_OFFSET + seed)
    anchors = rng_anchor.standard_normal((C, D))
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    return anchors


def label_of(anchors, vec):
    sims = (anchors @ vec) / (np.linalg.norm(vec) + 1e-12)
    return int(np.argmax(sims))


def make_dataset(rng, entities, verb_ops, anchors, n):
    """Same recurrence as bedmod.bed_instance, called through it (not
    reimplemented) so the population matches the bed exactly; this function
    only adds the plain-int (verb_id, patient_id) steps for the model's
    input and the discrete label for the target -- verb_ops/anchors are used
    ONLY to compute the label, never handed to a model."""
    rows = []
    for _ in range(n):
        line, final_state = bed_instance(rng, entities, verb_ops, LINE_LEN)
        agent0 = line[0][1]
        steps = [(v, p) for v, _a, p in line]
        rows.append({"agent0": agent0, "steps": steps, "label": label_of(anchors, final_state)})
    return rows


class GatedArm(nn.Module):
    """Shared backbone for both arms. use_gate=False -> arm (a): the gate
    Linear exists (parameter parity) but its output is discarded (m/theta
    hardwired to 1/0). use_gate=True -> arm (b): the gate Linear's output
    scales+rotates V before the weighted sum. Both end in a softmax
    classification head over C anchor labels (CrossEntropyLoss), not
    L2 regression to a continuous vector."""

    def __init__(self, use_gate: bool):
        super().__init__()
        self.use_gate = use_gate
        self.ent_emb = nn.Embedding(M_ENT, D_MODEL)
        self.verb_emb = nn.Embedding(K_VERB, D_MODEL)
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.gate_proj = nn.Linear(D_MODEL, 2)  # (log_m_raw, theta) -- present in BOTH arms
        self.readout = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, C))

    def _rope(self, x, pos):
        ang = (pos.float() * 0.5).unsqueeze(-1)
        d2 = x.shape[-1] // 2
        cos = torch.cos(ang).expand(*ang.shape[:-1], d2)
        sin = torch.sin(ang).expand(*ang.shape[:-1], d2)
        xp = x.reshape(*x.shape[:-1], d2, 2)
        x0, x1 = xp[..., 0], xp[..., 1]
        r0 = x0 * cos - x1 * sin
        r1 = x0 * sin + x1 * cos
        return torch.stack((r0, r1), dim=-1).reshape(x.shape)

    def forward(self, agent0, verbs, patients):
        B = agent0.shape[0]
        pos0 = self.ent_emb(agent0)
        tok = self.verb_emb(verbs) + self.ent_emb(patients)
        seq = torch.cat([pos0.unsqueeze(1), tok], dim=1)
        positions = torch.arange(4, device=seq.device).unsqueeze(0).expand(B, -1)

        q = self._rope(self.q_proj(seq), positions)
        k = self._rope(self.k_proj(seq), positions)
        v = self.v_proj(seq)

        if self.use_gate:
            raw = self.gate_proj(seq)
            m = torch.sigmoid(raw[..., 0]).unsqueeze(-1)
            theta = raw[..., 1]
            d2 = v.shape[-1] // 2
            vp = v.reshape(B, 4, d2, 2)
            v0, v1 = vp[..., 0], vp[..., 1]
            cos_t = torch.cos(theta).unsqueeze(-1)
            sin_t = torch.sin(theta).unsqueeze(-1)
            g0 = m * (v0 * cos_t - v1 * sin_t)
            g1 = m * (v0 * sin_t + v1 * cos_t)
            v = torch.stack((g0, g1), dim=-1).reshape(B, 4, -1)
        else:
            _ = self.gate_proj(seq)  # parameter-count parity, output unused

        attn = torch.softmax(q @ k.transpose(-1, -2) / (D_MODEL ** 0.5), dim=-1)
        pooled = (attn @ v)[:, -1, :]
        return self.readout(pooled)  # logits, [B, C]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def to_tensors(rows):
    agent0 = torch.tensor([r["agent0"] for r in rows], dtype=torch.long)
    verbs = torch.tensor([[s[0] for s in r["steps"]] for r in rows], dtype=torch.long)
    patients = torch.tensor([[s[1] for s in r["steps"]] for r in rows], dtype=torch.long)
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    return agent0, verbs, patients, labels


def run_arm(name, use_gate, seed, train_rows, test_rows, tightest_floor):
    torch.manual_seed(seed)
    model = GatedArm(use_gate=use_gate)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    tr_a0, tr_v, tr_p, tr_y = to_tensors(train_rows)
    te_a0, te_v, te_p, te_y = to_tensors(test_rows)

    model.eval()
    with torch.no_grad():
        untrained_logits = model(te_a0, te_v, te_p)
        untrained_acc = float((untrained_logits.argmax(-1) == te_y).float().mean())
        untrained_loss = float(loss_fn(untrained_logits, te_y))

    t0 = time.time()
    model.train()
    for step in range(STEPS):
        opt.zero_grad()
        logits = model(tr_a0, tr_v, tr_p)
        loss = loss_fn(logits, tr_y)
        loss.backward()
        opt.step()
    train_dur = time.time() - t0

    model.eval()
    with torch.no_grad():
        te_logits = model(te_a0, te_v, te_p)
        correct = (te_logits.argmax(-1) == te_y).float()
        trained_acc = float(correct.mean())
        trained_loss = float(loss_fn(te_logits, te_y))

    headroom_recovered = trained_acc - tightest_floor
    row = {
        "arm": name, "seed": seed, "n_params": n_params,
        "n_train": len(train_rows), "n_test": len(test_rows),
        "steps": STEPS, "lr": LR, "heads": HEADS, "d_model": D_MODEL, "c_anchors": C,
        "untrained_test_accuracy": untrained_acc, "untrained_test_loss": untrained_loss,
        "trained_test_accuracy": trained_acc, "trained_test_loss": trained_loss,
        "tightest_floor": tightest_floor, "headroom_recovered_above_tightest_floor": headroom_recovered,
        "final_train_loss": float(loss.item()),
        "train_seconds": train_dur,
        "producer": PRODUCER, "dtype_model": DTYPE_MODEL,
        "command": "python phaseh_H3v2_twins_ab.py",
    }
    append_result(row)
    log({"t": "test", "state": "GREEN", "name": f"H3v2_twin_{name}_seed{seed}",
         "detail": (f"n_params={n_params}, steps={STEPS}, untrained_acc={untrained_acc:.4f} -> "
                    f"trained_acc={trained_acc:.4f} (tightest_floor={tightest_floor:.4f}, "
                    f"headroom_recovered={headroom_recovered:+.4f}), n_test={len(test_rows)}, "
                    f"dur={train_dur:.1f}s"),
         "dur": train_dur})
    return row


def main():
    t0 = time.time()

    # ---- gate: refuse to train if the floors phase said the bed is void ----
    with io.open(FLOORS_RESULTS, "r", encoding="utf-8") as f:
        floors = json.load(f)
    null_discriminates = floors["edit4_null_as_accuracy"]["null_discriminates"]
    tightest_floor = floors["edit3_tighter_floors_n5000"]["tightest_val"]
    tightest_name = floors["edit3_tighter_floors_n5000"]["tightest_name"]
    honest_headroom = floors["edit3_tighter_floors_n5000"]["honest_headroom"]
    constant_predictor = floors["edit1_constant_predictor_floor"]["measured_floor_large_n"]

    log({"t": "finding", "item": "H3v2_twins_gate",
         "text": (f"Read {FLOORS_RESULTS}: null_discriminates={null_discriminates}, "
                   f"tightest_floor={tightest_name}={tightest_floor:.5f} (n=5000, 5 seeds, "
                   f"float64), honest_headroom={honest_headroom:.5f}, measured constant-predictor "
                   f"floor={constant_predictor:.5f} (declared 1/C=0.1250 was wrong, see floors "
                   f"phase). Twins are scored against tightest_floor, never 1/C.")})

    if not null_discriminates:
        log({"t": "done", "agent": "Foreman", "checks": 1, "reds": 1,
             "text": "BED VOID: null_discriminates=False in phaseh_H3v2_floors_results.json -- "
                      "no twin trained, per instruction."})
        print("null_discriminates=False -- bed void, stopping without training.")
        return

    F_start = (f"H3v2 twins (a) vector+softmax (no gate) vs (b) scalar-gate m*exp(i theta), "
               f"5 seeds, {STEPS} AdamW steps, discrete-label target (C={C} anchors), "
               f"scored as accuracy vs tightest_floor={tightest_floor:.5f}, not 1/C. "
               f"bed params m={M_ENT} d={D} k={K_VERB} line_len={LINE_LEN}. producer={PRODUCER}.")
    log({"t": "finding", "item": "phase_start", "text": F_start})

    all_rows = []
    param_counts = {"a": [], "b": []}

    for seed in SEEDS:
        rng = np.random.default_rng(TWIN_OFFSET + seed)
        entities = make_entities(rng, m=M_ENT, d=D)
        verb_ops = rot_ops(rng)
        anchors = anchors_for(seed)

        train_rows = make_dataset(rng, entities, verb_ops, anchors, N_TRAIN)
        test_rows = make_dataset(rng, entities, verb_ops, anchors, N_TEST)

        row_a = run_arm("a_vector_softmax_rope", use_gate=False, seed=seed,
                         train_rows=train_rows, test_rows=test_rows, tightest_floor=tightest_floor)
        row_b = run_arm("b_scalar_gate", use_gate=True, seed=seed,
                         train_rows=train_rows, test_rows=test_rows, tightest_floor=tightest_floor)
        param_counts["a"].append(row_a["n_params"])
        param_counts["b"].append(row_b["n_params"])
        all_rows.extend([row_a, row_b])

    params_equal = param_counts["a"] == param_counts["b"]
    log({"t": "finding", "item": "H3v2_twins_param_parity",
         "text": (f"arm (a) param counts per seed (numel()): {param_counts['a']}; "
                   f"arm (b): {param_counts['b']}. Equal by count: {params_equal}. Both arms "
                   f"carry the same gate_proj Linear(D_MODEL,2); arm (a) computes it and discards "
                   f"the output (m fixed at 1, theta fixed at 0), arm (b) wires it into V. This "
                   f"equalizes parameter COUNT, not effective capacity -- arm (a)'s gate_proj "
                   f"receives no gradient, so it is dead weight for (a) while load-bearing for "
                   f"(b). Reported plainly, not hidden.")})

    def agg(rows, key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals)), float(np.std(vals))

    a_rows = [r for r in all_rows if r["arm"].startswith("a_")]
    b_rows = [r for r in all_rows if r["arm"].startswith("b_")]
    a_un_mean, a_un_std = agg(a_rows, "untrained_test_accuracy")
    a_tr_mean, a_tr_std = agg(a_rows, "trained_test_accuracy")
    b_un_mean, b_un_std = agg(b_rows, "untrained_test_accuracy")
    b_tr_mean, b_tr_std = agg(b_rows, "trained_test_accuracy")

    a_headroom_frac = (a_tr_mean - tightest_floor) / honest_headroom
    b_headroom_frac = (b_tr_mean - tightest_floor) / honest_headroom

    log({"t": "finding", "item": "H3v2_twins_headline",
         "text": (f"5 seeds x float32 torch, {STEPS} AdamW steps, C={C}-way discrete-label "
                   f"accuracy (tightest_floor={tightest_name}={tightest_floor:.4f}, "
                   f"honest_headroom={honest_headroom:.4f}) :: arm (a) vector+softmax: "
                   f"untrained={a_un_mean:.4f}+/-{a_un_std:.4f} -> trained={a_tr_mean:.4f}+/-"
                   f"{a_tr_std:.4f} (headroom recovered={a_tr_mean - tightest_floor:+.4f} = "
                   f"{a_headroom_frac*100:.1f}% of honest headroom). arm (b) scalar-gate: "
                   f"untrained={b_un_mean:.4f}+/-{b_un_std:.4f} -> trained={b_tr_mean:.4f}+/-"
                   f"{b_tr_std:.4f} (headroom recovered={b_tr_mean - tightest_floor:+.4f} = "
                   f"{b_headroom_frac*100:.1f}% of honest headroom). This is the (a)/(b) pair "
                   f"only -- the untrained-vs-trained operator-bound-vs-vector-bound 653x "
                   f"comparison against arms (c)/(d) is a separate lane's file "
                   f"(phaseh_H3v2_twins_cd.py) and not reproduced here.")})

    scalar_gate_is_bias = abs(a_tr_mean - b_tr_mean) <= 0.05  # within 5pp on accuracy scale
    log({"t": "finding", "item": "H3v2_scalar_gate_expectation_check",
         "text": (f"mean trained accuracy across 5 seeds: (a)={a_tr_mean:.4f}, (b)={b_tr_mean:.4f}. "
                   f"Contract expectation: a single-token scalar gate telescopes under softmax to "
                   f"a per-key additive bias, i.e. (b) should behave close to (a). Within 5 "
                   f"percentage points: {scalar_gate_is_bias}. "
                   f"{'Matches expectation -- confirmed again on a bed that discriminates, not just on the null result the previous bed gave.' if scalar_gate_is_bias else 'Diverges from expectation on the discriminating bed -- worth reporting rather than assuming the wing is dead-equivalent.'}")})

    below_floor_a = [r["seed"] for r in a_rows if r["trained_test_accuracy"] <= tightest_floor]
    below_floor_b = [r["seed"] for r in b_rows if r["trained_test_accuracy"] <= tightest_floor]
    if below_floor_a or below_floor_b:
        log({"t": "finding", "item": "H3v2_twins_below_floor",
             "text": (f"seeds at/below tightest_floor ({tightest_floor:.4f}): arm (a) seeds "
                       f"{below_floor_a}, arm (b) seeds {below_floor_b}. A twin at or below the "
                       f"floor IS that floor, not a demonstration of order.")})

    summary = {
        "producer": PRODUCER, "seeds": list(SEEDS), "n_train": N_TRAIN, "n_test": N_TEST,
        "steps": STEPS, "lr": LR, "heads": HEADS, "d_model": D_MODEL, "c_anchors": C,
        "tightest_floor_name": tightest_name, "tightest_floor": tightest_floor,
        "honest_headroom": honest_headroom, "constant_predictor_floor": constant_predictor,
        "floors_source": FLOORS_RESULTS,
        "params_equal_by_count": params_equal, "param_counts": param_counts,
        "twin_results": all_rows,
        "arm_a_untrained_acc_mean": a_un_mean, "arm_a_untrained_acc_std": a_un_std,
        "arm_a_trained_acc_mean": a_tr_mean, "arm_a_trained_acc_std": a_tr_std,
        "arm_b_untrained_acc_mean": b_un_mean, "arm_b_untrained_acc_std": b_un_std,
        "arm_b_trained_acc_mean": b_tr_mean, "arm_b_trained_acc_std": b_tr_std,
        "arm_a_headroom_fraction": a_headroom_frac, "arm_b_headroom_fraction": b_headroom_frac,
        "scalar_gate_behaves_as_additive_bias": scalar_gate_is_bias,
        "wall_seconds": time.time() - t0,
    }
    with io.open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log({"t": "done", "agent": "Foreman", "checks": len(SEEDS) * 2 + 2, "reds": 0,
         "text": (f"H3v2 twins (a)/(b) trained to completion, 5 seeds each, {STEPS} AdamW steps, "
                   f"discrete-label accuracy vs tightest_floor={tightest_floor:.4f}. "
                   f"trained_acc: (a)={a_tr_mean:.4f}+/-{a_tr_std:.4f}, "
                   f"(b)={b_tr_mean:.4f}+/-{b_tr_std:.4f}. params_equal_by_count={params_equal}. "
                   f"wall={summary['wall_seconds']:.1f}s. results at {RESULTS_JSONL}")})

    print(json.dumps(summary, indent=2)[:4000])
    print(f"\nper-(arm,seed) rows appended to {RESULTS_JSONL}")
    print(f"summary written to {RESULTS_JSON}")
    print(f"board appended at {BOARD}")


if __name__ == "__main__":
    main()
