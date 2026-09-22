"""shell_bed.py -- floors for three shell-game state-tracking beds (S3, S4, A5).

Bed spec (verbatim, one line per bed):
S3: 3 cups, tokens = the 3 transpositions of {0,1,2} ((0 1),(0 2),(1 2)), ball starts at cup 0, label at prefix length t = cup holding ball after first t moves, L=64, tokens uniform iid, 3 classes.
S4: 4 cups, tokens = the 6 transpositions of {0,1,2,3}, same rule, L=64, 4 classes.
A5: 5 cups, tokens = the 20 distinct 3-cycles of {0,1,2,3,4} (a b c): a->b->c->a, ball starts at cup 0, L=64, 5 classes.

Move semantics: token is a permutation given as one cycle; ball at cup a maps to
the next cup in that cycle if a is in the cycle, else stays put.
"""
import sys
import os
import json
import time
import itertools

import numpy as np

CAM = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(CAM, "shell_bed_floors.json")
L = 64
POSITIONS = [8, 16, 32, 48, 64]  # 1-indexed prefix length
EVAL_N = 2000
EVAL_SEED = 12345
F2_SAMPLE_N = 20000
R_VALUES = [64, 256]
TRAIN_BUDGET_S = 150.0
TRAIN_BUDGET_S_CPU_FALLBACK = 90.0
LOG_EVERY = 200
BATCH = 256


# --------------------------------------------------------------- bed defs

def s3_moves():
    return [(0, 1), (0, 2), (1, 2)]


def s4_moves():
    return list(itertools.combinations(range(4), 2))


def a5_moves():
    pts = range(5)
    seen = set()
    cycles = []
    for a, b, c in itertools.permutations(pts, 3):
        key = frozenset([(a, b), (b, c), (c, a)])
        if key in seen:
            continue
        seen.add(key)
        cycles.append((a, b, c))
    return cycles


def move_table(n_cups, cycles):
    """next_cup[token, cup] -> cup after applying that token's cycle move."""
    n_tok = len(cycles)
    tab = np.tile(np.arange(n_cups), (n_tok, 1))
    for ti, cyc in enumerate(cycles):
        m = len(cyc)
        for i in range(m):
            a = cyc[i]
            b = cyc[(i + 1) % m]
            tab[ti, a] = b
    return tab.astype(np.int64)


BEDS = {
    "S3": dict(n_cups=3, cycles=s3_moves()),
    "S4": dict(n_cups=4, cycles=s4_moves()),
    "A5": dict(n_cups=5, cycles=a5_moves()),
}
for name, spec in BEDS.items():
    spec["n_tok"] = len(spec["cycles"])
    spec["table"] = move_table(spec["n_cups"], spec["cycles"])

BED_SPEC_TEXT = {
    "S3": "3 cups, tokens = 3 transpositions of {0,1,2}, ball starts cup 0, "
          "label[t]=cup after first t+1 moves (1-indexed), L=64, iid uniform tokens, 3 classes.",
    "S4": "4 cups, tokens = 6 transpositions of {0,1,2,3}, same rule, L=64, 4 classes.",
    "A5": "5 cups, tokens = 20 distinct 3-cycles of {0,1,2,3,4} (a b c): a->b->c->a, "
          "same rule, L=64, 5 classes.",
}


def gen_sequences(n_tok, n_seq, L, rng):
    return rng.integers(0, n_tok, size=(n_seq, L), dtype=np.int64)


def labels_from_tokens(tokens, table):
    """tokens: (N, L) int64. Returns labels (N, L): ball position after first t+1 moves."""
    N, Lx = tokens.shape
    n_cups = table.shape[1]
    ball = np.zeros(N, dtype=np.int64)
    labels = np.empty((N, Lx), dtype=np.int64)
    for t in range(Lx):
        tok_t = tokens[:, t]
        ball = table[tok_t, ball]
        labels[:, t] = ball
    return labels


# --------------------------------------------------------------- F0 chance

def compute_f0(bed_name, spec, tokens, labels):
    n_cups = spec["n_cups"]
    chance = 1.0 / n_cups
    per_pos = {}
    for p in POSITIONS:
        col = labels[:, p - 1]
        counts = np.bincount(col, minlength=n_cups)
        majority_freq = counts.max() / col.shape[0]
        per_pos[str(p)] = {"majority_class_freq": float(majority_freq)}
    return {"chance": chance, "per_position": per_pos}


# --------------------------------------------------------------- F1 multiset ceiling

def compute_f1(bed_name, spec, tokens, labels, rng):
    """Fully vectorized over (N sequences x R shuffles); loop only over the
    t sequential move-application steps (t <= 64)."""
    n_cups = spec["n_cups"]
    table = spec["table"]
    N = tokens.shape[0]
    out = {}
    for R in R_VALUES:
        per_pos = {}
        for p in POSITIONS:
            t = p  # prefix length = p
            prefix = tokens[:, :t]  # (N, t)
            true_lab = labels[:, p - 1]  # (N,)
            # R random permutations of the t moves, per sequence: (N, R, t)
            rand = rng.random((N, R, t), dtype=np.float32)
            perm_idx = np.argsort(rand, axis=2)  # (N, R, t) indices into [0,t)
            ball = np.zeros((N, R), dtype=np.int64)
            for tt in range(t):
                perm_tt = perm_idx[:, :, tt]  # (N, R) values in [0, t)
                tok_tt = np.take_along_axis(prefix, perm_tt, axis=1)  # (N, R)
                ball = table[tok_tt, ball]  # (N, R)
            # histogram over the R axis for each of N sequences
            onehot = (ball[:, :, None] == np.arange(n_cups)[None, None, :])  # (N,R,n_cups)
            hist = onehot.sum(axis=1)  # (N, n_cups)
            plugin_vals = hist.max(axis=1) / R
            argmax_pred = hist.argmax(axis=1)
            argmax_correct = (argmax_pred == true_lab)
            per_pos[str(p)] = {
                "plugin": float(plugin_vals.mean()),
                "argmax_acc": float(argmax_correct.mean()),
            }
        mean_ge32_plugin = float(np.mean([per_pos[str(p)]["plugin"] for p in (32, 48, 64)]))
        mean_ge32_argmax = float(np.mean([per_pos[str(p)]["argmax_acc"] for p in (32, 48, 64)]))
        out[f"R{R}"] = {
            "per_position": per_pos,
            "mean_ge32": {"plugin": mean_ge32_plugin, "argmax_acc": mean_ge32_argmax},
        }
    return out


# --------------------------------------------------------------- F2 last-w window Bayes

def compute_f2(bed_name, spec, tokens, labels, rng):
    n_cups = spec["n_cups"]
    n_tok = spec["n_tok"]
    table = spec["table"]
    N_eval = tokens.shape[0]
    out = {}
    # separate 20000-seq sample for the marginal prior
    sample_tokens = gen_sequences(n_tok, F2_SAMPLE_N, L, rng)
    sample_labels = labels_from_tokens(sample_tokens, table)
    for w in (4, 8):
        per_pos = {}
        for p in POSITIONS:
            t = p  # prefix length
            if t - w < 1:
                # position before window start is before the sequence begins;
                # ball is at cup 0 deterministically at "time 0"
                prior = np.zeros(n_cups, dtype=np.float64)
                prior[0] = 1.0
            else:
                # prior over ball position BEFORE the last w moves, i.e. at
                # prefix length (t - w), using empirical marginal from sample
                col = sample_labels[:, (t - w) - 1]
                counts = np.bincount(col, minlength=n_cups).astype(np.float64)
                prior = counts / counts.sum()
            window = tokens[:, max(t - w, 0):t]  # (N_eval, min(w,t))
            true_lab = labels[:, p - 1]
            # push prior through the window moves exactly, vectorized over N_eval
            dist = np.tile(prior, (N_eval, 1))  # (N_eval, n_cups)
            idx_n = np.arange(N_eval)
            for step in range(window.shape[1]):
                tok = window[:, step]  # (N_eval,)
                new_dist = np.zeros((N_eval, n_cups), dtype=np.float64)
                for c in range(n_cups):
                    target = table[tok, c]  # (N_eval,)
                    np.add.at(new_dist, (idx_n, target), dist[:, c])
                dist = new_dist
            pred = dist.argmax(axis=1)
            correct = int((pred == true_lab).sum())
            per_pos[str(p)] = {"acc": correct / N_eval}
        mean_ge32 = float(np.mean([per_pos[str(p)]["acc"] for p in (32, 48, 64)]))
        out[f"w{w}"] = {"per_position": per_pos, "mean_ge32": mean_ge32}
    return out


# --------------------------------------------------------------- F3 transformer twin

class TwinModel:
    def __init__(self, n_cups, n_tok, d_model=64, n_heads=4, n_layers=2, seq_len=64, device="cpu"):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn

        class Block(nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.ln1 = nn.LayerNorm(d)
                self.qkv = nn.Linear(d, 3 * d)
                self.proj = nn.Linear(d, d)
                self.ln2 = nn.LayerNorm(d)
                self.mlp = nn.Sequential(
                    nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
                )
                self.h = h
                self.d = d

            def forward(self, x):
                import torch.nn.functional as F
                B, T, D = x.shape
                h = self.h
                dh = D // h
                y = self.ln1(x)
                qkv = self.qkv(y).reshape(B, T, 3, h, dh).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn = attn.transpose(1, 2).reshape(B, T, D)
                x = x + self.proj(attn)
                x = x + self.mlp(self.ln2(x))
                return x

        class Net(nn.Module):
            def __init__(self, n_tok, n_cups, d, h, nl, seq_len):
                super().__init__()
                self.tok_emb = nn.Embedding(n_tok, d)
                self.pos_emb = nn.Embedding(seq_len, d)
                self.blocks = nn.ModuleList([Block(d, h) for _ in range(nl)])
                self.ln_f = nn.LayerNorm(d)
                self.head = nn.Linear(d, n_cups)
                self.seq_len = seq_len

            def forward(self, tok):
                B, T = tok.shape
                pos = torch.arange(T, device=tok.device).unsqueeze(0)
                x = self.tok_emb(tok) + self.pos_emb(pos)
                for blk in self.blocks:
                    x = blk(x)
                x = self.ln_f(x)
                return self.head(x)

        self.net = Net(n_tok, n_cups, d_model, n_heads, n_layers, seq_len).to(device)
        self.device = device
        self.n_params = sum(p.numel() for p in self.net.parameters())


def compute_f3(bed_name, spec, eval_tokens, eval_labels, device, budget_s):
    import torch
    import torch.nn.functional as F

    n_cups = spec["n_cups"]
    n_tok = spec["n_tok"]
    table = spec["table"]

    torch.manual_seed(0)
    model = TwinModel(n_cups, n_tok, d_model=64, n_heads=4, n_layers=2, seq_len=L, device=device)
    net = model.net
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)

    rng = np.random.default_rng(999)  # independent train-data stream, infinite/fresh each step
    log = []
    t0 = time.perf_counter()
    step = 0
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= budget_s:
            break
        batch_tokens = gen_sequences(n_tok, BATCH, L, rng)
        batch_labels = labels_from_tokens(batch_tokens, table)
        tok_t = torch.from_numpy(batch_tokens).to(device)
        lab_t = torch.from_numpy(batch_labels).to(device)
        logits = net(tok_t)  # (B, T, C)
        loss = F.cross_entropy(logits.reshape(-1, n_cups), lab_t.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % LOG_EVERY == 0:
            log.append([step, time.perf_counter() - t0, float(loss.item())])

    total_elapsed = time.perf_counter() - t0
    final_loss = float(loss.item())

    # eval
    net.eval()
    with torch.no_grad():
        tok_t = torch.from_numpy(eval_tokens).to(device)
        logits = net(tok_t)  # (N, T, C)
        preds = logits.argmax(dim=-1).cpu().numpy()  # (N, T)
    per_pos = {}
    for p in POSITIONS:
        acc = float((preds[:, p - 1] == eval_labels[:, p - 1]).mean())
        per_pos[str(p)] = {"acc": acc}
    mean_ge32 = float(np.mean([per_pos[str(p)]["acc"] for p in (32, 48, 64)]))

    # plateau flag: relative change of mean train loss between 3rd-to-last
    # quarter and last quarter of logged points < 0.02
    plateau = None
    if len(log) >= 4:
        n = len(log)
        q = n // 4
        last_q = log[n - q:] if q > 0 else log[-1:]
        third_last_q = log[n - 3 * q: n - 2 * q] if q > 0 else log[:1]
        if len(last_q) > 0 and len(third_last_q) > 0:
            mean_last = np.mean([r[2] for r in last_q])
            mean_third_last = np.mean([r[2] for r in third_last_q])
            if mean_third_last != 0:
                rel_change = abs(mean_last - mean_third_last) / abs(mean_third_last)
                plateau = bool(rel_change < 0.02)

    return {
        "n_params": model.n_params,
        "steps": step,
        "wall_s": total_elapsed,
        "final_loss": final_loss,
        "train_loss_log": log,
        "per_position_acc": per_pos,
        "mean_ge32_acc": mean_ge32,
        "plateau": plateau,
        "device": device,
        "budget_s": budget_s,
    }


# --------------------------------------------------------------- driver

def write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    t_start = time.perf_counter()
    result = {
        "eval_seed": EVAL_SEED,
        "eval_n": EVAL_N,
        "L": L,
        "positions": POSITIONS,
        "R_values": R_VALUES,
        "bed_spec": BED_SPEC_TEXT,
        "beds": {},
        "wall_times_s": {},
    }

    eval_data = {}
    for bed_name, spec in BEDS.items():
        t0 = time.perf_counter()
        rng = np.random.default_rng(EVAL_SEED)
        tokens = gen_sequences(spec["n_tok"], EVAL_N, L, rng)
        labels = labels_from_tokens(tokens, spec["table"])
        eval_data[bed_name] = (tokens, labels, rng)

        f0 = compute_f0(bed_name, spec, tokens, labels)
        f1 = compute_f1(bed_name, spec, tokens, labels, rng)
        f2 = compute_f2(bed_name, spec, tokens, labels, rng)

        result["beds"][bed_name] = {
            "n_cups": spec["n_cups"],
            "n_tok": spec["n_tok"],
            "F0": f0,
            "F1": f1,
            "F2": f2,
        }
        result["wall_times_s"][f"{bed_name}_cpu_floors"] = time.perf_counter() - t0
        write_json(result, JSON_PATH)
        print(f"[{bed_name}] CPU floors (F0,F1,F2) done, written to {JSON_PATH}", flush=True)

    result["wall_times_s"]["cpu_phase_total"] = time.perf_counter() - t_start
    write_json(result, JSON_PATH)
    print("CPU phase complete.", flush=True)

    # ---------------------------------------------------------- GPU phase
    sys.path.insert(0, r"C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad")
    import design4x5
    import torch

    gpu_busy_flag = os.path.join(CAM, "GPU_BUSY_CAMERON")
    while os.path.exists(gpu_busy_flag):
        print("[GPU] GPU_BUSY_CAMERON present, waiting 10s...", flush=True)
        time.sleep(10)

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        ok = design4x5.poll_until_free(timeout_s=900)
        if not ok:
            print("[GPU] poll_until_free timed out after 900s, falling back to CPU for F3", flush=True)
            cuda_ok = False

    device = "cuda" if cuda_ok else "cpu"
    budget_s = TRAIN_BUDGET_S if cuda_ok else TRAIN_BUDGET_S_CPU_FALLBACK
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "cpu"
    result["torch_version"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    result["device_used"] = device
    result["device_name"] = device_name
    result["f3_note"] = ("F3 ran on CUDA." if cuda_ok else
                          "CUDA unavailable or GPU never freed; F3 ran on CPU with a 90s/bed budget.")

    t_gpu0 = time.perf_counter()
    for bed_name, spec in BEDS.items():
        tokens, labels, _ = eval_data[bed_name]
        t0 = time.perf_counter()
        f3 = compute_f3(bed_name, spec, tokens, labels, device, budget_s)
        result["beds"][bed_name]["F3"] = f3
        result["wall_times_s"][f"{bed_name}_f3"] = time.perf_counter() - t0
        write_json(result, JSON_PATH)
        print(f"[{bed_name}] F3 done in {time.perf_counter()-t0:.1f}s, steps={f3['steps']}, "
              f"final_loss={f3['final_loss']:.4f}, mean_ge32_acc={f3['mean_ge32_acc']:.4f}", flush=True)

    result["wall_times_s"]["gpu_phase_total"] = time.perf_counter() - t_gpu0
    result["wall_times_s"]["total"] = time.perf_counter() - t_start
    write_json(result, JSON_PATH)
    print("ALL DONE.", flush=True)


if __name__ == "__main__":
    main()
