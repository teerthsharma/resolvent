"""GPT-style scale-ladder trainer. ALiBi reference attention arm + pluggable attention hook.
K1 copy of K0/wilson/train_ladder.py (sha d524b130...): + --eval_seed, harness sha, device memory, total wall time."""
import argparse, contextlib, glob, hashlib, importlib.util, json, math, os, subprocess, sys, tempfile, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 50304
D_HEAD = 64
RUNGS = {"R0": (4, 128), "R1": (6, 320), "R2": (8, 512), "R3": (12, 768)}

def get_slopes(n):
    def p2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        return [start * start ** i for i in range(n)]
    if math.log2(n).is_integer():
        return p2(n)
    c = 2 ** math.floor(math.log2(n))
    return p2(c) + get_slopes(2 * c)[0::2][:n - c]

_BIAS_CACHE = {}

class AlibiAttention(nn.Module):
    """Fixed-slope ALiBi attention: fused qkv, causal bias, nothing learned in the position path."""

    def __init__(self, d_model, n_heads, layer_idx, n_layers, ctx):
        super().__init__()
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("slopes", torch.tensor(get_slopes(n_heads), dtype=torch.float32), persistent=False)

    def _bias(self, T, device, dtype):
        # one cache for all layers: per-layer copies cost L x H x T^2 (4.8 GB at R3, ctx 4096)
        key = (self.n_heads, T, str(device), dtype)
        if key not in _BIAS_CACHE:
            i = torch.arange(T, dtype=torch.float32, device=device).view(T, 1)
            j = torch.arange(T, dtype=torch.float32, device=device).view(1, T)
            bias = -self.slopes.to(device).view(-1, 1, 1) * (i - j).view(1, T, T)
            bias = bias.masked_fill((j > i).view(1, T, T), float("-inf"))
            _BIAS_CACHE[key] = bias.unsqueeze(0).to(dtype)
        return _BIAS_CACHE[key]

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=self._bias(T, x.device, q.dtype))
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)

ATTENTION = {"alibi": AlibiAttention}

def load_external_attn(spec):
    path_str, cls_name = spec.rsplit(":", 1)
    with open(path_str, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    mod_spec = importlib.util.spec_from_file_location("external_attn", path_str)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return getattr(mod, cls_name), sha

class Block(nn.Module):
    def __init__(self, d_model, n_heads, layer_idx, n_layers, ctx, attn_cls):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn_cls(d_model, n_heads, layer_idx, n_layers, ctx)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, n_layers, d_model, ctx, attn_cls, vocab=VOCAB, skip_init=False):
        super().__init__()
        self.n_layers, self.d_model, self.ctx = n_layers, d_model, ctx
        n_heads = d_model // D_HEAD
        self.wte = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, i, n_layers, ctx, attn_cls) for i in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.wte.weight  # tied
        if not skip_init:
            self.apply(self._init_weights)
            std = 0.02 / math.sqrt(2 * n_layers)
            for block in self.blocks:
                nn.init.normal_(block.mlp[2].weight, mean=0.0, std=std)
                attn_out = getattr(block.attn, "out", None)
                if isinstance(attn_out, nn.Linear):
                    nn.init.normal_(attn_out.weight, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def build_model(n_layers, d_model, ctx, attn_cls, device="cpu"):
    dev = torch.device(device)
    with torch.device(dev):
        return GPT(n_layers, d_model, ctx, attn_cls, skip_init=(dev.type == "meta"))

# ---------------- data ----------------

def make_synthetic_shards(out_dir, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(2):
        rng.integers(0, 50257, size=200_000, dtype=np.uint16).tofile(os.path.join(out_dir, f"train_{i}.bin"))
    rng.integers(0, 50257, size=50_000, dtype=np.uint16).tofile(os.path.join(out_dir, "val_0.bin"))

def load_shards(data_dir, pattern):
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"no shards matching {pattern} in {data_dir}")
    return [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]

def _draw_batch(shards, rng, B, ctx):
    lengths = np.array([len(s) for s in shards], dtype=np.float64)
    probs = lengths / lengths.sum()
    shard = shards[rng.choice(len(shards), p=probs)]
    offsets = rng.integers(0, len(shard) - (ctx + 1) + 1, size=B)
    xs = np.stack([shard[o:o + ctx].astype(np.int64) for o in offsets])
    ys = np.stack([shard[o + 1:o + ctx + 1].astype(np.int64) for o in offsets])
    return torch.from_numpy(xs), torch.from_numpy(ys)

def sample_batch(shards, seed, k, B, ctx):
    return _draw_batch(shards, np.random.default_rng([seed, k]), B, ctx)

def build_eval_batches(shards, seed, n, B, ctx):
    rng = np.random.default_rng([seed, 999_999_999])
    return [_draw_batch(shards, rng, B, ctx) for _ in range(n)]

# ---------------- lr / checkpoint ----------------

def get_lr(step, total_steps, warmup, lr):
    if step <= warmup:
        return lr * step / max(1, warmup)
    if total_steps <= warmup:
        return lr
    progress = min(1.0, (step - warmup) / (total_steps - warmup))
    min_lr = 0.1 * lr
    return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

def raw_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model

def strip_prefix(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

def save_ckpt(path, model, optimizer, step, tokens, wall_s, args, device):
    ckpt = {
        "model": raw_model(model).state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "tokens": tokens, "wall_s": wall_s, "args": vars(args),
        "cpu_rng_state": torch.get_rng_state(),
    }
    if device.type == "cuda":
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)

# ---------------- training ----------------

def train(args):
    t_start = time.time()
    requested = args.device
    if requested == "cuda" and not args.no_poll:
        sys.path.insert(0, "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad")
        import design4x5
        if not design4x5.poll_until_free(timeout_s=600):
            print("GPU busy, refusing to run", file=sys.stderr)
            sys.exit(1)
    if args.deterministic:  # must precede cuBLAS init
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    device = torch.device("cuda") if (requested == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

    n_layers, d_model = RUNGS[args.rung]
    if args.layers is not None:
        n_layers = args.layers
    if args.width is not None:
        d_model = args.width
    n_heads = d_model // D_HEAD

    if args.attn in ATTENTION:
        attn_cls, attn_sha = ATTENTION[args.attn], None
    else:
        attn_cls, attn_sha = load_external_attn(args.attn)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.device(device):
        model = GPT(n_layers, d_model, args.ctx, attn_cls)
    params = sum(p.numel() for p in model.parameters())
    params_contract = 12 * n_layers * d_model * d_model + VOCAB * d_model

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt_kwargs = {"lr": args.lr, "betas": (0.9, 0.95), "eps": 1e-8}
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}], **opt_kwargs)

    if args.compile:
        model = torch.compile(model)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    step, tokens, wall_s, resumed_from_step = 0, 0, 0.0, None
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        raw_model(model).load_state_dict(strip_prefix(ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        step, tokens, wall_s = ckpt["step"], ckpt["tokens"], ckpt["wall_s"]
        torch.set_rng_state(ckpt["cpu_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        resumed_from_step = step

    log_f = open(os.path.join(args.out_dir, "log.jsonl"), "a")

    def log(rec):
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    log({"t": "config", **vars(args), "params": params, "params_contract": params_contract,
         "n_heads": n_heads, "torch_version": torch.__version__,
         "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
         "attn_spec": args.attn, "attn_sha256": attn_sha, "resumed_from_step": resumed_from_step,
         "harness_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
         "device_total_gib": torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if device.type == "cuda" else None})

    tokens_per_step = args.batch * args.accum * args.ctx
    if args.steps is not None:
        total_steps = args.steps
    else:
        target_tokens = args.tokens if args.tokens is not None else 20 * params
        total_steps = max(1, target_tokens // tokens_per_step)

    train_shards = load_shards(args.data_dir, "train_*.bin")
    val_shards = load_shards(args.data_dir, "val_*.bin")
    eval_batches = build_eval_batches(val_shards, args.seed if args.eval_seed is None else args.eval_seed,
                                      args.eval_batches, args.batch, args.ctx)

    use_autocast = device.type == "cuda" and args.dtype != "fp32"

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in eval_batches:
                x, y = x.to(device), y.to(device)
                with (torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()):
                    _, loss = model(x, y)
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    tok_s_hist, mfu_hist, peak_mem_gib = [], [], 0.0
    for step in range(step + 1, total_steps + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        lr = get_lr(step, total_steps, args.warmup, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        for micro in range(args.accum):
            k = (step - 1) * args.accum + micro
            x, y = sample_batch(train_shards, args.seed, k, args.batch, args.ctx)
            x, y = x.to(device), y.to(device)
            with (torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()):
                _, loss = model(x, y)
            loss = loss / args.accum
            loss.backward()
            loss_val += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_s = time.time() - t0
        wall_s += dt_s
        tokens += tokens_per_step
        tok_s = tokens_per_step / dt_s
        mfu = (6 * params + 6 * n_layers * args.ctx * d_model) * tok_s / (args.peak_tflops * 1e12)
        peak_mem_gib = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
        if device.type == "cuda" and torch.cuda.max_memory_allocated() > torch.cuda.get_device_properties(0).total_memory:
            # WDDM spills to shared system memory instead of raising OOM; such a run is ~15-30x slower
            log({"t": "abort", "step": step, "reason": "allocated > device memory (WDDM sysmem spill)",
                 "peak_mem_gib": peak_mem_gib, "tok_s": tok_s})
            sys.exit(3)
        tok_s_hist.append(tok_s)
        mfu_hist.append(mfu)
        log({"t": "step", "step": step, "loss": loss_val, "lr": lr, "tokens": tokens, "dt_s": dt_s,
             "wall_s": wall_s, "tok_s": tok_s, "mfu": mfu, "peak_mem_gib": peak_mem_gib})

        if step % args.eval_every == 0 or step == total_steps:
            log({"t": "eval", "step": step, "val_loss": evaluate(), "tokens": tokens})
        if step % args.ckpt_every == 0 or step == total_steps or step == args.stop_at:
            save_ckpt(ckpt_path, model, optimizer, step, tokens, wall_s, args, device)
        if step == args.stop_at:  # split a run without changing its schedule (resume with the same --steps)
            break

    def trimmed_mean(vals):
        tail = vals[10:] if len(vals) > 10 else vals
        return sum(tail) / len(tail) if tail else 0.0

    log({"t": "end", "step": step, "mean_tok_s_excl_first_10": trimmed_mean(tok_s_hist),
         "mean_mfu_excl_first_10": trimmed_mean(mfu_hist), "peak_mem_gib": peak_mem_gib,
         "wall_total_s": time.time() - t_start})
    log_f.close()

# ---------------- selftest ----------------

def _mark(ok, name, msg):
    print(f"{'PASS' if ok else 'FAIL'} ({name}) {msg}")
    return ok

def selftest_a():
    ok = True
    expected = {"R0": 7.2, "R1": 23.5, "R2": 50.9, "R3": 123.6}
    for rung, (L, d) in RUNGS.items():
        contract = 12 * L * d * d + VOCAB * d
        model = build_model(L, d, 1024, AlibiAttention, device="meta")
        measured = sum(p.numel() for p in model.parameters())
        got = round(contract / 1e6, 1)
        ok &= _mark(got == expected[rung], "a", f"{rung} params_contract={got}M expected={expected[rung]}M measured_params={measured}")
    return ok

def selftest_b():
    s8 = get_slopes(8)
    expect8 = [2.0 ** -k for k in range(1, 9)]
    ok = _mark(s8 == expect8, "b", f"get_slopes(8)={s8}")
    print(f"    get_slopes(5)={get_slopes(5)}")
    print(f"    get_slopes(12)={get_slopes(12)}")
    return ok

def selftest_c():
    torch.manual_seed(0)
    attn = AlibiAttention(320, 5, 0, 1, 1024).eval()
    x = torch.randn(2, 64, 320)
    with torch.no_grad():
        y = attn(x)
        B, T, D, H, dh = 2, 64, 320, 5, 64
        q, k, v = attn.qkv(x).split(D, dim=2)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        slopes = torch.tensor(get_slopes(H)).view(1, H, 1, 1)
        i = torch.arange(T).view(1, 1, T, 1).float()
        j = torch.arange(T).view(1, 1, 1, T).float()
        scores = scores - slopes * (i - j)
        scores = scores.masked_fill((j > i), float("-inf"))
        ref = torch.softmax(scores, dim=-1) @ v
        ref = ref.transpose(1, 2).contiguous().view(B, T, D)
        ref_out = attn.out(ref)
    diff = (y - ref_out).abs().max().item()
    return _mark(diff < 1e-5, "c", f"alibi_vs_reference max_abs_diff={diff:.2e}")

def selftest_d():
    torch.manual_seed(0)
    model = build_model(2, 128, 64, AlibiAttention, device="cpu").eval()
    T, t = 64, 30
    x = torch.randint(0, VOCAB, (1, T))
    with torch.no_grad():
        logits1, _ = model(x)
        x2 = x.clone()
        x2[0, t] = (x2[0, t] + 1) % VOCAB
        logits2, _ = model(x2)
    diff_before = (logits1[0, :t] - logits2[0, :t]).abs().max().item()
    diff_at_t = (logits1[0, t] - logits2[0, t]).abs().max().item()
    return _mark(diff_before < 1e-6 and diff_at_t > 1e-6, "d",
                 f"causality diff_before={diff_before:.2e} diff_at_t={diff_at_t:.2e}")

def _read_step_losses(log_path):
    losses = {}
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("t") == "step":
                losses[rec["step"]] = rec["loss"]
    return [losses[s] for s in sorted(losses)]

def selftest_e():
    script = os.path.abspath(__file__)
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as dA, tempfile.TemporaryDirectory() as dB:
        make_synthetic_shards(data_dir, seed=0)
        common = [sys.executable, script, "--layers", "2", "--width", "128", "--ctx", "64",
                  "--batch", "4", "--device", "cpu", "--data_dir", data_dir,
                  "--eval_batches", "1", "--eval_every", "1000", "--no_poll"]
        subprocess.run(common + ["--steps", "8", "--out_dir", dA], check=True, capture_output=True, text=True)
        subprocess.run(common + ["--steps", "4", "--out_dir", dB], check=True, capture_output=True, text=True)
        subprocess.run(common + ["--steps", "8", "--out_dir", dB, "--resume"], check=True, capture_output=True, text=True)
        lossesA = _read_step_losses(os.path.join(dA, "log.jsonl"))
        lossesB = _read_step_losses(os.path.join(dB, "log.jsonl"))
        losses_match = lossesA == lossesB
        ckA = torch.load(os.path.join(dA, "ckpt.pt"), map_location="cpu")["model"]
        ckB = torch.load(os.path.join(dB, "ckpt.pt"), map_location="cpu")["model"]
        params_match = set(ckA) == set(ckB) and all(torch.equal(ckA[k], ckB[k]) for k in ckA)
        return _mark(losses_match and params_match, "e",
                      f"resume lossesA={lossesA} lossesB={lossesB} params_equal={params_match}")

def run_selftest():
    results = [selftest_a(), selftest_b(), selftest_c(), selftest_d(), selftest_e()]
    return all(results)

# ---------------- cli ----------------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--rung", choices=list(RUNGS), default="R0")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--attn", default="alibi")
    p.add_argument("--data_dir", default="C:/Users/seal/datasets/fineweb_edu")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--tokens", type=int, default=None)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--ckpt_every", type=int, default=1_000_000_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop_at", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--no_poll", action="store_true")
    p.add_argument("--peak_tflops", type=float, default=26.77)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval_seed", type=int, default=None)  # fixed validation batches across seeds
    p.add_argument("--selftest", action="store_true")
    return p

def main():
    args = build_argparser().parse_args()
    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)
    train(args)

if __name__ == "__main__":
    main()
