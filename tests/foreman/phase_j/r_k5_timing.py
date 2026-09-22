"""Row R-K5: timing-only comparison of three attention forward+backward paths
at the grid shape (hidden 128, 3 layers, 8 heads, seq 512, batch 8, vocab
256), 50 steps after 10 warmup, torch.cuda.synchronize() around the timed
region. No training to convergence -- random data, random init, discarded
after timing.

Paths:
  (a-flash)    F.scaled_dot_product_attention(is_causal=True)          [design4x5 softmax_forward pattern]
  (a-explicit) same, but an explicit boolean causal attn_mask, is_causal=False
  (a'')        arms_j.fox_forward -- FoX float mask (forget-gate score bias)

SDPA backend is checked two ways per path: (1) torch.backends.cuda.sdp_kernel
context-manager probe (call each backend forced, see which raises), and (2)
torch.profiler with record_shapes, grepping kernel names in the trace for
flash/mem_efficient/math signatures.
"""
import json
import sys
import time

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
PHASE_J = SCRATCH + r"\phase_j"
REPO = r"C:\Users\seal\Desktop\New folder (32)"
sys.path.insert(0, PHASE_J)
sys.path.insert(0, SCRATCH)
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import design4x5 as D  # noqa: E402
import arms_j  # noqa: E402

HIDDEN, LAYERS, HEADS, SEQ, BATCH, VOCAB = 128, 3, 8, 512, 8, 256
DEV = "cuda"
D_HEAD = HIDDEN // HEADS
WARMUP, STEPS = 10, 50


class Attn(nn.Module):
    """Minimal stand-in exposing the same qkv/o_proj/n_heads/d_head surface
    design4x5's softmax_forward / arms_j.fox_forward expect, at the required
    shape -- avoids building the full CEQ model stack for a pure timing row."""

    def __init__(self, arm):
        super().__init__()
        self.n_heads, self.d_head = HEADS, D_HEAD
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN)
        if arm == "a2F":
            self.forget_head = nn.Linear(HIDDEN, HEADS)
        self.arm = arm

    def forward(self, x):
        if self.arm == "a-flash":
            return _flash_forward(self, x)
        elif self.arm == "a-explicit":
            return _explicit_forward(self, x)
        elif self.arm == "a2F":
            return arms_j.fox_forward(self, x)
        raise ValueError(self.arm)


def _shape(self, t, b, s):
    return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)


def _flash_forward(self, x):
    """design4x5.softmax_forward pattern, verbatim math."""
    b, s, d = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    o = F.scaled_dot_product_attention(_shape(self, q, b, s), _shape(self, k, b, s),
                                        _shape(self, v, b, s), is_causal=True)
    return self.o_proj(o.transpose(1, 2).reshape(b, s, d))


_CAUSAL_MASK_CACHE = {}


def _explicit_forward(self, x):
    """Same math, but is_causal=False plus an explicit boolean attn_mask
    (no gate) -- the K5 comparison point."""
    b, s, d = x.shape
    if s not in _CAUSAL_MASK_CACHE:
        m = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), diagonal=1)
        _CAUSAL_MASK_CACHE[s] = ~m  # True where attend allowed, SDPA bool mask semantics
    mask = _CAUSAL_MASK_CACHE[s]
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    o = F.scaled_dot_product_attention(_shape(self, q, b, s), _shape(self, k, b, s),
                                        _shape(self, v, b, s), attn_mask=mask, is_causal=False)
    return self.o_proj(o.transpose(1, 2).reshape(b, s, d))


class Layer(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.self_attn = Attn(arm)
        self.head = nn.Linear(HIDDEN, VOCAB)

    def forward(self, x):
        h = x + self.self_attn(x)
        return h


class Stack(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([Layer(arm) for _ in range(LAYERS)])
        self.lm_head = nn.Linear(HIDDEN, VOCAB)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def time_path(arm, label):
    torch.manual_seed(0)
    model = Stack(arm).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV)

    def step():
        opt.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        loss.backward()
        opt.step()
        return loss

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_per_step = (t1 - t0) * 1000.0 / STEPS

    # backend check 1: sdp_kernel forced-context probe
    backends_ok = {}
    x = torch.randn(2, HEADS, 16, D_HEAD, device=DEV)
    for name, flag in [("flash", "enable_flash"), ("mem_efficient", "enable_mem_efficient"),
                        ("math", "enable_math")]:
        kwargs = {"enable_flash": False, "enable_mem_efficient": False, "enable_math": False}
        kwargs[flag] = True
        try:
            with torch.backends.cuda.sdp_kernel(**kwargs):
                F.scaled_dot_product_attention(x, x, x, is_causal=True)
            backends_ok[name] = True
        except RuntimeError:
            backends_ok[name] = False

    # backend check 2: profiler trace, grep kernel names on one real step
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                             torch.profiler.ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages()]
    hit = {"flash": any("flash" in n.lower() for n in names),
           "efficient": any("efficient" in n.lower() for n in names),
           "math": any("fused_attention" not in n.lower() and
                       ("softmax" in n.lower() or "bmm" in n.lower()) for n in names)}

    del model, opt
    torch.cuda.empty_cache()
    return dict(label=label, ms_per_step=ms_per_step,
                sdp_kernel_forced_ok=backends_ok, profiler_kernel_hits=hit)


if __name__ == "__main__":
    D.poll_until_free(timeout_s=900)
    started = time.strftime("%H:%M:%S")
    results = []
    for arm, label in [("a-flash", "a-flash"), ("a-explicit", "a-explicit"), ("a2F", "a''")]:
        r = time_path(arm, label)
        print("[TIMED]", json.dumps(r), flush=True)
        results.append(r)
    finished = time.strftime("%H:%M:%S")

    ms = {r["label"]: r["ms_per_step"] for r in results}
    ratio_app_expl = ms["a''"] / ms["a-explicit"]
    ratio_expl_flash = ms["a-explicit"] / ms["a-flash"]
    ratio_app_flash = ms["a''"] / ms["a-flash"]
    bar_pass = ratio_app_expl <= 1.30
    out = dict(row="R-K5", started=started, finished=finished,
               grid=dict(hidden=HIDDEN, layers=LAYERS, heads=HEADS, seq=SEQ,
                         batch=BATCH, vocab=VOCAB),
               warmup=WARMUP, steps=STEPS,
               ms_per_step=ms,
               ratio_app_over_explicit=ratio_app_expl,
               ratio_explicit_over_flash=ratio_expl_flash,
               ratio_app_over_flash=ratio_app_flash,
               bar="(a'')/(a-explicit) <= 1.30",
               verdict="PASS" if bar_pass else "FAIL",
               backend_detail=results)
    print(json.dumps(out, indent=2), flush=True)
    with open(PHASE_J + r"\r_k5_result.json", "w") as f:
        json.dump(out, f, indent=2)
