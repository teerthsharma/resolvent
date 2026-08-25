"""Sizing arithmetic for a 0.5B-1B model whose attention is the signed path sum.

Every coefficient in here is CALIBRATED against the CUDA allocator on the box
that runs `tests/chase/test_scale_sizing.py`, not assumed. A sizing model that
has never been compared against a scale is a spreadsheet.

THE ONE NUMBER THAT DECIDES EVERYTHING

SDPA never forms the [S, S] score matrix -- a fused kernel streams tiles and its
activation memory is O(S). This operator materializes A explicitly, and autograd
retains it: measured **3.9 tensors of shape [B, H, S, S] per layer**, against
SDPA's zero. So the softmax arm's activations grow as O(S) and this one's grow as
O(S^2), and the ratio is not a constant -- it is a function of S. Measured on an
RTX 4060 Laptop (sm_89, 8.0 GiB, torch 2.5.1+cu121), B=4 d=256 L=4 H=4, fp32,
forward + backward:

    seq    softmax    signed     ratio
     128     36.1       51.9      1.44
     256     74.1      128.7      1.74
     512    143.5      379.9      2.65
    1024    285.8     1271.2      4.45
    2048    570.3     4596.0      8.06

The forward-only kernel figure already on record is 1.31x-1.94x. That is the
128-512 band. It does not describe training at 2048 and it must not be quoted as
if it did.
"""
from __future__ import annotations

import dataclasses
import math

# ---------------------------------------------------------------- coefficients
#
# C_RESIDUAL   activation elements per token per layer that BOTH arms pay: two
#              LayerNorms, the qkv output, the attention output, the projection,
#              the 4d MLP hidden and its GELU, and the residual adds.
# C_OPERATOR   [B,H,S,S] tensors retained per layer by the SIGNED arm only.
#              The masked-filled `w`, the `rho*w` product and the divided `A` are
#              each retained by something downstream; `w.abs()` is transient but
#              lands inside the same peak.
# C_HEAD       copies of the [B,S,V] logits alive at once -- the logits plus
#              cross_entropy's log-softmax.
#
# Fitted against the table in the module docstring; worst residual +6.0% at
# seq=256, best +1.3% at seq=128.
C_RESIDUAL = 18
C_OPERATOR = 3.9
C_HEAD = 2

#: EFFECTIVE bytes per activation element, (residual_term, operator_term).
#:
#: bf16 autocast DOES NOT HALVE THIS, and assuming it does was the one place this
#: module was optimistic. Measured on an RTX 4060 Laptop (sm_89, torch 2.5.1+cu121),
#: B=4 d=256 L=4 H=4, forward + backward, bf16 autocast peak against fp32 peak:
#:
#:     seq    softmax           signed            operator-term delta
#:      256   52.6/90.5 = 0.582  97.7/130.6 = 0.748   2.89 bytes/elem
#:      512   95.5/145.4 = 0.657 294.6/381.7 = 0.772  3.19 bytes/elem
#:     1024  175.1/286.6 = 0.611 1014.0/1272.1 = 0.797 3.36 bytes/elem
#:
#: The operator term lands at 2.89-3.36 effective bytes, not 2, and it RISES with
#: sequence length, so the large-S figure is the one carried. The residual term
#: fits 2.2. Autograd retains fp32 alongside the bf16 casts, the logits and
#: cross_entropy stay fp32, and LayerNorm is autocast-excluded.
DTYPE_MODES = {
    "fp32": (4.0, 4.0),
    "bf16_autocast": (2.2, 3.4),
}

HOPS = 3

#: Total VRAM in bytes, and the fraction of it an allocator can actually hand out
#: after the CUDA context, cuBLAS workspaces and fragmentation.
GPUS = {
    "A100-40GB": (40 * 1000 ** 3, 0.90),   # Colab reports 40960 MiB
    "A100-80GB": (80 * 1000 ** 3, 0.90),
    "L4-24GB": (22.5 * 1024 ** 3, 0.90),   # Colab reports ~22.5 GiB
    "T4-16GB": (15.0 * 1024 ** 3, 0.90),   # sm_75: CANNOT run the Triton kernel
    "RTX4060L-8GB": (8.0 * 1024 ** 3, 0.90),
}

#: Bytes of persistent state per parameter, weights included.
OPTIMIZERS = {
    "adamw_fp32": 16,          # p 4 + grad 4 + exp_avg 4 + exp_avg_sq 4
    "adamw_bf16_mixed": 18,    # the above, plus autocast's cached bf16 weight copy
    "adamw_8bit": 10,          # p 4 + grad 4 + two 1-byte moment states
    "sgd_momentum": 12,
}


@dataclasses.dataclass(frozen=True)
class Config:
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    vocab: int
    seq: int
    tied_embeddings: bool = True
    d_ff_mult: int = 4


def replace(cfg: Config, **kw) -> Config:
    return dataclasses.replace(cfg, **kw)


#: d_head is 64 and not 80, because `test_kernel_contracts.py::test_common_head_dims`
#: records head dims 80 and 96 as REJECTED by the kernel: only 16/32/64/128 are
#: supported. 20 heads of 64 is the nearest runnable factorization of d = 1280.
CFG_500M = Config(d_model=1280, n_layers=24, n_heads=20, d_head=64,
                  vocab=32000, seq=2048)

CFG_1B = Config(d_model=2048, n_layers=20, n_heads=32, d_head=64,
                vocab=32000, seq=2048)

#: The GPT-2-medium shape, which is what "300M" means in the literature: d 1024,
#: 24 layers, 16 heads of 64. d_head 64 is one of the four the kernel accepts.
#: `tied_embeddings=False` because `register_for_auto_class` breaks weight tying
#: in transformers 5.3.0 and the shipped rollback is to untie -- so the +vocab*d
#: is a cost this architecture actually pays, not a modelling choice.
CFG_300M = Config(d_model=1024, n_layers=24, n_heads=16, d_head=64,
                  vocab=32000, seq=2048, tied_embeddings=False)


# --------------------------------------------------------------------- params

def params(cfg: Config) -> int:
    """Parameter count. The operator itself carries none -- q, k, v and o belong
    to the block and are shared with the softmax arm, which is what makes the two
    arms comparable at all."""
    d, f = cfg.d_model, cfg.d_ff_mult * cfg.d_model
    per_layer = (3 * d * d + d * d) + (d * f + f * d) + 4 * d
    n = cfg.n_layers * per_layer
    n += cfg.vocab * d                 # token embedding
    n += cfg.seq * d                   # learned positions
    n += 2 * d                         # final LayerNorm
    if not cfg.tied_embeddings:
        n += cfg.vocab * d
    return n


# ---------------------------------------------------------------------- FLOPs

def flops_per_token(cfg: Config, *, arm: str, hops: int = HOPS,
                    include_head: bool = True) -> float:
    """Matmul FLOPs per token for one forward pass.

    Counts what `torch.utils.flop_counter.FlopCounterMode` counts: matmuls only.
    Elementwise work -- the abs, the row sum, the division, GELU, LayerNorm -- is
    excluded from both arms, so the ratio is not flattered by it.

    A NOTE ON CAUSALITY THAT THE RATIO HIDES. This counts the attention matmuls
    DENSE, for both arms, because that is what the counter does. In reality a
    fused causal SDPA kernel skips the upper triangle and does about half this
    work, while `ceq_operator` computes the full [S,S] product and then masks it.
    So the real attention-FLOP gap is about twice the gap reported here.
    """
    d, S = cfg.d_model, cfg.seq
    f = cfg.d_ff_mult * d
    proj = 2 * d * (3 * d) + 2 * d * d
    mlp = 2 * d * f + 2 * f * d
    if arm == "softmax":
        attn = 2 * S * d + 2 * S * d
    elif arm in ("signed", "sgate"):
        attn = 2 * S * d * (1 + hops)
    else:
        raise ValueError(arm)
    total = cfg.n_layers * (proj + mlp + attn)
    if include_head:
        total += 2 * d * cfg.vocab
    return float(total)


# --------------------------------------------------------------------- memory

def activation_bytes(cfg: Config, *, batch: int, arm: str, dtype: str = "bf16_autocast",
                     hops: int = HOPS, checkpointed: bool = False,
                     bytes_per: float | None = None) -> float:
    """Peak activation bytes for one forward + backward.

    `dtype` selects a MEASURED pair of effective bytes-per-element from
    `DTYPE_MODES`; `bytes_per` overrides both with a single number and exists so
    the calibration tests can drive the model at a known dtype. The logits are
    counted at 4 bytes regardless, because `cross_entropy` upcasts.
    """
    if bytes_per is not None:
        rb = ob = float(bytes_per)
    else:
        rb, ob = DTYPE_MODES[dtype]
    B, S, d, L = batch, cfg.seq, cfg.d_model, cfg.n_layers
    resid = C_RESIDUAL * L * B * S * d * rb
    head = C_HEAD * B * S * cfg.vocab * 4
    if arm == "softmax":
        return float(resid + head)
    if arm not in ("signed", "sgate"):
        raise ValueError(arm)
    # `sgate` runs softmax TWICE over the same [S, S] logits, so the retained
    # count was re-measured rather than assumed. RTX 4060 Laptop, B=4 d=256 L=4
    # H=4 fp32, fwd+bwd, backed out against the softmax arm at the same shape:
    # C_OPERATOR is 3.85 for `signed` and 3.90 for `sgate` at seq 1024, and the
    # measured totals are 1272.1 MiB against 1286.1 MiB = 1.011x. The second
    # softmax costs ~1%: `torch.softmax`'s backward needs only its own output,
    # and the masked-fill inputs it replaces were already retained.
    op_one = cfg.n_heads * B * S * S * ob
    if checkpointed:
        # one block's graph is alive at a time during recomputation
        return float(resid / L + C_OPERATOR * op_one + head)
    return float(resid + C_OPERATOR * L * op_one + head)


def state_bytes(cfg: Config, *, optimizer: str = "adamw_bf16_mixed") -> float:
    return float(params(cfg) * OPTIMIZERS[optimizer])


@dataclasses.dataclass(frozen=True)
class Fit:
    ok: bool
    gpu: str
    budget: float
    state: float
    activations: float
    cfg: Config
    batch: int
    arm: str

    @property
    def total(self) -> float:
        return self.state + self.activations

    def explain(self) -> str:
        g = 1024 ** 3
        return (
            "{arm} d={d} L={L} H={H} seq={S} batch={B}: params {p:,} -> state "
            "{st:.2f} GiB + activations {ac:.2f} GiB = {tot:.2f} GiB against a "
            "{bg:.2f} GiB budget on {gpu}. Headroom {hr:+.2f} GiB.".format(
                arm=self.arm, d=self.cfg.d_model, L=self.cfg.n_layers,
                H=self.cfg.n_heads, S=self.cfg.seq, B=self.batch,
                p=params(self.cfg), st=self.state / g, ac=self.activations / g,
                tot=self.total / g, bg=self.budget / g,
                gpu=self.gpu, hr=(self.budget - self.total) / g))


def fits(cfg: Config, *, batch: int, arm: str, gpu: str,
         optimizer: str = "adamw_bf16_mixed", dtype: str = "bf16_autocast",
         checkpointed: bool = False) -> Fit:
    total_vram, usable = GPUS[gpu]
    budget = total_vram * usable
    st = state_bytes(cfg, optimizer=optimizer)
    ac = activation_bytes(cfg, batch=batch, arm=arm, dtype=dtype,
                          checkpointed=checkpointed)
    return Fit(ok=(st + ac) <= budget, gpu=gpu, budget=budget, state=st,
               activations=ac, cfg=cfg, batch=batch, arm=arm)


def max_batch(cfg: Config, *, arm: str, gpu: str,
              optimizer: str = "adamw_bf16_mixed", dtype: str = "bf16_autocast",
              checkpointed: bool = False, cap: int = 4096) -> int:
    """Largest batch that fits. 0 means it does not fit at all."""
    b = 0
    while b < cap:
        if not fits(cfg, batch=b + 1, arm=arm, gpu=gpu, optimizer=optimizer,
                    dtype=dtype, checkpointed=checkpointed).ok:
            return b
        b += 1
    return cap


# ------------------------------------------------------------------ GPU-hours

#: Chinchilla-optimal tokens per parameter. Counted against NON-EMBEDDING
#: parameters, which is the convention the scaling-law literature uses and which
#: matters here: at d=1024 vocab 32000 UNTIED, the embeddings are 65.5M of
#: 369.7M and counting them would inflate the budget by 22%.
TOKENS_PER_PARAM = 20

#: A100-40GB bf16 dense peak, and the model-FLOPs utilization a well-tuned dense
#: transformer of this size reaches. 0.40 is deliberately at the optimistic end
#: for the CONTROL, because the control is the denominator: a generous baseline
#: makes the operator's multiple look smaller, not larger.
A100_PEAK_FLOPS, DEFAULT_MFU = 312e12, 0.40


def non_embedding_params(cfg: Config) -> int:
    d, f = cfg.d_model, cfg.d_ff_mult * cfg.d_model
    return cfg.n_layers * ((3 * d * d + d * d) + (d * f + f * d) + 4 * d)


def chinchilla_tokens(cfg: Config, per_param: int = TOKENS_PER_PARAM) -> float:
    return float(per_param * non_embedding_params(cfg))


def gpu_hours(cfg: Config, *, arm: str, tokens: float, hops: int = 2,
              checkpointed: bool = True, mfu: float = DEFAULT_MFU,
              peak_flops: float = A100_PEAK_FLOPS) -> float:
    """Hours to push `tokens` through one training run.

    `passes` is 3 forward-equivalents for a plain step -- one forward and a
    backward that costs two -- and 4 with gradient checkpointing, which recomputes
    every block's forward. Checkpointing defaults ON because
    `test_a_300m_sgate_model_trains_at_batch_1_on_a_colab_gpu` shows batch 1 is
    the alternative, and batch 1 is not a training run.

    Only meaningful for the SOFTMAX arm. The signed arm's MFU is not the
    control's -- it materializes [S,S] and is bandwidth-bound where a fused causal
    kernel is compute-bound -- so its cost is this number times a MEASURED
    wall-clock ratio, never this number at an assumed MFU.
    """
    passes = 4 if checkpointed else 3
    return tokens * flops_per_token(cfg, arm=arm, hops=hops) * passes / (
        peak_flops * mfu) / 3600.0


# ------------------------------------------------------------- the measurement

def measure_checkpointed_activation_bytes(kind, seq, device, *, bs, d, layers, heads):
    """Peak allocated bytes for forward + backward with every block recomputed.

    This is the stated rollback for the memory finding, run rather than asserted.
    `torch.utils.checkpoint` is stdlib-adjacent and already a dependency; nothing
    in the operator changes, so the signed property is untouched.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    from ceq import lm

    torch.manual_seed(0)
    m = lm.TinyLM(kind, d=d, n_layers=layers, n_heads=heads, seq=seq, seed=0).to(device)
    x = torch.randint(0, lm.VOCAB, (bs, seq), device=device)

    def fwd(idx):
        h = m.tok(idx) + m.pos(torch.arange(idx.shape[1], device=idx.device))[None]
        for blk in m.blocks:
            h = checkpoint(blk, h, use_reentrant=False)
        return m.head(m.norm(h))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    F.cross_entropy(fwd(x).reshape(-1, lm.VOCAB), x.reshape(-1)).backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del m, x
    torch.cuda.empty_cache()
    return peak


def report(cfg: Config = CFG_500M, *, gpus=("A100-40GB", "L4-24GB"),
           optimizer: str = "adamw_bf16_mixed") -> str:
    """The table this module exists to print."""
    g = 1024 ** 3
    out = ["params {:,}  state {:.2f} GiB @ {}".format(
        params(cfg), state_bytes(cfg, optimizer=optimizer) / g, optimizer)]
    fs = flops_per_token(cfg, arm="softmax")
    fg = flops_per_token(cfg, arm="signed")
    out.append("FLOP/token softmax {:.4e}  signed {:.4e}  ratio {:.3f}x".format(
        fs, fg, fg / fs))
    out.append("{:12} {:8} {:>7} {:>7} {:>9} {:>9}".format(
        "gpu", "arm", "maxB", "maxB+ck", "act@B1 GiB", "hdrm GiB"))
    for gpu in gpus:
        for arm in ("softmax", "signed"):
            f = fits(cfg, batch=1, arm=arm, gpu=gpu, optimizer=optimizer)
            out.append("{:12} {:8} {:7d} {:7d} {:9.2f} {:9.2f}".format(
                gpu, arm,
                max_batch(cfg, arm=arm, gpu=gpu, optimizer=optimizer),
                max_batch(cfg, arm=arm, gpu=gpu, optimizer=optimizer,
                          checkpointed=True),
                f.activations / g, (f.budget - f.total) / g))
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover
    print(report())
    print()
    print(report(CFG_1B))
