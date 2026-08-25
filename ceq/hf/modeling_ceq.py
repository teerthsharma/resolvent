"""A causal LM whose attention IS a signed path sum -- no softmax underneath.

    out = v + A v + A^2 v + ... + A^K v        A strictly causal, SIGNED

THE DEFAULT OPERATOR IS THE ONE THAT REACHED PARITY, AND IT IS `sgate`.

    sgate    A = rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)
    signed   A = rho * w / sum_j |w_ij|

`sgate` at rho=1.5 lam=0.10 hops=2 lr=1e-3 measured val loss **1.0334x** softmax,
median over 5 seeds (1.0199-1.0413), at **3,319,296 parameters on both arms** --
byte-level TinyStories, d=256, 4 layers, seq 128, 600 steps. `signed` measured
**1.337x** on the same harness with the gap WIDENING as the budget grew, and it
ships only as the negative control. Until 2026-08-25 this file implemented
`signed` alone, so the operator in the package was not the operator the number
was about; `test_the_package_implements_the_operator_that_reached_parity` now
compares bitwise against `ceq/lm.py`, where the parity run actually ran.

READ `COSTS` BELOW BEFORE ADOPTING ANYTHING, AND READ `COSTS["capability"]`
FIRST. Val-loss parity is the favourable number. On COGS generalization at
3,652,096 parameters in both arms the operator scored **0.0000 against softmax's
0.0293**, one-sided Fisher exact **p = 2.8e-05**, with the deficit already
present in-distribution (0.7734 against 0.9258). Training peak memory is up to
8.06x SDPA, wall clock 1.32x, there is no KV cache, and the property that
distinguishes this operator from softmax decays as `s^-1.389` in context length.

WHY THIS FILE DUPLICATES `ceq/attention.py` INSTEAD OF IMPORTING IT

The Hub copies the modeling file and its RELATIVE imports into a flat repository.
`from ceq.attention import ceq_operator` is an absolute import of a package that
will not be installed on the downloader's machine, so the operator has to live
here. Duplicated code drifts, so each operator has a bitwise oracle and they are
DIFFERENT files, because the two operators were written in different places:

  signed  vs `ceq/attention.py`, by
          `test_the_shipped_operator_is_bitwise_the_one_that_was_tested`
  sgate   vs `ceq/lm.py`, by
          `test_the_package_implements_the_operator_that_reached_parity`

`ceq/attention.py` contains NO sgate implementation, so it cannot be the oracle
for the shipped default; `ceq/lm.py` is where the 5-seed parity run ran and is.
The mask guard below is duplicated into `ceq/attention.py` as well, under one
parametrised test, for the same anti-drift reason.

WHY THERE IS NO TRITON IN THIS FILE

`ceq/mz_kernel.py` is FORWARD-ONLY -- it has no backward pass -- so it cannot
appear in a training step at all, and the training path is pure torch by
necessity rather than by preference. Two consequences worth stating plainly:

  * the sm_80 floor and the Linux-only PyPI wheel do NOT gate training, so a
    free-tier Colab T4 (sm_75) can run this;
  * a top-level `import triton` would make the checkpoint fail to LOAD, not
    merely to run slowly, on Windows, macOS and every CPU-only box -- and HF
    download counts are dominated by exactly those machines.

So pure torch is the ONLY path here, and it is the CPU fallback at the same time:
there is no `if q.is_cuda` branch anywhere in this file, which is what
`test_the_model_forwards_on_cpu_in_a_subprocess_with_cuda_hidden` checks in a
genuinely CUDA-less interpreter.

WHAT THIS MODEL CANNOT DO

Incremental decoding from a KV cache. `(A^h v)_i = sum_{j<i} A_ij (A^{h-1} v)_j`
needs the whole prefix's hop vectors, and during decode the operator row is
[1, N] so `A^2` is undefined. `use_cache` is forced to False in the config and
`generate()` recomputes the full prefix each step -- correct, and O(N^2) per
token. `ceq/hopcache.py` is the structure that fixes it at K vectors per
position; it is deliberately not wired in here.

RUN THE SMOKE TEST BEFORE TRUSTING ANY OF THIS: `python -m ceq.hf.smoke`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast

from .configuration_ceq import CEQConfig


# ------------------------------------------------------------- the price list

#: Every cost this module has been measured to carry, IN THE FILE THAT SHIPS.
#: The Hub copies `configuration_ceq.py` and `modeling_ceq.py` and nothing else,
#: so a cost recorded only in README.md does not reach the person who downloads
#: the checkpoint. Machine for the memory and wall-clock rows: RTX 4060 Laptop,
#: sm_89, 8.0 GiB, 24 SMs, torch 2.5.1+cu121, fp32.
#: `test_the_package_states_its_costs_in_the_file_that_ships` and
#: `test_every_cost_the_package_states_is_reachable_from_a_recorded_measurement`
#: pin these against DONE.md, which is the run log rather than a summary.
COSTS = {
    #: seq -> peak TRAINING memory (forward + backward) as a multiple of dense
    #: causal SDPA at identical parameters. SDPA never forms [S,S]; this operator
    #: materializes it and autograd retains a measured 3.9 tensors of [B,H,S,S]
    #: per layer, so activations are O(S^2) against O(S). The 1.31-1.94x figure
    #: that circulated earlier was INFERENCE, forward-only, seq 128-512.
    "train_peak_memory_vs_sdpa": {128: 1.44, 256: 1.74, 512: 2.65,
                                  1024: 4.45, 2048: 8.06},
    #: Seconds per run at the parity point against the softmax control:
    #: 21.3 s vs 16.1 s over 600 steps.
    "wall_clock_vs_softmax_at_parity": 1.32,
    #: Peak memory with `gradient_checkpointing_enable()`, seq 1024:
    #: 1287.5 -> 487.7 MiB, loss identical to 1e-4. MANDATORY, not optional --
    #: a 0.5B signed model at seq 2048 batch 1 needs 36.31 GiB against a
    #: 33.53 GiB A100-40GB budget and 20.25 GiB on an L4.
    "gradient_checkpointing_peak_ratio": 0.379,
    #: There is a Triton kernel in this repository and it CANNOT be used to
    #: train: it has no backward pass. The training path is pure torch, which is
    #: why a free-tier Colab T4 (sm_75) can run it and why nothing here imports
    #: triton at module scope.
    "kernel": "forward-only; the training path is pure torch",
    #: The distinguishing property -- whether a THIRD token can flip the sign of
    #: j's influence on i -- is a SHORT-CONTEXT property. Rate by context length
    #: s, with the fitted power law. Softmax reads exactly 0.0000 at every s, so
    #: the property is real; it is also diluted by 1/s because it is carried
    #: entirely by the k>=2 term of the path sum and a third token is 1 of ~s
    #: intermediates in that sum. No operating point escapes it: lam 0.05-2.00,
    #: rho 0.5/1.5/4.0, depth 2 and 3, and head width 16/32/64/128 all read
    #: 0.00000 at s=128 while holding 0.13-0.18 at s=8.
    #:
    #: THE FIRST VERSION OF THIS ENTRY HAD 0.10547 AT s=16 AND 0.03516 AT s=32.
    #: Both were wrong -- DONE.md:975 and DONE.md:1104 independently record
    #: 0.08887 and 0.02637 -- and the test that guards this dict only checked the
    #: endpoints, so it did not catch them. It checks every entry now.
    "content_conditional_sign_decay": {8: 0.17480, 16: 0.08887, 32: 0.02637,
                                       64: 0.01172, 128: 0.00391,
                                       "slope": -1.389, "r2": 0.9938,
                                       "softmax_at_every_s": 0.0},
    #: What "parity" means here, exactly and with its limits.
    "parity": {"ratio": 1.0334, "params": 3319296, "seeds": 5,
               "size": "3.3M", "seq": 128},
    #: AND THE CAPABILITY NUMBER, WHICH IS A LOSS. Val-loss parity is the
    #: favourable measurement and it is not the one that decides anything: at
    #: 3,652,096 parameters in BOTH arms, identical steps / lr / batch / seed /
    #: data / eval subsample, COGS generalization exact match over 512 items was
    #: softmax 0.0293 (15/512) against sgate 0.0000 (0/512), one-sided Fisher
    #: exact p = 2.8e-05. The deficit is ALREADY THERE in-distribution -- 0.7734
    #: against 0.9258 on the split both arms trained on -- so it is not a story
    #: about generalization. For scale, a published from-scratch encoder-decoder
    #: on the same task reaches 0.35 +/- 0.06.
    #:
    #: This entry exists because a price list that carries 1.0334 and omits this
    #: is advertising. `python -m ceq.capability`, `results/capability.json`,
    #: `tests/cameron/test_capability_result.py`.
    "capability": {"task": "COGS generalization, exact match, 512 items",
                   "cogs_gen_exact_match": {"softmax": 0.0293, "sgate": 0.0000},
                   "cogs_in_distribution": {"softmax": 0.9258, "sgate": 0.7734},
                   "fisher_one_sided_p": 2.8e-05,
                   "params_both_arms": 3652096,
                   "published_from_scratch_reference": 0.35,
                   "verdict": "a loss, not a tie, and already present in-distribution"},
    #: No KV cache exists for the multi-hop path sum, so `generate()` recomputes
    #: the whole prefix every step.
    "decode": "no KV cache; O(N^2) per token, use_cache is forced False",
    #: Loading this repository executes code from it.
    "trust_remote_code": "required; some organisations block it by policy",
    #: SIGNEDNESS IS EMERGENT, NOT STRUCTURAL, at the shipped lam = 0.10. An
    #: entry goes negative only when `softmax(w)_ij < lam * softmax(-w)_ij`, so
    #: it needs the entry to sit far below its own row. At `nn.Linear` default
    #: initialization the logits are too flat: measured negative fraction 3.91e-04
    #: (cpu) / 4.99e-04 (cuda) over 20 draws, max 1.085e-03, at a mean within-row
    #: logit spread of 0.68 -- against 5.25e-02 at unit-variance logits (spread
    #: 3.26) and 0.225 at 8x. The -0.1080 minimum on record is the TRAINED 3.3M
    #: model. At lam = 1.0 signedness is structural instead (rows sum to exactly
    #: zero) at the cost of annihilating the constant vector.
    "signedness": {"negative_fraction_at_nn_linear_init": {"cpu": 3.91e-04,
                                                           "cuda": 4.99e-04},
                   "negative_fraction_at_unit_variance_logits": 5.25e-02,
                   "min_A_trained_3p3M": -0.1080,
                   "structural_at_lam": 1.0},
    #: The residual gate (`ceq/hybrid.py`) at alpha = 0 is bitwise the model's
    #: own attention BY CONSTRUCTION -- it calls transformers'
    #: `sdpa_attention_forward` rather than reimplementing it. Reimplementing it
    #: was off by 1.490e-07 on a GQA model, which is what the earlier "bitwise"
    #: claim actually measured.
    "alpha_zero_gate": "bitwise the model's own sdpa at every head grouping",
}


# ---------------------------------------------------------------- observability

#: Counters a caller can read back. The most expensive bug this project has
#: produced twice is a constraint that was REPORTED applied and was not -- a
#: BlockMask that `to_dense()` showed sparse while the kernel computed fully
#: dense, and an `is_causal` that was correct in prefill and wrong at decode.
#: Neither raised. So the mask path here both raises on an unrecognised
#: convention and leaves a number behind that can be compared with what the
#: caller asked for.
#: COST OF THE COUNTER, stated rather than assumed: one reduction over the mask
#: per call. A padding mask is [B,1,1,S] and that is free; a full [B,1,S,S] mask
#: makes it O(S^2), which is 1/H of the operator's own [B,H,S,S] work and the
#: same order as the float-mask validation directly above it.
STATS = {"calls": 0, "masked_key_positions": 0}


def _keep_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Validate a padding mask and return it as a boolean "may attend" tensor.

    Two conventions are legal and they are the only two:

      * bool, True = attend;
      * additive float, 0.0 = attend and `<= finfo(dtype).min / 2` = masked.

    ANYTHING ELSE RAISES. `-10000.0` is the historical transformers mask value
    and is still emitted by live code; it is not below `finfo(float32).min / 2`
    (-1.7e38), so the previous implementation read it as "not masked" and
    returned an operator BITWISE EQUAL to the unmasked one while the masked
    columns kept 0.48290979862213135 of weight. The caller was told the
    constraint was applied. It was not.

    Causality is NOT taken from here -- both operators apply `tril(-1)`
    unconditionally -- so a caller who omits the mask loses padding, never
    causality.
    """
    if attention_mask is None:
        return None
    m = attention_mask
    if m.dtype == torch.bool:
        keep = m
    else:
        floor = torch.finfo(m.dtype).min / 2
        if bool(((m < 0) & (m > floor)).any()):
            raise ValueError(
                "unrecognised attention mask convention: the mask contains "
                "negative values above {:.3e}, which this operator cannot "
                "distinguish from 'not masked'. Legal masks are bool (True = "
                "attend) or additive float with 0.0 for attend and <= {:.3e} "
                "for masked. The historical -10000.0 convention silently "
                "produced a DENSE operator here; min value seen {:.6g}."
                .format(floor, floor, float(m.min())))
        keep = m >= floor
    STATS["masked_key_positions"] += int((~keep).sum())
    return keep


# --------------------------------------------------------------- the operator

def _strict_causal_mask(s: int, device) -> torch.Tensor:
    """Strictly lower triangular: token i reads j < i, never itself.

    Excluding the diagonal is what makes A nilpotent. Include it and the operator
    has a self-loop, rho(A) becomes the largest diagonal entry, and the finite
    path sum stops being exact.
    """
    return torch.ones(s, s, dtype=torch.bool, device=device).tril(-1)


def ceq_operator(query: torch.Tensor, key: torch.Tensor,
                 attention_mask: torch.Tensor | None = None,
                 *, rho: float = 0.9,
                 scaling: float | None = None,
                 eps: float = 0.0) -> torch.Tensor:
    """The signed, strictly causal coupling operator A.

    Signed by construction: the raw logits are the weights, normalized by their
    row L1 norm rather than pushed through a softmax. A non-negative operator has
    a non-negative influence Jacobian in any ordered semiring and therefore
    cannot express a negation -- measured minimum exactly 0.000e+00 over 40
    max-plus and 160 APPNP instances, against -9.000e-01 here.

    Row 0 has no predecessors, so its L1 norm is exactly zero; the clamp makes
    that row exactly zero instead of NaN.

    KNOWN CONDITIONING HAZARD, MEASURED AND NOT FIXED HERE. The forward is
    exactly scale-invariant -- every row is renormalized to L1 = rho -- but the
    backward is not: |dA/dq| scales as 1/l1, verified at exactly 10x per decade
    over six decades. At initialization the row L1 already spans 1.1e-03 (min) to
    1.67e+01 (median), a factor of 1.5e+04, and over 800 steps on real text the
    median rises to 53.07 while the minimum falls to 9.35e-07. See
    `tests/chase/test_signed_operator_trainability.py`. No NaN was produced in
    800 steps; the hazard is a conditioning spread inside one global grad clip,
    not an immediate blow-up. The rollback is `rho * w / (l1 + eps)`, which
    bounds the Jacobian by rho/eps and leaves the sign untouched.
    `eps = 0.0` is BITWISE the operator that shipped, so turning the rollback
    on is a decision rather than a default that quietly moved.
    """
    STATS["calls"] += 1
    s = query.shape[-2]
    scale = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])
    w = (query @ key.transpose(-2, -1)) * scale

    keep = _keep_mask(attention_mask)
    if keep is not None:
        w = w.masked_fill(~keep, 0.0)

    w = w.masked_fill(~_strict_causal_mask(s, w.device), 0.0)
    l1 = w.abs().sum(-1, keepdim=True) + eps
    return rho * w / l1.clamp_min(torch.finfo(w.dtype).tiny)


def sgate_operator(query: torch.Tensor, key: torch.Tensor,
                   attention_mask: torch.Tensor | None = None,
                   *, rho: float = 1.5, lam: float = 0.10,
                   scaling: float | None = None) -> torch.Tensor:
    """THE OPERATOR THAT REACHED PARITY. A difference of two softmaxes over the
    SAME logits:

        A = rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)

    Signed, because the negative half can dominate any given entry; strictly
    causal, because both halves are masked to `tril(-1)`; and sharp, because both
    halves keep `exp`. The L1 form it replaced has none of that sharpness: it is
    exactly homogeneous of degree zero, so `A(t*q, k) = A(q, k)` and peak row
    weight was 0.100336 at logit scales 0.25, 1.0, 4.0 and 16.0 alike -- spread
    exactly 0.000e+00 over a 64x sweep, against softmax 0.0504 -> 1.0000. Every
    query permanently lost a degree of freedom. That is why it lost 1.337x.

    `lam` trades the two remaining obstructions. At lam = 1 the halves each sum
    to 1, rows sum to exactly zero, and A annihilates the constant vector
    (measured row sum 0.000000e+00, |A @ 1| = 1.192e-07) -- the path sum can then
    add deviations but never signal level. At lam = 0 the negative half is gone
    and the operator is non-negative, which is tier 2 and provably cannot express
    a negation. 0.10 is the swept optimum.

    ROW 0, AND WHY THERE IS NO NaN. Row 0 has no predecessors, so every entry is
    filled with `finfo.min` before the softmax. `softmax` over a row of equal
    finite values is uniform rather than NaN, and the post-mask zeroes it. The
    all-masked-row case is handled here rather than left to the caller.

    THE BOUND THIS DOES NOT HAVE. `||A||_inf <= rho` still holds, but rho = 1.5
    at the shipped point, so the geometric tail bound `rho^(K+1)/(1-rho)` does
    not apply -- it is negative there, and `ceq.attention.truncation_bound`
    raises rather than returning -6.75. hops = 2 is a genuine truncation with no
    bound. Nilpotency is untouched (`CEQ.Nilpotent.pow_card_eq_zero` has no
    non-negativity and no magnitude hypothesis), but exactness through nilpotency
    needs hops >= S-1.

    Bitwise identical to `ceq.lm.Attention.operator` at kind="sgate" AND AT EQUAL
    (rho, lam) -- which is worth spelling out, because `ceq/lm.py` reads those
    from module globals and their COMMITTED values are the pre-campaign
    `RHO, HOPS = 0.9, 3` and `SGATE_LAM = 1.0`, not the parity point. So the file
    where the 5-seed run ran does not itself carry the numbers it ran with; the
    run set them, and the bitwise test sets them the same way. That is a defect
    in `ceq/lm.py` rather than here, and it is recorded as open.

    The division by `math.sqrt` rather than a multiplication by its reciprocal is
    deliberate: for a head dim that is not a power of two the two differ in the
    last bit and the bitwise test would fail for a reason that has nothing to do
    with the operator.
    """
    STATS["calls"] += 1
    s = query.shape[-2]
    w = (query @ key.transpose(-2, -1))
    w = w * scaling if scaling is not None else w / math.sqrt(query.shape[-1])

    keep = _strict_causal_mask(s, w.device)
    pad = _keep_mask(attention_mask)
    if pad is not None:
        # Padding must enter BEFORE the softmax. Zeroing afterwards would leave
        # the masked keys holding softmax mass that the surviving keys never get
        # back -- rows would sum to less than they should and the operator would
        # quietly shrink wherever a batch was padded.
        keep = keep & pad

    neg = torch.finfo(w.dtype).min
    pp = torch.softmax(w.masked_fill(~keep, neg), dim=-1).masked_fill(~keep, 0.0)
    pm = torch.softmax((-w).masked_fill(~keep, neg), dim=-1).masked_fill(~keep, 0.0)
    return rho * (pp - lam * pm) / (1.0 + lam)


def path_sum(a: torch.Tensor, v: torch.Tensor, hops: int) -> torch.Tensor:
    """v + A v + ... + A^hops v. A direct method, not a fixed-point search."""
    z = term = v
    for _ in range(hops):
        term = a @ term
        z = z + term
    return z


# ------------------------------------------------------------------ the blocks

class CEQAttention(nn.Module):
    """Carries the SAME parameters a softmax block would: qkv and an output
    projection, nothing extra. The operator itself has no parameters, which is
    what makes a matched-parameter comparison against softmax possible at all."""

    def __init__(self, config: CEQConfig):
        super().__init__()
        d = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.d_head = d // self.n_heads
        self.rho, self.hops, self.lam = config.rho, config.hops, config.lam
        #: Resolved to a CALLABLE at construction. A name looked up per forward
        #: would silently fall back on a typo; `CEQConfig` already rejects an
        #: unknown name, and this is the second place it cannot slip through.
        self.operator = config.operator
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, attention_mask=None):
        b, s, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def shape(t):
            return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        if self.operator == "sgate":
            a = sgate_operator(shape(q), shape(k), attention_mask,
                               rho=self.rho, lam=self.lam)
        else:
            a = ceq_operator(shape(q), shape(k), attention_mask, rho=self.rho)
        o = path_sum(a, shape(v), self.hops)
        return self.o_proj(o.transpose(1, 2).reshape(b, s, d))


class CEQBlock(nn.Module):
    def __init__(self, config: CEQConfig):
        super().__init__()
        d, f = config.hidden_size, config.intermediate_size
        self.input_layernorm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.self_attn = CEQAttention(config)
        self.mlp = nn.Sequential(nn.Linear(d, f), nn.GELU(), nn.Linear(f, d))

    def forward(self, x, attention_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask)
        return x + self.mlp(self.post_attention_layernorm(x))


# ------------------------------------------------------------------ the models

class CEQPreTrainedModel(PreTrainedModel):
    config_class = CEQConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CEQBlock"]
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


class CEQModel(CEQPreTrainedModel):
    def __init__(self, config: CEQConfig):
        super().__init__(config)
        d = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, d)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, d)
        self.layers = nn.ModuleList(CEQBlock(config)
                                    for _ in range(config.num_hidden_layers))
        self.norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                return_dict=True, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        s = inputs_embeds.shape[1]
        pos = torch.arange(s, device=inputs_embeds.device)
        x = inputs_embeds + self.embed_positions(pos)[None]

        # A padding mask arrives as [B, S]; the operator wants [B, 1, 1, S] so it
        # broadcasts over heads and query positions. Causality is NOT taken from
        # here -- `ceq_operator` applies tril(-1) unconditionally -- so a caller
        # who forgets the mask loses padding, never causality.
        m = None
        if attention_mask is not None and attention_mask.dim() == 2:
            m = attention_mask[:, None, None, :].bool()
        elif attention_mask is not None:
            m = attention_mask

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # THE MEMORY ROLLBACK. Recomputing the block in the backward pass
                # drops the retained [S,S] operator from ~3.9 tensors per layer to
                # one live block's worth. Measured on an RTX 4060 Laptop at
                # seq=1024: 1287.5 -> 487.7 MiB = 0.379x, loss identical to 1e-4.
                # This comment previously said "1271.2 MiB -> under 0.6x", which
                # disagreed with COSTS in the same file; DONE.md records 1287.5
                # and 0.379x, so that is what both now say. The operator is
                # unchanged, so the signed property is untouched.
                x = torch.utils.checkpoint.checkpoint(layer, x, m,
                                                      use_reentrant=False)
            else:
                x = layer(x, m)
        x = self.norm(x)
        if not return_dict:
            return (x,)
        return BaseModelOutput(last_hidden_state=x)


class CEQForCausalLM(CEQPreTrainedModel, GenerationMixin):
    """`GenerationMixin` is inherited EXPLICITLY.

    transformers 5.3.0 no longer has `PreTrainedModel` inherit it, and
    `can_generate()` returns True only when `"GenerationMixin" in
    str(cls.__bases__)`. A model that trains fine and cannot `.generate()` is a
    checkpoint nobody can use, and nothing raises to say so.
    """

    #: A MAPPING in transformers 5.x, not the list it was in 4.x. The 4.x form
    #: raises `AttributeError: 'list' object has no attribute 'keys'` from
    #: `modeling_utils.py:2489` at construction time, before any forward runs.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: CEQConfig):
        super().__init__(config)
        self.model = CEQModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # NO get_input_embeddings / get_output_embeddings OVERRIDES. `PreTrainedModel`
    # inherits `EmbeddingAccessMixin`, which finds them from `base_model_prefix`
    # plus the standard attribute names `embed_tokens` and `lm_head`. Overriding
    # them here breaks weight tying on RELOAD, silently: `save_pretrained` writes
    # only `model.embed_tokens.weight` (correct), and `from_pretrained` then
    # leaves `lm_head.weight` on the META device with `tied = False` and logs a
    # warning claiming both keys are in the checkpoint. Measured against a
    # `LlamaForCausalLM` control built the same way, which ties correctly.
    # Bound by `tests/chase/test_ceq_hub_package.py::
    # test_registering_for_auto_class_does_not_break_weight_tying` (a strict
    # xfail: the failure message IS the finding) and its rollback
    # `test_the_shipped_config_routes_around_the_tying_defect`.

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                labels=None, return_dict=True, **kwargs):
        # THE LOUD END OF THE TYING DEFECT, and the reason it is checked in the
        # forward rather than at load: at load nothing is wrong yet, and by the
        # time anything is wrong the caller already has a number. Measured with
        # `tie_word_embeddings=True` on transformers 5.3.0: `lm_head.weight`
        # stays on the META device, `save_pretrained` succeeds,
        # `from_pretrained` logs a warning and raises nothing, and the forward
        # RETURNS -- logits[0,0,0] = 1.0053620544046395e+30, uninitialized
        # memory, max abs difference 1.0053620544046395e+30 against the model
        # that was saved. One attribute read per forward buys that back.
        if self.lm_head.weight.is_meta:
            raise RuntimeError(
                "lm_head.weight is on the meta device: it was never materialized "
                "and this forward would return uninitialized memory, not logits "
                "(measured 1.0053620544046395e+30 with nothing raised). Cause: "
                "`PreTrainedModel.is_remote_code()` is `cls._auto_class is not "
                "None`, so `register_for_auto_class()` -- required to ship "
                "modeling code on the Hub -- routes the load onto a branch that "
                "refuses to tie the embeddings. Fix: set "
                "`tie_word_embeddings=False`, which is the shipped default and "
                "costs +7.9% parameters. See "
                "tests/chase/test_ceq_hub_package.py::"
                "test_registering_for_auto_class_does_not_break_weight_tying")
        h = self.model(input_ids=input_ids, attention_mask=attention_mask,
                       inputs_embeds=inputs_embeds).last_hidden_state
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1))
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """No cache, by construction -- see the module docstring. The whole
        prefix is re-read every step, which is exact and O(N^2) per token."""
        s = self.config.max_position_embeddings
        return {"input_ids": input_ids[:, -s:],
                "attention_mask": kwargs.get("attention_mask")}


CEQForCausalLM.register_for_auto_class("AutoModelForCausalLM")
CEQModel.register_for_auto_class("AutoModel")
