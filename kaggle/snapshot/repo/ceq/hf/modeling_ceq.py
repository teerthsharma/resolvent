"""A causal LM whose attention IS a signed path sum -- no softmax underneath.

    out = v + A v + A^2 v + ... + A^K v        A strictly causal, SIGNED

THE DEFAULT OPERATOR IS THE ONE THAT REACHED PARITY, AND IT IS `sgate`.

    sgate    A = rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)
    signed   A = rho * w / sum_j |w_ij|
    smprime  A_ij = G_ij exp(qk q_i.k_j) / Z_i^beta,  G the PATH PRODUCT

A THIRD OPERATOR IS SELECTABLE AND CARRIES NO NUMBER OF ITS OWN. `smprime` is
`ceq/arm_smprime.py` -- the workhorse arm `CEQ_V16_CONTRACT.md` names -- wired
in as PLUMBING so it can be trained at all. Nothing below applies to it: the
parity ratio, the COSTS table, the capability number and the decay rates are all
`sgate` measurements. It is the only operator in this file that is IMPORTED
rather than duplicated, so a checkpoint that selects it needs the `ceq` package
present; `sgate` and `signed` are untouched and the Hub copy still loads without
it. `V17_ARM_WIRING.md` is that wiring's receipt, including the proof that the
default path is bitwise what it was.

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
distinguishes this operator from softmax DECAYS with context length, from
0.17480 at s=8 to 0.00391 at s=128. The exponent is WITHDRAWN as a `floor = 1e-6` artifact; no replacement exponent is published because -0.958 (R^2 0.9990) and -1.221 (R^2 0.9662) disagree.

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
import statistics

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
                                       #: The exponent is WITHDRAWN, and the
                                       #: repair is DELETION rather than a
                                       #: substitution: the two candidate
                                       #: fits disagree, so any number put
                                       #: here would be indefensible. The
                                       #: RATES above are measured and stand.
                                       "slope": None, "r2": None,
                                       "exponent_status": (
                                           "WITHDRAWN -- the -1.389 / R^2 0.9938 pair was a "
                                           "floor=1e-6 artifact. No replacement is published: "
                                           "-0.958 (R^2 0.9990) on floor=0 rates and -1.221 "
                                           "(R^2 0.9662) from the audit disagree. The decay "
                                           "itself is real -- 0.17480 -> 0.00391 over s=8..128."),
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
    """At `sgate` and `signed`, carries the SAME parameters a softmax block
    would: qkv and an output projection, nothing extra. The operator itself has
    no parameters, which is what makes a matched-parameter comparison against
    softmax possible at all.

    AT `smprime` IT CARRIES MORE, AND THE EXCESS IS NAMED. The arm's
    parametrization is two per-position scalar heads (`m_head`, `theta_head`)
    and three scalar switches (`beta`, `qk`, `g`), which is
    `ceq/arm_smprime.py::ArmSMPrime`'s own -- `2*(d+1) + 3` parameters per block
    and nothing else. `V16_ARM_SMPRIME.md` section 1 records the same excess
    against a softmax control as `4,806 - 4,769 = 37` at `d_model = 16`, which
    is `2*16 + 5`. So a matched-parameter comparison at `smprime` is matched to
    within a stated, counted quantity and NOT exactly, and
    `tests/gate0/test_g10_arm_wiring.py` asserts the difference is that quantity
    and nothing more.

    THE HEADS ARE SHARED ACROSS HEADS, `nn.Linear(d, 1)`, because that is the
    arm's shape: one magnitude and one phase per POSITION. A per-head widening
    to `nn.Linear(d, n_heads)` would be a construction the arm does not have,
    and the round's first law strikes it.
    """

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
        if self.operator == "smprime":
            self.m_head = nn.Linear(d, 1)
            self.theta_head = nn.Linear(d, 1)
            #: `nn.Parameter`s because they are `nn.Parameter`s on the arm --
            #: `#5a` requires the three switches to be settable and the arm
            #: makes them trainable. `config.smp_*` names where they START.
            self.beta = nn.Parameter(torch.tensor(float(config.smp_beta)))
            self.qk = nn.Parameter(torch.tensor(float(config.smp_qk)))
            self.g = nn.Parameter(torch.tensor(float(config.smp_g)))

    def _smprime(self, q, k, v, x):
        """`ceq/arm_smprime.py`'s read-out, CALLED rather than reimplemented.

        THE IMPORT IS LOCAL AND THAT IS THE WHOLE COST OF THIS SEAM. Every other
        operator in this file is duplicated from the package so the Hub's flat
        copy is self-contained; a third duplicate would be a SECOND
        IMPLEMENTATION of a module the contract fixes as the workhorse
        ("Workhorse arm_smprime only"), and a duplicate that drifts is the
        failure this file already carries two bitwise oracles against. So the
        arm is imported instead, INSIDE the branch:

          * `import ceq.arm_smprime` at module scope would make the checkpoint
            fail to LOAD on every machine without this repository -- the same
            argument the module docstring makes about `import triton`;
          * as written, the Hub copy still imports and still runs at `sgate`
            and `signed`, and `operator="smprime"` raises ImportError there.

        Making `smprime` Hub-portable requires duplicating the arm, which is a
        construction for the author to rule on rather than one to build here.

        NO PATH SUM. The arm's read-out is `O_i = sum_j W_ij V_j`, one
        application of the operator, so `path_sum` is not called and `hops` has
        no meaning at this operator -- `CEQConfig` refuses to carry one.

        CAUSAL INCLUSIVE OF THE DIAGONAL (`j <= i`), where `ceq_operator` and
        `sgate_operator` are STRICTLY causal (`j < i`) because they need `A`
        nilpotent for the path sum. The arm takes no path sum, its window is
        `Ico (j+1) (i+1)` -- empty at `j = i`, so `G_ii = 1` -- and position `i`
        reading its own value is ordinary causal attention, not leakage.
        """
        from ceq.arm_smprime import readout as smprime_readout

        #: `[B, S] -> [B, 1, S]`: the gate is per position, and the singleton
        #: broadcasts against the operator's `[B, H, S, S]` over heads. Without
        #: the singleton a `[B, S, S]` path product would align B against H.
        u = self.m_head(x).squeeze(-1).unsqueeze(-2)
        th = self.theta_head(x).squeeze(-1).unsqueeze(-2)
        #: `.real` is `ArmSMPrime.forward`'s own read-out of the complex output.
        return smprime_readout(q, k, v, u, th,
                               beta=self.beta, qk=self.qk, g=self.g).real

    def forward(self, x, attention_mask=None):
        b, s, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def shape(t):
            return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        if self.operator == "smprime":
            if attention_mask is not None:
                # RAISES RATHER THAN IGNORING. `ceq/arm_smprime.py::operator`
                # takes no mask argument, so a padding mask handed to this
                # branch would be dropped in silence -- the constraint reported
                # applied and not applied, which is the bug `_keep_mask` above
                # exists because this project shipped twice. Threading a mask
                # through the arm is a change to the arm, not to this seam.
                raise NotImplementedError(
                    "operator='smprime' cannot apply a padding mask: "
                    "ceq/arm_smprime.py's operator takes no mask argument and "
                    "silently dropping it would report a constraint that was "
                    "not applied. Causality is unaffected (the path product is "
                    "masked to j <= i unconditionally). Train without padding, "
                    "or add the mask to the arm -- which is the arm's change to "
                    "make, not this file's.")
            o = self._smprime(shape(q), shape(k), shape(v), x)
        else:
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


# ------------------------------------------------ the beta column (RULING 2)

def beta_column(model) -> dict:
    """`smprime`'s `beta`, ONE SCALAR PER LAYER, with its gradient and its flag.

    GRANULARITY, STATED RATHER THAN CHOSEN. `ceq/arm_smprime.py::ArmSMPrime`
    carries `beta` as one scalar `nn.Parameter` per arm instance, shared across
    attention heads, and `CEQAttention` above carries that same parametrization
    once per layer. So the column is `[n_layers]` and "per instance" is PER
    LAYER. A `[n_layers, n_heads]` beta would be a construction the arm does not
    have -- the same argument section 2 of `V17_ARM_WIRING.md` makes about the
    gate heads -- so it is not built here.

    THREE LISTS AND NOT ONE, because the value alone cannot answer RULING 2's
    question. `beta` frozen by a bug and `beta` moved-and-returned both read
    exactly `1.0` at the end of a run and mean opposite things. `grad` says
    whether a gradient ever reached the parameter and `requires_grad` separates
    the two ways it can fail to: `requires_grad=False` (frozen at the flag) from
    a detached call site (flag still True, `grad` still `None`).

    `grad` is `None`, not `0.0`, when no backward has run or the parameter is
    out of the graph -- those are not a zero gradient and recording them as one
    would erase the distinction this function exists for. `None` survives JSON
    as `null`.

    A PURE READ: no tensor is written, no RNG advanced, no graph built. Empty
    lists on `sgate` and `signed`, which have no `beta`.
    """
    named = _beta_parameters(model)
    return {"name": [n for n, _ in named],
            "beta": [float(b) for _, b in named],
            "grad": [None if b.grad is None else float(b.grad)
                     for _, b in named],
            "requires_grad": [bool(b.requires_grad) for _, b in named]}


def beta_census(model, acc: list | None = None) -> list:
    """Integrated `|dL/dbeta_i|` over training, one running total per `beta`.

    RULING 2a's GRADIENT CENSUS, and the reason it is a separate call from
    `beta_column`: the trajectory is sampled every `log_every`, the census is an
    INTEGRAL and has to see every step. Call it once per step, immediately after
    `loss.backward()` and BEFORE `clip_grad_norm_`, so the quantity is the
    gradient the ruling names and not the clipped one.

    A `beta` whose gradient never arrives contributes exactly `0.0`, which is the
    PINNED-WITHOUT-SIGNAL witness: `beta` sitting at 1 with a census of `0.0` is
    a dial that was never exercised, and the card's word is "unused". `None` is
    not summed as a zero by accident -- a missing gradient IS a zero
    contribution to the integral, and the column's `grad` field is where the
    `None` itself is recorded.

    Mutates and returns `acc`, in `ceq/hf/train.py::_attach_row_l1_probe`'s own
    idiom (a plain list the caller owns), so no buffer is added to the model and
    no checkpoint or parameter count moves.
    """
    named = _beta_parameters(model)
    if acc is None:
        acc = []
    if not acc:
        acc.extend([0.0] * len(named))
    for i, (_, b) in enumerate(named):
        if b.grad is not None:
            acc[i] += abs(float(b.grad))
    return acc


def _beta_parameters(model):
    """`[(name, parameter)]` for every `smprime` `beta`, in layer order.

    THE NAME IS THE PER-INSTANCE IDENTITY AND IT HAS NO HEAD AXIS. RULING 2a's
    branch C asks WHERE the moved betas live, "which layers / heads". The answer
    the arm can give is LAYERS: `ceq/arm_smprime.py` carries one scalar `beta`
    per arm instance, shared across attention heads, so there is no per-head
    beta to report and manufacturing one would be a construction the arm does
    not have. The name is emitted rather than left implicit in the list position
    so that a record consumer joins by name, and so that the absence of a head
    index is visible in the artifact instead of only in prose.
    """
    layers = getattr(getattr(model, "model", model), "layers", [])
    return [("model.layers.{}.self_attn.beta".format(i), b.self_attn.beta)
            for i, b in enumerate(layers) if hasattr(b.self_attn, "beta")]


def beta_summary(series, *, init: float = 1.0, census=None, delta_beta=None,
                 k: float = 5.0) -> dict:
    """Read a logged `beta_column` series back into the two quantities RULING 2
    turns on, plus the final distribution the card must print.

    `5*delta_beta` IS NO LONGER THE PIN CRITERION -- RULING 10' RETIRED IT AND
    THIS FUNCTION'S `pinned` / `branch` / `pinned_fraction` ARE NOW A DIAGNOSTIC.
    The criterion is `beta_lrt`'s likelihood ratio against `3.841` and `ln n`.
    Why the old one went: it was defined off an identical-seed training pair, and
    the re-take measured that pair BITWISE on the certified device, so
    `delta_beta = 0`, the tolerance collapsed to exactly zero, and every run read
    the "moved" branch for a reason about the optimizer rather than about
    training (`n_degenerate_floor` below counts exactly that). RULING 10' does
    not patch the floor: Wilks' randomness is over the DATA, so a training pair
    is irrelevant to the test. `delta_beta` survives as the diagnostic it always
    was, `scripts/k_noise_floor.py` keeps measuring it, and NOTHING here is
    deleted -- a diagnostic that CONTRADICTS the LRT verdict is a finding, and it
    cannot contradict anything if it stops being computed. `criterion` in the
    returned dict says all of this to a consumer that never reads a docstring.

    `mobile`      a gradient reached `beta` at every logged step. FALSE is a
                  broken experiment: the run could not have moved `beta` no
                  matter what the data said.
    `moved`       `beta`'s value left `init` at some logged step.

    THE FOUR CASES, AND WHY THE FINAL VALUE COLLAPSES THREE OF THEM:

      mobile  moved   state          reading
      False   False   immobile       beta COULD NOT move -- a bug, not a result
      True    False   not_updated    beta had a live gradient and was never
                                     stepped (out of the optimizer)
      True    True    moved          beta moved; `final_displacement` says
                                     whether it came back
      False   True    moved          moved by something other than its gradient

    All three of the first cases can end at exactly `init`, so `final` alone
    reads the same for a frozen run and for the ruling's "beta pins at 1".

    NO TOLERANCE IS INVENTED HERE and no verdict is issued. "Pins at 1" is the
    author's reading of `final` and `final_displacement`, which are printed;
    this function reports whether the run was capable of answering the question
    at all. `max_displacement` is over the LOGGED steps only, so it is a lower
    bound on the true excursion -- an excursion entirely between two log points
    is not seen. Log every step if the excursion itself is the object.
    """
    cols = [c for c in series if c["beta"]]
    if not cols:
        return {"n_layers": 0, "n_logged": 0, "final": [], "state": "absent",
                "branch": None}
    grads = [g for c in cols for g in c["grad"] if g is not None]
    n_missing = sum(1 for c in cols for g in c["grad"] if g is None)
    max_abs_grad = max((abs(g) for g in grads), default=0.0)
    max_disp = max(abs(b - init) for c in cols for b in c["beta"])
    final = cols[-1]["beta"]
    mobile = n_missing == 0 and max_abs_grad > 0.0
    moved = max_disp > 0.0
    out = {"name": cols[-1].get("name", [None] * len(final)),
           "n_layers": len(final), "n_logged": len(cols), "init": init,
           "final": final, "final_min": min(final),
           "final_median": statistics.median(final), "final_max": max(final),
           "displacement": [abs(b - init) for b in final],
           "final_displacement": max(abs(b - init) for b in final),
           "max_displacement": max_disp, "max_abs_grad": max_abs_grad,
           "n_missing_grad": n_missing,
           "requires_grad": all(r for c in cols for r in c["requires_grad"]),
           "mobile": mobile, "moved": moved,
           "state": "moved" if moved else
                    ("not_updated" if mobile else "immobile")}
    out.update(_pinning(final, init, census, delta_beta, k))
    if out.get("pinned") is not None:
        #: BRANCH C's whole content: which instances left the corner. By NAME,
        #: because a flattened distribution cannot answer "where".
        out["moved_names"] = [n for n, pin in zip(out["name"], out["pinned"])
                              if not pin]
    return out


def _pinning(final, init, census, delta_beta, k) -> dict:
    """THE `5*delta_beta` DIAGNOSTIC. **NOT the pin criterion any more.**

    RULING 10' STRUCK IT AS THE CRITERION and replaced it with `beta_lrt`'s
    likelihood ratio against `3.841` (chi-squared, 1 dof, 0.95) and `ln n`. What
    this function computes is retained, unchanged, as a DIAGNOSTIC: it is read
    for whether it CONTRADICTS the LRT verdict, never for the verdict itself.
    `scripts/k_noise_floor.py` keeps measuring `delta_beta` and that is correct.

    The retired rule, kept here because the numbers below still implement it:
    `beta_i` was PINNED iff `|beta_i,final - 1| <= k * delta_beta_i`,
    `delta_beta` the end-of-training spread of that same parameter across the
    RULING 1 identical-seed pair, `k = 5` a HEURISTIC SCALE and not a CI.

    THREE BRANCHES, not two: `>= 95 %` pinned is A, `<= 5 %` is B, anything else
    is C and the card prints WHERE the moved betas live. `name` carries that
    identity; there is no head axis to carry (see `_beta_parameters`).

    THE CENSUS SETS THE CARD'S WORD, and it is set to the WEAKER one by default.
    "Preferred" is a COMPARATIVE claim -- softmax was chosen over the
    alternative -- so it needs the alternative to have been priced: every pinned
    beta's integrated `|dL/dbeta|` at least the MEDIAN of the moved betas'. With
    no moved beta there is no comparison group at all and the word stays
    "unused", which is also branch A's own sentence, so this rule can never
    upgrade a card sentence and can only refuse to.

    THE DEGENERACY THAT KILLED IT, and it was measured rather than predicted. If
    the identical-seed pair is bitwise on its path then `delta_beta_i = 0`, the
    tolerance is `0`, and PINNED collapses to `beta_i` being EXACTLY 1.0 -- so
    every beta reads NOT pinned and the run lands in branch B for a reason that
    is about the pair's determinism and not about training. The re-take then
    measured exactly that on the certified device. `n_degenerate_floor` counts
    those entries and still does, because a diagnostic reading `B` at
    `n_degenerate_floor > 0` is saying something about the pair, not the model.
    """
    criterion = ("DIAGNOSTIC ONLY. RULING 10' retired 5*delta_beta as the pin "
                 "criterion; the criterion is beta_lrt's Lambda against 3.841 "
                 "(chi2_1 at 0.95) and ln n. delta_beta survives as the "
                 "diagnostic it always was (scripts/k_noise_floor.py).")
    if census is not None:
        signal = [c > 0.0 for c in census]
    if delta_beta is None:
        out = {"criterion": criterion, "pinned": None, "branch": None,
               "branch_reason": "delta_beta not supplied; the retired RULING 2a "
                                "diagnostic needs the identical-seed pair's "
                                "per-parameter spread "
                                "(scripts/k_noise_floor.py). The VERDICT does "
                                "not wait on it -- see beta_lrt (RULING 10')."}
        if census is not None:
            out.update({"census": list(census), "signal": signal})
        return out
    pinned = [abs(b - init) <= k * d for b, d in zip(final, delta_beta)]
    frac = sum(pinned) / len(pinned)
    out = {"criterion": criterion,
           "delta_beta": list(delta_beta), "k": k, "pinned": pinned,
           "pinned_fraction": frac,
           "n_degenerate_floor": sum(1 for d in delta_beta if d == 0.0),
           "branch": "A" if frac >= 0.95 else ("B" if frac <= 0.05 else "C")}
    if census is not None:
        movedc = [c for c, pin in zip(census, pinned) if not pin]
        pinnedc = [c for c, pin in zip(census, pinned) if pin]
        bar = statistics.median(movedc) if movedc else None
        out.update({"census": list(census), "signal": signal,
                    "census_median_pinned":
                        statistics.median(pinnedc) if pinnedc else None,
                    "census_median_moved": bar,
                    "word": "preferred" if (bar is not None and pinnedc
                                            and all(c >= bar for c in pinnedc))
                            else "unused"})
    return out


@torch.no_grad()
def beta_substitution(model, **forward_kwargs) -> dict:
    """Forward the model twice -- at its trained `beta` and with `beta` set to
    1.0 in every layer -- and return the two losses and `|delta|`.

    WHY THIS AND NOT A THRESHOLD ON `beta` ITSELF. RULING 2 branches on "beta
    pins at 1" and RULING 1 fixes the only scale this round has measured: the
    training noise floor, `|delta|` in FINAL LOSS over two identical-seed
    chunks. That floor is a loss-space quantity and `beta` is a dimensionless
    exponent, so it cannot be carried into `beta`'s units -- the conversion
    factor is `dL/dbeta`, which is small exactly where the question is asked and
    makes the tolerance diverge. This function moves the PARAMETER into LOSS
    SPACE instead, which is the direction that works: it asks what the card's
    sentence actually claims -- that substituting the softmax corner for the
    trained model changes nothing anyone in this round can measure.

    FORWARD-ONLY, so RULING 1's bitwise regime applies and `delta` is exact
    rather than floor-limited. `beta` is restored on every path.

    L-LEAN: this reports one difference of two losses on whatever batch the
    caller passes. It is not an evaluation, not a cell and not a verdict.
    """
    layers = getattr(getattr(model, "model", model), "layers", [])
    attn = [b.self_attn for b in layers if hasattr(b.self_attn, "beta")]
    if not attn:
        return {"beta": [], "delta": None}
    was = [a.beta.detach().clone() for a in attn]
    try:
        trained = float(model(**forward_kwargs).loss)
        for a in attn:
            a.beta.fill_(1.0)
        at_one = float(model(**forward_kwargs).loss)
    finally:
        for a, w in zip(attn, was):
            a.beta.copy_(w)
    return {"beta": [float(w) for w in was], "loss_trained": trained,
            "loss_at_beta_one": at_one, "delta": abs(trained - at_one)}


# ------------------------------ RULING 10': "PINNED" BY LIKELIHOOD RATIO

#: chi-squared at 1 dof, 0.95 -- Wilks'. RULING 10' NAMES this constant and this
#: module does not choose it. `3.841`, `ln n` and the `2` in `Lambda` were on the
#: shelf; the ruling forbids inventing a fourth constant, a `k`, or an
#: interpolation between the first two, and `_lrt_verdict` below has none.
CHI2_1_AT_95 = 3.841


def _lrt_verdict(lam: float, ln_n: float) -> str:
    """RULING 10's three-way table, and nothing between its two constants.

        Lambda <= 3.841          PINNED
        Lambda >  ln n           MOVED
        otherwise                the interval verdict, verbatim

    A NEGATIVE `Lambda` READS PINNED, and that is correct rather than a guard:
    `Lambda < 0` means `beta_final` fits the HELD-OUT split WORSE than
    `beta == 1`, which is no evidence at all against `beta == 1`. It happens
    whenever `beta_final` is not the eval split's own maximiser -- the normal
    case, since it was fit on the training split. Wilks assumes the MLE; off it
    the statistic is CONSERVATIVE, biased toward PINNED, so the direction of the
    violation is the direction that claims less.

    THE INTERVAL IS EMPTY FOR `n <= 46`, because `ln 46 < 3.841 < ln 47`. On a
    small eval split every rejection is therefore a MOVED. That is a property of
    the two shelf constants and not a rule chosen here; it is reported by
    printing `ln n` beside every verdict rather than patched.
    """
    if lam <= CHI2_1_AT_95:
        return "PINNED"
    if lam > ln_n:
        return "MOVED"
    return "rejected at 0.95, below description-length"


def beta_lrt(model, **forward_kwargs) -> dict:
    """RULING 10': `Lambda = 2*[LL_eval(beta_final) - LL_eval(beta == 1)]`.

    TWO DETERMINISTIC FORWARD PASSES ON THE HELD-OUT SPLIT. No training, no
    identical-seed pair, no floor. RULING 2a's `5*delta_beta` is retired as the
    criterion because it broke on measurement -- the re-take read the pair
    BITWISE on the certified device, so its tolerance collapsed to exactly zero
    and every run took the "moved" branch for a reason about the optimizer.
    RULING 10' does not patch that: Wilks' randomness is over the DATA, so a
    bitwise training pair is IRRELEVANT to this test. `delta_beta` survives as
    the diagnostic it always was, in `beta_summary`.

    `LL = -n * loss`, because `CEQForCausalLM.forward` returns a MEAN
    cross-entropy and `n` is the number of scored targets. `n` is READ OFF
    `labels` -- `(labels[:, 1:] != -100).sum()`, matching the forward's own
    shift and `F.cross_entropy`'s default `ignore_index` -- and never inferred
    from a shape, because `n` sets `ln n` AND the power line and a guessed `n`
    is an invented tolerance wearing arithmetic's hat.

    THE SECOND PASS SETS EVERY `beta` TO EXACTLY `1.0` AND RESTORES. The restore
    is in a `finally` and copies back a pre-call clone, so the model is
    bit-identical afterwards under `torch.equal`; `.grad` is not touched
    (`torch.autograd.grad` does not accumulate into it), which matters because
    `.grad` is `beta_census`'s input and the census is what RULING 10' leaves
    standing as the scientific payload. `model.training` is saved and restored
    around an `eval()`.

    THE HESSIAN-DIAGONAL METHOD, STATED. `I_beta` is the OBSERVED Fisher
    information, taken by exact double backward: one `torch.autograd.grad(loss,
    betas, create_graph=True)`, then one `torch.autograd.grad(g_i, beta_i)` per
    parameter. That is the `i`-th diagonal entry of the Hessian of the mean loss
    and no off-diagonal term is formed or needed. Since `LL = -n*loss`,

        I_beta_total   = -d2 LL / dbeta2  =  n * d2 loss / dbeta2
        I_beta_per_obs = I_beta_total / n =      d2 loss / dbeta2

    Both are emitted BY NAME because the ruling's two formulas use different
    ones: the Wald form `(beta-1)^2 * I_beta` needs the TOTAL, and the power line
    `sqrt(3.841 / (n * I_beta))` needs the PER-OBSERVATION. They are the same
    number divided by `n`, and confusing them is a factor of `n`.

    A NON-POSITIVE `I_beta` IS REPORTED, NEVER CLAMPED. Observed information is
    only guaranteed non-negative AT an MLE, and `beta_final` is not one; an
    untrained model at `beta == 1` measures it negative on this box. Where it is
    non-positive, `wald` and `beta_min_detectable` are `None` and `note` says
    why. An `abs()` there would manufacture a resolution out of a curvature
    pointing the other way.

    THIS CALL IS NOT EXECUTABLE UNDER STRICT DETERMINISM ON CUDA, AND THE SPLIT
    RUNS EXACTLY WHERE RULING 1 SAYS THE HOLE IS. The two `Lambda` passes are
    FORWARD-ONLY and are fine: measured on the certified 4060, the forward runs
    under `use_deterministic_algorithms(True)` and repeats BITWISE. The
    Hessian-diagonal read takes a BACKWARD (twice), autograd differentiates
    `cumprod` with `cumsum`, and `cumsum_cuda_kernel` has no deterministic
    implementation -- so under strict mode this function RAISES, and under the
    round's `warn_only=True` regime it warns and proceeds. Consequence, flagged
    rather than worked around: `Lambda` and its verdict sit inside RULING 1's
    B2 regime, but `wald` and `beta_min_detectable` do NOT, and RULING 10' calls
    a verdict printed without its resolution a defect -- so a `Lambda` cell
    cannot be a B2 strict cell. That is a question for the deciding-cell list
    RULING 6f freezes, not something this function decides by catching the
    error. `V17_R10P_LRT.md` section 6.

    DOF, WHICH THE RULING'S TWO SENTENCES DO NOT AGREE ON, so both are emitted.
    `Lambda` is written over `beta == 1` -- every one of them -- and called a
    1-dof comparison. With one `beta` per LAYER those coincide only at
    `n_layers == 1`. `lambda_joint` carries `dof_joint = n_beta`; the
    `per_parameter` rows each hold ONE `beta` out at a time and are the 1-dof
    statistics `3.841` actually licenses, which is also the per-parameter
    reporting RULING 10' leaves standing. Nothing here picks a constant for
    `dof > 1`; the dof is printed instead.
    """
    named = _beta_parameters(model)
    labels = forward_kwargs.get("labels")
    if labels is None:
        raise ValueError(
            "beta_lrt needs `labels`: n is the eval count, and it sets both "
            "`ln n` and the power line. Inferring it from a shape would be an "
            "invented tolerance (RULING 10').")
    n = int((labels[:, 1:] != -100).sum())
    ln_n = math.log(n) if n > 0 else float("-inf")
    head = {"n_eval": n, "ln_n": ln_n, "chi2_1_at_95": CHI2_1_AT_95,
            "n_beta": len(named), "dof_joint": len(named)}
    if not named:
        return dict(head, ll_final=None, ll_beta_one=None, lambda_joint=None,
                    verdict_joint=None, per_parameter=[])

    betas = [b for _, b in named]
    was = [b.detach().clone() for b in betas]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            ll_final = -n * float(model(**forward_kwargs).loss)
            for b in betas:
                b.fill_(1.0)
            ll_joint = -n * float(model(**forward_kwargs).loss)
            ll_one = []
            for i in range(len(betas)):
                #: ONE beta out at a time: the genuinely 1-dof comparison.
                for j, (bj, wj) in enumerate(zip(betas, was)):
                    if j == i:
                        bj.fill_(1.0)
                    else:
                        bj.copy_(wj)
                ll_one.append(-n * float(model(**forward_kwargs).loss))
            for b, w in zip(betas, was):
                b.copy_(w)
        #: the Hessian-diagonal read, exact double backward
        loss = model(**forward_kwargs).loss
        g1 = torch.autograd.grad(loss, betas, create_graph=True)
        hess = [float(torch.autograd.grad(g1[i], b, retain_graph=True)[0])
                for i, b in enumerate(betas)]
    finally:
        with torch.no_grad():
            for b, w in zip(betas, was):
                b.copy_(w)
        model.train(was_training)

    rows = []
    for i, (name, _) in enumerate(named):
        lam = 2.0 * (ll_final - ll_one[i])
        i_total = n * hess[i]
        d = float(was[i]) - 1.0
        row = {"name": name, "beta_final": float(was[i]), "lambda": lam,
               "verdict": _lrt_verdict(lam, ln_n),
               "lrt_rejects_at_95": lam > CHI2_1_AT_95,
               "I_beta_per_obs": hess[i], "I_beta_total": i_total,
               "wald": None, "wald_rejects_at_95": None, "wald_agrees": None,
               "wald_rel_gap": None, "beta_min_detectable": None, "note": None}
        if i_total > 0.0:
            wald = d * d * i_total
            scale = max(abs(wald), abs(lam), 1e-300)
            row.update(wald=wald, wald_rejects_at_95=wald > CHI2_1_AT_95,
                       wald_agrees=(wald > CHI2_1_AT_95) == (lam > CHI2_1_AT_95),
                       wald_rel_gap=abs(wald - lam) / scale,
                       beta_min_detectable=math.sqrt(CHI2_1_AT_95 / i_total))
        else:
            row["note"] = (
                "observed information is {:.6g} <= 0, so the Wald form and the "
                "power line do not exist here: beta_final is not a maximiser of "
                "the eval log-likelihood along beta. Reported, not clamped."
                .format(i_total))
        rows.append(row)

    lam_joint = 2.0 * (ll_final - ll_joint)
    return dict(head, ll_final=ll_final, ll_beta_one=ll_joint,
                lambda_joint=lam_joint,
                verdict_joint=_lrt_verdict(lam_joint, ln_n),
                per_parameter=rows)


def lrt_report(res: dict) -> str:
    """RULING 10' printed, with BOTH constants and the resolution on every line.

    "Print both constants alongside the verdict always", and "a verdict printed
    without its minimum detectable departure is a defect" -- so the FORMATTER,
    not the caller's discipline, is what makes the defect unreachable. Where the
    resolution does not exist the field is still printed, as `undefined`, with
    the reason on its own line.
    """
    n, ln_n = res["n_eval"], res["ln_n"]
    both = "3.841 / ln n = {:.4f}".format(ln_n)
    out = ["RULING 10' LRT -- n_eval = {}, chi2_1(0.95) = {}, ln n = {:.4f}, "
           "n_beta = {}".format(n, CHI2_1_AT_95, ln_n, res["n_beta"])]
    if not res["per_parameter"]:
        out.append("  no beta on this operator: no verdict "
                   "(absence is not 'pinned')")
        return "\n".join(out)
    out.append("  JOINT  Lambda = {:+.4f}  [{}]  {}   dof = {}{}".format(
        res["lambda_joint"], both, res["verdict_joint"], res["dof_joint"],
        "" if res["dof_joint"] == 1 else
        "  <- 3.841 is the 1-dof constant; the per-parameter rows below are the "
        "1-dof statistics"))
    for r in res["per_parameter"]:
        resolution = ("undefined" if r["beta_min_detectable"] is None
                      else "{:.6f}".format(r["beta_min_detectable"]))
        wald = ("undefined" if r["wald"] is None
                else "{:+.4f} ({})".format(
                    r["wald"], "agrees" if r["wald_agrees"] else "DISAGREES"))
        out.append(
            "  {}  beta={:.6f}  Lambda={:+.4f}  [{}]  {}  |beta-1|_min={}  "
            "Wald={}  I_beta_total={:.6g}".format(
                r["name"], r["beta_final"], r["lambda"], both, r["verdict"],
                resolution, wald, r["I_beta_total"]))
        if r["note"]:
            out.append("      note: {}".format(r["note"]))
    out.append("  PINNED means indistinguishable AT THIS RESOLUTION, never "
               "exact: |beta-1|_min = sqrt(3.841 / (n * I_beta)).")
    return "\n".join(out)


CEQForCausalLM.register_for_auto_class("AutoModelForCausalLM")
CEQModel.register_for_auto_class("AutoModel")
