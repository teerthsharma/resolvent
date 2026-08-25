"""The residual gate: stock attention plus a signed multi-hop correction.

    out = stock_attention(q, k, v) + sum_{h=1..K} (alpha * A)^h v

At `alpha = 0` this is BITWISE stock attention. That is the adoptability bar and
it is not a convenience: a pretrained checkpoint's weights were fit for softmax
semantics, so swapping the operator outright destroys the model and any
perplexity measured afterwards is measuring the swap, not the operator.

With the gate, `alpha` starts at zero, the checkpoint keeps its exact behaviour,
and the cost of the signed term on untuned weights becomes a measurable curve
rather than a confound.

WHAT THE CORRECTION TERM IS. `A` is the strictly causal signed operator from
`ceq.attention`: diagonal excluded, rows L1-bounded by rho, entries free to be
negative. Scaling it by `alpha` scales the bound with it, so the L1 bound becomes
`alpha * rho` and every guarantee tightens rather than weakens as the gate opens
slowly.

The identity term is deliberately absent from the correction. Stock attention
already supplies the k=0 term of the path sum -- it is the value each position
reads for itself -- so including it again would double-count.

WHY THE SIGN MATTERS HERE. The minimum entry of the influence Jacobian of the
Kleene star of a non-negative matrix is exactly 0.000e+00 in any ordered
semiring. Softmax attention, APPNP and the max-plus star are all inside that
class, so none of them can let one token REDUCE another's contribution. The
correction term measured -9.000e-01 against a non-negative control at exactly
zero. That is the only thing this gate adds, and it is the only thing worth
adding.

GROUPED-QUERY ATTENTION. Qwen2.5 and most current checkpoints use fewer key/value
heads than query heads. `_repeat_kv` expands them explicitly rather than relying
on broadcasting, which would silently pair the wrong head with the wrong query.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .attention import ceq_operator

NAME = "ceq_hybrid"
DEFAULT_RHO = 0.9
DEFAULT_HOPS = 3


def _repeat_kv(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Expand kv heads to match query heads for grouped-query attention."""
    b, h, s, d = t.shape
    if h == n_heads:
        return t
    rep = n_heads // h
    return t[:, :, None].expand(b, h, rep, s, d).reshape(b, h * rep, s, d)


def _to_interface_layout(x: torch.Tensor) -> torch.Tensor:
    """`[B, H, S, D]` -> `[B, S, H, D]`, the layout AttentionInterface requires.

    `sdpa_attention_forward` ends with `attn_output.transpose(1, 2).contiguous()`
    because the caller then does `.reshape(*input_shape, -1)`. Returning the
    un-transposed tensor folds heads and sequence into each other and the model
    still runs: measured perplexity 89400.180 against 1.667 for the untouched
    checkpoint, with no error raised anywhere.

    The first version of this file omitted this and its parity test still passed,
    because that test compared the gated path against `stock_attention` -- both
    wrong in the same way. Internally consistent, externally garbage. Only parity
    against the real checkpoint caught it.
    """
    return x.transpose(1, 2).contiguous()


def stock_attention(query, key, value, attention_mask, scaling: float | None = None):
    """Plain causal SDPA. The thing `alpha = 0` must reproduce bitwise.

    `is_causal=True` is only correct when `q_len == kv_len`. During incremental
    decoding the query is one token against a cached `kv_len = N`, and torch
    aligns the causal mask to the TOP-LEFT of a 1xN grid -- so the single query
    attends to position 0 alone and nothing else.

    That bug is invisible to perplexity, which is teacher-forced and never
    decodes: measured perplexity matched stock to 0.0000% at alpha = 0 while
    greedy generation produced ' The following the 1: 1: 1: 1:' against stock's
    ' The capital of France is Paris.'. A generation check is not optional.
    """
    k = _repeat_kv(key, query.shape[1])
    v = _repeat_kv(value, query.shape[1])
    if attention_mask is not None:
        return F.scaled_dot_product_attention(query, k, v, attn_mask=attention_mask,
                                              scale=scaling)

    q_len, kv_len = query.shape[-2], k.shape[-2]
    if 1 < q_len < kv_len:
        # THE HALF OF THIS BUG THE FIRST FIX MISSED. `is_causal = (q_len ==
        # kv_len)` is right for a 1-token decode step, where "attend to
        # everything cached" IS the causal answer. It is WRONG for a chunk: at
        # q_len=4 against kv_len=16 it passes is_causal=False and every query
        # reads every key, including the ones after it. Measured behaviourally
        # rather than by inspecting a mask -- perturbing key 15 moved query 0's
        # output by 1.8396726846694946, where causality requires exactly 0.
        # Nothing raised, and teacher-forced perplexity cannot see it because it
        # never chunks. Refused rather than guessed at, because the caller is the
        # only one who knows the absolute positions of this chunk.
        raise NotImplementedError(
            "chunked prefill is refused: q_len={} against kv_len={} with no "
            "attention_mask. `is_causal` cannot express 'these {} queries are at "
            "absolute positions {}..{}', so torch would align the causal mask to "
            "the top-left and this call would silently attend to FUTURE keys "
            "(measured: query 0 moved by 1.84 when key {} was perturbed). Pass "
            "an explicit additive mask carrying the absolute positions, or call "
            "one token at a time."
            .format(q_len, kv_len, q_len, kv_len - q_len, kv_len - 1, kv_len - 1))

    return F.scaled_dot_product_attention(query, k, v, is_causal=q_len == kv_len,
                                          scale=scaling)


def ceq_hybrid_attention(module, query: torch.Tensor, key: torch.Tensor,
                         value: torch.Tensor, attention_mask=None, *,
                         alpha: float = 0.0, rho: float = DEFAULT_RHO,
                         hops: int = DEFAULT_HOPS, scaling: float | None = None,
                         **kwargs):
    """`transformers` AttentionInterface entry point for the gated form.

    THE STOCK HALF IS DELEGATED, NOT REIMPLEMENTED, and that is a correction.
    "alpha = 0 is bitwise stock" was asserted against `stock_attention` below --
    this file's own SDPA -- rather than against the attention the model actually
    runs, which is the same mistake as the output-layout bug: internally
    consistent, externally wrong. Measured against the real path on a Llama:
    0.000e+00 at 4 kv heads and **1.490e-07 max abs / 2.609e-07 relative at 2 kv
    heads**. The cause is grouped-query attention -- `_repeat_kv` materializes
    the expanded keys while transformers passes `enable_gqa=True` to torch, and
    the two reduce in different orders (4.768e-07 apart on bare tensors).
    Qwen2.5-0.5B, where the 9.2149-vs-9.2150 perplexity was measured, is a GQA
    model, so the recorded claim was about the wrong comparison.

    Calling `sdpa_attention_forward` makes alpha = 0 bitwise BY CONSTRUCTION
    rather than by measurement, at every head grouping and every transformers
    version. `stock_attention` survives for the callers that have no `module`.
    """
    if module is not None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        stock, _ = sdpa_attention_forward(module, query, key, value,
                                          attention_mask, scaling=scaling,
                                          **kwargs)
        if alpha == 0.0 or hops == 0:
            return stock, None
        out = stock.transpose(1, 2)          # back to [B, H, S, D] for the sum
    else:
        out = stock_attention(query, key, value, attention_mask, scaling)
        if alpha == 0.0 or hops == 0:
            return _to_interface_layout(out), None

    k = _repeat_kv(key, query.shape[1])
    v = _repeat_kv(value, query.shape[1])

    if query.shape[-2] != k.shape[-2]:
        raise NotImplementedError(
            "multi-hop decode is not implemented. During incremental decoding the "
            "operator row is [1, N] and A^2 is undefined, so the path sum cannot be "
            "formed from a standard KV cache. Falling back to hops=1 here would make "
            "the function behave differently in prefill and decode -- a silent "
            "inconsistency -- so it is refused instead. The fix is a K-slot hop "
            "cache: (A^h v)_i = sum_{j<i} A_ij (A^{h-1} v)_j, so caching the h-th "
            "hop vector per position makes decode exact at K times the value-cache "
            "memory and K attention rows per token. Not built yet.")

    a = alpha * ceq_operator(query, k, attention_mask, rho=rho, scaling=scaling)

    term = v
    corr = torch.zeros_like(v)
    for _ in range(hops):
        term = a @ term
        corr = corr + term
    return _to_interface_layout(out + corr), None


def ceq_mask(*args, **kwargs):
    """Registered alongside the attention function.

    Without this, transformers passes `attention_mask=None` and causal, padding,
    packing and sliding-window constraints are silently dropped.
    """
    from transformers.masking_utils import sdpa_mask
    return sdpa_mask(*args, **kwargs)


def register() -> None:
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register(NAME, ceq_hybrid_attention)
    AttentionMaskInterface.register(NAME, ceq_mask)


# ------------------------------------------------------------------ evaluation

def _install(model, alpha: float | None, rho: float, hops: int):
    """Swap every attention module's implementation.

    `alpha=None` restores the checkpoint's own implementation, which is the
    control every gated number is compared against.
    """
    import functools
    if alpha is None:
        model.set_attn_implementation("sdpa")
        return
    register()
    fn = functools.partial(ceq_hybrid_attention, alpha=alpha, rho=rho, hops=hops)
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register(NAME, fn)
    model.set_attn_implementation(NAME)


def run_with(model, inputs, alpha: float | None = 0.0, rho: float = DEFAULT_RHO,
             hops: int = DEFAULT_HOPS):
    _install(model, alpha, rho, hops)
    return model(**inputs)


@torch.no_grad()
def perplexity(model, batch, alpha: float | None = 0.0, rho: float = DEFAULT_RHO,
               hops: int = DEFAULT_HOPS) -> float:
    """Teacher-forced perplexity on one batch.

    A SANITY number, not a win condition. C3 stands: intercept wins invert at
    scale, so a perplexity delta on untuned weights says whether the gate is
    wired correctly and nothing about whether the operator is worth having.
    """
    _install(model, alpha, rho, hops)
    ids = batch["input_ids"]
    out = model(**batch, labels=ids)
    return float(torch.exp(out.loss))
