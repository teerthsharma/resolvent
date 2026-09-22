"""Phase J Addendum L, row FQ-ARM: arm (f_Q), the SU(2) leap.

Pattern copied from arms_j.py (a''/FoX wiring): (a)'s exact SDPA softmax
twin plus one extra piece, added purely at the score/value level so the
unchanged SDPA call is still the attention primitive.

(f_Q): per layer, one nn.Linear(d, 4) "quaternion head" (head-shared: the
same Linear reads the full hidden state, one quaternion per token, applied
identically to every attention head), normalised per token to a unit
quaternion. A log-depth prefix scan Pi over the sequence gives the
cumulative rotation Pi_i = q_i * q_{i-1} * ... * q_0 (su2.py's
prefix_scan, imported unedited -- it already exists in this directory, so
no private copy is written here). Each 4-block of V (head_dim split into
groups of 4) is left-multiplied by conj(Pi_j) before the unchanged SDPA
call; each 4-block of the SDPA output is left-multiplied by Pi_i after.
head_dim must be divisible by 4.

Extra params per layer: nn.Linear(d, 4) -> d*4 + 4 (weight + bias).

This file is new; arms_j.py is not edited.
"""
from __future__ import annotations

import os
import sys

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
PHASE_J = os.path.join(SCRATCH, "phase_j")
sys.path.insert(0, SCRATCH)
sys.path.insert(0, PHASE_J)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from su2 import qmul, qconj, qnormalize, prefix_scan  # noqa: E402 -- imported, not reimplemented

SU2_SOURCE = "su2.py (shared scratchpad/phase_j, pre-existing) -- imported unedited"


def _attach_quat_heads(model, hidden):
    """arms_j pattern: attach the extra per-layer module without touching
    the layer class itself."""
    for layer in model.model.layers:
        layer.self_attn.quat_head = nn.Linear(hidden, 4)
    return model


def quat_head_numel(model):
    """Extra params per layer (all layers carry the identical shape)."""
    per_layer = [sum(p.numel() for p in layer.self_attn.quat_head.parameters())
                 for layer in model.model.layers]
    return per_layer[0] if per_layer else 0, sum(per_layer)


def fq_forward(self, x, attention_mask=None):
    """(a)'s SDPA softmax twin plus the SU(2) leap. attn weights p_ij are
    exactly softmax(QK^T/sqrt(d)) with is_causal=True, unchanged; only V
    (pre-SDPA) and the SDPA output (post-SDPA) are quaternion-rotated."""
    if attention_mask is not None:
        raise NotImplementedError("f_Q twin: no padding-mask path needed")
    b, s, d = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    assert self.d_head % 4 == 0, "head_dim must be divisible by 4 for f_Q"
    n_blocks = self.d_head // 4

    def shape(t):
        return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,S,Dh]

    Q, K, V = shape(q), shape(k), shape(v)

    quat = qnormalize(self.quat_head(x))          # [B,S,4], unit quaternion per token
    Pi = prefix_scan(quat)                          # [B,S,4], Pi_i = q_i*...*q_0
    Pi_conj = qconj(Pi)                              # [B,S,4]

    def to_blocks(t):
        return t.reshape(b, self.n_heads, s, n_blocks, 4)

    V4 = to_blocks(V)
    Pi_conj_b = Pi_conj.view(b, 1, s, 1, 4).expand(b, self.n_heads, s, n_blocks, 4)
    Vr = qmul(Pi_conj_b, V4).reshape(b, self.n_heads, s, self.d_head)

    O = F.scaled_dot_product_attention(Q, K, Vr, is_causal=True)

    O4 = to_blocks(O)
    Pi_b = Pi.view(b, 1, s, 1, 4).expand(b, self.n_heads, s, n_blocks, 4)
    Or = qmul(Pi_b, O4).reshape(b, self.n_heads, s, self.d_head)

    return self.o_proj(Or.transpose(1, 2).reshape(b, s, d))


# --------------------------------------------------------------- RED-first test
def _direct_sum_reference(x_layer_input, attn_module, quat_head, n_heads, d_head):
    """Test oracle: attention output equals the direct sum over j of
    p_ij * Pi_i * conj(Pi_j) * v_j, computed with independent python loops
    (no SDPA, no vectorised qmul reuse) so it does not share a bug with
    fq_forward."""
    b, s, d = x_layer_input.shape
    assert b == 1
    q, k, v = attn_module.qkv(x_layer_input).chunk(3, dim=-1)
    q = q.view(s, n_heads, d_head)
    k = k.view(s, n_heads, d_head)
    v = v.view(s, n_heads, d_head)
    n_blocks = d_head // 4

    quat = qnormalize(quat_head(x_layer_input)).view(s, 4)
    Pi = [quat[0].clone()]
    for i in range(1, s):
        Pi.append(qmul(quat[i], Pi[-1]))
    Pi = torch.stack(Pi, dim=0)          # [S,4]
    Pi_conj = qconj(Pi)                   # [S,4]

    out = torch.zeros(s, n_heads, d_head, dtype=x_layer_input.dtype)
    for h in range(n_heads):
        scores = (q[:, h, :] @ k[:, h, :].T) / (d_head ** 0.5)
        causal = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        p = torch.softmax(scores, dim=-1)   # [S,S]
        for i in range(s):
            for blk in range(n_blocks):
                acc = torch.zeros(4, dtype=x_layer_input.dtype)
                for j in range(s):
                    if p[i, j] == 0:
                        continue
                    vj = v[j, h, blk * 4:(blk + 1) * 4]
                    rotated = qmul(Pi[i], qmul(Pi_conj[j], vj))
                    acc = acc + p[i, j] * rotated
                out[i, h, blk * 4:(blk + 1) * 4] = acc
    return attn_module.o_proj(out.reshape(1, s, n_heads * d_head))


class _StubAttn(nn.Module):
    """Minimal SDPA-softmax-twin attention module with the shapes fq_forward
    expects: qkv, o_proj, d_head, n_heads. Stands in for CEQAttention so the
    test does not depend on the rest of the repo."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o_proj = nn.Linear(d, d)


def _red_stub_forward(self, x, attention_mask=None):
    raise NotImplementedError("f_Q not yet wired -- RED stub")


def test_fq_attention_matches_direct_sum():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    d, n_heads = 8, 2   # d_head=4, one quaternion block per head, divisible by 4
    s = 8
    attn = _StubAttn(d, n_heads)
    quat_head = nn.Linear(d, 4)
    attn.quat_head = quat_head
    x = torch.randn(1, s, d)

    got = fq_forward(attn, x)
    want = _direct_sum_reference(x, attn, quat_head, n_heads, attn.d_head)
    max_err = (got - want).abs().max().item()
    assert max_err <= 1e-10, f"max_err={max_err} > 1e-10"
    return max_err


if __name__ == "__main__":
    # RED first: run with the stub to record the RED line, then GREEN with
    # the real implementation.
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    d, n_heads, s = 8, 2, 8
    attn = _StubAttn(d, n_heads)
    attn.quat_head = nn.Linear(d, 4)
    x = torch.randn(1, s, d)
    try:
        _red_stub_forward(attn, x)
        print("RED-first check FAILED: stub did not raise")
    except NotImplementedError as e:
        print(f"RED line: {type(e).__name__}: {e}")

    err = test_fq_attention_matches_direct_sum()
    print(f"GREEN: fq_forward matches direct-sum reference, max_err={err:.3e} (<=1e-10)")

    extra_per_layer = d * 4 + 4  # nn.Linear(d,4): weight d*4 + bias 4
    print(f"extra numel per layer (Linear(d,4), d={d}): {extra_per_layer}")
    print(f"su2 source: {SU2_SOURCE}")
