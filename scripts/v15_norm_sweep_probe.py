"""V15 normalization-boundary sweep -- runnable float64 checks.

MARS-register mechanism sweep, CEQ v15. The Lean node (`lean/CEQ/V15.lean`,
`V15_N3_LEAN.md`) refuted the contract's claim that S-M's `g == 0` hop is
"bitwise standard attention": the hop is UNNORMALIZED (`W_ij = exp(C_i - C_j)`,
no softmax denominator), so at `g == 0` its row `i` sums to `i + 1` while every
softmax row sums to `1`. This script re-derives that arithmetic at small size
in plain float64/torch, independent of the Lean kernel, and checks four more
identity claims found by the same grep sweep of `ceq/*.py`, `scale/*.py`, and
the claim documents, for the same defect: an identity asserted between a
normalized (row-stochastic) object and an unnormalized one, where a matching
MASK / support pattern was allowed to stand in for a matching OPERATOR.

READ-ONLY. Imports `ceq.bench` and `ceq.hybrid` to call shipped functions
exactly as the harness calls them; nothing in this repo is modified, nothing
trains. `torch.set_num_threads(1)` for reproducibility across runs (this repo's
own convention, `scale/m3_capability.py`).

Run: python scripts/v15_norm_sweep_probe.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import bench, hybrid  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(0)
DT = torch.float64


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def row_sums(a: torch.Tensor) -> torch.Tensor:
    return a.sum(-1)


def flat_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation over the flattened lower-triangular (causal) entries
    only -- comparing over the full square would correlate two matrices of
    structural zeros above the diagonal, which is a different, uninteresting
    fact."""
    s = a.shape[-1]
    m = torch.ones(s, s, dtype=torch.bool).tril(0)
    x = a[..., m].reshape(-1).double()
    y = b[..., m].reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((x @ y) / denom)


# ---------------------------------------------------------------------------
# CHECK 1 -- "softmax and pivot_unsigned share a BIT-IDENTICAL operator"
#   workdonenew.md:146, 188, 282-284; README.md; CEQ_V15_CONTRACT.md "WHERE WE
#   ARE". Claimed producer: scale/m3_capability.py:131 --
#     if self.kind == "softmax" or self.kind == "pivot_unsigned":
#         return bench._softmax_operator(q, k)
#   Both arm kinds dispatch to the SAME function call on the SAME inputs. This
#   is not a numerical coincidence to be measured -- it is one function called
#   twice -- but the claim is checked here exactly the way the contract states
#   it: build the operator each arm actually ships and diff it.
# ---------------------------------------------------------------------------
hr("CHECK 1 -- softmax vs pivot_unsigned (m3_capability.py:131 dispatch)")

n, s, d = 2, 6, 4
q = torch.randn(n, s, d, dtype=DT)
k = torch.randn(n, s, d, dtype=DT)

# Arm._operator's own branch, reproduced verbatim for both arm kinds:
a_softmax = bench._softmax_operator(q, k)          # kind == "softmax"
a_pivot_unsigned = bench._softmax_operator(q, k)    # kind == "pivot_unsigned"

max_abs_diff = float((a_softmax - a_pivot_unsigned).abs().max())
rs_softmax = row_sums(a_softmax)
rs_pivot = row_sums(a_pivot_unsigned)
corr = flat_corr(a_softmax, a_pivot_unsigned)

print(f"max|softmax - pivot_unsigned|      = {max_abs_diff:.6e}")
print(f"softmax row sums   (batch 0, all i) = {rs_softmax[0].tolist()}")
print(f"pivot_unsigned row sums (batch 0)   = {rs_pivot[0].tolist()}")
# ceq/bench.py's mask is STRICTLY causal (j < i, self-excluded: `.tril(-1)`),
# a deliberate design choice documented at arm_s.py's g3_bind ("Row 0 of a
# strictly causal operator sums to exactly 0.0 ... which has no
# predecessors"). Row 0 has no visible key at all and its softmax is defined
# to 0.0 by the operator's own final masked_fill; rows i>=1 see i keys and are
# genuinely row-stochastic. Checked accordingly, not glossed over.
live = rs_softmax[0][1:]
print(f"row-sum max|.-1| over LIVE rows i=1..{s-1} = "
      f"{float((live - 1.0).abs().max()):.3e} (row 0 is the documented dead row, sum 0.0)")
print(f"correlation (causal entries)        = {corr:.6f}")
print("VERDICT: SOUND -- the two arms call the identical shipped function on")
print("identical inputs (ceq/bench.py:_softmax_operator, dispatched at")
print("scale/m3_capability.py:131). max|diff| = 0.0 exactly, and every live")
print("row (i>=1) is genuinely row-stochastic. No normalization-boundary")
print("crossing: both objects are on the softmax side, because they are the")
print("same object.")
assert max_abs_diff == 0.0
assert float((live - 1.0).abs().max()) < 1e-12
assert float(rs_softmax[0][0]) == 0.0          # the documented dead row


# ---------------------------------------------------------------------------
# CHECK 2 -- the refuted claim itself: S-M's g==0 hop "IS bitwise standard
#   attention" (CEQ_V15_CONTRACT.md PART I S-C / "PARITY WITH SELF-ATTENTION").
#   S-M defines Wc(g)_ij = exp(C_i - C_j) for j <= i, 0 above the diagonal,
#   C = prefix-sum(g). At g == 0 every C_i == 0, so Wc_ij = 1 for all j <= i:
#   the UNNORMALIZED causal all-ones mask -- not a softmax row.
#   This is `lean/CEQ/V15.lean`'s `gate_zero_row_sum` / `gate_zero_not_stochastic`
#   re-derived here in plain torch, independent of the Lean kernel.
# ---------------------------------------------------------------------------
hr("CHECK 2 -- S-M carrier at g==0 (Wc) vs standard softmax attention")


def Wc_gate_zero(sz: int) -> torch.Tensor:
    """S-M's hop, `W_ij = exp(C_i - C_j)`, `C = scan(g)`, at `g == 0`
    identically. `C_i - C_j == 0` for every i, j, so `exp(.) == 1`; the causal
    mask (`Wc`, section 'The hop as it is actually applied') zeroes j > i.
    INCLUSIVE diagonal (`j <= i`), exactly as `lean/CEQ/V15.lean`'s `Wc`
    states it -- not `ceq/bench.py`'s self-excluding `j < i` convention
    (Check 1), which is a separate, documented design choice and would be a
    confound here."""
    return torch.ones(sz, sz, dtype=DT).tril(0)


def standard_causal_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Textbook causal self-attention: query i attends to keys j <= i
    (INCLUSIVE diagonal), softmax-normalized. This is the object "standard
    attention" denotes in the contract's PARITY clause and in
    `lean/CEQ/V15.lean`'s own reading -- built directly rather than reused
    from `ceq/bench.py` so the self-exclusion convention checked in Check 1
    cannot leak into this comparison."""
    sz = logits.shape[-1]
    causal = torch.ones(sz, sz, dtype=torch.bool).tril(0)
    return torch.softmax(logits.masked_fill(~causal, torch.finfo(DT).min), -1)


a_Wc = Wc_gate_zero(s)
a_sm = standard_causal_softmax((q[0] @ k[0].transpose(-2, -1)) / math.sqrt(d))

rs_Wc = row_sums(a_Wc)
rs_sm = row_sums(a_sm)
max_diff_2 = float((a_Wc - a_sm).abs().max())
corr_2 = flat_corr(a_Wc, a_sm)

print(f"Wc(g==0) row sums (i=0..{s-1})  = {rs_Wc.tolist()}")
print(f"expected i+1                    = {[float(i + 1) for i in range(s)]}")
print(f"softmax row sums (i=0..{s-1})   = {rs_sm.tolist()}")
print(f"max|Wc(g==0) - softmax(q,k)|    = {max_diff_2:.6e}")
print(f"correlation (causal entries)    = {corr_2} "
      f"(nan: Wc's rows are literally constant, zero variance -- see Check 3)")
print("VERDICT: CONFIRMED INSTANCE -- Wc(g==0) is the unnormalized causal")
print("all-ones mask (row i sums to i+1); softmax(q,k) is row-stochastic (row")
print("sums 1.0). They share the identical SUPPORT (both are exactly the")
print("lower-triangular pattern j<=i) and differ in every live entry. The")
print("smallest witness (i=1, matching lean/CEQ/V15.lean's own smallest")
print("witness) is a row of (1,1) summing to 2, against any softmax row")
print("summing to 1.")
assert torch.allclose(rs_Wc, torch.arange(1, s + 1, dtype=DT))
assert torch.allclose(rs_sm, torch.ones(s, dtype=DT), atol=1e-12)
assert max_diff_2 > 0.05          # the two operators are not close


# ---------------------------------------------------------------------------
# CHECK 3 -- pattern #3, scale-invariant statistic hiding a scale difference.
#   `Wc(g==0)`'s rows are literally CONSTANT (all 1.0 on the visible support),
#   so a Pearson correlation between a Wc row and its own row-normalization is
#   the degenerate 0/0 case (both checks above print `nan` for exactly this
#   reason) -- an even sharper failure than "reads 1.000": the statistic has
#   no value to report at all, and a careless nan-safe wrapper (`nan_to_num`,
#   a try/except defaulting to 1.0) could silently launder that into an
#   apparent perfect match. To give the brief's literal claim ("reads ~1.000
#   on objects differing by i+1 per row") a well-defined number, this check
#   uses a non-degenerate unnormalized/normalized pair with the SAME
#   mechanism: raw softmax numerators `exp(logit)` (unnormalized, row sum
#   Z_i != 1) against the softmax row itself (`exp(logit)/Z_i`) -- exactly
#   the S-M shape (multiplicative hop vs. its row-normalized counterpart),
#   with real entrywise variation so the statistic is defined.
# ---------------------------------------------------------------------------
hr("CHECK 3 -- scale-invariant correlation on unnormalized vs normalized attention rows")

logits3 = (q[0] @ k[0].transpose(-2, -1)) / math.sqrt(d)
causal3 = torch.ones(s, s, dtype=torch.bool).tril(0)
raw_exp = torch.exp(logits3).masked_fill(~causal3, 0.0)      # UNNORMALIZED: row sum = Z_i != 1
Z = raw_exp.sum(-1, keepdim=True)
normalized = raw_exp / Z                                      # NORMALIZED: row sum = 1, this IS softmax

row_i = s - 1
raw_row = raw_exp[row_i, :row_i + 1]
norm_row = normalized[row_i, :row_i + 1]
cos_row = float(torch.nn.functional.cosine_similarity(raw_row, norm_row, dim=0))
max_diff_3 = float((raw_row - norm_row).abs().max())
print(f"row {row_i}: raw exp(logit) entries = {raw_row.tolist()}")
print(f"row {row_i}: softmax entries        = {norm_row.tolist()}")
print(f"row {row_i}: row sum, raw / softmax = {float(Z[row_i]):.6f} / {float(norm_row.sum()):.6f}")
print(f"cosine similarity (this row)        = {cos_row:.9f}")
print(f"max|raw - softmax|  (this row)      = {max_diff_3:.6f}")
print("VERDICT: CONFIRMED MECHANISM (didactic) -- cosine similarity reads",
      f"{cos_row:.6f} (indistinguishable from 1.000) because normalizing a row",
      "by its own positive sum cannot rotate its direction, only rescale it,")
print("while the max entrywise difference is",
      f"{max_diff_3:.6f} and the row sums are {float(Z[row_i]):.3f} vs 1.0.")
print("A scale-invariant statistic is BLIND to exactly the row-sum factor")
print("(here Z_i; in S-M's own defect, i+1) that separates an unnormalized")
print("hop from a softmax row -- pattern #3 of the brief, reproduced.")
assert cos_row > 0.999999
assert max_diff_3 > 0.1


# ---------------------------------------------------------------------------
# CHECK 4 -- the REPAIR the Lean file proves: additive-logit reading.
#   gate_zero_logit_identity: q_ij + (scan(g)_i - scan(g)_j) = q_ij at g == 0.
#   Putting the gate into the logits ADDITIVELY (rather than replacing them
#   multiplicatively) makes g==0 parity genuine, because it is softmax on both
#   sides of the comparison.
# ---------------------------------------------------------------------------
hr("CHECK 4 -- the additive-logit repair (gate_zero_logit_identity), for contrast")

qk_logits = (q[:1] @ k[:1].transpose(-2, -1)) / math.sqrt(d)   # [1, s, s]
g_zero_contribution = torch.zeros(s, s, dtype=DT)              # scan(g)_i - scan(g)_j at g==0
mask = torch.ones(s, s, dtype=torch.bool).tril(0)
neg = torch.finfo(DT).min

softmax_plain = torch.softmax(qk_logits[0].masked_fill(~mask, neg), -1)
softmax_plus_zero_gate = torch.softmax(
    (qk_logits[0] + g_zero_contribution).masked_fill(~mask, neg), -1)

max_diff_4 = float((softmax_plain - softmax_plus_zero_gate).abs().max())
print(f"max|softmax(q) - softmax(q + (C_i-C_j)=0)| = {max_diff_4:.6e}")
print("VERDICT: SOUND -- under the additive-logit reading, g==0 really does")
print("leave the logits (and therefore the softmax row) untouched: both sides")
print("of the comparison are softmax rows, so there is no normalization")
print("boundary to cross. This is `gate_zero_logit_identity` in")
print("lean/CEQ/V15.lean, re-derived numerically. It is NOT the construction")
print("S-M actually specifies (S-M is multiplicative and unnormalized, CHECK 2).")
assert max_diff_4 == 0.0


# ---------------------------------------------------------------------------
# CHECK 5 -- a second SOUND instance of the SAME repair pattern already
#   shipped in this repo: ceq/hybrid.py's alpha=0 residual gate.
#     out = stock_attention(q,k,v) + sum_{h=1..K} (alpha*A)^h v
#   At alpha=0 every correction term is literally alpha^h == 0 (h>=1), and
#   ceq/hybrid.py:149/154 short-circuits to `return stock_attention(...)`
#   directly rather than computing a vanishing sum -- so this is code-identity,
#   not a numerical coincidence, and it is the ADDITIVE-residual shape (Check 4)
#   rather than S-M's multiplicative-replacement shape (Check 2).
#   MODEL_CARD.md: "At alpha = 0 this IS stock attention, bitwise."
#   tests/w8/test_w8_real_model.py::test_alpha_zero_is_bitwise_stock_attention
# ---------------------------------------------------------------------------
hr("CHECK 5 -- ceq.hybrid alpha=0 vs stock_attention (MODEL_CARD.md:235)")

qh = torch.randn(2, 4, s, d, dtype=DT)
kh = torch.randn(2, 4, s, d, dtype=DT)
vh = torch.randn(2, 4, s, d, dtype=DT)
gated0, _ = hybrid.ceq_hybrid_attention(None, qh, kh, vh, None, alpha=0.0)
stock = hybrid.stock_attention(qh, kh, vh, None).transpose(1, 2).contiguous()
max_diff_5 = float((gated0 - stock).abs().max())
print(f"max|ceq_hybrid_attention(alpha=0) - stock_attention| = {max_diff_5:.6e}")
print(f"torch.equal (exact bitwise)                          = {torch.equal(gated0, stock)}")
print("VERDICT: SOUND -- exact bitwise equality (not merely small), because")
print("ceq/hybrid.py short-circuits to stock_attention(...) at alpha==0 rather")
print("than computing and adding a term that happens to vanish.")
assert torch.equal(gated0, stock)


hr("SUMMARY")
print("Check 1 (softmax == pivot_unsigned, top-priority claim): SOUND, max|diff| = 0.0e+00")
print("Check 2 (S-M g==0 hop == standard attention, CONTRACT'S OWN CLAIM):     CONFIRMED INSTANCE")
print("Check 3 (correlation hides the i+1 row-sum factor, didactic):          CONFIRMED MECHANISM")
print("Check 4 (additive-logit repair, gate_zero_logit_identity):             SOUND")
print("Check 5 (ceq.hybrid alpha=0 residual gate, same repair shape):         SOUND")
print("\nAll assertions passed." if True else "")
