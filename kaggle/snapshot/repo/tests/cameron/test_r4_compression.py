"""R4 -- compress by forgetting.  VERDICT: DELETE.

REQUIREMENTS.md R4, verbatim:

    Per-token memory shrinks because the module *structurally cannot* retain
    what does not matter -- not because a pruning heuristic was bolted on.

    Falsifier: effective size of the settled state across many contexts. Near
    full width means no compression story. Task-relevant reconstruction must
    survive; reconstruction of crushed noise must not.

The deletion test is `test_crushed_noise_is_forgotten_more_than_a_retained_
token`. It is RED, and it is red because R4's falsifier as written is passed by
things that are not forgetting.

Run literally, R4 passes: effective size 2.3384 of 32, task-relevant R2
0.97333, crushed-noise R2 -0.11446, held out 300/100. Every clause satisfied.
Now the controls, same probe, same split:

    arm              eff size   task R2   crushed R2   RETAINED token R2   ctx mean R2
    attention          1.0059   0.99947     -0.11579           -0.10524      -0.10608
    row-stochastic     2.2887   0.98590     -0.11947           -0.11119      -0.10224
    perron             2.3384   0.97333     -0.11446           -0.11224      -0.10015

Two columns kill it.

1. The RETAINED token -- a filler the gate did NOT crush -- reconstructs at
   -0.11224, indistinguishable from the crushed one at -0.11446. So does the
   context mean, at -0.10015. The readout forgets everything equally. The
   clause "reconstruction of crushed noise must not survive" is satisfied by a
   32-wide readout summarizing 96 tokens whatever the gate does, which is the
   failure `Epsilon-Hollow` names in its own words about referenced_evictions:
   "It is not a measurement. It is the only value the expression can produce."

2. Standard attention scores -0.11579 on the same clause and 0.99947 on the
   task clause. The baseline passes R4 outright, at effective size 1.0059.

And the compression that looks real is not information loss. rho < 1 is exactly
the condition that makes I - A invertible, so the settled state is a bijection
of the input: (I - A) Z reproduces X to 7.699e-14, smallest singular value of
I - A is 0.011443, condition number 767.8. The certificate R1 and R3 need is
the same inequality that forbids R4, and no gate changes it -- a gate scales
columns of A, it cannot make I - A singular.

The one thing that does forget is eviction: remove the token from the key set
rather than gate its value to zero. Measured below, bitwise. It is also a rule
applied from outside the operator, which is what R4's first sentence excludes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import perron as pn
from test_r3_perturbation import (N_TOK, D_FEAT, RHO, context, edit,
                                  arm_attention, arm_perron, arm_row_stochastic)

N_CTX = 400
N_TRAIN = 300              # held-out split: a 32->32 fit on 400 samples overfits
READOUT = 0                # the sink row, the module's memory of the context
KEEP = 16                  # eviction budget for the bolted-on arm, 16 of 96

COMPRESSED = 0.25 * D_FEAT   # R4: "near full width means no compression story"
TASK_SURVIVES = 0.90         # R4: "task-relevant reconstruction must survive"
NOISE_CRUSHED = 0.10         # R4: "reconstruction of crushed noise must not"
SELECTIVITY = 0.25           # crushed must be forgotten THIS much more than kept


def _probe(feature, target):
    f, t = torch.stack(feature), torch.stack(target)
    return pn.linear_probe_r2_heldout(f[:N_TRAIN], t[:N_TRAIN],
                                      f[N_TRAIN:], t[N_TRAIN:])


def corpus(arm, device, n=N_CTX):
    """Per context: the readout row, plus four reconstruction targets --
    the causal token (task), a CRUSHED filler, a RETAINED filler (the negative
    control R4 does not name and needs), and the context mean."""
    read, task, crushed, kept, mean = [], [], [], [], []
    for s in range(n):
        x, c, f = context(s, device)
        j = (c + 1) % N_TOK
        j = (c + 2) % N_TOK if j == f else j
        read.append(arm(x)[READOUT])
        task.append(x[c])
        crushed.append(x[f])
        kept.append(x[j])
        mean.append(x.mean(0))
    return dict(read=read, task=task, crushed=crushed, kept=kept, mean=mean,
                size=float(np.median([pn.effective_size(arm(context(s, device)[0]))
                                      for s in range(24)])))


# ------------------------------------------------------------ R4's deletion test

def test_crushed_noise_is_forgotten_more_than_a_retained_token(device):
    """RED. This is the test that deletes R4.

    R4's falsifier says crushed noise must not reconstruct. It does not -- but
    neither does a filler token the gate KEPT, at -0.11224 against -0.11446,
    nor the context mean at -0.10015. A clause that a negative control passes
    identically is not measuring forgetting; it is measuring the width of the
    readout.
    """
    c = corpus(arm_perron, device)
    crushed, kept = _probe(c["read"], c["crushed"]), _probe(c["read"], c["kept"])
    assert kept - crushed >= SELECTIVITY, (
        f"crushed R2 {crushed:.5f} vs retained-token R2 {kept:.5f}: gap "
        f"{kept - crushed:+.5f} < {SELECTIVITY}. The readout forgets both "
        f"equally, so nothing was crushed selectively. R4 deleted."
    )


# ------------------------------------------- what is true, measured and green

def test_r4_passes_when_run_literally_and_so_does_the_baseline(device):
    """Both halves in one place. Every clause of R4 is satisfied by the module
    -- and by one softmax hop, which is the reason the clauses do not select a
    mechanism."""
    for arm in (arm_perron, arm_attention):
        c = corpus(arm, device)
        assert c["size"] <= COMPRESSED, c["size"]
        assert _probe(c["read"], c["task"]) >= TASK_SURVIVES
        assert _probe(c["read"], c["crushed"]) <= NOISE_CRUSHED


def test_the_state_is_a_bijection_so_nothing_is_destroyed(device):
    """The obstruction as the theorem it is. rho < 1 makes I - A invertible;
    an invertible map destroys nothing. Reconstruction is on the TOKEN axis,
    X = (I - A) Z, which is where the settling actually happens."""
    x, _, _ = context(0, device)
    a = pn.perron_operator(x, rho=RHO)
    ok, rho, _ = pn.certificate(a, torch.ones(N_TOK, dtype=a.dtype, device=a.device))
    z = pn.settle(a, x, RHO)
    eye = torch.eye(N_TOK, dtype=a.dtype, device=a.device)
    err = float(((eye - a) @ z - x).abs().max())
    smin = float(torch.linalg.svdvals(eye - a).min())
    assert ok and rho < 1.0
    assert err < 1e-12, f"(I-A)Z - X = {err:.3e}"
    assert smin > 0.0, f"smallest singular value {smin:.6f}"


def test_effective_size_looks_like_compression_and_certifies_nothing(device):
    """Both numbers are true at once, which is the trap. A participation ratio
    of 2.29 out of 32 is a 14x apparent squeeze on a state that is a bijection
    of a 29.02-of-32 input. Spectral concentration is not information loss."""
    x, _, _ = context(0, device)
    z = arm_row_stochastic(x)
    a = GAMMA_P = 0.9 * pn.transition(x)
    eye = torch.eye(N_TOK, dtype=x.dtype, device=x.device)
    assert pn.effective_size(z) <= COMPRESSED
    assert pn.effective_size(x) >= 0.75 * D_FEAT
    assert float(((eye - a) @ z - x).abs().max()) < 1e-12


def test_gating_a_column_to_zero_does_not_evict_the_token(device):
    """Why a gate cannot be the forgetting mechanism. Set a filler's salience
    to EXACTLY zero, so its column of A is exactly zero, and the settled state
    still moves when that token is edited -- through the softmax normalizer of
    every other row. Zero weight is not absence."""
    x, _, f = context(0, device)

    def hard(v):
        s = RHO * (pn.robust_z(v[:, pn.CONTROL]) > pn.OUTLIER_MADS).to(v.dtype)
        a = pn.transition(v) * s[None, :]
        assert float(a[:, f].abs().max()) == 0.0, "the column is not actually zero"
        return pn.settle(a, v, RHO)

    keep = [i for i in range(N_TOK) if i != f]
    moved = float((hard(edit(x, f, 0))[keep] - hard(x)[keep]).abs().max())
    assert moved > 0.0, "gating to zero happened to evict after all"


def test_eviction_does_forget_and_is_a_bolted_on_policy(device):
    """The honest alternative, and the reason it is not R4.

    Remove the token from the key set -- `NeMo-Relay`'s scope exit, and
    `foliation`'s collapse of a free face, refcount zero and no resident
    children -- and the settled state is BITWISE unchanged under any edit of
    the dropped token. Genuine structural forgetting, and a pruning step chosen
    by a rule outside the operator, which R4's first sentence excludes.
    """
    x, c, f = context(0, device)
    keep = torch.topk(pn.salience(x, RHO), KEEP).indices.sort().values
    kept = set(keep.tolist())
    assert f not in kept and c in kept, (f, c, sorted(kept)[:5])

    def evicted(v):
        sub = v[keep]
        return pn.settle(pn.perron_operator(sub, rho=RHO), sub, RHO)

    assert torch.equal(evicted(x), evicted(edit(x, f, 0))), "not bitwise"
    assert not torch.equal(evicted(x), evicted(edit(x, c, 0))), "also deaf to causal"


def test_eviction_at_matched_budget_beats_recency_and_random(device):
    """The null models the eviction arm has to beat, at the SAME budget.
    `foliation`'s own pool sweep records that at 8 plaques both it and LRU lose
    to same-budget random, so a budgeted policy reported without this control
    is not a measurement."""
    rng = np.random.default_rng(0)
    out = {}
    for name in ("salience", "recency", "random"):
        read, task = [], []
        for st in range(N_CTX):
            x, c, _ = context(st, device)
            if name == "salience":
                k = torch.topk(pn.salience(x, RHO), KEEP).indices
            elif name == "recency":
                k = torch.arange(N_TOK - KEEP, N_TOK, device=x.device)
            else:
                k = torch.tensor(rng.choice(N_TOK, KEEP, replace=False),
                                 device=x.device)
            sub = x[k.sort().values]
            read.append(pn.settle(pn.perron_operator(sub, rho=RHO), sub, RHO)[READOUT])
            task.append(x[c])
        out[name] = _probe(read, task)
    assert out["salience"] >= out["recency"] + 0.1, out
    assert out["salience"] >= out["random"] + 0.1, out
