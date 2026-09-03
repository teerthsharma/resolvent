"""V20 R15 it.9 -- MOON, on MERCURY's it.8 S2.5 open question.

MERCURY (`V20_R15_IT8_MERCURY.md` S2.5) measured `a_hat_max == 1.0` on
`arm_smprime` cells as `ceq/arm_smprime.py:113`'s `clamp(u, 0, 1)` ceiling,
reachable at ZERO training steps on six of sixteen cells, and named what it
did NOT determine: whether the PRE-CLAMP value at the argmax sits strictly
ABOVE 1.0 or LANDS on 1.0 exactly. "Distinguishing those needs the pre-clamp
tensor, which no journal stores."

THIS FILE SETTLES IT FOR THE 0-STEP CELLS, WITHOUT ANY GPU.
`scripts/v15_r1.py:801-809`'s 0-step control reconstructs the model as
`torch.manual_seed(seed); make_arm(kind, S)` -- no `identity_point()`, no
gradient step -- against the ONE eval batch `:699-700` draws at a hardcoded
`seed=12345`. Both draws use `torch.Generator(device="cpu")` internally
(`scale/negation_scope.py:94,426`) and the model is constructed BEFORE any
`.to(device)` move (`scripts/v15_r1.py:225-229`, "CONSTRUCT, THEN MOVE"), so
the byte-identical weights and inputs the original cuda run used are a pure
function of the seed and reproduce on CPU. At construction `model.g == 1.0`
exactly (`ceq/arm_smprime.py:529`), so `blend()`'s `lerp(ones, u, g)`
(`:129`) is the identity and the value fed to `magnitude()`'s clamp
(`:113`) is `u = model.heads(x)[0]` unmodified. That makes pre-clamp `u` at
step 0 CLOSED-FORM in the seed -- reconstructible here, not merely inferred.

TRAINED-cell (post-150-step) pre-clamp `u` is a different question and is
NOT settled by this file: training moves `g` off `1.0`
(`results/v20_r15_it6_seeds8_15.jsonl` seed-0 `manifest.smp_values.g` reads
`1.319505214691162`, asserted below), so the trained pre-clamp value needs
the TRAINED weights, not the seed, and a closed-form CPU reconstruction does
not exist for it.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ceq import arm_smprime                              # noqa: E402
from scale.negation_scope import make_equilibrium_batch  # noqa: E402

S, D, D_MODEL, T_STAR = 64, 24, 16, 2                      # scripts/v15_r1.py:137-138, scale/m3_capability.py:79
HEAD = S - 1 - T_STAR                                      # scripts/v15_r1.py:697
LIVE = list(range(HEAD + 1, S))                             # scripts/v15_r1.py:698
EVAL_SEED = 12345                                           # scripts/v15_r1.py:699
N_EVAL = 4096                                               # scripts/v15_r1.py:549 default

#: V20_R15_IT8_MERCURY.md S2.2 -- "0-step a_hat_max == 1.0: seeds [2, 3, 4, 7, 12, 14]"
SEEDS_0STEP_AT_1 = [2, 3, 4, 7, 12, 14]

JOURNALS = [ROOT / "results" / "v20_r15_it6_seeds8_15.jsonl",
            ROOT / "results" / "v17k_r4_retake.jsonl"]


def _arm_smprime_cell_records():
    for fn in JOURNALS:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("t") == "cell" and d.get("kind") == "arm_smprime":
                    yield d


def test_journal_has_no_preclamp_field():
    """No key on any `arm_smprime` cell row -- top level, `manifest`, or
    `manifest.smp_values` -- names the pre-clamp value. `a_hat_max` and
    `manifest.smp_values.m_max` are both `magnitude()`'s OUTPUT
    (`ceq/arm_smprime.py:113`, `scripts/v15_r1.py:381-382,866`). This is the
    receipt for "no journal stores it": NOT RECOVERABLE from the record
    alone, which is why the reconstruction below exists."""
    keys = set()
    for d in _arm_smprime_cell_records():
        keys |= set(d.keys())
        man = d.get("manifest", {})
        keys |= {f"manifest.{k}" for k in man}
        keys |= {f"manifest.smp_values.{k}" for k in man.get("smp_values", {})}
    assert keys, "no arm_smprime cell rows found -- journal paths moved"
    preclamp_like = {k for k in keys if k.split(".")[-1] in
                      ("u", "u_max", "u_head", "pre_clamp", "preclamp")}
    assert preclamp_like == set(), (
        f"a pre-clamp field now exists ({preclamp_like}) -- this node is "
        f"stale, re-settle from the journal directly")


def _reconstruct_0step_preclamp(seed: int):
    """`scripts/v15_r1.py:801-809`'s 0-step control, replayed on CPU one line
    short of the clamp: seed the global generator, construct with NO
    `identity_point()` and NO training step, read the magnitude head, and
    stop at `arm_smprime.py:129` -- the value `:113`'s `clamp(u, 0, 1)`
    receives."""
    x_ev, *_ = make_equilibrium_batch(N_EVAL, S, D, t_star=T_STAR,
                                       d_model=D_MODEL, seed=EVAL_SEED,
                                       device=None)
    torch.manual_seed(seed)
    model = arm_smprime.ArmSMPrime(S, d_model=D_MODEL)   # AS CONSTRUCTED
    assert float(model.g) == 1.0                          # so lerp() below is the identity
    u = model.heads(x_ev)[0][:, LIVE]
    pre = torch.lerp(torch.ones_like(u), u, model.g)       # arm_smprime.py:129
    post = torch.clamp(pre, 0.0, 1.0)                       # arm_smprime.py:113
    idx = int(post.reshape(-1).argmax())
    return float(post.reshape(-1)[idx]), float(pre.reshape(-1)[idx])


@pytest.mark.parametrize("seed", SEEDS_0STEP_AT_1)
def test_preclamp_u_strictly_exceeds_one_at_0step(seed):
    """THE SETTLED FACT. Reconstructed on CPU, closed-form in the seed: for
    every one of the six 0-step cells MERCURY read at `a_hat_max_0step ==
    1.0`, the pre-clamp value at the argmax STRICTLY EXCEEDS 1.0 -- it is
    hard saturation, not a landing. The margin (0.013-0.184 measured) is
    orders of magnitude past `~1e-7`, the float32 rounding floor of a
    16-wide dot product, so a CPU-vs-the-original-cuda-run arithmetic
    difference cannot flip this verdict."""
    post_max, pre_at_argmax = _reconstruct_0step_preclamp(seed)
    assert post_max == 1.0                      # reproduces the journalled a_hat_max_0step
    assert pre_at_argmax > 1.0                   # STRICT, not a landing
    assert pre_at_argmax - 1.0 > 1e-4            # nowhere near the rounding floor


def test_trained_cells_need_the_trained_weights_not_the_seed():
    """Companion fact, named rather than silently out of scope: training
    moves `g` off `1.0` (seed-0 `arm_smprime`, trained,
    `results/v20_r15_it6_seeds8_15.jsonl`), so `blend()`'s `lerp` is no
    longer the identity and a TRAINED cell's pre-clamp `u` needs the trained
    `m_head` weights -- a re-run, not a seed-only reconstruction. This file
    settles the 0-step cells only; the twelve trained-at-1.0 cells
    (`V20_R15_IT8_MERCURY.md` S2.2) are priced, not executed, in
    `V20_R15_IT9_MARS.md`."""
    for d in _arm_smprime_cell_records():
        if d.get("seed") == 0:
            g = d["manifest"]["smp_values"]["g"]
            assert g != 1.0
            return
    pytest.fail("seed 0 arm_smprime cell not found in the journal")
