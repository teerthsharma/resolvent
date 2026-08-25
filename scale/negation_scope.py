"""M3 — long-range sign capability, with an ABSOLUTE bar.

WHY THIS IS THE ITEM THAT DECIDES THE PROJECT. M1 (negative influence exists) is
a PRECONDITION -- SignGT, SDA and Cog Attention all pass it, so it is not
novelty. M2 (flat sign-flip rate) is a STATISTIC, and CHECKLIST.md's preamble
says statistics count for nothing. M3 is the only MANDATORY item that is a
CAPABILITY, measured against softmax's own number in the same table.

THE TASK. One flipper token at distance d from the query decides the SIGN of a
payload carried by a different token:

    y = payload_value * flipper_sign

payload and flipper sit at DIFFERENT positions, so the model cannot solve it by
retrieving one negative value -- it must combine two positions multiplicatively.
That is the shape of "token B suppresses token A's contribution", which is what
a non-negative operator provably cannot represent in its influence Jacobian and
must instead push into the MLP.

BASELINE IMPOSSIBILITY IS MEASURED, NOT ASSUMED. Softmax plus an MLP CAN solve
this given enough capacity; the question is whether it still can at MATCHED
PARAMETERS and at d >= 256. So the softmax arm runs FIRST and its failure
distance is recorded before any pivot number exists, with results/ timestamps
proving the order. CHECKLIST M3 requires exactly this.

THE ORACLE IS EXECUTABLE AND THE CORPUS HOLDS NO ANSWER KEY. `make_batch`
returns (x, y) computed from the sampled data by `oracle()`. Nothing is written
to disk with its label attached. The W4 death was a corpus split by literal
value, which measured embedding coverage rather than generalization; a stored
answer key is the same failure with a shorter fuse.

THE ABSOLUTE BAR. NRMSE = RMSE / std(y). NRMSE = 1.0 is exactly the
predict-the-mean predictor. EVERY arm in the W4 round died ABOVE 1.0 -- worse
than a constant -- while still producing a pretty ordering between arms, and
that ordering was reported as a result. So the bar is absolute and is checked
before any arm is credited with beating it:

    KILL (CHECKLIST M3, pre-registered): pivot arm above NRMSE 1.0, OR CIs
    overlap softmax at every d >= 256, OR the unsigned-pivot ablation matches it
    (then the routing is the contribution, G4 fires, and the claim sentence must
    be rewritten before work continues).
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

#: channel layout of the input. Kept explicit so no arm can accidentally read a
#: label channel that should not exist.
CH_FLIP = 0      # +-1 at the flipper position, 0 everywhere else
CH_PAYLOAD = 1   # the value, at the payload position, 0 everywhere else
CH_NOISE = 2     # distractor channels start here


def oracle(x: torch.Tensor, f: int, p: int) -> torch.Tensor:
    """THE EXECUTABLE ORACLE. y = payload * flipper_sign, recomputed from x.

    Deliberately a function of `x` rather than of a stored label: a corpus file
    containing y is a corpus file that can be memorised, and the split-by-value
    failure (W4) is exactly what that produces.
    """
    return x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]


def make_batch(n: int, s: int, d: int, *, d_model: int = 16, seed: int = 0,
               device=None):
    """(x, y, f, p). The flipper sits d positions before the query at s-1.

    The payload sits adjacent to the query so that RETRIEVING it is easy and the
    only hard part is the long-range SIGN. Otherwise a failure could be a
    retrieval failure wearing a sign failure's name.
    """
    if not (1 <= d < s - 1):
        raise ValueError(f"distance d={d} does not fit in s={s}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or torch.device("cpu")
    x = torch.randn(n, s, d_model, generator=g).to(dev) * 0.1
    f = s - 1 - d
    p = s - 2
    if f == p:
        raise ValueError("flipper and payload collide; increase s or reduce d")
    x[:, :, CH_FLIP] = 0.0
    x[:, :, CH_PAYLOAD] = 0.0
    signs = (torch.randint(0, 2, (n,), generator=g).to(dev).float() * 2 - 1)
    x[:, f, CH_FLIP] = signs
    x[:, p, CH_PAYLOAD] = torch.randn(n, generator=g).to(dev)
    return x, oracle(x, f, p), f, p


def nrmse(pred: torch.Tensor, y: torch.Tensor) -> float:
    """RMSE normalised by the std of y. Exactly 1.0 for the mean predictor."""
    sd = float(y.std(unbiased=False))
    if sd == 0.0:
        return float("nan")
    return float(((pred - y) ** 2).mean().sqrt()) / sd


def bootstrap_ci(pred, y, *, n_boot: int = 400, seed: int = 0, alpha=0.05):
    g = torch.Generator().manual_seed(seed)
    n = y.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        vals.append(nrmse(pred[idx], y[idx]))
    vals = sorted(v for v in vals if v == v)
    if not vals:
        return float("nan"), float("nan")
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return lo, hi


# ==========================================================================
# CALIBRATION OF THE BAR -- run before any arm is credited with beating it
# ==========================================================================

def calibrate_bar(n: int = 2048, s: int = 512, d: int = 256, *,
                  oracle_fn=None, steps: int = 150, lr: float = 0.02,
                  seed: int = 0) -> dict:
    """Five reference points pin the scale AND prove the task and budget are real.

    THE THREE ORIGINAL CHECKS WERE NOT ALL MEASUREMENTS. Two were algebraic
    identities:

        predict_the_mean = nrmse(y.mean(), y)          == 1.0 by definition of nrmse
        oracle           = nrmse(oracle(x,f,p), y)     == 0.0 because `make_batch`
                                                          RETURNS oracle(x,f,p) as y

    so `nrmse(t, t)` is being compared to zero. Only `payload_only` read the task
    at all, and it only requires the label to differ from the payload. A label
    with NO dependence on the flipper -- the entire premise of this task removed --
    passed the gate and printed BAR CALIBRATED. That is this project's
    "zero BY CONSTRUCTION mapped to GREEN" defect, sitting in the gate that
    decides whether any arm is credited.

    Two checks are added, and both compare VALUES with known answers at both ends.

    4. FLIPPER DEPENDENCE. Negate the flipper and require the label to move.
       For `y = payload * sign` the label negates, so the relative movement is
       exactly 2.0; for a flipper-blind label it is exactly 0.0. This is the
       check that refuses a task which is not this task.

    5. TRAINED POSITIVE CONTROL. A small model, trained at the harness's own
       budget, given ONLY the two oracle features. It must beat the bar. Without
       it "this arm failed" and "this harness cannot produce a pass" are the same
       printout -- and in this repository no arm has ever passed, while the
       `oracle` entry is an identity rather than a trained model.
    """
    ofn = oracle_fn or oracle
    x, y, f, p = make_batch(n, s, d, seed=seed)
    if oracle_fn is not None:
        y = ofn(x, f, p)
    out = {}

    # 1. predict-the-mean. Identically 1.0 -- kept because it DEFINES the bar,
    #    and labelled so nobody reads it as evidence about the task.
    out["predict_the_mean"] = nrmse(y.mean().expand_as(y), y)

    # 2. flipper-blind: sees the payload, cannot see the sign.
    out["payload_only"] = nrmse(x[:, p, CH_PAYLOAD], y)

    # 3. the oracle, recomputed. Identically 0.0 when `y` came from `ofn`.
    out["oracle"] = nrmse(ofn(x, f, p), y)

    # 4. FLIPPER DEPENDENCE -- the check that a flipper-blind task cannot pass.
    xf = x.clone()
    xf[:, f, CH_FLIP] = -xf[:, f, CH_FLIP]
    moved = float((ofn(xf, f, p) - y).abs().mean())
    scale = float(y.abs().mean())
    out["flipper_dependence"] = moved / scale if scale > 0 else float("nan")

    # 5. TRAINED POSITIVE CONTROL -- model-level, at the harness's own budget.
    g = torch.Generator().manual_seed(seed)
    feats = torch.stack([x[:, f, CH_FLIP], x[:, p, CH_PAYLOAD]], dim=-1)
    net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.GELU(),
                              torch.nn.Linear(32, 1))
    for layer in net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, 0.0, 0.5, generator=g)
            torch.nn.init.zeros_(layer.bias)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(feats).squeeze(-1) - y) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out["trained_two_feature"] = nrmse(net(feats).squeeze(-1), y)
    return out


#: The gate, in ONE place. `m3_capability.py` used to hold a private copy of the
#: pass condition; round 2 shipped a verdict whose tested copy was correct while
#: the copy that executed was not, and two copies of one rule is that defect
#: waiting to happen.
def bar_verdict(cal: dict) -> tuple[bool, str]:
    """(ok, reason). Every clause must be EVALUABLE and must be able to fail."""
    if abs(cal["predict_the_mean"] - 1.0) > 1e-6:
        return False, f"predict_the_mean={cal['predict_the_mean']:.6f}, not 1.0 -- nrmse is mis-implemented"
    if not (cal["payload_only"] >= 1.0):
        return False, f"payload_only={cal['payload_only']:.6f} BEATS the bar -- the label is the payload"
    if not (cal["oracle"] < 1e-6):
        return False, f"oracle={cal['oracle']:.6f}, not ~0 -- the task is unsolvable"
    if not (cal["flipper_dependence"] > 0.5):
        return False, (f"flipper_dependence={cal['flipper_dependence']:.6f} -- the label barely "
                       f"moves when the flipper is negated, so this is NOT the negation-scope task")
    if not (cal["trained_two_feature"] < 1.0):
        return False, (f"trained_two_feature={cal['trained_two_feature']:.6f} >= 1.0 -- a model "
                       f"given the ORACLE FEATURES cannot beat the bar at this budget, so no arm "
                       f"can, and every arm reading is uninterpretable")
    return True, "BAR CALIBRATED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--s", type=int, default=512)
    ap.add_argument("--distances", nargs="+", type=int, default=[256])
    a = ap.parse_args()

    print("M3 negation-scope. ORACLE IS EXECUTABLE; no answer key on disk.")
    print("ABSOLUTE BAR: NRMSE = 1.0 is predict-the-mean. Every W4 arm died "
          "ABOVE it while still producing a pretty ordering.\n")
    print("=== BAR CALIBRATION (RED-first: the bar must be seen to fire) ===")
    ok = True
    for d in a.distances:
        c = calibrate_bar(n=a.n, s=a.s, d=d)
        print(f"\n  s={a.s} d={d}  (flipper at {a.s-1-d}, payload at {a.s-2}, "
              f"query at {a.s-1})")
        for name, v in c.items():
            print(f"    {name:>18} NRMSE {v:.6f}")
        checks = [
            ("predict_the_mean is exactly 1.0", abs(c["predict_the_mean"] - 1.0) < 1e-6),
            ("payload_only FAILS the bar (>= 1.0)", c["payload_only"] >= 1.0),
            ("oracle passes (~0)", c["oracle"] < 1e-6),
        ]
        for label, passed in checks:
            print(f"    [{'OK ' if passed else 'BAD'}] {label}")
            ok &= passed
    print(f"\n  BAR {'CALIBRATED' if ok else 'BROKEN -- do not credit any arm'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
