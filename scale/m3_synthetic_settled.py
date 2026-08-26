"""M3 harness dry-run on SYNTHETIC arms whose verdict is known in advance.

WHY THIS FILE EXISTS. RULE 2 pins the real settled-vs-unsettled M3 run to
iteration 12. A null result there is worth nothing unless the harness has first
been shown to report a difference that was planted on purpose. Round 5 struck
four gates that each measured something adjacent to what was pre-registered; the
cheapest defence against a fifth is to hand the comparison machinery an answer it
cannot get wrong and check that it returns that answer.

THE ARMS. All three are the shipped `m3_capability.Arm` plus ONE extra scalar
weight, initialised to exactly zero, multiplying one extra input feature. They
differ only in what that feature carries:

    planted : the oracle value itself, y = payload * flipper_sign. The arm can
              learn to read the label directly. CORRECT ANSWER: SETTLED WINS.
    null    : nothing -- the feature branch is not taken at all, so the arm is
              BITWISE the twin at every training step. CORRECT ANSWER:
              NO DIFFERENCE.
    noise   : the oracle value SHUFFLED across the batch. Same marginal
              distribution, zero per-example information, one live extra
              parameter. Not asserted anywhere: it is the reading that says
              whether an extra useless parameter alone can move the verdict.

DECLARED CHEAT. `planted` reads the label. That is the entire point and no
number taken from it is a capability claim about any construction. The file
exists to measure the INSTRUMENT.

THE CONTRAST, AND WHY NOT THE SHIPPED ONE. `negation_scope.bootstrap_ci` is a
MARGINAL interval over one eval batch from one training run. Two marginal
intervals overlapping is not a test of a difference, and
`tests/chase/test_m3_capability_harness.py` already records that the interval is
narrower than the same arm's seed-to-seed spread, which makes the pre-registered
"CIs overlap softmax" kill decidable by the choice of seed. The instrument here
is instead a PAIRED bootstrap over training seeds:

    at seed s, twin and settled share the training batch, the eval batch and the
    initialisation RNG, so

        delta_s = NRMSE_twin(s) - NRMSE_settled(s)

    contains no batch or initialisation variance that is common to the pair. The
    bootstrap resamples the delta_s, so the interval covers exactly the variance
    the pairing does not remove.

    LIMIT: five seeds. A percentile interval over five paired values is coarse,
    and it cannot resolve an effect smaller than the seed-to-seed spread. It is
    the number of seeds the contract asks for and it is stated, not hidden.

TRAINING IS NOT REIMPLEMENTED. `scale/paired_arm.py::train_and_predict` is
`run_arm`'s loop verbatim and is called here unchanged; only the class it
instantiates is swapped, so a paired test between two arms is not a test of two
training loops. The 0-step gate comes from `m3_capability.run_arm` at steps=0,
which is the shipped gate's own code path.

THREADS ARE PINNED IN THIS FILE. The same command at 2 and at 20 threads moved
the published pivot_unsigned reading from 0.747528 to 0.747062
(results/m3_capability.txt, RUNs 2026-08-25 22:22:49 and 22:27:55).

BUCKETED (ADR-001). One unit is one (case, arm, seed) training run.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)                      # pinned HERE, not by the launcher

from scale import m3_capability as M3                             # noqa: E402
from scale import negation_scope as NS                            # noqa: E402
from scale import paired_arm as PA                                # noqa: E402
from scale.bucket import run_bucket, require_complete             # noqa: E402

NAME = "chase_m3_synth"

#: results/m3_capability.txt, RUN 2026-08-25 18:56:53, torch.get_num_threads()=2
PUBLISHED_SOFTMAX_2048 = dict(
    s=64, d=24, steps=150, n_train=2048, n_eval=512, seed=0,
    n_params=4769,
    nrmse0_train=1.003355, nrmse0_eval=1.000432,
    train_nrmse=0.790853, eval_nrmse=0.949529,
    ci_lo=0.891522, ci_hi=1.011235,
)

FEATURES = ("planted", "null", "noise")


# ---------------------------------------------------------------- the arms ---
class SyntheticSettledArm(M3.Arm):
    """A shipped arm plus one scalar-weighted extra feature.

    `hint_w` initialises to exactly 0.0, so the arm's output is BITWISE the
    twin's before training. Without that, the plant would trip the harness's own
    0-step gate (untrained NRMSE >= 1.0) and the dry-run would be exercising the
    abort path instead of the comparison path.
    """

    def __init__(self, kind: str, s: int, *, f: int, p: int,
                 shuffle: bool = False, feature: str | None = None, **kw):
        super().__init__(kind, s, **kw)
        if feature is None:
            feature = "noise" if shuffle else "planted"
        if feature not in FEATURES:
            raise ValueError(feature)
        self.f, self.p, self.feature = f, p, feature
        self.hint_w = nn.Parameter(torch.zeros(1))

    def _hint(self, x: torch.Tensor) -> torch.Tensor:
        feat = NS.oracle(x, self.f, self.p)
        if self.feature == "noise":
            # a fixed permutation of the SAME values: identical marginal
            # distribution, no per-example information. Deterministic in the
            # batch size so train and eval do not share a permutation.
            g = torch.Generator().manual_seed(90000 + feat.shape[0])
            feat = feat[torch.randperm(feat.shape[0], generator=g)]
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        if self.feature == "null":
            # The branch is not taken at all, so the arm is bitwise the twin and
            # `hint_w` never receives a gradient. `base + hint_w * 0` would be
            # equal for every finite base but not for -0.0, and the point of
            # this arm is that it is IDENTICAL, not nearly so.
            return base
        return base + self.hint_w * self._hint(x)


def _arm_factory(feature: str, f: int, p: int):
    """A class with `Arm(kind, s)`'s signature, for the class swap below."""

    class _Synth(SyntheticSettledArm):
        def __init__(self, kind, s):
            super().__init__(kind, s, f=f, p=p, feature=feature)

    return _Synth


@contextlib.contextmanager
def _as_arm_class(cls):
    """Swap the class `run_arm` and `train_and_predict` instantiate.

    Both call the module-global name `Arm`, so this reuses their loops verbatim
    rather than copying them. The swap is undone on every exit path.
    """
    old_m3, old_pa = M3.Arm, PA.Arm
    M3.Arm, PA.Arm = cls, cls
    try:
        yield
    finally:
        M3.Arm, PA.Arm = old_m3, old_pa


# ------------------------------------------------------------ the contrast ---
def shrink_toward(pred: torch.Tensor, y: torch.Tensor,
                  delta: float) -> torch.Tensor:
    """pred moved a fraction `delta` of the way to the truth.

    Every residual is scaled by (1 - delta), so NRMSE scales by exactly the same
    factor. Used to plant a margin that is known in closed form and depends on
    no training at all.
    """
    return y + (1.0 - delta) * (pred - y)


def verdict_of(ci_lo: float, ci_hi: float) -> str:
    """The verdict, as a pure function of the interval. Strict at zero: an
    interval touching zero is not an interval excluding it (G6)."""
    if ci_lo > 0.0:
        return "SETTLED WINS"
    if ci_hi < 0.0:
        return "TWIN WINS"
    return "NO DIFFERENCE"


def contrast(twin: list[float], settled: list[float], *, n_boot: int = 10000,
             seed: int = 0) -> dict:
    """Paired bootstrap over seeds on NRMSE_twin - NRMSE_settled.

    Positive delta means the settled arm has the LOWER error, i.e. it wins.
    """
    if len(twin) != len(settled):
        raise ValueError(f"unpaired: {len(twin)} vs {len(settled)}")
    d = [t - s for t, s in zip(twin, settled)]
    n = len(d)
    point = sum(d) / n
    g = torch.Generator().manual_seed(seed)
    reps = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        reps.append(sum(d[int(i)] for i in idx) / n)
    reps.sort()
    lo = reps[int(0.025 * len(reps))]
    hi = reps[min(len(reps) - 1, int(0.975 * len(reps)))]
    return dict(delta=point, ci_lo=lo, ci_hi=hi, n_seeds=n, n_boot=n_boot,
                per_seed_delta=d, verdict=verdict_of(lo, hi))


# ----------------------------------------------------------------- the run ---
def reproduce_softmax(**over) -> dict:
    """The published softmax reading, re-taken. Runs BEFORE any synthetic number."""
    cfg = {k: PUBLISHED_SOFTMAX_2048[k] for k in
           ("s", "d", "steps", "n_train", "n_eval", "seed")}
    cfg.update(over)
    xt, yt, _f, _p = NS.make_batch(cfg["n_train"], cfg["s"], cfg["d"],
                                   d_model=M3.D_MODEL, seed=cfg["seed"])
    xe, ye, _f2, _p2 = NS.make_batch(cfg["n_eval"], cfg["s"], cfg["d"],
                                     d_model=M3.D_MODEL, seed=cfg["seed"] + 12345)
    return M3.run_arm("softmax", xt, yt, xe, ye, s=cfg["s"], steps=cfg["steps"],
                      seed=cfg["seed"])


def _unit(p: dict) -> dict:
    """One arm at one seed: 0-step gate plus trained eval NRMSE.

    `feature=None` means the shipped twin, built by the unpatched class.
    """
    s, d, seed = p["s"], p["d"], p["seed"]
    f, pos = s - 1 - d, s - 2
    cls = None if p["feature"] is None else _arm_factory(p["feature"], f, pos)
    ctx = contextlib.nullcontext() if cls is None else _as_arm_class(cls)

    xt, yt, _a, _b = NS.make_batch(p["n_train"], s, d, d_model=M3.D_MODEL,
                                   seed=seed)
    xe, ye, _c, _e = NS.make_batch(p["n_eval"], s, d, d_model=M3.D_MODEL,
                                   seed=seed + 12345)
    with ctx:
        red = M3.run_arm(p["kind"], xt, yt, xe, ye, s=s, steps=0, seed=seed)
        pred, y_eval, npar = PA.train_and_predict(
            p["kind"], s=s, d=d, steps=p["steps"], n_train=p["n_train"],
            n_eval=p["n_eval"], seed=seed)
    return dict(eval_nrmse=NS.nrmse(pred, y_eval), n_params=npar,
                nrmse0_train=red["nrmse0_train"], nrmse0_eval=red["nrmse0_eval"])


def _key(p: dict) -> str:
    return (f"{p['kind']}_{p['feature'] or 'twin'}_s{p['s']}_d{p['d']}"
            f"_st{p['steps']}_ntr{p['n_train']}_nev{p['n_eval']}_sd{p['seed']}")


def _units(feature, *, kind, s, d, steps, n_train, n_eval, seeds):
    out = []
    for sd in seeds:
        p = dict(kind=kind, feature=feature, s=s, d=d, steps=steps,
                 n_train=n_train, n_eval=n_eval, seed=sd)
        out.append((_key(p), p))
    return out


def dry_run(*, arm: str = "planted", kind: str = "softmax", s: int = 64,
            d: int = 24, steps: int = 150, n_train: int = 2048,
            n_eval: int = 512, seed: int = 0, n_seeds: int = 5,
            bucket: bool = False, budget_s: float = 420.0,
            n_boot: int = 10000) -> dict:
    """One synthetic case against its twin, `n_seeds` paired training seeds.

    `arm` is a key of FEATURES, or "twin_offset" for the twin compared against
    ITSELF at a disjoint seed block -- two constructions that are identical by
    definition, differing only in initialisation and batch.
    """
    seeds = [seed + i for i in range(n_seeds)]
    common = dict(kind=kind, s=s, d=d, steps=steps, n_train=n_train,
                  n_eval=n_eval)
    if arm == "twin_offset":
        a_units = _units(None, seeds=[100 + t for t in seeds], **common)
    else:
        a_units = _units(arm, seeds=seeds, **common)
    b_units = _units(None, seeds=seeds, **common)

    if bucket:
        acc = run_bucket(NAME, b_units + a_units, _unit, budget_s=budget_s)
        # `run_bucket` counts `remaining` against EVERY key in the journal, and
        # the four cases share one journal and one twin block, so a later case
        # sees more units journalled than it asked for and `remaining` goes
        # NEGATIVE. Only a positive value means work is left; `require_complete`
        # is the authority on whether this case's own units are all present.
        if acc["remaining"] > 0:
            return dict(incomplete=acc)
        vals = require_complete(NAME, b_units + a_units)
        got = {k: vals[k] for k, _ in b_units + a_units}
    else:
        got = {k: _unit(p) for k, p in b_units + a_units}

    twin = [got[k]["eval_nrmse"] for k, _ in b_units]
    settled = [got[k]["eval_nrmse"] for k, _ in a_units]
    out = contrast(twin, settled, n_boot=n_boot, seed=0)
    out.update(arm=arm, kind=kind, seeds=seeds, twin_nrmse=twin,
               settled_nrmse=settled,
               nrmse0_twin=[got[k]["nrmse0_eval"] for k, _ in b_units],
               nrmse0_settled=[got[k]["nrmse0_eval"] for k, _ in a_units],
               n_params_twin=got[b_units[0][0]]["n_params"],
               n_params_settled=got[a_units[0][0]]["n_params"])
    return out


CASES = (("planted", "SETTLED WINS"),
         ("null", "NO DIFFERENCE"),
         ("noise", None),
         ("twin_offset", None))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=420.0)
    ap.add_argument("--no-bucket", action="store_true")
    a = ap.parse_args()

    print(f"=== M3 SYNTHETIC DRY-RUN  s={a.s} d={a.d} steps={a.steps} "
          f"n_train={a.n_train} n_eval={a.n_eval} seeds={a.seeds} "
          f"threads={torch.get_num_threads()} torch {torch.__version__} ===")

    print("\n=== SOFTMAX FIRST: the published reading, re-taken ===")
    got = reproduce_softmax()
    print(f"{'field':>14} {'published':>12} {'re-taken':>12} {'verdict':>12}")
    ok = True
    for field in ("n_params", "nrmse0_train", "nrmse0_eval", "train_nrmse",
                  "eval_nrmse", "ci_lo", "ci_hi"):
        w, g = PUBLISHED_SOFTMAX_2048[field], got[field]
        same = (w == g) if isinstance(w, int) else (f"{w:.6f}" == f"{g:.6f}")
        ok = ok and same
        ws = f"{w}" if isinstance(w, int) else f"{w:.6f}"
        gs = f"{g}" if isinstance(g, int) else f"{g:.6f}"
        print(f"{field:>14} {ws:>12} {gs:>12} {'MATCH' if same else 'DRIFT':>12}")
    if not ok:
        print("ABORT: the published softmax reading did not reproduce. Nothing "
              "downstream of this instrument may be credited.")
        return 1

    rows = []
    for arm, want in CASES:
        r = dry_run(arm=arm, s=a.s, d=a.d, steps=a.steps, n_train=a.n_train,
                    n_eval=a.n_eval, n_seeds=a.seeds,
                    bucket=not a.no_bucket, budget_s=a.budget)
        if "incomplete" in r:
            print(f"\n[{arm}] bucket not complete: {r['incomplete']}. "
                  "Re-run to continue; no verdict on a partial set.")
            return 3
        rows.append((arm, want, r))
        print(f"\n=== CASE {arm}  (correct answer: {want or 'not asserted'}) ===")
        print(f"  twin    NRMSE per seed: "
              f"{['%.6f' % v for v in r['twin_nrmse']]}")
        print(f"  settled NRMSE per seed: "
              f"{['%.6f' % v for v in r['settled_nrmse']]}")
        print(f"  0-step eval NRMSE twin/settled (gate needs >= 1.0): "
              f"{min(r['nrmse0_twin']):.6f} / {min(r['nrmse0_settled']):.6f}")
        print(f"  n_params twin={r['n_params_twin']} "
              f"settled={r['n_params_settled']}")
        print(f"  paired delta per seed: "
              f"{['%+.6f' % v for v in r['per_seed_delta']]}")
        print(f"  delta = {r['delta']:+.6f}  95% CI "
              f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]  B={r['n_boot']}")
        print(f"  VERDICT: {r['verdict']}"
              + ("" if want is None else
                 f"   [{'CORRECT' if r['verdict'] == want else 'WRONG'}]"))

    print("\n=== DRY-RUN SUMMARY ===")
    bad = 0
    for arm, want, r in rows:
        mark = "-" if want is None else \
            ("CORRECT" if r["verdict"] == want else "WRONG")
        bad += int(mark == "WRONG")
        print(f"  {arm:>12}  expected={str(want):>14}  got={r['verdict']:>14}"
              f"  {mark}")
    print(json.dumps({a_: r["verdict"] for a_, _w, r in rows}))
    if bad:
        print("THE HARNESS DID NOT RETURN A KNOWN ANSWER. A null result at "
              "iteration 12 would carry no information.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
