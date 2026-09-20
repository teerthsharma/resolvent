"""The RED test for the beta-HOMOTOPY leap: anneal beta 1 -> 0, deploy at beta = 0.

WHAT THIS FILE TESTS, AND IT IS NOT THE RETIRED RULE. The per-coordinate corner
rule `beta_d = 1 - alpha_d` is dead and retired at docs/CORNER_RULE_RETIREMENT.md;
nothing here re-opens it. beta is a SINGLE SCALAR shared by every coordinate in
this file, so no coordinate is ever assigned a corner. What is on trial is a
different claim about the same containment: that moving one global beta
CONTINUOUSLY from 1 to 0 across training, and shipping at exactly beta = 0, buys
something that a hard softmax-to-linear swap does not.

THE THREE ARMS, ALL EVALUATED AT beta = 0.

    A  "linear_only"  beta(t) = 0                       for every step
    B  "hard_swap"    beta(t) = 1 for t < steps//2, then 0
    C  "anneal"       beta(t) = 1 - t/(steps-1)         linear 1 -> 0

L-NULL, BY NAME. WHAT VARIES: the beta schedule, and only that. WHAT IS PINNED:
the seed, the initial weights (asserted identical across arms by digest, not
assumed -- see `pin_report`), the step count, the optimiser and its learning
rate, the batch contents, tau, lam/mu/nu, the latent width, the observation
width, S_LEN, HORIZON, the held-out bed, the evaluation beta (0 for every arm),
and the parameter count. An arm that differs in a second thing is not a schedule
ablation, so the difference is measured rather than intended. `hard_swap_refit`
varies a SECOND thing on purpose -- where the optimiser is rebuilt -- and is
therefore reported beside the verdict rather than inside it.

COST-MATCHED AT INFERENCE, AND THE HYPOTHESIS'S COST CLAIM IS FALSE. All arms
are read at beta = 0 for evaluation, so they are matched to each other. But the
hypothesis further claims the beta = 0 model is "plain linear attention with no
beta at inference and O(n) cost", AND THAT IS REFUTED HERE BY MEASUREMENT, see
`corner_cost` and `corner_rank`. beta enters ceq/arm_smprime.py:250 only as the
exponent of `mod.sum(-1)`, after :227 has already built an [n, n] exponential;
`Z**0 = 1` removes the DIVISION and nothing else. The beta = 0 corner is
`Real.exp (qk i j)` -- lean/CEQ/V16Domain.lean:387, whose own docstring at :384
calls it "the score with NO row normalizer" -- i.e. the unnormalised
exponential kernel, not a factorised `phi(q).phi(k)`. So the arms cost the same
as each other AND the same as softmax, and none of them is O(n).

THE CEILING, STATED AGAINST THIS FILE, TWICE. Externally: on 115,628 real
Lichess games every pi-JEPA arm lost to a "guess the most common square"
control at every context length (docs/CORNER_RULE_RETIREMENT.md, Kaggle
section). No beta schedule here touches that control. Internally: the reported
metric is NRMSE, so PREDICTING ZERO SCORES EXACTLY 1.0 by construction -- error
equal to the target's own RMS. An arm at or above 1.0 is beaten by a constant,
and several are. Read every margin below against that line, not against zero.

THE REUSE, AND WHY IT IS NOT A CONVENIENCE. `pi_jepa.fit` is called UNCHANGED.
beta rides in the `beta_assigned` buffer, which `PiJepa.beta_list()` re-reads on
every forward pass, and the schedule writes that buffer from inside the
`batches` callable that `fit` already calls FIRST on every step. So the collapse
detector, the EMA update, the stop-grad and the VICReg loss are the repo's own
and not a second copy of them that can drift from the original while this file
reports a comparison against it. The side effect is deliberate and is named in
`_schedule_then_batch` rather than hidden.

THE BEDS. `draw_bed` is pi_jepa's own synthetic AR(1) stream. `chess` is real
chess: seeded self-play games generated offline through `ceqjepa.beds.chess`'s
own `generate_selfplay_game` and encoded by its own `fen_to_vec`, so the board
is python-chess's and not a second notion of what a square holds. BOTH ARE RUN
AND BOTH ARE REPORTED, because the retirement card records one finding on which
the synthetic bed pointed the OPPOSITE way from real data.

WHAT THIS FILE CANNOT DO, AND IT IS RED RATHER THAN OMITTED. Real HUMAN games
are not on this machine. `tests/gate0/fixtures/games.pgn` is 36 games truncated
at 18 plies -- max 18 < S_LEN + HORIZON = 20 -- so it yields ZERO windows and
cannot be the bed; `human_pgn_windows` measures that rather than asserting it.
Self-play chess has real rules and real sequential structure but is not human
play, and every chess number below carries that limit.
"""
from __future__ import annotations

import hashlib
import random
from collections import OrderedDict

import numpy as np
import torch

from . import pi_jepa as pj
from .beds import chess as cb

__all__ = [
    "ARMS", "arm_beta", "schedule_table", "synthetic_bed", "chess_bed",
    "human_pgn_windows", "run_arm", "three_arms", "pin_report",
    "swap_recovery", "corner_cost", "corner_rank", "demo",
]

#: The three arms the hypothesis specified. The names are the report's row labels.
ARMS = ("linear_only", "hard_swap", "anneal")

#: A FOURTH ARM THAT EXISTS TO KEEP ARM B HONEST, and it is not optional.
#: `hard_swap` as specified keeps ONE optimiser across the swap, so Adam carries
#: its second-moment estimates through the jump. That is NOT what the prior art
#: does: T2R (arXiv 2103.13076 §2.2) calls its procedure "swap-then-finetune"
#: and finetunes as a SEPARATE STAGE, which means a fresh optimiser. Crediting
#: the anneal for beating a swap that no published method performs would be
#: defending the hypothesis rather than testing it, so `hard_swap_refit` runs
#: the same beta path as two fit() calls -- beta = 1 for the first half, then a
#: FRESH Adam at beta = 0 for the second -- and is reported beside arm B.
CONTROL_ARMS = ("hard_swap_refit",)
ALL_ARMS = ARMS + CONTROL_ARMS

#: Step budget. 600 rather than pi_jepa's 120 so the anneal has room to be a
#: path and not a jump: at 120 steps arm C moves beta by 1/119 per step, which
#: is already within a factor of two of arm B's single jump per-step size once
#: Adam's second moment has adapted. This is a CPU budget, not a scale claim.
STEPS = 600
BATCH = 64
SEEDS = (5501, 5502, 5503)

#: Held-out offset. The evaluation bed is drawn at seed + HELD_OUT_OFFSET, never
#: at the training seed, and for chess it is a DISJOINT SET OF GAMES.
HELD_OUT_OFFSET = 1

CHESS_X_DIM = cb.X_DIM            # 769: 12 piece planes over 64 squares + turn
CHESS_PLIES = pj.S_LEN + pj.HORIZON   # 20 plies per sequence


def arm_beta(arm, step, steps=STEPS):
    """The scalar beta arm `arm` reads at optimiser step `step`.

    All three agree at the LAST step: arm_beta(a, steps-1) == 0.0 for every a.
    That is the deployment condition and it is asserted in demo(), because an
    arm that ends anywhere else is not cost-matched with the others and its
    win would be a win for a more expensive model.
    """
    if arm == "linear_only":
        return 0.0
    if arm in ("hard_swap", "hard_swap_refit"):
        return 1.0 if step < steps // 2 else 0.0
    if arm == "anneal":
        return 1.0 - step / float(steps - 1)
    raise ValueError("unknown arm %r: the arms are %r" % (arm, ALL_ARMS))


def schedule_table(steps=STEPS):
    """beta at a few named steps, so the three schedules are readable as data."""
    marks = (0, steps // 4, steps // 2 - 1, steps // 2, 3 * steps // 4, steps - 1)
    return OrderedDict(
        (arm, OrderedDict((m, round(arm_beta(arm, m, steps), 6)) for m in marks))
        for arm in ARMS)


# ---------------------------------------------------------------------------
# THE BEDS
# ---------------------------------------------------------------------------

def synthetic_bed(seed, batch=BATCH):
    """pi_jepa's own AR(1) stream, in (context, target window, action) form."""
    x, a = pj.draw_bed(int(seed), batch, pj.S_LEN, pj.HORIZON, pj.X_DIM)
    ctx, tgt = pj._windows(x, pj.HORIZON, pj.S_LEN)
    return ctx, tgt, a


def _selfplay_sequences(seed, n_games, plies=CHESS_PLIES):
    """[n_games, plies, 769] of real board vectors from seeded self-play.

    Games shorter than `plies` are DROPPED rather than padded: a padded tail is
    a repeated terminal position, which is a constant the encoder can read off
    and would make the bed easier at exactly the positions the read's count
    dependence acts on. The draw continues until n_games survive, so the count
    is exact and the reader is told how many games were spent.
    """
    rng = random.Random(int(seed))
    out, spent = [], 0
    while len(out) < n_games:
        game = cb.generate_selfplay_game(rng, max_plies=plies + 4)
        spent += 1
        board = game.board()
        fens = [board.fen()]
        for mv in game.mainline_moves():
            board.push(mv)
            fens.append(board.fen())
        if len(fens) < plies:
            continue
        out.append(np.stack([cb.fen_to_vec(f) for f in fens[:plies]]))
    return torch.from_numpy(np.stack(out)).float(), spent


_CHESS_CACHE = {}


def chess_bed(seed, batch=BATCH):
    """Real-chess (context, target window, action=None). Action is None because
    this bed supplies no action and a fabricated one would be a second variable.

    Cached on (seed, batch). The draw is already deterministic -- random.Random
    is seeded and the move choice is its only consumer -- so the cache changes
    the cost and not the data; the `data_digest` in every row is what actually
    proves the arms saw the same games."""
    hit = _CHESS_CACHE.get((seed, batch))
    if hit is None:
        x, spent = _selfplay_sequences(seed, batch)
        hit = _CHESS_CACHE[(seed, batch)] = (x, spent)
    ctx, tgt = pj._windows(hit[0], pj.HORIZON, pj.S_LEN)
    return ctx, tgt, None


def human_pgn_windows(path="tests/gate0/fixtures/games.pgn"):
    """MEASURE what the repo's only real-human PGN can supply. Never assume it.

    Returns the game count, the ply histogram bounds and the number of windows
    of length S_LEN + HORIZON it yields. A zero here is the L-SURFACE RED for
    the human-data arm of this test: the check cannot run, so it is reported as
    unrun rather than skipped.
    """
    from ceq.kdata import iter_games, label_plies

    plies = [len(label_plies(g)) for _, g in iter_games(path)]
    need = pj.S_LEN + pj.HORIZON
    return dict(path=path, games=len(plies), total_plies=int(sum(plies)),
                min_plies=min(plies) if plies else 0,
                max_plies=max(plies) if plies else 0,
                need_plies=need,
                windows=int(sum(max(0, p - need + 1) for p in plies)))


BEDS = OrderedDict(synthetic=(synthetic_bed, pj.X_DIM),
                   chess=(chess_bed, CHESS_X_DIM))


# ---------------------------------------------------------------------------
# THE PINS: DIGESTS, SO "IDENTICAL" IS ASSERTED AND NOT INTENDED
# ---------------------------------------------------------------------------

def _digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        if t is None:
            h.update(b"<none>")
            continue
        t = t.detach().contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.to(torch.float64).numpy().tobytes())
    return h.hexdigest()[:16]


def _init_digest(model):
    """A digest of the INITIAL weights, excluding the beta buffer itself.

    beta_assigned is excluded on purpose: it is the variable. Every other
    parameter and buffer is included, so two arms with the same digest here
    differ in beta and in nothing else that the optimiser can see.
    """
    sd = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        if k == "beta_assigned":
            continue
        h.update(k.encode())
        h.update(sd[k].detach().to(torch.float64).numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# ONE ARM
# ---------------------------------------------------------------------------

def _schedule_then_batch(model, arm, steps, batches_by_step, seen, offset=0):
    """The `batches` callable fit() takes -- WITH A DELIBERATE SIDE EFFECT.

    It writes this step's beta into `model.beta_assigned` before returning the
    batch. That is legitimate and not a trick: fit() calls batches(step) FIRST
    on every step, and PiJepa.beta_list() re-reads the buffer on every forward,
    so the write lands for this step's read. Doing it this way means fit() --
    the collapse detector, the EMA, the stop-grad, the loss -- is the repo's
    own code and not a fork of it that reports against its own drift.

    `seen` collects the digest of every batch actually handed over, so the claim
    that the three arms saw the same data IN THE SAME ORDER is a measurement.
    """
    def batches(step):
        # `offset` maps a STAGE-LOCAL step back onto the GLOBAL schedule, so a
        # two-stage arm walks the same beta path as a one-stage arm and the two
        # differ only in where the optimiser is rebuilt.
        g = step + offset
        with torch.no_grad():
            model.beta_assigned.fill_(arm_beta(arm, g, steps))
        ctx, tgt, a = batches_by_step[g]
        seen.append(_digest(ctx, tgt, a if a is None else a.float()))
        return ctx, tgt, a
    return batches


@torch.no_grad()
def _eval_at_zero(model, ctx, tgt, a):
    """Held-out latent NRMSE READ AT beta = 0, whatever the arm trained at.

    NRMSE and not raw MSE, for pi_jepa's own reason at :972 -- a JEPA's target
    is the encoder's own output, so an encoder with small outputs is trivially
    easy to predict and ranking arms by absolute MSE ranks them by output
    scale. The target RMS is returned beside it so the correction is visible
    rather than taken on trust.
    """
    model.beta_assigned.fill_(0.0)
    ro = model(ctx, a)
    s_hat = ro.out[:, -1, :]
    s_tgt = model.online(tgt)[:, -1, :][:, model.cols.tolist()]
    mse = (s_hat - s_tgt).pow(2).sum(-1).mean()
    scale = s_tgt.pow(2).sum(-1).mean()
    return dict(mse=float(mse), nrmse=float((mse / scale).sqrt()),
                target_rms=float(scale.sqrt()))


def run_arm(arm, seed, bed="synthetic", steps=STEPS, batch=BATCH, lr=pj.LR):
    """Train one arm and score it at beta = 0. Returns the row plus its pins."""
    draw, x_dim = BEDS[bed]
    # THE DATA IS DRAWN ONCE, BEFORE THE MODEL, AND REPLAYED BY STEP INDEX.
    # Drawing inside the loop would make the stream depend on how many times the
    # RNG was touched, which the beta schedule does not change today but could
    # after any edit -- so the pin is structural, not a promise.
    train = draw(seed, batch)
    batches_by_step = [train] * steps
    held = draw(seed + HELD_OUT_OFFSET, batch)

    model = pj.build(seed, beta=[0.0] * pj.D_LATENT, x_dim=x_dim)
    init = _init_digest(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())

    seen = []
    row = dict(arm=arm, seed=seed, bed=bed, steps=steps,
               n_params=n_params, n_trainable=n_train,
               init_digest=init, data_digest=_digest(*train[:2]),
               beta_at_last_step=arm_beta(arm, steps - 1, steps),
               diverged=False, diverged_why=None)

    def _fit(n_steps, offset=0):
        """One fit() stage. A stage boundary is a FRESH Adam, which is what
        `hard_swap_refit` needs and what `fit` gives by construction."""
        return pj.fit(model,
                      batches=_schedule_then_batch(model, arm, steps,
                                                   batches_by_step, seen,
                                                   offset=offset),
                      steps=n_steps, lr=lr, seed=seed, detector=True)

    # A DIVERGED ARM IS A RESULT AND IS REPORTED AS ONE. It is not retried at a
    # kinder learning rate: tuning an arm until it survives, while its rivals
    # keep the original setting, is the unmatched comparison this project's own
    # MATCHED COUNTS rule forbids. NRMSE is set to inf so the arm simply loses.
    try:
        if arm == "hard_swap_refit":
            half = steps // 2
            a_res = _fit(half, offset=0)                 # beta = 1 stage
            res = _fit(steps - half, offset=half)        # fresh Adam, beta = 0
            res["loss_first"] = a_res["loss_first"]
            res["erank_min"] = min(res["erank_min"], a_res["erank_min"])
        else:
            res = _fit(steps, offset=0)
    except (RuntimeError, AssertionError, torch._C._LinAlgError) as exc:
        row.update(diverged=True, diverged_why="%s: %s" % (type(exc).__name__,
                                                           str(exc)[:160]),
                   loss_first=float("nan"), loss_last=float("inf"),
                   erank_min=float("nan"), std_min_last=float("nan"),
                   collapsed_at=None, mse=float("inf"), nrmse=float("inf"),
                   target_rms=float("nan"),
                   order_digest=hashlib.sha256("".join(seen).encode()).hexdigest()[:16],
                   steps_seen=len(seen))
        return row

    # THE EVAL IS GUARDED TOO, and that is not belt-and-braces. An arm can
    # finish fit() without raising and still carry weights whose beta = 0 read
    # overflows -- pi_jepa's own forward() then refuses it at :879 for a
    # non-finite imaginary part. That refusal IS the deployment failure this
    # test is about, so it is recorded as a diverged arm rather than crashing
    # the sweep and losing the eleven rows that did survive.
    try:
        score = _eval_at_zero(model, *held)
    except (RuntimeError, AssertionError, torch._C._LinAlgError) as exc:
        score = dict(mse=float("inf"), nrmse=float("inf"), target_rms=float("nan"))
        row.update(diverged=True,
                   diverged_why="at eval: %s: %s" % (type(exc).__name__,
                                                     str(exc)[:120]))
    if not np.isfinite(score["nrmse"]):
        row.update(diverged=True, diverged_why=row["diverged_why"]
                   or "held-out read is non-finite at beta = 0")
        score["nrmse"] = float("inf")
    row.update(loss_first=res["loss_first"], loss_last=res["loss_last"],
               erank_min=res["erank_min"], std_min_last=res["std_min_last"],
               collapsed_at=res["collapsed_at"],
               order_digest=hashlib.sha256("".join(seen).encode()).hexdigest()[:16],
               steps_seen=len(seen), **score)
    return row


# ---------------------------------------------------------------------------
# THE THREE ARMS, AND THE PIN REPORT THAT LICENSES COMPARING THEM
# ---------------------------------------------------------------------------

def three_arms(bed="synthetic", seeds=SEEDS, steps=STEPS, batch=BATCH,
               arms=ALL_ARMS):
    """Every arm at every seed. The rows are the report."""
    return [run_arm(arm, seed, bed=bed, steps=steps, batch=batch)
            for seed in seeds for arm in arms]


def pin_report(rows):
    """L-NULL, MEASURED. Per seed: do the arms share init, data, width and endpoint?

    WHAT IS COMPARED AND WHY IT IS NOT THE ORDER DIGEST. Each run replays ONE
    batch for every step (`batches_by_step = [train] * steps`), so the batch
    SEQUENCE carries no information beyond its length and two arms can only
    differ in it by one of them stopping early. An arm that diverges does stop
    early, so an order-digest equality test reports a data difference where the
    only difference is that one arm died -- which is a defect in the metric, not
    a second variable. The pin is therefore on `init_digest` (the weights before
    any step), `data_digest` (the batch itself), the parameter count and the
    final beta; truncation is reported separately, by name, in `truncated`.
    """
    out = OrderedDict()
    for seed in sorted({r["seed"] for r in rows}):
        grp = [r for r in rows if r["seed"] == seed]
        inits = {r["init_digest"] for r in grp}
        datas = {r["data_digest"] for r in grp}
        betas = {r["beta_at_last_step"] for r in grp}
        params = {r["n_params"] for r in grp}
        full = {r["steps_seen"] for r in grp if not r.get("diverged")}
        out[seed] = dict(arms=len(grp), init=sorted(inits), data=sorted(datas),
                         order=sorted({r["order_digest"] for r in grp}),
                         n_params=sorted(params), final_beta=sorted(betas),
                         steps_seen=sorted({r["steps_seen"] for r in grp}),
                         truncated=sorted(r["arm"] for r in grp
                                          if r.get("diverged")),
                         pinned=(len(inits) == 1 and len(datas) == 1
                                 and len(params) == 1 and betas == {0.0}
                                 and len(full) <= 1))
    return out


def interval(rows, arm, key="nrmse"):
    """The seed spread of one arm as an INTERVAL, plus its width. 'Beats' in this
    file means beyond this interval, and the number is printed so it can be
    checked rather than believed."""
    vals = sorted(r[key] for r in rows if r["arm"] == arm)
    return dict(arm=arm, n=len(vals), lo=vals[0], hi=vals[-1],
                mean=sum(vals) / len(vals), width=vals[-1] - vals[0],
                values=vals)


def verdict(rows, key="nrmse"):
    """BOUND or REFUTED, and which death if it died.

    The hypothesis is refuted unless `anneal` beats BOTH other arms by more than
    the seed spread. Lower NRMSE is better. 'Beats X' is `anneal.hi < X.lo`:
    the annealed arm's WORST seed must be better than the other arm's BEST, so
    the win survives the spread rather than being inside it.

    THE TWO DEATHS ARE NAMED SEPARATELY AND THE SHARPER ONE IS NOT SOFTENED.
    'hard_swap_already_gives_it' means anneal beat linear_only but NOT hard_swap:
    the swap already buys the benefit, continuity bought nothing, and the
    operator is a one-file convenience rather than an architecture.
    """
    present = [a for a in ALL_ARMS if any(r["arm"] == a for r in rows)]
    iv = {a: interval(rows, a, key) for a in present}
    an, lo_arm, sw = iv["anneal"], iv["linear_only"], iv["hard_swap"]
    beats_a = an["hi"] < lo_arm["lo"]
    beats_b = an["hi"] < sw["lo"]
    if beats_a and beats_b:
        v, death = "BOUND", None
    elif beats_a and not beats_b:
        v, death = "REFUTED", "hard_swap_already_gives_it"
    else:
        v, death = "REFUTED", "anneal_does_not_beat_plain_linear"
    out = dict(verdict=v, death=death, key=key, intervals=iv,
               anneal_beats_linear_only=beats_a, anneal_beats_hard_swap=beats_b,
               margin_vs_linear_only=lo_arm["lo"] - an["hi"],
               margin_vs_hard_swap=sw["lo"] - an["hi"],
               diverged=sorted({(r["arm"], r["seed"]) for r in rows
                                if r.get("diverged")}),
               # NRMSE = 1.0 IS the predict-zero control, by construction. Any
               # arm-seed at or above it was beaten by a constant, and saying
               # which ones is what keeps a 0.88-vs-0.94 margin honest.
               predict_zero_nrmse=1.0,
               beaten_by_constant=sorted((r["arm"], r["seed"]) for r in rows
                                         if r[key] >= 1.0))
    if "hard_swap_refit" in iv:
        # Reported SEPARATELY from the verdict, because the verdict is on the
        # test the hypothesis specified. This says whether a win over arm B
        # survives giving arm B the finetuning stage the prior art actually uses.
        rf = iv["hard_swap_refit"]
        out.update(anneal_beats_hard_swap_refit=an["hi"] < rf["lo"],
                   margin_vs_hard_swap_refit=rf["lo"] - an["hi"])
    return out


# ---------------------------------------------------------------------------
# THE MECHANISM THE HYPOTHESIS OFFERED, MEASURED DIRECTLY
# ---------------------------------------------------------------------------

def swap_recovery(seed=SEEDS[0], bed="synthetic", steps=STEPS, batch=BATCH,
                  window=40):
    """THE DISCONTINUITY THE HYPOTHESIS SAYS THE HARD SWAP CROSSES, in steps.

    The claim is mechanistic: a hard beta 1 -> 0 swap crosses a discontinuity
    and a continuous anneal does not. If that is the mechanism, arm B's loss
    must jump at the swap step and take a visible number of steps to come back.
    This runs arm B, reads its own loss trace around step steps//2, and reports
    the jump and how many steps it needs to return under its pre-swap level.

    A recovery inside a handful of steps means the 'discontinuity' is a
    transient the optimiser absorbs, and continuity is buying a few steps of
    budget rather than an architecture.
    """
    draw, x_dim = BEDS[bed]
    train = draw(seed, batch)
    model = pj.build(seed, beta=[0.0] * pj.D_LATENT, x_dim=x_dim)
    seen = []
    res = pj.fit(model, batches=_schedule_then_batch(model, "hard_swap", steps,
                                                     [train] * steps, seen),
                 steps=steps, lr=pj.LR, seed=seed, detector=True)
    tr, k = res["loss_trace"], steps // 2
    pre = tr[k - 1]
    post = tr[k:k + window]
    back = next((i for i, v in enumerate(post) if v <= pre), None)
    return dict(seed=seed, bed=bed, swap_step=k, loss_before_swap=pre,
                loss_at_swap=tr[k], jump_ratio=tr[k] / pre if pre else float("nan"),
                peak_after=max(post), steps_to_recover=back,
                recovered_within=window if back is not None else None,
                loss_last=tr[-1])


# ---------------------------------------------------------------------------
# THE DEPLOYMENT PREMISE: IS THE beta = 0 CORNER ACTUALLY O(n)?
# ---------------------------------------------------------------------------

def corner_cost(ns=(32, 64, 128, 256, 512), dk=pj.DK, d=pj.D_LATENT):
    """What the shipped read MATERIALISES at each beta, as a function of n.

    The hypothesis promises that shipping at beta = 0 gives 'plain linear
    attention with no beta at inference and O(n) cost'. beta enters
    ceq/arm_smprime.py:250 ONLY as the exponent of `mod.sum(-1)`, AFTER
    `numerator` at :227 has already built `e = torch.exp(w.masked_fill(up, NEG))`
    from `w = q @ k.transpose(-2,-1)`. That is an [n, n] tensor at every beta.
    So this measures `num.numel()` against n: an O(n) operator cannot have an
    n**2 intermediate, and the count is a fact about the tensor rather than a
    timing that a busy machine could blur.
    """
    import ceq.arm_smprime as arm
    rows = []
    for n in ns:
        g = torch.Generator().manual_seed(pj.SEED)
        q = torch.randn(1, n, dk, generator=g, dtype=torch.float64)
        k = torch.randn(1, n, dk, generator=g, dtype=torch.float64)
        v = torch.randn(1, n, d, generator=g, dtype=torch.float64)
        num, mod = arm.numerator(q, k)
        o0 = pj.read_per_coordinate(q, k, v, [0.0] * d).out
        o1 = pj.read_per_coordinate(q, k, v, [1.0] * d).out
        rows.append(dict(n=n, num_shape=tuple(num.shape),
                         num_numel=int(num.numel()),
                         numel_over_n=num.numel() / n,
                         numel_over_n2=num.numel() / n ** 2,
                         beta0_equals_beta1=bool(torch.equal(o0, o1)),
                         beta0_rms=float(o0.real.pow(2).mean().sqrt()),
                         beta1_rms=float(o1.real.pow(2).mean().sqrt())))
    return rows


def corner_rank(ns=(8, 16, 32, 64, 128, 256), dk=pj.DK, tol=1e-10):
    """WHETHER AN O(n) FORM COULD EXIST AT ALL, which is the sharper question.

    An O(n) linear attention needs the score to be an inner product of FINITE
    feature maps, `phi(q_i) . phi(k_j)` with phi into R^r, because that is what
    lets the row become a running r-by-d state instead of a matrix. A score of
    that form has matrix rank at most r FOR EVERY n. The beta = 0 corner is
    `Real.exp (qk i j)` -- lean/CEQ/V16Domain.lean:387, and its own docstring at
    :384 calls it 'the score with NO row normalizer'. This measures the rank of
    that score matrix as n grows at FIXED dk. Rank that keeps tracking n is a
    measurement that no finite phi exists, so the cost is not an artefact of how
    ceq/arm_smprime.py happens to be written.
    """
    rows = []
    for n in ns:
        g = torch.Generator().manual_seed(pj.SEED)
        q = torch.randn(n, dk, generator=g, dtype=torch.float64)
        k = torch.randn(n, dk, generator=g, dtype=torch.float64)
        w = torch.exp((q @ k.T) / dk ** 0.5)     # the unmasked exp kernel
        r = int(torch.linalg.matrix_rank(w, tol=tol))
        rows.append(dict(n=n, dk=dk, rank=r, rank_equals_n=(r == n),
                         rank_over_dk=r / dk))
    return rows


# ---------------------------------------------------------------------------
# THE SELF-CHECK
# ---------------------------------------------------------------------------

def demo():
    """The smallest thing that fails if the schedule logic breaks."""
    # 1. Every arm ships at beta = 0. An arm that does not is not cost-matched.
    for a in ARMS:
        assert arm_beta(a, STEPS - 1, STEPS) == 0.0, (a, "does not ship at 0")
    # 2. The arms are genuinely different paths, not three names for one.
    mid = [arm_beta(a, STEPS // 3, STEPS) for a in ARMS]
    assert len(set(mid)) == 3, ("the arms coincide at steps//3", mid)
    # 3. hard_swap is a step and anneal is not: the jump sizes differ by >100x.
    k = STEPS // 2
    swap_jump = abs(arm_beta("hard_swap", k, STEPS) - arm_beta("hard_swap", k - 1, STEPS))
    anneal_jump = abs(arm_beta("anneal", k, STEPS) - arm_beta("anneal", k - 1, STEPS))
    assert swap_jump == 1.0 and anneal_jump < 0.01, (swap_jump, anneal_jump)
    assert swap_jump / anneal_jump > 100, swap_jump / anneal_jump
    # 4. The buffer write actually reaches the read. This is the one wiring
    #    claim the whole file rests on, so it is measured, not trusted.
    m = pj.build(SEEDS[0], beta=[0.0] * pj.D_LATENT, x_dim=pj.X_DIM)
    ctx, _, a = synthetic_bed(SEEDS[0], 8)
    with torch.no_grad():
        m.beta_assigned.fill_(0.0)
        at0 = m(ctx, a).out.clone()
        m.beta_assigned.fill_(1.0)
        at1 = m(ctx, a).out.clone()
    assert not torch.equal(at0, at1), "beta_assigned does not reach the read"
    assert m.beta_list() == [1.0] * pj.D_LATENT, m.beta_list()
    # 5. A mid-anneal beta is strictly between the corners at the read, so the
    #    path is a path and not two endpoints with nothing in between.
    with torch.no_grad():
        m.beta_assigned.fill_(0.5)
        mid_out = m(ctx, a).out
    lo, hi = at0.abs().mean().item(), at1.abs().mean().item()
    assert min(lo, hi) < mid_out.abs().mean().item() < max(lo, hi), (lo, hi)
    # 6. The human PGN really is unusable: RED asserted, not assumed.
    hp = human_pgn_windows()
    assert hp["windows"] == 0 and hp["max_plies"] < hp["need_plies"], hp
    # 7. THE DEPLOYMENT PREMISE. beta = 0 must not be cheaper to materialise
    #    than beta = 1, and the score's rank must keep tracking n -- both are
    #    what make 'O(n) at beta = 0' false. Asserted here so a future edit that
    #    quietly made the corner factorisable would fail this file loudly.
    cc = corner_cost(ns=(32, 64, 128))
    assert all(r["num_numel"] == r["n"] ** 2 for r in cc), cc
    assert all(r["numel_over_n2"] == 1.0 for r in cc), cc
    cr = corner_rank(ns=(16, 64, 128))
    assert all(r["rank"] == r["n"] for r in cr), cr
    assert cr[-1]["rank"] > 4 * pj.DK, cr[-1]
    print("beta_homotopy demo: OK")
    print("  schedules:", dict(schedule_table()))
    print("  human pgn:", hp)
    print("  corner cost:", [(r["n"], r["num_numel"]) for r in cc])
    print("  corner rank:", [(r["n"], r["rank"]) for r in cr])


if __name__ == "__main__":
    demo()
