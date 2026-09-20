"""ceqjepa/causal_eval.py -- the outcome-perplexity gap harness for the causal claim.

WHAT IS BEING MEASURED. The mechanism claims to predict what an intervention
DOES, not merely to notice that one occurred. The empirical proxy (Richens &
Everitt, ICLR 2024, arXiv:2402.10877: a regret bound under a large class of
distributional shifts implies an approximate causal model; a do() is such a
shift) is the pair of outcome perplexities

    PPL_obs -- positions scored as they actually occurred
    PPL_do  -- positions scored after a move is FORCED, against the outcome
               that actually followed the forcing (for chess this is a real
               oracle: python-chess replays the forced move and finishes the
               game, so k_star_do is a fact, not a model output --
               ceqjepa.beds.chess_do)
    gap     = PPL_do - PPL_obs

=====================================================================
A LOW GAP IS MEANINGLESS UNLESS PPL_obs IS ITSELF GOOD.
A uniformly terrible model has a SMALL GAP and has learned NOTHING:
it was never on target, so there is nowhere for it to fall to. The
constant predictor in demo() case (3) posts gap = +0.000000 exactly
-- the same gap as the perfect-mechanism predictor beside it -- and
the two are separated ONLY by PPL_obs (2.0000 vs 1.1111). Gap is
never reported, returned, or read in isolation: (PPL_obs, gap)
always ship together, with the bootstrap SE of each.
=====================================================================

THE CONTROLS ARE THE ENTIRE POINT. A gap of +0.4 means nothing on its own;
it means something only against what these two post on the SAME draw:

  IGNORE-THE-INTERVENTION (the bar): predict that the interventional
    committor equals the observational one. This is exactly, mechanically,
    what a correlational model does -- its read is a function of the input,
    and the input did not change when the move was forced. Call it as
    causal_gap(f, f, bed), or via ignore_intervention_gap(f, bed). A model
    that cannot beat this has not been shown to represent the intervention.
  MARGINAL: predict the training-set outcome frequency regardless of
    position -- marginal_predictor(k_star_train, K). Its gap is ~0 by
    construction, which is the boxed warning above made concrete.

STANDARD ERROR, ON EVERY GAP. Every earlier number in this project was
reported without one and several were noise-dominated (the operator-vs-MLP
separation was +0.0264 with sd 0.0358). causal_gap bootstraps over eval
items and PAIRS the resample -- both arms are scored on the SAME resampled
indices, because the two arms are the same positions and an unpaired
bootstrap inflates the gap's SE by up to sqrt(2). demo() case (5) plants
that exact negative: a bed where the do-arm is identical to the obs-arm
must return se_gap = 0.0 EXACTLY, which an unpaired implementation cannot.

CALIBRATION, because perplexity hides it. The committor is a probability
(operator.committor attains 0 and 1 exactly), so a systematically
over-confident but well-RANKED predictor can post a decent PPL.
calibration_report gives one-vs-rest reliability buckets plus ECE.

SIBLING MODULES, read from source, not assumed:
  ceqjepa.operator.committor(P, absorbing_idx) -> q [..., n, k]  (exact solve)
  ceqjepa.intervene.committor_do(P, absorbing_idx, i, row) -> (q_do [n,k], den)
      NOTE: single [n,n] P only (intervene._blocks raises on a batched P), and
      den = (1 - P'[i,i]) / (1 - P[i,i]). committor_do_predictions() below is
      the only place this file touches that module.
  ceqjepa.beds.chess_do.build_intervention_dataset(...) -> [InterventionSample]
      obs_outcome is an EXACT one-hot; do_outcome_mean is an average over R
      rollouts, i.e. a SAMPLE of the post-intervention distribution and not
      the distribution (R=4 by default). To score a bed like that here, expand
      each position into its R realized rollouts as R separate hard k_star
      items -- that is exactly cross-entropy against the empirical
      interventional distribution, and it keeps the rollout noise visible in
      the bootstrap SE instead of hiding it inside a soft target.
      CAVEAT ON THAT ROUTE: R rollouts of the SAME position are correlated,
      and the item bootstrap below assumes independent draws. Expanding them
      to R items therefore reports an SE that is too SMALL, by up to sqrt(R)
      if the rollouts of a position agree with each other. Bootstrap over
      POSITIONS (resample whole positions, carrying their R rollouts) if that
      matters; this module resamples the items it is handed and cannot see
      which of them came from one board.

torch only. No numpy, no python-chess: this file never touches a board.
"""
import math

import torch
import torch.nn.functional as F

import ceqjepa.operator as op

__all__ = ["chance_level", "outcome_ppl", "causal_gap", "ignore_intervention_gap",
           "marginal_predictor", "committor_do_predictions", "calibration_report"]

#: Lower clamp on the predicted probability of the REALIZED class before its
#: log. Only a lower clamp is needed: log(0) = -inf is the one failure mode,
#: and committor() attains 0 exactly, so an unclamped mean goes to +inf on a
#: single confidently-wrong example. At 1e-6 one such example costs
#: log(1/eps) = 13.8 nats, which is visible without being fatal.
EPS_Q = 1e-6


def chance_level(n_outcomes):
    """PPL of the uniform 1/K predictor: exp(-log(1/K)) = K. 1.0 is perfect,
    K is no information. Chess win/draw/loss (K=3) -> 3.0."""
    return float(n_outcomes)


def _as_probs(q_pred):
    if torch.is_tensor(q_pred):
        return q_pred
    return torch.as_tensor(q_pred, dtype=torch.get_default_dtype())


def _logp(q_pred, k_star, eps=EPS_Q):
    """Per-item log probability of the realized class, [N]. The single
    quantity both outcome_ppl and the bootstrap are built from, so the point
    estimate and its SE cannot drift apart."""
    q_pred = _as_probs(q_pred)
    k_star = torch.as_tensor(k_star, dtype=torch.long, device=q_pred.device)
    if q_pred.dim() != 2:
        raise ValueError("q_pred must be [N, K], got %s" % (tuple(q_pred.shape),))
    if k_star.shape[0] != q_pred.shape[0]:
        raise ValueError("q_pred has %d rows but k_star has %d entries"
                         % (q_pred.shape[0], k_star.shape[0]))
    if int(k_star.max()) >= q_pred.shape[-1] or int(k_star.min()) < 0:
        raise ValueError("k_star out of range for K = %d outcomes" % q_pred.shape[-1])
    p = q_pred.gather(-1, k_star.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
    return torch.log(p)


def outcome_ppl(q_pred, k_star, eps=EPS_Q, verbose=False):
    """exp of the negative mean log predicted probability of the set actually
    reached: PPL = exp(-mean_i log q_pred[i, k_star[i]]).

    q_pred: [N, K] probabilities over the K absorbing sets (a committor, rows
    summing to 1 -- never logits). k_star: [N] long, the set actually reached.
    1.0 is perfect; chance_level(K) = K is no information; above K is worse
    than guessing. verbose=True prints the chance level alongside, so the
    number is readable against its own scale.
    """
    lp = _logp(q_pred, k_star, eps=eps)
    ppl = float(torch.exp(-lp.mean()))
    if verbose:
        k = _as_probs(q_pred).shape[-1]
        print("[causal_eval] outcome_ppl over %d items, K = %d absorbing sets: "
              "PPL = %.6f  (chance level, the uniform 1/%d predictor = %.4f; "
              "1.0 is perfect; clamp eps = %g)"
              % (lp.numel(), k, ppl, k, chance_level(k), eps))
    return ppl


def _bootstrap(lp_obs, lp_do, n_boot, seed):
    """Paired item bootstrap. Returns (se_obs, se_do, se_gap).

    PAIRED: one index draw scores BOTH arms, because the two arms are the same
    eval positions. Resampling them independently would inflate se_gap by up
    to sqrt(2) and would break the planted negative in demo() case (5).
    """
    n = lp_obs.numel()
    if n_boot <= 0 or n < 2:
        nan = float("nan")
        return nan, nan, nan
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    # ponytail: materialises [n_boot, n] indices at once. n_boot*n <= a few
    # million is nothing; chunk over n_boot if an eval ever gets big enough
    # to care.
    idx = torch.randint(0, n, (int(n_boot), n), generator=g)
    ppl_obs_b = torch.exp(-lp_obs.cpu()[idx].mean(dim=1))
    ppl_do_b = torch.exp(-lp_do.cpu()[idx].mean(dim=1))
    gap_b = ppl_do_b - ppl_obs_b
    return (float(ppl_obs_b.std(unbiased=True)),
            float(ppl_do_b.std(unbiased=True)),
            float(gap_b.std(unbiased=True)))


def _field(bed, name):
    v = bed.get(name) if isinstance(bed, dict) else getattr(bed, name, None)
    if v is None:
        raise ValueError(
            "causal_eval: the bed does not supply '%s'. A bed for this harness "
            "must expose k_star_obs [N] (the set actually reached) and "
            "k_star_do [N] (the set reached after the move was FORCED and the "
            "game played out) as attributes or dict keys, index-aligned: item i "
            "of one arm is the SAME position as item i of the other, which is "
            "what makes the paired bootstrap valid." % name)
    return torch.as_tensor(v, dtype=torch.long)


def causal_gap(predict_obs, predict_do, bed, n_boot=1000, seed=0, eps=EPS_Q,
               label="model", verbose=True):
    """PPL_obs, PPL_do and gap = PPL_do - PPL_obs, each with a bootstrap SE.

    predict_obs(bed) -> [N, K] probabilities for the observational arm.
    predict_do(bed)  -> [N, K] probabilities for the interventional arm.
    bed supplies k_star_obs [N] and k_star_do [N] (see _field).

    Passing the SAME callable twice IS the ignore-the-intervention control --
    see ignore_intervention_gap. A returned gap is never meaningful without
    the ppl_obs beside it in the same dict; both are always present, and
    verbose=True (the default) prints them together with the SE and the
    chance level, because a gap printed alone is the failure this module
    exists to prevent.
    """
    k_obs, k_do = _field(bed, "k_star_obs"), _field(bed, "k_star_do")
    q_obs, q_do = _as_probs(predict_obs(bed)), _as_probs(predict_do(bed))
    if q_obs.shape != q_do.shape:
        raise ValueError("the two arms returned different shapes: obs %s vs do %s"
                         % (tuple(q_obs.shape), tuple(q_do.shape)))
    lp_obs, lp_do = _logp(q_obs, k_obs, eps=eps), _logp(q_do, k_do, eps=eps)
    ppl_obs, ppl_do = float(torch.exp(-lp_obs.mean())), float(torch.exp(-lp_do.mean()))
    se_obs, se_do, se_gap = _bootstrap(lp_obs.detach(), lp_do.detach(), n_boot, seed)
    out = dict(label=label, n=int(lp_obs.numel()), n_outcomes=int(q_obs.shape[-1]),
               chance=chance_level(q_obs.shape[-1]),
               ppl_obs=ppl_obs, ppl_do=ppl_do, gap=ppl_do - ppl_obs,
               se_ppl_obs=se_obs, se_ppl_do=se_do, se_gap=se_gap, n_boot=int(n_boot))
    if verbose:
        print(format_gap(out))
    return out


def format_gap(r):
    """One line, gap never without its ppl_obs and its SE."""
    return ("[causal_eval] %-22s n=%-5d PPL_obs=%.4f+-%.4f  PPL_do=%.4f+-%.4f  "
            "gap=%+.4f+-%.4f  (chance=%.2f, perfect=1.00, %d bootstrap draws)"
            % (r["label"], r["n"], r["ppl_obs"], r["se_ppl_obs"], r["ppl_do"],
               r["se_ppl_do"], r["gap"], r["se_gap"], r["chance"], r["n_boot"]))


def ignore_intervention_gap(predict_obs, bed, **kw):
    """THE BAR. Score the interventional arm with the OBSERVATIONAL prediction:
    predict q(do a) == q(obs). This is what a correlational model does -- the
    input did not change when the move was forced, so neither did its read.
    Any claim that the mechanism represents the intervention is a claim that
    it beats THIS number, not that its own gap is small."""
    kw.setdefault("label", "IGNORE-INTERVENTION")
    return causal_gap(predict_obs, predict_obs, bed, **kw)


def marginal_predictor(k_star_train, n_outcomes):
    """Predict the training-set outcome frequency, regardless of position.
    Returns a callable(bed) -> [N, K], usable as either arm of causal_gap.
    Its gap is ~0 by construction and it has learned nothing: the module
    docstring's warning, available as a runnable control."""
    k = torch.as_tensor(k_star_train, dtype=torch.long)
    freq = torch.bincount(k, minlength=int(n_outcomes)).to(torch.float64)
    freq = freq / freq.sum()

    def predict(bed):
        n = _field(bed, "k_star_obs").shape[0]
        return freq.unsqueeze(0).expand(n, -1)
    return predict


def committor_do_predictions(P, absorbing_idx, alpha, row_idx, rows,
                             intervene_module=None):
    """The do-arm read of a built operator: for each item, clamp its row and
    mix the resulting committor field with that item's OWN attention weights.

    P: [N, n, n] built operators (boundary rows already identity).
    alpha: [N, n] the attention weights the observational read used -- passed
      in UNCHANGED, because alpha is a function of the encoder output and the
      encoder never saw the intervention. Holding it fixed is what makes the
      rank-1 edit the only thing that can move the prediction.
    row_idx: [N] long, the transient row do(a) clamps. rows: [N, n] the
      row-stochastic vector it is clamped to.
    Returns (q_do [N, K], den [N]); den = (1-P'_ii)/(1-P_ii), the
    Sherman-Morrison denominator, so a caller can see how near-singular any
    item's edit came. Raises through intervene's own guards -- a
    SingularTransientBlockError or an acausal-row ValueError is the finding,
    never something to catch and average over.
    """
    if intervene_module is None:
        import ceqjepa.intervene as intervene_module
    fn = getattr(intervene_module, "committor_do", None)
    if fn is None:
        raise RuntimeError(
            "causal_eval: %r does not expose committor_do(P, absorbing_idx, i, "
            "row) -> (q_do, den). Refusing to fabricate the do-arm prediction "
            "from an unread API." % (intervene_module,))
    # ponytail: one factorisation per ITEM. intervene.committor_do_batch
    # amortises across m candidate moves at ONE position, which is a different
    # axis than this one; N separate positions means N solves either way.
    qs, dens = [], []
    for b in range(P.shape[0]):
        q_do, den = fn(P[b], absorbing_idx, int(row_idx[b]), rows[b])
        qs.append(q_do)
        dens.append(float(den))
    q_field = torch.stack(qs)                                    # [N, n, K]
    return (torch.einsum("bn,bnk->bk", alpha.to(q_field.dtype), q_field),
            torch.tensor(dens, dtype=q_field.dtype))


def calibration_report(q_pred, k_star, n_bins=10):
    """One-vs-rest reliability buckets. Every (item, class) pair is one point
    (predicted probability q_pred[i,k], label 1{k_star[i]==k}), binned into
    n_bins equal-width buckets of predicted probability. Per bin: count, mean
    predicted probability, empirical frequency. A calibrated committor has
    mean_pred ~= empirical_freq in every non-empty bin.

    Also returns ECE, the count-weighted mean |mean_pred - empirical_freq|
    over non-empty bins -- the scalar perplexity hides, since PPL is a log
    score a systematically over-confident but well-ranked predictor still
    scores decently on.
    """
    q_pred = _as_probs(q_pred)
    k_star = torch.as_tensor(k_star, dtype=torch.long)
    _, K = q_pred.shape
    probs = q_pred.reshape(-1)
    labels = F.one_hot(k_star, num_classes=K).to(q_pred.dtype).reshape(-1)
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    total = probs.numel()
    bins, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        last = i == n_bins - 1
        mask = (probs >= lo) & ((probs <= hi) if last else (probs < hi))
        count = int(mask.sum())
        if count == 0:
            bins.append(dict(lo=lo, hi=hi, count=0, mean_pred=float("nan"),
                             empirical_freq=float("nan")))
            continue
        mean_pred, emp = float(probs[mask].mean()), float(labels[mask].mean())
        bins.append(dict(lo=lo, hi=hi, count=count, mean_pred=mean_pred,
                         empirical_freq=emp))
        ece += (count / total) * abs(mean_pred - emp)
    return dict(bins=bins, ece=ece, n_points=total)


# ---------------------------------------------------------------------------
# Self-check. Every case hand-computable; every printed number is RUN.
# ---------------------------------------------------------------------------
def _const_predictor(q_row, n):
    q = torch.as_tensor(q_row, dtype=torch.float64)
    return lambda bed: q.unsqueeze(0).expand(n, -1)


def demo():
    torch.manual_seed(0)
    f64 = torch.float64

    print("=== (1) outcome_ppl, against hand-computed values ===")
    q = torch.tensor([[0.5, 0.3, 0.2], [0.9, 0.05, 0.05], [1 / 3, 1 / 3, 1 / 3]], dtype=f64)
    ppl = outcome_ppl(q, torch.tensor([0, 0, 2]), verbose=True)
    by_hand = math.exp(-(math.log(0.5) + math.log(0.9) + math.log(1 / 3)) / 3)
    print("(1a) mixed 3-class      : %.10f   hand: %.10f" % (ppl, by_hand))
    assert abs(ppl - by_hand) < 1e-12

    p_perfect = outcome_ppl(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=f64),
                            torch.tensor([0, 1]))
    print("(1b) perfect predictor  : %.10f   hand: 1.0 exactly" % p_perfect)
    assert abs(p_perfect - 1.0) < 1e-12

    p_uniform = outcome_ppl(torch.full((5, 3), 1 / 3, dtype=f64), torch.tensor([0, 1, 2, 0, 1]))
    print("(1c) uniform over K=3   : %.10f   chance_level(3) = %.10f"
          % (p_uniform, chance_level(3)))
    assert abs(p_uniform - chance_level(3)) < 1e-12

    p_clamped = outcome_ppl(torch.tensor([[0.0, 1.0]], dtype=f64), torch.tensor([0]))
    print("(1d) confidently wrong  : %.4f   hand: 1/eps = %.4f (log(0) clamped, not +inf)"
          % (p_clamped, 1 / EPS_Q))
    assert abs(p_clamped - 1 / EPS_Q) < 1e-6

    print("\n=== (2) the do-arm read, against the exact recompute AND by hand ===")
    # 4-node causal chain: absorbing {0,1}, transient {2,3}.
    #   q2: 0.5*q2 = [0.2,0.3]              -> [0.40, 0.60]
    #   q3: 0.5*q3 = [0.1,0.1] + 0.3*q2     -> [0.44, 0.56]
    absorbing_idx = torch.tensor([0, 1])
    P = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.2, 0.3, 0.5, 0.0],
                      [0.1, 0.1, 0.3, 0.5]], dtype=f64)
    q_field = op.committor(P, absorbing_idx)
    print("(2a) committor row2 = %s  hand = [0.4, 0.6]" % q_field[2].tolist())
    print("(2a) committor row3 = %s  hand = [0.44, 0.56]" % q_field[3].tolist())
    assert torch.allclose(q_field[2], torch.tensor([0.4, 0.6], dtype=f64), atol=1e-14)
    assert torch.allclose(q_field[3], torch.tensor([0.44, 0.56], dtype=f64), atol=1e-14)

    # do(3 -> [0.9, 0, 0, 0.1]): the only way out of 3 is that row, so it
    # reaches outcome 0 with certainty: 0.9*q3' = [0.9,0] -> q3' = [1,0].
    # den = (1 - P'_33)/(1 - P_33) = 0.9/0.5 = 1.8, exactly.
    import ceqjepa.intervene as iv
    new_row = torch.tensor([0.9, 0.0, 0.0, 0.1], dtype=f64)
    q_do, den = iv.committor_do(P, absorbing_idx, 3, new_row)
    q_do_exact = op.committor(iv.clamp_row(P, 3, new_row), absorbing_idx)
    print("(2b) committor_do row3 = %s  hand = [1.0, 0.0]" % q_do[3].tolist())
    print("(2b) committor_do row2 = %s  (row 2 PRECEDES row 3: must be unchanged)"
          % q_do[2].tolist())
    print("(2b) den = %.12f  hand: (1-0.1)/(1-0.5) = 1.8" % den)
    assert torch.allclose(q_do[3], torch.tensor([1.0, 0.0], dtype=f64), atol=1e-14)
    assert torch.allclose(q_do[2], q_field[2], atol=1e-14)
    assert abs(den - 1.8) < 1e-14
    d = float((q_do - q_do_exact).abs().max())
    print("(2c) Sherman-Morrison vs operator.committor(clamp_row(P)): max|diff| = %.3e "
          "[RUN, real ceqjepa.intervene]" % d)
    assert d < 1e-12

    # committor_do_predictions: alpha reads position 3, so the do-arm read is
    # exactly q_do[3] = [1, 0] and the obs-arm read exactly q_field[3].
    N = 4
    alpha = F.one_hot(torch.full((N,), 3), num_classes=4).to(f64)
    q_pred_do, dens = committor_do_predictions(
        P.expand(N, -1, -1), absorbing_idx, alpha,
        torch.full((N,), 3, dtype=torch.long), new_row.expand(N, -1))
    print("(2d) committor_do_predictions[0] = %s  den = %.4f  (hand: [1.0, 0.0], 1.8)"
          % (q_pred_do[0].tolist(), dens[0]))
    assert torch.allclose(q_pred_do, torch.tensor([1.0, 0.0], dtype=f64).expand(N, -1), atol=1e-14)

    print("\n=== (3) THE MANDATORY FAILURE: a small gap for the wrong reason ===")
    # Bed where the intervention FLIPS the outcome on every item -- maximally
    # causal, so ignoring the intervention must be punished.
    bed = dict(k_star_obs=torch.tensor([0, 0, 1, 1]),
               k_star_do=torch.tensor([1, 1, 0, 0]))
    onehot90 = lambda ks: (F.one_hot(ks, 2).to(f64) * 0.8 + 0.1)   # 0.9 on the true class

    mech = causal_gap(lambda b: onehot90(_field(b, "k_star_obs")),
                      lambda b: onehot90(_field(b, "k_star_do")),
                      bed, label="MECHANISM (0.9 both)")
    ctrl = ignore_intervention_gap(lambda b: onehot90(_field(b, "k_star_obs")), bed)
    const = _const_predictor([0.5, 0.5], 4)
    flat = causal_gap(const, const, bed, label="CONSTANT [0.5,0.5]")
    marg = marginal_predictor(torch.tensor([0, 0, 1, 1]), 2)
    mg = causal_gap(marg, marg, bed, label="MARGINAL control")

    # Hand: mechanism 1/0.9 both arms; ignore-control scores the do arm at the
    # flipped class, 0.1 -> PPL 10; constant exp(-ln 0.5) = 2 both arms.
    assert abs(mech["ppl_obs"] - 1 / 0.9) < 1e-12 and abs(mech["gap"]) < 1e-12
    assert abs(ctrl["ppl_do"] - 10.0) < 1e-9
    assert abs(ctrl["gap"] - (10.0 - 1 / 0.9)) < 1e-9
    assert abs(flat["ppl_obs"] - 2.0) < 1e-12 and abs(flat["gap"]) < 1e-12
    assert abs(mg["ppl_obs"] - 2.0) < 1e-12 and abs(mg["gap"]) < 1e-12
    print("     hand: mechanism 1/0.9 = %.4f both arms, gap 0 exactly | ignore-control "
          "do-arm 1/0.1 = 10.0 | constant exp(-ln 0.5) = 2.0 both arms, gap 0 exactly"
          % (1 / 0.9))
    print("     THE TRAP, MEASURED: gap(MECHANISM) = %+.6f and gap(CONSTANT) = %+.6f are "
          "the SAME NUMBER." % (mech["gap"], flat["gap"]))
    print("     They are separated ONLY by PPL_obs: %.4f vs %.4f (chance = %.2f). A gap "
          "read alone ranks a predictor that has learned nothing level with a perfect "
          "one. This is why (PPL_obs, gap) ship together, always."
          % (mech["ppl_obs"], flat["ppl_obs"], flat["chance"]))
    print("     THE BAR, MEASURED: the ignore-the-intervention control pays %+.4f, so on "
          "this bed a mechanism claim is the claim that gap < %.4f -- not that gap is "
          "small." % (ctrl["gap"], ctrl["gap"]))

    print("\n=== (4) bootstrap SE against the delta method ===")
    # Half the items at 0.9, half at 0.2, N = 200, all realizing class 0.
    n = 200
    q_het = torch.zeros(n, 2, dtype=f64)
    q_het[:n // 2, 0], q_het[n // 2:, 0] = 0.9, 0.2
    q_het[:, 1] = 1.0 - q_het[:, 0]
    bed_het = dict(k_star_obs=torch.zeros(n, dtype=torch.long),
                   k_star_do=torch.zeros(n, dtype=torch.long))
    het = causal_gap(lambda b: q_het, lambda b: q_het, bed_het, n_boot=2000,
                     label="heteroscedastic")
    lp = torch.log(q_het[:, 0])
    delta = float(torch.exp(-lp.mean()) * lp.std(unbiased=False) / math.sqrt(n))
    print("(4a) bootstrap se_ppl_obs = %.6f   delta-method ppl*sd(logp)/sqrt(N) = %.6f   "
          "ratio = %.4f" % (het["se_ppl_obs"], delta, het["se_ppl_obs"] / delta))
    assert 0.85 < het["se_ppl_obs"] / delta < 1.15, "bootstrap SE is not the sampling SE"

    # A gap whose SE is NONZERO -- the headline number, shown doing its job.
    # Same predictor, but the do arm's realized class is flipped on the second
    # half, so the two arms genuinely differ and the gap can move under
    # resampling. The SE is what says whether such a gap is real: here the gap
    # is many SEs from zero, which is the report shape every eval must use.
    bed_split = dict(k_star_obs=torch.zeros(n, dtype=torch.long),
                     k_star_do=torch.cat([torch.zeros(n // 2, dtype=torch.long),
                                          torch.ones(n // 2, dtype=torch.long)]))
    split = causal_gap(lambda b: q_het, lambda b: q_het, bed_split, n_boot=2000,
                       label="nonzero-gap arms")
    print("(4b) se_gap = %.6f (NONZERO: the arms differ) -> gap = %+.4f is %.1f SEs from "
          "zero. PPL_obs = %.4f against chance %.2f, so the gap is readable at all."
          % (split["se_gap"], split["gap"], abs(split["gap"]) / split["se_gap"],
             split["ppl_obs"], split["chance"]))
    assert split["se_gap"] > 1e-3, "a gap between genuinely different arms must carry a SE"
    assert abs(split["gap"]) > 5 * split["se_gap"]

    print("\n=== (5) PLANTED NEGATIVE: the bootstrap must be PAIRED ===")
    print("(5a) same predictor, same k_star both arms -> the gap is identically 0 on "
          "every resample, so se_gap must be 0.0 EXACTLY.")
    print("     se_ppl_obs = %.6f (nonzero: each ARM does vary) but se_gap = %.6e"
          % (het["se_ppl_obs"], het["se_gap"]))
    assert het["se_ppl_obs"] > 1e-3, "the arms should vary; the check below is vacuous otherwise"
    assert het["se_gap"] == 0.0, "se_gap != 0 on an identical-arm bed: the bootstrap is UNPAIRED"
    # Show what an unpaired bootstrap would have reported on the same data.
    g = torch.Generator().manual_seed(0)
    i1 = torch.randint(0, n, (2000, n), generator=g)
    i2 = torch.randint(0, n, (2000, n), generator=g)
    unpaired = float((torch.exp(-lp[i2].mean(1)) - torch.exp(-lp[i1].mean(1))).std(unbiased=True))
    print("     an UNPAIRED bootstrap on the identical data reports se_gap = %.6f, i.e. "
          "%.2fx the arm SE and entirely spurious. [RUN]" % (unpaired, unpaired / het["se_ppl_obs"]))
    assert unpaired > 10 * max(het["se_gap"], 1e-9)

    print("\n=== (6) reliability buckets ===")
    q_cal = torch.tensor([[0.05, 0.95], [0.15, 0.85], [0.55, 0.45], [0.95, 0.05]], dtype=f64)
    cal = calibration_report(q_cal, torch.tensor([1, 0, 0, 0]), n_bins=10)
    # points: (.05,0) (.95,1) (.15,0) (.85,0) (.55,1) (.45,0) (.95,1) (.05,0)
    b0, b9 = cal["bins"][0], cal["bins"][9]
    print("(6a) bin [0.0,0.1): count=%d mean_pred=%.4f empirical=%.4f   hand: 2, 0.05, 0.0"
          % (b0["count"], b0["mean_pred"], b0["empirical_freq"]))
    print("(6b) bin [0.9,1.0]: count=%d mean_pred=%.4f empirical=%.4f   hand: 2, 0.95, 1.0"
          % (b9["count"], b9["mean_pred"], b9["empirical_freq"]))
    assert b0["count"] == 2 and abs(b0["mean_pred"] - 0.05) < 1e-12 and b0["empirical_freq"] == 0.0
    assert b9["count"] == 2 and abs(b9["mean_pred"] - 0.95) < 1e-12 and b9["empirical_freq"] == 1.0
    assert cal["n_points"] == q_cal.numel()
    # A perfectly calibrated set: 0.5 everywhere, half the labels 1 -> ECE 0.
    q_flat = torch.full((4, 2), 0.5, dtype=f64)
    cal2 = calibration_report(q_flat, torch.tensor([0, 1, 0, 1]), n_bins=10)
    print("(6c) ECE(this predictor) = %.6f   ECE(q=0.5 everywhere, balanced) = %.6f  "
          "hand: 0.0 exactly" % (cal["ece"], cal2["ece"]))
    assert cal2["ece"] == 0.0

    print("\nALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    demo()
