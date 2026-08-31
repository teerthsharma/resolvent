"""FOREMAN round 3 -- is 1.0334 at 3.3M a property of the operator or of the box?

The campaign's standing verdict is a median val-loss ratio of **1.0334** against
softmax at 3,319,296 matched parameters, 5/5 seeds, at `rho=1.5, lam=0.10,
hops=2, lr=1e-3`, d=256, 4 layers, **seq 128**, 600 steps, byte-level TinyStories.

The standard that governs is not the one the loop was run against. Parity at 3.3M
is FALSE unless it survives at 300M. The ORIGINAL row-L1 operator already showed
exactly the pathology in question -- 1.2191x at 250 steps became 1.337x at 600 --
and the README states in as many words that the same measurement was never
repeated on `sgate`.

Two independent things are asked here, and they are kept in separate tests
because they can disagree.

**THE SLOPE.** `compare_kinds` at three budgets, three widths and three context
lengths. C3 requires slope at >= 3 sizes; a ratio that is flat or falling is
evidence the operator scales, a ratio that climbs is the artifact verdict. These
are the `slow` tests. Nothing about them is a derivation.

**THE MECHANISM.** `A = rho*(softmax(w) - lam*softmax(-w))/(1+lam)`. The positive
branch concentrates on the MOST similar keys; the negative branch concentrates on
the LEAST similar ones. Training has a gradient reason to sharpen the first --
that is where the signal is read -- and no such reason for the second, because
the least-similar key of a row is whatever happens to sit at the bottom of the
logit bulk. If the negative branch does not sharpen, then as the context grows it
spreads over more and more tokens and converges on the prefix MEAN: a content-free
DC term holding a fixed `lam/(1+lam)` share of the row's absolute mass. At seq 128
that share is spread over at most 127 tokens. At the 1024-2048 a 300M run uses it
is spread over an order of magnitude more, and the operator degenerates toward
`rho/(1+lam) * softmax(w)` minus a constant -- which is softmax with a rescale,
carrying none of the signed property the module exists for.

That is a prediction with a number attached and it is measured in
`test_claim_the_negative_branch_sharpens_with_context_the_way_the_positive_one_does`
on a TRAINED model, since with random logits the two branches are symmetric by
construction and the question cannot even be posed.

Every test parametrizes over cpu and cuda, skipping cuda when absent.
"""
from __future__ import annotations

import math
import pathlib

import pytest
import torch

from ceq import bench, lm

CORPUS = pathlib.Path(__file__).resolve().parents[2] / "data" / "tinystories_20k.txt"

# The campaign configuration, verbatim from DONE.md iteration 16.
RHO, LAM, HOPS, LR = 1.5, 0.10, 2, 1e-3


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    return torch.device(request.param)


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS.exists():
        pytest.skip(f"{CORPUS} missing; run the W10 data step")
    return lm.ByteCorpus(CORPUS.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def campaign_knobs():
    """Every test in this file runs the operator the campaign actually shipped."""
    old = lm.RHO, lm.SGATE_LAM, lm.HOPS
    lm.RHO, lm.SGATE_LAM, lm.HOPS = RHO, LAM, HOPS
    yield
    lm.RHO, lm.SGATE_LAM, lm.HOPS = old


def _ratio(corpus, *, steps, device, seed, **model_kw) -> tuple[float, dict]:
    """sgate val loss / softmax val loss at a matched budget."""
    res = lm.compare_kinds(corpus, ("softmax", "sgate"), steps=steps,
                           device=device, seed=seed, lr=LR, **model_kw)
    return res["sgate"]["val_loss"] / res["softmax"]["val_loss"], res


# --------------------------------------------------------------- the instrument

DRAWS = 128   # the published numbers are k/128; 0.1641 = 21/128 exactly


def test_the_sign_flip_probe_still_reports_its_published_numbers(device):
    """Calibration. Nothing below is read until the instrument reproduces the
    numbers already on the record, so a change here is a change in the probe and
    not in the operator. `n_draws=128` is recovered from the published values:
    0.0469 = 6/128, 0.1641 = 21/128, 0.0234 = 3/128, all exact."""
    dev = torch.device(device)
    r = lambda **kw: bench.sign_flip_rate(n_draws=DRAWS, device=dev, **kw)
    assert r(kind="softmax", depth=1) == 0.0
    assert r(kind="signed", depth=1, hops=3) == pytest.approx(0.0469, abs=1e-4)
    assert r(kind="sgate", depth=1, hops=2) == pytest.approx(0.1641, abs=1e-4)
    assert r(kind="sgate", depth=1, hops=1) == pytest.approx(0.0234, abs=1e-4)


# ------------------------------------------------------------------ MECHANISM

def _branch_normalized_entropies(model, x, prefixes):
    """H(branch row i) / ln(i) for both softmax halves, averaged over every
    layer, head and batch element.

    Normalized by `ln(i)` because row `i` of a strictly causal operator has
    exactly `i` live entries, so `ln(i)` is that row's uniform ceiling. 1.0 means
    the branch is a flat average over the prefix and carries no content.
    """
    caught = []
    handles = [b.attn.register_forward_pre_hook(lambda m, inp: caught.append(inp[0]))
               for b in model.blocks]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    pos, neg = {i: [] for i in prefixes}, {i: [] for i in prefixes}
    for attn, xin in zip([b.attn for b in model.blocks], caught):
        q, k, _ = attn.qkv_heads(xin)
        w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        for i in prefixes:
            row = w[..., i, :i].double()            # the i live entries of row i
            for name, logits in (("pos", row), ("neg", -row)):
                p = torch.softmax(logits, dim=-1)
                h = -(p * p.clamp_min(1e-300).log()).sum(-1) / math.log(i)
                (pos if name == "pos" else neg)[i].append(float(h.mean()))
    return ({i: sum(v) / len(v) for i, v in pos.items()},
            {i: sum(v) / len(v) for i, v in neg.items()})


@pytest.mark.slow
def test_claim_the_negative_branch_sharpens_with_context_the_way_the_positive_one_does(
        device, corpus):
    """RED first.

    CLAIM UNDER TEST: both halves of `softmax(w) - lam*softmax(-w)` are
    selective, so the operator's behaviour at seq 128 is the same operator at
    seq 1024.

    KILL CONDITION: if the negative branch's normalized entropy sits at the
    uniform ceiling and stays there as the prefix grows while the positive
    branch's falls, then the negative half is a running MEAN, not a selection,
    and its fixed share of row mass buys the operator a constant. The signed
    property is then an artifact of short context and cannot survive 300M.
    """
    prefixes = [8, 16, 32, 64, 128, 255]
    res = lm.train_one("sgate", corpus, steps=400, device=device, seed=0,
                       lr=LR, seq=256, bs=16)
    m = res["model"]
    gen = torch.Generator().manual_seed(99)
    x, _ = corpus.batch("val", 16, 256, gen, device)
    pos, neg = _branch_normalized_entropies(m, x, prefixes)

    report = " ".join(f"{i}:pos={pos[i]:.4f}/neg={neg[i]:.4f}" for i in prefixes)
    assert neg[prefixes[-1]] < 0.95, (
        f"the negative branch is at the uniform ceiling: {report}. "
        f"softmax(-w) is not selecting a least-similar key, it is averaging the "
        f"whole prefix, so lam/(1+lam) of every row's mass is a DC term whose "
        f"information content falls as 1/S.")


@pytest.mark.slow
def test_claim_the_two_branches_stay_equally_selective_as_the_prefix_grows(
        device, corpus):
    """RED first. The sharper form of the same question: the GAP.

    If the two branches are symmetric the operator is a genuine two-sided
    selector at any length. If the gap `neg - pos` widens with prefix length,
    the operator is drifting toward `softmax(w) minus a constant` and the drift
    has a measured rate, which is the thing that extrapolates to 300M.
    """
    prefixes = [8, 255]
    res = lm.train_one("sgate", corpus, steps=400, device=device, seed=0,
                       lr=LR, seq=256, bs=16)
    pos, neg = _branch_normalized_entropies(
        res["model"], corpus.batch("val", 16, 256,
                                   torch.Generator().manual_seed(99), device)[0],
        prefixes)
    gap_short = neg[prefixes[0]] - pos[prefixes[0]]
    gap_long = neg[prefixes[-1]] - pos[prefixes[-1]]
    assert gap_long <= gap_short + 0.02, (
        f"branch-selectivity gap widens with context: {gap_short:.4f} at prefix "
        f"{prefixes[0]} -> {gap_long:.4f} at prefix {prefixes[-1]}. "
        f"pos={pos} neg={neg}")


def test_claim_content_conditional_sign_survives_a_longer_context(device):
    """RED first.

    `sign_flip_rate` measured 0.1641 for sgate at s=8 against softmax's exactly
    0.0000. s=8 is the probe's default, not a chosen operating point. If the rate
    decays toward softmax's zero as s grows, the module's one distinguishing
    property is a short-context artifact and the 300M question is already
    answered in the negative.
    """
    dev = torch.device(device)
    sweep = lambda **kw: {s: bench.sign_flip_rate(depth=1, hops=HOPS, s=s, i=s - 1,
                                                 j=1, c=s // 2, n_draws=256,
                                                 device=dev, **kw)
                          for s in (8, 16, 32, 64, 128)}
    rates, control = sweep(kind="sgate"), sweep(kind="softmax")
    nofloor = sweep(kind="sgate", floor=0.0)
    assert set(control.values()) == {0.0}, control
    assert rates[128] >= 0.5 * rates[8], (
        f"content-conditional sign rate decays with context: sgate={rates}, "
        f"softmax control={control}, sgate with the discard floor disabled="
        f"{nofloor}. The floor is not the cause -- the rates agree with it off. "
        f"softmax's value is exactly 0.0000 at every s, so decay toward it is "
        f"decay toward having no distinguishing property at all.")


def test_claim_a_signed_hop_coefficient_cannot_do_what_a_signed_matrix_does(device):
    """RED first. The novelty claim, restated against the paper that occupies it.

    README: "every published multi-hop propagation requires non-negativity".
    ParaFormer (arXiv:2512.14619, 16 Dec 2025) Eq. 9 is

        Z = sum_{k=0}^{K} gamma_k Ahat^k V,   Ahat = Softmax(QK^T / sqrt(d))

    with "{gamma_k in R | k=0,1,2,...,K} ... a set of learnable weights", i.e.
    multi-hop propagation with SIGNED hop coefficients over a NON-NEGATIVE base
    matrix. So the residual claim can only be that the BASE MATRIX is signed, not
    merely the coefficients.

    That residual is worth something only if the two are distinguishable. The
    influence Jacobian of the ParaFormer form is `sum_k gamma_k (Ahat^k)_ij` --
    every factor non-negative, every coefficient a constant -- but the MIXTURE
    over k is content-dependent, so a third token can move which hop dominates.
    This test asserts it cannot. If it is RED, the two constructions are not
    separable on the instrument this project uses to state the property at all.

    ROUND 4: IT IS RED, and the round-3 GREEN was an artifact of the probe.
    `sign_flip_rate` drew the `gam` vector and never referenced it; the
    `paraformer` arm fell through to `else: a = _softmax_operator(...)` followed
    by `h = a @ h`, so "0 flips in 2048 draws" was a single-hop SOFTMAX number,
    not a ParaFormer number. With Eq. 9 actually applied the arm reads
    **0.023926** here against sgate's 0.145020 on the same draws -- a factor of
    6.1, not the published ">= 84x". Kept RED rather than retuned: the claim it
    was written to defend is deleted. The corrected table with Clopper-Pearson
    intervals at every context length is in
    `tests/foreman/test_paraformer_ratio_across_context.py`.
    """
    dev = torch.device(device)
    n = 2048
    rate = bench.sign_flip_rate("paraformer", depth=1, hops=HOPS, n_draws=n,
                                device=dev)
    ceiling = bench.zero_success_upper_bound(n)      # 0.00146 at n = 2048
    sgate = bench.sign_flip_rate("sgate", depth=1, hops=HOPS, n_draws=n, device=dev)
    assert rate == 0.0 and sgate > 50 * ceiling, (
        f"ParaFormer's signed hop coefficients over a non-negative softmax "
        f"matrix reach a content-conditional sign rate of {rate}, against "
        f"softmax's exactly 0.0000, against sgate's {sgate} on the same draws. "
        f"Signed coefficients are not weaker than a signed matrix on this probe.")


def test_claim_some_knob_in_the_operator_family_arrests_the_context_decay(device):
    """RED first. If the decay is real, the next question is whether it is
    fixable from inside the operator.

    `lam` is the signed mass and `rho` the row scale -- the two knobs the
    campaign tuned. `lam = 0` deletes the negative half entirely and is the
    control INSIDE the family: it must read exactly 0.0000 at every context
    length, the same as softmax, or the probe is measuring something other than
    signedness.

    KILL CONDITION: if no setting of either knob holds even half of its own
    short-context rate out to s = 128, the decay is a property of the operator
    FAMILY and not of an operating point, and no amount of further tuning at
    3.3M changes what happens at 300M.
    """
    dev = torch.device(device)
    r = lambda s, **kw: bench.sign_flip_rate("sgate", depth=1, hops=HOPS, s=s,
                                             i=s - 1, j=1, c=s // 2, n_draws=256,
                                             device=dev, **kw)
    control = {s: r(s, lam=0.0) for s in (8, 128)}
    assert set(control.values()) == {0.0}, f"lam=0 is not the non-negative control: {control}"

    grid = {("lam", v): (r(8, lam=v), r(128, lam=v))
            for v in (0.05, 0.25, 1.00, 2.00)}
    grid.update({("rho", v): (r(8, rho=v), r(128, rho=v)) for v in (0.5, 4.0)})
    # head width is not an operator knob, but it is the axis that actually grows
    # from 3.3M to 300M: d_head there is 64-128, not the probe's default 16.
    grid.update({("d_head", v): (r(8, d=v), r(128, d=v)) for v in (32, 64, 128)})
    held = {k: v for k, v in grid.items() if v[1] >= 0.5 * v[0]}
    assert held, (
        f"no setting of rho or lam holds half its short-context sign rate out to "
        f"s=128: {grid}. The decay is a property of the operator family.")


def test_claim_a_pairwise_sign_holds_the_property_where_a_normalized_one_loses_it(device):
    """RED first. THE LEAP, stated as a falsifier rather than a plan.

    MECHANISM the decay tests establish: in `rho*(softmax(w) - lam*softmax(-w))`
    the SIGN of an entry is decided by a comparison between two GLOBALLY
    NORMALIZED quantities, `p+_ij` against `lam*p-_ij`. Both normalizers run over
    the whole prefix, so every entry's share -- and with it every token's
    leverage over any other entry's sign -- is diluted by adding tokens. The
    measured consequence is a rate falling as s^-1.4 with no knob that arrests it.

    THE MOVE: decide the sign from an UNNORMALIZED pairwise quantity and take
    only the MAGNITUDE from a normalized one:

        A = rho * sgn(w) * softmax(|w|)

    Row L1 is exactly rho, so the operator stays strictly causal with bounded
    rows and `CEQ.Nilpotent.pow_card_eq_zero` applies unchanged. `sgn(w_ij)`
    depends on the pair (i, j) alone; no third token can change it, and adding
    tokens cannot dilute it. This is Cog Attention's matrix (arXiv:2411.07176,
    `SignExp(p)/sum|SignExp|` = `sgn(p)*softmax(|p|)`), which is published and
    SINGLE-HOP -- the multi-hop path sum over it is the cell the project already
    identified as unoccupied.

    KILL CONDITION: if the pairwise-sign operator decays with context the same
    way, then the decay belongs to multi-hop signed propagation itself, the
    unoccupied cell is unoccupied for a reason, and there is no leap here.
    """
    dev = torch.device(device)
    r = lambda kind, s: bench.sign_flip_rate(kind, depth=1, hops=HOPS, s=s,
                                             i=s - 1, j=1, c=s // 2, n_draws=512,
                                             device=dev)
    held = {s: r("signmag", s) for s in (8, 16, 32, 64, 128)}
    lost = {s: r("sgate", s) for s in (8, 128)}
    assert held[128] >= 0.5 * held[8], (
        f"a pairwise sign decays too: signmag={held}, sgate={lost}. The decay is "
        f"a property of multi-hop signed propagation, not of the normalizer.")


def test_claim_the_property_does_not_live_only_in_the_multi_hop_term(device):
    """RED first. WHERE the property lives decides whether it can be kept.

    A third token c can only reach the pair (i, j) through a path that passes
    THROUGH c, and the first term of `J = sum_k A^k` containing such a path is
    k = 2. If the whole content-conditional sign rate is carried by k >= 2, then
    it is carried by a sum over ~s intermediate tokens of which c is exactly one,
    its share is 1/s by construction, and the decay measured elsewhere in this
    file is not a defect that can be tuned out -- it is the same fact as the
    property.
    """
    dev = torch.device(device)
    r = lambda kind, h: bench.sign_flip_rate(kind, depth=1, hops=h, n_draws=512,
                                             device=dev)
    one = {k: r(k, 1) for k in ("sgate", "signmag")}
    two = {k: r(k, 2) for k in ("sgate", "signmag")}
    assert min(one.values()) >= 0.5 * min(two.values()), (
        f"the property is a k>=2 effect: hops=1 {one} against hops=2 {two}. The "
        f"single-hop term A_ij does not contain token c at all, so every flip is "
        f"a 1-in-s share of the two-hop sum and must dilute as the context grows.")


def test_claim_the_context_decay_is_specific_to_the_signed_value_path(device):
    """RED first. The control that decides whether the decay means anything.

    `wrt="x"` differentiates the INPUT path, which the probe's own docstring
    records as sign-unconstrained for EVERY operator, softmax included. If
    softmax decays there at the same rate the signed operators decay on the value
    path, then what is being measured is the dilution of ONE token's leverage in
    a context of s -- a fact about attention, not about signedness -- and the
    finding has to be restated in those terms.
    """
    dev = torch.device(device)
    smx = {s: bench.sign_flip_rate("softmax", depth=1, hops=2, s=s, i=s - 1, j=1,
                                   c=s // 2, n_draws=512, device=dev, wrt="x")
           for s in (8, 32, 128)}
    assert smx[128] >= 0.5 * smx[8], (
        f"softmax decays on the sign-unconstrained input path too: {smx}. The "
        f"decay is dilution of a single token's leverage, common to attention, "
        f"and not a property of the signed operator.")


# ---------------------------------------------------------------------- SLOPE

@pytest.mark.slow
def test_claim_the_parity_ratio_does_not_widen_with_training_budget(device, corpus):
    """RED first. The exact pathology the L1 operator showed, re-measured on
    the operator that reached parity.

    L1 operator: 1.2191x at 250 steps -> 1.337x at 600. If sgate does the same
    thing more slowly, 1.0334 at 600 steps is a number about a budget, not about
    an operator, and 300M -- which is trained for orders of magnitude longer --
    is already lost.
    """
    budgets = (600, 1200, 2400)
    seen = {}
    for steps in budgets:
        rs = [_ratio(corpus, steps=steps, device=device, seed=s)[0] for s in (0, 1)]
        seen[steps] = sorted(rs)
    lo, hi = seen[budgets[0]][0], seen[budgets[-1]][0]
    print(f"BUDGET LADDER ratios by steps: {seen}")
    assert max(seen[budgets[-1]]) <= max(seen[budgets[0]]) + 0.01, (
        f"parity ratio widens with budget: {seen}. Same shape as the L1 "
        f"operator's 1.2191 -> 1.337.")


@pytest.mark.slow
def test_claim_the_parity_ratio_does_not_widen_with_context_length(device, corpus):
    """RED first. The axis the mechanism predicts, measured end to end.

    Every published number for this operator is at seq 128. A 300M run is at
    1024-2048. If the negative branch is a prefix mean, the ratio must climb here
    and the two tests fail together -- which is what makes this a mechanism and
    not a coincidence.
    """
    seqs = (64, 128, 256)
    seen = {s: _ratio(corpus, steps=600, device=device, seed=0, seq=s, bs=16)[0]
            for s in seqs}
    print(f"CONTEXT LADDER ratios by seq: {seen}")
    assert seen[seqs[-1]] <= seen[seqs[0]] + 0.01, (
        f"parity ratio widens with context length: {seen}")


@pytest.mark.slow
def test_claim_the_parity_ratio_does_not_widen_with_model_width(device, corpus):
    """RED first. C3's own requirement: slope at >= 3 sizes.

    3.3M -> 300M is ~10x in width. Three widths on a laptop cannot prove parity
    at 300M, but a ratio that climbs monotonically across them refutes it, and
    that is the only direction this box can settle.
    """
    widths = (128, 256, 384)
    seen = {d: _ratio(corpus, steps=600, device=device, seed=0, d=d)[0]
            for d in widths}
    print(f"WIDTH LADDER ratios by d_model: {seen}")
    assert seen[widths[-1]] <= seen[widths[0]] + 0.01, (
        f"parity ratio widens with width: {seen}")
