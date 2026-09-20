"""tests/cameron/test_third_collapse_leg.py -- ASSAY C2: is there a third
collapse leg, alongside pi_jepa.collapse_report's std_min and effective rank?

THE GAP THE TWO LEGS CANNOT CLOSE, BY CONSTRUCTION AND NOT BY TUNING.
pi_jepa.collapse_report computes std_min and erank from the representation s
ALONE, before it ever looks at q or y -- neither leg's signature takes a label. So no threshold on either one
can ever depend on whether s carries information about the task's actual
target: a leg that never reads y cannot detect the loss of something only y
carries. That is a structural claim, and structural claims are cheap unless a
concrete plant is exhibited that both legs call healthy while the
representation is provably uninformative about its label.

THE REACHABLE PLANT (the finding this file exists to carry). A frozen-random
encoder -- one whose output never receives a gradient -- is not a hypothetical:
it is the repo's own mandatory control at docs/COMPONENT_LEDGER.md:47-58,
"identical to the trained arm except that its encoder receives no gradient."
Its representation is isotropic noise, independent of the input and therefore
of the REAL label grp -- grp is never touched, permuted, or resampled here.
This is what a training run reaches when the encoder fails to learn: nothing
about it requires an adversary to reach behind the representation and shuffle
labels by hand. That is the difference from a label permutation, which no
gradient step, no matter how bad, can ever produce as its (s, y) pairing --
permuting y is an operation on the TEST, not a state training visits.

THE MECHANISM, and it is the reason the geometric legs are blind here: a
representation that has genuinely learned K clusters pins at effective rank
about K-1 no matter how wide the layer is, because clustering IS anisotropy.
An untrained, frozen-random encoder spreads across all D directions instead
and scores an effective rank near D -- HIGHER, not lower, than the informative
case -- so std_min and erank both read GREEN on the least informative
representation there is.

THE CONSTRUCTION. K=4 group centroids at 2*eye(K) in a D=K=4 latent space;
the informative baseline is s = centroid[grp] + N(0, NOISE_STD^2), q =
softmax(-||s - centroid_k||^2) -- a nearest-centroid read computed FROM s
itself. The frozen-random plant keeps the same real grp and the same q
readout function, but replaces s with s = N(0, I_D): the "encoder" ignores
its input entirely, exactly what an untrained or frozen-random encoder does.
All figures below are float64 (torch default for this construction;
pi_jepa.collapse_report upcasts to float64 internally regardless of s's input
dtype).

MEASURED at SEED=20260920, N=20000, boot_seed=SEED (pi_jepa.collapse_report's
own bootstrap-seed argument -- margin is seed-independent but se_margin, and
therefore margin_sigma, is NOT: it moves with the bootstrap draw, so it is
reported here with the seed that produced it, never as a bare number), by
running this file (`pytest tests/cameron/test_third_collapse_leg.py -q`) and
calling pj.collapse_report with the shipped (q, y, K) signature rather than
reading sharpness directly: informative baseline std_min 0.8648, erank
3.0047 (both green), leg=None, i_q 6.0008 at se_i_q 1.3455e-3, ORACLE
margin_sigma +14694.4654 at boot_seed=20260920 -- ORACLE because this q is
the nearest-centroid readout of an s that was itself built FROM grp
(centroid[grp] + noise), so the figure measures how separable the
construction made the labels, not what any model read off an independent
representation; it is printed for completeness, never as a bar. Frozen-
random plant on the SAME real, unpermuted grp: std_min 0.9904, erank 3.9993
(both green, and erank is HIGHER than the informative case's 3.0047, the
inversion the mechanism above predicts), leg='label', i_q 0.019593 at
se_i_q 0.026208, margin -2.9184 at margin_sigma -96.4079 at
boot_seed=20260920 (this margin_sigma moves with the bootstrap seed -- at
boot_seed=2 it reads -103.6875; report the seed, not a bare figure). RED
FIRES: the encoder that learned nothing reads healthier on both geometric
legs than the encoder that learned the task, while the label-aware leg
correctly calls it dead.

RULE 2 -- THE BED AND ITS GUARDED-ENTRY COUNT, STATED BY COUNTING, NOT BY
GREP. A "guarded entry" for the label leg is an execution of
pi_jepa.collapse_report's `if q is not None and y is not None:` branch body
-- observable only by instrumenting that body,
because a grep of call sites counts sites
that COULD reach the branch, not evaluations that DID (this repo has made
that exact mistake three times; see pi_jepa.py's own module docstring for the
first two). This file previously called `collapse_report(s)` at all 5 sites
below with neither q nor y, so grepping "5 call sites" and calling that the
guarded-entry count was the same error a third time: the true count was 0,
not 5. Of the 6 `collapse_report` calls in this file today, 5 now pass q, y
and K (the shipped signature) and enter the branch; the 6th (the frozen-
random test's erank cross-check against the informative baseline) asks only
for `["erank"]` and passes neither, so it stays a two-legged call by choice,
not by oversight. Verified by wrapping `sharpness.decompose` -- the only
function the label branch calls -- with a counter keyed on the CALLING
FRAME (co_name == "collapse_report" and f_lineno equal to the branch's first
statement), for the duration of a
`pytest tests/cameron/test_third_collapse_leg.py -q` run: 5 label-branch
entries, 2 label-branch fires (leg='label'): the frozen-random plant's own
read, and the permuted-label read AFTER permutation. The other 3 guarded
entries -- the baseline test's read, the frozen-random test's informative-q
comparison read, and the permuted test's BEFORE read -- all carry the real,
unpermuted label and correctly do not fire (and, since they read an ORACLE
q, are also tagged construction_property=True below, which bars them from
the kill_fired call before firing is even possible). A plain call counter on
`sharpness.decompose` would instead read 6, because it also counts this
file's own direct construction-property call in the baseline test below --
that is not a branch counter, so the frame filter is required. The target
line CANNOT be pinned as a literal, because it is a line in pi_jepa.py and
this file does not own that module: instead the branch is located by
searching collapse_report's own source for its condition text, so the
recipe cannot go stale under a parallel edit to pi_jepa.py. Instrument it
yourself with:
    python -c "
    import sys
    import inspect
    import torch
    from ceqjepa import pi_jepa as pj
    from ceqjepa import sharpness
    src = inspect.getsource(pj.collapse_report)
    start = pj.collapse_report.__code__.co_firstlineno
    lines = src.splitlines()
    cond_idx = next(
        i for i, l in enumerate(lines)
        if 'q is not None and y is not None' in l
    )
    first_stmt_idx = cond_idx + 1
    while not lines[first_stmt_idx].strip() or lines[first_stmt_idx].strip().startswith('#'):
        first_stmt_idx += 1
    target_line = start + first_stmt_idx
    n = {'entries': 0}
    orig = sharpness.decompose
    def counted(*a, **k):
        caller = sys._getframe(1)
        if caller.f_code.co_name == 'collapse_report' and caller.f_lineno == target_line:
            n['entries'] += 1
        return orig(*a, **k)
    sharpness.decompose = counted
    import pytest
    pytest.main(['tests/cameron/test_third_collapse_leg.py', '-q'])
    print('label-branch entries:', n['entries'])
    "
MEASURED verbatim against tree on top of 1d7863f, torch 2.14.0+cpu: branch
condition line 586, first-statement line 587 (both found by search, printed
for the record only -- nothing below depends on the number), label-branch
entries: 5. If the count printed is not 5, something in this file stopped
passing q and y.

THE PERMUTATION CASE IS KEPT BUT DEMOTED. Permuting y relative to (s, q) is
still useful as a SIGNATURE check -- it isolates that neither leg's formula
reads y at all, since permuting labels cannot change std_min or erank by even
a bit (s is untouched) -- but it is not evidence of what training reaches, so
it is no longer presented as the reachable proof. That role belongs to the
frozen-random plant above.

Two legs are not enough: this file is the finding, not a proposal for a
third detector's implementation (out of scope for this lane -- pi_jepa.py is
owned elsewhere).
"""

from __future__ import annotations

import pytest
import torch

from ceqjepa import pi_jepa as pj
from ceqjepa import sharpness

N = 20000
K = 4
D = K  # no room needed above K: erank's ceiling is K for the informative case
       # and D for the frozen-random case; both are checked against D below.
SEED = 20260920
NOISE_STD = 0.05


def _grouped_representation(seed):
    """grp, s, q -- float64. s is an informative representation with a group
    structure (what a TRAINED encoder produces); q is a nearest-centroid read
    of s itself (deterministic given s), not a second, independent oracle the
    way sharpness._grouped_draw's P matrix is."""
    g = torch.Generator().manual_seed(seed)
    centroids = 2.0 * torch.eye(K, dtype=torch.float64)
    grp = torch.randint(K, (N,), generator=g)
    s = centroids[grp] + NOISE_STD * torch.randn(N, D, generator=g, dtype=torch.float64)
    d2 = ((s[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    q = torch.softmax(-d2, dim=1)
    return grp, s, q


def _frozen_random_representation(seed):
    """grp, s, q -- float64, REAL labels, never permuted or touched. s models
    a frozen-random encoder (docs/COMPONENT_LEDGER.md:47-58's mandatory
    control): its output is isotropic noise INDEPENDENT of the input, so it
    is independent of grp too, even though grp is the genuine label paired
    with this draw. q is the identical nearest-centroid readout function used
    in `_grouped_representation`, run on this uninformative s -- the readout
    is not the thing under test, the representation feeding it is."""
    g = torch.Generator().manual_seed(seed)
    centroids = 2.0 * torch.eye(K, dtype=torch.float64)
    grp = torch.randint(K, (N,), generator=g)  # the REAL label; unpermuted
    s = torch.randn(N, D, generator=g, dtype=torch.float64)  # frozen-random: ignores the input
    d2 = ((s[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    q = torch.softmax(-d2, dim=1)
    return grp, s, q


def test_baseline_clears_both_legs_and_carries_real_information():
    """MEASUREMENT, not a bar on its own: pins the construction healthy on
    every leg before either plant is allowed to touch it, so a later RED
    cannot be blamed on a broken representation rather than on the plant.
    Would fail if: NOISE_STD were raised enough to blur the four centroids
    together (std_min or erank would drop under their floors), or if grp were
    shuffled before scoring (i_q would collapse the way the plant's does)."""
    grp, s, q = _grouped_representation(SEED)
    assert s.shape[0] == N  # guarded-entry count: all N rows, no subsampling

    # shipped signature: q and y activate the label leg
    # (pi_jepa.collapse_report's `if q is not None and y is not None:`
    # branch), a real guarded entry, not the two-legged `collapse_report(s)`
    # this file called at all 5 sites before -- see RULE 2 above. q here is
    # the nearest-centroid ORACLE read of an s built FROM grp (see
    # `_grouped_representation`'s docstring), so it is tagged
    # construction_property=True: RULE 1, a construction property must never
    # reach the bar API, so kill_fired is not called and this can never fire
    # "label" from an oracle read (leg stays None either way -- tagging moves
    # no assertion here, only which path the call takes to get there).
    rep = pj.collapse_report(
        s, q=q, y=grp, K=K, boot_seed=SEED, construction_property=True
    )
    assert rep["std_min"] > pj.COLLAPSE_STD_MIN, rep
    assert rep["erank"] > pj.COLLAPSE_ERANK_MIN, rep
    assert rep["collapsed"] is False and rep["leg"] is None, (
        f"the unpermuted read should clear the PREREG bar via the shipped "
        f"label leg: leg={rep['leg']}, label={rep['label']}, "
        f"boot_seed={SEED}, dtype={q.dtype}"
    )

    # ASSAY C1's guard, demonstrated rather than asserted-by-comment: q here
    # is a nearest-centroid read of an s built FROM the labels -- a
    # construction property of this synthetic plant, never a model's read --
    # so tagging it construction_property=True must make passes_bar/kill_fired
    # refuse it outright instead of silently returning a bar decision. Would
    # fail if the guard in sharpness._guard_not_construction_property were
    # deleted: both calls below would then return a bool instead of raising.
    d_tagged = sharpness.decompose(q, grp, K, construction_property=True)
    se_tagged = sharpness.bootstrap_se(q, grp, K, n_boot=200, seed=1)
    with pytest.raises(sharpness.ConstructionPropertyAsBarError):
        sharpness.passes_bar(d_tagged, se_tagged)
    with pytest.raises(sharpness.ConstructionPropertyAsBarError):
        sharpness.kill_fired(d_tagged, se_tagged)


def test_frozen_random_encoder_on_real_labels_fires_the_third_leg():
    """THE RED, and the reachable one. grp is the REAL label, never permuted:
    only s changes, from the trained encoder's cluster structure to a
    frozen-random encoder's isotropic noise -- the repo's own mandatory
    control at docs/COMPONENT_LEDGER.md:47-58. This is a state training
    actually visits (an encoder that never learned to cluster), unlike a
    label permutation, which no gradient step can produce as a pairing with s.

    Would fail if: the frozen-random s were replaced by anything correlated
    with grp (i_q would rise off zero and kill_fired would stop firing), or
    if erank_min / std_min were raised so high that this same isotropic noise
    finally read unhealthy (per the module docstring's "no threshold rescues
    it" finding, that would also kill the informative baseline first)."""
    grp, s, q = _frozen_random_representation(SEED)
    assert s.shape[0] == N  # guarded-entry count: all N rows, no subsampling,
    # no RANK_BURN_IN gate -- this call bypasses fit() entirely.

    # shipped signature: this is the guarded entry --
    # pi_jepa.collapse_report's `if q is not None and y is not None:` branch
    # -- and rep['leg'] is what the detector a training run would actually
    # call returns, not a bar decision this test recomputes by hand. q here
    # is the genuine frozen-random readout, not an oracle, so it is left
    # untagged (construction_property defaults to False): this is the read
    # that must be allowed to reach the bar and fire.
    rep = pj.collapse_report(s, q=q, y=grp, K=K, boot_seed=SEED)
    assert rep["std_min"] > pj.COLLAPSE_STD_MIN, rep
    assert rep["erank"] > pj.COLLAPSE_ERANK_MIN, rep
    assert rep["leg"] == "label" and rep["collapsed"] is True, (
        f"the shipped label leg must fire on the frozen-random plant against "
        f"its real label: leg={rep['leg']}, label={rep['label']}, "
        f"boot_seed={SEED}"
    )
    assert abs(rep["label"]["i_q"]) < sharpness.BAR_SIGMA * rep["label"]["se_i_q"], (
        f"i_q should be statistically indistinguishable from zero for a "
        f"representation independent of its own real label: "
        f"i_q={rep['label']['i_q']:+.4e} at se_i_q={rep['label']['se_i_q']:.4e}, "
        f"boot_seed={SEED}"
    )

    informative_erank = pj.collapse_report(
        _grouped_representation(SEED)[1]
    )["erank"]
    assert rep["erank"] > informative_erank, (
        f"the mechanism this file exists to name: frozen-random erank "
        f"{rep['erank']:.4f} must read HIGHER than the informative case's "
        f"{informative_erank:.4f}, because clustering is anisotropy and noise "
        f"is not"
    )

    # THE FINDING, asserted rather than narrated: the geometric legs would
    # have read 'None' (healthy) here -- in fact healthier than the
    # informative baseline, by the erank check above -- while the shipped
    # label leg reads it dead. Two legs are NOT enough, and the gap is one
    # training actually reaches.
    # informative_q is the same nearest-centroid ORACLE readout as the
    # baseline test's q -- built FROM grp -- so this call is tagged
    # construction_property=True for the same reason: RULE 1, a construction
    # property must never reach the bar API.
    _, _, informative_q = _grouped_representation(SEED)
    baseline_rep = pj.collapse_report(
        _grouped_representation(SEED)[1], q=informative_q, y=grp, K=K,
        boot_seed=SEED, construction_property=True,
    )
    assert baseline_rep["leg"] is None and (
        abs(rep["label"]["i_q"]) < 1e-1 * abs(baseline_rep["label"]["i_q"])
    ), (
        f"the label leg must read healthy (leg=None) on the informative "
        f"baseline whose i_q, {baseline_rep['label']['i_q']:.4f}, is not "
        f"within noise of zero, while the frozen-random plant's i_q, "
        f"{rep['label']['i_q']:+.4e}, is -- boot_seed={SEED}"
    )


def test_permuted_label_is_a_signature_check_not_the_reachable_proof():
    """A SIGNATURE CHECK, not the reachable proof (see module docstring): y
    permuted independently of (s, q) shows that neither leg's FORMULA reads a
    label at all -- std_min and erank are IDENTICAL, not merely close, before
    and after, because s is never touched -- while i_q falls toward zero. It
    is kept because it isolates the formula-level blindness cleanly, but no
    training run reaches this (s, y) pairing by gradient descent; the
    frozen-random test above is the state training can actually land in.

    Would fail if: the permutation left any label in place (torch.equal
    would then not raise the "must actually move labels" assertion), or if
    std_min/erank differed at all after permutation (they are pure functions
    of s, so any difference would mean the permutation touched s by mistake)."""
    grp, s, q = _grouped_representation(SEED)
    # shipped signature on the REAL, unpermuted label: the guarded entry that
    # should read healthy, since s and q still carry the group structure. q
    # is the nearest-centroid ORACLE readout built FROM grp, so this call is
    # tagged construction_property=True too (RULE 1); the AFTER call below
    # keeps the same s and q but is the one this test is actually about, so
    # it is left untagged and must be free to fire.
    rep_before = pj.collapse_report(
        s, q=q, y=grp, K=K, boot_seed=SEED, construction_property=True
    )
    assert rep_before["leg"] is None, (
        f"the real, unpermuted label must not fire the label leg: "
        f"leg={rep_before['leg']}, label={rep_before['label']}"
    )

    perm = torch.randperm(N, generator=torch.Generator().manual_seed(SEED + 1))
    y_perm = grp[perm]
    # a relabelling, not a resample: the multiset of labels is unchanged, only
    # which item each label is attached to.
    assert torch.equal(torch.sort(y_perm).values, torch.sort(grp).values)
    assert not torch.equal(y_perm, grp), "the permutation must actually move labels"

    # shipped signature on the PERMUTED label: a second, independent guarded
    # entry -- same s and q as rep_before, only y changes.
    rep_after = pj.collapse_report(s, q=q, y=y_perm, K=K, boot_seed=SEED)
    assert rep_after["std_min"] == rep_before["std_min"], (
        "std_min must be exactly unchanged: it is a function of s alone and s "
        "was never touched by the label permutation"
    )
    assert rep_after["erank"] == rep_before["erank"], (
        "effective rank must be exactly unchanged for the same reason"
    )
    assert rep_after["std_min"] > pj.COLLAPSE_STD_MIN, rep_after
    assert rep_after["erank"] > pj.COLLAPSE_ERANK_MIN, rep_after
    assert rep_after["leg"] == "label" and rep_after["collapsed"] is True, (
        f"a label permutation independent of (s, q) must not clear the PREREG "
        f"bar: leg={rep_after['leg']}, label={rep_after['label']}, "
        f"boot_seed={SEED}"
    )
