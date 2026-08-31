"""MARS standing attack #3 -- the skyline leaks the oracle.

FILED AT it.0, bound by Saturn, against `CEQ_V15_CONTRACT.md` PART III (BED-M's
"Skyline = the gated scan at 4,769 matched (m = 93)") and PART IV ("distance-to-
native-skyline" as a per-cell column every R-measurement reports).

THE MECHANISM, and where it is NOT what it first looks like. `BED-M`'s chain
family sets `a` (the per-step gate) and `b` (the per-step driver) DIRECTLY as
channels of `x` -- `scale/negation_scope.py` documents this explicitly and
repeatedly ("NO NEW CHANNEL. The encoding is the chain family's own: CH_DRIVE
carries the Rademacher coefficients a, CH_FLIP the Gaussian drivers b",
`:520-522`) and R1's own kill-condition text anticipates it ("diagnose by linear
probe on log a (should be near-exact)"). Raw, per-position `a_i` sitting in `x`
is therefore BY DESIGN, not a leak -- an attack that fired on that alone would be
attacking a documented, intentional observational task (a strawman this file
does not file).

The real risk is one order up. The SKYLINE is an oracle-informed reference: it
is allowed to know the true `a` exactly and compute the exact resolvent from it,
because it exists to define a ceiling, not to be a fair arm. If a COMPOSED
quantity that only the oracle should be able to produce -- a cumulative log-gate
sum, a partial resolvent output, anything of Neumann-order >= 1 -- ends up
sitting in `x` as well, then any arm scored "close to skyline" could be reading
the skyline's own answer off its input rather than composing anything, and
"distance-to-native-skyline" stops meaning what the round says it means. THE
NUMBER THAT WOULD BE WRONG: a small distance-to-skyline, credited to the arm's
architecture, that is actually the arm (or a shared corpus/caching path) reading
a pre-composed answer out of `x`.

CLASS. MISTAKES.md D (design-level failure), a SHARPER INSTANCE of D-2 ("an
oracle that is the arm's own resolvent"). D-2's rule is "write down what
computes the label and what computes the prediction; if they are the same
operator, the comparison is void." This attack extends that rule from a SHARED
OPERATOR to a SHARED CHANNEL: the comparison is equally void if the label's
computed intermediates leak into the compared object's input tensor, even when
the operators themselves differ.

STATUS, run at time of filing. `scale/negation_scope.py`'s chain family
(`make_equilibrium_batch`, `equilibrium_hop_reading`) is the live corpus/oracle
infrastructure the contract's "BED-M (Markov, exists)" refers to -- it is on
disk today, unlike BED-K and the interventional channel, which the contract
schedules as fresh builds. The composed-quantity containment census below
therefore RUNS FOR REAL against it now (not a SKIP) and reads clean: no channel
of `x` contains the composed (order>=1) oracle signal, by exact match or by
linear-probe R^2. The must-fire fixtures show the same census catching a planted
composed-quantity leak, and the v15-specific skyline module (not yet built) is
handled by its own SKIP.
"""
from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale.negation_scope import CH_DRIVE, CH_FLIP, make_equilibrium_batch  # noqa: E402

torch.manual_seed(0)


# ================================================================ THE CENSUS
def linear_r2(probe: np.ndarray, target: np.ndarray) -> float:
    """R^2 of the best linear fit `target ~ a*probe + b`. A REAL correlation
    check on REAL numbers -- least squares, not string matching."""
    probe = np.asarray(probe, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if probe.std() < 1e-12:
        return 0.0
    A = np.stack([probe, np.ones_like(probe)], axis=1)
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    pred = A @ coef
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def channel_containment_census(x: torch.Tensor, oracle_signal: torch.Tensor,
                                *, exact_tol: float = 1e-6,
                                r2_thresh: float = 0.999,
                                positions=None) -> dict:
    """For every channel `c` of `x`, and every position `i` in `positions`
    (default: all), does `x[:, i, c]` CONTAIN `oracle_signal[:, i]` -- exactly,
    or as a near-perfect linear function of it?

    Two containment modes, both real value checks: (a) exact numeric equality
    within `exact_tol`, fraction of (batch, position) pairs; (b) least-squares
    R^2 of a linear probe, pooled over the batch at each position. A channel
    flags as CONTAINS if either mode clears its threshold at ANY position --
    "never appear" is a claim about every position, not a lucky one.

    `positions` matters here specifically: `equilibrium_hop_reading`'s own
    recurrence RESETS wherever `a=0` (a design choice, `head` and earlier), so
    at those positions the "composed" signal degenerates to `z=b`, which is
    itself raw order-0 `CH_FLIP` -- a true but uninteresting match that has
    nothing to do with leakage. Restricting to the live band (`i > head`) is
    what makes a hit here mean "a real multi-step composition is sitting in x
    verbatim," which is the actual claim under test.
    """
    n, s, d = x.shape
    pos_iter = range(s) if positions is None else positions
    hits = {}
    for c in range(d):
        for i in pos_iter:
            xc = x[:, i, c].detach().cpu().numpy()
            o = oracle_signal[:, i].detach().cpu().numpy()
            exact_frac = float(np.mean(np.abs(xc - o) < exact_tol))
            r2 = linear_r2(xc, o)
            if exact_frac > 0.999 or r2 > r2_thresh:
                hits[(c, i)] = {"exact_frac": exact_frac, "r2": r2}
    return hits


def assert_no_containment(hits: dict, *, context: str = "") -> None:
    assert not hits, (
        f"composed oracle signal is recoverable from x at {len(hits)} "
        f"(channel, position) pair(s){' -- ' + context if context else ''}: "
        f"{dict(list(hits.items())[:5])}{' ...' if len(hits) > 5 else ''}. "
        f"An arm or skyline reading these channels does not need to compose "
        f"anything; it can copy the answer.")


# ============================================== INDEPENDENT COMPOSED SIGNAL
def composed_oracle_signal(x: torch.Tensor) -> torch.Tensor:
    """The order>=1 quantity a skyline is allowed to know and an arm's raw
    input is not: the running (1-hop) partial resolvent at every prefix
    length, `equilibrium_hop_reading(x[:, :i+1], k=1)` for each `i`.

    Deliberately routed through the corpus module's OWN PUBLIC function
    (`equilibrium_hop_reading`) rather than through the channel indices this
    file could otherwise hardcode -- the point of this attack is "a real
    containment check on actual tensors, not an assertion about a variable
    name," so the ground truth here is obtained the same way a downstream
    consumer would obtain it: by calling the corpus's own oracle, not by
    reading `CH_DRIVE`/`CH_FLIP` and asserting against itself.
    """
    n, s, _ = x.shape
    out = torch.zeros(n, s)
    for i in range(1, s + 1):
        # k = i-1 caps the truncation at the whole visible prefix, so this is
        # the full running composition up to position i-1, not a fixed budget.
        out[:, i - 1] = _equilibrium_hop_reading_prefix(x[:, :i, :], k=i - 1)
    return out


def _equilibrium_hop_reading_prefix(x_prefix: torch.Tensor, k: int) -> torch.Tensor:
    from scale.negation_scope import equilibrium_hop_reading
    return equilibrium_hop_reading(x_prefix, k)


# =================================================================== TESTS
S, D_MODEL, N = 10, 16, 32


def _batch(t_star: int = 4, seed: int = 0):
    #: `d` here is `make_batch`'s flipper DISTANCE (must satisfy 1<=d<s-1 and
    #: not collide with the payload at s-2), not the embedding width --
    #: `d_model` is the keyword for that.
    x, y, head, p = make_equilibrium_batch(N, S, 2, t_star=t_star, d_model=D_MODEL,
                                            seed=seed)
    return x, y, head, p


def test_raw_per_step_gate_is_present_by_documented_design_not_a_leak():
    """CONTROL, not an attack. Confirms `a` (order-0, single-step) is FUNCTIONAL
    input to the corpus's own public oracle -- not a check on `x[:, :, CH_DRIVE]`
    against itself (that would be circular: the same tensor equals itself by
    construction), but a black-box perturbation test through
    `equilibrium_oracle`, the module's own public API.

    Overwriting only channel `CH_DRIVE` at one live-band position must move the
    oracle's output; if it did not, `a` would not really be read as the gate
    and the boundary this file draws (order-0 disclosed, order>=1 not) would
    have nothing to stand on.
    """
    from scale.negation_scope import equilibrium_oracle

    x, y, head, p = _batch()
    live_pos = head + 1
    assert live_pos < S - 1, "fixture must leave the live band nonempty"
    base = equilibrium_oracle(x, head, p)
    bumped = x.clone()
    g = torch.Generator().manual_seed(999)
    bumped[:, live_pos, CH_DRIVE] = torch.randint(0, 2, (N,), generator=g).float() * 2 - 1
    moved = equilibrium_oracle(bumped, head, p)
    frac_moved = float((base != moved).float().mean())
    assert frac_moved > 0.05, (
        f"perturbing CH_DRIVE at a live-band position moved only "
        f"{frac_moved:.3f} of outputs; the channel documented as the gate does "
        f"not behave like one")


def test_composed_order1_signal_is_absent_from_x_on_the_live_bedm_corpus():
    """THE REAL ATTACK, run today. Not a SKIP: `scale/negation_scope.py` exists
    and is the corpus the contract's BED-M refers to. The composed (order>=1)
    running-resolvent signal, obtained through the module's own public
    `equilibrium_hop_reading` rather than by naming a channel, must not be
    recoverable from any channel of x at any position.

    READS CLEAN at time of filing over the LIVE BAND (positions `head+1..S-1`,
    where the recurrence actually composes more than one step -- see
    `channel_containment_census`'s docstring for why positions at or before
    `head` are excluded). This is reported as a finding, not assumed -- see the
    printed count in `demo()`.
    """
    x, y, head, p = _batch()
    oracle_signal = composed_oracle_signal(x)
    live = range(head + 1, S)
    assert len(live) >= 2, "fixture must leave a real multi-step live band"
    hits = channel_containment_census(x, oracle_signal, positions=live)
    assert_no_containment(hits, context="scale.negation_scope chain family, live band")


def test_containment_census_MUST_FIRE_on_a_planted_composed_leak():
    """FIRES. Same batch, one unused noise channel overwritten with the
    composed oracle signal -- exactly the shape a caching bug or an
    accidentally-shared buffer would produce. The census must find it and the
    real assertion must raise on it."""
    x, y, head, p = _batch()
    oracle_signal = composed_oracle_signal(x)
    leaky = x.clone()
    plant_channel = D_MODEL - 1
    assert plant_channel not in (CH_DRIVE, CH_FLIP), (
        "fixture must not overwrite a channel already asserted clean by the "
        "control test")
    leaky[:, :, plant_channel] = oracle_signal
    live = range(head + 1, S)

    hits = channel_containment_census(leaky, oracle_signal, positions=live)
    assert any(c == plant_channel for c, _ in hits), (
        f"planted leak at channel {plant_channel} was not detected: {hits}")
    with pytest.raises(AssertionError, match="composed oracle signal is recoverable"):
        assert_no_containment(hits)


def test_containment_census_MUST_FIRE_on_a_rescaled_planted_leak():
    """A second must-fire: the leak is an affine transform of the composed
    signal (`2.5 * oracle - 0.3`), not a verbatim copy, so the census has to be
    shown catching it via the R^2 probe rather than the exact-match branch
    alone -- an exact-equality-only check would miss a rescaled leak and still
    look green."""
    x, y, head, p = _batch()
    oracle_signal = composed_oracle_signal(x)
    leaky = x.clone()
    plant_channel = D_MODEL - 2
    leaky[:, :, plant_channel] = 2.5 * oracle_signal - 0.3
    live = range(head + 1, S)

    hits = channel_containment_census(leaky, oracle_signal, exact_tol=1e-9, positions=live)
    flagged = [(c, i) for (c, i) in hits if c == plant_channel]
    assert flagged, f"rescaled leak at channel {plant_channel} was not detected"
    assert all(hits[k]["exact_frac"] < 0.5 for k in flagged), (
        "fixture is wrong: an affine-rescaled leak should not exact-match, so a "
        "detection here that comes from the exact branch is not testing R^2 at all")


def _find_v15_skyline():
    for mod_name, attr in (
        ("ceq.beds.bed_m", "skyline"),
        ("ceq.beds.bed_m", "gated_scan_skyline"),
        ("scale.arm_pl", "skyline"),
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn
    return None


def test_composed_signal_absent_from_x_fed_to_the_v15_skyline():
    """The forward-looking half of the attack. SKIPS -- the v15-specific
    skyline implementation (the "gated scan at 4,769 matched, m=93" the
    contract predicts) has not been built: no `ceq.beds.bed_m` /
    `scale.arm_pl` skyline function is importable, `V15_LEDGER.md` NEXT is
    `it.3-4`, and BED-M's skyline is scheduled for the same it.6-7 window as
    ARM PL. When it lands, this test runs the same census used above against
    whatever tensor it reads `x` from, with no rewrite needed.
    """
    fn = _find_v15_skyline()
    if fn is None:
        pytest.skip(
            "no v15 skyline function found (tried ceq.beds.bed_m.skyline / "
            "gated_scan_skyline, scale.arm_pl.skyline). BED-M's skyline has not "
            "been built yet (V15_LEDGER.md NEXT=it.3-4). See "
            "test_containment_census_MUST_FIRE_* above for proof the census "
            "catches a planted composed-signal leak on the live chain corpus.")
    x, y, head, p = _batch()
    oracle_signal = composed_oracle_signal(x)
    hits = channel_containment_census(x, oracle_signal, positions=range(head + 1, S))
    assert_no_containment(hits, context="v15 skyline input")


def demo() -> None:
    x, y, head, p = _batch()
    oracle_signal = composed_oracle_signal(x)
    live = range(head + 1, S)
    hits = channel_containment_census(x, oracle_signal, positions=live)
    assert not hits, hits
    leaky = x.clone()
    leaky[:, :, D_MODEL - 1] = oracle_signal
    leak_hits = channel_containment_census(leaky, oracle_signal, positions=live)
    assert any(c == D_MODEL - 1 for c, _ in leak_hits)
    print(f"demo OK: live chain corpus, {len(live) * D_MODEL} (channel,position) "
          f"live-band pairs checked, 0 contain the composed order>=1 signal; "
          f"planted leak at channel {D_MODEL - 1} detected at {len(leak_hits)} pair(s)")


if __name__ == "__main__":
    demo()
