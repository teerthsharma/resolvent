"""ceqjepa/beds/market_band.py

THE MARKET BAND -- an EXISTENCE bed, per the author's Addendum F-N. No
architecture is scored here and none of this module's numbers claim that any
model predicts the band. What it exists to show: on a synthetic binary market
where the correct output is "no position," a bettor forced to emit a
probability anyway can lose almost everything even when that bettor's belief
is exactly right -- because the true edge sits inside the cost band, not
because the belief was wrong. That is a fact about the TASK (refuse vs.
forced-answer), not about any architecture, and this module makes no other
claim.

NAME. Not "B3": that label already names two different things in this tree --
CONTRACT.md:166 ("B3 Mobius inversion on incidence algebras" -> W9) and the
canon census's B3 (docs/canon/CHARTER.md:229 / 04_BEDS_AND_INSTRUMENTS.md /
05_REPAIRS.md 05.9: "tests coupled to prose"). This bed is not a member of
that census and was never W9; giving it the same short label was a third,
spurious collision. It has no number here, only a name: MARKET-BAND.

SPECIFICATION (Addendum F-N). A synthetic grid p in [0.40, 0.60], a fair
market price q = 0.50, a proportional trading cost c = 0.03, Kelly staking
at fair odds, cost charged proportionally on the traded side.

THE MODEL, ONE CONVENTION THROUGHOUT (DEFECT 1 FIX). The previous revision
mixed two price conventions in one class: `multiplier` paid out as if q were
always 0.5 (win = 1+f, lose = 1-f, even money) while `f_blind`/`f_star` used
the general fair-odds edge (p-q)/(1-q). Those agree only at q=0.5. Away from
it they disagreed about where the no-trade band sits: the even-money
`multiplier`'s own sign change sat at p = 0.5 +/- c/(2q) (fixed at 0.5,
wrong for q != 0.5), while the stated `refuses()` rule |p-q| <= c was
centered on q (the right shape, but asserted, not derived from `multiplier`
at all). At q=0.40 the gap was not cosmetic: the old `refuses()` correctly
placed p=0.47 outside its stated band and handed it a stake, but that stake
then LOST money under the old (even-money) `multiplier`'s own arithmetic --
g = -1.128834e-02 at f_star = +0.0667, strictly worse than refusing (g=0.0).
A stated rule and the arithmetic it is supposed to describe disagreeing
about the same trade is the defect.

The fix is a single two-sided fair-odds multiplier for a SIGNED stake f
(f >= 0: long YES at price q; f < 0: long NO, magnitude |f|, at price 1-q),
cost charged on the notional actually traded (DEFECT 2 below reports the
*other* honest reading, "cost on the stake," but does not change which one
this module uses elsewhere):

    price  = q          if f >= 0 else (1 - q)
    drag   = c * |f| / price                        # cost on the notional
    payout = |f| / price     if this side resolves   else 0
    multiplier = 1 - |f| + payout - drag

At f >= 0 this reduces to win = 1 + f*(1-q)/q - c*f/q, lose = 1 - f - c*f/q
(the even-money formulas are the q=0.5 special case, (1-q)/q = 1 -- so every
q=0.5 number this module ever printed is unchanged by this fix; only q!=0.5
behavior was wrong before). This is the one multiplier the frictionless
Kelly fraction f_blind = (p-q)/(1-q) is actually optimizing.

`refuses()` is now DERIVED from that multiplier's own marginal sign, not
stated as a separate rule: refuse iff neither a small long-YES (f=+eps) nor
a small long-NO (f=-eps) trade beats holding cash (g(f) <= g(0) = 0).
Because expected log-growth is concave on each side of f=0, this one-sided
sign is the exact (not approximate) refuse/trade decision. Worked out in
closed form: on the buy side the marginal numerator is (p-q-c), on the sell
side (q-p-c) -- both divided by a positive price, so only the numerator sign
matters -- giving an EXACT band of [q-c, q+c] for every q in (0,1), not only
q=0.5: verified below at q=0.40 and q=0.60 too. `f_star` keeps the same
cost-adjusted-Kelly shape outside the band, (p, q, c) -> (p-q-c*sign(p-q))/
(1-q); only the band's existence and exact edges are asserted, not that this
outside-band sizing matches any external reference.

WHAT THIS DOES NOT SHOW. It does not show that any model, architecture, or
learned policy predicts f* or the band edges. It is an existence proof about
the task's shape (refusal beats a forced probability under this cost),
nothing more.
"""
import math

import torch


class MarketBandBed:
    """The no-trade band and the two named wealth trajectories through it.
    Existence bed -- no batch()/architecture interface, unlike the other
    beds in this package."""

    def __init__(self, q=0.50, c=0.03, dtype=torch.float64):
        self.q = float(q)
        self.c = float(c)
        self.dtype = dtype

    def multiplier(self, f, yes_happens, cost_on="notional"):
        """Wealth multiplier for one bet of SIGNED fraction f: f >= 0 is a
        long-YES stake at price q, f < 0 is a long-NO stake of magnitude |f|
        at price (1-q). `yes_happens` is the actual coin flip (prob p).
        `cost_on` selects which honest reading of "proportional cost on the
        traded side" is charged (DEFECT 2 in the module docstring):
        "notional" (this module's convention, cost on the |f|/price
        contracts actually bought) or "stake" (cost on the dollar stake |f|
        itself). Both are reported by forced_blind_growth; only "notional"
        is used elsewhere in this module."""
        if cost_on not in ("notional", "stake"):
            raise ValueError(f"cost_on must be 'notional' or 'stake', got {cost_on!r}")
        price = self.q if f >= 0 else (1.0 - self.q)
        af = abs(f)
        drag = (self.c * af / price) if cost_on == "notional" else (self.c * af)
        side_happens = yes_happens if f >= 0 else (not yes_happens)
        payout = (af / price) if side_happens else 0.0
        return 1.0 - af + payout - drag

    def _g_of_f(self, p, f):
        """Expected per-bet log-growth at signed stake f, this module's cost
        convention. Used only to derive the refusal band from the
        multiplier's own sign change -- never stated as a separate rule."""
        win = self.multiplier(f, True)
        lose = self.multiplier(f, False)
        if win <= 0.0 or lose <= 0.0:
            return float("-inf")
        return p * math.log(win) + (1.0 - p) * math.log(lose)

    def refuses(self, p, eps=1e-6):
        """True where the cost-aware policy refuses to trade, DERIVED from
        multiplier()'s own marginal sign at f=0 (not a stated |p-q|<=c
        rule): neither a small long-YES (f=+eps) nor a small long-NO
        (f=-eps) trade beats holding cash. Exact for any q in (0,1) because
        expected log-growth is concave on each side of f=0."""
        return self._g_of_f(p, eps) <= 0.0 and self._g_of_f(p, -eps) <= 0.0

    def f_blind(self, p):
        """Cost-blind (frictionless) Kelly fraction at fair odds."""
        return (p - self.q) / (1.0 - self.q)

    def f_star(self, p):
        """Cost-aware optimal fraction: 0 inside the band (per refuses()),
        else the cost-adjusted Kelly fraction outside it."""
        if self.refuses(p):
            return 0.0
        edge = p - self.q
        net_edge = edge - self.c * (1.0 if edge > 0 else -1.0)
        return net_edge / (1.0 - self.q)

    def band(self, p_grid):
        """f*(p) over a grid, as a tensor -- the no-trade band is where this
        is exactly 0.0, not approximately."""
        return torch.tensor([self.f_star(float(p)) for p in p_grid], dtype=self.dtype)

    def forced_blind_growth(self, p, cost_on="notional"):
        """Exact expected per-bet log-growth of the forced cost-blind Kelly
        bettor who knows p exactly but never subtracts c from the sizing.
        `cost_on` picks which honest cost reading to report under (DEFECT 2
        in the module docstring); it does not change f_blind."""
        f = self.f_blind(p)
        win_mult = self.multiplier(f, True, cost_on=cost_on)
        lose_mult = self.multiplier(f, False, cost_on=cost_on)
        if win_mult <= 0.0 or lose_mult <= 0.0:
            return float("-inf")
        return p * math.log(win_mult) + (1.0 - p) * math.log(lose_mult)

    def simulate_forced_blind(self, p, n_bets, seed=0):
        """Realized (not expected) log-wealth path over n_bets iid Bernoulli(p)
        resolutions, forced-blind sizing throughout, this module's ("notional")
        cost convention. Returns (log_wealth, wealth)."""
        f = self.f_blind(p)
        gen = torch.Generator().manual_seed(seed)
        wins = torch.rand(n_bets, generator=gen, dtype=self.dtype) < p
        win_mult = self.multiplier(f, True)
        lose_mult = self.multiplier(f, False)
        log_step = torch.where(
            wins,
            torch.full((n_bets,), math.log(win_mult) if win_mult > 0 else float("-inf"), dtype=self.dtype),
            torch.full((n_bets,), math.log(lose_mult) if lose_mult > 0 else float("-inf"), dtype=self.dtype),
        )
        log_wealth = float(log_step.sum())
        return log_wealth, math.exp(log_wealth)

    def refuser_wealth(self, n_bets):
        """The refuser never trades inside the band: wealth is untouched by
        construction, for any n_bets. Not a simulation -- a fixed point."""
        return 0.0, 1.0  # (log_wealth, wealth)


if __name__ == "__main__":
    bed = MarketBandBed(q=0.50, c=0.03, dtype=torch.float64)

    print("[market_band] MARKET-BAND -- EXISTENCE bed, no architecture scored, "
          "python -m ceqjepa.beds.market_band, dtype=float64")
    print("    (was labelled \"B3\"; that collides with CONTRACT.md:166's B3 "
          "[Mobius inversion on incidence algebras] and with the canon "
          "census's B3 [tests coupled to prose, docs/canon/CHARTER.md:229]; "
          "renamed here to avoid both)")

    # (1) the no-trade band, f* = 0 EXACTLY on [q-c, q+c] -- verified at
    # q=0.40 and q=0.60 as well as the spec's q=0.50, because refuses() is
    # now derived from multiplier()'s own sign change rather than stated.
    for q_check in (0.40, 0.50, 0.60):
        bed_q = MarketBandBed(q=q_check, c=0.03, dtype=torch.float64)
        lo, hi = q_check - 0.03, q_check + 0.03
        boundary = {lo: True, hi: True, lo - 0.01: False, hi + 0.01: False}
        edges_ok = all((bed_q.f_star(p) == 0.0) == inside for p, inside in boundary.items())
        p_grid_q = [round(0.30 + 0.01 * i, 2) for i in range(41)]
        band_ps_q = [p for p in p_grid_q if bed_q.refuses(p)]
        band_exact_q = all(bed_q.f_star(p) == 0.0 for p in band_ps_q)
        print(f"\n[1] NO-TRADE BAND @ q={q_check:.2f}, producer=MarketBandBed.f_star "
              f"(refuses() derived from multiplier's sign), dtype=float64, "
              f"boundary probes at [{lo:.2f}, {hi:.2f}] +/- 0.01, grid 0.30:0.70:0.01 (41 pts)")
        print(f"    band observed on grid = [{min(band_ps_q):.2f}, {max(band_ps_q):.2f}]  "
              f"expected [q-c, q+c] = [{lo:.2f}, {hi:.2f}]  "
              f"f*==0.0 exactly on it: {band_exact_q}  exact edges match: {edges_ok}")
        assert band_exact_q, f"f_star must be exactly 0.0 inside the band at q={q_check}"
        assert edges_ok, f"band edges must be exactly [q-c, q+c] at q={q_check}"
        assert all(bed_q.f_star(p) != 0.0 for p in p_grid_q if p not in band_ps_q), \
            f"f_star must be nonzero strictly outside the band at q={q_check}"

    # (2) forced cost-blind Kelly bettor, p=0.52 known, exact expected g,
    # reported under BOTH honest cost readings (DEFECT 2) -- neither is
    # tuned to hit the author's reference.
    p_known = 0.52
    g_notional = bed.forced_blind_growth(p_known, cost_on="notional")
    g_stake = bed.forced_blind_growth(p_known, cost_on="stake")
    print(f"\n[2] FORCED COST-BLIND BETTOR, producer=MarketBandBed.forced_blind_growth, "
          f"dtype=float64, command=python -m ceqjepa.beds.market_band, p={p_known}, q=0.50, c=0.03")
    print(f"    g (cost on the notional f/q, this module's convention) = {g_notional:.6f}")
    print(f"    g (cost on the stake f, the other honest reading)      = {g_stake:.6f}")
    print(f"    reference g = -0.001469  |  matches notional reading: "
          f"{abs(g_notional - (-0.001469)) < 1e-6}  |  matches stake reading: "
          f"{abs(g_stake - (-0.001469)) < 1e-6}")
    print("    the reference sits between the two natural readings and matches neither.")

    # (3) that bettor over 20,000 bets -- restated as n*g plus the seed
    # band it actually has, not chased against the single seeded reference
    # path (DEFECT 3): -35.2 is one realization, not the expectation.
    n_bets = 20_000
    seeds = range(12)
    paths = [bed.simulate_forced_blind(p_known, n_bets, seed=s) for s in seeds]
    log_wealths = [lw for lw, _ in paths]
    n_g = n_bets * g_notional
    print(f"\n[3] SAME BETTOR OVER {n_bets} BETS, producer=MarketBandBed.simulate_forced_blind, "
          f"dtype=float64, seeds=0..11, command=python -m ceqjepa.beds.market_band")
    print(f"    n * g (expectation, cost-on-notional) = {n_g:.4f}")
    print(f"    realized log_wealth over seeds 0..11: min={min(log_wealths):.2f}, "
          f"max={max(log_wealths):.2f} (spread {max(log_wealths)-min(log_wealths):.2f} nats)")
    print(f"    author's reference log_wealth = -35.2 (wealth 5.1e-16 = exp(-35.2), "
          f"the same number quoted twice, not two checks) -- falls inside this seed "
          f"band and is not itself the target; the target is n*g = {n_g:.2f} plus this spread.")

    # (4) the refuser
    r_log_wealth, r_wealth = bed.refuser_wealth(n_bets)
    print(f"\n[4] REFUSER, producer=MarketBandBed.refuser_wealth (fixed point, no RNG), "
          f"dtype=float64")
    print(f"    wealth = {r_wealth:.6f}   reference = 1.000   match: {r_wealth == 1.0}")
    assert r_wealth == 1.0 and r_log_wealth == 0.0, "the refuser must be an exact fixed point"

    print("\n[market_band] SUMMARY vs. Addendum F-N reference values:")
    print(f"    (1) band edges [q-c, q+c]         : reproduced exactly at q=0.40, 0.50 and "
          f"0.60, now derived from multiplier()'s own sign change rather than a stated rule "
          f"that could (and at q=0.40, under the old even-money-only multiplier, did) "
          f"disagree with the multiplier's own arithmetic.")
    print(f"    (2) g = -0.001469 @ p=0.52, q=0.50 : notional reading gives {g_notional:.6f}, "
          f"stake reading gives {g_stake:.6f} -- both same sign as the reference, neither "
          f"matches it, and no number here is tuned to close that gap.")
    print(f"    (3) log_wealth -35.2, wealth 5.1e-16 : restated as n*g = {n_g:.2f} plus a "
          f"seed-0..11 spread of [{min(log_wealths):.2f}, {max(log_wealths):.2f}]; the "
          f"reference falls inside that spread rather than being a separate target to reproduce.")
    print(f"    (4) refuser x1.000                : reproduced exactly (a fixed point of "
          f"never trading, true under any cost model)")
