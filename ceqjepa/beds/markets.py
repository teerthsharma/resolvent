"""ceqjepa/beds/markets.py

MARKETS bed -- the phase DCM-1 was designed for. A YES/NO contract
resolving IS two absorbing sets, and the committor q = (I-Q)^{-1}R
(ceqjepa.operator.committor, unchanged, reused here rather than
reimplemented) IS the implied probability of resolving YES. This module
builds the market's transition graph BY HAND (the transition
probabilities are already known -- p, 1-p -- there is nothing to learn
them from), then hands the resulting P straight to
ceqjepa.operator.committor for the exact solve. No new solver is written.

DEFAULT PATH: no network, no API key. A CRR binomial price lattice over T
steps generates its own path and its own exact resolution label; q* is
exact by construction, not a numerical approximation.

Chart layout (matches ceqjepa.operator's convention: the first nA indices
are absorbing, and index 0 is forced self-absorbing under any causal mask
because state 0 sees only j <= 0):
    index 0        = NO  (absorbing)
    index 1        = YES (absorbing)
    index 2 .. n-1 = transient lattice nodes (s, j): s = steps remaining
                     to resolution (1..T), j = up-moves taken so far
                     (0..T-s). A node's two children sit at s-1 < s, so
                     they always land at a strictly smaller index --
                     that is what makes the hand-built P lower
                     triangular, the same causal shape build_operator
                     produces from logits.

THE PLANTED DEPARTURE FROM THE MARTINGALE, so the bed is not trivially
solved by reading the current price: the SAME graph is built twice, at
two different up-probabilities --
    p_true  = the physical probability that actually generates the label
              (a drift added on top of the risk-neutral rate)
    p_star  = (1-d)/(u-d), the risk-neutral / martingale-implied
              probability a zero-drift market would price the contract
              at
q_star (the training label) is committor(P_true)'s read; the price
control is committor(P_star)'s read at the SAME query node. Under the
martingale hypothesis the market-implied price IS the unconstrained YES
committor -- so this price control, not a constant predictor, is the
floor a model must beat. Both are computed once (the lattice is fixed;
there is nothing to re-solve per batch) and reported side by side below.

LEAKAGE GUARDS A REAL VENUE WOULD NEED (the synthetic bed needs none of
these today -- every path and its label are generated in the same
process call, so there is no wall-clock gap for information to leak
across):
    - timestamps: a real snapshot must carry its observation time and
      never be paired with a resolution, or a later price, that a trader
      at that time could not yet see.
    - resolution-time censoring: an outcome must not be visible to the
      model before the market's own resolution time -- joining settled
      outcomes back onto pre-resolution snapshots is the standard leak.
    - survivorship: a corpus built only from contracts that reached
      resolution (dropping delisted / cancelled / still-open ones)
      overstates predictability, because the hard-to-resolve tail was
      removed before the model ever saw it.
"""
import math

import torch

import ceqjepa.operator as op

NO, YES = 0, 1  # absorbing indices, canonical: NO must be 0 (state 0 is forced self-absorbing)


class MarketsBed:
    """A fixed CRR binomial-lattice YES/NO contract. batch(gen, B) matches
    ceqjepa.train.Bed's interface: (x, x_nx, q_star, v_idx)."""

    def __init__(self, T=8, S0=100.0, sigma=0.2, dt=1.0, drift=0.05,
                 threshold=None, dtype=torch.float64):
        assert T >= 1
        self.T = T
        self.S0 = float(S0)
        self.dtype = dtype
        self.u = math.exp(sigma * math.sqrt(dt))
        self.d = 1.0 / self.u
        self.p_star = (1.0 - self.d) / (self.u - self.d)          # zero-drift martingale prob
        self.p_true = min(max(self.p_star + drift, 1e-3), 1 - 1e-3)  # planted real-world drift
        self.threshold = self.S0 if threshold is None else float(threshold)
        self.nA = 2
        self.absorbing_idx = torch.tensor([NO, YES])

        # index layout: transient block s=1..T, j=0..T-s, offsets increasing with s
        offset, acc = {}, 2
        for s in range(1, T + 1):
            offset[s] = acc
            acc += (T - s + 1)
        self.n = acc
        self._offset = offset
        self._node_of = {}
        for s in range(1, T + 1):
            for j in range(0, T - s + 1):
                self._node_of[offset[s] + j] = (s, j)

        self.P_true = self._build_P(self.p_true)
        self.P_star = self._build_P(self.p_star)
        self.q_true = op.committor(self.P_true, self.absorbing_idx)   # [n,2], the exact label table
        self.q_price = op.committor(self.P_star, self.absorbing_idx)  # [n,2], the price-control table

    def _idx(self, s, j):
        return self._offset[s] + j

    def _terminal_price(self, k):
        """Price after all T steps with k total up-moves."""
        return self.S0 * (self.u ** k) * (self.d ** (self.T - k))

    def _terminal_label(self, k):
        return YES if self._terminal_price(k) >= self.threshold else NO

    def _build_P(self, p):
        """Hand-built causal, row-stochastic P for one fixed up-probability p.
        Absorbing rows are identity rows, exactly per operator's boundary
        convention; every transient row places mass only on strictly
        smaller indices (its two lattice children), so P is lower
        triangular by construction -- no softmax, no teleport needed:
        transient rows have zero self-mass, so (I-Q) is never singular."""
        n = self.n
        P = torch.zeros(n, n, dtype=self.dtype)
        P[NO, NO] = 1.0
        P[YES, YES] = 1.0
        for s in range(1, self.T + 1):
            for j in range(0, self.T - s + 1):
                i = self._idx(s, j)
                if s == 1:
                    lbl_down = self._terminal_label(j)
                    lbl_up = self._terminal_label(j + 1)
                    P[i, lbl_down] += (1.0 - p)
                    P[i, lbl_up] += p
                else:
                    P[i, self._idx(s - 1, j)] = 1.0 - p
                    P[i, self._idx(s - 1, j + 1)] = p
        return P

    def batch(self, gen, B):
        """One batch, ZERO solves (the lattice is fixed; committor was
        solved once, in __init__, for both p_true and p_star). Returns
        (x[B,3] f32, x_nx[B,3] f32, q_star[B,2] f32, v_idx[B])."""
        v_idx = torch.randint(2, self.n, (B,), generator=gen)
        s_j = [self._node_of[int(v)] for v in v_idx]
        s_t = torch.tensor([s for s, _ in s_j], dtype=self.dtype)
        j_t = torch.tensor([j for _, j in s_j], dtype=self.dtype)
        t_t = self.T - s_t                                   # steps already taken

        x = self._features(t_t, j_t)
        noise = 0.01 * torch.randn(B, generator=gen, dtype=self.dtype)
        x = x + torch.stack([noise, torch.zeros(B, dtype=self.dtype),
                              torch.zeros(B, dtype=self.dtype)], dim=-1)

        up = (torch.rand(B, generator=gen, dtype=self.dtype) < self.p_true).to(self.dtype)
        j_next, t_next = j_t + up, t_t + 1
        x_nx = self._features(t_next, j_next)

        q_star = self.q_true[v_idx]
        return x.float(), x_nx.float(), q_star.float(), v_idx

    def _features(self, t, j):
        """[len,3]: log-moneyness, time fraction elapsed, up-move fraction."""
        price = self.S0 * (self.u ** j) * (self.d ** (t - j))
        log_m = torch.log(price / self.threshold)
        time_frac = t / self.T
        up_frac = torch.where(t > 0, j / t.clamp_min(1), torch.zeros_like(t))
        return torch.stack([log_m, time_frac, up_frac], dim=-1)

    def price_control(self, v_idx):
        """The martingale-implied YES probability at these query nodes --
        the floor a model must beat, per the architecture spec, not a
        constant predictor."""
        return self.q_price[v_idx].float()


if __name__ == "__main__":
    torch.manual_seed(0)
    bed = MarketsBed(T=8, S0=100.0, sigma=0.2, drift=0.05)
    print(f"[markets] T={bed.T} n_states={bed.n} (2 absorbing + "
          f"{bed.T * (bed.T + 1) // 2} transient) u={bed.u:.6f} d={bed.d:.6f} "
          f"p_star(martingale)={bed.p_star:.6f} p_true(planted drift)={bed.p_true:.6f} "
          f"threshold={bed.threshold:.2f}")

    # self-check: P is row-stochastic and causal; q is a real probability.
    for name, P in (("P_true", bed.P_true), ("P_star", bed.P_star)):
        row_sums = P.sum(-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-10), \
            f"{name} is not row-stochastic"
        assert torch.equal(P, torch.tril(P)), f"{name} is not lower triangular (causal)"
    for name, q in (("q_true", bed.q_true), ("q_price", bed.q_price)):
        assert (q >= -1e-10).all() and (q <= 1 + 1e-10).all(), f"{name} outside [0,1]"
        assert torch.allclose(q.sum(-1), torch.ones(bed.n, dtype=bed.dtype), atol=1e-10), \
            f"{name} channels don't sum to 1"
    print("[markets] self-check: P_true/P_star row-stochastic + causal, "
          "q_true/q_price valid probabilities (sum to 1, in [0,1])")

    gen = torch.Generator().manual_seed(0)
    x, x_nx, q_star, v_idx = bed.batch(gen, 8)
    print("\n[markets] real batch, B=8, columns = [log_moneyness, time_frac, up_frac]:")
    print("  v_idx  :", v_idx.tolist())
    print("  x      :\n", x)
    print("  x_nx   :\n", x_nx)
    print("  q_star (P(NO), P(YES)):\n", q_star)

    eval_gen = torch.Generator().manual_seed(1)
    B_eval = 2048
    _, _, q_star_eval, v_idx_eval = bed.batch(eval_gen, B_eval)
    price_ctrl = bed.price_control(v_idx_eval)
    const_ctrl = q_star_eval.mean(0, keepdim=True).expand_as(q_star_eval)
    mse_price = ((price_ctrl - q_star_eval) ** 2).mean().item()
    mse_const = ((const_ctrl - q_star_eval) ** 2).mean().item()
    S_price_vs_const = 1.0 - mse_price / mse_const
    print(f"\n[markets] PRICE-CONTROL COMPARISON, eval_n={B_eval}:")
    print(f"  THE FLOOR: under the martingale hypothesis the market-implied price IS "
          f"the unconstrained YES committor -- the model must beat mse_price, not mse_const.")
    print(f"  mse(const predictor {const_ctrl[0].tolist()} vs q_star) = {mse_const:.6e}")
    print(f"  mse(price control (p_star={bed.p_star:.4f}) vs q_star (p_true={bed.p_true:.4f})) "
          f"= {mse_price:.6e}")
    print(f"  S(price vs const) = {S_price_vs_const:+.4f}  "
          f"(price is a much stronger control than the constant, but not zero-error, "
          f"because of the planted drift p_true - p_star = {bed.p_true - bed.p_star:+.4f})")
    assert mse_price < mse_const, "price control should beat the constant predictor by construction"
    assert mse_price > 1e-9, "price control should NOT be a perfect (zero-error) predictor -- " \
        "the planted drift must leave it something to lose to"
    print("\n[markets] ALL SELF-CHECKS PASSED")
