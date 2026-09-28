# P2-DJ bar — do candidate rankings outlive state prediction past T*? (registered 2026-09-27, before any run)

The question D-JEPA (arXiv 2609.24749) raises for Canon: its "decision-local prediction gap" scores a
model on the *ranking* of candidate futures, not on the state. On a chaotic world, does the ranking
of candidate actions stay predictable after the state has stopped being predictable? This bed
measures the **Bayes ceiling**: a perfect model, a Monte Carlo posterior, and no learning anywhere.
Every run records the sha256 of this file.

## World

- Lorenz-63 (sigma 10, rho 28, beta 8/3), RK4, dt 0.01, exactly as `sun/sun_maths.py`;
  lambda = 0.9059 (S2). T*(delta) = ln(1/delta)/lambda (S2's law, Delta = 1).
- delta in {1e-2, 1e-4}: T* = 5.08 and 10.17.
- Starts: burn-in 5,000 steps from (1,1,1), then states spaced by 200-600 steps (as S2).
  N = 2,000 starts per (family, delta, seed); seeds {0, 1, 2}.
- Observation y = x0 + delta*xi, xi ~ N(0, I3). Posterior ensemble: M = 64 members y + delta*xi_m
  (flat-prior posterior; ponytail: ignores the attractor prior, exact only as delta -> 0).
- K = 4 candidate actions, two families:
  - `impulse`: at t = 0 add u_k to the state, u_k = 1.0 * a unit vector drawn at random per start
    (transient consequence).
  - `forcing`: add f_k to dx/dt for all t, f_k = (k - 1.5)*2 * e_x, i.e. {-3, -1, +1, +3} e_x
    (persistent consequence).
- Consequence of action k at lead t: J_k(t) = x-coordinate of the actioned trajectory. Truly best
  = argmax_k J_k(t) on the true x0. Common random numbers across k.
- Leads: t/T* in {0.25, 0.5, ..., 3.0} (12 leads).

## Readings

- `hit_bayes(t)`: fraction of starts where argmax_k mean_m J_k^m(t) equals the truly best action.
- `hit_blind(t)`: the state-blind policy — pick argmax_k of the climatological mean of J_k(t) (from
  4,000 independent climatological starts per seed, no observation). For `impulse` this is chance
  1/K by symmetry; for `forcing` it is not.
- Excess E(t) = hit_bayes(t) - hit_blind(t): the ranking information that needs the state.
- `S(t)`: state MSE skill of the ensemble mean for the actioned trajectories, averaged over k:
  1 - E|mean - truth|^2 / E|truth - clim mean|^2.
- State horizon t_S: the first lead with S(t) < 0.1. Ranking horizon t_R: the first lead with
  E(t) < 0.1 * E(t_first).
- Spearman rank correlation between the posterior means and the true J_k, beside the top-1 hit.

## Bars (deciding: mean over 3 seeds)

- **Prediction (lead):** rankings do not outlive the state beyond their state-blind part: mean of
  E(t) over leads t >= 1.5 * t_S is <= 0.03, in both families at both delta.
- **Counter (D-JEPA-favourable):** mean of E(t) over leads t >= 1.5 * t_S is >= 0.08 in at least one
  family at both delta — the ranking carries state information past the state horizon.
- **Horizon scaling (secondary):** t_R(1e-4) - t_R(1e-2) is within +-30% of ln(100)/lambda = 5.08
  in the `impulse` family (the decision horizon is Lyapunov-governed).
- Neither prediction nor counter: OPEN, with the number.
- Null check that must fire: at t/T* = 3 in `impulse`, hit_bayes is within 0.03 of 1/K = 0.25.
  If it does not, the bed is broken and every row is void.

## Amendment A1 (2026-09-27, after House's shortcut hunt, before any run of `dj.py`)

Declared: House ran a pilot before this amendment (N = 300, M = 32, delta = 1e-2, one seed, point J
only). It saw E(t >= 2 T*) = 0 within noise for point J, and 0.17–0.30 for a cumulative consequence.
Every item below is therefore **post-pilot**. Where an item could be chosen for its outcome, that is
stated.

1. **Leads:** readings on an absolute grid every 0.25 time units, out to 4.5 T*. A T*-relative lead
   reads the nearest grid point. If t_S is never reached, or the band t >= 1.5 t_S holds fewer than 4
   T*-relative leads (step 0.25 T*), that cell is **void**.
2. **Bayes rule:** hit_bayes uses the top-1 Bayes rule argmax_k (1/M) sum_m 1[k = argmax_j J_j^m].
   The posterior-mean argmax is reported beside it.
3. **Blind policy:** the same vote rule with the posterior replaced by 64 climatological states under
   the same actions (and, for `impulse`, the same offsets). S(t) uses each action's own
   climatological mean.
4. **Consequences:**
   - point J (as registered, deciding);
   - **J^win** = mean of x over [t - 0.5 T*, t] (new, unpiloted, deciding alongside point J);
   - J^cum = mean of x over [0, t] (House's proposal; **descriptive only**, because it was piloted
     and because its ranking contains the predictable segment before T* by construction).

   The prediction and the counter are read on point J and on J^win separately, with the thresholds
   unchanged.
5. **Horizon scaling:** t_R is read on the absolute 0.25 grid, with E linearly interpolated.
6. **Ceiling check (must fire):** hit_bayes(0.25 T*) >= 0.95 in both families at both delta, on point
   J. If it fails, the bed is broken.
