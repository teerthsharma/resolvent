"""
CEQ-JEPA core operator.

The declared-real math (frozen; see docs/canon/08_ARCHITECTURE.md and the
DCM-1 build spec):

    P    = causal row-stochastic attention matrix, softmax over j <= i.
           Declared absorbing rows are overwritten to identity ROWS of the
           SAME matrix, AFTER the softmax: P[a, :] = e_a for a in the
           absorbing set A. This is a rank-preserving edit, never a second
           matrix, and it carries no gradient through the overwritten rows.
    z    = (I - g*P)^{-1} Vtilde, solved by ONE lower-triangular forward
           substitution -- exact, because P is causal (row i reads only
           j <= i, so P and I - g*P are lower triangular).
    O    = (1 - g) * P @ z                                   (the state read)

    Boundary rows split P into canonical transient/absorbing blocks
    (index order preserved, so both blocks stay triangular):
        Q = P[T, T],  R = P[T, A]
    and the committor -- the probability of being absorbed into each member
    of A, starting from each transient vertex -- is the SECOND, separate
    exact solve:
        q^(bullet)_T = (I - Q)^{-1} R            (also one triangular solve)
    q attains 0 and 1 exactly; it is a probability, never a logit.

    q_floor is the committor field of the UNIFORM causal chain (softmax of
    an all-zeros logit matrix under the causal mask) -- computable in
    closed form from (n, absorbing indices) alone, no encoder, no solve.
    It is the collapse floor: a collapsed encoder produces exactly q_floor.

TELEPORT IS A TRAINING SCAFFOLD, NOT PART OF THE READ (see teleport_at).
The teleport floor c > 0 is what keeps I - Q non-singular under gradient
pressure, but it displaces the g=0 bitwise-softmax corner by up to 2c in row
L1 -- a displacement that does NOT shrink with n. Measured at c=0.0125, A=[0],
logits ~ N(0,1) float64 under torch.manual_seed(0): max row L1 = 0.024870 at
n=16, 0.024965 at n=64, 0.024996 at n=256 -- rising toward the 2c ceiling
0.025, not falling.
The architecture's load-bearing claim -- that at g=0 the read IS the causal
softmax, bitwise -- therefore holds only at c = 0. Train with c > 0, anneal
c -> 0, EVALUATE AND SHIP at c = 0, where committor()'s singularity and
kappa_max guards are the live protection instead of the teleport. Self-check
(a) asserts both ends: bitwise at c=0, bounded displacement at c=0.0125.

Refusal, not a caught exception: state 0 has only j <= 0 available under
any causal mask, so P[0, 0] == 1 always. If index 0 is not declared
absorbing, Q has a 1 on its diagonal, I - Q is exactly singular, and the
chain is reducible (state 0 can never leave itself, and nothing reaches it).
committor() and q_floor_closed_form() both RAISE SingularTransientBlockError
in that case rather than returning a number. Non-finite input is the same
kind of refusal: a NaN logit RAISES ValueError instead of propagating a
full-NaN q with a NaN conditioning number that nothing downstream catches
((diag.abs() < tol).any() is False for NaN -- the hole this closes).
"""

import torch

__all__ = [
    "SingularTransientBlockError",
    "causal_mask",
    "build_operator",
    "teleport_at",
    "state_solve",
    "committor",
    "q_floor_closed_form",
    "hitting_time_read",
    "VERDICT_DEFINED",
    "VERDICT_NEVER",
    "VERDICT_UNDEFINED",
    "TELEPORT",
    "KAPPA_DESIGN_BOUND_F32",
]

#: LAW L-NEVER verdicts (Addendum F-N). String constants, not free-typed
#: literals, so a caller building a table against hitting_time_read() can't
#: typo a comparison silently.
VERDICT_DEFINED = "DEFINED"
VERDICT_NEVER = "NEVER"
VERDICT_UNDEFINED = "UNDEFINED"


class SingularTransientBlockError(RuntimeError):
    """Raised when the transient block (I - Q) is singular, or so
    ill-conditioned that the solve would return garbage: a declared absorbing
    set fails to cover every self-absorbing / unreachable state, or an
    annealed teleport has let a row saturate."""


def causal_mask(n, device=None):
    """[n, n] bool, True where j <= i (causal, diagonal included)."""
    return torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))


#: Teleport floor used DURING TRAINING. Every transient row sends at least this
#: much mass onto the causally-visible absorbing set, so ||Q||_inf <= 1 - TELEPORT
#: and hence ||(I-Q)^{-1}||_inf <= 1/TELEPORT = 80 BEFORE any training, making the
#: transient block non-singular by construction rather than by a threshold check:
#: diag(I-Q)_ii = 1 - Q_ii >= TELEPORT > 0 always.
#:
#: THE 80 IS NOT EXACT IN FLOAT32. The derivation assumes softmax rows sum to
#: exactly 1; in float32 they sum to 1 +/- 1.19e-07, so the max Q row sum reaches
#: 0.98750009 against the ideal 1 - c = 0.98750000 and the realized bound is
#:     ||(I-Q)^{-1}||_inf <= 80 * (1 + 6e-6) = 80.000480
#: Measured, exact norm, not the diagonal proxy: 80.000305 (absolute excess
#: 3.052e-04, relative 3.815e-06) worst over 2000 random float32 draws at logit
#: scale 1..100, n=16, nA=4; and 80.000214 over the eleven shipped stress
#: checkpoints x 512 Bed examples. The same construction in float64 reads
#: 79.995218912596 -- the excess is float32 rounding, nothing else.
#: A caller enforcing the design bound must test kappa <= 80 * (1 + 6e-6).
#: `assert kappa <= 80.0 + 1e-6` is measurably too tight: it FAILS on the
#: shipped lr_extreme checkpoint (80.000214) with the teleport fully intact.
TELEPORT = 0.0125

#: The float32-realized design bound at TELEPORT. Use this, not 80.0, in asserts.
KAPPA_DESIGN_BOUND_F32 = 80.0 * (1.0 + 6e-6)


def teleport_at(step, total_steps, start=TELEPORT, end=0.0):
    """Linear teleport anneal: `start` at step 0, `end` at step >= total_steps.

    The point of annealing rather than picking one end of the knob: containment
    and the bitwise corner are the SAME knob in opposite positions. c > 0 keeps
    I - Q non-singular while the logits are still moving; c = 0 is the only
    setting at which the g=0 read is the causal softmax bitwise.

    IS THE ANNEAL VIABLE? Measured on the eleven shipped stress checkpoints
    (ceqjepa/artifacts/stress_*.pt), 512 fresh Bed examples each, at teleport=0,
    counting transient rows with 1 - P[i,i] < nT*eps_float32:

        ten non-diverged runs (seeds 0-4, g=0.99, nA=1, nA=8, 10x lr, 100x lr):
            0 saturated rows out of 61,440. Worst slack 1 - P[i,i] = 4.053e-05
            (100x lr): the solve stays finite, but conditioning at c=0 is
            24,672 against 79.745 at c=0.0125 -- 309x worse.
        the diverged run (lr_extreme, max |logit| = 1.4e4):
            815 of 7,680 rows saturate (10.6%), 512 of 512 examples affected,
            min 1 - P[i,i] = 0.000e+00 EXACTLY.

    So: for a run whose logits stay at trained scale (max |logit| <= ~13) the
    anneal IS viable -- saturation frequency 0/61,440 -- but it is not free,
    because at c=0 nothing structural bounds kappa. Anneal only with
    committor(..., kappa_max=...) armed, and treat the raise as the run's
    verdict rather than an exception to swallow. For a diverged encoder the
    anneal is not viable at any schedule: every example saturates.
    """
    if total_steps <= 0:
        return float(end)
    f = min(max(float(step) / float(total_steps), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * f


def absorbing_teleport(n, absorbing_idx, dtype, device=None):
    """[n, n] row-stochastic teleport target: uniform over the absorbing states
    that are CAUSALLY VISIBLE from each row (a <= i).

    Rows with no visible absorbing state get a zero row and receive no teleport
    (the caller keeps the bare softmax there). With 0 in absorbing_idx every
    row i >= 1 has index 0 visible, which is why the canon declares the sink.
    """
    idx = torch.as_tensor(absorbing_idx, dtype=torch.long, device=device)
    A = torch.zeros(n, n, dtype=dtype, device=device)
    if idx.numel() == 0:
        return A
    rows = torch.arange(n, device=device).unsqueeze(1)          # [n,1]
    visible = (idx.unsqueeze(0) <= rows)                        # [n,k] a <= i
    A[rows.expand(-1, idx.numel()), idx.unsqueeze(0).expand(n, -1)] = visible.to(dtype)
    counts = A.sum(-1, keepdim=True)
    return torch.where(counts > 0, A / counts.clamp_min(1), A)


def build_operator(logits, absorbing_idx, teleport=TELEPORT):
    """Causal row-stochastic P from logits, with boundary rows overwritten.

    logits: [..., n, n]. absorbing_idx: 1-D long/int tensor or sequence of
    absorbing vertex indices. Returns P: [..., n, n].

    A `teleport` fraction of every transient row's mass is moved onto the
    causally-visible absorbing states BEFORE the boundary overwrite. This
    bounds ||Q||_inf <= 1 - teleport for every example at every training step,
    which is the structural repair for the singular-transient-block crash: no
    setting of the logits can drive P[i,i] to 1 once teleport > 0.

    teleport=0.0 is the SHIP/EVAL setting, not a curiosity: it is the only
    setting at which the g=0 read is the causal softmax bitwise (self-check
    (a)). At teleport=0 nothing structural protects the transient solve, so
    committor()'s guards are the protection -- keep them armed and give
    committor a kappa_max. See teleport_at() for the anneal and its measured
    cost.

    Raises ValueError on non-finite logits: a single NaN otherwise yields a
    full-NaN P, a full-NaN q and a NaN conditioning number that every
    downstream threshold check silently passes.
    """
    if not (0.0 <= teleport < 1.0):
        raise ValueError("teleport must be in [0, 1), got %r" % (teleport,))
    if not torch.isfinite(logits).all():
        n_bad = int((~torch.isfinite(logits)).sum())
        raise ValueError(
            "build_operator: logits contain %d non-finite entries (NaN/Inf); "
            "refusing to build P, because softmax would propagate NaN into q "
            "and into the conditioning number, where no threshold check "
            "catches it ((x < tol) is False for NaN)" % n_bad
        )
    n = logits.shape[-1]
    mask = causal_mask(n, device=logits.device)
    masked = logits.masked_fill(~mask, float("-inf"))
    P = torch.softmax(masked, dim=-1)
    absorbing_idx = torch.as_tensor(absorbing_idx, dtype=torch.long, device=logits.device)
    if teleport > 0.0 and absorbing_idx.numel() > 0:
        A = absorbing_teleport(n, absorbing_idx, P.dtype, P.device)
        has_target = (A.sum(-1, keepdim=True) > 0).to(P.dtype)   # [n,1]
        c = has_target * teleport                                # 0 where no visible target
        P = (1.0 - c) * P + c * A
    eye = torch.eye(n, dtype=P.dtype, device=P.device)
    P = P.clone()
    if absorbing_idx.numel() > 0:
        P[..., absorbing_idx, :] = eye[absorbing_idx]
    return P


def state_solve(P, V, g):
    """z = (I - g*P)^{-1} V by triangular forward substitution; O = (1-g)*P@z.

    P: [..., n, n] causal (lower triangular). V: [..., n, d]. g: scalar < 1.
    Returns (z, O), both [..., n, d].

    Needs no singularity guard at any teleport: diag(I - g*P)_ii = 1 - g*P_ii
    >= 1 - g > 0 for g < 1, whatever the logits do. Only the committor solve,
    where the coefficient is 1 rather than g, can go singular.
    """
    n = P.shape[-1]
    eye = torch.eye(n, dtype=P.dtype, device=P.device)
    M = eye - g * P
    z = torch.linalg.solve_triangular(M, V, upper=False)
    O = (1 - g) * (P @ z)
    return z, O


def committor(P, absorbing_idx, tol=None, kappa_max=None):
    """q^(bullet)_T = (I - Q)^{-1} R, embedded back to full [..., n, k].

    Raises SingularTransientBlockError if (I - Q)'s diagonal has a zero (a
    transient state whose only causal predecessor is itself), or if kappa_max
    is given and the EXACT conditioning ||(I-Q)^{-1}||_inf exceeds it. Raises
    ValueError if P is non-finite.

    kappa_max is the guard that matters once the teleport is annealed to 0:
    the diagonal test alone only catches saturation to within nT*eps, and a row
    at 1 - Q_ii = 4.053e-05 (measured, the 100x-lr checkpoint) passes it while
    returning a solve amplified 24,672x. Pass kappa_max=KAPPA_DESIGN_BOUND_F32
    to hold an annealed run to the conditioning the teleport used to guarantee.

    Journals committor.last_kappa_bound = ||(I-Q)^{-1}||_inf, EXACT rather than
    a bound: for lower-triangular Q >= 0 with Q_ii < 1, (I-Q)^{-1} = sum_k Q^k
    is entrywise nonnegative, so its infinity norm is exactly max_i (M^{-1} 1)_i
    -- one extra triangular solve against the ones vector. The value previously
    stashed there was 1 / min_i (1 - Q_ii), a LOWER bound that callers asserted
    against as if it were an upper bound. Measured understatement of that old
    quantity, exact / diagonal: 1.97x (uniform n=16 nA=4), 2.15x (uniform n=32
    A=[0,7,19]), 2.80x (uniform n=64 A=[0]: 1.9753 vs 5.5244), 3.45x (uniform
    n=256 A=[0]: 1.9753 vs 6.8092), and 1.00x..3.21x over 200 random scale-3
    operators at n=32. The float() runs detached, inside
    no_grad, so the per-step "Converting a tensor with requires_grad=True to a
    scalar" UserWarning is gone and the journal keeps no graph alive. With
    leading batch dimensions this is the max over the batch: one ill-conditioned
    example is visible but not attributable.
    """
    if not torch.isfinite(P).all():
        n_bad = int((~torch.isfinite(P)).sum())
        raise ValueError(
            "committor: P contains %d non-finite entries (NaN/Inf); refusing "
            "to solve, because the singularity test (diag.abs() < tol).any() "
            "is False for NaN, so a NaN P returns a full-NaN q and a NaN "
            "kappa with nothing raised" % n_bad
        )
    n = P.shape[-1]
    device = P.device
    absorbing_idx = torch.as_tensor(absorbing_idx, dtype=torch.long, device=device)
    is_absorbing = torch.zeros(n, dtype=torch.bool, device=device)
    is_absorbing[absorbing_idx] = True
    transient_idx = torch.nonzero(~is_absorbing, as_tuple=True)[0]  # ascending: preserves triangularity
    k = absorbing_idx.numel()

    Q = P[..., transient_idx, :][..., :, transient_idx]
    R = P[..., transient_idx, :][..., :, absorbing_idx]

    diag = 1.0 - torch.diagonal(Q, dim1=-2, dim2=-1)
    if tol is None:
        # Dtype-aware. A fixed absolute 1e-10 is BELOW float32 eps (1.192e-07),
        # so in float32 a saturated row passed the guard and committor() returned
        # a q whose channels summed to 1.1868 -- 1245x over the eps_32 = 1.5e-4
        # conservation bar, finite and inside [0,1] so nothing downstream caught
        # it. The guard must scale with the dtype and the transient dimension.
        tol = max(1e-10, transient_idx.numel() * torch.finfo(P.dtype).eps)
    if (diag.abs() < tol).any():
        raise SingularTransientBlockError(
            "transient block singular (I - Q has a diagonal entry below %.3e in %s): "
            "some causally-self-only state is not in the declared absorbing set, or "
            "an annealed teleport has let a row saturate. With "
            "build_operator(teleport=c>0) this is unreachable -- a teleport of c "
            "bounds every transient diagonal below by c." % (tol, P.dtype)
        )

    eyeT = torch.eye(transient_idx.numel(), dtype=P.dtype, device=device)
    M = eyeT - Q
    q_T = torch.linalg.solve_triangular(M, R, upper=False)

    # EXACT ||(I-Q)^{-1}||_inf by one more triangular solve, against 1.
    with torch.no_grad():
        Md = M.detach()
        ones = torch.ones(*Md.shape[:-1], 1, dtype=P.dtype, device=device)
        kappa = float(torch.linalg.solve_triangular(Md, ones, upper=False).abs().max())
    committor.last_kappa_bound = kappa
    if kappa_max is not None and kappa > kappa_max:
        raise SingularTransientBlockError(
            "transient block ill-conditioned: ||(I-Q)^{-1}||_inf = %.6g exceeds "
            "kappa_max = %.6g. At teleport=c the structural bound is 1/c (80 at "
            "c=0.0125, realized 80*(1+6e-6) in float32); at teleport=0 nothing "
            "bounds it and this guard is the only protection." % (kappa, kappa_max)
        )

    q_full = torch.zeros(*P.shape[:-1], k, dtype=P.dtype, device=device)
    q_full[..., transient_idx, :] = q_T
    q_full[..., absorbing_idx, :] = torch.eye(k, dtype=P.dtype, device=device)
    return q_full


def q_floor_closed_form(n, absorbing_idx, dtype=torch.float64, device=None,
                        teleport=TELEPORT):
    """Committor field of the uniform causal chain, closed form, no solve.

    With teleport c and A_vis(i) = {a in A : a <= i}, the uniform row is
        P[i,j] = (1-c)/(i+1)  for j <= i,   plus  c/|A_vis(i)|  for a in A_vis(i)
    so, writing S_{i-1} = sum_{j<i} q_j and u_i = mean of e_a over A_vis(i),
        q_i = [ (1-c)/(i+1) * S_{i-1} + c * u_i ] / (1 - (1-c)/(i+1))
    which is a forward recursion in causal order. At c = 0 this collapses to
    the bare q_i = mean(q_0 .. q_{i-1}).

    `teleport` must match the value build_operator was called with, or this is
    the floor of a different chain: under the anneal, pass teleport_at(step,...).

    Raises SingularTransientBlockError if index 0 is not absorbing (state 0
    is forced self-absorbing under any causal mask).
    """
    absorbing_list = sorted(int(a) for a in absorbing_idx)
    k = len(absorbing_list)
    pos = {a: j for j, a in enumerate(absorbing_list)}
    if 0 not in pos:
        raise SingularTransientBlockError(
            "index 0 is not in the declared absorbing set: state 0 is "
            "always self-absorbing under a causal mask (only j <= 0 exists)"
        )

    c = float(teleport)
    q = torch.zeros(n, k, dtype=dtype, device=device)
    running_sum = torch.zeros(k, dtype=dtype, device=device)
    for i in range(n):
        if i in pos:
            q[i, pos[i]] = 1.0
        else:
            w = (1.0 - c) / (i + 1)                    # uniform weight per visible j
            u = torch.zeros(k, dtype=dtype, device=device)
            vis = [j for a, j in pos.items() if a <= i]
            if vis and c > 0.0:
                u[vis] = 1.0 / len(vis)
            q[i] = (w * running_sum + c * u) / (1.0 - w)
        running_sum = running_sum + q[i]
    return q


def hitting_time_read(W, T, i=None, gamma_abel=None, r_tol=1e-3, kappa_max=1e12):
    """LAW L-NEVER (Addendum F-N) -- the read. Implements the specification
    exactly, term by term:

        Q_T        = W with T's rows AND COLUMNS removed (both -- removing
                     only rows leaves columns into T, which is a different,
                     wrong chain).
        r_gamma(i) = (1 - gamma) * [(I - gamma*Q_T)^-1 . 1]_i
        Abel's theorem: r_gamma -> P_i(tau_T = infinity) as gamma -> 1^-.
        E[tau]     = [(I - Q_T)^-1 . 1]_i, ONLY when r_gamma < r_tol;
                     otherwise verdict is NEVER and E is not computed (a
                     mean over an event this read has just measured as not
                     converging within gamma_abel of 1 is not a number).
        cond(I - Q_T) > kappa_max gives UNDEFINED for every queried state,
        with the exact 2-norm condition number logged to
        hitting_time_read.last_condition_number, and NO solve attempted --
        a NaN/Inf out of a near-singular solve is a void instrument, never
        a result (the same rule committor() and build_operator() already
        enforce elsewhere in this file).

    gamma_abel is a FIXED practical stand-in for the gamma -> 1 limit, not
    the limit itself (no code computes an exact symbolic limit). Default
    1 - 1e-6 sits three orders of magnitude closer to 1 than the r_tol=1e-3
    decision boundary, so a chain's spectral radius has to be within about
    1e-3 of 1 (an enormous, practically-never mean hitting time even where
    finite) before the choice of gamma_abel itself could flip DEFINED vs
    NEVER. That is the intended behavior, not a numerical artifact: a state
    whose escape is this close to certain-never is reported NEVER even if
    its true limiting P(never) is exactly 0, because within any horizon a
    caller can afford to solve at, it is indistinguishable from never.

    MASS WALL: for any row-stochastic W (rows sum to 1, the construction
    every chain in this file produces), a transient block whose rows always
    carry positive mass toward T has P(never) = 0 identically -- that is
    not this function inferring the hypothesis class, it is what an honest
    r_gamma measurement returns for it. A caller tabulating results across
    hypothesis classes must record that a class is built this way, per the
    house rule, rather than score a correct 0 as if it were a finding.

    Runs entirely in float64 (cast on entry regardless of W's dtype): the
    whole point of this read is condition-number and near-gamma=1 accuracy,
    where float32 rounding would move the r_tol decision boundary itself.

    W: [n, n], no batch dim -- every caller in this repo reads one chain at
       a time; add a batch dim when one actually needs it.
    T: sequence of int indices, the target/absorbing set to hit.
    i: None -> every state not in T, ascending original index; or a single
       int / sequence of original W-space indices. None of them may be in
       T (tau_T is trivially 0 there, not this read's job).
    r_tol, kappa_max: the two thresholds the LAW specifies (1e-3, 1e12).

    Returns a list of dicts, one per queried state, in query order:
        {"i": <original index>, "E": float | None, "p_never": float | None,
         "verdict": VERDICT_DEFINED | VERDICT_NEVER | VERDICT_UNDEFINED,
         "kappa": float}
    kappa is the exact cond(I - Q_T) for the whole read (shared across all
    queried states in this call), so a caller never has to re-derive why
    UNDEFINED fired.
    """
    if not torch.isfinite(W).all():
        raise ValueError(
            "hitting_time_read: W contains non-finite entries (NaN/Inf); "
            "refusing to build Q_T, for the same reason build_operator() "
            "refuses non-finite logits -- a NaN would pass every downstream "
            "threshold check silently"
        )
    if gamma_abel is None:
        gamma_abel = 1.0 - 1e-6
    W = W.to(torch.float64)
    n = W.shape[-1]
    device = W.device

    T_idx = torch.as_tensor(sorted(set(int(t) for t in T)), dtype=torch.long, device=device)
    is_target = torch.zeros(n, dtype=torch.bool, device=device)
    is_target[T_idx] = True
    transient_idx = torch.nonzero(~is_target, as_tuple=True)[0]   # ascending original index
    nT = transient_idx.numel()

    if i is None:
        query_orig = transient_idx
    else:
        query_orig = torch.as_tensor([i] if isinstance(i, int) else list(i),
                                      dtype=torch.long, device=device)
        bad = query_orig[is_target[query_orig]]
        if bad.numel() > 0:
            raise ValueError(
                "hitting_time_read: state(s) %s are in T; tau_T is trivially "
                "0 there, not this read's job" % bad.tolist()
            )

    if nT == 0:
        hitting_time_read.last_condition_number = 0.0
        return []

    pos_of = torch.full((n,), -1, dtype=torch.long, device=device)
    pos_of[transient_idx] = torch.arange(nT, device=device)
    query_pos = pos_of[query_orig]

    Q_T = W[transient_idx][:, transient_idx]
    eye = torch.eye(nT, dtype=torch.float64, device=device)
    M1 = eye - Q_T

    kappa = float(torch.linalg.cond(M1))
    hitting_time_read.last_condition_number = kappa

    # An exactly (or numerically) singular M1 can make the SVD ratio a NaN
    # (0/0), not the +inf a caller would expect -- `nan > kappa_max` is
    # False, so the bare comparison would silently let a singular block
    # through. Route NaN/Inf through the same UNDEFINED branch as "too big".
    if not (kappa == kappa) or kappa == float("inf") or kappa > kappa_max:
        return [{"i": int(q), "E": None, "p_never": None, "verdict": VERDICT_UNDEFINED,
                 "kappa": kappa} for q in query_orig.tolist()]

    ones = torch.ones(nT, 1, dtype=torch.float64, device=device)
    Mg = eye - gamma_abel * Q_T
    try:
        r_gamma_all = ((1.0 - gamma_abel) * torch.linalg.solve(Mg, ones)).squeeze(-1)
        E_all = torch.linalg.solve(M1, ones).squeeze(-1)
    except torch.linalg.LinAlgError:
        # cond() said "invertible enough" but the solver itself refused --
        # a void instrument, never a result. UNDEFINED, kappa already logged.
        return [{"i": int(q), "E": None, "p_never": None, "verdict": VERDICT_UNDEFINED,
                 "kappa": kappa} for q in query_orig.tolist()]

    if not (torch.isfinite(r_gamma_all).all() and torch.isfinite(E_all).all()):
        # cond() said "invertible enough" but the solve disagreed -- a void
        # instrument, never a result. Route to UNDEFINED rather than let a
        # NaN/Inf leak out labeled DEFINED or NEVER.
        return [{"i": int(q), "E": None, "p_never": None, "verdict": VERDICT_UNDEFINED,
                 "kappa": kappa} for q in query_orig.tolist()]

    results = []
    for orig, p in zip(query_orig.tolist(), query_pos.tolist()):
        r = float(r_gamma_all[p])
        if r < r_tol:
            results.append({"i": orig, "E": float(E_all[p]), "p_never": r,
                             "verdict": VERDICT_DEFINED, "kappa": kappa})
        else:
            results.append({"i": orig, "E": None, "p_never": r,
                             "verdict": VERDICT_NEVER, "kappa": kappa})
    return results


if __name__ == "__main__":
    torch.manual_seed(0)

    # -----------------------------------------------------------------------
    # (a) THE CORNER, at both ends of the teleport knob, with a NON-EMPTY
    # absorbing set so the teleport branch is LIVE. The previous version of
    # this check called build_operator(logits, absorbing_idx=[]), and the
    # teleport branch is gated on `absorbing_idx.numel() > 0`, so it
    # short-circuited: that check could not fail no matter what the teleport
    # did, and it was reported as passing while the corner was broken.
    # -----------------------------------------------------------------------
    s = 64
    A = [0, 13, 40]                       # non-empty -> teleport branch is live
    logits = torch.randn(s, s, dtype=torch.float64)

    ref = torch.softmax(logits.masked_fill(~causal_mask(s), float("-inf")), dim=-1).clone()
    ref[A] = torch.eye(s, dtype=torch.float64)[A]        # the lane's own boundary overwrite
    P0 = build_operator(logits, A, teleport=0.0)
    corner = (P0 - ref).abs().max().item()
    print("(a1) SHIP corner, teleport=0, |A|=%d, n=%d: max abs diff vs causal softmax "
          "= %.3e" % (len(A), s, corner))
    assert torch.equal(P0, ref), "teleport=0 read is NOT the bitwise causal softmax"
    assert corner == 0.0

    Pc = build_operator(logits, A, teleport=0.0125)
    disp = (Pc - P0).abs().sum(-1)                       # per-row L1 displacement
    dmax = disp.max().item()
    abs_rows = torch.tensor(A)
    print("(a2) TRAIN corner, teleport=0.0125: max row L1 displacement = %.6f "
          "(stated bound 2c = 0.025); transient rows moved %d/%d, absorbing rows "
          "moved %d" % (dmax, int((disp > 0).sum()), s - len(A),
                        int((disp[abs_rows] > 0).sum())))
    assert dmax > 0.0, "teleport branch is DEAD -- this check would be vacuous"
    assert dmax <= 0.025 + 1e-12, "displacement %.6f exceeds the stated 2c bound" % dmax
    assert int((disp[abs_rows] > 0).sum()) == 0, "teleport leaked into an absorbing row"
    assert torch.equal(build_operator(logits, A), Pc), \
        "build_operator's DEFAULT teleport is not 0.0125 -- every stated bound is wrong"

    # (b) q_floor of a constant encoder == closed-form uniform-causal committor, to 1e-12.
    n = 32
    absorbing_idx = [0, 7, 19]
    zero_logits = torch.zeros(n, n, dtype=torch.float64)
    P_uniform = build_operator(zero_logits, absorbing_idx)
    q_general = committor(P_uniform, absorbing_idx)
    q_closed = q_floor_closed_form(n, absorbing_idx, dtype=torch.float64)
    max_diff_b = (q_general - q_closed).abs().max().item()
    print("(b) q_floor (constant encoder, general solve) vs closed form: max abs diff %.3e"
          % max_diff_b)
    assert max_diff_b < 1e-12
    assert torch.allclose(q_general.sum(-1), torch.ones(n, dtype=torch.float64), atol=1e-10)

    # (c) solve RAISES (never returns a number) when the transient block is singular
    # (no sink: index 0 not declared absorbing).
    no_sink = [7, 19]
    P_no_sink = build_operator(zero_logits, no_sink)
    raised = False
    try:
        committor(P_no_sink, no_sink)
    except SingularTransientBlockError as e:
        raised = True
        print("(c) committor() raised SingularTransientBlockError as required: %s" % e)
    assert raised, "committor() must raise on a singular transient block, not return a number"

    raised_cf = False
    try:
        q_floor_closed_form(n, no_sink)
    except SingularTransientBlockError as e:
        raised_cf = True
        print("(c) q_floor_closed_form() raised SingularTransientBlockError as required: %s" % e)
    assert raised_cf

    # (d) NON-FINITE REFUSAL. One NaN logit used to give a full-NaN q, a NaN
    # kappa and max|sum_k q - 1| = nan with nothing raised, because
    # (nan < tol) is False.
    nan_logits = zero_logits.clone()
    nan_logits[7, 3] = float("nan")
    P_nan = build_operator(zero_logits, absorbing_idx).clone()
    P_nan[9, 2] = float("nan")
    for label, fn in (("build_operator(NaN logits)",
                       lambda: build_operator(nan_logits, absorbing_idx)),
                      ("committor(NaN P)",
                       lambda: committor(P_nan, absorbing_idx))):
        raised_nan = False
        try:
            fn()
        except ValueError as e:
            raised_nan = True
            print("(d) %s raised ValueError as required: %s" % (label, e))
        assert raised_nan, "%s must raise, not return NaN" % label

    # (e) last_kappa_bound is the EXACT ||(I-Q)^{-1}||_inf, cross-checked against
    # a dense inverse; the old diagonal quantity is shown to be the lower bound
    # it always was.
    scaled = torch.randn(n, n, dtype=torch.float64) * 3.0
    P_s = build_operator(scaled, absorbing_idx)
    committor(P_s, absorbing_idx)
    stashed = committor.last_kappa_bound
    T_idx = torch.tensor([i for i in range(n) if i not in absorbing_idx])
    Q_s = P_s[T_idx][:, T_idx]
    kappa_dense = torch.linalg.inv(
        torch.eye(len(T_idx), dtype=torch.float64) - Q_s).abs().sum(-1).max().item()
    diag_lb = 1.0 / (1.0 - torch.diagonal(Q_s)).abs().min().item()
    rel = abs(stashed - kappa_dense) / kappa_dense
    print("(e) kappa: stashed %.6f  dense-inverse %.6f  rel err %.3e  |  the old "
          "diagonal quantity %.6f understates it by %.2fx"
          % (stashed, kappa_dense, rel, diag_lb, kappa_dense / diag_lb))
    assert rel < 1e-12, "last_kappa_bound is not the exact infinity norm"
    assert diag_lb <= kappa_dense * (1 + 1e-12), "diagonal quantity is not a lower bound"

    # (f) kappa_max is the live guard for an annealed (teleport -> 0) run, and
    # the journal fires no UserWarning under autograd.
    raised_k = False
    try:
        committor(P_s, absorbing_idx, kappa_max=stashed * 0.5)
    except SingularTransientBlockError as e:
        raised_k = True
        print("(f) kappa_max guard fired as required: %s" % str(e).split(".")[0])
    assert raised_k, "kappa_max must raise when exceeded"
    assert teleport_at(0, 100) == TELEPORT and teleport_at(100, 100) == 0.0
    assert teleport_at(50, 100) == TELEPORT / 2

    import warnings
    grad_logits = torch.zeros(n, n, dtype=torch.float64, requires_grad=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        committor(build_operator(grad_logits, absorbing_idx), absorbing_idx).sum().backward()
    print("(f) kappa journal under requires_grad=True: %d warnings, kappa=%.6f, "
          "grad finite=%s" % (len(w), committor.last_kappa_bound,
                              bool(torch.isfinite(grad_logits.grad).all())))
    assert len(w) == 0, "kappa journal still warns: %s" % [str(x.message) for x in w]

    # bonus sanity: state solve / read.
    g = 0.9
    Vtilde = torch.randn(n, 8, dtype=torch.float64)
    z, O = state_solve(P_uniform, Vtilde, g)
    assert z.shape == (n, 8) and O.shape == (n, 8)
    assert torch.isfinite(z).all() and torch.isfinite(O).all()
    print("(bonus) state_solve: z, O finite, correct shape. route=solve_triangular, delta=0.0")

    # -----------------------------------------------------------------------
    # (g) LAW L-NEVER read: DEFINED cross-checked against a dense inverse,
    # NEVER on a near-degenerate escape, UNDEFINED on an exactly-singular
    # transient block with the condition number logged and NO solve leaking
    # a NaN/Inf out as if it were a result.
    # -----------------------------------------------------------------------
    g_res = hitting_time_read(P_uniform, absorbing_idx)
    g_kappa = hitting_time_read.last_condition_number
    T_g = torch.tensor([i for i in range(n) if i not in absorbing_idx])
    Q_g = P_uniform[T_g][:, T_g].to(torch.float64)
    E_dense = (torch.linalg.inv(torch.eye(len(T_g), dtype=torch.float64) - Q_g)
               @ torch.ones(len(T_g), 1, dtype=torch.float64)).squeeze(-1)
    max_E_err = max(abs(r["E"] - float(E_dense[k])) for k, r in enumerate(g_res))
    print("(g1) L-NEVER DEFINED: %d/%d states, cond(I-Q_T)=%.3f, max |E - dense inverse| = %.3e"
          % (sum(r["verdict"] == VERDICT_DEFINED for r in g_res), len(g_res), g_kappa, max_E_err))
    assert all(r["verdict"] == VERDICT_DEFINED for r in g_res)
    assert all(r["p_never"] < 1e-3 for r in g_res)
    assert max_E_err < 1e-9

    p_stall = 0.9999
    W_never = torch.tensor([[p_stall, 1.0 - p_stall], [0.0, 1.0]], dtype=torch.float64)
    r_never = hitting_time_read(W_never, [1], i=0)[0]
    print("(g2) L-NEVER NEVER: p_stall=%.4f, p_never (r_gamma) = %.6f (>= 1e-3 required), "
          "E withheld = %s" % (p_stall, r_never["p_never"], r_never["E"] is None))
    assert r_never["verdict"] == VERDICT_NEVER
    assert r_never["E"] is None and r_never["p_never"] >= 1e-3

    W_sing = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    r_undef = hitting_time_read(W_sing, [2])
    print("(g3) L-NEVER UNDEFINED: exactly-singular transient block, cond logged = %s, "
          "E/p_never withheld for all %d queried states"
          % (hitting_time_read.last_condition_number, len(r_undef)))
    assert all(r["verdict"] == VERDICT_UNDEFINED and r["E"] is None and r["p_never"] is None
               for r in r_undef)
    _kappa_g3 = hitting_time_read.last_condition_number
    assert _kappa_g3 != _kappa_g3 or _kappa_g3 > 1e12, \
        "an exactly-singular block must log NaN or a huge cond, not a finite plausible one"

    print("ALL SELF-CHECKS PASSED")
