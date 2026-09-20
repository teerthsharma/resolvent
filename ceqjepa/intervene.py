"""
do(a): the intervention machinery.

An intervention clamps ONE position's outgoing row of the causal operator to a
chosen move:

    P' = P + e_i (r_a - P_i)^T

a rank-1 edit. Restricted to the transient block T (ascending, so triangularity
survives) with t the position of i inside T:

    Q' = Q + e_t d^T,      d  = r_a[T] - Q[t]
    R' = R + e_t dR^T,     dR = r_a[A] - R[t]
    M  = I - Q,  M' = I - Q' = M - e_t d^T

Sherman-Morrison on M', with c = M^{-1} e_t (column t of the inverse) and
w = M^{-T} d (the inverse contracted with d):

    den        = 1 - d^T M^{-1} e_t = 1 - w[t]
    M'^{-1}    = M^{-1} + c w^T / den
    M'^{-1}e_t = c (1 + w[t]/den) = c / den          (since den + w[t] = 1)

so the intervened committor is the base committor plus ONE outer product:

    q'_T = M'^{-1} R' = q_T + c (R^T w + dR)^T / den                    (*)

DERIVED, and asserted in check (f): d is supported on T-indices <= t (r_a is
causal), so M' stays lower triangular, and the upper-triangular back-substitution
for w gives w[j] = 0 for j > t and w[t] = d[t] / (1 - Q[t,t]). Therefore

    den = (1 - P'[i,i]) / (1 - P[i,i])

-- the ratio of the new to the old diagonal slack. The update goes singular
exactly when the clamped row makes i self-absorbing, which is the same
degeneracy operator.committor() already refuses. It is not a new failure mode,
it is the old one seen through the update.

COST. Everything before (*) -- q_T, c, and the triangular structure -- is shared
by every candidate move. m candidates cost one shared solve plus one m-column
back-substitution for W, then m outer products; not m independent re-solves.
Whether that is FASTER is a measured question, not a derived one, because P is
triangular and a re-solve is already O(n^2) rather than O(n^3). Check (e)
measures it and prints the ratio whichever way it falls.
"""

import torch

from ceqjepa.operator import SingularTransientBlockError

__all__ = ["clamp_row", "committor_do", "committor_do_batch", "delta_q"]


def _check_rows(rows, i, n):
    """rows: [..., n]. Raises unless every row is a causal probability row for
    position i: nonneg, sums to 1, and ZERO on every j > i."""
    if rows.shape[-1] != n:
        raise ValueError("row has length %d, operator has n = %d" % (rows.shape[-1], n))
    if not torch.isfinite(rows).all():
        raise ValueError("clamped row contains non-finite entries")
    tail = rows[..., i + 1:]
    if tail.numel() and float(tail.abs().max()) != 0.0:
        col = tail.abs().amax(dim=0) if tail.dim() > 1 else tail.abs()
        bad = int(torch.nonzero(col, as_tuple=True)[0][0]) + i + 1
        raise ValueError(
            "clamped row puts mass %.6g on j = %d > i = %d: that is an acausal "
            "edit, the operator would stop being lower triangular, and every "
            "downstream claim (one triangular solve, exact committor, "
            "Sherman-Morrison) depends on it"
            % (float(tail.abs().max()), bad, i))
    if float(rows.min()) < 0.0:
        raise ValueError("clamped row has a negative entry %.6g" % float(rows.min()))
    tol = 10 * n * torch.finfo(rows.dtype).eps
    err = float((rows.sum(-1) - 1.0).abs().max())
    if err > tol:
        raise ValueError("clamped row is not stochastic: max |sum - 1| = %.3e > %.3e"
                         % (err, tol))


def _blocks(P, absorbing_idx, i):
    if P.dim() != 2 or P.shape[-1] != P.shape[-2]:
        raise ValueError("intervene: P must be a single square [n, n] operator, got %s"
                         % (tuple(P.shape),))
    if not torch.isfinite(P).all():
        raise ValueError("intervene: P contains non-finite entries")
    n = P.shape[-1]
    idx = torch.as_tensor(absorbing_idx, dtype=torch.long, device=P.device)
    is_abs = torch.zeros(n, dtype=torch.bool, device=P.device)
    is_abs[idx] = True
    if not (0 <= i < n):
        raise ValueError("intervened index %d out of range [0, %d)" % (i, n))
    if bool(is_abs[i]):
        raise ValueError(
            "index %d is DECLARED ABSORBING: clamping it is not a rank-1 edit of "
            "(I - Q) -- it moves i from A to T, changes the block partition, and "
            "changes the shape of q. Intervene on a transient position." % i)
    T = torch.nonzero(~is_abs, as_tuple=True)[0]
    t = int(torch.nonzero(T == i, as_tuple=True)[0])
    Q = P[T][:, T]
    R = P[T][:, idx]
    M = torch.eye(T.numel(), dtype=P.dtype, device=P.device) - Q
    return idx, T, t, Q, R, M


def _embed(q_T, T, idx, n, batch=None):
    k = idx.numel()
    shape = (n, k) if batch is None else (batch, n, k)
    q = torch.zeros(*shape, dtype=q_T.dtype, device=q_T.device)
    q[..., T, :] = q_T
    q[..., idx, :] = torch.eye(k, dtype=q_T.dtype, device=q_T.device)
    return q


def clamp_row(P, i, row):
    """P' = P with position i's outgoing distribution replaced by `row`.

    Raises ValueError if `row` is not a causal probability row for i -- in
    particular if it puts ANY mass on j > i.
    """
    row = torch.as_tensor(row, dtype=P.dtype, device=P.device)
    _check_rows(row, i, P.shape[-1])
    P2 = P.clone()
    P2[..., i, :] = row
    return P2


def _den_min(P):
    # sqrt(eps): the update amplifies by 1/den, so this keeps half the mantissa.
    # float64 -> 1.49e-08, float32 -> 3.45e-04.
    return float(torch.finfo(P.dtype).eps) ** 0.5


def committor_do(P, absorbing_idx, i, row, den_min=None, _den_perturb=0.0):
    """Committor of do(i -> row), by Sherman-Morrison on (I - Q).

    Returns (q_do [n, k], den). den is the Sherman-Morrison denominator, equal
    to (1 - P'[i,i]) / (1 - P[i,i]), so the caller can see how close the update
    came to singular. Raises SingularTransientBlockError if |den| falls below
    den_min (default sqrt(eps of P.dtype)) instead of returning a committor
    amplified by 1/den.
    """
    row = torch.as_tensor(row, dtype=P.dtype, device=P.device)
    idx, T, t, Q, R, M = _blocks(P, absorbing_idx, i)
    _check_rows(row, i, P.shape[-1])
    if den_min is None:
        den_min = _den_min(P)

    d = row[T] - Q[t]
    dR = row[idx] - R[t]

    q_T = torch.linalg.solve_triangular(M, R, upper=False)              # base
    e_t = torch.zeros(T.numel(), 1, dtype=P.dtype, device=P.device)
    e_t[t, 0] = 1.0
    c = torch.linalg.solve_triangular(M, e_t, upper=False)              # [nT,1]
    w = torch.linalg.solve_triangular(M.transpose(-1, -2), d.unsqueeze(-1),
                                      upper=True)                       # [nT,1]
    den = float(1.0 - w[t, 0]) + float(_den_perturb)
    if abs(den) < den_min:
        raise SingularTransientBlockError(
            "Sherman-Morrison denominator |den| = %.3e below %.3e: the clamped "
            "row drives P'[%d,%d] toward 1, i.e. makes the intervened position "
            "self-absorbing, and (I - Q') toward singular. The update would be "
            "amplified by 1/den = %.3e. Refusing."
            % (abs(den), den_min, i, i, 1.0 / max(abs(den), 1e-300)))
    q_do_T = q_T + (c @ ((w.transpose(-1, -2) @ R) + dR.unsqueeze(0))) / den
    return _embed(q_do_T, T, idx, P.shape[-1]), den


def committor_do_batch(P, absorbing_idx, i, rows, den_min=None):
    """m candidate moves at position i, ONE factorisation.

    rows: [m, n]. Returns (q_do [m, n, k], den [m]). The base solve q_T and the
    inverse column c are computed once; the m denominators come from a single
    m-column back-substitution, and each candidate is then one outer product.
    """
    rows = torch.as_tensor(rows, dtype=P.dtype, device=P.device)
    if rows.dim() != 2:
        raise ValueError("rows must be [m, n], got %s" % (tuple(rows.shape),))
    idx, T, t, Q, R, M = _blocks(P, absorbing_idx, i)
    _check_rows(rows, i, P.shape[-1])
    if den_min is None:
        den_min = _den_min(P)

    D = rows[:, T] - Q[t]                                               # [m,nT]
    DR = rows[:, idx] - R[t]                                            # [m,k]

    q_T = torch.linalg.solve_triangular(M, R, upper=False)              # shared
    e_t = torch.zeros(T.numel(), 1, dtype=P.dtype, device=P.device)
    e_t[t, 0] = 1.0
    c = torch.linalg.solve_triangular(M, e_t, upper=False)              # shared
    W = torch.linalg.solve_triangular(M.transpose(-1, -2), D.transpose(0, 1),
                                      upper=True)                       # [nT,m]
    den = 1.0 - W[t]                                                    # [m]
    if bool((den.abs() < den_min).any()):
        bad = torch.nonzero(den.abs() < den_min, as_tuple=True)[0]
        raise SingularTransientBlockError(
            "Sherman-Morrison denominator below %.3e for %d of %d candidate rows "
            "(first: row %d, den = %.3e): those moves make position %d "
            "self-absorbing. Refusing the whole batch."
            % (den_min, bad.numel(), rows.shape[0], int(bad[0]),
               float(den[bad[0]]), i))
    upd = (W.transpose(0, 1) @ R) + DR                                  # [m,k]
    q_do_T = q_T.unsqueeze(0) \
        + (c.squeeze(-1)[None, :, None] * upd[:, None, :]) / den[:, None, None]
    return _embed(q_do_T, T, idx, P.shape[-1], batch=rows.shape[0]), den


def delta_q(P, absorbing_idx, i, row, den_min=None):
    """The CONSEQUENCE of do(i -> row): q(do a) - q, a vector over the absorbing
    sets, per position. Returns (dq [n, k], den)."""
    idx, T, t, Q, R, M = _blocks(P, absorbing_idx, i)
    q_base = _embed(torch.linalg.solve_triangular(M, R, upper=False), T, idx,
                    P.shape[-1])
    q_do, den = committor_do(P, absorbing_idx, i, row, den_min=den_min)
    return q_do - q_base, den


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from ceqjepa.operator import build_operator, committor

    torch.manual_seed(0)
    DT = torch.float64

    def random_case(n, nA, scale=1.0):
        A = sorted({0} | set(torch.randperm(n - 1)[:nA - 1].add(1).tolist()))
        P = build_operator(torch.randn(n, n, dtype=DT) * scale, A)
        i = int(torch.randint(1, n, (1,)))
        while i in A:
            i = int(torch.randint(1, n, (1,)))
        return P, A, i

    def random_row(n, i, dtype=DT, onehot=False):
        r = torch.zeros(n, dtype=dtype)
        if onehot:
            r[int(torch.randint(0, i, (1,)))] = 1.0    # forced move, never onto i
        else:
            e = torch.rand(i + 1, dtype=dtype) + 1e-3
            r[: i + 1] = e / e.sum()
        return r

    # (a) EXACTNESS vs a full dense re-solve of the clamped operator.
    worst, worst_where, ndraw = 0.0, None, 0
    dens = []
    for n, nA in ((16, 3), (32, 4), (64, 5), (128, 3)):
        for draw in range(60):
            P, A, i = random_case(n, nA, scale=1.0 + 2.0 * (draw % 3))
            row = random_row(n, i, onehot=(draw % 2 == 0))
            q_fast, den = committor_do(P, A, i, row)
            q_ref = committor(clamp_row(P, i, row), A)
            e = float((q_fast - q_ref).abs().max())
            dens.append(den)
            ndraw += 1
            if e > worst:
                worst = e
                worst_where = (n, nA, i, "onehot" if draw % 2 == 0 else "dirichlet")
            assert torch.allclose(q_fast.sum(-1), torch.ones(n, dtype=DT), atol=1e-10), \
                "intervened committor is not a probability: sum over A != 1"
            assert float(q_fast.min()) >= -1e-12 and float(q_fast.max()) <= 1 + 1e-12
    print("(a) EXACTNESS: %d draws, n in {16,32,64,128}: worst |committor_do - dense "
          "re-solve| = %.3e   (at n=%d nA=%d i=%d %s); den range [%.4f, %.4f]"
          % (ndraw, worst, worst_where[0], worst_where[1], worst_where[2],
             worst_where[3], min(dens), max(dens)))
    assert worst <= 1e-10, "fast path is NOT exact: %.3e" % worst

    # (b) BATCH == m singles.
    P, A, i = random_case(64, 4)
    m = 32
    rows = torch.stack([random_row(64, i, onehot=(j % 3 == 0)) for j in range(m)])
    q_b, den_b = committor_do_batch(P, A, i, rows)
    singles = [committor_do(P, A, i, rows[j]) for j in range(m)]
    q_s = torch.stack([s[0] for s in singles])
    den_s = torch.tensor([s[1] for s in singles], dtype=DT)
    db = float((q_b - q_s).abs().max())
    dd = float((den_b - den_s).abs().max())
    print("(b) BATCH vs %d singles: max |dq| = %.3e, max |d den| = %.3e, bitwise = %s"
          % (m, db, dd, bool(torch.equal(q_b, q_s))))
    assert db <= 1e-14 and dd <= 1e-14

    # (c) PLANTED NEGATIVE: perturb den by 1e-3, check (a) must FAIL.
    P, A, i = random_case(32, 3)
    row = random_row(32, i)
    q_ref = committor(clamp_row(P, i, row), A)
    q_bad, den_bad = committor_do(P, A, i, row, _den_perturb=1e-3)
    e_bad = float((q_bad - q_ref).abs().max())
    q_ok, den_ok = committor_do(P, A, i, row)
    e_ok = float((q_ok - q_ref).abs().max())
    print("(c) PLANTED NEGATIVE: den %.6f -> %.6f (+1e-3): error %.3e -> %.3e; "
          "check (a) threshold 1e-10 %s"
          % (den_ok, den_bad, e_ok, e_bad,
             "FAILS as required" if e_bad > 1e-10 else "STILL PASSES -- CHECK IS DEAD"))
    assert e_ok <= 1e-10
    assert e_bad > 1e-10, "the check cannot fail: it is decorative, not a check"

    # (d) NEAR-SINGULAR REFUSAL.
    n = 32
    P, A, i = random_case(n, 3)
    near = torch.zeros(n, dtype=DT)
    near[: i + 1] = 1e-14 / i
    near[i] = 1.0 - 1e-14
    near = near / near.sum()
    raised = False
    try:
        committor_do(P, A, i, near)
    except SingularTransientBlockError as e:
        raised = True
        print("(d) NEAR-SINGULAR: raised as required -- %s" % str(e).split(":")[0])
    assert raised, "must refuse, not return an amplified committor"
    q_garbage, den_g = committor_do(P, A, i, near, den_min=0.0)   # what it refused
    print("    what the guard refused (den_min=0): den = %.3e, max |q| = %.3e, "
          "max |sum_k q - 1| = %.3e (a committor must lie in [0,1] and sum to 1)"
          % (den_g, float(q_garbage.abs().max()),
             float((q_garbage.sum(-1) - 1).abs().max())))
    assert float((q_garbage.sum(-1) - 1).abs().max()) > 1e-6, \
        "the refused value is fine -- then the refusal is theatre"
    bad_row = torch.zeros(n, dtype=DT)
    bad_row[i + 1] = 1.0
    raised_ac = False
    try:
        clamp_row(P, i, bad_row)
    except ValueError as e:
        raised_ac = True
        print("(d) ACAUSAL ROW: raised as required -- %s" % str(e)[:110])
    assert raised_ac

    # (e) TIMING: m=32 candidates, one factorisation + 32 rank-1 vs 32 re-solves.
    for n in (64, 256, 512):
        P, A, i = random_case(n, 4)
        rows = torch.stack([random_row(n, i, onehot=True) for _ in range(32)])
        committor_do_batch(P, A, i, rows)                       # warm
        t0 = time.perf_counter()
        for _ in range(20):
            committor_do_batch(P, A, i, rows)
        t_sm = (time.perf_counter() - t0) / 20
        t0 = time.perf_counter()
        for _ in range(20):
            for j in range(32):
                committor(clamp_row(P, i, rows[j]), A)
        t_full = (time.perf_counter() - t0) / 20
        t0 = time.perf_counter()
        for _ in range(20):
            for j in range(32):
                committor_do(P, A, i, rows[j])
        t_single = (time.perf_counter() - t0) / 20
        print("(e) TIMING n=%3d m=32: batch SM %8.3f ms | 32x committor(clamp) "
              "%8.3f ms (%6.2fx) | 32x single SM %8.3f ms (%6.2fx)"
              % (n, t_sm * 1e3, t_full * 1e3, t_full / t_sm, t_single * 1e3,
                 t_single / t_sm))

    # (f) the derived closed form for den, from a different direction.
    P, A, i = random_case(64, 4)
    row = random_row(64, i)
    _, den = committor_do(P, A, i, row)
    den_closed = (1 - float(row[i])) / (1 - float(P[i, i]))
    print("(f) den vs derived closed form (1-P'_ii)/(1-P_ii): %.15f vs %.15f, "
          "rel err %.3e" % (den, den_closed, abs(den - den_closed) / abs(den_closed)))
    assert abs(den - den_closed) / abs(den_closed) < 1e-12

    # (g) delta_q: the consequence. The null intervention must be exactly zero.
    dq, den0 = delta_q(P, A, i, P[i].clone())
    print("(g) delta_q of the NULL intervention do(i -> P_i): max |dq| = %.3e, "
          "den = %.15f (must be 1)" % (float(dq.abs().max()), den0))
    assert float(dq.abs().max()) < 1e-12 and abs(den0 - 1.0) < 1e-12
    # DEGENERATE REGIME, found by this check failing: below the SECOND absorbing
    # index only A[0]=0 is causally visible, so q = e_0 for every predecessor and
    # NO move at such an i can change anything. delta_q is then exactly zero --
    # a structural fact about the operator, asserted here so it stays true.
    i_low = 1 if A[1] > 1 else None
    if i_low is not None:
        dq_low, _ = delta_q(P, A, i_low, random_row(64, i_low, onehot=True))
        print("    DEGENERATE: i=%d is below the second absorbing index (%d), so only "
              "set 0 is causally visible: max |dq| = %.3e (zero to float64 roundoff)"
              % (i_low, A[1], float(dq_low.abs().max())))
        assert float(dq_low.abs().max()) < 1e-14

    i_hi = max(j for j in range(64) if j not in A)      # every set visible
    dq, _ = delta_q(P, A, i_hi, random_row(64, i_hi, onehot=True))
    aff = int((dq.abs().max(-1).values > 1e-12).sum())
    print("    a forced move at i=%d (above all %d absorbing indices) moves %d of 64 "
          "positions; max |dq| = %.4f" % (i_hi, len(A), aff, float(dq.abs().max())))
    assert aff > 0, "an intervention that changes nothing is not an intervention"
    assert float(dq[:i_hi].abs().max()) < 1e-14, \
        "intervention at i changed a position before i -- causality violated"

    # (h) THE DEVICE ARM. Every exactness figure above is float64 on CPU. The card
    # a claim gets defended on runs float32, and two things break there and nowhere
    # else: the narrower mantissa, and anything that silently disagrees across
    # devices. This arm exists so the GPU numbers ship with a producer instead of
    # being quoted from a scratch script -- STRUCK.md carries rows marked
    # "NO PRODUCER HAS EVER EXISTED" and this repo does not need another.
    # It SKIPS cleanly without CUDA rather than passing vacuously.
    print("(h) DEVICE ARM: the identities on CUDA, float64 and float32.")
    if not torch.cuda.is_available():
        print("    SKIPPED: no CUDA visible to this interpreter (torch %s). This is a"
              % torch.__version__)
        print("    SKIP, not a pass -- nothing about the GPU is asserted by this run.")
    else:
        # i_hi is DECLARED ABSORBING in this demo's A; clamping it is refused by
        # design (it would move i from A to T and change q's shape). Pick a
        # transient index above the second absorbing one, and a causal target.
        # n was rebound by the timing sweep above (it ends at 512) while P is
        # still the 64x64 operator -- derive both from P, never from the loop var.
        n_dev = P.shape[-1]
        i_dev = next(j for j in range(n_dev - 1, 0, -1) if j not in A and j > A[2])
        row_h = torch.zeros(n_dev, dtype=torch.float64); row_h[A[2]] = 1.0
        seen = {}
        for dev in ("cpu", "cuda"):
            for dt in (torch.float64, torch.float32):
                Pd = P.to(device=dev, dtype=dt)
                rd = row_h.to(device=dev, dtype=dt)
                idxd = torch.as_tensor(A, device=dev)
                q_sm, den_d = committor_do(Pd, idxd, i_dev, rd)
                q_dn = committor(clamp_row(Pd, i_dev, rd), idxd)
                seen[(dev, dt)] = (float((q_sm - q_dn).abs().max()), float(den_d))
                print("    %-4s %-8s  max|SM - dense| = %.3e   den = %.9f"
                      % (dev, str(dt).replace("torch.", ""), *seen[(dev, dt)]))
        for dt in (torch.float64, torch.float32):
            assert abs(seen[("cpu", dt)][1] - seen[("cuda", dt)][1]) < 1e-9, \
                "the Sherman-Morrison denominator disagrees across devices in %s" % dt
            assert seen[("cuda", dt)][0] < 1e3 * float(torch.finfo(dt).eps), \
                "the CUDA %s update does not match a dense re-solve to its own eps" % dt
        # the planted negative must fire ON THE GPU, in float32 -- the case that
        # actually protects a T4 run, not a float64 CPU rehearsal of it
        Pc = P.to(device="cuda", dtype=torch.float32)
        rc = row_h.to(device="cuda", dtype=torch.float32)
        idxc = torch.as_tensor(A, device="cuda")
        ref = committor(clamp_row(Pc, i_dev, rc), idxc)
        good = float((committor_do(Pc, idxc, i_dev, rc)[0] - ref).abs().max())
        bad = float((committor_do(Pc, idxc, i_dev, rc, _den_perturb=1e-3)[0] - ref).abs().max())
        print("    PLANTED NEGATIVE on cuda/float32: %.3e -> %.3e" % (good, bad))
        assert bad > 1e-6 > good, \
            "the GPU arm cannot tell a good Sherman-Morrison denominator from a bad one"

    print("ALL SELF-CHECKS PASSED")
