#!/usr/bin/env python3
"""X26 trajectory-state monitor: CUSUM null calibration and planted-bed lead time.

    python scripts/v13_cusum_arl_calib.py

numpy + standard library only. Fixed seeds throughout. Every assert names what
broke. Equation numbers in comments refer to V13_X26_TRAJECTORY_MONITOR.md.

This script exits non-zero on purpose: the final assert is the kill clause, and
it fires. Sections 1-6 print in full before it does. Nothing here touches the
training loop or the optimizer.
"""

import math
import numpy as np

# ---------------------------------------------------------------- constants

K = 0.5                      # CUSUM reference value (slack), null sd units
ARL0_TARGET = 1000.0         # the contract's headline operating point
SIEGMUND_OVERSHOOT = 1.166   # 2 * 0.5826, Brownian expected overshoot, Eq. (8)

M_CAL, T_CAL = 12000, 12000  # replicates / censoring horizon, idealized null
M_VER = 12000                # replicates for fresh-stream verification
M_OOC = 20000                # replicates for the out-of-control ARL1 check
BISECT_STEPS = 14
SEED_CAL, SEED_VER, SEED_AR = 11, 12, 13

# planted bed geometry, Eq. (12)
N_SEEDS = 8
BED_SEED0 = 20260831
STEPS_A = 2000               # arm A logged every step; abscissa unit = 2048 tokens
N_B = 630                    # arm B logged every 3rd step
B_STRIDE, B_BATCH_RATIO = 3.0, 1.07
TAU_A, TAU_B = 300.0, 330.0
SIGMA_NOISE = 0.02           # per-logged-point loss noise, sd in loss units
PHI_TRAIN = 0.35             # AR(1) coefficient of the per-arm training noise
AMP_SIGMA = 12.0             # transition amplitude, units of SIGMA_NOISE
U_MID, U_WIDTH = 1400.0, 40.0
DETREND_W, DETREND_G = 300, 150   # trailing reference window, guard gap
BURN_END = 900               # burn-in ends at this arm-A abscissa
N_NULL_BEDS = 2500           # end-to-end pipeline null
SEED_NULLBED = 4242
ARL0_MULT = 20.0             # deployable ARL0 = ARL0_MULT * horizon

T7_975 = 2.364624            # Student t, 7 df, two-sided 95%; scipy not available
SIGN_FLOOR_N8 = 2.0 * 0.5 ** 8   # 0.0078125, this project's N=8 sign-test floor


# ------------------------------------------------------- CUSUM, Eqs. (2)-(4)

def cusum_alarm_batch(x, k, h):
    """First alarm index per row of x. Returns x.shape[1] where no alarm.

    S+_t = max(0, S+_{t-1} + (x_t - mu0) - k)   Eq. (2)
    S-_t = max(0, S-_{t-1} - (x_t - mu0) - k)   Eq. (3)
    alarm at min{t : max(S+_t, S-_t) >= h}      Eq. (4)
    x is already centred on mu0 by the pipeline, so mu0 = 0 here.
    """
    n, m = x.shape
    sp = np.zeros(n)
    sn = np.zeros(n)
    out = np.full(n, m, np.int64)
    for t in range(m):
        sp = np.maximum(0.0, sp + x[:, t] - k)
        sn = np.maximum(0.0, sn - x[:, t] - k)
        hit = ((sp >= h) | (sn >= h)) & (out == m)
        out[hit] = t
    return out


def shewhart_alarm_batch(x, limit):
    """Memoryless comparator, Eq. (14): alarm at min{t : |x_t| > limit}. This is
    the ARL0-matched stand-in for 'the transition is visible in the raw curve'."""
    hit = np.abs(x) > limit
    return np.where(hit.any(1), hit.argmax(1), x.shape[1])


class IIDNull:
    """x_t ~ N(0,1) i.i.d. -- the null the classical ARL formulas assume."""

    def __init__(self, rng, m):
        self.rng, self.n = rng, m

    def draw(self):
        return self.rng.standard_normal(self.n)

    def keep(self, mask):
        self.n = int(mask.sum())


class AR1Null:
    """x_t = phi x_{t-1} + sqrt(1-phi^2) e_t. Marginal variance 1, matched to
    IIDNull, so any ARL difference is dependence and nothing else."""

    def __init__(self, rng, m, phi):
        self.rng, self.phi = rng, phi
        self.s = rng.standard_normal(m)
        self.inn = math.sqrt(1.0 - phi * phi)

    def draw(self):
        self.s = self.phi * self.s + self.inn * self.rng.standard_normal(self.s.size)
        return self.s

    def keep(self, mask):
        self.s = self.s[mask]


class Shifted:
    """Any stream plus a constant mean shift -- the out-of-control null."""

    def __init__(self, inner, delta):
        self.inner, self.delta = inner, delta

    def draw(self):
        return self.inner.draw() + self.delta

    def keep(self, mask):
        self.inner.keep(mask)


def first_alarm_times(stream, m, T, k, h):
    """Run lengths over m parallel replicates, censored at T (returned as T+1).
    Alarmed replicates leave the working set, which is what makes this cheap."""
    sp = np.zeros(m)
    sn = np.zeros(m)
    idx = np.arange(m)
    rl = np.full(m, T + 1, np.int64)
    for t in range(1, T + 1):
        x = stream.draw()
        sp = np.maximum(0.0, sp + x - k)
        sn = np.maximum(0.0, sn - x - k)
        hit = (sp >= h) | (sn >= h)
        if hit.any():
            rl[idx[hit]] = t
            keep = ~hit
            idx, sp, sn = idx[keep], sp[keep], sn[keep]
            stream.keep(keep)
            if idx.size == 0:
                break
    return rl


def sim_arl(seed, m, T, k, h, phi=0.0, delta=0.0):
    """(mean run length, censored fraction, standard error)."""
    rng = np.random.default_rng(seed)
    stream = AR1Null(rng, m, phi) if phi else IIDNull(rng, m)
    if delta:
        stream = Shifted(stream, delta)
    rl = first_alarm_times(stream, m, T, k, h)
    return rl.mean(), float((rl > T).mean()), rl.std(ddof=1) / math.sqrt(m)


# --------------------------------------------- Siegmund, Eqs. (8)-(10)

def siegmund_one_sided(k, h, delta):
    """Eq. (8). ARL of one CUSUM arm under a mean shift delta (null sd units)."""
    b = h + SIEGMUND_OVERSHOOT
    d = delta - k
    if abs(d) < 1e-12:
        return b * b
    return (math.exp(-2.0 * d * b) + 2.0 * d * b - 1.0) / (2.0 * d * d)


def siegmund_arl0(k, h):
    """Eq. (9). Two-sided in-control ARL: under the null only one arm can be far
    from zero at a time, so the alarm hazards add and ARL0 = ARL+ / 2."""
    return siegmund_one_sided(k, h, 0.0) / 2.0


def bisect(fn, target, lo, hi, steps=200):
    """fn monotone increasing in its argument."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def shewhart_limit_for_arl0(target):
    """Eq. (15). Two-sided normal individuals chart: ARL0 = 1 / (2 Phi(-L))."""
    return bisect(lambda l: 1.0 / (2.0 * 0.5 * math.erfc(l / math.sqrt(2.0))),
                  target, 0.5, 12.0)


def calibrate_h_iid(target, k, seed, m, T, lo=2.5, hi=12.0):
    """Bisect the simulated ARL0 on a common random-number stream. Re-seeding
    identically at every h makes ARL_hat(h) monotone: a replicate's first passage
    to a higher barrier cannot come earlier."""
    return bisect(lambda h: sim_arl(seed, m, T, k, h)[0], target, lo, hi, BISECT_STEPS)


def calibrate_on_beds(alarm_fn, x, target_fa, lo, hi):
    """Calibrate a threshold directly on realized pipeline output -- no formula,
    no independence assumption. Eq. (18)."""
    return bisect(lambda c: -(alarm_fn(x, c) < x.shape[1]).mean(), -target_fa, lo, hi, 40)


# ------------------------------------------------- planted bed, Eqs. (12)-(17)

def bed_abscissa():
    """Arm A logs every step; arm B logs every 3rd step at a 7% larger batch.
    Index i of A and index i of B therefore sit at different token counts --
    which is exactly the artefact 'matched abscissa' exists to remove."""
    u_a = np.arange(STEPS_A, dtype=float)
    u_b = np.arange(N_B, dtype=float) * B_STRIDE * B_BATCH_RATIO
    return u_a, u_b


def interp_weights(u_a, u_b):
    j = np.clip(np.searchsorted(u_b, u_a, side="right") - 1, 0, len(u_b) - 2)
    return j, (u_a - u_b[j]) / (u_b[j + 1] - u_b[j])


def ar1_rows(rng, n, m, phi):
    x = np.empty((n, m))
    x[:, 0] = rng.standard_normal(n)
    inn = math.sqrt(1.0 - phi * phi)
    for t in range(1, m):
        x[:, t] = phi * x[:, t - 1] + inn * rng.standard_normal(n)
    return x


def make_gap(rng, n, amp_sigma, width=U_WIDTH):
    """eps(t): the matched-abscissa gap L_A - L_B, Eqs. (1) and (12). amp_sigma = 0 is the
    no-change bed. Both arms carry AR(1) training noise; arm B is then linearly
    interpolated onto arm A's abscissa, which adds more."""
    u_a, u_b = bed_abscissa()
    j, a = interp_weights(u_a, u_b)
    la = 1.20 + 1.8 * np.exp(-u_a / TAU_A) + SIGMA_NOISE * ar1_rows(rng, n, len(u_a), PHI_TRAIN)
    drop = amp_sigma * SIGMA_NOISE / (1.0 + np.exp(-(u_b - U_MID) / width))
    lb = 1.18 + 1.8 * np.exp(-u_b / TAU_B) - drop + SIGMA_NOISE * ar1_rows(rng, n, len(u_b), PHI_TRAIN)
    return la - ((1.0 - a) * lb[:, j] + a * lb[:, j + 1]), la, lb


def detrend_filter(w_len=DETREND_W, gap=DETREND_G):
    """Eq. (5). Causal local-linear extrapolation from a trailing window stopping
    `gap` points short of t, so a change cannot leak into its own baseline. The
    predictor is a fixed linear filter, hence one matmul."""
    off = np.arange(-(gap + w_len), -gap, dtype=float)
    return np.linalg.pinv(np.column_stack([np.ones(w_len), off]))[0], gap + w_len


def detrend(eps, w_len=DETREND_W, gap=DETREND_G):
    w, start = detrend_filter(w_len, gap)
    win = np.lib.stride_tricks.sliding_window_view(eps, w_len, axis=-1)
    return start, eps[..., start:] - win[..., : eps.shape[-1] - start, :] @ w


def pipeline(r, burn_end=BURN_END, start=DETREND_G + DETREND_W):
    """Eqs. (6)-(7). Burn-in estimates mu0, sigma0 and phi_hat; the monitored
    statistic is the standardized, AR(1)-pre-whitened residual."""
    burn = r[..., : burn_end - start]
    mu = burn.mean(axis=-1, keepdims=True)
    sd = burn.std(axis=-1, ddof=1, keepdims=True)
    c = burn - mu
    phi = (c[..., :-1] * c[..., 1:]).sum(-1, keepdims=True) / (c * c).sum(-1, keepdims=True)
    rs = (r - mu) / sd                                                 # standardized, Eq. (6)
    z = (rs[..., 1:] - phi * rs[..., :-1]) / np.sqrt(1.0 - phi ** 2)   # pre-whitened, Eq. (7)
    return (rs[..., burn_end - start:], z[..., burn_end - start - 1:],
            phi.ravel(), sd.ravel())


def bed(amp, width=U_WIDTH):
    """The N_SEEDS planted (or, at amp=0, no-change) trajectories."""
    rows = [make_gap(np.random.default_rng(BED_SEED0 + s), 1, amp, width)[0]
            for s in range(N_SEEDS)]
    _, r = detrend(np.concatenate(rows, axis=0))
    return pipeline(r)


def lag1(x):
    c = x - x.mean(axis=-1, keepdims=True)
    return float(np.mean((c[..., :-1] * c[..., 1:]).sum(-1) / (c * c).sum(-1)))


def lrv_ratio(x, bw):
    """Bartlett long-run variance divided by gamma0, Eq. (11). This -- not the
    lag-1 autocorrelation -- is the quantity a cumulative statistic integrates."""
    c = x - x.mean(-1, keepdims=True)
    n = x.shape[-1]
    g0 = (c * c).mean(-1)
    s = g0.copy()
    for l in range(1, bw + 1):
        s = s + 2.0 * (1.0 - l / (bw + 1.0)) * (c[..., :-l] * c[..., l:]).sum(-1) / n
    return float((s / g0).mean())


# ---------------------------------------------------------------------- main

def main():
    out = []
    p = out.append
    failed = []

    def check(name, ok, detail):
        """Record a pre-registered check without aborting the report. Every entry
        here is asserted at the end; none of them is ever relaxed to pass."""
        if not ok:
            failed.append((name, detail))
        return ok

    p("=" * 79)
    p("X26 trajectory-state monitor -- CUSUM null calibration")
    p("=" * 79)

    # -- 1. matched abscissa ------------------------------------------------
    eps0, la, lb = make_gap(np.random.default_rng(BED_SEED0), 1, 0.0)
    u_a, u_b = bed_abscissa()
    idx_gap = la[0, :N_B] - lb[0, :N_B]                 # index-matched, the wrong thing
    p("")
    p("1. MATCHED ABSCISSA (Eq. 1)")
    p(f"   arm A: {STEPS_A} points, 1 step/log     arm B: {N_B} points, "
      f"{B_STRIDE:.0f} steps/log at {B_BATCH_RATIO:.2f}x batch")
    p(f"   abscissa span   A [{u_a[0]:.0f}, {u_a[-1]:.0f}]   B [{u_b[0]:.0f}, {u_b[-1]:.0f}]")
    p(f"   mean |gap|, index-matched      {np.abs(idx_gap).mean() / SIGMA_NOISE:8.2f} sigma_noise")
    p(f"   mean |gap|, matched-abscissa   {np.abs(eps0[0]).mean() / SIGMA_NOISE:8.2f} sigma_noise")
    assert np.abs(idx_gap).mean() > 5.0 * np.abs(eps0[0]).mean(), (
        "MATCHED-ABSCISSA BED IS DEGENERATE: index-matching and abscissa-matching give the "
        "same gap, so the bed cannot demonstrate the artefact it exists to remove")

    # -- 2. the detrender is a high-pass filter -----------------------------
    p("")
    p("2. HOW THE NULL IS MADE STATIONARY, AND WHAT IT COSTS (Eq. 5)")
    p("   Noiseless unit-amplitude transition pushed through the detrender:")
    p(f"   {'W':>5}{'G':>6}{'G/width':>9}{'peak residual / amplitude':>28}")
    step = 1.0 / (1.0 + np.exp(-(u_a - U_MID) / U_WIDTH))
    for w_len, gap in ((120, 20), (120, 150), (300, 150), (300, 300)):
        _, res = detrend(step[None, :], w_len, gap)
        p(f"   {w_len:5d}{gap:6d}{gap / U_WIDTH:9.2f}{res.max():28.3f}")
    p("   The guard gap G must exceed the transition duration or the trailing baseline")
    p("   climbs onto the ramp and cancels it. At G/width = 0.5 only 13% survives.")

    eps_null, _, _ = make_gap(np.random.default_rng(SEED_NULLBED), N_NULL_BEDS, 0.0)
    start, r_null = detrend(eps_null)
    rs_n, z_n, phi_n, sd_n = pipeline(r_null)
    horizon = z_n.shape[-1]
    phi_bar = float(phi_n.mean())
    z_ac = lag1(z_n)
    p("")
    p(f"   shipped geometry: W={DETREND_W}, G={DETREND_G}, burn-in [{start}, {BURN_END}), "
      f"horizon H = {horizon}")
    p(f"   residual sd / sigma_noise            {float(sd_n.mean()) / SIGMA_NOISE:8.3f}")
    p(f"   phi_hat on burn-in, mean of {N_NULL_BEDS} beds {phi_bar:8.4f}"
      f"   sd {float(phi_n.std(ddof=1)):.4f}")
    p(f"   lag-1 autocorr of z after Eq. (7)    {z_ac:+8.4f}   <- looks whitened")
    z_drift = float(z_n.mean())
    p(f"   monitored z: mean {z_drift:+.4f}, sd {float(z_n.std()):.4f}")
    check("C1 null-drift", abs(z_drift) < 0.20,
          f"monitored null mean z = {z_drift:+.4f}, above the pre-registered 0.20 sd. The "
          f"local-linear extrapolator is biased on a curved trend, and that bias consumes "
          f"{abs(z_drift) / K * 100:.0f}% of the k={K} slack before any change occurs.")
    for bw in (20, 50, 100, 200):
        p(f"   Bartlett LRV(z)/gamma0 at L={bw:3d}      {lrv_ratio(z_n, bw):8.2f}")
    p("   Zero-frequency power, not lag-1, is what a cumulative statistic integrates.")
    p("   Eq. (7) whitens lag 1 and leaves the low-frequency power in place.")

    # -- 3. calibration on the idealized i.i.d. null ------------------------
    h_sieg = bisect(lambda h: siegmund_arl0(K, h), ARL0_TARGET, 0.5, 25.0)
    h_iid = calibrate_h_iid(ARL0_TARGET, K, SEED_CAL, M_CAL, T_CAL)
    arl_ver, cens_ver, se_ver = sim_arl(SEED_VER, M_VER, T_CAL, K, h_iid)
    se_cal = ARL0_TARGET / math.sqrt(M_CAL)
    tol = 4.0 * math.hypot(se_cal, se_ver)          # pre-registered: 4 sd of the pair
    p("")
    p(f"3. CALIBRATION ON THE IDEALIZED i.i.d. NULL, k = {K}, TARGET ARL0 = "
      f"{ARL0_TARGET:.0f} (Eqs. 8-10)")
    p(f"   h from Siegmund Eq. (9)             {h_sieg:.4f}")
    p(f"   h from simulation (CRN bisection)   {h_iid:.4f}   M={M_CAL}, T={T_CAL}")
    p(f"   achieved ARL0, fresh stream         {arl_ver:.1f} +/- {se_ver:.1f} (1 se), "
      f"censored {cens_ver * 100:.3f}%")
    p(f"   pre-registered tolerance            +/-{tol:.1f} (4 sd of calibration+verification)")
    assert cens_ver < 1e-3, (
        f"ARL0 ESTIMATE IS CENSORED: {cens_ver * 100:.2f}% of runs hit T={T_CAL} without "
        "alarming, so the reported mean is a lower bound, not an ARL")
    assert abs(arl_ver - ARL0_TARGET) <= tol, (
        f"ARL0 CALIBRATION MISSED ITS TARGET: achieved {arl_ver:.1f} against target "
        f"{ARL0_TARGET:.0f}, outside the pre-registered +/-{tol:.1f}")

    arl_sieg = siegmund_arl0(K, h_iid)
    rel = abs(arl_sieg - arl_ver) / arl_ver
    p(f"   Siegmund ARL0 at h_iid              {arl_sieg:.1f}"
      f"   relative error vs simulation {rel * 100:.2f}%")
    assert rel < 0.10, (
        f"SIEGMUND APPROXIMATION DISAGREES WITH THE INDEPENDENT NULL: {arl_sieg:.1f} vs "
        f"simulated {arl_ver:.1f}, {rel * 100:.1f}% -- Eq. (9) cannot be used to set h")

    p("   out-of-control (Eq. 10), the lead-time scale:")
    for delta in (0.75, 1.0, 1.5):
        a1, _, _ = sim_arl(400 + int(delta * 100), M_OOC, 4000, K, h_iid, delta=delta)
        a1s = siegmund_one_sided(K, h_iid, delta)
        e1 = abs(a1s - a1) / a1
        p(f"     delta={delta:.2f}   simulated ARL1 {a1:7.2f}   Siegmund {a1s:7.2f}   {e1 * 100:5.1f}%")
        assert e1 < 0.15, (
            f"ARL1 FORMULA WRONG AT delta={delta}: Eq. (10) gives {a1s:.2f} against simulated "
            f"{a1:.2f}; the predicted lead time is not usable")

    # -- 4. autocorrelation, priced ----------------------------------------
    p("")
    p("4. WHAT AUTOCORRELATION COSTS IF IGNORED (Eq. 11)")
    p(f"   h = {h_iid:.4f} held fixed. Marginal variance is 1 in every row, so every")
    p("   difference below is dependence and nothing else.")
    p(f"   {'phi':>6}{'LRV/gamma0':>13}{'ARL0 achieved':>19}{'vs nominal':>13}"
      f"{'P(FA) over H=' + str(horizon):>20}")
    for phi in (0.0, 0.3, 0.5, phi_bar, 0.8):
        a, _, se = sim_arl(SEED_AR + int(phi * 1000), M_CAL, T_CAL, K, h_iid, phi=phi)
        tag = "  <- phi_hat on the bed" if abs(phi - phi_bar) < 1e-12 else ""
        p(f"   {phi:6.3f}{(1 + phi) / (1 - phi):13.2f}{a:12.1f} +/-{se:4.1f}"
          f"{a / ARL0_TARGET:12.3f}x{(1 - math.exp(-horizon / a)) * 100:19.1f}%{tag}")

    infl = math.sqrt((1 + phi_bar) / (1 - phi_bar))
    a_pw, _, _ = sim_arl(SEED_AR + 77, M_CAL, T_CAL, K, h_iid)   # pre-whitened == i.i.d.
    a_in, _, _ = sim_arl(SEED_AR + 78, M_CAL, T_CAL, K * infl, h_iid * infl, phi=phi_bar)
    p("")
    p(f"   Two textbook corrections at phi = {phi_bar:.4f}, on a TRUE AR(1):")
    p(f"     pre-whitening, Eq. (7)        ARL0 {a_pw:9.1f}   signal attenuation {1 / infl:.3f}x")
    p(f"     variance inflation {infl:.3f}x     ARL0 {a_in:9.1f}   signal attenuation {1 / infl:.3f}x")
    p("   Both restore ARL0 on a true AR(1) and both cost the same factor in delta.")
    p("   Neither restores it on the real pipeline, because the residual is not AR(1).")

    # -- 5. calibration on the pipeline itself ------------------------------
    arl0_dep = ARL0_MULT * horizon
    fa_target = 1.0 - math.exp(-1.0 / ARL0_MULT)
    h_dep_iid = bisect(lambda h: siegmund_arl0(K, h), arl0_dep, 0.5, 60.0)
    l_dep_iid = shewhart_limit_for_arl0(arl0_dep)
    h_pipe = calibrate_on_beds(lambda x, c: cusum_alarm_batch(x, K, c), z_n, fa_target, 2.0, 90.0)
    l_pipe = calibrate_on_beds(shewhart_alarm_batch, rs_n, fa_target, 1.0, 15.0)
    fa_h = float((cusum_alarm_batch(z_n, K, h_pipe) < horizon).mean())
    fa_l = float((shewhart_alarm_batch(rs_n, l_pipe) < horizon).mean())
    fa_naive = float((cusum_alarm_batch(z_n, K, h_dep_iid) < horizon).mean())
    arl_naive = -horizon / math.log(1.0 - fa_naive)
    p("")
    p(f"5. CALIBRATION ON THE PIPELINE ITSELF (Eq. 18), TARGET ARL0 = {arl0_dep:.0f} = "
      f"{ARL0_MULT:.0f}H")
    p(f"   target per-run false-alarm probability over H={horizon}: {fa_target * 100:.2f}%")
    p(f"   {'detector':<22}{'threshold from theory':>23}{'from pipeline':>16}{'inflation':>12}")
    p(f"   {'CUSUM k=0.5, h':<22}{h_dep_iid:23.3f}{h_pipe:16.3f}{h_pipe / h_dep_iid:11.2f}x")
    p(f"   {'Shewhart, L':<22}{l_dep_iid:23.3f}{l_pipe:16.3f}{l_pipe / l_dep_iid:11.2f}x")
    p(f"   realized null FA: CUSUM {fa_h * 100:.2f}%, Shewhart {fa_l * 100:.2f}% "
      f"({N_NULL_BEDS} beds)")
    p("")
    p(f"   USING THE THEORY VALUE h = {h_dep_iid:.3f} ON THE REAL PIPELINE:")
    p(f"     realized per-run false-alarm probability {fa_naive * 100:.1f}% "
      f"(nominal {fa_target * 100:.2f}%)")
    p(f"     implied ARL0 {arl_naive:.0f} against nominal {arl0_dep:.0f}"
      f"  -- short by {arl0_dep / arl_naive:.1f}x")
    p("   Dependence inflates an integrating detector's threshold far more than a")
    p("   memoryless one's. That asymmetry is the mechanism that kills the lead time.")

    # -- 6. must-fires ------------------------------------------------------
    rs_p, z_p, phi_p, _ = bed(AMP_SIGMA)
    rs_z, z_z, _, _ = bed(0.0)
    t_cusum = cusum_alarm_batch(z_p, K, h_pipe)
    t_shew = shewhart_alarm_batch(rs_p, l_pipe)
    t_cusum_null = cusum_alarm_batch(z_z, K, h_pipe)
    n_hit = int((t_cusum < horizon).sum())
    n_fa = int((t_cusum_null < horizon).sum())
    p("")
    p("6. MUST-FIRES (Eqs. 12-13)")
    p(f"   planted transition: amplitude {AMP_SIGMA:.0f} sigma_noise "
      f"({AMP_SIGMA * SIGMA_NOISE:.2f} nats), midpoint u={U_MID:.0f}, width {U_WIDTH:.0f}")
    p(f"   MUST-FIRE 1: planted bed alarms on {n_hit}/{N_SEEDS} seeds")
    p(f"   MUST-FIRE 2: no-change bed alarms on {n_fa}/{N_SEEDS} seeds "
      f"(pre-registered bound <= 2; P(>=3) = 0.0060 at FA {fa_target * 100:.2f}%)")
    assert n_hit == N_SEEDS, (
        f"MUST-FIRE 1 FAILED: only {n_hit}/{N_SEEDS} planted delayed-transition seeds alarmed "
        "inside the horizon")
    assert n_fa <= 2, (
        f"MUST-FIRE 2 FAILED: the no-change bed alarmed on {n_fa}/{N_SEEDS} seeds, above the "
        "pre-registered bound of 2; the monitor false-alarms on flat training")

    # -- 7. lead time -------------------------------------------------------
    off = BURN_END
    lead = (t_shew - t_cusum).astype(float)
    boot = np.random.default_rng(7).integers(0, N_SEEDS, (20000, N_SEEDS))
    lo, hi = np.percentile(lead[boot].mean(1), [2.5, 97.5])
    srt = np.sort(lead)
    half = T7_975 * lead.std(ddof=1) / math.sqrt(N_SEEDS)
    p("")
    p("7. LEAD TIME AGAINST THE ARL0-MATCHED COMPARATOR (Eqs. 14-17)")
    p("   seed   phi_hat   t_cusum   t_shewhart   lead")
    for s in range(N_SEEDS):
        p(f"   {s:4d}   {phi_p[s]:+.4f}   {t_cusum[s] + off:7d}   {t_shew[s] + off:10d}"
          f"   {lead[s]:+5.0f}")
    p(f"   mean {lead.mean():+.2f}   median {np.median(lead):+.2f}   sd {lead.std(ddof=1):.2f}")
    p(f"   percentile bootstrap 95% CI of the mean  [{lo:+.2f}, {hi:+.2f}]   B=20000")
    p(f"   Student t 95% CI of the mean             "
      f"[{lead.mean() - half:+.2f}, {lead.mean() + half:+.2f}]")
    p(f"   order-statistic 92.97% CI of the median  [{srt[1]:+.2f}, {srt[6]:+.2f}]  (L_(2), L_(7))")
    p(f"   sign test: {int((lead > 0).sum())}/{N_SEEDS} positive; the N=8 two-sided floor is "
      f"{SIGN_FLOOR_N8:.7f} and is reachable only at 8/8 or 0/8")
    p(f"   CI excludes zero: {'YES' if (lo > 0 or hi < 0) else 'NO'}")

    p("")
    p("8. SENSITIVITY TO TRANSITION WIDTH (all thresholds held at section 5)")
    p(f"   {'width':>7}{'G/width':>9}{'both fire':>11}{'median lead':>13}{'min':>7}{'max':>7}")
    for width in (40.0, 80.0, 120.0, 200.0):
        rs_w, z_w, _, _ = bed(AMP_SIGMA, width)
        tc = cusum_alarm_batch(z_w, K, h_pipe)
        ts = shewhart_alarm_batch(rs_w, l_pipe)
        both = (tc < horizon) & (ts < horizon)
        ld = (ts - tc)[both]
        p(f"   {width:7.0f}{DETREND_G / width:9.2f}{int(both.sum()):8d}/{N_SEEDS}"
          f"{np.median(ld) if ld.size else float('nan'):13.0f}"
          f"{ld.min() if ld.size else 0:7.0f}{ld.max() if ld.size else 0:7.0f}")

    check("C2 kill-clause lead time", lo > 0.0,
          f"lead-time bootstrap CI [{lo:+.2f}, {hi:+.2f}] does not exclude zero (mean "
          f"{lead.mean():+.2f}, {int((lead > 0).sum())}/{N_SEEDS} seeds positive). At matched "
          "ARL0 the CUSUM does not reliably precede the memoryless comparator.")

    p("")
    p("=" * 79)
    if failed:
        p(f"FAILED CHECKS: {len(failed)}. These are the result, not a tolerance to relax.")
        for name, detail in failed:
            p(f"  [{name}] {detail}")
        p("")
        p("VERDICT: the kill clause fires. The monitor is a plot and not a feature.")
    else:
        p("VERDICT: all pre-registered checks pass.")
    p("=" * 79)
    print("\n".join(out))

    assert not failed, "; ".join(f"{n}: {d}" for n, d in failed)


if __name__ == "__main__":
    main()
