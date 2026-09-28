# P2-F: Lorenz-63 forecasters -- skill before T*, reliability after T*. Bar: sun/canon/p2/BAR_F.md incl. Amendment A1
# (sha256 of the bar and of this script recorded in results_p2f.json). P2F_QUICK=1 -> smoke run, writes nothing.
import hashlib, itertools, json, math, os, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
QUICK = os.environ.get("P2F_QUICK") == "1"
BAR_SHA_IMPLEMENTED = "48a03f96b454f8048c6eea20f5e3b56602f65f1c8fef45c03ad77d058ad788eb"   # BAR_F.md with Amendment A1
LAM, DT, SUB, DELTA, L = 0.9059, 0.01, 5, 0.1, 20          # SUB: RK4 steps per observation, dt_obs = 0.05
DTO = DT * SUB
TSTAR = math.log(1 / DELTA) / LAM                           # 2.542
LEAD_FR = np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3])
H = np.round(LEAD_FR * TSTAR / DTO).astype(int)             # leads in observation steps (deviation: rounded to dt_obs grid)
HMAX = int(H.max())
BEFORE, AFTER = LEAD_FR <= 1, LEAD_FR > 1
SEEDS = [0, 1, 2]
T_TRAIN, T_TEST, BURN = (500, 200, 50) if QUICK else (10000, 500, 50)   # test cut 2,000 -> 500 units (coordinator deadline)
TF_WALL = 60 if QUICK else 600   # tf GPU wall cap per seed (coordinator deadline: <= 10 min)
CACHE = Path(os.environ.get('P2F_CACHE_DIR', '.'))
VAL_UNITS, STRIDE_EVAL, NENS, NPF = 1000, 5, 256, 512       # validation = last 1,000 train units (A1.3)
SAT_FRAC = 0.5          # canon fit: leads <= T* and before pooled ngrc RMS reaches SAT_FRAC * sigma_clim (fixed before any run)
GATE = 0.95             # A1.5
LIVE_BSS = 0.05         # A1.1
LW_A = 0.98             # Liu-West shrinkage for the particle filter's resample-move (fixed before any run)
NBOOT, BLOCK_UNITS = 1000, 10.0
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(28)


# ---------------- Lorenz-63, equations as sun/sun_maths.py (sigma 10, rho 28, beta 8/3, RK4 dt 0.01) ----------------
def lorenz_np(x):   # sun_maths.lorenz, copied (importing sun_maths runs its S1/S2 script)
    return np.array([10.0 * (x[1] - x[0]), x[0] * (28.0 - x[2]) - x[1], x[0] * x[1] - 8 / 3 * x[2]])


def rk4_np(x, dt):
    k1 = lorenz_np(x); k2 = lorenz_np(x + dt / 2 * k1); k3 = lorenz_np(x + dt / 2 * k2); k4 = lorenz_np(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_scalar(a, b, c, h=DT):
    # same arithmetic as rk4_np, unrolled on Python floats for speed
    def f(a, b, c):
        return 10.0 * (b - a), a * (28.0 - c) - b, a * b - 8 / 3 * c
    k1 = f(a, b, c)
    k2 = f(a + h / 2 * k1[0], b + h / 2 * k1[1], c + h / 2 * k1[2])
    k3 = f(a + h / 2 * k2[0], b + h / 2 * k2[1], c + h / 2 * k2[2])
    k4 = f(a + h * k3[0], b + h * k3[1], c + h * k3[2])
    return (a + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            b + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            c + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))


def rk4_torch(x):
    def f(x):
        a, b, c = x.unbind(-1)
        return torch.stack([10.0 * (b - a), a * (28.0 - c) - b, a * b - 8 / 3 * c], -1)
    k1 = f(x); k2 = f(x + DT / 2 * k1); k3 = f(x + DT / 2 * k2); k4 = f(x + DT * k3)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


_x = np.array([1.3, -2.0, 20.0])
assert np.allclose(rk4_scalar(*_x), rk4_np(_x, DT), rtol=0, atol=1e-12)
assert np.allclose(rk4_torch(torch.tensor(_x)).numpy(), rk4_np(_x, DT), rtol=0, atol=1e-12)


def trajectory(seed, T):
    """True state and noisy observation every dt_obs for T time units after a BURN-unit burn-in."""
    rng = np.random.default_rng(seed)
    a, b, c = (np.array([1.0, 1.0, 1.0]) + 1e-3 * rng.standard_normal(3)).tolist()
    for _ in range(round(BURN / DT)):
        a, b, c = rk4_scalar(a, b, c)
    n = round(T / DTO)
    true = np.empty((n, 3))
    for i in range(n):
        for _ in range(SUB):
            a, b, c = rk4_scalar(a, b, c)
        true[i] = a, b, c
    return true, true + DELTA * rng.standard_normal(true.shape)


def anchors(lo, hi, stride=1):
    """Anchor obs indices i with inputs [i-L+1, i] and targets i+HMAX inside [lo, hi)."""
    return np.arange(lo + L - 1, hi - HMAX, stride)


# ---------------- metrics ----------------
def mse(pred, truth):              # 3 coordinates pooled (mean over coordinates)
    return float(((pred - truth) ** 2).mean())


def murphy(p, o):
    """BS = REL - RES + UNC (+ within-bin residual); 10 equal-width bins on [0,1], last bin includes 1.0."""
    p = np.clip(p, 0, 1)
    k = np.minimum(np.floor(p * 10).astype(int), 9)
    N, ob = len(p), o.mean()
    n = np.bincount(k, minlength=10)
    nz = n > 0
    fb = np.bincount(k, p, 10)[nz] / n[nz]
    obk = np.bincount(k, o, 10)[nz] / n[nz]
    rel = float((n[nz] * (fb - obk) ** 2).sum() / N)
    res = float((n[nz] * (obk - ob) ** 2).sum() / N)
    unc, bs = float(ob * (1 - ob)), float(np.mean((p - o) ** 2))
    return dict(BS=bs, REL=rel, RES=res, UNC=unc, BSS=1 - bs / unc, resid=bs - (rel - res + unc))


# ---------------- NG-RC (Gauthier et al. 2021) and EDMD on the 2-tap delay state ----------------
class Delay:
    def __init__(self, obs):
        self.mu, self.sd = obs.mean(0), obs.std(0)

    def z(self, cur, prev):        # standardized 6-vector (current, previous observation)
        return np.concatenate([(cur - self.mu) / self.sd, (prev - self.mu) / self.sd], -1)


def monomials(z, deg):
    """[1, z, all unique monomials of degree 2..deg] of the columns of z."""
    cols = [np.ones(len(z))] + [z[:, i] for i in range(z.shape[1])]
    for d in range(2, deg + 1):
        for idx in itertools.combinations_with_replacement(range(z.shape[1]), d):
            cols.append(np.prod(z[:, idx], 1))
    return np.stack(cols, 1)


def ngrc_fit(obs, dl, alpha):
    F = monomials(dl.z(obs[1:-1], obs[:-2]), 2)               # 28 features
    Y = obs[2:] - obs[1:-1]                                    # next observation's increment
    return np.linalg.solve(F.T @ F + alpha * np.eye(F.shape[1]), F.T @ Y)


def ngrc_run(W, dl, obs, idx, clip=1e3):
    """Iterate from anchors idx; returns forecasts at every step 1..HMAX, (N,HMAX,3). Clip guards blow-ups (counted)."""
    cur, prev = obs[idx].copy(), obs[idx - 1].copy()
    out = np.empty((len(idx), HMAX, 3))
    for h in range(HMAX):
        nxt = np.clip(cur + monomials(dl.z(cur, prev), 2) @ W, -clip, clip)
        prev, cur = cur, np.nan_to_num(nxt, nan=0.0)
        out[:, h] = cur
    return out


def edmd_fit(obs, dl):
    G = monomials(dl.z(obs[1:-1], obs[:-2]), 3)               # 84-term dictionary
    A = monomials(dl.z(obs[2:], obs[1:-1]), 3)
    return np.linalg.lstsq(G, A, rcond=None)[0]               # K = G^+ A (row convention: A ~ G K)


def edmd_run(K, dl, obs, idx, project):
    """Lifted iteration psi <- psi K; linear block read out. project=True re-lifts the linear block every step."""
    p = monomials(dl.z(obs[idx], obs[idx - 1]), 3)
    out = np.empty((len(idx), HMAX, 3))
    for h in range(HMAX):
        p = p @ K
        lin = p[:, 1:7]
        if project:
            p = monomials(np.clip(np.nan_to_num(lin), -50, 50), 3)
        out[:, h] = lin[:, :3] * dl.sd + dl.mu
    return out


def diverged(fc):
    return (not np.all(np.isfinite(fc))) or np.abs(fc).max() > 1e3


# ---------------- transformer ----------------
class TF(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(3, 64)
        self.pos = nn.Parameter(0.02 * torch.randn(L, 64))
        layer = nn.TransformerEncoderLayer(64, 4, 256, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(64)
        self.state = nn.Linear(64, 3 * len(H))                 # per-lead state heads
        self.event = nn.Linear(64, len(H))                     # per-lead event heads

    def forward(self, x):
        h = self.norm(self.enc(self.inp(x) + self.pos))[:, -1]
        return self.state(h).view(-1, len(H), 3), self.event(h)


def bce(p, o):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@torch.no_grad()
def tf_predict(model, obs, idx, mu, sd):
    """Returns state (N,7,3) and event logits (N,7)."""
    model.eval()
    on = torch.tensor((obs - mu) / sd, dtype=torch.float32, device=DEV)
    off = torch.arange(-L + 1, 1, device=DEV)
    S, Z = [], []
    for ch in np.array_split(idx, max(1, len(idx) // 4096)):
        i = torch.tensor(ch, device=DEV)
        ps, pe = model(on[i[:, None] + off])
        S.append(ps.double().cpu().numpy() * sd + mu); Z.append(pe.double().cpu().numpy())
    model.train()
    return np.concatenate(S), np.concatenate(Z)


def tf_train(seed, obs, true, tr_idx, va_idx, mu, sd, var_clim, budgets, wall_cap):
    """Budget ladder; each budget early-stops (best validation loss checkpoint) on the validation slice (A1.3).
    The next budget is tried only if the validation gate S(0.25T*) < GATE; the test set is never consulted."""
    on = torch.tensor((obs - mu) / sd, dtype=torch.float32, device=DEV)
    tn = torch.tensor((true - mu) / sd, dtype=torch.float32, device=DEV)
    ev = torch.tensor(true[:, 0] > 0, dtype=torch.float32, device=DEV)
    off, Ht = torch.arange(-L + 1, 1, device=DEV), torch.tensor(H, device=DEV)
    tr_t = torch.tensor(tr_idx, device=DEV)
    va_truth, va_evt = true[va_idx[:, None] + H], (true[va_idx[:, None] + H, 0] > 0).astype(float)
    log, t0 = [], time.time()
    for steps in budgets:
        torch.manual_seed(seed)
        model = TF().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
        best, best_state, best_it = 1e9, None, 0
        tb = time.time()
        for it in range(steps):
            i = tr_t[torch.randint(len(tr_t), (512,), device=DEV)]
            ps, pe = model(on[i[:, None] + off])
            loss = ((ps - tn[i[:, None] + Ht]) ** 2).mean() + nn.functional.binary_cross_entropy_with_logits(
                pe, ev[i[:, None] + Ht])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            stop = time.time() - tb > wall_cap
            if (it + 1) % 2000 == 0 or it + 1 == steps or stop:
                vs, vz = tf_predict(model, obs, va_idx, mu, sd)
                vl = ((vs - va_truth) ** 2).mean() / var_clim + bce(sigmoid(vz), va_evt).mean()
                if vl < best:
                    best, best_it, best_state = vl, it + 1, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if stop:
                break
        model.load_state_dict(best_state)
        vs, _ = tf_predict(model, obs, va_idx, mu, sd)
        s_val = 1 - mse(vs[:, 0], va_truth[:, 0]) / var_clim
        log.append(dict(steps=steps, steps_done=it + 1, best_step=best_it, val_skill_lead0=s_val, val_loss=float(best),
                        wall_s=time.time() - tb))
        print(f"  tf seed {seed}: budget {steps} steps (best @ {best_it}) -> val S(0.25T*) {s_val:.4f}, "
              f"{time.time() - tb:.0f}s", flush=True)
        if s_val >= GATE or time.time() - t0 >= wall_cap:
            break
    return model, log


# ---------------- oracles (true model) ----------------
def propagate(x):
    """Integrate ensembles x (B,P,3) and yield (lead index, state) at the recorded RK4 step counts."""
    steps = {int(h * SUB): j for j, h in enumerate(H)}
    for s in range(1, HMAX * SUB + 1):
        x = rk4_torch(x)
        if s in steps:
            yield steps[s], x


def oracle1(obs, idx, seed):
    """Single-observation oracle: 256 members from last obs + N(0, delta^2)."""
    g = torch.Generator().manual_seed(1000 + seed)
    S, P = np.empty((len(idx), len(H), 3)), np.empty((len(idx), len(H)))
    for ch in np.array_split(np.arange(len(idx)), max(1, len(idx) // 500)):
        x = torch.tensor(obs[idx[ch]])[:, None] + DELTA * torch.randn(len(ch), NENS, 3, generator=g, dtype=torch.float64)
        for j, xs in propagate(x):
            S[ch, j] = xs.mean(1).numpy(); P[ch, j] = (xs[..., 0] > 0).double().mean(1).numpy()
    return S, P


def oracle20(obs, idx, seed):
    """20-observation oracle (A1.4): bootstrap particle filter over the window with the true model, NPF particles,
    Gaussian likelihood (delta), multinomial resampling when ESS < NPF/2 followed by a Liu-West resample-move
    (x <- a x + (1-a) mean + sqrt(1-a^2) * chol(cov) z, a = LW_A; preserves the cloud's mean and covariance and
    stops a deterministic model from collapsing onto duplicates). Forecast = importance-weighted ensemble."""
    g = torch.Generator().manual_seed(2000 + seed)
    S, P = np.empty((len(idx), len(H), 3)), np.empty((len(idx), len(H)))
    ess_final, n_resample = [], []
    for ch in np.array_split(np.arange(len(idx)), max(1, len(idx) // 250)):
        B = len(ch)
        win = torch.tensor(obs[idx[ch][:, None] + np.arange(-L + 1, 1)])          # (B,L,3)
        x = win[:, :1] + DELTA * torch.randn(B, NPF, 3, generator=g, dtype=torch.float64)   # posterior of obs 0, flat prior
        logw = torch.zeros(B, NPF, dtype=torch.float64)
        nres = torch.zeros(B)
        for k in range(1, L):
            for _ in range(SUB):
                x = rk4_torch(x)
            logw = logw - ((x - win[:, k:k + 1]) ** 2).sum(-1) / (2 * DELTA ** 2)
            w = torch.softmax(logw, 1)
            ess = 1 / (w ** 2).sum(1)
            rs = ess < NPF / 2
            if k < L - 1 and rs.any():
                b = rs.nonzero()[:, 0]
                pick = torch.multinomial(w[b], NPF, replacement=True, generator=g)
                xb = torch.gather(x[b], 1, pick[..., None].expand(-1, -1, 3))
                m = xb.mean(1, keepdim=True)
                C = ((xb - m).transpose(1, 2) @ (xb - m)) / (NPF - 1) + 1e-12 * torch.eye(3, dtype=torch.float64)
                Lc = torch.linalg.cholesky(C)
                z = torch.randn(len(b), NPF, 3, generator=g, dtype=torch.float64)
                x[b] = LW_A * xb + (1 - LW_A) * m + math.sqrt(1 - LW_A ** 2) * z @ Lc.transpose(1, 2)
                logw[b] = 0.0
                nres[b] += 1
        w = torch.softmax(logw, 1)
        ess_final.append((1 / (w ** 2).sum(1)).numpy()); n_resample.append(nres.numpy())
        for j, xs in propagate(x):
            S[ch, j] = (w[..., None] * xs).sum(1).numpy(); P[ch, j] = (w * (xs[..., 0] > 0).double()).sum(1).numpy()
    ess_final, n_resample = np.concatenate(ess_final), np.concatenate(n_resample)
    info = dict(n_particles=NPF, ess_final_mean=float(ess_final.mean()), ess_final_median=float(np.median(ess_final)),
                ess_final_min=float(ess_final.min()), ess_final_frac_below_10=float((ess_final < 10).mean()),
                resamples_per_window_mean=float(n_resample.mean()))
    return S, P, info


def logistic_1d(x_tr, y_tr, x_ap):
    """Platt / 2-parameter logistic (unregularized to machine precision: C = 1e6)."""
    m, s = x_tr.mean(), x_tr.std() + 1e-12
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(((x_tr - m) / s)[:, None], y_tr)
    return lr.predict_proba(((x_ap - m) / s)[:, None])[:, 1], dict(coef=float(lr.coef_[0, 0] / s),
                                                                    intercept=float(lr.intercept_[0] - lr.coef_[0, 0] * m / s))


# ---------------- one seed ----------------
def run_seed(seed, test_true, test_obs, te_idx):
    t0 = time.time()
    true, obs = trajectory(seed, T_TRAIN)
    n = len(obs)
    tr_hi = n - round((VAL_UNITS if not QUICK else 0.1 * T_TRAIN) / DTO)   # train [0, tr_hi), validation [tr_hi, n)
    tr_idx, va_idx = anchors(0, tr_hi), anchors(tr_hi, n, STRIDE_EVAL)
    clim_mu, var_c = true[:tr_hi].mean(0), true[:tr_hi].var(0)
    var_clim = float(var_c.mean())                             # A1.7: mean over coordinates of per-coordinate variance
    sig_clim = np.sqrt(var_c)
    base = float((true[:tr_hi, 0] > 0).mean())
    info = dict(seed=seed, n_train_anchors=len(tr_idx), n_val_anchors=len(va_idx), base_rate=base,
                clim_mean=clim_mu.tolist(), clim_var=var_c.tolist(), var_clim=var_clim)
    print(f"seed {seed}: trajectory {time.time() - t0:.0f}s", flush=True)
    skill = lambda p, t: 1 - mse(p, t) / var_clim

    dl = Delay(obs[:tr_hi])
    # NG-RC ridge alpha on the validation slice: mean state skill over leads <= T*
    alphas = 10.0 ** np.arange(-8, 3)
    a_scores = []
    for a in alphas:
        fc = ngrc_run(ngrc_fit(obs[:tr_hi], dl, a), dl, obs, va_idx)
        a_scores.append(np.mean([skill(fc[:, h - 1], true[va_idx + h]) for h in H[BEFORE]]))
    alpha = float(alphas[int(np.nanargmax(a_scores))])
    W = ngrc_fit(obs[:tr_hi], dl, alpha)
    info["ngrc"] = dict(alpha=alpha, val_scores={f"{a:.0e}": float(s) for a, s in zip(alphas, a_scores)})

    # canon: sigma0, lam_hat from ngrc error growth on the validation slice, leads <= T* only (A1.2)
    fv = ngrc_run(W, dl, obs, va_idx)
    err = fv - true[va_idx[:, None] + np.arange(1, HMAX + 1)]
    rms_c = np.sqrt((err ** 2).mean(0))                         # (HMAX,3)
    rms = np.sqrt((rms_c ** 2).mean(1))
    tau = np.arange(1, HMAX + 1) * DTO
    ok = (tau <= TSTAR) & (np.cumsum(rms >= SAT_FRAC * math.sqrt(var_clim)) == 0)
    hs = int(ok.sum())
    lam_hat, _ = np.polyfit(tau[:hs], np.log(rms[:hs]), 1)
    sigma0 = np.exp((np.log(rms_c[:hs]) - lam_hat * tau[:hs, None]).mean(0))
    assert tau[hs - 1] <= TSTAR
    info["canon"] = dict(lam_hat=float(lam_hat), lam_true=LAM, sigma0=sigma0.tolist(), fit_steps=hs,
                         fit_tau_max=float(tau[hs - 1]), rms_curve=rms.tolist())
    print(f"  ngrc alpha {alpha:.0e}; canon lam_hat {lam_hat:.4f} over tau <= {tau[hs - 1]:.2f}", flush=True)

    # EDMD: pure lifted iteration unless it diverges on validation, then project; the projected variant is also
    # reported as an extra, non-verdict arm `edmd-proj`.
    K = edmd_fit(obs[:tr_hi], dl)
    project = diverged(edmd_run(K, dl, obs, va_idx, False))
    info["edmd"] = dict(mode="project" if project else "lifted", lifted_diverged_on_val=bool(project))

    # ngrc-p (logistic per lead on ngrc features + ngrc forecast; C on validation) and ngrc-platt (A1.2: per-lead
    # 2-parameter logistic on the ngrc mean mu_x, fitted on train)
    lp_idx = tr_idx[::STRIDE_EVAL]
    ftr = ngrc_run(W, dl, obs, lp_idx)
    fte = ngrc_run(W, dl, test_obs, te_idx)
    def lp_x(o, idx, fc, j):
        return np.concatenate([monomials(dl.z(o[idx], o[idx - 1]), 2)[:, 1:], (fc[:, H[j] - 1] - clim_mu) / sig_clim], 1)
    p_ngp, p_platt, lp_info, platt_info = [], [], [], []
    for j in range(len(H)):
        Xtr, ytr = lp_x(obs, lp_idx, ftr, j), true[lp_idx + H[j], 0] > 0
        Xva, yva = lp_x(obs, va_idx, fv, j), true[va_idx + H[j], 0] > 0
        m_, s_ = Xtr.mean(0), Xtr.std(0) + 1e-12
        best = None
        for C in (1e-3, 1e-2, 1e-1, 1, 10):
            lr = LogisticRegression(C=C, max_iter=2000).fit((Xtr - m_) / s_, ytr)
            ll = bce(lr.predict_proba((Xva - m_) / s_)[:, 1], yva).mean()
            if best is None or ll < best[0]:
                best = (ll, C, lr)
        p_ngp.append(best[2].predict_proba((lp_x(test_obs, te_idx, fte, j) - m_) / s_)[:, 1])
        lp_info.append(dict(C=best[1], val_logloss=float(best[0])))
        pp, pi = logistic_1d(ftr[:, H[j] - 1, 0], ytr, fte[:, H[j] - 1, 0])
        p_platt.append(pp); platt_info.append(pi)
    info["ngrc_p"], info["ngrc_platt"] = lp_info, platt_info

    # transformer + tf-platt (per-lead logistic recalibration of tf's event logit on the validation slice)
    budgets = [3000] if QUICK else [30000]
    tmu, tsd = obs[:tr_hi].mean(0), obs[:tr_hi].std(0)
    model, tf_log = tf_train(seed, obs, true, tr_idx, va_idx, tmu, tsd, var_clim, budgets, TF_WALL)
    info["tf_budgets"] = tf_log
    _, vz = tf_predict(model, obs, va_idx, tmu, tsd)
    ts, tz = tf_predict(model, test_obs, te_idx, tmu, tsd)
    p_tfp, tfp_info = [], []
    for j in range(len(H)):
        pp, pi = logistic_1d(vz[:, j], true[va_idx + H[j], 0] > 0, tz[:, j])
        p_tfp.append(pp); tfp_info.append(pi)
    info["tf_platt"] = tfp_info

    # ---------------- test ----------------
    N = len(te_idx)
    info["ngrc_test_clipped_frac"] = float((np.abs(fte) >= 1e3).mean())
    ng = fte[:, H - 1]
    ed = edmd_run(K, dl, test_obs, te_idx, project)[:, H - 1]
    edp = edmd_run(K, dl, test_obs, te_idx, True)[:, H - 1]
    sig = np.minimum(sigma0 * np.exp(lam_hat * H * DTO)[:, None], sig_clim)          # (7,3)
    rho = np.sqrt(np.maximum(0, 1 - sig ** 2 / sig_clim ** 2))                         # A1.7 shrink
    cmu = clim_mu + rho * (ng - clim_mu)
    t_or = time.time()
    o1s, o1p = oracle1(test_obs, te_idx, seed)
    o20s, o20p, pf_info = oracle20(test_obs, te_idx, seed)
    info["oracle20_pf"] = pf_info
    print(f"  oracles {time.time() - t_or:.0f}s; PF final ESS mean {pf_info['ess_final_mean']:.1f} / {NPF}", flush=True)
    last = np.repeat(test_obs[te_idx][:, None], len(H), 1)
    det = lambda st: (np.nan_to_num(st[..., 0]) > 0).astype(float)
    arms = {
        "clim": (np.broadcast_to(clim_mu, (N, len(H), 3)), np.full((N, len(H)), base)),
        "persist": (last, det(last)),
        "oracle1": (o1s, o1p),
        "oracle20": (o20s, o20p),
        "tf": (ts, sigmoid(tz)),
        "tf-platt": (ts, np.stack(p_tfp, 1)),
        "ngrc": (ng, det(ng)),
        "ngrc-platt": (ng, np.stack(p_platt, 1)),
        "ngrc-p": (ng, np.stack(p_ngp, 1)),
        "edmd": (ed, det(ed)),
        "edmd-proj": (edp, det(edp)),
        "canon": (cmu, norm.cdf(cmu[..., 0] / sig[:, 0])),
    }
    truth = test_true[te_idx[:, None] + H]                      # (N,7,3)
    evt = (truth[..., 0] > 0).astype(float)
    res, raw = {}, {}
    for name, (st, pr) in arms.items():
        se = ((np.nan_to_num(st, nan=0.0, posinf=1e6, neginf=-1e6) - truth) ** 2).mean(-1)   # (N,7) pooled sq error
        res[name] = [dict(MSE=float(se[:, j].mean()), S=1 - float(se[:, j].mean()) / var_clim,
                          **murphy(pr[:, j], evt[:, j])) for j in range(len(H))]
        raw[name] = (se, np.asarray(pr, float))
    info["wall_s"] = time.time() - t0
    return info, res, raw, evt


def main():
    t_start = time.time()
    bar_sha = hashlib.sha256((HERE / "BAR_F.md").read_bytes()).hexdigest()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if bar_sha != BAR_SHA_IMPLEMENTED:
        print(f"!!! BAR_F.md sha {bar_sha} differs from the implemented bar {BAR_SHA_IMPLEMENTED}")
    print(f"bar_sha256 {bar_sha}\nscript_sha256 {script_sha}\nT* {TSTAR:.4f}; lead steps {H.tolist()} "
          f"(tau {(H * DTO).round(2).tolist()})", flush=True)
    if DEV.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(3.0 / (torch.cuda.get_device_properties(0).total_memory / 2**30))
    test_true, test_obs = trajectory(99, T_TEST)
    te_idx = anchors(0, len(test_obs), STRIDE_EVAL)
    per_seed, infos, raws = {}, {}, {}
    for s in SEEDS:
        ck = CACHE / f'p2f_seed{s}_{"quick" if QUICK else "full"}.pkl'
        if os.environ.get('P2F_USE_CACHE') == '1' and ck.exists():
            info, res, raw, evt = pickle.loads(ck.read_bytes())
        else:
            info, res, raw, evt = run_seed(s, test_true, test_obs, te_idx)
            ck.write_bytes(pickle.dumps((info, res, raw, evt)))
        infos[s], per_seed[s], raws[s] = info, res, raw
        print(f"  seed {s} done {info['wall_s']:.0f}s; " + ", ".join(f"{a} S0 {res[a][0]['S']:.3f}" for a in res),
              flush=True)
    arms = list(per_seed[SEEDS[0]])
    J = range(len(H))
    mean = {a: [{k: float(np.mean([per_seed[s][a][j][k] for s in SEEDS])) for k in per_seed[SEEDS[0]][a][j]}
                for j in J] for a in arms}
    exempt, deterministic = {"clim", "oracle1", "oracle20"}, {"persist", "ngrc", "edmd", "edmd-proj"}
    gate = {a: ("exempt" if a in exempt else ("learned" if mean[a][0]["S"] >= GATE else "VOID")) for a in arms}

    # block bootstrap CIs (A1.7): 10-time-unit blocks of the test trajectory, same resample for every seed and arm
    blk = np.floor(te_idx * DTO / BLOCK_UNITS).astype(int)
    blocks = [np.flatnonzero(blk == b) for b in np.unique(blk)]
    rng = np.random.default_rng(7)
    boot = {}                                                   # (arm, stat, j) -> list over draws of seed-mean
    ci_arms = ["canon", "tf", "tf-platt", "ngrc-platt", "oracle20"]
    for _ in range(NBOOT if not QUICK else 100):
        ii = np.concatenate([blocks[b] for b in rng.integers(len(blocks), size=len(blocks))])
        for a in ci_arms:
            for j in J:
                ms = [murphy(raws[s][a][1][ii, j], evt[ii, j]) for s in SEEDS]
                boot.setdefault((a, "REL", j), []).append(np.mean([m["REL"] for m in ms]))
                boot.setdefault((a, "BSS", j), []).append(np.mean([m["BSS"] for m in ms]))
                boot.setdefault((a, "MSE", j), []).append(np.mean([raws[s][a][0][ii, j].mean() for s in SEEDS]))
        for j in J:
            boot.setdefault(("ratio", "MSE", j), []).append(boot[("canon", "MSE", j)][-1] / boot[("tf", "MSE", j)][-1])
    ci = {f"{a}|{st}|{LEAD_FR[j]}": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for (a, st, j), v in boot.items()}
    c95 = lambda a, st, j: "[%.4f, %.4f]" % tuple(ci[f"{a}|{st}|{LEAD_FR[j]}"])

    print("\nper lead, mean over seeds (test, N=%d anchors per seed)" % len(te_idx))
    print(f"{'arm':10s} {'tau/T*':>6s} {'S':>8s} {'MSE':>8s} {'BS':>7s} {'REL':>7s} {'RES':>7s} {'UNC':>7s} {'BSS':>8s} "
          f"{'resid':>9s}")
    for a in arms:
        for j in J:
            m = mean[a][j]
            print(f"{a:10s} {LEAD_FR[j]:6.2f} {m['S']:8.4f} {m['MSE']:8.3f} {m['BS']:7.4f} {m['REL']:7.4f} "
                  f"{m['RES']:7.4f} {m['UNC']:7.4f} {m['BSS']:8.4f} {m['resid']:9.2e}"
                  + ("   (0/1 forecast: REL printed, never cited)" if a in deterministic and j == 0 else ""))
    print(f"\nlearned gate S(0.25T*) >= {GATE} (seed mean; per seed):")
    for a in arms:
        print(f"  {a:10s} {mean[a][0]['S']:.4f} [{gate[a]}]  " + " ".join(f"{per_seed[s][a][0]['S']:.4f}" for s in SEEDS))
    max_resid = max(abs(per_seed[s][a][j]["resid"]) for s in SEEDS for a in arms for j in J)
    print(f"Murphy check: max |BS - (REL - RES + UNC)| over all arms, leads, seeds = {max_resid:.3e} (within-bin term)")

    # ---------------- verdict (A1) ----------------
    live = [j for j in np.flatnonzero(AFTER) if mean["oracle20"][j]["BSS"] >= LIVE_BSS]
    print("\nlive leads > T* (oracle20 BSS >= %.2f): " % LIVE_BSS + ", ".join(
        f"{LEAD_FR[j]} ({mean['oracle20'][j]['BSS']:.4f}{' LIVE' if j in live else ' dead'}; oracle1 "
        f"{mean['oracle1'][j]['BSS']:.4f})" for j in np.flatnonzero(AFTER)))
    V = lambda a: gate[a] != "VOID"
    ratios = {j: mean["canon"][j]["MSE"] / mean["tf"][j]["MSE"] for j in np.flatnonzero(BEFORE)}
    match = V("canon") and V("tf") and all(0.8 <= r <= 1.25 for r in ratios.values())
    for j, r in ratios.items():
        print(f"Skill match {LEAD_FR[j]} T*: MSE(canon)/MSE(tf) = {mean['canon'][j]['MSE']:.4f}/{mean['tf'][j]['MSE']:.4f}"
              f" = {r:.4f} CI {c95('ratio', 'MSE', j)} in [0.8, 1.25] -> {0.8 <= r <= 1.25}")
    per_lead = {}
    for j in live:
        c, t, tp, npl = mean["canon"][j], mean["tf"][j], mean["tf-platt"][j], mean["ngrc-platt"][j]
        d = dict(
            prediction=bool(match and c["REL"] <= 0.01 and t["REL"] >= 0.05),
            house=bool(V("canon") and V("tf") and c["BSS"] >= t["BSS"] - 0.02),
            canon_claim=bool(V("canon") and V("ngrc-platt") and c["REL"] <= 0.01 and c["BSS"] >= npl["BSS"] - 0.02),
            counter=bool(V("tf") and t["REL"] <= 0.02),
            kill_tf_platt=bool(V("tf-platt") and V("canon") and tp["REL"] <= 0.02 and tp["BSS"] >= c["BSS"] - 0.02))
        per_lead[float(LEAD_FR[j])] = d
        L_ = LEAD_FR[j]
        print(f"\nlead {L_} T*:")
        print(f"  Prediction: skill match {match}; REL(canon) {c['REL']:.4f} {c95('canon', 'REL', j)} <= 0.01; "
              f"REL(tf) {t['REL']:.4f} {c95('tf', 'REL', j)} >= 0.05 -> {d['prediction']}")
        print(f"  Counter: REL(tf) {t['REL']:.4f} <= 0.02 -> {d['counter']}")
        print(f"  House addition: BSS(canon) {c['BSS']:.4f} {c95('canon', 'BSS', j)} >= BSS(tf) {t['BSS']:.4f} "
              f"{c95('tf', 'BSS', j)} - 0.02 -> {d['house']}")
        print(f"  Canon claim (A1.2): REL(canon) {c['REL']:.4f} <= 0.01 and BSS(canon) {c['BSS']:.4f} >= "
              f"BSS(ngrc-platt) {npl['BSS']:.4f} {c95('ngrc-platt', 'BSS', j)} - 0.02 -> {d['canon_claim']}")
        print(f"  KILL via tf-platt (A1.3): REL(tf-platt) {tp['REL']:.4f} {c95('tf-platt', 'REL', j)} <= 0.02 and "
              f"BSS(tf-platt) {tp['BSS']:.4f} {c95('tf-platt', 'BSS', j)} >= BSS(canon) {c['BSS']:.4f} - 0.02 "
              f"-> {d['kill_tf_platt']}")
    if not live:
        verdict = "OPEN (no live lead > T*)"
    elif not (V("canon") and V("tf")):
        verdict = "OPEN (void arm under L-LEARN: %s)" % ",".join(a for a in ("canon", "tf") if not V(a))
    else:
        kill = [d["counter"] or d["kill_tf_platt"] for d in per_lead.values()]
        win = [d["prediction"] and d["house"] and d["canon_claim"] for d in per_lead.values()]
        if all(kill):
            verdict = "KILL"
        elif all(win) and not any(kill):
            verdict = "PASS pending deep Koopman"
        else:
            verdict = "OPEN"
    print(f"\nVERDICT: {verdict}")

    peak = torch.cuda.max_memory_allocated() / 2**30 if DEV.type == "cuda" else 0.0
    j_live = live
    deciding = {float(LEAD_FR[j]): dict(REL_canon=mean["canon"][j]["REL"], REL_tf=mean["tf"][j]["REL"],
                                        REL_tf_platt=mean["tf-platt"][j]["REL"], BSS_canon=mean["canon"][j]["BSS"],
                                        BSS_tf=mean["tf"][j]["BSS"], BSS_tf_platt=mean["tf-platt"][j]["BSS"],
                                        BSS_ngrc_platt=mean["ngrc-platt"][j]["BSS"],
                                        BSS_oracle20=mean["oracle20"][j]["BSS"]) for j in np.flatnonzero(AFTER)}
    S_before = {a: float(np.mean([mean[a][j]["S"] for j in np.flatnonzero(BEFORE)])) for a in arms}
    out = dict(
        bar_sha256=bar_sha, script_sha256=script_sha, bar_sha_matches_implemented=bar_sha == BAR_SHA_IMPLEMENTED,
        verdict=verdict, live_leads=[float(LEAD_FR[j]) for j in j_live], per_live_lead=per_lead,
        skill_match=dict(ok=bool(match), mse_ratio={float(LEAD_FR[j]): r for j, r in ratios.items()}),
        deciding_after_Tstar=deciding, S_before_Tstar_mean=S_before, gate=gate,
        per_lead_mean=mean, per_lead_per_seed=per_seed, ci95_block_bootstrap=ci,
        seed_info=infos, murphy_max_resid=max_resid,
        config=dict(T_star=TSTAR, lead_fr=LEAD_FR.tolist(), lead_obs_steps=H.tolist(), lead_tau=(H * DTO).tolist(),
                    delta=DELTA, dt=DT, dt_obs=DTO, L=L, T_train=T_TRAIN, T_test=T_TEST, burn=BURN,
                    val_units=VAL_UNITS, test_stride_obs=STRIDE_EVAL, n_test_anchors=len(te_idx), n_ens_oracle1=NENS,
                    n_particles_oracle20=NPF, liu_west_a=LW_A, canon_sat_frac=SAT_FRAC, gate=GATE,
                    live_bss=LIVE_BSS, n_boot=NBOOT, block_units=BLOCK_UNITS, device=str(DEV)),
        deviations=[
            "Leads rounded to the dt_obs = 0.05 grid (obs steps %s, tau %s vs exact %s) so every arm, including the "
            "iterated ngrc/edmd, forecasts at the same instants." % (H.tolist(), (H * DTO).round(3).tolist(),
                                                                     (LEAD_FR * TSTAR).round(3).tolist()),
            "Validation slice = last 1,000 units of each training trajectory; all fitting uses the first 9,000 "
            "(tf-platt is fitted on the validation slice, as A1.3 says). Every hyperparameter (ridge alpha, "
            "ngrc-p C, tf budget and early-stopping checkpoint, canon sigma0/lam_hat, EDMD mode) is chosen on "
            "validation only; the test trajectory is used only for the reported numbers.",
            "Test trajectory cut from 2,000 to 500 time units (coordinator deadline, first cut in the ordered list); test "
            "anchors every 5 observations (0.25 time units) of the seed-99 trajectory, not every observation.",
            "canon: lam_hat is one exponent fitted log-linearly to the pooled (3-coordinate) ngrc RMS error on "
            "validation over leads tau <= T* and before the RMS first reaches %.1f x sqrt(Var_clim); sigma0 per "
            "coordinate at that lam_hat. sigma(tau) = min(sigma0 exp(lam_hat tau), sigma_clim) per coordinate; "
            "shrink rho = sqrt(max(0, 1 - sigma^2/sigma_clim^2)) per coordinate (A1.7)." % SAT_FRAC,
            "ngrc-p features: the 27 non-constant NG-RC features of the anchor plus the ngrc forecast at that lead "
            "(standardized); read as the strongest variant. ngrc-p, ngrc-platt and tf-platt take their state "
            "forecast (and so their L-LEARN gate) from ngrc / tf.",
            "ngrc-platt: per-lead logistic on the ngrc x forecast fitted on train anchors every 5 observations "
            "(36k), C = 1e6 (effectively unregularized).",
            "tf: trained on true-state targets and event labels of the train slice (MSE on standardized state + "
            "BCE), last-token readout; early stopping = best validation-loss checkpoint (evaluated every 2,000 "
            "steps); one budget of at most 30,000 steps capped at 10 min wall clock per seed (coordinator deadline, "
            "second cut); the planned 30k/90k/270k ladder up to 30 min was not run. steps_done recorded per seed.",
            "oracle1: last observation + N(0, delta^2), 256 members, state = ensemble mean.",
            "oracle20: bootstrap particle filter with %d particles initialised at obs[i-19] + N(0, delta^2); "
            "multinomial resampling when ESS < N/2 followed by a Liu-West resample-move (a = %.2f) that keeps the "
            "cloud's mean and covariance but adds jitter the deterministic true model does not have; forecast "
            "ensemble is importance-weighted. ESS recorded in seed_info.oracle20_pf." % (NPF, LW_A),
            "EDMD: pure lifted iteration of K unless it diverges on validation (non-finite or |x| > 1e3); mode "
            "recorded per seed. The re-lifted (project-to-dictionary) variant is printed as extra arm `edmd-proj` "
            "and takes no part in the verdict.",
            "Verdict aggregation over live leads: KILL iff (counter or tf-platt kill) holds at every live lead; PASS "
            "iff prediction, House addition and the A1.2 canon claim hold at every live lead and no kill condition "
            "holds at any; OPEN otherwise, and OPEN if canon or tf is void. The bar says 'per lead' without an "
            "aggregation rule for KILL; this is the reading fixed before the deciding run.",
            "The skill match (A1.6) replaces |dS| <= 0.05 and uses the seed-mean pooled MSE at each lead <= T*.",
            "CIs: 95%% percentile block bootstrap, %d draws, 10-time-unit blocks of test anchors, same draw for all "
            "seeds and arms, statistic = seed mean. The verdict uses point estimates; CIs are reported." % NBOOT,
        ],
        wall_clock_s=time.time() - t_start, peak_gpu_mem_gib=peak,
    )
    print(f"wall clock {out['wall_clock_s']:.0f}s; peak GPU memory {peak:.3f} GiB")
    if not QUICK:
        (HERE / "results_p2f.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
