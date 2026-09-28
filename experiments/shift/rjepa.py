# R-JEPA core (Foreman): a JEPA latent predictor + candidate-ranking heads on an exact-truth bed.
# Bar: BAR.md (registered before this file). Tests: test_rjepa.py.
import json, sys, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
K, D, RHO, SIG_E = 4, 2, 1.0, 0.02


# ---- theorem T-A ---------------------------------------------------------------------------------------
def linear_equivariant_resolvent(s, alpha, beta, gamma):
    """(I - gamma*(alpha I + beta J))^-1 s; None when the spectral radius of gamma*W is >= 1."""
    n = len(s)
    W = alpha * np.eye(n) + beta * np.ones((n, n))
    if max(abs(np.linalg.eigvals(gamma * W))) >= 1 - 1e-6:
        return None
    return np.linalg.solve(np.eye(n) - gamma * W, s)


# ---- bed `shift` ----------------------------------------------------------------------------------------
def make_bed(n, sigma, sigma_e=SIG_E, seed=0, mix=0.0):
    """mix = share of the error variance sigma^2 that is idiosyncratic per candidate (A4); 0 = all shared."""
    rng = np.random.default_rng([seed, int(sigma * 1000), int(sigma_e * 1000)] + ([int(mix * 1000)] if mix else []))
    sh = sigma * np.sqrt(1 - mix)
    sid = sigma_e if mix == 0 else np.sqrt(sigma_e ** 2 + sigma ** 2 * mix)
    y = 3 * rng.standard_normal((n, D))
    s = y - sh * rng.standard_normal((n, D))              # s | y ~ N(y, sh^2 I) exactly
    r = rng.standard_normal((n, K, D))
    a = -y[:, None] + RHO * r                             # planner aims at the goal from its own estimate
    z = s[:, None] + a + sid * rng.standard_normal((n, K, D))
    return dict(y=y, a=a, r=r, z=z, best=np.linalg.norm(z, axis=-1).argmin(1),
                zhat_true=y[:, None] + a, sigma=sh, sigma_e=sid, seed=seed)


def hit(pick, b):
    return float((pick == b["best"]).mean())


def dist_pick(zhat):
    return np.linalg.norm(zhat, axis=-1).argmin(1)


def bayes_pick(b, M=1024, chunk=256, max_rank=None):
    """Top-1 Bayes rule; max_rank restricts it to candidates whose plug-in distance rank <= max_rank
    (D-JEPA's Cor 1 reach at K = 4, eps = 0.2: ranks {0, 1})."""
    rng = np.random.default_rng([b["seed"], 7])
    out = []
    for i in range(0, len(b["y"]), chunk):
        y, a = b["y"][i:i + chunk], b["a"][i:i + chunk]
        s = y[:, None] - b["sigma"] * rng.standard_normal((len(y), M, D))
        z = s[:, :, None] + a[:, None] + b["sigma_e"] * rng.standard_normal((len(y), M, K, D))
        w = np.linalg.norm(z, axis=-1).argmin(-1)          # (n, M)
        P = np.stack([(w == k).mean(1) for k in range(K)], 1)
        if max_rank is not None:
            rk = np.linalg.norm(y[:, None] + a, axis=-1).argsort(1).argsort(1)
            P = np.where(rk <= max_rank, P, -1.0)
        out.append(P.argmax(1))
    return np.concatenate(out)


def hull_angle_share(P, n_dir=4096):
    """Share of directions u for which each candidate is argmax_k P_k.u: the exterior angle of its Voronoi cell,
    i.e. its limiting Bayes win probability as a shared isotropic error grows. 0 for hull-interior points."""
    th = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    U = np.stack([np.cos(th), np.sin(th)], 1)
    w = (P @ U.T).argmax(1)                                # (n, n_dir)
    return np.stack([(w == k).mean(1) for k in range(P.shape[1])], 1)


def hull_angle_pick(b):
    return hull_angle_share(b["r"], 2048).argmax(1)


# ---- JEPA predictor and heads ----------------------------------------------------------------------------
def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


def resolvent_apply(W, h, gamma):
    I = torch.eye(W.shape[-1], dtype=W.dtype, device=W.device)
    return torch.linalg.solve(I - gamma * W, h)


EPS, TAU = 0.2, 0.05                                      # D-JEPA's bound and softmax temperature


def dist_rank(z, mask):
    n = z.norm(dim=-1).masked_fill(~mask, torch.inf)
    r = n.argsort(1).argsort(1).to(z.dtype)
    return r / (mask.sum(1, keepdim=True) - 1).clamp(min=1).to(z.dtype)


class Head(nn.Module):
    def __init__(self, kind, H=64, din=D):
        super().__init__()
        self.kind = kind
        if kind.startswith("djepa"):                       # "djepa" = eps 0.2; "djepa4" = eps 4 (A6)
            self.eps = float(kind[5:]) if len(kind) > 5 else EPS
            self.enc = nn.Sequential(nn.Linear(din + 1, H), nn.LayerNorm(H), nn.GELU())
            layer = nn.TransformerEncoderLayer(H, 4, 128, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.down, self.up = nn.Linear(H, 8), nn.Linear(8, 1)
            nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
            return
        self.enc, self.out = mlp(din, H, H), mlp(2 * H, H, 1)
        if kind in ("hop1", "resolvent"):
            self.q, self.k = nn.Linear(H, H), nn.Linear(H, H)
            self.theta = nn.Parameter(torch.zeros(()))

    def forward(self, z, mask=None):
        """Scores, higher = better; masked candidates get -inf."""
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        if self.kind.startswith("djepa"):
            b = dist_rank(z, mask)
            h = self.tf(self.enc(torch.cat([z, b[..., None]], -1)), src_key_padding_mask=~mask)
            self.last_delta = self.eps * torch.tanh(self.up(torch.tanh(self.down(h)))).squeeze(-1)
            return (-(b + self.last_delta)).masked_fill(~mask, -torch.inf)
        h = self.enc(z)
        m = h
        if self.kind in ("hop1", "resolvent"):
            Kc = z.shape[1]
            valid = mask[:, None, :] & ~torch.eye(Kc, dtype=torch.bool, device=z.device)
            logit = (self.q(h) @ self.k(h).transpose(1, 2)) / h.shape[-1] ** 0.5
            W = torch.softmax(logit.masked_fill(~valid, -1e9), -1) * valid   # all-masked row -> zero row, no NaN
            g = 0.99 * torch.sigmoid(self.theta)
            m = h + g * W @ h if self.kind == "hop1" else resolvent_apply(W, h, g)
        s = self.out(torch.cat([h, m], -1)).squeeze(-1)
        return s.masked_fill(~mask, -torch.inf)


def latency(K=63, B=16, reps=200, warm=50):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for kind, din in (("djepa", 385), ("resolvent", 386)):
        m = Head(kind, din=din).to(dev).eval()
        x = torch.randn(B, K, din, device=dev)
        ts = []
        with torch.no_grad():
            for i in range(warm + reps):
                if dev == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter(); m(x)
                if dev == "cuda": torch.cuda.synchronize()
                if i >= warm: ts.append(time.perf_counter() - t0)
        out[kind] = float(np.median(ts) * 1e3)                # ms per batch of B starts
    out["device"] = dev
    return out


def fit(model, X, Y, loss_fn, epochs, bs=512, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(X) + bs - 1) // bs))
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=g).split(bs):
            opt.zero_grad(); loss_fn(model(X[idx]), Y[idx]).backward(); opt.step(); sched.step()
    return model


def jepa_predictor(tr, seed):
    X = torch.tensor(np.concatenate([np.repeat(tr["y"][:, None], K, 1), tr["a"]], -1).reshape(-1, 2 * D), dtype=torch.float32)
    Y = torch.tensor(tr["z"].reshape(-1, D), dtype=torch.float32)
    f = fit(mlp(2 * D, 64, D), X, Y, nn.functional.mse_loss, epochs=4, seed=seed)

    def pred(b):
        with torch.no_grad():
            x = torch.tensor(np.concatenate([np.repeat(b["y"][:, None], K, 1), b["a"]], -1), dtype=torch.float32)
            return f(x).numpy()
    return pred


def run_cell(sigma, seeds=(0, 1, 2), n_train=100_000, n_eval=4000, kinds=("point", "hop1", "resolvent", "djepa"), epochs=12):
    res = dict(sigma=sigma, bayes=[], dist=[], dist_ideal=[], hull=[], jepa_mse=[], bayes_rank_ge2=[],
               hit={k: [] for k in kinds}, agree={k: [] for k in kinds}, closure={k: [] for k in kinds})
    for sd in seeds:
        tr, ev = make_bed(n_train, sigma, seed=100 + sd), make_bed(n_eval, sigma, seed=sd)
        pred = jepa_predictor(tr, sd)
        zt, ze = pred(tr), pred(ev)
        bp = bayes_pick(ev)
        hb, hd = hit(bp, ev), hit(dist_pick(ze), ev)
        res["bayes"].append(hb); res["dist"].append(hd); res["dist_ideal"].append(hit(dist_pick(ev["zhat_true"]), ev))
        res["hull"].append(hit(hull_angle_pick(ev), ev))
        res["jepa_mse"].append(float(((ze - ev["zhat_true"]) ** 2).mean()))
        Xt, Yt = torch.tensor(zt), torch.tensor(tr["best"])
        rk = np.linalg.norm(ze, axis=-1).argsort(1).argsort(1)
        res["bayes_rank_ge2"].append(float((rk[np.arange(len(bp)), bp] >= 2).mean()))
        for k in kinds:
            m = Head(k)
            if k.startswith("djepa"):
                loss = lambda o, y, m=m: nn.functional.cross_entropy(o / TAU, y) + 0.1 * (m.last_delta ** 2).mean()
            else:
                loss = nn.functional.cross_entropy
            m = fit(m, Xt, Yt, loss, epochs=epochs, seed=sd)
            with torch.no_grad():
                p = m(torch.tensor(ze)).argmax(1).numpy()
            res["hit"][k].append(hit(p, ev)); res["agree"][k].append(float((p == bp).mean()))
            res["closure"][k].append((hit(p, ev) - hd) / (hb - hd) if hb > hd else float("nan"))
    return res


if __name__ == "__main__":
    t0 = time.time()
    sigmas = [float(x) for x in sys.argv[1:]] or [1.0]
    out = {}
    for sg in sigmas:
        out[str(sg)] = run_cell(sg)
        print(json.dumps(out[str(sg)]), flush=True)
    out["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(out, open("results_foreman.json", "w"), indent=1)
