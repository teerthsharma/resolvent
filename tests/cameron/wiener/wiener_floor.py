"""
L-WIENER row. Wiener (1942) optimal LINEAR one-step-ahead predictor,
reproduced against the fixed instance, then packaged as a reusable
floor-row function for any time-indexed bed.

CPU only. No git, no results/*.pt, no GPU.
"""
import numpy as np


# ---------------------------------------------------------------------------
# (1) Reproduce the fixed AR(2)+noise instance exactly as specified.
# ---------------------------------------------------------------------------

def generate_instance(seed=0, N=200_000, a1=1.2, a2=-0.5, proc_sd=0.5, obs_sd=0.3):
    """AR(2) hidden state x_t = a1*x_{t-1} + a2*x_{t-2} + eps_t (eps ~ N(0,proc_sd^2)),
    observed y_t = x_t + n_t (n ~ N(0, obs_sd^2)). Returns (x, y)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, proc_sd, N)
    x = np.zeros(N)
    for t in range(2, N):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + eps[t]
    y = x + rng.normal(0.0, obs_sd, N)
    return x, y


def autocov(series, max_lag, demean=True):
    """Empirical autocovariance gamma[0..max_lag] via FFT (biased/N estimator)."""
    s = series - series.mean() if demean else series
    n = len(s)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(s, nfft)
    acf_full = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    return acf_full / n


def wiener_weights(y, x_target, p):
    """Solve the Wiener-Hopf normal equations for the order-p linear MMSE
    predictor of x_target[t] from y[t-1..t-p] (causal, one-step-ahead,
    never uses y[t]). Uses empirical autocovariance of y and empirical
    cross-covariance of x_target with lagged y -> works with no ground-truth
    model, so the same code path serves any bed.
    """
    n = len(y)
    gy = autocov(y, p)  # gy[0..p]
    R = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            R[i, j] = gy[abs(i - j)]
    # cross-covariance r[i] = Cov(x_target[t], y[t-1-i]), i=0..p-1
    ym = y - y.mean()
    xm = x_target - x_target.mean()
    r = np.empty(p)
    for i in range(p):
        lag = i + 1
        r[i] = np.mean(xm[lag:] * ym[: n - lag])
    w = np.linalg.solve(R + 1e-12 * np.eye(p), r)
    return w, xm.mean() if False else x_target.mean(), y.mean()


def wiener_predict(y, x_target, p):
    """Fit order-p Wiener weights on the full series and score one-step-ahead
    MSE in-sample (N=200,000 makes train/held-out estimation-noise negligible
    at this order; this is a floor-row, not a generalization claim)."""
    n = len(y)
    w, xmean, ymean = wiener_weights(y, x_target, p)
    pred = np.full(n, xmean)
    for i in range(p):
        lag = i + 1
        pred[lag:] += w[i] * (y[: n - lag] - ymean)
    valid = slice(p, n)
    mse = np.mean((x_target[valid] - pred[valid]) ** 2)
    return mse, w


def last_value_mse(y, x_target):
    """Naive baseline: predict x_t with the last observation y_{t-1}."""
    pred = y[:-1]
    target = x_target[1:]
    return float(np.mean((target - pred) ** 2))


def oracle_mse(x, a1=1.2, a2=-0.5):
    """Best-possible one-step predictor: true AR coefficients on the true
    (noise-free) past state. Irreducible error = process-noise variance."""
    pred = a1 * x[1:-1] + a2 * x[:-2]
    target = x[2:]
    return float(np.mean((target - pred) ** 2))


def wiener_row(series, target=None, p_values=(8, 32)):
    """L-WIENER reusable row.

    series: the observed time-indexed sequence (regressors).
    target: what is being predicted at time t (defaults to series itself,
        i.e. plain self-prediction -- the only option available on a bed
        with no separate hidden ground-truth channel, e.g. chess/MDP betas).
    Returns dict: last_value, wiener_p8, wiener_p32 (MSE), base_rate.
    """
    series = np.asarray(series, dtype=float)
    tgt = series if target is None else np.asarray(target, dtype=float)
    row = {
        "n": len(series),
        "last_value_mse": last_value_mse(series, tgt),
        "base_rate": float(np.var(tgt)),  # predicting the unconditional mean
    }
    for p in p_values:
        mse, _ = wiener_predict(series, tgt, p)
        row[f"wiener_p{p}_mse"] = mse
    return row


if __name__ == "__main__":
    TARGET = dict(last_value=0.4594, wiener=0.3498, oracle=0.2502)

    x, y = generate_instance(seed=0, N=200_000)

    lv = last_value_mse(y, x)
    orc = oracle_mse(x)
    w8_mse, w8 = wiener_predict(y, x, 8)
    w32_mse, w32 = wiener_predict(y, x, 32)

    print("=== W1 - Wiener 1942, AR(2) instance, default_rng(0) ===")
    print(f"last-value MSE : {lv:.4f}   (target {TARGET['last_value']})")
    print(f"WIENER p=8 MSE : {w8_mse:.4f}   (target {TARGET['wiener']})")
    print(f"oracle MSE     : {orc:.4f}   (target {TARGET['oracle']})")
    print(f"WIENER p=32 MSE: {w32_mse:.4f}")
    print(f"gap to oracle p=8 : {w8_mse - orc:.4f}")
    print(f"gap to oracle p=32: {w32_mse - orc:.4f}")
    closes = (w32_mse - orc) < (w8_mse - orc)
    print(f"gap closes p8->p32: {closes}  "
          f"({w8_mse - orc:.4f} -> {w32_mse - orc:.4f})")

    for name, got, tgt in [("last-value", lv, TARGET["last_value"]),
                            ("wiener_p8", w8_mse, TARGET["wiener"]),
                            ("oracle", orc, TARGET["oracle"])]:
        rel = abs(got - tgt) / tgt
        status = "MATCH" if rel < 0.01 else ("CLOSE" if rel < 0.05 else "MISMATCH")
        print(f"  {name}: got {got:.4f} vs spec {tgt} -> {status} ({rel*100:.2f}% rel diff)")

    # self-check
    assert w8_mse < lv, "Wiener(8) must beat last-value (it strictly nests it)"
    assert orc < w8_mse, "oracle must beat any observation-only linear predictor"
    print("\nself-check OK: oracle < wiener_p8 < last-value")
