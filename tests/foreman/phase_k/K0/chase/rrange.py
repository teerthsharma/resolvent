"""R-RANGE instrument (rebuild, Chase K0). Mass of a head = decay rate of its resolvent kernel:
m = -slope of r -> mean_i ln|G(i, i-r)| over the chosen rows i, G = (I - gW)^-1, rows by transposed triangular solves.
Validated on the contract's Y2 Toeplitz head and on Y1 (independent routes from the pinned yukawa.py:
banded solve for Y1, multi-row log-mean for Y2)."""
import numpy as np
from scipy.linalg import solve_banded, solve_triangular


def fit_decay(prof, r):
    return float(-np.polyfit(r, np.log(prof), 1)[0])


def y1_decay(m, N=4001, r=np.arange(20, 60)):
    ab = np.zeros((3, N))
    ab[0, 1:], ab[1, :], ab[2, :-1] = -1.0, 2.0 + m * m, -1.0
    e = np.zeros(N); e[N // 2] = 1.0
    G = solve_banded((1, 1), ab, e)
    return fit_decay(G[N // 2 + r], r)


def resolvent_rows(W, g, rows):
    """rows of G = (I - gW)^-1 for lower-triangular W: solve (I - gW)^T Y = E_rows (upper)."""
    n = W.shape[0]
    E = np.zeros((n, len(rows))); E[rows, np.arange(len(rows))] = 1.0
    return solve_triangular((np.eye(n) - g * W).T, E, lower=False).T      # [len(rows), n]


def head_mass(W, g, rows, r):
    Gr = resolvent_rows(W, g, rows)
    rows = np.asarray(rows)
    prof = np.exp(np.mean(np.log(np.abs(Gr[np.arange(len(rows))[:, None], rows[:, None] - r[None, :]])), 0))
    return fit_decay(prof, r)


def toeplitz_head(lam, n):
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    K = np.where(j <= i, np.exp(-lam * np.clip(i - j, 0, None)), 0.0)
    return K / K.sum(1, keepdims=True)


def toeplitz_mass(lam, g, n=3000, rows=np.arange(2000, 2901, 100), r=np.arange(30, 400)):
    return head_mass(toeplitz_head(lam, n), g, rows, r)


def head_mass_tail(W, g, rows, r):
    """Replacement instrument (after head_mass returned m < 0 on content-bearing heads, breaks.py B2):
    tail mass T(r) = mean_i sum_{j <= i-r} (1-g) G(i, j) (all terms >= 0, summed upward from j = 0: no cancellation),
    m = -slope of ln T(r). T is non-increasing in r by construction, so m >= 0; on a Toeplitz head T is exactly geometric."""
    Gr = (1 - g) * resolvent_rows(W, g, rows)
    rows = np.asarray(rows)
    T = np.stack([np.cumsum(Gr[t, :rows[t] + 1])[rows[t] - r] for t in range(len(rows))])   # sum_{j <= i-r}
    return fit_decay(T.mean(0), r)


def toeplitz_mass_tail(lam, g, n=3000, rows=np.arange(2000, 2901, 100), r=np.arange(30, 400)):
    return head_mass_tail(toeplitz_head(lam, n), g, rows, r)
