"""Interventional corpora with oracle-measured outcomes.

Two domains, deliberately given IDENTICAL tensor shapes so that transfer between
them is possible at all:

  state  z in R^6
  action a in R^3      (the intervention)
  target z' in R^6     (measured, never computed analytically by the caller)

`physics`  -- stand-in for the MuJoCo arm of THEORY.md section 2.
`code`     -- Python source is edited and the CPython interpreter is run.
              The outcome is whatever the interpreter printed. No answer key.
"""

from __future__ import annotations

import contextlib
import functools
import io

import numpy as np

STATE_DIM = 6
ACT_DIM = 3

# --------------------------------------------------------------------------
# domain 1: executed Python source
# --------------------------------------------------------------------------

TEMPLATES = [
    # t=0 : y = -a + 30*u, built by loops so a closed form is not read off
    "def run(a, u):\n"
    "    acc = 0\n"
    "    for _ in range(a):\n"
    "        acc -= 1\n"
    "    for _ in range(u):\n"
    "        acc += 30\n"
    "    return acc\n"
    "print(run({a}, {u}))\n",
    # t=1 : y = -2a + 30*u
    "def run(a, u):\n"
    "    acc = 0\n"
    "    for _ in range(a):\n"
    "        acc -= 2\n"
    "    for _ in range(u):\n"
    "        acc += 30\n"
    "    return acc\n"
    "print(run({a}, {u}))\n",
    # t=2 : y = -a + 15*u + 5
    "def run(a, u):\n"
    "    acc = 5\n"
    "    for _ in range(a):\n"
    "        acc -= 1\n"
    "    for _ in range(u):\n"
    "        acc += 15\n"
    "    return acc\n"
    "print(run({a}, {u}))\n",
]


def render(a: int, u: int, t: int) -> str:
    return TEMPLATES[int(t) % len(TEMPLATES)].format(a=int(a), u=int(u))


@functools.lru_cache(maxsize=200_000)
def run_program(src: str) -> float:
    """Execute `src` and return the single number it printed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "<corpus>", "exec"), {"__builtins__": __builtins__}, {})
    return float(buf.getvalue().strip())


def code_state(a: int, u: int, t: int) -> np.ndarray:
    y = run_program(render(a, u, t))
    z = np.zeros(STATE_DIM)
    z[0] = a / 10.0
    z[1] = u
    z[2] = y / 30.0
    z[3 + (int(t) % 3)] = 1.0
    return z


def _clip_prog(a, u, t):
    return int(np.clip(a, 0, 40)), int(np.clip(u, 0, 1)), int(t % 3)


def code_corpus(n: int, mode: str, seed: int, edit_range=(-3, 3)) -> dict:
    """mode='observational' : the literal `a` is chosen by the hidden style `u`.
    mode='interventional'   : `a` is set by an edit, independent of `u`.
    """
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, n)
    t = rng.integers(0, 3, n)
    if mode == "observational":
        # hidden style u drives BOTH the literal and the outcome -> confounded
        a = 2 + 18 * u + rng.integers(-2, 3, n)
    elif mode == "interventional":
        a = rng.integers(0, 23, n)
    else:
        raise ValueError(mode)

    d_a = rng.integers(edit_range[0], edit_range[1] + 1, n)
    d_u = rng.integers(-1, 2, n)
    d_t = rng.integers(0, 3, n)

    z = np.zeros((n, STATE_DIM))
    z_next = np.zeros((n, STATE_DIM))
    act = np.zeros((n, ACT_DIM))
    y = np.zeros(n)
    for i in range(n):
        ai, ui, ti = _clip_prog(a[i], u[i], t[i])
        z[i] = code_state(ai, ui, ti)
        y[i] = z[i, 2] * 30.0
        bi, vi, si = _clip_prog(ai + d_a[i], ui + d_u[i], ti + d_t[i])
        z_next[i] = code_state(bi, vi, si)
        act[i] = [(bi - ai) / 10.0, vi - ui, (si - ti) / 2.0]
    return dict(z=z, a=act, z_next=z_next, y=y,
                a_lit=a.astype(float), u_lit=u.astype(float), t_lit=t.astype(float))


# --------------------------------------------------------------------------
# domain 2: physics stand-in
# --------------------------------------------------------------------------


def _physics_operators(seed: int = 99):
    rng = np.random.default_rng(seed)
    A0 = rng.normal(size=(STATE_DIM, STATE_DIM)) / np.sqrt(STATE_DIM)
    A0 *= 0.9 / max(abs(np.linalg.eigvals(A0)))
    Bk = [0.15 * rng.normal(size=(STATE_DIM, STATE_DIM)) / np.sqrt(STATE_DIM)
          for _ in range(ACT_DIM)]
    C = rng.normal(size=(STATE_DIM, ACT_DIM)) * 0.5
    c = rng.normal(size=STATE_DIM) * 0.1
    return A0, Bk, C, c


def shuffle_actions(a: np.ndarray, seed: int) -> np.ndarray:
    """Break the action->outcome relation while keeping the action marginal.

    A prior fitted on this has the same shape and scale as a real one and no
    valid dynamics: the null model for any transfer claim.
    """
    rng = np.random.default_rng(seed)
    return a[rng.permutation(len(a))]


def physics_corpus(n: int, seed: int, act_scale: float = 1.0) -> dict:
    A0, Bk, C, c = _physics_operators()
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, STATE_DIM))
    a = rng.uniform(-act_scale, act_scale, size=(n, ACT_DIM))
    z_next = z @ A0.T + a @ C.T + c
    for k in range(ACT_DIM):
        z_next += a[:, [k]] * (z @ Bk[k].T)
    return dict(z=z, a=a, z_next=z_next, ops=(A0, Bk, C, c))


# --------------------------------------------------------------------------
# the sigmoid-style bilinear action lift and ridge fit
# --------------------------------------------------------------------------


def lift(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """phi(z,a) = [z ; a (x) z ; a ; 1]"""
    z = np.atleast_2d(z)
    a = np.atleast_2d(a)
    outer = (a[:, :, None] * z[:, None, :]).reshape(len(z), -1)
    return np.concatenate([z, outer, a, np.ones((len(z), 1))], axis=1)


PHI_DIM = STATE_DIM + ACT_DIM * STATE_DIM + ACT_DIM + 1


def fit_bilinear(z, a, z_next, lam: float = 1e-3, prior_W=None, prior_w: float = 0.0):
    """Ridge fit of W.  With prior_W, penalise ||W - prior_W||^2 at weight
    prior_w instead of shrinking to zero (transfer from a prior fit)."""
    P = lift(z, a)
    G = P.T @ P + (lam + prior_w) * np.eye(P.shape[1])
    rhs = P.T @ z_next
    if prior_W is not None:
        rhs = rhs + prior_w * prior_W
    return np.linalg.solve(G, rhs)


def predict(W, z, a):
    return lift(z, a) @ W


def nrmse(pred, truth) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((pred - truth) ** 2))
                 / (np.sqrt(np.mean(truth ** 2)) + 1e-12))


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    return float((xc @ (y - y.mean())) / (xc @ xc))
