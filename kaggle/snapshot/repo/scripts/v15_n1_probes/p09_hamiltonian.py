"""[V-eq] probe 9 -- Hamiltonian neural nets / the conservation census.

Greydanus, Dzamba & Yosinski 2019, "Hamiltonian Neural Networks", arXiv:1906.01563:
  Eq (2)  dq/dt = dH/dp,  dp/dt = -dH/dq ;  S_H = (dH/dp, -dH/dq)
  Eq (3)  L_HNN = || dH_theta/dp - dq/dt ||_2 + || dH_theta/dq + dp/dt ||_2
  HYPOTHESES: canonical coordinates (q, p); H time-independent; gradients of the
  network taken by automatic differentiation (H is the network output, not the field).

Instance: H = (p^2 + q^2)/2 with a one-parameter model H_theta = theta (p^2+q^2)/2,
so dH_theta/dp = theta p and dH_theta/dq = theta q.  L_HNN vanishes iff theta = 1.
Then: the symplectic field conserves H; a field that is not a symplectic gradient
does not -- this is the conservation-census number.
"""
import numpy as np

rng = np.random.default_rng(8)
q, p = rng.normal(size=20), rng.normal(size=20)
dqdt, dpdt = p, -q                       # the true field of H = (p^2 + q^2)/2


def L_hnn(th):
    return (np.linalg.norm(th * p - dqdt) + np.linalg.norm(th * q + dpdt))


print("L_HNN(theta=1)    = %.3e" % L_hnn(1.0))
print("L_HNN(theta=1.01) = %.6f" % L_hnn(1.01))
print("L_HNN(theta=0.99) = %.6f" % L_hnn(0.99))

# sign control: the WRONG sign convention (dp/dt = +dH/dq) does not vanish anywhere
th = np.linspace(-2, 2, 40001)
wrong = np.array([np.linalg.norm(t * p - dqdt) + np.linalg.norm(t * q - dpdt) for t in th])
print("min over theta of the sign-flipped loss = %.6f at theta=%.3f (never 0)"
      % (wrong.min(), th[wrong.argmin()]))


def rk4(f, y, h, n):
    out = np.empty((n, 2))
    for i in range(n):
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = y
    return out


H0 = 0.5
y0 = np.array([1.0, 0.0])
symp = lambda y: np.array([y[1], -y[0]])            # ( dH/dp, -dH/dq )
nonc = lambda y: np.array([y[1] + 0.01 * y[0], -y[0]])
for f, tag in [(symp, "symplectic S_H"), (nonc, "unconstrained ")]:
    traj = rk4(f, y0, 1e-3, 200_000)
    Hs = 0.5 * (traj[:, 0] ** 2 + traj[:, 1] ** 2)
    print("%s : max |H - H_0| over 2e5 steps = %.3e" % (tag, np.abs(Hs - H0).max()))
