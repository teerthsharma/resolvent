"""[V-eq] probe 5 -- MWU / Hedge and the replicator dynamics.
Arora, Hazan, Kale 2012, Theory of Computing 8:121-164, Fig.1 Eq (2.1)
  w_i(t+1) = w_i(t)(1 - eta m_i(t)); p(t) = w(t)/Phi(t), Phi(t) = sum_i w_i(t)
  Thm 2.1 (hypotheses: m_i(t) in [-1,1], eta <= 1/2):
     sum_t m(t).p(t) <= sum_t m_i(t) + eta sum_t |m_i(t)| + ln n / eta
  Eq (2.6) Hedge: w_i(t+1) = w_i(t) exp(-eta m_i(t)); Thm 2.3 (eta <= 1).
Falniowski & Mertikopoulos 2024, arXiv:2402.09824, Eq (EW)/(III):
  y_i(t+d) = y_i(t) + d u_i(x(t)),  x_i = exp(y_i)/sum_j exp(y_j)
  <=> x_i(t+d) = x_i(t) exp(d u_i) / sum_j x_j(t) exp(d u_j)
  Euler-discretises (RD): xdot_i = x_i [u_i(x) - u(x)] with O(d^2) error.
"""
import numpy as np
rng = np.random.default_rng(4)
n, T, eta = 5, 300, 0.4
m = rng.uniform(-1, 1, size=(T, n))
w = np.ones(n); cost = 0.0
for t in range(T):
    p = w / w.sum()
    cost += m[t] @ p
    w = w * (1 - eta * m[t])
best = m.sum(0).argmin()
rhs = m[:, best].sum() + eta * np.abs(m[:, best]).sum() + np.log(n) / eta
print("MWU Thm 2.1  LHS=%.6f  RHS=%.6f  holds=%s" % (cost, rhs, cost <= rhs))

# (EW) == softmax of cumulative payoff, to machine precision  [contract's Lean #8]
u = rng.normal(size=n)                     # frequency-independent payoffs
d = 0.05
y = np.zeros(n); x = np.ones(n) / n
for _ in range(200):
    y = y + d * u
    x = x * np.exp(d * u); x /= x.sum()
sm = np.exp(y - y.max()); sm /= sm.sum()
print("EW iterate vs softmax(cumulative payoff):", np.abs(x - sm).max())

# replicator: Fisher -- d(mean payoff)/dt = Var(payoff)  (hypothesis: u constant)
def rd_step(x, u, dt):
    return x + dt * x * (u - x @ u)
x = rng.dirichlet(np.ones(n)); dt = 1e-6
ubar0 = x @ u; x1 = rd_step(x, u, dt); ubar1 = x1 @ u
var = x @ (u ** 2) - (x @ u) ** 2
print("Fisher: dubar/dt=%.10f  Var(f)=%.10f  rel.err=%.2e"
      % ((ubar1 - ubar0) / dt, var, abs((ubar1 - ubar0) / dt - var) / var))

# EW is an Euler discretisation of RD with O(delta^2) error -> ratio ~ 4 on halving
def ew_step(x, u, dd):
    z = x * np.exp(dd * u); return z / z.sum()
def err(dd):
    x0 = rng2.dirichlet(np.ones(n))
    return np.abs(ew_step(x0, u, dd) - (x0 + dd * x0 * (u - x0 @ u))).max()
rng2 = np.random.default_rng(7); e1 = err(0.02)
rng2 = np.random.default_rng(7); e2 = err(0.01)
print("EW-vs-RD one-step error ratio delta/(delta/2) = %.3f (expect ~4)" % (e1 / e2))
