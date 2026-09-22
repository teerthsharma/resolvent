"""Minimal float64 reimplementations, each from its published equation (see SOURCES.txt).
No third-party code is copied or executed; line comments name the equation reimplemented."""
import numpy as np


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softplus(z):
    return np.logaddexp(0.0, z)


# ours: design4x5.softmax_gated_forward -- o * sigmoid(o_gate_head(x)) per head, before o_proj
def a2_gate(Y, X, W, b):
    return Y * _sig(X @ W + b).transpose(0, 2, 1)[..., None]


# Qwen arXiv 2505.06708: Y' = Y (.) sigma(X W_theta), headwise = one scalar per head
def qwen_g1_headwise(Y, X, W):
    return Y * _sig(X @ W).transpose(0, 2, 1)[..., None]


# Qwen3-Next (modeling_qwen3_next.py L433-450): gate = second half of q_proj per head, head_dim wide
def qwen_g1_elementwise(Y, X, Wel):
    return Y * _sig(np.einsum("bsd,dhe->bhse", X, Wel))


# FoX arXiv 2503.02130 as built in arms_j.fox_forward: log f = log sigmoid(z) = -softplus(-z)
def fox_logf(z):
    return -_softplus(-z)


# Qwen3-Next GatedDeltaNet L858: g = -exp(A_log) * softplus(a + dt_bias)
def gdn_logdecay(a, A_log, dt_bias):
    return -np.exp(A_log) * _softplus(a + dt_bias)


# arms_j: D_ij = C_i - C_j, C = cumsum(log f), causal; returned as exp(D) (0 above diagonal)
def decay_mask_from_logf(logf):
    c = np.cumsum(logf)
    T = len(logf)
    D = c[:, None] - c[None, :]
    return np.where(np.tril(np.ones((T, T), bool)), np.exp(np.minimum(D, 0.0)), 0.0)  # D<=0 on/below diagonal


# Qwen3-Next torch_recurrent_gated_delta_rule loop body (L1038-1051), one head
def gdn_recurrent(q, k, v, logdecay, beta):
    S = np.zeros((k.shape[1], v.shape[1]))
    out = np.zeros((q.shape[0], v.shape[1]))
    for t in range(q.shape[0]):
        S = S * np.exp(logdecay[t])
        kv_mem = (S * k[t][:, None]).sum(0)
        delta = (v[t] - kv_mem) * beta[t]
        S = S + k[t][:, None] * delta[None, :]
        out[t] = (S * q[t][:, None]).sum(0)
    return out


def linear_attn_masked(q, k, v, Mk):
    return (Mk * (q @ k.T)) @ v


def softmax_attn_masked(q, k, v, Mk):
    logits = np.where(Mk > 0, q @ k.T + np.log(np.where(Mk > 0, Mk, 1.0)), -np.inf)
    logits -= logits.max(-1, keepdims=True)
    P = np.exp(logits)
    return (P / P.sum(-1, keepdims=True)) @ v


# Kimi Linear arXiv 2510.26692 KDA: S_t = (I - b k k^T) Diag(alpha) S_{t-1} + b k v^T
def kda_transition(alpha, beta, k):
    return (np.eye(len(k)) - beta * np.outer(k, k)) @ np.diag(alpha)


# Gated DeltaNet as shipped in Qwen3-Next: S <- alpha S, then delta rule => alpha (I - b k k^T)
def gdn_transition(alpha, beta, k):
    return alpha * (np.eye(len(k)) - beta * np.outer(k, k))


def householder_general(beta, k):
    k = k / np.linalg.norm(k)
    return np.eye(len(k)) - beta * np.outer(k, k)


# ours (Addendum L): unit quaternion -> SU(2)
def su2_from_quat(qv):
    a, b, c, d = qv / np.linalg.norm(qv)
    return np.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]])


# MiniMax lightning: kv_t = lambda kv_{t-1} + k_t v_t^T  =>  M_ts = lambda^(t-s)
def lightning_mask(lam, T):
    t = np.arange(T)
    return np.where(t[:, None] >= t[None, :], lam ** (t[:, None] - t[None, :]).clip(0), 0.0)


# ALiBi bias -m (t-s), exponentiated
def alibi_mask(m, T):
    t = np.arange(T)
    return np.where(t[:, None] >= t[None, :], np.exp(-m * (t[:, None] - t[None, :]).clip(0)), 0.0)


# NSA arXiv 2502.11089: o*_t = sum_c g_t^c Attn(q_t, K~^c, V~^c)
def nsa_combine(gs, outs):
    return sum(g * o for g, o in zip(gs, outs))
