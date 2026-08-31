"""BED-1 -- a multi-basin bed whose label is the SPLITTING PROBABILITY.

Third of the three beds CEQ_V15_CONTRACT.md PART III names, and the one the
C-TS bar (transition-state / exit accuracy) is measured on. BED-M's label is a
scan state, BED-K's is a memory kernel; BED-1's is the committor `q` -- the
probability that a trajectory started at a configuration reaches basin B before
basin A. Guards are the `q = 1/2` isocommittor surfaces and the itinerary of
guard crossings is the action vocabulary.

WHY NOT THE BARRIER HEIGHT. The contract's pre-v13 strike: with `m` parallel
saddles at the higher barrier the higher barrier becomes the FASTER channel
above

    T* = DeltaDeltaE-dagger / ln m

because the `m`-fold degeneracy multiplies the channel's rate while the
Boltzmann factor only exponentiates the barrier. At `DeltaDeltaE = 1.0` and
`m = 5`, `T* = 0.6213349345596119`; the landscape below reproduces the round's
re-derived readings exactly through its committor solve -- at `0.9 T*` the low
channel carries `1.6725e-01` against the high bundle's `1.3986e-01`, and at
`1.1 T*` it is `2.3151e-01` against `2.6799e-01`, the order reversed. A bed
labelled by barrier height therefore gives the WRONG answer above `T*` at every
temperature, and the two labelling schemes are made to disagree inside
tests/beds/test_bed_1.py rather than merely being described as capable of it.

THE LANDSCAPE. An 11-node energy graph, three basins and eight saddles:

      A(0.0) --S_lo(1.0)--------------------------- B(0.0)      the low channel
      A      --S_hi0..4(2.0)----------------------- B           m = 5 parallel
      A      --S_ac(1.5)-- C(0.9) --S_cb(1.5)------ B           via the intermediate

`C` is a metastable intermediate that sits exactly ON the `q = 1/2` surface --
a genuine transition-state basin, not a saddle -- which is what gives the CK
battery a lumping with real memory to detect.

RATES AND ARITHMETIC. Metropolis rates `k_ij = exp(-(V_j - V_i)_+ / T)` on each
edge, reversible with respect to `w_i = exp(-V_i / T)`. Because every downhill
rate is exactly 1, a barrier top flanked by two deeper basins has
`q = (q_left + q_right)/2` for ANY basin depths and ANY temperature: the
committor of the default landscape is the closed form `(0, 1/4, 1/2, 3/4, 1)`,
and the numerical solve is checked against it rather than only against its own
residual.

MISTAKES.md M-19 (a dynamical invariant estimated on a collapsed float64
orbit). BED-1 iterates NO real-valued map. Its trajectory is an integer state
sequence produced by comparing PRNG uniforms against a fixed cumulative row of
`P`, so no bit of an initial condition is lost per step and there is no
`mantissa_bits / log2(stretching rate)` horizon to exceed. `entropy_rate_chain`
is a CLOSED FORM in `P` and `pi` -- it is not estimated from an orbit at all --
and the symbolic entropies are block-frequency counts on that integer sequence.
The exact-rational requirement M-19 imposes on tent-map orbits is inapplicable
here, and this paragraph is the statement of arithmetic M-19's check demands
rather than its omission. Any future BED-1 variant driven by a chaotic map
inherits M-19 in full.

MISTAKES.md M-19, second constraint (Bollt et al. 2001: the Pesin deficit is
NON-MONOTONE in partition misplacement). `guards` is the `q = 1/2` level set,
computed from the committor before any trajectory exists. NO function in this
module ranks, searches, or argmins over the deficit, and none takes the deficit
as an input to guard selection. `pesin_deficit` is offered only as the
near-zero admissibility test the source supports.

SEEDING DISCIPLINE, matching bed_k.py: one `seed` argument, one RNG per call,
nothing touches global state. `build()` returns a manifest dict rather than an
object, for the same reason bed_k.build() does -- every field is meant to be
read by a caller that never imports this module's internals.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "E_LO", "E_HI", "M_PARALLEL", "crossover_temperature", "arrhenius_channel_rate",
    "build", "committor", "harmonic_residual", "reactive_flux", "channel_flux",
    "label_by_barrier", "label_by_committor", "simulate", "macrostate_of_node",
    "transition_matrix", "ck_error", "entropy_rate_chain", "block_entropy_rate",
    "pesin_deficit", "morse_census", "guard_itinerary", "conservation_census",
]

E_LO = 1.0            # the single low saddle
E_HI = 2.0            # the m parallel high saddles
E_MID = 1.5           # the two saddles flanking the intermediate basin
V_C = 0.9             # the intermediate basin itself
M_PARALLEL = 5

_NAMES = ["A", "B", "C", "S_lo"] + [f"S_hi{k}" for k in range(M_PARALLEL)] + ["S_ac", "S_cb"]
_V0 = np.array([0.0, 0.0, V_C, E_LO] + [E_HI] * M_PARALLEL + [E_MID, E_MID], dtype=np.float64)
_EDGES = ([("A", "S_lo"), ("S_lo", "B")]
          + [e for k in range(M_PARALLEL) for e in (("A", f"S_hi{k}"), (f"S_hi{k}", "B"))]
          + [("A", "S_ac"), ("S_ac", "C"), ("C", "S_cb"), ("S_cb", "B")])
_CHANNELS = {"lo": ["S_lo"], "hi": [f"S_hi{k}" for k in range(M_PARALLEL)],
             "trap": ["S_ac", "C", "S_cb"]}
_BARRIERS = {"lo": E_LO, "hi": E_HI, "trap": E_MID}


# --------------------------------------------------------------------------
# the strike, in closed form
# --------------------------------------------------------------------------


def crossover_temperature(delta_delta_barrier: float, m: int) -> float:
    """`T* = DeltaDeltaE-dagger / ln m`, above which the `m`-fold degenerate
    higher barrier is the faster channel. Undefined at `m = 1` (no degeneracy,
    no crossover)."""
    if m < 2:
        raise ValueError(f"a crossover needs m >= 2 parallel saddles, got {m}")
    return float(delta_delta_barrier / np.log(m))


def arrhenius_channel_rate(barrier: float, T: float, multiplicity: int = 1) -> float:
    """`multiplicity * exp(-barrier / T)` -- the bare escape rate over a channel
    of `multiplicity` equivalent saddles. This is the quantity the contract
    quotes; `channel_flux` reaches the same number through the committor and the
    two are compared in the test suite."""
    return float(multiplicity * np.exp(-barrier / T))


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


def build(T: float, seed: int = 0, jitter: float = 0.0,
          V_override: dict | None = None) -> dict:
    """The manifest. `T` sets the rates; `seed` and `jitter` seed a Gaussian
    perturbation of the node energies (`jitter = 0`, the default, gives the
    exact symmetric landscape and ignores the seed entirely -- so a caller can
    tell "the seed did nothing" from "the seed was ignored").

    `V_override` replaces named node energies after jittering. It exists for the
    Morse must-fire, which needs a landscape whose critical-point counts differ
    while the Euler characteristic cannot.

    Every field a caller needs sits at the top level, matching bed_k.build().
    """
    rng = np.random.default_rng(seed)
    V = _V0.copy()
    if jitter > 0.0:
        V = V + rng.normal(0.0, jitter, size=len(_V0))
    ix = {name: i for i, name in enumerate(_NAMES)}
    if V_override:
        for name, val in V_override.items():
            V[ix[name]] = float(val)
    elif jitter > 0.0:
        # Trust boundary: a jitter large enough to reorder an edge's endpoints
        # turns a saddle into a basin and silently changes what the bed is. The
        # committor would still solve; the landscape would no longer be the one
        # documented. Refuse rather than emit it.
        for u, v in _EDGES:
            if np.sign(V[ix[u]] - V[ix[v]]) != np.sign(_V0[ix[u]] - _V0[ix[v]]):
                raise ValueError(
                    f"jitter={jitter} reordered edge {u}-{v} "
                    f"({_V0[ix[u]]}->{_V0[ix[v]]} became {V[ix[u]]}->{V[ix[v]]}); "
                    "the perturbed landscape is not a perturbation of this bed"
                )

    n = len(_NAMES)
    K = np.zeros((n, n), dtype=np.float64)
    for u, v in _EDGES:
        i, j = ix[u], ix[v]
        K[i, j] = np.exp(-max(V[j] - V[i], 0.0) / T)
        K[j, i] = np.exp(-max(V[i] - V[j], 0.0) / T)
    L = K.copy()
    np.fill_diagonal(L, -K.sum(axis=1))

    A, B = [ix["A"]], [ix["B"]]
    interior = [i for i in range(n) if i not in A + B]
    q = committor(L, A, B)

    w = np.exp(-V / T)                 # unnormalized Boltzmann weight
    pi = w / w.sum()
    d = float(K.sum(axis=1).max())     # uniformization constant
    P = K / d
    np.fill_diagonal(P, 1.0 - K.sum(axis=1) / d)

    guards = [i for i in range(n) if abs(q[i] - 0.5) <= 1e-12]
    return dict(kind="bed_1", T=float(T), seed=seed, jitter=float(jitter), n=n,
                names=list(_NAMES), node_index=ix, V=V, edges=list(_EDGES),
                A=A, B=B, interior=interior, K=K, L=L, w=w, pi=pi, P=P, d=d,
                q=q, guards=guards,
                channels={c: [ix[nm] for nm in v] for c, v in _CHANNELS.items()},
                barriers=dict(_BARRIERS),
                params=dict(E_lo=E_LO, E_hi=E_HI, m=M_PARALLEL,
                            delta_delta=E_HI - E_LO,
                            T_star=crossover_temperature(E_HI - E_LO, M_PARALLEL)))


def committor(L: np.ndarray, A: list[int], B: list[int]) -> np.ndarray:
    """Solve the backward Kolmogorov / discrete harmonic problem: `(Lq)_i = 0`
    on the interior, `q = 0` on A, `q = 1` on B. float64 dense solve; the
    interior here is 9x9."""
    n = L.shape[0]
    interior = [i for i in range(n) if i not in list(A) + list(B)]
    rhs = -L[np.ix_(interior, list(B))].sum(axis=1)
    q = np.zeros(n, dtype=np.float64)
    q[interior] = np.linalg.solve(L[np.ix_(interior, interior)], rhs)
    q[list(B)] = 1.0
    return q


def harmonic_residual(bed: dict, q: np.ndarray | None = None) -> float:
    """`max_i |(Lq)_i|` over the interior -- contract item #14's numerical
    content. Zero to machine precision iff `q` is the committor."""
    q = bed["q"] if q is None else q
    return float(np.abs(bed["L"] @ q)[bed["interior"]].max())


# --------------------------------------------------------------------------
# transition-path theory: reactive flux and the channel labels
# --------------------------------------------------------------------------


def reactive_flux(bed: dict) -> np.ndarray:
    """One-way TPT reactive flux `J_ij = w_i k_ij (q_j - q_i)_+`. Unnormalized
    weights, so `2 * J` on a two-edge channel equals that channel's Arrhenius
    rate -- the convention the contract's quoted numbers are in."""
    q = bed["q"]
    return bed["w"][:, None] * bed["K"] * np.maximum(q[None, :] - q[:, None], 0.0)


def channel_flux(bed: dict) -> dict:
    """Reactive flux leaving basin A into each channel."""
    J = reactive_flux(bed)
    a = bed["A"][0]
    return {c: float(J[a, nodes].sum()) for c, nodes in bed["channels"].items()}


def label_by_barrier(bed: dict) -> str:
    """The WRONG labelling, kept so it can be shown to be wrong: the channel
    with the smallest barrier. Independent of temperature by construction, which
    is the whole defect."""
    return min(bed["barriers"], key=bed["barriers"].get)


def label_by_committor(bed: dict) -> str:
    """The bed's labelling: the channel carrying the largest reactive flux, i.e.
    the one the splitting probability actually routes through."""
    f = channel_flux(bed)
    return max(f, key=f.get)


# --------------------------------------------------------------------------
# dynamics -- integer state space, no iterated real-valued map (M-19)
# --------------------------------------------------------------------------


def simulate(bed: dict, n_walkers: int, n_steps: int, seed: int) -> np.ndarray:
    """`n_walkers` independent walkers started from `pi`, stepped together.
    Returns an `(n_steps, n_walkers)` int8 array of node indices.

    The state is an integer throughout; the only float64 involved is the fixed
    cumulative row of `P` compared against a PRNG uniform. Nothing is iterated,
    so nothing decays -- see this module's M-19 paragraph.
    """
    rng = np.random.default_rng(seed)
    cum = np.cumsum(bed["P"], axis=1)
    s = rng.choice(bed["n"], size=n_walkers, p=bed["pi"])
    out = np.empty((n_steps, n_walkers), dtype=np.int8)
    for t in range(n_steps):
        out[t] = s
        s = (rng.random(n_walkers)[:, None] > cum[s]).sum(axis=1)
    return out


def macrostate_of_node(bed: dict) -> np.ndarray:
    """0 = A-side (`q < 1/2`), 1 = the transition-state region (`q = 1/2`, i.e.
    the guard set), 2 = B-side. The committor macrostates, not energy ones."""
    return (np.sign(np.round(bed["q"] - 0.5, 12)) + 1).astype(np.int64)


def transition_matrix(sym: np.ndarray, tau: int, k: int) -> np.ndarray:
    """Row-stochastic `T-hat(tau)` from all overlapping `(t, t+tau)` pairs of a
    `(n_steps, n_walkers)` symbol array."""
    a, b = sym[:-tau].ravel(), sym[tau:].ravel()
    C = np.zeros((k, k), dtype=np.float64)
    np.add.at(C, (a, b), 1.0)
    return C / C.sum(axis=1, keepdims=True)


def ck_error(sym: np.ndarray, tau: int, n: int, k: int) -> float:
    """Chapman-Kolmogorov residual `max |T-hat(n.tau) - T-hat(tau)^n|`."""
    return float(np.abs(transition_matrix(sym, tau * n, k)
                        - np.linalg.matrix_power(transition_matrix(sym, tau, k), n)).max())


# --------------------------------------------------------------------------
# entropy instruments
# --------------------------------------------------------------------------


def entropy_rate_chain(bed: dict) -> float:
    """`h = -sum_i pi_i sum_j P_ij ln P_ij`, nats per step. CLOSED FORM in the
    manifest's own `P` and `pi` -- no orbit, no estimator, no horizon. This is
    the quantity that plays `lambda-hat`'s role in the deficit: the
    Kolmogorov-Sinai theorem bounds `h_sym(alpha) <= h` for every partition
    `alpha`, with equality at a generating one, and for a Markov chain the state
    partition is generating."""
    P, pi = bed["P"], bed["pi"]
    lg = np.where(P > 0.0, np.log(np.where(P > 0.0, P, 1.0)), 0.0)
    return float(-(pi[:, None] * P * lg).sum())


def block_entropy_rate(sym: np.ndarray, alphabet: int, L: int) -> float:
    """`H(L) - H(L-1)`, nats -- the block-entropy slope estimate of the symbolic
    entropy rate. Counts by `np.bincount` over the packed block code (the code
    is bounded by `alphabet**L`), so no sort is involved."""
    if alphabet ** L > 2 ** 40:
        raise ValueError(f"block alphabet {alphabet}**{L} is too large to count")
    S = sym.shape[0]
    H = []
    for Li in (L - 1, L):
        code = np.zeros((S - Li + 1, sym.shape[1]), dtype=np.int64)
        for k in range(Li):
            code = code * alphabet + sym[k:S - Li + 1 + k]
        c = np.bincount(code.ravel(), minlength=alphabet ** Li).astype(np.float64)
        c = c[c > 0]
        p = c / c.sum()
        H.append(float(-(p * np.log(p)).sum()))
    return H[1] - H[0]


def pesin_deficit(bed: dict, sym: np.ndarray, alphabet: int, L: int) -> float:
    """`h_chain - h_sym(partition)`. ADMISSIBLE USE: the near-zero test -- a
    deficit at machine/sampling scale says the partition resolves the dynamics.
    INADMISSIBLE USE, and not implemented anywhere in this module: ranking or
    searching over candidate partitions, which Bollt et al. (2001) rule out by
    proving the deficit non-monotone in misplacement (MISTAKES.md M-19)."""
    return entropy_rate_chain(bed) - block_entropy_rate(sym, alphabet, L)


# --------------------------------------------------------------------------
# Morse census
# --------------------------------------------------------------------------


def morse_census(bed: dict) -> dict:
    """Critical points of `V` on the graph by the lower-link criterion. For a
    1-complex the only indices are 0 and 1: a node whose lower link is empty is
    a minimum; a node whose lower link has `c >= 2` components carries `c - 1`
    index-1 critical points; `c == 1` is a regular point.

    Reports the Morse relation `m_0 - m_1 = chi = V - E` and the Betti numbers
    `b_0 = 1`, `b_1 = E - V + 1` that the Morse inequalities bound."""
    V, n = bed["V"], bed["n"]
    adj = {i: set() for i in range(n)}
    ix = bed["node_index"]
    for u, v in bed["edges"]:
        adj[ix[u]].add(ix[v])
        adj[ix[v]].add(ix[u])
    minima = index_1 = regular = 0
    for i in range(n):
        lower = {j for j in adj[i] if V[j] < V[i]}
        comps, seen = 0, set()
        for j in lower:                     # components of the lower link
            if j in seen:
                continue
            comps += 1
            stack = [j]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(adj[x] & lower)
        if comps == 0:
            minima += 1
        elif comps == 1:
            regular += 1
        else:
            index_1 += comps - 1
    n_edges = len(bed["edges"])
    return dict(n_nodes=n, n_edges=n_edges, minima=minima, index_1=index_1,
                regular=regular, euler=n - n_edges,
                betti_0=1, betti_1=n_edges - n + 1)


# --------------------------------------------------------------------------
# the guard itinerary -- the action vocabulary
# --------------------------------------------------------------------------


def guard_itinerary(bed: dict, traj: np.ndarray) -> dict:
    """Counts of guard traversals by channel and direction.

    A traversal is a maximal run of `q = 1/2` states entered from one side of
    the isocommittor surface and left on the other; its label is the channel of
    the first guard node in the run. `forward` is A-side -> B-side. A crossing
    with no guard state in between (possible only on a perturbed landscape,
    where an edge can straddle `q = 1/2`) lands in `direct`.
    """
    node_channel = {i: c for c, nodes in bed["channels"].items() for i in nodes}
    order = sorted(bed["channels"])
    cid = {c: k for k, c in enumerate(order)}
    side = np.sign(np.round(bed["q"] - 0.5, 12)).astype(np.int64)[traj]
    fwd = np.zeros(len(order), dtype=np.int64)
    bwd = np.zeros(len(order), dtype=np.int64)
    direct = 0
    for w in range(traj.shape[1]):
        nz = np.nonzero(side[:, w])[0]
        if len(nz) < 2:
            continue
        a, b = nz[:-1], nz[1:]
        changed = side[a, w] != side[b, w]
        a, b = a[changed], b[changed]
        direct += int((b == a + 1).sum())
        gap = b > a + 1
        a, b = a[gap], b[gap]
        if len(a) == 0:
            continue
        lab = np.array([cid[node_channel[int(traj[i + 1, w])]] for i in a], dtype=np.int64)
        going_up = side[a, w] < 0
        fwd += np.bincount(lab[going_up], minlength=len(order))
        bwd += np.bincount(lab[~going_up], minlength=len(order))
    return dict(forward={c: int(fwd[cid[c]]) for c in order},
                backward={c: int(bwd[cid[c]]) for c in order},
                total_forward=int(fwd.sum()), total_backward=int(bwd.sum()),
                direct=direct)


# --------------------------------------------------------------------------
# conservation census -- MISTAKES.md V-23: the failures are printed, not omitted
# --------------------------------------------------------------------------


def _row(name, value, tol, note=""):
    return dict(name=name, value=float(value), tol=float(tol),
                conserves=bool(float(value) <= float(tol)), note=note)


def conservation_census(bed: dict, traj: np.ndarray | None = None) -> list[dict]:
    """Every carrier this bed has, with its value, its tolerance and whether it
    conserves -- including the ones that do not.

    V-23 was filed this round against a plural conservation claim evidenced only
    on its passing members. The two rows below that read `conserves = False` are
    the reason this function returns a list rather than a boolean: the
    guard-crossing balance holds only in expectation at finite N, and a channel
    containing a metastable intermediate recrosses its own guard, so its raw
    crossing share exceeds its reactive-flux share. Both are properties of the
    bed, not defects to be tidied away.
    """
    P, pi, q = bed["P"], bed["pi"], bed["q"]
    flow = pi[:, None] * P
    J = reactive_flux(bed)
    net = J.sum(axis=1) - J.sum(axis=0)
    a, b = bed["A"][0], bed["B"][0]
    rows = [
        _row("transition rows sum to 1", np.abs(P.sum(axis=1) - 1.0).max(), 1e-12),
        _row("stationary distribution  pi P = pi", np.abs(pi @ P - pi).max(), 1e-12),
        _row("detailed balance  pi_i P_ij = pi_j P_ji", np.abs(flow - flow.T).max(), 1e-12),
        _row("committor harmonicity  (Lq)_int = 0", harmonic_residual(bed), 1e-12),
        _row("reactive flux divergence-free on interior",
             np.abs(net[bed["interior"]]).max(), 1e-12),
        _row("reactive flux  out(A) = in(B)", abs(net[a] + net[b]), 1e-12),
    ]
    if traj is not None:
        it = guard_itinerary(bed, traj)
        rows.append(_row(
            "guard-crossing balance  #(A->B) = #(B->A)",
            abs(it["total_forward"] - it["total_backward"]), 0.0,
            note=(f"equal only in expectation; the finite-N imbalance is O(sqrt(N)) -- "
                  f"{abs(it['total_forward'] - it['total_backward'])} on "
                  f"{it['total_forward'] + it['total_backward']} events is "
                  f"{abs(it['total_forward'] - it['total_backward']) / np.sqrt(it['total_forward'] + it['total_backward']):.2f} "
                  "sqrt-units. This carrier does not conserve at finite N and is "
                  "reported rather than omitted.")))
        f = channel_flux(bed)
        tot_f = sum(f.values())
        share_emp = it["forward"]["trap"] / max(it["total_forward"], 1)
        share_flux = f["trap"] / tot_f
        ratio_emp = it["forward"]["lo"] / max(it["forward"]["hi"], 1)
        rows.append(_row(
            "itinerary lo:hi ratio vs committor flux ratio",
            abs(ratio_emp / (f["lo"] / f["hi"]) - 1.0), 0.02))
        rows.append(_row(
            "itinerary share vs reactive flux, TRAP channel",
            abs(share_emp / share_flux - 1.0), 0.05,
            note=(f"{share_emp:.5f} of forward crossings against {share_flux:.5f} of the "
                  "reactive flux. A channel whose interior holds a metastable basin "
                  "recrosses its own guard, so raw crossing counts over-represent it; "
                  "the two direct channels' ratio (row above) agrees. This carrier does "
                  "not conserve and is reported rather than omitted.")))
    return rows
