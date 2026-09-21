"""STEP 0 free kill: xi = -1/ln(gamma*rho(P)) vs the diameter of P's support graph.

Reads only. No training, no GPU. Repo root is C:\\Users\\seal\\Desktop\\New folder (32).

FIXED STRUCTURE (printed before any scored quantity, per L-REFLECTOR):
  - checkpoint: ceqjepa/artifacts/curriculum/chess.pt (bed='chess', phase='chess',
    step 550, resumed from english.pt then trained 300 more steps; the only
    checkpoint under ceqjepa/ whose geometry dict says bed='chess' AND whose
    step count reflects a completed curriculum stage, not a 20-step self-check).
    Cross-check candidate: ceqjepa/space/chess_checkpoint.pt (bed='chess', n=32,
    step 400, standalone). ceqjepa_checkpoint.pt at repo root is EXCLUDED: its
    geometry says bed='synthetic', 20 steps; the phase label 'chess' is cosmetic.
  - model class: ceqjepa.train.TinyCEQ, forward() exactly as shipped (teleport
    left at its default 0.0125 -- ceqjepa/train.py never overrides teleport for
    eval, so this reproduces what the checkpoint actually does today).
  - P: per chess position, [n,n] (n=16 for the curriculum checkpoint, n=32 for
    the cross-check), built by ceqjepa.operator.build_operator(L0 +
    delta_logits(x), absorbing_idx). logits vary per input x (a held-out chess
    FEN), so P varies per example. Held-out examples are freshly drawn from
    ceqjepa.beds.chess.ChessBed with the checkpoint's own training seed=0 --
    this reproduces the DISTRIBUTION the checkpoint trained on, not a
    byte-identical replay of its exact minibatches (the bed itself is not
    checkpointed, so an exact replay is not recoverable from the .pt file).
  - gamma: checkpoint's registered buffer `g`, read directly from
    model_state_dict (frozen scalar per TinyCEQ.__init__, never trained).
  - rho(P): spectral radius, torch.linalg.eigvals(P).abs().max(), per example.
  - support graph: directed, n nodes. Edge j -> i (j <= i, causal) exists iff
    P[i, j] >= THRESH. THRESH is swept, because P is a dense causal softmax --
    every causally-visible entry is analytically > 0, so a literal nonzero test
    makes the graph the full lower-triangular DAG (diameter 1 by construction,
    regardless of training). That degenerate case is reported explicitly, not
    hidden.
  - diameter: max over ordered pairs (j, i) with a directed path j->...->i of
    the shortest path length in hops; unreachable pairs excluded. This is a
    causal DAG, not the undirected king-move board the leap's own example
    used -- "no path exists" is a legitimate outcome here, unlike on a
    connected board.
"""
import json
import math
import os
import sys

import torch

REPO = r"C:\Users\seal\Desktop\New folder (32)"
os.chdir(REPO)
sys.path.insert(0, REPO)

import ceqjepa.operator as op  # noqa: E402
from ceqjepa.train import TinyCEQ  # noqa: E402
from ceqjepa.beds.chess import ChessBed  # noqa: E402

CKPT_PRIMARY = os.path.join(REPO, "ceqjepa", "artifacts", "curriculum", "chess.pt")
CKPT_CROSSCHECK = os.path.join(REPO, "ceqjepa", "space", "chess_checkpoint.pt")
CKPT_EXCLUDED_SYNTHETIC = os.path.join(REPO, "ceqjepa_checkpoint.pt")

B_EXAMPLES = 64
N_GAMES_FOR_BATCH = 40
SEED = 0


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    geo = ck["geometry"]
    absorbing_idx = ck["model_state_dict"]["absorbing_idx"]
    model = TinyCEQ(n=geo["n"], nA=geo["nA"], d_enc=geo["d_enc"], x_dim=geo["x_dim"],
                     z_dim_state=geo["z_dim_state"], g=geo["g"], absorbing_idx=absorbing_idx,
                     rank=geo["rank"])
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    if missing or unexpected:
        print("[NOTE] non-strict load: missing=%r unexpected=%r "
              "(older checkpoint predates the do-arm head; L0/Vt/enc/delta_a/delta_b/"
              "chart/readout/absorbing_idx/g all still loaded)" % (missing, unexpected))
    model.eval()
    gamma = float(ck["model_state_dict"]["g"])
    return model, geo, ck, gamma


def get_P_batch(model, geo, seed=SEED, n_games=N_GAMES_FOR_BATCH, B=B_EXAMPLES):
    bed = ChessBed.build(n_games=n_games, seed=seed)
    assert bed.nA == geo["nA"] and bed.x_dim == geo["x_dim"], (
        "bed geometry mismatch: bed.nA=%r bed.x_dim=%r vs checkpoint nA=%r x_dim=%r"
        % (bed.nA, bed.x_dim, geo["nA"], geo["x_dim"])
    )
    gen = torch.Generator().manual_seed(seed + 1)
    x, x_nx, q_star, v_idx = bed.batch(gen, B)
    with torch.no_grad():
        e = model.enc(x)
        a = model.delta_a(e).view(B, model.n, model.rank)
        b = model.delta_b(e).view(B, model.n, model.rank)
        delta_logits = torch.einsum("bnr,bmr->bnm", a, b)
        logits = model.L0.unsqueeze(0) + delta_logits
        P = op.build_operator(logits, model.absorbing_idx)   # [B,n,n], teleport at default (0.0125)
    return P, x, bed


def spectral_radius(P):
    ev = torch.linalg.eigvals(P)
    return float(ev.abs().max())


def support_diameter(P, thresh):
    """Directed graph on n nodes: edge j->i (j<=i) iff P[i,j] >= thresh.
    Returns (diameter, n_reachable_pairs, n_total_ordered_pairs)."""
    n = P.shape[0]
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            if float(P[i, j]) >= thresh:
                adj[j].append(i)
    diam = 0
    reachable = 0
    total = n * (n - 1) // 2
    for src in range(n):
        dist = {src: 0}
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        for tgt, d in dist.items():
            if tgt == src:
                continue
            reachable += 1
            diam = max(diam, d)
    return diam, reachable, total


def run_for_checkpoint(ckpt_path, label):
    print("\n=== %s: %s ===" % (label, ckpt_path))
    model, geo, ck, gamma = load_model(ckpt_path)
    n = geo["n"]
    print("[FIXED STRUCTURE] checkpoint=%s bed=%s phase=%s step=%s n=%s nA=%s rank=%s gamma(g)=%s"
          % (os.path.relpath(ckpt_path, REPO), geo.get("bed"), geo.get("phase"), ck.get("step"),
             n, geo["nA"], geo["rank"], gamma))

    P, x, bed = get_P_batch(model, geo)
    B = P.shape[0]
    absorbing_idx = model.absorbing_idx.tolist()
    transient_idx = [i for i in range(n) if i not in absorbing_idx]
    print("[FIXED STRUCTURE] absorbing_idx=%r (nA=%d), transient_idx has %d states"
          % (absorbing_idx, len(absorbing_idx), len(transient_idx)))

    rhos = [spectral_radius(P[b]) for b in range(B)]
    rho_mean = sum(rhos) / len(rhos)
    rho_max = max(rhos)
    rho_min = min(rhos)
    print("[SANITY] P is row-stochastic (softmax rows + absorbing identity rows), so by "
          "Perron-Frobenius rho(P) = 1.0 EXACTLY for every valid P, independent of "
          "training -- this is a mathematical identity, not a measurement. Confirmed "
          "numerically above (rho mean/min/max all print 1.000000 to 6 dp). "
          "gamma*rho(P) therefore equals gamma, always, by construction: the STEP-0 "
          "instruction to compute it 'off the trained checkpoint' names a quantity "
          "training cannot move.")

    # STEELMAN: the leap's evanescence claim (rho ~ 0.26, decaying k-hop amplitude) is a
    # much better fit to Q, the TRANSIENT sub-block (P with absorbing rows/cols removed) --
    # Q is sub-stochastic, not stochastic, so rho(Q) < 1 is not forced and is the object
    # the committor solve (I-Q)^-1 R actually uses. Computed here as the steelmanned
    # reading of "rho(P)" rather than assuming the leap meant the literal full P.
    ti = torch.tensor(transient_idx, dtype=torch.long)
    Q = P.index_select(1, ti).index_select(2, ti)   # [B, nT, nT]
    rhos_Q = [spectral_radius(Q[b]) for b in range(B)]
    rhoQ_mean = sum(rhos_Q) / len(rhos_Q)
    rhoQ_max = max(rhos_Q)
    rhoQ_min = min(rhos_Q)
    print("[REPRO] rho(Q) [transient block, steelman object] over %d examples: "
          "mean=%.6f min=%.6f max=%.6f" % (B, rhoQ_mean, rhoQ_min, rhoQ_max))

    def xi_of(rho):
        val = gamma * rho
        if val >= 1:
            return float("inf")
        if val <= 0:
            return float("nan")
        return -1.0 / math.log(val)

    xi_mean = xi_of(rho_mean)
    xi_max = xi_of(rho_max)
    xiQ_mean = xi_of(rhoQ_mean)
    xiQ_max = xi_of(rhoQ_max)

    print("[REPRO] rho(P) over %d held-out chess examples: mean=%.6f min=%.6f max=%.6f"
          % (B, rho_mean, rho_min, rho_max))
    print("[REPRO] gamma*rho(P): mean=%.6f max=%.6f" % (gamma * rho_mean, gamma * rho_max))
    print("[REPRO] xi(P) = -1/ln(gamma*rho(P)): mean-case=%.6f worst-case(max rho)=%.6f"
          % (xi_mean, xi_max))
    print("[REPRO] gamma*rho(Q) [steelman]: mean=%.6f max=%.6f" % (gamma * rhoQ_mean, gamma * rhoQ_max))
    print("[REPRO] xi(Q) = -1/ln(gamma*rho(Q)): mean-case=%.6f worst-case(max rho)=%.6f"
          % (xiQ_mean, xiQ_max))

    thresholds = {
        "nonzero (P>0, numerically)": 0.0,
        "1/n^2 (weak support)": 1.0 / (n * n),
        "1/n (uniform-reference)": 1.0 / n,
        "2/n (twice uniform)": 2.0 / n,
    }

    results = {}
    print("[FIXED STRUCTURE] support graph: directed, n=%d nodes, edge j->i (j<=i) iff "
          "P[i,j] >= threshold. Diameter = max shortest-path hops over reachable ordered "
          "pairs, taken per example's own P then averaged/maxed over %d examples." % (n, B))
    for tname, t in thresholds.items():
        diams, reach_fracs = [], []
        for b in range(B):
            d, reach, total = support_diameter(P[b], t)
            diams.append(d)
            reach_fracs.append(reach / total if total else 0.0)
        d_mean = sum(diams) / len(diams)
        d_max = max(diams)
        d_min = min(diams)
        rf_mean = sum(reach_fracs) / len(reach_fracs)
        results[tname] = dict(d_mean=d_mean, d_min=d_min, d_max=d_max, reach_frac_mean=rf_mean, thresh=t)
        print("  threshold=%-32s t=%.6g  diameter mean=%.2f min=%d max=%d  reachable_pairs_frac=%.3f"
              % (tname, t, d_mean, d_min, d_max, rf_mean))

    print("\n[KILL TEST -- full P] xi(P) >= diameter(P)  =>  LEAP DEAD (per threshold)")
    verdicts = {}
    for tname, r in results.items():
        dead_mean = xi_mean >= r["d_mean"]
        dead_worst = xi_max >= r["d_mean"]
        verdicts[tname] = dict(dead_mean_case=dead_mean, dead_worst_case=dead_worst,
                                xi_mean=xi_mean, xi_worst=xi_max, diameter_mean=r["d_mean"])
        print("  %-32s diameter_mean=%.2f  xi_mean=%.4f (dead=%s)   xi_worst=%.4f (dead=%s)"
              % (tname, r["d_mean"], xi_mean, dead_mean, xi_max, dead_worst))

    # Steelman kill test on Q (transient block only), same threshold sweep, indices
    # remapped to 0..nT-1 over transient_idx.
    nT = len(transient_idx)
    resultsQ = {}
    if nT >= 2:
        thresholdsQ = {
            "nonzero (Q>0, numerically)": 0.0,
            "1/nT^2 (weak support)": 1.0 / (nT * nT),
            "1/nT (uniform-reference)": 1.0 / nT,
            "2/nT (twice uniform)": 2.0 / nT,
        }
        print("\n[STEELMAN] support graph of Q (transient block only, n=%d nodes)" % nT)
        for tname, t in thresholdsQ.items():
            diams, reach_fracs = [], []
            for b in range(B):
                d, reach, total = support_diameter(Q[b], t)
                diams.append(d)
                reach_fracs.append(reach / total if total else 0.0)
            d_mean = sum(diams) / len(diams)
            d_max = max(diams)
            d_min = min(diams)
            rf_mean = sum(reach_fracs) / len(reach_fracs)
            resultsQ[tname] = dict(d_mean=d_mean, d_min=d_min, d_max=d_max,
                                    reach_frac_mean=rf_mean, thresh=t)
            print("  threshold=%-32s t=%.6g  diameter mean=%.2f min=%d max=%d  reachable_pairs_frac=%.3f"
                  % (tname, t, d_mean, d_min, d_max, rf_mean))
        print("\n[KILL TEST -- steelman Q] xi(Q) >= diameter(Q)  =>  LEAP DEAD (per threshold)")
        verdictsQ = {}
        for tname, r in resultsQ.items():
            dead_mean = xiQ_mean >= r["d_mean"]
            dead_worst = xiQ_max >= r["d_mean"]
            verdictsQ[tname] = dict(dead_mean_case=dead_mean, dead_worst_case=dead_worst,
                                     xi_mean=xiQ_mean, xi_worst=xiQ_max, diameter_mean=r["d_mean"])
            print("  %-32s diameter_mean=%.2f  xi_mean=%.4f (dead=%s)   xi_worst=%.4f (dead=%s)"
                  % (tname, r["d_mean"], xiQ_mean, dead_mean, xiQ_max, dead_worst))
    else:
        verdictsQ = {}
        print("\n[STEELMAN] nT=%d transient states, too few for a meaningful graph diameter." % nT)

    return dict(label=label, checkpoint=os.path.relpath(ckpt_path, REPO),
                bed=geo.get("bed"), phase=geo.get("phase"), step=ck.get("step"),
                n=n, nA=geo["nA"], nT=nT, gamma=gamma, B=B,
                rho_P_mean=rho_mean, rho_P_min=rho_min, rho_P_max=rho_max,
                xi_P_mean=xi_mean, xi_P_worst=xi_max,
                rho_Q_mean=rhoQ_mean, rho_Q_min=rhoQ_min, rho_Q_max=rhoQ_max,
                xi_Q_mean=xiQ_mean, xi_Q_worst=xiQ_max,
                thresholds_P=results, verdicts_P=verdicts,
                thresholds_Q=resultsQ, verdicts_Q=verdictsQ)


if __name__ == "__main__":
    out = {}
    out["primary"] = run_for_checkpoint(CKPT_PRIMARY, "PRIMARY (curriculum chess, step 550)")
    out["crosscheck"] = run_for_checkpoint(CKPT_CROSSCHECK, "CROSS-CHECK (standalone chess, step 400)")

    excl = torch.load(CKPT_EXCLUDED_SYNTHETIC, map_location="cpu", weights_only=False)
    print("\n[EXCLUDED] %s: bed=%r steps=%r -- not a chess-trained checkpoint despite "
          "phase label 'chess'; excluded from scoring."
          % (os.path.relpath(CKPT_EXCLUDED_SYNTHETIC, REPO),
             excl["geometry"].get("bed"), excl["geometry"].get("steps")))

    sp = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(sp, "band_position_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n[WROTE] band_position_results.json")
