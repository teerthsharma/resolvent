"""Chase K1 bars: numerics (clause 6), bf16 on the fused path, BOS (clause 7), R-RANGE. Registered 2026-09-23 before any
measurement; RED run first with CHASE_STUB=1 (every bar raises). Instruments: instr.py. g = 0.999 throughout.
Board: {"t":"test","agent":"Chase","status":..,"name":"chase.k1.<bar>"}. Exit 1 if any bar is red.

bf16 ON THE FUSED PATH (K0's struck row-sum failure was measured on a materialized bf16 W; these bars measure fs5c).
Setup (K0's): B1 H8 D64, q, k, v ~ N(0,1) from torch.Generator seed 0 (float64 -> fp32), SSMax a = 1, b = 0
(R.ssmax_q), S in {1024, 4096}. "Leak" = R.fs5c called with fp32 inputs inside torch.autocast("cuda", bfloat16), i.e.
the hook's two guards removed (autocast then decides the dtype of every op inside, eager, not graphed).
 bf16_leak_fused_read1     PREDICTION: with V = 1 the leak's read stays max|x - 1| <= 1e-5 at both S (the fused path
                           normalizes every block with one shared log-normalizer, so rounded logits cannot make a row
                           sum exceed 1; the K0 failure needs W's entries rounded after normalization).
 bf16_leak_fused_pattern   PREDICTION: with V ~ N(0,1) the leak's read is wrong: max|x - x64| / max|x64| > 1e-3 at both
                           S (x64 = float64 dense solve per head, K0 R.resolvent path "dense").
 bf16_inputs_fused_raises  PREDICTION: fs5c with q, k, v cast to bf16 (no autocast; what the unguarded layer would receive
                           from bf16 projections) raises, or returns a non-finite x; it does not silently return finite.

CLAUSE 6 (RECORD_K): on every finished (f_R) R0 checkpoint in SP/phase_k/K1/cameron (rdepth.py runs: read at S = 1024 on
make_train(default_rng([13, 0])) and S = 16384 on make_test(default_rng([12, 16384, 0]))) and SP/phase_k/K1/wilson
(train_ladder runs: the first eval draw of 2 sequences at ctx), with qs, k captured from the layer's own fp32
computation under the trainer's bf16 autocast:
 c6_numerics_R0   >= 1 checkpoint, and on every (checkpoint, S): g * max_i rowsum(P) < 1 (P = fp32 max-shifted softmax
                  row blocks, c = 256, never a full W) AND the hook's read of V = 1 has max|x - 1| <= 1e-5.
 c6_power_R0      the two halves can fail on these checkpoints: on >= 1 (checkpoint, S) the bf16-rounded P gives
                  g * max rowsum >= 1, AND on >= 1 (checkpoint, S) the K0 fs5 forward (naive pivot) gives
                  max|x(V=1) - 1| > 1e-5. (If this is RED, c6_numerics is vacuous on these checkpoints.)
CLAUSE 7:
 c7_bos_R0        >= 1 checkpoint, and every resolvent-layer head at every (checkpoint, S) above: BOS share < 0.5, where
                  BOS share = mean over the batch and rows i in [S/2, S) of R(i,0) = (1-g) G(i,0) (the read with
                  V = e_0, fused path).
 c7_route_gamma_h ROUTE, registered now, runs only if c7_bos_R0 is RED: per failing (checkpoint, S), per head,
                  lam_h = the band-mass decay (below) of the head's single-hop W rows (qs, no resolvent), D_t = S / 2,
                  g_h = (e^lam_h - e^(1/D_t)) / (e^lam_h - 1) clipped to [0, .999]. PREDICTION: BOS share < 0.5 in
                  every head at g_h on the same checkpoint (no retraining).
R-RANGE (contract section 5). Instrument: band mass B(r) = mean over batch and 64 rows i in [S/2, S) of
sum_{i-r-32 < j <= i-r} K(i, j), direct window sums, position 0 (the absorber) never inside a band; K = (1-g) G for a
resolvent head (rows by the adjoint solve), K = W for a single hop; m = -slope of ln B(r) weighted by B (polyfit
w = sqrt(B)); range = 1/m (inf if m <= 0). Rows: 64 rows spread over [S/2, S); r in [16, S/2 - 32).
 rrange_band_y2        the band-mass instrument on the Y2 Toeplitz head (K0 rrange.toeplitz_head, n 3000, rows
                       2000..2900 step 100, r in [30, 400)): |m - ln(g + (1-g)e^lam)| <= 1e-4 and relative <= 1e-4,
                       lam .5 / 1, g .9 / .99 / .999.
 rrange_band_sink      the same Toeplitz head with a BOS sink of share sigma = 0.2 (W <- (1-s)W, W[:,0] += s):
                       |m - ln(g' + (1-g')e^lam)| <= 1e-4 and relative <= 1e-4, g' = g(1 - sigma), all six cells.
 rrange_tail_sink_fooled  K0's tail-mass instrument (rrange.head_mass_tail, position 0 included) on the sink head reads
                       m < 0.5 m' (range more than doubled) in >= 1 of the six cells. (Kill of the tail instrument on
                       heads with a sink; replacement: band mass.)
 rrange_fR_depth_R0    >= 1 finished Cameron (f_R) R0 checkpoint, and on each, >= 1 resolvent-layer head has band-mass
                       range >= the max chain depth (hops) of the S = 16384 bed make_test(default_rng([12, 16384, 0])).
                       (Contract prediction; its counter "no long-range head" is this bar's failure.)
 rrange_fR_span_R0     same checkpoints and bed: >= 1 resolvent head with range >= max_i (i - position of i's root),
                       in tokens (a hop spans ~16 tokens on bed_k, so the literal depth bar is 16x weaker).
 rrange_aL_slope_R0    >= 1 finished (a_L) checkpoint (Cameron aL, any depth; Wilson --attn alibi, any rung) and every
                       ALiBi head of every layer, single hop, read at S = 1024 (Cameron) / ctx (Wilson):
                       |ln(range * slope_h)| <= ln 2 (contract: "(a_L) heads have range ~ 1/slope").
"""
import json, math, os, sys, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BOARD = "C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
ROWS = os.path.join(HERE, "ckpt_rows.jsonl")
results = {}


def _stub():
    if os.environ.get("CHASE_STUB") == "1":
        raise NotImplementedError("stub")


def bar(name):
    def deco(fn):
        def run():
            try:
                ok, val = fn()
            except Exception as e:
                ok, val = False, "EXC " + type(e).__name__ + ": " + str(e)[:300]
                traceback.print_exc()
            if ok is None:
                print("SKIP " + name + ": " + json.dumps(val), flush=True)
                return
            st = "green" if ok else "red"
            print(("GREEN " if ok else "RED ") + name + ": " + json.dumps(val), flush=True)
            results[name] = st
            if os.environ.get("CHASE_BOARD", "1") == "1":
                with open(BOARD, "a") as f:
                    f.write(json.dumps({"t": "test", "agent": "Chase", "status": st, "name": "chase.k1." + name}) + "\n")
        run.__name__ = name
        return run
    return deco


def _qkv(S, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 8, S, 64, generator=g, dtype=torch.float64).float().cuda() for _ in range(3)]


def _leak(S):
    import torch
    import instr as I
    q, k, v = _qkv(S)
    qs = I.R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x1 = I.R.fs5c(qs, k, torch.ones_like(v), I.GAMMA)
        xv = I.R.fs5c(qs, k, v, I.GAMMA)
    return qs, k, v, x1.float(), xv.float()


@bar("bf16_leak_fused_read1")
def t_leak_read1():
    _stub()
    out = {}
    for S in (1024, 4096):
        _, _, _, x1, _ = _leak(S)
        out[S] = float((x1 - 1).abs().max())
    return all(math.isfinite(d) and d <= 1e-5 for d in out.values()), out


@bar("bf16_leak_fused_pattern")
def t_leak_pattern():
    _stub()
    import torch
    import instr as I
    out = {}
    for S in (1024, 4096):
        qs, k, v, _, xv = _leak(S)
        ref = torch.cat([I.R.resolvent(qs[:, h:h + 1].double(), k[:, h:h + 1].double(), v[:, h:h + 1].double(),
                                       I.GAMMA, path="dense") for h in range(8)], 1)
        out[S] = float((xv.double() - ref).abs().max() / ref.abs().max())
    return all(e > 1e-3 for e in out.values()), out


@bar("bf16_inputs_fused_raises")
def t_bf16_inputs():
    _stub()
    import torch
    import instr as I
    q, k, v = _qkv(1024)
    qs = I.R.ssmax_q(q, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
    try:
        x = I.R.fs5c(qs.bfloat16(), k.bfloat16(), torch.ones_like(v).bfloat16(), I.GAMMA)
    except Exception as e:
        return True, {"raised": type(e).__name__ + ": " + str(e)[:160]}
    fin = bool(torch.isfinite(x).all())
    return not fin, {"raised": None, "finite": fin, "dtype": str(x.dtype), "read1_dev": float((x.float() - 1).abs().max())}


def _y2(sigma):
    import instr as I
    out, ok = {}, True
    for lam in (0.5, 1.0):
        for g in (0.9, 0.99, 0.999):
            gp = g * (1 - sigma)
            m, _ = I.toeplitz_band_mass(lam, g, sigma)
            pred = math.log(gp + (1 - gp) * math.exp(lam))
            out[f"{lam}/{g}"] = {"m": m, "pred": pred, "abs": abs(m - pred), "rel": abs(m - pred) / pred}
            ok &= abs(m - pred) <= 1e-4 and abs(m - pred) / pred <= 1e-4
    return ok, out


@bar("rrange_band_y2")
def t_band_y2():
    _stub()
    return _y2(0.0)


@bar("rrange_band_sink")
def t_band_sink():
    _stub()
    return _y2(0.2)


@bar("rrange_tail_sink_fooled")
def t_tail_fooled():
    _stub()
    import instr as I
    out, n = {}, 0
    for lam in (0.5, 1.0):
        for g in (0.9, 0.99, 0.999):
            _, W = I.toeplitz_band_mass(lam, g, 0.2)
            gp = g * 0.8
            mp = math.log(gp + (1 - gp) * math.exp(lam))
            mt = I.toeplitz_tail_mass(W, g)
            out[f"{lam}/{g}"] = {"m_tail": mt, "m_true": mp, "ratio": mt / mp}
            n += mt < 0.5 * mp
    return n >= 1, out


# ---------------- checkpoint measurements (computed once, rows saved) ----------------

_M = {}


def measure():
    """One pass over every finished checkpoint; rows appended to ckpt_rows.jsonl with the checkpoint sha256."""
    if _M:
        return _M
    import torch
    import instr as I
    cks = I.find_ckpts()
    _M["cks"] = []
    for ck in cks:
        lengths = (1024, 16384) if ck["lane"] == "cameron" else None
        cap = I.capture(ck, lengths)
        rec = {"lane": ck["lane"], "arm": ck["arm"], "path": ck["path"], "sha256": I.sha(ck["path"]), "S": {}}
        for S, c in cap.items():
            row = {"a": c["a"], "b": c["b"]}
            if c["res"] is not None:
                qs, k = c["res"]["qs"].float(), c["res"]["k"].float()
                row.update(I.rowsums(qs, k))
                row.update(I.read1(qs, k))
                row["bos"] = I.bos_share(qs, k)
                rows, r = I.rows_for(S), I.r_for(S)
                Bm = I.band_mass(I.resolvent_rows(qs, k, rows), rows, r)
                row["res_m"] = [I.fit_mass(Bm[h], r) for h in range(Bm.shape[0])]
                row["res_range"] = [I.rng_of(m) for m in row["res_m"]]
                Bh = I.band_mass(I.hop_rows(qs, k, rows), rows, r)
                row["res_hop_m"] = [I.fit_mass(Bh[h], r) for h in range(Bh.shape[0])]
            if c["bed"] is not None:
                bed = c["bed"]
                row["bed_max_depth"] = int(bed["depth"].max())
                row["bed_max_span"] = int((np.arange(len(bed["root"])) - bed["root"]).max())
            al = []
            for e in c["alibi"]:
                rows, r = I.rows_for(S), I.r_for(S)
                Bm = I.band_mass(I.hop_rows(e["q"], e["k"], rows, e["slopes"]), rows, r)
                for h in range(Bm.shape[0]):
                    m = I.fit_mass(Bm[h], r)
                    al.append({"layer": e["layer"], "head": h, "slope": float(e["slopes"][h]), "m": m,
                               "range": I.rng_of(m)})
            row["alibi_heads"] = al
            rec["S"][S] = row
        with open(ROWS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        _M["cks"].append((ck, cap, rec))
        del cap
        torch.cuda.empty_cache()
    return _M


def _fr():
    return [(ck, cap, rec) for ck, cap, rec in measure()["cks"] if ck["arm"] == "fR"]


@bar("c6_numerics_R0")
def t_c6():
    _stub()
    fr = _fr()
    out, ok = {}, len(fr) >= 1
    for ck, _, rec in fr:
        for S, row in rec["S"].items():
            good = row["gamma_max_rowsum"] < 1 and row["finite"] and row["read1_dev"] <= 1e-5
            ok &= good
            out[f"{ck['lane']}:{os.path.basename(os.path.dirname(ck['path']))}:S{S}"] = {
                "g_max_rowsum": row["gamma_max_rowsum"], "read1_dev": row["read1_dev"], "ok": good}
    return ok, {"n_ckpt": len(fr), **out}


@bar("c6_power_R0")
def t_c6_power():
    _stub()
    rows = [row for _, _, rec in _fr() for row in rec["S"].values()]
    p_rs = sum(r["gamma_max_rowsum_bf16"] >= 1 for r in rows)
    p_rd = sum(r["read1_dev_fs5"] > 1e-5 for r in rows)
    return p_rs >= 1 and p_rd >= 1, {"cells": len(rows), "bf16_rowsum_fails": p_rs, "fs5_read1_fails": p_rd,
                                     "max_bf16_g_rowsum": max((r["gamma_max_rowsum_bf16"] for r in rows), default=None),
                                     "max_fs5_dev": max((r["read1_dev_fs5"] for r in rows), default=None)}


C7 = {}


@bar("c7_bos_R0")
def t_c7():
    _stub()
    fr = _fr()
    out, ok = {}, len(fr) >= 1
    for ck, _, rec in fr:
        for S, row in rec["S"].items():
            sh = row["bos"]["mean_2nd_half"]
            good = all(s < 0.5 for s in sh)
            ok &= good
            key = f"{ck['lane']}:{os.path.basename(os.path.dirname(ck['path']))}:S{S}"
            out[key] = {"share": sh, "last_row": row["bos"]["last_row"]}
            if not good:
                C7.setdefault("fail", []).append((ck, S))
    C7["ran"] = True
    return ok, {"n_ckpt": len(fr), **out}


@bar("c7_route_gamma_h")
def t_c7_route():
    _stub()
    if not C7.get("fail"):
        return None, {"reason": "c7_bos_R0 not red; route not run"}
    import torch
    import instr as I
    out, ok = {}, True
    by = {id(ck): (ck, cap, rec) for ck, cap, rec in _fr()}
    for ck, S in C7["fail"]:
        _, cap, rec = by[id(ck)]
        c = cap[S]["res"]
        qs, k = c["qs"].float(), c["k"].float()
        gh, sh = [], []
        for h, lam in enumerate(rec["S"][S]["res_hop_m"]):
            g = I.gamma_h(lam, S / 2)
            gh.append(g)
            sh.append(I.bos_share(qs[:, h:h + 1].contiguous(), k[:, h:h + 1].contiguous(), g=g)["mean_2nd_half"][0])
        ok &= all(s < 0.5 for s in sh)
        out[f"{ck['lane']}:S{S}"] = {"lam": rec["S"][S]["res_hop_m"], "g_h": gh, "share": sh}
    return ok, out


def _cam_fr():
    return [(ck, rec) for ck, _, rec in _fr() if ck["lane"] == "cameron"]


def _range_bar(key):
    rows = _cam_fr()
    out, ok = {}, len(rows) >= 1
    for ck, rec in rows:
        row = rec["S"][16384]
        need = row["bed_max_depth" if key == "depth" else "bed_max_span"]
        good = any(rg >= need for rg in row["res_range"])
        ok &= good
        out[os.path.basename(os.path.dirname(ck["path"]))] = {"range": row["res_range"], "need": need}
    return ok, {"n_ckpt": len(rows), **out}


@bar("rrange_fR_depth_R0")
def t_rr_depth():
    _stub()
    return _range_bar("depth")


@bar("rrange_fR_span_R0")
def t_rr_span():
    _stub()
    return _range_bar("span")


@bar("rrange_aL_slope_R0")
def t_rr_al():
    _stub()
    al = [(ck, rec) for ck, _, rec in measure()["cks"] if ck["arm"] == "aL"]
    out, ok = {}, len(al) >= 1
    for ck, rec in al:
        S = 1024 if ck["lane"] == "cameron" else ck["cfg"]["ctx"]
        hs = rec["S"][S]["alibi_heads"]
        good = all(abs(math.log(h["range"] * h["slope"])) <= math.log(2) if 0 < h["range"] < float("inf") else False
                   for h in hs)
        ok &= good
        out[f"{ck['lane']}:{os.path.basename(os.path.dirname(ck['path']))}"] = [
            (h["layer"], h["head"], round(h["range"], 2), round(1 / h["slope"], 2)) for h in hs]
    return ok, {"n_ckpt": len(al), **out}


if __name__ == "__main__":
    only = sys.argv[1:]
    synth = [t_leak_read1, t_leak_pattern, t_bf16_inputs, t_band_y2, t_band_sink, t_tail_fooled]
    ck = [t_c6, t_c6_power, t_c7, t_c7_route, t_rr_depth, t_rr_span, t_rr_al]
    tests = [t for t in synth + ck if not only or t.__name__ in only]
    from gpulock import gpu_lock
    if os.environ.get("CHASE_STUB") == "1":
        for t in tests:
            t()
    else:
        with gpu_lock("chase.k1 ckpt bars"):
            for t in tests:
                t()
    print("SUMMARY", json.dumps(results))
    sys.exit(1 if any(s == "red" for s in results.values()) else 0)
