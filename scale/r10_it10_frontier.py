"""R10 it.10 JUPITER: isotonic frontier fit + NRMSE=1.0 crossing, from measured cells only.

Reads the three it.8 capacity journals, fits NRMSE as a monotone-decreasing function
of log2(n_train) at steps=150, reports the 1.0 crossing, tests the monotonicity
assumption the fit rests on, and reports the sign of dNRMSE/d(steps) everywhere two
or more step rungs were measured.

No training, no allocation. Pure re-read of results/r10_it8_capacity_softmax_t*.jsonl.
Those journals are append-only and were still being written when this ran, so every
number is pinned to the sha256 and line count recorded in the meta record.
"""
import datetime, hashlib, json, math, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = {t: ROOT / f"results/r10_it8_capacity_softmax_t{t}.jsonl" for t in (2, 8, 32)}
BAR = 1.0
NS = [2048, 8192, 32768]
STEPS = [150, 600, 2400, 9600]
CEILING = {2: 0.7071067811865476, 8: 0.9354143466934853, 32: 0.9842509842514764}


# ---- two isotonic backends, distinct names, one explicit selection ----------------
def isofit_pava(x, y):
    """Pool-adjacent-violators for a DECREASING fit, unit weights. ~15 lines, no deps."""
    blocks = [[v, 1.0] for _, v in sorted(zip(x, y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:  # violates "decreasing" -> pool
            s = blocks[i]
            w = blocks.pop(i + 1)
            s[0] = (s[0] * s[1] + w[0] * w[1]) / (s[1] + w[1])
            s[1] += w[1]
            i = max(i - 1, 0)
        else:
            i += 1
    return [b[0] for b in blocks for _ in range(int(b[1]))]


try:
    import sklearn
    from sklearn.isotonic import IsotonicRegression

    def isofit_sklearn(x, y):
        return [float(v) for v in IsotonicRegression(increasing=False).fit_transform(x, y)]

    isofit = isofit_sklearn
    ISO_LIB = f"sklearn.isotonic.IsotonicRegression {sklearn.__version__}"
except ImportError:
    isofit = isofit_pava
    ISO_LIB = "in-repo pool-adjacent-violators (sklearn not installed)"

# the selected backend must agree with the independent one on a sequence that
# actually violates monotonicity, or the "which did you use" line means nothing
_VX, _VY = [11.0, 13.0, 15.0], [1.0, 0.6, 0.8]
assert all(abs(a - b) < 1e-12 for a, b in zip(isofit(_VX, _VY), [1.0, 0.7, 0.7])), "backend wrong"
assert all(abs(a - b) < 1e-12 for a, b in zip(isofit_pava(_VX, _VY), isofit(_VX, _VY)))

# ---- load every cell record, keeping file + 1-based line number as provenance ----
cells = []
SNAPSHOT = {}
for t, path in SRC.items():
    raw = path.read_bytes()
    SNAPSHOT[path.name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                           "lines": len(raw.decode().splitlines())}
    for ln, line in enumerate(raw.decode().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("t") == "cell":
            r["_src"] = f"{path.name}:{ln}"
            cells.append(r)


def rows(t, n, steps):
    return [c for c in cells if c["t_star"] == t and c["n_train"] == n and c["steps"] == steps]


def dedup(rs):
    """Collapse bit-identical duplicate journal rows (same seed, same numbers)."""
    seen, out = set(), []
    for r in rs:
        k = (r["seed"], r["threads"], r["eval_nrmse"], r["boot_lo"], r["boot_hi"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def point(t, n, steps, seeds=None):
    """Aggregate one (t*, n, steps) cell, or None when the cell was never run.

    One value per seed. Where a seed was run at two thread counts, the lower thread
    count wins, so a cell never double-counts one seed under float reduction drift.
    """
    rs = [r for r in dedup(rows(t, n, steps)) if seeds is None or r["seed"] in seeds]
    if not rs:
        return None
    by_seed = {}
    for r in rs:
        by_seed.setdefault(r["seed"], []).append(r)
    vals = [sorted(v, key=lambda r: r["threads"])[0]["eval_nrmse"] for v in by_seed.values()]
    return {
        "t_star": t, "n_train": n, "steps": steps,
        "nrmse": sum(vals) / len(vals),
        "n_seeds": len(by_seed), "seeds": sorted(by_seed),
        "threads": sorted({r["threads"] for r in rs}),
        "src": [r["_src"] for r in rs],
        "agg": "single run, 1 seed" if len(vals) == 1 else f"mean over {len(vals)} seeds",
    }


OUT = []


def emit(rec):
    OUT.append(rec)
    return rec


# ---- 0. noise floor: same (t*, n, steps, seed) rerun at a different thread count ----
drifts = []
for t in (2, 8, 32):
    for n in NS:
        for s in STEPS:
            by_seed = {}
            for r in dedup(rows(t, n, s)):
                by_seed.setdefault(r["seed"], []).append(r)
            for seed, rs in by_seed.items():
                if len({r["threads"] for r in rs}) > 1:
                    vs = sorted((r["threads"], r["eval_nrmse"], r["_src"]) for r in rs)
                    drifts.append({"t_star": t, "n_train": n, "steps": s, "seed": seed,
                                   "threads": [v[0] for v in vs], "eval_nrmse": [v[1] for v in vs],
                                   "abs_drift": max(v[1] for v in vs) - min(v[1] for v in vs),
                                   "src": [v[2] for v in vs]})
FLOOR = max((d["abs_drift"] for d in drifts), default=0.0)
emit({"t": "noise_floor", "mechanism": "same seed, same config, different thread count: "
      "float reduction order only", "pairs": drifts, "max_abs_drift": FLOOR,
      "note": "this is the resolution limit on any difference between two cells swept at "
              "different thread counts; the grid used threads 8, 6, 6, 12, 12"})

emit({"t": "meta", "round": "r10", "iteration": 10, "agent": "JUPITER",
      "read_at": datetime.datetime.now().isoformat(timespec="seconds"),
      "sources": SNAPSHOT, "bar": BAR, "isotonic_library": ISO_LIB,
      "isotonic_backends_available": ["pool-adjacent-violators (in-repo)"]
      + ([ISO_LIB] if isofit is not isofit_pava else []),
      "interpolation_variable": "log2(n_train)", "noise_floor_abs_nrmse": FLOOR,
      "note": "no training run; re-read of measured cells only. Source journals are "
              "append-only and were live at read time; numbers pin to the sha256 above."})


# ---- 1-3. per-rung series, monotonicity test, isotonic fit, crossing ----
def cross(xs, ys):
    """Crossing of BAR. xs ascending log2(n), ys the fitted curve. Always a dict."""
    if ys[0] < BAR:
        return {"status": "below_grid", "n_star": None,
                "note": f"already {ys[0]:.6f} < 1.0 at the smallest measured n={2 ** int(xs[0])}; "
                        "the crossing lies below the grid and is NOT extrapolated backwards"}
    for i in range(len(xs) - 1):
        if ys[i] >= BAR > ys[i + 1]:
            f = (ys[i] - BAR) / (ys[i] - ys[i + 1])
            x = xs[i] + f * (xs[i + 1] - xs[i])
            return {"status": "interpolated", "n_star": 2 ** x, "log2_n_star": x,
                    "bracket_n": [2 ** int(xs[i]), 2 ** int(xs[i + 1])],
                    "note": "linear interpolation in log2(n) between two measured n"}
    sl = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    x = xs[-1] + (BAR - ys[-1]) / sl
    return {"status": "extrapolated", "n_star": 2 ** x, "log2_n_star": x,
            "octaves_beyond_grid": x - xs[-1], "last_segment_slope_per_octave": sl,
            "note": "linear extrapolation in log2(n) past the largest measured n; "
                    "not bracketed by any measured cell"}


def do_rung(t, steps, seeds=None, tag=""):
    """Full treatment of one (t*, steps) rung. Emits point / monotonicity / isotonic /
    crossing records, and bails out with not_measured when fewer than two n exist."""
    pts = [(n, point(t, n, steps, seeds)) for n in NS]
    have = [(n, p) for n, p in pts if p is not None]
    missing = [n for n, p in pts if p is None]
    for n, p in pts:
        if p is None:
            emit({"t": "point", "t_star": t, "n_train": n, "steps": steps, "variant": tag,
                  "status": "not_measured", "nrmse": None,
                  "note": "cell never run; dropped as unaffordable in the it.8 sweep"})
        else:
            emit({"t": "point", **p, "variant": tag, "status": "measured",
                  "log2_n": math.log2(n)})
    if len(have) < 2:
        emit({"t": "crossing", "t_star": t, "steps": steps, "variant": tag,
              "status": "not_measured", "measured_n": [n for n, _ in have],
              "missing_n": missing,
              "note": "fewer than two n measured at this rung; no curve, no crossing"})
        return None
    xs = [math.log2(n) for n, _ in have]
    ys = [p["nrmse"] for _, p in have]
    viol = [{"from_n": have[i][0], "to_n": have[i + 1][0], "delta": ys[i + 1] - ys[i]}
            for i in range(len(ys) - 1) if ys[i + 1] > ys[i]]
    emit({"t": "monotonicity", "t_star": t, "steps": steps, "variant": tag,
          "n_train": [n for n, _ in have], "observed": ys,
          "monotone_decreasing": not viol, "violations": viol,
          "max_violation": max([v["delta"] for v in viol], default=0.0),
          "gaps": [ys[i] - ys[i + 1] for i in range(len(ys) - 1)],
          "min_gap_over_noise_floor": (min(ys[i] - ys[i + 1] for i in range(len(ys) - 1)) / FLOOR
                                       if FLOOR else None),
          "src": [s for _, p in have for s in p["src"]]})
    yh = isofit(xs, ys)
    d = max(abs(a - b) for a, b in zip(ys, yh))
    emit({"t": "isotonic", "t_star": t, "steps": steps, "variant": tag, "log2_n": xs,
          "y_observed": ys, "y_fitted": yh, "library": ISO_LIB,
          "is_identity": d == 0.0, "max_abs_change": d,
          "note": "fit changed nothing: observed sequence already monotone decreasing"
                  if d == 0.0 else "fit pooled at least one violating pair"})
    c = cross(xs, yh)
    rec = {"t": "crossing", "t_star": t, "steps": steps, "variant": tag, "bar": BAR, **c,
           "measured_n": [n for n, _ in have], "missing_n": missing,
           "n_seeds_per_n": [p["n_seeds"] for _, p in have],
           "threads_per_n": [p["threads"] for _, p in have],
           "one_hop_ceiling": CEILING[t], "headroom_below_bar": BAR - CEILING[t],
           "margin_at_largest_n": ys[-1] - BAR,
           "margin_over_noise_floor": abs(ys[-1] - BAR) / FLOOR if FLOOR else None,
           "src": [s for _, p in have for s in p["src"]]}
    if len(ys) >= 3:
        d2 = ys[0] - 2 * ys[1] + ys[2]
        d1, dd2 = ys[0] - ys[1], ys[1] - ys[2]
        r = dd2 / d1 if d1 > 0 else None
        asym = ys[2] - dd2 * r / (1 - r) if r is not None and 0 < r < 1 else None
        rec.update({"second_difference_log2n": d2, "convex_in_log2n": d2 > 0,
                    "decrement_ratio_per_2_octaves": r,
                    "geometric_continuation_asymptote": asym,
                    "geometric_continuation_crosses_bar": (asym < BAR) if asym is not None else None,
                    "asymptote_gap_to_bar": (asym - BAR) if asym is not None else None})
    else:
        rec["caveat"] = ("two points only: a straight line with no curvature check; "
                         f"n={missing} NOT MEASURED at this rung")
    emit(rec)
    return rec


primary = {t: do_rung(t, 150) for t in (2, 8, 32)}
for t in (2, 8, 32):
    for s in (600, 2400, 9600):
        do_rung(t, s)

# sensitivity: seed 0 only, the one seed present at every n and every rung
sens = {t: do_rung(t, 150, seeds={0}, tag="seed0_only") for t in (2, 8, 32)}

# ---- 4. steps axis: sign of dNRMSE/d(steps) at every cell with >= 2 rungs ----
for t in (2, 8, 32):
    for n in NS:
        pts = [(s, point(t, n, s)) for s in STEPS]
        have = [(s, p) for s, p in pts if p is not None]
        missing = [s for s, p in pts if p is None]
        if len(have) < 2:
            emit({"t": "steps_axis", "t_star": t, "n_train": n, "status": "not_measured",
                  "measured_steps": [s for s, _ in have], "missing_steps": missing,
                  "note": "fewer than two step rungs measured; no slope computable"})
            continue
        pairs = [{"steps_from": s0, "steps_to": s1, "nrmse_from": p0["nrmse"],
                  "nrmse_to": p1["nrmse"], "delta": p1["nrmse"] - p0["nrmse"],
                  "sign": "+" if p1["nrmse"] > p0["nrmse"] else "-",
                  "over_noise_floor": abs(p1["nrmse"] - p0["nrmse"]) / FLOOR if FLOOR else None,
                  "src": p0["src"] + p1["src"]}
                 for (s0, p0), (s1, p1) in zip(have, have[1:])]
        signs = {p["sign"] for p in pairs}
        emit({"t": "steps_axis", "t_star": t, "n_train": n, "status": "measured",
              "measured_steps": [s for s, _ in have], "missing_steps": missing, "pairs": pairs,
              "sign_of_dNRMSE_dsteps": "+" if signs == {"+"} else "mixed",
              "interpretation": "more steps -> higher NRMSE -> worse" if signs == {"+"} else "mixed",
              "n_seeds_per_rung": [p["n_seeds"] for _, p in have]})

# ---- seed inventory at the largest n, which was still filling in at read time ----
for t in (2, 8, 32):
    rs = rows(t, 32768, 150)
    dd = dedup(rs)
    emit({"t": "seed_inventory", "t_star": t, "n_train": 32768, "steps": 150,
          "journal_rows": len(rs), "rows_after_dedup": len(dd),
          "distinct_seeds": sorted({r["seed"] for r in dd}),
          "eval_nrmse_by_seed": {str(r["seed"]): r["eval_nrmse"] for r in dd},
          "spread": (max(r["eval_nrmse"] for r in dd) - min(r["eval_nrmse"] for r in dd)),
          "secs_per_row": [r["secs"] for r in rs], "src": [r["_src"] for r in rs],
          "note": "duplicate journal rows are bit-identical in eval_nrmse/boot_lo/boot_hi and "
                  "differ only in secs: one run journaled twice, not replicates. No interval "
                  "is computed over duplicate rows. Seed count here was rising at read time."})

# ---- self-check on the arithmetic that carries the conclusions ----
assert all(r["is_identity"] for r in OUT if r["t"] == "isotonic"), "isotonic changed the data"
_c = {(r["t_star"], r.get("variant", "")): r
      for r in OUT if r["t"] == "crossing" and r.get("steps") == 150}
assert _c[(2, "")]["status"] == "below_grid"
assert _c[(8, "")]["status"] == "interpolated" and 8192 < _c[(8, "")]["n_star"] < 32768
assert _c[(32, "")]["status"] == "extrapolated" and _c[(32, "")]["n_star"] > 32768
# a curve flat then dropping must cross where the drop is, not at the segment start
assert abs(cross([11.0, 13.0], [1.5, 0.5])["log2_n_star"] - 12.0) < 1e-12
# a rung with one measured n must not produce a number
assert _c.get((2, ""))["measured_n"] == NS
assert any(r["t"] == "crossing" and r["status"] == "not_measured" for r in OUT)

if __name__ == "__main__":
    out = ROOT / "results/r10_it10_frontier.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in OUT) + "\n")
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.values()}
    moved = [k for k, v in SNAPSHOT.items() if v["sha256"] != after[k]]
    print(f"wrote {out} ({len(OUT)} records) using {ISO_LIB}")
    print(f"noise floor (cross-thread, same seed): {FLOOR:.6e}")
    print("journals that changed WHILE this ran: " + (", ".join(moved) if moved else "none"))
    for r in OUT:
        if r["t"] in ("crossing", "monotonicity", "isotonic", "steps_axis", "seed_inventory",
                      "noise_floor"):
            print(json.dumps(r)[:300])
