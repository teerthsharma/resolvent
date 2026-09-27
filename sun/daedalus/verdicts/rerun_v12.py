"""Daedalus V1/V2: re-run Phase J arms a, f, a_alibi, a2F at split seed 0 by IMPORTING
the original code (design4x5.train_arm, rpos.alibi_forward, arms_j.train_a2F) with every
output path redirected under sun/daedalus/results/.

python sun/daedalus/verdicts/rerun_v12.py            # train 4 arms, append rows, write verdicts_12.json
python sun/daedalus/verdicts/rerun_v12.py --summarize  # rebuild verdicts_12.json from existing rows only
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import time

REPO = r"C:\Users\seal\Desktop\New folder (32)"
RES = os.path.join(REPO, "sun", "daedalus", "results")
ROWS = os.path.join(RES, "verdicts_lm_rows.jsonl")
OUT = os.path.join(RES, "verdicts_12.json")
GRID = os.path.join(REPO, "tests", "foreman", "design4x5", "design4x5_results.jsonl")
RPOS = os.path.join(REPO, "tests", "foreman", "phase_j", "N2", "foreman", "rpos_results.jsonl")
RECORD = os.path.join(REPO, "tests", "foreman", "phase_j", "record.jsonl")
PROTECTED = [GRID, RPOS, RECORD, os.path.join(REPO, "house-events.jsonl"),
             os.path.join(REPO, "tests", "foreman", "design4x5", "crn_order.json")]
ARMS = ["a", "f", "a_alibi", "a2F"]
STEPS = 3538          # round(20 * 724608 / (8 * 512)), the grid's Chinchilla budget
SS = 0
FOX_ROW = {0: "R-FoX", 1: "R-FoX-s1", 2: "R-FoX-s2"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def gpu_used_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], timeout=10).decode()
    used, total = (int(v) for v in out.strip().splitlines()[0].split(","))
    return used, total


def wait_for_room(need_mib=3500, timeout_s=1800):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used, total = gpu_used_mib()
        print("[GPU] used={} total={} need={}".format(used, total, need_mib), flush=True)
        if total - used >= need_mib:
            return True
        time.sleep(30)
    return False


def load_modules():
    for p in (os.path.join(REPO, "tests", "foreman", "design4x5"),
              os.path.join(REPO, "tests", "foreman", "eval"),
              os.path.join(REPO, "tests", "foreman", "phase_j", "N2", "foreman"),
              os.path.join(REPO, "tests", "foreman", "phase_j"), REPO):
        sys.path.insert(0, p)
    import design4x5 as D
    import crn_build as CB
    import rpos as RP
    import arms_j as J
    import q2_certificate as Q2
    import r1_gate as R1
    # redirect every write path the originals know about
    for m in (D, Q2, R1):
        m.BOARD = os.path.join(RES, "house-events.redirected.jsonl")
        m.RESULTS_PATH = os.path.join(RES, m.__name__ + "_results.redirected.jsonl")
    D.CRN_PATH = os.path.join(REPO, "tests", "foreman", "design4x5", "crn_order.json")  # read only
    D.OUT_DIR_TMPL = os.path.join(RES, "ckpt_{arm}_ss{ss}")
    J.OUT_DIR_TMPL = os.path.join(RES, "ckpt_{arm}_ss{ss}")
    RP.ROWS = os.path.join(RES, "rpos_results.redirected.jsonl")
    return D, CB, RP, J


def train_one(arm, D, RP, J, crn):
    import torch
    if arm == "a_alibi":   # rpos.run(): rebind D.softmax_forward, call train_arm("a")
        D.softmax_forward = RP.alibi_forward
        D.OUT_DIR_TMPL = os.path.join(RES, "ckpt_a_alibi_ss{ss}")
        try:
            rec = D.train_arm("a", SS, SS, STEPS, crn)
        finally:
            D.softmax_forward = RP._SOFTMAX
            D.OUT_DIR_TMPL = os.path.join(RES, "ckpt_{arm}_ss{ss}")
    elif arm == "a2F":     # run_j.run_row() minus the quintile read (needs a lost scratch ckpt)
        rec = J.train_a2F(seed=SS, split_seed=SS, steps=STEPS)
    else:
        rec = D.train_arm(arm, SS, SS, STEPS, crn)
    torch.cuda.empty_cache()
    return rec


def run():
    before = {p: sha(p) for p in PROTECTED}
    D, CB, RP, J = load_modules()
    crn = D.load_crn()
    text = D.RE._corpus_text(None, 64 * 1024 * 1024)
    tr, va, _ = CB.doc_order(text, SS)
    dg = CB.digest_order(tr, va)
    for arm in ARMS:
        if not wait_for_room():
            print("[HELD] no GPU room for", arm, flush=True)
            break
        used0, _ = gpu_used_mib()
        t0 = time.time()
        rec = train_one(arm, D, RP, J, crn)
        row = dict(arm=arm, split_seed=SS, seed=SS, steps=STEPS, n_params=rec["n_params"],
                   final_eval_loss=rec["final_eval_loss"], loss_last_train=rec["losses"][-1],
                   train_seconds=rec["seconds"], wall_seconds=time.time() - t0,
                   peak_mb=rec["peak_bytes"] / 2 ** 20, gpu_used_mib_before=used0,
                   crn_digest_file=D.crn_digest(crn, SS), crn_digest_recomputed=dg,
                   crn_match=(dg == D.crn_digest(crn, SS)),
                   deterministic=False, device=rec["device"],
                   ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with io.open(ROWS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print("[CELL]", json.dumps(row), flush=True)
    after = {p: sha(p) for p in PROTECTED}
    return {os.path.relpath(p, REPO): before[p] == after[p] for p in PROTECTED}


def jl(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def stored():
    grid = {(r["arm"], r["split_seed"]): r["final_eval_loss"] for r in jl(GRID) if r.get("stage") == "cell"}
    rpos = {(r["arm"], r["split_seed"]): r["final_eval_loss"] for r in jl(RPOS)}
    fox = {r["row"]: r["measured"]["loss_arm"] for r in jl(RECORD) if r.get("row") in FOX_ROW.values()}
    out = {}
    for ss in (0, 1, 2):
        a, f, al, fx = grid[("a", ss)], grid[("f", ss)], rpos[("a_alibi", ss)], fox[FOX_ROW[ss]]
        rec = (a - fx) / (a - f)
        out[str(ss)] = dict(loss_a=a, loss_f=f, loss_alibi=al, loss_a2F=fx,
                            delta_alibi_minus_f=al - f, V1_holds=al < f,
                            recovery=rec, V2_holds=rec >= 0.8)
    return out


def summarize(extra=None):
    rows = {}
    if os.path.exists(ROWS):
        for r in jl(ROWS):
            if r["split_seed"] == SS:
                rows[r["arm"]] = r      # last row per arm wins
    st = stored()
    s0 = st[str(SS)]
    rerun = all(a in rows for a in ARMS)
    L = {a: rows[a]["final_eval_loss"] for a in rows}
    use = L if rerun else dict(a=s0["loss_a"], f=s0["loss_f"], a_alibi=s0["loss_alibi"], a2F=s0["loss_a2F"])
    stored_map = dict(a=s0["loss_a"], f=s0["loss_f"], a_alibi=s0["loss_alibi"], a2F=s0["loss_a2F"])
    absdiff = {a: abs(L[a] - stored_map[a]) for a in L}
    cmd = "python sun/daedalus/verdicts/rerun_v12.py"
    prov = [GRID, RPOS, RECORD, ROWS, os.path.join(REPO, "sun", "daedalus", "verdicts", "rerun_v12.py"),
            "tests/foreman/design4x5/design4x5.py:train_arm", "tests/foreman/phase_j/N2/foreman/rpos.py:alibi_forward",
            "tests/foreman/phase_j/arms_j.py:train_a2F", "tests/foreman/eval/r3_eval.py:train_with_eval"]
    method = "rerun" if rerun else "stored"
    d1 = use["a_alibi"] - use["f"]
    rec = (use["a"] - use["a2F"]) / (use["a"] - use["f"])
    out = {
        "V1_alibi_beats_family": dict(
            method=method, split_seed=SS, steps=STEPS,
            rerun=dict(loss_a=L.get("a"), loss_f=L.get("f"), loss_alibi=L.get("a_alibi")),
            stored=dict(loss_a=s0["loss_a"], loss_f=s0["loss_f"], loss_alibi=s0["loss_alibi"]),
            abs_diff_rerun_vs_stored={k: absdiff.get(k) for k in ("a", "f", "a_alibi")},
            delta_alibi_minus_f=d1, stored_delta_alibi_minus_f=s0["delta_alibi_minus_f"],
            bar="eval loss(a_alibi) < eval loss(f) at ss0",
            reproduced=d1 < 0, provenance=prov, command=cmd),
        "V2_fox_recovers_cwin": dict(
            method=method, split_seed=SS, steps=STEPS,
            rerun=dict(loss_a=L.get("a"), loss_f=L.get("f"), loss_a2F=L.get("a2F")),
            stored=dict(loss_a=s0["loss_a"], loss_f=s0["loss_f"], loss_a2F=s0["loss_a2F"]),
            abs_diff_rerun_vs_stored={k: absdiff.get(k) for k in ("a", "f", "a2F")},
            recovery=rec, stored_recovery=s0["recovery"],
            bar="(loss(a) - loss(a2F)) / (loss(a) - loss(f)) >= 0.8 at ss0",
            reproduced=rec >= 0.8, provenance=prov, command=cmd),
        "wall_clock_s": sum(r["wall_seconds"] for r in rows.values()) if rows else None,
        "per_arm_wall_s": {a: r["wall_seconds"] for a, r in rows.items()},
        "gpu_peak_mb": max(r["peak_mb"] for r in rows.values()) if rows else None,
        "per_arm_peak_mb": {a: r["peak_mb"] for a, r in rows.items()},
        "crn_match_ss0": all(r["crn_match"] for r in rows.values()) if rows else None,
        "stored_all_seeds": st,
        "notes": [
            "default (non-deterministic) SDPA mode, not bitwise; one run per arm",
            "a2F: arms_j.train_a2F called directly; run_j.run_row's quintile read skipped "
            "(its twin checkpoint lived in a deleted scratchpad and is not needed for the verdict)",
            "stored a2F loss is record.jsonl R-FoX* measured.loss_arm; stored a/f from the design4x5 grid; "
            "stored a_alibi from rpos_results.jsonl",
        ],
    }
    if extra:
        out.update(extra)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in ("wall_clock_s", "gpu_peak_mb")}, indent=2))
    print("V1", method, d1, d1 < 0, " V2", rec, rec >= 0.8)


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    extra = None
    if "--summarize" not in sys.argv:
        extra = dict(protected_files_unchanged=run())
    summarize(extra)
