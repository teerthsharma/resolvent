"""SATURN's binding of MARS attack #1 (R10 it.13). Training-free, one process.

Section 3 of `results/r10_it13_mars_attack1.md` verbatim, plus three things the
round requires and MARS's script does not carry: an error count that cannot read
as a passing draw, a JSONL record, and a FALSIFIABILITY ARM -- the pre-fix corpus
(`b[:, s-1]` left as a live N(0,1) driver instead of zeroed) whose label variance
genuinely IS `t*+1`, run through the same three lines that produce `delta_1`.
Without that arm a measured -1.000 cannot be told from a harness that subtracts 1.
"""
import json, math, pathlib, sys, time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.negation_scope import (CH_FLIP, M3_TASKS, equilibrium_oracle,  # noqa: E402
                                  equilibrium_hop_reading, nrmse)

K, C8 = 200, 0.248958
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "r10_it13_saturn_binding.jsonl"


def stat(a):
    m = sum(a) / len(a)
    sd = (sum((v - m) ** 2 for v in a) / (len(a) - 1)) ** 0.5
    return m, sd, 1.96 * sd / len(a) ** 0.5



def main() -> None:
    """The binding run. Under a `__main__` guard because this module WRITES.

    The first version had no guard at all: lines 28-106 ran at import, so
    `import scale.it13_mars_binding` performed 600 draws (~73 s) and then
    OVERWROTE results/r10_it13_saturn_binding.jsonl -- the evidence file for
    the it.13 verdict -- with a fresh run. Caught by
    tests/loop/test_no_module_writes_a_file_at_import.py, whose message cites
    the measured precedent: importing scale/replay_census.py truncated a
    37-byte file and wrote 251 bytes of census output into it.
    """
    rows = []
    t0 = time.time()
    for t in (2, 8, 32):
        bf = M3_TASKS["e3_t%d" % t][0]
        V, H, Sh, P, errs = [], [], [], [], []
        for sd in range(20000, 20000 + K):
            try:
                x, y, f, p = bf(4096, 64, 24, d_model=16, seed=sd)
                V.append(float(y.var(unbiased=False)))
                h = equilibrium_hop_reading(x, 1)
                H.append(nrmse(h, y))
                if t == 8:
                    Sh.append(nrmse(C8 * h, y))
                # falsifiability arm: undo `b[:, s-1] = 0`, the one line MARS names.
                xp = x.clone()
                g = torch.Generator(device="cpu").manual_seed(sd + 999331)
                xp[:, x.shape[1] - 1, CH_FLIP] = torch.randn(x.shape[0], generator=g).to(x.dtype)
                P.append(float(equilibrium_oracle(xp, f, p).var(unbiased=False)))
            except Exception as e:                                   # noqa: BLE001
                errs.append({"seed": sd, "err": "%s: %s" % (type(e).__name__, e)})
        n_ok = len(V)
        if n_ok != K or len(H) != K or len(P) != K:
            print("  !! t*=%d: %d/%d draws completed, %d errored" % (t, n_ok, K, len(errs)))
        mV, sV, hV = stat(V)
        mH, sH, hH = stat(H)
        mP, sP, hP = stat(P)
        cf = math.sqrt((t - 1) / t)
        d1, d1p = mV - (t + 1), mP - (t + 1)
        fires = (d1 < 0) and (d1 + hV < 0) and (abs(d1 + 1.000) < 0.05)
        print("t*=%2d  mean Var(y)=%.6f  delta_1=%+.6f  CI [%+.6f, %+.6f]"
              % (t, mV, d1, d1 - hV, d1 + hV))
        print("       1-hop NRMSE=%.6f  closed form=%.6f  delta_2=%+.6f  CI [%+.6f, %+.6f]"
              % (mH, cf, mH - cf, mH - cf - hH, mH - cf + hH))
        print("       FALSIFIABILITY (pre-fix corpus, Var(y) truly t*+1): mean Var=%.6f  "
              "delta_1=%+.6f  CI [%+.6f, %+.6f]" % (mP, d1p, d1p - hP, d1p + hP))
        print("       sign<0=%s  CI excludes 0=%s  |delta_1+1|=%.6f<0.05=%s  -> %s"
              % (d1 < 0, d1 + hV < 0, abs(d1 + 1.0), abs(d1 + 1.0) < 0.05,
                 "FIRES" if fires else "MISS"))
        row = dict(t="binding", agent="SATURN", t_star=t, K=K, draws_ok=n_ok,
                   draws_errored=len(errs), errors=errs, seeds=[20000, 20000 + K - 1],
                   shape=dict(n=4096, s=64, d=24, d_model=16),
                   mean_var_y=mV, sd_var_y=sV, delta_1=d1, delta_1_ci=[d1 - hV, d1 + hV],
                   clause_1_sign_negative=bool(d1 < 0), clause_2_ci_excludes_0=bool(d1 + hV < 0),
                   clause_3_abs_dev=abs(d1 + 1.0), clause_3_ok=bool(abs(d1 + 1.0) < 0.05),
                   verdict="FIRES" if fires else "MISS",
                   mean_1hop_nrmse=mH, closed_form_1hop=cf, delta_2=mH - cf,
                   delta_2_ci=[mH - cf - hH, mH - cf + hH],
                   control_a_within_0002=bool(abs(mH - cf) < 0.002 and abs(mH - cf) + hH < 0.002),
                   prefix_mean_var_y=mP, prefix_delta_1=d1p,
                   prefix_delta_1_ci=[d1p - hP, d1p + hP],
                   harness_can_report_zero=bool(abs(d1p) < 0.05 and d1p - hP < 0 < d1p + hP))
        if t == 8:
            mS, sS, _ = stat(Sh)
            print("       CONTROL B: pinned NRMSE=%.6f  across-draw sd=%.6f  "
                  "95%% draw half-width=%.6f  (crossing margin 0.027628)"
                  % (mS, sS, 1.96 * sS))
            row.update(control_b_pinned_nrmse=mS, control_b_sd=sS,
                       control_b_half_width=1.96 * sS, control_b_filed_sd=0.00129,
                       control_b_withdraw_if_sd_above=0.014,
                       control_b_concession_holds=bool(sS <= 0.014))
        rows.append(row)

    # the trap arithmetic, from the shipped journals
    JOURNAL = {2: 1.4012436552018552, 8: 0.7130855231384859, 32: 0.368704564751893}
    print("\nTRAP CHECK (flipper_tol=0.05, bar_verdict default):")
    for t, meas in JOURNAL.items():
        a, b = 2.0 / math.sqrt(t), 2.0 / math.sqrt(t + 1)
        print("  t*=%2d measured=%.10f  2/sqrt(t*)=%.10f dev=%.6f %s | "
              "2/sqrt(t*+1)=%.10f dev=%.6f %s"
              % (t, meas, a, abs(meas - a), "PASS" if abs(meas - a) <= 0.05 else "ABORT",
                 b, abs(meas - b), "PASS" if abs(meas - b) <= 0.05 else "ABORT"))
        rows.append(dict(t="trap", agent="SATURN", t_star=t, journal_flipper_dependence=meas,
                         two_over_sqrt_t=a, dev_t=abs(meas - a),
                         two_over_sqrt_t_plus_1=b, dev_t_plus_1=abs(meas - b),
                         flipper_tol=0.05, verdict_under_t=abs(meas - a) <= 0.05,
                         verdict_under_t_plus_1=abs(meas - b) <= 0.05))

    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print("\nwrote %s  (%.1fs)" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
