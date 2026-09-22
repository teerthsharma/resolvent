import io
import json
import os
import sys
import time

SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
REPO = r"C:\Users\seal\Desktop\New folder (32)"
PHASE_J = os.path.join(SCRATCH, "phase_j")
sys.path.insert(0, SCRATCH)
sys.path.insert(0, REPO)

import design4x5 as D  # noqa: E402
import arms_j as J  # noqa: E402

RECORD = os.path.join(PHASE_J, "record.jsonl")
LOSS_A = {1: 1.292035, 2: 1.301034}
C_WIN = 0.232628
BAR = 0.5 * C_WIN
DEADLINE_H, DEADLINE_M = 6, 9


def append_record(row):
    with io.open(RECORD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def past_deadline():
    now = time.localtime()
    return (now.tm_hour, now.tm_min) >= (DEADLINE_H, DEADLINE_M)


def run_seed(ss):
    row_id = "R-FoX-s{}".format(ss)
    if past_deadline():
        append_record(dict(row=row_id, arms=["a2F", "a"], seeds=[ss], split_seed=ss,
                            machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
                            seconds_per_cell=None, tokens_seen=None, params_numel=None,
                            bar=dict(desc="C_forget >= 0.5*C_win(ss0)", threshold=BAR),
                            measured=None, verdict="SKIPPED: deadline", control=None,
                            quintile_profile=None, producer="run_fox_s12.py", output=None,
                            repro_class="SDPA tolerance", killed="deadline reached before start"))
        return None
    if not D.poll_until_free(1800):
        append_record(dict(row=row_id, arms=["a2F", "a"], seeds=[ss], split_seed=ss,
                            machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
                            seconds_per_cell=None, tokens_seen=None, params_numel=None,
                            bar=dict(desc="C_forget >= 0.5*C_win(ss0)", threshold=BAR),
                            measured=None, verdict="SKIPPED: card busy", control=None,
                            quintile_profile=None, producer="run_fox_s12.py", output=None,
                            repro_class="SDPA tolerance", killed="card never freed"))
        return None

    steps = round(20.0 * 724608 / (D.BATCH * D.SEQ))
    t0 = time.time()
    try:
        rec = J.train_a2F(seed=ss, split_seed=ss, steps=steps)
    except Exception as e:
        append_record(dict(row=row_id, arms=["a2F", "a"], seeds=[ss], split_seed=ss,
                            machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
                            seconds_per_cell=time.time() - t0, tokens_seen=None,
                            params_numel=None,
                            bar=dict(desc="C_forget >= 0.5*C_win(ss0)", threshold=BAR),
                            measured=None, verdict="CRASH: {}".format(repr(e)), control=None,
                            quintile_profile=None, producer="run_fox_s12.py", output=None,
                            repro_class="SDPA tolerance", killed="run crashed, recorded not hidden"))
        print("[{}] CRASH: {}".format(row_id, e), flush=True)
        return None

    wall = rec.get("seconds", time.time() - t0)
    loss_arm = rec["final_eval_loss"]
    c_measured = LOSS_A[ss] - loss_arm
    tokens_seen = steps * D.BATCH * D.SEQ
    verdict = "PASS" if c_measured >= BAR else "FAIL"
    append_record(dict(
        row=row_id, arms=["a2F", "a"], seeds=[ss], split_seed=ss,
        machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
        seconds_per_cell=wall, tokens_seen=tokens_seen,
        params_numel=dict(arm=rec["n_params"], extra=rec.get("extra_params"), a=724608),
        bar=dict(desc="C_forget >= 0.5*C_win(ss0)", threshold=BAR),
        measured=dict(loss_arm=loss_arm, C=c_measured),
        verdict=verdict, control=dict(loss_a_ss=LOSS_A[ss], C_win_ss0=C_WIN,
                                       C_win_5seed_mean=0.2470),
        quintile_profile=None, producer="run_fox_s12.py",
        output=rec["model_path"], repro_class="SDPA tolerance",
        killed=("FoX forgetting explains at least half of C_win" if verdict == "PASS"
                else "FoX forgetting does not reach half of C_win"),
    ))
    print("[{}] C={:.6f} verdict={}".format(row_id, c_measured, verdict), flush=True)
    return c_measured


if __name__ == "__main__":
    cs = []
    for ss in (1, 2):
        c = run_seed(ss)
        if c is not None:
            cs.append(c)
    if cs:
        mean_c = sum(cs) / len(cs)
        append_record(dict(
            row="R-FoX-summary", arms=["a2F", "a"], seeds=[0, 1, 2][:1 + len(cs)],
            split_seed=None, machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
            seconds_per_cell=None, tokens_seen=None, params_numel=None,
            bar=dict(desc="C_forget >= 0.5*C_win(ss0)", threshold=BAR),
            measured=dict(C_seeds_run=cs, C_seed0=0.23298124317789082,
                           mean_C_over_seeds_run=(0.23298124317789082 + sum(cs)) / (1 + len(cs))),
            verdict="INFO: summary row, see per-seed rows for PASS/FAIL",
            control=dict(C_win_ss0=C_WIN, C_win_5seed_mean=0.2470),
            quintile_profile=None, producer="run_fox_s12.py", output=None,
            repro_class="SDPA tolerance",
            killed="summary over the seeds actually run for (a2F)",
        ))
