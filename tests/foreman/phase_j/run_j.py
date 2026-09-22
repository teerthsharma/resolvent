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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import design4x5 as D  # noqa: E402
import arms_j as J  # noqa: E402
from ceq.hf.modeling_ceq import CEQAttention  # noqa: E402

RECORD = os.path.join(PHASE_J, "record.jsonl")
DEADLINE_H, DEADLINE_M = 6, 9
LOSS_A = 1.275684
C_WIN = 0.232628
BAR = 0.5 * C_WIN
TWIN_CKPT = os.path.join(SCRATCH, "d45_ckpt_a_ss0_pair", "model.pt")


def append_record(row):
    with io.open(RECORD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def past_deadline():
    now = time.localtime()
    return (now.tm_hour, now.tm_min) >= (DEADLINE_H, DEADLINE_M)


def eval_batches():
    data = None
    import r3_eval as RE
    data = RE.DocByteBatches(RE._corpus_text(None, 64 * 1024 * 1024), D.VOCAB,
                             val_frac=0.1, split_seed=0)
    eg = torch.Generator().manual_seed(20260921 + 0)
    return [data.batch("val", D.BATCH, D.SEQ, eg, D.DEVICE)[0]
            for _ in range(D.EVAL_BATCHES)]


def per_token_nll(model, x, fwd=None):
    ctx = CEQAttention.forward
    if fwd is not None:
        CEQAttention.forward = fwd
    try:
        with torch.no_grad():
            logits = model(input_ids=x).logits
    finally:
        CEQAttention.forward = ctx
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                           x[:, 1:].reshape(-1), reduction="none"
                           ).view(x.shape[0], -1).double()


def quintile_profile(d, nll_a):
    n = d.numel()
    order = torch.argsort(nll_a)
    qe = [round(n * i / 5) for i in range(6)]
    abs_strata = [float(d[order[qe[i]:qe[i + 1]]].mean()) for i in range(5)]
    rel_strata = [float((d[order[qe[i]:qe[i + 1]]] / nll_a[order[qe[i]:qe[i + 1]]]).mean())
                  for i in range(5)]
    return dict(absolute=abs_strata, relative=rel_strata)


def run_row(row_id, arm_key, train_fn, bar_desc, do_poll=True):
    if past_deadline():
        append_record(dict(row=row_id, arms=[arm_key, "a"], seeds=[0], split_seed=0,
                            machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
                            seconds_per_cell=None, tokens_seen=None, params_numel=None,
                            bar=bar_desc, measured=None, verdict="SKIPPED: deadline",
                            control=None, quintile_profile=None,
                            producer="run_j.py", output=None,
                            repro_class="SDPA tolerance", killed="deadline reached before start"))
        print("[{}] SKIPPED: deadline".format(row_id), flush=True)
        return

    if do_poll and not D.poll_until_free(1800):
        append_record(dict(row=row_id, arms=[arm_key, "a"], seeds=[0], split_seed=0,
                            machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
                            seconds_per_cell=None, tokens_seen=None, params_numel=None,
                            bar=bar_desc, measured=None, verdict="SKIPPED: card busy",
                            control=None, quintile_profile=None,
                            producer="run_j.py", output=None,
                            repro_class="SDPA tolerance", killed="card never freed"))
        return

    steps = round(20.0 * 724608 / (D.BATCH * D.SEQ))  # same Chinchilla budget as grid arm (a)
    t0 = time.time()
    try:
        rec = train_fn(seed=0, split_seed=0, steps=steps)
    except Exception as e:
        append_record(dict(row=row_id, arms=[arm_key, "a"], seeds=[0], split_seed=0,
                            machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
                            seconds_per_cell=time.time() - t0, tokens_seen=None,
                            params_numel=None, bar=bar_desc, measured=None,
                            verdict="CRASH: {}".format(repr(e)), control=None,
                            quintile_profile=None, producer="run_j.py", output=None,
                            repro_class="SDPA tolerance", killed="run crashed, recorded not hidden"))
        print("[{}] CRASH: {}".format(row_id, e), flush=True)
        return
    wall = rec.get("seconds", time.time() - t0)
    loss_arm = rec["final_eval_loss"]
    c_measured = LOSS_A - loss_arm
    tokens_seen = steps * D.BATCH * D.SEQ

    model, fwd = J.build_for_eval(arm_key, split_seed=0)
    blob = torch.load(rec["model_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(D.DEVICE).eval()

    from q2_certificate import softmax_forward as _sm
    m_a = D.RE.build(operator="sgate", hidden_size=D.HIDDEN, n_layers=D.LAYERS,
                     n_heads=D.HEADS, seq=D.SEQ, vocab_size=D.VOCAB)
    blob_a = torch.load(TWIN_CKPT, map_location="cpu", weights_only=False)
    m_a.load_state_dict(blob_a["state_dict"], strict=True)
    m_a.to(D.DEVICE).eval()

    ds, nlls = [], []
    for x in eval_batches():
        nll_a = per_token_nll(m_a, x, fwd=_sm)
        nll_arm = per_token_nll(model, x, fwd=fwd)
        ds.append((nll_a - nll_arm).reshape(-1))
        nlls.append(nll_a.reshape(-1))
    d = torch.cat(ds).cpu()
    nll_a_cat = torch.cat(nlls).cpu()
    qprof = quintile_profile(d, nll_a_cat)

    verdict = "PASS" if c_measured >= bar_desc["threshold"] else "FAIL"
    append_record(dict(
        row=row_id, arms=[arm_key, "a"], seeds=[0], split_seed=0,
        machine="RTX 4060" if D.DEVICE == "cuda" else "CPU",
        seconds_per_cell=wall, tokens_seen=tokens_seen,
        params_numel=dict(arm=rec["n_params"], extra=rec.get("extra_params"), a=724608),
        bar=bar_desc, measured=dict(loss_arm=loss_arm, C=c_measured, mean_d_t=float(d.mean())),
        verdict=verdict, control=dict(C_win_ss0=C_WIN, C_win_5seed_mean=0.2470),
        quintile_profile=qprof, producer="run_j.py",
        output=rec["model_path"], repro_class="SDPA tolerance",
        killed=("adaptive calibration explains at least half of C_win"
                if arm_key == "a_T" else "FoX forgetting explains at least half of C_win")
        if verdict == "PASS" else
        ("adaptive calibration does not reach half of C_win"
         if arm_key == "a_T" else "FoX forgetting does not reach half of C_win"),
    ))
    print("[{}] {} loss={:.6f} C={:.6f} bar={:.4f} verdict={} wall={:.1f}s".format(
        row_id, arm_key, loss_arm, c_measured, bar_desc["threshold"], verdict, wall), flush=True)


if __name__ == "__main__":
    os.makedirs(PHASE_J, exist_ok=True)
    run_row("R-TEMP", "a_T", J.train_a_T,
            dict(desc="C_temp >= 0.5*C_win(ss0)", threshold=BAR))
    run_row("R-FoX", "a2F", J.train_a2F,
            dict(desc="C_forget >= 0.5*C_win(ss0)", threshold=BAR))
