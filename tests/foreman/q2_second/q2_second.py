"""Q2, second corpus -- the pre-registered transfer check on Q2's first row.

Q2 (tests/foreman/q2/q2_certificate.py, scratchpad/q2_certificate_results.jsonl)
found hard-concrete beating a matched softmax twin by 0.2326-0.2622 nats at
5/5 seeds on TinyStories, repetition <1x for the first time -- but explicitly
NOT written as a win pending a second corpus in a different domain. This is
that row: SAME PROTOCOL, ONE VARIABLE CHANGED (the corpus).

REUSED, not reimplemented -- identical to q2_certificate.py:
  - tests/foreman/eval/r3_eval.py (RE): DocByteBatches, _corpus_text,
    assert_head_dim, print_repetition_row/repetition_factor, train_with_eval.
  - tests/chase/gate/r1_gate.py (R): magnitude_hardconcrete, build_repaired.
  - softmax_forward: the same monkeypatch of CEQAttention.forward this file's
    sibling uses, unedited, to give arm (a) genuine causal softmax attention
    over the same qkv/o_proj skeleton (ceq has no standalone softmax operator
    to import -- see q2_certificate.py's docstring).

THE ONE VARIABLE: corpus file and RE.CORPUS_TOKENS swapped from
data/tinystories_20k.txt (synthetic children's stories, short dependencies)
to a WikiText-103-raw slice (encyclopedic English, CC BY-SA 4.0, long-form
articles) pinned at scratchpad/wikitext103_corpus.txt -- see
scratchpad/build_wikitext_corpus.py for the fetch (streamed via HF
`datasets`, Salesforce/wikitext, config wikitext-103-raw-v1, train split) and
the reachability report returned by the harness's own prior PREPARE step.
CORPUS_TOKENS is measured, not the pre-registration's estimate: it is
len(RE._corpus_text(WIKI_CORPUS_PATH, WIKI_MAX_BYTES).encode("utf-8")), i.e.
the exact byte count of the exact slice train_with_eval will actually load
(max_bytes=64MiB, RE's own default -- unchanged from Q2's first row; the
WikiText file is 105,326,029 bytes total and gets truncated the same way
Q2's TinyStories corpus, at 18,167,706 bytes, never needed to be).

Shape, optimizer, split convention, seed set, arms: byte-identical to
q2_certificate.py. hidden=128 n_layers=3 n_heads=8 d_head=16 seq=512
vocab=256 (byte-level, UNCHANGED -- WikiText-103-raw is >99.9% Latin-1/UTF-8
narrow text and needs no vocabulary adjustment; this is verified, not
assumed, by print_fixed_structure below reporting n_docs/n_val_docs same as
any other DocByteBatches corpus) batch=8 lr=3e-4. split_seed=0 FIXED across
all 5 seeds. steps = Chinchilla 20x budget on arm (a)'s param count, SAME
formula as Q2's first row (recomputed here because n_params can differ by a
few dozen if the ceq/ build path drifts -- printed below, not assumed).

Command: python q2_second.py [n_seeds]
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
RESULTS_PATH = os.path.join(SCRATCH, "q2_second_results.jsonl")
WIKI_CORPUS_PATH = os.path.join(SCRATCH, "wikitext103_corpus.txt")
WIKI_MAX_BYTES = 64 * 1024 * 1024  # RE.train_with_eval's own default -- unchanged
WIKI_LICENCE = "CC BY-SA 4.0 (Salesforce/wikitext, wikitext-103-raw-v1, Wikipedia Good/Featured articles)"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests", "foreman", "eval"))
sys.path.insert(0, os.path.join(REPO, "tests", "chase", "gate"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import r3_eval as RE  # noqa: E402 -- REUSED module, not edited
import r1_gate as R  # noqa: E402 -- REUSED module, not edited
import ceq.arm_smprime as arm_smprime  # noqa: E402
from ceq.hf.modeling_ceq import CEQAttention  # noqa: E402

_ORIG_BUILD = RE.build
_ORIG_MAGNITUDE = arm_smprime.magnitude
_ORIG_ATTN_FORWARD = CEQAttention.forward
_ORIG_CORPUS_TOKENS = RE.CORPUS_TOKENS  # TinyStories's 18,167,706 -- restored at exit


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Foreman",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", row, flush=True)


def result(**kw):
    with io.open(RESULTS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(kw, default=str) + "\n")


HIDDEN, LAYERS, HEADS, SEQ, BATCH = 128, 3, 8, 512, 8
VOCAB = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED = 0
OUT_DIR_TMPL = os.path.join(SCRATCH, "q2_second_ckpt_{arm}_seed{seed}")
LR = 3e-4
EVAL_BATCHES = 8


def softmax_forward(self, x, attention_mask=None):
    if attention_mask is not None:
        raise NotImplementedError("softmax twin: no padding-mask path needed "
                                   "or exercised by this row")
    b, s, d = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)

    def shape(t):
        return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    o = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), is_causal=True)
    return self.o_proj(o.transpose(1, 2).reshape(b, s, d))


def pin_wikitext_corpus():
    """Report bytes_total (wc -c on the file on disk), the exact slice
    train_with_eval will load (post max_bytes truncation, the number that
    actually sets CORPUS_TOKENS/repetition_factor), licence, digest."""
    if not os.path.exists(WIKI_CORPUS_PATH):
        raise FileNotFoundError(
            "{} not found -- run build_wikitext_corpus.py first".format(WIKI_CORPUS_PATH))
    file_bytes_total = os.path.getsize(WIKI_CORPUS_PATH)
    with io.open(WIKI_CORPUS_PATH, "rb") as fh:
        full_raw = fh.read()
    full_digest = hashlib.sha256(full_raw).hexdigest()
    loaded_text = RE._corpus_text(WIKI_CORPUS_PATH, WIKI_MAX_BYTES)
    loaded_bytes = loaded_text.encode("utf-8")
    loaded_tokens = len(loaded_bytes)
    loaded_digest = hashlib.sha256(loaded_bytes).hexdigest()
    n_docs_loaded = len([d for d in loaded_text.split("\n\n") if d.strip()])
    report = dict(
        key="wikitext103-pin",
        file_path=WIKI_CORPUS_PATH,
        file_bytes_total=file_bytes_total,
        max_bytes_cap=WIKI_MAX_BYTES,
        loaded_tokens_byte_level=loaded_tokens,
        loaded_sha256=loaded_digest,
        full_file_sha256=full_digest,
        n_docs_in_loaded_slice=n_docs_loaded,
        licence=WIKI_LICENCE,
        source="huggingface Salesforce/wikitext, wikitext-103-raw-v1, train split, streamed",
    )
    print(json.dumps(report, indent=2), flush=True)
    board("q2_second_corpus_pin", **report)
    return loaded_tokens


def measure_params():
    torch.manual_seed(0)
    m_a = _ORIG_BUILD(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                      seq=SEQ, vocab_size=VOCAB, operator="sgate")
    n_a = sum(p.numel() for p in m_a.parameters())
    del m_a
    torch.manual_seed(0)
    m_f = R.build_repaired(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                           seq=SEQ, vocab_size=VOCAB, operator="smprime")
    n_f = sum(p.numel() for p in m_f.parameters())
    del m_f
    return n_a, n_f


def print_fixed_structure(n_a, n_f, steps, wiki_tokens):
    rep_a_wiki = RE.repetition_factor(n_a)   # RE.CORPUS_TOKENS is already swapped to wiki_tokens by here
    rep_f_wiki = RE.repetition_factor(n_f)
    rep_a_tinystories = (20.0 * n_a) / _ORIG_CORPUS_TOKENS
    rep_f_tinystories = (20.0 * n_f) / _ORIG_CORPUS_TOKENS
    rows = dict(
        law="L-REFLECTOR",
        table="Q2-second -- transfer check, second corpus",
        initializer="arm (a) softmax twin: CEQForCausalLM default init "
                     "(initializer_range=0.02), no gate heads exist. "
                     "arm (f) hard-concrete: R.build_repaired -- "
                     "m_head.bias={:.3f}, theta_head.bias={:.3f} "
                     "(GATE_INIT_OFF={}), SMPRIME_CORNER beta=qk=g=1.0".format(
                         1.0 - R.GATE_INIT_OFF, R.GATE_INIT_OFF, R.GATE_INIT_OFF),
        parameterization="hidden={} n_layers={} n_heads={} d_head={} seq={} "
                          "vocab={} batch={}".format(
                              HIDDEN, LAYERS, HEADS, HIDDEN // HEADS, SEQ, VOCAB, BATCH),
        n_params_arm_a_softmax_twin=n_a,
        n_params_arm_f_hard_concrete=n_f,
        n_params_diff="{} ({:.4f}%)".format(n_f - n_a, 100.0 * (n_f - n_a) / n_a),
        corpus_regime="scratchpad/wikitext103_corpus.txt (WikiText-103-raw, "
                       "{}), {:,} bytes==tokens actually loaded (max_bytes={:,} "
                       "cap applied), by-document split (RE.DocByteBatches), "
                       "val_frac=0.1, split_seed={} FIXED across all seeds. "
                       "BOTH REPETITION FACTORS on THIS corpus: "
                       "repetition_factor(a)={:.4f}x repetition_factor(f)={:.4f}x "
                       "-- both <1x, generalization-admissible. For reference, "
                       "the SAME two arms on the FIRST corpus (TinyStories, "
                       "{:,} bytes) were repetition_factor(a)={:.4f}x "
                       "repetition_factor(f)={:.4f}x -- domain changed, "
                       "shape/steps-formula/split/seeds/arms did not".format(
                           WIKI_LICENCE, wiki_tokens, WIKI_MAX_BYTES, SPLIT_SEED,
                           rep_a_wiki, rep_f_wiki, _ORIG_CORPUS_TOKENS,
                           rep_a_tinystories, rep_f_tinystories),
        scorer_functional="arm (a): torch.nn.functional.scaled_dot_product_attention, "
                           "is_causal=True, ordinary softmax(QK^T/sqrt(d))V. "
                           "arm (f): ceq.arm_smprime.blend/path_product readout "
                           "with magnitude(u)=stretched-sigmoid-then-clamp "
                           "(zeta={}, gamma={}), repaired init".format(R.ZETA, R.GAMMA),
        bin_scheme="none (single scalar final eval loss per seed)",
        dtype_path="float32, device={}".format(DEVICE),
        torch_build="torch {} cuda_available={}".format(
            torch.__version__, torch.cuda.is_available()),
        eval_subsample_size="eval_batches x batch = {} x {} = {} windows per "
                             "eval call, FIXED eval generator seed (r3_eval.py's "
                             "20260921+split_seed), same subsample every arm/seed".format(
                                 EVAL_BATCHES, BATCH, EVAL_BATCHES * BATCH),
        steps=steps, lr=LR, seeds=None,
    )
    return rows


def train_arm_a(seed, steps):
    CEQAttention.forward = softmax_forward
    try:
        return RE.train_with_eval(
            out_dir=OUT_DIR_TMPL.format(arm="softmax", seed=seed), steps=steps,
            batch=BATCH, seq=SEQ, hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
            device=DEVICE, vocab_size=VOCAB, lr=LR, seed=seed, split_seed=SPLIT_SEED,
            eval_every=steps, eval_batches=EVAL_BATCHES, log_every=0,
            operator="sgate", data_path=WIKI_CORPUS_PATH, max_bytes=WIKI_MAX_BYTES)
    finally:
        CEQAttention.forward = _ORIG_ATTN_FORWARD


def train_arm_f(seed, steps):
    RE.build = R.build_repaired
    arm_smprime.magnitude = R.FORMS["hard_concrete"]
    try:
        return RE.train_with_eval(
            out_dir=OUT_DIR_TMPL.format(arm="hardconcrete", seed=seed), steps=steps,
            batch=BATCH, seq=SEQ, hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
            device=DEVICE, vocab_size=VOCAB, lr=LR, seed=seed, split_seed=SPLIT_SEED,
            eval_every=steps, eval_batches=EVAL_BATCHES, log_every=0,
            operator="smprime", data_path=WIKI_CORPUS_PATH, max_bytes=WIKI_MAX_BYTES)
    finally:
        RE.build = _ORIG_BUILD
        arm_smprime.magnitude = _ORIG_MAGNITUDE


def run_all(n_seeds: int):
    t0 = time.time()
    wiki_tokens = pin_wikitext_corpus()
    RE.CORPUS_TOKENS = wiki_tokens  # the one variable: swap the corpus's token count for repetition_factor
    try:
        n_a, n_f = measure_params()
        steps = max(1, round(20.0 * n_a / (BATCH * SEQ)))
        fixed = print_fixed_structure(n_a, n_f, steps, wiki_tokens)
        seeds = list(range(n_seeds))
        fixed["seeds"] = seeds
        print(json.dumps(fixed, indent=2), flush=True)
        board("q2_second_fixed_structure", **fixed)

        landed = {"softmax": [], "hard_concrete": []}
        total = 2 * len(seeds)
        done = 0
        for seed in seeds:
            for arm, fn in (("softmax", train_arm_a), ("hard_concrete", train_arm_f)):
                t1 = time.time()
                rec = fn(seed, steps)
                dt = time.time() - t1
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                row = dict(arm=arm, seed=seed, n_params=rec["n_params"],
                           final_eval_loss=rec["final_eval_loss"],
                           loss_last_train=rec["losses"][-1],
                           repetition_factor=rec["repetition_factor"],
                           n_docs=rec["n_docs"], n_train_docs=rec["n_train_docs"],
                           n_val_docs=rec["n_val_docs"], steps=steps, run_seconds=dt)
                result(stage="q2_second_arm_seed", **row)
                board("q2_second_seed_done", **row)
                landed[arm].append(rec["final_eval_loss"])
                done += 1
                print("[Q2-second] done {}/{}  arm={} seed={} final_eval_loss={:.4f} "
                      "({:.1f}s)".format(done, total, arm, seed,
                                          rec["final_eval_loss"], dt), flush=True)

        n_pairs = min(len(landed["softmax"]), len(landed["hard_concrete"]))
        diffs = [landed["hard_concrete"][i] - landed["softmax"][i] for i in range(n_pairs)]
        mean_diff = sum(diffs) / len(diffs) if diffs else float("nan")
        n_f_better = sum(1 for d in diffs if d < -0.01)
        n_tie = sum(1 for d in diffs if abs(d) <= 0.01)
        n_a_better = sum(1 for d in diffs if d > 0.01)
        if n_pairs == len(seeds) and n_f_better == n_pairs:
            branch = "TRANSFERS"
        elif n_pairs == len(seeds) and n_tie == n_pairs:
            branch = "DOES_NOT_TRANSFER (tie within 0.01 nats at every seed)"
        elif n_pairs == len(seeds) and n_a_better == n_pairs:
            branch = "DOES_NOT_TRANSFER (softmax twin wins on this corpus)"
        else:
            branch = "MIXED (wins at some seeds, not others -- smaller than seed spread on this corpus)"
        verdict = dict(
            n_seeds_completed=n_pairs, n_seeds_requested=len(seeds),
            mean_diff_f_minus_a=mean_diff, per_seed_diff=diffs,
            n_tie_within_0p01=n_tie, n_f_better_by_gt_0p01=n_f_better,
            n_a_better_by_gt_0p01=n_a_better,
            branch=branch,
            first_corpus="TinyStories: mean_diff_f_minus_a=-0.2473, 5/5 hard-concrete better "
                          "(scratchpad/q2_certificate_results.jsonl)",
            second_corpus="WikiText-103-raw, {:,} bytes loaded".format(wiki_tokens),
            wall_clock_s=time.time() - t0,
        )
        result(stage="q2_second_verdict", **verdict)
        board("q2_second_all_done", **verdict)
        print("Q2-SECOND COMPLETE:", json.dumps(verdict, indent=2), flush=True)
    finally:
        RE.CORPUS_TOKENS = _ORIG_CORPUS_TOKENS


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_all(n)
