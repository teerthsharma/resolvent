"""design4x5 -- four arms x five split seeds, paired by common random numbers.

REUSED, not reimplemented (same contract as tests/foreman/q2/q2_certificate.py,
which this file imports FROM rather than copies):
  - tests/foreman/eval/r3_eval.py (RE): DocByteBatches, _corpus_text,
    train_with_eval, repetition_factor, assert_head_dim.
  - tests/chase/gate/r1_gate.py (R): build_repaired, FORMS["hard_concrete"],
    GATE_INIT_OFF, ZETA, GAMMA.
  - tests/foreman/q2/q2_certificate.py (Q2): softmax_forward (arm (a)'s
    genuine-softmax monkeypatch of CEQAttention.forward) and the
    HIDDEN/LAYERS/HEADS/SEQ/BATCH/VOCAB/LR/EVAL_BATCHES shape constants, so
    this design sits at the SAME certificate shape Q2 already ran at.
  - crn_build.py (this dir): doc_order()/digest_order(), so this file's CRN
    digest is computed the identical way build_crn() computed the one on
    disk -- not a second implementation that could drift.

FOUR ARMS:
  (f)  phase free    -- Q2's arm (f) unmodified: hard-concrete magnitude,
       theta_head trainable, repaired init, operator="smprime".
  (f0) phase FROZEN  -- byte-identical to (f) except CEQAttention._smprime
       is monkeypatched so the theta_head OUTPUT is multiplied by 0 before
       it reaches smprime_readout. theta_head's Linear is NOT deleted (its
       parameters still exist, still get a gradient of exactly zero since
       nothing downstream of the zeroed output depends on them) -- so
       numel() matches (f) to the parameter. Verified below, not assumed.
  (a)  SDPA softmax twin -- Q2's arm (a) unmodified.
  (a2) SDPA twin + per-head sigmoid output MASS gate -- (a)'s build plus one
       extra nn.Linear(d, n_heads) per layer (o_gate_head), forward patched
       to multiply the SDPA output by sigmoid(o_gate_head(x)) per head
       BEFORE o_proj. The attention weights themselves (softmax(QK^T/sqrt(d)))
       are never touched -- verified bitwise identical to (a) with the gate
       pinned at 1, below, before this arm is ever scored on real data.

CRN: crn_order.json (built + pairing-proved by crn_build.py, run before this
file and NOT re-derived here) fixes one document order per split_seed in
{0,1,2,3,4}. Model seed is TIED to split_seed (seed == split_seed) so all
four arms draw the identical document split, the identical train-batch
sequence (gen=manual_seed(seed+1), inside r3_eval.train_with_eval, untouched
by this file) and the identical eval subsample (eval_gen=manual_seed(20260921
+split_seed)) at a given split_seed. Every run record below carries the
16-hex-char CRN digest for its split_seed, read out of crn_order.json, not
recomputed -- a mismatch between the two is the pre-registered VOID kill.

(a2) GATE: does not run until Miller 2023 (softmax-off-by-one), Xiao et al.
2023 (attention sinks) and Qwen 2025 (gated attention) are fetched onto the
page. This file does not fetch citations (no network tool available to it)
-- so per the pre-registered instruction, (a2) is BUILT and INVARIANT-CHECKED
below but HELD from the cell loop, and cells run for (f), (f0), (a) only.

Command: python design4x5.py [--build] [--run] [--n_split_seeds N]
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
CRN_PATH = os.path.join(SCRATCH, "crn_order.json")
RESULTS_PATH = os.path.join(SCRATCH, "design4x5_results.jsonl")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests", "foreman", "eval"))
sys.path.insert(0, os.path.join(REPO, "tests", "chase", "gate"))
sys.path.insert(0, os.path.join(REPO, "tests", "foreman", "q2"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import r3_eval as RE  # noqa: E402 -- REUSED module, not edited
import r1_gate as R  # noqa: E402 -- REUSED module, not edited
import ceq.arm_smprime as arm_smprime  # noqa: E402
from ceq.hf.modeling_ceq import CEQAttention  # noqa: E402
from q2_certificate import softmax_forward  # noqa: E402 -- REUSED, arm (a)'s forward

_ORIG_BUILD = RE.build
_ORIG_MAGNITUDE = arm_smprime.magnitude
_ORIG_ATTN_FORWARD = CEQAttention.forward
_ORIG_SMPRIME = CEQAttention._smprime

HIDDEN, LAYERS, HEADS, SEQ, BATCH = 128, 3, 8, 512, 8
VOCAB = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 3e-4
EVAL_BATCHES = 8
OUT_DIR_TMPL = os.path.join(SCRATCH, "d45_ckpt_{arm}_ss{ss}")
ARMS = ["f", "f0", "a", "a2"]
SPLIT_SEEDS = [0, 1, 2, 3, 4]
A2_CITATIONS_REQUIRED = ["Miller 2023 (softmax-off-by-one)",
                         "Xiao et al. 2023 (attention sinks)",
                         "Qwen 2025 (gated attention)"]
PRIOR_ART_DOC = os.path.join(REPO, "docs", "PRIOR_ART_MASS_GATE.md")


def _a2_citations_on_page():
    """Checks the ACTUAL page on disk rather than trusting a board claim --
    a sibling lane (Wilson, per house-events.jsonl 02:47:24) wrote
    docs/PRIOR_ART_MASS_GATE.md with verbatim quotes from all three sources
    before this file re-checked. Read fresh every call so a stale import
    cannot report a page that has since changed."""
    if not os.path.exists(PRIOR_ART_DOC):
        return []
    with io.open(PRIOR_ART_DOC, encoding="utf-8") as fh:
        text = fh.read()
    markers = {"Miller 2023 (softmax-off-by-one)": "Miller" in text and "Off By One" in text,
               "Xiao et al. 2023 (attention sinks)": "Xiao" in text and "Attention Sinks" in text,
               "Qwen 2025 (gated attention)": "Qwen" in text and "Gated Attention" in text}
    return [name for name, present in markers.items() if present]


A2_CITATIONS_ON_PAGE = _a2_citations_on_page()


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Foreman",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", row, flush=True)


def result(**kw):
    with io.open(RESULTS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(kw, default=str) + "\n")


def already_landed():
    """(arm, split_seed) pairs already in design4x5_results.jsonl -- so a
    re-invocation of this file (e.g. a later Foreman pass, same protocol)
    RESUMES the 20-cell grid instead of re-running and duplicating cells
    that already landed a real GPU-hour."""
    done = set()
    if not os.path.exists(RESULTS_PATH):
        return done
    with io.open(RESULTS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("stage") == "cell":
                done.add((row["arm"], row["split_seed"]))
    return done


def load_crn():
    if not os.path.exists(CRN_PATH):
        raise RuntimeError("crn_order.json missing -- run crn_build.py first "
                            "(step 1: build+prove the CRN file BEFORE any cell)")
    with io.open(CRN_PATH, encoding="utf-8") as fh:
        crn = json.load(fh)
    return crn


def crn_digest(crn, split_seed):
    return crn["orders"][str(split_seed)]["digest"]


# ------------------------------------------------------------- (f0) wiring

def smprime_phase_frozen(self, q, k, v, x):
    """Byte-identical to CEQAttention._smprime (ceq/hf/modeling_ceq.py) except
    `th` is multiplied by 0 before smprime_readout. theta_head still runs
    (its Linear is still called, still holds parameters, still receives x) --
    only its OUTPUT is zeroed, so numel() is unchanged and the grad reaching
    theta_head's weights is exactly zero (frozen), not undefined."""
    from ceq.arm_smprime import readout as smprime_readout
    u = self.m_head(x).squeeze(-1).unsqueeze(-2)
    th = self.theta_head(x).squeeze(-1).unsqueeze(-2)
    th = th * 0.0  # PHASE FROZEN AT THETA IDENTICALLY 0 -- output only, params intact
    return smprime_readout(q, k, v, u, th, beta=self.beta, qk=self.qk, g=self.g).real


# ------------------------------------------------------------- (a2) wiring

def _attach_o_gate_heads(model):
    """Post-build step for arm (a2): one nn.Linear(d, n_heads) per attention
    layer, attached as a submodule (so .to(device) moves it and
    model.parameters() -- hence the optimizer -- sees it) BEFORE the caller's
    .to(device) runs. Not present on arm (a); this IS the extra mass-channel
    arm (a2) adds over (a), and its numel() is reported, not hidden."""
    d = HIDDEN
    for layer in model.model.layers:
        layer.self_attn.o_gate_head = nn.Linear(d, HEADS)
    return model


def softmax_gated_forward(self, x, attention_mask=None):
    """(a)'s softmax_forward plus a per-head sigmoid output gate applied to
    the SDPA output BEFORE o_proj. The attention weights softmax(QK^T/sqrt(d))
    are computed by the SAME F.scaled_dot_product_attention call as (a) and
    are never touched by the gate -- verified bitwise against (a) at gate=1
    in verify_a2_gate_invariant() below, before this arm is ever trained."""
    if attention_mask is not None:
        raise NotImplementedError("a2 twin: no padding-mask path needed")
    b, s, d = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)

    def shape(t):
        return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    o = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), is_causal=True)
    gate = torch.sigmoid(self.o_gate_head(x))               # [B, S, H]
    gate = gate.transpose(1, 2).unsqueeze(-1)                # [B, H, S, 1]
    o = o * gate
    return self.o_proj(o.transpose(1, 2).reshape(b, s, d))


def softmax_gated_forward_gate_pinned_1(self, x, attention_mask=None):
    """Same as softmax_gated_forward with the gate hard-pinned to the
    constant 1.0 tensor instead of sigmoid(o_gate_head(x)) -- used ONLY by
    verify_a2_gate_invariant() to prove the attention pattern is untouched."""
    if attention_mask is not None:
        raise NotImplementedError
    b, s, d = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)

    def shape(t):
        return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    o = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), is_causal=True)
    gate = torch.ones(b, self.n_heads, s, 1)
    o = o * gate
    return self.o_proj(o.transpose(1, 2).reshape(b, s, d))


def verify_a2_gate_invariant():
    """BUILD-TIME PROOF, not an assumption: arm (a2)'s output with its gate
    pinned at 1 is BITWISE IDENTICAL to arm (a)'s output on the same input,
    same weights. x*1.0 is exact in IEEE754, so this is not a tolerance
    check -- torch.equal, zero mismatched elements or the design does not
    proceed to score (a2)."""
    torch.manual_seed(12345)
    m = _ORIG_BUILD(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                    seq=SEQ, vocab_size=VOCAB, operator="sgate")
    _attach_o_gate_heads(m)
    x = torch.randn(2, SEQ, HIDDEN)
    attn = m.model.layers[0].self_attn
    with torch.no_grad():
        out_a = softmax_forward(attn, x)
        out_a2_pinned = softmax_gated_forward_gate_pinned_1(attn, x)
    ok = torch.equal(out_a, out_a2_pinned)
    row = dict(check="a2_gate_pinned_1_equals_arm_a", bitwise_equal=ok,
               max_abs_diff=float((out_a - out_a2_pinned).abs().max()))
    print("[VERIFY]", row, flush=True)
    board("a2_gate_invariant", **row)
    if not ok:
        raise RuntimeError("a2 attention-pattern invariant FAILED: gate at 1 "
                            "changed the output -- (a2) is not a pure mass "
                            "channel and must not be scored")
    return ok


# ------------------------------------------------------------- param count

def measure_params():
    """numel() for all four arms, printed BEFORE any arm is scored (L-REFLECTOR:
    fixed structure printed before any arm is scored)."""
    torch.manual_seed(0)
    m_a = _ORIG_BUILD(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                      seq=SEQ, vocab_size=VOCAB, operator="sgate")
    n_a = sum(p.numel() for p in m_a.parameters())
    del m_a

    torch.manual_seed(0)
    m_a2 = _ORIG_BUILD(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                       seq=SEQ, vocab_size=VOCAB, operator="sgate")
    _attach_o_gate_heads(m_a2)
    n_a2 = sum(p.numel() for p in m_a2.parameters())
    del m_a2

    torch.manual_seed(0)
    m_f = R.build_repaired(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                           seq=SEQ, vocab_size=VOCAB, operator="smprime")
    n_f = sum(p.numel() for p in m_f.parameters())
    del m_f

    torch.manual_seed(0)
    m_f0 = R.build_repaired(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                            seq=SEQ, vocab_size=VOCAB, operator="smprime")
    n_f0 = sum(p.numel() for p in m_f0.parameters())
    del m_f0

    row = dict(n_params_f=n_f, n_params_f0=n_f0, n_params_a=n_a, n_params_a2=n_a2,
               f_eq_f0="MATCH TO THE PARAMETER" if n_f == n_f0 else
                       "MISMATCH: {} vs {}".format(n_f, n_f0),
               a2_minus_a=n_a2 - n_a,
               a2_extra_is_o_gate_heads="{} params == {} layers x nn.Linear({},{}) "
                                        "incl. bias".format(
                                            n_a2 - n_a, LAYERS, HIDDEN, HEADS))
    print("[PARAMS]", json.dumps(row, indent=2), flush=True)
    board("params_measured", **row)
    if n_f != n_f0:
        raise RuntimeError("(f0) does not match (f) to the parameter -- design "
                            "requires reporting this and it is being reported, "
                            "not silently patched over")
    return n_f, n_f0, n_a, n_a2


# ------------------------------------------------------------- arm trainers

def train_arm(arm, seed, split_seed, steps, crn):
    dg = crn_digest(crn, split_seed)
    out_dir = OUT_DIR_TMPL.format(arm=arm, ss=split_seed)
    common = dict(out_dir=out_dir, steps=steps, batch=BATCH, seq=SEQ,
                  hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                  device=DEVICE, vocab_size=VOCAB, lr=LR, seed=seed,
                  split_seed=split_seed, eval_every=steps,
                  eval_batches=EVAL_BATCHES, log_every=0)
    if arm == "a":
        CEQAttention.forward = softmax_forward
        try:
            rec = RE.train_with_eval(operator="sgate", **common)
        finally:
            CEQAttention.forward = _ORIG_ATTN_FORWARD
    elif arm == "a2":
        CEQAttention.forward = softmax_gated_forward
        RE.build = lambda **kw: _attach_o_gate_heads(_ORIG_BUILD(**kw))
        try:
            rec = RE.train_with_eval(operator="sgate", **common)
        finally:
            CEQAttention.forward = _ORIG_ATTN_FORWARD
            RE.build = _ORIG_BUILD
    elif arm == "f":
        RE.build = R.build_repaired
        arm_smprime.magnitude = R.FORMS["hard_concrete"]
        try:
            rec = RE.train_with_eval(operator="smprime", **common)
        finally:
            RE.build = _ORIG_BUILD
            arm_smprime.magnitude = _ORIG_MAGNITUDE
    elif arm == "f0":
        RE.build = R.build_repaired
        arm_smprime.magnitude = R.FORMS["hard_concrete"]
        CEQAttention._smprime = smprime_phase_frozen
        try:
            rec = RE.train_with_eval(operator="smprime", **common)
        finally:
            RE.build = _ORIG_BUILD
            arm_smprime.magnitude = _ORIG_MAGNITUDE
            CEQAttention._smprime = _ORIG_SMPRIME
    else:
        raise ValueError(arm)
    rec["crn_digest"] = dg
    rec["arm"] = arm
    rec["split_seed"] = split_seed
    rec["seed"] = seed
    return rec


# ------------------------------------------------------------- GPU polling

def card_free(headroom_mib=1500, max_procs=0):
    """True iff nvidia-smi reports no compute process holding the card AND
    used memory is below headroom_mib. NEVER kills what it finds -- polls."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            timeout=10).decode(errors="ignore").strip()
        procs = [l for l in out.splitlines() if l.strip()]
        mem = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10).decode(errors="ignore").strip().splitlines()[0]
        used = int(mem.strip())
    except Exception as e:
        print("[POLL] nvidia-smi unavailable ({}), assuming not free".format(e), flush=True)
        return False
    free = (len(procs) <= max_procs) and (used <= headroom_mib)
    print("[POLL] compute_procs={} used_mib={} free={}".format(
        len(procs), used, free), flush=True)
    return free


def poll_until_free(timeout_s, interval_s=20):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if card_free():
            return True
        time.sleep(interval_s)
    return False


# ------------------------------------------------------------- driver

def print_fixed_structure(n_f, n_f0, n_a, n_a2, steps, crn):
    row = dict(
        law="L-REFLECTOR", table="design4x5: 4 arms x 5 split seeds, CRN-paired",
        arms=dict(
            f="phase free -- hard-concrete magnitude, theta_head trainable, "
              "operator=smprime, R.build_repaired init",
            f0="phase FROZEN at theta==0 -- same build as (f), "
               "CEQAttention._smprime monkeypatched so theta_head's output "
               "(not its parameters) is multiplied by 0",
            a="SDPA softmax twin -- operator=sgate skeleton, "
              "F.scaled_dot_product_attention(is_causal=True)",
            a2="SDPA twin + per-head sigmoid output gate -- (a) plus "
               "nn.Linear(hidden,n_heads) per layer, gate applied to the "
               "SDPA output before o_proj, attention pattern unchanged "
               "(bitwise-verified at gate=1, see a2_gate_invariant board event). "
               "HELD from the cell loop: citations unfetched (Miller 2023, "
               "Xiao et al. 2023, Qwen 2025) -- see gate policy below.",
        ),
        n_params=dict(f=n_f, f0=n_f0, a=n_a, a2=n_a2),
        f_f0_param_match=(n_f == n_f0),
        parameterization="hidden={} n_layers={} n_heads={} d_head={} seq={} "
                          "vocab={} batch={}".format(
                              HIDDEN, LAYERS, HEADS, HIDDEN // HEADS, SEQ, VOCAB, BATCH),
        steps=steps, lr=LR, eval_batches=EVAL_BATCHES,
        split_seeds=SPLIT_SEEDS,
        crn_digests={ss: crn_digest(crn, ss) for ss in SPLIT_SEEDS},
        a2_gate_policy=dict(
            required_citations=A2_CITATIONS_REQUIRED,
            on_page=A2_CITATIONS_ON_PAGE,
            held=(sorted(A2_CITATIONS_ON_PAGE) != sorted(A2_CITATIONS_REQUIRED)),
        ),
        dtype_path="float32, device={}".format(DEVICE),
        torch_build="torch {} cuda_available={}".format(
            torch.__version__, torch.cuda.is_available()),
    )
    print(json.dumps(row, indent=2), flush=True)
    board("design4x5_fixed_structure", **row)
    return row


def run_all(n_split_seeds, poll_timeout_s=0, poll_interval_s=20):
    crn = load_crn()
    n_f, n_f0, n_a, n_a2 = measure_params()
    verify_a2_gate_invariant()
    steps = max(1, round(20.0 * n_a / (BATCH * SEQ)))  # same Chinchilla budget as Q2
    fixed = print_fixed_structure(n_f, n_f0, n_a, n_a2, steps, crn)

    active_arms = ["f", "f0", "a"]
    a2_held = fixed["a2_gate_policy"]["held"]
    if a2_held:
        print("[GATE] (a2) HELD: citations unfetched. Running {} only.".format(
            active_arms), flush=True)
        board("a2_held", required=A2_CITATIONS_REQUIRED, on_page=A2_CITATIONS_ON_PAGE)
    else:
        active_arms = ARMS

    split_seeds = SPLIT_SEEDS[:n_split_seeds]
    all_cells = [(arm, ss) for ss in split_seeds for arm in active_arms]
    done = already_landed()
    cells = [c for c in all_cells if c not in done]
    print("[PLAN] {} cells total, {} already landed, {} to run: {}".format(
        len(all_cells), len(done), len(cells), cells), flush=True)
    board("design4x5_plan", n_cells=len(all_cells), already_landed=len(done),
          to_run=cells, a2_held=a2_held)

    landed = 0
    for arm, ss in cells:
        seed = ss  # TIED per the design
        if not card_free():
            print("[POLL] card busy, waiting up to {}s (interval {}s), "
                  "NOT queueing, NOT killing".format(poll_timeout_s, poll_interval_s),
                  flush=True)
            if not poll_until_free(poll_timeout_s, poll_interval_s):
                print("[HOLD] card still busy after {}s -- cell arm={} ss={} "
                      "NOT run this pass. {}/{} cells landed.".format(
                          poll_timeout_s, arm, ss, landed, len(cells)), flush=True)
                board("cell_held_card_busy", arm=arm, split_seed=ss,
                      landed_so_far=landed, total=len(cells))
                break
        t1 = time.time()
        rec = train_arm(arm, seed, ss, steps, crn)
        dt = time.time() - t1
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        row = dict(arm=arm, split_seed=ss, seed=seed, n_params=rec["n_params"],
                   final_eval_loss=rec["final_eval_loss"],
                   loss_last_train=rec["losses"][-1],
                   crn_digest=rec["crn_digest"], steps=steps, run_seconds=dt)
        result(stage="cell", **row)
        board("cell_done", **row)
        landed += 1
        print("[CELL] {}/{} arm={} ss={} final_eval_loss={:.4f} ({:.1f}s)".format(
            landed, len(cells), arm, ss, rec["final_eval_loss"], dt), flush=True)

    board("design4x5_pass_done", landed=landed, total=len(cells))
    return landed, len(cells)


if __name__ == "__main__":
    n_ss = 5
    if "--n_split_seeds" in sys.argv:
        n_ss = int(sys.argv[sys.argv.index("--n_split_seeds") + 1])
    landed, total = run_all(n_ss)
    print("design4x5: {}/{} cells landed this pass.".format(landed, total), flush=True)
