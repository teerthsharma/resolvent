"""The Kaggle-run autopilot envelope, as CODE, not as a model's judgement.

During a long training run a small polling model reads the training journal
and picks one of three tiers of response. This module is the fixed envelope
that judgement runs inside: `decide_line(raw, history, ...)` maps one parsed
journal line, plus everything seen before it, to exactly one `Decision`. It
never calls a model, never reads a clock, and never branches on a journal
line's free text -- every decision comes from a fixed set of typed fields the
line either has or does not, checked against thresholds and a pre-registered
config the journal cannot alter about itself.

TOTAL BY CONSTRUCTION. `decide()`'s trigger chain ends in "nothing matched,
line is well-formed and healthy" -> tier 0 / action NONE. Anything that does
NOT parse into a well-formed line -- bad JSON, a missing field, a field of the
wrong type -- never reaches `decide()` at all: `decide_line` routes it
straight to Tier-3 HALT. There is no third outcome where an unrecognised
input is quietly treated as "continue".

WHY THE DECIDING-CELL / BRANCH REGISTRY LIVES IN `AutopilotConfig`, NOT IN THE
JOURNAL. A journal line is the thing under attack (G0.9/MARS): if "is this
cell a deciding cell" or "is this branch pre-registered" were fields the
journal itself could set, a crafted line could simply claim `deciding_cell:
false` on a deciding cell, or invent a `branch_reached` name and have it
treated as authorised. Both live in `AutopilotConfig`, supplied out of band by
whoever starts the run, and no code path here reads a journal line to decide
membership in either set.

WHY TAMPER DETECTION IS A REMEMBERED SEAL, NOT A SELF-CLAIMED ROOT FIELD. An
earlier design put a `root` field on every line, self-declared by the writer.
That is exactly as forgeable as any other journal content: a party rewriting
the file can recompute a self-consistent root for its rewritten prefix and
carry on. `scale/merkle.py` already solved this for whole-file audits with
`sealed_root`/`sealed_lines` supplied externally (never inferred from the
file); `decide`/`decide_line` take the same two arguments, threaded forward
poll to poll by the caller (see `replay`), so the check is "does the prefix I
am holding right now still hash to what I sealed last time", independent of
anything the current file contents claim.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Union

from scale.merkle import merkle_root

__all__ = [
    "AutopilotConfig", "PollEvent", "Malformed", "Decision",
    "parse_line", "decide", "decide_line", "replay",
    "format_poll_line", "decisions_row", "false_correction_rate",
    "render_halt_md", "write_halt", "merkle_root",
]

# ------------------------------------------------------------- thresholds
# Every number here is [MODULE]: taken verbatim from the envelope table in
# the round prompt. Six were interpretation calls where the table's English
# was ambiguous, called out in V17_G08_G09_AUTOPILOT.md -- V17K_RULINGS.md
# RULING 6 (6a-6f) has since adjudicated all six; this module implements the
# ruling, not the earlier reading. See V17_R6_ENVELOPE.md for the diff.

TIER1_QUOTA_FRAC = 0.15        # "quota within 15%" -> pause
TIER1_SESSION_REMAIN_MIN = 20.0  # "session <20 min" -- RULING 6b: the SOURCE
                                # is Kaggle's own reported remaining-time
                                # number (PollEvent.kaggle_session_remaining_min),
                                # never a value computed from session_end_t-t.
TIER1_NO_CKPT_MIN = 30.0       # "no ckpt in 30" [min]
TIER1_STALE_POLLS = 2          # RULING 6a: stale journal = no write for 2
                                # CONSECUTIVE POLLS -- a poll count, not a
                                # time gap. Supersedes the old [ASSUMED]
                                # TIER1_STALE_MIN=15min elapsed-time reading.
TIER2_AHAT_MAX = 1.5           # seed divergence â_max threshold
TIER2_LAMBDA_POLLS = 3         # consecutive polls with λ̂>0
TIER3_AHAT_PHASE = 1.0         # â_max>1 on a phase arm
TIER3_KCERT_FRAC = 0.5         # K-CERT δ/tol>50%
LR_DECAY = 0.5
LR_FLOOR_DIVISOR = 8.0
HALT_LAST_N_POLLS = 20

_T2_SUFFIX = re.compile(r"\.t2\.\d+$")


# ----------------------------------------------------------------- config

@dataclass(frozen=True)
class AutopilotConfig:
    """Everything the decision function knows ahead of time, out of band from
    any journal line. `deciding_cells` and `branch_diagnostics` are the two
    registries a journal line must never be able to talk its way into.

    RULING 6f: deciding-cell membership is an explicit id list, FROZEN at
    launch; it may shrink mid-flight (a cell retired from consideration) but
    must never grow. `deciding_cells_launch`, when supplied, is that frozen
    launch set -- `__post_init__` makes growth beyond it a raised error at
    construction time, before any decision could ever see the larger set.
    Leaving it `None` opts a config out of the check entirely (e.g. the
    single launch-time config that establishes the set has nothing to be
    checked against)."""
    deciding_cells: frozenset
    branch_diagnostics: dict          # {branch_id: diagnostic_name}
    cell_queue: tuple                 # pre-registered schedule order
    lr0: float                        # initial lr, for the Tier-2 floor
    deciding_cells_launch: Optional[frozenset] = None

    def __post_init__(self):
        if self.deciding_cells_launch is not None:
            grew = frozenset(self.deciding_cells) - frozenset(self.deciding_cells_launch)
            if grew:
                raise ValueError(
                    "deciding_cells cannot grow beyond the launch-frozen set "
                    "(RULING 6f); attempted to add: {}".format(sorted(grew)))


# ------------------------------------------------------------- journal line

@dataclass(frozen=True)
class PollEvent:
    """One structurally-validated training-journal line. `note` and `raw` are
    carried for logging/hashing only -- no function on the decision path
    reads `.note` (see tests/gate0/test_g09_mars.py, which checks this both
    empirically and via bytecode inspection)."""
    raw: str
    t: float
    cell: str
    step: int
    loss: Optional[float]
    loss_nonfinite: bool
    lambda_hat_max: Optional[float]
    a_hat_max: Optional[float]
    phase_arm: bool
    r2: Optional[float]
    micro_batch: Optional[int]
    grad_accum: Optional[int]
    oom: bool
    kernel_alive: bool
    quota_remaining_frac: Optional[float]
    queue_idle: bool
    kaggle_session_remaining_min: Optional[float]  # RULING 6b: Kaggle's own
                                # reported remaining-session minutes, read
                                # directly -- not session_end_t-t, computed.
    last_ckpt_t: Optional[float]
    last_good_ckpt: Optional[str]
    lr: Optional[float]
    design_change_request: bool
    branch_reached: Optional[str]
    kcert_delta_over_tol: Optional[float]
    mem_used_frac: Optional[float]
    note: str = ""


@dataclass(frozen=True)
class Malformed:
    """A journal line that failed schema validation. Carries `.raw` so the
    merkle prefix check still sees it -- the file grew by one line whether or
    not that line could be parsed."""
    raw: str
    reason: str


_BOOL = ("loss_nonfinite", "phase_arm", "oom", "kernel_alive", "queue_idle",
         "design_change_request")
_NUM_OR_NONE = ("loss", "lambda_hat_max", "a_hat_max", "r2",
                "quota_remaining_frac", "kaggle_session_remaining_min", "last_ckpt_t", "lr",
                "kcert_delta_over_tol", "mem_used_frac")
_INT_OR_NONE = ("micro_batch", "grad_accum")
_STR_OR_NONE = ("last_good_ckpt", "branch_reached")


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_line(raw: str) -> Union[PollEvent, Malformed]:
    """Strict schema parse. Any missing field, wrong type, or invalid JSON
    returns `Malformed` -- this is the trust boundary the whole envelope
    stands on, so nothing here is inferred or defaulted except `note`."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        return Malformed(raw=raw, reason="invalid-json:" + str(e))
    if not isinstance(obj, dict):
        return Malformed(raw=raw, reason="not-an-object")

    try:
        if "t" not in obj or not _is_num(obj["t"]):
            raise KeyError("t")
        if "cell" not in obj or not isinstance(obj["cell"], str) or not obj["cell"]:
            raise KeyError("cell")
        if "step" not in obj or not isinstance(obj["step"], int) or isinstance(obj["step"], bool):
            raise KeyError("step")
        for f in _BOOL:
            if f not in obj or not isinstance(obj[f], bool):
                raise KeyError(f)
        for f in _NUM_OR_NONE:
            if f not in obj or not (obj[f] is None or _is_num(obj[f])):
                raise KeyError(f)
        for f in _INT_OR_NONE:
            if f not in obj:
                raise KeyError(f)
            v = obj[f]
            if not (v is None or (isinstance(v, int) and not isinstance(v, bool))):
                raise TypeError(f)
        for f in _STR_OR_NONE:
            if f not in obj or not (obj[f] is None or isinstance(obj[f], str)):
                raise KeyError(f)
        note = obj.get("note", "")
        if not isinstance(note, str):
            raise TypeError("note")
    except (KeyError, TypeError) as e:
        return Malformed(raw=raw, reason="field:" + str(e))

    return PollEvent(
        raw=raw, t=float(obj["t"]), cell=obj["cell"], step=obj["step"],
        loss=obj["loss"], loss_nonfinite=obj["loss_nonfinite"],
        lambda_hat_max=obj["lambda_hat_max"], a_hat_max=obj["a_hat_max"],
        phase_arm=obj["phase_arm"], r2=obj["r2"],
        micro_batch=obj["micro_batch"], grad_accum=obj["grad_accum"],
        oom=obj["oom"], kernel_alive=obj["kernel_alive"],
        quota_remaining_frac=obj["quota_remaining_frac"], queue_idle=obj["queue_idle"],
        kaggle_session_remaining_min=obj["kaggle_session_remaining_min"], last_ckpt_t=obj["last_ckpt_t"],
        last_good_ckpt=obj["last_good_ckpt"], lr=obj["lr"],
        design_change_request=obj["design_change_request"],
        branch_reached=obj["branch_reached"],
        kcert_delta_over_tol=obj["kcert_delta_over_tol"],
        mem_used_frac=obj["mem_used_frac"], note=note,
    )


# --------------------------------------------------------------- decision

@dataclass(frozen=True)
class Decision:
    tier: int                      # 0 (NONE), 1, 2, or 3
    action: str
    trigger: str
    value: object
    cell_old: Optional[str]
    cell_new: Optional[str]
    expected_effect: str           # factual, mechanical -- no hypothesis/verdict
    current_root: str
    manifest: Optional[dict] = None


def _lineage_root(cell: str) -> str:
    prev = None
    while prev != cell:
        prev = cell
        cell = _T2_SUFFIX.sub("", cell)
    return cell


def _tier2_count_for_lineage(history, cell: str) -> int:
    root = _lineage_root(cell)
    return sum(1 for e, d in history
              if isinstance(e, PollEvent) and d.tier == 2 and _lineage_root(e.cell) == root)


def _mint_cell(cell: str, count: int) -> str:
    return "{}.t2.{}".format(_lineage_root(cell), count + 1)


def _stale_poll_count(history, event: PollEvent) -> int:
    """RULING 6a: stale journal = no write for 2 CONSECUTIVE POLLS. A poll is
    'no write' when its raw journal text is byte-identical to the poll
    immediately before it -- the journal did not grow between those two
    polls. Returns the length of the run of consecutive no-write polls
    ending at `event` (0 if the poll before it already differs, i.e. the
    journal DID grow last time)."""
    raws = [e.raw for e, _ in history] + [event.raw]
    count = 0
    i = len(raws) - 1
    while i > 0 and raws[i] == raws[i - 1]:
        count += 1
        i -= 1
    return count


def _oom_count_for_shape(history, cell: str, micro_batch) -> int:
    """RULING 6e: OOM is keyed per (cell, shape). 'Shape' is read as the
    reported `micro_batch` at the time of the OOM -- `grad_accum` moves in
    lockstep to hold the effective batch invariant (see `_match_tier1`), so
    `micro_batch` alone identifies the shape. Counts prior OOM polls at this
    exact (cell, shape), regardless of what tier each one resolved to."""
    return sum(1 for e, _ in history
              if isinstance(e, PollEvent) and e.cell == cell and e.oom
              and e.micro_batch == micro_batch)


def _next_in_queue(config: AutopilotConfig, cell: str) -> Optional[str]:
    q = config.cell_queue
    if cell in q:
        i = q.index(cell)
        if i + 1 < len(q):
            return q[i + 1]
    return None


def _lambda_three_poll(history, event: PollEvent) -> bool:
    if event.lambda_hat_max is None or event.lambda_hat_max <= 0:
        return False
    same_cell = [e for e, _ in history if isinstance(e, PollEvent) and e.cell == event.cell]
    last_two = same_cell[-(TIER2_LAMBDA_POLLS - 1):]
    if len(last_two) < TIER2_LAMBDA_POLLS - 1:
        return False
    return all(e.lambda_hat_max is not None and e.lambda_hat_max > 0 for e in last_two)


def _match_tier2(history, event: PollEvent, config: AutopilotConfig):
    """Returns (name, value) for the first Tier-2 rule that matches, else
    None. Order follows the table: NaN/inf, then seed divergence by â_max,
    then by λ̂, then a pre-registered branch diagnostic."""
    if event.loss_nonfinite:
        return ("loss_nonfinite", True)
    if event.a_hat_max is not None and event.a_hat_max > TIER2_AHAT_MAX:
        return ("a_hat_max", event.a_hat_max)
    if _lambda_three_poll(history, event):
        return ("lambda_hat_max_3polls", event.lambda_hat_max)
    if event.branch_reached and event.branch_reached in config.branch_diagnostics:
        return ("branch_reached", event.branch_reached)
    return None


def _tier2_decision(name, value, event: PollEvent, config: AutopilotConfig,
                    count: int, current_root: str) -> Decision:
    if name == "loss_nonfinite":
        new_cell = _mint_cell(event.cell, count)
        old_lr = event.lr
        new_lr = None
        if old_lr is not None:
            new_lr = max(old_lr * LR_DECAY, config.lr0 / LR_FLOOR_DIVISOR)
        manifest = dict(old_cell=event.cell, new_cell=new_cell, reason="loss_nonfinite",
                        rollback_to_ckpt=event.last_good_ckpt, lr_old=old_lr, lr_new=new_lr,
                        grad_clip_registered=True)
        effect = "rolled back to last good checkpoint; lr {}->{} (floor lr0/{:g}); grad-clip registered".format(
            old_lr, new_lr, LR_FLOOR_DIVISOR)
        return Decision(tier=2, action="rollback+lr-decay+grad-clip", trigger="loss_nonfinite",
                        value=value, cell_old=event.cell, cell_new=new_cell,
                        expected_effect=effect, current_root=current_root, manifest=manifest)
    if name in ("a_hat_max", "lambda_hat_max_3polls"):
        new_cell = _mint_cell(event.cell, count)
        manifest = dict(old_cell=event.cell, new_cell=new_cell, reason=name,
                        rollback_to_ckpt=event.last_good_ckpt, scope="seed")
        return Decision(tier=2, action="seed-scoped-rollback", trigger=name, value=value,
                        cell_old=event.cell, cell_new=new_cell,
                        expected_effect="seed-scoped rollback to last good checkpoint for this lineage",
                        current_root=current_root, manifest=manifest)
    # name == "branch_reached": pre-registered diagnostic, no cell/manifest churn.
    diag = config.branch_diagnostics[event.branch_reached]
    manifest = dict(old_cell=event.cell, new_cell=event.cell, reason="branch_reached",
                    diagnostic=diag)
    return Decision(tier=2, action="diagnostic:" + diag, trigger="branch_reached", value=value,
                    cell_old=event.cell, cell_new=event.cell,
                    expected_effect="named diagnostic run; no training-state change",
                    current_root=current_root, manifest=manifest)


def _match_tier1(history, event: PollEvent, now_t: float, config: AutopilotConfig):
    """Returns a raw tuple for the first Tier-1 rule that matches, else None;
    `decide()` wraps it into a `Decision` (it already has `current_root`).

    `now_t` is accepted but unused since RULING 6a moved staleness off a
    wall-clock gap onto a poll count (`_stale_poll_count`, which reads
    `history` instead); kept in the signature for API stability and because
    the house rule ("time is passed in, never read internally") still governs
    any Tier-1 rule added later that does need it.
    """
    if not event.kernel_alive:
        return ("resume", "kernel_alive", False, event.cell, event.cell,
                "process resumed from last checkpoint; steps continue from saved step count")
    if event.oom:
        mb, ga = event.micro_batch, event.grad_accum
        new_mb = max(1, mb // 2)
        ratio = mb // new_mb if new_mb else 1
        new_ga = ga * ratio
        assert new_mb * new_ga == mb * ga, "effective batch must be invariant"
        return ("halve-micro-batch", "oom", True, event.cell, event.cell,
                "micro-batch {}->{}, grad-accum {}->{}; effective batch unchanged".format(
                    mb, new_mb, ga, new_ga))
    if (event.kaggle_session_remaining_min is not None
            and event.kaggle_session_remaining_min < TIER1_SESSION_REMAIN_MIN
            and event.last_ckpt_t is not None and (event.t - event.last_ckpt_t) >= TIER1_NO_CKPT_MIN):
        nxt = _next_in_queue(config, event.cell)
        return ("force-ckpt-end-chunk", "session_remaining+no_ckpt",
                (event.kaggle_session_remaining_min, event.t - event.last_ckpt_t),
                event.cell, nxt, "checkpoint written; chunk ended; next chunk scheduled")
    stale = _stale_poll_count(history, event)
    if stale >= TIER1_STALE_POLLS:
        return ("probe-restart", "stale_journal", stale, event.cell, event.cell,
                "probe issued; restart triggered if the probe fails")
    if event.quota_remaining_frac is not None and event.quota_remaining_frac <= TIER1_QUOTA_FRAC:
        return ("pause", "quota_remaining_frac", event.quota_remaining_frac,
                event.cell, event.cell, "training paused; resumes when the quota window resets")
    if event.queue_idle:
        nxt = _next_in_queue(config, event.cell)
        return ("next-cell", "queue_idle", True, event.cell, nxt, "advanced to next queued cell")
    return None


def _halt(trigger: str, value, event: PollEvent, current_root: str) -> Decision:
    return Decision(tier=3, action="HALT", trigger=trigger, value=value,
                    cell_old=event.cell, cell_new=None,
                    expected_effect="cell stopped; HALT.md written with last {} polls; "
                                    "awaiting operator".format(HALT_LAST_N_POLLS),
                    current_root=current_root)


def decide(history: list, new_event: PollEvent, *, now_t: float, config: AutopilotConfig,
          sealed_root: Optional[str] = None, sealed_lines: Optional[int] = None) -> Decision:
    """The decision for `new_event`, given everything decided before it.

    `history` is `[(PollEvent|Malformed, Decision), ...]` in poll order, never
    mutated. `now_t` and `sealed_root`/`sealed_lines` are passed in by the
    caller -- no clock read, no state hidden inside this function -- so the
    same three arguments always produce the same `Decision`.
    """
    if sealed_root is None:
        sealed_root = merkle_root([])
    if sealed_lines is None:
        sealed_lines = len(history)

    prefix_raw = [e.raw for e, _ in history]
    current_root = merkle_root(prefix_raw + [new_event.raw])

    # ---- Tier-3 guards, checked before any correction is attempted. ----
    if merkle_root(prefix_raw[:sealed_lines]) != sealed_root:
        return _halt("root-fails-to-extend", sealed_root, new_event, current_root)

    if new_event.phase_arm and new_event.a_hat_max is not None and new_event.a_hat_max > TIER3_AHAT_PHASE:
        return _halt("a_hat_max-on-phase-arm", new_event.a_hat_max, new_event, current_root)

    if new_event.design_change_request and new_event.cell in config.deciding_cells:
        return _halt("deciding-cell-design-change", new_event.cell, new_event, current_root)

    if new_event.kcert_delta_over_tol is not None and new_event.kcert_delta_over_tol > TIER3_KCERT_FRAC:
        return _halt("kcert-delta-over-tol", new_event.kcert_delta_over_tol, new_event, current_root)

    if new_event.oom and _oom_count_for_shape(history, new_event.cell, new_event.micro_batch) >= 1:
        # RULING 6e: OOM keyed per (cell, shape) -- a second OOM at a shape
        # already recorded (typically: halving didn't change anything, e.g.
        # the micro-batch floor was already hit) escalates instead of
        # halving again.
        return _halt("oom-repeat-same-shape", new_event.micro_batch, new_event, current_root)

    tier2 = _match_tier2(history, new_event, config)
    if tier2 is not None:
        name, value = tier2
        if new_event.cell in config.deciding_cells:
            # HARD RULE: Tier-2 on a deciding cell is a KILL clause. HALT,
            # never act -- even though a correction rule matched.
            return _halt("deciding-cell-tier2-kill-clause:" + name, value, new_event, current_root)
        count = _tier2_count_for_lineage(history, new_event.cell)
        if count >= 1:
            return _halt("two-tier2-on-cell:" + name, value, new_event, current_root)
        return _tier2_decision(name, value, new_event, config, count, current_root)

    tier1 = _match_tier1(history, new_event, now_t, config)
    if tier1 is not None:
        action, trigger, value, cell_old, cell_new, effect = tier1
        return Decision(tier=1, action=action, trigger=trigger, value=value,
                        cell_old=cell_old, cell_new=cell_new, expected_effect=effect,
                        current_root=current_root)

    return Decision(tier=0, action="NONE", trigger="none", value=None,
                    cell_old=new_event.cell, cell_new=None,
                    expected_effect="no correction needed", current_root=current_root)


def decide_line(raw: str, history: list, *, now_t: float, config: AutopilotConfig,
                sealed_root: Optional[str] = None, sealed_lines: Optional[int] = None):
    """The total function: `raw` journal text in, `(event, Decision)` out.

    A line that fails `parse_line` never reaches `decide()` -- it is Tier-3
    by construction, not by any rule inside the trigger chain.
    """
    if sealed_root is None:
        sealed_root = merkle_root([e.raw for e, _ in history])
    if sealed_lines is None:
        sealed_lines = len(history)

    parsed = parse_line(raw)
    if isinstance(parsed, Malformed):
        current_root = merkle_root([e.raw for e, _ in history] + [raw])
        decision = Decision(tier=3, action="HALT", trigger="malformed-line:" + parsed.reason,
                            value=parsed.reason, cell_old=None, cell_new=None,
                            expected_effect="cell stopped; HALT.md written with last {} polls; "
                                            "awaiting operator".format(HALT_LAST_N_POLLS),
                            current_root=current_root)
        return parsed, decision

    decision = decide(history, parsed, now_t=now_t, config=config,
                      sealed_root=sealed_root, sealed_lines=sealed_lines)
    return parsed, decision


def replay(raw_lines: list, now_ts: list, config: AutopilotConfig) -> list:
    """Drive `decide_line` across a whole journal, folding history and the
    seal forward poll by poll. Returns `[(event, Decision), ...]`."""
    history = []
    sealed_root, sealed_lines = merkle_root([]), 0
    out = []
    for raw, now_t in zip(raw_lines, now_ts):
        event, decision = decide_line(raw, history, now_t=now_t, config=config,
                                      sealed_root=sealed_root, sealed_lines=sealed_lines)
        history.append((event, decision))
        sealed_root, sealed_lines = decision.current_root, len(history)
        out.append((event, decision))
    return out


# ------------------------------------------------------------ audit output

def false_correction_rate(decisions: list) -> float:
    """Corrections (Tier-2 or Tier-3) taken, divided by polls. Zero on a
    clean chunk is a claim, not an assumption -- the counter increments on any
    real Tier-2/3 decision regardless of how it got there."""
    if not decisions:
        return 0.0
    corrections = sum(1 for d in decisions if d.tier in (2, 3))
    return corrections / len(decisions)


def format_poll_line(event: PollEvent, decision: Decision, sentence: str, eta: str = "n/a") -> str:
    """`[AUTOPILOT t cell step loss λ̂max âmax R² mem ETA root action=...] sentence`"""
    if decision.tier == 0:
        action = "NONE"
    elif decision.tier == 3:
        action = "T3:HALT"
    else:
        action = "T{}:{}".format(decision.tier, decision.action)
        if decision.cell_new:
            action += "->{}".format(decision.cell_new)
    mem = "n/a" if event.mem_used_frac is None else "{:.0%}".format(event.mem_used_frac)
    loss_field = "nonfinite" if event.loss_nonfinite else event.loss
    body = ("[AUTOPILOT t={t} cell={cell} step={step} loss={loss} lhat={lhat} "
           "ahat={ahat} r2={r2} mem={mem} eta={eta} root={root} action={action}]").format(
        t=event.t, cell=event.cell, step=event.step, loss=loss_field,
        lhat=event.lambda_hat_max, ahat=event.a_hat_max, r2=event.r2,
        mem=mem, eta=eta, root=decision.current_root[:12], action=action)
    return body + " " + sentence


def decisions_row(timestamp: str, decision: Decision) -> str:
    """One `DECISIONS.md` row: timestamp | trigger+value | tier | action |
    expected effect | cell old->new. No hypotheses, no verdicts -- callers
    supply `expected_effect` only through `Decision`, which is always the
    factual, mechanical phrasing already fixed at the point of decision."""
    return "| {ts} | {trig}={val} | T{tier} | {action} | {effect} | {old}->{new} |".format(
        ts=timestamp, trig=decision.trigger, val=decision.value, tier=decision.tier,
        action=decision.action, effect=decision.expected_effect,
        old=decision.cell_old or "-", new=decision.cell_new or "-")


def render_halt_md(history: list, decision: Decision) -> str:
    """The `HALT.md` body for a Tier-3 stop: the last `HALT_LAST_N_POLLS`
    polls plus the halting decision. Pure string-building -- writing it to
    disk is the orchestration layer's job, not this module's."""
    tail = history[-HALT_LAST_N_POLLS:]
    lines = ["# HALT", "", "trigger: {} = {}".format(decision.trigger, decision.value),
             "cell: {}".format(decision.cell_old), "", "## last {} polls".format(len(tail))]
    for event, d in tail:
        raw = event.raw if hasattr(event, "raw") else "<unavailable>"
        lines.append("- T{} {} :: {}".format(d.tier, d.action, raw))
    return "\n".join(lines) + "\n"


def write_halt(path, history: list, decision: Decision) -> None:
    from pathlib import Path
    Path(path).write_text(render_halt_md(history, decision), encoding="utf-8")
