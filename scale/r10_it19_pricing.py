"""MERCURY, R10 it.19: Phase 1c (it.20-24) priced BEFORE it runs. RULE 4.

THE METHOD IS THE ONE BINDING SINCE PHASE 0 (`PHASE2_CONTRACT_V_MAIN_4.md:192`):
a count of 600-step run-equivalents read off the source that will execute them,
times a seconds-per-RE rate re-measured on the machine at the moment of spend.
The contract's own anchors are 20.58 s/RE uncontended and 48.89 s/RE at four
seats; it.8 re-measured 18.01 uncontended on THIS box and that is the rate
imported here by object identity from `scale.r10_it8_pricing`, not re-picked.

WHAT A CELL IS HERE, said plainly, because it.8's answer (36 cells over 9 NESTED
trainings, nesting worth 24.7%) does not transfer.

    A Phase 1c cell is (stage, arm, rung, seed).

    it.21  S1 ingestion + manifest hash   12 rungs x 8 seeds          =  96
    it.22  S2 graph + edge-recovery J     12 rungs x 8 seeds          =  96
    it.23  a trained arm                  12 rungs x 3 arms x 8 seeds = 288
    it.24  the verdict                    reads the table, spends 0   =   1
                                                                       ----
                                                                        481

    AND NOTHING NESTS. it.8's 24.7% came from ONE axis: a 9600-step run read at
    150/600/2400/9600. Phase 1c has no steps axis -- steps is pinned at 150 by
    it.8's own overfitting measurement (train NRMSE fell to 0.054 while eval rose
    past 2.238 at n=2048, `r10_it8_pricing.OVERFIT_AT_2048`). The seed axis
    cannot nest: eight trajectories is the entire content of N=8. The arm axis
    cannot nest: three arms are three optimisations. So 288 cells are 288
    trainings and the it.8 discount is unavailable here. Priced as such.

THE SHAPE FORK, UNDECLARED AND WORTH 1.9x. The it.23 table can mean two grids and
the round has declared neither, so both are priced rather than one chosen quietly:

    R1  per-rung models   -- 288 cells, each rung trained at s = its graph size
    R2  one pooled model  -- 24 cells, every rung padded to s = 1024

R2 has ONE TWELFTH the cells and costs 1.9x MORE, because padding pays the s^2
operator on the ten rungs that are not 1024 nodes wide. A smaller cell count is
not a cheaper grid.

THE ARM THAT DOES NOT EXIST. `m3_capability.ARMS` is
("softmax","pivot_signed","pivot_unsigned","windowed_signed") at line 66 and
`Arm.__init__` raises ValueError on anything else; `m3_quintuple.ALL_CELLS`
carries twin/settled/argmax and no S1 or S2 cell. So `S1+S2-twin` is not a cell
whose c_step was not measured -- it is a cell with no code. Its calibration
compute is 38.2 s (two steps over the twelve rungs at the measured per-row rate)
and that is not the constraint. The build is, and it is not priced here, because
a cell with no implementation must not read as a cheap cell.

MEASURED, NOT ASSUMED -- and the timing instrument is reported failing first.
Every c_step below was taken this iteration, threads pinned, 6 steps after 2
warmup, on a box whose free RAM moved 3,964 -> 6,478 MiB inside the three-minute
window. Reproducing it.8's own cell at its own pin (n=2048, 8 threads) read
0.204828 s/step against it.8's 0.11137: THE SAME WORK, 1.84x SLOWER. So no
absolute step time taken in this session is a price. What survives contention is
a RATIO between two arms measured minutes apart, and the absolute anchor is taken
from the it.8 journal instead, where 8 seeds of the same cell were timed.

    twin / softmax   3.6180 at n=2048     2.3323 at n=8192   -- NOT constant
    d_model 256 / 16 4.3559 at s=64       3.3709 at s=1024   -- NOT constant

Both are priced as bands. A ratio quoted at one n and multiplied out at another
is the exact error it.8 recorded (a per-example unit fitted at n=2048 reads 1.43x
low at n=32768).

THE HOST BINDS AND THE BOX CANNOT HOLD TWO SEATS. `m3_capability.py:261` -- "no
.cuda() anywhere in this file". Peak host RSS fits 665.5 + 0.1102*n; at n=49152
the fit predicts 6,082 MiB against a MEASURED 6,584, under by 8.2%, with the
per-example slope rising 0.1102 -> 0.1409. Every RSS number above n=49152 in this
file is a FLOOR, not a price. Two n=49152 seats need 16,460 MiB against a 16,091
MiB box -- refused at any occupancy, not merely at a busy moment.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.r10_it8_pricing import S_PER_RE_UNCONTENDED, S_PER_RE_FOUR_SEATS

ROOT = pathlib.Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------- grid
#: (graph nodes, mean degree, interior rows). Rungs and mean degrees from
#: `scale/r10_corpus_spec.py:506`; `interior_size` read from the 12 rung rows of
#: `results/r10_it14_corpus_spec.jsonl`. They sum to 4,412 -- EXACTLY the leaked
#: eval-row count it.16 measured on the by-file split (4,412 of 10,105), which
#: identifies the tensorisation as one row per interior vertex rather than
#: leaving it undeclared.
RUNGS = ((64, 4.0, 57), (64, 6.0, 60), (128, 4.0, 117), (128, 6.0, 121),
         (256, 4.0, 237), (256, 6.0, 240), (256, 8.0, 239), (512, 4.0, 471),
         (512, 6.0, 485), (512, 8.0, 481), (1024, 4.0, 944), (1024, 6.0, 960))
ARMS = ("softmax", "flat-twin", "S1+S2-twin")
SEEDS = 8
STEPS = 150                       # pinned by it.8's overfitting finding
ROWS_TOTAL = sum(r[2] for r in RUNGS)

# ------------------------------------------------------- measured this round
#: THE ANCHOR. 8 seeds of one cell, threads=12, steps=150, n_train=49152, s=64,
#: d=24, softmax, task e3_t32. `results/r10_it8_capacity_softmax_t32.jsonl`
#: lines 54, 56-62, run 2026-08-30. mean 513.6163 s, sd 9.9279, min 501.58,
#: max 532.11. Verdicts 5 LEARNS / 3 NO READING -- see N8_WRONG_VERDICT.
CELL_A_SOFTMAX_S = 513.6163
CELL_A_SD_S = 9.9279

#: WHY N=8 IS NOT OPTIONAL, measured, same journal. At n=32768 (lines 32, 39, 41,
#: 42, 46-50) eight seeds split 2 LEARNS / 6 NO READING; at n=49152 they split
#: 5 / 3. A single draw therefore reports the wrong side of NRMSE=1.0 for 2 of 8
#: and 3 of 8 seeds respectively. N=1 buys all twelve rungs inside the budget
#: below and buys twelve coin flips.
N8_WRONG_VERDICT = {32768: 2 / 8, 49152: 3 / 8}

#: Per-ROW-per-STEP cost at the corpus shape (s = graph size), softmax,
#: threads=12, this iteration. n=64 reading used; the n=16 and n=12 readings at
#: the same s give 0.008532 and 0.007896, a +-4% spread.
C_ROW_STEP_1024 = 0.528094 / 64          # 0.00825147 s
C_ROW_STEP_256 = 0.034045 / 64           # 0.00053195 s
#: The operator is [n,s,s], so the s exponent should be 2. MEASURED over the
#: 256 -> 1024 pair: log(15.512)/log(4) = 1.978. Exactly 2 is used below; the
#: 1.1% difference moves only rungs that contribute 1.6% of the grid.
S_EXPONENT_MEASURED = math.log(C_ROW_STEP_1024 / C_ROW_STEP_256) / math.log(4.0)

#: twin / softmax, same session, same box, minutes apart. QuintArm(cell="twin")
#: against m3_capability.Arm("softmax"), s=64, d_model=16, threads=12.
TWIN_MULT_LO = 2.136828 / 0.916191       # 2.3323 at n=8192
TWIN_MULT_HI = 0.576326 / 0.159293       # 3.6180 at n=2048

#: d_model 256 / 16, the fork it.20's coherence line implies. Measured at both
#: shapes because the [n,s,s] operator and the [n,s,d^2] projections scale
#: differently in d. n_params goes 4,769 -> 197,249 (41.4x).
D256_MULT_S1024 = 0.460173 / 0.136512    # 3.3709, the corpus shape
D256_MULT_S64 = 0.693863 / 0.159293      # 4.3559, the chain shape

#: THE TIMING INSTRUMENT, REPORTED FAILING. it.8's n=2048 cell at it.8's own pin
#: (8 threads) re-measured this iteration. Same code, same shape, same box.
CONTENTION_NOW = 0.204828 / 0.11137      # 1.8391x
#: The rate the method actually requires: re-measured on the machine at the
#: moment of spend. Every second in this file was measured at this contention,
#: so this is the rate that converts them to work.
RATE_NOW = S_PER_RE_UNCONTENDED * CONTENTION_NOW

#: it.21 and it.22, measured. Corpus build is the `seconds` field summed over the
#: 12 rung rows of results/r10_it14_corpus_spec.jsonl; the hash is sha256 over
#: 196,135 bytes of the four source files, 20 repetitions; the census is the
#: Jaccard over a 3,066-edge planted set against a 2,944-edge prediction.
CORPUS_BUILD_S = 57.174
MANIFEST_HASH_S = 0.000487
CENSUS_J_S = 0.000334
S2_FORWARD_S = 0.094752                  # one s=1024 step over 12 rungs, threads=12

#: Peak host RSS. Fit from it.8; the n=49152 point MEASURED at it.10.
RSS_FIT = (665.5, 0.1102)
RSS_MEASURED = {2048: 901.2, 8192: 1567.9, 32768: 4275.1, 49152: 6584.0}
RSS_FLOOR_ABOVE = 49152
BOX_TOTAL_MIB = 16091
GATE_MARGIN = 1.25                        # scale/vram_gate.preflight default

#: DECLARED, not measured, and labelled so. One seat (the gate allows no more,
#: see fits_the_box) for six hours. Anchors: the largest contiguous buy this
#: round completed is the 8-seed n=49152 wave at 8 x 513.6163 = 4,109 s, and
#: it.8's entire spend was 5,352 s. 21,600 s is 5.26x the former.
DECLARED_BUDGET_S = 21600.0

#: E[max pairwise |cos|] over k=16 unit vectors in d=256, 4,000 draws, seed
#: 0x3a190000, this iteration: 0.175255, 95% CI [0.174506, 0.176004]. The MEAN
#: pairwise coherence over the same draws is 0.049933 [0.049825, 0.050041].
#: it.20's 0.1748 is therefore the MAX, not the mean -- 3.51x apart. This settles
#: which quantity the source measured (house-events.jsonl:10792 flagged it). It
#: does NOT settle which d the S1 stage runs at; that is a declaration.
COHERENCE_MAX_MC = (0.175255, 0.174506, 0.176004)
COHERENCE_MEAN_MC = (0.049933, 0.049825, 0.050041)
COHERENCE_SOURCE = (0.1748, 0.17446, 0.17513)


def re_of(seconds: float, rate: float = None) -> float:
    """Run-equivalents. The DEFAULT is the rate re-measured at spend, not the
    uncontended one: every second in this file was measured on the box as it is
    now, and dividing a contended numerator by an idle denominator overstates
    the work. `re_of(x, S_PER_RE_UNCONTENDED)` is available and is what the
    report quotes once, to name that trap rather than fall into it."""
    return seconds / (RATE_NOW if rate is None else rate)


def c_row_step(s: int) -> float:
    """Seconds per row per step at sequence length `s`. s^2, the operator's own
    algebra, anchored on the MEASURED s=1024 point."""
    return C_ROW_STEP_1024 * (s / 1024.0) ** 2


def cell_seconds(arm: str, rung, twin_mult: float = TWIN_MULT_LO) -> float:
    n, _deg, rows_ = rung
    mult = 1.0 if arm == "softmax" else twin_mult
    return STEPS * rows_ * c_row_step(n) * mult


def grid_r1(twin_mult: float = TWIN_MULT_LO):
    """(rung, arm) -> seconds for all 8 seeds. 288 cells, 288 trainings."""
    return {(rg, arm): SEEDS * cell_seconds(arm, rg, twin_mult)
            for rg in RUNGS for arm in ARMS}


def total_r1(twin_mult: float = TWIN_MULT_LO) -> float:
    return sum(grid_r1(twin_mult).values())


def total_r2(twin_mult: float = TWIN_MULT_LO) -> float:
    """24 cells: one pooled model per (arm, seed), every rung padded to s=1024."""
    per_cell = STEPS * ROWS_TOTAL * C_ROW_STEP_1024
    return SEEDS * per_cell * (1.0 + 2.0 * twin_mult)


def rung_cost(rg, twin_mult: float = TWIN_MULT_LO) -> float:
    return sum(SEEDS * cell_seconds(a, rg, twin_mult) for a in ARMS)


def affordable(budget: float = DECLARED_BUDGET_S, twin_mult: float = TWIN_MULT_LO):
    """(kept, dropped, spend), cheapest rung first. A rung is ATOMIC: 3 arms x 8
    seeds, because a rung bought at one arm or four seats decides nothing."""
    kept, dropped, spend = [], [], 0.0
    for rg in sorted(RUNGS, key=lambda r: rung_cost(r, twin_mult)):
        c = rung_cost(rg, twin_mult)
        if spend + c <= budget:
            kept.append(rg)
            spend += c
        else:
            dropped.append(rg)
    return kept, dropped, spend


def rss_mib(n: int) -> tuple[float, bool]:
    """(MiB, is_floor). Measured where measured; the linear fit elsewhere, and
    the fit is a FLOOR above n=49152 -- it read 8.2% low at the one point above
    its fitted range that anyone checked."""
    if n in RSS_MEASURED:
        return RSS_MEASURED[n], n > RSS_FLOOR_ABOVE
    a, b = RSS_FIT
    return a + b * n, n > RSS_FLOOR_ABOVE


def fits_the_box(n: int, seats: int) -> tuple[bool, float]:
    """Can the box hold `seats` concurrent cells at n rows? Against TOTAL, not
    free -- a False here is unconditional and no sibling exiting repairs it."""
    need = seats * rss_mib(n)[0] * GATE_MARGIN
    return need <= BOX_TOTAL_MIB, need


# --------------------------------------------------------------------- report
def rows() -> list[dict]:
    """Every priced object, machine-readable. Written from Python, never through
    a shell string -- four lines of house-events.jsonl do not parse for exactly
    that reason (it.18: lines 1899, 2937, 2938, 5871)."""
    out = [dict(t="header", iteration="r10.it19", agent="MERCURY",
                phase="1c", iterations="20-24", objects=481, trainings=288,
                rate_s_per_re=S_PER_RE_UNCONTENDED,
                rate_source="scale/r10_it8_pricing.S_PER_RE_UNCONTENDED, "
                            "measured it.8; contract anchors 20.58/48.89 at "
                            "PHASE2_CONTRACT_V_MAIN_4.md:197",
                budget_s=DECLARED_BUDGET_S, budget_kind="DECLARED",
                contention_now=CONTENTION_NOW)]
    for kind, s in (("s1_ingestion", CORPUS_BUILD_S + 96 * MANIFEST_HASH_S),
                    ("s2_census", 96 * (S2_FORWARD_S + CENSUS_J_S))):
        out.append(dict(t="stage", stage=kind, cells=96, seconds=s,
                        run_equivalents=re_of(s)))
    kept, dropped, spend = affordable()
    keptset = set(kept)
    for rg in RUNGS:
        n, deg, rw = rg
        for arm in ARMS:
            sec = SEEDS * cell_seconds(arm, rg)
            out.append(dict(t="cell", stage="it23_training", n_graph=n,
                            mean_deg=deg, rows=rw, arm=arm, seeds=SEEDS, s=n,
                            steps=STEPS, seconds=sec, run_equivalents=re_of(sec),
                            per_seed_seconds=sec / SEEDS,
                            status="priced" if rg in keptset else "DROPPED",
                            why=None if rg in keptset
                                else "over the declared 21,600 s single-seat budget",
                            code_exists=arm != "S1+S2-twin"))
    for rg in dropped:
        out.append(dict(t="dropped", n_graph=rg[0], mean_deg=rg[1], rows=rg[2],
                        arms=list(ARMS), seeds=SEEDS, seconds=rung_cost(rg),
                        run_equivalents=re_of(rung_cost(rg)),
                        why="unaffordable at the declared 21,600 s single-seat budget"))
    t1, t2 = total_r1(), total_r2()
    out.append(dict(t="total", reading="R1_per_rung", cells=288, seconds=t1,
                    hours=t1 / 3600, run_equivalents=re_of(t1),
                    rate_used_s_per_re=RATE_NOW,
                    run_equivalents_at_uncontended_rate=re_of(
                        t1, S_PER_RE_UNCONTENDED),
                    uncontended_wall_clock_s=t1 / CONTENTION_NOW,
                    run_equivalents_four_seats=re_of(t1, S_PER_RE_FOUR_SEATS)))
    out.append(dict(t="total", reading="R2_pooled_padded", cells=24, seconds=t2,
                    hours=t2 / 3600, run_equivalents=re_of(t2),
                    note="one twelfth the cells, %.2fx the cost" % (t2 / t1)))
    out.append(dict(t="total", reading="R1_at_twin_mult_ceiling",
                    twin_mult=TWIN_MULT_HI, seconds=total_r1(TWIN_MULT_HI),
                    run_equivalents=re_of(total_r1(TWIN_MULT_HI))))
    out.append(dict(t="total", reading="R1_if_S1_runs_at_d256",
                    d_mult=D256_MULT_S1024, seconds=t1 * D256_MULT_S1024,
                    run_equivalents=re_of(t1 * D256_MULT_S1024),
                    note="undeclared fork; measured multiplier, not a flop model"))
    for n in (32768, 49152):
        mib, floor = rss_mib(n)
        for seats in (1, 2, 3):
            ok, need = fits_the_box(n, seats)
            out.append(dict(t="host", n_train=n, peak_rss_mib=mib,
                            is_floor=floor, seats=seats, need_mib=need,
                            fits_total_box=ok, box_total_mib=BOX_TOTAL_MIB))
    out.append(dict(t="coherence", d=256, k=16, draws=4000, seed="0x3a190000",
                    quantity="E[max pairwise |cos|]", mc=list(COHERENCE_MAX_MC),
                    mean_pairwise=list(COHERENCE_MEAN_MC),
                    source=list(COHERENCE_SOURCE),
                    resolves="which quantity 0.1748 measures (max, not mean)",
                    does_not_resolve="which d the S1 stage runs at"))
    return out


def report() -> None:
    p = print
    t1, t2 = total_r1(), total_r2()
    p("=== R10 PHASE 1c (it.20-24) -- PRICED, MERCURY it.19 ===")
    p("rate USED for every RE below: %.2f s/RE -- the uncontended %.2f"
      % (RATE_NOW, S_PER_RE_UNCONTENDED))
    p("      re-measured at spend and multiplied by the %.4fx contention"
      % CONTENTION_NOW)
    p("      measured on this box this iteration. %.2f s/RE is imported by"
      % S_PER_RE_UNCONTENDED)
    p("      object identity from scale/r10_it8_pricing.py, not re-picked here.")
    p("      %.2f s/RE at four seats. The contract's own anchors are 20.58 and"
      % S_PER_RE_FOUR_SEATS)
    p("      48.89 at PHASE2_CONTRACT_V_MAIN_4.md:197. An RE is a 600-step run-")
    p("      equivalent -- that is the quantity the source measured.")
    p("      RE-MEASURED AT SPEND, which the method requires: it.8's own n=2048")
    p("      cell at it.8's own 8-thread pin reads 0.204828 s/step now against")
    p("      0.11137 then -- %.4fx. This box is at %.2f s/RE right now."
      % (CONTENTION_NOW, S_PER_RE_UNCONTENDED * CONTENTION_NOW))
    p("")
    p("WHAT A CELL IS: (stage, arm, rung, seed). 481 objects, 288 of them trainings.")
    p("  it.21 S1 ingestion + manifest hash   12 rungs x 8 seeds          =  96")
    p("  it.22 S2 graph + edge-recovery J     12 rungs x 8 seeds          =  96")
    p("  it.23 a trained arm                  12 rungs x 3 arms x 8 seeds = 288")
    p("  it.24 verdict                        reads the table            =   1")
    p("  Rungs are r10_corpus_spec.RUNGS (12); rows per rung are the")
    p("  `interior_size` fields of results/r10_it14_corpus_spec.jsonl, which sum")
    p("  to %d -- exactly the leaked-row count it.16 measured on the by-file"
      % ROWS_TOTAL)
    p("  split (4,412 of 10,105). That identifies the tensorisation as one row")
    p("  per interior vertex instead of leaving it undeclared.")
    p("  NOTHING NESTS. it.8 saved 24.7% on the steps axis; there is no steps")
    p("  axis here (150 is pinned by it.8's own overfitting measurement), the")
    p("  seed axis IS the N=8 requirement, and three arms are three optimisations.")
    p("")
    p("c_step EQUIVALENT FOR THE S1/S2 SHAPES -- MEASURED, threads=12, this iteration:")
    p("  per row per step, s=1024   %.8f s   (n=64; n=16 and n=12 give"
      % C_ROW_STEP_1024)
    p("                                            0.008532 and 0.007896)")
    p("  per row per step, s= 256   %.8f s   (n=64)" % C_ROW_STEP_256)
    p("  measured s exponent %.3f against the operator's algebraic 2"
      % S_EXPONENT_MEASURED)
    p("  twin / softmax     %.4f at n=8192, %.4f at n=2048  -- NOT constant,"
      % (TWIN_MULT_LO, TWIN_MULT_HI))
    p("                                                        priced at both ends")
    p("  d_model 256 / 16   %.4f at s=1024, %.4f at s=64    -- the undeclared fork"
      % (D256_MULT_S1024, D256_MULT_S64))
    p("")
    p("  THE INSTRUMENT FAILED FIRST AND IS REPORTED FAILING. Every absolute step")
    p("  time above was taken while the box's free RAM moved 3,964 -> 6,478 MiB;")
    p("  the it.8 reproduction is %.2fx slow. RATIOS survive that, absolutes do"
      % CONTENTION_NOW)
    p("  not, so the absolute anchor below comes from the it.8 journal instead.")
    p("  The peak-RSS reader used this iteration is a HIGH-WATER process reading:")
    p("  it agreed with it.8 on the FIRST configuration (894.0 vs 901.2 MiB,")
    p("  0.8%) and every later reading inherited that high-water mark. All but")
    p("  the first are discarded; the RSS prices below come from it.8 and it.10.")
    p("")
    p("  THE ARM THAT HAS NO CODE. m3_capability.ARMS is (softmax, pivot_signed,")
    p("  pivot_unsigned, windowed_signed) at line 66 and Arm.__init__ raises on")
    p("  anything else; m3_quintuple carries twin/settled/argmax and no S1 or S2")
    p("  cell. QuintArm is pinned at d_model=16 (measured: wq.in_features == 16),")
    p("  so it cannot ingest d=256 roles without a constructor change. The")
    p("  S1+S2-twin calibration COMPUTE is 38.2 s and is not the constraint. The")
    p("  build is, and it is unpriced here. A cell with no implementation is not")
    p("  a cheap cell; it is an unpriced one, and it is named as such in every row.")
    p("")
    p("ANCHOR, MEASURED 8 TIMES: %.4f s per (softmax, n=49152, 150 steps, 1 seed),"
      % CELL_A_SOFTMAX_S)
    p("  sd %.4f, range 501.58-532.11, threads=12," % CELL_A_SD_S)
    p("  results/r10_it8_capacity_softmax_t32.jsonl:54,56-62.")
    p("  N=8 IS NOT OPTIONAL: those 8 seeds split 5 LEARNS / 3 NO READING, so one")
    p("  draw lands the wrong side of NRMSE=1.0 %.1f%% of the time; at n=32768 the"
      % (100 * N8_WRONG_VERDICT[49152]))
    p("  same journal splits 2/6 -- %.1f%%. N=1 buys all twelve rungs inside the"
      % (100 * N8_WRONG_VERDICT[32768]))
    p("  budget below and buys twelve coin flips.")
    p("")
    p("TOTAL. Two readings of the it.23 table, neither declared by the round:")
    p("  R1  per-rung models,  288 cells : %9.0f s = %6.2f h = %8.1f RE"
      % (t1, t1 / 3600, re_of(t1)))
    p("  R2  one pooled model,  24 cells : %9.0f s = %6.2f h = %8.1f RE"
      % (t2, t2 / 3600, re_of(t2)))
    p("  R2 has one TWELFTH the cells and costs %.2fx more -- padding ten rungs"
      % (t2 / t1))
    p("  to s=1024 pays the s^2 operator on rows 64 to 512 wide.")
    p("  A SMALLER CELL COUNT IS NOT A CHEAPER GRID.")
    p("  R1 at the twin ceiling %.4f    : %9.0f s = %6.2f h"
      % (TWIN_MULT_HI, total_r1(TWIN_MULT_HI), total_r1(TWIN_MULT_HI) / 3600))
    p("")
    p("  WHICH RATE, AND THE TRAP IN CONVERTING AT THE WRONG ONE. The seconds")
    p("  above were built from c_row_step measured on THIS box AT %.2fx"
      % CONTENTION_NOW)
    p("  contention, so they are contended wall-clock, not work. Dividing them")
    p("  by the UNCONTENDED %.2f s/RE gives %.1f RE and DOUBLE-COUNTS the"
      % (S_PER_RE_UNCONTENDED, re_of(t1, S_PER_RE_UNCONTENDED)))
    p("  contention -- it prices idle-box units with a busy-box numerator. The")
    p("  work content is the contended rate:")
    p("    work        : %8.1f RE at the re-measured %.2f s/RE"
      % (re_of(t1), RATE_NOW))
    p("    wall-clock now, one seat  : %9.0f s = %6.2f h" % (t1, t1 / 3600))
    p("    wall-clock on an idle box : %9.0f s = %6.2f h  (%.1f RE x %.2f s/RE)"
      % (t1 / CONTENTION_NOW, t1 / CONTENTION_NOW / 3600,
         re_of(t1), S_PER_RE_UNCONTENDED))
    p("    at four seats %.2f s/RE, the same work is %9.0f s -- and four seats"
      % (S_PER_RE_FOUR_SEATS, re_of(t1) * S_PER_RE_FOUR_SEATS))
    p("    are not available here anyway: the host clause below allows ONE.")
    p("  R1 if S1 runs at d=256          : %9.0f s = %8.1f RE  (%.4fx, MEASURED"
      % (t1 * D256_MULT_S1024, re_of(t1 * D256_MULT_S1024), D256_MULT_S1024))
    p("                                                       at s=1024, not modelled)")
    p("")
    s1 = CORPUS_BUILD_S + 96 * MANIFEST_HASH_S
    s2 = 96 * (S2_FORWARD_S + CENSUS_J_S)
    p("it.21 AND it.22 ARE FREE, and that is a finding, not an omission:")
    p("  S1 ingestion, 96 cells : %8.2f s = %5.2f RE   corpus build %.3f s from"
      % (s1, re_of(s1), CORPUS_BUILD_S))
    p("    the twelve `seconds` fields of results/r10_it14_corpus_spec.jsonl,")
    p("    plus 96 x %.6f s of sha256 over 196,135 bytes of source."
      % MANIFEST_HASH_S)
    p("  S2 census,     96 cells : %8.2f s = %5.2f RE   one s=1024 forward per"
      % (s2, re_of(s2)))
    p("    seed at %.6f s, plus %.6f s of Jaccard over 3,066 planted edges."
      % (S2_FORWARD_S, CENSUS_J_S))
    p("  Together %.1f s = %.2f RE = %.3f%% of Phase 1c. THE PHASE IS it.23."
      % (s1 + s2, re_of(s1 + s2), 100 * (s1 + s2) / t1))
    p("  The manifest-hash refusal machinery it.21 requires costs 481 x %.6f s"
      % MANIFEST_HASH_S)
    p("  = %.3f s over the whole phase. It is free and there is no argument for"
      % (481 * MANIFEST_HASH_S))
    p("  skipping it.")
    p("")
    p("UNAFFORDABLE, cell by cell, with the arithmetic.")
    p("  Budget: %.0f s = %.1f RE. DECLARED, not measured, and labelled so: one"
      % (DECLARED_BUDGET_S, re_of(DECLARED_BUDGET_S)))
    p("  seat (the gate allows no more, below) x 6 h. The largest contiguous buy")
    p("  this round completed is the 8-seed n=49152 wave, 8 x %.4f = %.0f s;"
      % (CELL_A_SOFTMAX_S, 8 * CELL_A_SOFTMAX_S))
    p("  it.8's entire spend was 5,352 s. This budget is %.2fx the former."
      % (DECLARED_BUDGET_S / (8 * CELL_A_SOFTMAX_S)))
    p("  The budget is in WALL-CLOCK, so what it buys depends on the box: the")
    p("  kept set below is priced at the %.2fx contention measured at spend."
      % CONTENTION_NOW)
    kept, dropped, spend = affordable()
    p("")
    p("  THE CORNER. The two 1024-node rungs alone:")
    corner = sum(rung_cost(rg) for rg in RUNGS if rg[0] == 1024)
    for rg in RUNGS:
        if rg[0] != 1024:
            continue
        n, deg, rw = rg
        p("    n=%d deg=%.1f, %d rows: %d steps x %d rows x %.8f s/row-step"
          % (n, deg, rw, STEPS, rw, c_row_step(n)))
        p("      = %8.1f s per softmax seed, x%.4f = %8.1f s per twin seed"
          % (cell_seconds("softmax", rg), TWIN_MULT_LO,
             cell_seconds("flat-twin", rg)))
        p("      x 8 seeds x 3 arms = %9.0f s = %5.2f h = %7.1f RE"
          % (rung_cost(rg), rung_cost(rg) / 3600, re_of(rung_cost(rg))))
    p("    the pair           = %9.0f s = %5.2f h = %7.1f RE = %.1f%% of R1"
      % (corner, corner / 3600, re_of(corner), 100 * corner / t1))
    p("    (it.8's comparable corner, (9600, 32768) at three t*, was 79%.)")
    p("")
    p("  KEPT %d rungs = %d cells, %.0f s = %.1f RE (%.1f%% of R1):"
      % (len(kept), 3 * SEEDS * len(kept), spend, re_of(spend), 100 * spend / t1))
    for rg in sorted(kept):
        p("    n=%-5d deg=%.1f rows=%-4d  %8.0f s = %6.1f RE"
          % (rg[0], rg[1], rg[2], rung_cost(rg), re_of(rung_cost(rg))))
    drop_s = sum(rung_cost(rg) for rg in dropped)
    p("  DROPPED %d rungs = %d cells, %.0f s = %.2f h = %.1f RE (%.1f%% of R1),"
      % (len(dropped), 3 * SEEDS * len(dropped), drop_s, drop_s / 3600,
         re_of(drop_s), 100 * drop_s / t1))
    p("  every one named:")
    for rg in sorted(dropped):
        for arm in ARMS:
            c = SEEDS * cell_seconds(arm, rg)
            p("    n=%-5d deg=%.1f rows=%-4d arm=%-11s x8 seeds  %8.0f s = %6.1f RE%s"
              % (rg[0], rg[1], rg[2], arm, c, re_of(c),
                 "  [NO CODE]" if arm == "S1+S2-twin" else ""))
    p("")
    p("HOST RSS PER CELL, and whether the box holds the concurrency:")
    p("  fit RSS = %.1f + %.4f*n MiB (it.8). MEASURED 6,584 MiB at n=49152"
      % RSS_FIT)
    p("  against a predicted 6,082 -- 8.2% LOW -- with the per-example slope")
    p("  rising 0.1102 (8192->32768) to 0.1409 (32768->49152). ABOVE n=%d"
      % RSS_FLOOR_ABOVE)
    p("  EVERY NUMBER HERE IS A FLOOR, NOT A PRICE.")
    for n in (32768, 49152):
        mib, floor = rss_mib(n)
        p("    n=%-6d peak %7.1f MiB%s" % (n, mib, "  [FLOOR]" if floor else ""))
        for seats in (1, 2, 3):
            ok, need = fits_the_box(n, seats)
            p("      %d seat(s): needs %8.0f MiB (x%.2f gate margin) against the "
              "%d MiB BOX -- %s"
              % (seats, need, GATE_MARGIN, BOX_TOTAL_MIB,
                 "fits" if ok else "IMPOSSIBLE"))
    p("  Two n=49152 seats need 16,460 MiB against a 16,091 MiB box, so 2-seat")
    p("  concurrency there is refused at ANY occupancy -- no sibling exiting")
    p("  repairs it. CONCURRENCY = 1 and the eight seeds are strictly sequential.")
    p("  MEASURED LIVE, scale/vram_gate.preflight at 4,885 MiB available:")
    p("    preflight(0, name='it19-phase1c-1-seat-n49152', host_mib=6584)")
    p("      -> HOST DOES NOT FIT, needs 8,230 MiB, 4,885 available")
    p("    preflight(0, name='it19-phase1c-1-seat-n32768', host_mib=4275.1)")
    p("      -> HOST DOES NOT FIT, needs 5,344 MiB, 4,885 available")
    p("  The gate refused every seat at the moment this was priced. That refusal")
    p("  is the correct behaviour and it is the same one that stopped the n=65536")
    p("  octave at 9,859 needed against 8,808 available.")
    p("")
    p("  THE CORPUS SHAPE MOVES THE MEMORY QUESTION. Phase 1c's largest rung is")
    p("  960 rows at s=1024, not 49,152 rows at s=64, so what binds is the")
    p("  [n,s,s] operator: 960 x 1024^2 x 4 B = %.0f MiB for ONE materialised"
      % (960 * 1024 * 1024 * 4 / 2 ** 20))
    p("  operator, before autograd keeps a copy. That is why R2's padding is a")
    p("  memory decision as well as a %.2fx time decision, and why the row count"
      % (t2 / t1))
    p("  the it.8 fit is expressed in does not transfer to this phase at all.")
    p("")
    idle_kept, idle_dropped, idle_spend = affordable(
        DECLARED_BUDGET_S * CONTENTION_NOW)
    p("  ON AN IDLE BOX the same %.0f s buys %d of 12 rungs instead of %d --"
      % (DECLARED_BUDGET_S, len(idle_kept), len(kept)))
    p("  the affordability verdict is a function of the occupancy at spend, and")
    p("  the occupancy moved 3,964 -> 6,478 MiB inside three minutes today. The")
    p("  extra rung is n=%d deg=%.1f."
      % (idle_kept[-1][0], idle_kept[-1][1]))
    p("")
    p("DOES PHASE 1c FIT THIS BOX? NO at the full grid, YES for %d of 12 rungs"
      % len(kept))
    p("  contended and %d idle." % len(idle_kept))
    p("  R1 is %.1f h at one seat, and the box allows exactly one seat. At the"
      % (t1 / 3600))
    p("  declared %.0f s budget, %.1f%% of the grid is bought and %.1f%% is dropped."
      % (DECLARED_BUDGET_S, 100 * spend / t1, 100 * drop_s / t1))
    p("  The it.23 table as written -- 12 rungs, 3 arms, N=8 -- does not fit any")
    p("  budget this round has ever spent, by a factor of %.1f against it.8's own"
      % (t1 / 5352.0))
    p("  5,352 s.")


def demo() -> None:
    """One runnable check per claim a reader would otherwise have to trust."""
    # The rate is IMPORTED, not re-picked. Object identity, so a local copy fails.
    from scale import r10_it8_pricing as it8
    assert S_PER_RE_UNCONTENDED is it8.S_PER_RE_UNCONTENDED
    assert (S_PER_RE_UNCONTENDED, S_PER_RE_FOUR_SEATS) == (18.01, 48.89)

    # The tensorisation claim: interior rows sum to the leaked-row count it.16
    # measured. If either number moves this stops being evidence.
    assert ROWS_TOTAL == 4412, ROWS_TOTAL

    # Accounting closes: kept + dropped reconstructs the grid exactly.
    kept, dropped, spend = affordable()
    assert set(kept) | set(dropped) == set(RUNGS)
    assert not (set(kept) & set(dropped))
    assert abs(spend + sum(rung_cost(r) for r in dropped) - total_r1()) < 1e-6
    assert spend <= DECLARED_BUDGET_S
    assert dropped, "a budget that drops nothing is not a budget"

    # The affordability rule is greedy-by-cost. A greedy knapsack that SKIPS a
    # rung to fit a cheaper one further up would be an odd thing to publish, so
    # the cheaper prefix rule is checked to give the same answer here. If a
    # future grid makes these differ, the prefix rule is the one to keep.
    prefix, running = [], 0.0
    for rg in sorted(RUNGS, key=rung_cost):
        if running + rung_cost(rg) > DECLARED_BUDGET_S:
            break
        prefix.append(rg)
        running += rung_cost(rg)
    assert set(prefix) == set(kept), (sorted(prefix), sorted(kept))

    # Every dropped rung is named at CELL granularity, never summarised away.
    named = {(r[0], r[1], a) for r in dropped for a in ARMS}
    assert len(named) == 3 * len(dropped)

    # THE COUNTERINTUITIVE ONE, asserted because a reader will not believe it:
    # one twelfth the cells, more money.
    assert total_r2() > total_r1(), (total_r2(), total_r1())

    # MUST-FIRE on the host clause: it has to refuse the impossible, or it is the
    # vacuous GREEN this campaign catalogues. Two seats at n=49152 exceed the
    # BOX, not merely the free memory, so this cannot decay when siblings exit.
    ok2, need2 = fits_the_box(49152, 2)
    assert not ok2 and need2 > BOX_TOTAL_MIB, (ok2, need2)
    # MUST-NOT-FIRE: one n=32768 seat is possible on an idle box, or the clause
    # blocks everything and says nothing.
    assert fits_the_box(32768, 1)[0]

    # The RSS model must announce itself as a floor exactly where it was measured
    # to read low, and it must never be quoted as a price above that point.
    assert rss_mib(49152) == (6584.0, False)
    assert rss_mib(65536)[1] is True
    assert rss_mib(65536)[0] < 6584.0 + 0.1409 * (65536 - 49152), (
        "the linear fit must under-read the risen slope -- that is why it is a floor")

    # The s^2 law is MEASURED, not assumed: the two-point exponent must sit near
    # the operator's algebraic 2 or the cost model is not the operator's.
    assert 1.9 < S_EXPONENT_MEASURED < 2.1, S_EXPONENT_MEASURED

    # The ratios are bands, not constants. Equal ends would claim a stability
    # nothing measured.
    assert TWIN_MULT_HI > TWIN_MULT_LO * 1.4
    assert D256_MULT_S64 > D256_MULT_S1024

    # The coherence quantity: the source's number must be the MAX -- the MC
    # interval overlaps it and the MEAN is nowhere near. Getting this backwards
    # would import a floor 3.5x wrong.
    assert COHERENCE_MAX_MC[1] <= COHERENCE_SOURCE[2]
    assert COHERENCE_MAX_MC[2] >= COHERENCE_SOURCE[1]
    assert COHERENCE_MEAN_MC[2] < COHERENCE_SOURCE[1] / 3

    # N=8 is priced because a single draw is MEASURED wrong, not because 8 is
    # traditional. Unanimous splits would leave the 8x unjustified.
    assert 0.0 < N8_WRONG_VERDICT[32768] < 0.5
    assert 0.0 < N8_WRONG_VERDICT[49152] < 0.5

    # THE RATE DIRECTION. Seconds measured on a contended box, divided by the
    # UNCONTENDED rate, overstate the work -- an idle-box denominator under a
    # busy-box numerator. The report must never quote the larger number as the
    # work content, so the ordering is asserted rather than left to the reader.
    assert re_of(total_r1(), S_PER_RE_UNCONTENDED) > re_of(total_r1())
    assert abs(re_of(total_r1()) * S_PER_RE_UNCONTENDED
               - total_r1() / CONTENTION_NOW) < 1e-6

    # Affordability is a function of occupancy, and saying so requires the two
    # answers to actually differ. If they stopped differing the report's claim
    # that occupancy decides the grid would be decoration.
    idle_kept, _, _ = affordable(DECLARED_BUDGET_S * CONTENTION_NOW)
    assert len(idle_kept) > len(kept), (len(idle_kept), len(kept))
    assert set(kept) <= set(idle_kept)

    # The stages must not be summed into the trainings and vanish.
    stages = (CORPUS_BUILD_S + 96 * MANIFEST_HASH_S
              + 96 * (S2_FORWARD_S + CENSUS_J_S))
    assert stages / total_r1() < 0.001

    # Every priced object carries a status, and the arm with no code is flagged
    # in every row it appears in -- a missing cell must not read as a cheap one.
    cells = [r for r in rows() if r["t"] == "cell"]
    assert len(cells) == len(RUNGS) * len(ARMS) == 36
    assert all(r["code_exists"] is False for r in cells
               if r["arm"] == "S1+S2-twin")
    assert sum(r["seconds"] for r in cells) == total_r1()

    print("\nALL CHECKS PASSED")


def write_jsonl() -> pathlib.Path:
    p = ROOT / "results" / "r10_it19_priced_1c.jsonl"
    payload = rows()
    with p.open("w", encoding="utf-8") as f:
        for r in payload:
            f.write(json.dumps(r) + "\n")
    # A write that reports success while destroying its payload is this round's
    # own recorded failure (house-events.jsonl:1899, 2937, 2938, 5871). Read it
    # back and parse every line rather than trusting the write.
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        json.loads(line)
        n += 1
    assert n == len(payload), (n, len(payload))
    return p


if __name__ == "__main__":
    report()
    demo()
    if "--write" in sys.argv:
        print("\nwrote %s (%d rows, every line re-parsed)"
              % (write_jsonl(), len(rows())))
