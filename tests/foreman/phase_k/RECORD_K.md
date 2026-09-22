# Phase K — PROPAGATOR: process record

Contract: `PHASE_K_CONTRACT.md`. Instances: `instances/` (sha256 in `instances/SHA256SUMS`).
Data: `DATA_MANIFEST.json`. Lane producers and reports: `K0/<seat>/`, `S0/cameron/`.
Audit: `K0/inspector/PASS1.md` (rerun logs in `K0/inspector/rerun/`, checkpoints omitted).

## it.K0 (2026-09-23 01:37 – 02:5x): results as bound

Fellows on opus with nurses; one Inspector pass, 63 audited, 24 struck. Machine: RTX 4060
laptop (8 GB), torch 2.14.0+cu126, no flash attention (SDPA dispatches
EFFICIENT_ATTENTION).

**The author's instances reproduce.** `wald.py` and `yukawa.py` hash to their pinned
sha256 and re-run byte-identical to `wald.out` / `yukawa.out` (Inspector). Cameron's
independent rebuild of the wald bed and of Y3 reproduces every pinned number. Chase's
rebuild of Y1 and Y2 matches to ≤ 8.6e−16 (Y1) and ≤ 1.9e−13 relative (Y2). The
contract's instance kill (§6, last line) does not fire.

**R-DEPTH's prediction line is not a ceiling.**
- A hand-set 6-layer causal softmax construction (Foreman's C2 hybrid: pointer doubling
  plus a positional window) scores **0.3982** at R1 width on the pinned bed against the
  pinned L = 6 line of 0.2539. Its 27 heads are real softmax heads (0 mismatches against
  the idealised construction, margin 0.992).
- With chance credit on unresolved queries, perfect L = 6 doubling alone scores
  **0.2988** against a chance-credited line of 0.2739 (Cameron).
- The counter as written (> 0.5 beyond 2^L) does not fire at R1 width: the anchored
  hybrid scores 0.1934. At a window of the whole context (W = 4096) it scores 1.0.
- Width confounds R-SCALE: the centered-window slope is 1.2207 against the 1.2 edge of
  the doubling band (margin 0.021); a dilated bed gives a ratio of 1.2202 against its bar of 1.25 (Foreman).

**The logit scale is unidentified at one training length.** A resolvent fitted at a
single length (s_train = 11.50) scores 0.2551 at n = 16k, where s_99 = 14.50 is needed
(a = 1.082) (Cameron). The γ window on the pinned bed: γ = 0.9 gives 0.7798;
0.99–0.99999 give 1.0; 1 − 1e−10 gives 0.0605. γ = 0.999 is kept. The centered
resolvent is killed (0.5596 on its own bar; slopes 1.600 vs 1.528).

**The resolvent layer trains with exact gradients** (Chase). x = (1−γ)(I−γW)⁻¹V at β = 1;
backward by the transposed triangular solve, blocked in one reverse pass.
- Forward `fs5c` (pivot rebuilt from off-diagonal mass): 3.04e−6 / 4.08e−6 against
  float64 at S 1024. The unmodified fs5 stays RED at 3.68e−5.
- Backward `fusedc`: dq 2.95e−6, dk 9.74e−6, dv 2.15e−7. The kernel-normaliser adjoint
  stays RED at dq 1.14e−3.
- Gradchecks pass for softmax, sparsemax and the blocked adjoint; mutants fail.
- CUDA-graphed equals eager bitwise on 8 tensors.
- Cost, forward+backward against one SDPA forward+backward, B1 H8 D64: **2.27× at
  S 4096, 2.26× at S 8192**; the Inspector's idle-card rerun gives 2.30× / 2.26×.
  Reported, not scored.

**S0 f-lite** (Cameron, audited this pass). Exact to 2.899e−6 against dense smprime;
55,692 tok/s at 11.2M parameters and 1.63 GiB peak; **0.933×** the twin on the
Inspector's rerun (reported 0.92×; `test_cost` divides by a hard-coded twin figure).
HF patch parity stays RED at 6.690e−5 (gate-gradient precision; open).

**DATA is met** (Wilson). FineWeb-Edu `sample/10BT` files 000–003 at revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, licence ODC-By v1.0 (card line 632);
sha256 of all 4 parquet files matches the Hugging Face LFS record. Tokenised with
tiktoken 0.12.0 `gpt2`: 2,994,211,739 train tokens in 30 shards, 10,312,022 validation
tokens (the first 10,000 documents of file 000). Shards live outside the repository
(`C:/Users/seal/datasets/fineweb_edu/`); the Inspector re-hashed 3 of 31.

**Harness** (`K0/wilson/train_ladder.py`). `--attn alibi` is the ALiBi twin (a_L) with
no learned position path. GPU resume is bitwise only with `--deterministic`
(400/400 step losses, validation 6.742448, reproduced by the Inspector across sessions),
at 14.4% throughput. The default mode diverges at step 139 (max 2.22e−4), before the
resume point: run-to-run nondeterminism, not the resume path.

## Struck by the Inspector (to Open, not re-argued)

- Whether bed_k's random ids remove the shortcut. Foreman's `bedk_best_response`
  (0.3181) was declared RED and never run; Cameron's band for the beyond-2^L best
  response on bed_k (0.041–0.063, board 0.058–0.068) has no bar. The two disagree, and
  neither is bound.
- "The shortcut relies on ids = positions"; "no lower bound covers this bed" (sources
  not saved in the lane); "the counter fires at 48 slots"; Foreman's `id_range_gap`.
- Cameron's multi-length route (a = 1.26 → 1.0), the γ 0.99–0.99999 row at 16k, the
  pinned-seed depth claim, "extra heads do not beat perfect doubling".
- S0's 26M and 725k rows; the fp64-accumulation kill.
- Chase: the γ·ρ(W) < 1 check (it could not fail: W_00 = 1); bf16 rows with
  γ·rowsum ≥ 1 (19.3% / 21.2%, read of V = 1 at 2.53 / 1.91); BOS taking 51–99% of
  row 1023's read at γ = 0.999; tail-mass m > 0 on 16/16 content heads; log-kernel
  m < 0 on 7/8; unit self-weight counts; forward error at S 4096 / 8192 (reproduces,
  no bar); densec 9.41×; the pivot diagnosis of fs5's miss.
- Wilson: the selftest 8/8 (no RED; four checks could not fail); the MFU test
  (`test_mfu.py` edited after its RED; the 14 logs still reproduce every row).

Process notes: Chase dispatched no nurses. An Inspector nurse ran Chase's GPU bar file
during Wilson's `real_R1` run, which coincides with the R1 slowdown Wilson excluded.

## R-DEPTH as registered for it.K1 (before any K1 cell runs)

Each clause replaces a contract clause that K0 broke; the bound fact behind it is named.

1. **Bed.** bed_k: n distinct ids drawn from V = 32,768, parents by id, not position
   (Cameron, `K0_k1_bed`). Replaces the pinned bed, whose ids are positions.
2. **Floor row first (K1.F).** Before any arm trains, on bed_k at R0 width and at each
   test length: the position-only floor, chance-credited L-doubling, and the C2 hybrid
   at its best window. The (a_L) line is the maximum of the three, measured. Replaces
   "stays under 0.254", which a hand-set construction beats (0.3982). If the C2 hybrid
   exceeds 0.5 beyond 2^L on bed_k, the bed is void by the contract's own kill and a
   deeper bed is owed before any arm runs.
3. **Twin.** (a_L) is `train_ladder.py --attn alibi`: ALiBi with no absolute-position
   table, so no arm reads position except through the fixed prior.
4. **Logit scale.** (f_R) trains at n ∈ {256, 512, 1024}. Bar, registered now: (f_R)
   ≥ 0.95 at n = 16k after that training. The single-length fit is known to fail
   (0.2551).
5. **γ = 0.999**, fixed (inside the window at 4k and 16k).
6. **Numerics bar (Chase's replacement for γ·ρ(W)):** γ·max row-sum(W) < 1 and the read
   of V = 1 within 1 ± 1e−5, with the resolvent layer in fp32 outside autocast, on every
   R0 checkpoint.
7. **BOS.** Count BOS's share of each (f_R) head's read on the R0 checkpoints; bar:
   < 0.5 in the resolvent layer. If it fails, the per-head γ_h route is run next,
   registered before it runs.
8. **MFU.** Re-measured at R0 with the bar written before the sweep. Wilson's 14 logs
   (0.163–0.171 at R0) are the prior.
9. Any run compared bitwise to another uses `--deterministic`.

Kills stay as the contract states: (f_R) < 0.8 at the training depth is a learnability
kill.
