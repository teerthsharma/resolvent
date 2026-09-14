# PI-JEPA KAGGLE CARD

A heuristic beat the model. On 13,388 K+Q-vs-K positions taken from the same Lichess
corpus the encoder was trained on, a policy that plays the move minimising the bare
king's legal-move count — the box-shrinking heuristic, arm 2 — matched the move the
human actually played 0.3401 of the time, 95% Wilson interval [0.3321, 0.3482]. The
trained per-coordinate pi-JEPA probe, arm 3, matched 0.2882, [0.2806, 0.2959]. The
intervals are disjoint. The frozen-random encoder read through the identical probe
path, arm 4, matched 0.3279, [0.3200, 0.3359], and its interval is also disjoint from
the trained arm's: the trained encoder lost to the heuristic by 0.0519 and to its own
untrained control by 0.0397. Both statements survive restriction to the 910 positions
drawn from games the GPU run never saw.

One thing went the other way, and it is why this card is not a single sentence. On
the metric the kernel was built to decide — a linear probe's cross-entropy over the
human's from-square and to-square — the trained encoder bought 0.5997 nats of the
8.3178 available, against the frozen-random encoder's 0.2348. That is a win for
training on the probe's own loss, and it does not reach top-1 accuracy, where every
arm at every evaluated length sits below the single most frequent square.

## Provenance

| item | value | producing command |
| --- | --- | --- |
| kernel id | `melowdramtic/ceq-pi-jepa-chess-t4` | `cat kaggle_pkg/kernel-metadata.json` |
| kernel title | `CEQ pi-JEPA chess: trained vs frozen random` | same file |
| kernel type, accelerator | script, `NvidiaTeslaT4`, internet on | same file |
| kernel dataset | `thomaswesthead/120000-chess-games-lichess` | same file |
| kernel wall clock | RUNNING `2026-09-13T23:33:15Z` to COMPLETE `2026-09-14T00:01:35Z`; last `SAVED` line at t = 1716.27 s | `cat kernel_status.log`; tail of `ceq-pi-jepa-chess-trained-vs-frozen-random.log` |
| commit the kernel ran | `391a2d004fb5e77aa91febdfdb91c79a1a8db7bf` (`391a2d0`) | `/provenance/commit` in `pi_jepa_chess_results.json`; the log's first line prints the same SHA |
| kernel device | `Tesla T4`, `cuda`, torch `2.10.0+cu128`, CUDA `12.8`, Python `3.12.13`, numpy `2.0.2` | `/provenance/*` |
| commit this card was written at | `b4c6620` | `git rev-parse --short=7 HEAD` |
| local node | `WIN-16QAL06O9GB` | `python -c "import platform;print(platform.node())"` |
| local machine id | `60b8cf943ee0` | `python -c "import platform,hashlib;print(hashlib.sha256(platform.node().encode()).hexdigest()[:12])"` |
| local platform | `Windows-10-10.0.26200-SP0`, Python `3.11.9`, numpy `2.4.6`, torch `2.14.0+cpu`, python-chess `1.11.2` | `python -c "import platform,sys,numpy,torch,chess; print(...)"` |

Every cell below is read either from `pi_jepa_chess_results.json`, the file the T4
kernel wrote, by the key path given in its row, or from the stdout of a command
listed in **The runs**. No cell is transcribed from prose.

## The runs

| label | command | cwd | exit | wall clock | last line |
| --- | --- | --- | --- | --- | --- |
| K1 | the Kaggle script kernel above | `/kaggle/working` | COMPLETE | 1716.27 s to the last `SAVED` | `trained  acc_exact 0.0000  from 0.0123  to 0.0288  majority from 0.0247  NRMSE 6.8498` |
| L1 | `python score.py` | scratchpad `humanmoves/` | 0 | 44 s | `total 41.8s` |
| L2 | `python repro_eval.py` | scratchpad `humanmoves/` | 0 | 322 s | `total 319s` |
| L3 | `python -m pytest tests/curvature/test_chess_probe_protocol.py -q` | repo root | **1** | 30 s | `8 failed in 23.85s` |
| L4 | `python -c "from ceqjepa import pi_assign as pa; print(pa.FROZEN_MASK_DIGEST)"` | repo root | 0 | under 1 s | `854dc3aae168dd98e0e631e189aa30c0800cd8d1a61e69433b2b2ea21fad504a` |
| L5 | `grep -cniE "dtm|kqk|endgame|tablebase" kaggle_pkg/pi_jepa_chess.py` | scratchpad | 0 | under 1 s | `0` |

L1 and L2 read the checkpoint `pi_jepa_chess_weights.pt` that K1 wrote, on CPU,
through `load_state_dict(..., strict=True)`. L3 is red by intent and is the subject
of the last two limits.

**The cross-device control.** L2 rebuilds K1's own L = 16 evaluation on this machine
from the T4's weights and prints both numbers side by side. It read 43,714 games
(41,498 kept, 2,216 short, 0 illegal) into 4,096 held-out first windows and produced,
with the kernel's own figure in parentheses:

```
arm             from      to        exact       (kaggle in parentheses)
trained         0.0823(0.0823) 0.0608(0.0608) 0.0056(0.0056)
frozen_random   0.0437(0.0437) 0.0388(0.0388) 0.0039(0.0039)
untrained       0.0403(0.0403) 0.0464(0.0464) 0.0059(0.0059)
```

Nine cells of nine agree to four decimal places across a T4 and a CPU, so the local
reading path used in L1 is the reading path K1 used. Without this control the
human-move table would be a cross-machine number of the kind this repository has been
throwing away.

---

## Row 1 — WHAT WAS TRAINED, ON WHAT DATA, UNDER WHICH MASK

Source: K1, key paths `/config`, `/corpus`, `/split`, `/mask`, `/arms/*`.

| item | value | key path |
| --- | --- | --- |
| corpus file | `lichess_db_standard_rated_2013-01.pgn` | `/corpus/path` |
| games kept | 115,628 | `/corpus/n_games` |
| plies | 8,089,783 | `/corpus/n_plies` |
| dropped for length (under 20 plies) | 5,704 | `/corpus/dropped_short` |
| dropped for an illegal move | 0 | `/corpus/dropped_illegal` |
| corpus read time | 1,193.456 s | `/corpus/load_seconds` |
| split | by game, fracs (0.90, 0.10, 0.00), 8,089,783 position rows checked | `/split` |
| train, holdout games | 104,132 and 11,496 | `/split/n_train_games`, `/split/n_holdout_games` |
| observation width | 769 = 12 piece planes x 64 squares + side to move | `/config`, kernel header |
| encoder hidden width | 1,024 | `/config/hidden` |
| context S, horizon h | 16 and 4 | `/shipped_constants/S_LEN`, `/shipped_constants/HORIZON` |
| JEPA steps, batch | 3,000 steps x 512 games per arm | `/config/steps`, `/config/batch` |
| probe steps, batch, lr | 1,500 x 512 at 0.01, encoder and predictor frozen | `/config/probe_steps`, `/config/probe_batch`, `/config/probe_lr` |
| seed | 5501 | `/config/seed` |
| optimisation constants | LR 0.003, TAU 0.99, LAM 25.0, MU 25.0, NU 100.0 | `/shipped_constants` |
| collapse floors | std_min 0.001, effective rank 1.5 | `/shipped_constants` |

**The frozen mask.** Axis `D`, `axis_len` 9, `n_columns` 7, digest
`854dc3aae168dd98e0e631e189aa30c0800cd8d1a61e69433b2b2ea21fad504a`
(`/mask/digest`). The kernel's second printed line reads
`matches_FROZEN_MASK_DIGEST=True`, and L4 recomputes the constant in this tree at
`b4c6620` and returns the same 64 hex characters, so the mask that trained on the T4
is the mask this repository ships.

| index | coordinate | beta | status |
| --- | --- | --- | --- |
| 0 | `mean_value` | 1.0 | assigned |
| 1 | `prob_above` | 1.0 | assigned |
| 2 | `ratio_vu` | 1.0 | assigned |
| 3 | `count_above` | 0.0 | assigned |
| 4 | `sum_value` | 0.0 | assigned |
| 5 | `total_u` | 0.0 | assigned |
| 6 | `half_walk` | unprinted, refused | `COORDINATE-REFUSED-BY-R1` |
| 7 | `osc_shape` | unprinted, refused | `COORDINATE-REFUSED-BY-R1` |
| 8 | `log_shape` | 1.0 | assigned |

Two of the nine coordinates carry a refusal in place of a number, so the read is
7 columns wide: `/arms/trained/cols` is `[0, 1, 2, 3, 4, 5, 8]`. The refusals crossed
into the kernel as values, not as NaN and not as a silently shortened axis.

**The three arms, and how they differ.** All three carry 3,694,867 parameters
(`/arms/*/n_params`). The trained arm optimises 1,847,562 of them
(`/arms/trained/n_trainable`); the frozen-random arm optimises 257, the predictor
alone (`/arms/frozen_random/n_trainable`); the untrained arm receives no JEPA fit at
all (`/arms/untrained/fitted` is `false`, `/arms/untrained/seconds` is 0.0). The
trained encoder moved and the frozen one did not: `encoder_delta` 1.0669680833816528
against exactly 0.0.

**The guards held.** Preflight over real windows: std_min 0.006457493762014507
against a 0.001 floor, effective rank 7.99610496640477 of 9 against a 1.5 floor
(`/preflight`). Across 3,000 checked steps neither arm tripped either leg:
`/arms/trained/collapsed_at` and `/arms/frozen_random/collapsed_at` are both `null`,
`erank_min` 2.0258612355588115 trained and 7.509285913893922 frozen, `std_min_last`
0.9560204811337765 and 0.005861211607431702. Fit time 213.84 s and 224.05 s.

**VERDICT: void.** This row scores no claim. The only metric internal to it is the
JEPA latent loss, and that loss is circular, because the target is the encoder's own
moving representation. Its last values, 2.496544361114502 trained against
0.00020781744387932122 frozen (`/arms/*/loss_last`), rank the frozen arm first
because its target never moved: target RMS 0.046557 against the trained arm's
6.368760 at L = 16, a factor of 136.7950. A metric an encoder wins by shrinking
cannot rank encoders, and nothing in this row is reported as though it could.

---

## Row 2 — LENGTH GENERALISATION

Source: K1, key path `/arms/<arm>/eval/<length>`. Training length 16, evaluated at
16, 64 and 128, which are 1x, 4x and 8x.

| arm | params | trainable | L | n scored | acc from | majority from | acc to | majority to | acc exact | NRMSE | target RMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trained | 3,694,867 | 1,847,562 | 16 | 4,096 | 0.082275 | 0.094238 | 0.060791 | 0.075439 | 0.005615 | 0.353092 | 6.368760 |
| trained | 3,694,867 | 1,847,562 | 64 | 4,096 | 0.025635 | 0.039795 | 0.027344 | 0.031250 | 0.000732 | 1.839072 | 5.843023 |
| trained | 3,694,867 | 1,847,562 | 128 | 486 | 0.012346 | 0.024691 | 0.028807 | 0.022634 | 0.000000 | 6.849839 | 4.962314 |
| frozen_random | 3,694,867 | 257 | 16 | 4,096 | 0.043701 | 0.094238 | 0.038818 | 0.075439 | 0.003906 | 0.341025 | 0.046557 |
| frozen_random | 3,694,867 | 257 | 64 | 4,096 | 0.032471 | 0.039795 | 0.027588 | 0.031250 | 0.003418 | 2.501029 | 0.047297 |
| frozen_random | 3,694,867 | 257 | 128 | 486 | 0.026749 | 0.024691 | 0.028807 | 0.022634 | 0.002058 | 5.951863 | 0.046810 |
| untrained | 3,694,867 | 1,847,562 | 16 | 4,096 | 0.040283 | 0.094238 | 0.046387 | 0.075439 | 0.005859 | 4.509396 | 0.046557 |
| untrained | 3,694,867 | 1,847,562 | 64 | 4,096 | 0.015381 | 0.039795 | 0.024658 | 0.031250 | 0.000732 | 19.363043 | 0.047297 |
| untrained | 3,694,867 | 1,847,562 | 128 | 486 | 0.014403 | 0.024691 | 0.022634 | 0.022634 | 0.000000 | 38.827464 | 0.046810 |

Uniform chance is 0.015625 per square (`/arms/*/eval/*/uniform_chance`), and 1/4096 =
0.000244 for both squares at once. The majority column is the single most frequent
square in train at the ply the evaluation scores, from `/majority_control`: ply 15,
from-square 60, over 104,132 games; ply 63, from-square 62, over 53,583; ply 127,
from-square 37, over 5,419. A joint majority control is not printed by the run and
its cell is therefore empty here, which is the fourth red in L3.

**The probe cross-entropy leg.** Uniform chance over two 64-way heads is
2 ln 64 = 8.3178 nats. Final probe loss (`/arms/*/probe_loss_last`) and the nats it
buys, both to four decimals:

| arm | probe loss at step 1499 | nats bought |
| --- | --- | --- |
| trained | 7.718047618865967 | 0.5997 |
| frozen_random | 8.083013534545898 | 0.2348 |
| untrained | 8.078201293945312 | 0.2396 |

**VERDICT, three legs, because the row has three.**

- Probe cross-entropy at the training length: **win**, to the trained encoder,
  0.5997 nats against 0.2348. One run, one seed, no interval.
- Top-1 accuracy against the majority-square control: **win**, to the majority
  square. On the from-square the trained arm sits below the control at every length,
  by −0.011963, −0.014160 and −0.012346. Across the from-square and to-square cells
  together, seven of nine are negative; the two positives are the to-square at
  L = 128 on n = 486, +0.006173 for the trained and the frozen arm alike, both
  reading exactly 0.028807.
- Length generalisation as such: **void**. The evaluation does not vary length alone.
  The scored label sits at ply 15, 63 and 127 (`/majority_control/*/ply`), so a move
  from 1x to 8x is a move of game phase; and the scored population differs at each
  length, 11,496, 5,270 and 486 holdout games (`/split/holdout_games_by_length`).
  NRMSE cannot arbitrate either, because its denominator is per-arm: target RMS
  ratios of trained to frozen are 136.7950, 123.5390 and 106.0104 at the three
  lengths.

---

## Row 3 — THE HUMAN MOVE

Source: L1, `python score.py`, exit 0, 44 s. Target: the move the human actually
played. Position set: every K+Q-vs-K position in the full
`lichess_db_standard_rated_2013-01.pgn.zst` carrying 16 plies of history behind it.
121,332 games and 8,155,187 plies were scanned, 13,479 rows emitted and 0 dropped as
short; 13,388 of those carry an exact distance-to-mate label from the repository's
own enumeration and 91 are excluded as `drawn-by-defence`. Controls printed by the
same run: the fast position key agrees with `chess_steps._key_of` on 300 of 300
sampled positions, and the enumeration is 368,452 positions. Legal moves per
position: mean 15.65, median 18, min 1, max 35, so uniform-random chance is 0.0639.

**The run emits four arms, not three.** Arm 1 is the tablebase ceiling rather than a
competitor: it is what an optimal engine plays, scored against what a human played.

| arm | what it plays | hits / n | accuracy | 95% Wilson |
| --- | --- | --- | --- | --- |
| 1 `dtm_optimal` | the enumeration's optimal move | 5179/13388 | 0.3868 | [0.3786, 0.3951] |
| 2 `box_shrink` | fewest legal moves for the bare king, ties by dtm | 4553/13388 | 0.3401 | [0.3321, 0.3482] |
| 3 `trained` | trained pi-JEPA read, argmax over legal moves | 3858/13388 | 0.2882 | [0.2806, 0.2959] |
| 4 `frozen_random` | frozen-random read, identical path | 4390/13388 | 0.3279 | [0.3200, 0.3359] |

By side to move:

| arm | attacker to move, n = 6713 | 95% Wilson | defender to move, n = 6675 | 95% Wilson |
| --- | --- | --- | --- | --- |
| 1 `dtm_optimal` | 1421/6713 = 0.2117 | [0.2021, 0.2216] | 3758/6675 = 0.5630 | [0.5511, 0.5749] |
| 2 `box_shrink` | 1155/6713 = 0.1721 | [0.1632, 0.1813] | 3398/6675 = 0.5091 | [0.4971, 0.5210] |
| 3 `trained` | 345/6713 = 0.0514 | [0.0464, 0.0569] | 3513/6675 = 0.5263 | [0.5143, 0.5383] |
| 4 `frozen_random` | 441/6713 = 0.0657 | [0.0600, 0.0719] | 3949/6675 = 0.5916 | [0.5798, 0.6033] |

By whether the GPU run saw the game, under the same
`ceq.kdata.split_by_game(fracs=(0.90, 0.10, 0.00))` over 844 games:

| arm | held out, n = 910 | 95% Wilson | trained on, n = 12478 | 95% Wilson |
| --- | --- | --- | --- | --- |
| 1 `dtm_optimal` | 393/910 = 0.4319 | [0.4000, 0.4643] | 4786/12478 = 0.3836 | [0.3751, 0.3921] |
| 2 `box_shrink` | 327/910 = 0.3593 | [0.3288, 0.3910] | 4226/12478 = 0.3387 | [0.3304, 0.3470] |
| 3 `trained` | 264/910 = 0.2901 | [0.2616, 0.3204] | 3594/12478 = 0.2880 | [0.2801, 0.2960] |
| 4 `frozen_random` | 303/910 = 0.3330 | [0.3031, 0.3642] | 4087/12478 = 0.3275 | [0.3194, 0.3358] |

Two readings that are not the headline and are reported because they are what makes
the headline mean anything. On defender-to-move positions the frozen-random arm,
0.5916 [0.5798, 0.6033], stands above the tablebase itself, 0.5630 [0.5511, 0.5749]:
the bare king has few legal moves and a near-constant predictor does well there, so a
high number on that slice is not evidence of understanding. On attacker-to-move
positions, where the choice is wide, the trained arm reads 0.0514 [0.0464, 0.0569],
below the frozen-random control's 0.0657 [0.0600, 0.0719] and below the box-shrinking
heuristic's 0.1721 [0.1632, 0.1813] by a factor of 3.3478.

**VERDICT: win, to the box-shrinking heuristic.** 0.3401 [0.3321, 0.3482] against the
trained arm's 0.2882 [0.2806, 0.2959], intervals disjoint, on the pooled set and on
the held-out subset alike. The frozen-random control also beats the trained arm,
0.3279 [0.3200, 0.3359], intervals disjoint. The count of arms the trained encoder
beats is zero.

---

## What is cited rather than claimed

- **The single-softmax length failure is prior art, and its arm is calibration rather
  than a finding.** That a single softmax readout must disperse as the item count
  grows is Velickovic et al., arXiv:2410.01104, ICML 2025; the representational
  collapse and counting route through a decoder stack is Barbero et al.,
  arXiv:2406.04267, NeurIPS 2024, Theorem B.3 with Corollary B.10 and Proposition
  B.9. In this repository the single-exponent arms live only in
  `ceqjepa/t_length.py`, where they are labelled calibration instruments that
  replicate a theorem someone else proved. **This Kaggle run carries no
  single-exponent arm at all**: all three of its arms use the same frozen
  per-coordinate mask, and `/arms/*/cols` is `[0, 1, 2, 3, 4, 5, 8]` in every case.
  Nothing in this card replicates or tests that prior art.
- The normalised-versus-unnormalised attention axis is Katharopoulos et al.,
  arXiv:2006.16236; the length-generalisation line is Press et al., arXiv:2108.12409.
- Every unscoped "cannot" about softmax is bounded by arXiv:2511.20038, Theorem 4.3:
  length-generalizable softmax chain-of-thought transformers with relative positional
  encodings are Turing-complete.
- The nearest prior work to a per-coordinate exponent is Bakarji et al.,
  arXiv:2202.04643, Nature Computational Science 2022.
- These characterisations are relayed from this repository's own prior-art pass in
  `ceqjepa/t_length.py` and `ceqjepa/pi_assign.py`. No run in this card verifies them.

## Limits

- **The training covered no endgame target.** The kernel filters no material and
  loads no distance-to-mate label: L5 returns 0 hits for `dtm`, `kqk`, `endgame` and
  `tablebase` over the staged `pi_jepa_chess.py`. Row 3 sits on 13,388 positions out
  of 8,155,187 plies scanned, a fraction of 0.001642, and none of it is a statement
  about a target the training optimised.
- **Row 3's position set is mostly seen data.** 12,478 of 13,388 positions come from
  games the GPU run trained on; 910, a fraction of 0.0680, are held out. The headline
  reproduces on that held-out subset with wider intervals.
- **Opening-trunk contamination is measured and not removed.** 120,775 of 183,936
  holdout first-16-ply positions also occur in train, a fraction of 0.656614
  (`/holdout_fen_overlap`). The L = 16 accuracies in Row 2 are one pooled number over
  both strata, and the run prints no split of them.
- **The 8x column is thin.** 486 holdout games are long enough to fill a 132-ply
  window, and 486 is the n behind every L = 128 cell.
- **No Kaggle row carries an interval.** K1 is one run at one seed, 5501, on one
  device. Only Row 3, produced locally, carries Wilson intervals.
- **NRMSE is not comparable across arms.** Target RMS differs between the trained and
  frozen arms by factors of 136.7950, 123.5390 and 106.0104 at the three lengths.
- **The action channel is unused.** `N_ACTIONS` is 4 and `None` is passed throughout;
  this bed carries no intervention, so nothing here is a causal claim.
- **Two coordinates of nine are refused**, so every result is a property of a
  7-column read and not of the 9-dimensional latent.
- **The evaluation protocol is red, and the red is open.** L3 exits 1 with 8 failed
  of 8 collected. One verbatim line each:
  `the evaluation scores a different ply at every length: {16: [15], 64: [63], 128: [127]}`;
  `a different holdout population is scored at each length: {16: 240, 64: 180, 128: 120}`;
  `the probe's fit distribution over label plies is uniform over the game; the evaluation scores a point mass at ply L-1. fraction of probe training labels landing on each evaluated ply: {16: 0.00555, 64: 0.00595, 128: 0.00565}`;
  `acc_exact is printed as the verdict and the only control beside it is majority_from, a marginal`;
  `fen_overlap measures the contaminated fraction and evaluate() never sees it; the 1x accuracy is one pooled number over both strata`;
  `over 153 holdout positions the head predicts 3 distinct from-squares and 1 distinct to-squares; the single most-predicted from-square covers 0.8758 of them`;
  `the read rescales with the context length on identical rows: mean|read| {16: 0.4138384759426117, 64: 1.537669062614441, 128: 3.1158440113067627}, ratio to 1x {16: 1.0, 64: 3.715626148854449, 128: 7.529130790001379}`;
  `after 1500 steps the probe cross-entropy is 8.1351 against a uniform-chance 8.3178: the read buys 0.1827 nats out of the 8.3178 available`.
  The first three are structural and hold for K1 exactly as printed: K1's own
  `/majority_control/*/ply` is 15, 63 and 127, and its own
  `/split/holdout_games_by_length` is 11,496, 5,270 and 486.
- **Five of those eight run at smoke size, not at the kernel's size.** The fixture
  sets `cfg.hidden = 32`, `cfg.max_games = 1500` and `cfg.batch = 8`, against K1's
  `/config/hidden` 1024, `/config/max_games` 120000 and `/config/batch` 512, and it
  reads a 6 MB PGN prefix rather than the whole file. That test file's docstring
  states it "runs the shipped width"; the fixture does not. Its figures — the 8.1351
  cross-entropy, the 0.8758 constant-prediction share, the 7.5291 rescaling ratio —
  are properties of a hidden-32 pipeline and do not transfer to Row 2, whose
  hidden-1024 counterpart on the cross-entropy is the 7.718047618865967 above.
- **This card changes no code and repairs no red.** L3 is recorded as it stands.

## The honest sentence

In one Kaggle T4 run at seed 5501 over 115,628 Lichess games, the trained
per-coordinate pi-JEPA encoder's linear probe bought 0.5997 nats of the 8.3178
available against a frozen-random encoder's 0.2348, and still predicted the human's
moved-from square less often than the single most frequent from-square at every
evaluated length; and on 13,388 K+Q-vs-K positions from the same corpus a heuristic
minimising the bare king's legal-move count matched the human's played move 0.3401 of
the time, [0.3321, 0.3482], against that trained probe's 0.2882, [0.2806, 0.2959].

Nothing else measured here is scored.
