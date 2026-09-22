# FOREMAN K0: the R-DEPTH shortcut

**Verdict: a shortcut exists, and the 2^L line is not a ceiling.** A hand-set 6-layer causal construction scores **0.3982** on the pinned wald.py bed. The pinned L = 6 line is **0.2539**. The construction uses real softmax heads at the R1 width, and its pointers match the idealized construction exactly (0 mismatches).

The counter as written is a different question: it asks for (a_L) > 0.5 on tokens deeper than 2^L.
- At R1's slot budget the construction reaches 0.1934 (anchored window) or 0.3276 (centered window), so the counter does **not** fire there.
- It crosses 0.5 once the window reaches 48 slots (576 dims): 0.5596 on the pinned bed, and 0.537–0.576 on 8 beds with the bed_k structure.
- So the bed is void as pinned. It uses ids equal to positions, and it scores against the 2^L ceiling. The counter's 0.5 line holds only at R1 width.

**Lower bound: none that covers this bed.** Every bound read is conditional, asymptotic, or outside this bed's regime. None pins a constant.

Bed used: the pinned `wald.py` (sha256 31271d64…), `make_bed(4096, 16)`, seed 23, executed untouched. The target is the root id for every token. The 2^L count starts from wald.py's "layer 1 = 2 hops".

## 1. Constructions tried

Every construction is a per-layer information-flow simulation of a causal transformer. In one layer, a head fetches one token's pre-layer state at an address known before the layer, and the MLP combines what was fetched. Widths are priced: one pointer is 12 bits.

**C0. Pointer doubling (the author's best response).** The update is A ← A[A].
- The explicit simulation reproduces 0.06640625 / 0.25390625 / 0.985595703125 at L = 4 / 6 / 8.
- It scores 0.0000 on depth > 2^L.

**C1. Leak probes (positional or local features, no chain following).** Scored on depth > 64 (n = 3,056). The bar was 1/16 + 3 SE = 0.0763.

| Probe | depth > 64 |
|---|---|
| Most recent root | 0.0566 |
| Position mod 16 | 0.0560 |
| Local count table: (pos mod 16, own gap, parent gap) → root rank, fit on 40 fresh beds | 0.0625 |

**No leak.** Every probe scores at chance, which matches exchangeability: lane labels are i.i.d. and independent of position.

**C2. Hybrid content-plus-window pointer jumping (the shortcut).**

What a token t knows before the layer is v1 = A(t). The layer then does three things:
- One content head fetches v2 = A(v1). This is plain doubling.
- W positional heads fetch A(u) for the u in a window near v1. wald.py's ids *are* positions, so "the token whose id is v1 − j" is an exact content lookup.
- A ReLU multiplexer then picks the slot at offset v1 − v2, which gives v3 = A(v2). The token gets a triple jump whenever v2 falls in the window.

Width: attention 12W dims, MLP about 12W for k = 3. The k = 4 variant chases one more hop at an MLP cost of W²·12.

There are two window placements:
- **Anchored:** [v1 − W, v1).
- **Centered:** W slots around v1 − 16·hops(t). Hop counts are additive, so a token can track them.

Why it works: each hop spans a Geom(1/16) number of positions, so ancestors are positionally local.

Pinned bed, L = 6, all tokens / depth > 64:

| W (attention dims) | anchored k = 3 | centered k = 3 | centered k = 4 (MLP W²·12) |
|---|---|---|---|
| 8 (96) | 0.3064 / 0.0704 | 0.3232 / 0.0929 | 0.3376 / 0.1122 |
| 26 (312), R1 = 320 | **0.3982 / 0.1934** | **0.4983 / 0.3276** | 0.5547 / 0.4031 (8,112 hidden) |
| 32 (384) | 0.4241 / 0.2281 | 0.5713 / 0.4254 | 0.6658 / **0.5520** (12,288 hidden) |
| 48 (576) | 0.4822 / 0.3060 | 0.6714 / **0.5596** | 0.8186 / 0.7569 |
| 128 (1,536) | 0.6697 / **0.5573** | 0.9783 / 0.9709 | 1.0000 / 1.0000 |
| 512 (6,144) | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |

The same construction at other depths, rungs and beds:
- **L = 4 (R0).** Doubling scores 0.0664. At W = 8 the construction scores 0.0784 anchored and 0.0813 centered; at W = 16, 0.0911 and 0.1008.
- **Replication, 8 fresh beds at depth > 64.** W = 26 anchored: 0.175–0.201. W = 64 centered: 0.670–0.758. W = 128 centered: 0.950–0.968.
- **Real softmax heads.** 27 causal heads, d_head = 12, ±1 binary id codes, β = 4, float64, with an explicit ReLU multiplexer. **0 pointer mismatches** against the idealized run across all 6 layers. Accuracy 0.398193 / 0.193390, minimum decode margin 0.992.
- **Cameron's K1 bed structure** (`bed_k.make_test`, imported read-only), 8 beds per n, depth > 64, zero credit. This assumes a positional channel exists:

| n | centered W = 26 | anchored W = 26 | centered W = 48 |
|---|---|---|---|
| 4,096 | 0.318 | 0.187 | 0.560 |
| 8,192 | 0.140 | 0.081 | 0.238 |
| 16,384 | 0.066 | 0.037 | 0.114 |

  Cameron's stated best response beyond 2^L, 0.041–0.068, is therefore not the best response whenever a positional channel exists.

**C3. Ladder rungs (R-SCALE).** Bed: n = 16,384, 16 lanes, seed 3000. The slot budget is W = d // 12, which gives 10 / 26 / 42 slots for R0–R2. D_c is the number of depths whose accuracy is at least 0.5.

| Arm | D_c at R0 / R1 / R2 | log-slope vs 2^L |
|---|---|---|
| Doubling | 17 / 65 / 257 | 0.980 |
| Centered hybrid | 22 / 128 / 649 | **1.221** |
| Anchored hybrid | 21 / 101 / 470 | 1.121 |

The centered slope clears R-SCALE's counter line of 1.2, but only narrowly.

**C4. Dilated bed (more lanes).** n = 16,384, seed 3001, L = 6, W = 26, centered. Ratio of the construction's all-token accuracy over doubling:
- 16 lanes: 1.918×
- 64 lanes: **1.220×**
- 128 lanes: 1.109×

The k = 4 ratios are 2.206× / 1.232× / 1.115×. With 64 lanes, the depth > 64 score falls to 0.0749.

**Considered and not built:**
- **Move-to-front rank tracking over S_16 with a K-ary parallel prefix.** Ranks are countable in O(1) layers. But depth is 2 + ⌈log_K 4096⌉ + 1, so L = 6 needs K = 16, i.e. an MLP composing 16 permutations. That costs about 16^16 width, so this route is dominated by C2.
- **Extra heads on lower-level pointers.** The sumset of known jump lengths at most doubles per layer. Only positional windows escape this.
- **Exclusion or local-statistics guessing on unresolved tokens.** C1 shows this is chance. The only gain is chance credit (Cameron, 0.2988 at L = 6).

## 2. Lower-bound status

The primary sources were fetched by nurses; the quotes and theorem numbers are theirs.

- **Sanford–Hsu–Telgarsky 2024** (arXiv 2402.09268, read at arxiv.org/html/2402.09268).
  - Upper bound (Thm 4.2): depth ⌊log₂ k⌋ + 2, m = O(1), H = 1.
  - Lower bound (Cor. 4.3): **conditional** on Conjecture 2.4, one cycle vs two cycles in MPC. It applies only for k = Θ(N^ξ) with ξ ∈ (0, 1/2] and m·H = O(k^{1−ε}), and gives L = Ω(log k).
  - Does not cover this bed. Here k = 269 ≈ N^0.67 exceeds the stated range. The hybrid's width, 312 or more, exceeds k^{1−ε}. The Ω carries no constant.
  - Their unconditional results cover multi-layer RNNs and one-layer transformers with chain of thought only.
  - §4.2 and App. G.4: trained models learn the doubling construction.
- **Chen–Peng–Wu 2024** (arXiv 2412.02975, read at arxiv.org/html/2412.02975v1).
  - Thm 1.1 is **unconditional**, but it covers L-sequential function composition laid out as z_{L−1}…z_0 followed by the query. That layout defeats causal composition, and an O(log L)-layer encoder solves it (Cor 1.3).
  - Its condition HdpL ≤ n^{2^{−4L}} is vacuous at L = 6.
  - Does not cover this bed. Here parents precede children, so causal doubling works.
- **Peng–Narayanan–Papadimitriou 2024** (arXiv 2402.08164, Thm 1). This is a one-layer bound only (H(d+1)p < n log n), and the authors say it "break[s] down for multiple layers".
- **Merrill–Sabharwal 2025** (arXiv 2503.03961v3, Thm 2). ⌈log₂ n⌉ unrolled layers solve connectivity. Fixed depth fails "under common complexity conjectures" (TC⁰ vs L), at log precision. The result is asymptotic.
- **Liu et al. 2023** (arXiv 2210.10749, Thm 4). Non-solvable semiautomata cannot be simulated at constant depth unless TC⁰ = NC¹.
  - A reduction of this bed to that setting is my own and unsourced: the block-synchronous sub-family, where every block of 16 tokens holds one token per lane, encodes an arbitrary S_16 word locally.
  - It is asymptotic and worst-case. The i.i.d. bed puts negligible mass on that sub-family.
- **Saparov et al. 2024** (arXiv 2412.04703). Trained transformers learn "exponential path-merging". Their task has no positional locality for the hybrid to exploit.

**Status:** no lower bound read covers this bed at its n, k and widths. No source gives a constant. The 2^L line is also beaten outright, at every window of 8 slots or more.

## 3. Verdict on R-DEPTH's counter

- The clause "(a_L) stays under its doubling ceiling (0.254 at L = 6)" is **struck**. Two independent routes break it: Cameron's chance credit gives 0.2988, and this construction gives 0.3982 with 27 real heads at R1 width.
- The counter "(a_L) > 0.5 beyond 2^L":
  - Does **not** fire from any construction here at R1's slot budget (best: 0.3276 centered, k = 3).
  - **Does** fire at 48 slots centered (576 dims), or at 32 slots with the k = 4 multiplexer (384 attention dims, 12,288 MLP hidden units).
  - Is n-diluted: the same W = 48 construction scores 0.560 at 4k but 0.114 at 16k.
- The bed as pinned is therefore void as a ceiling statement. The counter's 0.5 threshold is safe only at R1 width, and only if the twin has no positional channel.

## 4. RED path and exact RED line per finding

All tests live in `SP/phase_k/K0/foreman/`. Every bar was written before its result, and the board holds each red before its green.

| Test (board name) | File | Exact RED line | Green |
|---|---|---|---|
| foreman.k0.doubling_sim | test_shortcut.py | `FileNotFoundError: [Errno 2] No such file or directory: '…\foreman\results.json'` | 01:52:46 |
| foreman.k0.no_local_leak | test_shortcut.py | same line (red 01:48:36) | 01:52:46 |
| foreman.k0.hybrid_beats_2L | test_shortcut.py | same line | 01:52:46 |
| foreman.k0.hybrid_at_R1_width | test_shortcut.py | same line | 01:52:46 |
| foreman.k0.rscale_width_confound | test_shortcut2.py | `FileNotFoundError: … 'results2.json'` (red 01:53:27) | 01:54:37 |
| foreman.k0.dilated_bed | test_shortcut2.py | same line | 01:54:37 |
| foreman.k0.softmax_heads_exact | test_shortcut3.py | `FileNotFoundError: … 'results3.json'` (red 01:55:05) | 01:57:17 |
| foreman.k0.id_range_gap | test_idrange.py | declared stub; no failing run was observed | PASS own 0.9375, parent 0.9374 |
| foreman.k0.bedk_best_response | test_bedk.py | declared stub; no failing run was observed | PASS 0.3181 > 0.068 |

The clock times in the headers of test_shortcut2.py ("01:56") and test_shortcut3.py ("01:57") are estimates, and they run late. The record is the file mtimes (01:53:19 and 01:55:05) and the board timestamps. Both precede their results.

## 5. Replacement route (the bed dies as pinned)

1. **Ids.** Use Cameron's `bed_k`: n distinct ids drawn from V = 32,768, with a NULL parent for roots, at both training and test time.
   - This closes the exact positional addressing the hybrid needs.
   - It also closes the train/test id gap: under wald.py's convention, 93.75% of test ids at 16k never appear when training at n = 1,024.
   - The (a_L) twin must then carry **no** absolute-position channel. If any arm gets one, its ceiling is the C2 table.
2. **Ceiling line.** Replace (depth ≤ 2^L).mean() with the C2 hybrid at the twin's own slot budget (d // 12), plus chance credit.
3. **Counter.** Restate it on D_c: the twin fires only if D_c(a_L) exceeds the hybrid's D_c at the same (L, d). On n = 16k with 16 lanes, that is 22 / 128 / 649 for R0–R2 (centered).
   - "Accuracy beyond 2^L" is n-diluted and should not carry the counter.
4. **R-SCALE.** The ladder's width growth makes the counter fire by construction (slope 1.221). Fit D_c against 2^L at a fixed slot budget, or against the hybrid's width-priced D_c(L, d).
5. **Optional dilation.** 64 lanes at n = 16,384 cut the hybrid's gain from 1.92× to 1.22×. Depth is then capped near 290, which is fine for R0/R1 (L = 4, 6) and gives no room for L = 8.

## OPEN

- **Arithmetic not built at weight level.** The per-token 12-bit arithmetic is idealized: code(v1 − j), one-hot(t − A(t)), and the centered offset 16·hops. The softmax check covers the heads and the multiplexer only.
- **Positional channel on bed_k.** With random ids, an ALiBi-only twin would have to synthesize absolute positions, for example 1/(t+1) from uniform attention. Whether it can do so cheaply enough to regain C2 is unmeasured.
- **Trained twin.** Whether a trained (a_L) finds C2 at all is unknown; that is K1's read.
- **S_16 reduction.** The NC¹-hardness reduction is unsourced and only asymptotic.
- **No constant-level bound.** No lower bound at the constant level exists in anything read.
- **Narrow margins.** T5 (1.2207 against 1.2) and T6 (1.2202 against 1.25) both pass by small margins.

**Standing state.** The shortcut is real and realized with softmax heads; the pinned ceiling and the "under 0.254" clause are struck. Cameron's `bed_k` (random ids) is the replacement bed, and his "beyond-2^L best response 0.041–0.068" holds only while the twin has no positional channel. Default for K1: run R-DEPTH on `bed_k`, with an ALiBi twin that has no absolute position embedding, and price the ceiling and the counter by the C2 table and D_c at the twin's width.
