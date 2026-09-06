# CHARTER — how the canon is written, and why it never changes

Written 2026-09-05 at HEAD `99777ab`. This file is the constitution of `docs/canon/`.
Every book in the canon is written under it, attacked under it, and repaired under it
before it is born. After birth a book is never edited: a flaw found later is recorded in
`CORRECTIONS.md` as a row that names the verse whose **pre-written replacement now
applies**. If no replacement was pre-written, the row says so, and that absence is itself
the defect recorded — the book failed its own standard, and the correction is a new verse
appended to `CORRECTIONS.md`, never a silent edit of the old one.

---

## 0. The instruction the canon serves, verbatim

The author, 2026-09-05:

> now start a workflow i want you to just write theoretical plans and such i want
> everything that breaks here fixed in at least documentation and plans to develop it to
> be more accurate and more faster to train then just normal base self attention

> i want the docs to be like bible or quran so unchanging that if it finds a flaw or a
> kill there is already solution for a formula if that disproves

The author, 2026-09-05, clarifying the idea and the use it is for:

> there is no need to test your job is to deltee redundant and make a bible level
> archithureal doc about the idea , let me clear the idea , a casuality / consequences
> understanding attention module which predicts for next eullbiburma phase while
> travesing transiets states , it must be engineers wetdream cause the main usecase of
> this is guessing prediction market trades , juggling 4d chess (yes 4d chess exists but
> we will train on chess data) so yea

Read as the canon reads it: **the module** is a causality- and consequence-understanding
attention layer that predicts the **next equilibrium phase** of a context while the
context **traverses transient states**; **the engineering** must be complete enough to
build from the page — interfaces, shapes, the forward pass, the solve, the training
loop, the data pipelines, the budgets — which is book 08; **the two uses that decide
it** are prediction-market trades and chess, trained on chess data, where "4D" names the
multi-constraint, multi-horizon reading the canon operationalises as $K$ constraint sets
as absorbing rows and the horizon dial $\gamma$ — which is book 09. Every other book
serves those two.

The author, 2026-09-03 (`docs/sources/BRIEF.md`; `docs/CEQ_SHAPE.md` §1.2):

> the main attention mechanism i want is causality for ai to learn consequences for it
> to predict safest move under multiple constraint something self attention is bad at
> and any other benefit coming from next transient state phase equilibrium predictor
> based attention

The immutable north star (`CEQ_V20_R15_CONTRACT.md:42-45`):

> attention EQUAL to self-attention on its own ground, built FROM softmax and AdamW,
> capable on ground they cannot occupy — predicting the NEXT STATE toward equilibrium,
> not the next token.

**The three sentences are one goal with three faces.** Equality on softmax's ground is
already earned (the $\gamma=0$ corner is softmax bitwise, `RUN[coord]`,
`docs/CEQ_SHAPE.md` §2.2 Prop. 1). What remains is the other ground: consequences under
intervention, the safest move under $K$ constraints, the next transient state — and on
that ground the canon must plan an operator that is **more accurate** and **faster to
train** than base self-attention, with both words defined below so that no verse can
claim them by accident.

---

## 1. The two words, defined so they cannot be claimed by accident

The record's largest failure mechanism is `MISTAKES.md` **D-1**, racing a baseline at
its proven optimum: every bed in fifteen rounds asked for one scalar at one position, the
shape on which a single attention layer is asymptotically Bayes-optimal. The second is
**R-SKY** (`CEQ_V16_CONTRACT.md:209`): a deeper softmax stack computes the same
resolvent by iteration, so "beats softmax" is banned wherever softmax is a fellow
approximator. The definitions below are the fence around those two holes.

**Base self-attention** means the lane's own `softmaxAttn` (`ceq/lm.py`), which is the
canon's operator at $\gamma=0$ bitwise, at matched parameters (Ruling 3: exact counts in
every table header, a $0.032$ per cent residual is matched, no re-architecting to close
it) — **and** its depth skyline: the same head stacked to depth
$\lfloor\log_2 t^\star\rfloor+2$, which is the honest control on any bed whose label is a
$t^\star$-hop quantity.

**More accurate** means: on a registered bed whose label class is vector-valued and
interventional — the jointly determined state $z^\star$, the displacement field
$\Delta z$ under $\mathrm{do}(a)$, the $K+1$ reach-avoid committors, the argmin over
candidate moves — the operator's distance-to-floor (L-FLOOR: the exact restricted-view
floor, the Hankel ceiling, the Fano bound, whichever the bed's book assigns) is below
base self-attention's by more than $\mathrm{MDE}_N$ at the realised paired sd, at matched
parameters, with the depth skyline's number printed in the same row. Two forms are
licensed and no third: **(A1)** at depth 1, the per-layer statement — one solve does what
$L$ learned hops do; **(A2)** at matched parameters against the depth skyline, where the
skyline's learned hops fall short of the exact solve by more than $\mathrm{MDE}_N$.
Everything else is a reproduction-versus-non-reproduction contrast and is VOID as a
capability number (`docs/CEQ_SHAPE.md` §3.8).

**Faster to train** means two measured numbers, both on the certified device, both with
synchronised timers, interleaved arms and run order recorded (the record's own hazard:
`torch.cuda.synchronize` absent from `scripts/v15_r1.py`, and run order the strongest
correlate of seconds at $\rho=+0.7029$, `V20_R15_JOURNAL.md` C17 at `99777ab`):
**(T1) GPU-seconds-to-floor** — wall-clock from initialisation to a fixed
distance-to-floor, at matched parameters, ratio against base self-attention and against
the depth skyline; and **(T2) draws-to-floor** — the number of training draws to the same
point, the sample-efficiency half. A per-step cost model (one triangular solve of depth
$s$ versus $L$ attention layers) is a **DERIVED** statement until the cost law of the
kernel book is **RUN**; a per-step number alone never licenses "faster to train".

A verse that says "more accurate" or "faster to train" without naming which of A1, A2,
T1, T2 it means, and which bed, floor, $N$ and matched count it is measured on, is
struck at refutation (V-17, a threshold imported out of its units; P-7, vocabulary with no
referent).

---

## 2. The verse — the unit of the canon

Every load-bearing formula, claim, threshold and plan step is a **verse**. A verse has
every field below, in this order, or it is not a verse and is struck.

```
### <BOOK>.<n> — <title>
**Statement.**   the formula or claim, exactly, in LaTeX where it is mathematics
**Hypotheses.**  every condition under which the statement is asserted; the geometry
                 (s, d, n, N, K, t*) at which its inequalities are non-vacuous, printed
**Evidence.**    one class per number: RUN · READ path:line @ <sha> · CITED [V]/[V-eq]/[U]
                 <bib key> · DERIVED (steps) · [FITTED] R² · [ASSUMED] reason ·
                 NOT MEASURED — needs <instrument>
**Mechanism.**   the MISTAKES.md mechanism(s) the verse is designed against, by id
**Kill.**        the single decidable number or condition that refutes the statement,
                 with its threshold FROZEN here (M-2); the instrument that decides it;
                 its price in GPU-s and evenings; and the control that would make the
                 kill fire on a known-bad plant (V-15: no condemning rule without a
                 planted negative)
**If killed.**   the replacement formula or route, ALREADY DERIVED here — not named,
                 derived — with its own Hypotheses, Evidence and Kill (a replacement
                 whose kill is the same as the verse's is not a replacement: V-9)
**Terminal.**    the sentence that survives when the verse and every replacement are
                 dead: what the canon still licenses, and what it withdraws
```

**Depth rule.** Every verse's fallback chain (Statement → If killed → its If killed → …)
ends in a Terminal within at most three replacements, and each link's kill must be
strictly cheaper or strictly more decisive than the one before it, so the chain is
walkable in evenings, cheapest-refutation-first. A chain with a "to be determined" link
is not a chain (P-4, claimed scaffolding that does not exist).

**Reachability rule.** A kill that cannot fire on the bed as registered (V-11, a
precondition satisfied at every real draw; V-10, a gate satisfied by construction) is not
a kill. Every kill names the draw, seed rule and $N$ at which it is decidable, and the
realised-sd clause that makes its $\mathrm{MDE}_N$ honest (M-3, M-9).

**Number rule.** No number without an evidence class. Struck constants of `STRUCK.md`
appear nowhere. A number whose only home is a commit message or a chat is `NOT
MEASURED` (P-1, P-2). A `[V]` citation is inadmissible for a load-bearing statement;
load-bearing needs `[V-eq]` — the theorem, its hypotheses, one numeric instance — or a
DERIVED block (L-EQ, P-10).

**Skyline rule.** No sentence of the form "softmax cannot" without its vacuity inequality
printed at the verse's geometry (V-25; `docs/CEQ_SHAPE.md` §3). No sentence of the form
"beats softmax" outside forms A1/A2 with the skyline number in the row (R-SKY, D-1).

**Voice.** Third person; no author-voice; no hedging words in place of a number; exact
counts; one idea per sentence. Headings and tables carry the structure; prose carries the
argument. LaTeX for mathematics.

---

## 3. Evidence classes and the citation pin

`READ path:line @ 99777ab` is the canon's pin. Round reports were deleted from the
working tree on 2026-09-05 (`docs/sources/README.md`); every one of them is readable at
`99777ab` with `git show 99777ab:<path>`, and `docs/CEQ_SHAPE.md`'s own `READ` class is
pinned to `207e7b9`. A book that cites a deleted file cites it at the pin, and the pin
makes the citation permanent — which is the property the canon needs and the working tree
never had.

Running existing repository code to read a number is allowed and expected (every number
needs provenance). **No code file is written** — no `.py`, `.sh`, `.lean`, `.ipynb`, no
scratch helper in the tree — the author's standing rule, 2026-09-03: *"do not write any
code only docs."* Where a number would need new code, the verse says
`NOT MEASURED — needs <the instrument, specified in the instruments book>`. No git
writes of any kind by any writer, refuter or repairer: git is the shared state
(`CONTRACT.md` D-1), and the coordinator commits.

---

## 4. The laws in force

One line each, source at `99777ab`; the mechanism each pays for is in `MISTAKES.md`.

| law | one line | source |
|---|---|---|
| D-1 dependency | work is a DAG; parallel dispatch only on nodes with no shared repository state; git is the shared state | `CONTRACT.md:27-33` |
| D-2 skills are modes | planet names label responsibilities inside documents, never concurrent processes | `CONTRACT.md:35-38` |
| D-3 loop gate | the iteration count comes from the DAG's critical path | `CONTRACT.md:40-44` |
| D-4 order | contracts scheduled behind an unreached round are staged, not started | `CONTRACT.md:55-57` |
| L-DOM | every theorem that gates a run ships a domain census; no overlap means decoration | `CEQ_V16_CONTRACT.md:54-56` |
| L-SIGN | a counter-prediction of equal specificity beside every prediction | `CEQ_V16_CONTRACT.md:58-61` |
| L-DIAG | a contract prescribes what a diagnostic must distinguish, never which statistic | `CEQ_V16_CONTRACT.md:63-66` |
| L-FLOOR | every capability number ships beside its information floor | `CEQ_V20_R15_CONTRACT.md:60-63` |
| L-CERT | every sparsity mask ships its certificate, F0 exact or F1 with $\delta$ printed | `CEQ_V20_R15_CONTRACT.md:64-67` |
| L-EQ | `[V]` inadmissible for a load-bearing statement; `[V-eq]` needs theorem, hypotheses, one instance | `CEQ_V15_CONTRACT.md:51-55` |
| L-LEAN | the arm may not be trained before its identity theorems are green | `CEQ_V15_CONTRACT.md:57-58` |
| L-G2 | journals never move and are never deleted; a superseded cell keeps a supersede marker | `V17K_RULINGS.md:62-64` |
| FOUND-not-NAMED | an arm is FOUND iff `results/` holds a record with its `kind`; a named arm is struck | `CEQ_V20_R15_CONTRACT.md:49, :266` |
| R-SKY | the depth skyline is read beside every bed; "beats softmax" is not licensed where softmax is a fellow approximator | `CEQ_V16_CONTRACT.md:209` |
| Ruling 1 | CUDA determinism with `warn_only=True`; bitwise for replay and every deciding forward cell | `V17K_RULINGS.md:39-45` |
| Ruling 2/2a | $\beta$ learnable, logged per instance; no sentence transfers across corners without a bind at that corner | `V17K_RULINGS.md:47-54` |
| Ruling 3 | matched parameters as defined in §1 | `V17K_RULINGS.md:56-59` |
| Ruling 7 | a bed is generator + seed + hash, regenerated in-notebook | `V17K_RULINGS.md:83-86` |
| Ruling 10′ | "pinned" is a likelihood ratio on held-out data with the minimum detectable departure printed | `V17K_RULINGS.md:389-436` |
| D-CALIB | the counter is the point estimate for every unscored prediction (9 checked, 9 adverse; `V16_CALIBRATION.md:96-100`) | `docs/PLAN.md` §5.5 |
| the 2026-09-03 supersession | a new primitive is licensed; "no new primitive that is not a repair" is lifted for the canon; every other law stands | `docs/PLAN.md` §5.2 |

The standing laws that bind this file bind every book. The canon adds one: **L-VERSE**
— nothing load-bearing outside a verse; nothing in a verse without its Kill, If-killed and
Terminal.

---

## 5. The census of what is broken — every item is owned by a book

Each row must be closed by a verse in the named book: the break stated with its evidence,
the fix as a plan with its price and its verification number, and the verse's kill (the
number that would show the fix did not work) with its replacement.

| # | what is broken | evidence | owning book |
|---|---|---|---|
| B1 | environment drift: the certified device is `torch 2.5.1+cu121` (`COSTS.md` §1); the box now runs `torch 2.14.0`, `numpy 2.4.6`; `torchvision` circular import, `transformers.PreTrainedModel` import fails, `ripser`/`persim` compiled against numpy 1 — **13 collection errors**, 0 tests run at root (`pytest -q`, 2026-09-05, RUN) | RUN this session | 05 REPAIRS |
| B2 | the device certificate is stale by construction: `results/k_cert_local.json` at `ab5b485` on torch 2.5.1; no certificate exists for the installed stack | `COSTS.md` §1 | 05 REPAIRS, 03 KERNEL |
| B3 | tests coupled to prose: 56 test files open round-report documents and assert their line numbers or hashes; deleted with the documents on 2026-09-05 | RUN scan this session; `docs/sources/README.md` | 05 REPAIRS (test policy), 04 INSTRUMENTS |
| B4 | the label class: every bed asked one scalar at one position (D-1); Q6 is F4 on both wings because no state axis exists | `docs/CEQ_SHAPE.md` §4.4 | 04 INSTRUMENTS, 01 ACCURACY |
| B5 | no CEQ arm has been trained; BED-S has no cell, no realised sd, no measured $t^\star$; every BED-S number is a floor formula | `docs/CEQ_SHAPE.md` §8 | 04, 06 PREDICTIONS |
| B6 | the calibration column: 9 checked predictions, 9 adverse, 7 of 8 signed rows optimistic ($p=0.0352$) | `V16_CALIBRATION.md:96-100` | 06 PREDICTIONS |
| B7 | D-APPROX: every "$X$ cannot represent $Y$" needs an approximation bound before it reads as "cannot fit" — the largest unclosed gap in the record's logic | `docs/CEQ_SHAPE.md` §4.4 | 01 ACCURACY |
| B8 | obstruction theorems vacuous at the record's geometry: Peng Thm 1 needs $n\log n>H(d+1)p$, reads $384<544$ at $s=64,d=16$; Chen $n^{2^{-4L}}=1.30$ at $L=1$ | `docs/CEQ_SHAPE.md` §8 | 01 ACCURACY |
| B9 | the Cheeger sentence at `MATHEMATICS.md:455` cites a reversible-chain theorem for a causal, non-reversible $P$ | `docs/CEQ_SHAPE.md` §8 | 01 ACCURACY |
| B10 | Bellman optimality can fail in multichain constrained MDPs with several unsafe sets — the safest-move rule's class | `misra-2023-safety-constrained-mdp` [V] | 01 ACCURACY |
| B11 | the hop$_k\to$committor reduction is unwritten; the TD-skyline claim rests on two abstracts; two skylines have no cell | `docs/CEQ_SHAPE.md` §3.9 | 01 ACCURACY, 04 |
| B12 | the corner descent: $\beta$ pins to the softmax corner and $\hat\gamma$ is predicted PINNED on at least 6 of 8 seeds (Bet C counter); R1 read 3 of 8 seeds diverging with $\hat a_{\max}$ up to $285.07$ | `docs/PLAN.md` §5.5; `workdonenewseal.md` §4 | 02 TRAINING |
| B13 | determinism: `solve_triangular` observed bitwise but undocumented; `cumsum` raises under deterministic mode; training inherits the backward hole | `docs/CEQ_SHAPE.md` §8; Ruling 1 | 03 KERNEL |
| B14 | timers unsynchronised, arms not interleaved, run order the strongest correlate of seconds ($\rho=+0.7029$); every price is a per-op floor under a $2.0\times$–$6.6\times$ dispatch gap | `V20_R15_JOURNAL.md` C17 @ `99777ab`; `scale/m3_flops.py:101-121` | 03 KERNEL, 05 |
| B15 | the kernels do not exist: chunked solve, CSR path, Mapper schedule are `NOT MEASURED`; the dense control is not runnable at $n=2048,s=4096$ ($\approx137$ GB); the $n=32768$ and $16384$ reproductions were dropped to the memory ceiling | `docs/CEQ_SHAPE.md` §8, §4.4 | 03 KERNEL |
| B16 | harness debts in `scripts/v15_r1.py`: `S, D = 64, 24` module constants at `:137`, no `seq_len` flag; `lambda_hat` at `:383` averages $\log m$ so one $m_k=0$ sends it to $-\infty$ (one bit) | `docs/CEQ_SHAPE.md` §4.4 | 05 REPAIRS |
| B17 | Gate 0 circularity: rulings 1–3 blocked on numbers only the Kaggle run produces, and nothing launches until Gate 0 is green | `V17K_RULINGS.md` open-items table | 05 REPAIRS |
| B18 | five author rulings open: clause-1 tail; the F0–F4 rubric (cited, no text); the Kaggle attach behind the chess witness; the gate's blindness to F4 cells; TERMINAL vs NOT-PUT | `docs/CEQ_SHAPE.md` §4.4 | 05 REPAIRS, 04 |
| B19 | the HuggingFace package scheduled and unbuilt; no trained checkpoint; parameter counts $25{,}736{,}232$ vs $25{,}728{,}000$; three Kaggle sources UNPINNED | `COSTS.md` §0; `docs/CEQ_SHAPE.md` §4.4 | 05 REPAIRS |
| B20 | W2 `arm_phase` is NAMED not FOUND (0 records by `kind`); the route back is one `make_arm` branch, $41$–$130$ GPU-s | `docs/CEQ_SHAPE.md` §4.4 | 04 INSTRUMENTS |
| B21 | the read is occupied: ChaCAL (Fagnou et al. 2024) is the same resolvent, same triangular solve, same $\gamma=0$ parity; the delta is boundary rows, committor read, interventional re-solve, certificate, Lean — and K-E1 predicts a column sink reproduces the row condition | `docs/CEQ_SHAPE.md` §7.2, §5.5 Bet F | 01 ACCURACY, 04 |
| B22 | the bibliography is verified at the identifier, not the equation; ChaCAL's diagonal convention rests on one HTML fetch and decides two controls | `docs/CEQ_SHAPE.md` §8 | 01, 04 |
| B23 | the magnitude interval: closed vs half-open, filed two-sided because $[0,1)$ costs reachability of $a=\pm1$, two-thirds of BED-M's support; D-R3, two mutually inverse registrations for one bed | `docs/CEQ_SHAPE.md` §4.4 | 04 INSTRUMENTS |
| B24 | the process became the product: round 15 produced 102 files and a 40-row index of corrections to its own corrections; the paper's sources were never committed until 2026-09-05 | `git log`; `docs/sources/README.md` | 05 REPAIRS (documentation discipline), this CHARTER |
| B25 | "faster to train" has never been defined or measured in the record; the only cost numbers are per-op floors and the it.3 ratios withdrawn at C17 | `V20_R15_JOURNAL.md` C17 @ `99777ab` | 02 TRAINING, 03 KERNEL, 04 |
| B26 | no architecture document exists: the module's interfaces, tensor shapes, forward pass, solve, boundary rows, heads, training loop and budgets are scattered across `ceq/arm_smprime.py`, `ceq/lm.py`, `ceq/hf/modeling_ceq.py`, `ceq/hf/train.py` and `MATHEMATICS.md`, and no page states the module an engineer would build | `git ls-files ceq/` @ `99777ab` | 08 ARCHITECTURE |
| B27 | the two uses the author names — prediction-market trades and chess — have no bed, no data pipeline, no label, no floor and no matched baseline in the record; the chess witness of round 15 is behind three UNPINNED Kaggle sources and was never run | `COSTS.md` §0; `CEQ_V20_R15_CONTRACT.md` Phase C @ `99777ab` | 09 CHESS_AND_MARKETS, 04 |

---

## 6. The books and their owners

Planet names label responsibility inside documents (D-2). No book holds two of
*derive / execute / adjudicate*.

| book | file | planet · role | scope in one sentence |
|---|---|---|---|
| 00 | `00_NORTH_STAR.md` | coordinator (written last, from 01–09) | the goal, the two definitions, the master decision tree over every book's kills, the first five evenings, the sentence licensed on every leaf |
| 01 | `01_THEORY_ACCURACY.md` | JUPITER · MYCROFT, derivations | the exact class of the resolvent read with boundary rows; the approximation bounds (D-APPROX) that turn "cannot represent" into "cannot fit"; the hop→committor reduction written; every obstruction with its vacuity line at the planned geometry; Cheeger on a named lazy or symmetrised surrogate; the Bellman hazard resolved as a one-step filter or a proven multichain condition; the ChaCAL delta as theorems; the LEAPABLE fields as named theorem targets with Lean grades — each verse with the replacement formula pre-derived |
| 02 | `02_THEORY_TRAINING.md` | JUPITER-II · MYCROFT, derivations | why and when training finds the resolvent solution faster than a depth-$L$ stack learns $L$ hops: landscape, identifiability of $P$ from next-state and committor labels, the $\gamma$/$\beta$ corner-descent problem with its pre-written fixes (fixed-$\gamma$ regime, annealed dial, committor supervision, interventional pairs), sample-complexity statements with constants, the AdamW regime — T2 as theorem-shaped verses with kills |
| 03 | `03_KERNEL.md` | NEPTUNE · LINUS, systems | the fast path: chunked lower-triangular solve fused with causal softmax, memory layout, FLOP and byte model versus $L$ layers, the determinism regime, the cost-law experiment with synchronised timers and interleaved arms, the $n$ ceiling, a Triton design as a specification — T1 as verses with a fallback chain fused → chunked → CSR → dense-per-chunk → cost sentence withdrawn |
| 04 | `04_BEDS_AND_INSTRUMENTS.md` | SATURN · WATSON, instruments | the bed ladder (BED-S in-class and exact; a joint-consistency bed outside the per-row class; the chess witness on natural ground), labels with floors, the admission census, the exact protocol that measures A1/A2/T1/T2, $N$ and $\mathrm{MDE}_N$ from realised sd, the identity manifest, the calibration column, the test policy (tests assert code and journals, never prose) — each bed with its reroute pre-written |
| 05 | `05_REPAIRS.md` | MERCURY · LESTRADE, pricing and scheduling | every row of §5 as a verse: the break, the fix as a plan, the price in evenings and GPU-s, the verification number, ordered cheapest-decisive-first; the environment pin, device re-certification, harness debts, Gate-0 circularity resolution, the HF package, the Kaggle attach, the documentation discipline that keeps the tree at its 2026-09-05 size |
| 06 | `06_PREDICTIONS.md` | VENUS · IRENE, competing predictions | for every cell the books schedule: prediction, counter of equal specificity, SPLIT band, the deciding number, what each outcome triggers in the tree — under D-CALIB the counter is the point estimate; the T1/T2 ratios predicted as ranges with what makes each wrong |
| 07 | `07_ATTACKS.md` | MARS · MORIARTY, standing adversary | the consolidated attacks on books 01–06, 08 and 09 from the refutation stage, each a runnable or decidable statement with a frozen firing number, and the branch of the tree it sends the reader down |
| 08 | `08_ARCHITECTURE.md` | NEPTUNE-II · LINUS, systems, with MYCROFT's derivations cited | the module as an engineer builds it: the CEQ attention layer's interfaces and tensor shapes; the causal three-corner $P$ with $\beta$, QK and the gate; the boundary rows (goal set, $K$ constraint sets, the declared BOS sink) as identity rows; the one triangular solve $z=(I-\gamma P)^{-1}V$ with $K+1$ value channels; the reads — next transient state $z$, the committor vector $q^{(\bullet)}$, the displacement $\Delta z$ under $\mathrm{do}(a)$ by the interventional re-solve, the safest move as the argmin over candidate moves of the max committor across constraint sets, the equilibrium-phase head (which absorbing set the mass flows to, and when a position is a decision point); the transient-state traversal — a sequence read as a trajectory through transient states toward an absorbing equilibrium, the horizon dial $\gamma$, the certificate $\gamma^{K+1}/(1-\gamma)$; integration into a transformer block (residual, normalisation, MLP, multi-head, the $\gamma=0$ corner bitwise softmax so a pretrained softmax stack loads unchanged); the training objectives and their weights; the AdamW regime; the inference API; the compute and memory budget per layer on the certified 4060 and on a Kaggle T4/P100; the kernel hooks (book 03) and the instruments (book 04) named where they attach; pseudocode blocks inside the markdown as specification only — every architectural decision as a verse with its kill and its pre-derived alternative design |
| 09 | `09_CHESS_AND_MARKETS.md` | SATURN-II · WATSON, instruments, with LESTRADE's prices | the two uses as beds and as products: **chess** — PGN to FEN sequences, a move as $\mathrm{do}(a)$, labels (legality, next-FEN, eval-$\Delta$ sign, mate-in-$k$ and material as constraint sets, the terminal outcomes as absorbing sets), the oracles recomputed locally (a chess engine as the exact oracle where one exists), the local data route (lichess database dumps or python-chess-generated games, hashed and pinned per Ruling 7) before any Kaggle attach, floors and matched baselines (a depth-$L$ softmax transformer at matched parameters, a next-token-trained control), what "4D" adds as $K$ and $\gamma$; **prediction markets** — the state as the order book and price path, contracts resolving into absorbing sets, the committor as the probability of resolving in one set before another, the safest trade under constraints (drawdown, liquidity, exposure) as the argmin rule, the interventional re-solve for "if this trade is placed, what moves", the honest floors (market-implied probabilities; Brier and log score against them; the near-efficiency of liquid markets stated as the counter), leakage and look-ahead guards, the data sources that can be pinned locally, matched baselines, and PnL under constraints as a scored prediction with its counter; for both, the first cell at $0$ GPU-s and the first GPU cell priced — every bed verse with its reroute pre-written and its Terminal |
| — | `CORRECTIONS.md` | append-only, empty at birth | the only file in the canon that ever changes |

**Length.** A book is 300 to 900 lines. Every verse earns its place; a verse that no
kill can reach is deleted before birth.

**Reading order for a writer.** This CHARTER; `docs/CEQ_SHAPE.md` §2 (the shape), §3
(obstructions), §4.4 (where the campaign stands), §8 (limits) and the sections of §5 and
§6 in the book's scope — the file is 484 KB, read it by section with `grep -n '^## '` and
`sed -n`, never whole; `docs/PLAN.md` §5.2 (laws), §5.5 (bets), §5.10 (what dies);
`MISTAKES.md` headings (`grep -nE '^### '`), then the entries the book's mechanisms name;
`MATHEMATICS.md` for the record's own derivations; `docs/sources/design/` for the six
refutations already survived; `docs/sources/sweep/` for prior art; `results/` for any
number cited as RUN; `docs/references.bib` for every bib key used.

---

## 7. Refutation and repair

**MARS attacks every verse of a book** and returns structured findings: verse id; the
flaw in one sentence; the mechanism id; the number or condition that shows it; whether
the pre-written replacement survives the same attack; the required repair; severity
(`strike` = the verse is false or unreachable as written; `repair` = the verse stands
with a stated fix; `note`). A finding is itself a decidable statement — a number, a
counterexample, a line — never an opinion. MARS defaults to `strike` when a kill is
satisfied by construction, when a replacement dies with its verse, when a number has no
evidence class, when "more accurate" or "faster" appears outside A1/A2/T1/T2, and when a
chain has no Terminal.

**The repairer rewrites the book** applying every `strike` and `repair`, keeps every
finding **verbatim** in a closing section `## Attacks answered` with the repair beside
it, and leaves any finding it cannot repair standing as an open verse whose Terminal
sentence is the one now licensed. Nothing is softened; a finding that reads more gently
after repair than when filed is a defect (`V20_R15_JOURNAL.md` corrections-index rule @
`99777ab`).

**The critic** reads all six repaired books and answers only: is every §5 row closed by a
verse; does every verse have a Kill, If-killed and Terminal; does every replacement carry
its own kill; does every number carry a class; do any two books contradict each other on
a number or a definition. Its gaps go to the coordinator, who closes them in the books
before birth and writes 00 and 07.

**Birth** is the commit. From that commit the books are unchanging and `CORRECTIONS.md`
is the only door.
