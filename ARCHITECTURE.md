# CEQ — architecture diagrams

Diagrams only. The architecture-and-state document is **`workdonenew.md`**;
`ARCH.md` does not exist and must not be created. Every number appearing here is
carried from `workdonenew.md` and `results/*.jsonl`.

---

## 1. The arm — what actually runs

All four arms share one geometry, one parameter count (`4769`), one MLP and one
readout. Only the operator and the presence of a second hop differ.

```mermaid
flowchart TD
    X["x&nbsp;&nbsp;[n, s, d_model=16]"] --> WQ["wq&nbsp;· x"]
    X --> WK["wk&nbsp;· x"]
    WQ --> OP{"_operator(q, k)"}
    WK --> OP
    OP -->|"softmax<br/>pivot_unsigned"| SM["bench._softmax_operator<br/><b>identical tensor</b>"]
    OP -->|"pivot_signed"| SG0["_causal_sgate_operator<br/>window = 0"]
    OP -->|"windowed_signed"| SG8["_causal_sgate_operator<br/>window = 8"]
    SM --> A["a&nbsp;&nbsp;[n, s, s]<br/>strictly lower triangular<br/>sub-stochastic, nilpotent"]
    SG0 --> A
    SG8 --> A
    A --> H1["hop 1&nbsp;&nbsp;z = x + a·x"]
    H1 --> BR{"arm has hop 2?"}
    BR -->|"softmax: no"| Z["z"]
    BR -->|"pivot_*: yes"| P2["hop2 = a[:, P] · a[P, :]<br/>P = top-k by key-norm, K = 8<br/><b>rank ≤ K</b>"]
    BR -->|"windowed_signed: yes"| W2["z + a·(a·x)<br/>banded, reach 2w = 16"]
    P2 --> Z2["z = z + hop2·x"]
    W2 --> Z2
    Z --> MLP["mlp: 16 → 128 → GELU → 16"]
    Z2 --> MLP
    MLP --> RO["readout → [n, s]"]
    RO --> TAIL["<b>[:, s−1]</b><br/>only the final position<br/>reaches the loss"]
```

**The tail matters.** `Arm.forward` returns `out[:, s-1]`, and the MLP is
position-wise, so only row `s−1` of every operator reaches the loss. Statistics
computed over the whole matrix describe 63 rows that are computed and discarded.

---

## 2. Why the second hop carries nothing

```mermaid
flowchart LR
    LBL["chain label<br/>z_i = a_i·z_{i−1} + b_i<br/>negation_scope.py:307-331"] --> EXP["t-hop term =<br/><b>a_{s−1}···a_{s−t}·b</b><br/>a PATH PRODUCT"]
    ATT["attention hop<br/>Σ_j α_ij x_j"] --> SUM["composing hops =<br/>sums of sums"]
    EXP --> GAP{"shape match?"}
    SUM --> GAP
    GAP -->|"no"| OUT["term is the wrong SHAPE<br/>not the wrong scale"]
    OUT --> R1["K sweep: monotone worse"]
    OUT --> R2["gain sweep: no γ beats 0"]
    OUT --> R3["deflation: returns to baseline"]
    OUT --> R4["MMSE: all 64 modes in the null band"]
    OUT --> R5["gated hop: worse than both controls"]
```

Five independent routes, one conclusion. The full table with every number is in
`workdonenew.md` §6.

---

## 3. The verdict machinery — what each instrument refuses

Every gate is a refusal, not a score. This is the part of the repository that
holds.

```mermaid
flowchart TD
    J["results/*.jsonl<br/>cell rows carry<br/>t_star, n_train, seed, threads, steps"] --> BS["it11_verdict.by_seed"]
    BS -->|"pools across thread counts?"| NO1["<b>REFUSED</b><br/>drift 2.345e-3 measured"]
    BS --> V{"N ≥ 8 distinct seeds<br/>at one thread count?"}
    V -->|"no"| INS["<b>INSUFFICIENT</b><br/>no CI reported"]
    V -->|"yes"| T["tost(arm, reference)"]
    V --> C["cap_verdict(t*, values)"]
    T --> M{"Δ_eq ≥ 2 × 2.345e-3?"}
    M -->|"no"| NO2["<b>REFUSED</b><br/>margin below the<br/>reduction-order floor"]
    M -->|"yes"| PAR["PARITY / UNDERPOWERED /<br/>NOT EQUIVALENT / NO VERDICT<br/>+ achieved power"]
    T -.->|"p > 0.05 therefore equal"| NO3["<b>REFUSED</b><br/>difference test is not parity"]
    C --> F{"CI below √((t*−h)/t*)?"}
    F -->|"yes"| MH["PROVEN MULTI-HOP<br/>proven_hops = h"]
    F -->|"no"| OK["consistent with 1 hop"]
    C -.->|"mean ≥ 1.0"| NA["<b>ĥ = n/a</b><br/>no hop count above the bar"]
```

---

## 4. The claim ladder, and where it stands

```mermaid
flowchart LR
    NS["NORTH STAR<br/>TOST-equal on their ground,<br/>capable on ground they cannot occupy"] --> LAD{"C-PAR ∧ (C-CAP ∨ C-TS)"}
    LAD --> PAR["<b>C-PAR</b><br/>TOST-parity at matched params"]
    LAD --> CAP["<b>C-CAP</b><br/>CI below the 1-hop floor"]
    LAD --> TS["<b>C-TS</b><br/>transition-state accuracy"]
    PAR --> PS["UNREACHABLE at N=8<br/>half-width 0.8807σ vs 0.5σ margin<br/>needs <b>N = 70</b>"]
    CAP --> CS["<b>0 of 9 cells</b><br/>ĥ peaks at 0.389"]
    TS --> TSS["NOT BUILT<br/>bed blocked on Round 11"]
    PS --> SC["<b>SCOREBOARD 0 / 39</b>"]
    CS --> SC
    TSS --> SC
```

---

## 5. The DAG, and what blocks what

```mermaid
flowchart TD
    R11["Round 11<br/>registration + the reading<br/><b>NOT STARTED</b>"] --> BED["BED-1<br/>multi-basin bed + oracle"]
    BED --> DET["fan-out detector<br/>+ α-instrument"]
    DET --> HEADS["basin head, barrier head"]
    HEADS --> S5["S5′ reading<br/>exit accuracy, CK gate"]
    S5 --> VER["verdict"]
    VER --> TRN["local 4060 training"]
    TRN --> TOST2["parity table"]
    R11 -.->|"D-4 blocks everything"| BED
    subgraph DONE["done, parallel-safe (D-1)"]
        A1["Tier-6 prior art"]
        A2["D-3 loop forensics"]
        A3["Tier-1/2 derivations"]
        A8["--arm patch, verified 0.000e+00"]
        A13["claim audit 44/9/1"]
        A17["prediction adjudicator"]
    end
    subgraph BLOCKED["blocked behind BED-1"]
        X27["X₂₇ fractal layer"]
        X30["X₃₀ cheap proxies"]
        X31["X₃₁ chaos guard"]
    end
    BED --> X27
    BED --> X30
    BED --> X31
```

---

## 6. The measurement discipline, as a loop

```mermaid
flowchart LR
    P["file a prediction<br/>while the journal holds<br/><b>zero</b> cells"] --> H["hash it"]
    H --> ADJ["write the adjudicator<br/>before the data exists"]
    ADJ --> RUN["run the cell<br/>one thread lane"]
    RUN --> REF{"N = 8 in one lane?"}
    REF -->|"no"| STOP["<b>REFUSE</b><br/>report the seed count"]
    REF -->|"yes"| READ["adjudicate by script"]
    READ --> REC["record the verdict<br/><b>without editing the prediction</b>"]
    STOP --> RUN
```

This loop ran once this round and returned **4/4 held** on a negative result. It
refused twice on the way — at 6/8 seeds, and again when the completing seeds
arrived in the wrong thread lane, which was the author's own error.

---

## 7. Where to look

| question | file |
|---|---|
| what is true, and how far from the goal | `workdonenew.md` |
| every failure mechanism, with its check | `MISTAKES.md` — 54 entries |
| this round's derivations and corrections | `V13_DAG_TASKLIST.md` |
| independent audit of this round's numbers | `V13_CLAIM_AUDIT.md` |
| the numbers themselves | `results/*.jsonl` |
| the proved core | `lean/CEQ/` |
