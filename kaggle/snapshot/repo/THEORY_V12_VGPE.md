# THEORY_V12_VGPE.md — VGPE as a weighted automaton, and what its capacity actually is

Round 12, Mycroft. Normal English by RULE 4's artifact exemption.

This chapter states three things and each one is measured rather than asserted:
what VGPE's positional code is as a formal object, what the repo's existing
Hankel instrument can and cannot say about its capacity, and what the `d = 256`
capacity number is once the Welch line from `MATHEMATICS.md:766` is carried onto
`so(d)` instead of onto role vectors.

**It is written under a prior-art verdict that goes against the round.** The
novelty claim as specified — path-ordered non-abelian transport as a positional
encoding for attention, with RoPE recovered as a special case — is **OCCUPIED**,
and the occupant states both halves in one sentence. `PRIOR_ART.md` §5 carries
the fetch. Nothing below should be read as a claim of priority; it is a claim
about what the object *is*, which is a separate and still-answerable question.

---

## 0. THE VERDICT THIS CHAPTER IS WRITTEN UNDER

*PaTH Attention: Position Encoding via Accumulating Householder Transformations*,
Yang, Shen, Wen, Tan, Mishra, Ren, Panda, Kim, **arXiv:2505.16381**,
NeurIPS 2025, §2.1, scores attention as

```
    A_ij  ∝  exp( k_j^T ( prod_{s=j+1..i} H_s ) q_i )
```

and says of it, verbatim, *"RoPE is thus a special case of the above with a
static transition matrix `H_s = R`"*. That is the VGPE score formula
`⟨q_i, ρ(path i→j) k_j⟩` and the RoPE-recovery identity, in the same section of
the same paper. The mechanism is not open.

What is left after §5 of `PRIOR_ART.md` is narrow and is stated there. This
chapter proceeds because **the capacity question is not the novelty question**:
nothing found has priced a path-ordered orthogonal code in Hankel rank, and the
repo already owns the instrument that does it.

---

## 1. THE OBJECT

`scale/vgpe.py:201` defines the code. Each action type `a` carries a generator
`Ω_a ∈ so(d)`, mapped into `SO(d)` by Cayley (`scale/vgpe.py:111`), and a path
is a word in the free monoid over the action alphabet:

```
    U_a = (I − Ω_a)^{-1} (I + Ω_a) ∈ SO(d)
    ρ(w) = P ∏_{a ∈ w} U_a                       (scale/vgpe.py:201)
    score(i,j) = ⟨q_i, ρ(path i→j) k_j⟩
```

Backward edges traverse `U^T`, which is the inverse **because** `U` is
orthogonal; `scale/vgpe.py:216` free-reduces words so that a retraced walk is
exactly `()` and a genuine cycle is not. RoPE is `|Σ| = 1` on a chain, and the
constant it needs is `θ_i = 10000^{−2(i−1)/d}` — RoFormer eq. (15), §3.2.2 —
which `scale/vgpe.py:149` implements and which reads `[1.0, 0.1, 0.01, 0.001]`
at `d = 8`, i.e. `10000^{−(i−1)/4}`, matching the paper's constant exactly.

---

## 2. THE CODE IS A WEIGHTED AUTOMATON, AND ITS RANK IS `d`

### 2.1 The statement, with its factorisation

Fix one head and one query/key pair. The score as a function of the path is a
scalar function on the free monoid:

```
    f(w) = q^T ρ(w) k = q^T ( ∏_{a ∈ w} U_a ) k
```

This is literally a linear representation `(λ, {M_a}, γ)` with `λ = q^T`,
`M_a = U_a`, `γ = k`, of dimension `d`. By Fliess / Carlyle–Paz — the theorem
`ceq/hankel.py` is built on, module docstring lines 11–17 — the Hankel rank of a
rational series equals the dimension of its minimal linear representation, so

```
    rank_R(H_f)  ≤  d
```

The factorisation is not abstract. Because `ρ(uv) = ρ(u) ρ(v)`,

```
    H_f[u,v] = q^T ρ(u) ρ(v) k = ⟨ ρ(u)^T q , ρ(v) k ⟩
```

so `H_f = A B` with `A`'s rows the **transported queries** `ρ(u)^T q ∈ R^d` and
`B`'s columns the **transported keys** `ρ(v) k ∈ R^d`. The rank-`d`
factorisation of the Hankel matrix *is* the pair of transported vectors. Rank
equals `d` exactly whenever those two families each span `R^d`, which is
generic.

### 2.2 What `ceq/hankel.py` can actually measure

The instrument takes `f: Σ* → R` as a Python callable over a `str` alphabet and
a finite word block:

| function | line | what it returns for this series |
|---|---|---|
| `words_upto` | `ceq/hankel.py:81` | the block index, all words of length `≤ n` |
| `hankel_block` | `ceq/hankel.py:108` | `H[u,v] = f(uv)`, dense, float64 |
| `rank_real` | `ceq/hankel.py:131` | the rank **and** `gap_ratio`, the margin |
| `myhill_nerode_classes` | `ceq/hankel.py:152` | distinct residuals by exact equality |
| `rank_plus_lower` | `ceq/hankel.py:279` | `NEG_ENTRY` — see §2.4 |
| `additive_nonneg_certificate` | `ceq/hankel.py:324` | not applicable — see §2.4 |

`rank_real`'s own docstring is the load-bearing part: *"a gap near 1 means the
rank was not measured, it was chosen"*. Every number below is quoted with its
gap.

### 2.3 THE MEASUREMENT

Gauge from `scale/vgpe.py:128` `random_generators(A, d, seed=20260830,
scale=0.25)` — the battery's own seed and scale — through `cayley`; `q, k` unit,
`torch.Generator` seed 7. python 3.11.9, torch 2.5.1+cu121, numpy 1.26.4,
float64, this box.

```
    d    |Σ|   n   block     rank   bound(=d)   gap_ratio
     4    2    3   15x15       4        4       2.631528e+14
     4    2    4   31x31       4        4       5.616284e+14
     8    2    3   15x15       8        8       1.809204e+13
     8    2    4   31x31       8        8       8.215602e+13
    16    2    3   15x15      15       16       inf          <- BLOCK-LIMITED
    16    2    4   31x31      16       16       2.608537e+14
```

**The `d = 16, n = 3` row is the control that the instrument is honest.** A
`15x15` block cannot exhibit rank 16, and it reports 15 with `gap = inf` — the
saturated reading, not a claim. Widening the block to `31x31` recovers 16.

**Alphabet size does not raise the rank.** At `d = 8`:

```
    |Σ|   n   block     rank   gap_ratio
     2    3   15x15       8     1.809204e+13
     3    2   13x13       8     2.772809e+14
     4    2   21x21       8     5.007888e+14
```

Three different action alphabets, one rank, and the rank is `d`.

### 2.4 The two columns that are structurally unavailable here

**`rank_+` is dead, and for the right reason.** Attention scores are signed. The
`d = 8, |Σ| = 2, n = 3` block has minimum entry `-0.6964291002329377`, and
`rank_plus_lower` returns the `NEG_ENTRY` sentinel: *"negative entry: no
nonnegative factorisation of any size"*. The nonnegative-rank machinery that
`tests/cameron/test_hankel_mustfire.py:102` and `:122` certify is therefore not
applicable to a VGPE score series at all. It is not a gap in the measurement; it
is the measurement, and `tests/cameron/test_hankel_mustfire.py:181` is the test
that requires absence be reported as absence rather than as zero.

**The Nerode column is vacuous on a real-valued series.**
`myhill_nerode_classes` separates residuals by exact float equality. On the
`d = 8, |Σ| = 2, n = 2` instance it returns **7**, and `words_upto("ab", 2)` has
**7** elements — it counted the prefixes. Every residual of a generic
real-valued series is distinct, so the deterministic-state column saturates at
the block size and carries no information about VGPE. That is a property of the
instrument on continuous series, and
`tests/cameron/test_hankel_mustfire.py:193` is the self-test that exists because
the Nerode reading can mislead.

### 2.5 THE CAPACITY CLAIM, AND IT CUTS AGAINST THE ROUND

Non-commutativity **does not buy Hankel rank**. The bound is `d` whether the
alphabet has one action type or four, and it is `d` for RoPE too. What changes
between RoPE and VGPE is not the number of states but the number of positions
those states must serve:

```
    RoPE, |Σ| = 1 :  n+1 distinct positions of length ≤ n,   served by d states
    VGPE, |Σ| = A :  (A^{n+1}−1)/(A−1) distinct positions,   served by d states
```

At `A = 4, n = 8` that is 87,381 distinct paths against 256 states. **The free
monoid is a coverage liability at fixed `d`, not a capacity gain.** Any claim
that VGPE "carries more structure" has to be a claim about *which* rank-`d`
series is realisable, never about the rank, and this chapter can supply the
bound but not that claim.

### 2.6 The instrument is better conditioned on VGPE than on RoPE

Run against RoPE's own gauge (`rope_thetas(8)` through `rope_generator`,
`scale/vgpe.py:149`, `:160`), `|Σ| = 1`, `n = 12`, block `13x13`:

```
    σ_i/σ_0 = 1.000e+00  8.416e-01  7.952e-01  3.684e-02  5.125e-03
              2.154e-08  6.522e-11  1.537e-16  ...
    threshold/σ_0 = 2.887e-15        ->   rank_real reports 7, not 8
```

The true rank is 8 — four frequency blocks, two exponentials each, and all four
amplitudes are `O(10^-1)`: `1.819966e-01, 6.788512e-02, 7.144116e-02,
3.354234e-01`. The eighth mode is nevertheless **below the float64 threshold**,
because `θ_4 = 0.001` sweeps 0.024 rad over the whole block and is numerically
indistinguishable from the low-order behaviour already spanned. RoPE's
log-uniform `θ` spectrum spans three decades by construction, and that is what
wrecks the conditioning; the VGPE gauges above measure clean at
`10^13`–`10^14`. **Reported as observed: this is a statement about the
instrument's conditioning, not about either architecture's expressivity.**

---

## 3. THE M1 CAPACITY LINE AT `d = 256`

### 3.1 Where the existing line lives, and what it is not

The repo's capacity argument is `MATHEMATICS.md:762`, §14, *"THE COHERENCE FLOOR
AND THE CALIBRATED SCRAMBLE CONTROL"*. Its statement is `MATHEMATICS.md:766`:
any `k` unit vectors in `R^d` obey
`max_{i<j} |⟨u_i,u_j⟩| ≥ sqrt((k−d)/(d(k−1)))`, which is `MATHEMATICS.md:772`
**exactly 0 for `k ≤ d`**. Its M1 reading is `MATHEMATICS.md:777`: at
`d = 256, k = 16` the floor is 0 and random role vectors carry max coherence
`0.174795`, so *"a crosstalk-shaped failure ... is a training or design defect,
not dimension starvation"*.

**Correction to the round's framing: there is no Johnson–Lindenstrauss line in
this repository.** A case-insensitive search for `lindenstrauss` across every
`.md` and `.py` file in the worktree returns **zero** matches. The M1 line is
Welch and only Welch. Anything below that reads as JL is new here and is
labelled as such.

### 3.2 The dimension

Generators are the vectors, and `so(d)` is a real inner-product space under
Frobenius — `scale/vgpe.py:123` `skew` is described in the code as *"the
orthogonal projection in Frobenius"*, so the geometry is the right one.

```
    dim so(256)  =  256 · 255 / 2  =  65280 / 2  =  32640          [verified]
```

For scale: `d² = 65536`, and `32640 + 32896 = 65536` with `32896 = 256·257/2`
the symmetric complement. The arithmetic closes.

### 3.3 The Welch floor does not bind, and saying it does would be wrong

`MATHEMATICS.md:766` with `d ← D = 32640`:

```
    k = 16          0.000000e+00
    k = 256         0.000000e+00
    k = 4096        0.000000e+00
    k = 32640       0.000000e+00       <- still exactly zero
    k = 32641       3.063725e-05       <- first nonzero
    k = 65280       3.913932e-03
    k = 1000000     5.444014e-03
    k -> ∞          5.535093e-03  =  1/sqrt(D)
```

**Up to 32,640 action types can carry exactly mutually orthogonal generators at
`d = 256`, and the Welch floor never exceeds `1/sqrt(32640) = 5.535093e-03` no
matter how large the alphabet gets.** Welch is therefore not the binding
constraint on the action alphabet at any plausible size, and quoting `32640` as
"the capacity" would be quoting a bound that is never reached in practice.

### 3.4 What binds instead: coherence of drawn generators

Same routes as `MATHEMATICS.md:783`–`795`, at `D = 32640`, 400 trials, one seed
stream (`numpy` `default_rng(20260830)`):

```
    k     Welch      union-over-C(k,2) UB   Monte Carlo max      sd
      4   0.000e+00       1.047803e-02       9.290684e-03    2.927e-03
     16   0.000e+00       1.712750e-02       1.560419e-02    2.180e-03
     64   0.000e+00       2.159236e-02       1.997241e-02    1.805e-03
    256   0.000e+00       2.523577e-02       2.367753e-02    1.519e-03

    mean |⟨u,v⟩| exact  Γ(D/2)/(√π Γ((D+1)/2))  =  4.416399e-03
    sqrt(2/(πD))                                =  4.416365e-03
```

The union bound sits above the sampled maximum in every row, as a bound must —
the same shape as `MATHEMATICS.md:786`–`793`, and for the same reason the
author's `sqrt(2 ln k / d)` form is the wrong one there: the maximum runs over
`C(k,2)` pairs, not over `k` vectors.

**The dense reading is roomy.** At `k = 256` action types, worst-case generator
overlap is `2.367753e-02`, roughly a seventh of M1's own `0.174795` at
`d = 256, k = 16`, because `D = 32640` is 128× the role space. On the dense
parametrisation, action-alphabet interference is not a real constraint.

### 3.5 THE COLLAPSE — the affordable parametrisation has `D = 384`

The dense reading is the wrong one, and `V12_PRICING.md` is why.
`V12_PRICING.md:129` prices the dense build, and `V12_PRICING.md:151` prices the
consistent dense arm at **93.6156%** of arm total at `n = 1`, with
`V12_PRICING.md:153` recording that *"the dense transport alone is 14.3×
`F_base`"* and that *"the dense path is never free at any `n`"*. The affordable
arm is the blockwise one. `V12_PRICING.md:89` and `:108` establish that the
round's own `768 FLOPs/edge` reproduces only as the `b = 2` **abelian** rate, and
`V12_PRICING.md:118` that the non-abelian `b = 4` quaternion rate is `1,792`.

Blockwise parametrisation constrains the generator to block-diagonal, and the
dimension collapses accordingly:

```
    dense              so(256)              32640
    blockwise b=4      so(4)^{d/4}      6 ×   64  =   384      85× smaller
    unit quaternion    su(2)^{d/4}      3 ×   64  =   192     170× smaller
    RoPE               so(2)^{d/2}      1 ×  128  =   128     255× smaller
```

The `su(2)` row is the left-multiplication-only quaternion action,
`dim su(2) = 3`; the `so(4)` row is the full block, `dim so(4) = 6`. Which of
the two `V12_PRICING.md`'s `b = 4` row denotes is that file's call, not this
one's, so both are carried.

At those dimensions the coherence numbers change character completely:

```
    D = 384  (so(4)^64)      mean |⟨u,v⟩| = 4.074339e-02
      k=  4   Welch 0.000e+00    UB 9.660269e-02   MC 8.446483e-02  sd 2.654e-02
      k= 16   Welch 0.000e+00    UB 1.579077e-01   MC 1.426924e-01  sd 1.980e-02
      k= 64   Welch 0.000e+00    UB 1.990717e-01   MC 1.836696e-01  sd 1.561e-02
      k=256   Welch 0.000e+00    UB 2.326623e-01   MC 2.160998e-01  sd 1.297e-02

    D = 192  (su(2)^64)      mean |⟨u,v⟩| = 5.765738e-02
      k= 16   Welch 0.000e+00    UB 2.233153e-01   MC 2.006609e-01  sd 2.765e-02
      k=256   Welch 3.615508e-02  UB 3.290342e-01  MC 3.031641e-01  sd 1.806e-02

    D = 128  (so(2)^128, RoPE's own)   mean |⟨u,v⟩| = 7.066157e-02
      k= 16   Welch 0.000e+00    UB 2.735042e-01   MC 2.453292e-01  sd 3.370e-02
```

**M1's own number is the anchor.** `MATHEMATICS.md:777` reports `0.174795` as
the random worst-case overlap at `d = 256, k = 16` — the level at which the
project already treats crosstalk as a live design concern. The affordable VGPE
alphabet at `k = 16` reads `0.1426924` (`so(4)` blocks) to `0.2006609`
(quaternion blocks), **bracketing M1's figure**. So:

> **The action alphabet the affordable parametrisation supports at `d = 256` is
> on the order of 16 action types before drawn generators interfere at the level
> M1 already calls a design concern, and at most 384 before the Welch floor
> itself stops being zero. It is not 32,640. The 32,640 figure describes a
> parametrisation that `V12_PRICING.md:151` prices at 93.6156% of the arm.**

### 3.6 And the alphabet buys no Hankel rank anyway

§2.3 measured rank `= d` at `|Σ| = 2, 3, 4`. Combining:

* **Welch bounds the alphabet** — how many *distinct* generators fit.
* **Hankel bounds the series** — how much *structure* a code of width `d` can
  express over paths, and that ceiling is `d = 256` regardless of alphabet.

Both are necessary conditions and neither is sufficient. Nothing here says a
trained VGPE reaches either.

---

## 4. THE WILSON-LOOP SPECTRUM AS THE PROBE

### 4.1 Conjugation invariance is proved, and it is the easy half

`scale/vgpe.py:284` `wilson_spectrum(us, loops)` returns `{tr ρ(loop)}`. Its
docstring gives the whole argument: conjugation passes through the path-ordered
product because *"the V^T V pairs between consecutive letters cancel"*, and
trace is a similarity invariant. So for `U_a ↦ V U_a V^T`,

```
    tr ρ_V(w) = tr( V ρ(w) V^T ) = tr ρ(w)                for every word w
```

This is why the spectrum measures the connection rather than the frame. A probe
that read entries of `ρ` would report the coordinates, and
`tests/watson/test_vgpe_binds.py:294`
(`test_a_coordinate_probe_moves_under_conjugation`) is the test that shows a
coordinate probe does exactly that. V12 should not describe the trace probe as
"chosen"; it is forced by the invariance requirement.

### 4.2 Completeness is a citation, not a result of this repo

The claim that the *full family* of Wilson loops determines the connection up to
gauge is **Giles' reconstruction theorem** — R. Giles, *"Reconstruction of gauge
potentials from Wilson loops"*, Phys. Rev. D **24** (1981) 2160. **[U]** — the
title, journal, volume, year and page were reached through the APS listing only;
the primary text was **NOT REACHED** and no statement of the theorem is quoted
here. Anything this project writes about *completeness* rests on that citation
and it must be fetched before publication.

**And it is not what the tests establish.** The Watson battery tests the
**necessary** direction — invariance — plus a weak empirical converse. It does
not test completeness and no test in this repository does. Writing "complete
gauge invariant" without that sentence attached would be an overclaim of exactly
the kind `MISTAKES.md` tracks.

### 4.3 What Watson's numbers do establish

`tests/watson/test_vgpe_binds.py`, 11/11 passing, float64, torch 2.5.1+cu121,
bar `1e-12` fixed before the runs:

| what | test | line | measured |
|---|---|---|---|
| conjugated twin reads the same spectrum | `test_the_conjugated_twin_reads_the_same_wilson_spectrum` | `:254` | `7.105e-15` max abs tr diff |
| the probe separates two different gauges | `test_the_wilson_probe_is_not_vacuous` | `:276` | non-vacuous |
| a coordinate probe does **not** survive conjugation | `test_a_coordinate_probe_moves_under_conjugation` | `:294` | moves |
| the connection stays in `SO(d)` at depth 512 | `test_orthogonality_holds_at_depth_512_and_the_additive_controls_do_not` | `:165` | `3.220e-15` |
| retraced walk is exactly the identity | `test_the_tree_roundtrip_is_the_identity_and_a_loop_is_holonomy` | `:209` | `2.286e-15` |
| a non-orthogonal connection breaks it | `test_a_non_orthogonal_connection_breaks_the_tree_roundtrip` | `:234` | RED holds |
| non-commutativity is nonzero and collapsible | `test_the_commutator_dial_reads_nonzero_and_the_collapse_zeroes_it` | `:313` | `commutator_norms`, `scale/vgpe.py:295`, `:308` |
| RoPE recovery, tolerance not bitwise | `test_the_rope_recovery_bind_is_a_tolerance_not_a_bitwise_identity` | `:125` | `6.217249e-14` abs / `3.253493e-15` rel, **0 of 4 bitwise** |
| a perturbed `θ` breaks recovery | `test_a_perturbed_theta_breaks_the_rope_recovery_bind` | `:146` | RED holds at `1e-9` |
| Cayley doubles the angle without the warp | `test_the_naive_warp_doubles_the_angle` | `:79` | `θ ↦ 2 arctan θ` |

The theory predicts three things and the battery reads all three: **invariance**
(`:254`), **non-vacuity** (`:276`), and **frame-dependence of the alternative**
(`:294`). Together they say the trace probe measures structure and a coordinate
readout does not. They do **not** say the trace probe measures *all* the
structure. `:276`'s separation of two gauges is one instance, not a converse.

The `:313` pairing is what makes the apparatus falsifiable rather than
decorative: `abelian_collapse` (`scale/vgpe.py:308`) zeroes the commutator dial,
and if the round's benefit survives the collapse then the non-commutativity was
never load-bearing. That control is the round's real gate, and it belongs to the
controls file, not to this one.

---

## 5. LIMITS

Collected once, at the end.

The Hankel measurements of §2.3, §2.4 and §2.6 are at `d ∈ {4, 8, 16}` with
blocks up to `31x31`, on drawn gauges from one seed, on one box. **They are not
at `d = 256`.** Measuring rank 256 needs a block of at least `257x257`, i.e.
`≥ 66049` evaluations of `f` at words of length up to `2n`; at `|Σ| = 2` that is
`n = 8`, and each dense `256×256` compose costs `2·d³ = 33,554,432` FLOPs by
`V12_PRICING.md:103`'s own convention. The `≤ d` bound is proved in §2.1 and
holds at every `d`; the claim that it is **attained** is measured only up to
`d = 16`.

`rank_real`'s threshold is `max(shape)·eps·σ_0` and its verdict is a numerical
decision — §2.6 is a live demonstration that the decision can go the wrong way
by one, on RoPE's own gauge, at `σ_7/σ_0 = 1.537e-16` against a threshold of
`2.887e-15`.

§3.4 and §3.5 are arithmetic about **independent uniform** unit vectors in
`so(d)` and say nothing about the generators any trained arm holds — the same
limitation `MATHEMATICS.md:829` records for §14, carried forward rather than
quietly dropped. The Monte Carlo is 400 trials at one seed. Real learned
generators are neither independent nor uniform, and if training concentrates
them the effective `D` is smaller than the nominal one and every figure in §3.5
is optimistic. The `D = 384` and `D = 192` rows further depend on which
blockwise parametrisation the arm actually ships, which is `V12_PRICING.md`'s
determination and could move.

§4.2's completeness claim is **[U]** and unfetched. §0's verdict rests on
`PRIOR_ART.md` §5, whose own status markers apply.

Nothing in this chapter measures a trained model, an accuracy, or a benefit.
Every number is a property of the construction.
