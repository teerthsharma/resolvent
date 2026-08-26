# STATE — CEQ v8.2, ROUND 6, THE HILBERT ROUND

| field | value |
|---|---|
| contract | `LOOP_PROMPT.md` (v8.2). Round 5 archived `LOOP_PROMPT_ROUND5_ARCHIVE.md` |
| promise | **`HILBERT`** — D2 verbatim, or an honest `BROKEN` |
| iteration | **7 complete, 8 next** |
| phase | **B — birth gates and the probe (iterations 5–9)** |
| scoreboard | **4** — Star delta +2, κ<1 measured +2 *(qualified: Birkhoff route dead, structural β earned it)* |
| RULE 2 clock | **running.** M3 settled-vs-unsettled EXECUTES by **iteration 12** or breach review |
| register | **caveman, all agents, every iteration.** Artifacts stay normal English |
| autonomy | user meets the loop at **iteration 30**; no check-ins |

## THE ONE NEXT ACTION (round 6, iteration 8)

**THE G2 EVENT. Dispatch WILSON. A published number has moved and G2 is a global
stop.**

    re-taken   D_FR slope in k = -0.4137  [-0.4573, -0.3712]
    published                     -0.4137  [-0.4579, -0.3704]

Fixed `manual_seed(4242)`, `B=2000`, 6/6 k, 2000/2000 usable reps. **The point
estimate reproduces exactly and the interval endpoints do not.** The round-5
verdict is untouched — both intervals lie below `-0.30` — **but G2 does not
condition on whether the verdict survives.**

**This is WILSON's, and it is explicitly NOT a Dr House trigger:** a bootstrap
interval that will not reproduce is an engineering and provenance question, not a
missing leap. If Wilson finds it is instrumentation rather than fact, it goes to
the **Health Inspector** next, who is the better engineer.

**What he must settle, in order:**
  1. **Is the published interval or the re-taken one correct?** Not which is
     preferred — which is *right*, and what produced the other.
  2. **What moved it** at a fixed seed: a torch/BLAS version change, thread-count
     dependence (this repo has a measured history of exactly that), a resampling
     index change, or an edit to the estimator since publication.
  3. **Does anything else published from the same bootstrap drift?** One drifted
     endpoint pair with a stable point estimate suggests the resampler, not the
     statistic — so the same check belongs on every bootstrap CI in the record.
  4. **Whether the seven publication sites must be corrected or the re-take is
     the error.** Correcting a published number requires the G2 procedure, not an
     edit.

**Running concurrently, nothing claimed for any of it:** Foreman on ARM S's birth
gates; Cameron on the F-green matched re-run (**+3/−5, the round's pivot**).

**RULE 2: four iterations after this one.** Chase's harness is calibrated and
correct in both directions, so the money run is unblocked — but it must be at
**n_train=8192**, print **all five seeds**, and carry the **~0.05 NRMSE resolution
floor in its pre-registration**.

## Open REDs

Carried from round 5, all still open:

1. **K1's sign-flip clause** — the only pre-registered kill left genuinely
   undecided. 8 events in 2400 draws, `0.003333 [0.001440, 0.006557]`; the
   unsigned arm's interval `[0, 0.010195]` **contains** it. **Round 6 closes it
   by SPRT (K-H, +1)** rather than by 20,000 fixed draws.
2. **Every probe number ever taken was random-init.** The trained-projection
   re-run is Phase D and is the oldest open item in the project.
3. **What peak attention tracks** once the key-norm confound is removed —
   retention 2.9% / 13.3% / 58.9%, matched by **rank not value**, so those are
   **upper** bounds.
4. **The provenance bind is RED deliberately** — nine shipped cost figures whose
   run evidence lives in `DONE_ARCHIVE_ROUND1.md`. Repair by **re-pointing**,
   never by pasting.
