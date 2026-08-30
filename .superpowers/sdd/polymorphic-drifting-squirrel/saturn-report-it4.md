# SATURN (WATSON) — iteration 4 report

Unit: test the two-procedures hypothesis on the three headline interval
families, then name the producer beside every published interval.

**Status: DONE_WITH_CONCERNS.**

Worktree fast-forwarded `e5badcc → 6484bee` before any edit.

**Constraint compliance, checked mechanically.** `git diff --cached --name-only
| grep -cE "capability_table|m3_quintuple|eprocess|results/|hf_artifact"` → **0**.
Per-seed values were read through `git show HEAD:results/m3_quintuple_v2.jsonl`,
never from the working tree, because a live run is appending to it. **No wall
clock was read and no arm was trained** — both procedures are deterministic
arithmetic over ten floats. The `3.0×`, `5×`, `1.61×`, `2.1×`, `15.5 h` and
`5.9 h` figures in M-8 and P-8 are unedited and still labelled READ.

---

## 1. The hypothesis is CONFIRMED, exactly, on all three contrasts

Both families reproduce from the same five per-seed deltas. Every one of the
twelve bounds matches to the printed digit.

| contrast | shipped `contrast(n_boot=10000, seed=0)` | exact over all 3125 |
|---|---|---|
| `settled − twin` | `[−0.048587, +0.031557]` = **family A** | `[−0.042903, +0.031557]` = **family B** |
| `settled − softmax` | `[+0.066232, +0.147110]` = **family A** | `[+0.068181, +0.147110]` = **family B** |
| `argmax − softmax` | `[−0.134115, −0.102204]` = **family A** | `[−0.134115, −0.102786]` = **family B** |

**No published interval is orphaned. No number is wrong.** At five seeds the
paired resample space is finite — `5**5 = 3125` tuples with **126 distinct
means**, measured — so the exact percentile is a *different estimator*, not an
approximation of the Monte-Carlo one.

Two independent runs of this, which is why I trust it. The first used the 6-dp
values printed in `results/m3_quintuple.txt:41-47` and reproduced 5 of 6 bounds
exactly with the sixth off by `1e-6` — precisely what input rounding predicts.
The second used full-precision floats from the journal and reproduced **all
six**. The residual in run one was the rounding of my inputs, not the procedure.

---

## 2. The finding is bigger than "provenance failure", and worse

**This was already known, already diagnosed, and already written down — in four
places, including the file that ships.**

`ceq/hf_artifact/README.md:80`, limits paragraph (e), verbatim:

> *"THE INTERVALS ARE MONTE-CARLO, AND THE PROSE DELIVERABLES CARRY A SECOND
> FAMILY. At five seeds the paired bootstrap distribution is finite (5**5 = 3125
> resamples, 126 distinct values), so the exact percentile is computable and
> differs from the B=10000 draw…"*

The same paragraph is in `results/capability_table_v0.md:64` and `v1.md:80`. The
root `README.md:492-498` states it again. `tests/chase/test_capability_table.py`
states it a fourth time — *"Both are defensible estimators; only one of them was
labelled."*

**And the table three lines above paragraph (e) still prints `95% CI` with no
estimator.** Worse: the JSON behind that table **already carries
`n_boot: 10000, boot_seed: 0`** (verified in `results/capability_table_v1.json`).
The provenance exists, is correct, and is **dropped at render time**.

So this is not P-1. **It is P-8 — a caveat that is present, correctly worded, in
the right place, and below the number that gets quoted — recurring on the
repository's most-quoted result, while I was writing P-8.** I have rewritten
M-9's note accordingly; the old "unresolved" text had become stale by my own
P-3.

---

## 3. What I changed, and what I deliberately did not

**Changed.**

* `CHECKLIST.md:1168` and `:1206` — the two unlabelled rows. `:1167` already
  named its estimator; `:1168` is the row the headline gets quoted from and did
  not. Both now name the procedure and give the other family's reading.
* `MISTAKES.md` M-9 — replaced the stale "unresolved" note with the resolution,
  the reproduction table, and the P-8 reading.
* `tests/cameron/test_published_intervals_have_producers.py` — 12 tests.

**Not changed, deliberately.** The root `README.md` was **already correct** —
its column is headed `exact 95 % CI` and its preamble at `:298-301` names the
enumeration explicitly. Nothing to do. `DONE.md` carries both families but is a
13,900-line archive; labelling three lines there is churn against a document
nobody quotes from, and it carrying both families is itself useful evidence.

**The minimal binder, and why it is minimal.** `tests/chase/test_capability_table.py`
already binds family A to the journal and already requires `n_boot`/`boot_seed`
per row. Family B was bound **nowhere** — it existed only as transcribed digits
in three prose documents, which is exactly how an estimator came to look like a
discrepancy. The new file adds that half and nothing else. It recomputes both
procedures, asserts each against its published family, asserts the two
**disagree on the real data**, and **rejects the mis-pairing** — so it cannot
pass while claiming the wrong document uses the wrong estimator. Without that
last clause the file would be circular.

---

## 4. For Mercury — two changes in files I am not allowed to touch

Neither needs a recomputation or a re-run; both are text.

1. **`render()` drops the estimator.** The JSON row already has
   `n_boot: 10000, boot_seed: 0`. The markdown column header is `95% CI`. Make
   it `95% CI (percentile bootstrap, B=10000, seed=0)` — or emit the two fields
   the JSON already holds. This is the whole fix, and it lands in
   `ceq/hf_artifact/README.md`, `capability_table_v0.md` and `v1.md` on the next
   normal cut.
2. **Limits paragraph (e) has gone stale.** It carries `+0.146551` as *"an open
   defect, not adopted"*. That endpoint was diagnosed and corrected:
   `CHECKLIST.md:1240` records it reproduces at `numpy.random.default_rng(0)`
   but mixed generator states between its endpoints, and `README.md:460` records
   the CHECKLIST line as now reading `+0.147110` — which it does. The paragraph
   describing the fix as open is now itself the stale claim.
   `README.md:455-463` lists two further stale clauses in the same string.

---

## 5. Test summary

| file | result |
|---|---|
| `tests/cameron/test_published_intervals_have_producers.py` | **12 passed** (new) |
| `tests/cameron/test_mistakes_citations_resolve.py` | 7 passed |
| `tests/cameron/test_c1_propagate_registration.py` | 23 passed |
| `tests/cameron/test_impact_hetero_is_not_its_own_baseline.py` | 6 passed |
| `tests/cameron/test_bar_control_sees_the_arms_preprocessing.py` | 5 passed |

53 passed, 0 failed. The full suite was not run.

The citation binder earned itself again mid-change: it failed on
`tests/cameron/test_published_intervals_have_producers.py` because `MISTAKES.md`
cited the new file before it was tracked. That is the forward-reference case I
did not anticipate when writing it.

---

## 6. What I could not validate

**The reproduction is of the two procedures, not of the run.** I recomputed both
interval families from journalled per-seed NRMSE values; I did not retrain a
single arm, so I have shown that the published intervals follow from the
journal, not that the journal follows from the weights. If a per-seed value in
`m3_quintuple_v2.jsonl` is itself wrong, both families inherit it identically
and every assertion in the new test still passes. That is a real hole and this
work does not touch it.

I read the committed blob rather than the working file. If Mercury's live run
has appended or rewritten rows at the published keys since `6484bee`, my
reproduction is of the committed state and could diverge from what the next
table cut reads — `git status` showed `results/m3_quintuple_v2.jsonl` clean at
the time I read it, but that is a point-in-time check on a file under active
write, not a guarantee.

**The `126 distinct means` and `3125` figures are mine and measured; the claim
that the exact percentile is the "right" estimator is nobody's.** I have shown
both are computable and both are published; I have not argued which one should
ship, and M-9's point stands regardless — at N=5 the finest achievable two-sided
p is `0.0625` under either procedure, so neither family reaches the `0.05` the
project quotes.

I did not verify paragraph (e)'s claim that `+0.146551` is *"absent from the
97.5th percentile at every Monte-Carlo seed 0..399"* — I confirmed only that the
endpoint no longer appears in `CHECKLIST.md:1168` and that two documents record
it as corrected. If that 400-seed sweep was itself never run, the stale
paragraph is hiding a second issue and I would not have seen it.

Finally, I labelled `CHECKLIST.md` and left `DONE.md` alone on a judgment about
which documents get quoted. If `DONE.md` is in fact cited as a source of record,
that judgment is wrong and its three rows (`:1844`, `:1884`, `:2198`) need the
same treatment — `:1884` being family A and the other two family B is the
clearest single piece of evidence that both were journalled deliberately, and it
would be a shame to lose that by labelling it carelessly.
