# CEQ v15.1 → v15.2 — X₃₅′ WAVE-THEORETIC HIDDEN-CAUSE INFERENCE

Filed verbatim from the author's message of 2026-08-31, mid-round at R11 it.12.
Extends `CEQ_V15_1_DELTA.md`'s X₃₅. Both stay verbatim-of-record under RUL-5.

X₃₅ posed hidden-cause inference as **detection** — a residual and an onset.
X₃₅′ poses it as an **inverse problem**: the residual is a field, the hidden
cause is a source, and the machinery for recovering sources from fields already
exists and is exact. The onset detector does not go away; it acquires a floor to
be measured against.

---

## INSTRUMENTS

### (a) Exact source solve

`ĥ = (I − A) r`. `[RUN 8.9e-16, two sources]`

The residual `r` is what the propagator produced from an unobserved source. If
the forward map is the resolvent `(I − A)⁻¹`, the source is recovered by applying
its inverse, which is `(I − A)` — a *subtraction*, not a solve. This is why the
figure is `8.9e-16` and not a convergence tolerance.

### (b) Time-reversal localization

`Wᵀ r`, with **Wiener deconvolution when the noise is colored**.
`[RUN: exact at sd 0.3]`

The adjoint of the propagator focuses a back-propagated field at the source. It
is not exact where the exact inverse is, and it is robust where the exact inverse
is not — which is what makes the pair worth carrying rather than either alone.

### (c) Kramers–Kronig causality residual on every learned kernel

**SHIPS ONLY after its rebuilt must-fire.** A planted **anticipating** kernel
MUST read nonzero.

> The author's own KK probe read `0.000` on a planted anticipating kernel and is
> therefore **VOID** — a causality test that does not fire on an acausal kernel
> is measuring nothing. Filed by the author against the author.

The instrument is rebuilt from scratch or it does not ship. Nothing downstream
may cite a KK number until the planted-anticipating must-fire is seen to fire.

### (d) Cramér–Rao localization floor

Per bed, **printed beside every onset CI**. Detector accuracy is reported as
**distance-to-CRB**, not as raw error.

This is what converts "the detector found it at index 36 instead of 37" from an
anecdote into a statement: either the detector is at the information-theoretic
floor for that noise level, or it is a stated distance above it, and the two are
completely different findings.

---

## MUST-FIRES (L-SCOPE, production-path residuals)

1. planted **single** source
2. planted **two** sources — superposition
3. **no plant** — flat residual at the calibrated false-alarm rate
4. **anticipating kernel** — KK fires
5. **noise sweep** against the CRB curve

---

## LEAN #15 `[M]` — `resolvent_inverse_is_difference`

`(I − A)⁻¹` has inverse `(I − A)` — trivial — and hence
**`source = (I − A) · residual`**, the subtraction lemma.

The theorem is easy and the consequence is not: it says the inverse problem has a
**closed form** on this architecture, with no iteration, no regularization and no
conditioning question, because the forward operator's inverse is a first-order
difference. That is a property of the resolvent carrier specifically, and it is
the sharpest existing argument for the carrier.

## PRIOR ART `[U]`

- inverse-source / time-reversal (Fink)
- Titchmarsh / Kramers–Kronig in learned kernels
- Cramér–Rao bounds for delay estimation

---

## KILLS

- **Any KK or CRB number quoted before its must-fire ⇒ STRUCK.**
- **Detector worse than the adjoint at high noise ⇒ the exact inverse is retired
  to the noiseless regime, stated.** The exact solve is not defended past where
  it stops winning.

---

## RELATION TO X₃₅ AND TO THE ROUND'S ORDER

X₃₅a's Shewhart onset detector stands. X₃₅′ gives it (d) a floor to be measured
against and (a)/(b) two estimators to be compared with. The X₃₅ kill still binds
first and is the stricter of the two: **no-plant residual fails flatness ⇒ X₃₅
VOID**, because the detector cannot outrun the model it subtracts. A CRB
comparison against a void detector measures nothing, so must-fire 3 gates the
rest.
