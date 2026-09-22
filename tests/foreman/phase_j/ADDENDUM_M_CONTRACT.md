# PHASE J — ADDENDUM M (THE VOLUME)

**Time as a spatial axis. Stack the sequence's representations into a solid (position, feature, time) and slice it obliquely: motion becomes a straight line whose slope is velocity, a tangle is a crossing, a phase boundary is a slope change, and the barcode is the t-axis cut where the slope is constant. This addendum adds the slice as an instrument beside the rotation fit of Addendum L, with a null, and the oblique-slice family as a read.**

Author: Seal (Teerth Sharma). Contract date: 2026-09-22. Amends PHASE_J_ADDENDUM_L_LEAP.md §3 (DRIFT) and PHASE_J_HYPERSPACE_v2.md §6 (R-PHASE, R-HOLE). Source: the spatiotemporal-volume reel ("xy + t", bradleytangonan, 2026-09-18) — the tesseract note drawn by someone else — and the mathematics that already exists for it.

---

## 0. Standing

- Addendum L stands: the leap is the piecewise-constant SU(2) connection; Q1–Q3 unrun; the three make-or-break counts unchanged.
- The reel's content, read from the page: a video's frames stacked into a solid with time as depth, cut at an angle so that one image shows different places at different times. Comments: *frame by frame*; *replace an axis with time*; *time becomes a place you can move in*. Caption "xy + t": either the slicing plane or the surface z = xy + t — a saddle sliding through time. Both readings are used below; neither is asserted about the artist.
- Security, L-REPRO, L-REFLECTOR, L-WIENER, the bed precondition, the DPI row — all unchanged. **DPI caution pinned above every row here: a slice of the volume cannot contain more than the frames did. This addendum adds instruments, not capability.**

---

## 1. The object: the spatiotemporal volume of a run

- **Definition.** V[p, f, t]: position p along the sequence, feature f (or a fixed projection of the residual stream), time t ∈ {layer index} or t ∈ {training step}. Two volumes, both kept: V_layer (p, f, layer) per forward pass; V_train (p, f, step) for a fixed probe batch across training.
- **Slices.** The (p, t) slice at fixed f is the epipolar-plane image of the run (Bolles, Baker, Marimont 1987 [U]): a representation moving at constant velocity along p traces a straight line of slope 1/v; a crossing is an occlusion — the tangle. The (f, t) slice at fixed p is a token's worldline through feature space.
- **Oblique slices.** A plane a·p + b·f + c·t = d through V mixes place and time — the reel's cut. The family of slices is parameterized by the plane's normal; the reads below fix which normals are used, so no slice is chosen after seeing the result.

---

## 2. Instruments

### M1 DRIFT-EPI (beside DRIFT-ROT of Addendum L)
- Local orientation of the (p, t) slice by the **structure tensor** J = G_σ ∗ (∇V ∇Vᵀ); the dominant eigenvector gives the worldline slope, the coherence (λ₁−λ₂)/(λ₁+λ₂) says whether a slope exists (Wanner & Goldluecke 2012 for light-field EPIs [U]).
- Drift at (p, t) = the slope; a **phase boundary** = a change in slope beyond the CI; a **tangle** = a crossing = coherence collapse at a point.
- **Null:** shuffle the t-axis of the probe batch; coherence must fall below 0.2 everywhere, else the instrument reads structure that is not there and is void.
- **Agreement rule:** DRIFT-EPI and DRIFT-ROT must agree on ≥ 80% of boundaries; below 50% one of them is struck by the Tutte-delta diagnostic (Δ of the slope sequence must be noise between boundaries).

### M2 BARCODE-FROM-VOLUME
- Cut the t-axis of V_train into segments where the slope is constant (piecewise-constant fit, BIC-penalized). The segmentation is compared to the barcode the gate produces (segment ids from the kernel's two vectors). Prediction: Jaccard of boundary sets ≥ 0.6 on planted chains; counter: ≤ 0.3 — the gate's segments and the run's phases are different objects, and the page says which one the word "phase" refers to.

### M3 SADDLE-IN-TIME (the literal caption)
- Track the negatively-curved decision points of R-CURV (κ_i < 0 by the exact transport LP) across t. A *transient equilibrium phase* is a saddle that persists for a run of steps and then moves. Report: number of saddles, mean lifetime, displacement per step. Prediction: saddle lifetime is bimodal (transient vs held); counter: unimodal — no phase structure in the saddles, and New Note's box is a drawing.

### M4 OBLIQUE READ
- For the consequence coordinates c_i(t) over t, the oblique slice with normal (1, 0, −v) reads the coordinates in the co-moving frame of velocity v — the frame in which a drifting representation stands still. Pre-registered v: the median slope from M1. Row: RES of c_i in the co-moving frame vs the lab frame on planted chains with drift injected; prediction: co-moving RES higher by ≥ 20%; counter: no gain — drift is not a coherent motion, and M4 is dropped.

---

## 3. Where the rows plug in

- R-PHASE (near-1 eigenvalue counts per segment) gains M2's segmentation as its second definition of "segment"; the two must agree on boundaries or R-PHASE reports "segment definition unstable".
- R-HOLE (Betti drift) runs on the same segments; Δβ is reported against M1's slope changes.
- DRIFT (Addendum L) becomes DRIFT-ROT; M1 is DRIFT-EPI; both in every drift table.
- R-CURV gains the time axis through M3.

---

## 4. Kills

- M1 null fails (coherence > 0.2 on the shuffled axis) → DRIFT-EPI void.
- M1 and DRIFT-ROT agree on < 50% → one struck by the delta diagnostic; if both fail it, "drift" leaves the page until an instrument survives its null.
- M2 Jaccard ≤ 0.3 → "phase" is defined once, by whichever object survives its null, and the other loses the word.
- M3 unimodal → the transient-equilibrium box is a drawing; R-CURV stays static.
- M4 no gain → dropped without a sentence.

---

## 5. Room, fetch orders, iteration

- Chase — V construction and the (p, t) slice, structure tensor, M1 null. Cameron — M2 segmentation and its Jaccard against the kernel barcode; M3 saddle tracking; M4. Foreman — the probe batch and V_train logging in every training run from now on (a reflector row: the volume is logged, not reconstructed). Wilson — the DPI caution pinned; agreement-rule arbitration; majority-of-three checkers.
- Fetch before any table (L-EQ): Bolles–Baker–Marimont 1987 (epipolar-plane images); Wanner–Goldluecke 2012 (structure-tensor orientation on EPIs); Hägerstrand 1970 (the space-time cube) [U]; the reel itself is a page line with its date, not a source.
- it.M0 — V logging; M1 with its null on planted chains with injected drift. it.M1 — M2 against the kernel barcode. it.M2 — M3 on R-CURV's saddles. it.M3 — M4. it.M4 — tables; DRIFT-EPI and DRIFT-ROT reconciled or one struck.

---

## 6. Reference instances

None claimed. The reel was read from its page (caption, likes, comments) in the desktop browser; no frame was measured. Every number above is a prediction with a null.
