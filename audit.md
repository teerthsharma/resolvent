# Audit — status at 2026-09-27

Every line cites its commit or file. Nothing here is newly measured.

## 1. The research (Phase K, PROPAGATOR)

| iteration | verdict | commit |
|---|---|---|
| K0 | Instances reproduce and 2,994,211,739 FineWeb-Edu tokens are sharded. The resolvent layer is exact at 2.27× one SDPA forward+backward. R-DEPTH's ceiling line is broken: a hand-set construction scores 0.3982 against the 0.2539 line | e5c3cc7 |
| K1-a | Two chain beds were voided by hand-built window shortcuts. R-DEPTH now scores only the far band past 2·c_max·2^L | 49f5a2e |
| K1 | **R-DEPTH fails at R0.** The registered resolvent arm hits its learnability kill (0.2026 / 0.2354 < 0.8). The γ-anneal arm learns on 1 of 2 seeds and scores 0.0784 on the far band at 16k, against 0.95. The R0 language model is worse than the ALiBi twin at both seeds. Audit: 340 claims, 96 struck | 1554a74 |
| K2.0 | Sharpening the logit scale lifts depth > 160 at 4k from 0.2855 to 0.8750, short of 0.9. γ is not the wall. The hard-pointer ceiling at 16k is 0.4498 / 0.2292 | 8e1bd52, corrected c9daf85 |
| K2.1 pilot | The sparsemax arm learns (0.9971), but its pointer ceiling fails (0.8949 / 0.4513 / 0.2498). The registered branch is **IDENTITY** | 946f075 |
| K2 identity grid | 4 runs finished locally, 14 runs + 1 control + 6 walks on Kaggle. **Both Kaggle kernels report COMPLETE** (`kaggle kernels status`, checked today). **Outputs not yet downloaded or scored** | 7390fc5 |

**North star, unchanged since Phase J:** 8 of 19 gates (42%); the weakest leg is 30% (`docs/STATUS.md`).
- The language-model win belongs to relative position: ALiBi recovers 108.8% of it, FoX 100.2%.
- Prediction is tied by a histogram.
- One result holds: order, 0.8620 against 0.2860.

## 2. The website (teerthsharma.github.io/resolvent)

| change | state | commit |
|---|---|---|
| Theme matched to teerthsharma.vercel.app (sky ground, white cards, Instrument Sans, blue primary) | live | fb5fe46 |
| Viz engine (`docs/assets/viz/core.js`): figures mount by heading id, so no canon file is edited | live | e77db80 |
| Proof atlas: 207 theorems and lemmas, 73 definitions, 523 dependencies, parsed from `lean/CEQ` | live | e77db80 |
| Visual Status page (19-gate board) and visual North Star page (Born-series convergence) | live | e77db80, d543aaf |
| Dense pages, the canon included, moved unchanged under **Paper** | live | e77db80 |
| **Home rebuilt as a nine-chapter scroll story**, with a pinned stage that crossfades per chapter; old home kept as "The experiment, in full" | live | 8c19735 |
| Accent changed from violet to copper | live | 8c19735 |

**Author's asks not yet met:**
- Every page except Paper should show, not tell. Still text-heavy: What died (`FAILS.md`) and every Paper sub-page, which is by design.
- "Every maths and every proof visualised": only Status, North Star, the atlas and the home story carry figures. The canon's ~500 sections carry none.
- Story transitions were checked as stills, not under live scrolling; the browser pane was hidden and throttled to 1–2 fps.
- Dark-mode island snow sits over text (cosmetic).

**Process notes:**
- Three agent lanes (Wilson, Foreman, Cameron) were stopped at the author's request. All web work since has been done by hand.
- One near-miss: on this case-insensitive disk, writing `docs/status.md` overwrote `docs/STATUS.md`. It was restored from git before any commit, and visual pages now live in `docs/see/`.

## 3. Owed, in order

1. **K2 grid:** download both kernels' outputs (`tests/foreman/phase_k/K2_KAGGLE_HANDOFF.md`). Check the id_aL4_s0 T4-vs-4060 control first, then run `test_k2_grid_kaggle.py id`, then an Inspector pass.
2. **Site:** a visual What died page; then canon figures, one chapter at a time; a live-scroll check on a visible screen.
3. **Phase J debts:** re-measure contrasts against the ALiBi twin; bring the README current before any scale claim.
