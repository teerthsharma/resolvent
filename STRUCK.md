# STRUCK

Every constant this project has struck: withdrawn from the record, never
silently replaced. A number enters the list the moment an audit strikes it.

**This file is a pointer, not a copy.** The registry is a module-level dict in
`tests/loop/test_no_struck_constant_ships.py:47`, asserted non-empty by
`inspector.py:459` and scanned against the shipped code and the lead documents
by that same test -- a second hand-maintained strike list is exactly how a
strike gets dropped from one copy and survives in the other.

Regenerate this file from the registry; never edit the table by hand:

```
python scripts/render_struck.py > STRUCK.md
```

Rendered from `tests/loop/test_no_struck_constant_ships.py` at `c54a77d`. 12 entries.

| constant | cite | why it was struck |
| --- | --- | --- |
| `-1.389` | `tests/loop/test_no_struck_constant_ships.py:48` | M2 decay exponent. WITHDRAWN as a `floor = 1e-6` artifact (MODEL_CARD.md:615, 2026-09-10; the earlier README.md:136 citation moved once README.md's own text shifted past it -- re-grep for the exponent before trusting either line number). NO REPLACEMENT IS PUBLISHED — least squares on the floor=0 rates gives -0.958 (R^2 0.9990) while the audit that forced the correction reports -1.221 (R^2 0.9662), and those disagree. The honest repair is DELETION of the claim, never substitution of a number nobody can defend. |
| `0.9938` | `tests/loop/test_no_struck_constant_ships.py:57` | R^2 belonging to the withdrawn -1.389; withdrawn with it. |
| `-0.5173` | `tests/loop/test_no_struck_constant_ships.py:58` | Lower endpoint of the 'live rows only' K1 slope interval published in D1.md and done5.md. UNVERIFIED: the interval exists in no .py, .json, .jsonl or .txt in the tree, and no producer computing a live-rows slope with a confidence interval could be located. The only file that computes a live-rows slope, scale/foreman_theta_tv.py, runs on the 120-draw three-point data, emits -0.3323 rather than -0.4654, contains no resampling machinery at all, and exits 1. The 400-draw producer, scale/arm_a_k1.py, has no live-rows path. Round 6 iteration 14. |
| `-0.4160` | `tests/loop/test_no_struck_constant_ships.py:68` | Upper endpoint of the same unverified interval; withdrawn with it. |
| `-0.4654` | `tests/loop/test_no_struck_constant_ships.py:69` | The 'live rows only' K1 point slope. UNVERIFIED for the same reason as -0.5173: no producer emits it. The verdict does NOT depend on it -- the as-computed slope -0.4137 [-0.4579, -0.3704] lies entirely below the -0.30 trigger on its own, and that pair reproduces exactly from the shipped producer. |
| `-1.826` | `tests/loop/test_no_struck_constant_ships.py:76` | M2 slope as first reported. Contradicted by measurement at iteration 20 — the shipped operator reads -1.298. Propagated in a RUN voice at iteration 17. |
| `1.471448` | `tests/loop/test_no_struck_constant_ships.py:80` | M5 tail norm at s=128. FABRICATED — iteration 35 swept 1,800 settings (dim, seed, generator layout, rho, lam, hops) and ZERO produced it. Measured: 0.880500 at hops=2, 0.882030 at hops=4. |
| `1.343174` | `tests/loop/test_no_struck_constant_ships.py:85` | M5 tail norm at s=512, same fabrication. Measured: 1.292741. |
| `0.743864` | `tests/loop/test_no_struck_constant_ships.py:86` | U1/N3 pilot Spearman rho between the harmonic-measure rank and the masking-displacement rank, disclosed at tests/cameron/test_harmonic_attribution.py:47. STRUCK at R9 iteration 2: NO PRODUCER HAS EVER EXISTED. The battery calls nine names on scale/negation_scope.py -- absorbing_boundary_kernel, harmonic_measure, harmonic_label_batch, train_control_arm, masking_displacement, rank_crosscheck, dead_control_arm, u1_attribution_run and PREREGISTERED_RHO_FLOOR -- and scale/negation_scope.py has never defined any of them: `git log -S"def <name>(" --all --oneline -- scale/negation_scope.py` returns 0 commits for each of the nine, against 1 for the control `def nrmse(`, which that file does define. For the lead name the repo-wide form `git log -S"def absorbing_boundary_kernel(" --all --oneline -- "*.py" ":!tests/"` also returns 0, against 3 for the control `def path_product(`. (The bare `git log -S` across ALL refs is not an absence proof: it finds this audit's own recordings, and `def harmonic_measure(` in scale/kirchhoff.py since 22036de, an unrelated function of the same name. Search corrected 2026-09-11.) This is not a deleted producer and not a figure transcribed from a sibling task: the number has no possible source in any state this repository has ever been in. The 1.471448 class, with no floor -- 1.471448 at least had a sweep that could look for it. |
| `0.656532` | `tests/loop/test_no_struck_constant_ships.py:109` | Lower edge of the U1/N3 pilot bootstrap CI; struck with 0.743864. |
| `0.816955` | `tests/loop/test_no_struck_constant_ships.py:110` | Upper edge of the U1/N3 pilot bootstrap CI; struck with 0.743864. |
| `5.4944e-13` | `tests/loop/test_no_struck_constant_ships.py:111` | Karcher residual in float64. STRUCK by the health inspection at round 5 iteration 21: asserted in a [RUN] voice with NO LIVE PRODUCER. It appeared only in a code comment and in prose. It came from a throwaway float64 check whose OWN output was defective -- that script printed `nan` for both means because it omitted the norm clamp the probe has, and 5.4944e-13 was its `min` at settings nobody recorded. The probe reproduces 7.481e-09 / 8.155e-09 / 8.405e-09 at published settings, and 7.307e-13 / 7.958e-13 / 8.405e-13 at --tol 1e-15 --steps 400. 5.4944e-13 is not the mean, min or max at any k. Order of magnitude right, so stale rather than fabricated -- but a number asserted [RUN] with no producer is the 1.471448 class. |
