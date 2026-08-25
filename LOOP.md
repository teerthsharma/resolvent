# Loop protocol — consequence-equilibrium attention

The Ralph loop feeds an identical prompt every iteration. Nothing carries over in
conversation. **This file and the backlog are the only memory.** An iteration that
does not write to them accomplished nothing.

Target deliverable: a Hugging Face release. Not a paper, not a benchmark table — a
thing on the Hub that someone can load and run.

## Each iteration, in order

1. Read `STATE.md`. It names the single current task. Do not re-plan.
2. Read `BACKLOG.md`. Do not start anything not in it.
3. Do the task. Test first — RED before GREEN, no exceptions.
4. Update `STATE.md`: what happened, what is next, what broke.
5. Move the task to `DONE.md` with its test name and measured result.
6. If the task revealed new work, add it to `BACKLOG.md` with a size estimate.

## Rules that do not change between iterations

- **No finding without a RED test that failed first.** An iteration reporting a
  result with no failing test behind it has produced an opinion, not a result.
- **Every number carries provenance.** Which commit, which machine, what control it
  was compared against. A number without a control is not a measurement.
- **Reuse before build.** The assets are listed in `THEORY.md` §0. Three of the four
  design decisions already have working code. An iteration that reimplements one of
  them has gone backwards.
- **The 0D-salience schedule from `triton-lang/kernels#22` is the control.** Any new
  schedule that does not beat it on a measured axis is not an improvement.
- **Declining is a valid outcome.** `caustic`'s governor always enters the do-nothing
  candidate. An iteration may conclude a task is not worth doing — record why in
  `DONE.md` and move on.
- **Blocked is not stuck.** If a task cannot proceed, write the blocker in `STATE.md`,
  move to the next backlog item, and do not spin.

## Completion promise

Output the promise ONLY when every one of these is true and checkable in the repo:

1. A `transformers`-loadable module exists and runs a forward pass on CPU.
2. The intervention-generalization eval exists, runs, and reports a number against a
   dense-attention control on the same data.
3. Every claim in the model card has a test name and a reproduction command.
4. `README.md` and the model card state the limits first, in the house style.

Do not output the promise to escape the loop. A false promise ends the run with a
broken artifact, which is worse than iteration 50 with an honest one.
