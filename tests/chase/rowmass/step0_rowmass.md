# Row mass surgery -- stopped at step (1)

Run: `python step0_rowmass.py` (CPU, no training, no GPU touched). L-REPRO: re-run
reproduces byte-identical output, all five directories and the source trace
are read off disk, nothing is sampled or seeded.

## Fixed structure (printed before any scored quantity, per L-REFLECTOR)

Five directories exist, one per seed, at
`q2_ckpt_hardconcrete_seed{0,1,2,3,4}` under the scratchpad
(`OUT_DIR_TMPL` in `tests/foreman/q2/q2_certificate.py:87`). Each contains
exactly one file, `run_record.json` (272,706-272,884 bytes), and nothing
else -- no `.pt`/`.pth`/`.bin`/`.safetensors`/`.ckpt`.

Per seed, `run_record.json` gives: `operator=smprime` (arm (f)),
`steps=3538`, `hidden_size=128`, `n_layers=3`, `n_heads=8`, `d_head=16`,
`seq=512`, `batch=8`, `n_params=725391`, and a `final_eval_loss` around
1.04 nats (1.0391-1.0439 across the five seeds). Its keys are training
curves and scalars: `losses`, `grad_norms`, `row_l1_min`, `eval_losses`,
`final_eval_loss`, plus doc-split and repetition bookkeeping. There is no
`state_dict`, no tensor blob, no `G`, no `theta_head`, no `beta` value
anywhere in the file.

## Why there is no checkpoint

`q2_certificate.py:train_arm_f` (lines ~184-193) calls
`tests/foreman/eval/r3_eval.py:train_with_eval(out_dir=..., ...)`. That
function's own docstring says it is `` `ceq.hf.train.train()`, minus
checkpoint resume/save_every (not needed for these short scratchpad
runs...) ``. Its body (lines 178-274) builds the model, trains it in
memory, computes a `record` dict of scalars/lists, and writes only that
dict to `run_record.json`:

```
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_record.json"), "w") as fh:
        json.dump(record, fh, indent=2)
return record
```

`torch.save` and `.state_dict()` do not occur anywhere in `r3_eval.py`
(grep-confirmed). The real checkpoint-saving path,
`ceq/hf/train.py:_atomic_torch_save` / `torch.save` at line 268, belongs to
`ceq.hf.train.train()`, a different function this row's producer explicitly
bypassed. The trained `nn.Module` for every arm (f) seed was discarded when
`train_with_eval` returned. `results/m3_quintuple_v2_weights/` and
`results/m3_quintuple_v2_cuda_weights/` do carry real `.pt` weight files,
but for a different experiment (`twin`/`argmaxste`/`settled`/`softmax`
arms, `k/s/d/st` naming) -- none is `hardconcrete`/`smprime`, none is arm
(f) from this Q2 certificate.

## Verdict

No arm (f) model checkpoint survived, at any of the five seeds. Steps
(2) trained beta, (3) row mass, (4) exact-zero fraction, (5) the W5 delta
diagnostic, and (6) the pre-registered prediction all require forward-
passing a live trained model on a held-out batch and reading `G`,
`theta_head`, and `beta` out of it. None of that exists on disk. Per the
row's own stop rule -- "If no (f) checkpoint survived, say so immediately
- the row cannot proceed on a rebuild and must not fake one" -- this row
stops here. Retraining is both out of scope for a checkpoint-read row and
would not be reading the checkpoint the row was asked to read.

## What would unblock it

Re-run `q2_certificate.py`'s arm (f) training with `out_dir` pointed at a
`train_with_eval` call that also does `torch.save(model.state_dict(), ...)`
(or swap in `ceq.hf.train.train()`, which already has
`_atomic_torch_save`), for at least one seed, on the same fixed shapes
(hidden=128, layers=3, heads=8, d_head=16, seq=512). That is new training,
outside this row's no-training constraint, and is left for whoever owns
the next GPU slot.
