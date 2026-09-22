"""Train-step cost of f-lite vs the softmax twin vs the dense family, same harness as scale/probe.py.
`python measure.py ARM hidden layers heads seq batch` -> one JSON line (one fresh CUDA context).
`python measure.py all` -> each (arm, shape) in its own subprocess, rows to cost.json."""
import sys, time, json, subprocess, torch
SP = r"C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
REPO = r"C:/Users/seal/Desktop/New folder (32)"
SHAPES = ((384, 6, 6, 1024, 8), (512, 8, 8, 1024, 8), (128, 3, 8, 512, 8))


def run(arm, hidden, layers, heads, seq, batch, steps=12):
    sys.path.insert(0, SP); sys.path.insert(0, REPO)
    import r3_eval as RE, r1_gate as R, q2_certificate as Q2
    from ceq import arm_smprime
    from ceq.hf.modeling_ceq import CEQAttention
    import flite
    torch.manual_seed(0)
    try:
        if arm == "twin":
            CEQAttention.forward = Q2.softmax_forward
            m = RE.build(hidden_size=hidden, n_layers=layers, n_heads=heads, seq=seq, vocab_size=256, operator="sgate")
        else:
            arm_smprime.magnitude = R.FORMS["hard_concrete"]
            if arm == "flite":
                CEQAttention.forward = flite.flite_forward
            m = R.build_repaired(hidden_size=hidden, n_layers=layers, n_heads=heads, seq=seq, vocab_size=256, operator="smprime")
        m = m.cuda().train(); n = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        x = torch.randint(0, 256, (batch, seq), device="cuda")
        torch.cuda.reset_peak_memory_stats(); ts, losses = [], []
        for s in range(steps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            loss = m(input_ids=x, labels=x).loss; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0); losses.append(loss.item())
        dt = sorted(ts[3:])[len(ts[3:]) // 2]
        return dict(arm=arm, params=n, hidden=hidden, layers=layers, heads=heads, seq=seq, batch=batch,
                    tok_per_s=round(batch * seq / dt), step_s=round(dt, 4),
                    peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    loss_first=round(losses[0], 4), loss_last=round(losses[-1], 4))
    except torch.cuda.OutOfMemoryError:
        return dict(arm=arm, hidden=hidden, layers=layers, heads=heads, seq=seq, batch=batch, oom=True)


if __name__ == "__main__":
    if sys.argv[1] == "all":
        rows = []
        for shape in SHAPES:
            for arm in ("twin", "flite"):
                out = subprocess.run([sys.executable, __file__, arm, *map(str, shape)],
                                     capture_output=True, text=True)
                row = json.loads(out.stdout.strip().splitlines()[-1])
                rows.append(row); print(json.dumps(row), flush=True)
        json.dump(rows, open("cost.json", "w"), indent=1)
    else:
        print(json.dumps(run(sys.argv[1], *map(int, sys.argv[2:7]))))
