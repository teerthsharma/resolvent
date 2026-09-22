import sys, time, json, torch
sys.path.insert(0, r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad")
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")
import design4x5 as D

CEQAttention = D.CEQAttention
softmax_forward = D.softmax_forward
ORIG_FORWARD = D._ORIG_ATTN_FORWARD
ORIG_BUILD = D._ORIG_BUILD

if not D.poll_until_free(timeout_s=1800):
    print(json.dumps({"error": "GPU not free"}))
    sys.exit(1)

LAYERS, HEADS, SEQ, BATCH, VOCAB = D.LAYERS, D.HEADS, D.SEQ, D.BATCH, D.VOCAB
DEVICE = D.DEVICE

results = []
for hidden in (256, 384, 512):
    torch.manual_seed(0)
    model = ORIG_BUILD(hidden_size=hidden, n_layers=LAYERS, n_heads=HEADS,
                        seq=SEQ, vocab_size=VOCAB, operator="sgate").to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    CEQAttention.forward = softmax_forward
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    try:
        # warmup
        for _ in range(3):
            loss = model(input_ids=x, labels=x).loss
            opt.zero_grad(); loss.backward(); opt.step()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            loss = model(input_ids=x, labels=x).loss
            opt.zero_grad(); loss.backward(); opt.step()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t100 = time.time() - t0
    finally:
        CEQAttention.forward = ORIG_FORWARD
    proj = t100 / 100 * 3538
    results.append(dict(hidden=hidden, params=n_params, t100=t100, projected_3538=proj))
    print(json.dumps(results[-1]), flush=True)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

best = min(results, key=lambda r: abs(r["projected_3538"] - 514.8))
print(json.dumps({"results": results, "best": best}, indent=2))
with open("probe_result.json", "w") as f:
    json.dump({"results": results, "best": best}, f, indent=2)
