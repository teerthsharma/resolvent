"""House leap, own instance (CPU, no training). Does the trained softmax twin carry a
relative-recency prior in its position-only layer-0 logits? Approximation: token
embeddings omitted, so this reads what the absolute position table alone can express."""
import torch, json
out = {}
for tag, p in (("twin_a", "d45_ckpt_a_ss0_pair/model.pt"), ("fox", "phase_j/ckpt_a2F_ss0/model.pt"),
               ("family_f", "d45_ckpt_f_ss0_pair/model.pt")):
    sd = torch.load(p, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
    E = sd["model.embed_positions.weight"].double()
    rows = []
    for L in range(3):
        pre = f"model.layers.{L}."
        X = torch.nn.functional.layer_norm(E, (E.shape[1],), sd[pre + "input_layernorm.weight"].double(),
                                           sd[pre + "input_layernorm.bias"].double())
        Wqkv = sd[pre + "self_attn.qkv.weight"].double()
        Q, K = X @ Wqkv[:128].T, X @ Wqkv[128:256].T
        n = E.shape[0]; i, j = torch.tril_indices(n, n)
        delta = i - j
        for h in range(8):
            P = (Q[:, h*16:(h+1)*16] @ K[:, h*16:(h+1)*16].T / 4.0)[i, j]
            tot = ((P - P.mean())**2).sum()
            means = torch.zeros(n, dtype=torch.float64).index_add_(0, delta, P) / torch.bincount(delta, minlength=n).double()
            r2 = float(1 - ((P - means[delta])**2).sum() / tot)
            near = float(means[1:17].mean()); far = float(means[64:257].mean())
            rows.append({"layer": L, "head": h, "toeplitz_R2": round(r2, 4), "recency_gap_nats": round(near - far, 3)})
    out[tag] = rows
    gaps = [r["recency_gap_nats"] for r in rows]; r2s = [r["toeplitz_R2"] for r in rows]
    print(f"{tag:9s} recency gap (near 1-16 minus far 64-256), max {max(gaps):6.2f}  mean {sum(gaps)/len(gaps):6.2f}   Toeplitz R2 max {max(r2s):.3f} mean {sum(r2s)/len(r2s):.3f}")
    print("          layer-0 gaps by head:", [r["recency_gap_nats"] for r in rows if r["layer"] == 0])
json.dump(out, open("phase_j/N1/house/leap_instance.json", "w"), indent=1)
