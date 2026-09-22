import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
SP = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
REPO = r"C:\Users\seal\Desktop\New folder (32)"
sys.path.insert(0, SP); sys.path.insert(0, REPO)
import torch, math
import abstention_deciles as AD
import design4x5 as D
D.DEVICE = "cpu"

model, blob = AD.build_arm("f")
batches = AD.eval_batches()


def capture_input(x, L):
    """Run the model on x, grab the pre-attn hidden state for layer L via a
    forward pre-hook, and return it."""
    at = model.model.layers[L].self_attn
    captured = {}

    def hook(module, inputs):
        captured["h"] = inputs[0]

    handle = at.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=x)
    finally:
        handle.remove()
    return captured["h"], at


def compute_case(window, L, head, i):
    x = batches[window // 8][window % 8:window % 8 + 1]  # [1, 512]
    h, at = capture_input(x, L)  # h: [1, 512, 128]

    B, S, _ = h.shape
    q, k, v = at.qkv(h).chunk(3, dim=-1)
    q = q.view(B, S, 8, 16).transpose(1, 2)  # [1,8,512,16]
    k = k.view(B, S, 8, 16).transpose(1, 2)
    u = at.m_head(h).squeeze(-1)  # [1,512]
    beta, qk, g = float(at.beta), float(at.qk), float(at.g)

    qh = q[0, head].double()   # [512,16]
    kh = k[0, head].double()   # [512,16]
    uv = u[0].double()          # [512]

    # magnitude m_t = clamp(sigmoid(1 + g*(u_t - 1)) * 1.2 - 0.1, 0, 1)
    m = torch.sigmoid(1.0 + g * (uv - 1.0)) * 1.2 - 0.1
    m = m.clamp(0.0, 1.0)

    scale = 1.0 / math.sqrt(16)

    # logits w_ij for j <= i, and R_ij = prod_{t=j+1}^{i} m_t
    qi = qh[i]  # [16]
    w = torch.empty(i + 1, dtype=torch.float64)
    for j in range(i + 1):
        w[j] = qk * float(qi @ kh[j]) * scale

    R = torch.empty(i + 1, dtype=torch.float64)
    R[i] = 1.0
    running = 1.0
    # R_ij for j from i-1 down to 0: R[j] = R[j+1] * m[j+1]
    for j in range(i - 1, -1, -1):
        running = running * float(m[j + 1])
        R[j] = running

    Z = float((R * torch.exp(w)).sum())
    w_ii = float(w[i])
    W_ii = math.exp(w_ii) / (Z ** beta)

    m_min = float(m[1:512].min()) if S > 1 else float("nan")

    return dict(W_ii=W_ii, w_ii=w_ii, Z=Z, beta=beta, m_min=m_min)


caseA = compute_case(window=22, L=2, head=1, i=242)
caseB = compute_case(window=0, L=0, head=4, i=0)

given_A = 2.0676820532
given_B = 1.2068454693

print("=== Case A: L=2 window=22 head=1 i=242 ===")
print(f"W_ii = {caseA['W_ii']:.10f}")
print(f"w_ii = {caseA['w_ii']:.10f}")
print(f"Z_i  = {caseA['Z']:.10f}")
print(f"beta = {caseA['beta']:.10f}")
print(f"abs diff from given ({given_A}) = {abs(caseA['W_ii'] - given_A):.10e}")
print(f"min m_t over t=1..511 (layer 2, window 22) = {caseA['m_min']:.10f}")

print()
print("=== Case B: L=0 window=0 head=4 i=0 ===")
print(f"W_ii = {caseB['W_ii']:.10f}")
print(f"w_ii = {caseB['w_ii']:.10f}")
print(f"Z_i  = {caseB['Z']:.10f}")
print(f"beta = {caseB['beta']:.10f}")
print(f"abs diff from given ({given_B}) = {abs(caseB['W_ii'] - given_B):.10e}")
