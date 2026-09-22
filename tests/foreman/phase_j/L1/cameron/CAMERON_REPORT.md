# Cameron — final report (received ~23:32 local), claims as filed

All paths under this directory (SPJ/L1/cameron).

Verdict as filed: cost leg passes — exact Mode B' 1.04x SDPA forward, 1.12x forward+backward, inside 1.3x. Bed leg still RED: board bed retired; shell-game replacement passes A1, A3, A4 but not A2 (twin plateau).

F1 Cost forward GREEN. Mode B' 1.041x SDPA at S 4096 fp32 B1 H8 D64, computing the contract's G_ij = q_i...q_{j+1}. Old scan 5.97 ms was launch overhead; one tl.associative_scan launch = 0.019 ms (API checked in installed Triton 3.7.1 core.py:2770: tuple input, jit combine, reverse=).
- RED test_costb_fused.py IMPL=author: "RED author: fold: max |out - contract G_ij fold| = 3.775e+00 > 1e-4 | scan: max |Pi_fp32 - Pi_f64| = 1.927e+00 > 1e-4 | ratio: Mode B' fwd 11.422 ms / SDPA 4.593 ms = 2.487x > 1.3x"
- GREEN: "GREEN fused: fold 2.21e-06, scan 6.63e-05, ratio 1.041x"
- Stages (costb_fused_stages.json): head 0.037, scan 0.019, pre-rotate 0.070, post-rotate 0.066, SDPA 4.34 ms.
- By length (costb_extra.json): 1.132 S1024, 1.065 S2048, 1.035 S4096, 0.978 S8192; f64 parity at S4096 1.49e-5.
- Baseline: SDPA default dispatch here is EFFICIENT_ATTENTION at 4.55-4.75 ms, not MATH (MATH 25.4 ms); flash and cuDNN not compiled. Author JSON also records EFFICIENT_ATTENTION.
- torch.compile on plain contract-order eager code: 1.069x (eager 1.810x), fold err 1.42e-6 (costb_compile.py, costb_compile.json) — "no custom kernel" can stand.
- Kill: costb_impl.py retired as the cost-row object; replaced by costb_fused.py measured by test_costb_fused.py. Fusing rotations into k22 tile loads retired (saves at most 0.136 of 4.72 ms, 2.9%, and swaps stock SDPA for a custom kernel).

F2 Cost forward+backward GREEN. Hand-written backward: scan backward = Pi_k [reverse cumsum of conj(Pi_i) g_i] conj(Pi_{k-1}), exact for unit quaternions; rotation gradient a sum of g conj(v).
- RED test_costb_fwdbwd.py IMPL=author: "RED author: grad: max rel err vs contract f64 = 1.634e+00 > 1e-3 | ratio: fwd+bwd 37.142 ms / SDPA 17.574 ms = 2.113x > 1.3x"
- GREEN: "GREEN fused: grad 9.87e-07, fwd+bwd ratio 1.121x" (costb_fwdbwd_fused.json).

F3 costb_impl.py does not compute the contract's G_ij: Pi_i = q_0...q_i, so Pi_i conj(Pi_j) = C (q_{j+1}...q_i) C^-1 with C = q_0...q_j (CPU S=16 f64 check 3.3e-16). test_costb.py compared the implementation with itself. Arms import su2.py (contract order): defect confined to the cost row. RED/GREEN = the fold clauses of F1.

F4 Board bed retired; shell-game bed replaces it, still RED.
- RED test_bed_admissible.py ../../board_bed_result.json: "INADMISSIBLE board ply>=40: A1: no commuting-control ceiling measured | A2: no softmax twin trained on the bed | A4: no SU(2) construction on the bed | A3: best floor rule_free_tracker = 0.9968, +0.10 = 1.0968 > 1.0"
- Board state is mostly last-write-wins per square; softmax reads the latest write (tracker 0.9968).
- Shell game: each token swaps two cups; label = cup holding the ball; groups S3, S4 inside SO(3).
- Checks: A1 commuting ceiling measured <= 0.90; A2 twin plateaued and >= 0.90 at prefix 8; A3 best control + 0.10 <= 1.0; A4 exact SU(2) construction >= best control + 0.10.
- Floors at prefix >= 32 (S3 / S4): chance 0.333/0.250; multiset achievable (R=256) 0.333/0.252; multiset upper bound 0.364/0.282; last-8 window 0.327/0.239; twin 150 s (2 layers d64, 105k params) 0.340/0.243; twin longer 0.332 (150 s) / 0.255 (240 s); exact SU(2) construction on su2.prefix_scan fp32 L 64 every position 1.000/1.000. Sources shell_bed_floors.json, shell_bed_floors_long.json, fq_construction.json.
- S3, S4 pass A1, A3, A4. Fail A2: "INADMISSIBLE S3 ... A2: twin loss did not plateau; INADMISSIBLE S4 ... A2: twin loss did not plateau". S4 quarter losses 1.2388/1.1857/1.1689/1.1594; S3 1.0065/0.989/0.9769/0.9063, prefix-16 acc 0.326 -> 0.563.
- A5 out: twin plateaued but 0.4785 at prefix 8; no A5 construction built.
- No bar written.

OPEN (as filed): A2 plateau — default pre-register plateau on >=32 accuracy (change < 0.02 across 150/300/600 s) and rerun S4 twin at 600 s, then bar = twin floor + 0.10 on S4 (cleared prefix-8 in both runs: 0.9175, 0.9375). Whether f_Q learns S4 is Foreman's row. Scan fp32 error 6.6e-5 at S4096 without renormalisation (1.5x headroom under 1e-4). GPU contention: SDPA read 7.50 ms in the last run vs 4.3-4.75 earlier while a5_anchor_train.py shared the card; ratios alternated runs. Cameron's first author fwd+bwd RED (3.599x) ran without polling for a free GPU while Foreman's a5_bed.py held the card; superseded by the clean 2.113x run.
