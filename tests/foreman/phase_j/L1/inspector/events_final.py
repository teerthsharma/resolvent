import json
B = r"C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
CAM = "SPJ/L1/cameron/"
E = [
# ---- Cameron: GPU re-runs (costb_fused.py md5 edd8af50..., the 23:11:14 file; copies run from SPJ/L1/inspector/runs/cam)
("Cameron", "clean", CAM + "test_costb_fused.py IMPL=author (RED)",
 "GPU re-run after poll_until_free: exit 1, 'RED author: fold: max |out - contract G_ij fold| = 3.775e+00 > 1e-4 | scan: max |Pi_fp32 - Pi_f64| = 1.927e+00 > 1e-4 | ratio: Mode B' fwd 10.575 ms / SDPA 6.283 ms = 1.683x > 1.3x'; fold and scan errors bitwise equal to the log, ratio 1.683x vs logged 2.487x (SDPA baseline 6.28 ms vs 4.59 ms, different clock state), still > 1.3x"),
("Cameron", "clean", CAM + "test_costb_fused.py IMPL=fused (GREEN) on the post-23:11 costb_fused.py",
 "GPU re-run x2: exit 0, 'GREEN fused: fold 2.21e-06, scan 6.63e-05, ratio 1.088x' (7.498/6.890 ms) and 'ratio 1.034x' (6.987/6.755 ms); errors bitwise equal to the log; logged 1.041x sits inside the re-run spread; the forward GREEN now holds for the edited file"),
("Cameron", "clean", CAM + "test_costb_fwdbwd.py IMPL=author (RED)",
 "GPU re-run: exit 1, 'RED author: grad: max rel err vs contract f64 = 1.634e+00 > 1e-3 | ratio: fwd+bwd 44.556 ms / SDPA 25.460 ms = 1.750x > 1.3x'; grad error bitwise equal, ratio 1.750x vs logged 2.113x, still > 1.3x"),
("Cameron", "clean", CAM + "test_costb_fwdbwd.py IMPL=fused (GREEN)",
 "GPU re-run x2: exit 0, 'GREEN fused: grad 9.87e-07, fwd+bwd ratio 1.081x' (27.964/25.858 ms) and '1.093x' (27.756/25.388 ms); logged 1.121x; all inside 1.3x"),
# ---- Cameron: findings and report claims
("Cameron", "clean", "finding F1: COST leg survives, Mode B' 1.041x SDPA forward",
 "REDs 4729, 4742 precede GREENs 4730, 4743 and finding 4744; the GREEN re-runs on the current file at 1.034x-1.088x; supersedes pass-1 'not re-run: GPU'"),
("Cameron", "clean", "finding F2: forward+backward 1.121x",
 "RED 4742 precedes GREEN 4743 and finding 4744; fused fwd+bwd GREEN at 23:11:39 was produced after the 23:11:14 edit; re-run 1.081x / 1.093x"),
("Cameron", "clean", "F1 baseline: SDPA default dispatch EFFICIENT_ATTENTION, flash and cuDNN unavailable, MATH ~25 ms",
 "re-run sdpa_ms_by_backend: EFFICIENT_ATTENTION 5.890 / 6.661 / 6.718 ms, MATH 26.5 / 28.5 / 28.1 ms, FLASH and CUDNN 'No available kernel'; backend identity holds; the 4.55-4.75 ms level is not reproduced at this clock state (SM 1890-2670 of 3105 MHz, 77-84 C)"),
("Cameron", "clean", "F1 citation: Triton 3.7.1 core.py:2770 associative_scan(..., reverse=)",
 "installed triton 3.7.1, language/core.py line 2770: 'def associative_scan(input, axis, combine_fn, reverse=False, ...)'"),
("Cameron", "struck", "F1 kill: rotation fusion into k22 tile loads retired (saves at most 0.136 of 4.72 ms, 2.9%)",
 "report-only; no RED, GREEN or finding event on the board; the 0.070+0.066 stage times come from costb_fused_stages.json with no test; unbound"),
("Cameron", "clean", "finding F3: costb_impl builds Pi_i = q_0...q_i, fold err 3.775; test_costb.py compared the implementation with itself",
 "RED fold clause 4729 precedes finding 4745; re-run fold err 3.7753911924862122 bitwise; SPJ/test_costb.py:29,80 builds its reference with its own hillis_steele_prefix_quat, Pi[i] = q0*...*qi, the same order as costb_impl; agrees with Wilson [V] costb_impl oldest-left"),
("Cameron", "struck", "F3 evidence: CPU S=16 f64 check Pi_i conj(Pi_j) = C (q_{j+1}...q_i) C^-1 to 3.3e-16",
 "no script in SPJ/L1/cameron produces 3.3e-16 (the string appears only in CAMERON_REPORT.md) and no test event on the board; unbound"),
("Cameron", "clean", CAM + "test_bed_admissible.py RED on board_bed_result, shell_bed_floors, shell_bed_floors_long",
 "CPU re-run: exit 1 x3, output identical to pass-1 run (line endings only differ): 'INADMISSIBLE S3 prefix>=32 (best control multiset 0.3641): A2: twin loss did not plateau | A2: twin never learned the task (short-prefix acc 0.7945 < 0.90)'"),
("Cameron", "clean", "finding F4 floors: chance 0.333/0.250, multiset 0.333/0.252 (upper 0.364/0.282), last-8 0.327/0.239, twin 0.340/0.243, longer twin 0.332/0.255, prefix-8 0.9175/0.9375, A5 0.4785",
 "every number equals shell_bed_floors.json / shell_bed_floors_long.json; RED 4776 precedes finding 4778; long-run RED 4794 precedes the report"),
("Cameron", "struck", "F4 claim: S3 and S4 pass A4 (exact SU(2) construction >= best control + 0.10)",
 "test_bed_admissible.py:8-9,54-56 reads A4 from fq_construction.json, and fq_construction.py has no RED or GREEN event (pass-1 struck); A4 rests on an unbound construction"),
("Cameron", "clean", "finding F4: board bed retired (A3 0.9968 + 0.10 > 1.0)",
 "carried from pass 1; re-run 'A3: best floor rule_free_tracker = 0.9968, +0.10 = 1.0968 > 1.0' unchanged; 0.996844 equals Wilson [V]"),
("Cameron", "clean", "Cameron claims vs Wilson [V]",
 "no contradiction: costb_impl oldest-left, EFFICIENT_ATTENTION observed, old scan 5.97 ms, BOARD-BED 0.996844 all agree; Wilson's flash flag is flash_sdp_enabled(), Cameron's is a kernel dispatch attempt"),
# ---- Chase rivals
("Chase", "clean", "SPJ/L1/chase_rivals/test_rivals.py rivals_stub (RED 0/10)",
 "CPU re-run: '{\"module\": \"rivals_stub\", \"passed\": 0, \"total\": 10}', every row NotImplementedError, identical to red_run2.txt; exit status 0, so RED is readable only from the passed count"),
("Chase", "clean", "SPJ/L1/chase_rivals/test_rivals.py rivals_impl (GREEN 10/10)",
 "CPU re-run: '{\"module\": \"rivals_impl\", \"passed\": 10, \"total\": 10}', every row identical to green_run.txt (T1a 0.000e+00 ... T6 0.000e+00)"),
("Chase", "struck", "board test events 4797-4799 (rivals RED 0/9, RED 0/10, GREEN 10/10)",
 "schema is test(status red/green, name); these carry 'state' and 'file', no 'status' and no 'name'; house.py reads e.status, so no red or green reaches the board; under 'a finding whose test name never appears with status: red is unbound' they bind nothing"),
("Chase", "struck", "finding 4800: a2 is Qwen headwise G1 (T1a 0.0, T1b 2.50 / 4.4e-16, T6)",
 "its RED is logged only under 'state', never 'status: red'; unbound (re-run reproduces the numbers)"),
("Chase", "struck", "finding 4801: GDN decay is FoX gate up to a per-head power; recurrence = FoX-masked linear attention (T2a, T2b)",
 "RED only under 'state'; unbound"),
("Chase", "struck", "finding 4802: KDA/GDN non-commuting but per-token spectrum in [0,1]; unit-circle spectrum not shipped (T3a-T3c)",
 "RED only under 'state'; unbound; T3c row (test_rivals.py 23:35:03) was written after rivals_impl.py (23:33:33) existed"),
("Chase", "struck", "finding 4803: MiniMax lightning = exp(ALiBi) = constant-f FoX; MLA/DSA/MoBA/NSA order-blind (T4, T5)",
 "RED only under 'state'; unbound"),
("Chase", "clean", "report M1-M10 literature facts with a SOURCES.txt line (Qwen PPL 6.026/5.761/5.792, 46.7%->4.8%, NSA 9.0x/6.0x/11.6x, MoBA 6.5x, KDA 75% / 6.3x, licenses, RoPE bases)",
 "each number has a line in SOURCES.txt; SOURCES.txt is Chase's restatement, the nurses' verbatim fetch outputs are not on disk; source correctness not ruled"),
("Chase", "struck", "report numbers with no source line: KDA prefill 2.9x; Qwen3-Next '10 times inference throughput for context over 32K'; '12 x (3 GDN + 1 gated attention)'",
 "absent from SOURCES.txt and from every board event; unbound"),
("Chase", "struck", "report claim 3: learned phase ships in no lab, retire",
 "no finding or test event; a negative over ten sampled sources; unbound"),
("Chase", "clean", "Chase rivals claims vs Wilson [V]",
 "no overlapping [V] fact contradicted; rivals_impl a2_gate = Y * sigmoid(XW + b) per head matches design4x5.py:211-213 softmax_gated_forward"),
]
if __name__ == "__main__":
    with open(B, "a", encoding="utf-8") as f:
        for c, v, n, w in E:
            f.write(json.dumps({"t": "audit", "agent": "Inspector", "cites": c, "verdict": v, "name": n, "why": w}) + "\n")
    print(len(E), sum(v == "struck" for _, v, _, _ in E))
