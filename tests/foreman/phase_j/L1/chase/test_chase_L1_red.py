"""Chase it.L1 RED tests: each asserts a claim as the record, code or commit
states it today. Every one is expected to FAIL now; a failure is the finding."""
import ast, inspect, json, math, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SPJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np
import indep

REC = os.path.join(SPJ, "record_L.jsonl")


def row(name):
    for line in open(REC, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # the LEAN-SCOUT line carries an unescaped Windows path
        if r.get("row") == name:
            return r
    raise KeyError(name)


def _spj_import(mod):
    if SPJ not in sys.path:
        sys.path.append(SPJ)
    return __import__(mod)


# PF-GAMMA: the record's unit-lattice eps=0.03 value is called P_min
def test_pf_record_eps003_is_lattice_minimum():
    rec = row("PF-GAMMA")["measured"]["unit_lattice_n400"]["eps=0.03"]["P_min"]
    dp = indep.pf_lattice_dp(0.03)
    assert rec <= dp * 1.001, f"record P_min {rec} vs DP global min {dp}"


# PF-GAMMA reprice: 'within 20% of 1/3' on the token lattice the gate lives on
def test_pf_repriced_bar_on_token_lattice():
    dp = indep.pf_lattice_dp(0.03)
    assert abs(dp - 1 / 3) / (1 / 3) <= 0.20, f"token-lattice min per boundary at eps=0.03 is {dp}, bar 1/3"


# DRIFT-SYNTH: the registered 1e-9 rad noiseless bar, run_horn as it stands
def test_drift_noiseless_registered_1e9():
    rh = _spj_import("run_horn")
    err, _ = rh.case_a()
    assert err <= 1e-9, f"run_horn case_a noiseless angle error {err} rad > 1e-9"


# Q3 control: the two orders must be two different computations
def test_q3_control_composes_two_orders():
    cq = _spj_import("chase_q2q3")
    tree = ast.parse(inspect.getsource(cq.q3_gate_diffs))
    rhs = {t.targets[0].id: ast.dump(t.value) for t in ast.walk(tree)
           if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name)}
    same = [k for k in ("scalar", "phase") if rhs[f"out_ab_{k}"] == rhs[f"out_ba_{k}"]]
    assert not same, f"out_ab_* and out_ba_* are the identical expression for {same}: the control never composes two gates"


# Q3 control claim applied to the named competitor: FoX (gate in the logits), value only at position 0
def test_q3_fox_read_is_order_blind():
    rng = np.random.default_rng(7)
    qk = rng.standard_normal(4)
    vals = np.zeros((4, 4)); vals[0] = [1.0, 2.0, 3.0, 4.0]
    def read(logf):
        c = np.cumsum(logf); s = qk + (c[-1] - c); p = np.exp(s - s.max()); p /= p.sum()
        return p @ vals
    la, lb = math.log(0.8), math.log(0.3)
    d = float(np.abs(read([0, la, lb, 0]) - read([0, lb, la, 0])).max())
    assert d == 0.0, f"FoX read distinguishes AB from BA: diff {d}"


# Q2 as the contract states it: the stated generators close at 60 (non-BFS test)
def test_q2_contract_generators_are_icosahedral():
    r = indep.q2_stated()
    ang = r["rotation_angle_ab_deg"]
    assert any(abs(ang - t) < 1e-6 for t in r["allowed_in_I"]), f"rotation angle of ab = {ang} deg, not an icosahedral angle"


# commit 3932ced: 'On the A5 word problem at 150 steps all three sit at or below chance, 1/60'
def test_a5_commit_all_arms_at_or_below_chance():
    m = row("L1-A5")["measured"]
    over = {k: m[k]["mean_acc_pos64"] for k in ("a_softmax_twin", "a2_fox_twin", "fq_quat_twin") if m[k]["mean_acc_pos64"] > 1 / 60}
    assert not over, f"above 1/60: {over}"


# commit 5b7e6b8: 'the baseline itself ran the math backend'
def test_costb_record_backend_matches_producer():
    rec = row("COST-B'")["measured"]["backend_observed"]
    prod = json.load(open(os.path.join(SPJ, "costb_timing_result.json")))["backend_observed"]
    assert rec.startswith(prod), f"record says {rec[:4]!r}, producer JSON says {prod!r}"


# PF-GAMMA's board GREEN: its test file must hold at least one collectable test
def test_pf_test_file_is_collectable():
    tree = ast.parse(open(os.path.join(SPJ, "test_pf.py"), encoding="utf-8").read())
    n = sum(isinstance(f, ast.FunctionDef) and f.name.startswith("test_") for f in tree.body)
    assert n >= 1, f"test_pf.py defines {n} test functions; pytest collects nothing"


# commit 3932ced's A5 numbers must reproduce when a5_bed.py is re-run at the same seeds (copy in beds/, paths redirected)
def test_a5_rerun_reproduces_record():
    new = json.load(open(os.path.join(HERE, "beds", "a5_bed_results.json")))
    m = row("L1-A5")["measured"]
    diffs = {}
    for k in ("a_softmax_twin", "a2_fox_twin", "fq_quat_twin"):
        a = np.array(m[k]["per_seed_final_train_loss"]); b = np.array([r["final_train_loss"] for r in new["results"][k]])
        diffs[k] = float(np.abs(a - b).max())
    assert max(diffs.values()) < 1e-3, f"max |record - rerun| per-seed train loss: {diffs}"


# commit 5b7e6b8 headline: 'the board bar is unsatisfiable per square'. Search per-square accuracies on a
# 0.001 grid for (a), (a''), (f_Q) meeting BARD-ARMS as registered: f_Q >= a + 0.10, f_Q >= a'' + 0.10, f_Q >= floor (iii)
def test_board_bar_unsatisfiable_per_square():
    floor = row("BOARD-BED")["measured"]["floor_iii_rule_free_tracker"]["ply>=40"]["per_square_acc"]
    g = np.round(np.arange(0, 1.0005, 0.001), 3)
    fq = g[g >= floor]
    feasible = [(a, fq.max()) for a in g if fq.max() >= a + 0.10]
    assert not feasible, f"{len(feasible)} feasible (a=a'', f_Q) points, e.g. a=a''={feasible[-1][0]}, f_Q={feasible[-1][1]}; floor {floor}"


# commit 5b7e6b8 compares the scan (5.97 ms) with SDPA's 4.38 ms but prices the ratio on SDPA = 5.86 ms:
# the same SDPA call timed standalone and inside Mode B' must agree within 10%
def test_costb_sdpa_timed_consistently():
    m = json.load(open(os.path.join(SPJ, "costb_timing_result.json")))
    a, b = m["ms_sdpa_fwd"], m["stage_ms_fwd"]["sdpa"]
    assert abs(a - b) / b <= 0.10, f"standalone SDPA {a:.3f} ms vs in-B' SDPA stage {b:.3f} ms; ratio on the stage = {m['ms_bprime_fwd']/b:.3f}"
