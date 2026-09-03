"""MERCURY it.28 marker node. Three RED against V20_R15_THEORY_TABLE.md, three GREEN
controls against the unmutated repo. Every RED goes GREEN on a table edit alone.
If a control ever goes RED the defect was 'repaired' by moving the code, not by citing it.
"""
import io, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"

def _lines():
    return io.open(TABLE, encoding="utf-8").read().splitlines()

def _cell(name):
    """Lines of one §2 cell, heading exclusive."""
    ls = _lines(); out = []; on = False
    for ln in ls:
        if ln.startswith("### "):
            on = name in ln
            continue
        if ln.startswith("## "):
            on = False
        if on: out.append(ln)
    assert out, f"cell {name!r} not found"
    return out

def _sec5():
    ls = _lines(); i = next(k for k, l in enumerate(ls) if l.startswith("## §5"))
    return ls[i:]

PTR = re.compile(r"`[\w./-]+\.(?:py|lean|md|jsonl):\s*[\d*]")

# ---------- RED: the table as it stands ----------

def test_m28a_q6_w3_armpl_forward_carries_a_pointer():
    """Q6/W1 cites ArmSMPrime.forward @ ceq/arm_smprime.py:577. Q6/W3 asserts the
    identical fact about ArmPL.forward and cites nothing."""
    for ln in _cell("CELL Q6/W3"):
        if "`ArmPL.forward`" in ln:
            assert PTR.search(ln), f"ArmPL.forward asserted with no path:line: {ln.strip()}"

def test_m28b_q2_w3_planted_negative_cites_its_generator():
    """The constant 1.0845223424 is measured on planted negative PN-2 `sign_flip_gate`.
    The name carries no pointer, so the leap cannot open the negative it rests on."""
    for ln in _cell("CELL Q2/W3"):
        if "`sign_flip_gate`" in ln:
            head = ln.split("`sign_flip_gate`")[1].split(".")[0]
            assert PTR.search(head), f"sign_flip_gate named with no path:line: {ln.strip()}"

def test_m28c_sec5_secs_cites_its_producer():
    """§5 rules the whole arena price un-synchronised on the basis of the `secs` field
    and never says where `secs` is written."""
    for ln in _sec5():
        if "`secs`" in ln:
            assert PTR.search(ln), f"secs asserted with no path:line: {ln.strip()}"

# ---------- GREEN: controls, unmutated repo ----------

def test_m28a_control_armpl_forward_is_at_arm_pl_405():
    src = io.open(ROOT / "ceq" / "arm_pl.py", encoding="utf-8").read().splitlines()
    assert src[357].startswith("class ArmPL")      # :358
    assert "def forward" in src[404]                # :405

def test_m28b_control_sign_flip_gate_has_no_producer_py():
    """The name resolves nowhere but tests -- which is why the missing pointer bites."""
    hits = [p for p in ROOT.rglob("*.py")
            if "sign_flip_gate" in io.open(p, encoding="utf-8", errors="ignore").read()]
    assert hits, "sign_flip_gate vanished from the tree"
    assert all("test" in p.name for p in hits), [str(p) for p in hits]

def test_m28c_control_secs_is_a_journalled_field():
    src = io.open(ROOT / "scripts" / "v15_r1.py", encoding="utf-8").read().splitlines()
    assert 'r["secs"]' in src[1005]                  # :1006
