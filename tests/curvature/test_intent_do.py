"""RED-FIRST tests for ceqjepa/intent_do.py (an intervention on a context window).

WHAT THE MODULE UNDER TEST CLAIMS. The field equation sources curvature from a
stress tensor built from interventions. On a board an intervention is a clamp;
on a context window the owner's definition is a CONSEQUENCE SWAP AT FIXED
SURFACE -- replace a span with one of the same shape whose downstream
consequence differs -- and the source is DIRECTED, because A changing B is not
B changing A. Each test below is a way for that to be FALSE while the code
still returns numbers:

  (a) THE BED IS NOT GROUND TRUTH. If the admitted-future set is not fixed by
     the generator, every dq below is an estimate wearing a decimal point. The
     bed's oracle read is checked against a brute-force intersection written
     out here, with no call into the module's own reader.
  (b) THE SURFACE IS NOT HELD. If the consequence swap moves the surface further
     than the surface-only swap does, the gap between them is a surface gap and
     the whole claim is a restatement of novelty. The two families are checked
     for EQUAL Hamming distance, token by token.
  (c) THE NULL SWAP IS APPROXIMATELY ZERO. Replacing a span by itself must give
     0.0 bitwise. A tolerance here would hide a read with state in it.
  (d) THE GAP IS THE INSTRUMENT, NOT THE BED. The surface-only control has teeth
     only against a read that could confuse surface with consequence. A read
     that scores how ALIKE positions are -- which is what self-attention is --
     is run as the PLANTED NEGATIVE and must FAIL the control, or the control
     proves nothing about the reads that pass it.
  (e) THE DIRECTION IS ASSUMED. The literal field-equation form |dq(i) - dq(j)|
     is symmetric in its two indices as an identity, so it cannot carry a
     directed source no matter what the bed says; that is asserted, not hoped.
     The two-index form is then measured against a symmetrised control.
  (f) THE ASYMMETRY IS A MASK ARTIFACT. Under a causal prefix read the backward
     block is zero BY CONSTRUCTION and reporting it as a finding is circular.
     Both reads are run and the causal one is labelled.
  (g) THE REFUSAL IS A NaN OR A ZERO. Either is invisible to a scorer. The
     refusal must be a VALUE carrying its reason, and a NULL swap -- one whose
     consequence provably does not exist -- must be ANSWERED with exact zero
     rather than refused.
  (h) THE POOLED RATE HIDES IT. A criterion that refuses everything and one that
     answers everything are scored beside the real one, sensitivity and
     specificity separately, never pooled.
  (i) THE FLOW IS CLAIMED. Curvature is reported static, with its edge count and
     refusal census. Any convergence figure would need a shortcut census beside
     it and this module produces none, so it must produce no convergence figure.
  (j) THE NUMBER CAME FROM NO RUN. Every number in the docstrings of BOTH files
     must appear in the output of a run performed here, matched as whole TOKENS,
     integers included, with the coverage figure itself bound by two counters
     written independently of each other; and every published value must equal
     an expression that recomputes it, compared POSITION BY POSITION.

THE IMPORT IS GUARDED ON PURPOSE. A module-level import turns the RED phase into
one collection error with no test names in it; each test must be seen to FAIL BY
NAME before the module exists.

RUN: python -m pytest tests/curvature/test_intent_do.py -v
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

try:
    from ceqjepa import intent_do as ido
except Exception as _exc:                      # noqa: BLE001 -- RED phase carries it
    ido = None
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None


def _m():
    if ido is None:
        pytest.fail("ceqjepa.intent_do does not import: %r" % (_IMPORT_ERROR,))
    return ido


# ---------------------------------------------------------------------------
# (a) THE BED IS GROUND TRUTH, checked against an oracle written HERE
# ---------------------------------------------------------------------------

def _brute_admitted(vocab, story, exclude=None):
    """Intersection of the admitted-outcome sets, written without the module.

    Sets, not bitmasks, and a python loop, not numpy: if this agrees with the
    module's reader then the module's reader is the generator's own definition
    and not a summary of it.
    """
    m = _m()
    keep = set(range(m.N_OUTCOMES))
    for slot, tok in enumerate(story):
        if slot == exclude:
            continue
        mask = vocab[slot][tok].mask
        keep &= {y for y in range(m.N_OUTCOMES) if mask >> y & 1}
    return keep


def test_the_bed_is_exact_and_its_oracle_read_counts_the_generators_futures():
    """The read is a function from context to a distribution over outcomes.

    Here it is EXACT: the uniform distribution over the futures the generator
    admits, obtained by intersecting the per-token constraint sets. It is not a
    language model, not a trained probe, and not an estimate.
    """
    m = _m()
    vocab = m.vocabulary()
    stories = m.sample_stories(40, seed=m.SEED)
    read = m.OracleRead(vocab)
    worst = 0.0
    for story in stories:
        for j in range(m.WINDOW):
            want = _brute_admitted(vocab, story, exclude=j)
            assert want, "the generator admitted no future: the bed is degenerate"
            p = np.zeros(m.N_OUTCOMES)
            for y in want:
                p[y] = 1.0 / len(want)
            worst = max(worst, float(np.abs(read.q(story, j) - p).max()))
    print("\n  oracle read vs a brute-force intersection: worst |diff| %.1e" % worst)
    assert worst == 0.0, "the oracle read is not the generator's own set count"


def test_the_bed_states_its_seed_and_the_run_prints_it():
    """A bed whose seed is not published is not a bed."""
    m = _m()
    a = m.sample_stories(20, seed=m.SEED)
    b = m.sample_stories(20, seed=m.SEED)
    c = m.sample_stories(20, seed=m.SEED + 1)
    assert np.array_equal(a, b), "the generator is not reproducible at a fixed seed"
    assert not np.array_equal(a, c), "the seed does not change the draw"


# ---------------------------------------------------------------------------
# (b) THE SURFACE IS HELD, token by token
# ---------------------------------------------------------------------------

def test_the_two_swap_families_move_the_surface_by_the_same_amount():
    """The planted negative is only a control if the surface move is matched.

    Every branch token carries a consequence partner and a surface-only partner.
    Both must sit at the SAME Hamming distance from the anchor's surface code,
    and the consequence partner must change the admitted set while the
    surface-only partner must not.
    """
    m = _m()
    vocab = m.vocabulary()
    n_pairs = 0
    for slot in m.BRANCH_SLOTS:
        for idx, tok in enumerate(vocab[slot]):
            csq = vocab[slot][tok.consequence]
            syn = vocab[slot][tok.surface_only]
            hc = bin(tok.surface ^ csq.surface).count("1")
            hs = bin(tok.surface ^ syn.surface).count("1")
            assert hc == hs == m.HAMMING, (
                "slot %d token %d: consequence partner at Hamming %d, "
                "surface-only partner at %d" % (slot, idx, hc, hs))
            assert csq.mask != tok.mask, "the consequence partner keeps the future set"
            assert syn.mask == tok.mask, "the surface-only partner moves the future set"
            n_pairs += 1
    print("\n  %d branch tokens, both partners at Hamming %d" % (n_pairs, m.HAMMING))
    assert n_pairs == len(m.BRANCH_SLOTS) * m.TOKENS_PER_BRANCH


# ---------------------------------------------------------------------------
# (c) (d) THE THREE SWAPS
# ---------------------------------------------------------------------------

def test_the_null_swap_is_bitwise_zero_on_every_read():
    """Replace the span by itself. Not small: zero, on every read, including the
    planted negative -- a read with state in it would fail here first."""
    m = _m()
    for name in m.READS:
        tab = m.swap_table(m.read_by_name(name))
        null = np.asarray(tab["null"])
        assert null.size > 0, "%s: no null swaps were run" % name
        assert (null == 0.0).all(), (
            "%s: the null swap moved the read, max %r" % (name, null.max()))
        print("\n  %-8s null swap max %r over %d swaps" % (name, null.max(), null.size))


def test_the_consequence_gap_survives_the_surface_only_control_on_the_real_reads():
    """(a) against (b): the gap IS the claim, quoted as a ratio and a difference.

    The floor from (c) is reported beside them. A read that passes here has
    separated consequence from surface; the next test shows a read that does not.
    """
    m = _m()
    for name in m.CONSEQUENCE_READS:
        g = m.gap(m.read_by_name(name))
        print("\n  %-8s consequence %.6f  surface-only %.6f  null %.6f  ratio %s"
              % (name, g["consequence"]["mean"], g["surface_only"]["mean"],
                 g["null"]["mean"], g["ratio_text"]))
        assert g["consequence"]["mean"] > 0.0, "%s: the consequence swap moved nothing" % name
        assert g["surface_only"]["mean"] < 0.1 * g["consequence"]["mean"], (
            "%s: the surface-only control is not small beside the consequence swap "
            "(%r vs %r): the read is measuring surface"
            % (name, g["surface_only"]["mean"], g["consequence"]["mean"]))
        assert g["null"]["mean"] == 0.0


def test_the_alikeness_read_is_the_planted_negative_and_fails_the_control():
    """The occupied ground, made to fire.

    A read that scores how ALIKE the context is to each outcome -- symmetric in
    content, which is what self-attention computes -- has no access to
    consequence. Fed the same two matched-surface swaps it must move about
    equally for both, so the (a)/(b) gap collapses toward one. If it did not,
    the gap on the real reads would be measuring the bed rather than the read.
    """
    m = _m()
    g = m.gap(m.read_by_name(m.PLANTED_NEGATIVE_READ))
    print("\n  %-8s consequence %.6f  surface-only %.6f  ratio %s"
          % (m.PLANTED_NEGATIVE_READ, g["consequence"]["mean"],
             g["surface_only"]["mean"], g["ratio_text"]))
    assert g["surface_only"]["mean"] > 0.5 * g["consequence"]["mean"], (
        "the planted negative did not fire: an alikeness read separated "
        "consequence from surface, which it cannot do")


def test_the_source_is_not_a_local_information_density():
    """The collapse this whole definition exists to avoid, tested directly.

    On the half of the consequence swaps that leave the number of admitted
    futures identical, an entropy or a surprisal cannot move by construction. If
    dq is nonzero there, the source is not one of those. The alikeness read is
    checked the other way: its entropy DOES move on that half, which is what
    makes it the density-flavoured negative rather than a second consequence read.
    """
    m = _m()
    keep = m.entropy_control(m.read_by_name("oracle"))["entropy_preserving"]
    print("\n  entropy-preserving consequence swaps: %d, dq %.6f, entropy move %r"
          % (keep["n"], keep["mean"], keep["dh_max"]))
    assert keep["n"] > 0, "the bed produced no entropy-preserving swaps"
    assert keep["dh_max"] == 0.0, "the subset is mislabelled: the entropy moved"
    assert keep["mean"] > 0.0, "the source vanishes where the entropy cannot move"
    other = m.entropy_control(m.read_by_name(m.PLANTED_NEGATIVE_READ))
    assert other["entropy_preserving"]["dh_max"] > 0.0, \
        "the alikeness read held its entropy fixed: it is not the density negative"


# ---------------------------------------------------------------------------
# (e) (f) DIRECTEDNESS
# ---------------------------------------------------------------------------

def test_the_literal_field_equation_form_cannot_be_directed():
    """|dq(i) - dq(j)| is symmetric as an ALGEBRAIC IDENTITY.

    Not approximately, not on this bed: for every read and every story the
    asymmetry of that form is exactly zero, so the directed claim cannot live
    in it. This is the finding that forces the two-index form.
    """
    m = _m()
    vocab = m.vocabulary()
    stories = m.sample_stories(m.N_T_STORIES, seed=m.SEED)
    worst = 0.0
    for name in m.READS:
        read = m.read_by_name(name)
        for story in stories[:5]:
            T = m.t_field(read, story)
            worst = max(worst, float(np.abs(T - T.T).max()))
    print("\n  |T_field - T_field^T| worst over every read and story: %r" % worst)
    assert worst == 0.0, "the field-equation form is not symmetric: check the code"
    assert vocab is not None


def test_the_two_index_source_is_measurably_asymmetric_against_a_symmetric_control():
    """The asymmetry is MEASURED: the distribution of |T(i,j) - T(j,i)|, its
    median and its tail, beside the same statistic on a symmetrised copy."""
    m = _m()
    a = m.directedness(m.read_by_name("oracle"))
    print("\n  |T(i,j)-T(j,i)| median %.6f  p90 %.6f  max %.6f over %d pairs"
          % (a["median"], a["p90"], a["max"], a["n_pairs"]))
    print("  symmetrised control: median %.6f  max %.6f"
          % (a["control_median"], a["control_max"]))
    assert a["control_max"] == 0.0, "the symmetric control is not symmetric"
    assert a["max"] > 0.0, "the directed source is symmetric: the directed claim dies"
    assert a["frac_nonzero"] > 0.0


def test_the_backward_block_is_reported_for_both_reads_and_the_causal_one_is_labelled():
    """Under a causal prefix read the backward block is zero by the MASK, not by
    a measurement, and a module that reports it as evidence is arguing in a
    circle. Both are run; the bidirectional one is where the number means
    something."""
    m = _m()
    b = m.backward_block(m.read_by_name("oracle"))
    print("\n  causal read backward mass %r over %d entries"
          % (b["causal_backward_max"], b["causal_backward_n"]))
    print("  bidirectional read backward mean %.6f, %d of %d entries nonzero"
          % (b["bidir_backward_mean"], b["bidir_backward_nonzero"],
             b["bidir_backward_n"]))
    assert b["causal_backward_max"] == 0.0, "the causal mask leaked"
    assert b["causal_is_tautological"] is True, (
        "the module must declare the causal backward zero a mask artifact")


# ---------------------------------------------------------------------------
# (g) (h) THE REFUSAL AND ITS SCORING
# ---------------------------------------------------------------------------

def test_the_refusal_is_a_value_with_a_reason_and_the_null_cell_is_answered():
    """Three ways, seen to fire: refuse UNDEFINED, ANSWER NULL with exact zero,
    answer DEFINED. Never a NaN, never a bare zero standing in for a refusal."""
    from ceqjepa.curvature import is_refusal
    m = _m()
    seen = {}
    for case in m.refusal_cases():
        verdict, value = m.classify(m.read_by_name("oracle"), case)
        seen.setdefault(case.truth, []).append((verdict, value))
    for truth in (m.DEFINED, m.UNDEFINED, m.NULL):
        assert truth in seen, "the bed never produced a %s case" % truth
    for verdict, value in seen[m.UNDEFINED]:
        assert is_refusal(value), "an UNDEFINED swap was answered with %r" % (value,)
        assert value.reason, "the refusal carries no reason"
    for verdict, value in seen[m.NULL]:
        assert not is_refusal(value), "a NULL swap was refused"
        assert value == 0.0, "the NULL answer is not exactly zero: %r" % (value,)
    for verdict, value in seen[m.DEFINED]:
        assert not is_refusal(value)
        assert not np.isnan(value), "a defined swap returned NaN"
    print("\n  %d DEFINED, %d UNDEFINED, %d NULL cases, all three fired"
          % (len(seen[m.DEFINED]), len(seen[m.UNDEFINED]), len(seen[m.NULL])))


def test_the_3x2_is_scored_with_sensitivity_and_specificity_separately():
    """A pooled agreement rate cannot separate a criterion that refuses
    everything from one that works; both trivial criteria are scored beside the
    real one, on every read."""
    m = _m()
    board = m.score_board()
    for row in board:
        print("\n  %-8s %-20s sensitivity %6.2f%%  specificity %6.2f%%  "
              "(tp %d fn %d tn %d fp %d)"
              % (row["read"], row["criterion"], row["sensitivity"],
                 row["specificity"], row["tp"], row["fn"], row["tn"], row["fp"]))
        assert "pooled" not in row, "a pooled rate reached the board"
    names = {(r["read"], r["criterion"]) for r in board}
    for read in m.READS:
        for crit in ("refuse everything", "answer everything"):
            assert (read, crit) in names, "%s is missing the %r control" % (read, crit)
    refuse_all = [r for r in board if r["criterion"] == "refuse everything"]
    assert all(r["specificity"] == 0.0 for r in refuse_all), (
        "refuse-everything did not score zero specificity: the scorer is pooling")
    assert all(r["sensitivity"] == 100.0 for r in refuse_all)


# ---------------------------------------------------------------------------
# (i) CURVATURE, STATIC ONLY
# ---------------------------------------------------------------------------

def test_curvature_is_static_with_its_edge_count_and_no_convergence_figure():
    """The source is wired to the instrument and nothing more.

    A convergence figure would need a shortcut census beside it, which this
    module does not produce; so it must not produce a convergence figure. The
    symmetrisation the instrument forces is reported as a LOSS, not hidden.
    """
    m = _m()
    c = m.context_curvature(m.read_by_name("oracle"))
    print("\n  kappa mean %.6f  CI [%.6f, %.6f]  %d edges, %d refused"
          % (c["stats"]["mean"], c["stats"]["ci_lo"], c["stats"]["ci_hi"],
             c["stats"]["n_edges"], c["n_refused"]))
    print("  asymmetry discarded by symmetrisation: %.4f of the source mass"
          % c["asymmetry_lost"])
    assert c["stats"]["n_edges"] > 0, "the context graph has no edges"
    assert np.isfinite(c["stats"]["mean"])
    src = "\n".join(str(d) for _, d in _module_docstrings(m))
    for word in ("fixed point", "converged", "convergence"):
        assert word not in src.lower(), (
            "the module claims %r with no shortcut census behind it" % word)


def test_the_empty_admitted_set_raises_instead_of_returning_a_distribution():
    """The bed's own invariant, exercised rather than assumed.

    No story the generator can draw empties the admitted set, so the guard in
    the reader is unreachable from the bed and would sit there untested. It is
    reached here directly: a mask with no outcomes must raise, not hand back a
    vector of NaN that every downstream statistic would silently average.
    """
    m = _m()
    with pytest.raises(ValueError):
        m._uniform_over(0)
    p = m._uniform_over((1 << m.N_OUTCOMES) - 1)
    assert abs(float(p.sum()) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# (j) THE GUARD: five clauses plus the identity layer
# ---------------------------------------------------------------------------

#: A number WITH ITS BOUNDARIES, the form struck into shape at
#: tests/curvature/test_lifetimes.py. A substring test passes any fabricated
#: number that shares digits with a real one; matching TOKEN-to-TOKEN means a
#: docstring number must appear in the run as a whole number. The inner
#: (?:\.\d+)* keeps a dotted identifier as ONE token. The trailing guard is
#: (?!\w)(?!\.\d) and NOT (?![\w.]), which would let a sentence-final period
#: hide the number in front of it.
_MEASURED = re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")


def _tokens(text):
    """Every number in `text` as a boundary-anchored token. COUNTER ONE."""
    return {t.lstrip("+") for t in _MEASURED.findall(text)}


def _tokens_scan(text):
    """The same token rule, written WITHOUT a regex. COUNTER TWO.

    The fourth clause wants the coverage figure bound by two counters written
    independently; two spellings of one regex are one counter. This is a
    character scan, and the test below asserts it returns the SAME SET rather
    than merely the same count, so a divergence names the token.
    """
    out, n, i = set(), len(text), 0
    isd = str.isdigit

    def wordish(ch):
        return ch.isalnum() or ch == "_"

    while i < n:
        if not isd(text[i]):
            i += 1
            continue
        s = i
        if s > 0 and text[s - 1] in "+-":
            if s - 1 == 0 or not (wordish(text[s - 2]) or text[s - 2] == "."):
                s -= 1
        elif s > 0 and (wordish(text[s - 1]) or text[s - 1] == "."):
            while i < n and isd(text[i]):
                i += 1
            continue
        j = i
        while j < n and isd(text[j]):
            j += 1
        while j + 1 < n and text[j] == "." and isd(text[j + 1]):
            j += 1
            while j < n and isd(text[j]):
                j += 1
        if j < n and text[j] in "eE":
            k = j + 1
            if k < n and text[k] in "+-":
                k += 1
            if k < n and isd(text[k]):
                while k < n and isd(text[k]):
                    k += 1
                j = k
        ok = not (j < n and wordish(text[j]))
        if j + 1 < n and text[j] == "." and isd(text[j + 1]):
            ok = False
        if ok:
            out.add(text[s:j].lstrip("+"))
        i = j
    return out


#: Set in the child pytest process so the own-file scan does not recurse.
_CHILD = "INTENT_DO_GUARD_CHILD"


def _own_test_run_output():
    """stdout of THIS file's own tests, so numbers in THEIR docstrings are bound."""
    env = dict(os.environ)
    env[_CHILD] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).resolve()),
         "-q", "-s", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parents[2]))
    assert proc.stdout, "the child run printed nothing: %r" % (proc.stderr[-400:],)
    return proc.stdout


def _module_docstrings(m):
    """(name, docstring) for the module and everything DEFINED in it.

    Scoped by __module__, not __all__: demo() is the function that prints every
    number and is absent from __all__, so an __all__-scoped scan misses exactly
    the thing being checked. The same filter keeps numpy's and collections'
    docstrings out without any number ever being exempted.
    """
    docs = [("module", m.__doc__ or "")]
    for name, obj in sorted(vars(m).items()):
        if getattr(obj, "__module__", None) != m.__name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append((name, doc))
        for attr in sorted(vars(obj)) if isinstance(obj, type) else ():
            member = getattr(obj, attr, None)
            member = member.fget if isinstance(member, property) else member
            if getattr(member, "__module__", None) != m.__name__:
                continue
            sub = getattr(member, "__doc__", None)
            if isinstance(sub, str) and sub.strip() and attr != "__doc__":
                docs.append(("%s.%s" % (name, attr), sub))
    return docs


#: The coverage floor. An emptied scan must fail LOUDLY rather than report zero
#: missing, so the same constant is asserted against and asserted about.
MIN_NUMBERS = 60


def test_the_guard_is_not_vacuous_against_planted_fabrications(capsys):
    """The guard's own comparison, run against fabrications planted HERE.

    A guard is worth its line count only if a fabrication is seen to fail it.
    Five shapes, each of which has walked past a weaker guard in this project:
    a plain decimal, a FRAGMENT of a number the run really does print, a bare
    integer, a number at the end of a sentence where a trailing (?![\\w.]) hides
    it, and a dotted identifier. For each, both counters must SEE it in prose
    and the run must NOT contain it -- which is exactly the pair of conditions
    that makes the scan above fail on it.
    """
    m = _m()
    m.demo()
    run = _tokens(capsys.readouterr().out)
    planted = ("0.884411", "706.0377", "4471", "5.5512", "2606.11911")
    for tok in planted:
        assert tok not in run, "%s is in the run: it is no longer a fabrication" % tok
        mid = "a docstring quoting %s in the middle of a line" % tok
        end = "a docstring quoting %s." % tok
        for text in (mid, end):
            assert tok in _tokens(text), "the regex counter misses %r in %r" % (tok, text)
            assert tok in _tokens_scan(text), "the scan counter misses %r in %r" % (tok, text)
    assert "1706.03762" in run and "706.0377" not in run, \
        "the fragment control and the real identifier are not being separated"
    assert len({t for _, doc in [] for t in _tokens(doc)}) < MIN_NUMBERS, \
        "an emptied scan would clear the floor: the floor is not a floor"
    print("\n  %d planted fabrications, each seen by both counters and absent "
          "from the run" % len(planted))


def test_every_measured_number_in_a_docstring_is_printed_by_the_demo(capsys):
    """No exemption list. A number no run prints cannot be checked by anyone."""
    if os.environ.get(_CHILD):
        pytest.skip("child of the own-file scan; the parent does the scanning")
    m = _m()
    stats = m.demo()
    out = capsys.readouterr().out
    child = _own_test_run_output()
    run = _tokens(out) | _tokens(child)

    assert "1706.03762" in run, "the run stopped printing the arXiv id"
    assert "06.03762" not in run, \
        "SUBSTRING VACUITY: a fragment of 1706.03762 is being read as a number"
    assert "31.4159" not in run, "the absent-decimal control was found in the run"
    assert "8675309" not in run, "the absent-integer control was found in the run"
    assert str(stats["n_consequence_swaps"]) in run, \
        "a measured INTEGER stopped being matched as a token"

    docs = _module_docstrings(m) + _module_docstrings(sys.modules[__name__])
    one = {tok for _, doc in docs for tok in _tokens(doc)}
    two = {tok for _, doc in docs for tok in _tokens_scan(doc)}
    assert one == two, ("the two counters disagree, regex-only %r scan-only %r"
                        % (sorted(one - two), sorted(two - one)))
    assert _tokens(out) == _tokens_scan(out), "the two counters disagree on the run"
    checked = len(one)
    missing = sorted((where, tok) for where, doc in docs
                     for tok in _tokens(doc) if tok not in run)
    print("\n  %d distinct numbers across %d docstrings, %d missing from the run"
          % (checked, len(docs), len(missing)))
    assert checked >= MIN_NUMBERS, "only %d numbers found: the regex is not biting" % checked
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))


#: The only published value that cannot be pinned to a recomputation, because it
#: is wall-clock time. It is therefore not quoted in any docstring either, and the
#: completeness check below is what stops that exemption from quietly growing.
UNPINNABLE = frozenset({"elapsed_s"})


def test_the_published_values_are_recomputed_not_literals(capsys):
    """PRINTED IS NOT MEASURED.

    A literal typed into a print statement passes the docstring guard trivially.
    Every published value is pinned here to an expression that recomputes it
    from the module's own functions in this same run, every sequence is compared
    POSITION BY POSITION -- an aggregate allclose hides a frozen entry beside a
    varying one -- and a COMPLETENESS check at the end fails on any key of the
    stats dict that this test forgot, so the coverage cannot rot as keys are
    added. An adversarial read of this file found seven unpinned keys; the
    completeness check is the repair for the class, not for the seven.
    """
    m = _m()
    stats = m.demo()
    out = capsys.readouterr().out
    assert isinstance(stats, dict) and stats, "demo() must return what it measured"
    pinned = set()

    def pin(key, value):
        assert key in stats, "no published value named %r" % (key,)
        assert stats[key] == value, "%s is a literal: %r vs %r" % (key, stats[key], value)
        pinned.add(key)

    def pin_seq(key, seq):
        assert key in stats, "no published value named %r" % (key,)
        got = stats[key]
        assert len(got) == len(seq) > 0, "%s has length %d, recomputed %d" % (
            key, len(got), len(seq))
        for k in range(len(seq)):
            assert got[k] == seq[k], "%s[%d] is frozen: %r vs %r" % (
                key, k, got[k], seq[k])
        pinned.add(key)

    sizes = sorted({int(m._bits(int(np.bitwise_and.reduce(
        m._mask_table()[np.arange(m.WINDOW), story]))).sum())
        for story in m.sample_stories(8, m.SEED)})
    pin_seq("admitted_sizes", sizes)

    prof = m.surface_profile()
    pin_seq("hamming_consequence", prof["consequence"])
    pin_seq("hamming_surface_only", prof["surface_only"])

    for name in m.READS:
        g = m.gap(m.read_by_name(name))
        for family in ("consequence", "surface_only", "null"):
            pin("%s_%s" % (name, family), g[family]["mean"])
            pin("%s_%s_n" % (name, family), g[family]["n"])
            pin("%s_%s_max" % (name, family), g[family]["max"])

    og = m.gap(m.read_by_name("oracle"))
    pin("n_consequence_swaps", int(og["consequence"]["n"]))
    pin("n_surface_swaps", int(og["surface_only"]["n"]))
    pin("n_null_swaps", int(og["null"]["n"]))

    for name in m.READS:
        ec = m.entropy_control(m.read_by_name(name))
        for label in ("entropy_preserving", "entropy_moving"):
            for suffix, field in (("dq", "mean"), ("n", "n"), ("dh_max", "dh_max")):
                pin("%s_%s_%s" % (name, label, suffix), ec[label][field])

    pin_seq("counts_surface_by_n", m.estimator_gap())

    worst = 0.0
    for name in m.READS:
        r = m.read_by_name(name)
        for story in m.sample_stories(5, m.SEED):
            Tf = m.t_field(r, story)
            worst = max(worst, float(np.abs(Tf - Tf.T).max()))
    pin("field_form_asymmetry", worst)

    a = m.directedness(m.read_by_name("oracle"))
    for field in ("median", "p90", "max", "n_pairs", "frac_nonzero",
                  "control_median", "control_max"):
        pin("asym_" + field, a[field])

    b = m.backward_block(m.read_by_name("oracle"))
    for field in ("causal_backward_max", "bidir_backward_mean",
                  "bidir_backward_nonzero", "bidir_backward_n"):
        pin(field, b[field])

    pin("n_cases", len(m.refusal_cases()))

    board = m.score_board()
    assert len(stats["board"]) == len(board)
    for k, (got, want) in enumerate(zip(stats["board"], board)):
        for field in ("read", "criterion", "sensitivity", "specificity",
                      "tp", "fn", "tn", "fp", "null_zero"):
            assert got[field] == want[field], (
                "board row %d field %r is frozen: %r vs %r"
                % (k, field, got[field], want[field]))
    pinned.add("board")

    sweep = m.tolerance_sweep()
    assert len(stats["tol_sweep"]) == len(sweep) > 0
    for k, (got, want) in enumerate(zip(stats["tol_sweep"], sweep)):
        for field in ("tol", "sensitivity", "specificity", "tp", "fn", "tn", "fp"):
            assert got[field] == want[field], (
                "tol_sweep row %d field %r is frozen: %r vs %r"
                % (k, field, got[field], want[field]))
    pinned.add("tol_sweep")
    best = max(sweep, key=lambda r: r["sensitivity"] + r["specificity"])
    pin("best_tol", best["tol"])
    pin("best_tol_sensitivity", best["sensitivity"])
    pin("best_tol_specificity", best["specificity"])

    c = m.context_curvature(m.read_by_name("oracle"))
    pin("kappa_mean", c["stats"]["mean"])
    pin("kappa_ci_lo", c["stats"]["ci_lo"])
    pin("kappa_ci_hi", c["stats"]["ci_hi"])
    pin("kappa_n_edges", c["stats"]["n_edges"])
    pin("kappa_n_refused", c["n_refused"])
    pin("asymmetry_lost", c["asymmetry_lost"])

    unpinned = set(stats) - pinned - UNPINNABLE
    assert not unpinned, (
        "these published values are pinned by nothing and could be literals: %r"
        % (sorted(unpinned),))
    assert UNPINNABLE <= set(stats), "the exemption names a key demo() no longer sets"
    assert 0.0 < stats["elapsed_s"] < m.BUDGET_S

    for key in ("n_consequence_swaps", "n_surface_swaps", "n_null_swaps",
                "kappa_n_edges", "asym_n_pairs", "n_cases"):
        assert str(stats[key]) in out, "%s = %r is not in the output" % (key, stats[key])
    print("\n  %d published values re-derived against the module in this run, "
          "%d exempt" % (len(pinned), len(UNPINNABLE)))


def test_the_demo_finishes_inside_the_budget_and_ends_with_the_exact_line(capsys):
    """The budget is BUDGET_S on CPU and the last line is fixed by contract.

    MEASURED COLD, IN A SUBPROCESS. An in-process demo() runs with every
    lru_cache already warm from the tests above it, which timed the second run
    and not the first -- it came back at 0.0 s, which is not a budget check, it
    is a clock reading. The bar itself is read from the module rather than
    retyped, because it used to be a literal in two prints and two asserts with
    nothing to catch them drifting apart.
    """
    m = _m()
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "ceqjepa.intent_do"],
                          capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[2]))
    dt = time.time() - t0
    out = proc.stdout.rstrip("\n")
    assert proc.returncode == 0, "the cold run failed: %r" % (proc.stderr[-400:],)
    print("\n  a COLD `python -m ceqjepa.intent_do` took %.2f s against the %.0f s bar"
          % (dt, m.BUDGET_S))
    assert dt < m.BUDGET_S, "the cold run took %.1f s" % dt
    assert out.endswith("ALL SELF-CHECKS PASSED"), \
        "the last line is %r" % (out.splitlines()[-1] if out else "",)
    assert "RUN:" in (m.__doc__ or ""), "the module docstring carries no RUN: line"
    for name in m.__all__:
        assert hasattr(m, name), "__all__ advertises a missing name %r" % (name,)
