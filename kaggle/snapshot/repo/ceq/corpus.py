"""An interventional corpus of executed programs.

The oracle is CPython. No answer key is written anywhere in this file: every
label is the value the interpreter actually produced.

WHY THIS DOMAIN. `p(y | do(x))` is not identifiable from observational text, so
the transition operator has to be fitted where interventions are real and
outcomes are measured. Physics simulation was tried and failed the null: a
physics prior scored 0.3667 against 0.3654 for a sham prior with its action
column permuted and 0.3654 for matched-lambda scratch -- indistinguishable from
regularization strength. Executed code has a free oracle and an unconfounded
intervention: edit a token, run it again, read the result.

WHAT AN INTERVENTION IS HERE. A single token edit to the program source, with
the outcome re-measured by executing the edited program. That is a `do()`, not a
correlation: the edit is applied, not observed.

THE NEGATION GATE. Every program ends with

    if flag: acc = -acc

so one token inverts the sign of the whole outcome. That operation is exactly
what a non-negative influence Jacobian cannot express -- and the minimum
influence entry of the Kleene star of a non-negative matrix is exactly zero in
any ordered semiring. The corpus is built so that the property under test is the
property the task requires.

`exec` is used rather than `subprocess` for speed; it is the same interpreter.
`execute_subprocess` exists so a test can prove the two agree rather than assume
it.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any

SEQ_LEN = 12

# token ids: a tiny explicit vocabulary, so nothing about tokenization is hidden
PAD, OP_ADD, OP_SUB, FLAG_0, FLAG_1 = 0, 1, 2, 3, 4
N_BASE = 5
N_LIT = 12                      # literal values 0..11 get ids N_BASE..N_BASE+11
VOCAB = N_BASE + N_LIT

TEMPLATE = (
    "acc = 0\n"
    "for _ in range({n}):\n"
    "    acc {op}= {c}\n"
    "if {flag}:\n"
    "    acc = -acc\n"
    "print(acc)\n"
)


def source(n: int, op: str, c: int, flag: int) -> str:
    return TEMPLATE.format(n=n, op=op, c=c, flag=bool(flag))


def execute(src: str) -> int:
    """Run the program and return what it printed. The interpreter is the label."""
    out: list[str] = []
    g: dict[str, Any] = {"print": lambda *a: out.append(" ".join(map(str, a)))}
    exec(compile(src, "<corpus>", "exec"), g)  # noqa: S102 - generated source, fixed template
    return int(out[-1])


def execute_subprocess(src: str) -> int:
    """Same program in an isolated CPython. Exists so a test can check that the
    `exec` shortcut did not change the oracle."""
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=30)
    return int(r.stdout.strip())


def tokenize(n: int, op: str, c: int, flag: int, filler: list[int]) -> list[int]:
    """Program -> token ids, padded to SEQ_LEN with distractor tokens.

    The causal tokens sit at fixed positions but are surrounded by filler drawn
    from the same vocabulary, so an arm cannot win by reading a position it was
    handed. Filler is literal-valued and therefore indistinguishable in type
    from the token that matters.
    """
    t = [N_BASE + n, OP_ADD if op == "+" else OP_SUB, N_BASE + c,
         FLAG_1 if flag else FLAG_0]
    return (t + filler)[:SEQ_LEN]


def build(n_train: int, n_test: int, seed: int = 0) -> dict:
    """Interventional pairs, split by COMPOSITION rather than by vocabulary.

    Train sees every (op, flag) cell except ('-', 1). Test is that cell alone.
    Literals 1..9 appear on BOTH sides, so every token id is trained and no arm
    is asked to read an embedding row it never updated.

    That distinction is not cosmetic. An earlier version of this file split by
    literal value -- train 1..3, test 7..9 -- and every arm landed near the
    predict-the-mean baseline (attention 0.954, appnp 0.779, signed 0.777)
    because the test tokens indexed embedding rows that training never touched.
    The experiment measured embedding coverage, not generalization.

    The held-out cell requires composing a subtraction with a negation: two sign
    flips, seen separately in training and never together. That composition is
    precisely what a non-negative influence Jacobian cannot express, so the
    held-out cell is the property under test rather than an arbitrary slice.
    """
    import random

    rng = random.Random(seed)
    HELD_OUT = ("-", 1)

    def one(cells: list[tuple[str, int]]) -> dict:
        n = rng.randint(1, 4)
        c = rng.randint(1, 9)
        op, flag = rng.choice(cells)
        y_base = execute(source(n, op, c, flag))

        # the intervention: edit exactly one token, then re-execute
        which = rng.choice(("c", "n"))
        n2, c2 = (n, rng.choice([x for x in range(1, 10) if x != c]))             if which == "c" else (rng.choice([x for x in (1, 2, 3, 4) if x != n]), c)
        src = source(n2, op, c2, flag)
        filler = [N_BASE + rng.randrange(N_LIT) for _ in range(SEQ_LEN)]
        return dict(src=src, y=execute(src), y_base=y_base,
                    tokens=tokenize(n2, op, c2, flag, filler),
                    n=n2, op=op, c=c2, flag=flag, edited=which)

    train_cells = [(o, f) for o in "+-" for f in (0, 1) if (o, f) != HELD_OUT]
    return dict(train=[one(train_cells) for _ in range(n_train)],
                test=[one([HELD_OUT]) for _ in range(n_test)],
                held_out=HELD_OUT,
                vocab=VOCAB)
