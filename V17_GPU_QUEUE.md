# CEQ v17-K — GPU WORK REGISTER

Author's instruction: **do not spend hours of local GPU time; anything that
needs a GPU is marked for Kaggle unless it cannot legally run there.**

The one thing that cannot move is fixed by the round's own KILL clause —
*"any deciding number taken from Kaggle ⇒ struck (Kaggle reproduces)"* — and by
RULING 4, which refuses tolerance widening. A deciding cell run on Kaggle is
not a cheaper deciding cell; it is a struck one.

| item | needs GPU | runs where | why |
|---|---|---|---|
| K-CERT device certificate | yes | **local (in flight)** + **Kaggle at L2** | the certificate is per-device by definition; both boxes get one |
| R1 training noise floor — two identical-seed chunks, `\|Δ\|` final loss | yes | **KAGGLE** | it bounds *training* claims, and training runs on Kaggle. Measuring it locally would bound the wrong device. Not a deciding number |
| Q1 reproduction | yes | **KAGGLE** | a reproduction by construction; `\|Δ\|` against the local cell is the whole point |
| ~~Q2 reproduction~~ | — | **DROPPED** | RULING 8. `n=16,384` does not fit the certified 4060 (10.578 GiB vs 7.996), so its deciding cell could only exist on Kaggle, where the KILL clause strikes it |
| Q3 training segments | yes | **KAGGLE** | the round exists to run these there |
| Q4 capability table at trained scale | yes | **KAGGLE** | downstream of Q3's checkpoint |
| **R4 re-take: R1′ and every deciding cell on the certified 4060** | yes | **LOCAL — cannot move** | KILL clause. A deciding number from Kaggle is struck, and RULING 4 refuses widening the tolerance instead. **MEASURED: 2.65 GPU-min** for 24 cells, one invocation, 0.92 GiB peak — the `~2 GPU-h` estimate was 45x over-booked (it priced a 9600-step ladder; the journal holds 150-step cells). Running |
| gradient-finiteness / determinism spot checks | briefly | local, seconds | smallest binding shape only |

## STANDING RULE FOR EVERY NODE

If a task needs the GPU for more than a few minutes and its output is **not** a
deciding number, it is Kaggle's. Write the harness, test it on CPU at a tiny
shape, and hand it to the notebook. Do not burn the local card on work the
Kaggle session is going to redo anyway.
