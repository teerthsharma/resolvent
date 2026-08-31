/-
  CEQ — Consequence-Equilibrium Attention, verified core
  ------------------------------------------------------

  Three load-bearing facts for the architecture in THEORY.md, each stated at the
  strength it can actually be proved at.

  | module        | fact                                                        | why it is load-bearing |
  |---------------|-------------------------------------------------------------|------------------------|
  | `Contraction` | row-stochastic `P` ⇒ `γP` is a sup-norm contraction of modulus `γ`; and a witness showing `σ_max` may exceed 1 | well-posedness of the equilibrium, and a correction to the metric `sigmoid` §5 reports |
  | `Occupancy`   | `(I − A)·∑_{k<N} A^k = I − A^N`, exact inverse when `A^N = 0` | the resolvent-equals-occupancy reading, finite and honest |
  | `OrbitBound`  | injective ground relation ⇒ `err ≥ n − m`                     | the no-ground-truth training objective |
  | `OracleSeparation` | non-negative, symmetric-support, one positive entry ⇒ NOT nilpotent, so `oracle ≠ resolvent` | the round-8 ladder is only creditable if its oracle is not the arm's own forward |

  No `sorry` anywhere. Where the full statement was out of reach it was weakened to a
  provable corollary and the weakening is documented in the module header, following
  the convention already used in `Epsilon-Hollow`'s `AetherVerified`.
-/

import CEQ.Contraction
import CEQ.Occupancy
import CEQ.OrbitBound
import CEQ.Nilpotent
import CEQ.Refcount
import CEQ.OracleSeparation

import CEQ.V15
