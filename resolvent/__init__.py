"""rjepa: candidate-set resolvents, a D-JEPA-spec relational operator and two exact-truth ranking beds."""
from .djepa import RelationalOperator, dj_loss, rank01
from .heads import Head
from .resolvent import (build_A, linear_equivariant_resolvent, neumann_resolvent, resolvent_apply,
                        set_resolvent, stochastic_resolvent)

__version__ = "0.1.0"
__all__ = ["Head", "RelationalOperator", "build_A", "dj_loss", "linear_equivariant_resolvent",
           "neumann_resolvent", "rank01", "resolvent_apply", "set_resolvent", "stochastic_resolvent"]
