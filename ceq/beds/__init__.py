"""ceq.beds -- corpora whose label depends on more than the previous state.

Drop-in sibling of ceq/corpus.py (BED-M, the chain corpus). BED-M's label at
position i is a function of position i-1 and a running scan state; every
corpus in the campaign before this package is built that way. BED-K
(bed_k.py) is the first whose label contains a DELAYED CAUSE.
"""
__all__ = ["bed_k"]
