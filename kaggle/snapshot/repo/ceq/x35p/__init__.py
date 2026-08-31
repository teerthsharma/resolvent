"""X35' — wave-theoretic hidden-cause inference (CEQ v15.2).

Instruments, per `CEQ_V15_2_DELTA.md`:

  source.py  (a) exact source solve  h = (I - A) r
             (b) time-reversal localization  W^T r, Wiener-deconvolved
  kk.py      (c) Kramers-Kronig causality residual on a learned kernel
  crb.py     (d) localization floor, printed beside every onset CI

Created by the coordinator to unblock two nodes writing into this package
concurrently. Each module owns its own file; append your export below rather
than rewriting this one.
"""
