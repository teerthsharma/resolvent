"""ceq.certs -- certificates. Instruments that refuse rather than guess.

A certificate here is a reading with a stated rejection region: an input it
cannot measure makes it raise `CertificateRefused`, never return a plausible
number (MISTAKES.md V-16). Nothing in this package trains, fits, or selects a
threshold from the data it judges.
"""
__all__ = ["topological"]
