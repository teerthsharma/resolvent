"""X35-prime (c) -- the Kramers-Kronig causality residual, REBUILT.

CEQ_V15_2_DELTA.md struck the author's own KK probe: it read `0.000` on a
planted anticipating kernel, and a causality test that does not fire on an
acausal kernel is measuring nothing (MISTAKES.md V-16). This module is the
rebuild, and it ships only because its planted-anticipating must-fire fires --
`tests/x35p/test_kk_crb.py`.

THE RELATION. A real sequence h is causal iff its odd part is `sgn(k)` times
its even part. The DFT of the even part is `Re H` and the DFT of the odd part
is `i Im H`, so for a causal h the imaginary part of the spectrum is a
function of the real part alone -- the discrete Kramers-Kronig relation. The
residual is what is left when you throw `Im H` away, rebuild it from `Re H`
under the causal hypothesis, and compare:

    residual = || Im H - Im H_hat ||_2 / sqrt(M),   H_hat = F[ P_causal F^-1 Re H ]

`P_causal` is the standard fold: keep index 0 and Nyquist, double the first
half, zero the second. This is the whole probe. It consumes `H(omega)` and
never looks at the taps, which is the point -- it applies to a learned kernel
observed only through its frequency response.

WHAT THE NUMBER IS, EXACTLY. `H - H_hat` is the DFT of the odd extension of
h's anticausal taps, so by Parseval the residual has a closed form:

    residual = sqrt(2) * || anticausal taps ||_2

computed in the time domain, and `kk_residual` returns it as `closed_form`
beside the frequency-domain reading. The two are computed by different routes
and their agreement is a check, not an identity (MISTAKES.md V-3). It also
means the probe is CALIBRATED rather than thresholded: a kernel with an
anticipating tap of amplitude `a` reads `a*sqrt(2)`, so the reading is graded
in the violation and has no tuning knob at all.

THE GUARD BAND IS NOT OPTIONAL. Causality is a property of a sequence RELATIVE
TO ITS TIME ORIGIN, and the DFT is a transform on a circle where there is no
origin unless you put one there. `kk_residual` therefore takes an explicit
`lags` axis and REFUSES a grid too small to separate a negative lag from a
long positive one. Both refusals are load-bearing; see the DIAGNOSIS section
below for the four naive implementations they exist to prevent, three of which
are measured to read 0.000 on a kernel that anticipates by five samples.

float64 throughout; `np.fft` on real input returns complex128.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "kk_residual", "kk_residual_from_spectrum", "kernel_from_matrix",
    "broken_bare_array", "broken_magnitude", "broken_no_padding",
    "broken_bin_normalized",
]


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------


def _causal_project(x: np.ndarray) -> np.ndarray:
    """Fold a circularly-even sequence onto the causal half: `x[0]` and the
    Nyquist bin keep their weight, the first half doubles, the second half
    goes to zero. `F[_causal_project(F^-1 Re H)]` is the unique causal
    spectrum with the given real part."""
    m = x.size
    half = m // 2
    y = np.zeros_like(x)
    y[0] = x[0]
    y[1:half] = 2.0 * x[1:half]
    y[half] = x[half]
    return y


def _grid(lags: np.ndarray, n_freq) -> int:
    lo, hi = int(lags.min()), int(lags.max())
    if n_freq is None:
        span = max(hi + 1, -lo, 1)
        n_freq = 1 << int(np.ceil(np.log2(4 * span)))
    n_freq = int(n_freq)
    if n_freq % 2:
        raise ValueError(f"n_freq must be even, got {n_freq}")
    half = n_freq // 2
    if hi >= half or -lo >= half:
        raise ValueError(
            f"grid too small: lags span [{lo}, {hi}] on {n_freq} bins, which "
            f"leaves no guard band. A negative lag would alias into the causal "
            f"half of the circle and read as a long positive delay; use "
            f"n_freq > {2 * max(hi + 1, -lo)}."
        )
    return n_freq


def kk_residual_from_spectrum(spectrum: np.ndarray) -> dict:
    """The probe, on `H(omega)` alone. `spectrum` is the DFT of the kernel on
    a grid whose bin 0 is lag 0 (see `kk_residual` for the guard band that
    makes that meaningful).

    `residual` is the RMS-over-frequency absolute residual, in the kernel's
    own units. `relative` divides by the RMS of `Im H`; it is reported beside
    the absolute and never instead of it, because a normalizer that grows with
    the numerator can hide the very violation being measured.
    """
    H = np.asarray(spectrum, dtype=np.complex128)
    m = H.size
    even = np.fft.ifft(H.real).real
    H_hat = np.fft.fft(_causal_project(even))
    d = H.imag - H_hat.imag
    res = float(np.linalg.norm(d) / np.sqrt(m))
    im_rms = float(np.linalg.norm(H.imag) / np.sqrt(m))
    return dict(residual=res,
                relative=float(res / im_rms) if im_rms > 0.0 else float("nan"),
                im_rms=im_rms, n_freq=int(m), causal_spectrum=H_hat)


def kk_residual(lags, taps, *, n_freq=None) -> dict:
    """The probe on an explicitly-lagged kernel. `lags[k]` is the lag of
    `taps[k]`; negative entries are anticipating.

    The lag axis is REQUIRED and there is no default. A bare tap array with an
    implied origin at index 0 is causal by construction whatever it contains,
    which is `broken_bare_array` below and is the reading the struck probe
    produced.
    """
    lags = np.asarray(lags, dtype=np.int64)
    taps = np.asarray(taps, dtype=np.float64)
    if lags.shape != taps.shape:
        raise ValueError(f"lags {lags.shape} and taps {taps.shape} disagree")
    m = _grid(lags, n_freq)
    h = np.zeros(m, dtype=np.float64)
    np.add.at(h, lags % m, taps)
    out = kk_residual_from_spectrum(np.fft.fft(h))
    anti = float(np.linalg.norm(taps[lags < 0]))
    out.update(spectrum=np.fft.fft(h), kernel=h,
               anticausal_norm=anti, closed_form=float(np.sqrt(2.0) * anti))
    return out


def kernel_from_matrix(K: np.ndarray, row: int) -> tuple[np.ndarray, np.ndarray]:
    """`(lags, taps)` for one row of a kernel matrix, with `lag = i - j`.
    Lets the probe run on any `K` this repo builds -- `bed_k.kernel_matrix`,
    or an attention operator read off as a matrix -- without the caller
    inventing a lag convention."""
    K = np.asarray(K, dtype=np.float64)
    j = np.arange(K.shape[1], dtype=np.int64)
    return row - j, K[row].copy()


# --------------------------------------------------------------------------
# DIAGNOSIS -- four naive implementations, kept so the failure is measured
# rather than asserted. NOTHING SHIPPING MAY CALL THESE. They exist to answer
# "why did the original read 0.000", and each one's reading is filed in
# V15_X35P_KK_CRB.md.
# --------------------------------------------------------------------------


def broken_bare_array(taps, *, n_freq=None) -> dict:
    """CANDIDATE 1 -- the origin-forgetting bug. The kernel is handed over as
    a bare array and index 0 is taken to be lag 0. Every finitely-supported
    sequence is causal once you forget where its time origin is, because a
    shift of the origin is a linear phase and linear phase preserves
    causality. Reads 0 on every input, causal or not."""
    taps = np.asarray(taps, dtype=np.float64)
    return kk_residual(np.arange(taps.size), taps, n_freq=n_freq)


def broken_magnitude(lags, taps, *, n_freq=None) -> dict:
    """CANDIDATE 2 -- the residual taken on `|H|` instead of on `Re/Im`. The
    KK partner of `|H|` is the PHASE, and reconstructing it gives the
    minimum-phase spectrum `exp(A)` with `Re A = log|H|`, whose magnitude is
    the input magnitude by construction. Comparing magnitudes therefore has no
    rejection region at all (MISTAKES.md V-10), and the phase -- which is
    where causality lives -- is never looked at."""
    lags = np.asarray(lags, dtype=np.int64)
    taps = np.asarray(taps, dtype=np.float64)
    m = _grid(lags, n_freq)
    h = np.zeros(m)
    np.add.at(h, lags % m, taps)
    mag = np.abs(np.fft.fft(h))
    log_mag = np.log(np.maximum(mag, 1e-300))
    A = np.fft.fft(_causal_project(np.fft.ifft(log_mag).real))
    res = float(np.linalg.norm(np.abs(np.exp(A)) - mag) / np.sqrt(m))
    return dict(residual=res, n_freq=int(m))


def broken_no_padding(lags, taps) -> dict:
    """CANDIDATE 3 -- FFT length set to the kernel's own support with no zero
    padding, and no guard-band check. A negative lag then wraps into the
    causal half of the circle and is indistinguishable from a long positive
    delay. Reads 0 exactly when every anticausal tap lands below the Nyquist
    index, which is what happens whenever the anticipation is comparable to
    the causal extent."""
    lags = np.asarray(lags, dtype=np.int64)
    taps = np.asarray(taps, dtype=np.float64)
    m = int(lags.max() - lags.min() + 1)
    m += m % 2
    h = np.zeros(m)
    np.add.at(h, lags % m, taps)
    out = kk_residual_from_spectrum(np.fft.fft(h))
    out["wrapped_index"] = (lags % m).tolist()
    return out


def broken_bin_normalized(lags, taps, *, n_freq=None) -> dict:
    """CANDIDATE 4 -- correct numerator, divided by the bin count `M` instead
    of `sqrt(M)`. The reading then shrinks as `1/sqrt(M)` purely because the
    spectrum was sampled more finely. This one is ELIMINATED as the mechanism
    below: it shrinks, but it does not reach 0.000 at any grid a person would
    run."""
    out = kk_residual(lags, taps, n_freq=n_freq)
    m = out["n_freq"]
    out["residual"] = out["residual"] * np.sqrt(m) / m
    return out
