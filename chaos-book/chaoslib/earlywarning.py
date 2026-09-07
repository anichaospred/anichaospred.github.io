r"""Early-warning indicators for an approaching bifurcation, and their honest limits.

A system in a shallowing potential well relaxes more slowly, and a slower
relaxation shows up in a record as **more variance** and **more
autocorrelation**. That is critical slowing down, and it is the basis of every
early-warning indicator in the literature. This module provides the indicators,
the exact reference values they should take, and the two competing timing laws
that decide whether a warning can arrive at all.

Sign and units, once, because both are easy to get wrong:

* ``restoring_rate`` :math:`\lambda` is the eigenvalue of the linearised drift
  and is **negative** for a stable state. Its magnitude is a rate, per unit
  time.
* ``noise_std`` :math:`\sigma` is the diffusion coefficient in
  :math:`\mathrm{d}x = f(x)\,\mathrm{d}t + \sigma\,\mathrm{d}W`, in state units
  per **square root** of time -- the same convention as
  :func:`chaoslib.integrate.rk4_stochastic`.

The exact Ornstein-Uhlenbeck relations this module leans on, for
:math:`\mathrm{d}x = \lambda x\,\mathrm{d}t + \sigma\,\mathrm{d}W` with
:math:`\lambda < 0`:

.. math::
   \operatorname{var} x = \frac{\sigma^2}{2|\lambda|},
   \qquad
   \operatorname{corr}(x_t, x_{t+\Delta t}) = e^{\lambda \Delta t}.

Both diverge to their critical values as :math:`\lambda \to 0`, which is the
signal; both are also *linear* statements about a *nonlinear* escape problem,
which is why :func:`noise_advanced_fold` exists.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]

__all__ = [
    "detrend",
    "lag1_autocorrelation",
    "sliding_variance",
    "sliding_lag1_autocorrelation",
    "kendall_tau",
    "ar1_restoring_rate",
    "ou_stationary_std",
    "kramers_escape_time",
    "escape_times",
    "fold_delay",
    "noise_advanced_fold",
]

# First zero of the Airy function Ai, in modulus. Sets the deterministic delay
# past a fold that is swept at a finite rate; see fold_delay.
_AIRY_FIRST_ZERO = 2.338107410459767


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def detrend(series: Array, axis: int = 0) -> Array:
    """Remove a least-squares straight line along ``axis``.

    Indicators are computed on residuals, not on the raw record. A record that
    is drifting towards a new state has a *trend*, and the variance of a
    trending series is dominated by the trend rather than by the fluctuations
    the indicator is meant to measure -- which inflates the variance estimate
    exactly when the system is closest to tipping, i.e. produces the answer
    the analyst was hoping for by construction.
    """
    y = np.moveaxis(np.asarray(series, dtype=float), axis, 0)
    n = y.shape[0]
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    flat = y.reshape(n, -1)
    slope = (t[:, None] * (flat - flat.mean(axis=0))).sum(axis=0) / (t**2).sum()
    residual = flat - flat.mean(axis=0) - t[:, None] * slope
    return np.moveaxis(residual.reshape(y.shape), 0, axis)


def lag1_autocorrelation(series: Array, axis: int = 0) -> Array:
    r"""Lag-1 autocorrelation along ``axis``, after removing the mean.

    For a record sampled every :math:`\Delta t` from an OU process this
    estimates :math:`e^{\lambda \Delta t}`, so it depends on the **sampling
    interval as well as the dynamics**: the same system sampled twice as often
    reports a higher autocorrelation. Comparing an autocorrelation across
    records of different sampling is meaningless; convert to a rate with
    :func:`ar1_restoring_rate` first.
    """
    y = np.moveaxis(np.asarray(series, dtype=float), axis, 0)
    a = y - y.mean(axis=0)
    num = (a[:-1] * a[1:]).sum(axis=0)
    den = np.sqrt((a[:-1] ** 2).sum(axis=0) * (a[1:] ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _sliding(series: Array, width: int, step: int, reduce, detrend_first: bool):
    y = np.asarray(series, dtype=float)
    if width < 3:
        raise ValueError("window width must be at least 3 samples")
    if y.shape[0] < width:
        raise ValueError(
            f"record of {y.shape[0]} samples is shorter than the "
            f"{width}-sample window"
        )
    starts = np.arange(0, y.shape[0] - width + 1, max(int(step), 1))
    centres = starts + (width - 1) / 2.0
    out = []
    for s in starts:
        chunk = y[s: s + width]
        out.append(reduce(detrend(chunk) if detrend_first else chunk))
    return centres, np.asarray(out)


def sliding_variance(
    series: Array, width: int, step: int = 1, detrend_first: bool = True
) -> tuple[Array, Array]:
    """Variance in a sliding window of ``width`` samples.

    Returns ``(centres, values)`` with ``centres`` in **samples** from the
    start of the record, so multiply by the sampling interval for a time.
    Leading axes beyond the first are carried through, so an ensemble of
    records reduces in one call.
    """
    return _sliding(series, width, step, lambda c: c.var(axis=0), detrend_first)


def sliding_lag1_autocorrelation(
    series: Array, width: int, step: int = 1, detrend_first: bool = True
) -> tuple[Array, Array]:
    """Lag-1 autocorrelation in a sliding window; see :func:`sliding_variance`."""
    return _sliding(series, width, step, lag1_autocorrelation, detrend_first)


def kendall_tau(y: Array, x: Array | None = None) -> float:
    r"""Kendall's rank correlation :math:`\tau_b` between ``y`` and ``x``.

    With ``x`` omitted this is the trend statistic used throughout the
    early-warning literature: :math:`\tau` against the sample index, which is
    +1 for any monotonically rising indicator and -1 for any falling one,
    whatever its shape.

    Ties are handled as :math:`\tau_b` (the denominator is reduced by the tied
    pairs on each side), so it agrees with :func:`scipy.stats.kendalltau`,
    which the tests assert on random data.

    :math:`\tau` says nothing about *significance*. A rising indicator computed
    on overlapping windows has strongly autocorrelated samples, so the
    null distribution of :math:`\tau` is far wider than the independent-sample
    one, and a threshold must be calibrated against surrogate records from the
    null rather than taken from a table. Chapter 27 does that calibration and
    the false-alarm rates it reports are the reason.
    """
    a = np.asarray(y, dtype=float).ravel()
    b = np.arange(a.size, dtype=float) if x is None else np.asarray(x, float).ravel()
    if a.size != b.size:
        raise ValueError("y and x must have the same length")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2:
        return float("nan")
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(a.size, k=1)
    concordant = float((da[iu] * db[iu]).sum())
    n_a = float((da[iu] != 0).sum())
    n_b = float((db[iu] != 0).sum())
    if n_a == 0 or n_b == 0:
        return float("nan")
    return concordant / np.sqrt(n_a * n_b)


# --------------------------------------------------------------------------
# Exact reference values
# --------------------------------------------------------------------------
def ar1_restoring_rate(series: Array, dt: float, axis: int = 0) -> Array:
    r"""Restoring rate :math:`\lambda = \ln \alpha / \Delta t` from the lag-1
    autocorrelation :math:`\alpha`.

    Negative for a stable state. Returns ``nan`` where the estimated
    :math:`\alpha` is non-positive, which happens on short records of a rapidly
    mixing system and is not a physical rate.
    """
    alpha = np.asarray(lag1_autocorrelation(series, axis=axis), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(alpha > 0.0, np.log(alpha) / float(dt), np.nan)


def ou_stationary_std(noise_std: float, restoring_rate: float) -> float:
    r"""Exact stationary standard deviation :math:`\sigma/\sqrt{2|\lambda|}` of
    :math:`\mathrm{d}x = \lambda x\,\mathrm{d}t + \sigma\,\mathrm{d}W`.

    This is the value a variance-based indicator is estimating. It **diverges**
    as :math:`\lambda \to 0`, which is why variance is an indicator at all --
    and it is a linearisation, so it is only the answer while
    :math:`\sigma/\sqrt{2|\lambda|}` stays small compared with the distance to
    the saddle. Those two conditions fail together, at the distance
    :func:`noise_advanced_fold` computes.
    """
    lam = abs(float(restoring_rate))
    if lam == 0.0:
        return float("inf")
    return float(noise_std) / np.sqrt(2.0 * lam)


def kramers_escape_time(
    barrier: float,
    noise_std: float,
    curvature_well: float,
    curvature_saddle: float,
) -> float:
    r"""Kramers' mean first-passage time out of a potential well,

    .. math::
       \tau = \frac{2\pi}{\sqrt{V''(x_w)\,|V''(x_s)|}}
              \exp\!\left(\frac{2\Delta V}{\sigma^2}\right).

    ``curvature_well`` is :math:`V''(x_w) > 0` and ``curvature_saddle`` is
    :math:`|V''(x_s)|`, both positive.

    The exponential is asymptotically exact as :math:`2\Delta V/\sigma^2 \to
    \infty`; the prefactor is not, and at the moderate barriers a tipping
    problem actually involves it is optimistic. Measured against an ensemble at
    :math:`2\Delta V/\sigma^2` between 2 and 6 the formula overestimates
    :math:`\tau` by about 60 % (chapter 27, section 2), while the **slope** of
    :math:`\ln\tau` against :math:`1/\sigma^2` matches :math:`2\Delta V` to
    0.2 %. Use it for scaling, not for a date.
    """
    stiffness = float(curvature_well) * abs(float(curvature_saddle))
    if stiffness <= 0.0:
        return float("nan")
    return float(
        2.0 * np.pi / np.sqrt(stiffness)
        * np.exp(2.0 * float(barrier) / float(noise_std) ** 2)
    )


def escape_times(
    series: Array, threshold: float, times: Array, above: bool = True
) -> Array:
    """First time each record crosses ``threshold``; ``nan`` where it never does.

    ``series`` is ``(time, member)`` (or just ``(time,)``) and ``times`` is the
    matching time axis.

    **The ``nan`` is the point.** Averaging only over the members that did
    escape is a censored estimator and it biases the mean escape time *low*,
    because the slow escapes are exactly the ones missing. In chapter 27 a run
    in which 84 % of members escaped biased the fitted Kramers slope by 12 %,
    while dropping that one case brought the fit to 0.2 % of the exact value.
    Check ``np.isnan(...).any()`` before taking a mean.
    """
    y = np.asarray(series, dtype=float)
    t = np.asarray(times, dtype=float)
    if y.shape[0] != t.size:
        raise ValueError(
            f"series has {y.shape[0]} time samples but times has {t.size}"
        )
    crossed = y > float(threshold) if above else y < float(threshold)
    ever = crossed.any(axis=0)
    first = t[crossed.argmax(axis=0)]
    return np.where(ever, first, np.nan)


# --------------------------------------------------------------------------
# The two timing laws
# --------------------------------------------------------------------------
def fold_delay(rate: float) -> float:
    r"""How far **past** a fold a noiseless system is carried before it leaves,
    for a parameter swept at ``rate``.

    .. math:: \mu_{\text{tip}} - \mu_c \simeq |a_1|\,3^{-1/6}\,\gamma^{2/3},

    where :math:`|a_1| = 2.3381\ldots` is the modulus of the first zero of the
    Airy function. It comes from the fold normal form
    :math:`\dot u = \sqrt3 u^2 + \gamma\tau`, which linearises under
    :math:`u = -w'/w` to Airy's equation, so the escape is the first zero of
    :math:`w`. The constant is :math:`1.9469\ldots`; a swept run measures
    1.9239 at :math:`\gamma = 2.5\times10^{-4}` and approaches it from below as
    the rate falls (chapter 27, section 4).

    This delay and the noise advance of :func:`noise_advanced_fold` have
    **opposite signs**, so a measured tipping point is a competition and
    neither law can be checked without controlling the other.
    """
    return float(_AIRY_FIRST_ZERO * 3.0 ** (-1.0 / 6.0) * float(rate) ** (2.0 / 3.0))


def noise_advanced_fold(
    noise_std: float, rate: float, quantile: float = 0.5
) -> float:
    r"""How far **before** a fold noise carries a system out of the well.

    Returns :math:`d^* = \mu_c - \mu_{\text{tip}}`, the distance to the fold at
    which the cumulative escape probability reaches ``quantile``.

    Integrating Kramers' rate along the ramp with the near-fold forms
    :math:`\Delta V = \tfrac43 3^{-1/4} d^{3/2}` and
    :math:`V''(x_w) = |V''(x_s)| = 2\cdot3^{1/4}\sqrt d` gives a closed form,
    because the substitution :math:`w = d^{3/2}` makes the integral
    elementary:

    .. math::
       d^* = \left[\frac{\sigma^2}{\tfrac83 3^{-1/4}}
             \ln \frac{2\cdot3^{1/4}\,\sigma^2}
             {3\pi\,\tfrac83 3^{-1/4}\,\gamma\,\ln\frac{1}{1-q}}\right]^{2/3}.

    **The logarithm is the whole point.** Dropping it leaves
    :math:`d^* \propto \sigma^{4/3}`, which is the scaling usually quoted; with
    it, a power law fitted over :math:`\sigma \in [0.04, 0.19]` reads
    :math:`\sigma^{1.69}`. The full expression predicts the measured median
    tipping tilt to **3 %** across that range at
    :math:`\gamma = 2.5\times10^{-5}`, where the bare :math:`\sigma^{4/3}` form
    is wrong by a factor rising from 1.8 to 3.2 (chapter 27, section 4).

    Returns ``nan`` where the logarithm is non-positive -- a ramp so fast, or
    noise so weak, that escape before the fold is not expected at all, and the
    deterministic :func:`fold_delay` governs instead.
    """
    sigma2 = float(noise_std) ** 2
    gamma = float(rate)
    q = float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    a = (8.0 / 3.0) * 3.0 ** -0.25            # 2 dV / sigma^2 = a d^(3/2) / sigma^2
    argument = (
        2.0 * 3.0**0.25 * sigma2
        / (3.0 * np.pi * a * gamma * np.log(1.0 / (1.0 - q)))
    )
    if argument <= 1.0:
        return float("nan")
    return float((sigma2 / a * np.log(argument)) ** (2.0 / 3.0))
