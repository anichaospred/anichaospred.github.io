r"""Deterministic forecast verification, and the awkward question underneath it.

Every score in this book so far has compared a forecast against a *truth*. No
forecast centre has ever had one. What they have is observations, which are the
truth plus an error -- and often the very observations that were assimilated to
produce the analysis the forecast started from.

This module holds the deterministic scores (anomaly correlation, the mean-square
error and its decomposition, skill horizons) together with the corrections that
relate a score computed against observations to the score you actually wanted.
The probabilistic scores live in :mod:`chaoslib.ensemble`.

Two identities are worth knowing before reading anything below.

* An unbiased forecast whose variance matches the truth's has
  :math:`\mathrm{MSE} = 2\sigma^2(1-r)`, so it beats a climatological forecast
  exactly when :math:`r > 1/2`. **That is where the conventional anomaly-
  correlation threshold comes from**, and it is a statement about undamped
  forecasts rather than a law of nature.
* Verifying against observations with independent error of variance
  :math:`\sigma_o^2` inflates the mean-square error by exactly
  :math:`\sigma_o^2` and attenuates the anomaly correlation by exactly
  :math:`[1+\sigma_o^2/\sigma_t^2]^{-1/2}`. Both are correctable **if** the
  errors really are independent, which for assimilated observations they are not.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]

__all__ = [
    "anomaly_correlation",
    "mse_decomposition",
    "mse_skill_score",
    "skill_horizon",
    "acc_threshold_for_climatological_skill",
    "optimal_damping",
    "correct_mse_for_observation_error",
    "correct_acc_for_observation_error",
]


def _anomalies(field: Array, climatology: Array | None) -> Array:
    field = np.asarray(field, dtype=float)
    if climatology is None:
        return field - field.mean(axis=0, keepdims=True)
    return field - np.asarray(climatology, dtype=float)


def anomaly_correlation(
    forecast: Array, truth: Array, climatology: Array | None = None
) -> float:
    r"""Anomaly correlation between forecast and truth, about a climatology.

    .. math::
        \mathrm{ACC} = \frac{\sum (f-c)(t-c)}
                            {\sqrt{\sum (f-c)^2 \sum (t-c)^2}}

    The operational skill metric, and the one the familiar "useful to about a
    week" figures are quoted in. It measures whether the forecast got the
    *pattern* of departures from climatology right, and is deliberately blind to
    a uniform amplitude error -- scaling every forecast anomaly by a constant
    leaves it unchanged, which is exactly why it must be read alongside a
    mean-square error rather than instead of one.

    ``forecast`` and ``truth`` have matching shapes; ``climatology`` broadcasts
    against them, and ``None`` means "subtract the sample mean over the first
    axis", which is the usual convention when a long climatology is unavailable.
    """
    f = _anomalies(forecast, climatology).ravel()
    t = _anomalies(truth, climatology).ravel()
    denominator = np.sqrt(float(f @ f) * float(t @ t))
    if denominator == 0.0:
        return float("nan")
    return float(f @ t / denominator)


def mse_decomposition(
    forecast: Array, truth: Array
) -> tuple[float, float, float]:
    r"""Split the mean-square error into bias, amplitude and phase parts.

    .. math::
        \mathrm{MSE} = \underbrace{(\bar f - \bar t)^2}_{\text{bias}}
          + \underbrace{(\sigma_f - \sigma_t)^2}_{\text{amplitude}}
          + \underbrace{2\sigma_f\sigma_t(1-r)}_{\text{phase}}

    an identity, not an approximation, with :math:`r` the correlation and the
    standard deviations taken about the respective means. Returns
    ``(bias, amplitude, phase)``, which sum to the MSE to machine precision.

    The split is worth making because the three parts are fixed by different
    things. **Bias** is a model or observing-system problem and is removable by
    subtraction. **Amplitude** is a calibration problem -- a forecast whose
    anomalies are systematically too large loses to one that hedges, and
    :func:`optimal_damping` says by how much. **Phase** is the part that
    represents actual predictive information going away, and it is the only one
    of the three that chaos forces on you.
    """
    f = np.asarray(forecast, dtype=float).ravel()
    t = np.asarray(truth, dtype=float).ravel()
    if f.shape != t.shape:
        raise ValueError("forecast and truth must have the same shape")
    bias = float(f.mean() - t.mean())
    sigma_f = float(f.std())
    sigma_t = float(t.std())
    if sigma_f == 0.0 or sigma_t == 0.0:
        correlation = 0.0
    else:
        centred_f, centred_t = f - f.mean(), t - t.mean()
        correlation = float(
            (centred_f @ centred_t) / (f.size * sigma_f * sigma_t)
        )
    return (
        bias**2,
        (sigma_f - sigma_t) ** 2,
        2.0 * sigma_f * sigma_t * (1.0 - correlation),
    )


def mse_skill_score(
    forecast: Array, truth: Array, reference: Array | None = None
) -> float:
    r"""Skill against a reference forecast: :math:`1 - \mathrm{MSE}_f/\mathrm{MSE}_r`.

    ``reference=None`` uses the climatological forecast -- the mean of ``truth``
    -- which is the usual baseline and the one the anomaly-correlation threshold
    is calibrated against. Positive means better than the reference, zero means
    no better, and negative means a user would have done better ignoring the
    forecast entirely.
    """
    f = np.asarray(forecast, dtype=float).ravel()
    t = np.asarray(truth, dtype=float).ravel()
    reference_values = (
        np.full_like(t, t.mean())
        if reference is None
        else np.asarray(reference, dtype=float).ravel()
    )
    mse_forecast = float(np.mean((f - t) ** 2))
    mse_reference = float(np.mean((reference_values - t) ** 2))
    if mse_reference == 0.0:
        return float("nan")
    return 1.0 - mse_forecast / mse_reference


def skill_horizon(
    times: Array, scores: Array, threshold: float, decreasing: bool = True
) -> float:
    r"""Where a skill curve first crosses ``threshold``, by linear interpolation.

    ``decreasing=True`` (the default) suits scores that fall as the forecast
    ages -- anomaly correlation, skill score. ``False`` suits scores that rise,
    such as RMSE or CRPS.

    Returns ``nan`` if the curve never crosses, and **the first** crossing if it
    crosses several times. Interpolating rather than reporting the last grid
    point that still qualified matters more than it sounds: with skill sampled
    every six hours, rounding down to the grid quantises every horizon in the
    book to the sampling interval, and differences between scores smaller than
    that vanish entirely.
    """
    times = np.asarray(times, dtype=float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if times.shape != scores.shape:
        raise ValueError("times and scores must have the same shape")
    crossed = scores < threshold if decreasing else scores > threshold
    if not np.any(crossed):
        return float("nan")
    index = int(np.argmax(crossed))
    if index == 0:
        return float(times[0])
    before, after = scores[index - 1], scores[index]
    if after == before:
        return float(times[index])
    fraction = (threshold - before) / (after - before)
    return float(times[index - 1] + fraction * (times[index] - times[index - 1]))


def acc_threshold_for_climatological_skill(damped: bool = False) -> float:
    r"""The anomaly correlation at which a forecast stops beating climatology.

    For an **undamped** forecast -- unbiased, with anomaly variance matching the
    truth's -- :math:`\mathrm{MSE} = 2\sigma^2(1-r)` against the climatological
    :math:`\sigma^2`, so the forecast wins exactly while :math:`r > 1/2`.
    The conventional operational threshold of 0.6 is this number plus a margin.

    For an **optimally damped** forecast, :func:`optimal_damping` shows
    :math:`\mathrm{MSE} = \sigma^2(1-r^2)`, which beats climatology for *any*
    non-zero correlation. So the threshold is 0, and the familiar 0.6 is a
    statement about a particular post-processing choice rather than about
    predictability.

    Returned as a function rather than a constant because the number depends on
    that choice, and writing 0.5 as a literal somewhere hides the dependence.
    """
    return 0.0 if damped else 0.5


def optimal_damping(forecast: Array, truth: Array) -> tuple[float, float]:
    r"""The variance-minimising rescaling of a forecast's anomalies.

    Regressing truth on forecast gives the multiplier
    :math:`a = r\,\sigma_t/\sigma_f` and the resulting
    :math:`\mathrm{MSE} = \sigma_t^2(1-r^2)`. Returns ``(a, mse_ratio)`` where
    the ratio is against the climatological MSE, so it equals :math:`1-r^2`.

    Damping is not a trick for flattering a score. A least-squares-optimal point
    forecast of a partly unpredictable quantity *should* be closer to
    climatology than the truth is: pushing full-amplitude anomalies is
    over-confidence in exactly the sense chapter 17 measures for ensembles. What
    is lost is the realism of the field -- a damped forecast has too little
    variance and too few extremes -- which is why operational centres issue
    undamped deterministic forecasts and take the score penalty knowingly.
    """
    f = np.asarray(forecast, dtype=float).ravel()
    t = np.asarray(truth, dtype=float).ravel()
    centred_f = f - f.mean()
    centred_t = t - t.mean()
    variance = float(centred_f @ centred_f)
    if variance == 0.0:
        return 0.0, 1.0
    multiplier = float(centred_f @ centred_t) / variance
    sigma_f = float(centred_f.std())
    sigma_t = float(centred_t.std())
    correlation = (
        0.0 if sigma_f == 0.0 or sigma_t == 0.0
        else float((centred_f @ centred_t) / (f.size * sigma_f * sigma_t))
    )
    return multiplier, 1.0 - correlation**2


def correct_mse_for_observation_error(
    mse_against_observations: float, observation_variance: float
) -> float:
    r"""Remove the observation error's contribution to a verification score.

    With :math:`y = t + \varepsilon` and :math:`\varepsilon` **independent** of
    the forecast error,

    .. math::
        \mathbb{E}\,(f-y)^2 = \mathbb{E}\,(f-t)^2 + \sigma_o^2 ,

    exactly -- the cross term vanishes. So the correction is a subtraction, and
    the whole difficulty lies in that word *independent*.

    It fails, badly and in the flattering direction, when the observations being
    verified against were **assimilated** into the analysis the forecast started
    from. The analysis was pulled towards those observations, so the forecast
    error and the observation error share a component, the cross term is
    negative, and the score is optimistic rather than pessimistic. Chapter 22
    measures the size of that. The operational remedy is to verify against
    observations withheld from the assimilation, which costs exactly the
    observations you would most like to have used.

    A negative result is returned as-is rather than clipped to zero: it means
    the assumed :math:`\sigma_o^2` is too large or the independence assumption
    has failed, and silently clipping would hide both.
    """
    return float(mse_against_observations) - float(observation_variance)


def correct_acc_for_observation_error(
    acc_against_observations: float,
    observation_variance: float,
    truth_variance: float,
) -> float:
    r"""Undo the attenuation of an anomaly correlation by observation noise.

    Independent observation error leaves the covariance alone but inflates the
    verifying field's variance, so

    .. math::
        \mathrm{ACC}_{\text{obs}} = \mathrm{ACC}_{\text{true}}
          \left(1 + \frac{\sigma_o^2}{\sigma_t^2}\right)^{-1/2} ,

    the classical attenuation-by-measurement-error result. This multiplies by
    the inverse factor.

    Note which way it goes: noisy verification makes a forecast look **worse**
    in anomaly correlation, and makes it look worse in mean-square error too --
    but the *skill score* against a climatology verified the same way is nearly
    unaffected, because the reference is penalised as well. Three scores, three
    different sensitivities to the same noise, which is the practical reason to
    quote more than one.

    ``truth_variance`` is the variance of the *true* field. Using the observed
    field's variance instead over-corrects, since that already includes
    :math:`\sigma_o^2`.
    """
    ratio = float(observation_variance) / float(truth_variance)
    return float(acc_against_observations) * np.sqrt(1.0 + ratio)
