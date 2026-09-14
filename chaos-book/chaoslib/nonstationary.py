r"""Trends in predictability, and how to attribute and detect them.

Forecast skill has improved for fifty years. Two entirely different things could
have caused that -- the forecasting system got better, or the atmosphere became
easier to forecast -- and a skill record on its own cannot separate them. This
module holds the four tools chapter 28 needs to try.

**The conceptual difficulty comes first.** A Lyapunov exponent is a limit as
:math:`T\to\infty`. It is a property of an *attractor*, and a system whose
forcing is changing does not have one. So "the predictability of the 1980s"
is not a well-posed quantity in the sense chapter 7 defines, and every number
in this module is a **finite-sample estimate of a finite-time quantity**. That
is not a technicality to be apologised for: it is why
:func:`trend_detection_power` exists and why its answers are large.

Four tools.

1. :func:`horizon_law_decomposition` -- splits a change in the forecast horizon
   into the parts due to instability, to the amplitude of what is being
   forecast, and to the accuracy of the initial state. Exact, from the horizon
   law of :func:`chaoslib.errorgrowth.horizon_law`.
2. :func:`factorial_attribution` -- the 2x2 design an operational centre
   actually runs: a frozen forecasting system re-run on a later atmosphere.
   Returns the two single-factor effects **and the interaction**, which is not a
   correction term here but a result: the return on a better observing system
   depends on how unstable the atmosphere is.
3. :func:`shift_share` -- was the mean instability change because states of a
   given kind became more unstable, or because the system now visits unstable
   states more often? The decomposition is exact and symmetric in the two
   epochs.
4. :func:`linear_trend`, :func:`trend_standard_error`,
   :func:`trend_detection_power`, :func:`minimum_record_length` -- how long a
   record has to be before a trend of a given size is detectable at all.

Sign convention, once: every "horizon" here is a **time**, so a *positive*
trend means forecasts got longer, i.e. predictability improved. A rising
:math:`\lambda_1` therefore appears as a *negative* horizon trend.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.floating]

__all__ = [
    "linear_trend",
    "trend_standard_error",
    "trend_detection_power",
    "minimum_record_length",
    "horizon_law_decomposition",
    "factorial_attribution",
    "shift_share",
    "epoch_means",
    "breakeven_accuracy",
    "effective_sample_size",
]


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------
def horizon_law_decomposition(
    saturation: tuple[float, float],
    delta0: tuple[float, float],
    rate: tuple[float, float],
    fraction: float = 0.5,
) -> dict[str, float]:
    r"""Split a change in the forecast horizon into its three causes.

    The horizon law (:func:`chaoslib.errorgrowth.horizon_law`) is

    .. math::
        T = \frac{1}{\lambda}\ln\frac{f\,\delta_\infty}{\delta_0}
          = \frac{L}{\lambda},
        \qquad L = \ln f + \ln\delta_\infty - \ln\delta_0 ,

    so a horizon can change for three reasons that have nothing to do with each
    other: the flow became more unstable (:math:`\lambda`), the *thing being
    forecast* grew in amplitude (:math:`\delta_\infty`), or the initial state
    became better known (:math:`\delta_0`).

    Each argument is a ``(before, after)`` pair. The split uses the exact
    symmetric identity

    .. math::
        \frac{L_1}{\lambda_1} - \frac{L_0}{\lambda_0}
        = \frac{L_1 + L_0}{2}\Bigl(\frac1{\lambda_1}-\frac1{\lambda_0}\Bigr)
        + (L_1 - L_0)\,\frac{1}{2}
          \Bigl(\frac1{\lambda_1}+\frac1{\lambda_0}\Bigr),

    which holds term by term for any values -- there is no linearisation and no
    residual -- and :math:`L_1 - L_0` then separates *additively* into the
    amplitude and accuracy pieces because :math:`L` is a sum of logarithms.
    The result is order-independent, unlike the more common "hold one fixed and
    vary the other" attribution, which gives two different answers depending on
    which you hold.

    Returns ``total``, ``instability``, ``amplitude``, ``accuracy``, and
    ``horizon_before`` / ``horizon_after``. The three components sum to
    ``total`` to machine precision; :func:`shift_share` and this function are
    both tested on that identity rather than on a reference value.

    A trap worth naming: ``amplitude`` is **positive** when the climate's
    variability grows, because a fixed absolute analysis error is a smaller
    *relative* error against a larger signal. A warming climate can hand a
    forecasting system a free improvement that has nothing to do with the
    forecasting system and nothing to do with the flow becoming more
    predictable.
    """
    sat0, sat1 = (float(v) for v in saturation)
    d00, d01 = (float(v) for v in delta0)
    lam0, lam1 = (float(v) for v in rate)
    if min(sat0, sat1, d00, d01) <= 0.0 or min(lam0, lam1) <= 0.0:
        raise ValueError("saturations, initial errors and rates must be positive")

    f = float(fraction)
    log_before = np.log(f) + np.log(sat0) - np.log(d00)
    log_after = np.log(f) + np.log(sat1) - np.log(d01)
    inv_mean = 0.5 * (1.0 / lam1 + 1.0 / lam0)

    instability = 0.5 * (log_after + log_before) * (1.0 / lam1 - 1.0 / lam0)
    amplitude = np.log(sat1 / sat0) * inv_mean
    accuracy = -np.log(d01 / d00) * inv_mean

    return {
        "horizon_before": float(log_before / lam0),
        "horizon_after": float(log_after / lam1),
        "total": float(log_after / lam1 - log_before / lam0),
        "instability": float(instability),
        "amplitude": float(amplitude),
        "accuracy": float(accuracy),
    }


def breakeven_accuracy(
    saturation: tuple[float, float],
    rate: tuple[float, float],
    delta0_before: float,
    fraction: float = 0.5,
    horizon_before: float | None = None,
) -> dict[str, float]:
    r"""How much better the initial state must be known just to stand still.

    Setting the horizon after equal to the horizon before in the horizon law
    and solving for the later initial error,

    .. math::
        \frac{1}{\lambda_1}\ln\frac{f\,\delta_\infty^{(1)}}{\delta_0^{(1)}}
        = \frac{1}{\lambda_0}\ln\frac{f\,\delta_\infty^{(0)}}{\delta_0^{(0)}}
        \;\Longrightarrow\;
        \delta_0^{(1)} = f\,\delta_\infty^{(1)}
          \left(\frac{\delta_0^{(0)}}{f\,\delta_\infty^{(0)}}
          \right)^{\lambda_1/\lambda_0} .

    Returns ``delta0_after``, the ``ratio``
    :math:`\delta_0^{(0)}/\delta_0^{(1)}` by which the analysis error must
    shrink, ``decades`` :math:`= \log_{10}` of that ratio, and the
    ``horizon_held``.

    ``horizon_before`` overrides the horizon the law would predict for the
    earlier climate, and passing the **measured** one matters: the law can be
    a good several per cent for one climate and excellent for another, and
    since the requirement is an exponential in :math:`\lambda_1 T`, a ten per
    cent error in :math:`T` moves the answer by a factor of two. Chapter 28
    reports both.

    The exponent :math:`\lambda_1/\lambda_0` is why the answer is brutal. The
    requirement is not proportional to the change in instability; the initial
    error has to be raised to that power, so a system whose leading exponent
    doubles needs its analysis error **squared** (relative to saturation) to
    hold the same horizon. Chapter 20's :math:`\ln 10/\lambda` says a decade of
    analysis error buys a fixed amount of lead; this says how many of those
    decades a changing climate can consume before any of them shows up as
    improved skill.
    """
    sat0, sat1 = (float(v) for v in saturation)
    lam0, lam1 = (float(v) for v in rate)
    d0 = float(delta0_before)
    if min(sat0, sat1, lam0, lam1, d0) <= 0.0:
        raise ValueError("saturations, rates and the initial error must be positive")
    f = float(fraction)
    target = (
        np.log(f * sat0 / d0) / lam0
        if horizon_before is None
        else float(horizon_before)
    )
    d1 = f * sat1 * np.exp(-lam1 * target)
    return {
        "delta0_after": float(d1),
        "ratio": float(d0 / d1),
        "decades": float(np.log10(d0 / d1)),
        "horizon_held": float(target),
    }


def factorial_attribution(
    old_system_old_climate: float,
    old_system_new_climate: float,
    new_system_old_climate: float,
    new_system_new_climate: float,
) -> dict[str, float]:
    r"""Attribute a skill change to the system, the climate, and their interaction.

    The 2x2 an operational centre can actually run. Two of the four cells are
    the historical record -- the old system forecasting the old atmosphere, the
    new system forecasting the new one -- and the other two are **reforecasts**:
    a frozen system re-run on a period it never operated in. Those two cells are
    the whole reason a reforecast archive exists.

    With :math:`T_{sc}` the horizon for system :math:`s` on climate :math:`c`,

    .. math::
        \underbrace{T_{11} - T_{00}}_{\text{observed}}
        = \underbrace{(T_{10} - T_{00})}_{\text{system}}
        + \underbrace{(T_{01} - T_{00})}_{\text{climate}}
        + \underbrace{(T_{11} - T_{10} - T_{01} + T_{00})}_{\text{interaction}} ,

    exactly. ``system`` and ``climate`` are each measured with the other factor
    held at its old value, which is what makes them separately meaningful; the
    ``interaction`` is what is left, and it is **not** a nuisance. By the
    horizon law the horizon is :math:`L/\lambda`, so a fixed improvement in
    :math:`\ln\delta_0` buys :math:`\Delta\ln\delta_0/\lambda` of horizon --
    less of it in a more unstable atmosphere. The interaction is therefore
    negative whenever the climate became less predictable, and its magnitude is
    the *declining return* on observing-system investment. Chapter 20's
    :math:`\ln 10/\lambda` per decade of analysis error is not a constant of
    nature; this is the term that says so.

    Also returned are the two ``main_*`` effects, each averaged over the other
    factor's two levels. Those sum to ``observed`` exactly with no interaction
    term at all, which is why they are the wrong thing to quote when the
    question is causal: they silently distribute half the interaction into each.
    """
    t00 = float(old_system_old_climate)
    t01 = float(old_system_new_climate)
    t10 = float(new_system_old_climate)
    t11 = float(new_system_new_climate)
    return {
        "observed": t11 - t00,
        "system": t10 - t00,
        "climate": t01 - t00,
        "interaction": t11 - t10 - t01 + t00,
        "main_system": 0.5 * ((t10 - t00) + (t11 - t01)),
        "main_climate": 0.5 * ((t01 - t00) + (t11 - t10)),
    }


def shift_share(
    groups_before: Array,
    values_before: Array,
    groups_after: Array,
    values_after: Array,
    n_groups: int,
) -> dict[str, Array | float]:
    r"""Split a change in a mean into a within-group and a reweighting part.

    The question this answers is the one asked of every observed change in
    atmospheric predictability: did the *dynamics* change, or did the
    *circulation* change so that the system now spends more time in states that
    were always the unstable ones? Both raise the mean local growth rate and
    they are different physics.

    With :math:`p_b` the occupancy of group :math:`b` and :math:`\mu_b` the
    group mean,

    .. math::
        \langle\mu\rangle_1 - \langle\mu\rangle_0
        = \underbrace{\sum_b \frac{p_b^1+p_b^0}{2}(\mu_b^1-\mu_b^0)}_{\text{within}}
        + \underbrace{\sum_b (p_b^1-p_b^0)\frac{\mu_b^1+\mu_b^0}{2}}_{\text{between}} ,

    which is an algebraic identity, not an approximation: expanding the right
    side cancels to :math:`\sum_b (p_b^1\mu_b^1 - p_b^0\mu_b^0)`. The symmetric
    form is used rather than the more common
    :math:`\sum_b (p^1_b-p^0_b)\mu^0_b + \sum_b p^1_b(\mu^1_b-\mu^0_b)` because
    that one is exact too but *order-dependent*: swapping the epochs changes the
    split. Here it only changes the sign.

    ``groups_*`` are integer group labels in ``[0, n_groups)``, from a binning
    that must be **comparable across the two epochs** -- that is the whole
    scientific content of the choice, and a binning defined on unnormalised
    amplitudes will report a reweighting that is nothing but the change in
    units. A group empty in one epoch contributes only to ``between``, with its
    occupied mean used for both, since there is no within-group change to
    measure.

    Returns ``total``, ``within``, ``between``, the two occupancy vectors and
    the two group-mean vectors.
    """
    gb = np.asarray(groups_before, dtype=int).ravel()
    ga = np.asarray(groups_after, dtype=int).ravel()
    vb = np.asarray(values_before, dtype=float).ravel()
    va = np.asarray(values_after, dtype=float).ravel()
    if gb.shape != vb.shape or ga.shape != va.shape:
        raise ValueError("group labels and values must have matching shapes")
    n = int(n_groups)
    if n < 1:
        raise ValueError("n_groups must be at least 1")
    if gb.size and (gb.min() < 0 or gb.max() >= n):
        raise ValueError("group labels must lie in [0, n_groups)")
    if ga.size and (ga.min() < 0 or ga.max() >= n):
        raise ValueError("group labels must lie in [0, n_groups)")

    count_b = np.bincount(gb, minlength=n).astype(float)
    count_a = np.bincount(ga, minlength=n).astype(float)
    sum_b = np.bincount(gb, weights=vb, minlength=n)
    sum_a = np.bincount(ga, weights=va, minlength=n)
    p_b = count_b / max(gb.size, 1)
    p_a = count_a / max(ga.size, 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mu_b = np.where(count_b > 0, sum_b / np.where(count_b > 0, count_b, 1), np.nan)
        mu_a = np.where(count_a > 0, sum_a / np.where(count_a > 0, count_a, 1), np.nan)

    # A group seen in only one epoch has no within-group change; carry the
    # occupied mean across so its whole contribution lands in `between`.
    mu_b_f = np.where(np.isnan(mu_b), mu_a, mu_b)
    mu_a_f = np.where(np.isnan(mu_a), mu_b, mu_a)
    both_empty = (count_b == 0) & (count_a == 0)
    mu_b_f = np.where(both_empty, 0.0, mu_b_f)
    mu_a_f = np.where(both_empty, 0.0, mu_a_f)

    within = float(np.sum(0.5 * (p_a + p_b) * (mu_a_f - mu_b_f)))
    between = float(np.sum((p_a - p_b) * 0.5 * (mu_a_f + mu_b_f)))
    return {
        "total": float(va.mean() - vb.mean()),
        "within": within,
        "between": between,
        "occupancy_before": p_b,
        "occupancy_after": p_a,
        "group_mean_before": mu_b,
        "group_mean_after": mu_a,
    }


# --------------------------------------------------------------------------
# Trends and their detectability
# --------------------------------------------------------------------------
def epoch_means(values: Array, n_epochs: int) -> Array:
    """Mean of ``values`` within each of ``n_epochs`` equal consecutive blocks.

    ``values`` has shape ``(n_epochs * n_per_epoch, ...)``; the leading axis is
    split. Reshaping rather than looping so an epoch axis can be added to a
    stack of per-case diagnostics without copying.
    """
    v = np.asarray(values, dtype=float)
    n = int(n_epochs)
    if v.shape[0] % n:
        raise ValueError(
            f"leading axis {v.shape[0]} is not divisible by n_epochs={n}"
        )
    return np.nanmean(v.reshape(n, v.shape[0] // n, *v.shape[1:]), axis=1)


def effective_sample_size(series: Array, max_lag: int | None = None) -> float:
    r"""Number of *independent* cases in an autocorrelated record.

    .. math::
        n_{\rm eff} = \frac{n}{1 + 2\sum_{k\ge1}\rho_k}

    with the sum truncated at the first non-positive :math:`\rho_k` (the
    initial-positive-sequence rule), which is what keeps the estimator from
    being dominated by the noise in the long-lag autocorrelations.

    This is the correction that decides how long a record has to be. An
    operational centre runs a forecast every day, but two forecasts launched a
    day apart verify against an atmosphere that has barely changed, so their
    errors are strongly correlated and they are not two independent
    measurements of the atmosphere's predictability. Treating them as
    independent inflates the apparent significance of any trend --
    :func:`trend_detection_power` takes ``n_per_epoch``, and the number to pass
    it is this one, not the number of forecasts.

    Exact reference values a test can use: white noise gives :math:`n`, and an
    AR(1) process with lag-one correlation :math:`\phi` gives
    :math:`n(1-\phi)/(1+\phi)` in the long-record limit.
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        raise ValueError("need at least four finite values")
    x = x - x.mean()
    var = float(np.mean(x**2))
    if var <= 0.0:
        return float(n)
    top = n // 4 if max_lag is None else min(int(max_lag), n - 1)
    total = 0.0
    for k in range(1, max(top, 1) + 1):
        rho = float(np.mean(x[:-k] * x[k:]) / var)
        if rho <= 0.0:
            break
        total += rho
    return float(n / (1.0 + 2.0 * total))


def linear_trend(x: Array, y: Array) -> dict[str, float]:
    r"""Ordinary least-squares trend with its standard error and t-statistic.

    Returns ``slope``, ``intercept``, ``stderr``, ``tstat``, ``pvalue``
    (two-sided), ``df`` and ``n``. Non-finite pairs are dropped, which is what
    makes this safe on a horizon record where some cases never crossed the
    threshold -- but dropping them is **not** neutral: a censored horizon is a
    long one, so dropping it biases the record short exactly where
    predictability is highest. Chapter 28 reports the censored count alongside
    every trend for that reason.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    n = x.size
    if n < 3:
        raise ValueError("need at least three finite pairs for a trend and its error")

    xm, ym = x.mean(), y.mean()
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0.0:
        raise ValueError("x has no variance; the slope is not identifiable")
    slope = float(np.sum((x - xm) * (y - ym)) / sxx)
    intercept = float(ym - slope * xm)
    residual = y - (intercept + slope * x)
    df = n - 2
    s2 = float(np.sum(residual**2) / df)
    stderr = float(np.sqrt(s2 / sxx))
    tstat = float(slope / stderr) if stderr > 0.0 else np.inf
    pvalue = float(2.0 * stats.t.sf(abs(tstat), df))
    return {
        "slope": slope,
        "intercept": intercept,
        "stderr": stderr,
        "tstat": tstat,
        "pvalue": pvalue,
        "df": float(df),
        "n": float(n),
    }


def trend_standard_error(
    noise_std: float, n_epochs: int, n_per_epoch: int = 1
) -> float:
    r"""Standard error of an OLS trend on ``n_epochs`` equally spaced epochs.

    With the epoch index running :math:`0,1,\dots,n-1` and :math:`m` independent
    cases per epoch, :math:`S_{xx} = m\,n(n^2-1)/12` and

    .. math:: \operatorname{se}(b) = \frac{\sigma}{\sqrt{m\,n(n^2-1)/12}} .

    The :math:`n^{-3/2}` is the number that governs everything in this part of
    the chapter: **a longer record is worth far more than a bigger sample within
    each epoch.** Doubling the cases per year buys a factor :math:`\sqrt2`;
    doubling the length of the record buys :math:`2\sqrt2`.
    """
    n, m = int(n_epochs), int(n_per_epoch)
    if n < 3:
        raise ValueError("a trend needs at least three epochs")
    if m < 1:
        raise ValueError("n_per_epoch must be at least one")
    sxx = m * n * (n * n - 1) / 12.0
    return float(float(noise_std) / np.sqrt(sxx))


def trend_detection_power(
    slope: float,
    noise_std: float,
    n_epochs: int,
    n_per_epoch: int = 1,
    alpha: float = 0.05,
) -> float:
    r"""Probability that a trend of size ``slope`` is found significant.

    A two-sided t-test at level ``alpha`` on the OLS slope, with the
    non-centrality :math:`\delta = b/\operatorname{se}(b)` from
    :func:`trend_standard_error` and :math:`\nu = nm-2` degrees of freedom. The
    power is evaluated from the **non-central** t distribution rather than the
    usual normal approximation, which overstates power for short records --
    precisely the regime the chapter is about.

    Returned as a probability in :math:`[0,1]`, and always finite: SciPy's
    ``nct`` returns ``nan`` for a large ``df`` with a large non-centrality, so
    that corner falls back to the normal limit, which the t distribution
    matches to many digits by then. Without the guard a long-record power
    lands in a chapter as ``nan``.

    It counts *significance*, not
    correctness of sign; for the small non-centralities that arise here those
    differ, and a record can be significant with the wrong sign. Chapter 28
    measures that rate by Monte Carlo rather than reading it off this function.
    """
    n, m = int(n_epochs), int(n_per_epoch)
    se = trend_standard_error(noise_std, n, m)
    df = n * m - 2
    if df < 1:
        raise ValueError("need at least three observations in total")
    nc = float(slope) / se
    crit = float(stats.t.isf(0.5 * float(alpha), df))
    upper = float(stats.nct.sf(crit, df, nc))
    lower = float(stats.nct.cdf(-crit, df, nc))
    power = upper + lower
    if np.isfinite(power):
        return power
    # scipy's non-central t returns nan for large df combined with a large
    # non-centrality -- exactly the easy-detection corner. There the t
    # distribution is the standard normal to many digits, so fall back to it
    # rather than propagating a nan into a record-length scan.
    upper = float(stats.norm.sf(crit - nc))
    lower = float(stats.norm.cdf(-crit - nc))
    return upper + lower


def minimum_record_length(
    slope: float,
    noise_std: float,
    n_per_epoch: int = 1,
    power: float = 0.8,
    alpha: float = 0.05,
    max_epochs: int = 10_000,
) -> int:
    """Shortest record reaching ``power`` for a trend of size ``slope``.

    Increasing ``n_epochs`` raises the power monotonically (the non-centrality
    grows like :math:`n^{3/2}` while the critical value falls), so a plain
    upward scan is exact rather than a heuristic. Returns ``max_epochs`` if the
    target is not reached, and the chapter says so when it happens rather than
    quoting the cap as an answer.
    """
    for n in range(3, int(max_epochs) + 1):
        if trend_detection_power(slope, noise_std, n, n_per_epoch, alpha) >= power:
            return n
    return int(max_epochs)
