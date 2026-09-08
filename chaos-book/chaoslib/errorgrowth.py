"""Error growth beyond the exponential regime.

The exponential picture -- error :math:`\\propto e^{\\lambda t}`, horizon set by
the doubling time -- is only the *small-error* limit. Real forecast errors
saturate at the climatological difference between two randomly chosen states,
and the approach to saturation is what actually sets the useful forecast
horizon. Lorenz's logistic model captures both regimes with one parameter each.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

Array = NDArray[np.floating]

__all__ = [
    "logistic_error_growth",
    "fit_logistic_error_growth",
    "saturation_level",
    "predictability_horizon",
    "lagged_forecast_difference",
    "cascade_rates",
    "cascade_growth",
    "cascade_contamination_time",
    "horizon_law",
    "initialised_error_variance",
    "initialised_advantage",
    "initialised_useful_lead",
    "KOLMOGOROV_ALPHA",
]

#: Growth-rate exponent implied by Kolmogorov scaling. The eddy turnover time
#: at scale :math:`\ell` is :math:`\tau \sim \varepsilon^{-1/3}\ell^{2/3}`,
#: so the growth rate goes as :math:`\ell^{-2/3}` and doubles every
#: :math:`2/3` of an octave. This is the exponent for which Lorenz's argument
#: gives a finite predictability limit.
KOLMOGOROV_ALPHA = 2.0 / 3.0


def logistic_error_growth(
    t: Array, e0: float, growth_rate: float, saturation: float
) -> Array:
    r"""Lorenz's logistic error-growth model.

    Solves :math:`\dot E = \lambda E (1 - E/E_\infty)`, giving

    .. math::
        E(t) = \frac{E_\infty}{1 + (E_\infty/E_0 - 1)\,e^{-\lambda t}} .

    Small :math:`E` recovers pure exponential growth at rate :math:`\lambda`;
    large :math:`t` approaches the saturation value :math:`E_\infty`. The single
    nonlinear term is a crude stand-in for the fact that errors cannot exceed
    the spread of the attractor itself.
    """
    t = np.asarray(t, dtype=float)
    if e0 <= 0.0:
        raise ValueError("e0 must be positive")
    return saturation / (1.0 + (saturation / e0 - 1.0) * np.exp(-growth_rate * t))


def fit_logistic_error_growth(
    t: Array, error: Array, saturation: float | None = None, space: str = "log"
) -> tuple[float, float, float]:
    r"""Fit :math:`(E_0, \lambda, E_\infty)` to a measured error curve.

    If ``saturation`` is supplied it is held fixed (the usual case -- it is
    better measured directly from the climatological spread than fitted, since
    the tail of the curve constrains it weakly). Returns
    ``(e0, growth_rate, saturation)``.

    **Fit in log space, which is why that is the default.** An error curve spans
    the whole way from the initial perturbation to saturation -- ten orders of
    magnitude is normal -- and least squares on :math:`E` weights each point by
    :math:`E`. The handful of points near saturation then contribute residuals
    of order :math:`E_\infty` while the entire exponential phase contributes
    residuals of order :math:`E_0`, so the optimiser never sees the exponential
    phase at all, and :math:`\lambda` -- which is *defined* by that phase --
    comes out of a fit that ignored it.

    Measured on the 1024-member Lorenz 63 twin experiment of chapter 9, started
    at :math:`\delta_0 = 10^{-6}` and against an early-time exponential rate of
    0.921: ``space="log"`` returns :math:`\lambda = 0.919` and
    :math:`E_0 = 3.6\times10^{-6}`, while ``space="linear"`` returns 0.748 and
    :math:`5.3\times10^{-5}`. So the linear fit is 19 % out in the rate and
    fifty-fold out in the amplitude, and it produces a curve that looks entirely
    convincing on a linear plot. The log-space rate is also stable across
    initial amplitudes (0.920, 0.919, 0.921 at
    :math:`\delta_0 = 10^{-8}, 10^{-6}, 10^{-4}`) where the linear one is not
    (0.716, 0.748, 0.722). ``space="linear"`` is kept only so that chapter 9 can
    show this happening.

    Note also that the fitted :math:`\lambda` is not the Lyapunov exponent even
    when the fit is done properly: fitted over the nonlinear range it comes out
    12 % below :math:`\lambda_1` for Lorenz 63 and 26 % below for Lorenz 96.
    They are different quantities.
    """
    t = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    e_sat = float(np.nanmax(error)) if saturation is None else float(saturation)
    if space not in ("log", "linear"):
        raise ValueError(f"space must be 'log' or 'linear', not {space!r}")
    if space == "log" and np.any(error <= 0.0):
        raise ValueError("log-space fitting needs strictly positive errors")

    fit_free = saturation is None

    def _predict(tt: Array, log_e0: float, rate: float, sat: float) -> Array:
        return logistic_error_growth(tt, np.exp(log_e0), rate, sat)

    def _residual_target(values: Array) -> Array:
        return np.log(values) if space == "log" else values

    if fit_free:
        def model(tt: Array, log_e0: float, rate: float, sat: float) -> Array:
            return _residual_target(_predict(tt, log_e0, rate, sat))

        p0 = (np.log(max(error[0], 1e-12)), 1.0, e_sat)
        popt, _ = curve_fit(
            model, t, _residual_target(error), p0=p0, maxfev=40000
        )
        return float(np.exp(popt[0])), float(popt[1]), float(popt[2])

    def model_fixed(tt: Array, log_e0: float, rate: float) -> Array:
        return _residual_target(_predict(tt, log_e0, rate, e_sat))

    p0 = (np.log(max(error[0], 1e-12)), 1.0)
    popt, _ = curve_fit(
        model_fixed, t, _residual_target(error), p0=p0, maxfev=40000
    )
    return float(np.exp(popt[0])), float(popt[1]), e_sat


def saturation_level(
    trajectory: Array, seed: int | None = 0, statistic: str = "rms"
) -> float:
    r"""Climatological error saturation: distance between random state pairs.

    The level a forecast error tends to once all skill is gone -- equivalently
    the error of a random draw from climatology. Computed from independently
    shuffled pairs of states from a long trajectory, which is what "two randomly
    chosen states of the attractor" means operationally.

    **Match the statistic to the error curve you are comparing against**, and
    this is not a nicety. ``"rms"`` returns
    :math:`\sqrt{\langle\|x-y\|^2\rangle}` and ``"mean"`` returns
    :math:`\langle\|x-y\|\rangle`; by Jensen's inequality the second is
    always the smaller, and how much smaller depends on the attractor. For
    Lorenz 96 (:math:`N=40, F=8`) the ratio is 0.995 -- distances concentrate in
    high dimension and the choice hardly matters. For Lorenz 63 it is **0.889**,
    because the two lobes give a broad distribution of pair distances. Comparing
    an ensemble-*mean* error curve against the *RMS* saturation therefore makes
    the curve appear to stop growing at 89 % of saturation, which then shows up
    as a spurious 12 % error in the logistic model's shape. Chapter 9 measures
    this; it cost real time to find.

    The RMS form also satisfies an exact identity worth checking an
    implementation against: for independent draws from one distribution,
    :math:`\langle\|x-y\|^2\rangle = 2\langle\|x-\bar x\|^2\rangle`, so
    the RMS saturation is :math:`\sqrt2` times the RMS spread about the mean.
    """
    traj = np.asarray(trajectory, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(traj.shape[0])
    distance = np.sqrt(np.sum((traj - traj[idx]) ** 2, axis=-1))
    if statistic == "rms":
        return float(np.sqrt(np.mean(distance**2)))
    if statistic == "mean":
        return float(np.mean(distance))
    raise ValueError(f"statistic must be 'rms' or 'mean', not {statistic!r}")


def predictability_horizon(
    t: Array, error: Array, threshold_frac: float = 0.5
) -> float:
    """First time the error exceeds ``threshold_frac`` of its saturation value.

    A blunter but more operationally meaningful figure than the doubling time:
    it answers "how long is this forecast worth using", and it is finite even
    when growth is not exponential. Returns ``inf`` if the threshold is never
    crossed within the record.
    """
    t = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    target = threshold_frac * np.nanmax(error)
    crossed = np.flatnonzero(error >= target)
    return float(t[crossed[0]]) if crossed.size else float("inf")


def lagged_forecast_difference(forecasts: Array, gap: int = 1) -> Array:
    r"""Lorenz's (1982) error-growth estimator, which never uses the truth.

    ``forecasts`` has shape ``(n_leads, n_starts, n_state)``:
    ``forecasts[L, m]`` is the forecast started from analysis :math:`m` at lead
    :math:`L`, so it verifies at analysis time :math:`m + L`. That is the shape
    an operational archive has.

    For a verification time :math:`v`, two forecasts valid at :math:`v` were
    started :math:`\text{gap}` analyses apart: one at :math:`v - L` with lead
    :math:`L`, the other at :math:`v - L - \text{gap}` with lead
    :math:`L + \text{gap}`. Their RMS difference, averaged over :math:`v`, is
    returned as a function of :math:`L`.

    **Why this is worth having.** A forecast cannot be verified against the
    truth, because there is no truth -- only an analysis, which is itself a
    forecast corrected by observations, and whose own error is a large fraction
    of a short-range forecast's. Twin experiments are unavailable for the same
    reason. But two forecasts of *different lead* valid at the *same time* can
    be differenced with no reference to anything external, and their difference
    grows at the same rate the error does: at the moment the older forecast
    reaches :math:`v - L`, it differs from the younger one's starting analysis
    by roughly a ``gap``-cycle forecast error, and both are then integrated
    forward together for :math:`L` cycles.

    The curve is *offset* below the true error curve -- it starts from a
    forecast difference rather than from the analysis error -- but the offset
    does not matter, because the growth **rate** is what the method is for.
    Validated in chapter 13 against the truth on a synthetic Lorenz 96 archive:
    fitted over :math:`E \in [0.02, 0.2]\,E_\infty`, the estimated rate is
    within 2 % of the true one for analysis errors of 0.5 % and 2 % of the
    climatological spread, and within 5 % at 6 %.

    **What it cannot see.** The two forecasts come from the same model, so any
    systematic model error is common to both and cancels algebraically in the
    difference -- adding a constant bias to an entire archive leaves this
    function's output unchanged to round-off, which the tests assert. The
    method estimates the growth of *initial-condition* error and is blind to
    model error by construction -- see chapter 21. It also needs an exponential
    phase to fit: when the analysis error is already a large fraction of
    saturation (44 % in chapter 13's most degraded configuration) there is
    nothing left to measure, and that is a failure of the archive rather than of
    the estimator.
    """
    array = np.asarray(forecasts, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            "forecasts must have shape (n_leads, n_starts, n_state), got "
            f"{array.shape}"
        )
    n_leads, n_starts = array.shape[0], array.shape[1]
    gap = int(gap)
    if gap < 1:
        raise ValueError("gap must be at least 1")

    out = np.full(max(0, n_leads - gap), np.nan)
    for lead in range(out.size):
        verification = np.arange(lead + gap, n_starts)
        if verification.size == 0:
            continue
        older = array[lead + gap, verification - lead - gap]
        younger = array[lead, verification - lead]
        out[lead] = float(
            np.sqrt(np.mean((older - younger) ** 2, axis=-1)).mean()
        )
    return out


# --------------------------------------------------------------------------
# The upscale error cascade (Lorenz 1969)
# --------------------------------------------------------------------------
def cascade_rates(
    n_bands: int, alpha: float = KOLMOGOROV_ALPHA, rate0: float = 1.0
) -> Array:
    r"""Growth rate of each octave band, largest scale first.

    Band :math:`n` has scale :math:`L\,2^{-n}` and growth rate

    .. math:: \lambda_n = \lambda_0\,2^{\alpha n},

    so :math:`\alpha` is how much faster small scales grow than large ones.
    :math:`\alpha = 2/3` is Kolmogorov (:data:`KOLMOGOROV_ALPHA`);
    :math:`\alpha = 0` is a system whose growth rate does not depend on scale,
    which is what Lorenz 63 and Lorenz 96 are. Everything in this chapter turns
    on which of those two cases holds.
    """
    return float(rate0) * 2.0 ** (float(alpha) * np.arange(int(n_bands)))


def _cascade_rhs(t: float, e: Array, rates: Array, coupling: float) -> Array:
    """Logistic growth in each band, forced from the band below it in scale."""
    e = np.clip(np.asarray(e, dtype=float), 0.0, 1.0)
    upscale = np.zeros_like(e)
    upscale[:-1] = e[1:]  # band n is fed by band n+1, one octave smaller
    return rates * (e + coupling * upscale) * (1.0 - e)


def cascade_growth(
    n_bands: int,
    alpha: float = KOLMOGOROV_ALPHA,
    rate0: float = 1.0,
    coupling: float = 1.0,
    seed_amplitude: float = 1.0,
    seed_band: int | None = None,
    t_final: float = 40.0,
    n_times: int = 400,
) -> tuple[Array, Array]:
    r"""Error in every octave band over time, for Lorenz's cascade argument.

    Each band's error :math:`e_n` is measured as a fraction of *its own*
    saturation level, so :math:`e_n \in [0, 1]`, and obeys

    .. math::
        \frac{de_n}{dt} = \lambda_n\bigl(e_n + \kappa\,e_{n+1}\bigr)
                          \bigl(1 - e_n\bigr).

    The first term is ordinary logistic growth -- exponential at rate
    :math:`\lambda_n` until it saturates -- and the second is the **upscale
    cascade**: band :math:`n` is contaminated by the band one octave smaller
    than it. With a single band the second term is absent and this reduces
    *exactly* to :func:`logistic_error_growth`, which the tests check.

    ``seed_band`` defaults to the smallest (``n_bands - 1``), which is the
    physically meaningful case: an observing system has a resolution, and about
    scales finer than it we know nothing, so the error there starts at
    saturation (``seed_amplitude = 1``). **Adding a band therefore represents
    improving the observing resolution by one octave**, not reducing an error
    amplitude, and that distinction is the whole point of the chapter.

    Returns ``(times, errors)`` with ``errors`` of shape
    ``(n_times, n_bands)``, band 0 first.
    """
    from scipy.integrate import solve_ivp

    rates = cascade_rates(n_bands, alpha, rate0)
    initial = np.zeros(int(n_bands), dtype=float)
    initial[int(n_bands) - 1 if seed_band is None else int(seed_band)] = float(
        seed_amplitude
    )
    times = np.linspace(0.0, float(t_final), int(n_times))
    solution = solve_ivp(
        _cascade_rhs,
        (0.0, float(t_final)),
        initial,
        args=(rates, float(coupling)),
        t_eval=times,
        rtol=1e-10,
        atol=1e-14,
    )
    return solution.t, solution.y.T


def cascade_contamination_time(
    n_bands: int,
    alpha: float = KOLMOGOROV_ALPHA,
    rate0: float = 1.0,
    coupling: float = 1.0,
    seed_amplitude: float = 1.0,
    seed_band: int | None = None,
    threshold: float = 0.5,
    t_max: float = 4000.0,
) -> float:
    r"""Time for the **largest** scale to reach ``threshold`` of saturation.

    This is the number Lorenz's argument is about. Seeding the smallest resolved
    band at saturation and adding bands one at a time, it

    * **converges** for :math:`\alpha > 0` -- measured 1.4466 at
      :math:`\alpha = 2/3`, 2.3035 at :math:`1/3`, 1.1220 at :math:`1`, with the
      increment from one more octave falling below :math:`10^{-4}` by sixteen
      bands. Resolving finer scales stops buying lead time;
    * **diverges** for :math:`\alpha = 0`, growing without bound at a settled
      0.281 per octave out to 128 bands.

    So a finite predictability limit follows from small scales growing faster
    than large ones, and from nothing else. Note that the naive estimate
    :math:`\sum_n \lambda_n^{-1} = (1 - 2^{-\alpha})^{-1}` gets the
    *convergence* right and the constant wrong (2.70 against 1.4466 at
    :math:`\alpha = 2/3`), because the bands overlap in time -- band
    :math:`n-1` starts growing well before band :math:`n` saturates.

    Returns ``nan`` if the threshold is not reached within ``t_max``.
    """
    from scipy.integrate import solve_ivp

    rates = cascade_rates(n_bands, alpha, rate0)
    initial = np.zeros(int(n_bands), dtype=float)
    initial[int(n_bands) - 1 if seed_band is None else int(seed_band)] = float(
        seed_amplitude
    )

    def reached(t: float, e: Array, *_: object) -> float:
        return float(e[0]) - float(threshold)

    reached.terminal = True  # type: ignore[attr-defined]
    reached.direction = 1.0  # type: ignore[attr-defined]

    solution = solve_ivp(
        _cascade_rhs,
        (0.0, float(t_max)),
        initial,
        args=(rates, float(coupling)),
        events=reached,
        rtol=1e-10,
        atol=1e-16,
    )
    hits = solution.t_events[0]
    return float(hits[0]) if hits.size else float("nan")


# --------------------------------------------------------------------------
# Initialised prediction of a slow component (chapter 26)
# --------------------------------------------------------------------------
def initialised_error_variance(
    lead: Array, memory: float, analysis_error: float, climatological_sd: float
) -> Array:
    r"""Error variance of an initialised forecast of an Ornstein-Uhlenbeck
    component with memory :math:`T`.

    .. math::
        \operatorname{var}\varepsilon(\ell)
          = \epsilon^2 e^{-2\ell/T} + 2\sigma^2\!\left(1 - e^{-2\ell/T}\right)

    Forecast and truth share their predictable part
    :math:`S_0 e^{-\ell/T}` up to the analysis error :math:`\epsilon`, and
    their unpredictable parts are independent -- which is why the saturation is
    :math:`2\sigma^2` and not :math:`\sigma^2`. Getting that factor of two
    wrong shifts the useful lead of :func:`initialised_useful_lead` by about a
    fifth of a memory time, which is how it was caught.
    """
    lead = np.asarray(lead, dtype=float)
    decay = np.exp(-2.0 * lead / float(memory))
    return (
        float(analysis_error) ** 2 * decay
        + 2.0 * float(climatological_sd) ** 2 * (1.0 - decay)
    )


def initialised_advantage(
    lead: Array, memory: float, analysis_error: float, climatological_sd: float
) -> Array:
    r"""Error-variance advantage of initialising over drawing from climatology.

    .. math::
        \operatorname{var}\varepsilon_{\rm uninit}
          - \operatorname{var}\varepsilon_{\rm init}
          = \left(2\sigma^2 - \epsilon^2\right) e^{-2\ell/T}

    An uninitialised forecast has error variance :math:`2\sigma^2` at *every*
    lead, so the whole lead-dependence sits in the initialised arm and the
    advantage decays as a pure exponential with rate :math:`2/T`. **That makes
    the decay of forecast skill a direct measurement of the component's
    memory**, needing no autocorrelation estimate -- which matters because a
    reservoir driven by fast weather carries a small high-frequency component
    that biases a lag-1 estimate of :math:`T` low by a factor of three or more
    (chapter 26, section 1).

    Measured against the model at :math:`\epsilon = 0.3\sigma`, the fitted
    decay rate is 0.87 to 1.01 of :math:`2/T` for reservoirs with
    :math:`T \ge 8`, and 1.24 at :math:`T = 2` where the reservoir is not
    slow enough for the weather to look white.
    """
    lead = np.asarray(lead, dtype=float)
    return (
        2.0 * float(climatological_sd) ** 2 - float(analysis_error) ** 2
    ) * np.exp(-2.0 * lead / float(memory))


def initialised_useful_lead(
    memory: float,
    analysis_error: float,
    climatological_sd: float,
    fraction: float = 0.95,
) -> float:
    r"""Lead at which an initialised forecast's RMSE reaches ``fraction`` of an
    uninitialised one's -- the point past which initialising has stopped paying.

    .. math::
        \ell^\ast = -\frac{T}{2}
          \ln \frac{2\sigma^2\left(1 - f^2\right)}{2\sigma^2 - \epsilon^2}

    **The useful lead is strictly proportional to the memory.** The threshold
    and the analysis error set only the coefficient: at
    :math:`\epsilon = 0.3\sigma` and :math:`f = 0.95` it is
    :math:`1.141\,T`. So no observing system, however good, buys decadal skill
    in a component whose memory is a season -- the coefficient improves like a
    logarithm while the memory multiplies.

    Returns ``inf`` when the analysis is already better than the threshold
    demands (:math:`\epsilon^2 \ge 2\sigma^2(1-f^2)`... in which case the
    forecast never degrades that far, which cannot happen for
    :math:`f < 1` and finite :math:`\epsilon`), and ``nan`` for an analysis
    no better than climatology.
    """
    sigma2 = 2.0 * float(climatological_sd) ** 2
    eps2 = float(analysis_error) ** 2
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must lie strictly between 0 and 1")
    if eps2 >= sigma2:
        return float("nan")
    ratio = sigma2 * (1.0 - float(fraction) ** 2) / (sigma2 - eps2)
    if ratio >= 1.0:
        return 0.0
    return float(-0.5 * float(memory) * np.log(ratio))


def horizon_law(
    saturation: float,
    delta0: float,
    rate: float,
    fraction: float = 0.5,
) -> float:
    r"""Predictability horizon from exponential growth alone,

    .. math:: T = \frac{1}{\lambda}\ln\frac{f\,\delta_\infty}{\delta_0},

    the time for an error :math:`\delta_0` growing at :math:`\lambda` to reach a
    fraction :math:`f` of its saturation value.

    **This is the book's thesis in one formula, and chapter 3 measures it as
    one.** It contains no fitted constant, and it holds across the whole
    hierarchy: measured against the logistic map (one variable, discrete time),
    Lorenz 63 (three), and Lorenz 96 at twelve and forty variables, over six
    decades of :math:`\delta_0`, :math:`\lambda T` agrees with
    :math:`\ln(f\delta_\infty/\delta_0)` to **9 %** at worst. A one-variable map
    and a forty-variable flow obey the same law with each system's own
    :math:`\lambda` and :math:`\delta_\infty` substituted.

    Two cautions, both measured in chapter 3 rather than asserted.

    The law says the horizon grows **without bound** as
    :math:`\delta_0 \to 0`, like :math:`\ln(1/\delta_0)`. That is a property of
    every system in this library, and it is *not* a property of the atmosphere:
    a system with a spectrum of scales has a **finite** limit that no
    observation accuracy defeats (:func:`cascade_contamination_time`, and
    chapter 12). The law transfers up the hierarchy; the unboundedness does not
    survive the addition of a scale spectrum.

    And :math:`\lambda` is not the only number that matters. Across Lorenz 96
    from eight to forty variables :math:`\lambda_1` stays within 13 % while the
    number of unstable directions grows from 2 to 13 -- so a low-order model can
    get this law exactly right and still be wrong about the *dimension* of the
    forecast error, which is what sets the ensemble size (chapter 19).
    """
    if float(rate) <= 0.0:
        return float("nan")
    ratio = float(fraction) * float(saturation) / float(delta0)
    if ratio <= 1.0:
        return 0.0
    return float(np.log(ratio) / float(rate))

