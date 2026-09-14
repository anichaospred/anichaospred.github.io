r"""Invariant measures, time averages, and how long a run has to be.

"Climate" is not a trajectory. It is a probability measure :math:`\nu` on state
space -- the *invariant measure* -- and every climatological statement is an
integral against it,

.. math:: \langle A \rangle_\nu = \int A(x)\,d\nu(x).

A model run does not give you :math:`\nu`. It gives you one trajectory, and the
**Birkhoff ergodic theorem** is the licence to use the one for the other: if the
system is ergodic, the time average along :math:`\nu`-almost every trajectory
converges to the ensemble average,

.. math::
    \bar A_T = \frac{1}{T}\int_0^T A(x(t))\,dt \;\longrightarrow\;
    \langle A \rangle_\nu .

The theorem is a limit. It gives **no rate**, it makes no promise about the
trajectory you actually ran, and it says nothing about how to tell from inside a
run whether it has converged. This module is the arithmetic of those three
omissions:

* :func:`batch_variance` and :func:`autocorrelation_time` -- the rate, measured
  two ways that do not always agree;
* :func:`budget_variance` and :func:`budget_penalty` -- one long run against
  many short ones at fixed cost;
* :func:`lorenz63_moment_residuals` -- exact identities that :math:`\nu`
  satisfies and no finite run does, so the residual is a convergence diagnostic
  that needs no reference run;
* :func:`trailing_average_bias` -- what a time average estimates when there is
  no invariant measure to estimate, because the forcing is moving.

Two systems here have an invariant measure in **closed form**, which is what
makes the rest testable: the logistic map at :math:`r=4`
(:func:`logistic_invariant_density`, the arcsine law) and a gradient flow with
additive noise (:func:`boltzmann_density`). Everything else is measured against
those.

Notation follows ``NOTATION.md``: :math:`\nu` is the invariant measure,
:math:`\bar A_T` a time average over a window :math:`T`, and
:math:`\tau_{\rm int}` the integrated autocorrelation time. Note that the
double-well tilt of chapters 27 and 30 is also written :math:`\mu`; this module
never uses :math:`\mu` for a measure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = ArrayLike
Floats = NDArray[np.float64]

__all__ = [
    "time_average",
    "running_time_average",
    "trailing_average",
    "trailing_average_bias",
    "block_means",
    "batch_variance",
    "autocorrelation_time",
    "sampling_error",
    "budget_variance",
    "budget_penalty",
    "power_law_fit",
    "empirical_measure",
    "wasserstein1",
    "total_variation",
    "logistic_invariant_density",
    "logistic_invariant_cdf",
    "logistic_invariant_quantile",
    "boltzmann_density",
    "rotation_orbit",
    "star_discrepancy",
    "occupancy",
    "lorenz63_moment_residuals",
]


# =========================================================================
# time averages
# =========================================================================
def time_average(values: Array, dt: float = 1.0, axis: int = 0) -> Floats:
    r"""Trapezoidal time average :math:`\bar A_T = T^{-1}\int_0^T A\,dt`.

    ``values`` is sampled on a uniform grid of spacing ``dt`` along ``axis``,
    and :math:`T = (n-1)\,\Delta t` is the span of the samples -- **not**
    :math:`n\,\Delta t`, which is the commonest off-by-one here.

    The trapezoidal rule rather than :func:`numpy.mean` for a reason that
    matters in :func:`lorenz63_moment_residuals`: the exact identities this
    module checks equate a time average to a *boundary* term, and the rectangle
    rule commits an error of exactly the boundary size,
    :math:`\tfrac12(A_0 + A_T)\Delta t/T`, which is the same order as the
    quantity being tested. Measured on Lorenz 63 over 20 time units at
    :math:`\Delta t = 0.005`, the trapezoidal rule satisfies the first moment
    identity to :math:`7\times10^{-7}` and the rectangle rule to
    :math:`1.7\times10^{-6}`; at :math:`\Delta t = 0.0025` the gap widens to
    three orders of magnitude.
    """
    a = np.asarray(values, dtype=float)
    n = a.shape[axis]
    if n < 2:
        raise ValueError("need at least two samples to average over a window")
    return np.trapezoid(a, dx=float(dt), axis=axis) / (float(dt) * (n - 1))


def running_time_average(values: Array, dt: float = 1.0) -> Floats:
    r"""Cumulative time average :math:`\bar A_t` for every :math:`t`, by
    trapezoid.

    The first entry is ``values[0]`` by convention -- a zero-length window
    averages to its own endpoint -- so the returned array has the same length
    as the input and can be plotted against the same time axis. This is the
    curve that makes Birkhoff convergence visible: it should settle, and the
    envelope of its wandering is the sampling error of the run.
    """
    a = np.asarray(values, dtype=float).ravel()
    if a.size < 2:
        return a.copy()
    increments = 0.5 * (a[1:] + a[:-1]) * float(dt)
    integral = np.concatenate(([0.0], np.cumsum(increments)))
    elapsed = np.arange(a.size, dtype=float) * float(dt)
    out = np.empty_like(a)
    out[0] = a[0]
    out[1:] = integral[1:] / elapsed[1:]
    return out


def trailing_average(values: Array, window: int) -> Floats:
    r"""Boxcar average over the trailing ``window`` samples, ``nan``-padded.

    Entry :math:`i` is the mean of samples :math:`i-w+1 \ldots i`, so the
    result is aligned to the **end** of its window and the first :math:`w-1`
    entries are ``nan``. That alignment is the one an observational record
    uses -- a thirty-year normal is quoted at the end of the thirty years --
    and it is the reason the estimate lags a moving climate by half a window
    (:func:`trailing_average_bias`).

    A ``window`` of :math:`w` samples spans :math:`(w-1)\Delta t` of continuous
    time, the same convention as :func:`time_average`. Passing :math:`w\Delta t`
    to :func:`trailing_average_bias` instead is wrong by one sample -- small,
    but it is a bias in a quantity whose whole interest is that it is exact.
    """
    a = np.asarray(values, dtype=float).ravel()
    w = int(window)
    if w < 1 or w > a.size:
        raise ValueError(f"window {w} outside 1..{a.size}")
    kernel = np.ones(w, dtype=float) / w
    valid = np.convolve(a, kernel, mode="valid")
    out = np.full(a.size, np.nan)
    out[w - 1 :] = valid
    return out


def trailing_average_bias(rate: float, window: float) -> float:
    r"""Bias of a trailing time average of a **drifting** mean, exactly
    :math:`-bT/2`.

    If the underlying mean moves linearly, :math:`\langle A\rangle(t) = a + bt`,
    a trailing average over a window of length :math:`T` returns the value the
    mean had at the window's *midpoint*, not at its end:

    .. math::
        \frac{1}{T}\int_{t-T}^{t}(a + bs)\,ds = a + bt - \frac{bT}{2}.

    The bias is exact, independent of the variability about the mean, and does
    **not** shrink as the record lengthens -- lengthening the window makes it
    worse. That is the difference between a slow estimator and an inconsistent
    one, and it is why a moving climate has to be sampled across an ensemble
    rather than along a trajectory (chapter 30, section 6; chapter 28).

    ``window`` is a continuous span. For a boxcar of :math:`w` samples that is
    :math:`(w-1)\Delta t`; see :func:`trailing_average`.

    Measured on Lorenz 63 with :math:`\rho` ramped at
    :math:`3\times10^{-3}\,\mathrm{TU}^{-1}`, the trailing mean of :math:`z`
    lags the ensemble mean by :math:`-0.6015` at :math:`T = 400` against
    :math:`-0.6004` predicted: 0.2 %.
    """
    return -0.5 * float(rate) * float(window)


# =========================================================================
# how fast does a time average converge?
# =========================================================================
def block_means(values: Array, block: int) -> Floats:
    """Means of consecutive non-overlapping blocks of ``block`` samples.

    A trailing partial block is discarded rather than averaged over fewer
    samples, because an unequal-length block carries a different variance and
    silently biases :func:`batch_variance` low.
    """
    a = np.asarray(values, dtype=float)
    b = int(block)
    if b < 1:
        raise ValueError("block must be at least one sample")
    n = a.shape[0] // b
    if n < 1:
        raise ValueError(f"block {b} longer than the record ({a.shape[0]})")
    trimmed = a[: n * b]
    return trimmed.reshape(n, b, *a.shape[1:]).mean(axis=1)


def batch_variance(
    values: Array, windows: Array, dt: float = 1.0
) -> dict[str, Floats]:
    r"""Variance of a time average as a function of the averaging window.

    For each window :math:`T` in ``windows`` (in time units), splits the record
    into non-overlapping blocks of that length and returns the variance across
    block means. Three arrays come back, keyed ``window``, ``variance``,
    ``blocks``, plus

    .. math:: \tau_{\rm eff}(T) = \frac{T\,\operatorname{var}(\bar A_T)}
                                       {\operatorname{var}(A)},

    the **effective decorrelation time implied by the measurement itself**. For
    a record whose autocovariance is absolutely integrable,
    :math:`\tau_{\rm eff}` plateaus at
    :math:`\tau_{\rm int} = \int_{-\infty}^{\infty}\rho(s)\,ds` and the time
    average converges at the Monte Carlo rate :math:`T^{-1/2}`.

    **It does not always plateau, and that is the interesting case.** Measured
    on Lorenz 63 at :math:`\Delta t = 0.01` over 60,000 time units:
    :math:`\tau_{\rm eff}` for :math:`x` settles at 1.008 TU, matching
    :func:`autocorrelation_time`'s 0.992 to 1.7 %, while for :math:`z` it
    *falls* from 0.126 at :math:`T=0.5` to 0.0098 at :math:`T=500` -- the
    average converges far faster than :math:`T^{-1/2}` over that range, because
    :math:`z` differs from a total time derivative by a constant
    (:func:`lorenz63_moment_residuals`) and its autocovariance integrates to
    nearly zero. A single :math:`\tau` is then not a property of the system but
    of the window, and quoting one is how a run gets declared converged when it
    is not.

    The variance of an estimated variance is :math:`2/(n_{\rm blocks}-1)` in
    relative terms, so a window that leaves fewer than about 30 blocks is
    reported but should not be fitted.
    """
    a = np.asarray(values, dtype=float).ravel()
    total_var = float(a.var())
    ws, variances, counts, taus = [], [], [], []
    for w in np.atleast_1d(np.asarray(windows, dtype=float)):
        block = int(round(float(w) / float(dt)))
        if block < 1 or block > a.size:
            continue
        means = block_means(a, block)
        if means.size < 2:
            continue
        v = float(means.var(ddof=1))
        span = block * float(dt)
        ws.append(span)
        variances.append(v)
        counts.append(float(means.size))
        taus.append(span * v / total_var if total_var > 0 else np.nan)
    return {
        "window": np.asarray(ws),
        "variance": np.asarray(variances),
        "blocks": np.asarray(counts),
        "tau_eff": np.asarray(taus),
    }


def autocorrelation_time(
    series: Array, dt: float = 1.0, max_lag: int | None = None
) -> float:
    r"""Integrated autocorrelation time
    :math:`\tau_{\rm int} = \Delta t\,(1 + 2\sum_{k\ge1}\rho_k)`.

    The sum is truncated at the first non-positive :math:`\rho_k` -- the
    initial-positive-sequence rule -- which is the same convention as
    :func:`chaoslib.nonstationary.effective_sample_size`, and the two are exact
    reciprocals: :math:`\tau_{\rm int} = n\,\Delta t / n_{\rm eff}`. The tests
    assert that to machine precision, so a change to one convention cannot
    silently diverge from the other.

    It returns a time, so :math:`\operatorname{var}(\bar A_T) =
    \tau_{\rm int}\operatorname{var}(A)/T` and the number is directly
    comparable with :func:`batch_variance`'s ``tau_eff``.

    **The truncation is the weakness, and it is one-sided.** Stopping at the
    first negative lag discards exactly the negative lobes that an oscillating
    record uses to cancel its own variance, so the estimator is biased *high*
    whenever the signal oscillates. On Lorenz 63 it returns 0.233 TU for
    :math:`z` against a measured 0.0098 -- an error bar too large by a factor
    of five. For a noisy bistable record the same rule fails the other way
    (chapter 30, section 5): it sees only the within-well wobble, whose
    correlation time is 2.1 time units, and reports an error bar 15 times too
    small for a climatology whose ergodic time is 279.
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        raise ValueError("need at least four finite values")
    x = x - x.mean()
    var = float(np.mean(x**2))
    if var <= 0.0:
        return float(dt)
    top = n // 4 if max_lag is None else min(int(max_lag), n - 1)
    total = 0.0
    for k in range(1, max(top, 1) + 1):
        rho = float(np.mean(x[:-k] * x[k:]) / var)
        if rho <= 0.0:
            break
        total += rho
    return float(dt) * (1.0 + 2.0 * total)


def sampling_error(variance: float, tau: float, duration: float) -> float:
    r"""Standard error of a time average,
    :math:`\sqrt{\tau_{\rm int}\operatorname{var}(A)/T}`.

    The Monte Carlo law with the correlation time doing the work of the sample
    size: a record of length :math:`T` is worth :math:`T/\tau_{\rm int}`
    independent draws. Halving the error costs four times the integration,
    whatever the model.
    """
    t = float(duration)
    if t <= 0.0:
        return float("nan")
    return float(np.sqrt(float(tau) * float(variance) / t))


def budget_variance(
    variance: float,
    tau: float,
    budget: float,
    members: int,
    spin_up: float = 0.0,
) -> float:
    r"""Variance of a climatology estimated by ``members`` runs at fixed cost.

    A total integration budget :math:`C` is split into :math:`M` runs of
    :math:`C/M` each, of which the first :math:`T_s` must be discarded as
    spin-up. The usable record is :math:`C - M T_s`, so

    .. math::
        \operatorname{var}(\bar A) = \frac{\tau_{\rm int}\operatorname{var}(A)}
                                          {C - M T_s},

    which **rises monotonically with** :math:`M`. For an ergodic system at
    fixed cost the optimum is one long run, and the ensemble's only cost is
    the spin-up paid :math:`M` times over.

    Returns ``inf`` when the budget cannot cover the spin-ups.

    Measured on Lorenz 63 with :math:`C = 2000` and :math:`T_s = 20`, over
    2,000 repetitions per member count, the variance of :math:`\bar x` rose by
    1.073, 1.058, 1.121, 1.347 and 1.920 for :math:`M = 2, 5, 10, 25, 50`
    against the 1.010, 1.042, 1.100, 1.320 and 1.980 predicted: the spin-up
    accounting is the whole of the effect, to within 6 %.

    Two caveats. The formula assumes each segment is long compared with
    :math:`\tau_{\rm int}`, which at :math:`M = 50` means 20 time units
    against 1 -- shorten the segments further and :math:`\tau_{\rm eff}`
    itself grows, and the penalty exceeds this prediction. And the conclusion
    inverts entirely when the ergodic time exceeds the run length, which is
    section 5.
    """
    usable = float(budget) - int(members) * float(spin_up)
    if usable <= 0.0:
        return float("inf")
    return float(float(tau) * float(variance) / usable)


def budget_penalty(budget: float, members: int, spin_up: float) -> float:
    r"""Variance inflation from splitting a budget, relative to one long run:
    :math:`(C - T_s)/(C - M T_s)`.

    Parameter-free -- no correlation time, no variance -- so a measured
    inflation can be compared against it directly, and any excess is
    attributable to something other than the spin-up accounting.
    """
    single = float(budget) - float(spin_up)
    split = float(budget) - int(members) * float(spin_up)
    if split <= 0.0 or single <= 0.0:
        return float("inf")
    return float(single / split)


def power_law_fit(x: Array, y: Array) -> dict[str, float]:
    r"""Ordinary least squares of :math:`\ln y` on :math:`\ln x`.

    Returns ``exponent``, ``prefactor`` and ``r2`` for :math:`y = c\,x^{p}`.
    Non-positive or non-finite pairs are dropped, because a single zero makes
    the whole fit ``nan`` and the usual symptom is a convergence exponent
    quietly reported as ``nan`` in a table.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    a, b = np.log(a[keep]), np.log(b[keep])
    if a.size < 2:
        return {"exponent": float("nan"), "prefactor": float("nan"), "r2": float("nan")}
    slope, intercept = np.polyfit(a, b, 1)
    fit = slope * a + intercept
    ss_res = float(np.sum((b - fit) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "exponent": float(slope),
        "prefactor": float(np.exp(intercept)),
        "r2": float(r2),
    }


# =========================================================================
# measures, and distances between them
# =========================================================================
def empirical_measure(samples: Array, edges: Array) -> dict[str, Floats]:
    r"""The empirical measure of ``samples`` on the bins given by ``edges``.

    Returns ``probability`` (summing to one over the bins, so samples outside
    the range are dropped and the rest renormalised), ``density``
    (probability per unit length) and ``centre``.

    The distinction is the subject of section 1. For a measure with a density
    the two carry the same information and the density is the natural object.
    For the measure on a strange attractor there **is** no density -- the
    measure is singular, concentrated on a set of dimension 2.06 in a
    three-dimensional space (chapter 8) -- so the returned ``density`` depends
    on the bin width and does not converge as the bins shrink, while
    ``probability`` on a fixed partition converges perfectly well. A
    climatological PDF is always the second thing wearing the first thing's
    name.
    """
    x = np.asarray(samples, dtype=float).ravel()
    e = np.asarray(edges, dtype=float).ravel()
    counts, _ = np.histogram(x[np.isfinite(x)], bins=e)
    total = counts.sum()
    prob = counts / total if total > 0 else counts.astype(float)
    widths = np.diff(e)
    return {
        "probability": prob,
        "density": prob / widths,
        "centre": 0.5 * (e[1:] + e[:-1]),
        "width": widths,
    }


def wasserstein1(a: Array, b: Array) -> float:
    r"""1-Wasserstein (earth mover's) distance between two sets of samples,

    .. math:: W_1(\nu_a,\nu_b) = \int_{-\infty}^{\infty}|F_a(s)-F_b(s)|\,ds ,

    computed exactly for the two empirical distributions, with no binning.

    It is used here rather than a relative entropy for one reason: two
    *singular* measures -- a long run and a scattered ensemble on the same
    strange attractor -- generically have disjoint support, so
    :math:`D(p\|q)` is infinite and a histogram-based estimate of it reports
    whatever the bin width was. :math:`W_1` needs no density, is finite between
    any two measures with a first moment, and carries the units of the
    variable, so a value can be read against the spread of the variable itself.
    """
    sa = np.sort(np.asarray(a, dtype=float).ravel())
    sb = np.sort(np.asarray(b, dtype=float).ravel())
    sa = sa[np.isfinite(sa)]
    sb = sb[np.isfinite(sb)]
    if sa.size == 0 or sb.size == 0:
        raise ValueError("both samples must be non-empty")
    grid = np.sort(np.concatenate([sa, sb]))
    widths = np.diff(grid)
    cdf_a = np.searchsorted(sa, grid[:-1], side="right") / sa.size
    cdf_b = np.searchsorted(sb, grid[:-1], side="right") / sb.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * widths))


def total_variation(p: Array, q: Array) -> float:
    r"""Total-variation distance :math:`\tfrac12\sum_i|p_i - q_i|` between two
    binned measures, in :math:`[0, 1]`.

    Bin-dependent by construction -- unlike :func:`wasserstein1` -- which is
    exactly why it is the wrong tool for comparing two samples of a singular
    measure and the right one for comparing two *coarse-grained* climatologies
    on a partition that has been fixed in advance.
    """
    a = np.asarray(p, dtype=float).ravel()
    b = np.asarray(q, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    return float(0.5 * np.sum(np.abs(a - b)))


# =========================================================================
# invariant measures in closed form
# =========================================================================
def logistic_invariant_density(x: Array) -> Floats:
    r"""The invariant density of the logistic map at :math:`r = 4`,

    .. math:: p(x) = \frac{1}{\pi\sqrt{x(1-x)}},\qquad 0 < x < 1 .

    The arcsine law. It follows from the conjugacy
    :math:`x = \sin^2(\pi\theta/2)` to the doubling map on :math:`[0,1]`, whose
    invariant density is uniform; the density is the Jacobian of that change of
    variable. Its moments are exact -- :math:`\langle x\rangle = 1/2`,
    :math:`\langle x^2\rangle = 3/8`, :math:`\operatorname{var} = 1/8` -- which
    is what makes this the one system in the book where the convergence of a
    time average can be measured against truth rather than against a longer
    run.

    Note where the mass is: the density **diverges** at both ends, so the
    orbit spends most of its time near 0 and 1 and comparatively little in the
    middle. A uniform prior on the state of this system is badly wrong, and
    nothing about the equation announces that.
    """
    v = np.asarray(x, dtype=float)
    out = np.full(v.shape, np.nan)
    inside = (v > 0.0) & (v < 1.0)
    out[inside] = 1.0 / (np.pi * np.sqrt(v[inside] * (1.0 - v[inside])))
    return out


def logistic_invariant_cdf(x: Array) -> Floats:
    r"""Distribution function of the arcsine law,
    :math:`F(x) = \tfrac{2}{\pi}\arcsin\sqrt{x}`, clipped to :math:`[0,1]`."""
    v = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return (2.0 / np.pi) * np.arcsin(np.sqrt(v))


def logistic_invariant_quantile(u: Array) -> Floats:
    r"""Inverse of :func:`logistic_invariant_cdf`,
    :math:`F^{-1}(u) = \sin^2(\pi u/2)`.

    Inverse-transform sampling from it gives an *exact* draw from the invariant
    measure, which is the reference the measured orbit is compared against.
    """
    v = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return np.sin(0.5 * np.pi * v) ** 2


def boltzmann_density(x: Array, potential: Array, noise_std: float) -> Floats:
    r"""Stationary density of :math:`dx = -V'(x)\,dt + \sigma\,dW`, normalised
    on the supplied grid:

    .. math:: p(x) = Z^{-1}\exp\!\left(-\frac{2V(x)}{\sigma^2}\right).

    Exact for a one-dimensional gradient flow with additive noise, and the
    invariant measure of the double well that chapters 27 and 30 use. It pins
    the drift and the integrator's noise convention **together**: a factor of
    two lost in either one shows up as a visibly wrong width.

    ``potential`` is :math:`V` evaluated on the same grid as ``x``; the
    normalisation is by trapezoid over that grid, so it must cover the support
    to the accuracy wanted.
    """
    v = np.asarray(potential, dtype=float)
    grid = np.asarray(x, dtype=float)
    sigma = float(noise_std)
    if sigma <= 0.0:
        raise ValueError("noise_std must be positive")
    shifted = -2.0 * (v - v.min()) / sigma**2
    weight = np.exp(shifted)
    norm = float(np.trapezoid(weight, grid))
    return weight / norm


# =========================================================================
# ergodic but not mixing: the rotation
# =========================================================================
def rotation_orbit(alpha: float, n: int, x0: float = 0.0) -> Floats:
    r"""The orbit :math:`x_k = \{x_0 + k\alpha\}` of the circle rotation,
    :math:`k = 1 \ldots n`.

    For irrational :math:`\alpha` this is **uniquely ergodic**: every orbit,
    without exception, equidistributes on :math:`[0,1)`, and time averages of
    continuous observables converge to the uniform average. It is also
    completely non-chaotic -- :math:`\lambda_1 = 0`, two nearby points stay
    exactly as far apart forever -- and not mixing.

    So ergodicity is not chaos, and the two have opposite consequences for
    sampling. Because the orbit is *deliberately* uniform rather than
    independent, its averages converge like :math:`n^{-1}` up to a logarithm
    (:func:`star_discrepancy`) instead of the :math:`n^{-1/2}` of a chaotic or
    random sample: at :math:`n = 1000` the golden-ratio rotation estimates
    :math:`\langle x\rangle` to :math:`1\times10^{-5}` where an i.i.d. sample
    of the same size manages :math:`9\times10^{-3}`. Quasi-periodic components
    of a climate -- the seasonal cycle, the diurnal cycle -- are cheap to
    average over for exactly this reason; the chaotic part is not.

    For *rational* :math:`\alpha = p/q` the orbit is periodic with :math:`q`
    points and is not ergodic at all, which is the limiting case the tests use.
    """
    k = np.arange(1, int(n) + 1, dtype=float)
    return np.mod(float(x0) + float(alpha) * k, 1.0)


def star_discrepancy(points: Array) -> float:
    r"""Star discrepancy of a point set in :math:`[0,1)`,

    .. math::
        D_N^* = \sup_{t\in[0,1]}\left|\frac{\#\{x_i < t\}}{N} - t\right|
              = \frac{1}{2N} + \max_i\left|x_{(i)} - \frac{2i-1}{2N}\right| ,

    the second form exact for the sorted sample and what is computed.

    This is the sampling error of the *worst* indicator observable, so it
    bounds the error of averaging any function of bounded variation
    (Koksma's inequality). For an i.i.d. sample it decays like
    :math:`N^{-1/2}`; for a rotation by a badly approximable irrational like
    :math:`(\sqrt5-1)/2` it decays like :math:`\log N / N`.
    """
    x = np.sort(np.asarray(points, dtype=float).ravel())
    n = x.size
    if n == 0:
        raise ValueError("need at least one point")
    i = np.arange(1, n + 1, dtype=float)
    return float(1.0 / (2 * n) + np.max(np.abs(x - (2 * i - 1) / (2 * n))))


def occupancy(series: Array, threshold: float = 0.0) -> float:
    """Fraction of samples above ``threshold`` -- the measure of one regime.

    For a symmetric double well with a symmetric noise this is exactly 1/2 on
    the invariant measure, whatever the barrier height, so any departure in a
    finite run is pure sampling error with a known answer to compare against.
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x > float(threshold)))


# =========================================================================
# exact identities the invariant measure satisfies
# =========================================================================
def lorenz63_moment_residuals(
    traj: Array,
    dt: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> dict[str, float]:
    r"""Four exact moment identities of the Lorenz 63 invariant measure, and
    how far a finite run is from each.

    For any bounded :math:`B(x)`, the time average of :math:`\dot B` is a pure
    boundary term, :math:`\overline{\dot B}_T = (B_T - B_0)/T`, which vanishes
    as :math:`T\to\infty` **without any appeal to ergodicity or mixing**.
    Applying that to :math:`x^2/2`, :math:`z`, :math:`z^2/2` and :math:`y^2/2`
    gives, on the invariant measure,

    .. math::
        \langle xy\rangle = \langle x^2\rangle, \qquad
        \langle xy\rangle = \beta\langle z\rangle, \qquad
        \langle xyz\rangle = \beta\langle z^2\rangle, \qquad
        \rho\langle xy\rangle = \langle y^2\rangle + \langle xyz\rangle .

    Two things come out of this, and the returned dictionary separates them.

    ``finite_*`` are the identities **with** their boundary terms, e.g.

    .. math::
        \overline{xy}_T - \overline{x^2}_T
            = \frac{x_T^2 - x_0^2}{2\sigma T},

    which hold for *every* :math:`T`, exactly, and are therefore a check on the
    integrator rather than on convergence. Measured with the trapezoidal rule
    on a 20 time-unit run at :math:`\Delta t = 0.005`, they close to
    :math:`7\times10^{-7}` against terms of order 60.

    ``limit_*`` are the same identities with the boundary term dropped -- what
    a modeller would actually assert about the climate. These close only as the
    run lengthens, and their residuals are a **convergence diagnostic that
    needs no reference run**: they say, from inside a single trajectory,
    whether it is yet long enough to be quoting climatology from. On a run of
    2,000 time units, :math:`\overline{x^2}/\beta` and :math:`\bar z` agree to
    :math:`4\times10^{-5}` relative, while :math:`\bar x` -- whose exact value
    is :math:`0` by the symmetry :math:`(x,y,z)\to(-x,-y,z)` -- is still
    :math:`-0.37`.

    ``mean_x`` is returned alongside for that comparison: it is the one
    quantity here whose true value is known exactly *and* the slowest of them
    all to converge.
    """
    a = np.asarray(traj, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected a (time, 3) trajectory, got {a.shape}")
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    span = float(dt) * (a.shape[0] - 1)
    if span <= 0.0:
        raise ValueError("trajectory must span a positive time")

    m_xy = float(time_average(x * y, dt))
    m_x2 = float(time_average(x * x, dt))
    m_y2 = float(time_average(y * y, dt))
    m_z = float(time_average(z, dt))
    m_z2 = float(time_average(z * z, dt))
    m_xyz = float(time_average(x * y * z, dt))

    b_x2 = (x[-1] ** 2 - x[0] ** 2) / (2.0 * float(sigma) * span)
    b_z = (z[-1] - z[0]) / span
    b_z2 = (z[-1] ** 2 - z[0] ** 2) / (2.0 * span)
    b_y2 = (y[-1] ** 2 - y[0] ** 2) / (2.0 * span)

    limit_x2 = m_xy - m_x2
    limit_z = m_xy - float(beta) * m_z
    limit_z2 = m_xyz - float(beta) * m_z2
    limit_y2 = float(rho) * m_xy - m_y2 - m_xyz

    return {
        "finite_x2": float(limit_x2 - b_x2),
        "finite_z": float(limit_z - b_z),
        "finite_z2": float(limit_z2 - b_z2),
        "finite_y2": float(limit_y2 - b_y2),
        "limit_x2": float(limit_x2),
        "limit_z": float(limit_z),
        "limit_z2": float(limit_z2),
        "limit_y2": float(limit_y2),
        "mean_x": float(time_average(x, dt)),
        "mean_z": float(m_z),
        "mean_x2": float(m_x2),
        "duration": span,
    }
