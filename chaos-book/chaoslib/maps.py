r"""Bifurcations, cascades and the routes to chaos in one-dimensional maps.

A map :math:`x_{n+1} = f_r(x_n)` is the cheapest system in the book, and the
only one whose route into chaos can be drawn in full: the entire
period-doubling cascade fits in one figure. What makes it worth a chapter
rather than a footnote is that the *numbers* describing that route are
**universal** -- shared by every smooth unimodal map, and by physical systems
that look nothing like a map.

Three things are computed here.

**The cascade.** The parameter values :math:`R_n` at which the
:math:`2^n`-cycle is *superstable* -- meaning the cycle contains the critical
point :math:`x_c` where :math:`f'(x_c) = 0`, so the cycle multiplier is exactly
zero. Their spacings shrink geometrically, and

.. math::
    \delta = \lim_{n\to\infty}\frac{R_{n-1} - R_{n-2}}{R_n - R_{n-1}}
           = 4.669\,201\,609\ldots

is the Feigenbaum constant. Superstable points, not bifurcation points, are what
this module locates: see :func:`superstable_cascade` for why that choice is a
numerical necessity rather than a preference.

**The Lyapunov exponent.** For a 1-D map the whole of chapter 7's machinery
collapses to :math:`\lambda(r) = \langle \ln|f'_r(x)|\rangle` along the orbit --
no QR decomposition, no tangent linear model, one line. Plotted against
:math:`r` it shows what the bifurcation diagram only hints at: that "chaotic" is
a property of a parameter *value*, and that the chaotic and periodic sets are
interleaved at every scale.

**Intermittency.** Just below the tangent bifurcation that creates the
3-cycle, the orbit spends long stretches almost periodic before bursting into
chaos and returning, with no change in any parameter. The mean laminar stretch
diverges as :math:`(r_c - r)^{-1/2}`, which is the cleanest available model of a
regime that persists for a while and then does not.

Conventions follow ``NOTATION.md``: :math:`r` is the map parameter, :math:`f_r`
the map, :math:`x_c` its critical point, :math:`\lambda` the Lyapunov exponent
in nats per iteration.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import brentq

Array = np.ndarray
MapFn = Callable[..., Array]

__all__ = [
    "map_orbit",
    "cobweb_path",
    "bifurcation_points",
    "map_lyapunov_exponent",
    "iterate_n",
    "superstable_cascade",
    "feigenbaum_ratios",
    "cycle_multiplier",
    "period_three_threshold",
    "laminar_phases",
    "FEIGENBAUM_DELTA",
]

#: The Feigenbaum constant, to the precision quoted in the literature.
#: *[citation needed: Feigenbaum (1978); Briggs (1991) for the high-precision value]*
FEIGENBAUM_DELTA = 4.669201609102990


# --------------------------------------------------------------------------
# Orbits
# --------------------------------------------------------------------------
def map_orbit(
    fmap: MapFn,
    x0: float | Array,
    n_iter: int,
    n_discard: int = 0,
    **params: float,
) -> Array:
    r"""Iterate ``fmap`` and return the orbit :math:`x_0, x_1, \ldots`.

    ``x0`` may be an array, in which case every entry is advanced in lockstep
    and the result has shape ``(n_iter + 1, *x0.shape)``. That vectorisation is
    what makes a live bifurcation diagram affordable in the browser: a thousand
    parameter values advance in a thousand-element NumPy operation rather than a
    thousand Python loops.

    ``n_discard`` transient iterations are run first and not returned.
    """
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(int(n_discard)):
        x = np.asarray(fmap(x, **params), dtype=float)
    out = np.empty((int(n_iter) + 1, *x.shape), dtype=float)
    out[0] = x
    for i in range(1, int(n_iter) + 1):
        x = np.asarray(fmap(x, **params), dtype=float)
        out[i] = x
    return out if x.shape else out.reshape(-1)


def cobweb_path(
    fmap: MapFn, x0: float, n_steps: int, **params: float
) -> tuple[Array, Array]:
    r"""Vertices of the cobweb (staircase) construction, ready to plot.

    The construction alternates: go vertically from :math:`(x_n, x_n)` to
    :math:`(x_n, f(x_n))`, then horizontally to :math:`(f(x_n), f(x_n))` on the
    diagonal. Returns ``(xs, ys)`` for a single polyline of
    :math:`2n_{\text{steps}} + 1` vertices.

    Reading a cobweb is reading a stability calculation: the staircase spirals
    *into* a fixed point when :math:`|f'(x^*)| < 1` and *away* from it when
    :math:`|f'(x^*)| > 1`, which is the whole content of linear stability
    analysis done graphically.
    """
    x = float(x0)
    xs = [x]
    ys = [x]
    for _ in range(int(n_steps)):
        y = float(np.asarray(fmap(x, **params)))
        xs.extend([x, y])
        ys.extend([y, y])
        x = y
    return np.asarray(xs), np.asarray(ys)


# --------------------------------------------------------------------------
# The bifurcation diagram and the Lyapunov exponent
# --------------------------------------------------------------------------
def bifurcation_points(
    fmap: MapFn,
    r_values: Array,
    param: str = "r",
    x0: float = 0.4,
    n_discard: int = 600,
    n_keep: int = 250,
) -> tuple[Array, Array]:
    r"""Attractor samples for each parameter value, flattened for scattering.

    Returns ``(r_flat, x_flat)``, each of length ``len(r_values) * n_keep``,
    suitable for a single scatter call with a small marker.

    All parameter values are advanced simultaneously, so the cost is
    ``n_discard + n_keep`` vector operations regardless of how finely
    :math:`r` is sampled.

    ``n_discard`` matters more than it looks. Near the accumulation point
    :math:`r_\infty` and just inside a periodic window, convergence onto the
    attractor is slow, and too small a value paints transient smear that is
    easily mistaken for chaos.
    """
    r = np.asarray(r_values, dtype=float)
    x = np.full(r.shape, float(x0))
    for _ in range(int(n_discard)):
        x = np.asarray(fmap(x, **{param: r}), dtype=float)
    keep = int(n_keep)
    xs = np.empty((keep, r.size), dtype=float)
    for i in range(keep):
        x = np.asarray(fmap(x, **{param: r}), dtype=float)
        xs[i] = x
    return np.repeat(r, keep), xs.T.reshape(-1)


def map_lyapunov_exponent(
    fmap: MapFn,
    dfmap: MapFn,
    r_values: Array,
    param: str = "r",
    x0: float = 0.4,
    n_discard: int = 1000,
    n_iter: int = 3000,
    floor: float = -20.0,
) -> Array:
    r"""Lyapunov exponent of a 1-D map, in nats per iteration.

    .. math::
        \lambda(r) = \lim_{N\to\infty}\frac{1}{N}\sum_{n=0}^{N-1}
                     \ln\bigl|f'_r(x_n)\bigr|

    This is chapter 7's definition with the tangent space collapsed to one
    dimension: :math:`\ln|f'|` *is* the local stretching rate, so no
    re-orthonormalisation is needed and no QR decomposition appears.

    Exact values worth checking an implementation against: :math:`\lambda = \ln 2`
    at :math:`r = 4` for the logistic map (the map is then conjugate to a full
    binary shift), :math:`\lambda = 0` at every bifurcation point, and
    :math:`\lambda = -\infty` at every superstable parameter, where the orbit
    hits :math:`f' = 0` exactly.

    That last case is a genuine :math:`-\infty`, not a numerical artefact, so
    the per-iteration logarithm is clipped at ``floor`` to keep the returned
    array finite and plottable. Reduce ``floor`` if you need to resolve deep
    superstable dips; set it to ``-np.inf`` to let them diverge.
    """
    r = np.asarray(r_values, dtype=float)
    x = np.full(r.shape, float(x0))
    for _ in range(int(n_discard)):
        x = np.asarray(fmap(x, **{param: r}), dtype=float)
    total = np.zeros(r.shape, dtype=float)
    for _ in range(int(n_iter)):
        slope = np.abs(np.asarray(dfmap(x, **{param: r}), dtype=float))
        with np.errstate(divide="ignore"):
            total += np.maximum(np.log(slope), floor)
        x = np.asarray(fmap(x, **{param: r}), dtype=float)
    return total / int(n_iter)


# --------------------------------------------------------------------------
# The period-doubling cascade
# --------------------------------------------------------------------------
def iterate_n(fmap: MapFn, x: float | Array, n: int, **params: float) -> Array:
    r"""Apply ``fmap`` exactly ``n`` times: the :math:`n`-th iterate :math:`f^n(x)`."""
    out = np.asarray(x, dtype=float)
    for _ in range(int(n)):
        out = np.asarray(fmap(out, **params), dtype=float)
    return out


def superstable_cascade(
    fmap: MapFn,
    x_critical: float,
    r_lo: float,
    r_hi: float,
    n_max: int = 8,
    param: str = "r",
    delta_guess: float = FEIGENBAUM_DELTA,
    scan: int = 60,
    base_period: int = 1,
) -> Array:
    r"""Parameters :math:`R_0 \ldots R_{n_{\max}}` at which the
    :math:`p\,2^n`-cycle is superstable, for base period :math:`p`.

    A cycle is *superstable* when it contains the critical point
    :math:`x_c`. Since :math:`f'(x_c) = 0`, the cycle multiplier
    :math:`\prod_i f'(x_i)` is then exactly zero, and :math:`R_n` is a solution
    of the smooth scalar equation

    .. math::
        g_n(r) \equiv f_r^{p\,2^n}(x_c) - x_c = 0 .

    ``base_period`` selects which cascade is followed. The default ``p = 1`` is
    the main cascade :math:`1, 2, 4, 8, \ldots`. Setting ``p = 3`` with
    ``r_lo``/``r_hi`` inside the period-3 window follows that window's *own*
    cascade :math:`3, 6, 12, 24, \ldots`, which is how the self-similarity of the
    bifurcation diagram is turned from an observation into a measurement: the
    sub-cascade has the same :math:`\delta`. That is the content of the
    renormalisation argument, and it holds for every periodic window.

    **Why superstable points rather than bifurcation points.** The obvious
    target is the bifurcation value :math:`b_n` where the :math:`2^{n-1}`-cycle
    loses stability, i.e. where its multiplier reaches :math:`-1`. Locating that
    requires first finding the cycle -- itself a root-find in a basin that
    shrinks like :math:`\delta^{-n}` -- and then detecting a *marginal*
    condition, which is ill-conditioned by construction. Superstable points need
    neither: :math:`x_c` is known in closed form, so :math:`g_n` is evaluated by
    plain iteration. The two sequences interleave
    (:math:`b_n < R_n < b_{n+1}`) and have the *same* limit and the *same*
    :math:`\delta`, so nothing is lost.

    **Bracketing is the whole difficulty.** :math:`g_n` vanishes at every
    :math:`R_k` with :math:`k \le n`, because a :math:`2^k`-cycle is also a
    :math:`2^n`-cycle traversed repeatedly -- in particular
    :math:`g_n(R_{n-1}) = 0`. An expanding bracket started at :math:`R_{n-1}`
    therefore converges straight back onto the root already known. What is used
    instead: scan upward from just above :math:`R_{n-1}` and bracket the
    **first** sign change, whose location is predicted to within a few percent
    by extrapolating the previous spacing with ``delta_guess``. There is no root
    of :math:`g_n` strictly between :math:`R_{n-1}` and :math:`R_n`, so the
    first sign change is the one wanted.

    ``r_hi`` must not exceed the accumulation point :math:`r_\infty` by much:
    above it, :math:`g_n` acquires further roots inside the periodic windows of
    the chaotic band.

    Returns however many levels were found, which may be fewer than
    ``n_max + 1``. Double precision runs out around :math:`n = 8`: the spacings
    shrink by :math:`\delta` each level while :math:`g_n` needs :math:`2^n`
    compositions, so the condition number grows like :math:`\delta^n`.
    """
    found: list[float] = []
    for n in range(int(n_max) + 1):
        period = int(base_period) * 2**n

        def g(r: float, _p: int = period) -> float:
            return float(iterate_n(fmap, x_critical, _p, **{param: r}) - x_critical)

        if n == 0:
            lo, hi = float(r_lo), float(r_hi)
        else:
            prev = found[-1]
            gap = (
                (found[-1] - found[-2]) / float(delta_guess)
                if len(found) >= 2
                else float(r_hi) - prev
            )
            lo = prev + 1e-6 * abs(gap)
            hi = min(float(r_hi), prev + 3.0 * abs(gap))
            if not hi > lo:
                break

        grid = np.linspace(lo, hi, int(scan))
        values = np.array([g(float(r)) for r in grid])
        crossings = np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0.0)[0]
        if crossings.size == 0:
            break
        i = int(crossings[0])
        found.append(
            float(brentq(g, grid[i], grid[i + 1], xtol=1e-14, rtol=8.9e-16))
        )
    return np.asarray(found, dtype=float)


def feigenbaum_ratios(cascade: Sequence[float] | Array) -> Array:
    r"""Successive spacing ratios :math:`(R_{n-1}-R_{n-2})/(R_n-R_{n-1})`.

    The sequence converges to :math:`\delta`; its last entry is the best
    available estimate from a given cascade. Convergence is itself geometric, so
    each extra level gains roughly one significant figure.
    """
    r = np.asarray(cascade, dtype=float)
    if r.size < 3:
        raise ValueError("need at least three cascade parameters for one ratio")
    gaps = np.diff(r)
    return gaps[:-1] / gaps[1:]


def cycle_multiplier(
    fmap: MapFn, dfmap: MapFn, x_start: float, period: int, **params: float
) -> float:
    r"""Multiplier :math:`\prod_{i=0}^{p-1} f'(x_i)` of a :math:`p`-cycle.

    The cycle is stable for :math:`|{\cdot}| < 1`, superstable at
    :math:`0`, and loses stability through period doubling at :math:`-1` (and
    through a tangent bifurcation at :math:`+1`). ``x_start`` must lie on the
    cycle; the product is invariant to which point of it is used.
    """
    x = float(x_start)
    product = 1.0
    for _ in range(int(period)):
        product *= float(np.asarray(dfmap(x, **params)))
        x = float(np.asarray(fmap(x, **params)))
    return product


# --------------------------------------------------------------------------
# Intermittency
# --------------------------------------------------------------------------
def period_three_threshold() -> float:
    r"""The exact logistic-map parameter at which the 3-cycle is born.

    .. math:: r_c = 1 + 2\sqrt{2} = 3.828427\ldots

    At :math:`r_c` the third iterate :math:`f^3` is *tangent* to the diagonal at
    three points simultaneously -- :math:`f^3(x) = x` with
    :math:`(f^3)'(x) = +1` -- so a stable and an unstable 3-cycle appear
    together out of nothing. That is a tangent (saddle-node) bifurcation, a
    different mechanism from the period doubling of :func:`superstable_cascade`,
    and it is the one that produces intermittency.
    """
    return 1.0 + 2.0 * np.sqrt(2.0)


def laminar_phases(
    fmap: MapFn,
    r: float,
    period: int = 3,
    tolerance: float = 0.02,
    n_iter: int = 200_000,
    n_discard: int = 5_000,
    x0: float = 0.4,
    min_length: int = 4,
    param: str = "r",
) -> Array:
    r"""Lengths of the near-periodic (laminar) stretches of an orbit.

    An iterate is counted as laminar when
    :math:`|f^{p}(x_n) - x_n| < \text{tolerance}`: the orbit is close to
    returning to itself after :math:`p` steps, so it is shadowing a cycle that
    does not quite exist. Runs shorter than ``min_length`` are dropped as noise
    rather than laminar phases.

    Below :math:`r_c` from :func:`period_three_threshold` the 3-cycle has not
    been born, but :math:`f^3` still passes very close to the diagonal, and an
    orbit entering that narrow channel takes many iterations to traverse it.
    Type-I intermittency theory gives a mean laminar length diverging as
    :math:`\langle L\rangle \sim (r_c - r)^{-1/2}` *[citation needed: Pomeau &
    Manneville (1980)]*, the square root coming from the local normal form
    :math:`x \mapsto x + a x^2 + \epsilon`.

    The behaviour is worth the attention it gets in this book because it is a
    regime that ends **without any parameter changing** -- and because the
    diverging timescale as :math:`r \to r_c^-` is the mechanism behind
    critical-slowing-down early-warning indicators.
    """
    x = float(x0)
    kwargs = {param: float(r)}
    for _ in range(int(n_discard)):
        x = float(np.asarray(fmap(x, **kwargs)))

    lengths: list[int] = []
    run = 0
    for _ in range(int(n_iter)):
        near = abs(float(iterate_n(fmap, x, int(period), **kwargs)) - x) < tolerance
        if near:
            run += 1
        else:
            if run >= int(min_length):
                lengths.append(run)
            run = 0
        x = float(np.asarray(fmap(x, **kwargs)))
    if run >= int(min_length):
        lengths.append(run)
    return np.asarray(lengths, dtype=float)
