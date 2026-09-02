r"""Fractal dimension of an attractor, measured from a sampled trajectory.

The Kaplan-Yorke dimension in :mod:`chaoslib.lyapunov` is computed from the
Lyapunov spectrum -- i.e. from the *dynamics*. The estimator here works from a
sampled trajectory only, so the two give genuinely independent numbers for the
same attractor. Agreement between them is evidence both are right; for Lorenz 63
both land near 2.06.

Two traps this module is built to keep the reader out of. Neither announces
itself, and -- measured on Lorenz 63 rather than assumed -- **either sign of
error is available**, which is why neither can be corrected for after the fact:

1. **The scaling window.** :math:`C(r) \sim r^{D_2}` only holds over a limited
   range of :math:`r`. On a 4000-sample Lorenz 63 trajectory, fitting only
   radii above 30 % of the attractor diameter -- where :math:`C \to 1` --
   returns **0.19**; fitting only below 0.2 % of it, where the pair counts are
   quantised and erratic, returns **2.51**; fitting the whole available range
   returns 1.92. The window that works, roughly 1--5 % of the diameter, gives
   2.057. So too coarse biases low, too fine biases high, and the
   plausible-looking whole-range fit is the most dangerous of the three. The
   radii and :math:`C(r)` are returned so that a chapter can plot the fit and
   let the reader judge the window rather than trust it.
2. **Temporal correlation.** Consecutive trajectory samples are close together
   because they are consecutive, not because the attractor is dense there. The
   excess pairs appear as a *bump* in :math:`C(r)` at the distance the
   trajectory covers between samples, so the bias depends on where that
   distance falls relative to the fit window. For Lorenz 63 at
   :math:`\Delta t = 0.01` the step is 1.3 % of the diameter -- inside the usual
   window -- and the bump's rising flank steepens the local slope from 1.92 to
   2.22 there, biasing :math:`D_2` **high**: 2.139 with no Theiler window
   against 2.039 with one of 200 samples. Sample densely enough to put the step
   scale *below* the window and the classic opposite bias appears instead, the
   estimator reporting the trajectory's smooth one-dimensional curve. The
   ``theiler`` window excludes pairs closer than a given number of samples in
   time, and is the fix in both cases.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]

__all__ = [
    "correlation_sum",
    "correlation_dimension",
    "local_slopes",
    "fit_dimension",
    "box_occupancy",
    "renyi_dimension",
    "renyi_spectrum",
    "delay_embed",
    "cantor_set",
    "koch_curve",
    "sierpinski_triangle",
    "REFERENCE_DIMENSIONS",
    "REFERENCE_WINDOWS",
    "MIN_BOX_OCCUPANCY",
]

#: Box-counting dimensions of the reference sets below, known in closed form.
#: Estimators are checked against these rather than against each other.
REFERENCE_DIMENSIONS = {
    "cantor": np.log(2.0) / np.log(3.0),        # 0.630930
    "koch": np.log(4.0) / np.log(3.0),          # 1.261860
    "sierpinski": np.log(3.0) / np.log(2.0),    # 1.584963
}

#: Mean points per occupied box below which box counting stops measuring the
#: set and starts measuring the sample. Calibrated on
#: :func:`sierpinski_triangle` with 200,000 points, where the local slope reads
#: 1.582, 1.568, 1.549, 1.451, 1.127 at occupancies of 186, 63, 21, 7.3, 2.8
#: against an exact 1.585 -- gradual degradation that turns sharp below about
#: ten, and reaches zero at one point per box, where the box count equals the
#: sample size and stops responding to the scale at all.
#:
#: This applies to a **random sample** of a set. For a deterministic
#: construction such as :func:`koch_curve`, whose points are the curve's own
#: vertices rather than a sample of its length, occupancy is not the binding
#: constraint -- the construction depth is, since no structure exists below
#: :math:`3^{-\text{depth}}` to be resolved.
MIN_BOX_OCCUPANCY = 10.0

#: Scaling windows, in units of the largest extent, over which box counting
#: recovers each reference set's exact dimension. Found by inspecting local
#: slopes, not by assumption -- see the chapter 8 notebook, which plots them.
#: Every one is a compromise: the Cantor set needs 2.5 decades because its local
#: slope oscillates with period :math:`\ln 3` (lacunarity) rather than sitting
#: on a plateau, so the fit has to average over several periods of the
#: oscillation instead of finding a flat stretch.
REFERENCE_WINDOWS = {
    "cantor": (3e-5, 1e-2),
    "koch": (3e-4, 8e-3),
    "sierpinski": (3e-3, 3e-2),
}


def _prepare(points: Array, max_points: int) -> Array:
    """Subsample uniformly in time so the pair loop stays affordable.

    Uniform (stride) subsampling rather than random: it preserves the time
    ordering that ``theiler`` relies on, and it additionally decorrelates
    neighbouring samples, which is what we want anyway.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[0] > max_points:
        stride = int(np.ceil(pts.shape[0] / max_points))
        pts = pts[::stride]
    return pts


def correlation_sum(
    points: Array,
    radii: Array,
    theiler: int = 0,
    max_points: int = 4000,
) -> Array:
    r"""Grassberger-Procaccia correlation sum :math:`C(r)`.

    .. math::
        C(r) = \frac{2}{N(N-1-2w)}
               \sum_{j > i + w}\Theta\bigl(r - \|x_i - x_j\|\bigr)

    the fraction of state pairs closer than :math:`r`, excluding pairs within
    ``theiler`` samples (:math:`w`) of each other in time.

    Computed row by row and accumulated, rather than by forming the full
    :math:`N \times N` distance matrix: at ``max_points=4000`` that matrix would
    be ~128 MB, which Pyodide will not tolerate. Each row is still fully
    vectorised.
    """
    pts = _prepare(points, max_points)
    radii = np.asarray(radii, dtype=float)
    n = pts.shape[0]
    w = max(0, int(theiler))

    counts = np.zeros(radii.size, dtype=np.int64)
    total = 0
    for i in range(n - 1):
        j0 = i + 1 + w
        if j0 >= n:
            break
        d = np.sqrt(np.sum((pts[j0:] - pts[i]) ** 2, axis=-1))
        d.sort()
        # searchsorted over the sorted row: O(m log m + len(radii) log m), far
        # cheaper than one comparison pass per radius.
        counts += np.searchsorted(d, radii, side="right")
        total += d.size

    if total == 0:
        raise ValueError(
            "no pairs survived the Theiler window; reduce `theiler` or supply "
            "a longer trajectory"
        )
    return counts / total


def local_slopes(radii: Array, c: Array) -> Array:
    r"""Local slope :math:`d\ln C/d\ln r`, the honest diagnostic of scaling.

    A genuine fractal shows a plateau here; the plateau's value is
    :math:`D_2` and its extent is the scaling window. Reporting a single fitted
    number without ever looking at this plot is how wrong dimensions get
    published.
    """
    radii = np.asarray(radii, dtype=float)
    c = np.asarray(c, dtype=float)
    ok = c > 0.0
    if ok.sum() < 3:
        raise ValueError("need at least 3 non-empty radii")
    return np.gradient(np.log(c[ok]), np.log(radii[ok]))


def correlation_dimension(
    points: Array,
    n_radii: int = 32,
    fit_range: tuple[float, float] = (0.008, 0.05),
    theiler: int = 0,
    max_points: int = 4000,
) -> tuple[float, Array, Array]:
    r"""Correlation dimension :math:`D_2`: slope of :math:`\ln C` vs :math:`\ln r`.

    ``fit_range`` gives the scaling window as fractions of the attractor's
    diameter. The default ``(0.008, 0.05)`` is the measured plateau for Lorenz 63
    (local slope 1.94--2.10 there); a different attractor needs a different
    window, found by inspecting :func:`local_slopes` first.

    ``theiler`` excludes temporally adjacent pairs. For a trajectory already
    subsampled to roughly one sample per Lyapunov time, ``0`` is adequate;
    for densely sampled input use a window of order one Lyapunov time in
    samples.

    Returns ``(D2, radii, C)``.
    """
    pts = _prepare(points, max_points)
    diameter = float(np.sqrt(np.sum((pts.max(axis=0) - pts.min(axis=0)) ** 2)))
    radii = np.logspace(
        np.log10(fit_range[0] * diameter),
        np.log10(fit_range[1] * diameter),
        n_radii,
    )
    c = correlation_sum(pts, radii, theiler=theiler, max_points=max_points)

    ok = c > 0.0
    if ok.sum() < 3:
        raise ValueError(
            "fewer than 3 non-empty radii in the fit window; widen `fit_range` "
            "or supply a longer trajectory"
        )
    slope = np.polyfit(np.log(radii[ok]), np.log(c[ok]), 1)[0]
    return float(slope), radii, c


def fit_dimension(
    radii: Array, c: Array, fit_range: tuple[float, float]
) -> tuple[float, int]:
    r"""Slope of :math:`\ln C` against :math:`\ln r` over a sub-range.

    Returns ``(D2, n_points_used)``. ``fit_range`` is given in the same units as
    ``radii`` (absolute, not fractions of the diameter).

    Separated from :func:`correlation_dimension` because the two halves of that
    computation have wildly different costs: forming :math:`C(r)` is
    :math:`O(N^2)` in the number of trajectory samples, while re-fitting a
    window on an existing curve is microseconds. A chapter that lets the reader
    drag the scaling window computes :math:`C(r)` **once** over a wide range and
    calls this on every drag.
    """
    radii = np.asarray(radii, dtype=float)
    c = np.asarray(c, dtype=float)
    lo, hi = float(min(fit_range)), float(max(fit_range))
    inside = (radii >= lo) & (radii <= hi) & (c > 0.0)
    if int(inside.sum()) < 3:
        raise ValueError(
            f"only {int(inside.sum())} usable radii in [{lo:g}, {hi:g}]; widen "
            "the window or supply a longer trajectory"
        )
    slope = np.polyfit(np.log(radii[inside]), np.log(c[inside]), 1)[0]
    return float(slope), int(inside.sum())


# --------------------------------------------------------------------------
# Box counting and the Renyi dimensions
# --------------------------------------------------------------------------
def box_occupancy(points: Array, scale: float) -> Array:
    r"""Point counts of the occupied boxes of a grid of the given ``scale``.

    The set is first rescaled so its **largest** extent is 1, so ``scale`` is a
    fraction of that extent. Returns one count per *occupied* box; empty boxes
    are not represented, since neither the box count nor any
    :math:`\sum_i p_i^q` needs them.

    ``len(box_occupancy(...))`` is the box count :math:`N(\varepsilon)` and
    ``points.shape[0] / len(...)`` is the mean occupancy -- the number that
    decides whether a box-counting estimate means anything (see
    :data:`MIN_BOX_OCCUPANCY`).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    lower = pts.min(axis=0)
    extent = float((pts.max(axis=0) - lower).max())
    if not extent > 0.0:
        raise ValueError("the point set has zero extent")
    unit = (pts - lower) / extent
    index = np.floor(unit / float(scale)).astype(np.int64)
    _, counts = np.unique(index, axis=0, return_counts=True)
    return counts


def renyi_dimension(
    points: Array,
    q: float = 0.0,
    fit_range: tuple[float, float] = (0.004, 0.06),
    n_scales: int = 12,
) -> tuple[float, Array, Array, Array]:
    r"""Generalised (Renyi) dimension :math:`D_q` by box counting.

    .. math::
        D_q = \lim_{\varepsilon\to0}\frac{1}{q-1}
              \frac{\ln \sum_i p_i^{\,q}}{\ln\varepsilon},
        \qquad
        D_1 = \lim_{\varepsilon\to0}\frac{\sum_i p_i\ln p_i}{\ln\varepsilon},

    with :math:`p_i` the fraction of points in occupied box :math:`i`. The
    special cases are the three dimensions usually quoted: :math:`q=0` is the
    **box-counting** dimension (which counts occupied boxes and ignores how
    full they are), :math:`q=1` the **information** dimension, and :math:`q=2`
    a box-based version of the correlation dimension. They obey
    :math:`D_0 \ge D_1 \ge D_2`, with equality only for a uniform measure --
    an exact inequality, asserted in the tests.

    Returns ``(D_q, scales, exponents, occupancy)`` where ``exponents`` is the
    quantity whose slope against :math:`\ln\varepsilon` is :math:`D_q`, and
    ``occupancy`` is the mean number of points per occupied box at each scale.

    **Box counting is the estimator to reach for when you can afford it and the
    wrong one otherwise.** It recovers the closed-form dimensions of the
    reference sets in this module to within 0.01 given a few hundred thousand
    points in one or two dimensions. In three dimensions at achievable sample
    sizes it starves: boxes fine enough to resolve structure hold about one
    point each, and the measured slope collapses toward zero. The ``occupancy``
    return exists so that this is visible rather than inferred -- keep the fit
    to scales where it exceeds :data:`MIN_BOX_OCCUPANCY`. For a
    three-dimensional attractor, :func:`correlation_dimension` uses all
    :math:`N^2/2` pairs instead of spreading :math:`N` points over boxes and
    does far better on the same data.

    A second, smaller bias is worth knowing about because it does not vanish
    with more points. The set is rescaled into the unit box, so a grid of scale
    :math:`\varepsilon` lays down :math:`(1/\varepsilon + 1)^d` boxes rather
    than :math:`\varepsilon^{-d}`, and the extra row along each edge biases the
    slope low by :math:`O(\varepsilon)`. On a uniform square, windows of
    :math:`(0.02, 0.15)`, :math:`(0.008, 0.06)` and :math:`(0.004, 0.03)` return
    1.925, 1.978 and 1.983 against the exact 2 -- tracking the
    :math:`(1/\varepsilon+1)^2` prediction of 1.885, 1.952 and 1.976 rather
    than converging on 2 from a fixed window. Prefer the finest window the
    sample supports.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    n = pts.shape[0]
    q = float(q)
    scales = np.logspace(
        np.log10(float(fit_range[0])), np.log10(float(fit_range[1])), int(n_scales)
    )

    exponents = np.empty(scales.size, dtype=float)
    occupancy = np.empty(scales.size, dtype=float)
    for k, scale in enumerate(scales):
        counts = box_occupancy(pts, float(scale))
        probability = counts / n
        occupancy[k] = n / counts.size
        if abs(q - 1.0) < 1e-12:
            exponents[k] = float((probability * np.log(probability)).sum())
        else:
            exponents[k] = float(
                np.log((probability**q).sum()) / (q - 1.0)
            )

    slope = np.polyfit(np.log(scales), exponents, 1)[0]
    return float(slope), scales, exponents, occupancy


def renyi_spectrum(
    points: Array,
    q_values: Array = (0.0, 1.0, 2.0),
    fit_range: tuple[float, float] = (0.004, 0.06),
    n_scales: int = 12,
) -> Array:
    r""":math:`D_q` for each requested :math:`q`, sharing one point set.

    Useful mainly for checking the ordering :math:`D_0 \ge D_1 \ge D_2`, which
    holds for any measure and is therefore a check on the estimator rather than
    on the attractor.
    """
    return np.array(
        [
            renyi_dimension(points, float(q), fit_range, n_scales)[0]
            for q in np.atleast_1d(np.asarray(q_values, dtype=float))
        ]
    )


# --------------------------------------------------------------------------
# Delay embedding: dimension from one observed variable
# --------------------------------------------------------------------------
def delay_embed(series: Array, dimension: int, lag: int) -> Array:
    r"""Time-delay (Takens) embedding of a scalar series.

    Builds the vectors
    :math:`\bigl(s_n,\ s_{n+\ell},\ \ldots,\ s_{n+(m-1)\ell}\bigr)` for
    :math:`m =` ``dimension`` and :math:`\ell =` ``lag``, returned with shape
    ``(len(series) - (m-1)*lag, m)``.

    This is what makes a measured dimension possible at all. A real observing
    system does not deliver the state vector; it delivers a few scalar time
    series. Takens' theorem says that for almost any smooth observable, an
    embedding of dimension :math:`m > 2D` reconstructs a set diffeomorphic to
    the original attractor, so its dimension, its Lyapunov exponents and its
    topology survive the reconstruction *[citation needed: Takens (1981)]*.

    In practice :math:`D` is what you are trying to measure, so the criterion
    cannot be applied directly. What is done instead: increase :math:`m` and
    watch the dimension estimate. It rises while the embedding is too small to
    hold the set without self-intersection -- the estimate is then capped by
    :math:`m` rather than by the attractor -- and **saturates** once the
    embedding is adequate. Chapter 8 measures 1.72, 1.95, 2.00, 2.01 for
    :math:`m = 2, 3, 4, 5` on the :math:`x` component of Lorenz 63, against
    2.058 from the full state.

    The ``lag`` matters less than :math:`m` but is not free: too short and the
    coordinates are nearly equal, collapsing the reconstruction onto the
    diagonal; too long and they are dynamically unrelated. Values from 0.1 to
    0.3 in Lorenz 63 model time units all work.
    """
    s = np.asarray(series, dtype=float).ravel()
    m, lag = int(dimension), int(lag)
    if m < 1 or lag < 1:
        raise ValueError("dimension and lag must both be at least 1")
    length = s.size - (m - 1) * lag
    if length < 2:
        raise ValueError(
            f"series of {s.size} samples is too short for dimension {m} at lag {lag}"
        )
    return np.stack([s[i * lag : i * lag + length] for i in range(m)], axis=-1)


# --------------------------------------------------------------------------
# Reference sets whose dimension is known in closed form
# --------------------------------------------------------------------------
def cantor_set(
    n_points: int = 200_000, depth: int = 18, seed: int = 0
) -> Array:
    r"""Points on the middle-thirds Cantor set. :math:`D_0 = \ln2/\ln3`.

    Every point of the set has a base-3 expansion using only the digits 0 and
    2, so a point is generated by drawing ``depth`` such digits -- no recursive
    subdivision, and every point is an independent exact sample of the set
    truncated at :math:`3^{-\text{depth}}`.
    """
    rng = np.random.default_rng(int(seed))
    digits = 2 * rng.integers(0, 2, size=(int(n_points), int(depth)))
    weights = 3.0 ** -np.arange(1, int(depth) + 1)
    return (digits * weights).sum(axis=1)[:, None]


def koch_curve(depth: int = 7) -> Array:
    r"""Vertices of the Koch curve. :math:`D_0 = \ln4/\ln3`.

    Built by explicit subdivision, so the vertex count is
    :math:`4^{\text{depth}} + 1` and the points are the curve's corners rather
    than a sample of its length. That is adequate for box counting because the
    corners are dense in the curve at the scale the subdivision reaches.
    """
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    rotation = np.array(
        [[0.5, -np.sqrt(3.0) / 2.0], [np.sqrt(3.0) / 2.0, 0.5]]
    )
    for _ in range(int(depth)):
        grown = [points[0]]
        for a, b in zip(points[:-1], points[1:]):
            step = (b - a) / 3.0
            first, second = a + step, a + 2.0 * step
            grown += [first, first + rotation @ step, second, b]
        points = np.array(grown)
    return points


def sierpinski_triangle(
    n_points: int = 200_000,
    depth: int = 24,
    seed: int = 1,
    probabilities: Array | None = None,
) -> Array:
    r"""Points on the Sierpinski triangle. :math:`D_0 = \ln3/\ln2`.

    The chaos game, vectorised. Iterating
    :math:`p \mapsto (p + v_c)/2` for a randomly chosen vertex :math:`v_c`
    gives, after :math:`k` steps,
    :math:`p_k = \sum_{i=1}^{k} 2^{-i} v_{c_i} + 2^{-k} p_0`. The dependence on
    :math:`p_0` decays as :math:`2^{-k}`, so for ``depth`` of 24 it is below
    :math:`10^{-7}` and every point can be drawn independently from its own
    sequence of vertex choices -- turning a sequential loop into one array
    operation.

    ``probabilities`` biases the vertex choice, which leaves the **support**
    unchanged -- still the whole triangle, still :math:`D_0 = \ln3/\ln2` -- but
    makes the invariant measure on it non-uniform, so the Renyi dimensions
    separate. That case has closed forms, which is what makes it worth having:
    for contraction ratio :math:`1/2`,

    .. math::
        D_1 = -\frac{\sum_i p_i \ln p_i}{\ln 2},
        \qquad
        D_2 = -\frac{\ln \sum_i p_i^2}{\ln 2}.

    Note that a *measured* :math:`D_0` falls too once the weights are skewed,
    even though the support has not changed. That is not an error in the
    estimator: :math:`D_0` counts a box holding one point the same as a box
    holding a million, so rarely visited parts of the support are simply never
    sampled. It is the reason :math:`D_0` is the least robust of the three to
    estimate, and the reason this book quotes :math:`D_2`.
    """
    rng = np.random.default_rng(int(seed))
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    if probabilities is None:
        choices = rng.integers(0, 3, size=(int(n_points), int(depth)))
    else:
        weightsp = np.asarray(probabilities, dtype=float)
        if weightsp.shape != (3,) or not np.isclose(weightsp.sum(), 1.0):
            raise ValueError("probabilities must be three values summing to 1")
        choices = rng.choice(3, size=(int(n_points), int(depth)), p=weightsp)
    weights = 2.0 ** -np.arange(1, int(depth) + 1)
    return (vertices[choices] * weights[None, :, None]).sum(axis=1)
