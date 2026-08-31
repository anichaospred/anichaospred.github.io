"""Fractal dimension of an attractor, measured from a sampled trajectory.

The Kaplan-Yorke dimension in :mod:`chaoslib.lyapunov` is computed from the
Lyapunov spectrum -- i.e. from the *dynamics*. The estimator here works from a
sampled trajectory only, so the two give genuinely independent numbers for the
same attractor. Agreement between them is evidence both are right; for Lorenz 63
both land near 2.06.

Two traps this module is built to keep the reader out of, because both bias
:math:`D_2` badly **low** and neither announces itself:

1. **The scaling window.** :math:`C(r) \\sim r^{D_2}` only holds over a limited
   range of :math:`r`. Above it :math:`C \\to 1` and the measured slope falls to
   zero; below it there are too few pairs and the slope is noise. For Lorenz 63
   the window is roughly 1--5 % of the attractor's diameter -- which is why the
   defaults here are small, and why the radii and :math:`C(r)` are returned so a
   chapter can plot the fit and let the reader judge the window rather than
   trust it.
2. **Temporal correlation.** Consecutive trajectory samples are close together
   because they are consecutive, not because the attractor is dense there.
   Counting those pairs measures the trajectory's smoothness, not the
   attractor's geometry. The ``theiler`` window excludes pairs closer than a
   given number of samples in time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]

__all__ = ["correlation_sum", "correlation_dimension", "local_slopes"]


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
