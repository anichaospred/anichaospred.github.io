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
]


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
    t: Array, error: Array, saturation: float | None = None
) -> tuple[float, float, float]:
    r"""Fit :math:`(E_0, \lambda, E_\infty)` to a measured error curve.

    If ``saturation`` is supplied it is held fixed (the usual case -- it is
    better measured directly from the climatological spread than fitted, since
    the tail of the curve constrains it weakly). Returns
    ``(e0, growth_rate, saturation)``.

    Fitting the full curve with this model, rather than a straight line to the
    early portion, is the honest way to get :math:`\lambda` when the record
    already includes the bend-over.
    """
    t = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    e_sat = float(np.nanmax(error)) if saturation is None else float(saturation)

    if saturation is None:
        def model(tt: Array, e0: float, rate: float, sat: float) -> Array:
            return logistic_error_growth(tt, e0, rate, sat)

        p0 = (max(error[0], 1e-12), 1.0, e_sat)
        popt, _ = curve_fit(model, t, error, p0=p0, maxfev=20000)
        return float(popt[0]), float(popt[1]), float(popt[2])

    def model_fixed(tt: Array, e0: float, rate: float) -> Array:
        return logistic_error_growth(tt, e0, rate, e_sat)

    p0 = (max(error[0], 1e-12), 1.0)
    popt, _ = curve_fit(model_fixed, t, error, p0=p0, maxfev=20000)
    return float(popt[0]), float(popt[1]), e_sat


def saturation_level(trajectory: Array, seed: int | None = 0) -> float:
    r"""Climatological error saturation: RMS distance between random state pairs.

    The level a forecast error tends to once all skill is gone -- equivalently
    the error of a random draw from climatology. Computed as the RMS distance
    between independently shuffled pairs of states from a long trajectory, which
    is what "two randomly chosen states of the attractor" means operationally.
    """
    traj = np.asarray(trajectory, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(traj.shape[0])
    return float(np.sqrt(np.mean(np.sum((traj - traj[idx]) ** 2, axis=-1))))


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
