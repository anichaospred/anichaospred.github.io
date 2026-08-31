"""Information-theoretic measures of predictability.

The dynamical-systems view asks how fast trajectories separate. The
information-theoretic view asks a subtly different and often more useful
question: *how much does the forecast tell us that climatology did not?* A
forecast has skill exactly as long as the forecast distribution is
distinguishable from the climatological one, and relative entropy is the natural
currency for that.

All quantities are returned in **nats** (natural log). Divide by ``ln 2`` for
bits.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]

__all__ = [
    "shannon_entropy",
    "relative_entropy",
    "mutual_information",
    "predictive_information",
    "gaussian_relative_entropy",
]

_EPS = np.finfo(float).tiny


def shannon_entropy(p: Array, axis: int | None = None) -> Array | float:
    r"""Shannon entropy :math:`-\sum_i p_i \ln p_i` of a normalised histogram.

    Zero-probability bins contribute zero (the :math:`p\ln p \to 0` limit) and
    are masked rather than allowed to produce ``nan``.
    """
    p = np.asarray(p, dtype=float)
    terms = np.where(p > 0.0, p * np.log(np.maximum(p, _EPS)), 0.0)
    out = -terms.sum(axis=axis)
    return float(out) if np.ndim(out) == 0 else out


def relative_entropy(p: Array, q: Array) -> float:
    r"""Kullback-Leibler divergence :math:`D(p\|q) = \sum_i p_i \ln(p_i/q_i)`.

    Read as the information gained by using the forecast distribution ``p``
    instead of the climatology ``q``. It is not symmetric, and that asymmetry is
    the right way round for forecast evaluation: what matters is the cost of
    believing climatology when the forecast is true.

    Requires ``q > 0`` wherever ``p > 0`` (otherwise the divergence is
    infinite); raises rather than silently returning ``inf``.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    support = p > 0.0
    if np.any(q[support] <= 0.0):
        raise ValueError(
            "q has zero probability where p is positive: D(p||q) is infinite"
        )
    return float(np.sum(p[support] * np.log(p[support] / q[support])))


def mutual_information(joint: Array) -> float:
    r"""Mutual information of a 2-D joint histogram.

    .. math::
        I(X;Y) = \sum_{ij} p_{ij}\,\ln\frac{p_{ij}}{p_i\,p_j}

    Used for predictability as :math:`I` between the initial state and the state
    at lead time :math:`\tau`: it decays to zero exactly when the forecast
    carries no more information than climatology, giving a horizon that needs no
    assumption of Gaussianity or of exponential growth.
    """
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint must be a 2-D histogram")
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    outer = px * py
    support = joint > 0.0
    return float(
        np.sum(joint[support] * np.log(joint[support] / outer[support]))
    )


def predictive_information(
    initial: Array, later: Array, bins: int = 32
) -> float:
    """Mutual information between a state component now and at a later time.

    Convenience wrapper that builds the 2-D histogram for you. ``bins`` is the
    usual bias/variance trade-off: too many bins and finite-sample noise inflates
    ``I`` towards ``ln(n)``; the chapter that uses this should show the
    dependence rather than hide it behind a default.
    """
    joint, _, _ = np.histogram2d(
        np.asarray(initial, dtype=float).ravel(),
        np.asarray(later, dtype=float).ravel(),
        bins=bins,
    )
    return mutual_information(joint)


def gaussian_relative_entropy(
    mean_f: Array, cov_f: Array, mean_c: Array, cov_c: Array
) -> float:
    r"""Relative entropy between two multivariate Gaussians, in closed form.

    .. math::
        D = \tfrac12\left[\operatorname{tr}(\Sigma_c^{-1}\Sigma_f)
            + (\mu_c-\mu_f)^{\!\top}\Sigma_c^{-1}(\mu_c-\mu_f)
            - k + \ln\frac{\det\Sigma_c}{\det\Sigma_f}\right]

    The two terms separate the two distinct ways an ensemble forecast can be
    informative: the **signal** component (its mean differs from climatology)
    and the **dispersion** component (it is sharper than climatology). Both
    matter, and conflating them is a standard source of confusion in
    predictability studies.
    """
    mean_f = np.atleast_1d(np.asarray(mean_f, dtype=float))
    mean_c = np.atleast_1d(np.asarray(mean_c, dtype=float))
    cov_f = np.atleast_2d(np.asarray(cov_f, dtype=float))
    cov_c = np.atleast_2d(np.asarray(cov_c, dtype=float))
    k = mean_f.size

    solve_c = np.linalg.solve(cov_c, cov_f)
    dmu = mean_c - mean_f
    quad = float(dmu @ np.linalg.solve(cov_c, dmu))
    sign_c, logdet_c = np.linalg.slogdet(cov_c)
    sign_f, logdet_f = np.linalg.slogdet(cov_f)
    if sign_c <= 0 or sign_f <= 0:
        raise ValueError("covariance matrices must be positive definite")
    return 0.5 * (np.trace(solve_c) + quad - k + logdet_c - logdet_f)
