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
    "gaussian_information_components",
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


def mutual_information(joint: Array, correction: str = "none") -> float:
    r"""Mutual information of a joint histogram, in nats.

    .. math::
        I(X;Y) = \sum_{ij} p_{ij}\,\ln\frac{p_{ij}}{p_i\,p_j}

    ``correction="miller_madow"`` subtracts the leading finite-sample bias,
    :math:`(B_{xy} - B_x - B_y + 1)/2N` with :math:`B` the counts of *occupied*
    bins. The plug-in estimator is biased **upward**, and badly: on independent
    samples, where the true value is exactly zero, it returns 0.046 nats at
    :math:`N = 2000` with 16 bins and 0.493 with 64. The correction brings those
    to 0.011 and 0.274 -- a factor of two to four, not a cure. Below the
    resulting floor an apparent mutual information is measuring the estimator.

    The correction does not damage a real signal: on a bivariate Gaussian of
    correlation 0.5, whose exact value is :math:`-\tfrac12\ln(1-r^2) = 0.1438`,
    the plug-in estimate at 64 bins is 0.1484 and the corrected one 0.1439.
    """
    joint = np.asarray(joint, dtype=float)
    total = joint.sum()
    if total <= 0:
        raise ValueError("the joint histogram is empty")
    p = joint / total
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nonzero = p > 0.0
    plug_in = float(
        np.sum(p[nonzero] * np.log(p[nonzero] / (px @ py)[nonzero]))
    )
    if correction == "none":
        return plug_in
    if correction != "miller_madow":
        raise ValueError(
            f"correction must be 'none' or 'miller_madow', not {correction!r}"
        )
    occupied_joint = int(np.count_nonzero(joint))
    occupied_x = int(np.count_nonzero(joint.sum(axis=1)))
    occupied_y = int(np.count_nonzero(joint.sum(axis=0)))
    bias = (occupied_joint - occupied_x - occupied_y + 1) / (2.0 * total)
    return plug_in - bias


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


def gaussian_information_components(
    mean_f: Array, cov_f: Array, mean_c: Array, cov_c: Array
) -> tuple[float, float, float]:
    r"""Split the Gaussian relative entropy into signal and dispersion.

    Returns ``(total, signal, dispersion)`` with

    .. math::
        D_{\rm signal} &= \tfrac12(\mu_c-\mu_f)^{\!\top}\Sigma_c^{-1}(\mu_c-\mu_f),\\
        D_{\rm disp} &= \tfrac12\Bigl[\operatorname{tr}(\Sigma_c^{-1}\Sigma_f)
                        - k + \ln\frac{\det\Sigma_c}{\det\Sigma_f}\Bigr],

    which sum to :func:`gaussian_relative_entropy` exactly -- asserted to
    machine precision in the tests, as is :math:`D_{\rm disp} \ge 0` with
    equality only when :math:`\Sigma_f = \Sigma_c`.

    The two answer different questions *[citation needed: DelSole (2004)]*.
    **Signal** is "my forecast says something other than the climatological
    mean"; **dispersion** is "my forecast is sharper than climatology". A
    forecast can be informative either way, and for a well-initialised ensemble
    the second dominates overwhelmingly: chapter 10 measures 8.5 of 9.0 nats as
    dispersion at lead zero, with the signal term hovering near 0.4 throughout.

    Like the total, both components are **invariant under any invertible linear
    transformation of the state** -- change your units, rotate your basis,
    reweight your variables, and the numbers do not move -- to
    :math:`2\times10^{-15}` nats, which is round-off. RMS error is not: under the
    same transformations chapter 10 measures it varying by a factor of 3.7 with
    no rescaling and 371 at the extreme, so the number it reports is a statement
    about the units as much as about the forecast. That invariance is the reason to reach for an information measure at
    all, and it is the exact counterpart of chapter 16's finding that singular
    vectors *do* depend on the norm.

    A warning about using this on a full state vector. The forecast covariance
    of an ensemble on a low-dimensional attractor is near-singular -- a
    400-member Lorenz 63 ensemble collapses onto a set of dimension 2.06 inside
    a 3-dimensional space -- so :math:`\det\Sigma_f` is set by whatever
    regularisation is applied rather than by the dynamics, and the dispersion
    term with it. Chapter 10 shows the resulting decay rate changing by a factor
    of two between two reasonable choices. Use it on scalars or on a
    well-conditioned subspace.
    """
    mean_f = np.atleast_1d(np.asarray(mean_f, dtype=float))
    mean_c = np.atleast_1d(np.asarray(mean_c, dtype=float))
    cov_f = np.atleast_2d(np.asarray(cov_f, dtype=float))
    cov_c = np.atleast_2d(np.asarray(cov_c, dtype=float))
    k = mean_f.size

    sign_c, logdet_c = np.linalg.slogdet(cov_c)
    sign_f, logdet_f = np.linalg.slogdet(cov_f)
    if sign_c <= 0 or sign_f <= 0:
        raise ValueError("covariance matrices must be positive definite")

    dmu = mean_c - mean_f
    signal = 0.5 * float(dmu @ np.linalg.solve(cov_c, dmu))
    dispersion = 0.5 * float(
        np.trace(np.linalg.solve(cov_c, cov_f)) - k + logdet_c - logdet_f
    )
    return signal + dispersion, signal, dispersion

