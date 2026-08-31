"""Data assimilation on the book's low-order systems.

Four schemes, in the order the book introduces them, all sharing
one interface so a chapter can swap between them and compare honestly:

* :func:`kalman_filter_update` -- the linear-Gaussian optimum, the yardstick.
* :func:`three_dvar_update` -- 3D-Var: a *fixed* background covariance,
  minimising the same cost function but never learning about the flow.
* :func:`four_dvar_analysis` -- 4D-Var: fits a whole trajectory to observations
  spread over a window, using the adjoint to get the gradient.
* :func:`enkf_update` -- the ensemble Kalman filter: the background covariance is
  estimated from the ensemble, so it *is* flow-dependent, at the cost of
  sampling noise (hence localisation and inflation).

Notation follows Kalnay (2003): ``xb`` background, ``xa`` analysis, ``B``/``P``
background covariance, ``R`` observation error covariance, ``H`` observation
operator, ``y`` observations.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from chaoslib.adjoint import tangent_linear_propagator
from chaoslib.integrate import rk4

Array = NDArray[np.floating]
RHS = Callable[..., Array]
Jacobian = Callable[..., Array]

__all__ = [
    "kalman_filter_update",
    "three_dvar_update",
    "four_dvar_analysis",
    "enkf_update",
    "gaspari_cohn",
    "analysis_rmse",
]


def kalman_filter_update(
    xb: Array, b_cov: Array, y: Array, h_op: Array, r_cov: Array
) -> tuple[Array, Array]:
    r"""One linear-Gaussian Kalman analysis step.

    .. math::
        \mathbf{K} = \mathbf{B}\mathbf{H}^{\top}
                     (\mathbf{H}\mathbf{B}\mathbf{H}^{\top}+\mathbf{R})^{-1},
        \quad x^a = x^b + \mathbf{K}(y - \mathbf{H}x^b)

    Returns ``(xa, analysis_covariance)``. This is the exact posterior when the
    model is linear and all errors Gaussian -- the reference every other scheme
    in this module is an approximation to, and the benchmark used to test them.
    """
    xb = np.asarray(xb, dtype=float).ravel()
    b_cov = np.atleast_2d(np.asarray(b_cov, dtype=float))
    h_op = np.atleast_2d(np.asarray(h_op, dtype=float))
    r_cov = np.atleast_2d(np.asarray(r_cov, dtype=float))
    y = np.asarray(y, dtype=float).ravel()

    bht = b_cov @ h_op.T
    gain = bht @ np.linalg.inv(h_op @ bht + r_cov)
    xa = xb + gain @ (y - h_op @ xb)
    p_a = (np.eye(xb.size) - gain @ h_op) @ b_cov
    return xa, p_a


def three_dvar_update(
    xb: Array, b_cov: Array, y: Array, h_op: Array, r_cov: Array
) -> Array:
    r"""3D-Var analysis: minimiser of the standard cost function.

    .. math::
        J(x) = \tfrac12 (x-x^b)^{\top}\mathbf{B}^{-1}(x-x^b)
             + \tfrac12 (y-\mathbf{H}x)^{\top}\mathbf{R}^{-1}(y-\mathbf{H}x)

    Algebraically identical to the Kalman analysis *for the same*
    :math:`\mathbf{B}`. The difference in practice is entirely that 3D-Var holds
    :math:`\mathbf{B}` fixed for all time instead of evolving it -- so it cannot
    know that today's background error lies along the flow's growing directions.
    That single limitation is what motivates everything after it.
    """
    xa, _ = kalman_filter_update(xb, b_cov, y, h_op, r_cov)
    return xa


def four_dvar_analysis(
    rhs: RHS,
    jacobian: Jacobian,
    xb: Array,
    b_cov: Array,
    observations: list[tuple[float, Array]],
    h_op: Array,
    r_cov: Array,
    dt: float = 0.01,
    max_iterations: int = 60,
    **params: float,
) -> Array:
    r"""Incremental 4D-Var over an assimilation window.

    Minimises

    .. math::
        J(x_0) = \tfrac12 (x_0-x^b)^{\top}\mathbf{B}^{-1}(x_0-x^b)
          + \tfrac12 \sum_k (y_k - \mathbf{H}x_k)^{\top}\mathbf{R}^{-1}
                            (y_k - \mathbf{H}x_k)

    where the sum runs over observation times inside the window and :math:`x_k`
    is the *model trajectory* launched from :math:`x_0`. The gradient

    .. math::
        \nabla J = \mathbf{B}^{-1}(x_0-x^b)
          - \sum_k \mathbf{M}_k^{\top}\mathbf{H}^{\top}\mathbf{R}^{-1}
                   (y_k - \mathbf{H}x_k)

    costs one adjoint application per observation time -- *not* one model run per
    degree of freedom. That asymmetry is the entire reason variational
    assimilation is affordable at operational size.

    ``observations`` is a list of ``(time, y)`` pairs with times measured from
    the window start.

    Minimised with L-BFGS-B on the analytic gradient. Fixed-step steepest
    descent is not a viable substitute here: with tight observation errors
    :math:`\mathbf{R}^{-1}` makes the gradient large, and any step size big
    enough to converge in a reasonable number of iterations sends the Lorenz
    trajectory to overflow. Real systems use a quasi-Newton or conjugate-gradient
    minimiser for exactly this reason.

    Returns the analysed initial state.
    """
    from scipy.optimize import minimize

    xb = np.asarray(xb, dtype=float).ravel()
    b_cov = np.atleast_2d(np.asarray(b_cov, dtype=float))
    h_op = np.atleast_2d(np.asarray(h_op, dtype=float))
    r_inv = np.linalg.inv(np.atleast_2d(np.asarray(r_cov, dtype=float)))
    b_inv = np.linalg.inv(b_cov)

    prepared = [
        (float(t_obs), np.asarray(y_obs, dtype=float).ravel())
        for t_obs, y_obs in observations
    ]

    def cost_and_gradient(x0: Array) -> tuple[float, Array]:
        departure = x0 - xb
        cost = 0.5 * float(departure @ (b_inv @ departure))
        grad = b_inv @ departure

        for t_obs, y_obs in prepared:
            if t_obs <= 0.0:
                # Observation at the window start: no propagation, and the
                # propagator below is the identity.
                x_at_obs = x0
            else:
                n_steps = max(1, int(round(t_obs / dt)))
                # linspace, not arange*dt: the trajectory must end exactly at
                # t_obs so that it matches the tangent-linear propagator.
                grid = np.linspace(0.0, t_obs, n_steps + 1)
                x_at_obs = rk4(rhs, x0, grid, **params)[-1]
            innovation = y_obs - h_op @ x_at_obs
            weighted = r_inv @ innovation
            cost += 0.5 * float(innovation @ weighted)
            propagator = tangent_linear_propagator(
                rhs, jacobian, x0, t_obs, dt=dt, **params
            )
            grad -= propagator.T @ (h_op.T @ weighted)

        if not np.isfinite(cost) or not np.all(np.isfinite(grad)):
            # A diverged trajectory must be reported as a barrier, not as NaN,
            # or the line search will happily step further into the blow-up.
            return float("inf"), np.zeros_like(x0)
        return cost, grad

    result = minimize(
        cost_and_gradient,
        xb.copy(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iterations)},
    )
    return np.asarray(result.x, dtype=float)


def gaspari_cohn(distance: Array, cutoff: float) -> Array:
    """Gaspari-Cohn localisation weights: a compactly supported 5th-order function.

    Zero beyond ``2*cutoff/2`` (i.e. beyond ``cutoff``), smooth, and positive
    definite -- the last property is why this specific polynomial is used rather
    than a Gaussian truncated by hand, which would not guarantee a valid
    covariance.

    Localisation exists because an ensemble of ~20 members cannot estimate
    covariances between distant points: those entries are pure sampling noise,
    and multiplying them away is cheaper than a bigger ensemble.
    """
    d = np.abs(np.asarray(distance, dtype=float)) / (cutoff / 2.0)
    w = np.zeros_like(d)

    near = d <= 1.0
    mid = (d > 1.0) & (d <= 2.0)
    dn = d[near]
    w[near] = (
        -0.25 * dn**5 + 0.5 * dn**4 + 0.625 * dn**3 - 5.0 / 3.0 * dn**2 + 1.0
    )
    dm = d[mid]
    w[mid] = (
        1.0 / 12.0 * dm**5
        - 0.5 * dm**4
        + 0.625 * dm**3
        + 5.0 / 3.0 * dm**2
        - 5.0 * dm
        + 4.0
        - 2.0 / 3.0 / dm
    )
    return np.clip(w, 0.0, 1.0)


def enkf_update(
    ensemble: Array,
    y: Array,
    h_op: Array,
    r_cov: Array,
    inflation: float = 1.0,
    localisation: Array | None = None,
    seed: int | None = 0,
) -> Array:
    r"""Stochastic (perturbed-observation) ensemble Kalman filter analysis.

    ``ensemble`` has shape ``(n_members, n_state)``. The background covariance is
    the sample covariance of the ensemble, so it is flow-dependent for free --
    the whole appeal of the method.

    ``inflation`` multiplies the background *perturbations* before the update, the
    standard remedy for the systematic under-dispersion of small ensembles.
    ``localisation``, if given, is an ``(n_state, n_state)`` weight matrix applied
    elementwise to the background covariance (see :func:`gaspari_cohn`).

    Each member assimilates its own perturbed observation, drawn from
    :math:`\mathcal{N}(y, \mathbf{R})`; without that perturbation the analysis
    ensemble is systematically under-dispersed.
    """
    ens = np.atleast_2d(np.asarray(ensemble, dtype=float))
    h_op = np.atleast_2d(np.asarray(h_op, dtype=float))
    r_cov = np.atleast_2d(np.asarray(r_cov, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n_members, n_state = ens.shape
    if n_members < 2:
        raise ValueError("EnKF needs at least 2 members")

    mean = ens.mean(axis=0)
    pert = inflation * (ens - mean)
    p_b = pert.T @ pert / (n_members - 1)
    if localisation is not None:
        p_b = p_b * np.atleast_2d(np.asarray(localisation, dtype=float))

    pbht = p_b @ h_op.T
    gain = pbht @ np.linalg.inv(h_op @ pbht + r_cov)

    rng = np.random.default_rng(seed)
    noise = rng.multivariate_normal(np.zeros(y.size), r_cov, size=n_members)
    background = mean + pert
    innovations = (y + noise) - background @ h_op.T
    return background + innovations @ gain.T


def analysis_rmse(analysis: Array, truth: Array) -> float:
    """RMS error of an analysis (or analysis ensemble mean) against the truth."""
    a = np.asarray(analysis, dtype=float)
    if a.ndim == 2:
        a = a.mean(axis=0)
    return float(
        np.sqrt(np.mean((a - np.asarray(truth, dtype=float).ravel()) ** 2))
    )
