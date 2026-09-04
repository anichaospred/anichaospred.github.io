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
    "four_dvar_cost",
    "four_dvar_hessian",
    "incremental_four_dvar",
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


def four_dvar_cost(
    rhs: RHS,
    jacobian: Jacobian,
    x0: Array,
    xb: Array,
    b_cov: Array,
    observations: list[tuple[float, Array]],
    h_op: Array,
    r_cov: Array,
    dt: float = 0.01,
    **params: float,
) -> tuple[float, Array]:
    r"""The 4D-Var cost function and its adjoint gradient at ``x0``.

    .. math::
        J(x_0) = \tfrac12 (x_0-x^b)^{\top}\mathbf{B}^{-1}(x_0-x^b)
          + \tfrac12 \sum_k d_k^{\top}\mathbf{R}^{-1} d_k,
        \qquad d_k = y_k - \mathbf{H}\mathcal{M}_{0\to k}(x_0)

    with gradient

    .. math::
        \nabla J = \mathbf{B}^{-1}(x_0-x^b)
          - \sum_k \mathbf{M}_k^{\top}\mathbf{H}^{\top}\mathbf{R}^{-1} d_k .

    The sum runs over observation times inside the window, measured from the
    window start, and :math:`x_k` is the *model trajectory* launched from
    :math:`x_0` -- so the model enters as a strong constraint, and information
    from an observation of one variable reaches every variable the dynamics
    couple it to.

    Each term of the gradient costs **one adjoint application**, not one model
    run per degree of freedom. That asymmetry is the entire reason variational
    assimilation is affordable at operational size.

    Public, and not merely an implementation detail of
    :func:`four_dvar_analysis`, because the cost surface and the gradient are
    objects worth looking at rather than only minimising:
    :func:`chaoslib.adjoint.gradient_test` consumes this function directly.

    A diverged trajectory returns ``(inf, 0)`` rather than ``(nan, nan)``. A line
    search treats infinity as a barrier and backs off; NaN it will step straight
    past, and the minimiser then wanders in a region where the cost is undefined.

    Returns ``(J, grad)``.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    xb = np.asarray(xb, dtype=float).ravel()
    h_op = np.atleast_2d(np.asarray(h_op, dtype=float))
    b_inv = np.linalg.inv(np.atleast_2d(np.asarray(b_cov, dtype=float)))
    r_inv = np.linalg.inv(np.atleast_2d(np.asarray(r_cov, dtype=float)))

    departure = x0 - xb
    cost = 0.5 * float(departure @ (b_inv @ departure))
    grad = b_inv @ departure

    for t_raw, y_raw in observations:
        t_obs = float(t_raw)
        y_obs = np.asarray(y_raw, dtype=float).ravel()
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
        return float("inf"), np.zeros_like(x0)
    return cost, grad


def four_dvar_hessian(
    rhs: RHS,
    jacobian: Jacobian,
    x0: Array,
    b_cov: Array,
    observation_times: list[float],
    h_op: Array,
    r_cov: Array,
    dt: float = 0.01,
    **params: float,
) -> Array:
    r"""The Gauss-Newton Hessian of the 4D-Var cost at ``x0``.

    .. math::
        \mathbf{A}^{-1} = \mathbf{B}^{-1}
          + \sum_k \mathbf{M}_k^{\top}\mathbf{H}^{\top}\mathbf{R}^{-1}
                   \mathbf{H}\mathbf{M}_k

    Its inverse :math:`\mathbf{A}` is the analysis-error covariance under the
    usual Gaussian assumptions, and two properties of it carry the chapter.

    **It does not depend on the observed values.** Only on where and when the
    observations are taken, and on the trajectory. So the analysis uncertainty
    can be computed *before* any observation exists, which is what makes
    observation-targeting studies (chapter 16) possible at all.

    **It is flow-dependent even though** :math:`\mathbf{B}` **is not.** The
    propagators carry the local dynamics into it, so the analysis-error
    covariance rotates with the flow. The 3D-Var Hessian, by contrast, is
    :math:`\mathbf{B}^{-1}+\mathbf{H}^{\top}\mathbf{R}^{-1}\mathbf{H}` at every
    point of the attractor and at every time, forever. This is the structural
    difference between the two schemes, and it survives the fact that they
    minimise the same-looking cost function.

    Gauss-Newton because the term involving the model's *second* derivative is
    dropped. That term is what can make the true Hessian indefinite once the
    window is long enough for nonlinearity to matter; dropping it is precisely
    what keeps the incremental inner problem convex, and it is an approximation,
    not an identity.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    h_op = np.atleast_2d(np.asarray(h_op, dtype=float))
    r_inv = np.linalg.inv(np.atleast_2d(np.asarray(r_cov, dtype=float)))
    hessian = np.linalg.inv(np.atleast_2d(np.asarray(b_cov, dtype=float)))

    for t_raw in observation_times:
        t_obs = float(t_raw)
        propagator = tangent_linear_propagator(
            rhs, jacobian, x0, max(t_obs, 0.0), dt=dt, **params
        )
        forward = h_op @ propagator
        hessian = hessian + forward.T @ (r_inv @ forward)
    return hessian


def incremental_four_dvar(
    rhs: RHS,
    jacobian: Jacobian,
    xb: Array,
    b_cov: Array,
    observations: list[tuple[float, Array]],
    h_op: Array,
    r_cov: Array,
    dt: float = 0.01,
    outer_iterations: int = 3,
    **params: float,
) -> tuple[Array, list[float]]:
    r"""Incremental 4D-Var: a short sequence of quadratic inner problems.

    The operational formulation. Rather than minimising the nonlinear cost
    directly, linearise about the current trajectory and minimise for an
    *increment*:

    .. math::
        J(\delta x) = \tfrac12(\delta x + x^{(k)} - x^b)^{\top}\mathbf{B}^{-1}
                      (\cdots)
          + \tfrac12\sum_k (d_k - \mathbf{H}\mathbf{M}_k\delta x)^{\top}
            \mathbf{R}^{-1}(\cdots),

    which is quadratic, hence convex, hence solvable by a linear method whose
    convergence does not depend on how nonlinear the model is. The outer loop
    re-linearises and repeats.

    Written out, this is exactly **Gauss-Newton**: the increment solves
    :math:`\mathbf{A}^{-1}\delta x = -\nabla J(x^{(k)})` with
    :math:`\mathbf{A}^{-1}` the Hessian of :func:`four_dvar_hessian`, because
    the right-hand side of the inner normal equations *is* minus the outer
    gradient. Recognising that is worth more than memorising the incremental
    algorithm: it says at once that one outer iteration is exact for a linear
    model, that convergence is quadratic near the solution, and that the whole
    scheme inherits Gauss-Newton's failure mode when the dropped
    second-derivative term is large.

    Here the inner problem is solved by a direct factorisation, which is honest
    at :math:`n=3` and impossible at :math:`n=10^8`; operationally the inner
    loop is itself an iterative conjugate-gradient minimisation that never forms
    :math:`\mathbf{A}^{-1}` and never applies :math:`\mathbf{M}` more than a few
    dozen times.

    Returns ``(xa, costs)``, where ``costs`` holds :math:`J` at the start of each
    outer iteration and once more at the analysis -- so it has
    ``outer_iterations + 1`` entries and its decrease is the convergence trace.
    """
    x = np.asarray(xb, dtype=float).ravel().copy()
    xb = np.asarray(xb, dtype=float).ravel()
    times = [float(t) for t, _ in observations]
    costs: list[float] = []

    for _ in range(int(outer_iterations)):
        cost, grad = four_dvar_cost(
            rhs, jacobian, x, xb, b_cov, observations, h_op, r_cov,
            dt=dt, **params,
        )
        costs.append(cost)
        if not np.isfinite(cost):
            break
        hessian = four_dvar_hessian(
            rhs, jacobian, x, b_cov, times, h_op, r_cov, dt=dt, **params
        )
        x = x + np.linalg.solve(hessian, -grad)

    final, _ = four_dvar_cost(
        rhs, jacobian, x, xb, b_cov, observations, h_op, r_cov, dt=dt, **params
    )
    costs.append(final)
    return x, costs


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
    history: list[float] | None = None,
    **params: float,
) -> Array:
    r"""Strong-constraint 4D-Var over an assimilation window.

    Minimises :func:`four_dvar_cost` from the background, returning the analysed
    initial state. Pass a list as ``history`` to collect :math:`J` at each
    iterate.

    Minimised with L-BFGS-B on the analytic gradient. Fixed-step steepest
    descent is not a viable substitute here: with tight observation errors
    :math:`\mathbf{R}^{-1}` makes the gradient large, and any step size big
    enough to converge in a reasonable number of iterations sends the Lorenz
    trajectory to overflow. Real systems use a quasi-Newton or conjugate-gradient
    minimiser for exactly this reason -- or the Gauss-Newton outer loop of
    :func:`incremental_four_dvar`.
    """
    from scipy.optimize import minimize

    xb = np.asarray(xb, dtype=float).ravel()

    def cost_and_gradient(x0: Array) -> tuple[float, Array]:
        value, grad = four_dvar_cost(
            rhs, jacobian, x0, xb, b_cov, observations, h_op, r_cov,
            dt=dt, **params,
        )
        if history is not None:
            history.append(value)
        return value, grad

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
