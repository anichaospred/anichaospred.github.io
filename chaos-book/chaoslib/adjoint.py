"""Tangent-linear and adjoint models, and the optimal perturbations they give.

The operational core of predictability work. Given a nonlinear model
:math:`x(t+\\tau) = \\mathcal{M}(x(t))`, the **tangent linear model** (TLM)
propagates a small perturbation,

.. math:: \\delta x(t+\\tau) \\approx \\mathbf{M}\\,\\delta x(t),

and its **adjoint** :math:`\\mathbf{M}^{\\top}` propagates a *sensitivity*
backwards in time. Two things follow, and they are the subject of chapters
15-16:

* the gradient of any scalar forecast metric with respect to the initial state
  costs one adjoint integration, not one per degree of freedom -- which is what
  makes 4D-Var and adjoint sensitivity feasible at all;
* the leading **singular vectors** of :math:`\\mathbf{M}` are the perturbations
  that grow fastest over a *finite* time :math:`\\tau`, and they are generally
  *not* the leading Lyapunov vectors (which describe asymptotic growth).

For the low-order systems in this book :math:`\\mathbf{M}` is small enough to
form explicitly, so the adjoint really is the transpose and the reader can
verify the identity numerically instead of taking it on faith.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from chaoslib.integrate import rk4
from chaoslib.lyapunov import _rk4_step_with_tangent

Array = NDArray[np.floating]
RHS = Callable[..., Array]
Jacobian = Callable[..., Array]

__all__ = [
    "tangent_linear_propagator",
    "adjoint_propagator",
    "adjoint_identity_residual",
    "tangent_linear_error",
    "singular_vectors",
    "gradient_test",
    "leading_singular_value",
]


def tangent_linear_propagator(
    rhs: RHS,
    jacobian: Jacobian,
    x0: Array,
    tau: float,
    dt: float = 0.01,
    **params: float,
) -> Array:
    r"""Build the TLM propagator :math:`\mathbf{M}(x_0, \tau)` explicitly.

    Integrates the identity matrix through the linearised flow along the
    *nonlinear* trajectory starting at ``x0`` -- the linearisation point moves,
    which is exactly what makes :math:`\mathbf{M}` state-dependent and hence
    predictability flow-dependent.

    Returns an ``(n, n)`` matrix. Columns are the images of the unit
    perturbations, so ``M @ dx`` is the propagated perturbation.
    """
    x = np.asarray(x0, dtype=float).ravel()
    m = np.eye(x.size)
    if tau <= 0.0:
        # A zero-length window propagates nothing: the propagator is the
        # identity. Forcing a step here (the obvious `max(1, ...)`) silently
        # advances the tangent by one dt and corrupts every gradient that
        # includes an observation at the window start -- which, in cycling
        # 4D-Var, is the normal case.
        return m
    n_steps = max(1, int(round(tau / dt)))
    # Step with tau/n_steps rather than dt, so the propagator covers EXACTLY
    # tau even when tau is not an integer multiple of dt. Using dt directly
    # makes M correspond to n_steps*dt, a different interval from the one the
    # nonlinear model was integrated over.
    step = tau / n_steps
    for _ in range(n_steps):
        x, m = _rk4_step_with_tangent(rhs, jacobian, x, m, step, **params)
    return m


def adjoint_propagator(propagator: Array) -> Array:
    r"""The adjoint of a TLM propagator: its transpose (real Euclidean inner product).

    Stated as a named function rather than an inline ``.T`` because the *concept*
    is the point: the adjoint is defined by the identity
    :math:`\langle \mathbf{M}x, y\rangle = \langle x, \mathbf{M}^{\top}y\rangle`,
    and it is only the transpose because we have chosen the plain Euclidean inner
    product. Under a weighted norm (energy, say) the adjoint acquires the weight
    matrices, which is where real adjoint models get subtle.
    """
    return np.asarray(propagator, dtype=float).T


def adjoint_identity_residual(
    propagator: Array, n_trials: int = 32, seed: int | None = 0
) -> float:
    r"""Max relative violation of the adjoint identity over random vector pairs.

    Checks :math:`\langle \mathbf{M}x, y\rangle = \langle x,
    \mathbf{M}^{\top}y\rangle` for random ``x``, ``y``. Should come out at
    machine precision; anything larger means the adjoint is not the adjoint of
    the TLM actually being used -- historically the single most common bug in
    variational assimilation systems, which is why this test is standard
    practice.
    """
    m = np.asarray(propagator, dtype=float)
    mt = adjoint_propagator(m)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        x = rng.normal(size=m.shape[1])
        y = rng.normal(size=m.shape[0])
        lhs = float((m @ x) @ y)
        rhs_val = float(x @ (mt @ y))
        scale = max(abs(lhs), abs(rhs_val), np.finfo(float).tiny)
        worst = max(worst, abs(lhs - rhs_val) / scale)
    return worst


def tangent_linear_error(
    rhs: RHS,
    jacobian: Jacobian,
    x0: Array,
    tau: float,
    amplitudes: Array,
    dt: float = 0.01,
    seed: int | None = 0,
    **params: float,
) -> tuple[Array, Array]:
    r"""Validate the TLM against finite differences of the nonlinear model.

    For each amplitude :math:`\alpha`, compares the nonlinear difference
    :math:`\mathcal{M}(x_0 + \alpha d) - \mathcal{M}(x_0)` with the linear
    prediction :math:`\alpha \mathbf{M} d` and returns the relative discrepancy.

    The textbook signature of a *correct* TLM is that this ratio falls linearly
    in :math:`\alpha` (the neglected term is :math:`O(\alpha^2)`) until it hits
    round-off. A TLM with a sign or index error instead shows an error that
    plateaus at O(1) -- which is why this test catches what an eyeball check of
    the Jacobian does not.

    Returns ``(amplitudes, relative_error)``.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    amps = np.atleast_1d(np.asarray(amplitudes, dtype=float))
    rng = np.random.default_rng(seed)
    d = rng.normal(size=x0.size)
    d /= np.linalg.norm(d)

    t_grid = np.arange(max(1, int(round(tau / dt))) + 1) * dt
    base = rk4(rhs, x0, t_grid, **params)[-1]
    m = tangent_linear_propagator(rhs, jacobian, x0, tau, dt=dt, **params)
    linear = m @ d

    errors = np.empty(amps.size)
    for i, a in enumerate(amps):
        nonlinear = (rk4(rhs, x0 + a * d, t_grid, **params)[-1] - base) / a
        errors[i] = np.linalg.norm(nonlinear - linear) / np.linalg.norm(linear)
    return amps, errors


def singular_vectors(
    propagator: Array, n_vectors: int = 1, weight: Array | None = None
) -> tuple[Array, Array, Array]:
    r"""Leading singular values and vectors of a TLM propagator.

    Returns ``(sigma, initial_vectors, final_vectors)`` where ``sigma`` are the
    largest ``n_vectors`` singular values, ``initial_vectors`` the corresponding
    optimal initial perturbations (as *columns*), and ``final_vectors`` their
    evolved shapes.

    Amplification over the window is :math:`\sigma`, not :math:`e^{\lambda\tau}`:
    for short windows singular-vector growth substantially exceeds what the
    leading Lyapunov exponent would suggest, which is precisely why operational
    centres perturb along singular vectors rather than random directions.

    Parameters
    ----------
    weight
        Optional inner-product weights defining the norm in which growth is
        measured -- a 1-D array of diagonal weights, or a symmetric positive
        definite matrix. ``None`` (the default) means the Euclidean norm.

    **Singular vectors are norm-dependent, and that is not a technicality.**
    "Which perturbation grows fastest" has no answer until you say *fastest in
    what measure*. Under the weighted inner product
    :math:`\langle u,v\rangle_E = u^{\top}\mathbf{E}v` the problem is

    .. math::
        \max_{v}\ \frac{(\mathbf{M}v)^{\top}\mathbf{E}(\mathbf{M}v)}
                          {v^{\top}\mathbf{E}v},

    solved here as the ordinary SVD of
    :math:`\mathbf{E}^{1/2}\mathbf{M}\mathbf{E}^{-1/2}`, with the vectors mapped
    back to state space. Operational centres compute singular vectors in a
    total-energy norm and get different structures from what a Euclidean norm
    would give; the returned ``sigma`` is the amplification *in the norm you
    asked for*.
    """
    m = np.asarray(propagator, dtype=float)
    k = int(n_vectors)

    if weight is None:
        u, s, vt = np.linalg.svd(m)
        return s[:k], vt[:k].T, u[:, :k]

    w = np.asarray(weight, dtype=float)
    if w.ndim == 1:
        if np.any(w <= 0.0):
            raise ValueError("diagonal weights must be positive")
        root = np.diag(np.sqrt(w))
        inv_root = np.diag(1.0 / np.sqrt(w))
    else:
        vals, vecs = np.linalg.eigh(w)
        if np.any(vals <= 0.0):
            raise ValueError("weight matrix must be positive definite")
        root = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        inv_root = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    u, s, vt = np.linalg.svd(root @ m @ inv_root)
    # Map back: a unit vector in the transformed space corresponds to a unit
    # E-norm vector in state space.
    initial = inv_root @ vt[:k].T
    final = inv_root @ u[:, :k]
    return s[:k], initial, final


def gradient_test(
    cost_and_gradient: Callable[[Array], tuple[float, Array]],
    x0: Array,
    direction: Array | None = None,
    alphas: Array | None = None,
) -> tuple[Array, Array]:
    r"""The standard adjoint gradient test.

    .. math::
        \Phi(\alpha) = \frac{J(x_0+\alpha h) - J(x_0)}
                            {\alpha\, h^{\top}\nabla J(x_0)}

    If :math:`\nabla J` really is the gradient of :math:`J`, then
    :math:`\Phi(\alpha)\to 1` as :math:`\alpha\to 0`, since the numerator is
    :math:`\alpha h^{\top}\nabla J + O(\alpha^2)`. This is the test operational
    centres run on an adjoint before they trust it, and it is stronger than
    comparing a few components against finite differences because it exercises
    the whole gradient at once, in one direction, at many magnitudes.

    Plotted as :math:`|\Phi-1|` against :math:`\alpha` on log axes it makes a
    **V**. The right branch falls with slope 1: it is the neglected
    :math:`O(\alpha^2)` curvature, divided by :math:`\alpha`. The left branch
    rises like :math:`1/\alpha`: the numerator is a difference of two nearly
    equal numbers, so its relative error is roughly
    :math:`\varepsilon|J|/(\alpha|h^{\top}\nabla J|)`. The floor between them is
    the best the test can do, and *where* the floor sits is set by how accurately
    :math:`J` itself can be evaluated -- not by the gradient.

    The diagnostic value is in the shape, not the floor. A gradient that is
    merely *nearly* right -- a missing term, a wrong transpose, a factor
    somewhere -- **plateaus**: :math:`|\Phi-1|` descends the curvature branch,
    levels off at the gradient's own relative error, and stays there over many
    decades of :math:`\alpha`, because that discrepancy is a property of the
    gradient rather than of the arithmetic. Only at the very smallest
    :math:`\alpha` does cancellation error reassert itself and lift the curve
    again, so a wrong gradient does eventually turn upward too -- the honest
    discriminant is the **depth** of the trough, not its existence. Measured on
    the chapter-18 configuration, a correct gradient descends over five decades
    to :math:`3\times10^{-8}` while one that is wrong by 1 % sits flat at
    :math:`10^{-2}` across nearly the whole range.

    ``direction`` defaults to a fixed pseudo-random unit vector, so the test is
    reproducible. ``alphas`` defaults to 21 points spanning
    :math:`10^{-14}` to :math:`10^{-1}`.

    Returns ``(alphas, phi)``.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    if direction is None:
        rng = np.random.default_rng(0)
        direction = rng.normal(size=x0.size)
        direction = direction / np.linalg.norm(direction)
    direction = np.asarray(direction, dtype=float).ravel()
    if alphas is None:
        alphas = np.logspace(-14.0, -1.0, 21)
    alphas = np.asarray(alphas, dtype=float).ravel()

    base_cost, base_grad = cost_and_gradient(x0)
    slope = float(direction @ base_grad)
    if slope == 0.0:
        raise ValueError(
            "the directional derivative vanishes; the gradient test is "
            "undefined along this direction -- pick another"
        )

    phi = np.empty(alphas.size)
    for i, alpha in enumerate(alphas):
        perturbed, _ = cost_and_gradient(x0 + alpha * direction)
        phi[i] = (perturbed - base_cost) / (alpha * slope)
    return alphas, phi


def leading_singular_value(propagator: Array) -> float:
    """Largest singular value: the maximum possible perturbation amplification."""
    return float(
        np.linalg.svd(np.asarray(propagator, dtype=float), compute_uv=False)[0]
    )
