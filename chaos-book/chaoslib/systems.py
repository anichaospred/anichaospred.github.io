"""Right-hand sides of the low-order dynamical systems used throughout the book.

Every continuous-time system is exposed as ``f(t, x, **params) -> dx/dt`` so it
drops straight into :func:`scipy.integrate.solve_ivp` and into
:mod:`chaoslib.integrate`. Discrete maps are exposed as ``f(x, **params) ->
x_next``.

Conventions (see NOTATION.md):

* State vectors are 1-D ``numpy`` arrays; time is a scalar.
* The signature keeps ``t`` first even for autonomous systems, so the whole
  module is uniform and ``solve_ivp``-compatible.
* Every function is vectorised over a leading ensemble axis where that costs
  nothing, so an ensemble of ``N`` members integrates without a Python loop.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]

__all__ = [
    "lorenz63",
    "lorenz63_jacobian",
    "lorenz63_fixed_points",
    "lorenz63_hopf_rho",
    "lorenz96",
    "lorenz96_jacobian",
    "pendulum",
    "pendulum_energy",
    "pendulum_period_exact",
    "double_pendulum",
    "double_pendulum_energy",
    "logistic_map",
    "logistic_map_derivative",
    "sine_map",
    "sine_map_derivative",
    "cubic_map",
    "cubic_map_derivative",
    "henon_map",
]


# --------------------------------------------------------------------------
# Lorenz (1963)
# --------------------------------------------------------------------------
def lorenz63(
    t: float,
    x: Array,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> Array:
    r"""Lorenz (1963) convection model.

    .. math::
        \dot X = \sigma (Y - X), \quad
        \dot Y = X(\rho - Z) - Y, \quad
        \dot Z = XY - \beta Z

    Physically, :math:`X` measures the intensity of the convective overturning,
    :math:`Y` the temperature difference between ascending and descending
    branches, and :math:`Z` the departure of the vertical temperature profile
    from linear. Time is in "model time units" (MTU); for the classical
    parameters 1 MTU is conventionally read as roughly 5 atmospheric days.

    Accepts either a single state of shape ``(3,)`` or an ensemble of shape
    ``(..., 3)``; the last axis is the state axis.
    """
    x = np.asarray(x, dtype=float)
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    return np.stack(
        [sigma * (Y - X), X * (rho - Z) - Y, X * Y - beta * Z], axis=-1
    )


def lorenz63_jacobian(
    x: Array,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> Array:
    """Jacobian :math:`\\partial f_i/\\partial x_j` of :func:`lorenz63`.

    Needed by the tangent-linear/adjoint machinery and by the Benettin
    algorithm. Note ``trace(J) = -(sigma + 1 + beta)`` for every state, which is
    why the Lyapunov exponents of this system must sum to exactly that value --
    a strong independent check on any spectrum calculation.
    """
    x = np.asarray(x, dtype=float)
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    zero = np.zeros_like(X)
    one = np.ones_like(X)
    return np.stack(
        [
            np.stack([-sigma * one, sigma * one, zero], axis=-1),
            np.stack([rho - Z, -one, -X], axis=-1),
            np.stack([Y, X, -beta * one], axis=-1),
        ],
        axis=-2,
    )


def lorenz63_fixed_points(
    rho: float = 28.0, beta: float = 8.0 / 3.0
) -> tuple[Array, Array | None, Array | None]:
    r"""The fixed points of :func:`lorenz63`: the origin and :math:`C^\pm`.

    The origin is always a fixed point. For :math:`\rho > 1` a symmetric pair
    appears at

    .. math::
        C^\pm = \bigl(\pm\sqrt{\beta(\rho-1)},\; \pm\sqrt{\beta(\rho-1)},\;
                      \rho-1\bigr),

    representing steady clockwise and counter-clockwise convective rolls.
    Returns ``(origin, C_plus, C_minus)``, with the pair ``None`` when
    :math:`\rho \le 1`.
    """
    origin = np.zeros(3)
    if rho <= 1.0:
        return origin, None, None
    s = float(np.sqrt(beta * (rho - 1.0)))
    return origin, np.array([s, s, rho - 1.0]), np.array([-s, -s, rho - 1.0])


def lorenz63_hopf_rho(sigma: float = 10.0, beta: float = 8.0 / 3.0) -> float:
    r"""Critical :math:`\rho` at which :math:`C^\pm` lose stability (Hopf).

    .. math::
        \rho_H = \frac{\sigma(\sigma + \beta + 3)}{\sigma - \beta - 1}

    For the classical :math:`\sigma=10,\ \beta=8/3` this gives
    :math:`\rho_H \approx 24.74`: above it the convective rolls are unstable and
    the trajectory wanders chaotically between them.
    """
    return sigma * (sigma + beta + 3.0) / (sigma - beta - 1.0)


# --------------------------------------------------------------------------
# Lorenz (1996)
# --------------------------------------------------------------------------
def lorenz96(t: float, x: Array, forcing: float = 8.0) -> Array:
    r"""Lorenz (1996) model on a cyclic chain of :math:`N` sites.

    .. math::
        \dot x_k = -x_{k-2} x_{k-1} + x_{k-1} x_{k+1} - x_k + F

    A crude but much-used analogue of a mid-latitude latitude circle: the
    quadratic terms conserve energy and mimic advection, :math:`-x_k` is
    dissipation, and :math:`F` is the forcing. Indices are cyclic. With
    :math:`N=40, F=8` the system is chaotic and 1 time unit is conventionally
    read as 5 atmospheric days.

    Accepts shape ``(N,)`` or ``(..., N)``; the last axis is the site axis, so
    an ensemble integrates without a Python loop.
    """
    x = np.asarray(x, dtype=float)
    return (
        (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1))
        * np.roll(x, 1, axis=-1)
        - x
        + forcing
    )


def lorenz96_jacobian(x: Array, forcing: float = 8.0) -> Array:
    """Jacobian of :func:`lorenz96` for a single state of shape ``(N,)``.

    Built explicitly (rather than by finite differences) so the tangent-linear
    and adjoint models are exact. Only four entries per row are non-zero.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    j = np.zeros((n, n))
    idx = np.arange(n)
    # d/dx_k of ( (x_{k+1} - x_{k-2}) * x_{k-1} - x_k + F )
    j[idx, (idx - 2) % n] = -x[(idx - 1) % n]
    j[idx, (idx - 1) % n] = x[(idx + 1) % n] - x[(idx - 2) % n]
    j[idx, idx] = -1.0
    j[idx, (idx + 1) % n] = x[(idx - 1) % n]
    return j


# --------------------------------------------------------------------------
# Pendulums
# --------------------------------------------------------------------------
def pendulum(t: float, y: Array, g: float = 9.81, length: float = 1.0) -> Array:
    r"""Exact (nonlinear) simple pendulum, state :math:`(\theta, \omega)`.

    .. math:: \ddot\theta = -(g/L)\sin\theta

    One degree of freedom, so the phase space is 2-D and energy conservation
    confines motion to a 1-D level curve. By Poincare-Bendixson this system can
    therefore never be chaotic, no matter how nonlinear -- the point chapter 4
    is built around.
    """
    y = np.asarray(y, dtype=float)
    theta, omega = y[..., 0], y[..., 1]
    return np.stack([omega, -(g / length) * np.sin(theta)], axis=-1)


def pendulum_energy(
    y: Array, g: float = 9.81, length: float = 1.0, mass: float = 1.0
) -> Array:
    r"""Total energy per pendulum, :math:`\tfrac12 m L^2\omega^2 + mgL(1-\cos\theta)`.

    Measured with the pivot as the datum. Conserved exactly by the true
    dynamics, so its drift is a direct diagnostic of integrator error.
    """
    y = np.asarray(y, dtype=float)
    theta, omega = y[..., 0], y[..., 1]
    return 0.5 * mass * length**2 * omega**2 + mass * g * length * (
        1.0 - np.cos(theta)
    )


def pendulum_period_exact(
    theta0: float, g: float = 9.81, length: float = 1.0
) -> float:
    r"""Exact period of a pendulum released from rest at :math:`\theta_0`.

    .. math:: T = 4\sqrt{L/g}\; K\!\left(\sin^2(\theta_0/2)\right)

    with :math:`K` the complete elliptic integral of the first kind (SciPy's
    ``ellipk`` takes the parameter :math:`m = k^2`, not the modulus :math:`k` --
    a classic source of factor-of-two errors). As
    :math:`\theta_0 \to 0` this tends to the small-angle result
    :math:`2\pi\sqrt{L/g}`.
    """
    from scipy.special import ellipk

    return float(
        4.0 * np.sqrt(length / g) * ellipk(np.sin(theta0 / 2.0) ** 2)
    )


def double_pendulum(
    t: float,
    y: Array,
    g: float = 9.81,
    l1: float = 1.0,
    l2: float = 1.0,
    m1: float = 1.0,
    m2: float = 1.0,
) -> Array:
    r"""Double pendulum, state :math:`(\theta_1, \theta_2, \omega_1, \omega_2)`.

    The full Euler-Lagrange equations for two point masses on rigid massless
    rods. Two degrees of freedom give a 4-D phase space; energy conservation
    reduces it to 3-D, which is the minimum dimension in which the
    stretch-and-fold mechanism of chaos can operate. Hence: one pendulum never
    chaotic, two pendulums chaotic at sufficient energy.
    """
    y = np.asarray(y, dtype=float)
    th1, th2, w1, w2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    delta = th1 - th2
    sin_d, cos_d = np.sin(delta), np.cos(delta)
    denom = 2.0 * m1 + m2 - m2 * np.cos(2.0 * delta)

    dw1 = (
        -g * (2.0 * m1 + m2) * np.sin(th1)
        - m2 * g * np.sin(th1 - 2.0 * th2)
        - 2.0 * sin_d * m2 * (w2**2 * l2 + w1**2 * l1 * cos_d)
    ) / (l1 * denom)
    dw2 = (
        2.0
        * sin_d
        * (
            w1**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(th1)
            + w2**2 * l2 * m2 * cos_d
        )
    ) / (l2 * denom)
    return np.stack([w1, w2, dw1, dw2], axis=-1)


def double_pendulum_energy(
    y: Array,
    g: float = 9.81,
    l1: float = 1.0,
    l2: float = 1.0,
    m1: float = 1.0,
    m2: float = 1.0,
) -> Array:
    """Total energy of the double pendulum, with the pivot as the datum.

    Conserved by the true dynamics; used as the integrator-accuracy check and
    as the control parameter separating regular from chaotic motion.
    """
    y = np.asarray(y, dtype=float)
    th1, th2, w1, w2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    kinetic = 0.5 * m1 * (l1 * w1) ** 2 + 0.5 * m2 * (
        (l1 * w1) ** 2
        + (l2 * w2) ** 2
        + 2.0 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)
    )
    potential = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
    return kinetic + potential


# --------------------------------------------------------------------------
# Discrete maps
# --------------------------------------------------------------------------
def logistic_map(x: Array, r: float = 3.9) -> Array:
    r"""Logistic map :math:`x_{n+1} = r\,x_n(1 - x_n)` on :math:`[0,1]`.

    The cheapest system that exhibits the full period-doubling route to chaos,
    with accumulation point :math:`r_\infty \approx 3.5699` and Feigenbaum
    ratio :math:`\delta \approx 4.669`.
    """
    x = np.asarray(x, dtype=float)
    return r * x * (1.0 - x)


def logistic_map_derivative(x: Array, r: float = 3.9) -> Array:
    r"""Derivative :math:`f'(x) = r(1 - 2x)` of the logistic map.

    Needed for the map Lyapunov exponent :math:`\lambda = \langle\ln|f'|\rangle`
    and for cycle multipliers. Note :math:`f'(1/2) = 0`: the critical point is
    where a cycle becomes *superstable*.
    """
    x = np.asarray(x, dtype=float)
    return r * (1.0 - 2.0 * x)


def sine_map(x: Array, r: float = 0.9) -> Array:
    r"""Sine map :math:`x_{n+1} = r\sin(\pi x_n)` on :math:`[0,1]`.

    A unimodal map of a completely different functional form from the logistic
    map, with a different critical exponent structure at the origin -- and
    therefore a genuinely independent test of Feigenbaum universality. Its own
    cascade parameters are unrelated to the logistic map's
    (:math:`R_0 = 1/2` against :math:`R_0 = 2`), yet the *ratios* of successive
    cascade spacings converge to the same :math:`\delta`. Chaotic for
    :math:`r \gtrsim 0.8655`; the map leaves :math:`[0,1]` for :math:`r > 1`.
    """
    x = np.asarray(x, dtype=float)
    return r * np.sin(np.pi * x)


def sine_map_derivative(x: Array, r: float = 0.9) -> Array:
    r"""Derivative :math:`f'(x) = \pi r\cos(\pi x)` of the sine map."""
    x = np.asarray(x, dtype=float)
    return np.pi * r * np.cos(np.pi * x)


def cubic_map(x: Array, r: float = 2.3) -> Array:
    r"""Cubic map :math:`x_{n+1} = r\,x_n(1 - x_n^2)` on :math:`[0,1]`.

    A third unimodal family, included for the same reason as
    :func:`sine_map`: universality is a claim about a *class* of maps, and one
    example cannot demonstrate it. Its critical point is at
    :math:`x_c = 1/\sqrt{3}`, not :math:`1/2`.
    """
    x = np.asarray(x, dtype=float)
    return r * x * (1.0 - x * x)


def cubic_map_derivative(x: Array, r: float = 2.3) -> Array:
    r"""Derivative :math:`f'(x) = r(1 - 3x^2)` of the cubic map."""
    x = np.asarray(x, dtype=float)
    return r * (1.0 - 3.0 * x * x)


def henon_map(xy: Array, a: float = 1.4, b: float = 0.3) -> Array:
    r"""Henon map :math:`(x,y) \mapsto (1 - a x^2 + y,\; b x)`.

    A 2-D invertible map with a strange attractor -- the minimal example of
    stretch-and-fold in a discrete system, and a convenient testbed for
    fractal-dimension estimators.
    """
    xy = np.asarray(xy, dtype=float)
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([1.0 - a * x**2 + y, b * x], axis=-1)
