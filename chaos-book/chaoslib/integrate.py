"""Time integration for the book's low-order systems.

Two entry points, deliberately:

* :func:`rk4` -- a fixed-step, fully vectorised Runge-Kutta 4. Used wherever an
  *ensemble* is integrated, because it steps all members simultaneously with no
  Python loop over members, and wherever a fixed, uniform time grid matters
  (Lyapunov algorithms, error-growth curves).
* :func:`solve` -- a thin wrapper over :func:`scipy.integrate.solve_ivp` for
  single trajectories where adaptive stepping and tight tolerances are what we
  want (phase portraits, exact-period checks).

Both return trajectories with **time on the leading axis**, so
``traj[i]`` is the state at ``t[i]`` and ``traj[..., k]`` is component ``k``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]
RHS = Callable[..., Array]

__all__ = ["rk4", "rk4_stochastic", "solve", "trajectory_grid"]


def trajectory_grid(t_final: float, dt: float) -> Array:
    """Uniform time grid ``[0, t_final]`` with step as close to ``dt`` as fits.

    The number of steps is rounded, then ``dt`` is recomputed so the grid lands
    exactly on ``t_final`` -- keeping later diagnostics (doubling times, growth
    rates) free of a ragged final interval.
    """
    n_steps = max(1, int(round(t_final / dt)))
    return np.linspace(0.0, t_final, n_steps + 1)


def rk4(
    rhs: RHS,
    x0: Array,
    t: Array,
    **params: float,
) -> Array:
    """Classical RK4 on a prescribed time grid, vectorised over leading axes.

    Parameters
    ----------
    rhs
        Right-hand side ``f(t, x, **params)`` from :mod:`chaoslib.systems`.
    x0
        Initial state, shape ``(..., n)``. Any leading axes are treated as
        independent ensemble members and stepped simultaneously.
    t
        Strictly increasing time grid, shape ``(n_t,)``. Steps may be uneven.

    Returns
    -------
    Array
        Shape ``(n_t, *x0.shape)`` -- time on the leading axis.

    Fixed-step RK4 is chosen over an adaptive solver for ensembles because every
    member must sit on the *same* time grid for spread and error diagnostics to
    mean anything, and because a Python-level loop over members would dominate
    the cost in Pyodide.
    """
    x0 = np.asarray(x0, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.empty((t.size, *x0.shape), dtype=float)
    out[0] = x0
    x = x0
    for i in range(t.size - 1):
        h = t[i + 1] - t[i]
        k1 = rhs(t[i], x, **params)
        k2 = rhs(t[i] + 0.5 * h, x + 0.5 * h * k1, **params)
        k3 = rhs(t[i] + 0.5 * h, x + 0.5 * h * k2, **params)
        k4 = rhs(t[i] + h, x + h * k3, **params)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i + 1] = x
    return out


def solve(
    rhs: RHS,
    x0: Array,
    t: Array,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    method: str = "RK45",
    **params: float,
) -> Array:
    """Adaptive integration of a single trajectory, sampled on ``t``.

    Tight default tolerances: several diagnostics in this book (finite-time
    Lyapunov fits, energy-conservation checks) measure quantities that a loose
    solver would swamp with truncation error.

    Returns shape ``(n_t, n)`` -- time on the leading axis, matching
    :func:`rk4`.
    """
    from scipy.integrate import solve_ivp

    t = np.asarray(t, dtype=float)
    # Bind params as KEYWORDS in a closure rather than passing solve_ivp's
    # positional `args`: the latter would silently depend on dict ordering
    # matching the RHS signature order.
    def _f(tt: float, xx: Array) -> Array:
        return rhs(tt, xx, **params)

    sol = solve_ivp(
        _f,
        (float(t[0]), float(t[-1])),
        np.asarray(x0, dtype=float).ravel(),
        t_eval=t,
        rtol=rtol,
        atol=atol,
        method=method,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.y.T


def rk4_stochastic(
    rhs: RHS,
    x0: Array,
    t: Array,
    noise_std: float = 0.0,
    seed: int | None = 0,
    **params: float,
) -> Array:
    r"""Fixed-step integration of :math:`dx = f(x)\,dt + \sigma\,dW`.

    RK4 on the drift, Euler-Maruyama on the noise: each step adds
    :math:`\sigma\sqrt{\Delta t}\,\xi` with :math:`\xi` standard normal,
    independently per component and per member. Leading axes of ``x0`` are
    ensemble members, as in :func:`rk4`.

    ``noise_std`` is the diffusion coefficient :math:`\sigma`, in state units
    per square root of time -- **not** per unit time, which is the usual place
    to go wrong. Doubling it quadruples the variance injected per unit time.

    With ``noise_std=0`` this reduces to :func:`rk4` exactly, bit for bit,
    which the tests assert -- so it can be used for both arms of a
    perfect-model/imperfect-model comparison without the deterministic arm
    picking up a different discretisation.

    **The noise term is first-order** while the drift is fourth. That is the
    standard trade-off for additive noise and it is adequate when the noise
    amplitude is the quantity being varied, as in chapter 21, but it means the
    step cannot be enlarged on the strength of RK4's accuracy alone. For
    Ornstein-Uhlenbeck, :math:`dx = -ax\,dt + \sigma\,dW`, the exact stationary
    variance is :math:`\sigma^2/2a`; the tests check the integrator reproduces
    it, which is what pins the :math:`\sqrt{\Delta t}` scaling.
    """
    x = np.asarray(x0, dtype=float).copy()
    t = np.asarray(t, dtype=float)
    out = np.empty((t.size, *x.shape), dtype=float)
    out[0] = x
    rng = np.random.default_rng(seed)
    sigma = float(noise_std)

    for i in range(t.size - 1):
        h = t[i + 1] - t[i]
        k1 = rhs(t[i], x, **params)
        k2 = rhs(t[i] + 0.5 * h, x + 0.5 * h * k1, **params)
        k3 = rhs(t[i] + 0.5 * h, x + 0.5 * h * k2, **params)
        k4 = rhs(t[i] + h, x + h * k3, **params)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if sigma != 0.0:
            x = x + sigma * np.sqrt(h) * rng.normal(size=x.shape)
        out[i + 1] = x
    return out

