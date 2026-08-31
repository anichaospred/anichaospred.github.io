"""Lyapunov exponents, finite-time growth rates, and doubling times.

Three distinct things that get conflated in casual usage, and are kept separate
here:

* the **global (asymptotic) spectrum** :math:`\\lambda_1 \\ge \\lambda_2 \\ge
  \\dots`, a property of the attractor -- :func:`lyapunov_spectrum`;
* the **finite-time (local) exponent**, a property of a particular state and
  lead time, which is what actually governs whether *today's* forecast is
  predictable -- :func:`finite_time_exponents`;
* the **twin-trajectory fit**, the cheap empirical estimate obtained by
  perturbing a trajectory and fitting the slope of :math:`\\ln\\|\\delta\\|`
  over its exponential window -- :func:`twin_trajectory_growth`. This is the
  one the pendulum and Lorenz chapters use pedagogically, because the reader can
  see the line being fitted.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from chaoslib.integrate import rk4

Array = NDArray[np.floating]
RHS = Callable[..., Array]
Jacobian = Callable[..., Array]

__all__ = [
    "lyapunov_spectrum",
    "finite_time_exponents",
    "twin_trajectory_growth",
    "fit_growth_rate",
    "doubling_time",
    "kaplan_yorke_dimension",
    "ks_entropy",
]


def lyapunov_spectrum(
    rhs: RHS,
    jacobian: Jacobian,
    x0: Array,
    dt: float = 0.01,
    t_final: float = 1000.0,
    t_transient: float = 50.0,
    n_exponents: int | None = None,
    **params: float,
) -> Array:
    r"""Full Lyapunov spectrum by the Benettin algorithm.

    Integrates the trajectory together with a set of tangent vectors, and
    re-orthonormalises the tangent set (Gram-Schmidt, here via a QR
    decomposition) at every step. The time-averaged logarithms of the diagonal
    of :math:`R` converge to the exponents:

    .. math::
        \lambda_i = \lim_{T\to\infty} \frac{1}{T}\sum_n \ln |R_{ii}^{(n)}|

    Re-orthonormalisation is not optional: without it every tangent vector
    collapses onto the leading direction and floating point overflows within a
    few Lyapunov times.

    The transient is discarded before accumulating, so ``x0`` need not lie on the
    attractor. Returns the exponents in descending order, in units of
    inverse time.

    A free correctness check for any dissipative system: the exponents must sum
    to the (state-independent, where it is constant) trace of the Jacobian --
    :math:`-(\sigma+1+\beta)` for Lorenz 63.
    """
    x = np.asarray(x0, dtype=float).ravel()
    n = x.size
    k = n if n_exponents is None else int(n_exponents)

    # Advance onto the attractor first; tangent vectors are irrelevant here.
    if t_transient > 0.0:
        n_tr = max(1, int(round(t_transient / dt)))
        x = rk4(rhs, x, np.arange(n_tr + 1) * dt, **params)[-1]

    q = np.eye(n)[:, :k]
    totals = np.zeros(k)
    n_steps = max(1, int(round(t_final / dt)))

    for _ in range(n_steps):
        # State and tangent advance together through the same RK4 stages, so the
        # tangent propagator is the exact derivative of the numerical map.
        x, q = _rk4_step_with_tangent(rhs, jacobian, x, q, dt, **params)

        q, r = np.linalg.qr(q)
        diag = np.abs(np.diag(r))
        # A zero on the diagonal means the tangent set has degenerated; guard so
        # a single bad step does not poison the whole average with -inf.
        diag = np.where(diag > 0.0, diag, np.finfo(float).tiny)
        totals += np.log(diag)

    return totals / (n_steps * dt)


def _rk4_step_with_tangent(
    rhs: RHS,
    jacobian: Jacobian,
    x: Array,
    v: Array,
    dt: float,
    **params: float,
) -> tuple[Array, Array]:
    """One RK4 step of the state AND of its tangent, through the same stages.

    This differentiates the *discrete* RK4 map rather than the continuous flow:
    each tangent stage uses the Jacobian evaluated at the corresponding
    intermediate state,

    .. code-block:: text

        K1 = J(x)               V
        K2 = J(x + h/2 k1)     (V + h/2 K1)
        K3 = J(x + h/2 k2)     (V + h/2 K2)
        K4 = J(x + h k3)       (V + h  K3)

    so the propagator returned is the exact Jacobian of the numerical map the
    nonlinear model actually takes. Freezing ``J`` at the start of the step
    instead -- the tempting simplification -- leaves an O(dt) inconsistency
    between the tangent and nonlinear models that no amount of reducing the
    perturbation amplitude removes; it shows up as a *floor* in the
    finite-difference check of :func:`chaoslib.adjoint.tangent_linear_error`.
    """
    k1 = rhs(0.0, x, **params)
    k2 = rhs(0.0, x + 0.5 * dt * k1, **params)
    k3 = rhs(0.0, x + 0.5 * dt * k2, **params)
    k4 = rhs(0.0, x + dt * k3, **params)

    j1 = jacobian(x, **params)
    j2 = jacobian(x + 0.5 * dt * k1, **params)
    j3 = jacobian(x + 0.5 * dt * k2, **params)
    j4 = jacobian(x + dt * k3, **params)

    cap1 = j1 @ v
    cap2 = j2 @ (v + 0.5 * dt * cap1)
    cap3 = j3 @ (v + 0.5 * dt * cap2)
    cap4 = j4 @ (v + dt * cap3)

    x_new = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    v_new = v + (dt / 6.0) * (cap1 + 2.0 * cap2 + 2.0 * cap3 + cap4)
    return x_new, v_new


def finite_time_exponents(
    rhs: RHS,
    jacobian: Jacobian,
    states: Array,
    tau: float,
    dt: float = 0.01,
    **params: float,
) -> Array:
    r"""Leading finite-time Lyapunov exponent over lead time ``tau``.

    For each state in ``states`` (shape ``(m, n)``), propagates the tangent
    linear model for ``tau`` and returns

    .. math::
        \lambda(x, \tau) = \frac{1}{\tau}\,
            \ln \|\mathbf{M}(x,\tau)\|_2 ,

    the largest singular value of the propagator. This is the quantity that
    varies by a factor of several around a single attractor and is the honest
    basis for "flow-dependent predictability": the asymptotic
    :math:`\lambda_1` describes the attractor, not today's forecast.
    """
    states = np.atleast_2d(np.asarray(states, dtype=float))
    n_steps = max(1, int(round(tau / dt)))
    out = np.empty(states.shape[0])

    for i, x0 in enumerate(states):
        x = x0.copy()
        m = np.eye(x.size)
        for _ in range(n_steps):
            x, m = _rk4_step_with_tangent(rhs, jacobian, x, m, dt, **params)
        out[i] = np.log(np.linalg.svd(m, compute_uv=False)[0]) / (n_steps * dt)
    return out


def twin_trajectory_growth(
    rhs: RHS,
    x0: Array,
    delta0: float,
    t: Array,
    seed: int | None = 42,
    **params: float,
) -> tuple[Array, Array]:
    """Separation between a control run and a minimally perturbed twin.

    Returns ``(separation, perturbed_initial_state)``. The perturbation is an
    isotropic random vector of exactly length ``delta0`` -- random rather than
    aligned with the leading Lyapunov vector, because that is the honest analogue
    of an analysis error, and because the reader should see the initial
    *non*-exponential phase while the perturbation rotates into the growing
    subspace.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float).ravel()
    direction = rng.normal(size=x0.size)
    direction /= np.linalg.norm(direction)
    x0_pert = x0 + delta0 * direction

    pair = rk4(rhs, np.stack([x0, x0_pert]), t, **params)
    separation = np.linalg.norm(pair[:, 1] - pair[:, 0], axis=-1)
    return separation, x0_pert


def fit_growth_rate(
    t: Array,
    separation: Array,
    lower_mult: float = 10.0,
    upper_frac: float = 0.1,
) -> tuple[float, float]:
    r"""Fit :math:`\ln\|\delta\| = \lambda t + c` over the exponential window.

    The window is bounded **below** by ``lower_mult`` times the *initial*
    separation -- early on the perturbation is still rotating into the growing
    subspace, and its norm can even shrink -- and **above** by ``upper_frac``
    times the largest separation reached, since nonlinear saturation bends the
    curve over.

    Anchoring the lower bound to :math:`\delta_0` rather than to saturation is
    the important part. A window expressed purely as a fraction of saturation
    (say 5--50 %) sits in the *nonlinear* phase whenever :math:`\delta_0` is
    tiny, and returns a growth rate far below :math:`\lambda_1`.

    Note what this function does and does not measure: one twin pair yields a
    *finite-time* exponent along one trajectory. For Lorenz 63 individual
    estimates scatter over roughly 0.5--1.3 even for a 12-MTU record, because
    local growth is bursty. Recovering the asymptotic :math:`\lambda_1` requires
    averaging over initial conditions (or using
    :func:`lyapunov_spectrum`, which does the time-averaging properly).

    Returns ``(lambda, intercept)``. Raises if fewer than three points fall in
    the window rather than returning a meaningless fit.
    """
    t = np.asarray(t, dtype=float)
    sep = np.asarray(separation, dtype=float)
    positive = sep[sep > 0.0]
    if positive.size == 0:
        raise ValueError("separation is zero everywhere")
    lo = lower_mult * float(positive[0])
    hi = upper_frac * float(np.nanmax(sep))
    mask = (sep > lo) & (sep < hi)
    if mask.sum() < 3:
        raise ValueError(
            "exponential window contains <3 points; loosen `lower_mult`/"
            "`upper_frac` or lengthen the integration"
        )
    slope, intercept = np.polyfit(t[mask], np.log(sep[mask]), 1)
    return float(slope), float(intercept)


def doubling_time(growth_rate: float) -> float:
    r"""Error-doubling time :math:`\ln 2/\lambda` for a growth rate.

    Only meaningful while growth is exponential; once errors approach
    saturation the doubling time is no longer a useful summary of the forecast
    horizon (see :mod:`chaoslib.errorgrowth`).
    """
    if growth_rate <= 0.0:
        return float("inf")
    return float(np.log(2.0) / growth_rate)


def kaplan_yorke_dimension(exponents: Array) -> float:
    r"""Kaplan-Yorke (Lyapunov) dimension from a spectrum.

    .. math::
        D_{KY} = j + \frac{\sum_{i=1}^{j}\lambda_i}{|\lambda_{j+1}|}

    where :math:`j` is the largest index whose partial sum is still
    non-negative. Interpolates between integer dimensions: for Lorenz 63 it
    gives :math:`\approx 2.06`, the quantitative sense in which the strange
    attractor is "a surface, but not quite".
    """
    lam = np.sort(np.asarray(exponents, dtype=float))[::-1]
    cumulative = np.cumsum(lam)
    j = int(np.searchsorted(-cumulative, 0.0, side="left"))
    if j == 0:
        return 0.0
    if j >= lam.size:
        return float(lam.size)
    return float(j + cumulative[j - 1] / abs(lam[j]))


def ks_entropy(exponents: Array) -> float:
    r"""Kolmogorov-Sinai entropy as the sum of positive exponents.

    .. math:: h_{KS} = \sum_{\lambda_i > 0} \lambda_i

    By Pesin's identity this is the rate at which the system generates new
    information -- equivalently, the rate at which knowledge of the initial
    state is destroyed. Units: nats per unit time.
    """
    lam = np.asarray(exponents, dtype=float)
    return float(lam[lam > 0.0].sum())
