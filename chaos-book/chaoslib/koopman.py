r"""The Koopman operator: linear representations of nonlinear dynamics.

Every chapter before this one has treated nonlinearity as the obstacle. The
Koopman viewpoint removes it exactly, and charges for the removal in dimension.

Let :math:`\mathcal{M}^\tau` be the (nonlinear) flow map. For an **observable**
:math:`g` -- any scalar function of the state -- define

.. math:: (\mathcal{K}^\tau g)(x) = g\!\left(\mathcal{M}^\tau(x)\right).

:math:`\mathcal{K}^\tau` is **linear** in :math:`g`,
:math:`\mathcal{K}(ag + bh) = a\mathcal{K}g + b\mathcal{K}h`, no matter how
nonlinear :math:`\mathcal{M}` is. That is not an approximation and there is no
small parameter: it is a change of the space being acted on, from a
three-dimensional state space to an infinite-dimensional space of functions on
it.

Nothing is free. The operator is linear on a space with no finite basis, and
every practical method replaces it by its compression onto a finite
**dictionary** of observables. This module is that compression and its
diagnostics:

* :func:`edmd` -- the regression that defines extended dynamic mode
  decomposition, and :func:`dmd` -- its special case on the state variables
  alone;
* :func:`closure_residual` -- whether the dictionary is *invariant*, which is
  the one property that decides whether the compression is exact or merely
  fitted;
* :func:`linear_rollout` against :func:`relift_rollout` -- the two ways to
  integrate a Koopman model, which are not the same model, and
  :func:`off_manifold_residual`, which measures what separates them;
* :func:`operator_correlation` -- the autocorrelation function the operator
  predicts, which survives long after its trajectories do not.

The control case is :func:`chaoslib.systems.slow_manifold`, whose Koopman
operator closes exactly on three observables
(:func:`chaoslib.systems.slow_manifold_koopman_matrix`), so every routine here
can be checked against machine precision before being pointed at an attractor
where nothing closes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = ArrayLike
Floats = NDArray[np.float64]

__all__ = [
    "standardise",
    "pick_centres",
    "rbf_features",
    "monomial_features",
    "monomial_powers",
    "edmd",
    "dmd",
    "closure_residual",
    "linear_rollout",
    "relift_rollout",
    "off_manifold_residual",
    "continuous_eigenvalues",
    "spectral_radius",
    "operator_correlation",
]


# =========================================================================
# dictionaries
# =========================================================================
def standardise(states: Array, reference: Array | None = None) -> Floats:
    """Centre and scale each component by the reference sample's statistics.

    Every dictionary here is scale-sensitive -- a radial basis function has one
    width for all directions, and a monomial of degree 7 in a variable of size
    25 is of order :math:`10^9` -- so the state is whitened first and the
    dictionary built on the whitened coordinates. Skipping this is the usual
    reason an EDMD fit is reported as rank-deficient.
    """
    x = np.asarray(states, dtype=float)
    ref = x if reference is None else np.asarray(reference, dtype=float)
    scale = ref.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (x - ref.mean(axis=0)) / scale


def pick_centres(states: Array, count: int, seed: int = 0) -> Floats:
    """``count`` states drawn without replacement, as radial-basis centres.

    Drawing from the trajectory rather than from a grid is what keeps the
    dictionary on the attractor: a grid over the bounding box of Lorenz 63
    spends most of its basis functions on a region the system never visits
    (chapter 30 measured that region as almost all of the box).
    """
    x = np.asarray(states, dtype=float)
    k = int(count)
    if k < 0 or k > x.shape[0]:
        raise ValueError(f"count {k} outside 0..{x.shape[0]}")
    rng = np.random.default_rng(seed)
    return x[rng.choice(x.shape[0], size=k, replace=False)]


def rbf_features(states: Array, centres: Array, width: float = 0.7) -> Floats:
    r"""Dictionary of a constant, the state itself, and Gaussian bumps:

    .. math::
        g(x) = \left(1,\; x,\; e^{-\|x-c_j\|^2/2w^2}\right)_{j=1\ldots k}.

    The **constant is not padding.** :math:`g \equiv 1` satisfies
    :math:`\mathcal{K}g = g` for every dynamical system whatsoever, so it is an
    exact Koopman eigenfunction with eigenvalue 1, and including it is what
    lets the fitted operator have that eigenvalue exactly rather than
    approximately. Its eigenvalue is the one attached to the invariant measure
    (chapter 30), and :func:`spectral_radius` returning exactly 1 is the
    signature.

    The **state is included too**, so that the state can be read straight back
    off the dictionary vector at columns ``1 .. n`` with no reconstruction
    step -- which is what :func:`linear_rollout` and :func:`relift_rollout`
    both rely on.
    """
    x = np.asarray(states, dtype=float)
    c = np.asarray(centres, dtype=float).reshape(-1, x.shape[-1])
    ones = np.ones((x.shape[0], 1))
    if c.shape[0] == 0:
        return np.hstack([ones, x])
    squared = ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=-1)
    return np.hstack([ones, x, np.exp(-squared / (2.0 * float(width) ** 2))])


def monomial_powers(dimension: int, order: int) -> NDArray[np.int64]:
    """Exponent tuples of every monomial of total degree at most ``order``.

    Ordered by total degree then lexicographically, so the constant is first
    and the state variables are next -- the same column convention as
    :func:`rbf_features`.
    """
    from itertools import product

    rows = [
        p
        for p in product(range(int(order) + 1), repeat=int(dimension))
        if sum(p) <= int(order)
    ]
    rows.sort(key=lambda p: (sum(p), tuple(-q for q in p)))
    return np.asarray(rows, dtype=np.int64)


def monomial_features(states: Array, order: int = 2) -> Floats:
    r"""Dictionary of all monomials :math:`x_1^{a}x_2^{b}\cdots` of total
    degree at most ``order``.

    The dictionary that *closes* for :func:`chaoslib.systems.slow_manifold`:
    at order 2 it contains :math:`\{1, x_1, x_2, x_1^2, x_1x_2, x_2^2\}`, which
    spans the invariant subspace :math:`\{x_1, x_2, x_1^2\}` and three more
    directions the dynamics simply does not couple into it.

    It is a poor dictionary on an attractor, and the reason is conditioning,
    not principle: the number of terms grows as
    :math:`\binom{n+d}{d}` and the high-degree columns become collinear on a
    set of dimension 2.06. Use :func:`rbf_features` there.
    """
    x = np.asarray(states, dtype=float)
    powers = monomial_powers(x.shape[-1], order)
    return np.stack([np.prod(x**p, axis=-1) for p in powers], axis=-1)


# =========================================================================
# the compression
# =========================================================================
def edmd(
    features_x: Array, features_y: Array, rcond: float = 1e-10
) -> dict[str, Floats | float]:
    r"""Extended DMD: least squares for the operator on a dictionary.

    Given dictionary values at successive times, :math:`G_x` and :math:`G_y`
    with one row per snapshot pair, solves

    .. math:: \mathbf{K} = \arg\min_{\mathbf{K}}\;\|G_y - G_x\mathbf{K}\|_F ,

    so that :math:`g(x_{n+1})^{\!\top} \approx g(x_n)^{\!\top}\mathbf{K}`.
    Returns ``operator``, its ``eigenvalues``, the relative ``residual``, and
    the ``rank`` the solve actually used.

    **The residual is the quantity to read, not the eigenvalues.** A small
    residual means the dictionary is nearly invariant under :math:`\mathcal{K}`
    and the eigenvalues approximate Koopman eigenvalues; a large one means the
    fit is a projection of something that leaves the dictionary, and its
    eigenvalues describe the projection rather than the dynamics. Measured on
    :func:`chaoslib.systems.slow_manifold`, a dictionary containing
    :math:`x_1^2` gives a residual of :math:`9\times10^{-16}` and one without
    it :math:`7\times10^{-3}`, from the same data.

    ``rcond`` matters here more than in most least squares: a dictionary of
    radial basis functions on an attractor is deliberately overcomplete and its
    Gram matrix is near-singular by construction, so the cut-off is choosing
    how much of the dictionary to believe, not guarding against an accident.
    """
    gx = np.asarray(features_x, dtype=float)
    gy = np.asarray(features_y, dtype=float)
    if gx.shape != gy.shape:
        raise ValueError(f"feature shapes differ: {gx.shape} vs {gy.shape}")
    operator, _, rank, _ = np.linalg.lstsq(gx, gy, rcond=float(rcond))
    denominator = float(np.linalg.norm(gy))
    residual = float(np.linalg.norm(gy - gx @ operator))
    return {
        "operator": operator,
        "eigenvalues": np.linalg.eigvals(operator),
        "residual": residual / denominator if denominator > 0 else np.nan,
        "rank": float(rank),
        "size": float(gx.shape[1]),
    }


def dmd(states_x: Array, states_y: Array, rcond: float = 1e-10) -> dict:
    """Plain dynamic mode decomposition: :func:`edmd` on the state variables
    and a constant.

    DMD is the :math:`d = 1` case of EDMD and nothing more. It is exact for a
    linear system, exact for a linear system observed through an affine
    transformation, and an approximation everywhere else -- which is worth
    stating plainly, because DMD is routinely applied to nonlinear fields and
    its output described as "the modes of the system".
    """
    x = np.asarray(states_x, dtype=float)
    y = np.asarray(states_y, dtype=float)
    gx = np.hstack([np.ones((x.shape[0], 1)), x])
    gy = np.hstack([np.ones((y.shape[0], 1)), y])
    return edmd(gx, gy, rcond=rcond)


def closure_residual(
    operator: Array, features_x: Array, features_y: Array
) -> float:
    r"""How far the dictionary is from being Koopman-invariant, relative.

    .. math::
        \frac{\|G_y - G_x\mathbf{K}\|_F}{\|G_y\|_F} .

    Zero exactly when :math:`\mathcal{K}` maps the span of the dictionary into
    itself, so this is the numerical form of the question "does my dictionary
    close?" -- the only question that separates an exact linear representation
    from a fitted one. It is reported by :func:`edmd` as ``residual``; this
    function exists to evaluate an operator fitted on **one** data set against
    **another**, which is how the chapter checks for overfitting.
    """
    gx = np.asarray(features_x, dtype=float)
    gy = np.asarray(features_y, dtype=float)
    k = np.asarray(operator, dtype=float)
    denominator = float(np.linalg.norm(gy))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(gy - gx @ k) / denominator)


# =========================================================================
# the two ways to integrate a Koopman model
# =========================================================================
def linear_rollout(operator: Array, features0: Array, steps: int) -> Floats:
    r"""Iterate the operator in observable space:
    :math:`g_{n+1}^{\!\top} = g_n^{\!\top}\mathbf{K}`.

    This is the Koopman model **as a linear model**: one matrix, applied
    repeatedly, with no reference to the state at any point. It is
    unconditionally stable whenever the spectral radius is at most one, which
    for a dictionary containing the constant it essentially always is.

    Returns an array of shape ``(steps + 1, members, size)``.

    Its weakness is not accuracy but geometry, and
    :func:`off_manifold_residual` measures it: the set of dictionary vectors
    that are *the dictionary of some state* is an :math:`n`-dimensional
    surface inside the :math:`m`-dimensional dictionary space, and
    :math:`\mathbf{K}` does not preserve it. Measured on Lorenz 63 with 304
    observables, the rolled-out vector is 9 % off that surface after a quarter
    of a time unit and 44 % off after one.
    """
    k = np.asarray(operator, dtype=float)
    g = np.atleast_2d(np.asarray(features0, dtype=float))
    out = np.empty((int(steps) + 1, *g.shape), dtype=float)
    out[0] = g
    for i in range(int(steps)):
        g = g @ k
        out[i + 1] = g
    return out


def relift_rollout(
    operator: Array, x0: Array, steps: int, lift, state_slice: slice
) -> Floats:
    r"""Iterate with a re-lift each step: advance, read the state back out,
    and rebuild the dictionary from it.

    .. math::
        x_{n+1} = P\left(g(x_n)^{\!\top}\mathbf{K}\right),

    where :math:`P` selects the state entries. Returns ``(steps + 1, members,
    n)``.

    **This is not a linear model.** The composition lift-advance-project is as
    nonlinear as the lift, and every claim that a Koopman model "linearises the
    dynamics" refers to :func:`linear_rollout` while every implementation that
    forecasts competitively does this instead. Chapter 31 measures the gap: at
    one time unit on Lorenz 63 the re-lifted rollout is 24 times more accurate,
    and it is the linearity that was traded for it. The re-lifted model also
    loses the stability guarantee -- its error passes the climatological level
    and keeps going, because nothing now constrains it to the attractor.
    """
    k = np.asarray(operator, dtype=float)
    x = np.atleast_2d(np.asarray(x0, dtype=float))
    out = np.empty((int(steps) + 1, *x.shape), dtype=float)
    out[0] = x
    for i in range(int(steps)):
        x = (lift(x) @ k)[:, state_slice]
        out[i + 1] = x
    return out


def off_manifold_residual(features: Array, lift, state_slice: slice) -> Floats:
    r"""How far a dictionary vector is from being any state's dictionary.

    .. math::
        \frac{\left\|g - \mathrm{dict}\!\left(P g\right)\right\|}
             {\left\|\mathrm{dict}\!\left(P g\right)\right\|}

    per member: read the state out of the vector, rebuild the dictionary from
    that state, and compare. Zero for any vector that lies on the lifted
    manifold, and growing with the linear rollout's error because it is its
    cause. This is the diagnostic that distinguishes "the operator is
    inaccurate" from "the operator is fine and the iteration has left the set
    it was fitted on" -- two failures with different fixes.
    """
    g = np.atleast_2d(np.asarray(features, dtype=float))
    rebuilt = lift(g[:, state_slice])
    denominator = np.linalg.norm(rebuilt, axis=1)
    denominator = np.where(denominator > 0.0, denominator, 1.0)
    return np.linalg.norm(g - rebuilt, axis=1) / denominator


# =========================================================================
# reading the spectrum
# =========================================================================
def continuous_eigenvalues(eigenvalues: Array, tau: float) -> NDArray[np.complex128]:
    r"""Convert discrete-time Koopman eigenvalues to growth rates and
    frequencies, :math:`\ln\lambda/\tau`.

    The real part is a decay rate per unit time and the imaginary part an
    angular frequency. Note the branch: :math:`\ln` is many-valued, so any
    frequency faster than the Nyquist rate :math:`\pi/\tau` is aliased into the
    principal branch and reported as a slower one. A mode reported near the
    Nyquist frequency should be treated as unresolved, not as a measurement.

    Eigenvalues at exactly zero (a rank-deficient fit will produce them) return
    :math:`-\infty`, which is the correct statement -- that direction is
    annihilated in one step -- and is why the array comes back complex rather
    than raising.
    """
    lam = np.asarray(eigenvalues, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(lam) / float(tau)


def spectral_radius(operator: Array) -> float:
    """Largest eigenvalue modulus: the growth rate of the linear rollout.

    Should be exactly 1 for any dictionary containing the constant function,
    and a value above 1 means the fitted operator will diverge under iteration
    no matter how small its one-step residual is -- the same failure chapter 29
    found in a reservoir whose spectral radius exceeded one.
    """
    return float(np.abs(np.linalg.eigvals(np.asarray(operator, dtype=float))).max())


def operator_correlation(
    operator: Array,
    features: Array,
    column: int,
    steps: int,
) -> Floats:
    r"""The autocorrelation function the operator predicts, normalised to 1 at
    lag zero.

    For an observable :math:`g_i` in the dictionary,

    .. math::
        C(n\tau) = \frac{\left\langle g_i(x)\,
                   \left(\mathbf{K}^n g\right)_i(x)\right\rangle}
                        {\left\langle g_i(x)^2\right\rangle},

    with the average over the supplied snapshots -- an estimate of
    :math:`\langle g_i(x)\,g_i(\mathcal{M}^{n\tau}x)\rangle_\nu` against the
    invariant measure of chapter 30.

    This is what a Koopman model is *for*. Its trajectories on Lorenz 63 are
    useless past two time units, and its correlation function is right to three
    decimal places over the same interval -- because a decaying correlation is
    a statement about the spectrum, which the compression captures, and a
    trajectory is a statement about a particular orbit, which it does not.
    """
    k = np.asarray(operator, dtype=float)
    g = np.asarray(features, dtype=float)
    reference = g[:, int(column)]
    variance = float(np.mean(reference**2))
    if variance <= 0.0:
        raise ValueError("the reference observable has zero variance")
    out = np.empty(int(steps) + 1, dtype=float)
    advanced = g.copy()
    for n in range(int(steps) + 1):
        if n:
            advanced = advanced @ k
        out[n] = float(np.mean(reference * advanced[:, int(column)]) / variance)
    return out
