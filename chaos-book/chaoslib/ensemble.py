"""Ensemble generation and probabilistic forecast verification.

The through-line of the applied half of the book: a single forecast is a point
estimate of a distribution we cannot observe, an ensemble is a sample of it, and
verification is the discipline of checking whether that sample was honest.

Two distinctions the diagnostics here are built to keep visible:

* **spread vs. error** -- a well-calibrated ensemble has RMS spread equal to the
  RMS error of its own mean. Under-spread is the characteristic failure of
  operational systems.
* **accuracy vs. calibration** -- CRPS rewards both; the rank histogram
  isolates calibration alone.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]
RHS = Callable[..., Array]

__all__ = [
    "gaussian_perturbations",
    "bred_vectors",
    "singular_vector_ensemble",
    "ensemble_spread",
    "outside_span_fraction",
    "ensemble_mean_error",
    "rank_histogram",
    "crps",
    "brier_score",
    "reliability_diagram",
    "brier_decomposition",
    "value_score",
]


def gaussian_perturbations(
    x0: Array, n_members: int, delta0: float, seed: int | None = 42
) -> Array:
    """``n_members`` initial states, each ``x0`` plus an iid Gaussian perturbation.

    Perturbation amplitude ``delta0`` is the standard deviation *per component*.
    Isotropic Gaussian sampling is the naive scheme on purpose: chapters 16-17
    contrast it with singular vectors and bred vectors, and the reader should
    first see what the naive choice costs.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float).ravel()
    return x0 + delta0 * rng.normal(size=(n_members, x0.size))


def outside_span_fraction(increment: Array, ensemble: Array,
                          tol: float = 1e-10) -> float:
    r"""What fraction of ``increment`` lies outside the ensemble's own subspace.

    An ensemble of :math:`k` members spans a :math:`(k-1)`-dimensional affine
    subspace, and a *global* ensemble filter can only move the state inside it:
    its analysis is :math:`\bar x + \mathbf{X}w`, so the increment is
    :math:`\mathbf{X}` times something. This returns
    :math:`\|(\mathbf{I}-\mathbf{\Pi})\,\delta x\| / \|\delta x\|` with
    :math:`\mathbf{\Pi}` the orthogonal projector onto that span -- zero to
    machine precision for a global filter, and strictly positive for a *local*
    one, which is the quantitative form of the rank argument in chapter 19.

    ``increment`` and ``ensemble`` are both ``(n_members, n_state)``.

    **The projector is built from an explicit SVD, not from** ``np.linalg.pinv``.
    The perturbation matrix is deliberately rank deficient, and `pinv`'s default
    cutoff is relative to the largest singular value: when the numerically-zero
    one lands just above that cutoff it is *retained* and inverted to something
    of order :math:`10^{13}`, and the resulting projector acquires a spurious
    direction with enormous weight. That happened at three of seven ensemble
    sizes here, returning :math:`10^{-2}` where the answer is exactly zero --
    erratic across sizes, and entirely a property of the diagnostic rather than
    of the filter.
    """
    increment = np.atleast_2d(np.asarray(increment, dtype=float)).T
    ensemble = np.atleast_2d(np.asarray(ensemble, dtype=float))
    perturbations = (ensemble - ensemble.mean(axis=0)).T
    left, singular, _ = np.linalg.svd(perturbations, full_matrices=False)
    keep = singular > singular[0] * float(tol)
    basis = left[:, keep]
    residual = increment - basis @ (basis.T @ increment)
    norm = np.linalg.norm(increment)
    if norm == 0.0:
        return 0.0
    return float(np.linalg.norm(residual) / norm)


def bred_vectors(
    rhs: RHS,
    x0: Array,
    n_vectors: int,
    amplitude: float,
    cycle_time: float,
    n_cycles: int = 8,
    dt: float = 0.01,
    orthogonalise: bool = False,
    seed: int | None = 0,
    **params: float,
) -> Array:
    r"""Bred vectors: the breeding cycle of Toth & Kalnay (1993).

    Perturb, integrate the perturbed and control runs forward by ``cycle_time``,
    take the difference, rescale it back to ``amplitude``, repeat. After a few
    cycles the perturbation has forgotten how it started and points along the
    locally fastest-growing direction of the *nonlinear* flow.

    This needs **no adjoint and no tangent linear model** -- it is the cheapest
    way to find growing directions, and it is why breeding was the operational
    scheme at NCEP while ECMWF ran singular vectors. It is a finite-amplitude,
    finite-time approximation to the leading Lyapunov vector.

    **Independently bred vectors collapse onto each other.** They are all
    converging to the *same* leading direction, so an ensemble built from several
    of them samples one direction several times, and its spread is a fiction.
    Chapter 17 measures the collapse. ``orthogonalise=True`` re-orthogonalises
    the set after **each** cycle, which turns breeding into the Benettin
    construction of :mod:`chaoslib.lyapunov` and converges the set to the leading
    Lyapunov *vectors*; the operational cure was geographic masking
    *[citation needed]*. It requires ``n_vectors <= n_state``.

    Returns ``(n_vectors, n_state)`` perturbations, each of norm ``amplitude``.
    """
    from chaoslib.integrate import rk4

    x0 = np.asarray(x0, dtype=float).ravel()
    if orthogonalise and int(n_vectors) > x0.size:
        # QR of an (n, k) matrix with k > n returns only n orthonormal columns,
        # so the ensemble would silently come back smaller than requested --
        # which is worse than refusing, because the caller's member count is
        # then wrong everywhere downstream.
        raise ValueError(
            f"cannot orthogonalise {n_vectors} vectors in {x0.size} dimensions; "
            "at most n_state mutually orthogonal directions exist"
        )
    n_steps = max(1, int(round(cycle_time / dt)))
    grid = np.linspace(0.0, cycle_time, n_steps + 1)
    rng = np.random.default_rng(seed)

    perturbations = rng.normal(size=(int(n_vectors), x0.size))
    perturbations *= amplitude / np.linalg.norm(
        perturbations, axis=1, keepdims=True
    )

    control = x0.copy()
    for _cycle in range(int(n_cycles)):
        advanced = rk4(rhs, control + perturbations, grid, **params)[-1]
        control = rk4(rhs, control, grid, **params)[-1]
        perturbations = advanced - control
        if orthogonalise and n_vectors > 1:
            # Gram-Schmidt every cycle, which is the Benettin construction of
            # chaoslib.lyapunov: the vectors then converge to the leading
            # Lyapunov *vectors* rather than all to the leading one. Doing it
            # once at the end instead would be useless -- by then the set is
            # nearly rank one, and orthogonalising a collapsed set manufactures
            # its extra directions out of rounding error.
            basis, _ = np.linalg.qr(perturbations.T)
            perturbations = basis.T[: int(n_vectors)]
        norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
        perturbations = perturbations * (amplitude / np.where(norms > 0, norms, 1.0))
    return perturbations


def singular_vector_ensemble(
    propagator: Array, n_pairs: int, amplitude: float
) -> Array:
    r"""An ensemble of :math:`\pm` pairs along the leading singular vectors.

    Takes ``n_pairs`` leading singular vectors of a tangent-linear propagator
    (chapter 16) and returns ``2 * n_pairs`` perturbations: each vector and its
    negative, scaled to ``amplitude``.

    The pairing is not decoration. Singular vectors are defined up to sign, and
    including both signs makes the ensemble mean equal the control state exactly,
    so the ensemble is centred by construction rather than by luck. Operational
    singular-vector ensembles are built this way.

    These perturbations maximise growth over the window the propagator spans, and
    **maximising growth is not the same as sampling the analysis-error
    distribution** -- which is what a calibrated ensemble needs to do. Chapter 17
    measures the difference.
    """
    from chaoslib.adjoint import singular_vectors

    _sigma, initial, _final = singular_vectors(propagator, n_vectors=int(n_pairs))
    columns = np.asarray(initial, dtype=float)
    members = []
    for index in range(int(n_pairs)):
        direction = columns[:, index]
        direction = direction / np.linalg.norm(direction)
        members.append(amplitude * direction)
        members.append(-amplitude * direction)
    return np.asarray(members)


def ensemble_spread(ensemble: Array) -> Array:
    r"""RMS spread about the ensemble mean, per time.

    .. math::
        s(t) = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}\|x_i(t)-\bar x(t)\|^2}

    ``ensemble`` has shape ``(n_t, n_members, n_state)``. The :math:`N-1`
    normalisation is deliberate: the ensemble mean is itself estimated from the
    sample, and with the small ensembles used here (N ~ 20) the difference from
    :math:`N` is not negligible.
    """
    ens = np.asarray(ensemble, dtype=float)
    mean = ens.mean(axis=1, keepdims=True)
    n = ens.shape[1]
    if n < 2:
        raise ValueError("spread needs at least 2 members")
    sq = np.sum((ens - mean) ** 2, axis=-1)
    return np.sqrt(sq.sum(axis=1) / (n - 1))


def ensemble_mean_error(ensemble: Array, truth: Array) -> Array:
    """Euclidean error of the ensemble mean against the truth run, per time.

    Paired with :func:`ensemble_spread`: for a perfectly reliable ensemble the
    two curves coincide, so plotting them together is the single most
    informative ensemble diagnostic there is.
    """
    ens = np.asarray(ensemble, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return np.linalg.norm(ens.mean(axis=1) - truth, axis=-1)


def rank_histogram(ensemble: Array, truth: Array, seed: int | None = 0) -> Array:
    """Counts of the truth's rank within the sorted ensemble (Talagrand diagram).

    ``ensemble`` shape ``(n_cases, n_members)``, ``truth`` shape ``(n_cases,)``;
    returns ``n_members + 1`` bin counts. A flat histogram means the truth is
    statistically indistinguishable from a member -- the definition of
    reliability. U-shaped means under-spread; dome-shaped, over-spread.

    Ties are broken randomly, which is the standard correction: breaking them
    consistently (e.g. always low) manufactures a spurious end-bin spike.
    """
    ens = np.atleast_2d(np.asarray(ensemble, dtype=float))
    truth = np.asarray(truth, dtype=float).ravel()
    if ens.shape[0] != truth.size:
        raise ValueError("ensemble and truth disagree on the number of cases")

    rng = np.random.default_rng(seed)
    n_members = ens.shape[1]
    below = np.sum(ens < truth[:, None], axis=1)
    equal = np.sum(ens == truth[:, None], axis=1)
    ranks = below + np.array(
        [rng.integers(0, e + 1) if e > 0 else 0 for e in equal]
    )
    return np.bincount(ranks, minlength=n_members + 1)


def crps(ensemble: Array, truth: Array) -> Array:
    r"""Continuous ranked probability score, by the fair energy-form estimator.

    .. math::
        \mathrm{CRPS} = \frac{1}{N}\sum_i |x_i - y|
                        - \frac{1}{2N^2}\sum_{i,j}|x_i - x_j|

    This is the standard (biased-but-conventional) sample estimator. Units are
    those of the variable, and lower is better; it reduces to the absolute error
    for a deterministic forecast, which is what makes it a fair comparison
    between ensemble and single-run systems.

    ``ensemble`` shape ``(n_cases, n_members)``; returns one score per case.
    """
    ens = np.atleast_2d(np.asarray(ensemble, dtype=float))
    truth = np.asarray(truth, dtype=float).ravel()
    n = ens.shape[1]
    term1 = np.mean(np.abs(ens - truth[:, None]), axis=1)
    # Pairwise term via a sort: sum_ij |x_i - x_j| = 2 * sum_k (2k - N + 1) x_(k)
    xs = np.sort(ens, axis=1)
    k = np.arange(n)
    weights = 2.0 * k - n + 1.0
    term2 = np.sum(weights * xs, axis=1) / (n * n)
    return term1 - term2


def brier_score(probabilities: Array, outcomes: Array) -> float:
    r"""Brier score :math:`\frac{1}{n}\sum (p_i - o_i)^2` for a binary event.

    ``outcomes`` must be 0/1. Lower is better; 0.25 is the score of always
    forecasting a climatological probability of 0.5, which is the reference a
    forecast system has to beat before it is worth anything.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    o = np.asarray(outcomes, dtype=float).ravel()
    if p.shape != o.shape:
        raise ValueError("probabilities and outcomes must have the same shape")
    if not np.all(np.isin(o, (0.0, 1.0))):
        raise ValueError("outcomes must be 0 or 1")
    return float(np.mean((p - o) ** 2))


def reliability_diagram(
    probabilities: Array, outcomes: Array, bin_edges: Array | None = None
) -> tuple[Array, Array, Array]:
    r"""Binned forecast probability against observed frequency.

    Returns ``(mean_forecast, observed_frequency, counts)``, one entry per
    non-empty bin. A perfectly reliable forecast lies on the diagonal: when it
    says 30 %, the event happens 30 % of the time.

    ``bin_edges`` defaults to the **distinct forecast values**, not to a fixed
    number of equal-width bins. An ensemble of :math:`k` members can only issue
    the probabilities :math:`0, 1/k, \ldots, 1`, so those values *are* the natural
    bins -- and with that choice the Brier decomposition of
    :func:`brier_decomposition` is exact rather than approximate. Equal-width
    binning of already-discrete forecasts introduces a within-bin variance term
    that shows up as a residual in the decomposition and is pure artefact.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    o = np.asarray(outcomes, dtype=float).ravel()
    if p.shape != o.shape:
        raise ValueError("probabilities and outcomes must have the same shape")

    if bin_edges is None:
        values = np.unique(p)
        indices = np.searchsorted(values, p)
        n_bins = values.size
    else:
        edges = np.asarray(bin_edges, dtype=float).ravel()
        indices = np.clip(np.digitize(p, edges[1:-1]), 0, edges.size - 2)
        n_bins = edges.size - 1

    counts = np.bincount(indices, minlength=n_bins).astype(float)
    forecast = np.bincount(indices, weights=p, minlength=n_bins)
    observed = np.bincount(indices, weights=o, minlength=n_bins)
    keep = counts > 0
    return (
        forecast[keep] / counts[keep],
        observed[keep] / counts[keep],
        counts[keep],
    )


def brier_decomposition(
    probabilities: Array, outcomes: Array, bin_edges: Array | None = None
) -> tuple[float, float, float]:
    r"""Murphy's decomposition of the Brier score into three parts.

    .. math::
        \mathrm{BS} = \underbrace{\tfrac1n\sum_k n_k(\bar p_k-\bar o_k)^2}_{\text{reliability}}
          - \underbrace{\tfrac1n\sum_k n_k(\bar o_k-\bar o)^2}_{\text{resolution}}
          + \underbrace{\bar o(1-\bar o)}_{\text{uncertainty}}

    Returns ``(reliability, resolution, uncertainty)``. Lower reliability is
    better (it measures calibration error); *higher* resolution is better (it
    measures the forecast's ability to separate events from non-events);
    uncertainty depends only on the climatological base rate and no forecast
    system can change it.

    The distinction is the practical content of the chapter. **Reliability is
    fixable after the fact** -- a systematically overconfident forecast can be
    recalibrated by relabelling its probabilities, with no new information.
    **Resolution cannot be**: it is the information the forecast actually
    carries, and no post-processing creates any.

    With the default binning (the distinct forecast values) the identity above
    holds *exactly*, and :func:`chaoslib.ensemble.brier_score` equals
    ``reliability - resolution + uncertainty`` to machine precision. That is
    asserted as a test, because an approximate decomposition is very easy to
    write by accident and its residual is easy to mistake for a real effect.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    o = np.asarray(outcomes, dtype=float).ravel()
    if not np.all(np.isin(o, (0.0, 1.0))):
        raise ValueError("outcomes must be 0 or 1")
    forecast, observed, counts = reliability_diagram(p, o, bin_edges)
    total = counts.sum()
    base_rate = float(o.mean())
    reliability = float(np.sum(counts * (forecast - observed) ** 2) / total)
    resolution = float(np.sum(counts * (observed - base_rate) ** 2) / total)
    uncertainty = float(base_rate * (1.0 - base_rate))
    return reliability, resolution, uncertainty


def value_score(
    probabilities: Array, outcomes: Array, cost_loss: Array
) -> Array:
    r"""Relative economic value of a probabilistic forecast (Richardson 2000).

    A user faces a cost :math:`C` to protect against an event that would
    otherwise cost them :math:`L`. Only the ratio :math:`\alpha = C/L` matters,
    and the optimal strategy is to protect whenever the forecast probability
    exceeds :math:`\alpha`. Writing :math:`h, f, m` for the frequencies of
    protected-and-event, protected-and-not, and unprotected-and-event, the
    expected expense per unit :math:`L` is :math:`\alpha(h+f) + m`, against
    :math:`\min(\alpha, \bar o)` for someone who only knows the climatology and
    :math:`\alpha\bar o` for someone with a perfect forecast. The relative value

    .. math::
        V = \frac{E_{\text{clim}} - E_{\text{forecast}}}
                 {E_{\text{clim}} - E_{\text{perfect}}}

    is 1 for a perfect forecast, 0 for one no better than climatology, and
    negative for one that is worse than useless.

    ``cost_loss`` may be an array, and this is the whole point: **V is a curve,
    not a number.** A deterministic forecast forces one threshold on every user,
    so it has value near a single :math:`\alpha`; a probabilistic forecast lets
    each user choose their own, so it has value across a range. That is the
    argument for ensembles that no accuracy score can make, and chapter 17
    measures it.

    Returns one value per requested cost-loss ratio.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    o = np.asarray(outcomes, dtype=float).ravel()
    if p.shape != o.shape:
        raise ValueError("probabilities and outcomes must have the same shape")
    if not np.all(np.isin(o, (0.0, 1.0))):
        raise ValueError("outcomes must be 0 or 1")
    ratios = np.atleast_1d(np.asarray(cost_loss, dtype=float))
    base_rate = float(o.mean())

    values = np.empty(ratios.size)
    for index, alpha in enumerate(ratios):
        # Protect when p >= alpha. The >= rather than > matters at the
        # boundaries: with discrete ensemble probabilities, alpha frequently
        # lands exactly on an attainable forecast value, and the two
        # conventions then differ by a whole bin's worth of cases.
        protect = p >= alpha
        hits = float(np.mean(protect & (o == 1.0)))
        false_alarms = float(np.mean(protect & (o == 0.0)))
        misses = float(np.mean(~protect & (o == 1.0)))
        expense = alpha * (hits + false_alarms) + misses
        climate = min(alpha, base_rate)
        perfect = alpha * base_rate
        denominator = climate - perfect
        values[index] = (
            np.nan if denominator == 0.0 else (climate - expense) / denominator
        )
    return values
