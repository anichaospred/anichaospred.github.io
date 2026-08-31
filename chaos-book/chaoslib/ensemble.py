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

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]

__all__ = [
    "gaussian_perturbations",
    "ensemble_spread",
    "ensemble_mean_error",
    "rank_histogram",
    "crps",
    "brier_score",
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
