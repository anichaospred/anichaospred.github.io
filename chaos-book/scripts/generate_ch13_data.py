#!/usr/bin/env python3
"""Precompute chapter 13's synthetic operational forecast archive.

The chapter validates Lorenz's (1982) lagged-forecast estimator, which infers
error growth from a forecast archive **without ever using the truth**. The only
way to check such an estimator is to run it on a system where the truth is
known, so this script builds a synthetic operational setup on Lorenz 96 --
cycling EnKF analyses every six hours, then forecasts to thirty days from every
analysis -- and records both the true error curve and the truth-free estimate.

Four observing configurations, spanning analysis errors from 0.5 % to 37 % of
the climatological spread, so the chapter can ask how the estimator behaves as
the analysis degrades -- and where it stops having anything to work with.

**Everything is per component**: the RMS over the 40 sites, which is the
operational convention (an RMS error in the units of the field, per grid point).
``errorgrowth.saturation_level`` returns a *norm* over the whole state vector, so
it is divided by sqrt(N) here. Mixing the two is a factor of 6.3 at N = 40, and
it silently made a fully saturated forecast look like it had plateaued at 16 %
of saturation on the first attempt.

Run from chaos-book/:
    python3 scripts/generate_ch13_data.py        # ~30 seconds
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import assimilate, errorgrowth, integrate, systems  # noqa: E402

N_SITES, FORCING, DT = 40, 8.0, 0.01
CYCLE_STEPS = 5          # 0.05 time units = 6 hours at 5 days per unit
N_SPINUP, N_CYCLES = 200, 600
MAX_LEAD = 120           # 6 time units = 30 days
LOCALISATION_CUTOFF = 8.0

# (label, observed every nth site, observation sigma, members, inflation)
#
# The inflation is 1.02 everywhere, and getting that wrong cost a whole run.
# The first attempt raised it for the sparser networks -- 1.08 and 1.10, on the
# reasoning that a worse-constrained ensemble needs more spread -- and both
# configurations then DIVERGED, giving analysis errors of 61 % and 76 % of the
# climatological spread and, in some tuning combinations, floating-point
# overflow. Retuned to 1.02 the same networks track perfectly well: 10 observed
# sites gives 6.2 % rather than 61 %.
#
# So the sparse cases are not an observability limit, and it would have been
# easy to present them as one. Inflation compensates for sampling error in the
# background covariance; when the observations are sparse the analysis is
# already close to the background, and additional inflation amplifies an
# ensemble spread that the observations cannot then pull back.
CONFIGURATIONS = (
    ("dense", 1, 0.1, 30, 1.02),
    ("operational", 2, 0.3, 30, 1.02),
    ("sparse", 4, 0.5, 30, 1.02),
    ("degraded", 5, 1.0, 30, 1.02),
)


def _step(state: np.ndarray, n_steps: int) -> np.ndarray:
    return integrate.rk4(
        systems.lorenz96, state,
        integrate.trajectory_grid(n_steps * DT, DT), forcing=FORCING,
    )[-1]


def _emit(name: str, values, fmt: str = ".6e", per_line: int = 6) -> None:
    items = [
        'float("nan")' if not np.isfinite(v) else format(float(v), fmt)
        for v in np.ravel(values)
    ]
    print(f"{name} = (")
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i : i + per_line]) + ",")
    print(")")


def archive(obs_every: int, sigma: float, members: int, inflation: float):
    """Cycle an EnKF, then forecast from every analysis to every lead."""
    rng = np.random.default_rng(0)
    truth = _step(
        systems.lorenz96_uniform_state(FORCING, N_SITES)
        + rng.normal(0.0, 0.1, N_SITES),
        30000,
    )
    sites = np.arange(0, N_SITES, obs_every)
    h_op = np.eye(N_SITES)[sites]
    r_cov = sigma**2 * np.eye(sites.size)
    separation = np.abs(
        np.arange(N_SITES)[:, None] - np.arange(N_SITES)[None, :]
    )
    separation = np.minimum(separation, N_SITES - separation)
    localisation = assimilate.gaspari_cohn(separation, LOCALISATION_CUTOFF)

    ensemble = truth + rng.normal(0.0, 2.0, (members, N_SITES))
    analyses, truths = [], []
    for cycle in range(N_SPINUP + N_CYCLES):
        truth = _step(truth, CYCLE_STEPS)
        ensemble = _step(ensemble, CYCLE_STEPS)
        observation = h_op @ truth + rng.normal(0.0, sigma, sites.size)
        ensemble = assimilate.enkf_update(
            ensemble, observation, h_op, r_cov,
            inflation=inflation, localisation=localisation,
            seed=int(rng.integers(1_000_000_000)),
        )
        if cycle >= N_SPINUP:
            analyses.append(ensemble.mean(axis=0).copy())
            truths.append(truth.copy())

    analyses = np.array(analyses)
    truths = np.array(truths)

    forecasts = np.empty((MAX_LEAD + 1, analyses.shape[0], N_SITES))
    verifying = np.empty_like(forecasts)
    running_f, running_t = analyses.copy(), truths.copy()
    forecasts[0], verifying[0] = running_f, running_t
    for lead in range(1, MAX_LEAD + 1):
        running_f = _step(running_f, CYCLE_STEPS)
        running_t = _step(running_t, CYCLE_STEPS)
        forecasts[lead], verifying[lead] = running_f, running_t

    return analyses, truths, forecasts, verifying


def run(label: str, obs_every: int, sigma: float, members: int, inflation: float):
    started = time.perf_counter()
    analyses, truths, forecasts, verifying = archive(
        obs_every, sigma, members, inflation
    )
    n_analyses = analyses.shape[0]

    # Per component throughout: RMS over the 40 sites.
    saturation = errorgrowth.saturation_level(
        truths, statistic="mean"
    ) / np.sqrt(N_SITES)
    analysis_error = float(
        np.sqrt(np.mean((analyses - truths) ** 2, axis=-1)).mean()
    )

    # The TRUE error curve, which needs the truth.
    true_error = np.sqrt(
        np.mean((forecasts - verifying) ** 2, axis=-1)
    ).mean(axis=1)

    # Lorenz (1982): for verification cycle v, compare the forecast started at
    # v-L-1 (lead L+1) with the one started at v-L (lead L). Both verify at v,
    # and neither uses the truth.
    lagged = np.full(MAX_LEAD, np.nan)
    for lead in range(MAX_LEAD):
        v = np.arange(lead + 1, n_analyses)
        difference = forecasts[lead + 1, v - lead - 1] - forecasts[lead, v - lead]
        lagged[lead] = float(
            np.sqrt(np.mean(difference**2, axis=-1)).mean()
        )

    leads = np.arange(MAX_LEAD + 1) * CYCLE_STEPS * DT
    print(f"\n# --- {label}: {N_SITES // obs_every} of {N_SITES} sites, "
          f"sigma_o = {sigma}, {members} members ---")
    print(f"#   analysis RMSE {analysis_error:.5f} = "
          f"{100 * analysis_error / saturation:.2f}% of saturation {saturation:.4f}")
    print(f"#   half saturation at "
          f"{np.interp(0.5 * saturation, true_error, leads):.2f} time units "
          f"({5 * np.interp(0.5 * saturation, true_error, leads):.1f} days)")

    def rate(x, y, lo, hi):
        w = (y > lo * saturation) & (y < hi * saturation)
        if int(w.sum()) < 5:
            return float("nan")
        return float(np.polyfit(x[w], np.log(y[w]), 1)[0])

    for lo, hi in ((0.02, 0.1), (0.02, 0.2), (0.05, 0.3), (0.1, 0.4)):
        true_rate = rate(leads, true_error, lo, hi)
        lagged_rate = rate(leads[: lagged.size], lagged, lo, hi)
        print(f"#   [{lo:g},{hi:g}]*sat: true {true_rate:6.3f}, "
              f"lagged {lagged_rate:6.3f}, ratio {lagged_rate / true_rate:6.4f}")

    key = label.upper()
    print(f"{key}_ANALYSIS_ERROR = {analysis_error:.6f}")
    print(f"{key}_SATURATION = {saturation:.6f}")
    print(f"{key}_N_OBS = {N_SITES // obs_every}")
    print(f"{key}_SIGMA_O = {sigma}")
    _emit(f"{key}_TRUE", true_error, ".6e", per_line=8)
    _emit(f"{key}_LAGGED", lagged, ".6e", per_line=8)
    print(f"#   {time.perf_counter() - started:.1f}s")
    return leads


if __name__ == "__main__":
    print("# --- shared axis ---")
    print(f"CYCLE_HOURS = 6")
    print(f"LAMBDA1_L96 = 1.67")
    grid = None
    for spec in CONFIGURATIONS:
        grid = run(*spec)
    print()
    _emit("LEADS", grid, ".4f", per_line=10)
    print(f"CONFIG_LABELS = {tuple(c[0] for c in CONFIGURATIONS)}")
