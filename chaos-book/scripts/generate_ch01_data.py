#!/usr/bin/env python3
r"""Precompute chapter 1's demonstrations.

Everything on Lorenz 63, because chapter 1 has to be readable before chapter 6
has introduced anything. :math:`\rho` stands in for a boundary forcing: it is
the Rayleigh number, so raising it is turning up the heating, and the attractor
it produces is a different attractor rather than a different point on the same
one.

Four blocks.

1. **One forecast, three ways of being right.** A single run, an ensemble and a
   climatological forecast of the same case, scored by RMSE and by CRPS against
   lead. Which one "wins" depends on the lead *and* on the score, which is the
   chapter's opening point and the whole reason chapter 22 exists.
2. **Predictability of the first kind**: information about the future carried by
   today's state, as a function of lead. Averaged over 32 starting points --
   a single start is not monotone, because whether a forecast blob sits where
   the climatology is thick or thin is an accident of where it started.
3. **Predictability of the second kind**: information carried by the forcing,
   for several changes in :math:`\rho`. Constant in lead, by construction.
4. **The crossover**, and the distributions that make it visible.

Both kinds are measured in **nats, by the same estimator, on the same variable**,
which is what makes them comparable at all. The estimator's noise floor is
measured rather than assumed: two independent climatology samples of the same
size give a positive answer, and no decaying curve can go below it.

Run from chaos-book/:
    python3 scripts/generate_ch01_data.py        # ~2 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import ensemble, information, integrate, systems  # noqa: E402

DT = 0.01
RHO_BASE = 28.0
RHO_SHIFTS = (29.0, 30.0, 32.0, 36.0)
CLIMATE_TIME = 900.0
SPINUP = 4000
BINS = tuple(np.round(np.linspace(-5.0, 70.0, 41), 4))
VARIABLE, VARIABLE_NAME = 2, "z"

N_STARTS, MEMBERS, MAX_LEAD = 32, 3000, 20.0
LEADS = tuple(np.round(np.arange(0.0, MAX_LEAD + 1e-9, 0.5), 3))
SPREAD0 = 0.05

SCORE_CASES, SCORE_MEMBERS = 200, 50
SCORE_LEADS = tuple(np.round(np.arange(0.0, 12.0 + 1e-9, 0.25), 3))


def _emit(name: str, values, fmt: str = ".6e", per_line: int = 8) -> None:
    items = [
        'float("nan")' if not np.isfinite(v) else format(float(v), fmt)
        for v in np.ravel(values)
    ]
    print(f"{name} = (")
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i : i + per_line]) + ",")
    print(")")


def _scalar(name: str, value, fmt: str = ".6f") -> None:
    """Emit one float, spelling a non-finite one as `float("nan")`."""
    value = float(value)
    if not np.isfinite(value):
        print(f'{name} = float("nan")')
    else:
        print(f"{name} = {format(value, fmt)}")


def _climatology(rho: float, seed: int = 0) -> np.ndarray:
    start = np.array([1.0, 1.0, 20.0]) + np.random.default_rng(seed).normal(
        0.0, 0.1, 3
    )
    return integrate.rk4(
        systems.lorenz63, start, integrate.trajectory_grid(CLIMATE_TIME, DT),
        rho=rho,
    )[SPINUP:]


# ==========================================================================
# 1. one forecast, three ways of being right
# ==========================================================================
def three_ways(base: np.ndarray) -> None:
    print("# --- 1. one forecast, three ways of being right ---")
    rng = np.random.default_rng(10)
    stride = base.shape[0] // SCORE_CASES
    starts = base[::stride][:SCORE_CASES]
    climate_mean = base.mean(axis=0)
    climate_sigma = float(base[:, VARIABLE].std())

    grid = np.linspace(0.0, SCORE_LEADS[-1],
                       int(round(SCORE_LEADS[-1] / DT)) + 1)
    truth = integrate.rk4(systems.lorenz63, starts, grid, rho=RHO_BASE)

    # The single run and the ensemble start from the SAME analysis error, so
    # the comparison is about what is done with it, not about who was given a
    # better initial condition.
    perturbations = rng.normal(0.0, SPREAD0, (SCORE_CASES, SCORE_MEMBERS, 3))
    members = starts[:, None, :] + perturbations
    control = members[:, 0, :]
    ens_runs = integrate.rk4(
        systems.lorenz63, members.reshape(-1, 3), grid, rho=RHO_BASE,
    ).reshape(-1, SCORE_CASES, SCORE_MEMBERS, 3)
    ctl_runs = integrate.rk4(systems.lorenz63, control, grid, rho=RHO_BASE)

    single_rmse, ens_rmse, clim_rmse = [], [], []
    single_crps, ens_crps, clim_crps = [], [], []
    climate_sample = base[:, VARIABLE]
    for lead in SCORE_LEADS:
        index = int(round(lead / DT))
        verify = truth[index][:, VARIABLE]
        ctl = ctl_runs[index][:, VARIABLE]
        ens = ens_runs[index][:, :, VARIABLE]
        single_rmse.append(float(np.sqrt(np.mean((ctl - verify) ** 2))))
        ens_rmse.append(
            float(np.sqrt(np.mean((ens.mean(axis=1) - verify) ** 2)))
        )
        clim_rmse.append(
            float(np.sqrt(np.mean((climate_mean[VARIABLE] - verify) ** 2)))
        )
        # CRPS of a one-member ensemble is the absolute error, which is what
        # makes it a fair comparison between a single run and an ensemble.
        single_crps.append(float(ensemble.crps(ctl[:, None], verify).mean()))
        ens_crps.append(float(ensemble.crps(ens, verify).mean()))
        # A climatological "ensemble": a random draw from the long run.
        draw = rng.choice(climate_sample, size=(SCORE_CASES, SCORE_MEMBERS))
        clim_crps.append(float(ensemble.crps(draw, verify).mean()))

    print(f"#   {SCORE_CASES} cases, {SCORE_MEMBERS} members, variable "
          f"{VARIABLE_NAME}; climatological sigma {climate_sigma:.3f}")
    print(f"#   {'lead':>5} {'RMSE single':>12} {'RMSE ens':>9} {'RMSE clim':>10} "
          f"| {'CRPS single':>12} {'CRPS ens':>9} {'CRPS clim':>10}")
    for slot in range(0, len(SCORE_LEADS), 8):
        print(f"#   {SCORE_LEADS[slot]:5.2f} {single_rmse[slot]:12.4f} "
              f"{ens_rmse[slot]:9.4f} {clim_rmse[slot]:10.4f} | "
              f"{single_crps[slot]:12.4f} {ens_crps[slot]:9.4f} "
              f"{clim_crps[slot]:10.4f}")

    def crossing(better, worse):
        for lead, a, b in zip(SCORE_LEADS, better, worse):
            if a >= b:
                return float(lead)
        return float("nan")

    rmse_cross = crossing(single_rmse, clim_rmse)
    crps_cross = crossing(single_crps, clim_crps)
    print(f"#   the single run stops beating climatology at lead "
          f"{rmse_cross:.2f} by RMSE and {crps_cross:.2f} by CRPS")
    print(f"SCORE_LEADS = {SCORE_LEADS}")
    print(f"SCORE_CASES = {SCORE_CASES}")
    print(f"SCORE_MEMBERS = {SCORE_MEMBERS}")
    _scalar("CLIMATE_SIGMA", climate_sigma)
    _scalar("RMSE_CROSSING", rmse_cross, ".4f")
    _scalar("CRPS_CROSSING", crps_cross, ".4f")
    for name, values in (
        ("RMSE_SINGLE", single_rmse), ("RMSE_ENS", ens_rmse),
        ("RMSE_CLIM", clim_rmse), ("CRPS_SINGLE", single_crps),
        ("CRPS_ENS", ens_crps), ("CRPS_CLIM", clim_crps),
    ):
        _emit(name, values, ".6f", per_line=10)


# ==========================================================================
# 2 & 3. the two kinds, in the same units
# ==========================================================================
def two_kinds(base: np.ndarray) -> None:
    print("\n# --- 2. how much does today's state tell you? ---")
    bins = np.asarray(BINS)
    reference = base[:, VARIABLE]

    # The estimator's noise floor: two independent samples of the SAME
    # distribution, at the ensemble's own size. No decaying curve can go below
    # this, and a curve that appears to is measuring its own bias.
    rng = np.random.default_rng(4)
    floors = [
        information.binned_relative_entropy(
            rng.choice(reference, MEMBERS), reference, bins
        )
        for _ in range(40)
    ]
    floor = float(np.mean(floors))
    print(f"#   estimator noise floor at {MEMBERS} members, "
          f"{len(bins) - 1} bins: {floor:.4f} nats")

    stride = base.shape[0] // N_STARTS
    starts = base[::stride][:N_STARTS]
    members = np.repeat(starts, MEMBERS, axis=0) + np.random.default_rng(
        1
    ).normal(0.0, SPREAD0, (N_STARTS * MEMBERS, 3))
    grid = np.linspace(0.0, MAX_LEAD, int(round(MAX_LEAD / DT)) + 1)
    started = time.perf_counter()
    evolved = integrate.rk4(systems.lorenz63, members, grid, rho=RHO_BASE)
    print(f"#   {N_STARTS} starts x {MEMBERS} members to lead {MAX_LEAD:g} "
          f"({time.perf_counter() - started:.0f}s)")

    first, spread = [], []
    for lead in LEADS:
        index = int(round(lead / DT))
        per_start = [
            information.binned_relative_entropy(
                evolved[index, s * MEMBERS : (s + 1) * MEMBERS, VARIABLE],
                reference, bins,
            )
            for s in range(N_STARTS)
        ]
        first.append(float(np.mean(per_start)))
        spread.append(float(np.std(per_start)))

    # Monotone WITHIN SAMPLING ERROR, not to an absolute tolerance. The mean
    # over 32 starts has a standard error of 0.07 nats at the top of the curve,
    # so a fixed 5e-3 tolerance flags the two upward steps as violations when
    # both are a third of one standard error. What is being tested is whether
    # information ever demonstrably increases, and it does not.
    error = [v / np.sqrt(N_STARTS) for v in spread]
    rises = [
        (LEADS[i], first[i + 1] - first[i])
        for i in range(len(first) - 1)
        if first[i + 1] - first[i] > error[i]
    ]
    print(f"#   information falls {first[0]:.3f} -> {first[-1]:.3f} nats "
          f"(floor {floor:.4f}); steps that rise by more than one standard "
          f"error: {len(rises)} of {len(first) - 1}")
    monotone = not rises

    print("\n# --- 3. how much does the forcing tell you? ---")
    second = []
    for rho in RHO_SHIFTS:
        shifted = _climatology(rho, seed=int(rho))
        value = information.binned_relative_entropy(
            shifted[:, VARIABLE], reference, bins
        )
        second.append(value)
        print(f"#   rho {RHO_BASE:g} -> {rho:g}: mean {VARIABLE_NAME} "
              f"{shifted[:, VARIABLE].mean():6.2f} against "
              f"{reference.mean():6.2f}, information {value:.4f} nats")

    print("\n# --- 4. the crossover ---")
    for rho, value in zip(RHO_SHIFTS, second):
        crossing = next(
            (float(l) for l, d in zip(LEADS, first) if d < value), float("nan")
        )
        print(f"#   rho -> {rho:g} ({value:.4f} nats): today's state is worth "
              f"more only out to lead {crossing:.1f}")

    reference_hist = np.histogram(reference, bins=bins)[0]
    print(f"LEADS = {LEADS}")
    print(f"BINS = {BINS}")
    print(f"RHO_BASE = {RHO_BASE}")
    print(f"RHO_SHIFTS = {RHO_SHIFTS}")
    print(f"N_STARTS = {N_STARTS}")
    print(f"MEMBERS = {MEMBERS}")
    print(f"VARIABLE_NAME = {VARIABLE_NAME!r}")
    _scalar("NOISE_FLOOR", floor)
    print(f"FIRST_MONOTONE = {bool(monotone)}")
    _emit("FIRST_KIND", first, ".6f", per_line=10)
    _emit("FIRST_KIND_SPREAD", spread, ".6f", per_line=10)
    _emit("FIRST_KIND_ERROR", error, ".6f", per_line=10)
    _emit("SECOND_KIND", second, ".6f", per_line=6)
    _emit("CROSSINGS", [
        next((float(l) for l, d in zip(LEADS, first) if d < value), float("nan"))
        for value in second
    ], ".4f", per_line=6)
    _emit("CLIM_HIST", reference_hist, ".1f", per_line=10)

    # Distributions for the figure: the forecast at three leads, plus the
    # shifted climatology, all on the same bins.
    # ONE start, not the pool. Pooling all 32 starts makes the lead-0 histogram
    # thirty-two spikes at thirty-two different values of z -- which is a
    # picture of the climatology being sampled, not of a forecast being sharp,
    # and it contradicts the very point the figure is making. The quantitative
    # curve above averages over starts; this illustration is a single case, and
    # the notebook says so.
    show_leads = (0.0, 5.0, 12.0)
    show_start = 0
    columns = slice(show_start * MEMBERS, (show_start + 1) * MEMBERS)
    print(f"SHOW_LEADS = {show_leads}")
    print(f"SHOW_START = {show_start}")
    _scalar("SHOW_START_Z", float(starts[show_start, VARIABLE]), ".4f")
    for lead in show_leads:
        index = int(round(lead / DT))
        counts = np.histogram(evolved[index, columns, VARIABLE], bins=bins)[0]
        print(f"#   lead {lead:4.1f}: one start, z spans "
              f"{evolved[index, columns, VARIABLE].min():6.2f} to "
              f"{evolved[index, columns, VARIABLE].max():6.2f}")
        _emit(f"FORECAST_HIST_{int(lead * 10):03d}", counts, ".1f", per_line=10)
    warm = _climatology(RHO_SHIFTS[-1], seed=int(RHO_SHIFTS[-1]))
    _emit("WARM_HIST", np.histogram(warm[:, VARIABLE], bins=bins)[0], ".1f",
          per_line=10)
    _scalar("WARM_RHO", RHO_SHIFTS[-1], ".1f")


if __name__ == "__main__":
    began = time.perf_counter()
    print(f"# Lorenz 63, rho = {RHO_BASE:g}; variable {VARIABLE_NAME}; "
          f"dt = {DT}")
    base_climatology = _climatology(RHO_BASE)
    print(f"# climatology: {base_climatology.shape[0]} states over "
          f"{CLIMATE_TIME:g} TU")
    three_ways(base_climatology)
    two_kinds(base_climatology)
    print(f"\n# total {time.perf_counter() - began:.0f}s")
