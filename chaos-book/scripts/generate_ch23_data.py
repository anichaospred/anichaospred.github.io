#!/usr/bin/env python3
r"""Precompute chapter 23's boundary-forced predictability experiments.

Lorenz 63 with a slowly oscillating Rayleigh number,
:math:`\rho(t) = 28 + A\sin(2\pi t/T)`. With :math:`T` an order of magnitude
longer than the trajectory's own predictability time, :math:`\rho` is a slow
*boundary condition* rather than part of the fast dynamics -- the cheapest
caricature of ENSO, the seasonal cycle, or any other slow driver.

The chapter's whole argument is one decomposition. Measure the forecast
distribution's distance from **two** different climatologies:

* the **unconditional** climatology, pooled over all forcing phases;
* the **phase-conditioned** climatology, given the forcing phase at the
  verification time.

Against the first, forecast information decays to a **floor**, not to zero.
Against the second it decays to the estimator's noise. The floor is what the
forcing knew all along, and it is the whole of seasonal forecasting in one
number.

Five blocks.

1. **The forced climatology**, conditioned on phase: how much the phase alone
   tells you, and how strongly that varies around the cycle.
2. **The two decay curves**, and where the boundary term overtakes the initial
   condition.
3. **Windows of opportunity**: the same measurement resolved by *launch* phase,
   because predictability is not uniform around the cycle.
4. **The knob**: forcing amplitude and period.
5. **A control**: amplitude zero must reproduce chapter 1's unforced result, or
   the decomposition is measuring something other than the forcing.

Run from chaos-book/:
    python3 scripts/generate_ch23_data.py        # ~8 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import information, integrate, systems  # noqa: E402

DT = 0.01
RHO_MEAN, RHO_AMPLITUDE, PERIOD = 28.0, 6.0, 40.0
CLIMATE_TIME, SPINUP = 4000.0, 4000
BINS = tuple(np.round(np.linspace(-5.0, 70.0, 41), 4))
N_PHASE = 8

N_STARTS, MEMBERS, MAX_LEAD = 120, 2000, 30.0
LEADS = tuple(np.round(np.arange(0.0, MAX_LEAD + 1e-9, 1.0), 3))
SPREAD0 = 0.05

AMPLITUDES = (0.0, 2.0, 4.0, 6.0, 9.0)
# Short periods included deliberately: at 40 and above the boundary
# information is nearly flat, which reads as 'period does not matter'.
# It does -- but only through whether the system has time to follow the
# forcing at all, which needs periods comparable to its own adjustment
# time to show.
PERIODS = (2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)


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
    value = float(value)
    if not np.isfinite(value):
        print(f'{name} = float("nan")')
    else:
        print(f"{name} = {format(value, fmt)}")


def _run(amplitude: float, period: float, total: float = CLIMATE_TIME):
    grid = integrate.trajectory_grid(total, DT)
    trajectory = integrate.rk4(
        systems.lorenz63_forced, np.array([1.0, 1.0, 20.0]), grid,
        rho_mean=RHO_MEAN, rho_amplitude=amplitude, period=period,
    )
    return trajectory[SPINUP:], grid[SPINUP:]


def _phase_of(times, period):
    return (np.asarray(times) % period) / period


def _conditional(trajectory, times, period, bins):
    """Phase-conditioned climatologies, and how much each says beyond the pool."""
    phase = _phase_of(times, period)
    edges = np.linspace(0.0, 1.0, N_PHASE + 1)
    pooled = trajectory[:, 2]
    samples, boundary, means = [], [], []
    for i in range(N_PHASE):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        samples.append(pooled[mask])
        boundary.append(
            information.binned_relative_entropy(pooled[mask], pooled, bins)
        )
        means.append(float(pooled[mask].mean()))
    return samples, np.asarray(boundary), np.asarray(means)


# ==========================================================================
# 1. the forced climatology
# ==========================================================================
def forced_climatology(trajectory, times, bins):
    print("# --- 1. what the forcing phase alone tells you ---")
    samples, boundary, means = _conditional(trajectory, times, PERIOD, bins)
    pooled = trajectory[:, 2]
    print(f"#   forced climatology: {pooled.size} states, z mean "
          f"{pooled.mean():.2f}, sd {pooled.std():.2f}")
    print(f"#   {'phase':>12} {'mean z':>8} {'D(phase||pooled)':>18}")
    edges = np.linspace(0.0, 1.0, N_PHASE + 1)
    for i in range(N_PHASE):
        print(f"#   {edges[i]:.2f}-{edges[i+1]:.2f} {means[i]:8.2f} "
              f"{boundary[i]:18.4f}")
    print(f"#   mean {boundary.mean():.4f} nats, range {boundary.min():.4f} "
          f"to {boundary.max():.4f} ({boundary.max()/boundary.min():.1f}x)")
    print(f"N_PHASE = {N_PHASE}")
    print(f"BINS = {BINS}")
    print(f"PERIOD = {PERIOD}")
    print(f"RHO_MEAN = {RHO_MEAN}")
    print(f"RHO_AMPLITUDE = {RHO_AMPLITUDE}")
    _scalar("BOUNDARY_MEAN", boundary.mean())
    _emit("BOUNDARY_BY_PHASE", boundary, ".6f", per_line=8)
    _emit("PHASE_MEAN_Z", means, ".4f", per_line=8)
    _emit("POOLED_HIST", np.histogram(pooled, bins=np.asarray(bins))[0], ".1f",
          per_line=10)
    for i in (1, 5):
        _emit(f"PHASE_HIST_{i}",
              np.histogram(samples[i], bins=np.asarray(bins))[0], ".1f",
              per_line=10)
    return samples, boundary


# ==========================================================================
# 2 & 3. the two decay curves, and windows of opportunity
# ==========================================================================
def decay(trajectory, times, samples, boundary, bins):
    print("\n# --- 2. two decay curves ---")
    pooled = trajectory[:, 2]
    rng = np.random.default_rng(0)
    lead_steps = int(round(MAX_LEAD / DT))
    start_index = np.linspace(
        0, trajectory.shape[0] - lead_steps - 10, N_STARTS
    ).astype(int)

    # Integrate each launch ONCE to the longest lead and sample it.
    started = time.perf_counter()
    runs, launch_phase = [], []
    for index in start_index:
        members = trajectory[index] + rng.normal(0.0, SPREAD0, (MEMBERS, 3))
        t_start = float(times[index])
        grid = np.linspace(t_start, t_start + MAX_LEAD, lead_steps + 1)
        runs.append(integrate.rk4(
            systems.lorenz63_forced, members, grid,
            rho_mean=RHO_MEAN, rho_amplitude=RHO_AMPLITUDE, period=PERIOD,
        ))
        launch_phase.append(float(_phase_of(t_start, PERIOD)))
    print(f"#   {N_STARTS} launches x {MEMBERS} members to lead {MAX_LEAD:g} "
          f"({time.perf_counter() - started:.0f}s)")

    against_pooled, against_phase = [], []
    per_launch_first = np.zeros((N_STARTS, len(LEADS)))
    for slot, lead in enumerate(LEADS):
        step = int(round(lead / DT))
        pooled_row, phase_row = [], []
        for j, index in enumerate(start_index):
            forecast = runs[j][step][:, 2]
            verify_phase = _phase_of(float(times[index]) + lead, PERIOD)
            which = min(int(verify_phase * N_PHASE), N_PHASE - 1)
            pooled_row.append(
                information.binned_relative_entropy(forecast, pooled, bins)
            )
            phase_row.append(
                information.binned_relative_entropy(
                    forecast, samples[which], bins
                )
            )
            per_launch_first[j, slot] = phase_row[-1]
        against_pooled.append(float(np.mean(pooled_row)))
        against_phase.append(float(np.mean(phase_row)))

    floor = float(boundary.mean())
    crossing = next(
        (float(l) for l, v in zip(LEADS, against_phase) if v < floor),
        float("nan"),
    )
    print(f"#   {'lead':>6} {'vs pooled':>11} {'vs phase':>10}")
    for slot in range(0, len(LEADS), 3):
        print(f"#   {LEADS[slot]:6.1f} {against_pooled[slot]:11.4f} "
              f"{against_phase[slot]:10.4f}")
    print(f"#   boundary floor {floor:.4f}; forecast-vs-pooled plateaus at "
          f"{np.mean(against_pooled[-5:]):.4f}")
    print(f"#   residual first-kind at lead {LEADS[-1]:g}: "
          f"{against_phase[-1]:.4f}; floor + residual = "
          f"{floor + against_phase[-1]:.4f}")
    print(f"#   the forcing overtakes the initial state at lead {crossing:.1f}")
    print(f"LEADS = {LEADS}")
    print(f"N_STARTS = {N_STARTS}")
    print(f"MEMBERS = {MEMBERS}")
    _scalar("CROSSING", crossing, ".4f")
    _scalar("PLATEAU", float(np.mean(against_pooled[-5:])))
    _emit("AGAINST_POOLED", against_pooled, ".6f", per_line=8)
    _emit("AGAINST_PHASE", against_phase, ".6f", per_line=8)

    print("\n# --- 3. windows of opportunity ---")
    order = np.argsort(launch_phase)
    edges = np.linspace(0.0, 1.0, N_PHASE + 1)
    horizons, counts = [], []
    threshold = 0.25
    for i in range(N_PHASE):
        rows = [
            j for j in order
            if edges[i] <= launch_phase[j] < edges[i + 1]
        ]
        counts.append(len(rows))
        if not rows:
            horizons.append(float("nan"))
            continue
        curve = per_launch_first[rows].mean(axis=0)
        horizons.append(next(
            (float(l) for l, v in zip(LEADS, curve) if v < threshold),
            float("nan"),
        ))
    horizons = np.asarray(horizons)
    finite = horizons[np.isfinite(horizons)]
    print(f"#   lead at which first-kind information falls below "
          f"{threshold:g} nats, by LAUNCH phase")
    for i in range(N_PHASE):
        print(f"#     {edges[i]:.2f}-{edges[i+1]:.2f} ({counts[i]:2d} launches): "
              f"{horizons[i]:5.1f}")
    if finite.size:
        print(f"#   spread {finite.min():.1f} to {finite.max():.1f} TU "
              f"({finite.max()/finite.min():.2f}x) from "
              f"{min(counts)}-{max(counts)} launches per bin")
        # A spread computed from a handful of launches per bin could be noise.
        # Split the launches in half and ask whether the two halves agree about
        # WHICH phases are the predictable ones; a real signal survives, a
        # sampling artefact does not.
        half_a, half_b = [], []
        for i in range(N_PHASE):
            rows = [j for j in order if edges[i] <= launch_phase[j] < edges[i + 1]]
            for target, subset in ((half_a, rows[0::2]), (half_b, rows[1::2])):
                if subset:
                    curve = per_launch_first[subset].mean(axis=0)
                    target.append(next(
                        (float(l) for l, v in zip(LEADS, curve) if v < threshold),
                        float("nan"),
                    ))
                else:
                    target.append(float("nan"))
        pair = np.array([half_a, half_b])
        usable = np.all(np.isfinite(pair), axis=0)
        correlation = (
            float(np.corrcoef(pair[0][usable], pair[1][usable])[0, 1])
            if usable.sum() > 2 else float("nan")
        )
        print(f"#   split-half correlation across phase bins: "
              f"{correlation:+.3f} ({usable.sum()} usable bins)")
        _scalar("WINDOW_SPLIT_HALF", correlation, ".4f")
        _emit("WINDOW_HALF_A", half_a, ".4f", per_line=8)
        _emit("WINDOW_HALF_B", half_b, ".4f", per_line=8)
    _scalar("WINDOW_THRESHOLD", threshold, ".3f")
    _emit("WINDOW_HORIZON", horizons, ".4f", per_line=8)
    _emit("WINDOW_COUNTS", counts, ".1f", per_line=8)
    _emit("LAUNCH_PHASE", launch_phase, ".5f", per_line=8)


# ==========================================================================
# 4 & 5. the knob, and the zero-amplitude control
# ==========================================================================
def knob(bins):
    print("\n# --- 4. the knob: forcing amplitude and period ---")
    print(f"#   {'amplitude':>10} {'boundary (nats)':>16}")
    amp_boundary = []
    for amplitude in AMPLITUDES:
        trajectory, times = _run(amplitude, PERIOD)
        _s, boundary, _m = _conditional(trajectory, times, PERIOD, bins)
        amp_boundary.append(float(boundary.mean()))
        print(f"#   {amplitude:10.1f} {amp_boundary[-1]:16.4f}")
    print(f"#   {'period':>10} {'boundary (nats)':>16}")
    period_boundary = []
    for period in PERIODS:
        trajectory, times = _run(RHO_AMPLITUDE, period)
        _s, boundary, _m = _conditional(trajectory, times, period, bins)
        period_boundary.append(float(boundary.mean()))
        print(f"#   {period:10.1f} {period_boundary[-1]:16.4f}")

    print("\n# --- 5. the zero-amplitude control ---")
    print(f"#   with no forcing the phase label is meaningless, so the "
          f"phase-conditioned climatology must equal the pooled one and the "
          f"boundary information must collapse to the estimator's floor.")
    print(f"#   measured: {amp_boundary[0]:.5f} nats at amplitude 0, against "
          f"{amp_boundary[AMPLITUDES.index(RHO_AMPLITUDE)]:.4f} at "
          f"amplitude {RHO_AMPLITUDE:g} "
          f"({amp_boundary[AMPLITUDES.index(RHO_AMPLITUDE)]/amp_boundary[0]:.0f}x)")
    print(f"AMPLITUDES = {AMPLITUDES}")
    print(f"PERIODS = {PERIODS}")
    _emit("AMPLITUDE_BOUNDARY", amp_boundary, ".6f", per_line=6)
    _emit("PERIOD_BOUNDARY", period_boundary, ".6f", per_line=6)


if __name__ == "__main__":
    began = time.perf_counter()
    bins_array = np.asarray(BINS)
    print(f"# Lorenz 63, rho(t) = {RHO_MEAN:g} + {RHO_AMPLITUDE:g} "
          f"sin(2 pi t / {PERIOD:g})")
    base_trajectory, base_times = _run(RHO_AMPLITUDE, PERIOD)
    phase_samples, boundary_info = forced_climatology(
        base_trajectory, base_times, bins_array
    )
    decay(base_trajectory, base_times, phase_samples, boundary_info, bins_array)
    knob(bins_array)
    print(f"\n# total {time.perf_counter() - began:.0f}s")
