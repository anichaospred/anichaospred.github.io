#!/usr/bin/env python3
"""Precompute chapter 9's error-growth curves.

Same split as chapter 8, for the same reason: the expensive half is knob-free
and the interactive half is a fit over a stored curve. Integrating a
1024-member twin-experiment ensemble is seconds natively and tens of seconds
under Pyodide; re-fitting a model to the resulting curve is microseconds. So the
curves are computed here and the notebook fits them live, which is also the
honest arrangement -- the reader sees one measured curve and changes the model,
rather than a curve that moves when the model does.

**Why the ensembles are large.** The Lorenz 63 mean error curve is
*non-monotonic* -- the error falls back for a stretch on the approach to
saturation -- and that survives 1024 members, so it is a property of the
attractor's two lobes and not sampling noise. Fewer members make it worse
without making it go away: the local growth rate dips to -2.4 per time unit at
64 members against -1.6 at 1024. Anything read off the local rate needs the
large ensemble.

**Two statistics that must not be mixed.** The ensemble *mean* error is compared
against the *mean* pair distance, not the RMS one. By Jensen the mean is smaller,
by a factor of 0.889 for Lorenz 63 (the two lobes give a broad distribution of
pair distances) and 0.995 for Lorenz 96 (distances concentrate in 40
dimensions). Mixing them makes the Lorenz 63 curve look as though it stops
growing at 89 % of saturation, which then appears as a 12 % error in the
logistic model's *shape*. That mistake accounted for most of an apparent
discrepancy before it was caught.

Run from chaos-book/:
    python3 scripts/generate_ch09_data.py        # ~2 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import errorgrowth, integrate, systems  # noqa: E402

DT = 0.01
AMPLITUDES = (1e-8, 1e-6, 1e-4, 1e-2, 1e-1)
STORE_EVERY = 5

SYSTEMS = {
    "L63": dict(
        rhs=systems.lorenz63,
        start=np.array([1.0, 1.0, 20.0]),
        spin=3000.0,
        members=1024,
        t_final=32.0,
        params={},
        lambda1=0.9056,
    ),
    "L96": dict(
        rhs=systems.lorenz96,
        start=systems.lorenz96_uniform_state(8.0, 40) + np.r_[0.01, np.zeros(39)],
        spin=800.0,
        members=256,
        t_final=18.0,
        params=dict(forcing=8.0),
        lambda1=1.67,
    ),
}


def _emit(name: str, values, fmt: str = ".6e", per_line: int = 6) -> None:
    items = [
        'float("nan")' if not np.isfinite(v) else format(float(v), fmt)
        for v in np.ravel(values)
    ]
    print(f"{name} = (")
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i : i + per_line]) + ",")
    print(")")


def run(key: str) -> None:
    spec = SYSTEMS[key]
    started = time.perf_counter()
    trajectory = integrate.rk4(
        spec["rhs"], spec["start"],
        integrate.trajectory_grid(spec["spin"], DT), **spec["params"],
    )[5000:]

    saturation_mean = errorgrowth.saturation_level(trajectory, statistic="mean")
    saturation_rms = errorgrowth.saturation_level(trajectory, statistic="rms")
    spread = float(
        np.sqrt(np.mean(np.sum((trajectory - trajectory.mean(axis=0)) ** 2, axis=-1)))
    )
    print(f"\n# --- {key} ---")
    print(f"#   {trajectory.shape[0]} states; saturation mean {saturation_mean:.4f}, "
          f"rms {saturation_rms:.4f}, ratio {saturation_mean / saturation_rms:.4f}")
    print(f"#   RMS spread about the mean {spread:.4f}; sqrt(2)*spread "
          f"{np.sqrt(2) * spread:.4f} vs rms saturation {saturation_rms:.4f}")
    print(f"{key}_SAT_MEAN = {saturation_mean:.5f}")
    print(f"{key}_SAT_RMS = {saturation_rms:.5f}")
    print(f"{key}_LAMBDA1 = {spec['lambda1']}")

    stride = max(1, trajectory.shape[0] // spec["members"])
    bank = trajectory[::stride][: spec["members"]]
    grid = integrate.trajectory_grid(spec["t_final"], DT)
    truth = integrate.rk4(spec["rhs"], bank, grid, **spec["params"])

    times = grid[::STORE_EVERY]
    _emit(f"{key}_TIMES", times, ".4f", per_line=10)

    rng = np.random.default_rng(0)
    curves = []
    for amplitude in AMPLITUDES:
        direction = rng.normal(size=bank.shape)
        direction *= amplitude / np.sqrt(
            np.sum(direction**2, axis=-1, keepdims=True)
        )
        perturbed = integrate.rk4(
            spec["rhs"], bank + direction, grid, **spec["params"]
        )
        error = np.sqrt(np.sum((perturbed - truth) ** 2, axis=-1)).mean(axis=1)
        curves.append(error[::STORE_EVERY])
        monotone = bool(np.all(np.diff(error) > -1e-12))
        print(
            f"#   delta0 = {amplitude:.0e}: error {error[0]:.3e} -> "
            f"{error[-1]:.4f}, monotone {monotone}"
        )
    print(f"{key}_AMPLITUDES = {AMPLITUDES}")
    _emit(f"{key}_ERRORS", np.array(curves).ravel(), ".6e",
          per_line=times.size)

    # The local growth rate against error amplitude, from the smallest seed.
    reference = curves[AMPLITUDES.index(1e-8)]
    rate = np.gradient(np.log(reference), times)
    fraction = reference / saturation_mean
    window = (fraction > 0.02) & (fraction < 0.85)
    slope, intercept = np.polyfit(fraction[window], rate[window], 1)
    print(
        f"#   dlnE/dt vs E/E_mean over [0.02, 0.85]: intercept {intercept:.4f} "
        f"(lambda_1 = {spec['lambda1']}), slope {slope:.4f}, "
        f"ratio {slope / intercept:+.4f} (logistic requires -1)"
    )
    print(f"{key}_RATE_SLOPE = {slope:.5f}")
    print(f"{key}_RATE_INTERCEPT = {intercept:.5f}")

    # Early-time exponential rate, fitted well below saturation.
    early = (reference > 3.0 * reference[0]) & (reference < 1e-3 * saturation_mean)
    if early.sum() > 4:
        early_rate = float(np.polyfit(times[early], np.log(reference[early]), 1)[0])
        print(f"#   early-time exponential rate {early_rate:.4f} "
              f"({int(early.sum())} points) against lambda_1 = {spec['lambda1']}")
        print(f"{key}_EARLY_RATE = {early_rate:.5f}")
    print(f"#   {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    for key in SYSTEMS:
        run(key)
