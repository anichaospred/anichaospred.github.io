#!/usr/bin/env python3
"""Precompute chapter 21's error-growth curves for three error sources.

Truth is Lorenz 96 at :math:`F = 8`. Three ways to be wrong about it, each with
a different short-lead growth law:

* **initial-condition error** -- perfect model, perturbed start:
  :math:`E \\sim \\delta_0 e^{\\lambda t}`, exponential from the outset;
* **deterministic model bias** -- perfect start, wrong :math:`F`:
  :math:`E = (b/\\lambda)(e^{\\lambda t}-1)`, which is **linear in t** while
  :math:`\\lambda t \\ll 1`;
* **stochastic model error** -- perfect start, noise added to the tendency:
  :math:`E = \\sigma\\sqrt{(e^{2\\lambda t}-1)/2\\lambda}`, **diffusive**,
  :math:`\\sim\\sigma\\sqrt t`.

All three end up exponential at the same rate, so the distinction is visible
only at short lead -- which is exactly where an operational forecast lives.

And the combined sweep, which is the chapter's point: lead time as a function of
initial-condition accuracy **and** model bias together. With a perfect model,
reducing the initial error keeps buying lead time. With any bias at all it stops
paying, and the level it stops at is set by the bias alone.

Everything is per component (RMS over the 40 sites) and compared against a
saturation level computed the same way -- ``saturation_level`` returns a norm
over the whole state, so it is divided by sqrt(N). That factor of 6.3 has caused
trouble twice already in this book.

Run from chaos-book/:
    python3 scripts/generate_ch21_data.py        # ~3 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import errorgrowth, integrate, systems  # noqa: E402

N_SITES, FORCING, DT = 40, 8.0, 0.01
T_FINAL, STORE_EVERY = 6.0, 5
N_STARTS = 48
LAMBDA1 = 1.67

IC_AMPLITUDES = (1e-2, 1e-3, 1e-4, 1e-6, 1e-8)
BIASES = (0.01, 0.05, 0.2)
NOISES = (0.02, 0.05, 0.2)
# The combined sweep: every initial amplitude against every bias, plus zero.
SWEEP_BIASES = (0.0,) + BIASES


def _emit(name: str, values, fmt: str = ".6e", per_line: int = 6) -> None:
    items = [
        'float("nan")' if not np.isfinite(v) else format(float(v), fmt)
        for v in np.ravel(values)
    ]
    print(f"{name} = (")
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i : i + per_line]) + ",")
    print(")")


def _truth(states: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return integrate.rk4(systems.lorenz96, states, grid, forcing=FORCING)


def error_curve(
    bank: np.ndarray, truth: np.ndarray, grid: np.ndarray,
    amplitude: float = 0.0, bias: float = 0.0, noise: float = 0.0,
) -> np.ndarray:
    """Mean RMS error of a run that is wrong in the requested ways."""
    rng = np.random.default_rng(0)
    start = bank.copy()
    if amplitude > 0.0:
        perturbation = rng.normal(size=bank.shape)
        perturbation *= amplitude / np.sqrt(
            np.mean(perturbation**2, axis=-1, keepdims=True)
        )
        start = start + perturbation
    run = integrate.rk4_stochastic(
        systems.lorenz96, start, grid, noise_std=noise, seed=11,
        forcing=FORCING + bias,
    )
    return np.sqrt(np.mean((run - truth) ** 2, axis=-1)).mean(axis=1)


def main() -> None:
    started = time.perf_counter()
    spun = _truth(
        systems.lorenz96_uniform_state(FORCING, N_SITES)
        + np.r_[0.01, np.zeros(N_SITES - 1)],
        integrate.trajectory_grid(300.0, DT),
    )
    climatology = spun[5000:]
    saturation = errorgrowth.saturation_level(
        climatology, statistic="mean"
    ) / np.sqrt(N_SITES)
    stride = max(1, climatology.shape[0] // N_STARTS)
    bank = climatology[::stride][:N_STARTS]

    grid = integrate.trajectory_grid(T_FINAL, DT)
    truth = _truth(bank, grid)
    times = grid[::STORE_EVERY]

    print(f"# truth: Lorenz 96, N={N_SITES}, F={FORCING}; {N_STARTS} start states")
    print(f"# saturation (per component, mean) = {saturation:.5f}")
    print(f"SATURATION = {saturation:.6f}")
    print(f"LAMBDA1 = {LAMBDA1}")
    print(f"IC_AMPLITUDES = {IC_AMPLITUDES}")
    print(f"BIASES = {BIASES}")
    print(f"NOISES = {NOISES}")
    print(f"SWEEP_BIASES = {SWEEP_BIASES}")
    _emit("TIMES", times, ".4f", per_line=10)

    def sub(curve):
        return curve[::STORE_EVERY]

    print("\n# --- three error sources, one at a time ---")
    for label, values, kwargs_name in (
        ("IC", IC_AMPLITUDES, "amplitude"),
        ("BIAS", BIASES, "bias"),
        ("NOISE", NOISES, "noise"),
    ):
        rows = []
        for value in values:
            curve = sub(error_curve(bank, truth, grid, **{kwargs_name: value}))
            rows.append(curve)
            window = (times > 0.05) & (times < 0.4)
            log_slope = float(
                np.polyfit(np.log(times[window]), np.log(curve[window]), 1)[0]
            )
            print(f"#   {label} {value:.0e}: E(0.1) = {curve[2]:.3e}, "
                  f"E(end) = {curve[-1]:.3f}, short-lead d lnE/d lnt = "
                  f"{log_slope:+.3f}")
        _emit(f"{label}_CURVES", np.array(rows).ravel(), ".6e",
              per_line=times.size)

    print("\n# --- the combined sweep: does better initialisation still pay? ---")
    table = np.full((len(IC_AMPLITUDES), len(SWEEP_BIASES)), np.nan)
    threshold = 0.3 * saturation
    for i, amplitude in enumerate(IC_AMPLITUDES):
        for j, bias in enumerate(SWEEP_BIASES):
            curve = sub(
                error_curve(bank, truth, grid, amplitude=amplitude, bias=bias)
            )
            hit = np.nonzero(curve >= threshold)[0]
            if hit.size and hit[0] > 0:
                k = int(hit[0])
                table[i, j] = float(
                    np.interp(threshold, [curve[k - 1], curve[k]],
                              [times[k - 1], times[k]])
                )
        print(f"#   delta0 = {amplitude:.0e}: " + "  ".join(
            f"bias {b:g} -> {table[i, j]:.3f}"
            for j, b in enumerate(SWEEP_BIASES)
        ))
    _emit("SWEEP_LEADS", table.ravel(), ".5f", per_line=len(SWEEP_BIASES))

    # The floor each bias imposes, with a perfect initial condition.
    floors = []
    for bias in SWEEP_BIASES[1:]:
        curve = sub(error_curve(bank, truth, grid, bias=bias))
        hit = np.nonzero(curve >= threshold)[0]
        k = int(hit[0])
        floors.append(
            float(np.interp(threshold, [curve[k - 1], curve[k]],
                            [times[k - 1], times[k]]))
        )
        print(f"#   bias {bias:g} with a PERFECT initial condition: "
              f"{floors[-1]:.3f} time units")
    _emit("BIAS_FLOORS", floors, ".5f")
    print(f"# total {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
