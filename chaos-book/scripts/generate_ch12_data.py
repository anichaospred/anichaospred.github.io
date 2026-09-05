#!/usr/bin/env python3
"""Precompute chapter 12's sweeps and its two-scale Lorenz 96 experiments.

The cascade model itself is cheap -- 16 bands over 6 time units is 87 ms, so
chapter 12 integrates it live and the reader can drive alpha and the number of
octaves directly. What is precomputed here is everything that sweeps a
parameter, plus the two-scale Lorenz 96 runs, which are not cheap.

**Why the two-scale runs cost so much.** The fast subsystem is stiff: with
``time_ratio = 10`` it evolves ten times faster than the slow one, which forces
a step of :math:`10^{-3}` where single-scale Lorenz 96 is happy at
:math:`10^{-2}`. One time unit of a 32-member ensemble of the 264-variable
system is 271 ms, so the six experiments below at 6 time units each, run twice
(truth and perturbed), come to about a minute natively and ten times that under
Pyodide.

The 32-member ensemble is not optional. A single realisation gives a
non-monotonic time-to-threshold -- the first version of this experiment had the
slow-seeded case reaching the threshold *sooner* from a larger initial error
than from a smaller one, which is pure sampling noise in a chaotic system.
Averaging over 32 decorrelated base states makes every curve monotone.

Run from chaos-book/:
    python3 scripts/generate_ch12_data.py        # ~3 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import errorgrowth, integrate, systems  # noqa: E402

# ---- cascade sweeps ------------------------------------------------------
BAND_COUNTS = (2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64)
ALPHAS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
AMPLITUDES = (1.0, 1e-2, 1e-4, 1e-8, 1e-12, 1e-16)

# ---- two-scale Lorenz 96 -------------------------------------------------
N_SLOW, N_FAST, FORCING = 8, 32, 20.0
DT = 0.001
N_MEMBERS = 32
T_FINAL = 6.0
STORE_EVERY = 50  # keep 121 samples over 6 time units
SEED_AMPLITUDES = (1e-1, 1e-3, 1e-5, 1e-7, 1e-9)


def _emit(name: str, values, fmt: str = ".6g", per_line: int = 8) -> None:
    # float("nan") rather than a bare nan: the output of this script is pasted
    # into a notebook cell, and `nan` is not a Python name. Caught the hard way
    # -- the export reported "some cells failed to execute" with
    # NameError: name 'nan' is not defined.
    items = [
        'float("nan")' if not np.isfinite(v) else format(float(v), fmt)
        for v in np.ravel(values)
    ]
    print(f"{name} = (")
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i : i + per_line]) + ",")
    print(")")


def _scalar(name: str, value, fmt: str = ".6f") -> None:
    """Emit one float, spelling a non-finite one as `float("nan")`.

    A bare `nan` in the generated data is valid Python only where numpy is in
    scope under that name, which in a marimo data cell it is not -- so it
    becomes a NameError in the reader's browser. It is invisible to
    `grep marimo-error` and does not change the exporter's exit code; only
    stderr reports it. `tests/test_notebooks.py` now catches it in the notebook,
    and this catches it at the source.
    """
    value = float(value)
    if not np.isfinite(value):
        print(f'{name} = float("nan")')
    else:
        print(f"{name} = {format(value, fmt)}")


def cascade_sweeps() -> None:
    print("# --- cascade: contamination time vs resolved octaves ---")
    table = np.empty((len(ALPHAS), len(BAND_COUNTS)))
    for i, alpha in enumerate(ALPHAS):
        for j, bands in enumerate(BAND_COUNTS):
            table[i, j] = errorgrowth.cascade_contamination_time(bands, alpha=alpha)
        print(
            f"#   alpha={alpha:.4f}: "
            + " ".join(f"{v:.4f}" for v in table[i])
        )
    print(f"BAND_COUNTS = {BAND_COUNTS}")
    _emit("CASCADE_ALPHAS", ALPHAS)
    _emit("CASCADE_TIMES", table.ravel(), ".5f", per_line=len(BAND_COUNTS))

    # Per-octave increments, which shrink as 2^(-2 alpha) -- the square of what
    # a naive sum-of-timescales argument predicts.
    print("\n# per-octave increment ratios, against 2^(-2 alpha):")
    for alpha in ALPHAS[1:]:
        octaves = np.arange(6, 22, 2)
        curve = np.array(
            [errorgrowth.cascade_contamination_time(n, alpha=alpha) for n in octaves]
        )
        increments = np.diff(curve) / 2.0
        ratio = increments[-1] / increments[-2]
        print(
            f"#   alpha={alpha:.4f}: measured {ratio:.4f}, "
            f"2^(-2a) = {2.0 ** (-2.0 * alpha):.4f}"
        )

    print("\n# --- cascade: initial amplitude, at the finest and coarsest band ---")
    for label, band in (("FINE", None), ("COARSE", 0)):
        row = [
            errorgrowth.cascade_contamination_time(
                16, alpha=2.0 / 3.0, seed_amplitude=a, seed_band=band
            )
            for a in AMPLITUDES
        ]
        print(f"#   seeded at the {'finest' if band is None else 'coarsest'} band: "
              + " ".join(f"{v:.4f}" for v in row))
        _emit(f"AMP_{label}", row, ".5f")
    _emit("AMP_VALUES", AMPLITUDES, ".3g")

    print("\n# --- cascade: one run's band-by-band history, for the strip plot ---")
    for label, alpha in (("KOLMOGOROV", 2.0 / 3.0), ("FLAT", 0.0)):
        times, errors = errorgrowth.cascade_growth(
            10, alpha=alpha, t_final=6.0, n_times=61
        )
        _emit(f"STRIP_{label}", errors.ravel(), ".5f", per_line=10)
    _emit("STRIP_TIMES", times, ".4f")


def _ensemble_bank() -> np.ndarray:
    """Decorrelated two-scale states, one time unit apart."""
    state = systems.lorenz96_two_scale_state(N_SLOW, N_FAST, FORCING, seed=0)
    grid = integrate.trajectory_grid(30.0, DT)
    state = integrate.rk4(
        systems.lorenz96_two_scale, state, grid,
        n_slow=N_SLOW, n_fast=N_FAST, forcing=FORCING,
    )[-1]
    bank = [state]
    step = integrate.trajectory_grid(1.0, DT)
    for _ in range(N_MEMBERS - 1):
        state = integrate.rk4(
            systems.lorenz96_two_scale, state, step,
            n_slow=N_SLOW, n_fast=N_FAST, forcing=FORCING,
        )[-1]
        bank.append(state)
    return np.array(bank)


def _advance(states: np.ndarray, n_steps: int) -> np.ndarray:
    """Advance without storing the interior: the full history of a 32-member
    ensemble of 264 variables at dt=1e-3 would be 400 MB."""
    grid = integrate.trajectory_grid(n_steps * DT, DT)
    return integrate.rk4(
        systems.lorenz96_two_scale, states, grid,
        n_slow=N_SLOW, n_fast=N_FAST, forcing=FORCING,
    )[-1]


def two_scale_experiments() -> None:
    print("\n# --- two-scale Lorenz 96 ---")
    started = time.perf_counter()
    bank = _ensemble_bank()
    slow_bank, fast_bank = systems.lorenz96_two_scale_split(bank, N_SLOW)
    saturation_slow = float(np.sqrt(2.0) * slow_bank.std())
    saturation_fast = float(np.sqrt(2.0) * fast_bank.std())
    print(f"#   {N_MEMBERS} base states in {time.perf_counter() - started:.1f}s")
    print(f"#   climatological RMS difference: slow {saturation_slow:.4f}, "
          f"fast {saturation_fast:.4f}")
    _scalar("TWO_SCALE_SAT_SLOW", saturation_slow, ".5f")
    _scalar("TWO_SCALE_SAT_FAST", saturation_fast, ".5f")
    print(f"TWO_SCALE_AMPLITUDES = {SEED_AMPLITUDES}")

    n_samples = int(T_FINAL / (DT * STORE_EVERY)) + 1
    sample_times = np.arange(n_samples) * DT * STORE_EVERY
    _emit("TWO_SCALE_TIMES", sample_times, ".4f")

    rng = np.random.default_rng(1)
    for where in ("FAST", "SLOW"):
        slow_curves, fast_curves = [], []
        for amplitude in SEED_AMPLITUDES:
            block_size = N_SLOW * N_FAST if where == "FAST" else N_SLOW
            block = rng.normal(0.0, 1.0, (N_MEMBERS, block_size))
            block *= amplitude / np.sqrt(
                np.mean(block**2, axis=1, keepdims=True)
            )
            perturbation = np.zeros_like(bank)
            if where == "FAST":
                perturbation[:, N_SLOW:] = block
            else:
                perturbation[:, :N_SLOW] = block

            truth = bank.copy()
            run = bank + perturbation
            slow_error, fast_error = [], []
            for _ in range(n_samples):
                difference = run - truth
                d_slow, d_fast = systems.lorenz96_two_scale_split(
                    difference, N_SLOW
                )
                slow_error.append(
                    float(np.sqrt(np.mean(d_slow**2, axis=-1)).mean())
                )
                fast_error.append(
                    float(np.sqrt(np.mean(d_fast**2, axis=-1)).mean())
                )
                truth = _advance(truth, STORE_EVERY)
                run = _advance(run, STORE_EVERY)
            slow_curves.append(slow_error)
            fast_curves.append(fast_error)
            print(
                f"#   {where} seed {amplitude:.0e}: slow error "
                f"{slow_error[0]:.2e} -> {slow_error[-1]:.3f}   "
                f"({time.perf_counter() - started:.0f}s elapsed)"
            )
        _emit(f"TWO_SCALE_{where}_SLOWERR", np.array(slow_curves).ravel(),
              ".5e", per_line=n_samples)
        _emit(f"TWO_SCALE_{where}_FASTERR", np.array(fast_curves).ravel(),
              ".5e", per_line=n_samples)

    # Leading Lyapunov exponent of the coupled system, by twin trajectories.
    amplitude = 1e-8
    block = rng.normal(0.0, 1.0, bank.shape)
    block *= amplitude / np.sqrt(np.mean(block**2, axis=1, keepdims=True))
    truth, run = bank.copy(), bank + block
    separation, times = [], []
    for i in range(60):
        separation.append(float(np.sqrt(np.mean((run - truth) ** 2, axis=-1)).mean()))
        times.append(i * 25 * DT)
        truth = _advance(truth, 25)
        run = _advance(run, 25)
    separation = np.array(separation)
    times = np.array(times)
    window = (separation > 10.0 * amplitude) & (separation < 1e-3)
    rate = float(np.polyfit(times[window], np.log(separation[window]), 1)[0])
    print(f"#   coupled lambda_1 = {rate:.3f} per time unit "
          f"(doubling {np.log(2) / rate:.4f} TU), fitted over {int(window.sum())} points")
    _scalar("TWO_SCALE_LAMBDA1", rate, ".4f")
    _emit("TWO_SCALE_SEP_TIMES", times, ".4f")
    _emit("TWO_SCALE_SEP", separation, ".5e")
    print(f"# total {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    cascade_sweeps()
    two_scale_experiments()
