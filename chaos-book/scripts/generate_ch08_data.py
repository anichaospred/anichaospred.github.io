#!/usr/bin/env python3
"""Precompute chapter 8's curves.

This chapter splits along an unusually clean line. Every expensive computation
is **knob-free** -- forming :math:`C(r)` for Lorenz 63 is :math:`O(N^2)` in the
sample and does not depend on anything the reader chooses -- while the one thing
the reader *does* choose, the scaling window, costs a `np.polyfit` over a stored
curve. So the curves are precomputed here and the notebook re-fits them on every
slider drag, which is both instant and exactly the pedagogical point: the reader
sees one fixed curve and moves the window across it, rather than watching the
curve itself change.

What is computed:

* the Grassberger-Procaccia correlation sum for Lorenz 63 over a **wide** radius
  range -- deliberately wider than the scaling window, so the reader can drag
  the fit into the saturated and the noise-dominated regions and watch it return
  a wrong answer;
* the same for three Theiler windows, since excluding temporally adjacent pairs
  is the other trap and cannot be applied after the fact;
* the same for the Henon map;
* box-counting data for the three reference sets whose dimension is known in
  closed form, and for Lorenz 63, where it starves;
* the correlation dimension of a delay embedding of the :math:`x` component
  alone, over embedding dimension and lag.

Run from chaos-book/:
    python3 scripts/generate_ch08_data.py        # ~3 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import dimension, integrate, lyapunov, systems  # noqa: E402

DT = 0.01
MAX_POINTS = 4000
THEILERS = (0, 10, 50)
EMBED_DIMENSIONS = (2, 3, 4, 5, 6)
EMBED_LAGS = (10, 20, 30)


def _emit(name: str, values, fmt: str = ".6g", per_line: int = 8) -> None:
    items = [format(float(v), fmt) for v in np.ravel(values)]
    print(f"{name} = (")
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i : i + per_line]) + ",")
    print(")")


def l63_trajectory(t_final: float = 800.0) -> np.ndarray:
    grid = integrate.trajectory_grid(t_final=t_final, dt=DT)
    return integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), grid
    )[3000:]


def henon_points(n: int = 60_000) -> np.ndarray:
    state = np.array([0.1, 0.0])
    out = []
    for i in range(n):
        state = systems.henon_map(state)
        if i > 1000:
            out.append(state.copy())
    return np.asarray(out)


def correlation_curves() -> None:
    """C(r) over a wide range, for three Theiler windows."""
    trajectory = l63_trajectory()
    prepared = trajectory[:: max(1, trajectory.shape[0] // MAX_POINTS)][:MAX_POINTS]
    diameter = float(
        np.sqrt(((prepared.max(axis=0) - prepared.min(axis=0)) ** 2).sum())
    )
    # Deliberately wider than the scaling window: 3e-4 of the diameter is well
    # into the noise floor and 1.2 is past total saturation.
    radii = np.logspace(
        np.log10(3e-4 * diameter), np.log10(1.2 * diameter), 60
    )
    print(f"# --- Lorenz 63 correlation sums ---")
    print(f"# {prepared.shape[0]} samples, attractor diameter {diameter:.4f}")
    print(f"L63_DIAMETER = {diameter:.6f}")
    _emit("L63_RADII", radii)
    for theiler in THEILERS:
        started = time.perf_counter()
        c = dimension.correlation_sum(
            trajectory, radii, theiler=theiler, max_points=MAX_POINTS
        )
        _emit(f"L63_C_THEILER_{theiler}", c, ".6e")
        nonzero = int((c > 0).sum())
        print(
            f"#   theiler={theiler:3d}: {nonzero}/{c.size} non-empty radii "
            f"({time.perf_counter() - started:.1f}s)"
        )

    # The dimension both ways, at the vetted window, for the comparison table.
    d2, _, _ = dimension.correlation_dimension(
        trajectory, theiler=50, max_points=MAX_POINTS
    )
    spectrum = lyapunov.lyapunov_spectrum(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        dt=DT,
        t_final=400.0,
        t_transient=30.0,
    )
    print(f"L63_D2 = {d2:.4f}")
    print(f"L63_DKY = {lyapunov.kaplan_yorke_dimension(spectrum):.4f}")
    print(f"L63_HKS = {lyapunov.ks_entropy(spectrum):.4f}")
    _emit("L63_SPECTRUM", spectrum, ".6f")


def henon_curve() -> None:
    points = henon_points()
    diameter = float(np.sqrt(((points.max(axis=0) - points.min(axis=0)) ** 2).sum()))
    radii = np.logspace(np.log10(3e-4 * diameter), np.log10(1.2 * diameter), 50)
    c = dimension.correlation_sum(points, radii, theiler=1, max_points=MAX_POINTS)
    d2, _, _ = dimension.correlation_dimension(
        points, fit_range=(0.005, 0.06), theiler=1, max_points=MAX_POINTS
    )
    print(f"\n# --- Henon map, {points.shape[0]} points ---")
    print(f"HENON_DIAMETER = {diameter:.6f}")
    _emit("HENON_RADII", radii)
    _emit("HENON_C", c, ".6e")
    print(f"HENON_D2 = {d2:.4f}")


def box_curves() -> None:
    """Box counting where it works, and where it starves."""
    print("\n# --- box counting: reference sets with exact dimensions ---")
    builders = {
        "cantor": lambda: dimension.cantor_set(200_000, 18, seed=0),
        "koch": lambda: dimension.koch_curve(8),
        "sierpinski": lambda: dimension.sierpinski_triangle(200_000, 24, seed=1),
    }
    for name, build in builders.items():
        points = build()
        low, high = dimension.REFERENCE_WINDOWS[name]
        # Two decades either side of the vetted window, so the reader can drag
        # out of it in both directions.
        scales = np.logspace(np.log10(low / 30.0), np.log10(min(0.5, high * 30.0)), 26)
        exponents = {}
        for q in (0.0, 1.0, 2.0):
            _, _, values, occupancy = dimension.renyi_dimension(
                points, q=q, fit_range=(scales[0], scales[-1]), n_scales=scales.size
            )
            exponents[q] = values
        fitted, _, _, _ = dimension.renyi_dimension(
            points, q=0.0, fit_range=(low, high), n_scales=14
        )
        upper = name.upper()
        print(f"# {name}: n={points.shape[0]}, fitted D_0={fitted:.4f}, "
              f"exact {dimension.REFERENCE_DIMENSIONS[name]:.5f}")
        _emit(f"{upper}_SCALES", scales)
        _emit(f"{upper}_EXP_Q0", exponents[0.0])
        _emit(f"{upper}_EXP_Q1", exponents[1.0])
        _emit(f"{upper}_EXP_Q2", exponents[2.0])
        _emit(f"{upper}_OCCUPANCY", occupancy, ".4g")

    print("\n# --- box counting on Lorenz 63, where it starves ---")
    trajectory = l63_trajectory()[::4]
    scales = np.logspace(np.log10(2e-3), np.log10(0.25), 20)
    _, _, values, occupancy = dimension.renyi_dimension(
        trajectory, q=0.0, fit_range=(scales[0], scales[-1]), n_scales=scales.size
    )
    print(f"L63_BOX_N = {trajectory.shape[0]}")
    _emit("L63_BOX_SCALES", scales)
    _emit("L63_BOX_EXP_Q0", values)
    _emit("L63_BOX_OCCUPANCY", occupancy, ".4g")


def embedding_sweep() -> None:
    """D_2 of a delay embedding of x(t) alone."""
    trajectory = l63_trajectory()
    series = trajectory[:, 0]
    print("\n# --- delay embedding of the x component ---")
    print(f"EMBED_DIMENSIONS = {EMBED_DIMENSIONS}")
    print(f"EMBED_LAGS = {EMBED_LAGS}")
    table = np.empty((len(EMBED_DIMENSIONS), len(EMBED_LAGS)))
    for i, m in enumerate(EMBED_DIMENSIONS):
        for j, lag in enumerate(EMBED_LAGS):
            embedded = dimension.delay_embed(series, m, lag)
            value, _, _ = dimension.correlation_dimension(
                embedded, theiler=50, max_points=3500
            )
            table[i, j] = value
        print(
            f"#   m={m}: " + "  ".join(
                f"lag {lag} -> {table[i, j]:.4f}" for j, lag in enumerate(EMBED_LAGS)
            )
        )
    _emit("EMBED_D2", table.ravel(), ".4f", per_line=len(EMBED_LAGS))


if __name__ == "__main__":
    correlation_curves()
    henon_curve()
    box_curves()
    embedding_sweep()
