#!/usr/bin/env python3
"""Precompute chapter 14's turbulence diagnostics.

Four things, of which the third is a negative result and is meant to be:

1. **Conservation**, over an inviscid run, as the solver's warrant.
2. **Vorticity snapshots** at a few times, so the reader can see what
   two-dimensional turbulence does that Lorenz 96 does not -- vortices merging,
   filaments being drawn out.
3. **The spectrum at three resolutions, with local slopes.** There is no
   inertial range at 64, 128 or 256 grid points, and the local slopes say so
   loudly. This is recorded rather than tuned away: the chapter's knob is
   resolution, and what resolution buys here is *not* a power law.
4. **The upscale error cascade.** Seed a perturbation in one narrow shell and
   follow the error spectrum. This works, and it is chapter 12's postulated
   cascade in an actual fluid.

Snapshots are stored as coarsened fields -- a 128-point field is 16 K doubles,
and three of them at full resolution would dominate the notebook. They are
decimated to 64 points for display, which is plenty for a printed figure.

Run from chaos-book/:
    python3 scripts/generate_ch14_data.py        # ~3 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import systems, turbulence  # noqa: E402

DT = 0.0015
VISCOSITY = 2.0e-4
DISPLAY = 64          # snapshots decimated to this many points per side

RESOLUTIONS = (64, 128, 256)
# Long enough for the inverse cascade to be VISIBLE. At 1200 steps the field
# is indistinguishable from its initial condition (enstrophy 0.91, spectral
# peak still at its initial k=9) and a figure built from it supports no claim
# about vortex merger at all -- which a first pass through this chapter made
# anyway. By 10000 steps enstrophy is 0.31 and the peak has migrated to k=4.
SNAPSHOT_STEPS = (0, 3000, 10000)
CASCADE_SEED_K = 25.0
CASCADE_BANDS = ((3, 8), (8, 18), (18, 32))
CASCADE_CHUNK, CASCADE_CHUNKS = 200, 9


def _emit(name: str, values, fmt: str = ".6e", per_line: int = 8) -> None:
    items = [
        'float("nan")' if not np.isfinite(v) else format(float(v), fmt)
        for v in np.ravel(values)
    ]
    print(f"{name} = (")
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i : i + per_line]) + ",")
    print(")")


def conservation() -> None:
    print("# --- 1. inviscid conservation, the solver's warrant ---")
    grid = turbulence.spectral_grid(96)
    state = turbulence.random_vorticity(grid, peak=12.0, seed=0)
    energy0 = turbulence.energy(state, grid)
    enstrophy0 = turbulence.enstrophy(state, grid)
    steps, energies, enstrophies = [], [], []
    running = state
    for chunk in range(9):
        if chunk:
            running = turbulence.advance_vorticity(
                running, grid, 0.002, 100, viscosity=0.0
            )
        steps.append(chunk * 100)
        energies.append(abs(turbulence.energy(running, grid) - energy0) / energy0)
        enstrophies.append(
            abs(turbulence.enstrophy(running, grid) - enstrophy0) / enstrophy0
        )
    print(f"#   after {steps[-1]} steps: energy drift {energies[-1]:.2e}, "
          f"enstrophy drift {enstrophies[-1]:.2e}")
    print(f"CONSERVE_STEPS = {tuple(steps)}")
    _emit("CONSERVE_ENERGY", energies, ".4e")
    _emit("CONSERVE_ENSTROPHY", enstrophies, ".4e")


def snapshots() -> None:
    print("\n# --- 2. vorticity snapshots ---")
    grid = turbulence.spectral_grid(128)
    state = turbulence.random_vorticity(grid, peak=10.0, seed=1)
    stride = 128 // DISPLAY
    fields, times, peaks, enstrophies = [], [], [], []
    running, done = state, 0
    for target in SNAPSHOT_STEPS:
        if target > done:
            running = turbulence.advance_vorticity(
                running, grid, DT, target - done, viscosity=VISCOSITY
            )
            done = target
        field = turbulence.vorticity_field(running)[::stride, ::stride]
        fields.append(field / np.abs(field).max())
        times.append(target * DT)
        wavenumbers, spectrum = turbulence.energy_spectrum(running, grid)
        peak = int(wavenumbers[1:][np.argmax(spectrum[1:])])
        peaks.append(peak)
        enstrophies.append(turbulence.enstrophy(running, grid))
        print(f"#   t = {target * DT:5.2f}: enstrophy {enstrophies[-1]:.4f}, "
              f"energy {turbulence.energy(running, grid):.4e}, "
              f"spectral peak k = {peak}")
    print(f"SNAPSHOT_TIMES = {tuple(round(t, 4) for t in times)}")
    print(f"SNAPSHOT_N = {DISPLAY}")
    print(f"SNAPSHOT_PEAKS = {tuple(peaks)}")
    _emit("SNAPSHOT_ENSTROPHY", enstrophies, ".4f")
    _emit("SNAPSHOTS", np.array(fields).ravel(), ".4f", per_line=DISPLAY)

    # The spectral dynamic range, which is the honest contrast with Lorenz 96.
    wavenumbers, spectrum = turbulence.energy_spectrum(running, grid)
    usable = (wavenumbers > 0) & (spectrum > 0)
    ks, es = wavenumbers[usable], spectrum[usable]
    top = int(np.argmax(es))
    print(f"#   final spectrum: peak k={ks[top]:.0f}, falls "
          f"{np.log10(es[top] / es[-1]):.2f} decades over "
          f"{np.log2(ks[-1] / ks[top]):.1f} octaves above the peak")
    print(f"FLOW_DECADES = {np.log10(es[top] / es[-1]):.4f}")
    print(f"FLOW_OCTAVES = {np.log2(ks[-1] / ks[top]):.4f}")
    _emit("FLOW_SPEC_K", ks, ".1f", per_line=16)
    _emit("FLOW_SPEC_E", es, ".6e")


def spectra() -> None:
    print("\n# --- 3. spectra at three resolutions, and the local slopes ---")
    print("#     (there is no inertial range here; that is the point)")
    for n in RESOLUTIONS:
        started = time.perf_counter()
        grid = turbulence.spectral_grid(n)
        state = turbulence.random_vorticity(grid, peak=10.0, seed=2)
        # Scale the viscosity so the dissipation scale tracks the grid.
        viscosity = VISCOSITY * (128.0 / n) ** 2
        state = turbulence.advance_vorticity(
            state, grid, DT, 1200, viscosity=viscosity
        )
        wavenumbers, spectrum = turbulence.energy_spectrum(state, grid)
        slopes = turbulence.local_spectral_slope(wavenumbers, spectrum)

        near_three = [
            int(k) for k in wavenumbers[3:]
            if np.isfinite(slopes[int(k)]) and abs(slopes[int(k)] - 3.0) < 0.4
        ]
        octaves = 0.0
        if near_three:
            runs, current = [], [near_three[0]]
            for a, b in zip(near_three, near_three[1:]):
                if b == a + 1:
                    current.append(b)
                else:
                    runs.append(current)
                    current = [b]
            runs.append(current)
            best = max(runs, key=len)
            octaves = float(np.log2(best[-1] / best[0])) if best[0] > 0 else 0.0
        print(f"#   n={n:4d} (kmax={n // 3}, nu={viscosity:.1e}, "
              f"{time.perf_counter() - started:.1f}s): widest stretch with "
              f"|slope-3|<0.4 spans {octaves:.2f} octaves")
        print(f"#     slopes at k=4,8,16,24: " + " ".join(
            f"{slopes[k]:+.2f}" if k < slopes.size and np.isfinite(slopes[k])
            else "  n/a" for k in (4, 8, 16, 24)
        ))
        print(f"SPEC_{n}_KMAX = {n // 3}")
        print(f"SPEC_{n}_OCTAVES = {octaves:.4f}")
        _emit(f"SPEC_{n}_K", wavenumbers, ".1f", per_line=16)
        _emit(f"SPEC_{n}_E", spectrum, ".6e")
        _emit(f"SPEC_{n}_SLOPE", np.nan_to_num(slopes, nan=np.nan), ".4f")
    print(f"RESOLUTIONS = {RESOLUTIONS}")

    # Lorenz 96's spectrum, for the contrast that motivates the chapter.
    from chaoslib import integrate, spatial

    l96 = integrate.rk4(
        systems.lorenz96,
        systems.lorenz96_uniform_state(8.0, 40) + np.r_[0.01, np.zeros(39)],
        integrate.trajectory_grid(200.0, 0.01), forcing=8.0,
    )[5000:]
    modes, power = spatial.spatial_power_spectrum(l96)
    usable = (modes > 0) & (power > 0)
    ms, ps = modes[usable], power[usable]
    top = int(np.argmax(ps))
    print(f"#   Lorenz 96: peak m={ms[top]:.0f}, falls "
          f"{np.log10(ps[top] / ps[-1]):.2f} decades over "
          f"{np.log2(ms[-1] / ms[top]):.1f} octaves above the peak")
    print(f"L96_DECADES = {np.log10(ps[top] / ps[-1]):.4f}")
    print(f"L96_OCTAVES = {np.log2(ms[-1] / ms[top]):.4f}")
    print(f"L96_PEAK = {int(ms[top])}")
    _emit("L96_MODES", modes, ".1f", per_line=16)
    _emit("L96_POWER", power, ".6e")


def cascade() -> None:
    print("\n# --- 4. the upscale error cascade ---")
    grid = turbulence.spectral_grid(96)
    state = turbulence.random_vorticity(grid, peak=10.0, seed=3)
    state = turbulence.advance_vorticity(state, grid, DT, 400, viscosity=VISCOSITY)
    perturbed = state + turbulence.band_perturbation(
        grid, centre=CASCADE_SEED_K, amplitude=1e-4, seed=4
    )

    truth, run = state, perturbed
    times, totals, fractions = [], [], []
    for chunk in range(CASCADE_CHUNKS):
        if chunk:
            truth = turbulence.advance_vorticity(
                truth, grid, DT, CASCADE_CHUNK, viscosity=VISCOSITY
            )
            run = turbulence.advance_vorticity(
                run, grid, DT, CASCADE_CHUNK, viscosity=VISCOSITY
            )
        wavenumbers, spectrum = turbulence.energy_spectrum(run - truth, grid)
        total = spectrum.sum()
        times.append(chunk * CASCADE_CHUNK * DT)
        totals.append(np.sqrt(total))
        fractions.append([
            spectrum[(wavenumbers >= lo) & (wavenumbers < hi)].sum() / total
            for lo, hi in CASCADE_BANDS
        ])
        print(f"#   t={times[-1]:5.2f}: error {totals[-1]:.3e}, bands " + " ".join(
            f"{v:.3f}" for v in fractions[-1]
        ))
    print(f"CASCADE_BANDS = {CASCADE_BANDS}")
    print(f"CASCADE_SEED_K = {CASCADE_SEED_K}")
    _emit("CASCADE_TIMES", times, ".4f")
    _emit("CASCADE_TOTAL", totals, ".6e")
    _emit("CASCADE_FRACTIONS", np.array(fractions).ravel(), ".5f",
          per_line=len(CASCADE_BANDS))


if __name__ == "__main__":
    started = time.perf_counter()
    conservation()
    snapshots()
    spectra()
    cascade()
    print(f"\n# total {time.perf_counter() - started:.0f}s")
