#!/usr/bin/env python3
r"""Precompute chapter 2's Richardson experiments.

Richardson integrated the primitive equations by hand in 1922 and got a
six-hour surface-pressure change of 145 hPa, against an observed change of
essentially nothing. The equations were right. Five blocks establish what went
wrong, in a system small enough to see all of it.

1. **The two modes.** The 1-D rotating shallow-water dispersion relation
   :math:`\omega^2 = f^2 + gHk^2`, measured against the model, and the
   balanced state that carries no fast mode at all.
2. **The tendency was real.** A spurious divergent wind of a metre per second
   forces a surface-pressure tendency of about 9 hPa per hour -- exactly linear
   in the wind. Extrapolate it over six hours and you get Richardson's order of
   magnitude.
3. **Extrapolating it was the error.** The tendency oscillates, so the actual
   change is bounded by :math:`A/\omega` however long you wait, while the
   extrapolation :math:`AT` grows without limit.
4. **Two kinds of imbalance.** Divergent imbalance shows up in the pressure
   tendency immediately; rotational imbalance does not show up in it at all,
   and still ruins the forecast. A small pressure tendency is not balance.
5. **The timestep, and why filtering was worth it.** The stability threshold
   measured against the gravity-wave speed, and the factor the 1950 filtered
   equations bought.

Run from chaos-book/:
    python3 scripts/generate_ch02_data.py        # ~2 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import integrate, shallowwater as sw, turbulence  # noqa: E402

N, LENGTH = 256, 1.0e7
DEPTH, CORIOLIS, GRAVITY = 8000.0, 1.0e-4, 9.81
AMPLITUDE, MODE = 120.0, 4
DT, SIX_HOURS = 30.0, 6 * 3600.0
SPURIOUS = (0.0, 0.5, 1.0, 2.0, 5.0)
MODES = (1, 2, 4, 8, 16)
INTERVALS = (0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0)
DEFICITS = (0.0, 0.1, 0.25, 0.5, 1.0)
DEPTHS = (8000.0, 4000.0, 2000.0, 1000.0, 500.0)
GRID = sw.shallow_water_grid(N, LENGTH)


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


def _height(amplitude=AMPLITUDE, mode=MODE, depth=DEPTH):
    return depth + amplitude * np.sin(
        2.0 * np.pi * mode * GRID["x"] / GRID["length"]
    )


def _balanced(amplitude=AMPLITUDE, mode=MODE, depth=DEPTH):
    return sw.geostrophic_balance(
        _height(amplitude, mode, depth), GRID, CORIOLIS, GRAVITY
    )


def _run(state, seconds, dt=DT):
    return integrate.rk4(
        sw.shallow_water_1d, state, integrate.trajectory_grid(seconds, dt),
        grid=GRID, coriolis=CORIOLIS, gravity=GRAVITY,
    )


def _tendency(state):
    return sw.shallow_water_split(
        sw.shallow_water_1d(0.0, state, GRID, CORIOLIS, GRAVITY), GRID
    )


# ==========================================================================
# 1. the two modes
# ==========================================================================
def two_modes():
    print("# --- 1. the two modes ---")
    c = sw.gravity_wave_speed(DEPTH, GRAVITY)
    radius = sw.rossby_radius(DEPTH, CORIOLIS, GRAVITY)
    print(f"#   c = {c:.1f} m/s, Rossby radius {radius/1e3:.0f} km, "
          f"dx = {GRID['dx']/1e3:.1f} km")
    exact, measured, periods = [], [], []
    for mode in MODES:
        k = 2.0 * np.pi * mode / LENGTH
        omega = float(sw.inertia_gravity_frequency(k, DEPTH, CORIOLIS, GRAVITY))
        period = 2.0 * np.pi / omega
        state = sw.shallow_water_state(
            np.zeros(N), np.zeros(N), _height(amplitude=1.0, mode=mode)
        )
        # Resolve the period, but never step past the CFL limit -- which is set
        # by the *shortest* wave on the grid and not by the mode being measured,
        # so the slowest mode here is the one that needs the smallest step. That
        # is section 5's point arriving early, and it broke this script first.
        step = min(period / 200.0, 0.8 * sw.cfl_timestep(GRID, DEPTH, 0.0, GRAVITY))
        times = integrate.trajectory_grid(12.0 * period, step)
        run = _run(state, 12.0 * period, step)
        series = run[:, 0] - run[:, 0].mean()
        crossings = np.where(np.diff(np.signbit(series)))[0]
        assert crossings.size > 5, f"mode {mode}: only {crossings.size} crossings"
        slope = np.polyfit(np.arange(crossings.size), times[crossings], 1)[0]
        exact.append(omega)
        measured.append(np.pi / slope)
        periods.append(period / 3600.0)
        print(f"#   m={mode:3d}: omega {omega:.6e} measured {np.pi/slope:.6e} "
              f"ratio {np.pi/slope/omega:.5f}  period {period/3600:6.2f} h")

    balanced = _balanced()
    du, dv, dh = _tendency(balanced)
    wind = float(np.abs(sw.shallow_water_split(balanced, GRID)[1]).max())
    six = _run(balanced, SIX_HOURS)
    _, _, h0 = sw.shallow_water_split(six[0], GRID)
    _, _, h1 = sw.shallow_water_split(six[-1], GRID)
    print(f"#   balanced jet {wind:.1f} m/s; tendency |du/dt| {np.abs(du).max():.3e}, "
          f"|dh/dt| {np.abs(dh).max():.3e}; 6-h drift "
          f"{np.abs(h1-h0).max():.3e} m")

    print(f"MODES = {MODES}")
    print(f"DEPTH = {DEPTH}")
    print(f"CORIOLIS = {CORIOLIS}")
    print(f"GRAVITY = {GRAVITY}")
    print(f"AMPLITUDE = {AMPLITUDE}")
    print(f"MODE = {MODE}")
    print(f"LENGTH = {LENGTH}")
    print(f"N_POINTS = {N}")
    _scalar("WAVE_SPEED", c, ".2f")
    _scalar("ROSSBY_RADIUS", radius, ".1f")
    _scalar("DX", GRID["dx"], ".1f")
    _scalar("BALANCED_JET", wind, ".2f")
    _scalar("BALANCED_TENDENCY", float(np.abs(du).max()), ".3e")
    _scalar("BALANCED_DRIFT_6H", float(np.abs(h1 - h0).max()), ".3e")
    _emit("OMEGA_EXACT", exact, ".6e")
    _emit("OMEGA_MEASURED", measured, ".6e")
    _emit("OMEGA_PERIOD_HOURS", periods, ".4f")
    _scalar(
        "OMEGA_WORST",
        100.0 * max(abs(m / e - 1.0) for m, e in zip(measured, exact)), ".3f",
    )
    _scalar("INERTIAL_PERIOD_HOURS", 2.0 * np.pi / CORIOLIS / 3600.0, ".2f")
    # the balanced height and wind, for the figure
    stride = max(1, N // 256)
    _emit("PROFILE_X", GRID["x"][::stride] / 1e3, ".2f")
    _emit("PROFILE_HEIGHT", _height()[::stride] - DEPTH, ".4f")
    _emit("PROFILE_WIND",
          sw.shallow_water_split(balanced, GRID)[1][::stride], ".4f")


# ==========================================================================
# 2 and 3. the tendency, and extrapolating it
# ==========================================================================
def the_tendency():
    print("# --- 2. a spurious divergent wind, and the tendency it forces ---")
    tendencies, six_hour = [], []
    for wind in SPURIOUS:
        state = _balanced().copy()
        state[:N] = wind * np.cos(2.0 * np.pi * MODE * GRID["x"] / LENGTH)
        _, _, dh = _tendency(state)
        rate = float(np.abs(dh).max()) * 3600.0
        run = _run(state, SIX_HOURS)
        _, _, h = sw.shallow_water_split(run, GRID)
        change = float(np.abs(h[-1] - h[0]).max())
        tendencies.append(sw.surface_pressure_change(rate, DEPTH))
        six_hour.append(sw.surface_pressure_change(change, DEPTH))
        print(f"#   u' = {wind:4.1f} m/s: dp/dt {tendencies[-1]:8.3f} hPa/h, "
              f"6-h actual {six_hour[-1]:7.3f} hPa, extrapolated "
              f"{6*tendencies[-1]:8.2f} hPa")
    linear = [t / tendencies[2] for t in tendencies]
    print(f"#   tendency / tendency(1 m/s): "
          + ", ".join(f"{v:.4f}" for v in linear))

    print("# --- 3. extrapolating an oscillation ---")
    state = _balanced().copy()
    state[:N] = 1.0 * np.cos(2.0 * np.pi * MODE * GRID["x"] / LENGTH)
    _, _, dh0 = _tendency(state)
    amplitude = float(np.abs(dh0).max())
    omega = float(
        sw.inertia_gravity_frequency(
            2.0 * np.pi * MODE / LENGTH, DEPTH, CORIOLIS, GRAVITY
        )
    )
    longest = max(INTERVALS) * 3600.0
    run = _run(state, longest)
    times = integrate.trajectory_grid(longest, DT)
    _, _, h = sw.shallow_water_split(run, GRID)
    excursion = np.abs(h - h[0]).max(axis=-1)
    bound = amplitude / omega
    extrap, actual = [], []
    for hours in INTERVALS:
        i = int(np.argmin(np.abs(times - hours * 3600.0)))
        extrap.append(sw.surface_pressure_change(amplitude * hours * 3600.0, DEPTH))
        actual.append(sw.surface_pressure_change(float(excursion[i]), DEPTH))
        print(f"#   T = {hours:5.1f} h: extrapolated {extrap[-1]:8.2f} hPa, "
              f"actual {actual[-1]:6.2f} hPa, ratio {extrap[-1]/actual[-1]:6.1f}, "
              f"omega T {omega*hours*3600:6.2f}")
    peak = sw.surface_pressure_change(float(excursion.max()), DEPTH)
    bound_hpa = sw.surface_pressure_change(bound, DEPTH)
    print(f"#   peak excursion {peak:.3f} hPa against the bound A/omega "
          f"{bound_hpa:.3f} hPa: ratio {peak/bound_hpa:.4f}")

    print(f"SPURIOUS = {SPURIOUS}")
    print(f"INTERVALS = {INTERVALS}")
    _emit("TENDENCY_HPA_PER_HOUR", tendencies, ".6f")
    _emit("SIX_HOUR_ACTUAL", six_hour, ".6f")
    _emit("TENDENCY_LINEARITY", linear, ".5f")
    _emit("EXTRAPOLATED", extrap, ".6f")
    _emit("ACTUAL", actual, ".6f")
    _scalar("BOUND_HPA", bound_hpa, ".4f")
    _scalar("PEAK_HPA", peak, ".4f")
    _scalar("BOUND_RATIO", peak / bound_hpa, ".4f")
    _scalar("OMEGA_MODE", omega, ".6e")
    _scalar("PERIOD_MODE_HOURS", 2.0 * np.pi / omega / 3600.0, ".3f")
    # The height at the point where the initial tendency is *largest*, not at
    # an arbitrary one. With u' = u_0 cos(kx) the flux divergence vanishes at
    # x = 0, so sampling there understated the oscillation sevenfold and made
    # the A/omega bound look far too generous.
    probe = int(np.argmax(np.abs(dh0)))
    stride = max(1, times.size // 600)
    print(f"#   probe at x = {GRID['x'][probe]/1e3:.0f} km (index {probe}), "
          f"where |dh/dt| is largest")
    _scalar("PROBE_X_KM", GRID["x"][probe] / 1e3, ".1f")
    _emit("SERIES_TIME", times[::stride] / 3600.0, ".4f")
    _emit("SERIES_PRESSURE",
          sw.surface_pressure_change(h[::stride, probe] - h[0, probe], DEPTH),
          ".6f")
    _emit("SERIES_EXTRAPOLATION",
          sw.surface_pressure_change(amplitude * times[::stride], DEPTH), ".6f")


# ==========================================================================
# 4. two kinds of imbalance
# ==========================================================================
def two_imbalances():
    print("# --- 4. divergent versus rotational imbalance ---")
    rot_tendency, rot_six = [], []
    for deficit in DEFICITS:
        state = _balanced().copy()
        state[N : 2 * N] *= 1.0 - deficit
        _, _, dh = _tendency(state)
        run = _run(state, SIX_HOURS)
        _, _, h = sw.shallow_water_split(run, GRID)
        rot_tendency.append(
            sw.surface_pressure_change(float(np.abs(dh).max()) * 3600.0, DEPTH)
        )
        rot_six.append(
            sw.surface_pressure_change(float(np.abs(h[-1] - h[0]).max()), DEPTH)
        )
        print(f"#   v deficit {deficit:.2f}: dp/dt(0) {rot_tendency[-1]:.4f} "
              f"hPa/h, 6-h change {rot_six[-1]:7.3f} hPa")
    print(f"DEFICITS = {DEFICITS}")
    _emit("ROTATIONAL_TENDENCY", rot_tendency, ".6e")
    _emit("ROTATIONAL_SIX_HOUR", rot_six, ".6f")


# ==========================================================================
# 5. the timestep
# ==========================================================================
def the_timestep():
    print("# --- 5. the stability threshold and the wave speed ---")

    def blows_up(dt, depth, steps=900):
        state = _balanced(depth=depth).copy()
        state[:N] = 1.0 * np.cos(2.0 * np.pi * MODE * GRID["x"] / LENGTH)
        reference = float(np.abs(state[2 * N:] - depth).max())
        current = state
        for _ in range(steps):
            current = integrate.rk4(
                sw.shallow_water_1d, current, np.array([0.0, dt]),
                grid=GRID, coriolis=CORIOLIS, gravity=GRAVITY,
            )[-1]
            if not np.isfinite(current).all():
                return True
            if float(np.abs(current[2 * N:] - depth).max()) > 20.0 * reference:
                return True
        return False

    speeds, thresholds, products = [], [], []
    for depth in DEPTHS:
        t0 = time.time()
        c = sw.gravity_wave_speed(depth, GRAVITY)
        lo, hi = 1.0, 4000.0
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            if blows_up(mid, depth):
                hi = mid
            else:
                lo = mid
        speeds.append(c)
        thresholds.append(lo)
        products.append(lo * c / GRID["dx"])
        print(f"#   H={depth:6.0f} c={c:6.1f}: dt* {lo:7.1f} s, "
              f"dt* c / dx = {products[-1]:.4f}  ({time.time()-t0:.0f}s)")
    spread = 100.0 * (max(products) / min(products) - 1.0)
    print(f"#   Courant constant {min(products):.3f}-{max(products):.3f} "
          f"({spread:.1f} % over a factor of {speeds[0]/speeds[-1]:.0f} in c)")

    jet = float(np.abs(sw.shallow_water_split(_balanced(), GRID)[1]).max())
    gain = (speeds[0] + jet) / jet
    print(f"#   filtering: c + U = {speeds[0]+jet:.0f} -> U = {jet:.0f} m/s, "
          f"a factor of {gain:.1f} in timestep")
    print(f"#   six hours: {SIX_HOURS/thresholds[0]:.0f} steps unfiltered, "
          f"{SIX_HOURS/(thresholds[0]*gain):.0f} filtered")

    print("# --- 5b. the filtered model conserves what it should ---")
    sgrid = turbulence.spectral_grid(64, length=2.0 * np.pi)
    z_hat = turbulence.random_vorticity(sgrid, seed=0)
    e0 = turbulence.energy(z_hat, sgrid)
    q0 = turbulence.enstrophy(z_hat, sgrid)
    advanced = turbulence.advance_vorticity(z_hat, sgrid, 0.005, 400, viscosity=0.0)
    e1 = turbulence.energy(advanced, sgrid)
    q1 = turbulence.enstrophy(advanced, sgrid)
    print(f"#   energy drift {100*abs(e1/e0-1):.6f} %, "
          f"enstrophy drift {100*abs(q1/q0-1):.6f} %")

    print(f"DEPTHS = {DEPTHS}")
    _emit("CFL_SPEED", speeds, ".4f")
    _emit("CFL_THRESHOLD", thresholds, ".4f")
    _emit("CFL_PRODUCT", products, ".6f")
    _scalar("CFL_SPREAD", spread, ".1f")
    _scalar("CFL_COURANT", float(np.mean(products)), ".4f")
    _scalar("FILTER_GAIN", gain, ".2f")
    _scalar("STEPS_UNFILTERED", SIX_HOURS / thresholds[0], ".0f")
    _scalar("STEPS_FILTERED", SIX_HOURS / (thresholds[0] * gain), ".0f")
    _scalar("BVE_ENERGY_DRIFT", 100.0 * abs(e1 / e0 - 1.0), ".6f")
    _scalar("BVE_ENSTROPHY_DRIFT", 100.0 * abs(q1 / q0 - 1.0), ".6f")


def main() -> None:
    started = time.time()
    print("# Generated by scripts/generate_ch02_data.py -- do not edit by hand.")
    print("# Chapter 2: a short history of numerical weather prediction.")
    two_modes()
    the_tendency()
    two_imbalances()
    the_timestep()
    print(f"# total {time.time() - started:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
