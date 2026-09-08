#!/usr/bin/env python3
r"""Precompute chapter 3's hierarchy census and its two laws.

The book's organising claim is that a small model teaches something true about
a large one. That is testable, and this chapter tests it: **dimensionless
relationships transfer between rungs; dimensional constants do not; and some
properties exist at no rung below a particular structure.**

Four blocks.

1. **A census.** Every rung's leading exponent, exponent sum against the exact
   trace, saturation level, Kaplan-Yorke dimension and unstable dimension.
2. **What transfers.** :math:`\lambda T = \ln(f\delta_\infty/\delta_0)`
   measured over four rungs -- a one-variable discrete map through a
   forty-variable flow -- and six decades of initial error, with no fitted
   constant.
3. **What does not.** Across Lorenz 96 from eight to forty variables
   :math:`\lambda_1` barely moves while the unstable dimension multiplies.
4. **Where the ladder breaks.** A Kolmogorov cascade has a horizon that ten
   decades of initial accuracy cannot move; every single-scale system's grows
   like :math:`\ln(1/\delta_0)` for ever.

Run from chaos-book/:
    python3 scripts/generate_ch03_data.py        # ~4 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import errorgrowth, integrate, lyapunov, maps, systems  # noqa: E402

CASES = 48
DECADES = (8, 6, 4, 2)
FRACTION = 0.5
L96_SIZES = (8, 12, 20, 40)
CASCADE_DECADES = (2, 4, 6, 8, 10, 12)
CASCADE_BANDS = 24


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


def _attractor(rhs, start, dt, spinup, **params):
    spun = integrate.rk4(rhs, start, integrate.trajectory_grid(spinup, dt), **params)
    return spun[int(0.3 * spun.shape[0]):]


def _flow_rung(name, rhs, jacobian, start, dt, spinup, horizon, trace, **params):
    """Census entry and measured horizons for one continuous-time rung."""
    attractor = _attractor(rhs, start, dt, spinup, **params)
    saturation = errorgrowth.saturation_level(attractor, seed=0)
    spectrum = lyapunov.lyapunov_spectrum(
        rhs, jacobian, attractor[0], dt=dt, t_final=1500.0, t_transient=20.0,
        **params
    )
    rate = float(spectrum[0])
    starts = attractor[:: max(1, attractor.shape[0] // CASES)][:CASES]
    times = integrate.trajectory_grid(horizon, dt)
    rng = np.random.default_rng(1)
    direction = rng.normal(size=starts.shape)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    control = integrate.rk4(rhs, starts, times, **params)

    horizons, curves = [], []
    for decade in DECADES:
        delta0 = 10.0 ** (-decade) * saturation
        twin = integrate.rk4(rhs, starts + delta0 * direction, times, **params)
        error = np.sqrt(((twin - control) ** 2).sum(axis=-1)).mean(axis=1)
        crossed = error > FRACTION * saturation
        horizons.append(
            float(times[int(np.argmax(crossed))]) if crossed.any() else float("nan")
        )
        if decade == DECADES[0]:
            stride = max(1, times.size // 300)
            curves = (times[::stride], error[::stride] / saturation)
    print(f"#   {name:16s} n={np.size(start):4d} lam1 {rate:7.4f} "
          f"sum-trace {abs(spectrum.sum() - trace):.1e} sat {saturation:8.3f} "
          f"D_KY {lyapunov.kaplan_yorke_dimension(spectrum):6.2f} "
          f"n_pos {lyapunov.unstable_dimension(spectrum):3d}")
    return dict(
        name=name, n=int(np.size(start)), rate=rate, spectrum=spectrum,
        saturation=saturation, horizons=horizons, curve=curves,
        ky=float(lyapunov.kaplan_yorke_dimension(spectrum)),
        unstable=int(lyapunov.unstable_dimension(spectrum)),
        trace_error=float(abs(spectrum.sum() - trace)),
    )


def _map_rung():
    """The one-variable discrete rung: the logistic map at r = 3.9."""
    rate = float(
        maps.map_lyapunov_exponent(
            systems.logistic_map, systems.logistic_map_derivative,
            np.array([3.9]),
        )[0]
    )
    rng = np.random.default_rng(3)
    x = rng.uniform(0.05, 0.95, 4000)
    for _ in range(500):
        x = systems.logistic_map(x, r=3.9)
    saturation = float(np.abs(x[:2000] - x[2000:]).mean())
    horizons, curve = [], None
    for decade in DECADES:
        delta0 = 10.0 ** (-decade) * saturation
        a, b = x.copy(), x + delta0
        errors = []
        for _ in range(90):
            errors.append(float(np.abs(b - a).mean()))
            a = systems.logistic_map(a, r=3.9)
            b = systems.logistic_map(b, r=3.9)
        errors = np.asarray(errors)
        crossed = errors > FRACTION * saturation
        horizons.append(
            float(int(np.argmax(crossed))) if crossed.any() else float("nan")
        )
        if decade == DECADES[0]:
            curve = (np.arange(errors.size, dtype=float), errors / saturation)
    print(f"#   {'logistic map':16s} n=   1 lam1 {rate:7.4f} "
          f"(per iteration)            sat {saturation:8.3f}")
    return dict(
        name="logistic map", n=1, rate=rate, spectrum=np.array([rate]),
        saturation=saturation, horizons=horizons, curve=curve,
        ky=float("nan"), unstable=1, trace_error=float("nan"),
    )


def main() -> None:
    started = time.time()
    print("# Generated by scripts/generate_ch03_data.py -- do not edit by hand.")
    print("# Chapter 3: the hierarchy of models.")
    print("# --- 1 and 2. the census, and the horizon law ---")
    rng = np.random.default_rng(0)
    rungs = [_map_rung()]
    rungs.append(_flow_rung(
        "Lorenz 63", systems.lorenz63, systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]), 0.005, 300.0, 34.0,
        -(10.0 + 1.0 + 8.0 / 3.0),
    ))
    for size, horizon in ((12, 24.0), (40, 24.0)):
        rungs.append(_flow_rung(
            f"Lorenz 96 ({size})", systems.lorenz96, systems.lorenz96_jacobian,
            systems.lorenz96_uniform_state(8.0, size)
            + rng.normal(0.0, 0.5, size),
            0.01, 200.0, horizon, -float(size), forcing=8.0,
        ))

    predicted = [np.log(FRACTION * 10.0 ** d) for d in DECADES]
    print(f"#   the law: lambda1 * T against ln(f delta_inf / delta_0)")
    print(f"#   {'rung':>16} " + " ".join(f"{'1e-%d' % d:>9}" for d in DECADES))
    worst = 0.0
    for rung in rungs:
        scaled = [rung["rate"] * h for h in rung["horizons"]]
        rung["scaled"] = scaled
        for value, target in zip(scaled, predicted):
            if np.isfinite(value):
                worst = max(worst, abs(value / target - 1.0))
        print(f"#   {rung['name']:>16} " + " ".join(f"{v:9.3f}" for v in scaled))
    print(f"#   {'predicted':>16} " + " ".join(f"{v:9.3f}" for v in predicted))
    print(f"#   worst departure across all rungs and decades: {100*worst:.1f} %")

    print(f"DECADES = {DECADES}")
    print(f"FRACTION = {FRACTION}")
    print("RUNG_NAMES = " + repr(tuple(r["name"] for r in rungs)))
    _emit("RUNG_N", [r["n"] for r in rungs], ".0f")
    _emit("RUNG_RATE", [r["rate"] for r in rungs], ".6f")
    _emit("RUNG_SATURATION", [r["saturation"] for r in rungs], ".6f")
    _emit("RUNG_KY", [r["ky"] for r in rungs], ".4f")
    _emit("RUNG_UNSTABLE", [r["unstable"] for r in rungs], ".0f")
    _emit("RUNG_TRACE_ERROR", [r["trace_error"] for r in rungs], ".3e")
    _emit("RUNG_HORIZON", np.array([r["horizons"] for r in rungs]), ".6f")
    _emit("RUNG_SCALED", np.array([r["scaled"] for r in rungs]), ".6f")
    _emit("LAW_PREDICTED", predicted, ".6f")
    _scalar("LAW_WORST", 100.0 * worst, ".1f")
    for index, rung in enumerate(rungs):
        _emit(f"CURVE_T_{index}", rung["curve"][0], ".4f")
        _emit(f"CURVE_E_{index}", rung["curve"][1], ".6e")

    print("# --- 3. the rate transfers; the dimension does not ---")
    rates, unstable, kys = [], [], []
    for size in L96_SIZES:
        start = (systems.lorenz96_uniform_state(8.0, size)
                 + np.random.default_rng(10 + size).normal(0.0, 0.5, size))
        attractor = _attractor(systems.lorenz96, start, 0.01, 200.0, forcing=8.0)
        spectrum = lyapunov.lyapunov_spectrum(
            systems.lorenz96, systems.lorenz96_jacobian, attractor[0],
            dt=0.01, t_final=1500.0, t_transient=20.0, forcing=8.0,
        )
        rates.append(float(spectrum[0]))
        unstable.append(lyapunov.unstable_dimension(spectrum))
        kys.append(float(lyapunov.kaplan_yorke_dimension(spectrum)))
        print(f"#   N={size:3d}: lambda1 {rates[-1]:7.4f}  unstable {unstable[-1]:3d}  "
              f"D_KY {kys[-1]:6.2f}")
    spread = 100.0 * (max(rates) / min(rates) - 1.0)
    print(f"#   lambda1 spread {spread:.1f} % over a factor of "
          f"{L96_SIZES[-1]/L96_SIZES[0]:.0f} in N, while the unstable dimension "
          f"goes {unstable[0]} -> {unstable[-1]}")
    print(f"L96_SIZES = {L96_SIZES}")
    _emit("L96_RATE", rates, ".6f")
    _emit("L96_UNSTABLE", unstable, ".0f")
    _emit("L96_KY", kys, ".4f")
    _scalar("L96_RATE_SPREAD", spread, ".1f")
    _scalar("L96_UNSTABLE_SLOPE",
            float(np.polyfit(L96_SIZES, unstable, 1)[0]), ".4f")
    _scalar("L96_KY_SLOPE", float(np.polyfit(L96_SIZES, kys, 1)[0]), ".4f")

    print("# --- 4. where the ladder breaks ---")
    cascade = {}
    for alpha, key in ((errorgrowth.KOLMOGOROV_ALPHA, "SPECTRUM"), (0.0, "SINGLE")):
        times = [
            errorgrowth.cascade_contamination_time(
                n_bands=CASCADE_BANDS, alpha=alpha,
                seed_amplitude=10.0 ** (-d), seed_band=CASCADE_BANDS - 1,
                threshold=0.5, t_max=4000.0,
            )
            for d in CASCADE_DECADES
        ]
        cascade[key] = times
        print(f"#   alpha {alpha:.3f}: " + " ".join(f"{t:8.3f}" for t in times)
              + f"   total gain over ten decades {times[-1]-times[0]:8.3f}")
    print(f"CASCADE_DECADES = {CASCADE_DECADES}")
    print(f"CASCADE_BANDS = {CASCADE_BANDS}")
    _scalar("KOLMOGOROV_ALPHA", errorgrowth.KOLMOGOROV_ALPHA, ".6f")
    _emit("CASCADE_SPECTRUM", cascade["SPECTRUM"], ".6f")
    _emit("CASCADE_SINGLE", cascade["SINGLE"], ".6f")
    _scalar("CASCADE_SPECTRUM_GAIN",
            cascade["SPECTRUM"][-1] - cascade["SPECTRUM"][0], ".4f")
    _scalar("CASCADE_SINGLE_GAIN",
            cascade["SINGLE"][-1] - cascade["SINGLE"][0], ".4f")
    _scalar("CASCADE_LIMIT", float(np.mean(cascade["SPECTRUM"])), ".4f")
    print(f"# total {time.time() - started:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
