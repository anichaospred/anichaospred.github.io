#!/usr/bin/env python3
"""Precompute chapter 10's information curves.

Ensemble forecasts from many start points, then the relative entropy of the
forecast distribution against climatology as a function of lead -- for Lorenz 63
and Lorenz 96, and for a scalar observable in each. About 30 s natively, so a
few minutes under Pyodide, and knob-free.

**Why scalars.** The obvious thing is the relative entropy of the *full* state
distribution, and it does not work. A 500-member Lorenz 63 ensemble collapses
onto a set of dimension 2.06 embedded in three dimensions, so the forecast
covariance is near-singular and ``det Sigma_f`` -- which the dispersion term
depends on logarithmically -- is set by whatever regularisation is applied. This
script records that failure alongside the scalar result, at two regularisation
levels, because the decay rate it produces differs by a factor of two between
them and that is worth showing rather than describing.

**The rate to compare against.** A scalar's forecast variance grows as
:math:`e^{2\\lambda_1 t}` while it is small, and the dispersion term is
:math:`\\tfrac12\\ln(v_c/v_f)`, so :math:`dD/dt \\to -\\lambda_1`. Note
:math:`\\lambda_1`, **not** :math:`h_{KS}`: the two are nearly equal for
Lorenz 63, which has a single positive exponent, and differ six-fold for
Lorenz 96 (1.67 against 10.21), so Lorenz 96 is what settles which one a scalar
observable actually tracks.

Run from chaos-book/:
    python3 scripts/generate_ch10_data.py        # ~1 minute
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import information, integrate, systems  # noqa: E402

DT = 0.01
N_STARTS, N_MEMBERS, SPREAD0 = 80, 500, 1e-3
SAMPLE_EVERY = 20

CASES = {
    "L63": dict(
        rhs=systems.lorenz63,
        start=np.array([1.0, 1.0, 20.0]),
        spin=3000.0,
        t_final=12.0,
        params={},
        lambda1=0.9056,
        h_ks=0.9010,
        variables=(0, 2),
        unit="MTU",
    ),
    "L96": dict(
        rhs=systems.lorenz96,
        start=systems.lorenz96_uniform_state(8.0, 40) + np.r_[0.01, np.zeros(39)],
        spin=600.0,
        t_final=7.0,
        params=dict(forcing=8.0),
        lambda1=1.67,
        h_ks=10.21,
        variables=(0, 20),
        unit="TU",
    ),
}

# Regularisations for the full-state attempt, to show it is the binding choice.
FLOORS = (1e-12, 1e-6)


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
    spec = CASES[key]
    started = time.perf_counter()
    climatology = integrate.rk4(
        spec["rhs"], spec["start"],
        integrate.trajectory_grid(spec["spin"], DT), **spec["params"],
    )[5000:]
    dimension = climatology.shape[-1]
    stride = max(1, climatology.shape[0] // N_STARTS)
    starts = climatology[::stride][:N_STARTS]
    grid = integrate.trajectory_grid(spec["t_final"], DT)
    times = grid[::SAMPLE_EVERY]

    print(f"\n# --- {key}: {climatology.shape[0]} climatological states, "
          f"{N_STARTS} starts x {N_MEMBERS} members ---")
    print(f"{key}_LAMBDA1 = {spec['lambda1']}")
    print(f"{key}_HKS = {spec['h_ks']}")
    print(f"{key}_VARIABLES = {spec['variables']}")
    _emit(f"{key}_TIMES", times, ".4f", per_line=10)

    scalar_stats = {
        var: (float(climatology[:, var].mean()), float(climatology[:, var].var()))
        for var in spec["variables"]
    }
    cov_full = np.cov(climatology.T)
    mean_full = climatology.mean(axis=0)

    scalar = {var: [] for var in spec["variables"]}
    full = {floor: [] for floor in FLOORS}
    rng = np.random.default_rng(0)

    for state in starts:
        ensemble0 = state + rng.normal(0.0, SPREAD0, (N_MEMBERS, dimension))
        trajectory = integrate.rk4(
            spec["rhs"], ensemble0, grid, **spec["params"]
        )
        rows_scalar = {var: [] for var in spec["variables"]}
        rows_full = {floor: [] for floor in FLOORS}
        for k in range(0, trajectory.shape[0], SAMPLE_EVERY):
            member = trajectory[k]
            for var in spec["variables"]:
                mean_c, var_c = scalar_stats[var]
                column = member[:, var]
                rows_scalar[var].append(
                    information.gaussian_information_components(
                        [column.mean()], [[column.var()]], [mean_c], [[var_c]]
                    )
                )
            if key == "L63":
                sample_cov = np.cov(member.T)
                for floor in FLOORS:
                    try:
                        rows_full[floor].append(
                            information.gaussian_relative_entropy(
                                member.mean(axis=0),
                                sample_cov + floor * np.eye(dimension),
                                mean_full, cov_full,
                            )
                        )
                    except Exception:
                        rows_full[floor].append(float("nan"))
        for var in spec["variables"]:
            scalar[var].append(rows_scalar[var])
        if key == "L63":
            for floor in FLOORS:
                full[floor].append(rows_full[floor])

    for var in spec["variables"]:
        stack = np.array(scalar[var]).mean(axis=0)     # (time, 3)
        total, signal, dispersion = stack[:, 0], stack[:, 1], stack[:, 2]
        window = (total > 0.15 * total[0]) & (total < 0.95 * total[0]) & (times > 0.2)
        slope = float(np.polyfit(times[window], total[window], 1)[0])
        print(f"#   variable {var}: D(0) = {total[0]:.3f} nats "
              f"(signal {signal[0]:.3f}, dispersion {dispersion[0]:.3f}); "
              f"dD/dt = {slope:+.4f} nats/{spec['unit']}")
        print(f"#     against -lambda_1 = {-spec['lambda1']:+.4f} "
              f"(ratio {slope / -spec['lambda1']:.3f}), "
              f"-h_KS = {-spec['h_ks']:+.4f} (ratio {slope / -spec['h_ks']:.3f})")
        _emit(f"{key}_V{var}_TOTAL", total, ".6e", per_line=8)
        _emit(f"{key}_V{var}_SIGNAL", signal, ".6e", per_line=8)
        _emit(f"{key}_V{var}_DISPERSION", dispersion, ".6e", per_line=8)
        print(f"{key}_V{var}_SLOPE = {slope:.5f}")

    if key == "L63":
        print("#   full-state attempt, which is dominated by the regularisation:")
        for floor in FLOORS:
            curve = np.array(full[floor]).mean(axis=0)
            window = (times > 0.3) & (times < 4.0)
            slope = float(np.polyfit(times[window], curve[window], 1)[0])
            print(f"#     floor {floor:.0e}: D(0) = {curve[0]:.2f}, "
                  f"dD/dt = {slope:+.4f} nats/MTU")
            # .upper(): "1e-12" -> "1E12M"... the exponent marker must be
            # upper-cased too, or the emitted name is L63_FULL_1eM12 while the
            # notebook naturally writes L63_FULL_1EM12. That mismatch cost an
            # export: the cell raised NameError while `grep marimo-error`
            # returned 0 and every figure still rendered.
            tag = (
                f"{floor:.0e}".replace("-", "M").replace("+", "P")
                .replace(".", "").upper()
            )
            _emit(f"L63_FULL_{tag}", curve, ".6e", per_line=8)
            print(f"L63_FULL_{tag}_SLOPE = {slope:.5f}")
        print("FLOOR_TAGS = " + repr(tuple(
            f"{f:.0e}".replace("-", "M").replace("+", "P").replace(".", "").upper()
            for f in FLOORS
        )))
        print(f"FLOOR_VALUES = {FLOORS}")

    # Mutual information of a scalar with itself at a lag, and its bias.
    if key == "L63":
        series = climatology[:, 0]
        lags = (0, 20, 50, 100, 200, 300, 400, 600, 800, 1200)
        bin_counts = (16, 32, 64, 128)
        print("#   I(x_0 ; x_t): plug-in and Miller-Madow, by bin count")
        plug, corrected = [], []
        for bins in bin_counts:
            for lag in lags:
                a = series[:-lag] if lag else series
                b = series[lag:] if lag else series
                joint, _, _ = np.histogram2d(a, b, bins=bins)
                plug.append(information.mutual_information(joint))
                corrected.append(
                    information.mutual_information(joint, correction="miller_madow")
                )
        print(f"MI_LAGS = {tuple(l * DT for l in lags)}")
        print(f"MI_BINS = {bin_counts}")
        _emit("MI_PLUGIN", plug, ".6e", per_line=len(lags))
        _emit("MI_CORRECTED", corrected, ".6e", per_line=len(lags))
        for bi, bins in enumerate(bin_counts):
            tail = plug[bi * len(lags) + len(lags) - 1]
            print(f"#     bins={bins:3d}: I at zero lag {plug[bi*len(lags)]:.3f}, "
                  f"floor at the longest lag {tail:.4f}")

    print(f"#   {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    for name in CASES:
        run(name)
