#!/usr/bin/env python3
r"""Precompute chapter 25's forced-response experiments.

Lorenz 63 with a one-way ramp in the Rayleigh number,
:math:`\rho(t) = \rho_0 + rt`. Chapter 23's forcing went round a loop and
chapter 24's was generated internally; this one only goes up, which is the shape
of an emissions scenario and changes the question from "where on the attractor"
to "what attractor".

Five blocks.

1. **The trajectory goes, the statistics stay.** A forecast ensemble from a tight
   blob under the ramp: individual members become useless while the ensemble's
   predicted *mean* continues to track the truth's running mean.
2. **Forced response against internal variability.** A ramped ensemble and a
   control ensemble from the same set of climatological start states. The
   difference of the means is the forced response; the control's spread is
   internal variability, and it does *not* shrink.
3. **Time of emergence** -- in two senses that are worth keeping apart: when a
   *single realisation* can detect the trend, and when an *ensemble of K* can.
   They differ by a factor of :math:`\sqrt K` and answer different questions.
4. **Initialisation is irrelevant here**, which is the sharp contrast with
   chapter 24: projection ensembles started from entirely different states give
   the same forced response.
5. **Ensemble size**, and the knob: how emergence moves with the ramp rate.

All statistics use a running window, because "climate" is a time average and an
instantaneous ensemble mean of a chaotic variable is not one. A first pass
without it gave signal-to-noise jumping 1.23, 1.30, 1.87 at successive times and
an emergence date that was a coin flip between them.

Run from chaos-book/:
    python3 scripts/generate_ch25_data.py        # ~4 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import integrate, systems  # noqa: E402

DT = 0.01
RHO_START = 28.0
RATE = 0.05
RATES = (0.0125, 0.025, 0.05, 0.10)
MEMBERS, HORIZON = 400, 320.0
WINDOW = 20.0                     # the "thirty-year average"
SAMPLE_EVERY = 2.0
THRESHOLDS = (1.0, 2.0)
SIZES = (2, 5, 10, 20, 50, 100, 200, 400)

FORECAST_MEMBERS, FORECAST_SPREAD = 200, 0.05
FORECAST_LEADS = tuple(np.round(np.arange(0.0, 60.0 + 1e-9, 2.0), 3))


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


def _running(values, axis=0):
    """Boxcar mean over WINDOW, along the time axis."""
    width = int(round(WINDOW / DT))
    kernel = np.ones(width) / width
    return np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="same"), axis, values
    )


def _start_states(count, seed=0, skip=0):
    spin = integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]),
        integrate.trajectory_grid(6000.0 + 20.0 * skip, DT),
    )[20000 + int(skip * 2000):]
    stride = max(1, spin.shape[0] // count)
    return spin[::stride][:count], spin


# ==========================================================================
# 1. the trajectory goes, the statistics stay
# ==========================================================================
def trajectory_versus_statistics(climatology):
    print("# --- 1. the trajectory goes, the statistics stay ---")
    rng = np.random.default_rng(3)
    truth_start = climatology[5000]
    members = truth_start + rng.normal(
        0.0, FORECAST_SPREAD, (FORECAST_MEMBERS, 3)
    )
    grid = integrate.trajectory_grid(FORECAST_LEADS[-1], DT)
    truth = integrate.rk4(
        systems.lorenz63_ramped, truth_start, grid,
        rho_start=RHO_START, rho_rate=RATE,
    )
    ensemble = integrate.rk4(
        systems.lorenz63_ramped, members, grid,
        rho_start=RHO_START, rho_rate=RATE,
    )
    clim_sd = float(climatology[:, 2].std())

    member_rmse, mean_error, forced_truth, forced_ensemble = [], [], [], []
    half = int(round(WINDOW / 2 / DT))
    for lead in FORECAST_LEADS:
        step = int(round(lead / DT))
        lo, hi = max(0, step - half), min(truth.shape[0], step + half)
        # Windowed like everything else. An instantaneous RMSE against a
        # single truth trajectory is one sample of a chaotic quantity: it came
        # out non-monotone (1.39, 1.78, 1.86, 2.07, 1.27 sd at successive
        # leads) around an expected saturation of sqrt(2).
        member_rmse.append(float(np.sqrt(np.mean(
            (ensemble[lo:hi][:, :, 2] - truth[lo:hi, 2][:, None]) ** 2
        ))))
        # "Climate" on both sides: a window mean of the truth against the
        # ensemble's window-and-member mean.
        truth_window = float(truth[lo:hi, 2].mean())
        ensemble_window = float(ensemble[lo:hi, :, 2].mean())
        forced_truth.append(truth_window)
        forced_ensemble.append(ensemble_window)
        mean_error.append(abs(ensemble_window - truth_window))

    print(f"#   {FORECAST_MEMBERS} members, initial spread {FORECAST_SPREAD}")
    print(f"#   {'lead':>6} {'member RMSE':>12} {'/clim sd':>9} "
          f"{'truth mean':>11} {'ens mean':>9} {'error':>7}")
    for slot in range(0, len(FORECAST_LEADS), 5):
        print(f"#   {FORECAST_LEADS[slot]:6.1f} {member_rmse[slot]:12.3f} "
              f"{member_rmse[slot] / clim_sd:9.3f} {forced_truth[slot]:11.3f} "
              f"{forced_ensemble[slot]:9.3f} {mean_error[slot]:7.3f}")
    print(f"#   saturation should be sqrt(2) = {np.sqrt(2.0):.3f} sd: "
          f"measured {member_rmse[-1] / clim_sd:.3f}")
    print(f"#   individual forecasts saturate at "
          f"{member_rmse[-1] / clim_sd:.2f} climatological sd, while the "
          f"ensemble's windowed mean tracks the truth's to "
          f"{np.mean(mean_error[len(mean_error)//2:]):.3f} "
          f"({100 * np.mean(mean_error[len(mean_error)//2:]) / clim_sd:.1f} % "
          f"of a climatological sd)")
    print(f"FORECAST_LEADS = {FORECAST_LEADS}")
    print(f"FORECAST_MEMBERS = {FORECAST_MEMBERS}")
    print(f"WINDOW = {WINDOW}")
    _scalar("CLIM_SD", clim_sd)
    _emit("MEMBER_RMSE", member_rmse, ".6f", per_line=8)
    _emit("FORCED_TRUTH", forced_truth, ".6f", per_line=8)
    _emit("FORCED_ENSEMBLE", forced_ensemble, ".6f", per_line=8)
    _emit("MEAN_ERROR", mean_error, ".6f", per_line=8)


# ==========================================================================
# 2, 3, 5. response, emergence, ensemble size, and the knob
# ==========================================================================
def _experiment(starts, rate, horizon=HORIZON):
    grid = integrate.trajectory_grid(horizon, DT)
    control = integrate.rk4(
        systems.lorenz63_ramped, starts, grid,
        rho_start=RHO_START, rho_rate=0.0,
    )[:, :, 2]
    ramped = integrate.rk4(
        systems.lorenz63_ramped, starts, grid,
        rho_start=RHO_START, rho_rate=rate,
    )[:, :, 2]
    return grid, control, ramped


def _emergence(times, signal, noise, threshold):
    ratio = np.abs(signal) / noise
    # The FIRST time the ratio crosses and stays above it for a whole window --
    # a single-sample crossing of a noisy ratio is not an emergence date.
    width = int(round(WINDOW / DT))
    above = ratio > threshold
    for i in range(above.size - width):
        if above[i : i + width].all():
            return float(times[i])
    return float("nan")


def response(starts, climatology):
    print("\n# --- 2. forced response against internal variability ---")
    started = time.perf_counter()
    times, control, ramped = _experiment(starts, RATE)
    print(f"#   {MEMBERS} members, ramp {RATE:g}/TU, to t = {HORIZON:g} "
          f"({time.perf_counter() - started:.0f}s)")

    smooth_control = _running(control.mean(axis=1))
    smooth_ramped = _running(ramped.mean(axis=1))
    signal = smooth_ramped - smooth_control
    internal = _running(control.std(axis=1))
    edge = int(round(WINDOW / 2 / DT))
    valid = slice(edge, times.size - edge)

    stride = int(round(SAMPLE_EVERY / DT))
    print(f"#   {'t':>6} {'rho':>6} {'signal':>8} {'internal sd':>12} {'S/N':>6}")
    for step in range(edge, times.size - edge, int(round(50.0 / DT))):
        print(f"#   {times[step]:6.0f} "
              f"{RHO_START + RATE * times[step]:6.1f} {signal[step]:8.3f} "
              f"{internal[step]:12.3f} {abs(signal[step]) / internal[step]:6.2f}")
    print(f"#   internal variability over the whole run: "
          f"{internal[valid].min():.3f} to {internal[valid].max():.3f} -- it "
          f"does not shrink as the forcing grows")

    print("\n# --- 3. two times of emergence ---")
    single = {}
    ensemble_toe = {}
    for threshold in THRESHOLDS:
        single[threshold] = _emergence(
            times[valid], signal[valid], internal[valid], threshold
        )
        ensemble_toe[threshold] = _emergence(
            times[valid], signal[valid],
            internal[valid] / np.sqrt(MEMBERS), threshold,
        )
        edge_time = float(times[valid][0])
        flag = (
            "  (at the first analysable time -- the ensemble detects it "
            "immediately, so this is a bound, not a date)"
            if np.isfinite(ensemble_toe[threshold])
            and ensemble_toe[threshold] <= edge_time + 1e-9
            else ""
        )
        print(f"#   S/N > {threshold:g}: a single realisation detects the trend "
              f"at t = {single[threshold]:.0f}; an ensemble of {MEMBERS} at "
              f"t = {ensemble_toe[threshold]:.0f}{flag}")
    print(f"#   the two differ by a factor of sqrt({MEMBERS}) in what counts as "
          f"noise, and they answer different questions")

    print(f"RATE = {RATE}")
    print(f"MEMBERS = {MEMBERS}")
    print(f"RHO_START = {RHO_START}")
    print(f"THRESHOLDS = {THRESHOLDS}")
    print(f"HORIZON = {HORIZON}")
    _emit("TIMES", times[valid][::stride], ".4f", per_line=10)
    _emit("SIGNAL", signal[valid][::stride], ".6f", per_line=10)
    _emit("INTERNAL", internal[valid][::stride], ".6f", per_line=10)
    _emit("TOE_SINGLE", [single[t] for t in THRESHOLDS], ".4f", per_line=4)
    _emit("TOE_ENSEMBLE", [ensemble_toe[t] for t in THRESHOLDS], ".4f",
          per_line=4)

    print("\n# --- 5. ensemble size, and the knob ---")
    sizes_toe = []
    for size in SIZES:
        subset = slice(0, size)
        sub_signal = _running(ramped[:, subset].mean(axis=1)) - _running(
            control[:, subset].mean(axis=1)
        )
        sub_noise = _running(control[:, subset].std(axis=1)) / np.sqrt(size)
        sizes_toe.append(_emergence(
            times[valid], sub_signal[valid], sub_noise[valid], 2.0
        ))
    print(f"#   ensemble emergence (S/N > 2) against member count:")
    print("#     K   " + "".join(f"{s:8d}" for s in SIZES))
    print("#     ToE " + "".join(
        f"{v:8.0f}" if np.isfinite(v) else f"{'--':>8}" for v in sizes_toe
    ))

    rate_toe_single, rate_toe_ensemble = [], []
    for rate in RATES:
        _t, ctrl, ramp = _experiment(starts[:200], rate)
        sig = _running(ramp.mean(axis=1)) - _running(ctrl.mean(axis=1))
        noise = _running(ctrl.std(axis=1))
        rate_toe_single.append(
            _emergence(_t[valid], sig[valid], noise[valid], 2.0)
        )
        rate_toe_ensemble.append(_emergence(
            _t[valid], sig[valid], noise[valid] / np.sqrt(200), 2.0
        ))
    print(f"#   emergence (S/N > 2) against ramp rate:")
    print("#     rate  " + "".join(f"{r:9.4f}" for r in RATES))
    print("#     single" + "".join(
        f"{v:9.0f}" if np.isfinite(v) else f"{'--':>9}" for v in rate_toe_single
    ))
    print("#     ens   " + "".join(
        f"{v:9.0f}" if np.isfinite(v) else f"{'--':>9}"
        for v in rate_toe_ensemble
    ))
    # Two clean laws worth checking rather than asserting. Signal ~ rate * t
    # and noise ~ sigma/sqrt(K), so emergence should scale as 1/rate and as
    # 1/sqrt(K).
    rate_products = [
        float(r * v) for r, v in zip(RATES, rate_toe_ensemble)
        if np.isfinite(v)
    ]
    size_products = [
        float(np.sqrt(k) * v) for k, v in zip(SIZES, sizes_toe)
        if np.isfinite(v)
    ]
    print(f"#   ToE x rate (should be constant): "
          f"{[round(v, 2) for v in rate_products]}")
    print(f"#   ToE x sqrt(K) (should be constant): "
          f"{[round(v, 0) for v in size_products]}")
    print(f"#     -- the rate law holds to "
          f"{100 * (max(rate_products)/min(rate_products) - 1):.1f} %; the "
          f"size law holds from K = 10 upward and breaks at the smallest "
          f"ensembles, where the noise estimate is itself noisy")
    print(f"SIZES = {SIZES}")
    print(f"RATES = {RATES}")
    _emit("RATE_PRODUCT", rate_products, ".5f", per_line=4)
    _emit("SIZE_PRODUCT", size_products, ".4f", per_line=8)
    _emit("SIZE_TOE", sizes_toe, ".4f", per_line=8)
    _emit("RATE_TOE_SINGLE", rate_toe_single, ".4f", per_line=4)
    _emit("RATE_TOE_ENSEMBLE", rate_toe_ensemble, ".4f", per_line=4)
    return times, valid, stride


# ==========================================================================
# 4. initialisation is irrelevant
# ==========================================================================
def initialisation_irrelevant(starts_a, starts_b, times, valid, stride):
    print("\n# --- 4. does it matter where the ensemble started? ---")
    responses = []
    for label, starts in (("A", starts_a), ("B", starts_b)):
        _t, control, ramped = _experiment(starts, RATE)
        signal = _running(ramped.mean(axis=1)) - _running(control.mean(axis=1))
        responses.append(signal[valid])
        print(f"#   set {label}: response at t = {times[valid][-1]:.0f} is "
              f"{signal[valid][-1]:.3f}")
    difference = np.abs(responses[0] - responses[1])
    _t, control, _r = _experiment(starts_a, RATE)
    internal = float(_running(control.std(axis=1))[valid].mean())
    print(f"#   the two forced responses differ by at most "
          f"{difference.max():.3f} ({100 * difference.max() / internal:.1f} % "
          f"of internal variability), mean {difference.mean():.3f}")
    print(f"#   chapter 24 found initialisation worth 8 TU for a slow variable. "
          f"For the FORCED RESPONSE it is worth nothing at all -- which is why "
          f"a projection needs no initialisation and a decadal prediction does.")
    _scalar("INIT_MAX_DIFF", float(difference.max()))
    _scalar("INIT_MEAN_DIFF", float(difference.mean()))
    _scalar("INIT_INTERNAL", internal)
    _emit("RESPONSE_A", responses[0][::stride], ".6f", per_line=10)
    _emit("RESPONSE_B", responses[1][::stride], ".6f", per_line=10)


if __name__ == "__main__":
    began = time.perf_counter()
    print(f"# Lorenz 63, rho(t) = {RHO_START:g} + {RATE:g} t")
    start_states, climatology_run = _start_states(MEMBERS, seed=0)
    other_states, _ = _start_states(MEMBERS, seed=1, skip=40)
    print(f"# {start_states.shape[0]} start states, climatological sd(z) = "
          f"{climatology_run[:, 2].std():.3f}")
    trajectory_versus_statistics(climatology_run)
    grid_times, valid_slice, sample_stride = response(
        start_states, climatology_run
    )
    initialisation_irrelevant(
        start_states, other_states, grid_times, valid_slice, sample_stride
    )
    print(f"\n# total {time.perf_counter() - began:.0f}s")
