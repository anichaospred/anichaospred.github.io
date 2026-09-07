#!/usr/bin/env python3
r"""Precompute chapter 27's bistability and tipping experiments.

The tilted double well, :math:`\dot x = x - x^3 + \mu + \sigma\,\xi`, whose
left state is destroyed at the exact fold :math:`\mu_c = 2/(3\sqrt3)`. Five
blocks, each of which measures a law rather than illustrating one.

1. **Two states, one system.** The stationary density is exactly Boltzmann,
   :math:`p \propto e^{-2V/\sigma^2}`. Measuring the ratio of the two lobes'
   populations against that integral tests the drift and the integrator's
   noise convention together.
2. **Noise-induced transitions.** Kramers' escape time is exponential in
   :math:`1/\sigma^2` with slope exactly :math:`2\Delta V`. One noise level is
   deliberately left censored to show what a censored mean does to the fit.
3. **Critical slowing down.** Variance and lag-1 autocorrelation against their
   exact Ornstein-Uhlenbeck values, and the point at which the linear theory
   stops describing its own system.
4. **Two timing laws with opposite signs.** A swept fold is left *late* by
   :math:`2.3381\cdot3^{-1/6}\gamma^{2/3}` (an Airy constant) and *early* by
   the noise. Both are measured; the noise law needs a rate slow enough that
   the deterministic delay does not contaminate it.
5. **Early warning as a detection problem.** Kendall's :math:`\tau` on
   windowed variance and autocorrelation, with the alarm threshold calibrated
   against a null so the false-alarm rate is fixed at 5 % by construction --
   then run against three scenarios, two of which defeat it.

Run from chaos-book/:
    python3 scripts/generate_ch27_data.py        # ~5 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import earlywarning as ew, integrate, systems  # noqa: E402

MU_C, X_C = systems.double_well_fold()

# --- section 2: Kramers -----------------------------------------------------
KRAMERS_MU = 0.15
KRAMERS_SIGMAS = (0.20, 0.22, 0.25, 0.28, 0.32)
KRAMERS_MEMBERS, KRAMERS_HORIZON, KRAMERS_DT = 400, 2000.0, 0.01

# --- section 3: critical slowing down ---------------------------------------
CSD_MUS = (0.0, 0.15, 0.25, 0.32, 0.36, 0.375, 0.382)
CSD_SIGMA, CSD_SAMPLE, CSD_HORIZON, CSD_MEMBERS, CSD_DT = 0.08, 0.25, 500.0, 200, 0.01

# --- section 4: the two timing laws -----------------------------------------
DELAY_RATES = (0.004, 0.002, 0.001, 0.0005, 0.00025)
ADVANCE_SIGMAS = (0.04, 0.06, 0.09, 0.13, 0.19)
ADVANCE_RATE, ADVANCE_MEMBERS, ADVANCE_DT = 2.5e-5, 200, 0.01

# --- section 5: early warning ------------------------------------------------
EWS_RATE, EWS_SIGMA, EWS_DT, EWS_SAMPLE = 1.0e-4, 0.06, 0.02, 2.0
EWS_MEMBERS, EWS_WIDTH, EWS_STEP = 200, 200, 10        # 400 TU window, 20 TU step
EWS_HOLD, EWS_NOISE_MU, EWS_NOISE_SIGMA = 0.30, 0.25, 0.13
EWS_LEADS = (0.0, 200.0, 400.0, 800.0, 1200.0, 1600.0, 2000.0)
EWS_FALSE_ALARM = 5.0                                   # per cent, by construction


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


# ==========================================================================
# 1. two states, one system: the Boltzmann density
# ==========================================================================
def boltzmann_density():
    print("# --- 1. two states, one system ---")
    edges = np.linspace(-2.0, 2.0, 81)
    centres = 0.5 * (edges[:-1] + edges[1:])
    fine = np.linspace(-2.5, 2.5, 20001)
    grid = integrate.trajectory_grid(400.0, 0.01)

    measured_rows, ratios, exact_ratios = [], [], []
    cases = ((0.0, 0.45), (0.15, 0.45))
    for mu, sigma in cases:
        run = integrate.rk4_stochastic(
            systems.double_well, np.linspace(-1.5, 1.5, 400), grid,
            noise_std=sigma, seed=1, mu=mu,
        )
        sample = run[grid.size // 4:].ravel()
        density, _ = np.histogram(sample, bins=edges, density=True)
        measured_rows.append(density)
        weight = np.exp(
            -2.0 * systems.double_well_potential(fine, mu) / sigma**2
        )
        ratios.append((sample > 0).sum() / (sample < 0).sum())
        exact_ratios.append(weight[fine > 0].sum() / weight[fine < 0].sum())
        print(f"#   mu {mu:.2f} sigma {sigma:.2f}: {sample.size/1e6:.0f}M samples, "
              f"right/left {ratios[-1]:.3f} vs exact {exact_ratios[-1]:.3f}")

    print(f"DENSITY_CASES = {cases}")
    _emit("DENSITY_CENTRES", centres, ".4f")
    _emit("DENSITY_MEASURED", np.stack(measured_rows), ".6f")
    _emit("DENSITY_RATIO", ratios, ".4f")
    _emit("DENSITY_RATIO_EXACT", exact_ratios, ".4f")
    _scalar(
        "DENSITY_WORST_ERROR",
        100.0 * max(abs(m / e - 1.0) for m, e in zip(ratios, exact_ratios)),
        ".2f",
    )


# ==========================================================================
# 2. noise-induced transitions: Kramers
# ==========================================================================
def kramers():
    print("# --- 2. noise-induced transitions ---")
    points = systems.double_well_fixed_points(KRAMERS_MU)
    well, saddle = points[0], points[1]
    barrier = systems.double_well_barrier(KRAMERS_MU)
    curvature_well = 3.0 * well**2 - 1.0
    curvature_saddle = 1.0 - 3.0 * saddle**2
    grid = integrate.trajectory_grid(KRAMERS_HORIZON, KRAMERS_DT)

    taus, escaped, predicted = [], [], []
    for sigma in KRAMERS_SIGMAS:
        t0 = time.time()
        run = integrate.rk4_stochastic(
            systems.double_well, np.full(KRAMERS_MEMBERS, well), grid,
            noise_std=sigma, seed=7, mu=KRAMERS_MU,
        )
        times = ew.escape_times(run, saddle, grid)
        finite = np.isfinite(times)
        # The mean over the members that escaped: censored where finite.all()
        # is False, which is exactly the case kept below to show the damage.
        taus.append(float(times[finite].mean()))
        escaped.append(float(finite.mean()))
        predicted.append(
            ew.kramers_escape_time(barrier, sigma, curvature_well, curvature_saddle)
        )
        print(f"#   sigma {sigma:.2f}: tau {taus[-1]:8.2f} "
              f"(Kramers {predicted[-1]:9.2f}, ratio {taus[-1]/predicted[-1]:.3f}), "
              f"escaped {escaped[-1]:6.1%}  ({time.time()-t0:.0f}s)")

    inverse = np.array([1.0 / s**2 for s in KRAMERS_SIGMAS])
    log_tau = np.log(taus)
    complete = np.array(escaped) >= 1.0
    slope_all = float(np.polyfit(inverse, log_tau, 1)[0])
    slope_clean = float(np.polyfit(inverse[complete], log_tau[complete], 1)[0])
    print(f"#   slope all {slope_all:.5f}, uncensored only {slope_clean:.5f}, "
          f"exact 2dV = {2*barrier:.5f}")

    print(f"KRAMERS_MU = {KRAMERS_MU}")
    print(f"KRAMERS_SIGMAS = {KRAMERS_SIGMAS}")
    _scalar("KRAMERS_BARRIER", barrier)
    _scalar("KRAMERS_WELL", well, ".4f")
    _scalar("KRAMERS_SADDLE", saddle, ".4f")
    _emit("KRAMERS_TAU", taus, ".4f")
    _emit("KRAMERS_TAU_PREDICTED", predicted, ".4f")
    _emit("KRAMERS_ESCAPED", escaped, ".4f")
    _scalar("KRAMERS_SLOPE_ALL", slope_all)
    _scalar("KRAMERS_SLOPE_CLEAN", slope_clean)
    _scalar("KRAMERS_SLOPE_EXACT", 2.0 * barrier)
    _scalar("KRAMERS_SLOPE_BIAS", 100.0 * abs(slope_all / (2 * barrier) - 1.0), ".1f")
    _scalar("KRAMERS_SLOPE_ERROR", 100.0 * abs(slope_clean / (2 * barrier) - 1.0), ".1f")
    _scalar(
        "KRAMERS_PREFACTOR_RATIO",
        float(np.mean([t / p for t, p, c in zip(taus, predicted, complete) if c])),
        ".3f",
    )


# ==========================================================================
# 3. critical slowing down, and where it stops
# ==========================================================================
def critical_slowing_down():
    print("# --- 3. critical slowing down ---")
    every = int(round(CSD_SAMPLE / CSD_DT))
    grid = integrate.trajectory_grid(CSD_HORIZON, CSD_DT)
    rows = []
    for mu in CSD_MUS:
        well = systems.double_well_fixed_points(mu)[0]
        saddle = systems.double_well_fixed_points(mu)[1]
        rate = systems.double_well_restoring_rate(mu)
        run = integrate.rk4_stochastic(
            systems.double_well, np.full(CSD_MEMBERS, well), grid,
            noise_std=CSD_SIGMA, seed=11, mu=mu,
        )
        series = run[grid.size // 5:: every]
        left = float((run > saddle).any(axis=0).mean())   # members that escaped
        sd = float(series.std(axis=0).mean())
        alpha = float(np.mean(ew.lag1_autocorrelation(series)))
        rows.append((
            mu, rate, ew.ou_stationary_std(CSD_SIGMA, rate), sd,
            float(np.exp(rate * CSD_SAMPLE)), alpha, left,
            saddle - well,
        ))
        print(f"#   mu {mu:.3f} lam {rate:7.4f}  sd {sd:.4f} "
              f"(OU {rows[-1][2]:.4f})  ac {alpha:.4f} (OU {rows[-1][4]:.4f})  "
              f"escaped {left:6.1%}  basin {rows[-1][7]:.4f}")

    print(f"CSD_MUS = {CSD_MUS}")
    print(f"CSD_SIGMA = {CSD_SIGMA}")
    print(f"CSD_SAMPLE = {CSD_SAMPLE}")
    _emit("CSD_RATE", [r[1] for r in rows], ".6f")
    _emit("CSD_SD_PREDICTED", [r[2] for r in rows], ".6f")
    _emit("CSD_SD", [r[3] for r in rows], ".6f")
    _emit("CSD_AC_PREDICTED", [r[4] for r in rows], ".6f")
    _emit("CSD_AC", [r[5] for r in rows], ".6f")
    _emit("CSD_ESCAPED", [r[6] for r in rows], ".4f")
    _emit("CSD_BASIN", [r[7] for r in rows], ".6f")
    clean = [r for r in rows if r[6] < 0.02]
    _scalar("CSD_CLEAN_MU_MAX", max(r[0] for r in clean), ".3f")
    _scalar(
        "CSD_CLEAN_WORST",
        100.0 * max(abs(r[3] / r[2] - 1.0) for r in clean), ".1f",
    )
    _scalar(
        "CSD_CLEAN_WORST_AC",
        100.0 * max(abs(r[5] / r[4] - 1.0) for r in clean), ".1f",
    )


# ==========================================================================
# 4. two timing laws with opposite signs
# ==========================================================================
def timing_laws():
    print("# --- 4a. the deterministic delay past a fold ---")
    delays, ratios = [], []
    for rate in DELAY_RATES:
        grid = integrate.trajectory_grid(1.3 * MU_C / rate, 0.005)
        run = integrate.rk4(
            systems.double_well_ramped, np.array([-1.0]), grid,
            mu_start=0.0, mu_rate=rate,
        )[:, 0]
        tipped = rate * grid[int(np.argmax(run > 0.0))]
        delays.append(tipped - MU_C)
        ratios.append(delays[-1] / ew.fold_delay(rate))
        print(f"#   rate {rate:9.6f}: mu_tip {tipped:.5f}, delay {delays[-1]:+.5f}, "
              f"delay/law {ratios[-1]:.4f}")
    print(f"DELAY_RATES = {DELAY_RATES}")
    _emit("DELAY_MU", [MU_C + d for d in delays], ".6f")
    _emit("DELAY", delays, ".6f")
    _emit("DELAY_RATIO", ratios, ".4f")
    _scalar("DELAY_LAW_CONSTANT", ew.fold_delay(1.0), ".4f")
    _scalar("DELAY_BEST_RATIO", ratios[-1], ".4f")

    print("# --- 4b. the noise advance before a fold ---")
    grid = integrate.trajectory_grid(1.15 * MU_C / ADVANCE_RATE, ADVANCE_DT)
    gaps, predictions, naive = [], [], []
    for sigma in ADVANCE_SIGMAS:
        t0 = time.time()
        run = integrate.rk4_stochastic(
            systems.double_well_ramped, np.full(ADVANCE_MEMBERS, -1.0), grid,
            noise_std=sigma, seed=5, mu_start=0.0, mu_rate=ADVANCE_RATE,
        )
        times = ew.escape_times(run, 0.0, grid)
        assert np.isfinite(times).all(), "censored: lengthen the advance run"
        gaps.append(MU_C - ADVANCE_RATE * float(np.median(times)))
        predictions.append(ew.noise_advanced_fold(sigma, ADVANCE_RATE))
        naive.append((3.0 / 8.0 * 3.0**0.25 * sigma**2) ** (2.0 / 3.0))
        print(f"#   sigma {sigma:.2f}: gap {gaps[-1]:.5f}, closed form "
              f"{predictions[-1]:.5f} (ratio {gaps[-1]/predictions[-1]:.3f}), "
              f"bare sigma^4/3 {naive[-1]:.5f} (ratio {gaps[-1]/naive[-1]:.3f})"
              f"  ({time.time()-t0:.0f}s)")
    exponent = float(
        np.polyfit(np.log(ADVANCE_SIGMAS), np.log(gaps), 1)[0]
    )
    print(f"#   fitted exponent {exponent:.3f} against 4/3 = {4/3:.3f}")

    print(f"ADVANCE_SIGMAS = {ADVANCE_SIGMAS}")
    print(f"ADVANCE_RATE = {ADVANCE_RATE}")
    _emit("ADVANCE_GAP", gaps, ".6f")
    _emit("ADVANCE_PREDICTED", predictions, ".6f")
    _emit("ADVANCE_NAIVE", naive, ".6f")
    _scalar("ADVANCE_EXPONENT", exponent, ".3f")
    _scalar(
        "ADVANCE_WORST",
        100.0 * max(abs(g / p - 1.0) for g, p in zip(gaps, predictions)), ".1f",
    )
    _scalar("ADVANCE_NAIVE_LOW", min(g / n for g, n in zip(gaps, naive)), ".2f")
    _scalar("ADVANCE_NAIVE_HIGH", max(g / n for g, n in zip(gaps, naive)), ".2f")
    _scalar("ADVANCE_DELAY_AT_RATE", ew.fold_delay(ADVANCE_RATE), ".6f")


# ==========================================================================
# 5. early warning as a detection problem
# ==========================================================================
def _ews_run(rhs, sigma, seed, **params):
    grid = integrate.trajectory_grid(1.15 * MU_C / EWS_RATE, EWS_DT)
    out = integrate.rk4_stochastic(
        rhs, np.full(EWS_MEMBERS, -1.0), grid, noise_std=sigma, seed=seed, **params
    )
    every = int(round(EWS_SAMPLE / EWS_DT))
    return grid[::every], out[::every]


def _held_ramp(t, x, mu_rate=0.0, mu_cap=1.0):
    """A ramp that stops short of the fold and holds. Not a chaoslib system:
    it exists only to make the point that a *stopped* approach is
    indistinguishable to the indicator, and belongs to this experiment."""
    return x - x**3 + min(mu_rate * t, mu_cap)


def _trend_taus(times, series, lead=0.0):
    """Kendall tau of the windowed variance and autocorrelation, using only
    each member's record up to ``lead`` before it tips."""
    tip = ew.escape_times(series, 0.0, times)
    variance = np.full(series.shape[1], np.nan)
    autocorrelation = np.full(series.shape[1], np.nan)
    for k in range(series.shape[1]):
        stop = times.size if not np.isfinite(tip[k]) else int(
            np.searchsorted(times, tip[k] - lead)
        )
        record = series[:stop, k]
        if record.size < EWS_WIDTH + EWS_STEP:
            continue
        _, v = ew.sliding_variance(record, EWS_WIDTH, EWS_STEP)
        _, a = ew.sliding_lag1_autocorrelation(record, EWS_WIDTH, EWS_STEP)
        variance[k] = ew.kendall_tau(v)
        autocorrelation[k] = ew.kendall_tau(a)
    return variance, autocorrelation, tip


def early_warning():
    print("# --- 5. early warning as a detection problem ---")
    scenarios = {}
    specs = (
        ("null", systems.double_well, EWS_SIGMA, 101, {"mu": 0.0}),
        ("ramp", systems.double_well_ramped, EWS_SIGMA, 102,
         {"mu_start": 0.0, "mu_rate": EWS_RATE}),
        ("short", _held_ramp, EWS_SIGMA, 103,
         {"mu_rate": EWS_RATE, "mu_cap": EWS_HOLD}),
        ("noise", systems.double_well, EWS_NOISE_SIGMA, 104, {"mu": EWS_NOISE_MU}),
    )
    for name, rhs, sigma, seed, params in specs:
        t0 = time.time()
        times, series = _ews_run(rhs, sigma, seed, **params)
        variance, autocorrelation, tip = _trend_taus(times, series)
        scenarios[name] = {
            "times": times, "series": series, "variance": variance,
            "autocorrelation": autocorrelation, "tip": tip,
        }
        print(f"#   {name:>6}: tipped {np.isfinite(tip).mean():6.1%}, "
              f"median tau(var) {np.nanmedian(variance):6.3f}, "
              f"tau(ac) {np.nanmedian(autocorrelation):6.3f}  "
              f"({time.time()-t0:.0f}s)")

    thresholds, alarms = {}, {}
    for key in ("variance", "autocorrelation"):
        thresholds[key] = float(
            np.nanpercentile(scenarios["null"][key], 100.0 - EWS_FALSE_ALARM)
        )
        alarms[key] = [
            float(np.nanmean(scenarios[n][key] > thresholds[key]))
            for n, *_ in specs
        ]
        print(f"#   {key}: threshold {thresholds[key]:.3f} -> alarms "
              + ", ".join(f"{n} {a:.1%}" for (n, *_), a in zip(specs, alarms[key])))

    # lead time: how early can the alarm be raised and still fire?
    lead_rates = []
    for lead in EWS_LEADS:
        variance, _, _ = _trend_taus(
            scenarios["ramp"]["times"], scenarios["ramp"]["series"], lead=lead
        )
        lead_rates.append(float(np.nanmean(variance > thresholds["variance"])))
    print("#   lead (TU) -> alarm rate: "
          + ", ".join(f"{l:.0f}:{r:.1%}" for l, r in zip(EWS_LEADS, lead_rates)))

    # one illustrative realisation, subsampled for the notebook
    times = scenarios["ramp"]["times"]
    tip = scenarios["ramp"]["tip"]
    pick = int(np.argmin(np.abs(tip - np.median(tip))))
    record = scenarios["ramp"]["series"][:, pick]
    stop = int(np.searchsorted(times, tip[pick]))
    centres, variance = ew.sliding_variance(record[:stop], EWS_WIDTH, EWS_STEP)
    _, autocorrelation = ew.sliding_lag1_autocorrelation(
        record[:stop], EWS_WIDTH, EWS_STEP
    )
    stride = max(1, times.size // 400)
    print(f"#   example member {pick}: tips at {tip[pick]:.0f} TU "
          f"(mu {EWS_RATE*tip[pick]:.4f}), {centres.size} windows")

    print(f"EWS_RATE = {EWS_RATE}")
    print(f"EWS_SIGMA = {EWS_SIGMA}")
    print(f"EWS_SAMPLE = {EWS_SAMPLE}")
    print(f"EWS_WIDTH = {EWS_WIDTH}")
    print(f"EWS_HOLD = {EWS_HOLD}")
    print(f"EWS_NOISE_MU = {EWS_NOISE_MU}")
    print(f"EWS_NOISE_SIGMA = {EWS_NOISE_SIGMA}")
    print(f"EWS_FALSE_ALARM = {EWS_FALSE_ALARM}")
    print(f"EWS_LEADS = {EWS_LEADS}")
    print("EWS_SCENARIOS = ('null', 'ramp', 'short', 'noise')")
    _emit("EWS_TIPPED", [float(np.isfinite(scenarios[n]['tip']).mean())
                         for n, *_ in specs], ".4f")
    for key in ("variance", "autocorrelation"):
        tag = "VAR" if key == "variance" else "AC"
        _scalar(f"EWS_THRESHOLD_{tag}", thresholds[key], ".4f")
        _emit(f"EWS_MEDIAN_TAU_{tag}",
              [float(np.nanmedian(scenarios[n][key])) for n, *_ in specs], ".4f")
        _emit(f"EWS_ALARM_{tag}", alarms[key], ".4f")
    _emit("EWS_LEAD_ALARM", lead_rates, ".4f")
    _scalar("EWS_MEDIAN_TIP", float(np.median(tip)), ".1f")
    _scalar("EWS_MEDIAN_TIP_MU", EWS_RATE * float(np.median(tip)), ".4f")
    _scalar("EWS_FOLD_TIME", MU_C / EWS_RATE, ".1f")
    _scalar("EWS_PREDICTED_MU",
            MU_C - ew.noise_advanced_fold(EWS_SIGMA, EWS_RATE), ".4f")
    _emit("EWS_EXAMPLE_TIME", times[:stop][::stride], ".2f")
    _emit("EWS_EXAMPLE_X", record[:stop][::stride], ".4f")
    _emit("EWS_EXAMPLE_TIP", [tip[pick]], ".2f")
    _emit("EWS_EXAMPLE_CENTRES", centres * EWS_SAMPLE, ".2f")
    _emit("EWS_EXAMPLE_VARIANCE", variance, ".6e")
    _emit("EWS_EXAMPLE_AC", autocorrelation, ".4f")


def main() -> None:
    started = time.time()
    print("# Generated by scripts/generate_ch27_data.py -- do not edit by hand.")
    print(f"# Chapter 27: bistability, tipping, and early warning.")
    _scalar("MU_C", MU_C, ".6f")
    _scalar("X_C", X_C, ".6f")
    boltzmann_density()
    kramers()
    critical_slowing_down()
    timing_laws()
    early_warning()
    print(f"# total {time.time() - started:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
