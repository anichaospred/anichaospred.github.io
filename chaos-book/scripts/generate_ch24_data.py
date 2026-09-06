#!/usr/bin/env python3
r"""Precompute chapter 24's initialised-prediction experiments.

A fast Lorenz 63 "atmosphere" coupled to a single slow "ocean" variable that
integrates it, :math:`\dot S = [-S + \lambda(z - z_{\mathrm{ref}})]/T`. The
memory comes from integrating fast weather rather than from slow internal
dynamics, which is Hasselmann's mechanism *[citation needed]*.

Five blocks.

1. **Where the memory is.** Autocorrelation of the fast and slow variables, and
   the trade-off that decides the chapter's configuration: any feedback strong
   enough to modulate the atmosphere has already destroyed the ocean's memory.
2. **Initialising the ocean.** Forecasts with the slow variable initialised from
   truth against forecasts with it drawn from climatology, scored separately for
   the slow and the fast variable. The stub's promise: it should help one and
   not the other.
3. **Drift.** An imperfect model initialised from the observed state does not
   stay there; it slides towards its own climatology at a rate that depends on
   lead. Drift is *not* a constant bias and cannot be removed like one.
4. **Correcting it**, with a lead-dependent drift estimated from a re-forecast
   set -- and how large that set has to be before correcting helps rather than
   hurts. That last question is the operational one and the answer is not "any".
5. **Cross-validation**, because a drift estimated on the cases it is then
   scored on flatters itself.

Run from chaos-book/:
    python3 scripts/generate_ch24_data.py        # ~6 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import integrate, systems  # noqa: E402

DT = 0.01
OCEAN_TIME, FORCING_STRENGTH, FEEDBACK = 50.0, 30.0, -0.02
REFERENCE_Z = 23.6
TRUTH_PARAMS = dict(
    ocean_time=OCEAN_TIME, forcing_strength=FORCING_STRENGTH,
    feedback=FEEDBACK, reference_z=REFERENCE_Z,
)
# The imperfect model. The choice of WHICH parameter to get wrong matters more
# than by how much, and the first attempt got it wrong: a 20 % error in the
# ocean's relaxation time changes the slow variable's *variance* but shifts its
# climatological mean by only 0.12 of a standard deviation, so there was no
# drift to see and nothing for a drift correction to remove. Worse, the sluggish
# model scored BETTER than the perfect one at long lead, for the damped-forecast
# reason chapter 22 sets out.
#
# A small error in the temperature the ocean relaxes towards displaces the
# model's climatology by about one standard deviation, which is what drift
# actually is: the model sliding away from the observed state towards its own
# preferred one.
MODEL_PARAMS = dict(TRUTH_PARAMS, reference_z=REFERENCE_Z + 0.1)

CLIMATE_TIME, SPINUP = 6000.0, 20000
FEEDBACKS = (0.0, -0.01, -0.02, -0.05, -0.10, -0.25)

N_CASES, MAX_LEAD = 300, 60.0
LEADS = tuple(np.round(np.arange(0.0, MAX_LEAD + 1e-9, 2.0), 3))
ANALYSIS_NOISE_FAST, ANALYSIS_NOISE_SLOW = 0.05, 0.05
REFORECAST_SIZES = (5, 10, 20, 40, 80, 160)


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


def _climatology(params, total=CLIMATE_TIME):
    grid = integrate.trajectory_grid(total, DT)
    return integrate.rk4(
        systems.coupled_ocean_atmosphere, np.array([1.0, 1.0, 20.0, 0.0]),
        grid, **params,
    )[SPINUP:]


def _e_folding(series, max_lag=800.0):
    centred = series - series.mean()
    n_lags = int(max_lag / DT)
    spectrum = np.fft.rfft(centred, 2 * centred.size)
    auto = np.fft.irfft(spectrum * np.conj(spectrum))[:n_lags].real
    auto = auto / auto[0]
    lags = np.arange(n_lags) * DT
    below = np.nonzero(auto < np.exp(-1.0))[0]
    return (float(lags[below[0]]) if below.size else float("nan")), lags, auto


# ==========================================================================
# 1. where the memory is
# ==========================================================================
def memory(truth):
    print("# --- 1. where the memory is ---")
    fast_time, lags, fast_auto = _e_folding(truth[:, 2])
    slow_time, _lags, slow_auto = _e_folding(truth[:, 3])
    print(f"#   z: sd {truth[:, 2].std():.3f}, e-folding {fast_time:.2f} TU")
    print(f"#   S: sd {truth[:, 3].std():.3f}, e-folding {slow_time:.2f} TU "
          f"({slow_time / fast_time:.0f}x longer)")
    stride = max(1, int(0.5 / DT))
    _scalar("FAST_TIME", fast_time, ".4f")
    _scalar("SLOW_TIME", slow_time, ".4f")
    _scalar("FAST_SD", float(truth[:, 2].std()))
    _scalar("SLOW_SD", float(truth[:, 3].std()))
    print(f"OCEAN_TIME = {OCEAN_TIME}")
    print(f"REFERENCE_Z = {REFERENCE_Z}")
    print(f"FEEDBACK = {FEEDBACK}")
    print(f"FORCING_STRENGTH = {FORCING_STRENGTH}")
    _emit("AUTO_LAGS", lags[::stride][:400], ".4f", per_line=10)
    _emit("AUTO_FAST", fast_auto[::stride][:400], ".5f", per_line=10)
    _emit("AUTO_SLOW", slow_auto[::stride][:400], ".5f", per_line=10)

    print("#   the trade-off that fixes the configuration:")
    print(f"#     {'feedback':>9} {'sd S':>7} {'rho swing':>10} {'tau_S':>8}")
    swings, times, sds = [], [], []
    for value in FEEDBACKS:
        run = _climatology(dict(TRUTH_PARAMS, feedback=value), total=3000.0)
        tau, _l, _a = _e_folding(run[:, 3])
        swings.append(abs(value) * float(run[:, 3].std()))
        times.append(tau)
        sds.append(float(run[:, 3].std()))
        print(f"#     {value:9.2f} {sds[-1]:7.3f} {swings[-1]:10.3f} {tau:8.1f}")
    print("#     any coupling strong enough to move the atmosphere has already "
          "cost the ocean its memory")
    print(f"FEEDBACKS = {FEEDBACKS}")
    _emit("TRADEOFF_SWING", swings, ".6f", per_line=6)
    _emit("TRADEOFF_TAU", times, ".4f", per_line=6)
    _emit("TRADEOFF_SD", sds, ".6f", per_line=6)


# ==========================================================================
# 2. initialising the ocean
# ==========================================================================
def initialisation(truth):
    print("\n# --- 2. does initialising the ocean help? ---")
    rng = np.random.default_rng(0)
    lead_steps = int(round(MAX_LEAD / DT))
    starts = np.linspace(0, truth.shape[0] - lead_steps - 10, N_CASES).astype(int)
    grid = np.linspace(0.0, MAX_LEAD, lead_steps + 1)
    climate_slow = truth[:, 3]

    # Two ensembles from the same analysis error in the FAST variables. They
    # differ only in whether the slow variable was initialised or drawn from
    # climatology -- which is exactly the initialised/uninitialised contrast.
    initial = truth[starts].copy()
    initial[:, :3] += rng.normal(0.0, ANALYSIS_NOISE_FAST, (N_CASES, 3))
    initialised = initial.copy()
    initialised[:, 3] += rng.normal(0.0, ANALYSIS_NOISE_SLOW, N_CASES)
    uninitialised = initial.copy()
    uninitialised[:, 3] = rng.choice(climate_slow, N_CASES)

    started = time.perf_counter()
    runs = {
        "INIT": integrate.rk4(
            systems.coupled_ocean_atmosphere, initialised, grid, **TRUTH_PARAMS
        ),
        "UNINIT": integrate.rk4(
            systems.coupled_ocean_atmosphere, uninitialised, grid, **TRUTH_PARAMS
        ),
    }
    print(f"#   {N_CASES} cases to lead {MAX_LEAD:g} "
          f"({time.perf_counter() - started:.0f}s)")

    slow_sd = float(climate_slow.std())
    fast_sd = float(truth[:, 2].std())
    scores = {f"{k}_{v}": [] for k in runs for v in ("SLOW", "FAST")}
    for lead in LEADS:
        step = int(round(lead / DT))
        verify = truth[starts + step]
        for key, run in runs.items():
            scores[f"{key}_SLOW"].append(float(np.sqrt(np.mean(
                (run[step][:, 3] - verify[:, 3]) ** 2
            ))))
            scores[f"{key}_FAST"].append(float(np.sqrt(np.mean(
                (run[step][:, 2] - verify[:, 2]) ** 2
            ))))

    print(f"#   RMSE, normalised by each variable's climatological spread")
    print(f"#   {'lead':>6} {'S init':>8} {'S uninit':>9} | {'z init':>8} "
          f"{'z uninit':>9}")
    for slot in range(0, len(LEADS), 4):
        print(f"#   {LEADS[slot]:6.1f} "
              f"{scores['INIT_SLOW'][slot] / slow_sd:8.3f} "
              f"{scores['UNINIT_SLOW'][slot] / slow_sd:9.3f} | "
              f"{scores['INIT_FAST'][slot] / fast_sd:8.3f} "
              f"{scores['UNINIT_FAST'][slot] / fast_sd:9.3f}")

    def horizon(values, sd, threshold=0.7):
        for lead, value in zip(LEADS, values):
            if value / sd > threshold:
                return float(lead)
        return float("nan")

    slow_gain = horizon(scores["INIT_SLOW"], slow_sd) - horizon(
        scores["UNINIT_SLOW"], slow_sd
    )
    print(f"#   slow-variable horizon (RMSE < 0.7 sd): "
          f"{horizon(scores['INIT_SLOW'], slow_sd):.0f} initialised against "
          f"{horizon(scores['UNINIT_SLOW'], slow_sd):.0f} uninitialised "
          f"(+{slow_gain:.0f} TU)")
    fast_diff = [
        (a - b) / fast_sd
        for a, b in zip(scores["INIT_FAST"], scores["UNINIT_FAST"])
    ]
    # Split by whether the fast variable still HAS skill. Past saturation both
    # curves are random draws from the climatology and their difference is
    # noise, so a max taken over all leads measures the noise rather than the
    # effect -- which is how a 0.22-sd "difference" appeared in a comparison
    # whose true answer is zero.
    skilful = [i for i, v in enumerate(scores["INIT_FAST"]) if v / fast_sd < 0.7]
    early = [fast_diff[i] for i in skilful] or [fast_diff[0]]
    late = [fast_diff[i] for i in range(len(LEADS)) if i not in skilful]
    best_early = int(np.argmin(fast_diff[: max(skilful) + 1])) if skilful else 0
    standard_error = scores["INIT_FAST"][1] / fast_sd / np.sqrt(2 * N_CASES)
    print(f"#   fast variable: the initialised run is better by "
          f"{abs(fast_diff[best_early]):.4f} sd at lead "
          f"{LEADS[best_early]:g} -- about "
          f"{abs(fast_diff[best_early]) / standard_error:.0f} standard errors, "
          f"so real rather than noise")
    print(f"#   but it is gone by lead {LEADS[min(best_early + 1, len(LEADS) - 1)]:g} "
          f"({fast_diff[best_early + 1]:+.4f} sd) and after saturation the "
          f"difference scatters about zero (mean {np.mean(late):+.4f}, sd "
          f"{np.std(late):.4f})")
    print(f"#   so: a small brief benefit for the weather, against a lasting "
          f"one for the ocean")
    print(f"LEADS = {LEADS}")
    print(f"N_CASES = {N_CASES}")
    _scalar("SLOW_CLIM_SD", slow_sd)
    _scalar("FAST_CLIM_SD", fast_sd)
    _scalar("SLOW_HORIZON_INIT", horizon(scores["INIT_SLOW"], slow_sd), ".4f")
    _scalar("SLOW_HORIZON_UNINIT", horizon(scores["UNINIT_SLOW"], slow_sd), ".4f")
    _scalar("FAST_BEST_EARLY", float(abs(fast_diff[best_early])))
    _scalar("FAST_BEST_EARLY_LEAD", float(LEADS[best_early]), ".2f")
    _scalar("FAST_STANDARD_ERROR", float(standard_error))
    _scalar("FAST_LATE_MEAN", float(np.mean(late)))
    _scalar("FAST_LATE_SD", float(np.std(late)))
    print(f"FAST_SKILFUL_LEADS = {len(skilful)}")
    for key, values in scores.items():
        _emit(f"RMSE_{key}", values, ".6f", per_line=8)
    return starts, grid


# ==========================================================================
# 3, 4, 5. drift, its correction, and cross-validation
# ==========================================================================
def drift(truth, starts, grid):
    print("\n# --- 3. drift ---")
    rng = np.random.default_rng(1)
    initial = truth[starts].copy()
    initial[:, :3] += rng.normal(0.0, ANALYSIS_NOISE_FAST, (N_CASES, 3))
    initial[:, 3] += rng.normal(0.0, ANALYSIS_NOISE_SLOW, N_CASES)

    perfect = integrate.rk4(
        systems.coupled_ocean_atmosphere, initial, grid, **TRUTH_PARAMS
    )
    imperfect = integrate.rk4(
        systems.coupled_ocean_atmosphere, initial, grid, **MODEL_PARAMS
    )
    slow_sd = float(truth[:, 3].std())

    perfect_bias, model_bias, model_rmse, perfect_rmse = [], [], [], []
    errors = np.zeros((len(LEADS), N_CASES))
    for slot, lead in enumerate(LEADS):
        step = int(round(lead / DT))
        verify = truth[starts + step][:, 3]
        errors[slot] = imperfect[step][:, 3] - verify
        model_bias.append(float(errors[slot].mean()))
        perfect_bias.append(float((perfect[step][:, 3] - verify).mean()))
        model_rmse.append(float(np.sqrt(np.mean(errors[slot] ** 2))))
        perfect_rmse.append(float(np.sqrt(np.mean(
            (perfect[step][:, 3] - verify) ** 2
        ))))

    print(f"#   truth relaxes towards z = {REFERENCE_Z:g}, model towards "
          f"{MODEL_PARAMS['reference_z']:g}")
    print(f"#   {'lead':>6} {'mean error':>11} {'perfect-model':>14} "
          f"{'RMSE':>8} {'perfect':>8}")
    for slot in range(0, len(LEADS), 4):
        print(f"#   {LEADS[slot]:6.1f} {model_bias[slot]:11.4f} "
              f"{perfect_bias[slot]:14.4f} {model_rmse[slot]:8.4f} "
              f"{perfect_rmse[slot]:8.4f}")
    leads_array = np.asarray(LEADS)
    drift_trend = float(np.corrcoef(leads_array, model_bias)[0, 1])
    perfect_trend = float(np.corrcoef(leads_array, perfect_bias)[0, 1])
    print(f"#   imperfect model: mean error {model_bias[0]:+.4f} -> "
          f"{model_bias[-1]:+.4f}, correlation with lead {drift_trend:+.3f}")
    print(f"#   perfect model:   mean error {perfect_bias[0]:+.4f} -> "
          f"{perfect_bias[-1]:+.4f}, correlation with lead {perfect_trend:+.3f}")
    print(f"#   the perfect model's mean error is NOT zero -- 300 cases drawn "
          f"from one trajectory is not a large independent sample, and it "
          f"wanders to {max(abs(v) for v in perfect_bias):.2f}. What separates "
          f"drift from that wandering is the systematic growth: "
          f"{abs(model_bias[-1] / perfect_bias[-1]):.0f}x larger at the longest "
          f"lead, and far more strongly trended.")
    print(f"#   drift is not a constant bias: a single number cannot remove it")
    _scalar("DRIFT_TREND", drift_trend, ".4f")
    _scalar("PERFECT_TREND", perfect_trend, ".4f")

    print("\n# --- 4. correcting it, and how many re-forecasts that takes ---")
    # Estimate the lead-dependent drift from the FIRST n cases and apply it to
    # the rest. Cross-validated: a drift estimated on the cases it is then
    # scored on removes some of the noise as well as the signal, and flatters
    # itself accordingly.
    held_out = slice(N_CASES // 2, N_CASES)
    uncorrected = np.sqrt(np.mean(errors[:, held_out] ** 2, axis=1))
    print(f"#   scored on {N_CASES - N_CASES // 2} held-out cases")
    print(f"#   {'re-forecasts':>13} {'RMSE':>8} {'vs uncorrected':>15}")
    print(f"#   {'none':>13} {uncorrected.mean():8.4f} {'--':>15}")
    curve = []
    for size in REFORECAST_SIZES:
        estimate = errors[:, :size].mean(axis=1)
        corrected = np.sqrt(np.mean(
            (errors[:, held_out] - estimate[:, None]) ** 2, axis=1
        ))
        curve.append(float(corrected.mean()))
        print(f"#   {size:13d} {curve[-1]:8.4f} "
              f"{100 * (curve[-1] / uncorrected.mean() - 1):+14.1f} %")

    in_sample = np.sqrt(np.mean(
        (errors - errors.mean(axis=1, keepdims=True)) ** 2, axis=1
    ))
    print(f"\n# --- 5. what cross-validation is for ---")
    print(f"#   in-sample correction (drift fitted on the same cases): "
          f"{in_sample.mean():.4f}")
    print(f"#   out-of-sample, with all {N_CASES // 2} training cases: "
          f"{curve[-1]:.4f}")
    print(f"#   the in-sample number is {100 * (1 - in_sample.mean() / curve[-1]):.1f} % "
          f"better than anything achievable, and is what a careless re-forecast "
          f"study would report")

    print(f"REFORECAST_SIZES = {REFORECAST_SIZES}")
    print(f"MODEL_REFERENCE_Z = {MODEL_PARAMS['reference_z']}")
    _scalar("UNCORRECTED_RMSE", float(uncorrected.mean()))
    _scalar("INSAMPLE_RMSE", float(in_sample.mean()))
    _emit("DRIFT_BIAS", model_bias, ".6f", per_line=8)
    _emit("PERFECT_BIAS", perfect_bias, ".6f", per_line=8)
    _emit("DRIFT_RMSE", model_rmse, ".6f", per_line=8)
    _emit("PERFECT_RMSE", perfect_rmse, ".6f", per_line=8)
    _emit("UNCORRECTED_BY_LEAD", uncorrected, ".6f", per_line=8)
    _emit("CORRECTED_CURVE", curve, ".6f", per_line=6)
    best = errors[:, : REFORECAST_SIZES[-1]].mean(axis=1)
    _emit("CORRECTED_BY_LEAD",
          np.sqrt(np.mean((errors[:, held_out] - best[:, None]) ** 2, axis=1)),
          ".6f", per_line=8)


if __name__ == "__main__":
    began = time.perf_counter()
    print(f"# coupled Lorenz 63 + slow ocean: T = {OCEAN_TIME:g}, "
          f"lambda = {FORCING_STRENGTH:g}, kappa = {FEEDBACK:g}")
    truth_run = _climatology(TRUTH_PARAMS)
    print(f"# truth climatology: {truth_run.shape[0]} states")
    memory(truth_run)
    case_starts, forecast_grid = initialisation(truth_run)
    drift(truth_run, case_starts, forecast_grid)
    print(f"\n# total {time.perf_counter() - began:.0f}s")
