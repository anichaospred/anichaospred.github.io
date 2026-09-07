#!/usr/bin/env python3
r"""Precompute chapter 26's Earth-system-prediction experiments.

Chapter 24 coupled the fast system to *one* slow variable. This chapter has a
hierarchy of them, :math:`T_k = 2, 8, 32, 128`, and -- the structural
difference from every earlier chapter -- the forcing is **not prescribed**. It
is a carbon reservoir that emissions fill and a sink drains, with the sink's
efficiency depending on the climate. So predicting the forcing is part of the
forecast.

Five blocks.

1. **A hierarchy of memories.** The memory of each reservoir is a *parameter*;
   what is measured is the amplitude, Hasselmann's
   :math:`\sigma_S = \sigma_z\sqrt{\tau_{\rm int}/T}`, and the departure from
   it where the weather is not white on the reservoir's own timescale.
2. **What initialising buys, and for how long.** Initialised and uninitialised
   forecasts of every reservoir. The advantage decays as
   :math:`e^{-2\ell/T}` exactly, so the useful lead is :math:`1.141\,T` and
   *nothing but the memory* sets the scale.
3. **The forcing as a state variable.** The carbon-climate feedback amplifies
   the equilibrium by :math:`1/(1-g)`; past :math:`g = 1` the sink saturates
   and carbon accumulates at a fixed rate for ever, which is a different thing
   from an explosion.
4. **Three sources of spread, not two.** Internal variability, carbon-feedback
   strength, and emissions scenario, decomposed against lead -- and which of
   them dominates depends on *which reservoir you ask about*.
5. **An emergent constraint that fails.** The observable matches its exact
   theoretical expectation to a few per cent with a within-record correlation
   of 0.99, and still carries essentially no information about the response,
   because it is not monotone in the parameter it is supposed to measure.

Run from chaos-book/:
    python3 scripts/generate_ch26_data.py        # ~5 minutes
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import earlywarning as ew, ensemble, errorgrowth, integrate, systems  # noqa: E402

DT, KEEP = 0.01, 10                       # integrate at 0.01, store at 0.1 TU
STORE_DT = DT * KEEP
TIMESCALES = (2.0, 8.0, 32.0, 128.0)
N_SLOW = len(TIMESCALES)

# --- section 2: initialised versus uninitialised ---------------------------
CONTROL_TU, CASES, LEAD_TU = 20000.0, 200, 400.0
FAST_ERROR, ANALYSIS_FRACTION, SKILL_THRESHOLD = 0.1, 0.30, 0.95

# --- section 3: the carbon-climate feedback --------------------------------
SINK_TIME, CARBON_FORCING, EMISSIONS = 50.0, 1.0, 0.16
FEEDBACK_ALPHAS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
SATURATION_ALPHAS = (0.16, 0.20)
SATURATION_TU = 12000.0

# --- section 4: three sources of spread ------------------------------------
SPREAD_ALPHAS, SPREAD_EMISSIONS, SPREAD_STARTS = (0.02, 0.04, 0.06), (0.14, 0.16, 0.18), 24
SPREAD_TU = 600.0

# --- section 5: the emergent constraint ------------------------------------
WINDOW_LO, WINDOW_HI, CONSTRAINT_TU = 1000.0, 2000.0, 2600.0
TRANSIENT_WINDOW = (50.0, 250.0)
ONE_PARAMETER_ALPHAS = tuple(np.round(np.linspace(0.015, 0.095, 17), 4))
TWO_PARAMETER_ALPHAS = (0.02, 0.035, 0.05, 0.065, 0.08)
TWO_PARAMETER_SINKS = (30.0, 50.0, 80.0)
MAX_GAIN = 0.8


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


def _control_run():
    run = integrate.rk4(
        systems.earth_system, systems.earth_system_state(n_slow=N_SLOW),
        integrate.trajectory_grid(CONTROL_TU, DT), slow_timescales=TIMESCALES,
    )[::KEEP]
    return run[int(500.0 / STORE_DT):]           # discard spin-up


# ==========================================================================
# 1. a hierarchy of memories
# ==========================================================================
def memory_hierarchy(control):
    print("# --- 1. a hierarchy of memories ---")
    _, reservoirs, _ = systems.earth_system_split(control, n_slow=N_SLOW)
    z = control[:, 2]
    anomaly = z - z.mean()
    # integral timescale of the driving, from its own autocorrelation
    lags = np.arange(1, 400)
    acf = np.array(
        [np.corrcoef(anomaly[:-l], anomaly[l:])[0, 1] for l in lags]
    )
    tau_int = STORE_DT * (0.5 + acf.sum())
    sd = reservoirs.std(axis=0)
    predicted = z.std() * np.sqrt(tau_int / np.asarray(TIMESCALES))
    scaled = sd * np.sqrt(np.asarray(TIMESCALES))
    for T, s, p, sc in zip(TIMESCALES, sd, predicted, scaled):
        print(f"#   T {T:6.1f}: sd {s:.4f}  Hasselmann {p:.4f}  "
              f"ratio {s/p:.3f}  sd*sqrt(T) {sc:.4f}")
    print(f"#   sigma_z {z.std():.4f}, integral timescale of z {tau_int:.5f} TU")

    print(f"TIMESCALES = {TIMESCALES}")
    _scalar("SIGMA_Z", z.std(), ".4f")
    _scalar("TAU_INT", tau_int, ".6f")
    _emit("RESERVOIR_SD", sd, ".6f")
    _emit("RESERVOIR_SD_PREDICTED", predicted, ".6f")
    _emit("RESERVOIR_SD_RATIO", sd / predicted, ".4f")
    # a short illustrative window, subsampled for the notebook
    lo = int(2000.0 / STORE_DT)
    span = int(1200.0 / STORE_DT)
    stride = max(1, span // 500)
    _emit("SAMPLE_TIME",
          np.arange(0, span, stride) * STORE_DT, ".2f")
    _emit("SAMPLE_Z", control[lo: lo + span: stride, 2], ".3f")
    _emit("SAMPLE_RESERVOIRS",
          reservoirs[lo: lo + span: stride].T, ".4f")


# ==========================================================================
# 2. what initialising buys
# ==========================================================================
def initialised_versus_forced(control):
    print("# --- 2. initialised versus uninitialised ---")
    _, reservoirs, _ = systems.earth_system_split(control, n_slow=N_SLOW)
    clim_sd = reservoirs.std(axis=0)
    lead_steps = int(round(LEAD_TU / STORE_DT))
    cases = np.linspace(0, control.shape[0] - lead_steps - 1, CASES).astype(int)
    rng = np.random.default_rng(0)

    initialised = control[cases].copy()
    uninitialised = control[cases].copy()
    fast_error = rng.normal(0.0, FAST_ERROR, (CASES, 3))
    initialised[:, :3] += fast_error
    uninitialised[:, :3] += fast_error          # same weather analysis in both
    initialised[:, 3:3 + N_SLOW] += rng.normal(
        0.0, ANALYSIS_FRACTION * clim_sd, (CASES, N_SLOW)
    )
    uninitialised[:, 3:3 + N_SLOW] = control[
        rng.integers(0, control.shape[0], CASES)
    ][:, 3:3 + N_SLOW]

    grid = integrate.trajectory_grid(LEAD_TU, DT)
    run_i = integrate.rk4(systems.earth_system, initialised, grid,
                          slow_timescales=TIMESCALES)[::KEEP]
    run_u = integrate.rk4(systems.earth_system, uninitialised, grid,
                          slow_timescales=TIMESCALES)[::KEEP]
    leads = np.arange(run_i.shape[0]) * STORE_DT
    truth = np.stack(
        [control[c: c + run_i.shape[0]] for c in cases], axis=1
    )

    slopes, measured_leads, predicted_leads = [], [], []
    err_i_all, err_u_all, advantage_all = [], [], []
    for k, T in enumerate(TIMESCALES):
        err_i = ((run_i[:, :, 3 + k] - truth[:, :, 3 + k]) ** 2).mean(axis=1)
        err_u = ((run_u[:, :, 3 + k] - truth[:, :, 3 + k]) ** 2).mean(axis=1)
        advantage = err_u - err_i
        window = (leads > 0.15 * T) & (leads < 1.6 * T) & (advantage > 0)
        slope = float(np.polyfit(leads[window], np.log(advantage[window]), 1)[0])
        ratio = np.sqrt(err_i / err_u)
        hit = int(np.argmax(ratio > SKILL_THRESHOLD))
        measured = leads[hit] if ratio[hit] > SKILL_THRESHOLD else float("nan")
        predicted = errorgrowth.initialised_useful_lead(
            T, ANALYSIS_FRACTION * clim_sd[k], clim_sd[k], SKILL_THRESHOLD
        )
        slopes.append(slope)
        measured_leads.append(measured)
        predicted_leads.append(predicted)
        # store on a lead grid the notebook can plot
        keep = slice(0, None, max(1, leads.size // 400))
        err_i_all.append(err_i[keep])
        err_u_all.append(err_u[keep])
        advantage_all.append(advantage[keep])
        print(f"#   T {T:6.1f}: slope {slope:9.5f} vs -2/T {-2/T:9.5f} "
              f"(ratio {slope/(-2/T):.3f})   useful lead {measured:7.1f} vs "
              f"{predicted:7.1f} ({measured/predicted:.3f})")

    print(f"CASES = {CASES}")
    print(f"ANALYSIS_FRACTION = {ANALYSIS_FRACTION}")
    print(f"SKILL_THRESHOLD = {SKILL_THRESHOLD}")
    _emit("CLIM_SD", clim_sd, ".6f")
    _emit("LEADS", leads[slice(0, None, max(1, leads.size // 400))], ".2f")
    _emit("ERROR_INITIALISED", np.stack(err_i_all), ".6e")
    _emit("ERROR_UNINITIALISED", np.stack(err_u_all), ".6e")
    _emit("ADVANTAGE", np.stack(advantage_all), ".6e")
    _emit("ADVANTAGE_SLOPE", slopes, ".6f")
    _emit("ADVANTAGE_SLOPE_LAW", [-2.0 / T for T in TIMESCALES], ".6f")
    _emit("USEFUL_LEAD", measured_leads, ".4f")
    _emit("USEFUL_LEAD_PREDICTED", predicted_leads, ".4f")
    _scalar("USEFUL_LEAD_COEFFICIENT",
            errorgrowth.initialised_useful_lead(
                1.0, ANALYSIS_FRACTION, 1.0, SKILL_THRESHOLD), ".4f")


# ==========================================================================
# 3. the forcing as a state variable
# ==========================================================================
def carbon_feedback():
    print("# --- 3. the forcing as a state variable ---")
    alphas = np.asarray(FEEDBACK_ALPHAS)
    members = np.stack([
        systems.earth_system_state(n_slow=N_SLOW) for _ in alphas
    ])
    t0 = time.time()
    run = integrate.rk4(
        systems.earth_system, members,
        integrate.trajectory_grid(3000.0, DT), slow_timescales=TIMESCALES,
        emissions=EMISSIONS, sink_time=SINK_TIME,
        sink_sensitivity=alphas, carbon_forcing=CARBON_FORCING,
    )
    tail = run[int(0.6 * run.shape[0]):]
    carbon = tail[:, :, -1].mean(axis=0)
    z_mean = tail[:, :, 2].mean(axis=0)
    baseline = EMISSIONS * SINK_TIME
    gains, predicted = [], []
    for a, c, zm in zip(alphas, carbon, z_mean):
        g = systems.earth_system_loop_gain(
            EMISSIONS, SINK_TIME, a, CARBON_FORCING)
        p = systems.earth_system_amplification(
            EMISSIONS, SINK_TIME, a, CARBON_FORCING)
        gains.append(g)
        predicted.append(p)
        print(f"#   alpha {a:.2f} (g {g:.2f}): C {c:8.4f}  amplification "
              f"{c/baseline:6.3f} vs {p:6.3f} (ratio {c/baseline/p:.4f})  "
              f"<z> {zm:7.3f}")
    print(f"#   ({time.time()-t0:.0f}s)")

    print("# --- 3b. past g = 1 the sink saturates ---")
    sat = np.asarray(SATURATION_ALPHAS)
    grid = integrate.trajectory_grid(SATURATION_TU, 0.02)
    t0 = time.time()
    run_s = integrate.rk4(
        systems.earth_system,
        np.stack([systems.earth_system_state(n_slow=N_SLOW) for _ in sat]),
        grid, slow_timescales=TIMESCALES, emissions=EMISSIONS,
        sink_time=SINK_TIME, sink_sensitivity=sat,
        carbon_forcing=CARBON_FORCING,
    )
    late = grid > 0.75 * SATURATION_TU
    rates, asymptotes = [], []
    for i, a in enumerate(sat):
        rate = float(np.polyfit(grid[late], run_s[late, i, -1], 1)[0])
        asymptote = EMISSIONS - 1.0 / (SINK_TIME * a * CARBON_FORCING)
        rates.append(rate)
        asymptotes.append(asymptote)
        print(f"#   alpha {a:.2f} (g {systems.earth_system_loop_gain(EMISSIONS, SINK_TIME, a, CARBON_FORCING):.2f}): "
              f"C({SATURATION_TU:.0f}) {run_s[-1, i, -1]:8.1f}  late dC/dt "
              f"{rate:.6f}  asymptote {asymptote:.6f}  ratio {rate/asymptote:.3f}")
    print(f"#   ({time.time()-t0:.0f}s)")

    print(f"FEEDBACK_ALPHAS = {FEEDBACK_ALPHAS}")
    print(f"EMISSIONS = {EMISSIONS}")
    print(f"SINK_TIME = {SINK_TIME}")
    print(f"CARBON_FORCING = {CARBON_FORCING}")
    print(f"SATURATION_ALPHAS = {SATURATION_ALPHAS}")
    print(f"SATURATION_TU = {SATURATION_TU}")
    _scalar("CARBON_BASELINE", baseline, ".4f")
    _emit("FEEDBACK_GAIN", gains, ".6f")
    _emit("FEEDBACK_CARBON", carbon, ".6f")
    _emit("FEEDBACK_AMPLIFICATION", carbon / baseline, ".6f")
    _emit("FEEDBACK_AMPLIFICATION_PREDICTED", predicted, ".6f")
    _emit("FEEDBACK_Z", z_mean, ".4f")
    _scalar(
        "FEEDBACK_WORST_BELOW_HALF",
        100.0 * max(
            abs(c / baseline / p - 1.0)
            for g, c, p in zip(gains, carbon, predicted) if g <= 0.5
        ), ".2f",
    )
    _scalar(
        "FEEDBACK_WORST_ABOVE_HALF",
        100.0 * max(
            abs(c / baseline / p - 1.0)
            for g, c, p in zip(gains, carbon, predicted) if g > 0.5
        ), ".1f",
    )
    _emit("SATURATION_RATE", rates, ".6f")
    _emit("SATURATION_ASYMPTOTE", asymptotes, ".6f")
    _emit("SATURATION_CARBON", run_s[-1, :, -1], ".2f")
    stride = max(1, grid.size // 400)
    _emit("SATURATION_TIME", grid[::stride], ".1f")
    _emit("SATURATION_TRACK", run_s[::stride, :, -1].T, ".3f")


# ==========================================================================
# 4. three sources of spread
# ==========================================================================
def spread_decomposition():
    print("# --- 4. three sources of spread ---")
    spin = integrate.rk4(
        systems.earth_system, systems.earth_system_state(n_slow=N_SLOW),
        integrate.trajectory_grid(8000.0, DT), slow_timescales=TIMESCALES,
    )[100000:]
    base = spin[::spin.shape[0] // SPREAD_STARTS][:SPREAD_STARTS].copy()
    base[:, -1] = 0.0

    combos = list(itertools.product(
        range(len(SPREAD_ALPHAS)), range(len(SPREAD_EMISSIONS)),
        range(SPREAD_STARTS),
    ))
    t0 = time.time()
    run = integrate.rk4(
        systems.earth_system, np.stack([base[s] for _, _, s in combos]),
        integrate.trajectory_grid(SPREAD_TU, DT), slow_timescales=TIMESCALES,
        emissions=np.array([SPREAD_EMISSIONS[e] for _, e, _ in combos]),
        sink_time=SINK_TIME, carbon_forcing=CARBON_FORCING,
        sink_sensitivity=np.array([SPREAD_ALPHAS[a] for a, _, _ in combos]),
    )[::KEEP]
    times = np.arange(run.shape[0]) * STORE_DT
    print(f"#   {len(combos)} members ({time.time()-t0:.0f}s)")

    stride = max(1, times.size // 300)
    internal_all, feedback_all, scenario_all, crossovers = [], [], [], []
    for k, T in enumerate(TIMESCALES):
        cube = run[:, :, 3 + k].reshape(
            times.size, len(SPREAD_ALPHAS), len(SPREAD_EMISSIONS), SPREAD_STARTS
        )
        internal = cube.var(axis=3).mean(axis=(1, 2))
        group = cube.mean(axis=3)
        feedback = group.mean(axis=2).var(axis=1)
        scenario = group.mean(axis=1).var(axis=1)
        total = internal + feedback + scenario
        fraction = internal / total
        # Smooth before locating the crossing. The share is noisy, and the
        # *first* dip below one half is biased early -- it reported 78 TU for
        # a curve a reader would place near 130.
        width = int(round(20.0 / STORE_DT))
        smooth = np.convolve(
            fraction, np.ones(width) / width, mode="same"
        )
        valid = slice(width, fraction.size - width)
        below = int(np.argmax(smooth[valid] < 0.5))
        crossovers.append(
            times[valid][below] if smooth[valid][below] < 0.5 else float("nan")
        )
        internal_all.append(internal[::stride])
        feedback_all.append(feedback[::stride])
        scenario_all.append(scenario[::stride])
        print(f"#   T {T:6.1f}: internal share {100*fraction[1]:5.1f}% at lead "
              f"{times[1]:.1f}, {100*fraction[-1]:5.2f}% at {times[-1]:.0f}; "
              f"crosses 50% at {crossovers[-1]:6.1f}")

    print(f"SPREAD_ALPHAS = {SPREAD_ALPHAS}")
    print(f"SPREAD_EMISSIONS = {SPREAD_EMISSIONS}")
    print(f"SPREAD_STARTS = {SPREAD_STARTS}")
    _emit("SPREAD_TIME", times[::stride], ".2f")
    _emit("SPREAD_INTERNAL", np.stack(internal_all), ".6e")
    _emit("SPREAD_FEEDBACK", np.stack(feedback_all), ".6e")
    _emit("SPREAD_SCENARIO", np.stack(scenario_all), ".6e")
    _emit("SPREAD_CROSSOVER", crossovers, ".4f")
    _emit(
        "SPREAD_INTERNAL_SHARE_END",
        [
            i[-1] / (i[-1] + f[-1] + s[-1])
            for i, f, s in zip(internal_all, feedback_all, scenario_all)
        ],
        ".6f",
    )


# ==========================================================================
# 5. an emergent constraint that fails
# ==========================================================================
def _constraint_experiment(alphas, sinks, window=(WINDOW_LO, WINDOW_HI)):
    run = integrate.rk4(
        systems.earth_system,
        np.stack([systems.earth_system_state(n_slow=N_SLOW) for _ in alphas]),
        integrate.trajectory_grid(CONSTRAINT_TU, DT),
        slow_timescales=TIMESCALES, emissions=EMISSIONS, sink_time=sinks,
        sink_sensitivity=alphas, carbon_forcing=CARBON_FORCING,
    )
    lo, hi = int(window[0] / DT), int(window[1] / DT)
    rows = []
    for i in range(len(alphas)):
        carbon = run[lo:hi:KEEP, i, -1]
        reservoir = run[lo:hi:KEEP, i, 3]
        growth = np.gradient(carbon, STORE_DT)
        detrended_r = ew.detrend(reservoir)
        detrended_g = ew.detrend(growth)
        observable = float(np.polyfit(detrended_r, detrended_g, 1)[0])
        raw = float(np.polyfit(reservoir - reservoir.mean(),
                               growth - growth.mean(), 1)[0])
        expected = float(
            carbon.mean() * alphas[i]
            / (sinks[i] * (1.0 + alphas[i] * reservoir.mean()) ** 2)
        )
        response = float(run[int(0.9 * run.shape[0]):, i, -1].mean())
        rows.append((
            observable, expected, response, response / (EMISSIONS * sinks[i]),
            float(np.corrcoef(detrended_r, detrended_g)[0, 1]), raw,
        ))
    return np.array(rows)


def emergent_constraint():
    print("# --- 5. an emergent constraint that fails ---")
    a1 = np.asarray(ONE_PARAMETER_ALPHAS)
    s1 = np.full_like(a1, SINK_TIME)
    t0 = time.time()
    one = _constraint_experiment(a1, s1)
    print(f"#   one-parameter ensemble, {a1.size} members ({time.time()-t0:.0f}s)")
    worst = 100.0 * np.abs(one[:, 0] / one[:, 1] - 1.0).max()
    print(f"#   observable vs its exact expectation: worst {worst:.1f} %, "
          f"within-record correlation {one[:, 4].min():.3f}..{one[:, 4].max():.3f}")
    print(f"#   observable monotone in alpha: "
          f"{bool(np.all(np.diff(one[:, 0]) > 0))}; peaks at alpha "
          f"{a1[int(np.argmax(one[:, 0]))]:.3f}")
    print(f"#   raw (undetrended) observable has the wrong sign throughout: "
          f"{bool(np.all(one[:, 5] < 0))}")
    r_one = float(np.corrcoef(one[:, 0], one[:, 2])[0, 1])
    print(f"#   r(observable, response) = {r_one:+.4f}")

    pairs = [
        (a, s) for a, s in itertools.product(
            TWO_PARAMETER_ALPHAS, TWO_PARAMETER_SINKS)
        if systems.earth_system_loop_gain(
            EMISSIONS, s, a, CARBON_FORCING) < MAX_GAIN
    ]
    a2 = np.array([p[0] for p in pairs])
    s2 = np.array([p[1] for p in pairs])
    t0 = time.time()
    two = _constraint_experiment(a2, s2)
    print(f"#   two-parameter ensemble, {len(pairs)} members with g < "
          f"{MAX_GAIN} ({time.time()-t0:.0f}s)")
    r_two = float(np.corrcoef(two[:, 0], two[:, 2])[0, 1])
    r_amp = float(np.corrcoef(two[:, 0], two[:, 3])[0, 1])
    print(f"#   r(observable, response) = {r_two:+.4f}   "
          f"r(observable, amplification) = {r_amp:+.4f}")

    best = None
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if abs(two[i, 0] - two[j, 0]) < 0.03 * two[:, 0].max():
                gap = abs(two[i, 2] - two[j, 2])
                if best is None or gap > best[0]:
                    best = (gap, i, j)
    _, i, j = best
    print(f"#   degenerate pair: (alpha {a2[i]:.3f}, tau {s2[i]:.0f}) -> "
          f"{two[i, 2]:.2f} and (alpha {a2[j]:.3f}, tau {s2[j]:.0f}) -> "
          f"{two[j, 2]:.2f}; observables {100*abs(two[i,0]/two[j,0]-1):.1f} % "
          f"apart, responses {100*abs(two[i,2]/two[j,2]-1):.0f} % apart")

    t0 = time.time()
    early = _constraint_experiment(a1, s1, window=TRANSIENT_WINDOW)
    print(f"#   the same observable on a TRANSIENT window "
          f"{TRANSIENT_WINDOW} ({time.time()-t0:.0f}s)")
    print(f"#     undetrended: wrong sign throughout: "
          f"{bool(np.all(early[:, 5] < 0))}")
    detrended_ratio = early[:, 0] / early[:, 1]
    print(f"#     detrended:   ratio to the exact value ranges "
          f"{detrended_ratio.min():+.2f} to {detrended_ratio.max():+.2f} "
          f"(sign wrong for {int((detrended_ratio < 0).sum())} of "
          f"{detrended_ratio.size} members)")

    naive = ensemble.emergent_constraint(
        one[:, 0], one[:, 2], observed=float(np.median(one[:, 0])),
        observed_error=0.1 * float(np.median(one[:, 0])),
    )
    print(f"#   what the regression *claims* for the one-parameter ensemble: "
          f"reduction {100*naive['reduction']:.0f} %")

    print(f"ONE_PARAMETER_ALPHAS = {ONE_PARAMETER_ALPHAS}")
    print(f"TWO_PARAMETER_ALPHAS = {TWO_PARAMETER_ALPHAS}")
    print(f"TWO_PARAMETER_SINKS = {TWO_PARAMETER_SINKS}")
    print(f"MAX_GAIN = {MAX_GAIN}")
    print(f"WINDOW_LO = {WINDOW_LO}")
    print(f"WINDOW_HI = {WINDOW_HI}")
    _emit("ONE_OBSERVABLE", one[:, 0], ".6e")
    _emit("ONE_EXPECTED", one[:, 1], ".6e")
    _emit("ONE_RESPONSE", one[:, 2], ".4f")
    _emit("ONE_CORRELATION", one[:, 4], ".4f")
    _emit("ONE_RAW_OBSERVABLE", one[:, 5], ".6e")
    _scalar("ONE_WORST_DEPARTURE", worst, ".1f")
    _scalar("ONE_PEAK_ALPHA", a1[int(np.argmax(one[:, 0]))], ".3f")
    _scalar("R_ONE", r_one, ".4f")
    _emit("TWO_ALPHA", a2, ".4f")
    _emit("TWO_SINK", s2, ".1f")
    _emit("TWO_GAIN", [systems.earth_system_loop_gain(
        EMISSIONS, s, a, CARBON_FORCING) for a, s in zip(a2, s2)], ".4f")
    _emit("TWO_OBSERVABLE", two[:, 0], ".6e")
    _emit("TWO_RESPONSE", two[:, 2], ".4f")
    _emit("TWO_AMPLIFICATION", two[:, 3], ".4f")
    _scalar("R_TWO", r_two, ".4f")
    _scalar("R_AMPLIFICATION", r_amp, ".4f")
    _emit("DEGENERATE_ALPHA", [a2[i], a2[j]], ".4f")
    _emit("DEGENERATE_SINK", [s2[i], s2[j]], ".1f")
    _emit("DEGENERATE_OBSERVABLE", [two[i, 0], two[j, 0]], ".6e")
    _emit("DEGENERATE_RESPONSE", [two[i, 2], two[j, 2]], ".4f")
    _scalar("DEGENERATE_OBSERVABLE_GAP",
            100.0 * abs(two[i, 0] / two[j, 0] - 1.0), ".1f")
    _scalar("DEGENERATE_RESPONSE_GAP",
            100.0 * abs(two[i, 2] / two[j, 2] - 1.0), ".0f")
    _scalar("CLAIMED_REDUCTION", 100.0 * naive["reduction"], ".0f")
    print(f"TRANSIENT_WINDOW = {TRANSIENT_WINDOW}")
    _emit("TRANSIENT_OBSERVABLE", early[:, 0], ".6e")
    _emit("TRANSIENT_RAW", early[:, 5], ".6e")
    _emit("TRANSIENT_EXPECTED", early[:, 1], ".6e")
    _scalar("TRANSIENT_RAW_ALL_NEGATIVE",
            1.0 if bool(np.all(early[:, 5] < 0)) else 0.0, ".0f")
    _scalar("TRANSIENT_WRONG_SIGN_COUNT",
            float((early[:, 0] / early[:, 1] < 0).sum()), ".0f")
    _scalar("TRANSIENT_BEST_RATIO",
            float((early[:, 0] / early[:, 1]).max()), ".2f")


def main() -> None:
    started = time.time()
    print("# Generated by scripts/generate_ch26_data.py -- do not edit by hand.")
    print("# Chapter 26: Earth system prediction.")
    t0 = time.time()
    control = _control_run()
    print(f"# control run {control.shape[0]*STORE_DT:.0f} TU "
          f"({time.time()-t0:.0f}s)")
    memory_hierarchy(control)
    initialised_versus_forced(control)
    carbon_feedback()
    spread_decomposition()
    emergent_constraint()
    print(f"# total {time.time() - started:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
