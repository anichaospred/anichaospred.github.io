#!/usr/bin/env python3
r"""Precompute chapter 22's verification experiments.

Forecasts from a real analysis: a cycling LETKF on Lorenz 96 (:math:`N=40`,
:math:`F=8`, all sites observed every 0.05 TU with :math:`\sigma_o=1`, chapter
19's configuration), so that "the truth", "the observations" and "the analysis"
are three different objects and the chapter can score against each in turn.

Five blocks.

1. **Skill curves.** Anomaly correlation, RMSE, CRPS and a threshold Brier
   score against lead, from one set of forecasts.
2. **The horizon depends on the score, and on the threshold.** The same
   forecasts give materially different answers to "how far ahead is this
   useful", and the conventional 0.6 is a choice rather than a measurement.
3. **Verifying against something other than the truth.** Independent
   observations inflate the mean-square error by exactly :math:`\sigma_o^2` and
   attenuate the anomaly correlation by exactly
   :math:`(1+\sigma_o^2/\sigma_t^2)^{-1/2}`, both correctable. The **verifying
   analysis** is a different matter: it shares a model and a background with the
   forecast, and flatters it in a way no correction removes.
4. **Where the error goes.** Murphy's bias/amplitude/phase split against lead,
   and what optimal damping would recover.
5. **Practical against intrinsic**, which is chapter 12's promise. The horizon
   as a function of initial-error amplitude across seven decades, for a
   single-scale system and a two-scale one. One of them keeps paying and the
   other stops.

Run from chaos-book/:
    python3 scripts/generate_ch22_data.py        # ~6 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import (  # noqa: E402
    assimilate, ensemble, integrate, systems, verification,
)

N_SITES, FORCING, DT = 40, 8.0, 0.01
OBS_SIGMA = 1.0
CYCLE_STEPS = 5
SPINUP_CYCLES, N_CASES, CASE_STRIDE = 200, 400, 3
MEMBERS = 20
LOCALISATION, INFLATION = 8.0, 1.02
BIASED_FORCING = 8.4
MEAN_OFFSET = 0.5
LEADS = tuple(np.round(np.arange(0.0, 5.0 + 1e-9, 0.1), 3))
ACC_THRESHOLDS = tuple(np.round(np.arange(0.3, 0.95 + 1e-9, 0.05), 3))
EVENT_QUANTILE = 0.7

# Block 5
AMPLITUDES = tuple(np.logspace(-8, -1, 8))
ONE_SCALE_DT, ONE_SCALE_T, ONE_SCALE_CASES = 0.005, 14.0, 24
TWO_SCALE_DT, TWO_SCALE_T, TWO_SCALE_CASES = 0.001, 8.0, 24
N_SLOW, N_FAST, TWO_SCALE_FORCING = 8, 32, 20.0
LAMBDA1_L96 = 1.67

_H_OP = np.eye(N_SITES)
_CYCLE_GRID = np.linspace(0.0, CYCLE_STEPS * DT, CYCLE_STEPS + 1)


def _scalar(name: str, value: float, fmt: str = ".6f") -> None:
    """Emit one float, spelling a non-finite one as `float("nan")`.

    A bare `nan` in the generated data is valid Python only where numpy has
    already been imported under that name, which in a marimo data cell it has
    not -- so it becomes a NameError in the reader's browser. It is also
    invisible to `grep marimo-error` and does not change the exporter's exit
    code; only stderr reports it. This has now caught two chapters.
    """
    if not np.isfinite(value):
        print(f'{name} = float("nan")')
    else:
        print(f"{name} = {format(float(value), fmt)}")


def _emit(name: str, values, fmt: str = ".6e", per_line: int = 8) -> None:
    items = [
        'float("nan")' if not np.isfinite(v) else format(float(v), fmt)
        for v in np.ravel(values)
    ]
    print(f"{name} = (")
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i : i + per_line]) + ",")
    print(")")


def _advance(states, steps=CYCLE_STEPS):
    grid = np.linspace(0.0, steps * DT, steps + 1)
    return integrate.rk4(systems.lorenz96, states, grid, forcing=FORCING)[-1]


def build_archive():
    """A cycling LETKF, keeping truth, analysis and observations separately.

    The three verifying fields the chapter needs. `analyses` are what a centre
    would actually have; `truth` is what nobody has; `observations` are the ones
    that were assimilated at each analysis time.
    """
    start = systems.lorenz96_uniform_state(FORCING, N_SITES) + np.r_[
        0.01, np.zeros(N_SITES - 1)
    ]
    attractor = integrate.rk4(
        systems.lorenz96, start, integrate.trajectory_grid(140.0, DT),
        forcing=FORCING,
    )[4000:]
    rng = np.random.default_rng(0)
    weights = assimilate.ring_localisation(N_SITES, LOCALISATION)
    truth = attractor[0].copy()
    members = truth + rng.normal(0.0, 1.0, (MEMBERS, N_SITES))

    truths, analyses, observations, ensembles = [], [], [], []
    for cycle in range(SPINUP_CYCLES + N_CASES * CASE_STRIDE):
        truth = _advance(truth)
        members = _advance(members)
        observation = truth + rng.normal(0.0, OBS_SIGMA, N_SITES)
        members = assimilate.letkf_update(
            members, observation, _H_OP, OBS_SIGMA**2,
            inflation=INFLATION, weights=weights,
        )
        truths.append(truth.copy())
        analyses.append(members.mean(axis=0).copy())
        observations.append(observation.copy())
        if cycle >= SPINUP_CYCLES and (cycle - SPINUP_CYCLES) % CASE_STRIDE == 0:
            ensembles.append(members.copy())
    return (np.array(truths), np.array(analyses), np.array(observations),
            ensembles, np.asarray(attractor))


# ==========================================================================
# 1 & 2. skill curves, and the horizon by score and threshold
# ==========================================================================
def skill_curves(truths, analyses, ensembles, attractor):
    print("# --- 1. skill against lead, from one set of forecasts ---")
    climatology = attractor.mean(axis=0)
    climate_variance = float(attractor.var())
    # TWO different indices, and conflating them was a real bug in this script.
    # `truths` and `analyses` hold ONE ENTRY PER ANALYSIS CYCLE, 0.05 TU apart;
    # the forecast trajectory holds one entry per TIMESTEP, 0.01 TU apart. Using
    # the timestep index to look up the verifying truth put it five times too
    # far into the future at every lead, which silently produced skill curves
    # for forecasts nobody had made.
    cycle_interval = CYCLE_STEPS * DT
    lead_step = [int(round(l / DT)) for l in LEADS]          # into the forecast
    lead_cycle = [int(round(l / cycle_interval)) for l in LEADS]  # into the archive
    assert all(
        abs(c * cycle_interval - l) < 1e-9 for c, l in zip(lead_cycle, LEADS)
    ), "every lead must land on an analysis cycle"

    starts = list(range(SPINUP_CYCLES, len(truths) - lead_cycle[-1] - 1,
                        CASE_STRIDE))[:N_CASES]
    print(f"#   {len(starts)} forecast cases, climatological variance "
          f"{climate_variance:.3f}")
    grid = np.linspace(0.0, LEADS[-1], lead_step[-1] + 1)

    acc, rmse, skill, crps_curve, brier = [], [], [], [], []
    forecast_store, truth_store, ens_store = [], [], []
    # Integrate each case ONCE to the longest lead and sample it, rather than
    # re-integrating per lead: 51 leads x 233 cases of redundant work otherwise.
    started = time.perf_counter()
    trajectories = np.array([
        integrate.rk4(systems.lorenz96, analyses[start], grid, forcing=FORCING)
        for start in starts
    ])
    for slot in range(len(LEADS)):
        forecasts = trajectories[:, lead_step[slot]]
        verifying = np.array([truths[s + lead_cycle[slot]] for s in starts])
        forecast_store.append(forecasts)
        truth_store.append(verifying)
        acc.append(verification.anomaly_correlation(
            forecasts, verifying, climatology))
        rmse.append(float(np.sqrt(np.mean((forecasts - verifying) ** 2))))
        skill.append(verification.mse_skill_score(
            forecasts, verifying,
            np.broadcast_to(climatology, verifying.shape)))
    print(f"#   deterministic scores: {time.perf_counter() - started:.0f}s")

    # Ensemble scores need the whole ensemble propagated, which is 20x the work,
    # so they are computed on a subset of the cases.
    subset = starts[: len(ensembles)][:120]
    threshold = float(np.quantile(attractor[:, 0], EVENT_QUANTILE))
    started = time.perf_counter()
    ensemble_runs = [
        integrate.rk4(systems.lorenz96, ensembles[case], grid, forcing=FORCING)
        for case in range(len(subset))
    ]
    for slot in range(len(LEADS)):
        scores, probabilities, outcomes = [], [], []
        for case, start in enumerate(subset):
            members = ensemble_runs[case][lead_step[slot]]
            verifying = truths[start + lead_cycle[slot]]
            scores.append(float(ensemble.crps(members.T, verifying).mean()))
            probabilities.append(float(np.mean(members[:, 0] > threshold)))
            outcomes.append(float(verifying[0] > threshold))
        crps_curve.append(float(np.mean(scores)))
        base = float(np.mean(outcomes))
        reference = base * (1.0 - base)
        brier.append(
            float("nan") if reference == 0.0
            else 1.0 - ensemble.brier_score(probabilities, outcomes) / reference
        )
    print(f"#   ensemble scores ({len(subset)} cases): "
          f"{time.perf_counter() - started:.0f}s")

    print(f"#   {'lead':>5} {'ACC':>7} {'RMSE':>7} {'MSE-SS':>8} {'CRPS':>7} "
          f"{'Brier SS':>9}")
    for slot in range(0, len(LEADS), 5):
        print(f"#   {LEADS[slot]:5.1f} {acc[slot]:7.4f} {rmse[slot]:7.4f} "
              f"{skill[slot]:8.4f} {crps_curve[slot]:7.4f} {brier[slot]:9.4f}")

    print(f"LEADS = {LEADS}")
    _scalar("CLIMATE_VARIANCE", climate_variance)
    _scalar("CLIMATE_SIGMA", float(np.sqrt(climate_variance)))
    print(f"N_FORECAST_CASES = {len(starts)}")
    print(f"N_ENSEMBLE_CASES = {len(subset)}")
    _scalar("EVENT_THRESHOLD", threshold)
    _emit("SCORE_ACC", acc, ".6f", per_line=10)
    _emit("SCORE_RMSE", rmse, ".6f", per_line=10)
    _emit("SCORE_MSESS", skill, ".6f", per_line=10)
    _emit("SCORE_CRPS", crps_curve, ".6f", per_line=10)
    _emit("SCORE_BRIERSS", brier, ".6f", per_line=10)

    print("\n# --- 2. the horizon depends on the score, and on the threshold ---")
    leads = np.asarray(LEADS)
    sigma = np.sqrt(climate_variance)
    definitions = (
        ("ACC < 0.6", verification.skill_horizon(leads, np.asarray(acc), 0.6)),
        ("ACC < 0.5", verification.skill_horizon(leads, np.asarray(acc), 0.5)),
        ("MSE skill < 0", verification.skill_horizon(
            leads, np.asarray(skill), 0.0)),
        ("RMSE > 0.7 sigma", verification.skill_horizon(
            leads, np.asarray(rmse), 0.7 * sigma, decreasing=False)),
        ("RMSE > 1.0 sigma", verification.skill_horizon(
            leads, np.asarray(rmse), sigma, decreasing=False)),
        ("Brier skill < 0", verification.skill_horizon(
            leads, np.asarray(brier), 0.0)),
    )
    for name, value in definitions:
        print(f"#     {name:18s} {value:6.2f} TU")
    spread = max(v for _n, v in definitions) / min(v for _n, v in definitions)
    print(f"#   widest-to-narrowest: {spread:.2f}x, from the same forecasts")
    print("HORIZON_LABELS = " + repr(tuple(n for n, _v in definitions)))
    _emit("HORIZON_VALUES", [v for _n, v in definitions], ".5f", per_line=6)
    _scalar("HORIZON_SPREAD", spread)

    threshold_horizons = [
        verification.skill_horizon(leads, np.asarray(acc), t)
        for t in ACC_THRESHOLDS
    ]
    print(f"#   ACC threshold sweep: "
          f"{ACC_THRESHOLDS[0]:.2f} -> {threshold_horizons[0]:.2f} TU, "
          f"{ACC_THRESHOLDS[-1]:.2f} -> {threshold_horizons[-1]:.2f} TU")
    print(f"ACC_THRESHOLDS = {ACC_THRESHOLDS}")
    _emit("ACC_THRESHOLD_HORIZON", threshold_horizons, ".5f", per_line=7)
    return (forecast_store, truth_store, starts, lead_cycle, climatology,
            climate_variance)


# ==========================================================================
# 3. verifying against something other than the truth
# ==========================================================================
def imperfect_truth(forecast_store, truth_store, starts, lead_cycle, analyses,
                    observations, climatology, climate_variance):
    print("\n# --- 3. verifying against observations, and against the analysis ---")
    rng = np.random.default_rng(77)

    rows = {k: [] for k in ("TRUTH", "INDEP", "ANALYSIS")}
    accs = {k: [] for k in ("TRUTH", "INDEP", "ANALYSIS")}
    for slot in range(len(LEADS)):
        forecasts = forecast_store[slot]
        truth = truth_store[slot]
        independent = truth + rng.normal(0.0, OBS_SIGMA, truth.shape)
        analysis = np.array([analyses[s + lead_cycle[slot]] for s in starts])
        for key, field in (("TRUTH", truth), ("INDEP", independent),
                           ("ANALYSIS", analysis)):
            rows[key].append(float(np.mean((forecasts - field) ** 2)))
            accs[key].append(verification.anomaly_correlation(
                forecasts, field, climatology))

    print(f"#   {'lead':>5} {'MSE truth':>10} {'MSE indep':>10} "
          f"{'inflation':>10} {'MSE analysis':>13} {'excess':>8}")
    for slot in range(0, len(LEADS), 5):
        inflation = rows["INDEP"][slot] - rows["TRUTH"][slot]
        excess = rows["ANALYSIS"][slot] - rows["TRUTH"][slot]
        print(f"#   {LEADS[slot]:5.1f} {rows['TRUTH'][slot]:10.4f} "
              f"{rows['INDEP'][slot]:10.4f} {inflation:10.4f} "
              f"{rows['ANALYSIS'][slot]:13.4f} {excess:8.4f}")
    print(f"#   sigma_o^2 = {OBS_SIGMA ** 2:.4f}; the independent-observation "
          f"inflation should equal it at every lead")
    # The analysis flattery is only interesting relative to the error being
    # measured. In absolute terms it is a few hundredths at every lead, which
    # is negligible against an MSE of 20 and decisive against one of 0.4.
    fractions = [
        (rows["ANALYSIS"][slot] - rows["TRUTH"][slot]) / rows["TRUTH"][slot]
        for slot in range(1, len(LEADS))
    ]
    print(f"#   verifying against the analysis, as a FRACTION of the true MSE:")
    for slot in (1, 3, 5, 10, 20, 40):
        print(f"#     lead {LEADS[slot]:4.1f}: "
              f"{100 * fractions[slot - 1]:+7.2f} %")
    _emit("ANALYSIS_FRACTION", fractions, ".6f", per_line=10)

    # The trap: verify the ANALYSIS against the observations that made it.
    assimilated_mse = float(np.mean([
        np.mean((analyses[s] - observations[s]) ** 2) for s in starts
    ]))
    analysis_truth_mse = rows["TRUTH"][0]
    corrected = verification.correct_mse_for_observation_error(
        assimilated_mse, OBS_SIGMA**2
    )
    print(f"#   analysis vs its OWN assimilated observations: "
          f"{assimilated_mse:.4f}")
    print(f"#     true analysis MSE {analysis_truth_mse:.4f}; "
          f"expected {analysis_truth_mse + OBS_SIGMA ** 2:.4f} if independent")
    print(f"#     correction gives {corrected:.4f} -- "
          f"{'NEGATIVE, i.e. impossible' if corrected < 0 else 'positive'}")
    _scalar("ASSIM_MSE", assimilated_mse)
    _scalar("ASSIM_CORRECTED", corrected)
    _scalar("ANALYSIS_TRUE_MSE", analysis_truth_mse)
    print(f"OBS_VARIANCE = {OBS_SIGMA ** 2:.6f}")
    for key in ("TRUTH", "INDEP", "ANALYSIS"):
        _emit(f"MSE_{key}", rows[key], ".6f", per_line=10)
        _emit(f"ACCV_{key}", accs[key], ".6f", per_line=10)
    _emit("ACC_CORRECTED",
          [verification.correct_acc_for_observation_error(
              a, OBS_SIGMA**2, climate_variance) for a in accs["INDEP"]],
          ".6f", per_line=10)


# ==========================================================================
# 4. where the error goes
# ==========================================================================
def error_budget(forecast_store, truth_store, analyses, starts, lead_cycle,
                 truths, climate_variance) -> None:
    print("\n# --- 4. bias, amplitude and phase ---")
    cycle_interval = CYCLE_STEPS * DT
    lead_step = [int(round(l / DT)) for l in LEADS]
    grid = np.linspace(0.0, LEADS[-1], lead_step[-1] + 1)

    # A perfect-model twin experiment has NO bias and NO amplitude error, by
    # construction -- the forecast is a genuine trajectory of the same system.
    # Running only that made the decomposition a column of zeros under a
    # heading about where error comes from. A wrong-forcing run gives it
    # something to find, and connects to chapter 21.
    biased_trajectories = np.array([
        integrate.rk4(systems.lorenz96, analyses[start], grid,
                      forcing=BIASED_FORCING)
        for start in starts
    ])

    # Three cases: the perfect twin (all phase, by construction), a
    # wrong-forcing model (which turns out to show up as AMPLITUDE, not bias),
    # and a forecast with a constant offset added -- the canonical model bias
    # that every operational system has, included so the split can be seen to
    # find what it is for.
    keys = ("PERFECT", "BIASED", "OFFSET")
    parts = {k: {"bias": [], "amplitude": [], "phase": []} for k in keys}
    damped, undamped = {k: [] for k in keys}, {k: [] for k in keys}
    for slot in range(len(LEADS)):
        verifying = truth_store[slot]
        sets = {
            "PERFECT": forecast_store[slot],
            "BIASED": biased_trajectories[:, lead_step[slot]],
            "OFFSET": forecast_store[slot] + MEAN_OFFSET,
        }
        for key, forecasts in sets.items():
            b, a, ph = verification.mse_decomposition(forecasts, verifying)
            parts[key]["bias"].append(b)
            parts[key]["amplitude"].append(a)
            parts[key]["phase"].append(ph)
            _multiplier, ratio = verification.optimal_damping(
                forecasts, verifying
            )
            damped[key].append(ratio)
            undamped[key].append(
                float(np.mean((forecasts - verifying) ** 2)) / climate_variance
            )

    print(f"#   perfect model against a wrong-forcing model "
          f"(F = {BIASED_FORCING:g} against {FORCING:g})")
    print(f"#   {'lead':>5} {'bias^2':>9} {'ampl':>8} {'phase':>9} "
          f"{'bias share':>11} | {'undamped':>9} {'damped':>8}")
    for key in keys:
        print(f"#   -- {key} --")
        for slot in range(0, len(LEADS), 10):
            total = (parts[key]["bias"][slot] + parts[key]["amplitude"][slot]
                     + parts[key]["phase"][slot])
            print(f"#   {LEADS[slot]:5.1f} {parts[key]['bias'][slot]:9.4f} "
                  f"{parts[key]['amplitude'][slot]:8.4f} "
                  f"{parts[key]['phase'][slot]:9.4f} "
                  f"{parts[key]['bias'][slot] / total:11.4f} | "
                  f"{undamped[key][slot]:9.4f} {damped[key][slot]:8.4f}")

    leads = np.asarray(LEADS)
    undamped_horizon = verification.skill_horizon(
        leads, np.asarray(undamped["PERFECT"]), 1.0, decreasing=False
    )
    damped_horizon = verification.skill_horizon(
        leads, np.asarray(damped["PERFECT"]), 1.0, decreasing=False
    )
    print(f"#   MSE reaches the climatological value at lead "
          f"{undamped_horizon:.2f} undamped; damped it never does within "
          f"{LEADS[-1]:g} TU (ratio {damped['PERFECT'][-1]:.4f} at the end)")
    # What the decomposition detected, relative to the perfect twin.
    last = -1
    print(f"#   at lead {LEADS[last]:g}, relative to the perfect twin:")
    for key in ("BIASED", "OFFSET"):
        print(f"#     {key:7s} bias^2 x{parts[key]['bias'][last] / max(parts['PERFECT']['bias'][last], 1e-12):8.1f}"
              f"   amplitude x{parts[key]['amplitude'][last] / max(parts['PERFECT']['amplitude'][last], 1e-12):7.1f}"
              f"   phase x{parts[key]['phase'][last] / parts['PERFECT']['phase'][last]:5.2f}")
    print(f"MEAN_OFFSET = {MEAN_OFFSET}")
    print(f"BUDGET_KEYS = {keys}")
    print(f"BIASED_FORCING = {BIASED_FORCING}")
    _scalar("UNDAMPED_HORIZON", undamped_horizon)
    _scalar("DAMPED_HORIZON", damped_horizon)
    for key in keys:
        for name in ("bias", "amplitude", "phase"):
            _emit(f"BUDGET_{key}_{name.upper()}", parts[key][name], ".6f",
                  per_line=10)
        _emit(f"RATIO_{key}_DAMPED", damped[key], ".6f", per_line=10)
        _emit(f"RATIO_{key}_UNDAMPED", undamped[key], ".6f", per_line=10)


# ==========================================================================
# 5. practical against intrinsic
# ==========================================================================
def practical_against_intrinsic() -> None:
    print("\n# --- 5. what better initial conditions buy ---")

    def sweep(rhs, bank, dt, t_final, params, n_cases, split=None, label=""):
        leads = np.arange(0.0, t_final + 1e-9, 0.05)
        grid = np.linspace(0.0, t_final, int(round(t_final / dt)) + 1)
        indices = [int(round(l / dt)) for l in leads]
        starts = bank[:n_cases]
        horizons = []
        for amplitude in AMPLITUDES:
            rng = np.random.default_rng(0)
            perturbation = rng.normal(size=starts.shape)
            perturbation *= amplitude / np.linalg.norm(
                perturbation, axis=1, keepdims=True
            )
            truth = integrate.rk4(rhs, starts, grid, **params)
            run = integrate.rk4(rhs, starts + perturbation, grid, **params)
            curve = []
            for i in indices:
                a, b = run[i], truth[i]
                if split is not None:
                    a, b = split(a)[0], split(b)[0]
                curve.append(verification.anomaly_correlation(a, b))
            horizons.append(
                verification.skill_horizon(leads, np.array(curve), 0.6)
            )
        print(f"#   {label} ({n_cases} cases)")
        print("#     delta0  " + "".join(f"{a:9.0e}" for a in AMPLITUDES))
        print("#     horizon " + "".join(f"{h:9.2f}" for h in horizons))
        gains = np.diff(horizons)
        print(f"#     mean gain per decade of delta0: {np.mean(gains):+.3f} TU "
              f"(range {np.min(gains):+.2f} to {np.max(gains):+.2f})")
        return horizons

    start = systems.lorenz96_uniform_state(FORCING, N_SITES) + np.r_[
        0.01, np.zeros(N_SITES - 1)
    ]
    bank_one = integrate.rk4(
        systems.lorenz96, start,
        integrate.trajectory_grid(200.0, ONE_SCALE_DT), forcing=FORCING,
    )[4000::1200]
    started = time.perf_counter()
    one = sweep(systems.lorenz96, bank_one, ONE_SCALE_DT, ONE_SCALE_T,
                {"forcing": FORCING}, ONE_SCALE_CASES,
                label=f"single-scale L96 (N={N_SITES}, F={FORCING:g})")
    print(f"#     predicted ln(10)/lambda1 = "
          f"{np.log(10.0) / LAMBDA1_L96:.3f} TU per decade "
          f"({time.perf_counter() - started:.0f}s)")

    two_start = systems.lorenz96_two_scale_state(
        N_SLOW, N_FAST, TWO_SCALE_FORCING, seed=0
    )
    bank_two = integrate.rk4(
        systems.lorenz96_two_scale, two_start,
        integrate.trajectory_grid(70.0, TWO_SCALE_DT),
        n_slow=N_SLOW, n_fast=N_FAST, forcing=TWO_SCALE_FORCING,
    )[20000::1800]
    started = time.perf_counter()
    two = sweep(
        systems.lorenz96_two_scale, bank_two, TWO_SCALE_DT, TWO_SCALE_T,
        {"n_slow": N_SLOW, "n_fast": N_FAST, "forcing": TWO_SCALE_FORCING},
        TWO_SCALE_CASES,
        split=lambda s: systems.lorenz96_two_scale_split(s, N_SLOW),
        label="two-scale L96, slow variables scored",
    )
    print(f"#     ({time.perf_counter() - started:.0f}s)")
    print(f"#   seven decades of better initialisation buy "
          f"{one[0] - one[-1]:.2f} TU in the single-scale system and "
          f"{two[0] - two[-1]:.2f} TU in the two-scale one")
    print(f"AMPLITUDES = {tuple(float(a) for a in AMPLITUDES)}")
    print(f"LAMBDA1_L96 = {LAMBDA1_L96}")
    print(f"ONE_SCALE_CASES = {ONE_SCALE_CASES}")
    print(f"TWO_SCALE_CASES = {TWO_SCALE_CASES}")
    _emit("HORIZON_ONE_SCALE", one, ".5f", per_line=8)
    _emit("HORIZON_TWO_SCALE", two, ".5f", per_line=8)


if __name__ == "__main__":
    began = time.perf_counter()
    print(f"# L96 N={N_SITES} F={FORCING}; LETKF radius {LOCALISATION:g}, "
          f"inflation {INFLATION}, {MEMBERS} members, sigma_o {OBS_SIGMA}")
    truths_, analyses_, observations_, ensembles_, attractor_ = build_archive()
    fstore, tstore, starts_, leadcyc_, clim_, climvar_ = skill_curves(
        truths_, analyses_, ensembles_, attractor_
    )
    imperfect_truth(fstore, tstore, starts_, leadcyc_, analyses_,
                    observations_, clim_, climvar_)
    error_budget(fstore, tstore, analyses_, starts_, leadcyc_, truths_,
                 climvar_)
    practical_against_intrinsic()
    print(f"\n# total {time.perf_counter() - began:.0f}s")
