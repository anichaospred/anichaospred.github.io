#!/usr/bin/env python3
r"""Precompute chapter 28's non-stationary predictability experiments.

Forecast skill has improved for fifty years. Two entirely different things
could have caused that, and a skill record cannot separate them. This chapter
builds the separation in a model where both factors are known exactly, and then
asks what it would take to do it from a record.

The atmosphere is Lorenz 96 at N = 40 with the forcing ramped from F = 6 to
F = 10 over 4,000 time units -- about fifty-five years at the conventional five
days per time unit. The forecasting system is perfect-model and its analysis
error is set by hand, so "the system" and "the atmosphere" are two independent
knobs rather than a confound.

Six blocks.

1. **The ramped experiment** -- a *frozen* forecasting system on a changing
   atmosphere, verified two ways, plus the quasi-static check that licenses
   reading the ramp as a sequence of climates.
2. **The stationary snapshots** -- Lyapunov spectrum, Kolmogorov-Sinai entropy,
   unstable dimension, saturation and horizon at each epoch's forcing, and the
   exact identities that check them.
3. **The horizon-law decomposition** -- instability against amplitude against
   accuracy, exactly.
4. **The 2x2 attribution** -- the frozen-system reforecast, the interaction
   term, and the break-even accuracy.
5. **Dynamics or circulation** -- the shift-share decomposition, the classifier
   correlations that decide whether it can say anything, and the trend in
   finite-time growth as a function of the lead time asked about.
6. **Detectability** -- the per-case scatter, the correlation between
   consecutive forecasts, and the record length a trend needs.

Run from chaos-book/:
    python3 scripts/generate_ch28_data.py        # ~13 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import (  # noqa: E402
    errorgrowth,
    lyapunov,
    nonstationary,
    integrate,
    spatial,
    systems,
    verification,
)

# --- the atmosphere ------------------------------------------------------
SITES, DT = 40, 0.01
F_START, F_END = 6.0, 10.0
TOTAL = 4000.0                    # time units of record; ~55 years at 5 d/TU
RATE = (F_END - F_START) / TOTAL  # 0.001 per time unit
N_EPOCH = 10
DAYS_PER_TU = 5.0

# --- the forecasting system ---------------------------------------------
DELTA0_OLD = 0.05                 # analysis error, per component
DELTA0_NEW = 0.005                # one decade better
LEAD, LAUNCH_EVERY = 15.0, 4.0
FRACTION = 0.5                    # of saturation: the useful-horizon threshold

# --- diagnostics ---------------------------------------------------------
# Enough to bracket every positive exponent AND the Kaplan-Yorke crossing at
# the strongest forcing. With 24 the crossing sat outside the range and
# kaplan_yorke_dimension returned the cap, 24.000, at nine of ten forcings --
# a number that looked like a measurement and was the truncation.
SPECTRUM_EXPONENTS = 34
SPECTRUM_TIME = 1000.0
CLIMATE_TIME = 1500.0
CLIMATE_SPINUP = 500.0
FTLE_STATES = 300
FTLE_TAUS = (0.5, 1.0, 2.5, 5.0)
FTLE_TAU_MAIN = 2.5
HORIZON_CASES = 400
SHIFT_SHARE_GROUPS = 6
DENSE_EVERY = 0.2                 # one launch per day, for the decorrelation
DENSE_SPAN = 600.0


# =========================================================================
# emit helpers
# =========================================================================
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
    v = float(value)
    text = 'float("nan")' if not np.isfinite(v) else format(v, fmt)
    print(f"{name} = {text}")


def _log(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


# =========================================================================
# building blocks
# =========================================================================
def stationary_climate(forcing: float) -> np.ndarray:
    """A long stationary trajectory at fixed forcing, transient discarded."""
    x0 = systems.lorenz96_uniform_state(forcing, SITES) + 0.01 * np.sin(
        np.arange(SITES)
    )
    traj = integrate.rk4(
        systems.lorenz96,
        x0,
        integrate.trajectory_grid(CLIMATE_TIME, DT),
        forcing=forcing,
    )
    return traj[int(round(CLIMATE_SPINUP / DT)) :]


def horizon_set(
    forcing: float,
    traj: np.ndarray,
    delta0: float,
    saturation: float,
    threshold_saturation: float | None = None,
    n_cases: int = HORIZON_CASES,
    seed: int = 11,
) -> dict[str, object]:
    """Per-case and mean-curve useful horizons for one climate and one system.

    The threshold defaults to ``FRACTION`` of *this* climate's saturation --
    a normalised score, which is what operational verification uses. Passing
    ``threshold_saturation`` fixes it to another climate's instead, which is
    the other convention and gives a different trend from the same forecasts.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(traj.shape[0], size=n_cases, replace=False)
    starts = traj[idx]
    perturbed = starts + delta0 * rng.normal(size=starts.shape)
    t = integrate.trajectory_grid(LEAD, DT)
    control = integrate.rk4(systems.lorenz96, starts, t, forcing=forcing)
    twin = integrate.rk4(systems.lorenz96, perturbed, t, forcing=forcing)
    error = np.linalg.norm(control - twin, axis=-1)

    sat = saturation if threshold_saturation is None else threshold_saturation
    threshold = FRACTION * sat
    per_case = np.array(
        [
            verification.skill_horizon(t, error[:, j], threshold, decreasing=False)
            for j in range(n_cases)
        ]
    )
    mean_curve = np.sqrt(np.mean(error**2, axis=1))
    return {
        "per_case": per_case,
        "mean_curve": mean_curve,
        "times": t,
        "horizon": verification.skill_horizon(
            t, mean_curve, threshold, decreasing=False
        ),
    }


def state_indices(states: np.ndarray, mean: float, sd: float) -> dict[str, np.ndarray]:
    """Candidate large-scale classifiers of "what kind of day is this".

    Every one is dimensionless or standardised, so that a comparison between
    two climates of different amplitude is not a comparison of units. That
    discipline is the whole content of the choice; see
    :func:`chaoslib.nonstationary.shift_share`.
    """
    anomaly = states - mean
    energy = np.sum(anomaly**2, axis=-1)
    gradient = np.sum(np.diff(states, append=states[:, :1], axis=-1) ** 2, axis=-1)
    return {
        "amplitude": np.sqrt(energy) / (sd * np.sqrt(SITES)),
        "roughness": gradient / np.sum(states**2, axis=-1),
        "centroid": spatial.spectral_centroid(states),
        "peak": np.max(np.abs(anomaly), axis=-1) / sd,
        "minimum": states.min(axis=-1) / sd,
        "skewness": (
            ((states - states.mean(axis=-1, keepdims=True)) ** 3).mean(axis=-1)
            / states.std(axis=-1) ** 3
        ),
        "advection": (
            np.sum(
                states
                * np.roll(states, 1, -1)
                * (np.roll(states, -1, -1) - np.roll(states, 2, -1)),
                axis=-1,
            )
            / np.sum(states**2, axis=-1)
        ),
    }


# =========================================================================
# 1. the ramped experiment
# =========================================================================
def ramped_experiment() -> dict[str, object]:
    """A frozen forecasting system on a ramping atmosphere.

    Two vectorisation notes, both of which cost time to find.

    The truth is stepped in ``LAUNCH_EVERY``-long segments on a **lead-time**
    grid with the forcing offset carried in ``forcing_start``, not on an
    absolute grid. ``trajectory_grid(...) + s`` does not have exactly uniform
    steps once ``s`` is large -- ``250.01 - 250.0`` is not ``0.01`` in binary
    floating point -- so integrating on a shifted grid is a *different*
    integration, and in a chaotic system the difference grows at
    :math:`\\lambda_1`.

    All launches are then integrated in **one** call, each member carrying its
    own ``forcing_start`` :math:`= F_0 + r s_i`; on a lead grid that reproduces
    the absolute ramp exactly, and a test asserts it against the scalar case.
    Without it a thousand launches is a Python loop.
    """
    x = systems.lorenz96_uniform_state(F_START, SITES) + 0.01 * np.sin(
        np.arange(SITES)
    )
    x = integrate.rk4(
        systems.lorenz96, x, integrate.trajectory_grid(200.0, DT), forcing=F_START
    )[-1]

    n_launch = int(round(TOTAL / LAUNCH_EVERY))
    launch_times = np.arange(n_launch) * LAUNCH_EVERY
    launch_states = np.empty((n_launch, SITES))
    n_clim = int(round(LAUNCH_EVERY / DENSE_EVERY))
    climate_states = np.empty((n_launch * n_clim, SITES))
    segment = integrate.trajectory_grid(LAUNCH_EVERY, DT)
    stride = int(round(DENSE_EVERY / DT))

    for i in range(n_launch):
        launch_states[i] = x
        seg = integrate.rk4(
            systems.lorenz96_ramped,
            x,
            segment,
            forcing_start=F_START + RATE * launch_times[i],
            forcing_rate=RATE,
        )
        climate_states[i * n_clim : (i + 1) * n_clim] = seg[:-1:stride]
        x = seg[-1]

    rng = np.random.default_rng(20)
    perturbed = launch_states + DELTA0_OLD * rng.normal(size=launch_states.shape)
    t_lead = integrate.trajectory_grid(LEAD, DT)
    offsets = (F_START + RATE * launch_times)[:, None]
    error = np.empty((t_lead.size, n_launch))
    chunk = 200
    for a in range(0, n_launch, chunk):
        b = min(a + chunk, n_launch)
        control = integrate.rk4(
            systems.lorenz96_ramped,
            launch_states[a:b],
            t_lead,
            forcing_start=offsets[a:b],
            forcing_rate=RATE,
        )
        twin = integrate.rk4(
            systems.lorenz96_ramped,
            perturbed[a:b],
            t_lead,
            forcing_start=offsets[a:b],
            forcing_rate=RATE,
        )
        error[:, a:b] = np.linalg.norm(control - twin, axis=-1)

    per_epoch = n_launch // N_EPOCH
    clim_per_epoch = climate_states.shape[0] // N_EPOCH
    forcing_mid = np.empty(N_EPOCH)
    saturation = np.empty(N_EPOCH)
    horizon_norm = np.empty(N_EPOCH)
    horizon_fixed = np.empty(N_EPOCH)
    curves = np.sqrt(
        np.mean(error.reshape(t_lead.size, N_EPOCH, per_epoch) ** 2, axis=2)
    )
    per_case = np.empty((N_EPOCH, per_epoch))

    for e in range(N_EPOCH):
        block = slice(e * per_epoch, (e + 1) * per_epoch)
        clim = climate_states[e * clim_per_epoch : (e + 1) * clim_per_epoch]
        forcing_mid[e] = F_START + RATE * launch_times[block].mean()
        saturation[e] = errorgrowth.saturation_level(clim, statistic="rms")
        for j, col in enumerate(range(block.start, block.stop)):
            per_case[e, j] = verification.skill_horizon(
                t_lead, error[:, col], FRACTION * saturation[e], decreasing=False
            )

    for e in range(N_EPOCH):
        horizon_norm[e] = verification.skill_horizon(
            t_lead, curves[:, e], FRACTION * saturation[e], decreasing=False
        )
        horizon_fixed[e] = verification.skill_horizon(
            t_lead, curves[:, e], FRACTION * saturation[0], decreasing=False
        )

    return {
        "forcing_mid": forcing_mid,
        "saturation": saturation,
        "horizon_norm": horizon_norm,
        "horizon_fixed": horizon_fixed,
        "curves": curves,
        "times": t_lead,
        "per_case": per_case,
        "n_launch": n_launch,
        "per_epoch": per_epoch,
    }


# =========================================================================
# main
# =========================================================================
def main() -> None:
    started = time.time()
    print("# Generated by scripts/generate_ch28_data.py -- do not edit by hand.")
    print("# Chapter 28: has predictability changed over time?")

    # -- 1. the ramped experiment ----------------------------------------
    _log("ramped experiment ...")
    ramp = ramped_experiment()
    _log(f"  {time.time() - started:.0f}s")

    print("# --- 1. the ramped experiment: a frozen system on a changing atmosphere ---")
    print(f"#   {ramp['n_launch']} launches, {ramp['per_epoch']} per epoch, "
          f"delta0 = {DELTA0_OLD}, lead {LEAD:g} TU")
    for e in range(N_EPOCH):
        print(f"#   epoch {e} F={ramp['forcing_mid'][e]:5.2f} "
              f"sat={ramp['saturation'][e]:6.2f} "
              f"H_norm={ramp['horizon_norm'][e]:5.2f} "
              f"H_fixed={ramp['horizon_fixed'][e]:5.2f}")
    _scalar("SITES", SITES, ".0f")
    _scalar("DT", DT, ".2f")
    _scalar("F_START", F_START, ".1f")
    _scalar("F_END", F_END, ".1f")
    _scalar("TOTAL", TOTAL, ".0f")
    _scalar("RATE", RATE, ".6f")
    _scalar("N_EPOCH", N_EPOCH, ".0f")
    _scalar("DAYS_PER_TU", DAYS_PER_TU, ".1f")
    _scalar("DELTA0_OLD", DELTA0_OLD, ".4f")
    _scalar("DELTA0_NEW", DELTA0_NEW, ".4f")
    _scalar("LEAD", LEAD, ".1f")
    _scalar("FRACTION", FRACTION, ".2f")
    _scalar("N_LAUNCH", ramp["n_launch"], ".0f")
    _scalar("PER_EPOCH", ramp["per_epoch"], ".0f")
    _emit("EPOCH_FORCING", ramp["forcing_mid"], ".4f")
    _emit("EPOCH_SATURATION", ramp["saturation"], ".4f")
    _emit("EPOCH_HORIZON_NORM", ramp["horizon_norm"], ".4f")
    _emit("EPOCH_HORIZON_FIXED", ramp["horizon_fixed"], ".4f")
    _emit("RAMP_TIMES", ramp["times"][::10], ".3f", per_line=10)
    _emit("RAMP_CURVES", ramp["curves"][::10].T, ".4f")

    epochs = np.arange(N_EPOCH, dtype=float)
    trend_norm = nonstationary.linear_trend(epochs, ramp["horizon_norm"])
    trend_fixed = nonstationary.linear_trend(epochs, ramp["horizon_fixed"])
    print(f"#   trend (normalised)  {trend_norm['slope']:+.4f} "
          f"+- {trend_norm['stderr']:.4f} TU/epoch, t = {trend_norm['tstat']:.1f}")
    print(f"#   trend (fixed thr.)  {trend_fixed['slope']:+.4f} "
          f"+- {trend_fixed['stderr']:.4f} TU/epoch, t = {trend_fixed['tstat']:.1f}")
    for key, tr in (("NORM", trend_norm), ("FIXED", trend_fixed)):
        _scalar(f"TREND_{key}_SLOPE", tr["slope"], ".5f")
        _scalar(f"TREND_{key}_STDERR", tr["stderr"], ".5f")
        _scalar(f"TREND_{key}_T", tr["tstat"], ".3f")
        _scalar(f"TREND_{key}_P", tr["pvalue"], ".3e")

    # -- 2. the stationary snapshots -------------------------------------
    print("# --- 2. the stationary snapshots at each epoch's forcing ---")
    forcings = ramp["forcing_mid"]
    lam1 = np.empty(N_EPOCH)
    ks_entropy = np.empty(N_EPOCH)
    unstable = np.empty(N_EPOCH)
    ky_dim = np.empty(N_EPOCH)
    sat_stat = np.empty(N_EPOCH)
    sd_stat = np.empty(N_EPOCH)
    mean_stat = np.empty(N_EPOCH)
    energy_lhs = np.empty(N_EPOCH)
    energy_rhs = np.empty(N_EPOCH)
    horizon_stat = np.empty(N_EPOCH)
    horizon_case = np.empty(N_EPOCH)
    horizon_scatter = np.empty(N_EPOCH)
    ftle_main = np.empty(N_EPOCH)
    ftle_main_sd = np.empty(N_EPOCH)
    endpoint = {}

    for e, forcing in enumerate(forcings):
        _log(f"snapshot F = {forcing:.3f} ...")
        traj = stationary_climate(forcing)
        sat_stat[e] = errorgrowth.saturation_level(traj, statistic="rms")
        sd_stat[e] = float(traj.std())
        mean_stat[e] = float(traj.mean())
        energy_lhs[e], energy_rhs[e] = systems.lorenz96_energy_balance(traj, forcing)

        spectrum = lyapunov.lyapunov_spectrum(
            systems.lorenz96,
            systems.lorenz96_jacobian,
            traj[-1],
            dt=DT,
            t_final=SPECTRUM_TIME,
            t_transient=20.0,
            n_exponents=SPECTRUM_EXPONENTS,
            forcing=forcing,
        )
        lam1[e] = spectrum[0]
        ks_entropy[e] = lyapunov.ks_entropy(spectrum)
        unstable[e] = lyapunov.unstable_dimension(spectrum)
        ky_dim[e] = lyapunov.kaplan_yorke_dimension(spectrum)

        result = horizon_set(forcing, traj, DELTA0_OLD, sat_stat[e])
        horizon_stat[e] = result["horizon"]
        horizon_case[e] = float(np.nanmean(result["per_case"]))
        horizon_scatter[e] = float(np.nanstd(result["per_case"]))

        rng = np.random.default_rng(3)
        states = traj[rng.choice(traj.shape[0], size=FTLE_STATES, replace=False)]
        main = lyapunov.finite_time_exponents(
            systems.lorenz96,
            systems.lorenz96_jacobian,
            states,
            tau=FTLE_TAU_MAIN,
            dt=DT,
            forcing=forcing,
        )
        ftle_main[e] = float(main.mean())
        ftle_main_sd[e] = float(main.std())

        if e in (0, N_EPOCH - 1):
            by_tau = {}
            for tau in FTLE_TAUS:
                by_tau[tau] = (
                    main
                    if tau == FTLE_TAU_MAIN
                    else lyapunov.finite_time_exponents(
                        systems.lorenz96,
                        systems.lorenz96_jacobian,
                        states,
                        tau=tau,
                        dt=DT,
                        forcing=forcing,
                    )
                )
            endpoint[e] = {
                "forcing": forcing,
                "states": states,
                "ftle": by_tau,
                "indices": state_indices(states, mean_stat[e], sd_stat[e]),
                "saturation": sat_stat[e],
                "lam1": lam1[e],
                "traj_tail": traj[-1],
                "per_case": result["per_case"],
                "mean_curve": result["mean_curve"],
                "times": result["times"],
            }
        _log(f"  lam1 {lam1[e]:.4f}  KS {ks_entropy[e]:.3f}  "
             f"unstable {unstable[e]:.0f}  H {horizon_stat[e]:.3f}  "
             f"({time.time() - started:.0f}s)")

    for e in range(N_EPOCH):
        print(f"#   F={forcings[e]:5.2f} lam1={lam1[e]:6.4f} KS={ks_entropy[e]:7.3f} "
              f"n_unst={unstable[e]:3.0f} D_KY={ky_dim[e]:6.3f} sat={sat_stat[e]:6.3f} "
              f"H={horizon_stat[e]:5.3f} FTLE={ftle_main[e]:6.4f}")
    _emit("STAT_LAMBDA1", lam1, ".5f")
    _emit("STAT_KS", ks_entropy, ".4f")
    _emit("STAT_UNSTABLE", unstable, ".0f")
    _emit("STAT_KY", ky_dim, ".4f")
    _emit("STAT_SATURATION", sat_stat, ".4f")
    _emit("STAT_SD", sd_stat, ".4f")
    _emit("STAT_MEAN", mean_stat, ".4f")
    _emit("STAT_HORIZON", horizon_stat, ".4f")
    _emit("STAT_HORIZON_CASE", horizon_case, ".4f")
    _emit("STAT_HORIZON_SCATTER", horizon_scatter, ".4f")
    _emit("STAT_FTLE", ftle_main, ".5f")
    _emit("STAT_FTLE_SD", ftle_main_sd, ".5f")

    # The quasi-static check. The ramp changes F by 0.015 over a 15 TU
    # forecast, so reading it as a sequence of stationary climates should be
    # legitimate -- but that is a claim about the model, not a definition, and
    # it is what licenses using stationary snapshots for the rest of the
    # chapter. Compare the frozen-system horizon measured *in the ramp*
    # against the same measurement in a stationary climate at each epoch's
    # forcing.
    quasi = np.abs(ramp["horizon_norm"] - horizon_stat)
    print(f"#   quasi-static check: |ramp - stationary| horizon, mean "
          f"{quasi.mean():.4f} TU ({100.0 * (quasi / horizon_stat).mean():.2f}% "
          f"of the horizon), worst {quasi.max():.4f} TU")
    _emit("QUASI_STATIC_DIFF", quasi, ".4f")
    _scalar("QUASI_STATIC_MEAN", quasi.mean(), ".4f")
    _scalar("QUASI_STATIC_PERCENT", 100.0 * (quasi / horizon_stat).mean(), ".3f")
    _scalar("QUASI_STATIC_WORST", quasi.max(), ".4f")
    _scalar("RAMP_DF_PER_FORECAST", RATE * LEAD, ".4f")
    _scalar("FTLE_TAU_MAIN", FTLE_TAU_MAIN, ".2f")
    _scalar("FTLE_STATES", FTLE_STATES, ".0f")

    # the exact identities
    resid = np.abs(energy_lhs - energy_rhs) / energy_lhs
    print(f"#   energy identity <x^2> = F<x>: worst relative residual "
          f"{resid.max():.2e}")
    _emit("STAT_ENERGY_LHS", energy_lhs, ".4f")
    _emit("STAT_ENERGY_RHS", energy_rhs, ".4f")
    _scalar("ENERGY_WORST", resid.max(), ".3e")

    _log("full spectra for the trace identity ...")
    trace_sum = np.empty(2)
    trace_forcing = np.array([forcings[0], forcings[-1]])
    for j, e in enumerate((0, N_EPOCH - 1)):
        full = lyapunov.lyapunov_spectrum(
            systems.lorenz96,
            systems.lorenz96_jacobian,
            endpoint[e]["traj_tail"],
            dt=DT,
            t_final=400.0,
            t_transient=20.0,
            forcing=forcings[e],
        )
        trace_sum[j] = float(full.sum())
    print(f"#   sum of all {SITES} exponents: {trace_sum[0]:.5f} at F={forcings[0]:.2f}, "
          f"{trace_sum[1]:.5f} at F={forcings[-1]:.2f}, exact -{SITES}")
    _emit("TRACE_FORCING", trace_forcing, ".4f")
    _emit("TRACE_SUM", trace_sum, ".5f")
    _scalar("TRACE_EXACT", -float(SITES), ".1f")

    # relative changes over the ramp: the "which predictability" table
    def _change(values) -> float:
        return 100.0 * (values[-1] / values[0] - 1.0)

    print("#   change over the ramp: "
          f"lam1 {_change(lam1):+.1f}%  KS {_change(ks_entropy):+.1f}%  "
          f"unstable {_change(unstable):+.1f}%  D_KY {_change(ky_dim):+.1f}%  "
          f"sat {_change(sat_stat):+.1f}%  horizon {_change(horizon_stat):+.1f}%  "
          f"FTLE(tau={FTLE_TAU_MAIN:g}) {_change(ftle_main):+.1f}%")
    for name, values in (
        ("LAMBDA1", lam1), ("KS", ks_entropy), ("UNSTABLE", unstable),
        ("KY", ky_dim), ("SATURATION", sat_stat), ("HORIZON", horizon_stat),
        ("FTLE", ftle_main),
    ):
        _scalar(f"CHANGE_{name}", _change(values), ".3f")

    # -- 3. the horizon-law decomposition --------------------------------
    print("# --- 3. the horizon-law decomposition ---")
    rms0 = DELTA0_OLD * np.sqrt(SITES)
    frozen = nonstationary.horizon_law_decomposition(
        (sat_stat[0], sat_stat[-1]), (rms0, rms0), (lam1[0], lam1[-1]), FRACTION
    )
    print(f"#   frozen system: total {frozen['total']:+.4f} = instability "
          f"{frozen['instability']:+.4f} + amplitude {frozen['amplitude']:+.4f} "
          f"+ accuracy {frozen['accuracy']:+.4f}")
    for key, value in frozen.items():
        _scalar("DECOMP_" + key.upper(), value, ".5f")
    _scalar("DELTA0_RMS_OLD", rms0, ".5f")
    _scalar("DELTA0_RMS_NEW", DELTA0_NEW * np.sqrt(SITES), ".5f")

    improved = nonstationary.horizon_law_decomposition(
        (sat_stat[0], sat_stat[-1]),
        (rms0, DELTA0_NEW * np.sqrt(SITES)),
        (lam1[0], lam1[-1]),
        FRACTION,
    )
    print(f"#   improved system: total {improved['total']:+.4f} = instability "
          f"{improved['instability']:+.4f} + amplitude {improved['amplitude']:+.4f} "
          f"+ accuracy {improved['accuracy']:+.4f}")
    for key, value in improved.items():
        _scalar("IMPROVED_" + key.upper(), value, ".5f")

    # -- 4. the 2x2 attribution ------------------------------------------
    print("# --- 4. the 2x2: a frozen system re-run on a later atmosphere ---")
    _log("2x2 attribution ...")
    cells = np.empty((2, 2))
    cell_scatter = np.empty((2, 2))
    for ci, e in enumerate((0, N_EPOCH - 1)):
        traj = stationary_climate(forcings[e])
        for si, delta0 in enumerate((DELTA0_OLD, DELTA0_NEW)):
            result = horizon_set(forcings[e], traj, delta0, sat_stat[e])
            cells[si, ci] = result["horizon"]
            cell_scatter[si, ci] = float(np.nanstd(result["per_case"]))
    print(f"#   old system: {cells[0, 0]:.4f} (old climate) -> {cells[0, 1]:.4f} (new)")
    print(f"#   new system: {cells[1, 0]:.4f} (old climate) -> {cells[1, 1]:.4f} (new)")
    attribution = nonstationary.factorial_attribution(
        cells[0, 0], cells[0, 1], cells[1, 0], cells[1, 1]
    )
    for key, value in attribution.items():
        print(f"#   {key:14s} {value:+.4f} TU")
        _scalar("ATTR_" + key.upper(), value, ".5f")
    _emit("CELLS", cells, ".4f", per_line=2)
    _emit("CELL_SCATTER", cell_scatter, ".4f", per_line=2)

    predicted = np.log(DELTA0_OLD / DELTA0_NEW) * (1.0 / lam1[-1] - 1.0 / lam1[0])
    print(f"#   interaction predicted by the horizon law: {predicted:+.4f} TU "
          f"(measured {attribution['interaction']:+.4f}, "
          f"{100.0 * abs(predicted / attribution['interaction'] - 1.0):.1f}% apart)")
    _scalar("ATTR_INTERACTION_PREDICTED", predicted, ".5f")
    _scalar(
        "ATTR_INTERACTION_MISMATCH",
        100.0 * abs(predicted / attribution["interaction"] - 1.0),
        ".2f",
    )
    decade_old = np.log(10.0) / lam1[0]
    decade_new = np.log(10.0) / lam1[-1]
    print(f"#   ln10/lambda: {decade_old:.4f} TU ({decade_old * DAYS_PER_TU:.1f} d) "
          f"-> {decade_new:.4f} TU ({decade_new * DAYS_PER_TU:.1f} d) per decade")
    _scalar("DECADE_OLD", decade_old, ".5f")
    _scalar("DECADE_NEW", decade_new, ".5f")

    # What a decade of analysis error actually bought, per climate. This is
    # the quantity the horizon law calls 1/lambda_1, and comparing them is how
    # the 13% gap in the interaction above is accounted for rather than
    # excused: the law's lambda is asymptotic, and a two-to-four time unit
    # forecast samples a finite-time growth rate that is not exactly it.
    gain = np.log(DELTA0_OLD / DELTA0_NEW)
    lam_eff = np.array([gain / (cells[1, 0] - cells[0, 0]),
                        gain / (cells[1, 1] - cells[0, 1])])
    print(f"#   effective rate from the measured cells: {lam_eff[0]:.4f} "
          f"(old, lambda_1 = {lam1[0]:.4f}) and {lam_eff[1]:.4f} "
          f"(new, lambda_1 = {lam1[-1]:.4f})")
    _emit("LAMBDA_EFFECTIVE", lam_eff, ".5f")

    breakeven = nonstationary.breakeven_accuracy(
        (sat_stat[0], sat_stat[-1]), (lam1[0], lam1[-1]), rms0, FRACTION
    )
    measured = nonstationary.breakeven_accuracy(
        (sat_stat[0], sat_stat[-1]), (lam1[0], lam1[-1]), rms0, FRACTION,
        horizon_before=float(cells[0, 0]),
    )
    print(f"#   break-even (law): analysis error must fall by "
          f"{breakeven['ratio']:.1f}x ({breakeven['decades']:.2f} decades)")
    print(f"#   break-even (anchored on the measured horizon "
          f"{cells[0, 0]:.3f} TU): {measured['ratio']:.1f}x "
          f"({measured['decades']:.2f} decades)")
    for key, value in breakeven.items():
        _scalar("BREAKEVEN_" + key.upper(), value, ".6f")
    for key, value in measured.items():
        _scalar("BREAKEVEN_MEASURED_" + key.upper(), value, ".6f")

    # -- 5. dynamics or circulation --------------------------------------
    print("# --- 5. dynamics or circulation, and the lead time asked about ---")
    first, last = endpoint[0], endpoint[N_EPOCH - 1]
    ftle_before = first["ftle"][FTLE_TAU_MAIN]
    ftle_after = last["ftle"][FTLE_TAU_MAIN]

    correlations = {}
    for name in first["indices"]:
        r_before = float(np.corrcoef(first["indices"][name], ftle_before)[0, 1])
        r_after = float(np.corrcoef(last["indices"][name], ftle_after)[0, 1])
        correlations[name] = (r_before, r_after)
        print(f"#   corr({name:10s}, FTLE) = {r_before:+.3f} (early), "
              f"{r_after:+.3f} (late)")
    names = tuple(correlations)
    print("INDEX_NAMES = " + repr(names))
    _emit("INDEX_CORR_EARLY", [correlations[n][0] for n in names], ".4f")
    _emit("INDEX_CORR_LATE", [correlations[n][1] for n in names], ".4f")
    _scalar(
        "INDEX_CORR_MAX",
        max(max(abs(a), abs(b)) for a, b in correlations.values()),
        ".4f",
    )

    best = max(correlations, key=lambda n: abs(correlations[n][0]))
    index_before = first["indices"][best]
    index_after = last["indices"][best]
    edges = np.quantile(
        np.concatenate([index_before, index_after]),
        np.linspace(0.0, 1.0, SHIFT_SHARE_GROUPS + 1),
    )
    groups_before = np.clip(
        np.digitize(index_before, edges[1:-1]), 0, SHIFT_SHARE_GROUPS - 1
    )
    groups_after = np.clip(
        np.digitize(index_after, edges[1:-1]), 0, SHIFT_SHARE_GROUPS - 1
    )
    share = nonstationary.shift_share(
        groups_before, ftle_before, groups_after, ftle_after, SHIFT_SHARE_GROUPS
    )
    print(f"#   shift-share on '{best}': total {share['total']:+.4f} = within "
          f"{share['within']:+.4f} + between {share['between']:+.4f} "
          f"({100.0 * share['within'] / share['total']:.1f}% within)")
    print("SHIFT_SHARE_INDEX = " + repr(best))
    _scalar("SHARE_TOTAL", share["total"], ".5f")
    _scalar("SHARE_WITHIN", share["within"], ".5f")
    _scalar("SHARE_BETWEEN", share["between"], ".5f")
    _scalar(
        "SHARE_WITHIN_PERCENT", 100.0 * share["within"] / share["total"], ".2f"
    )
    _emit("SHARE_OCCUPANCY_EARLY", share["occupancy_before"], ".4f")
    _emit("SHARE_OCCUPANCY_LATE", share["occupancy_after"], ".4f")
    _emit("SHARE_GROUP_MEAN_EARLY", share["group_mean_before"], ".4f")
    _emit("SHARE_GROUP_MEAN_LATE", share["group_mean_after"], ".4f")
    _scalar("SHIFT_SHARE_GROUPS", SHIFT_SHARE_GROUPS, ".0f")

    tau_early = np.array([first["ftle"][tau].mean() for tau in FTLE_TAUS])
    tau_late = np.array([last["ftle"][tau].mean() for tau in FTLE_TAUS])
    tau_change = 100.0 * (tau_late / tau_early - 1.0)
    for j, tau in enumerate(FTLE_TAUS):
        print(f"#   tau={tau:4.1f} TU: {tau_early[j]:6.4f} -> {tau_late[j]:6.4f} "
              f"({tau_change[j]:+.1f}%), ratio to lambda_1 "
              f"{tau_early[j] / lam1[0]:.2f} -> {tau_late[j] / lam1[-1]:.2f}")
    print(f"#   lambda_1 itself: {lam1[0]:.4f} -> {lam1[-1]:.4f} "
          f"({_change(lam1):+.1f}%)")
    _emit("FTLE_TAUS", FTLE_TAUS, ".2f")
    _emit("FTLE_TAU_EARLY", tau_early, ".5f")
    _emit("FTLE_TAU_LATE", tau_late, ".5f")
    _emit("FTLE_TAU_CHANGE", tau_change, ".3f")
    _emit("FTLE_HIST_EARLY", ftle_before, ".4f", per_line=10)
    _emit("FTLE_HIST_LATE", ftle_after, ".4f", per_line=10)

    # -- 6. detectability ------------------------------------------------
    print("# --- 6. how long a record does a trend need? ---")
    _log("dense launches for the decorrelation ...")
    forcing_mid = float(forcings[N_EPOCH // 2])
    traj = stationary_climate(forcing_mid)
    sat_mid = errorgrowth.saturation_level(traj, statistic="rms")
    n_dense = int(round(DENSE_SPAN / DENSE_EVERY))
    stride = int(round(DENSE_EVERY / DT))
    starts = traj[: n_dense * stride : stride]
    rng = np.random.default_rng(77)
    perturbed = starts + DELTA0_OLD * rng.normal(size=starts.shape)
    t_lead = integrate.trajectory_grid(LEAD, DT)
    dense = np.empty(n_dense)
    chunk = 200
    for a in range(0, n_dense, chunk):
        b = min(a + chunk, n_dense)
        control = integrate.rk4(
            systems.lorenz96, starts[a:b], t_lead, forcing=forcing_mid
        )
        twin = integrate.rk4(
            systems.lorenz96, perturbed[a:b], t_lead, forcing=forcing_mid
        )
        err = np.linalg.norm(control - twin, axis=-1)
        for j in range(b - a):
            dense[a + j] = verification.skill_horizon(
                t_lead, err[:, j], FRACTION * sat_mid, decreasing=False
            )

    # The autocorrelation function itself, from all of the launches rather
    # than the 400 emitted for the figure: 400 points give a lag-k standard
    # error of 0.05, which is the same size as the correlations being read.
    _centred = dense - np.nanmean(dense)
    _var = float(np.nanmean(_centred**2))
    acf_lags = np.arange(0, 21)
    acf = np.array(
        [1.0]
        + [
            float(np.nanmean(_centred[:-k] * _centred[k:]) / _var)
            for k in range(1, 21)
        ]
    )
    positive = np.flatnonzero(acf[1:] <= 0.0)
    tau_int = 1.0 + 2.0 * acf[1 : (int(positive[0]) + 1 if positive.size else 21)].sum()
    print(f"#   horizon autocorrelation: "
          f"{', '.join(f'lag {k} {v:+.3f}' for k, v in zip(acf_lags[1:6], acf[1:6]))}"
          f"; integrated autocorrelation time {tau_int:.2f} launches "
          f"({tau_int * DENSE_EVERY * DAYS_PER_TU:.1f} days)")
    _emit("ACF_LAGS", acf_lags, ".0f", per_line=11)
    _emit("ACF", acf, ".4f", per_line=7)
    _scalar("ACF_TAU_INT", tau_int, ".3f")
    _scalar("ACF_TAU_DAYS", tau_int * DENSE_EVERY * DAYS_PER_TU, ".2f")

    n_eff = nonstationary.effective_sample_size(dense)
    ratio = n_eff / n_dense
    per_year = 365.0 / DAYS_PER_TU * DENSE_EVERY / DENSE_EVERY  # launches per year
    launches_per_year = 365.0 / DAYS_PER_TU / DENSE_EVERY
    independent_per_year = launches_per_year * ratio
    scatter = float(np.nanstd(dense))
    print(f"#   {n_dense} launches {DENSE_EVERY:g} TU apart at F={forcing_mid:.2f}: "
          f"horizon {np.nanmean(dense):.3f} +- {scatter:.3f} TU")
    print(f"#   effective sample size {n_eff:.1f} of {n_dense} "
          f"({100.0 * ratio:.1f}%); {launches_per_year:.0f} launches per year "
          f"-> {independent_per_year:.1f} independent")
    _scalar("DENSE_EVERY", DENSE_EVERY, ".2f")
    _scalar("DENSE_N", n_dense, ".0f")
    _scalar("DENSE_FORCING", forcing_mid, ".4f")
    _scalar("DENSE_MEAN", float(np.nanmean(dense)), ".4f")
    _scalar("DENSE_SCATTER", scatter, ".4f")
    _scalar("DENSE_NEFF", n_eff, ".2f")
    _scalar("DENSE_RATIO", ratio, ".4f")
    _scalar("LAUNCHES_PER_YEAR", launches_per_year, ".2f")
    _scalar("INDEPENDENT_PER_YEAR", independent_per_year, ".2f")
    _emit("DENSE_HORIZONS", dense[:240], ".4f", per_line=10)
    del per_year

    years = TOTAL / (365.0 / DAYS_PER_TU)
    trend_per_decade = np.array([0.02, 0.05, 0.10, 0.20])
    horizon_ref = float(np.nanmean(dense))
    required_naive = np.empty(trend_per_decade.size)
    required_eff = np.empty(trend_per_decade.size)
    for j, frac in enumerate(trend_per_decade):
        slope = frac * horizon_ref / 10.0  # TU per year
        required_naive[j] = nonstationary.minimum_record_length(
            slope, scatter, int(round(launches_per_year))
        )
        required_eff[j] = nonstationary.minimum_record_length(
            slope, scatter, max(int(round(independent_per_year)), 1)
        )
        print(f"#   {100 * frac:4.0f}%/decade: {required_eff[j]:.0f} years "
              f"(naive, all launches independent: {required_naive[j]:.0f})")
    _emit("TREND_PER_DECADE", trend_per_decade, ".4f")
    _emit("REQUIRED_YEARS", required_eff, ".0f")
    _emit("REQUIRED_YEARS_NAIVE", required_naive, ".0f")
    _scalar("RECORD_YEARS", years, ".1f")
    _scalar("HORIZON_REF", horizon_ref, ".4f")

    # sign errors and power in short records, by Monte Carlo on the model's
    # own trend and scatter -- the analytic power counts significance, not
    # whether the significant answer had the right sign.
    rng = np.random.default_rng(101)
    slope_true = 0.05 * horizon_ref / 10.0
    m_eff = max(int(round(independent_per_year)), 1)
    mc_years = np.array([5, 10, 20, 40])
    mc_power = np.empty(mc_years.size)
    mc_wrong = np.empty(mc_years.size)
    trials = 4000
    for j, n_year in enumerate(mc_years):
        x = np.repeat(np.arange(n_year, dtype=float), m_eff)
        signal = slope_true * x
        noise = rng.normal(scale=scatter, size=(trials, x.size))
        detected = 0
        wrong = 0
        for k in range(trials):
            tr = nonstationary.linear_trend(x, signal + noise[k])
            if tr["pvalue"] < 0.05:
                detected += 1
                if tr["slope"] * slope_true < 0.0:
                    wrong += 1
        mc_power[j] = detected / trials
        mc_wrong[j] = wrong / trials
        analytic = nonstationary.trend_detection_power(
            slope_true, scatter, int(n_year), m_eff
        )
        print(f"#   {n_year:3d} years: Monte Carlo power {mc_power[j]:.3f} "
              f"(analytic {analytic:.3f}), significant-and-wrong-sign "
              f"{mc_wrong[j]:.4f}")
    _emit("MC_YEARS", mc_years, ".0f")
    _emit("MC_POWER", mc_power, ".4f")
    _emit("MC_WRONG_SIGN", mc_wrong, ".5f")
    _scalar("MC_SLOPE_TRUE", slope_true, ".6f")
    _scalar("MC_TRIALS", trials, ".0f")
    _scalar("MC_M_EFF", m_eff, ".0f")

    _log(f"total {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
