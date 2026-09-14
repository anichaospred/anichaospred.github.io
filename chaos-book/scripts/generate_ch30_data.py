#!/usr/bin/env python3
r"""Precompute chapter 30's ergodic-theory experiments.

A climate model is run once for five hundred years. A forecast ensemble is run
a hundred times for five. Both are used to state "the climate". This chapter
asks what object they are both estimating, when they estimate the same one, and
how long each has to be before the answer means anything.

Six blocks, one per section.

1. **The measure** -- one long Lorenz 63 run against a scattered ensemble
   sampled at a single instant, compared with the Wasserstein distance; and the
   demonstration that the object they agree on has no density.
2. **The rate** -- how fast a time average converges, measured by batch means
   for three observables of the same run, against the standard integrated
   autocorrelation estimate; and the exact moment identities as a
   reference-free convergence diagnostic.
3. **Ergodic is not chaotic** -- an irrational rotation, the logistic map at
   r = 4 whose invariant measure is known exactly, and an i.i.d. sample:
   discrepancy and mean error against sample size.
4. **The budget** -- one long run against many short ones at fixed total cost,
   measured, against the parameter-free spin-up penalty.
5. **When the ergodic time exceeds the record** -- a noisy double well, whose
   invariant measure is exactly Boltzmann, and the error bar a single run
   reports about a climatology it has no access to.
6. **No invariant measure at all** -- Lorenz 63 with a ramped Rayleigh number:
   the trailing average lags, the ensemble does not.

Run from chaos-book/:
    python3 scripts/generate_ch30_data.py        # ~12 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import (  # noqa: E402
    dimension,
    earlywarning,
    ergodic,
    integrate,
    lyapunov,
    nonstationary,
    systems,
)

# --- the reference system ------------------------------------------------
DT = 0.01
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
LONG = 60000.0          # the reference run: ~820 years at 5 d/TU
SPINUP = 100.0
DAYS_PER_TU = 5.0

# --- section 1: the measure ----------------------------------------------
ENSEMBLE_MEMBERS = 4000
ENSEMBLE_SPREAD = 300.0   # TU of independent spin-up before the snapshot
BINS_Z = 60
BOX_SIDES = (4, 8, 16, 32, 64, 128)
MIN_PER_BOX = 20.0   # below this the box count is sampling, not geometry

# --- section 2: the rate -------------------------------------------------
WINDOWS = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
IDENTITY_WINDOWS = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)

# --- section 3: ergodic is not chaotic -----------------------------------
GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0
SAMPLE_SIZES = (100, 300, 1000, 3000, 10000, 30000, 100000)
REALISATIONS = 200

# --- section 4: the budget ------------------------------------------------
BUDGET = 2000.0
SPIN_COST = 20.0
MEMBER_COUNTS = (1, 2, 5, 10, 25, 50)
BUDGET_REPS = 2000

# --- section 5: the bistable case ----------------------------------------
WELL_NOISE = 0.35
WELL_MEMBERS = 400
WELL_TOTAL = 20000.0
WELL_WINDOWS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
NOISE_LADDER = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

# --- section 6: the ramp -------------------------------------------------
RAMP_RHO0, RAMP_RATE, RAMP_TOTAL = 28.0, 0.003, 4000.0
RAMP_MEMBERS = 200
RAMP_WINDOWS = (50.0, 100.0, 200.0, 400.0, 800.0)
# Measured once on the unforced attractor over rho in [26, 40]; the same fit
# systems.earth_system uses. It converts a ramp in rho into a drift in <z>.
Z_SLOPE, Z_INTERCEPT = 1.0007, -4.4603


# =========================================================================
# emitters -- every float goes through these, never a bare f-string format,
# so a non-finite value is spelled float("nan") rather than becoming a
# NameError in the reader's browser
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
def reference_run() -> tuple[np.ndarray, np.ndarray]:
    """One long Lorenz 63 trajectory, spin-up already discarded."""
    t = integrate.trajectory_grid(t_final=LONG + SPINUP, dt=DT)
    traj = integrate.rk4(systems.lorenz63, np.array([1.0, 1.0, 20.0]), t)
    keep = int(SPINUP / DT)
    return t[keep:] - t[keep], traj[keep:]


def scattered_ensemble(seed: int = 0) -> np.ndarray:
    """States of many trajectories at one instant, from scattered starts.

    The ensemble analogue of the long run: independent initial conditions,
    integrated long enough that the memory of where they started is gone, and
    then read at a single time. This is the estimator a climate ensemble uses.
    """
    rng = np.random.default_rng(seed)
    start = rng.normal(loc=(0.0, 0.0, 25.0), scale=(15.0, 20.0, 12.0),
                       size=(ENSEMBLE_MEMBERS, 3))
    t = integrate.trajectory_grid(t_final=ENSEMBLE_SPREAD, dt=DT)
    return integrate.rk4(systems.lorenz63, start, t)[-1]


# =========================================================================
# 1. the measure
# =========================================================================
def block_measure(traj: np.ndarray) -> dict:
    _log("block 1: the invariant measure")
    tic = time.time()
    snapshot = scattered_ensemble()
    _log(f"  {ENSEMBLE_MEMBERS} scattered members in {time.time() - tic:.0f} s")

    out: dict = {"snapshot": snapshot}
    # the two estimators, variable by variable
    for axis, name in ((0, "x"), (2, "z")):
        long_series = traj[:, axis]
        thin = long_series[::100]          # 1 TU apart: a fair sample size
        w1 = ergodic.wasserstein1(thin, snapshot[:, axis])
        out[f"w1_{name}"] = w1
        out[f"spread_{name}"] = float(long_series.std())
        out[f"mean_long_{name}"] = float(ergodic.time_average(long_series, DT))
        out[f"mean_snap_{name}"] = float(snapshot[:, axis].mean())
        _log(f"  {name}: W1 = {w1:.4f} ({100 * w1 / long_series.std():.2f} % of sd), "
             f"means {out[f'mean_long_{name}']:.4f} / {out[f'mean_snap_{name}']:.4f}")

    # a control: the same comparison against a measure that is genuinely
    # different, so the W1 above can be read against something
    rng = np.random.default_rng(1)
    gaussian = rng.normal(traj[:, 2].mean(), traj[:, 2].std(), size=ENSEMBLE_MEMBERS)
    out["w1_gaussian"] = ergodic.wasserstein1(traj[::100, 2], gaussian)
    _log(f"  control: W1 to a Gaussian of the same mean and sd = "
         f"{out['w1_gaussian']:.4f}")

    # the histograms themselves
    edges = np.linspace(traj[:, 2].min(), traj[:, 2].max(), BINS_Z + 1)
    out["edges_z"] = edges
    out["hist_long_z"] = ergodic.empirical_measure(traj[::10, 2], edges)["density"]
    out["hist_snap_z"] = ergodic.empirical_measure(snapshot[:, 2], edges)["density"]
    out["tv_z"] = ergodic.total_variation(
        ergodic.empirical_measure(traj[::10, 2], edges)["probability"],
        ergodic.empirical_measure(snapshot[:, 2], edges)["probability"],
    )

    # ... and the object they agree on has no density. A measure with a
    # density fills its support, so box counting returns the dimension of the
    # space; a singular measure returns less. The uniform control is what makes
    # that readable: run through the same estimator it must return exactly 3,
    # and any shortfall below 3 for Lorenz 63 is then geometry, not method.
    sample = traj[::2]
    rng_box = np.random.default_rng(99)
    control = rng_box.random((sample.shape[0], 3))
    for label, points in (("box", sample), ("control", control)):
        low = points.min(axis=0)
        span = float((points.max(axis=0) - low).max())
        counts = []
        for side in BOX_SIDES:
            index = np.clip(
                np.floor((points - low) / span * side).astype(np.int64), 0, side - 1
            )
            key = (index[:, 0] * side + index[:, 1]) * side + index[:, 2]
            counts.append(float(np.unique(key).size))
        counts = np.asarray(counts)
        usable = points.shape[0] / counts >= MIN_PER_BOX
        fit = ergodic.power_law_fit(np.asarray(BOX_SIDES)[usable], counts[usable])
        out[f"{label}_occupied"] = counts
        out[f"{label}_usable"] = usable.astype(float)
        out[f"{label}_dimension"] = fit["exponent"]
        out[f"{label}_r2"] = fit["r2"]
        _log(f"  {label}: occupied {counts.astype(int)}, D_0 = "
             f"{fit['exponent']:.4f} (r2 {fit['r2']:.5f}) over sides "
             f"{np.asarray(BOX_SIDES)[usable]}")
    out["box_sides"] = np.asarray(BOX_SIDES, dtype=float)
    out["box_points"] = float(sample.shape[0])

    # a thinned trace of the reference run, to draw behind the ensemble
    # scatter: the point of that panel is that the members lie on the attractor
    # the run traces out, which 600 dots alone do not say
    out["reference_xz"] = traj[::600, ::2][:5000]

    # the dimension of the measure, independently: Kaplan-Yorke from the
    # dynamics and the box-counting slope from the sample
    spectrum = lyapunov.lyapunov_spectrum(
        systems.lorenz63, systems.lorenz63_jacobian, traj[-1], t_final=2000.0, dt=DT
    )
    out["spectrum"] = spectrum
    out["d_ky"] = lyapunov.kaplan_yorke_dimension(spectrum)
    out["trace"] = -(SIGMA + 1.0 + BETA)
    _log(f"  spectrum {np.round(spectrum, 4)}, sum {spectrum.sum():.5f} "
         f"(exact {out['trace']:.5f}), D_KY = {out['d_ky']:.4f}")
    return out


# =========================================================================
# 2. the rate
# =========================================================================
def block_rate(traj: np.ndarray) -> dict:
    _log("block 2: how fast does a time average converge?")
    out: dict = {}
    series = {
        "x": traj[:, 0],
        "z": traj[:, 2],
        "x2": traj[:, 0] ** 2,
    }
    for name, values in series.items():
        batch = ergodic.batch_variance(values, WINDOWS, DT)
        tau_int = ergodic.autocorrelation_time(values, DT)
        out[f"batch_window_{name}"] = batch["window"]
        out[f"batch_var_{name}"] = batch["variance"]
        out[f"batch_tau_{name}"] = batch["tau_eff"]
        out[f"batch_blocks_{name}"] = batch["blocks"]
        out[f"tau_int_{name}"] = tau_int
        out[f"var_{name}"] = float(values.var())
        out[f"mean_{name}"] = float(ergodic.time_average(values, DT))
        # The asymptotic tau: the longest window that still leaves 120 blocks.
        # A variance estimated from n blocks has relative error sqrt(2/n), so
        # 30 blocks is 26 % and the rule first written here picked exactly that
        # noise -- it returned tau = 0.756 for x where the neighbouring windows
        # agree on 1.00.
        usable = batch["blocks"] >= 120
        tau_plateau = float(batch["tau_eff"][usable][-1]) if usable.any() else np.nan
        out[f"tau_plateau_{name}"] = tau_plateau
        # Fit only where tau_eff has stopped rising and the block count is
        # still large: below ~10 TU the short windows are in the transient of
        # the autocorrelation and drag the exponent towards zero.
        fitted = usable & (batch["window"] >= 10.0)
        law = ergodic.power_law_fit(batch["window"][fitted], batch["variance"][fitted])
        out[f"law_exponent_{name}"] = law["exponent"]
        out[f"law_r2_{name}"] = law["r2"]
        _log(f"  {name}: var {values.var():9.3f}  tau_int {tau_int:.4f}  "
             f"tau_plateau {tau_plateau:.4f}  ratio {tau_int / tau_plateau:6.1f}  "
             f"var ~ T^{law['exponent']:.3f}")

    # the autocorrelation functions themselves, for the figure
    lags = np.arange(0, int(5.0 / DT) + 1, 5)
    out["acf_lag"] = lags * DT
    for name, values in series.items():
        centred = values - values.mean()
        var = float(np.mean(centred**2))
        out[f"acf_{name}"] = np.array(
            [float(np.mean(centred[: len(centred) - k] * centred[k:]) / var)
             if k else 1.0 for k in lags]
        )

    # what the difference costs: run length for a target standard error
    target = {"x": 0.05, "z": 0.05, "x2": 1.0}
    for name in series:
        need_plateau = (out[f"tau_plateau_{name}"] * out[f"var_{name}"]
                        / target[name] ** 2)
        need_naive = out[f"tau_int_{name}"] * out[f"var_{name}"] / target[name] ** 2
        out[f"need_{name}"] = need_plateau
        out[f"need_naive_{name}"] = need_naive
        _log(f"  {name}: to reach se = {target[name]}, {need_plateau:.0f} TU "
             f"(the standard recipe would demand {need_naive:.0f})")
    out["target"] = np.array([target[k] for k in ("x", "z", "x2")])

    # the exact moment identities, as a reference-free convergence diagnostic
    rows = []
    for span in IDENTITY_WINDOWS:
        n = int(span / DT) + 1
        if n > traj.shape[0]:
            continue
        res = ergodic.lorenz63_moment_residuals(traj[:n], DT, SIGMA, RHO, BETA)
        rows.append([
            span, res["finite_x2"], res["finite_z"], res["limit_x2"],
            res["limit_z"], res["mean_x"], res["mean_z"], res["mean_x2"] / BETA,
        ])
        _log(f"  T = {span:7.0f}: finite {res['finite_x2']:+.2e} {res['finite_z']:+.2e}"
             f" | limit {res['limit_x2']:+.4f} {res['limit_z']:+.4f} | "
             f"<x> {res['mean_x']:+.4f}  <z> {res['mean_z']:.4f} vs "
             f"<x^2>/beta {res['mean_x2'] / BETA:.4f}")
    rows = np.asarray(rows)
    out["identity_span"] = rows[:, 0]
    out["identity_finite_x2"] = rows[:, 1]
    out["identity_finite_z"] = rows[:, 2]
    out["identity_limit_x2"] = rows[:, 3]
    out["identity_limit_z"] = rows[:, 4]
    out["identity_mean_x"] = rows[:, 5]
    out["identity_mean_z"] = rows[:, 6]
    out["identity_mean_x2b"] = rows[:, 7]

    # Running averages for the convergence figure, sampled logarithmically:
    # the figure's x axis is logarithmic, so uniform thinning spends 90 % of
    # its points on the last decade and 9,000 lines of the notebook on them.
    picks = np.unique(
        np.geomspace(1.0, traj.shape[0] - 1, 500).astype(np.int64)
    )
    out["running_time"] = picks * DT
    for name in ("x", "z"):
        out[f"running_{name}"] = ergodic.running_time_average(series[name], DT)[picks]
    return out


# =========================================================================
# 3. ergodic is not chaotic
# =========================================================================
def logistic_orbit(n: int, x0: float) -> np.ndarray:
    x = np.empty(n)
    v = float(x0)
    for i in range(n):
        v = 4.0 * v * (1.0 - v)
        x[i] = v
    return x


def block_sampling() -> dict:
    _log("block 3: ergodic is not chaotic")
    out: dict = {}
    rng = np.random.default_rng(12)
    biggest = max(SAMPLE_SIZES)
    starts = rng.uniform(0.05, 0.95, size=REALISATIONS)
    offsets = rng.uniform(0.0, 1.0, size=REALISATIONS)

    # one long orbit per realisation, then truncate: each sample size sees the
    # same trajectory, which is how a modeller actually shortens a run
    logistic = np.array([logistic_orbit(biggest, s) for s in starts])
    rotation = np.array([ergodic.rotation_orbit(GOLDEN, biggest, o) for o in offsets])
    random = rng.random((REALISATIONS, biggest))

    # the exact answers: <x> = 1/2 for all three, by the arcsine law for the
    # logistic map and by uniformity for the other two
    # Discrepancy is defined against a *target* measure, and only the rotation
    # and the random sample target the uniform one. The logistic orbit targets
    # the arcsine law, so it is pushed through its own distribution function
    # first -- which is uniform exactly when the orbit is sampling the arcsine
    # law correctly. Measuring its raw discrepancy instead reports how far the
    # arcsine law is from uniform (a constant, ~0.2) and says nothing about
    # convergence at all.
    for label, block in (("rotation", rotation), ("logistic", logistic),
                         ("random", random)):
        uniformised = (
            ergodic.logistic_invariant_cdf(block) if label == "logistic" else block
        )
        errors, discrepancies = [], []
        for n in SAMPLE_SIZES:
            means = block[:, :n].mean(axis=1)
            errors.append(float(np.sqrt(np.mean((means - 0.5) ** 2))))
            discrepancies.append(
                float(np.mean([ergodic.star_discrepancy(row)
                               for row in uniformised[:20, :n]]))
            )
        out[f"err_{label}"] = np.asarray(errors)
        out[f"disc_{label}"] = np.asarray(discrepancies)
        law = ergodic.power_law_fit(SAMPLE_SIZES, errors)
        disc_law = ergodic.power_law_fit(SAMPLE_SIZES, discrepancies)
        out[f"exp_{label}"] = law["exponent"]
        out[f"exp_disc_{label}"] = disc_law["exponent"]
        _log(f"  {label:9s} rms error ~ N^{law['exponent']:+.3f} "
             f"(r2 {law['r2']:.4f}), discrepancy ~ N^{disc_law['exponent']:+.3f}; "
             f"at N=1000 error {errors[SAMPLE_SIZES.index(1000)]:.2e}")

    out["sizes"] = np.asarray(SAMPLE_SIZES, dtype=float)
    # the logistic map's measure against its closed form, as a check that the
    # orbit really is sampling the arcsine law and not floating-point debris
    exact = ergodic.logistic_invariant_quantile(
        (np.arange(biggest, dtype=float) + 0.5) / biggest
    )
    out["w1_logistic"] = ergodic.wasserstein1(logistic[0], exact)
    out["mean_logistic"] = float(logistic.mean())
    out["var_logistic"] = float(logistic.var())
    _log(f"  logistic orbit vs the exact arcsine law: W1 = {out['w1_logistic']:.5f}, "
         f"mean {out['mean_logistic']:.5f} (exact 0.5), "
         f"var {out['var_logistic']:.5f} (exact 0.125)")

    # the density, for the figure
    grid = np.linspace(0.002, 0.998, 200)
    out["arcsine_grid"] = grid
    out["arcsine_density"] = ergodic.logistic_invariant_density(grid)
    edges = np.linspace(0.0, 1.0, 61)
    out["arcsine_edges"] = edges
    out["arcsine_measured"] = ergodic.empirical_measure(
        logistic[:20].ravel(), edges
    )["density"]
    return out


# =========================================================================
# 4. one long run or many short ones?
# =========================================================================
def block_budget(traj: np.ndarray) -> dict:
    _log("block 4: one long run or many short ones, at fixed cost?")
    out: dict = {}
    rng = np.random.default_rng(21)
    pool = traj[::2000]                       # 20 TU apart: independent starts
    measured, predicted, sampled = [], [], []
    for members in MEMBER_COUNTS:
        per_run = BUDGET / members
        if per_run <= SPIN_COST:
            continue
        tic = time.time()
        idx = rng.integers(0, pool.shape[0], size=(BUDGET_REPS, members))
        t = integrate.trajectory_grid(t_final=per_run, dt=DT)
        runs = integrate.rk4(systems.lorenz63, pool[idx], t)
        kept = runs[int(SPIN_COST / DT) :, ..., 0]      # (time, reps, members)
        means = kept.mean(axis=(0, 2))
        measured.append(float(means.var(ddof=1)))
        predicted.append(ergodic.budget_penalty(BUDGET, members, SPIN_COST))
        sampled.append(BUDGET - members * SPIN_COST)
        _log(f"  M = {members:3d}: {per_run:6.0f} TU each, {sampled[-1]:5.0f} TU kept, "
             f"var {measured[-1]:.5f} in {time.time() - tic:.0f} s")
    measured = np.asarray(measured)
    out["members"] = np.asarray(MEMBER_COUNTS[: measured.size], dtype=float)
    out["budget_var"] = measured
    out["budget_ratio"] = measured / measured[0]
    out["budget_predicted"] = np.asarray(predicted)
    out["budget_sampled"] = np.asarray(sampled)
    _log(f"  measured inflation {np.round(out['budget_ratio'], 3)}")
    _log(f"  spin-up alone predicts {np.round(out['budget_predicted'], 3)}")

    # the residual: short segments have a larger effective correlation time
    # than long ones, which the spin-up accounting knows nothing about
    excess = out["budget_ratio"] / out["budget_predicted"]
    out["budget_excess"] = excess
    # 500 TU rather than the full 1980: a window that long leaves only 30
    # blocks of the reference run and the variance estimate is then 26 % noise.
    batch = ergodic.batch_variance(
        traj[:, 0], [500.0, BUDGET / MEMBER_COUNTS[-1] - SPIN_COST], DT
    )
    out["tau_long_segment"] = float(batch["tau_eff"][0])
    out["tau_short_segment"] = float(batch["tau_eff"][1])
    _log(f"  excess over the spin-up prediction {np.round(excess, 3)}; "
         f"tau_eff {out['tau_long_segment']:.3f} (long segment) vs "
         f"{out['tau_short_segment']:.3f} (short)")
    out["budget"] = BUDGET
    out["spin_cost"] = SPIN_COST
    return out


# =========================================================================
# 5. when the ergodic time exceeds the record
# =========================================================================
def block_bistable() -> dict:
    _log("block 5: when the ergodic time exceeds the record")
    out: dict = {}
    barrier = systems.double_well_barrier(0.0)
    tau_kramers = earlywarning.kramers_escape_time(barrier, WELL_NOISE, 2.0, 1.0)
    out["barrier"] = barrier
    out["noise"] = WELL_NOISE
    out["tau_kramers"] = tau_kramers
    out["exponent"] = 2.0 * barrier / WELL_NOISE**2
    _log(f"  barrier {barrier:.4f}, 2dV/s^2 = {out['exponent']:.3f}, "
         f"Kramers tau = {tau_kramers:.1f} TU")

    tic = time.time()
    t = integrate.trajectory_grid(t_final=WELL_TOTAL, dt=DT)
    x0 = np.where(np.arange(WELL_MEMBERS) % 2 == 0, 1.0, -1.0)
    path = integrate.rk4_stochastic(
        systems.double_well, x0, t, noise_std=WELL_NOISE, seed=17
    )
    _log(f"  {WELL_MEMBERS} members x {WELL_TOTAL:.0f} TU in {time.time() - tic:.0f} s")
    out["grand_mean"] = float(path.mean())
    out["occupancy"] = ergodic.occupancy(path)
    out["typical"] = float(np.abs(path).mean())

    # the exact invariant measure, and what the run measured
    grid = np.linspace(-2.2, 2.2, 601)
    out["well_grid"] = grid
    out["well_exact"] = ergodic.boltzmann_density(
        grid, systems.double_well_potential(grid), WELL_NOISE
    )
    edges = np.linspace(-2.2, 2.2, 89)
    out["well_edges"] = edges
    out["well_measured"] = ergodic.empirical_measure(path.ravel(), edges)["density"]
    short = ergodic.empirical_measure(path[: int(100.0 / DT), 0], edges)["density"]
    out["well_short"] = short
    _log(f"  grand mean {out['grand_mean']:+.4f} (exact 0), "
         f"occupancy {out['occupancy']:.4f} (exact 0.5)")

    # the estimator's error against window length, and the error bar a single
    # window reports for itself
    rms, naive, bias = [], [], []
    for window in WELL_WINDOWS:
        block = int(window / DT)
        means = ergodic.block_means(path, block)      # (blocks, members)
        rms.append(float(np.sqrt(np.mean(means**2))))
        bias.append(float(means.mean()))
        segments = path[: block, : min(40, WELL_MEMBERS)]
        naive.append(float(np.mean([
            ergodic.sampling_error(
                float(seg.var()), ergodic.autocorrelation_time(seg, DT), window
            )
            for seg in segments.T
        ])))
        _log(f"  T = {window:6.0f}: true rms error {rms[-1]:.4f}, "
             f"self-reported {naive[-1]:.4f}, overconfidence "
             f"{rms[-1] / naive[-1]:6.1f}x")
    out["well_windows"] = np.asarray(WELL_WINDOWS, dtype=float)
    out["well_rms"] = np.asarray(rms)
    out["well_naive"] = np.asarray(naive)
    out["well_bias"] = np.asarray(bias)
    out["well_overconfidence"] = out["well_rms"] / out["well_naive"]

    # the ergodic time implied by the measurement itself: for T >> tau the rms
    # error is |x| sqrt(tau/T), so tau = T (rms/|x|)^2
    implied = out["well_windows"] * (out["well_rms"] / out["typical"]) ** 2
    out["well_tau_implied"] = implied
    out["tau_erg"] = float(implied[-1])
    _log(f"  implied ergodic time by window: {np.round(implied, 1)}; "
         f"taking the longest, {out['tau_erg']:.0f} TU against Kramers' "
         f"{tau_kramers:.0f} ({100 * (out['tau_erg'] / tau_kramers - 1):+.1f} %)")

    # the within-well correlation time, which is what the naive estimator sees
    within = path[: int(100.0 / DT), 0]
    out["tau_within"] = ergodic.autocorrelation_time(within, DT)
    out["restoring"] = abs(systems.double_well_restoring_rate(0.0))
    _log(f"  within-well tau {out['tau_within']:.3f} TU against a restoring "
         f"rate of {out['restoring']:.2f} (1/rate = {1 / out['restoring']:.2f})")

    # how the requirement scales with the noise: the Kramers exponential
    ladder = np.asarray(NOISE_LADDER)
    out["noise_ladder"] = ladder
    out["tau_ladder"] = np.array([
        earlywarning.kramers_escape_time(barrier, s, 2.0, 1.0) for s in ladder
    ])
    _log(f"  tau over the noise ladder: {np.round(out['tau_ladder'], 0)}")
    return out


# =========================================================================
# 6. no invariant measure at all
# =========================================================================
def block_ramp(traj: np.ndarray) -> dict:
    _log("block 6: a forcing that moves, and no invariant measure to estimate")
    out: dict = {}
    starts = traj[::2000][:RAMP_MEMBERS]
    t = integrate.trajectory_grid(t_final=RAMP_TOTAL, dt=DT)
    tic = time.time()
    ens = integrate.rk4(
        systems.lorenz63_ramped, starts, t, rho_start=RAMP_RHO0, rho_rate=RAMP_RATE
    )
    _log(f"  {RAMP_MEMBERS} ramped members x {RAMP_TOTAL:.0f} TU in "
         f"{time.time() - tic:.0f} s")
    z = ens[..., 2]
    ensemble_mean = z.mean(axis=1)
    rho = systems.lorenz63_ramp(t, RAMP_RHO0, RAMP_RATE)
    drift = Z_SLOPE * RAMP_RATE

    out["rho_start"] = RAMP_RHO0
    out["rho_end"] = float(rho[-1])
    out["ramp_rate"] = RAMP_RATE
    out["drift"] = drift
    out["ramp_total"] = RAMP_TOTAL
    out["members"] = RAMP_MEMBERS
    out["ens_early"] = float(z[: int(50.0 / DT)].mean())
    out["ens_late"] = float(z[-int(50.0 / DT) :].mean())
    out["fit_early"] = Z_SLOPE * rho[0] + Z_INTERCEPT
    out["fit_late"] = Z_SLOPE * rho[-1] + Z_INTERCEPT
    _log(f"  rho {rho[0]:.1f} -> {rho[-1]:.1f}; ensemble <z> "
         f"{out['ens_early']:.3f} -> {out['ens_late']:.3f} against the "
         f"stationary fit {out['fit_early']:.3f} -> {out['fit_late']:.3f}")

    single = z[:, 0]
    lags, predictions = [], []
    for window in RAMP_WINDOWS:
        w = int(window / DT)
        trail = ergodic.trailing_average(single, w)
        span = (w - 1) * DT
        use = np.arange(single.size) > int(1000.0 / DT)
        lag = float(np.nanmean((trail - ensemble_mean)[use]))
        lags.append(lag)
        predictions.append(ergodic.trailing_average_bias(drift, span))
        _log(f"  window {window:6.0f}: measured lag {lag:+.4f}, "
             f"predicted {predictions[-1]:+.4f} "
             f"({100 * abs(lag / predictions[-1] - 1):.1f} % apart)")
    out["ramp_windows"] = np.asarray(RAMP_WINDOWS, dtype=float)
    out["ramp_lag"] = np.asarray(lags)
    out["ramp_predicted"] = np.asarray(predictions)

    # the ensemble's own error, for comparison: it is a sampling error and it
    # shrinks with members, unlike the lag
    out["ens_stderr"] = float(z[int(2000.0 / DT)].std() / np.sqrt(RAMP_MEMBERS))
    out["ens_spread"] = float(z[int(2000.0 / DT)].std())

    # the curves for the figure
    step = 400
    out["ramp_time"] = t[::step]
    out["ramp_rho_curve"] = rho[::step]
    out["ramp_ens_mean"] = ensemble_mean[::step]
    out["ramp_single"] = single[::step]
    for window in (100.0, 400.0):
        w = int(window / DT)
        out[f"ramp_trail_{int(window)}"] = ergodic.trailing_average(single, w)[::step]

    # the one-member ensemble is not the point; the drift is. How many members
    # would a trailing average need to match the ensemble's accuracy? None --
    # it is biased, so beyond a certain lead the comparison has no answer.
    out["ramp_bias_at_400"] = float(out["ramp_predicted"][RAMP_WINDOWS.index(400.0)])
    return out


# =========================================================================
# main
# =========================================================================
def main() -> None:
    started = time.time()
    _log("chapter 30 -- ergodic theory and invariant measures")
    tic = time.time()
    _, traj = reference_run()
    _log(f"reference run: {LONG:.0f} TU in {time.time() - tic:.0f} s, "
         f"{traj.shape[0]} samples")

    measure = block_measure(traj)
    rate = block_rate(traj)
    sampling = block_sampling()
    budget = block_budget(traj)
    bistable = block_bistable()
    ramp = block_ramp(traj)

    print("# Generated by scripts/generate_ch30_data.py -- do not edit by hand.")
    print("# Chapter 30: ergodic theory and invariant measures.")

    # -- constants --------------------------------------------------------
    print("# --- the reference system ---")
    _scalar("DT", DT, ".2f")
    _scalar("LONG_RUN", LONG, ".0f")
    _scalar("DAYS_PER_TU", DAYS_PER_TU, ".1f")
    _scalar("RHO", RHO, ".1f")
    _scalar("BETA", BETA, ".6f")

    # -- 1. the measure ---------------------------------------------------
    print("# --- 1. one long run against a scattered ensemble ---")
    _scalar("ENSEMBLE_MEMBERS", ENSEMBLE_MEMBERS, ".0f")
    for key in ("w1_x", "w1_z", "spread_x", "spread_z", "mean_long_x",
                "mean_snap_x", "mean_long_z", "mean_snap_z", "w1_gaussian",
                "tv_z", "d_ky", "trace"):
        _scalar(key.upper(), measure[key], ".5f")
    _emit("SPECTRUM", measure["spectrum"], ".5f")
    _scalar("SPECTRUM_SUM", measure["spectrum"].sum(), ".5f")
    _emit("EDGES_Z", measure["edges_z"], ".3f", per_line=10)
    _emit("HIST_LONG_Z", measure["hist_long_z"], ".5f")
    _emit("HIST_SNAP_Z", measure["hist_snap_z"], ".5f")
    _emit("BOX_SIDES", measure["box_sides"], ".0f", per_line=10)
    _emit("BOX_OCCUPIED", measure["box_occupied"], ".0f", per_line=10)
    _emit("BOX_USABLE", measure["box_usable"], ".0f", per_line=10)
    _emit("CONTROL_OCCUPIED", measure["control_occupied"], ".0f", per_line=10)
    _emit("CONTROL_USABLE", measure["control_usable"], ".0f", per_line=10)
    _scalar("BOX_DIMENSION", measure["box_dimension"], ".4f")
    _scalar("BOX_R2", measure["box_r2"], ".5f")
    _scalar("CONTROL_DIMENSION", measure["control_dimension"], ".4f")
    _scalar("CONTROL_R2", measure["control_r2"], ".5f")
    _scalar("BOX_POINTS", measure["box_points"], ".0f")
    _scalar("MIN_PER_BOX", MIN_PER_BOX, ".0f")
    _emit("REFERENCE_X", measure["reference_xz"][:, 0], ".2f", per_line=12)
    _emit("REFERENCE_Z", measure["reference_xz"][:, 1], ".2f", per_line=12)
    _emit("SNAPSHOT_X", measure["snapshot"][:600, 0], ".3f", per_line=10)
    _emit("SNAPSHOT_Z", measure["snapshot"][:600, 2], ".3f", per_line=10)

    # -- 2. the rate ------------------------------------------------------
    print("# --- 2. the rate at which a time average converges ---")
    for name in ("x", "z", "x2"):
        upper = name.upper()
        _emit(f"BATCH_WINDOW_{upper}", rate[f"batch_window_{name}"], ".3f")
        _emit(f"BATCH_VAR_{upper}", rate[f"batch_var_{name}"], ".6e")
        _emit(f"BATCH_TAU_{upper}", rate[f"batch_tau_{name}"], ".6f")
        _emit(f"BATCH_BLOCKS_{upper}", rate[f"batch_blocks_{name}"], ".0f", per_line=11)
        _scalar(f"TAU_INT_{upper}", rate[f"tau_int_{name}"], ".5f")
        _scalar(f"TAU_PLATEAU_{upper}", rate[f"tau_plateau_{name}"], ".5f")
        _scalar(f"VAR_{upper}", rate[f"var_{name}"], ".4f")
        _scalar(f"MEAN_{upper}", rate[f"mean_{name}"], ".5f")
        _scalar(f"LAW_EXPONENT_{upper}", rate[f"law_exponent_{name}"], ".4f")
        _scalar(f"NEED_{upper}", rate[f"need_{name}"], ".0f")
        _scalar(f"NEED_NAIVE_{upper}", rate[f"need_naive_{name}"], ".0f")
        _emit(f"ACF_{upper}", rate[f"acf_{name}"], ".4f", per_line=10)
    _emit("ACF_LAG", rate["acf_lag"], ".3f", per_line=10)
    _emit("TARGET_SE", rate["target"], ".2f")
    _emit("RUNNING_TIME", rate["running_time"], ".1f", per_line=10)
    _emit("RUNNING_X", rate["running_x"], ".4f", per_line=10)
    _emit("RUNNING_Z", rate["running_z"], ".4f", per_line=10)
    _emit("IDENTITY_SPAN", rate["identity_span"], ".0f")
    _emit("IDENTITY_FINITE_X2", rate["identity_finite_x2"], ".3e")
    _emit("IDENTITY_FINITE_Z", rate["identity_finite_z"], ".3e")
    _emit("IDENTITY_LIMIT_X2", rate["identity_limit_x2"], ".5f")
    _emit("IDENTITY_LIMIT_Z", rate["identity_limit_z"], ".5f")
    _emit("IDENTITY_MEAN_X", rate["identity_mean_x"], ".5f")
    _emit("IDENTITY_MEAN_Z", rate["identity_mean_z"], ".5f")
    _emit("IDENTITY_MEAN_X2B", rate["identity_mean_x2b"], ".5f")

    # -- 3. ergodic is not chaotic ----------------------------------------
    print("# --- 3. a rotation, a chaotic map, and a random sample ---")
    _emit("SAMPLE_SIZES", sampling["sizes"], ".0f", per_line=10)
    for label in ("rotation", "logistic", "random"):
        _emit(f"ERR_{label.upper()}", sampling[f"err_{label}"], ".4e")
        _emit(f"DISC_{label.upper()}", sampling[f"disc_{label}"], ".4e")
        _scalar(f"EXP_{label.upper()}", sampling[f"exp_{label}"], ".4f")
        _scalar(f"EXP_DISC_{label.upper()}", sampling[f"exp_disc_{label}"], ".4f")
    _scalar("W1_LOGISTIC", sampling["w1_logistic"], ".5f")
    _scalar("MEAN_LOGISTIC", sampling["mean_logistic"], ".5f")
    _scalar("VAR_LOGISTIC", sampling["var_logistic"], ".5f")
    _scalar("REALISATIONS", REALISATIONS, ".0f")
    _emit("ARCSINE_GRID", sampling["arcsine_grid"][::4], ".4f", per_line=10)
    _emit("ARCSINE_DENSITY", sampling["arcsine_density"][::4], ".4f", per_line=10)
    _emit("ARCSINE_EDGES", sampling["arcsine_edges"], ".4f", per_line=10)
    _emit("ARCSINE_MEASURED", sampling["arcsine_measured"], ".4f", per_line=10)

    # -- 4. the budget ----------------------------------------------------
    print("# --- 4. one long run or many short ones ---")
    _scalar("BUDGET", budget["budget"], ".0f")
    _scalar("SPIN_COST", budget["spin_cost"], ".0f")
    _scalar("BUDGET_REPS", BUDGET_REPS, ".0f")
    _emit("BUDGET_MEMBERS", budget["members"], ".0f", per_line=10)
    _emit("BUDGET_VAR", budget["budget_var"], ".6f")
    _emit("BUDGET_RATIO", budget["budget_ratio"], ".4f")
    _emit("BUDGET_PREDICTED", budget["budget_predicted"], ".4f")
    _emit("BUDGET_EXCESS", budget["budget_excess"], ".4f")
    _emit("BUDGET_SAMPLED", budget["budget_sampled"], ".0f", per_line=10)
    _scalar("TAU_LONG_SEGMENT", budget["tau_long_segment"], ".4f")
    _scalar("TAU_SHORT_SEGMENT", budget["tau_short_segment"], ".4f")

    # -- 5. the bistable case ---------------------------------------------
    print("# --- 5. when the ergodic time exceeds the record ---")
    for key in ("barrier", "noise", "tau_kramers", "exponent", "grand_mean",
                "occupancy", "typical", "tau_erg", "tau_within", "restoring"):
        _scalar(key.upper(), bistable[key], ".5f")
    _scalar("WELL_MEMBERS", WELL_MEMBERS, ".0f")
    _scalar("WELL_TOTAL", WELL_TOTAL, ".0f")
    _emit("WELL_GRID", bistable["well_grid"][::4], ".4f", per_line=10)
    _emit("WELL_EXACT", bistable["well_exact"][::4], ".5f", per_line=10)
    _emit("WELL_EDGES", bistable["well_edges"], ".4f", per_line=10)
    _emit("WELL_MEASURED", bistable["well_measured"], ".5f", per_line=10)
    _emit("WELL_SHORT", bistable["well_short"], ".5f", per_line=10)
    _emit("WELL_WINDOWS", bistable["well_windows"], ".0f")
    _emit("WELL_RMS", bistable["well_rms"], ".5f")
    _emit("WELL_NAIVE", bistable["well_naive"], ".5f")
    _emit("WELL_BIAS", bistable["well_bias"], ".5f")
    _emit("WELL_OVERCONFIDENCE", bistable["well_overconfidence"], ".3f")
    _emit("WELL_TAU_IMPLIED", bistable["well_tau_implied"], ".2f")
    _emit("NOISE_LADDER", bistable["noise_ladder"], ".3f")
    _emit("TAU_LADDER", bistable["tau_ladder"], ".4e")

    # -- 6. the ramp ------------------------------------------------------
    print("# --- 6. a forcing that moves ---")
    for key in ("rho_start", "rho_end", "ramp_rate", "drift", "ramp_total",
                "ens_early", "ens_late", "fit_early", "fit_late", "ens_stderr",
                "ens_spread", "ramp_bias_at_400"):
        _scalar(f"RAMP_{key.upper()}" if not key.startswith("ramp") else key.upper(),
                ramp[key], ".5f")
    _scalar("RAMP_MEMBERS", ramp["members"], ".0f")
    _emit("RAMP_WINDOWS", ramp["ramp_windows"], ".0f")
    _emit("RAMP_LAG", ramp["ramp_lag"], ".5f")
    _emit("RAMP_PREDICTED", ramp["ramp_predicted"], ".5f")
    _emit("RAMP_TIME", ramp["ramp_time"], ".1f", per_line=10)
    _emit("RAMP_RHO_CURVE", ramp["ramp_rho_curve"], ".3f", per_line=10)
    _emit("RAMP_ENS_MEAN", ramp["ramp_ens_mean"], ".4f", per_line=10)
    _emit("RAMP_SINGLE", ramp["ramp_single"], ".4f", per_line=10)
    _emit("RAMP_TRAIL_100", ramp["ramp_trail_100"], ".4f", per_line=10)
    _emit("RAMP_TRAIL_400", ramp["ramp_trail_400"], ".4f", per_line=10)

    _log(f"done in {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
