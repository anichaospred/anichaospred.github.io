#!/usr/bin/env python3
r"""Precompute chapter 31's Koopman experiments.

A nonlinear system is *exactly* a linear operator acting on observables. The
trade is dimension, and this chapter measures its exchange rate.

Five blocks.

1. **The exact case** -- the slow-manifold system, whose Koopman operator
   closes on three observables. The closed-form solution, the exact generator
   and its spectrum, and what happens to the EDMD residual when the dictionary
   is one observable short.
2. **EDMD on an attractor** -- dictionary size against one-step accuracy on
   Lorenz 63, in and out of sample, with the spectral radius.
3. **The trade** -- rollout error against lead time for each dictionary size,
   and the lead at which each crosses a skill threshold. The exchange rate of
   dimension for forecast range.
4. **The mechanism** -- the linear rollout against re-lifting, and the
   off-manifold residual that separates them. Linearity or accuracy.
5. **The spectrum** -- the autocorrelation function the operator predicts
   against the measured one, and the eigenvalue at exactly one.

Run from chaos-book/:
    python3 scripts/generate_ch31_data.py        # ~8 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import (  # noqa: E402
    ergodic,
    integrate,
    koopman,
    lyapunov,
    systems,
)

# --- the exact case ------------------------------------------------------
MU, LAM = -0.05, -1.0
SLOW_DT, SLOW_TOTAL = 0.01, 40.0
MONOMIAL_ORDERS = (1, 2, 3, 4)

# --- Lorenz 63 -----------------------------------------------------------
DT = 0.01
TAU = 0.05                  # the snapshot interval the operator advances
TOTAL = 3000.0
SPINUP = 50.0
N_TRAIN = 25000             # snapshot pairs used to fit
N_TEST = 8000
DICTIONARY_SIZES = (0, 10, 30, 100, 300, 1000)
RBF_WIDTH = 0.7
REFERENCE_SIZE = 300        # the dictionary used for the mechanism block
LEAD_STEPS = (1, 2, 5, 10, 20, 40, 80, 160, 320)
N_LAUNCH = 214              # forecast start points, spaced through the test set
THRESHOLD = 0.5             # of the climatological error: the useful horizon
TIGHT = 0.2
DAYS_PER_TU = 5.0
CORRELATION_STEPS = 120


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
# 1. the exact case
# =========================================================================
def block_exact() -> dict:
    _log("block 1: a system whose Koopman operator closes")
    out: dict = {}
    t = integrate.trajectory_grid(t_final=SLOW_TOTAL, dt=SLOW_DT)
    x0 = np.array([1.2, 0.7])
    numeric = integrate.rk4(systems.slow_manifold, x0, t, mu=MU, lam=LAM)
    exact = systems.slow_manifold_solution(t, x0, MU, LAM)
    out["solution_error"] = float(np.abs(numeric - exact).max())

    matrix = systems.slow_manifold_koopman_matrix(MU, LAM)
    out["generator"] = matrix
    out["generator_eigs"] = np.sort(np.linalg.eigvals(matrix).real)
    out["generator_exact"] = np.sort(np.array([MU, LAM, 2.0 * MU]))
    _log(f"  closed form vs RK4: {out['solution_error']:.2e}; "
         f"generator spectrum {np.round(out['generator_eigs'], 6)} against "
         f"exact {np.round(out['generator_exact'], 6)}")

    def lift(state):
        state = np.asarray(state, dtype=float)
        return np.stack(
            [state[..., 0], state[..., 1], state[..., 0] ** 2], axis=-1
        )

    closing = lift(numeric)
    fit = koopman.edmd(closing[:-1], closing[1:])
    truncated = koopman.edmd(numeric[:-1], numeric[1:])
    out["closing_residual"] = fit["residual"]
    out["truncated_residual"] = truncated["residual"]
    out["fitted_rates"] = np.sort(
        koopman.continuous_eigenvalues(fit["eigenvalues"], SLOW_DT).real
    )
    _log(f"  EDMD residual with x1^2 {fit['residual']:.2e}, without "
         f"{truncated['residual']:.2e}; fitted rates "
         f"{np.round(out['fitted_rates'], 6)}")

    # the whole trajectory, from the linear operator alone
    steps = numeric.shape[0] - 1
    rolled = koopman.linear_rollout(fit["operator"], closing[:1], steps)
    out["rollout_error"] = float(np.abs(rolled[:, 0, :2] - numeric).max())
    out["off_manifold"] = float(
        koopman.off_manifold_residual(rolled[-1], lift, slice(0, 2)).max()
    )
    _log(f"  linear rollout over {SLOW_TOTAL:.0f} TU: max error "
         f"{out['rollout_error']:.2e}, off-manifold {out['off_manifold']:.2e}")

    # Monomial dictionaries: the residual falls with order but never reaches
    # machine precision, because each order leaks into the next -- the
    # hierarchy never closes, it only leaks less.
    orders, residuals, sizes = [], [], []
    for order in MONOMIAL_ORDERS:
        gx = koopman.monomial_features(numeric[:-1], order)
        gy = koopman.monomial_features(numeric[1:], order)
        result = koopman.edmd(gx, gy)
        orders.append(float(order))
        residuals.append(result["residual"])
        sizes.append(result["size"])
        _log(f"  monomials to order {order}: {int(result['size']):2d} terms, "
             f"residual {result['residual']:.3e}")
    out["monomial_order"] = np.asarray(orders)
    out["monomial_residual"] = np.asarray(residuals)
    out["monomial_size"] = np.asarray(sizes)

    # Closure is a property of the SPAN, not of the size: adding an observable
    # can destroy it. d/dt(x1 x2) = (mu+lam) x1 x2 - lam x1^3, so admitting
    # x1*x2 opens a leak into degree three that the dictionary cannot absorb,
    # and admitting x1^3 as well closes it again.
    ladder = (
        ("x1, x2", lambda s: [s[:, 0], s[:, 1]]),
        ("x1, x2, x1^2", lambda s: [s[:, 0], s[:, 1], s[:, 0] ** 2]),
        ("1, x1, x2, x1^2", lambda s: [np.ones(len(s)), s[:, 0], s[:, 1],
                                       s[:, 0] ** 2]),
        ("1, x1, x2, x1^2, x1 x2", lambda s: [np.ones(len(s)), s[:, 0], s[:, 1],
                                              s[:, 0] ** 2, s[:, 0] * s[:, 1]]),
        ("1, x1, x2, x1^2, x1 x2, x1^3",
         lambda s: [np.ones(len(s)), s[:, 0], s[:, 1], s[:, 0] ** 2,
                    s[:, 0] * s[:, 1], s[:, 0] ** 3]),
    )
    ladder_size, ladder_residual = [], []
    for label, builder in ladder:
        g = np.stack(builder(numeric), axis=-1)
        result = koopman.edmd(g[:-1], g[1:])
        ladder_size.append(result["size"])
        ladder_residual.append(result["residual"])
        _log(f"  [{label:28s}] {int(result['size'])} terms, residual "
             f"{result['residual']:.3e}")
    out["ladder_size"] = np.asarray(ladder_size)
    out["ladder_residual"] = np.asarray(ladder_residual)

    # the leak, verified against its closed form rather than asserted
    x1, x2 = numeric[:, 0], numeric[:, 1]
    derivative = np.gradient(x1 * x2, SLOW_DT)
    closed_form = (MU + LAM) * (x1 * x2) - LAM * x1**3
    out["leak_residual"] = float(np.abs(derivative - closed_form)[5:-5].max())
    _log(f"  d/dt(x1 x2) against (mu+lam)x1x2 - lam x1^3: "
         f"{out['leak_residual']:.2e}")

    # the trajectory and the parabola it collapses onto, for the figure
    out["slow_time"] = t[::20]
    out["slow_traj"] = numeric[::20]
    ensemble = integrate.rk4(
        systems.slow_manifold,
        np.stack([np.linspace(-1.6, 1.6, 9), np.full(9, 2.6)], axis=-1),
        t[: int(12.0 / SLOW_DT)], mu=MU, lam=LAM,
    )
    out["slow_ensemble"] = ensemble[::20]
    return out


# =========================================================================
# the shared Lorenz 63 data
# =========================================================================
def lorenz_snapshots() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = integrate.trajectory_grid(t_final=TOTAL, dt=DT)
    traj = integrate.rk4(systems.lorenz63, np.array([1.0, 1.0, 20.0]), t)
    traj = traj[int(SPINUP / DT) :]
    snapshots = traj[:: int(round(TAU / DT))]
    whitened = koopman.standardise(snapshots)
    return traj, snapshots, whitened


def fit_dictionary(whitened: np.ndarray, size: int) -> dict:
    centres = koopman.pick_centres(whitened[:N_TRAIN], size, seed=11)

    def lift(state):
        return koopman.rbf_features(state, centres, RBF_WIDTH)

    gx = lift(whitened[:N_TRAIN])
    gy = lift(whitened[1 : N_TRAIN + 1])
    fit = koopman.edmd(gx, gy)
    test_x = whitened[N_TRAIN : N_TRAIN + N_TEST]
    test_y = whitened[N_TRAIN + 1 : N_TRAIN + N_TEST + 1]
    fit["test_residual"] = koopman.closure_residual(
        fit["operator"], lift(test_x), lift(test_y)
    )
    fit["radius"] = koopman.spectral_radius(fit["operator"])
    fit["lift"] = lift
    fit["centres"] = centres
    return fit


# =========================================================================
# 2 and 3. the dictionary sweep and the trade
# =========================================================================
def block_sweep(whitened: np.ndarray) -> dict:
    _log("blocks 2-3: dictionary size, accuracy and lead time")
    out: dict = {}
    climate_variance = float(whitened.var(axis=0).sum())
    starts = np.linspace(
        N_TRAIN + 100, N_TRAIN + N_TEST - max(LEAD_STEPS) - 2, N_LAUNCH
    ).astype(int)
    out["climate_variance"] = climate_variance
    out["starts"] = float(starts.size)

    sizes, train, test, radius, ranks = [], [], [], [], []
    curves, horizons, tight = [], [], []
    for size in DICTIONARY_SIZES:
        tic = time.time()
        fit = fit_dictionary(whitened, size)
        sizes.append(fit["size"])
        train.append(fit["residual"])
        test.append(fit["test_residual"])
        radius.append(fit["radius"])
        ranks.append(fit["rank"])

        g = fit["lift"](whitened[starts])
        errors, step = [], 0
        for lead in LEAD_STEPS:
            while step < lead:
                g = g @ fit["operator"]
                step += 1
            truth = whitened[starts + lead]
            errors.append(
                float(np.sqrt(((g[:, 1:4] - truth) ** 2).sum(axis=1).mean()
                              / climate_variance))
            )
        curves.append(errors)
        horizons.append(_crossing(np.asarray(LEAD_STEPS) * TAU, errors, THRESHOLD))
        tight.append(_crossing(np.asarray(LEAD_STEPS) * TAU, errors, TIGHT))
        _log(f"  {int(fit['size']):5d} observables: train {fit['residual']:.3e}, "
             f"test {fit['test_residual']:.3e}, radius {fit['radius']:.6f}, "
             f"horizon {horizons[-1]:.3f} TU ({time.time() - tic:.0f} s)")

    out["sizes"] = np.asarray(sizes)
    out["train_residual"] = np.asarray(train)
    out["test_residual"] = np.asarray(test)
    out["radius"] = np.asarray(radius)
    out["rank"] = np.asarray(ranks)
    out["leads"] = np.asarray(LEAD_STEPS, dtype=float) * TAU
    out["curves"] = np.asarray(curves)
    out["horizon"] = np.asarray(horizons)
    out["horizon_tight"] = np.asarray(tight)
    _log(f"  horizons at {THRESHOLD:.1f}: {np.round(out['horizon'], 3)}")
    _log(f"  dimension x{out['sizes'][-1] / out['sizes'][0]:.0f} buys lead "
         f"x{out['horizon'][-1] / out['horizon'][0]:.1f}")
    return out


def _crossing(leads: np.ndarray, errors, level: float) -> float:
    """Lead at which the error curve first reaches ``level``, interpolated in
    the logarithm of the error -- the curve is exponential, so a linear
    interpolation systematically overestimates the crossing."""
    e = np.asarray(errors, dtype=float)
    above = np.nonzero(e >= level)[0]
    if above.size == 0:
        return float(leads[-1])
    i = int(above[0])
    if i == 0:
        return float(leads[0])
    lo, hi = np.log(e[i - 1]), np.log(e[i])
    if hi <= lo:
        return float(leads[i])
    weight = (np.log(level) - lo) / (hi - lo)
    return float(leads[i - 1] + weight * (leads[i] - leads[i - 1]))


# =========================================================================
# 4. the mechanism
# =========================================================================
def block_mechanism(whitened: np.ndarray, lam1: float) -> dict:
    _log("block 4: the linear rollout against re-lifting")
    out: dict = {}
    fit = fit_dictionary(whitened, REFERENCE_SIZE)
    lift = fit["lift"]
    climate_variance = float(whitened.var(axis=0).sum())
    starts = np.linspace(
        N_TRAIN + 100, N_TRAIN + N_TEST - max(LEAD_STEPS) - 2, N_LAUNCH
    ).astype(int)

    g = lift(whitened[starts])
    x = whitened[starts].copy()
    linear, relift, drift, step = [], [], [], 0
    for lead in LEAD_STEPS:
        while step < lead:
            g = g @ fit["operator"]
            x = (lift(x) @ fit["operator"])[:, 1:4]
            step += 1
        truth = whitened[starts + lead]
        linear.append(
            float(np.sqrt(((g[:, 1:4] - truth) ** 2).sum(axis=1).mean()
                          / climate_variance))
        )
        relift.append(
            float(np.sqrt(((x - truth) ** 2).sum(axis=1).mean()
                          / climate_variance))
        )
        drift.append(float(koopman.off_manifold_residual(g, lift, slice(1, 4)).mean()))
        _log(f"  lead {lead * TAU:6.2f} TU: linear {linear[-1]:.4f}, "
             f"re-lifted {relift[-1]:.4f}, off-manifold {drift[-1]:.4f}")

    out["leads"] = np.asarray(LEAD_STEPS, dtype=float) * TAU
    out["linear"] = np.asarray(linear)
    out["relift"] = np.asarray(relift)
    out["drift"] = np.asarray(drift)
    out["size"] = fit["size"]
    out["horizon_linear"] = _crossing(out["leads"], linear, THRESHOLD)
    out["horizon_relift"] = _crossing(out["leads"], relift, THRESHOLD)

    # the re-lifted error grows exponentially; at what rate, against lambda_1?
    window = (out["leads"] >= 0.25) & (out["relift"] < 0.5)
    if window.sum() >= 2:
        slope = float(np.polyfit(out["leads"][window],
                                 np.log(out["relift"][window]), 1)[0])
    else:
        slope = float("nan")
    out["relift_growth"] = slope
    out["lambda1"] = lam1
    _log(f"  re-lifted error grows at {slope:.3f} per TU against "
         f"lambda_1 = {lam1:.3f}; horizons {out['horizon_linear']:.3f} "
         f"(linear) and {out['horizon_relift']:.3f} (re-lifted)")

    # does the re-lifted rollout stay on the attractor? the linear one must,
    # because its spectral radius is one
    long_steps = 2000
    far_linear = koopman.linear_rollout(
        fit["operator"], lift(whitened[starts[:40]]), long_steps
    )[-1][:, 1:4]
    far_relift = koopman.relift_rollout(
        fit["operator"], whitened[starts[:40]], long_steps, lift, slice(1, 4)
    )[-1]
    spread = whitened.std(axis=0)
    out["far_linear_spread"] = float(np.abs(far_linear).max() / spread.max())
    out["far_relift_spread"] = float(np.abs(far_relift).max() / spread.max())
    out["far_steps"] = float(long_steps)
    _log(f"  after {long_steps * TAU:.0f} TU the linear rollout reaches "
         f"{out['far_linear_spread']:.2f} standard deviations and the "
         f"re-lifted one {out['far_relift_spread']:.2f}")
    return out


# =========================================================================
# 5. the spectrum
# =========================================================================
def block_spectrum(whitened: np.ndarray, traj: np.ndarray) -> dict:
    _log("block 5: what the spectrum does say")
    out: dict = {}
    fit = fit_dictionary(whitened, REFERENCE_SIZE)
    lift = fit["lift"]

    magnitudes = np.sort(np.abs(fit["eigenvalues"]))[::-1]
    out["eig_magnitude"] = magnitudes[:12]
    out["radius"] = fit["radius"]
    rates = koopman.continuous_eigenvalues(fit["eigenvalues"], TAU)
    order = np.argsort(-rates.real)
    out["rate_real"] = rates.real[order][:12]
    out["rate_imag"] = np.abs(rates.imag[order][:12])
    out["eig_real"] = np.real(fit["eigenvalues"])
    out["eig_imag"] = np.imag(fit["eigenvalues"])
    _log(f"  leading |eigenvalue| {magnitudes[0]:.8f} (exact 1), next "
         f"{magnitudes[1]:.6f}; leading rates {np.round(out['rate_real'][:4], 4)}")

    # the correlation function: measured against the operator's prediction
    sample = whitened[:N_TRAIN]
    features = lift(sample)
    predicted = koopman.operator_correlation(
        fit["operator"], features, 1, CORRELATION_STEPS
    )
    reference = sample[:, 0]
    measured = np.array([
        float(np.mean(reference * whitened[n : N_TRAIN + n, 0])
              / np.mean(reference**2))
        for n in range(CORRELATION_STEPS + 1)
    ])
    out["correlation_lag"] = np.arange(CORRELATION_STEPS + 1) * TAU
    out["correlation_measured"] = measured
    out["correlation_predicted"] = predicted
    worst = float(np.abs(measured - predicted)[: int(2.0 / TAU) + 1].max())
    out["correlation_error_2tu"] = worst
    _log(f"  correlation function: worst error over the first 2 TU {worst:.4f}")

    # chapter 30's integrated autocorrelation time, from each
    out["tau_measured"] = ergodic.autocorrelation_time(traj[:, 0], DT)
    out["tau_operator"] = TAU * (
        1.0 + 2.0 * np.sum(predicted[1:][np.cumprod(predicted[1:] > 0.0, dtype=bool)])
    )
    _log(f"  tau_int from the trajectory {out['tau_measured']:.4f} TU, from the "
         f"operator's correlation function {out['tau_operator']:.4f} TU")
    return out


# =========================================================================
# main
# =========================================================================
def main() -> None:
    started = time.time()
    _log("chapter 31 -- the Koopman operator")
    exact = block_exact()

    tic = time.time()
    traj, snapshots, whitened = lorenz_snapshots()
    _log(f"Lorenz 63: {snapshots.shape[0]} snapshots {TAU} TU apart in "
         f"{time.time() - tic:.0f} s")
    spectrum_l63 = lyapunov.lyapunov_spectrum(
        systems.lorenz63, systems.lorenz63_jacobian, traj[-1], t_final=1500.0, dt=DT
    )
    lam1 = float(spectrum_l63[0])
    _log(f"  lambda_1 = {lam1:.4f}, spectrum sum {spectrum_l63.sum():.5f} "
         f"(exact {-(10.0 + 1.0 + 8.0 / 3.0):.5f})")

    sweep = block_sweep(whitened)
    mechanism = block_mechanism(whitened, lam1)
    spectrum = block_spectrum(whitened, traj)

    print("# Generated by scripts/generate_ch31_data.py -- do not edit by hand.")
    print("# Chapter 31: the Koopman operator.")

    print("# --- constants ---")
    _scalar("DT", DT, ".2f")
    _scalar("TAU", TAU, ".3f")
    _scalar("DAYS_PER_TU", DAYS_PER_TU, ".1f")
    _scalar("N_TRAIN", N_TRAIN, ".0f")
    _scalar("N_TEST", N_TEST, ".0f")
    _scalar("RBF_WIDTH", RBF_WIDTH, ".2f")
    _scalar("THRESHOLD", THRESHOLD, ".2f")
    _scalar("TIGHT", TIGHT, ".2f")
    _scalar("LAMBDA1", lam1, ".5f")
    _scalar("MU", MU, ".3f")
    _scalar("LAM", LAM, ".3f")
    _scalar("SLOW_TOTAL", SLOW_TOTAL, ".0f")

    print("# --- 1. a system whose Koopman operator closes ---")
    for key in ("solution_error", "closing_residual", "truncated_residual",
                "rollout_error", "off_manifold"):
        _scalar(key.upper(), exact[key], ".4e")
    _emit("GENERATOR", exact["generator"], ".6f", per_line=3)
    _emit("GENERATOR_EIGS", exact["generator_eigs"], ".8f")
    _emit("GENERATOR_EXACT", exact["generator_exact"], ".8f")
    _emit("FITTED_RATES", exact["fitted_rates"], ".8f")
    _emit("MONOMIAL_ORDER", exact["monomial_order"], ".0f")
    _emit("MONOMIAL_SIZE", exact["monomial_size"], ".0f")
    _emit("MONOMIAL_RESIDUAL", exact["monomial_residual"], ".4e")
    _emit("LADDER_SIZE", exact["ladder_size"], ".0f", per_line=10)
    _emit("LADDER_RESIDUAL", exact["ladder_residual"], ".4e")
    _scalar("LEAK_RESIDUAL", exact["leak_residual"], ".4e")
    _emit("SLOW_TIME", exact["slow_time"], ".2f", per_line=10)
    _emit("SLOW_X1", exact["slow_traj"][:, 0], ".4f", per_line=10)
    _emit("SLOW_X2", exact["slow_traj"][:, 1], ".4f", per_line=10)
    _emit("SLOW_ENS_X1", exact["slow_ensemble"][:, :, 0].T, ".4f", per_line=10)
    _emit("SLOW_ENS_X2", exact["slow_ensemble"][:, :, 1].T, ".4f", per_line=10)
    _scalar("SLOW_ENS_MEMBERS", exact["slow_ensemble"].shape[1], ".0f")

    print("# --- 2 and 3. the dictionary sweep and the trade ---")
    _emit("SIZES", sweep["sizes"], ".0f", per_line=10)
    _emit("TRAIN_RESIDUAL", sweep["train_residual"], ".4e")
    _emit("TEST_RESIDUAL", sweep["test_residual"], ".4e")
    _emit("RADIUS", sweep["radius"], ".8f")
    _emit("RANK", sweep["rank"], ".0f", per_line=10)
    _emit("LEADS", sweep["leads"], ".3f", per_line=10)
    _emit("CURVES", sweep["curves"], ".5f")
    _emit("HORIZON", sweep["horizon"], ".4f")
    _emit("HORIZON_TIGHT", sweep["horizon_tight"], ".4f")
    _scalar("CLIMATE_VARIANCE", sweep["climate_variance"], ".4f")
    _scalar("N_LAUNCH", sweep["starts"], ".0f")

    print("# --- 4. the mechanism ---")
    _emit("MECH_LEADS", mechanism["leads"], ".3f", per_line=10)
    _emit("MECH_LINEAR", mechanism["linear"], ".5f")
    _emit("MECH_RELIFT", mechanism["relift"], ".5f")
    _emit("MECH_DRIFT", mechanism["drift"], ".5f")
    for key in ("size", "horizon_linear", "horizon_relift", "relift_growth",
                "far_linear_spread", "far_relift_spread", "far_steps"):
        _scalar(f"MECH_{key.upper()}", mechanism[key], ".5f")

    print("# --- 5. the spectrum ---")
    _emit("EIG_MAGNITUDE", spectrum["eig_magnitude"], ".8f")
    _emit("EIG_REAL", spectrum["eig_real"], ".5f", per_line=10)
    _emit("EIG_IMAG", spectrum["eig_imag"], ".5f", per_line=10)
    _emit("RATE_REAL", spectrum["rate_real"], ".5f")
    _emit("RATE_IMAG", spectrum["rate_imag"], ".5f")
    _emit("CORRELATION_LAG", spectrum["correlation_lag"], ".3f", per_line=10)
    _emit("CORRELATION_MEASURED", spectrum["correlation_measured"], ".5f",
          per_line=10)
    _emit("CORRELATION_PREDICTED", spectrum["correlation_predicted"], ".5f",
          per_line=10)
    _scalar("CORRELATION_ERROR_2TU", spectrum["correlation_error_2tu"], ".5f")
    _scalar("SPECTRUM_RADIUS", spectrum["radius"], ".8f")
    _scalar("TAU_MEASURED", spectrum["tau_measured"], ".5f")
    _scalar("TAU_OPERATOR", spectrum["tau_operator"], ".5f")

    _log(f"done in {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
