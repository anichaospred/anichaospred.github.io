#!/usr/bin/env python3
r"""Precompute chapter 29's emulator experiments.

Does a model *fitted* to a chaotic system inherit its dynamics, or only its
short-term forecasts? Chapter 3 supplies the diagnostics that make the question
answerable -- the Lyapunov spectrum, the unstable dimension, the horizon law --
so this chapter is a comparison of censuses rather than an argument.

The emulator is a reservoir computer: a fixed random recurrent network driven
by Lorenz 96, with a linear readout fitted by ridge regression. Training is one
linear solve, which is what makes it runnable in a browser, and its tangent map
is analytic, which is what makes its Lyapunov spectrum computable.

Four blocks.

1. **The truth's census** -- what the emulator has to reproduce.
2. **The emulator's census**, for a well-chosen configuration: one-step error,
   rollout stability, climatology, and the leading eight exponents.
3. **A configuration sweep**, containing the real failure modes: a spectral
   radius above one, an input scaling that leaves the climatology perfect and
   the dynamics badly wrong, and a training set short enough that the *lowest*
   one-step error comes with among the worst rollouts.
4. **What the checkable diagnostics tell you** about the ones you cannot check.

Run from chaos-book/:
    python3 scripts/generate_ch29_data.py        # ~6 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import errorgrowth, integrate, learning, lyapunov, systems  # noqa: E402

SITES, FORCING, DT = 8, 8.0, 0.01
TRAIN, WASH, ROLL = 60000, 2000, 20000
N_EXPONENTS = 8
SPECTRUM_STEPS = 30000
TRACK_CASES = 8

CONFIGS = (
    ("baseline", dict(n_reservoir=1000, spectral_radius=0.4,
                      input_scaling=0.15, ridge=1e-5, n_train=TRAIN)),
    ("larger reservoir", dict(n_reservoir=2000, spectral_radius=0.4,
                              input_scaling=0.15, ridge=1e-5, n_train=TRAIN)),
    ("small reservoir", dict(n_reservoir=150, spectral_radius=0.4,
                             input_scaling=0.15, ridge=1e-5, n_train=TRAIN)),
    ("radius 0.9", dict(n_reservoir=1000, spectral_radius=0.9,
                        input_scaling=0.15, ridge=1e-5, n_train=TRAIN)),
    ("radius 1.4", dict(n_reservoir=1000, spectral_radius=1.4,
                        input_scaling=0.15, ridge=1e-5, n_train=TRAIN)),
    ("input scaling 1.5", dict(n_reservoir=1000, spectral_radius=0.4,
                               input_scaling=1.5, ridge=1e-5, n_train=TRAIN)),
    ("barely regularised", dict(n_reservoir=1000, spectral_radius=0.4,
                                input_scaling=0.15, ridge=1e-12, n_train=TRAIN)),
    ("4,000 steps", dict(n_reservoir=1000, spectral_radius=0.4,
                         input_scaling=0.15, ridge=1e-5, n_train=4000)),
    ("12,000 steps", dict(n_reservoir=1000, spectral_radius=0.4,
                          input_scaling=0.15, ridge=1e-5, n_train=12000)),
)


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


def _truth():
    rng = np.random.default_rng(0)
    start = systems.lorenz96_uniform_state(FORCING, SITES) + rng.normal(
        0.0, 0.5, SITES
    )
    spun = integrate.rk4(
        systems.lorenz96, start, integrate.trajectory_grid(200.0, 0.01),
        forcing=FORCING,
    )
    return integrate.rk4(
        systems.lorenz96, spun[-1],
        integrate.trajectory_grid((TRAIN + WASH + ROLL + 1) * DT, DT),
        forcing=FORCING,
    )


def _train(data, scaling, n_reservoir, spectral_radius, input_scaling,
           ridge, n_train, seed=1):
    network = learning.echo_state_network(
        SITES, n_reservoir=n_reservoir, spectral_radius=spectral_radius,
        input_scaling=input_scaling, sparsity=0.03, seed=seed,
    )
    inputs = data[: n_train + WASH]
    states = learning.reservoir_drive(network, inputs, scaling)
    targets = data[WASH + 1 : n_train + WASH + 1] - data[WASH : n_train + WASH]
    readout = learning.fit_readout(
        states[WASH:], data[WASH : n_train + WASH], targets, scaling, ridge=ridge
    )
    features = learning.readout_features(
        states[WASH:], data[WASH : n_train + WASH], scaling
    )
    relative = float(
        np.sqrt(((features @ readout - targets) ** 2).mean()) / targets.std()
    )
    return network, readout, states[-1], relative


def _synchronise(network, data, index, scaling, n_warm=600):
    """Reservoir state consistent with ``data[index - 1]``, by *driving* the
    reservoir with the truth.

    This must not be done with :func:`chaoslib.learning.esn_rollout`, which
    runs the emulator **autonomously** and therefore synchronises the reservoir
    to a trajectory that has already left the data. Doing it wrong is not
    visibly wrong: the rollout still looks like Lorenz 96, it merely starts
    from the wrong state, and the measured tracking time comes out meaningless.
    It cost this script two wrong answers before being caught.
    """
    driven = learning.reservoir_drive(
        network, data[index - n_warm : index], scaling
    )
    return driven[-1]


def _tracking(network, readout, scaling, data, saturation, start_index):
    """Lead time at which a free rollout reaches half of saturation, averaged
    over several launches -- one launch is a sample of a chaotic quantity."""
    leads = []
    for case in range(TRACK_CASES):
        launch = start_index + case * 1200
        state = _synchronise(network, data, launch, scaling)
        rolled, _ = learning.esn_rollout(
            network, readout, state, data[launch], 1500, scaling
        )
        reference = data[launch : launch + rolled.shape[0]]
        error = np.sqrt(((rolled - reference) ** 2).sum(axis=-1))
        crossed = error > 0.5 * saturation
        if crossed.any():
            leads.append(float(int(np.argmax(crossed)) * DT))
    return float(np.mean(leads)) if leads else float("nan")


def main() -> None:
    started = time.time()
    print("# Generated by scripts/generate_ch29_data.py -- do not edit by hand.")
    print("# Chapter 29: machine learning and data-driven prediction.")

    data = _truth()
    scaling = learning.standardiser(data[:TRAIN])
    print("# --- 1. the truth's census ---")
    truth_spectrum = lyapunov.lyapunov_spectrum(
        systems.lorenz96, systems.lorenz96_jacobian, data[0], dt=0.01,
        t_final=2000.0, t_transient=20.0, forcing=FORCING,
    )
    saturation = errorgrowth.saturation_level(data[:ROLL], seed=0)
    print("#   spectrum " + " ".join(f"{v:7.3f}" for v in truth_spectrum))
    print(f"#   sum {truth_spectrum.sum():.5f} (exact {-float(SITES):.1f}), "
          f"unstable {lyapunov.unstable_dimension(truth_spectrum)}, "
          f"D_KY {lyapunov.kaplan_yorke_dimension(truth_spectrum):.3f}, "
          f"saturation {saturation:.3f}, mean {data.mean():.4f}, "
          f"sd {data.std():.4f}")

    print(f"SITES = {SITES}")
    print(f"FORCING = {FORCING}")
    print(f"DT = {DT}")
    print(f"TRAIN = {TRAIN}")
    print(f"ROLL = {ROLL}")
    print("CONFIG_NAMES = " + repr(tuple(name for name, _ in CONFIGS)))
    _emit("TRUTH_SPECTRUM", truth_spectrum, ".6f")
    _scalar("TRUTH_SUM", truth_spectrum.sum(), ".5f")
    _scalar("TRUTH_SUM_EXACT", -float(SITES), ".1f")
    _scalar("TRUTH_UNSTABLE", lyapunov.unstable_dimension(truth_spectrum), ".0f")
    _scalar("TRUTH_KY", lyapunov.kaplan_yorke_dimension(truth_spectrum), ".4f")
    _scalar("TRUTH_SATURATION", saturation, ".4f")
    _scalar("TRUTH_MEAN", data.mean(), ".4f")
    _scalar("TRUTH_SD", data.std(), ".4f")
    _scalar("TRUTH_LAMBDA1", truth_spectrum[0], ".5f")

    print("# --- 2 and 3. the emulators ---")
    rows = []
    launch = TRAIN + WASH
    for name, kwargs in CONFIGS:
        t0 = time.time()
        network, readout, _, relative = _train(data, scaling, **kwargs)
        final = _synchronise(network, data, launch, scaling)
        rolled, _ = learning.esn_rollout(
            network, readout, final, data[launch], ROLL, scaling
        )
        survived = rolled.shape[0] - 1
        stable = survived >= ROLL - 1
        sd_ratio = float(rolled.std() / data.std()) if survived > 200 else float("nan")
        mean_shift = (
            float((rolled.mean() - data.mean()) / data.std())
            if survived > 200 else float("nan")
        )
        if stable:
            spectrum = learning.esn_lyapunov_spectrum(
                network, readout, final, data[launch], scaling, dt=DT,
                n_exponents=N_EXPONENTS, n_steps=SPECTRUM_STEPS,
                n_transient=2000,
            )
            track = _tracking(network, readout, scaling, data, saturation, launch)
        else:
            spectrum = np.full(N_EXPONENTS, np.nan)
            track = float("nan")
        finite = np.isfinite(spectrum).all()
        spectrum_error = (
            100.0 * float(np.abs(spectrum - truth_spectrum).max()
                          / np.abs(truth_spectrum).max()) if finite else float("nan")
        )
        unstable = int((spectrum > 0.0).sum()) if finite else -1
        rows.append(dict(
            name=name, relative=relative, survived=survived, stable=stable,
            sd_ratio=sd_ratio, mean_shift=mean_shift, spectrum=spectrum,
            spectrum_error=spectrum_error, unstable=unstable, track=track,
            n_reservoir=kwargs["n_reservoir"],
            spectral_radius=kwargs["spectral_radius"],
            input_scaling=kwargs["input_scaling"], ridge=kwargs["ridge"],
            n_train=kwargs["n_train"],
        ))
        print(f"#   {name:20s} 1-step {relative:.2e} survived {survived:6d} "
              f"sd {sd_ratio:6.3f} lam1 {spectrum[0]:7.3f} "
              f"({spectrum[0]/truth_spectrum[0]:5.3f}) unstable {unstable:2d} "
              f"specerr {spectrum_error:6.1f}% track {track:5.2f} TU "
              f"({time.time()-t0:.0f}s)")

    _emit("CFG_RESERVOIR", [r["n_reservoir"] for r in rows], ".0f")
    _emit("CFG_RADIUS", [r["spectral_radius"] for r in rows], ".4f")
    _emit("CFG_INPUT_SCALING", [r["input_scaling"] for r in rows], ".4f")
    _emit("CFG_RIDGE", [r["ridge"] for r in rows], ".3e")
    _emit("CFG_TRAIN", [r["n_train"] for r in rows], ".0f")
    _emit("CFG_ONESTEP", [r["relative"] for r in rows], ".6e")
    _emit("CFG_SURVIVED", [r["survived"] for r in rows], ".0f")
    _emit("CFG_SD_RATIO", [r["sd_ratio"] for r in rows], ".4f")
    _emit("CFG_MEAN_SHIFT", [r["mean_shift"] for r in rows], ".4f")
    _emit("CFG_LAMBDA1", [r["spectrum"][0] for r in rows], ".6f")
    _emit("CFG_UNSTABLE", [r["unstable"] for r in rows], ".0f")
    _emit("CFG_SPECTRUM_ERROR", [r["spectrum_error"] for r in rows], ".4f")
    _emit("CFG_TRACK", [r["track"] for r in rows], ".4f")
    _emit("CFG_SPECTRUM",
          np.array([r["spectrum"] for r in rows]), ".6f")

    # The blow-up trace, and a tracking curve for the baseline against the
    # truth's own error growth from the same initial error -- so the emulator's
    # model error can be compared with an equivalent initial-condition error.
    unstable_cfg = next(name for name, k in CONFIGS if k["spectral_radius"] > 1.0)
    kwargs = dict(next(k for name, k in CONFIGS if name == unstable_cfg))
    network, readout, _, _ = _train(data, scaling, **kwargs)
    final = _synchronise(network, data, launch, scaling)
    diverged, _ = learning.esn_rollout(
        network, readout, final, data[launch], ROLL, scaling
    )
    print(f"#   blow-up trace: {diverged.shape[0] - 1} steps, "
          f"max|u| reaches {np.abs(diverged).max():.3e}")
    stride = max(1, diverged.shape[0] // 400)
    print("BLOWUP_NAME = " + repr(unstable_cfg))
    _emit("BLOWUP_TIME", np.arange(diverged.shape[0])[::stride] * DT, ".4f")
    _emit("BLOWUP_TRACE", np.abs(diverged[::stride]).max(axis=-1), ".6e")
    _scalar("BLOWUP_STEPS", diverged.shape[0] - 1, ".0f")
    _scalar("BLOWUP_TU", (diverged.shape[0] - 1) * DT, ".2f")

    base_kwargs = dict(next(k for name, k in CONFIGS if name == "baseline"))
    network, readout, _, _ = _train(data, scaling, **base_kwargs)
    state = _synchronise(network, data, launch, scaling)
    rolled, _ = learning.esn_rollout(
        network, readout, state, data[launch], 2000, scaling
    )
    reference = data[launch : launch + rolled.shape[0]]
    emulator_error = np.sqrt(((rolled - reference) ** 2).sum(axis=-1))
    # the truth's own growth from an initial error of the same size
    delta0 = float(emulator_error[1])
    rng = np.random.default_rng(11)
    kick = rng.normal(size=SITES)
    kick *= delta0 / np.linalg.norm(kick)      # delta0 is now a vector norm
    times = integrate.trajectory_grid(2000 * DT, DT)
    control = integrate.rk4(systems.lorenz96, data[launch], times, forcing=FORCING)
    twin = integrate.rk4(
        systems.lorenz96, data[launch] + kick, times, forcing=FORCING
    )
    truth_error = np.sqrt(((twin - control) ** 2).sum(axis=-1))
    print(f"#   baseline tracking curve: emulator error at step 1 "
          f"{delta0:.3e}; the truth grows the same error to half saturation "
          f"in {DT*int(np.argmax(truth_error > 0.5*saturation)):.2f} TU, the "
          f"emulator in "
          f"{DT*int(np.argmax(emulator_error > 0.5*saturation)):.2f} TU")
    # From step 1, not step 0: the emulator's error at step 0 is identically
    # zero (it starts from the truth), so subsampling from there makes the two
    # curves appear to start at different errors when the whole point is that
    # they start at the same one.
    stride = max(1, emulator_error.size // 400)
    _emit("TRACK_TIME", np.arange(emulator_error.size)[1::stride] * DT, ".4f")
    _emit("TRACK_EMULATOR", emulator_error[1::stride], ".6e")
    _emit("TRACK_TRUTH", truth_error[: emulator_error.size][1::stride], ".6e")
    _scalar("TRACK_DELTA0", delta0, ".6e")
    _scalar("TRACK_TRUTH_HALF",
            DT * int(np.argmax(truth_error > 0.5 * saturation)), ".3f")
    _scalar("TRACK_EMULATOR_HALF",
            DT * int(np.argmax(emulator_error > 0.5 * saturation)), ".3f")

    # The stable subset, emitted separately: the figures that compare spectra
    # can only include configurations that have one, and filtering in the
    # notebook would mean carrying nan through every plot.
    stable_rows = [r for r in rows if np.isfinite(r["spectrum_error"])]
    print("CFG_NAMES_OK = " + repr(tuple(r["name"] for r in stable_rows)))
    _emit("CFG_ONESTEP_OK", [r["relative"] for r in stable_rows], ".6e")
    _emit("CFG_SD_RATIO_OK", [r["sd_ratio"] for r in stable_rows], ".6f")
    _emit("CFG_TRACK_OK", [r["track"] for r in stable_rows], ".4f")
    _emit("CFG_UNSTABLE_OK", [r["unstable"] for r in stable_rows], ".0f")
    _emit("CFG_SPECTRUM_ERROR_OK",
          [r["spectrum_error"] for r in stable_rows], ".4f")
    _emit("CFG_SPECTRUM_OK",
          np.array([r["spectrum"] for r in stable_rows]), ".6f")

    print("# --- 4. what the checkable diagnostics tell you ---")
    onestep = np.array([r["relative"] for r in stable_rows])
    specerr = np.array([r["spectrum_error"] for r in stable_rows])
    tracks = np.array([r["track"] for r in stable_rows])
    sds = np.array([r["sd_ratio"] for r in stable_rows])
    r_spec = float(np.corrcoef(np.log(onestep), specerr)[0, 1])
    r_track = float(np.corrcoef(np.log(onestep), tracks)[0, 1])
    r_sd = float(np.corrcoef(np.abs(sds - 1.0), specerr)[0, 1])
    print(f"#   over {len(stable_rows)} stable configurations:")
    print(f"#     corr(log one-step error, spectrum error) = {r_spec:+.3f}")
    print(f"#     corr(log one-step error, tracking time)  = {r_track:+.3f}")
    print(f"#     corr(|sd ratio - 1|,     spectrum error) = {r_sd:+.3f}")
    best_onestep = stable_rows[int(np.argmin(onestep))]
    best_track = stable_rows[int(np.argmax(tracks))]
    print(f"#     lowest one-step error: {best_onestep['name']} "
          f"({best_onestep['relative']:.2e}), tracking {best_onestep['track']:.2f} TU")
    print(f"#     longest tracking:      {best_track['name']} "
          f"({best_track['relative']:.2e}), tracking {best_track['track']:.2f} TU")
    _scalar("CORR_ONESTEP_SPECTRUM", r_spec, ".4f")
    _scalar("CORR_ONESTEP_TRACK", r_track, ".4f")
    _scalar("CORR_SD_SPECTRUM", r_sd, ".4f")
    print("BEST_ONESTEP_NAME = " + repr(best_onestep["name"]))
    _scalar("BEST_ONESTEP_VALUE", best_onestep["relative"], ".6e")
    _scalar("BEST_ONESTEP_TRACK", best_onestep["track"], ".4f")
    print("BEST_TRACK_NAME = " + repr(best_track["name"]))
    _scalar("BEST_TRACK_VALUE", best_track["relative"], ".6e")
    _scalar("BEST_TRACK_TRACK", best_track["track"], ".4f")
    _scalar("N_STABLE", len(stable_rows), ".0f")
    print(f"# total {time.time() - started:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
