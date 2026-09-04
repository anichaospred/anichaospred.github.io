#!/usr/bin/env python3
r"""Precompute chapter 19's ensemble-assimilation experiments.

Cycling assimilation on Lorenz 96, :math:`N=40`, :math:`F=8`: all 40 sites
observed every 0.05 TU (the six-hour analogue) with :math:`\sigma_o = 1`, against
a climatological spread of about 3.6. A well-configured filter reaches an
analysis error near 0.18, so the useful dynamic range of these experiments is
roughly a factor of twenty.

**Every comparison is paired.** One truth trajectory, one array of observation
noise, and one pool of initial ensemble perturbations, all drawn before the
sweeps begin and sliced rather than redrawn -- so a run with 5 members sees
exactly the same truth and the same observations as a run with 40. Redrawing per
configuration costs about as much scatter as the effects being measured, which
was learned the hard way in chapter 18's window sweep.

**Every scheme is given its own best inflation**, and where a localisation radius
applies, its own best radius, chosen from a fixed set that is reported. Comparing
a tuned filter against an untuned one is the commonest way to make a filter
comparison say whatever you wanted it to say.

Seven blocks.

0. **What is being estimated.** The forecast-error correlation, time-averaged over
   a well-configured run -- *not* the climatological correlation of the state,
   which is nearly diagonal and supports no claim about what localisation
   preserves. Conflating the two produced a figure whose "truth" panel showed no
   structure at all under a caption asserting a band.
1. **Sampling error.** RMS error in the far field of a sampled correlation
   matrix, against :math:`k^{-1/2}`. This is the whole case for localisation.
2. **Rank.** The global filter's increment lies exactly in the span of the
   :math:`k-1` ensemble perturbations; the local filter's does not. Measured as
   the fraction of the increment lying outside that span.
3. **The map**: analysis error over ensemble size and localisation radius. The
   filter-divergence cliff, and the ridge of optimal radius running along it.
4. **Inflation** over ensemble size, at a fixed radius.
5. **Schemes**: the deterministic local filter against the stochastic
   perturbed-observation filter, each at its own best configuration.
6. **Hybrids**: blending in a static covariance as a second, quite separate cure
   for rank deficiency -- and how it compares with localisation, and with both
   together.

Run from chaos-book/:
    python3 scripts/generate_ch19_data.py        # ~15 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import assimilate, ensemble as ens_tools, integrate, systems  # noqa: E402

N_SITES, FORCING, DT = 40, 8.0, 0.01
OBS_SIGMA = 1.0
CYCLE_STEPS = 5                    # 0.05 TU between analyses
SPINUP_CYCLES, CYCLES = 150, 500
MEMBERS = (5, 8, 10, 15, 20, 30, 40)
CUTOFFS = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, None)
INFLATIONS = (1.0, 1.02, 1.05, 1.1, 1.2, 1.4)
INFLATION_GRID = (1.0, 1.01, 1.02, 1.04, 1.08, 1.15, 1.25, 1.4)
REFERENCE_CUTOFF = 8.0
BETAS = (0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0)
STATIC_SCALE = 0.02               # tuned; see block 6
HYBRID_MEMBERS = (5, 10, 20)

_H_OP = np.eye(N_SITES)
_R_COV = np.eye(N_SITES) * OBS_SIGMA**2
_STEP_GRID = np.linspace(0.0, CYCLE_STEPS * DT, CYCLE_STEPS + 1)


def _emit(name: str, values, fmt: str = ".6e", per_line: int = 8) -> None:
    items = [
        'float("nan")' if not np.isfinite(v) else format(float(v), fmt)
        for v in np.ravel(values)
    ]
    print(f"{name} = (")
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i : i + per_line]) + ",")
    print(")")


def _advance(states):
    return integrate.rk4(
        systems.lorenz96, states, _STEP_GRID, forcing=FORCING
    )[-1]


def _spin_up():
    start = systems.lorenz96_uniform_state(FORCING, N_SITES)
    start = start + np.r_[0.01, np.zeros(N_SITES - 1)]
    return integrate.rk4(
        systems.lorenz96, start, integrate.trajectory_grid(140.0, DT),
        forcing=FORCING,
    )[4000:]


class Experiment:
    """One truth run, one noise realisation, one perturbation pool -- shared.

    Holding these fixed is what makes the sweeps comparable. The observation
    noise is drawn for every cycle in advance from its own generator, so it does
    not depend on how many members a configuration happens to use.
    """

    def __init__(self, attractor, seed: int = 0):
        total = SPINUP_CYCLES + CYCLES
        self.truth = np.empty((total + 1, N_SITES))
        self.truth[0] = attractor[0]
        for step in range(total):
            self.truth[step + 1] = _advance(self.truth[step])
        noise_rng = np.random.default_rng(1000 + seed)
        self.obs = self.truth[1:] + noise_rng.normal(
            0.0, OBS_SIGMA, (total, N_SITES)
        )
        pool_rng = np.random.default_rng(2000 + seed)
        self.pool = pool_rng.normal(0.0, 1.0, (max(MEMBERS), N_SITES))
        self.climatology = np.cov(attractor[::5].T)
        self.clim_std = float(attractor.std())

    def run(
        self, n_members: int, cutoff, inflation: float,
        scheme: str = "letkf", beta: float = 1.0, static=None,
    ) -> float:
        """Mean analysis error over the scoring cycles, or nan if it blew up."""
        ensemble = self.truth[0] + self.pool[:n_members]
        weights = (
            None if cutoff is None
            else assimilate.ring_localisation(N_SITES, cutoff)
        )
        errors = []
        for cycle in range(SPINUP_CYCLES + CYCLES):
            ensemble = _advance(ensemble)
            if not np.all(np.isfinite(ensemble)) or np.abs(ensemble).max() > 1e4:
                return float("nan")
            observation = self.obs[cycle]
            if scheme == "letkf":
                ensemble = assimilate.letkf_update(
                    ensemble, observation, _H_OP, OBS_SIGMA**2,
                    inflation=inflation, weights=weights,
                )
            else:
                background = None if beta >= 1.0 else assimilate.hybrid_covariance(
                    ensemble, static, beta, inflation=inflation
                )
                ensemble = assimilate.enkf_update(
                    ensemble, observation, _H_OP, _R_COV, inflation=inflation,
                    localisation=weights, seed=cycle,
                    background_cov=background,
                )
            if cycle >= SPINUP_CYCLES:
                errors.append(float(np.sqrt(np.mean(
                    (ensemble.mean(axis=0) - self.truth[cycle + 1]) ** 2
                ))))
        return float(np.mean(errors))

    def best(self, inflations=INFLATIONS, **kwargs) -> tuple[float, float]:
        """Best error over a set of inflations, with the inflation that won."""
        scored = [(self.run(inflation=i, **kwargs), i) for i in inflations]
        finite = [(e, i) for e, i in scored if np.isfinite(e)]
        return min(finite) if finite else (float("nan"), float("nan"))


# ==========================================================================
# 0. what the covariance being estimated actually looks like
# ==========================================================================
def error_correlation(experiment) -> None:
    r"""The forecast-error correlation the filter is trying to estimate.

    This is *not* the climatological correlation of the state, and conflating the
    two was a mistake worth recording: the climatological correlation of Lorenz 96
    is nearly diagonal, and a figure of it supports no claim about what
    localisation preserves. What a filter estimates is the covariance of
    **forecast errors**, accumulated here over the scoring cycles of a
    well-configured run.

    It is not a smooth band either. The profile has a *negative* lobe near
    separation 2, which is half the wavelength of the dominant Lorenz 96 mode --
    chapter 11 puts that at :math:`m^*=8`, so :math:`N/m^* = 5` sites per
    wavelength and an anticorrelation at 2.5. The structure localisation has to
    preserve is a wave, not a blob.
    """
    print("# --- 0. the forecast-error correlation ---")
    weights = assimilate.ring_localisation(N_SITES, REFERENCE_CUTOFF)
    index = np.arange(N_SITES)
    separation = np.minimum(
        np.abs(index[:, None] - index[None, :]),
        N_SITES - np.abs(index[:, None] - index[None, :]),
    )

    def profile(matrix):
        return [float(matrix[separation == d].mean()) for d in range(N_SITES // 2 + 1)]

    # The target: time-averaged forecast-error covariance from a large ensemble.
    ensemble = experiment.truth[0] + experiment.pool[:40]
    accumulated = np.zeros((N_SITES, N_SITES))
    counted = 0
    for cycle in range(SPINUP_CYCLES + CYCLES):
        ensemble = _advance(ensemble)
        if cycle >= SPINUP_CYCLES:
            error = ensemble.mean(axis=0) - experiment.truth[cycle + 1]
            accumulated += np.outer(error, error)
            counted += 1
        ensemble = assimilate.letkf_update(
            ensemble, experiment.obs[cycle], _H_OP, OBS_SIGMA**2,
            inflation=1.02, weights=weights,
        )
    covariance = accumulated / counted
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    true_corr = covariance / scale

    print(f"#   time-averaged over {counted} cycles; correlation by separation:")
    print("#     " + "  ".join(
        f"{d}:{v:+.3f}" for d, v in enumerate(profile(true_corr)[:8])
    ))
    print(f"#   rms beyond 12 sites: "
          f"{np.sqrt((true_corr[separation >= 12] ** 2).mean()):.4f}"
          f"  (its OWN sampling floor is about "
          f"{1.0 / np.sqrt(counted):.3f}, so this is a bound, not a value)")
    print(f"ERRCORR_CYCLES = {counted}")
    print(f"ERRCORR_FLOOR = {1.0 / np.sqrt(counted):.6f}")
    _emit("ERRCORR_TRUE", true_corr.ravel(), ".5f", per_line=N_SITES)
    _emit("ERRCORR_PROFILE_TRUE", profile(true_corr), ".5f", per_line=11)

    # What a filter of each size actually sees, at one cycle of its own run.
    samples = (5, 20)
    for n_members in samples:
        members = experiment.truth[0] + experiment.pool[:n_members]
        for cycle in range(SPINUP_CYCLES + 100):
            members = _advance(members)
            members = assimilate.letkf_update(
                members, experiment.obs[cycle], _H_OP, OBS_SIGMA**2,
                inflation=1.05, weights=weights,
            )
        members = _advance(members)
        estimate = np.corrcoef((members - members.mean(axis=0)).T)
        far = float(np.sqrt((estimate[separation >= 12] ** 2).mean()))
        print(f"#   k={n_members:2d} sample estimate: rms beyond 12 sites {far:.4f}")
        _emit(f"ERRCORR_K{n_members}", estimate.ravel(), ".5f", per_line=N_SITES)
        _emit(f"ERRCORR_PROFILE_K{n_members}", profile(estimate), ".5f", per_line=11)
        print(f"ERRCORR_FAR_K{n_members} = {far:.6f}")
    print(f"ERRCORR_SAMPLES = {samples}")


# ==========================================================================
# 1. sampling error
# ==========================================================================
def sampling_error() -> None:
    print("# --- 1. sampling error in the covariance ---")
    truth_corr = assimilate.ring_localisation(N_SITES, 8.0)
    factor = np.linalg.cholesky(truth_corr + 1e-9 * np.eye(N_SITES))
    index = np.arange(N_SITES)
    separation = np.minimum(
        np.abs(index[:, None] - index[None, :]),
        N_SITES - np.abs(index[:, None] - index[None, :]),
    )
    far = separation >= 12
    assert np.abs(truth_corr[far]).max() == 0.0, "far field must be exactly zero"

    sizes = (5, 8, 10, 15, 20, 30, 40, 80, 160, 320)
    rng = np.random.default_rng(0)
    rms, ratio = [], []
    for n_members in sizes:
        errors = []
        for _trial in range(40):
            draw = rng.normal(size=(n_members, N_SITES)) @ factor.T
            errors.append(np.corrcoef(draw.T)[far])
        value = float(np.sqrt(np.mean(np.square(np.concatenate(errors)))))
        rms.append(value)
        ratio.append(value * np.sqrt(n_members))
    print("#   RMS spurious correlation beyond 12 sites (true value: exactly 0)")
    print("#     k      " + "".join(f"{s:8d}" for s in sizes))
    print("#     rms    " + "".join(f"{v:8.4f}" for v in rms))
    print("#     x sqrt(k)" + "".join(f"{v:8.3f}" for v in ratio))
    print(f"SAMPLING_SIZES = {sizes}")
    _emit("SAMPLING_RMS", rms, ".6f", per_line=10)
    _emit("SAMPLING_RATIO", ratio, ".6f", per_line=10)


# ==========================================================================
# 2. rank
# ==========================================================================
def rank_and_span(experiment) -> None:
    print("\n# --- 2. rank: where the increment is allowed to live ---")
    rng = np.random.default_rng(7)
    observation = experiment.obs[0]
    print(f"#   {'k':>4} {'global':>10} {'c=4':>10} {'c=8':>10} {'c=20':>10}"
          "   (fraction of the increment outside the ensemble span)")
    keys = ("GLOBAL", "C4", "C8", "C20")
    configs = (None, 4.0, 8.0, 20.0)
    table = {key: [] for key in keys}
    for n_members in MEMBERS:
        members = experiment.truth[0] + experiment.pool[:n_members]
        members = members + rng.normal(0.0, 0.3, members.shape)
        row = []
        for key, cutoff in zip(keys, configs):
            weights = (
                None if cutoff is None
                else assimilate.ring_localisation(N_SITES, cutoff)
            )
            increment = assimilate.letkf_update(
                members, observation, _H_OP, OBS_SIGMA**2, weights=weights
            ) - members
            fraction = ens_tools.outside_span_fraction(increment, members)
            table[key].append(fraction)
            row.append(fraction)
        print(f"#   {n_members:4d} " + "".join(f"{v:10.2e}" for v in row))
    print(f"RANK_MEMBERS = {MEMBERS}")
    print(f"RANK_KEYS = {keys}")
    for key in keys:
        _emit(f"RANK_{key}", table[key], ".6e", per_line=7)


# ==========================================================================
# 3. the map
# ==========================================================================
def localisation_map(experiment) -> None:
    print("\n# --- 3. the map: ensemble size against localisation radius ---")
    print(f"#   climatological spread {experiment.clim_std:.4f}, "
          f"observation error {OBS_SIGMA}")
    labels = tuple("inf" if c is None else f"{c:g}" for c in CUTOFFS)
    print("#   " + f"{'k':>4} " + "".join(f"{lab:>9}" for lab in labels) + "   best")
    errors = np.full((len(MEMBERS), len(CUTOFFS)), np.nan)
    inflations = np.full((len(MEMBERS), len(CUTOFFS)), np.nan)
    best_cutoff = []
    for i, n_members in enumerate(MEMBERS):
        started = time.perf_counter()
        for j, cutoff in enumerate(CUTOFFS):
            errors[i, j], inflations[i, j] = experiment.best(
                n_members=n_members, cutoff=cutoff
            )
        finite = np.isfinite(errors[i])
        pick = int(np.nanargmin(np.where(finite, errors[i], np.inf)))
        best_cutoff.append(labels[pick])
        print(f"#   {n_members:4d} " + "".join(
            f"{v:9.4f}" if np.isfinite(v) else f"{'blew up':>9}"
            for v in errors[i]
        ) + f"   {labels[pick]:>4}  ({time.perf_counter() - started:.0f}s)")
    print(f"#   optimal radius by ensemble size: "
          f"{dict(zip(MEMBERS, best_cutoff))}")
    print(f"MAP_MEMBERS = {MEMBERS}")
    print(f"MAP_CUTOFF_LABELS = {labels}")
    print(f"MAP_BEST_CUTOFF = {tuple(best_cutoff)}")
    print(f"CLIM_STD = {experiment.clim_std:.6f}")
    print(f"OBS_SIGMA = {OBS_SIGMA}")
    print(f"INFLATIONS_TRIED = {INFLATIONS}")
    _emit("MAP_ERROR", errors.ravel(), ".6f", per_line=len(CUTOFFS))
    _emit("MAP_INFLATION", inflations.ravel(), ".4f", per_line=len(CUTOFFS))


# ==========================================================================
# 4. inflation
# ==========================================================================
def inflation_map(experiment) -> None:
    print("\n# --- 4. inflation against ensemble size ---")
    print(f"#   at localisation radius {REFERENCE_CUTOFF:g}")
    print("#   " + f"{'k':>4} " + "".join(f"{i:>8.2f}" for i in INFLATION_GRID))
    errors = np.full((len(MEMBERS), len(INFLATION_GRID)), np.nan)
    penalties: list[float] = []
    for i, n_members in enumerate(MEMBERS):
        for j, inflation in enumerate(INFLATION_GRID):
            errors[i, j] = experiment.run(
                n_members=n_members, cutoff=REFERENCE_CUTOFF, inflation=inflation
            )
        # The cost of NOT inflating, relative to the best. The full-range
        # worst/best ratio is a worse statistic: it is dominated by the absurd
        # end of the range (inflation 1.4 is nobody's candidate), and it made a
        # configuration whose sensible range spans 5 per cent look like a
        # factor of two.
        penalties.append(float(errors[i, 0] / np.nanmin(errors[i])))
        print(f"#   {n_members:4d} " + "".join(
            f"{v:8.4f}" if np.isfinite(v) else f"{'--':>8}" for v in errors[i]
        ) + f"   no-inflation penalty x{penalties[-1]:.2f}")
    print(f"INFLATION_GRID = {INFLATION_GRID}")
    print(f"INFLATION_CUTOFF = {REFERENCE_CUTOFF}")
    _emit("INFLATION_ERROR", errors.ravel(), ".6f", per_line=len(INFLATION_GRID))
    _emit("INFLATION_PENALTY", penalties, ".5f", per_line=7)


# ==========================================================================
# 5. schemes
# ==========================================================================
def schemes(experiment) -> None:
    print("\n# --- 5. deterministic against stochastic ---")
    print("#   each at its own best radius AND its own best inflation")
    print(f"#   {'k':>4} {'LETKF':>18} {'stochastic EnKF':>22} {'ratio':>7}")
    det, sto, det_cut, sto_cut = [], [], [], []
    labels = tuple("inf" if c is None else f"{c:g}" for c in CUTOFFS)
    for n_members in MEMBERS:
        rows = {}
        for scheme in ("letkf", "enkf"):
            scored = [
                (experiment.best(n_members=n_members, cutoff=c, scheme=scheme), c)
                for c in CUTOFFS
            ]
            finite = [((e, i), c) for (e, i), c in scored if np.isfinite(e)]
            rows[scheme] = min(finite, key=lambda item: item[0][0])
        (de, di), dc = rows["letkf"]
        (se, si), sc = rows["enkf"]
        det.append(de); sto.append(se)
        det_cut.append(labels[CUTOFFS.index(dc)])
        sto_cut.append(labels[CUTOFFS.index(sc)])
        print(f"#   {n_members:4d}  {de:.4f} @c={det_cut[-1]:>3},i={di:.2f}"
              f"   {se:.4f} @c={sto_cut[-1]:>3},i={si:.2f}   {se / de:7.3f}")
    print(f"SCHEME_MEMBERS = {MEMBERS}")
    print(f"SCHEME_DET_CUTOFF = {tuple(det_cut)}")
    print(f"SCHEME_STO_CUTOFF = {tuple(sto_cut)}")
    _emit("SCHEME_DET", det, ".6f", per_line=7)
    _emit("SCHEME_STO", sto, ".6f", per_line=7)


# ==========================================================================
# 6. hybrids
# ==========================================================================
def hybrids(experiment) -> None:
    print("\n# --- 6. hybrids: a static covariance as the other cure ---")
    # The static covariance has to be TUNED, not just climatological. Raw
    # climatology has a spread of 3.6 against a background error near 0.3, and
    # using it unscaled makes pure 3D-Var 0.94 instead of 0.43 -- which would
    # have flattered every hybrid in this block by a factor of two.
    print("#   choosing the static covariance scale (pure 3D-Var, k=10):")
    scales, scale_errors = (1.0, 0.25, 0.05, 0.02, 0.01, 0.003), []
    for scale in scales:
        value, _ = experiment.best(
            n_members=10, cutoff=None, scheme="enkf", beta=0.0,
            static=experiment.climatology * scale,
        )
        scale_errors.append(value)
        print(f"#     scale {scale:6.3f} (std "
              f"{np.sqrt(scale * experiment.climatology[0, 0]):5.2f}): "
              f"{value:.4f}")
    static = experiment.climatology * STATIC_SCALE
    print(f"#   using scale {STATIC_SCALE}")
    print(f"STATIC_SCALES = {scales}")
    print(f"STATIC_SCALE = {STATIC_SCALE}")
    _emit("STATIC_SCALE_ERROR", scale_errors, ".6f", per_line=6)

    print(f"#   beta sweep, no localisation:")
    print("#   " + f"{'beta':>6} " + "".join(f"{f'k={k}':>10}" for k in HYBRID_MEMBERS))
    errors = np.full((len(BETAS), len(HYBRID_MEMBERS)), np.nan)
    for i, beta in enumerate(BETAS):
        for j, n_members in enumerate(HYBRID_MEMBERS):
            errors[i, j], _ = experiment.best(
                n_members=n_members, cutoff=None, scheme="enkf", beta=beta,
                static=static,
            )
        print(f"#   {beta:6.2f} " + "".join(
            f"{v:10.4f}" if np.isfinite(v) else f"{'blew up':>10}"
            for v in errors[i]
        ))
    print(f"HYBRID_BETAS = {BETAS}")
    print(f"HYBRID_MEMBERS = {HYBRID_MEMBERS}")
    _emit("HYBRID_ERROR", errors.ravel(), ".6f", per_line=len(HYBRID_MEMBERS))

    print("#   the two cures, separately and together, at k=5:")
    # Each cure gets its best beta as well as its best inflation. Fixing
    # beta = 0.5 for the combined case understated it: with localisation already
    # supplying the rank, the best blend is a different one.
    combos = (
        ("NONE", None, (1.0,)),
        ("LOC", REFERENCE_CUTOFF, (1.0,)),
        ("HYBRID", None, BETAS[:-1]),
        ("BOTH", REFERENCE_CUTOFF, BETAS),
        ("STATIC", None, (0.0,)),
    )
    values, chosen = [], []
    for label, cutoff, betas in combos:
        scored = [
            (experiment.best(
                n_members=5, cutoff=cutoff, scheme="enkf", beta=b, static=static
            ), b)
            for b in betas
        ]
        finite = [((e, i), b) for (e, i), b in scored if np.isfinite(e)]
        (value, inflation), beta = min(finite, key=lambda item: item[0][0])
        values.append(value)
        chosen.append(beta)
        print(f"#     {label:7s} {value:.4f} @beta {beta:.2f}, "
              f"inflation {inflation:.2f}")
    print(f"CURE_LABELS = {tuple(c[0] for c in combos)}")
    print(f"CURE_BETA = {tuple(chosen)}")
    _emit("CURE_ERROR", values, ".6f", per_line=5)


if __name__ == "__main__":
    started = time.perf_counter()
    attractor = _spin_up()
    experiment = Experiment(attractor)
    print(f"# L96 N={N_SITES} F={FORCING}; analyses every "
          f"{CYCLE_STEPS * DT:g} TU; {CYCLES} scored cycles after "
          f"{SPINUP_CYCLES} spin-up")
    print(f"# climatological spread {experiment.clim_std:.4f}")
    print(f"CYCLES = {CYCLES}")
    print(f"CYCLE_INTERVAL = {CYCLE_STEPS * DT}")
    print(f"N_SITES = {N_SITES}")
    error_correlation(experiment)
    sampling_error()
    rank_and_span(experiment)
    localisation_map(experiment)
    inflation_map(experiment)
    schemes(experiment)
    hybrids(experiment)
    print(f"\n# total {time.perf_counter() - started:.0f}s")
