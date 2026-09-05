#!/usr/bin/env python3
r"""Precompute chapter 17's probabilistic-forecast experiments.

Forecasts from a *real* analysis: a cycling LETKF on Lorenz 96 (:math:`N=40`,
:math:`F=8`, all sites observed every 0.05 TU with :math:`\sigma_o=1`, chapter
19's configuration) supplies both the analysis and its own analysis ensemble, so
the analysis-error distribution the ensembles are supposed to sample is the one
they are actually scored against.

Five constructions, all with the same member count and the same total
perturbation amplitude, so the comparison is about *direction* and nothing else:

* **EDA** -- the LETKF analysis ensemble itself, which samples the analysis-error
  distribution by construction. The operational answer.
* **Gaussian** -- isotropic random perturbations. The naive baseline.
* **Bred** -- the breeding cycle of Toth & Kalnay, run **along the analysis
  cycle** rather than as a separate excursion. This matters: an earlier version
  bred forward from the analysis state for 5 TU, which produces vectors belonging
  to a state 5 TU downstream of the one being perturbed. Breeding is maintained
  here as persistent state across the cycling loop, which is what the
  operational scheme did.
* **Bred, orthogonalised** -- the same, re-orthogonalised each cycle, which is
  the Benettin construction and prevents the collapse block 1 measures.
* **Singular vectors** -- :math:`\pm` pairs of the leading singular vectors of
  the tangent-linear propagator over the forecast window (chapter 16).

Five blocks.

1. **Breeding collapse**, on Lorenz 63 and Lorenz 96, and *how* the
   collapsed set is degenerate -- into two antipodal clusters, since breeding
   fixes a direction but not a sign. Independently bred vectors
   converge on each other, and how fast depends on how well-separated the leading
   Lyapunov exponent is: about 2 e-foldings on Lorenz 63, about 8 on Lorenz 96,
   which has thirteen positive exponents of similar size.
2. **The five constructions scored** -- spread against error, CRPS, and rank
   histograms, at a range of leads.
3. **Ensemble size** -- diminishing returns in CRPS.
4. **The Brier decomposition** on a threshold event, and what recalibration can
   and cannot repair.
5. **Value** -- the cost-loss curve, probabilistic against deterministic.

Run from chaos-book/:
    python3 scripts/generate_ch17_data.py        # ~8 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import adjoint, assimilate, ensemble, integrate, systems  # noqa: E402

N_SITES, FORCING, DT = 40, 8.0, 0.01
OBS_SIGMA = 1.0
CYCLE_STEPS = 5
SPINUP_CYCLES, N_CASES, CASE_STRIDE = 200, 500, 3
MEMBERS = 20
LEADS = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0)
SV_WINDOW = 0.5
SIZES = (4, 6, 10, 14, 20, 30, 40)
LOCALISATION, INFLATION = 8.0, 1.02
METHODS = ("EDA", "GAUSS", "BRED", "BREDORTH", "SV")
EVENT_LEAD, EVENT_SITE = 1.0, 0
COST_LOSS = np.round(np.linspace(0.05, 0.95, 19), 4)
LAMBDA1_L96, LAMBDA1_L63 = 1.67, 0.906

_H_OP = np.eye(N_SITES)
_CYCLE_GRID = np.linspace(0.0, CYCLE_STEPS * DT, CYCLE_STEPS + 1)


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


def _forecast(states, tau):
    if tau <= 0.0:
        return np.asarray(states, dtype=float)
    grid = np.linspace(0.0, tau, int(round(tau / DT)) + 1)
    return integrate.rk4(systems.lorenz96, states, grid, forcing=FORCING)[-1]


def _mean_pairwise_angle(vectors):
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.abs(unit @ unit.T)
    upper = np.triu_indices(unit.shape[0], 1)
    return float(np.degrees(np.arccos(np.clip(cosine[upper], 0.0, 1.0))).mean())


# ==========================================================================
# 1. breeding collapse
# ==========================================================================
def breeding_collapse() -> None:
    print("# --- 1. do independently bred vectors stay distinct? ---")
    # Vector count differs by system because it must: there are at most
    # n_state mutually orthogonal directions, so the orthogonalised comparison
    # cannot use 4 vectors on a 3-variable system. `bred_vectors` refuses rather
    # than silently returning a smaller set, which is how this was caught.
    configs = (
        ("L63", systems.lorenz63, 3, 0.005, {}, 0.25, LAMBDA1_L63,
         (2, 4, 6, 10, 16, 25, 40), 3),
        ("L96", systems.lorenz96, N_SITES, DT, {"forcing": FORCING}, 0.05,
         LAMBDA1_L96, (2, 6, 12, 25, 50, 100, 200), 4),
    )
    for (label, rhs, n_state, dt, params, cycle, lam, cycle_counts,
         n_vectors) in configs:
        if label == "L63":
            start = np.array([1.0, 1.0, 20.0])
        else:
            start = systems.lorenz96_uniform_state(FORCING, N_SITES) + np.r_[
                0.01, np.zeros(N_SITES - 1)
            ]
        attractor = integrate.rk4(
            rhs, start, integrate.trajectory_grid(80.0, dt), **params
        )[3000:]
        bases = attractor[:: attractor.shape[0] // 9][:8]

        angles, orth_angles, random_angles = [], [], []
        for n_cycles in cycle_counts:
            plain, orthogonal, naive = [], [], []
            for index, base in enumerate(bases):
                rng = np.random.default_rng(index)
                start_set = rng.normal(size=(n_vectors, n_state))
                start_set *= 1e-3 / np.linalg.norm(
                    start_set, axis=1, keepdims=True
                )
                naive.append(_mean_pairwise_angle(start_set))
                plain.append(_mean_pairwise_angle(ensemble.bred_vectors(
                    rhs, base, n_vectors, 1e-3, cycle, n_cycles=n_cycles,
                    dt=dt, seed=index, **params,
                )))
                orthogonal.append(_mean_pairwise_angle(ensemble.bred_vectors(
                    rhs, base, n_vectors, 1e-3, cycle, n_cycles=n_cycles,
                    dt=dt, seed=index, orthogonalise=True, **params,
                )))
            angles.append(float(np.mean(plain)))
            orth_angles.append(float(np.mean(orthogonal)))
            random_angles.append(float(np.mean(naive)))

        efolds = [n * cycle * lam for n in cycle_counts]
        print(f"#   {label} (n={n_state}, lambda1={lam}, cycle {cycle:g} TU); "
              f"mean pairwise angle of {n_vectors} bred vectors, 8 base states")
        print(f"#     e-folds " + "".join(f"{v:8.2f}" for v in efolds))
        print(f"#     plain   " + "".join(f"{v:8.1f}" for v in angles))
        print(f"#     orth.   " + "".join(f"{v:8.1f}" for v in orth_angles))
        print(f"#     random  {random_angles[0]:8.1f}  (unbred reference)")
        print(f"{label}_CYCLES = {tuple(cycle_counts)}")
        print(f"{label}_CYCLE_TIME = {cycle}")
        print(f"{label}_LAMBDA1 = {lam}")
        print(f"{label}_RANDOM_ANGLE = {random_angles[0]:.4f}")
        print(f"{label}_N_VECTORS = {n_vectors}")
        _emit(f"{label}_EFOLDS", efolds, ".5f", per_line=7)
        _emit(f"{label}_ANGLE", angles, ".4f", per_line=7)
        _emit(f"{label}_ANGLE_ORTH", orth_angles, ".4f", per_line=7)


def breeding_degeneracy() -> None:
    r"""*How* a collapsed bred set is degenerate, which is worse than "narrow".

    Breeding fixes a direction but not a sign: the rescaling step normalises the
    difference, and nothing prefers :math:`+v` to :math:`-v`. So a fully bred set
    does not collapse to a point cloud around one direction -- it collapses to
    **two antipodal clusters**, and an ensemble of two points is not a sample of
    a distribution at all.

    This is what makes the plain-bred rank histogram in block 2 spiked at the
    cluster edges rather than merely U-shaped, and it is why "under-dispersed"
    undersells the problem.
    """
    print("\n# --- 1b. how the collapsed set is degenerate ---")
    start = systems.lorenz96_uniform_state(FORCING, N_SITES) + np.r_[
        0.01, np.zeros(N_SITES - 1)
    ]
    attractor = integrate.rk4(
        systems.lorenz96, start, integrate.trajectory_grid(80.0, DT),
        forcing=FORCING,
    )[3000:]

    for label, orthogonalise in (("BRED", False), ("BREDORTH", True)):
        projections = []
        for index, base in enumerate(attractor[::1200][:6]):
            vectors = ensemble.bred_vectors(
                systems.lorenz96, base, MEMBERS, 1e-3, 0.05, n_cycles=200,
                dt=DT, seed=index, orthogonalise=orthogonalise, forcing=FORCING,
            )
            unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            projections.append(unit @ unit[0])
        stacked = np.concatenate(projections)
        aligned = float(np.mean(stacked > 0.9))
        anti = float(np.mean(stacked < -0.9))
        middle = float(np.mean(np.abs(stacked) <= 0.9))
        print(f"#   {label:9s} projection on the leading member: "
              f"|cos| > 0.9 in {100 * (aligned + anti):5.1f} % of members "
              f"({100 * aligned:.0f} % aligned, {100 * anti:.0f} % anti-aligned), "
              f"{100 * middle:5.1f} % in between")
        print(f"DEGEN_{label}_ALIGNED = {aligned:.6f}")
        print(f"DEGEN_{label}_ANTI = {anti:.6f}")
        print(f"DEGEN_{label}_MIDDLE = {middle:.6f}")
        _emit(f"DEGEN_{label}_HIST",
              np.histogram(stacked, bins=np.linspace(-1.0, 1.0, 21))[0],
              ".1f", per_line=20)
    print("DEGEN_BINS = " + repr(tuple(np.round(np.linspace(-1.0, 1.0, 21), 3))))


# ==========================================================================
# the cycling run: analyses, analysis ensembles, and live bred vectors
# ==========================================================================
def build_cases():
    """One cycling LETKF, with a breeding cycle maintained alongside it."""
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
    # Breeding runs ALONG the analysis cycle, as persistent state. Amplitude is
    # arbitrary here; the perturbations are rescaled to the analysis-error size
    # when an ensemble is built from them.
    bred = rng.normal(size=(MEMBERS, N_SITES))
    bred *= 1e-3 / np.linalg.norm(bred, axis=1, keepdims=True)
    bred_orth = bred.copy()

    cases = []
    for cycle in range(SPINUP_CYCLES + N_CASES * CASE_STRIDE):
        control = members.mean(axis=0)
        advanced = _advance(control + bred)
        advanced_orth = _advance(control + bred_orth)
        truth = _advance(truth)
        members = _advance(members)
        control_next = _advance(control)

        bred = advanced - control_next
        bred *= 1e-3 / np.linalg.norm(bred, axis=1, keepdims=True)
        bred_orth = advanced_orth - control_next
        basis, _ = np.linalg.qr(bred_orth.T)
        bred_orth = basis.T[:MEMBERS]
        bred_orth *= 1e-3 / np.linalg.norm(bred_orth, axis=1, keepdims=True)

        observation = truth + rng.normal(0.0, OBS_SIGMA, N_SITES)
        members = assimilate.letkf_update(
            members, observation, _H_OP, OBS_SIGMA**2,
            inflation=INFLATION, weights=weights,
        )
        if cycle >= SPINUP_CYCLES and (cycle - SPINUP_CYCLES) % CASE_STRIDE == 0:
            cases.append((members.copy(), truth.copy(), bred.copy(),
                          bred_orth.copy()))
    return cases


def _build_ensembles(analysis_ensemble, bred, bred_orth, amplitude, case_index,
                     n_members=MEMBERS):
    """The five constructions, all at the same total perturbation amplitude."""
    mean = analysis_ensemble.mean(axis=0)

    def rescale(perturbations, count):
        chosen = np.asarray(perturbations)[:count]
        scale = amplitude / np.sqrt(np.mean(np.sum(chosen**2, axis=1)))
        return mean + chosen * scale

    built = {"EDA": analysis_ensemble[:n_members]}
    gaussian = np.random.default_rng(500 + case_index).normal(
        size=(n_members, N_SITES)
    )
    built["GAUSS"] = rescale(gaussian, n_members)
    built["BRED"] = rescale(bred, n_members)
    built["BREDORTH"] = rescale(bred_orth, n_members)
    propagator = adjoint.tangent_linear_propagator(
        systems.lorenz96, systems.lorenz96_jacobian, mean, SV_WINDOW,
        dt=DT, forcing=FORCING,
    )
    built["SV"] = rescale(
        ensemble.singular_vector_ensemble(propagator, n_members // 2, 1.0),
        n_members,
    )
    return built


# ==========================================================================
# 2. the five constructions, scored
# ==========================================================================
def score_constructions(cases):
    print("\n# --- 2. five constructions, scored ---")
    amplitude = float(np.sqrt(np.mean([
        np.sum((e.mean(axis=0) - t) ** 2) for e, t, _b, _o in cases
    ])))
    per_component = amplitude / np.sqrt(N_SITES)
    print(f"#   {len(cases)} cases, {MEMBERS} members; analysis error "
          f"{per_component:.4f} per component ({amplitude:.4f} whole-state)")
    print(f"#   all constructions rescaled to the SAME total amplitude, so the "
          f"comparison is about direction only")

    spread = {m: [[] for _ in LEADS] for m in METHODS}
    error = {m: [[] for _ in LEADS] for m in METHODS}
    score = {m: [[] for _ in LEADS] for m in METHODS}
    ranks = {m: np.zeros(MEMBERS + 1) for m in METHODS}
    rank_lead = int(np.argmin(np.abs(np.asarray(LEADS) - 1.0)))

    started = time.perf_counter()
    for index, (analysis, truth, bred, bred_orth) in enumerate(cases):
        built = _build_ensembles(analysis, bred, bred_orth, amplitude, index)
        for name, members in built.items():
            for slot, lead in enumerate(LEADS):
                forecast = _forecast(members, lead)
                verify = _forecast(truth, lead)
                spread[name][slot].append(
                    float(np.sqrt(np.mean(forecast.var(axis=0, ddof=1))))
                )
                error[name][slot].append(
                    float(np.sqrt(np.mean((forecast.mean(axis=0) - verify) ** 2)))
                )
                score[name][slot].append(
                    float(ensemble.crps(forecast.T, verify).mean())
                )
                if slot == rank_lead:
                    ranks[name] += ensemble.rank_histogram(
                        forecast.T, verify, seed=index
                    )
    print(f"#   ({time.perf_counter() - started:.0f}s)")

    print(f"#   {'method':>9} {'lead':>5} {'spread':>8} {'error':>8} "
          f"{'sp/er':>7} {'CRPS':>8}")
    for name in METHODS:
        for slot, lead in enumerate(LEADS):
            sp = float(np.sqrt(np.mean(np.square(spread[name][slot]))))
            er = float(np.sqrt(np.mean(np.square(error[name][slot]))))
            print(f"#   {name:>9} {lead:5.2f} {sp:8.4f} {er:8.4f} "
                  f"{sp / er:7.3f} {np.mean(score[name][slot]):8.4f}")

    print(f"LEADS = {LEADS}")
    print(f"METHODS = {METHODS}")
    print(f"MEMBERS = {MEMBERS}")
    print(f"N_CASES = {len(cases)}")
    print(f"ANALYSIS_ERROR = {per_component:.6f}")
    print(f"RANK_LEAD = {LEADS[rank_lead]}")
    for name in METHODS:
        _emit(f"SPREAD_{name}",
              [np.sqrt(np.mean(np.square(v))) for v in spread[name]], ".6f",
              per_line=len(LEADS))
        _emit(f"ERROR_{name}",
              [np.sqrt(np.mean(np.square(v))) for v in error[name]], ".6f",
              per_line=len(LEADS))
        _emit(f"CRPS_{name}", [np.mean(v) for v in score[name]], ".6f",
              per_line=len(LEADS))
        _emit(f"RANKS_{name}", ranks[name], ".1f", per_line=MEMBERS + 1)
    return amplitude


# ==========================================================================
# 3. ensemble size
# ==========================================================================
def size_sweep(cases, amplitude) -> None:
    print("\n# --- 3. diminishing returns in ensemble size ---")
    lead = 1.0
    print(f"#   CRPS at lead {lead}, EDA and Gaussian")
    curves = {"EDA": [], "GAUSS": []}
    for n_members in SIZES:
        for name in curves:
            values = []
            for index, (analysis, truth, bred, bred_orth) in enumerate(cases):
                if name == "EDA" and n_members > analysis.shape[0]:
                    # Cannot draw more EDA members than the filter ran. Sampling
                    # with replacement would duplicate members and understate
                    # the spread, which is exactly the quantity being measured.
                    values.append(np.nan)
                    continue
                built = _build_ensembles(
                    analysis, bred, bred_orth, amplitude, index,
                    n_members=n_members,
                )
                forecast = _forecast(built[name], lead)
                values.append(float(ensemble.crps(
                    forecast.T, _forecast(truth, lead)
                ).mean()))
            curves[name].append(float(np.nanmean(values)))
    print(f"#     k     " + "".join(f"{s:9d}" for s in SIZES))
    for name in curves:
        print(f"#     {name:6s}" + "".join(
            f"{v:9.4f}" if np.isfinite(v) else f"{'--':>9}" for v in curves[name]
        ))
    # Fit CRPS(k) = a + b/k, the expected form of the finite-ensemble penalty.
    for name in curves:
        finite = np.isfinite(curves[name])
        sizes = np.asarray(SIZES, dtype=float)[finite]
        values = np.asarray(curves[name])[finite]
        slope, intercept = np.polyfit(1.0 / sizes, values, 1)
        print(f"#     {name}: CRPS = {intercept:.4f} + {slope:.4f}/k"
              f"  (asymptote {intercept:.4f}; k=20 penalty "
              f"{100 * slope / 20 / intercept:.1f} %)")
        print(f"SIZE_{name}_ASYMPTOTE = {intercept:.6f}")
        print(f"SIZE_{name}_SLOPE = {slope:.6f}")
        _emit(f"SIZE_{name}", curves[name], ".6f", per_line=len(SIZES))
    print(f"SIZES = {SIZES}")
    print(f"SIZE_LEAD = {lead}")


# ==========================================================================
# 4. the Brier decomposition
# ==========================================================================
def brier_block(cases, amplitude) -> None:
    print("\n# --- 4. reliability against resolution ---")
    truths, probabilities = [], {m: [] for m in METHODS}
    for index, (analysis, truth, bred, bred_orth) in enumerate(cases):
        built = _build_ensembles(analysis, bred, bred_orth, amplitude, index)
        verify = _forecast(truth, EVENT_LEAD)
        truths.append(float(verify[EVENT_SITE]))
        for name, members in built.items():
            forecast = _forecast(members, EVENT_LEAD)
            probabilities[name].append(forecast[:, EVENT_SITE])

    threshold = float(np.percentile(truths, 70.0))
    outcomes = (np.asarray(truths) > threshold).astype(float)
    print(f"#   event: site {EVENT_SITE} above {threshold:.3f} at lead "
          f"{EVENT_LEAD} TU; base rate {outcomes.mean():.3f}")
    print(f"#   {'method':>9} {'BS':>8} {'REL':>8} {'RES':>8} {'UNC':>8} "
          f"{'identity':>10}")
    stored = {}
    for name in METHODS:
        probs = np.array([
            float(np.mean(np.asarray(row) > threshold))
            for row in probabilities[name]
        ])
        stored[name] = probs
        score = ensemble.brier_score(probs, outcomes)
        rel, res, unc = ensemble.brier_decomposition(probs, outcomes)
        print(f"#   {name:>9} {score:8.4f} {rel:8.4f} {res:8.4f} {unc:8.4f} "
              f"{abs(score - (rel - res + unc)):10.1e}")
        _emit(f"BRIER_{name}", [score, rel, res, unc], ".6f", per_line=4)
        # TWO binnings, deliberately. The decomposition above needs the exact
        # (distinct-value) bins for its identity to hold; a diagram drawn on
        # those bins is unreadable, because most of the 21 attainable
        # probabilities attract only one or two of the cases and their observed
        # frequency is then 0 or 1. The coarse binning is for the figure only.
        coarse = np.linspace(0.0, 1.0, 9)
        forecast_bin, observed_bin, counts = ensemble.reliability_diagram(
            probs, outcomes, bin_edges=coarse
        )
        _emit(f"RELDIAG_F_{name}", forecast_bin, ".5f", per_line=11)
        _emit(f"RELDIAG_O_{name}", observed_bin, ".5f", per_line=11)
        _emit(f"RELDIAG_N_{name}", counts, ".1f", per_line=11)

    # Recalibration: relabel each forecast probability with the observed
    # frequency it actually attained. Reliability should collapse; resolution
    # should not move, because the ordering of the cases is untouched.
    worst = max(METHODS, key=lambda m: ensemble.brier_decomposition(
        stored[m], outcomes)[0])
    probs = stored[worst]
    forecast_bin, observed_bin, _counts = ensemble.reliability_diagram(
        probs, outcomes
    )
    lookup = dict(zip(np.round(forecast_bin, 10), observed_bin))
    recalibrated = np.array([lookup[round(float(v), 10)] for v in probs])
    before = ensemble.brier_decomposition(probs, outcomes)
    after = ensemble.brier_decomposition(recalibrated, outcomes)
    print(f"#   recalibrating {worst} (its reliability was worst):")
    print(f"#     BS  {ensemble.brier_score(probs, outcomes):.4f} -> "
          f"{ensemble.brier_score(recalibrated, outcomes):.4f}")
    print(f"#     REL {before[0]:.5f} -> {after[0]:.5f}   "
          f"RES {before[1]:.5f} -> {after[1]:.5f}")
    print("#     NOTE: in-sample. The relabelling is fitted on the same cases it")
    print("#     is scored on, which is why reliability comes out at EXACTLY zero")
    print("#     rather than merely small. A real system fits the mapping on a")
    print("#     training period and applies it out of sample, where reliability")
    print("#     improves but does not vanish. The claim being tested here is the")
    print("#     structural one -- that resolution is untouched -- not the size of")
    print("#     the gain.")
    print(f"RECAL_METHOD = '{worst}'")
    print(f"RELDIAG_BINS = 8")
    print(f"EVENT_THRESHOLD = {threshold:.6f}")
    print(f"EVENT_BASE_RATE = {float(outcomes.mean()):.6f}")
    print(f"EVENT_LEAD = {EVENT_LEAD}")
    print(f"EVENT_SITE = {EVENT_SITE}")
    _emit("RECAL_BEFORE", [ensemble.brier_score(probs, outcomes), *before],
          ".6f", per_line=4)
    _emit("RECAL_AFTER",
          [ensemble.brier_score(recalibrated, outcomes), *after], ".6f",
          per_line=4)
    return stored, outcomes


# ==========================================================================
# 5. value
# ==========================================================================
def value_block(stored, outcomes) -> None:
    print("\n# --- 5. the cost-loss curve ---")
    # The chapter's own recommended forecast: the method with the most
    # RESOLUTION, recalibrated. Picking on raw Brier score instead selected a
    # different method from the one section 4 concludes with, which would have
    # left the chapter arguing against itself between two consecutive sections.
    best = max(
        METHODS,
        key=lambda m: ensemble.brier_decomposition(stored[m], outcomes)[1],
    )
    forecast_bin, observed_bin, _counts = ensemble.reliability_diagram(
        stored[best], outcomes
    )
    lookup = dict(zip(np.round(forecast_bin, 10), observed_bin))
    probabilistic = np.array(
        [lookup[round(float(v), 10)] for v in stored[best]]
    )
    # The deterministic counterpart: the same information, forced through one
    # threshold. Thresholding the RAW probabilities, since a deterministic
    # system never had a probability to recalibrate.
    deterministic = (stored[best] > 0.5).astype(float)

    prob_value = ensemble.value_score(probabilistic, outcomes, COST_LOSS)
    det_value = ensemble.value_score(deterministic, outcomes, COST_LOSS)
    print(f"#   probabilistic: {best}, recalibrated; deterministic: {best}'s "
          f"own ensemble majority")
    print(f"#     alpha " + "".join(f"{a:7.2f}" for a in COST_LOSS[::3]))
    print(f"#     prob  " + "".join(f"{v:7.2f}" for v in prob_value[::3]))
    print(f"#     det   " + "".join(f"{v:7.2f}" for v in det_value[::3]))
    print(f"#   positive value: prob {(prob_value > 0.05).sum()}/"
          f"{COST_LOSS.size}, det {(det_value > 0.05).sum()}/{COST_LOSS.size}")
    print(f"#   peak: prob {np.nanmax(prob_value):.3f}, "
          f"det {np.nanmax(det_value):.3f}; "
          f"worst: prob {np.nanmin(prob_value):.3f}, "
          f"det {np.nanmin(det_value):.3f}")
    print(f"VALUE_METHOD = '{best}'")
    print(f"VALUE_PROB_POSITIVE = {int((prob_value > 0.05).sum())}")
    print(f"VALUE_DET_POSITIVE = {int((det_value > 0.05).sum())}")
    print(f"COST_LOSS = {tuple(COST_LOSS)}")
    _emit("VALUE_PROB", prob_value, ".6f", per_line=10)
    _emit("VALUE_DET", det_value, ".6f", per_line=10)


if __name__ == "__main__":
    started = time.perf_counter()
    print(f"# L96 N={N_SITES} F={FORCING}; LETKF radius {LOCALISATION:g}, "
          f"inflation {INFLATION}, {MEMBERS} members")
    breeding_collapse()
    breeding_degeneracy()
    built_cases = build_cases()
    amp = score_constructions(built_cases)
    size_sweep(built_cases, amp)
    stored_probs, event_outcomes = brier_block(built_cases, amp)
    value_block(stored_probs, event_outcomes)
    print(f"\n# total {time.perf_counter() - started:.0f}s")
