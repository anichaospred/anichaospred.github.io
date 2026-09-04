#!/usr/bin/env python3
r"""Precompute chapter 18's variational-assimilation diagnostics.

One configuration throughout: Lorenz 63, observations of the full state every
0.1 TU with :math:`\sigma_o = 2`, and a background covariance
:math:`\mathbf{B} = \Sigma_{\mathrm{clim}}/64`, giving a background error near
1.0 per component. That last choice matters and was not the first one tried.
With :math:`\mathbf{B} = \Sigma_{\mathrm{clim}}/4` -- a background as bad as a
climatological guess -- the 4D-Var cost function is so multimodal that a third of
the cases end up *worse* than the background, Gauss-Newton makes essentially no
progress, and every measurement below is dominated by minimiser failure rather
than by the physics it is meant to isolate. An operational background is a short
forecast, not a climatology, and the chapter is written for that regime.

Six blocks.

1. **The cost landscape** at three window lengths, with :math:`J_b` and
   :math:`J_o` stored *separately* so the notebook's slider can weaken
   :math:`\mathbf{B}` and re-mix them without recomputing a trajectory:
   :math:`\mathbf{B}\to c\mathbf{B}` divides :math:`J_b` by :math:`c` and
   leaves :math:`J_o` untouched. The count of local minima on the slice is
   emitted with each, because it rises from 2 to 16 as the window grows from 0.5
   to 2.0 TU and that is the mechanism behind block 5's failure rate.
2. **The gradient test**, run on the true gradient and two broken ones -- and run
   at a point *displaced* from the background, because at :math:`x_0=x^b` the
   term :math:`\mathbf{B}^{-1}(x_0-x^b)` vanishes and the test cannot see an
   error in it at all. That blind spot is measured here rather than asserted.
3. **Observing one variable.** 3D-Var against 4D-Var when only :math:`x` is
   measured.
4. **The Hessian**: the analysis-error covariance around the attractor, and the
   chapter's central comparison -- the same six observations concentrated at the
   analysis time against spread over a window, as a function of forecast lead.
5. **Window length**, the chapter's knob.
6. **Convergence**: the incremental (Gauss-Newton) outer loop against L-BFGS,
   from backgrounds of three different qualities.

Run from chaos-book/:
    python3 scripts/generate_ch18_data.py        # ~10 minutes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaoslib import adjoint, assimilate, integrate, systems  # noqa: E402

DT = 0.01
SPACING = 0.1
OBS_SIGMA = 2.0
B_DIVISOR = 64.0
N_CASES = 24
WINDOWS = (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0)
REFERENCE_WINDOW = 0.5
LANDSCAPE_WINDOWS = (0.5, 1.0, 2.0)
HESSIAN_TIMES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
LEADS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5)
LAMBDA1 = 0.906


def _emit(name: str, values, fmt: str = ".6e", per_line: int = 8) -> None:
    items = [
        'float("nan")' if not np.isfinite(v) else format(float(v), fmt)
        for v in np.ravel(values)
    ]
    print(f"{name} = (")
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i : i + per_line]) + ",")
    print(")")


def _attractor() -> np.ndarray:
    return integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]),
        integrate.trajectory_grid(80.0, DT),
    )[2000:]


def _propagate(x0, t_obs: float) -> np.ndarray:
    if t_obs <= 0.0:
        return np.asarray(x0, dtype=float)
    grid = np.linspace(0.0, t_obs, int(round(t_obs / DT)) + 1)
    return integrate.rk4(systems.lorenz63, x0, grid)[-1]


def _observations(truth0, times, h_op, rng, sigma=OBS_SIGMA):
    n_obs = np.atleast_2d(h_op).shape[0]
    return [
        (float(t), np.atleast_2d(h_op) @ _propagate(truth0, float(t))
         + rng.normal(0.0, sigma, n_obs))
        for t in times
    ]


# ==========================================================================
# 1. the cost landscape
# ==========================================================================
def _count_local_minima(field: np.ndarray) -> int:
    """Strict interior local minima of a 2-D grid, 8-neighbour.

    Vectorised: a cell is a strict minimum when it is below all eight shifted
    copies of itself. The loop version of this is 61x61x9 Python comparisons per
    field and was measurably slower than the whole cost evaluation it summarises.
    """
    inner = field[1:-1, 1:-1]
    lower = np.ones_like(inner, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = field[1 + di : field.shape[0] - 1 + di,
                            1 + dj : field.shape[1] - 1 + dj]
            lower &= inner < shifted
    return int(lower.sum())


def landscape(attractor, b_cov) -> None:
    print("# --- 1. the cost landscape, at three window lengths ---")
    h_op = np.eye(3)
    r_cov = np.eye(3) * OBS_SIGMA**2
    b_inv, r_inv = np.linalg.inv(b_cov), np.linalg.inv(r_cov)

    truth0 = attractor[600]
    rng = np.random.default_rng(3)
    xb = truth0 + rng.multivariate_normal(np.zeros(3), b_cov)

    span, n_grid = 10.0, 61
    xs = np.linspace(truth0[0] - span, truth0[0] + span, n_grid)
    ys = np.linspace(truth0[1] - span, truth0[1] + span, n_grid)

    # Jb does not depend on the window, so it is computed once. It is also a
    # pure quadratic, but it is stored on the same grid as Jo rather than
    # rebuilt in the notebook, so that the slider mixes two arrays of identical
    # provenance.
    jb = np.empty((n_grid, n_grid))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            departure = np.array([xv, yv, truth0[2]]) - xb
            jb[i, j] = 0.5 * float(departure @ (b_inv @ departure))
    print(f"LAND_N = {n_grid}")
    print(f"LAND_WINDOWS = {LANDSCAPE_WINDOWS}")
    print(f"LAND_TRUTH = {tuple(round(float(v), 6) for v in truth0)}")
    print(f"LAND_BACKGROUND = {tuple(round(float(v), 6) for v in xb)}")
    _emit("LAND_X", xs, ".4f", per_line=10)
    _emit("LAND_Y", ys, ".4f", per_line=10)
    _emit("LAND_JB", jb.ravel(), ".5e", per_line=n_grid)

    analyses, minima_o, minima_j = [], [], []
    for window in LANDSCAPE_WINDOWS:
        started = time.perf_counter()
        times = np.arange(0.0, window + 1e-9, SPACING)
        observations = _observations(truth0, times, h_op, np.random.default_rng(3))
        # ONE integration per grid point, sampled at the observation times --
        # not one integration per observation. The window-2.0 grid takes about
        # two and a half minutes the naive way and eight seconds this way, and
        # the sampled states are identical because both march at exactly dt.
        n_steps = int(round(window / DT))
        grid = np.linspace(0.0, window, n_steps + 1)
        indices = [int(round(float(t) / DT)) for t in times]
        jo = np.empty((n_grid, n_grid))
        for i, yv in enumerate(ys):
            for j, xv in enumerate(xs):
                trajectory = integrate.rk4(
                    systems.lorenz63, np.array([xv, yv, truth0[2]]), grid
                )
                total = 0.0
                for index, (_t, y_obs) in zip(indices, observations):
                    innovation = y_obs - h_op @ trajectory[index]
                    total += 0.5 * float(innovation @ (r_inv @ innovation))
                jo[i, j] = total

        xa = assimilate.four_dvar_analysis(
            systems.lorenz63, systems.lorenz63_jacobian, xb, b_cov,
            observations, h_op, r_cov, dt=DT, max_iterations=200,
        )
        analyses.append(xa)
        minima_o.append(_count_local_minima(jo))
        minima_j.append(_count_local_minima(jb + jo))
        print(f"#   window {window:.1f} ({times.size:2d} obs, "
              f"{time.perf_counter() - started:5.1f}s): local minima of Jo "
              f"{minima_o[-1]:3d}, of J {minima_j[-1]:3d}; "
              f"|xa-truth| {np.linalg.norm(xa - truth0):.4f}")
        _emit(f"LAND_JO_{int(round(window * 10)):03d}", jo.ravel(), ".5e",
              per_line=n_grid)

    print(f"#   |xb - truth| = {np.linalg.norm(xb - truth0):.4f}")
    print(f"LAND_KEYS = {tuple(f'{int(round(w * 10)):03d}' for w in LANDSCAPE_WINDOWS)}")
    print(f"LAND_MINIMA_JO = {tuple(minima_o)}")
    print(f"LAND_MINIMA_J = {tuple(minima_j)}")
    _emit("LAND_ANALYSES", np.asarray(analyses).ravel(), ".6f", per_line=3)


# ==========================================================================
# 2. the gradient test, and its blind spot
# ==========================================================================
def gradient_curves(attractor, b_cov) -> None:
    print("\n# --- 2. the gradient test ---")
    h_op, r_cov = np.eye(3), np.eye(3) * OBS_SIGMA**2
    truth0 = attractor[900]
    rng = np.random.default_rng(4)
    xb = truth0 + rng.multivariate_normal(np.zeros(3), b_cov)
    observations = _observations(
        truth0, np.arange(0.0, 0.8 + 1e-9, 0.2), h_op, rng
    )
    b_inv = np.linalg.inv(b_cov)

    def exact(x0):
        return assimilate.four_dvar_cost(
            systems.lorenz63, systems.lorenz63_jacobian, x0, xb, b_cov,
            observations, h_op, r_cov, dt=DT,
        )

    def scaled(x0):
        """Every component 1 % too large -- a units or weighting slip."""
        value, grad = exact(x0)
        return value, grad * 1.01

    def no_background(x0):
        """The background term present in J but missing from grad J -- the
        classic omission when the two are coded in different places."""
        value, grad = exact(x0)
        return value, grad - b_inv @ (np.asarray(x0, dtype=float).ravel() - xb)

    alphas = np.logspace(-14.0, -1.0, 27)
    middle = (alphas > 4e-12) & (alphas < 6e-7)
    displaced = xb + np.array([1.5, -1.0, 2.0])
    print(f"#   test point displaced from xb by "
          f"{np.linalg.norm(displaced - xb):.3f}")
    for label, fn in (("EXACT", exact), ("SCALED", scaled), ("NOBG", no_background)):
        _, phi = adjoint.gradient_test(fn, displaced, alphas=alphas)
        error = np.abs(phi - 1.0)
        print(f"#   {label:6s}: floor {error.min():.3e} at "
              f"alpha={alphas[int(np.argmin(error))]:.1e}, middle decades span "
              f"a factor {error[middle].max() / error[middle].min():.3g}")
        _emit(f"GRAD_{label}", error, ".6e")
    _emit("GRAD_ALPHAS", alphas, ".6e")

    # The blind spot, measured. BOTH curves are emitted: the figure has to plot
    # the exact curve *at the same test point* to make the claim, and a first
    # pass plotted the displaced-point curve against this one, which separated
    # them visibly at large alpha and contradicted the very thing being shown.
    _, phi_exact = adjoint.gradient_test(exact, xb, alphas=alphas)
    _, phi_nobg = adjoint.gradient_test(no_background, xb, alphas=alphas)
    blind = bool(np.array_equal(phi_exact, phi_nobg))
    print(f"#   run AT xb instead, the broken and correct curves are bitwise "
          f"identical: {blind}")
    print(f"GRAD_BLIND_AT_BACKGROUND = {blind}")
    _emit("GRAD_BLIND_EXACT", np.abs(phi_exact - 1.0), ".6e")
    _emit("GRAD_BLIND_NOBG", np.abs(phi_nobg - 1.0), ".6e")
    print(f"SQRT_EPS = {np.sqrt(np.finfo(float).eps):.6e}")


# ==========================================================================
# 3. observing one variable
# ==========================================================================
def partial_observation(attractor, b_cov) -> None:
    print("\n# --- 3. observing x alone ---")
    h_op, r_cov = np.array([[1.0, 0.0, 0.0]]), np.array([[OBS_SIGMA**2]])
    times = np.arange(0.0, REFERENCE_WINDOW + 1e-9, SPACING)

    errors = {"BKG": [], "THREE": [], "FOUR": []}
    for case in range(N_CASES):
        rng = np.random.default_rng(100 + case)
        truth0 = attractor[400 + 137 * case]
        xb = truth0 + rng.multivariate_normal(np.zeros(3), b_cov)
        observations = _observations(truth0, times, h_op, rng)
        errors["BKG"].append(xb - truth0)
        errors["THREE"].append(assimilate.three_dvar_update(
            xb, np.diag(np.diag(b_cov)), observations[0][1], h_op, r_cov
        ) - truth0)
        errors["FOUR"].append(assimilate.four_dvar_analysis(
            systems.lorenz63, systems.lorenz63_jacobian, xb, b_cov,
            observations, h_op, r_cov, dt=DT, max_iterations=150,
        ) - truth0)

    print(f"#   RMS error per component, {N_CASES} cases, window "
          f"{REFERENCE_WINDOW} ({times.size} observations of x)")
    for key in ("BKG", "THREE", "FOUR"):
        per = np.sqrt(np.mean(np.asarray(errors[key]) ** 2, axis=0))
        print(f"#     {key:6s} x {per[0]:7.4f}  y {per[1]:7.4f}  z {per[2]:7.4f}")
        _emit(f"PARTIAL_{key}", per, ".6f", per_line=3)
    identical = bool(np.array_equal(
        np.asarray(errors["THREE"])[:, 1:], np.asarray(errors["BKG"])[:, 1:]
    ))
    print(f"#   3D-Var left y and z bitwise unchanged: {identical}")
    print(f"PARTIAL_THREE_UNCHANGED = {identical}")
    print(f"PARTIAL_CASES = {N_CASES}")
    print(f"PARTIAL_WINDOW = {REFERENCE_WINDOW}")
    print(f"PARTIAL_NOBS = {times.size}")


# ==========================================================================
# 4. the Hessian
# ==========================================================================
def hessian_structure(attractor, b_cov) -> None:
    print("\n# --- 4. the analysis-error covariance ---")
    h_op, r_cov = np.eye(3), np.eye(3) * OBS_SIGMA**2
    r_inv = np.linalg.inv(r_cov)
    n_obs = len(HESSIAN_TIMES)
    climatology = np.cov(attractor.T)
    starts = list(range(0, 6000, 150))

    def aniso_of(cov):
        values = np.linalg.eigvalsh(cov)
        return float(np.sqrt(values[-1] / values[0]))

    # Everything here is measured at TWO background qualities, because the first
    # pass measured it at one and reported a conclusion that does not survive the
    # other. With a loose B the observations shape the analysis covariance and
    # its least-uncertain axis lands within a few degrees of chapter 16's leading
    # singular vector; with a tight B the background shapes it instead and the
    # angle opens to more than 50 degrees. The alignment is real but conditional,
    # and saying so is the whole point of measuring it twice.
    for label, divisor in (("TIGHT", B_DIVISOR), ("LOOSE", 4.0)):
        b_here = climatology / divisor
        three_cov = np.linalg.inv(np.linalg.inv(b_here) + h_op.T @ r_inv @ h_op)
        # The control: the SAME number of observations, all at the analysis time.
        dense_cov = np.linalg.inv(
            np.linalg.inv(b_here) + n_obs * h_op.T @ r_inv @ h_op
        )

        aniso, aligned, orthogonal, traces = [], [], [], []
        ratios = {lead: [] for lead in LEADS}
        for start in starts:
            state = attractor[start]
            analysis_cov = np.linalg.inv(assimilate.four_dvar_hessian(
                systems.lorenz63, systems.lorenz63_jacobian, state, b_here,
                HESSIAN_TIMES, h_op, r_cov, dt=DT,
            ))
            values, vectors = np.linalg.eigh(analysis_cov)
            propagator = adjoint.tangent_linear_propagator(
                systems.lorenz63, systems.lorenz63_jacobian, state,
                HESSIAN_TIMES[-1], dt=DT,
            )
            _, initial_vectors, _ = adjoint.singular_vectors(propagator, n_vectors=1)
            leading = initial_vectors[:, 0]
            aniso.append(aniso_of(analysis_cov))
            aligned.append(float(np.degrees(np.arccos(
                min(1.0, abs(float(vectors[:, 0] @ leading)))))))
            orthogonal.append(float(np.degrees(np.arccos(
                min(1.0, abs(float(vectors[:, -1] @ leading)))))))
            traces.append(float(np.trace(analysis_cov)))
            for lead in LEADS:
                prop = np.eye(3) if lead == 0.0 else adjoint.tangent_linear_propagator(
                    systems.lorenz63, systems.lorenz63_jacobian, state, lead, dt=DT
                )
                spread = float(np.trace(prop @ analysis_cov @ prop.T))
                concentrated = float(np.trace(prop @ dense_cov @ prop.T))
                ratios[lead].append(spread / concentrated)

        # PAIRED ratios, summarised by quartiles. The ratio of the medians is a
        # different and much smaller number here (0.18 against 0.71 at lead 1.0)
        # because the traces are heavy-tailed across the attractor; the paired
        # ratio is the one that answers "for this state, which is better".
        median = [float(np.median(ratios[lead])) for lead in LEADS]
        lower = [float(np.percentile(ratios[lead], 25)) for lead in LEADS]
        upper = [float(np.percentile(ratios[lead], 75)) for lead in LEADS]

        print(f"#   --- B = clim/{divisor:g} ({label}) ---")
        print(f"#   3D-Var  (1 obs) : anisotropy {aniso_of(three_cov):.3f}, "
              f"trace {np.trace(three_cov):.4f} -- one ellipse, everywhere, always")
        print(f"#   {n_obs} obs at t=0  : anisotropy {aniso_of(dense_cov):.3f}, "
              f"trace {np.trace(dense_cov):.4f}")
        print(f"#   {n_obs} obs spread : anisotropy {min(aniso):.2f}-{max(aniso):.2f} "
              f"(median {np.median(aniso):.2f}), trace median "
              f"{np.median(traces):.4f}")
        print(f"#   angle to chapter 16's leading singular vector, "
              f"{len(starts)} points:")
        print(f"#     least-uncertain axis  median {np.median(aligned):5.1f} deg "
              f"[{min(aligned):.0f}, {max(aligned):.0f}]")
        print(f"#     most-uncertain  axis  median {np.median(orthogonal):5.1f} deg "
              f"[{min(orthogonal):.0f}, {max(orthogonal):.0f}]")
        print(f"#   paired ratio of forecast-error trace, spread / concentrated:")
        print(f"#     lead   " + "".join(f"{l:8.2f}" for l in LEADS))
        print(f"#     median " + "".join(f"{v:8.3f}" for v in median))
        print(f"#     IQR    " + "".join(
            f"{lo:5.2f}-{hi:.2f}" for lo, hi in zip(lower, upper)))

        print(f"HESS_{label}_DIVISOR = {divisor}")
        print(f"HESS_{label}_THREE_ANISO = {aniso_of(three_cov):.6f}")
        print(f"HESS_{label}_THREE_TRACE = {float(np.trace(three_cov)):.6f}")
        print(f"HESS_{label}_DENSE_ANISO = {aniso_of(dense_cov):.6f}")
        print(f"HESS_{label}_DENSE_TRACE = {float(np.trace(dense_cov)):.6f}")
        _emit(f"HESS_{label}_ANISO", aniso, ".5f", per_line=10)
        _emit(f"HESS_{label}_ALIGNED", aligned, ".4f", per_line=10)
        _emit(f"HESS_{label}_ORTHOGONAL", orthogonal, ".4f", per_line=10)
        _emit(f"HESS_{label}_TRACE", traces, ".5f", per_line=10)
        _emit(f"LEAD_{label}_MEDIAN", median, ".6f", per_line=7)
        _emit(f"LEAD_{label}_LOWER", lower, ".6f", per_line=7)
        _emit(f"LEAD_{label}_UPPER", upper, ".6f", per_line=7)

    print(f"HESS_TIMES = {tuple(HESSIAN_TIMES)}")
    print(f"HESS_POINTS = {len(starts)}")
    print(f"HESS_LABELS = ('TIGHT', 'LOOSE')")
    print(f"LEADS = {LEADS}")

    # The two exact equivalences, on a linear system.
    a_mat = np.array([[-0.5, 2.0, 0.0], [-2.0, -0.5, 0.3], [0.0, -0.3, -0.2]])
    lin_rhs = lambda t, x, **k: a_mat @ np.asarray(x)   # noqa: E731
    lin_jac = lambda x, **k: a_mat                       # noqa: E731
    tau = 0.7
    b_lin = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, -0.4], [0.5, -0.4, 2.0]])
    h_lin = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r_lin = np.diag([0.5, 0.8])
    xb_lin, y_lin = np.array([1.0, -2.0, 0.5]), np.array([2.3, -0.4])
    grid = np.linspace(0.0, tau, int(round(tau / DT)) + 1)
    propagator = adjoint.tangent_linear_propagator(lin_rhs, lin_jac, xb_lin, tau, dt=DT)
    kalman_mean, kalman_cov = assimilate.kalman_filter_update(
        integrate.rk4(lin_rhs, xb_lin, grid)[-1],
        propagator @ b_lin @ propagator.T, y_lin, h_lin, r_lin,
    )
    xa_inc, _ = assimilate.incremental_four_dvar(
        lin_rhs, lin_jac, xb_lin, b_lin, [(tau, y_lin)], h_lin, r_lin,
        dt=DT, outer_iterations=1,
    )
    mean_gap = float(np.abs(
        integrate.rk4(lin_rhs, xa_inc, grid)[-1] - kalman_mean).max())
    cov_gap = float(np.abs(
        propagator @ np.linalg.inv(assimilate.four_dvar_hessian(
            lin_rhs, lin_jac, xb_lin, b_lin, [tau], h_lin, r_lin, dt=DT
        )) @ propagator.T - kalman_cov).max())
    print(f"#   linear model, one outer step: |mean - Kalman| = {mean_gap:.2e}, "
          f"|M A M^T - Kalman cov| = {cov_gap:.2e}")
    print(f"KALMAN_MEAN_GAP = {mean_gap:.6e}")
    print(f"KALMAN_COV_GAP = {cov_gap:.6e}")


# ==========================================================================
# 5. window length
# ==========================================================================
def window_sweep(attractor, b_cov) -> None:
    print("\n# --- 5. window length, the chapter's knob ---")
    h_op, r_cov = np.eye(3), np.eye(3) * OBS_SIGMA**2

    # Fixed truths, fixed background errors, NESTED observations: a longer window
    # contains the shorter windows' observations, so the only thing that varies
    # is the window. A first pass drew a fresh background per window; its
    # background-error column wandered between 4.6 and 7.3 and made every
    # difference between windows uninterpretable.
    all_times = np.arange(0.0, max(WINDOWS) + 1e-9, SPACING)
    cases = []
    for case in range(N_CASES):
        rng = np.random.default_rng(200 + case)
        truth0 = attractor[500 + 173 * case]
        xb = truth0 + rng.multivariate_normal(np.zeros(3), b_cov)
        noise = rng.normal(0.0, OBS_SIGMA, (all_times.size, 3))
        cases.append((truth0, xb, [
            (float(t), h_op @ _propagate(truth0, float(t)) + noise[i])
            for i, t in enumerate(all_times)
        ]))

    background = float(np.sqrt(np.mean(
        [np.sum((xb - truth0) ** 2) for truth0, xb, _ in cases]
    )))
    print(f"#   background error, identical for every window: {background:.4f}")
    print(f"#   {'window':>7} {'nobs':>5} {'analysis':>9} {'worse':>7} {'s':>6}")

    errors, worse_fraction = [], []
    for window in WINDOWS:
        started = time.perf_counter()
        n_obs = int(round(window / SPACING)) + 1
        squared, worse = [], 0
        for truth0, xb, observations in cases:
            xa = assimilate.four_dvar_analysis(
                systems.lorenz63, systems.lorenz63_jacobian, xb, b_cov,
                observations[:n_obs], h_op, r_cov, dt=DT, max_iterations=120,
            )
            squared.append(float(np.sum((xa - truth0) ** 2)))
            worse += float(np.linalg.norm(xa - truth0)) > float(
                np.linalg.norm(xb - truth0))
        errors.append(float(np.sqrt(np.mean(squared))))
        worse_fraction.append(worse / len(cases))
        print(f"#   {window:7.2f} {n_obs:5d} {errors[-1]:9.4f} "
              f"{worse_fraction[-1]:7.2f} {time.perf_counter() - started:6.1f}")

    best = int(np.argmin(errors))
    print(f"#   optimum at {WINDOWS[best]:.2f} TU = "
          f"{WINDOWS[best] * LAMBDA1:.2f} e-folding times: error "
          f"{errors[best]:.4f}, against {errors[0]:.4f} at the shortest window "
          f"and {errors[-1]:.4f} at {WINDOWS[-1]:.1f}")
    print(f"WINDOWS = {WINDOWS}")
    print(f"WINDOW_BACKGROUND = {background:.6f}")
    print(f"WINDOW_BEST = {WINDOWS[best]}")
    print(f"WINDOW_CASES = {N_CASES}")
    print(f"LAMBDA1 = {LAMBDA1}")
    _emit("WINDOW_ERROR", errors, ".6f", per_line=8)
    _emit("WINDOW_WORSE", worse_fraction, ".4f", per_line=8)


# ==========================================================================
# 6. convergence
# ==========================================================================
def convergence(attractor, _b_cov) -> None:
    print("\n# --- 6. incremental (Gauss-Newton) against L-BFGS ---")
    climatology = np.cov(attractor.T)
    h_op, r_cov = np.eye(3), np.eye(3) * OBS_SIGMA**2
    truth0 = attractor[1500]
    outer = 6
    qualities = (("GOOD", 64.0), ("FAIR", 16.0), ("POOR", 4.0))
    # The same observations for all three, so only the background quality varies.
    observations = _observations(
        truth0, np.arange(0.0, REFERENCE_WINDOW + 1e-9, SPACING),
        h_op, np.random.default_rng(21),
    )

    for label, divisor in qualities:
        b_cov = climatology / divisor
        rng = np.random.default_rng(9)
        xb = truth0 + rng.multivariate_normal(np.zeros(3), b_cov)
        xa_inc, inc_costs = assimilate.incremental_four_dvar(
            systems.lorenz63, systems.lorenz63_jacobian, xb, b_cov, observations,
            h_op, r_cov, dt=DT, outer_iterations=outer,
        )
        history: list[float] = []
        xa_lbfgs = assimilate.four_dvar_analysis(
            systems.lorenz63, systems.lorenz63_jacobian, xb, b_cov, observations,
            h_op, r_cov, dt=DT, max_iterations=200, history=history,
        )
        # A running minimum: L-BFGS evaluates trial points that are worse, and a
        # plot of raw evaluations is not a convergence plot.
        running = np.minimum.accumulate(np.asarray(history, dtype=float))
        print(f"#   {label} background (clim/{divisor:g}, "
              f"|xb-truth| = {np.linalg.norm(xb - truth0):.3f}):")
        print(f"#     incremental {outer} outer: J {inc_costs[0]:8.2f} -> "
              f"{inc_costs[-1]:9.4f},  error {np.linalg.norm(xa_inc - truth0):.4f}")
        print(f"#     L-BFGS {len(history):3d} evals: J {running[0]:8.2f} -> "
              f"{running[-1]:9.4f},  error "
              f"{np.linalg.norm(xa_lbfgs - truth0):.4f}")
        print(f"#     |xa_inc - xa_lbfgs| = "
              f"{np.linalg.norm(xa_inc - xa_lbfgs):.3e}")
        print(f"#     inc trace: " + " ".join(f"{c:.4g}" for c in inc_costs))
        print(f"INC_{label}_DIVISOR = {divisor}")
        print(f"INC_{label}_BACKGROUND_ERROR = "
              f"{float(np.linalg.norm(xb - truth0)):.6f}")
        print(f"INC_{label}_GAP = {float(np.linalg.norm(xa_inc - xa_lbfgs)):.6e}")
        print(f"INC_{label}_EVALS = {len(history)}")
        _emit(f"INC_{label}_COSTS", inc_costs, ".6e", per_line=8)
        _emit(f"LBFGS_{label}_COSTS", running, ".6e", per_line=8)
    print(f"INC_OUTER = {outer}")
    print(f"INC_LABELS = {tuple(q[0] for q in qualities)}")


if __name__ == "__main__":
    started = time.perf_counter()
    trajectory = _attractor()
    background_cov = np.cov(trajectory.T) / B_DIVISOR
    print(f"# B = clim/{B_DIVISOR:g}; sigma_b per component = "
          f"{np.sqrt(np.diag(background_cov)).round(4).tolist()}")
    print(f"# observation error sigma_o = {OBS_SIGMA}")
    print(f"B_DIVISOR = {B_DIVISOR}")
    print(f"OBS_SIGMA = {OBS_SIGMA}")
    print(f"SPACING = {SPACING}")
    landscape(trajectory, background_cov)
    gradient_curves(trajectory, background_cov)
    partial_observation(trajectory, background_cov)
    hessian_structure(trajectory, background_cov)
    window_sweep(trajectory, background_cov)
    convergence(trajectory, background_cov)
    print(f"\n# total {time.perf_counter() - started:.0f}s")
