# `chaoslib` — what the library already provides

Shared, Pyodide-safe numerics for every chapter. Import from here rather than
re-implementing, so the equation printed in a chapter is provably the equation being
stepped, and one test suite covers all of it.

**Dependencies:** NumPy, SciPy, Plotly. Nothing else, and nothing compiled — see
[`architecture.md`](architecture.md) for why.

**Conventions:** continuous systems are `f(t, x, **params)`; trajectories carry **time
on the leading axis**; information-theoretic quantities are in **nats**; symbols follow
[`NOTATION.md`](../NOTATION.md).

## `systems` — right-hand sides and Jacobians

| Function | Notes |
|---|---|
| `lorenz63(t, x, sigma, rho, beta)` | vectorised over a leading ensemble axis |
| `lorenz63_jacobian(x, …)` | `trace = -(sigma+1+beta)` for every state |
| `lorenz63_fixed_points(rho, beta)` | returns `(origin, C_plus, C_minus)`; the pair is `None` for $\rho \le 1$ |
| `lorenz63_hopf_rho(sigma, beta)` | $\approx 24.7368$ for the classical parameters |
| `lorenz96(t, x, forcing)`, `lorenz96_jacobian(x, …)` | cyclic; last axis is the site axis |
| `pendulum`, `pendulum_energy`, `pendulum_period_exact` | exact period via `ellipk`; note SciPy takes $m=k^2$ |
| `double_pendulum`, `double_pendulum_energy` | full Euler–Lagrange equations |
| `logistic_map(x, r)`, `henon_map(xy, a, b)` | discrete maps |

## `integrate` — time stepping

| Function | Use it for |
|---|---|
| `rk4(rhs, x0, t, **params)` | **ensembles** — steps all members on one grid, no Python loop; and anywhere a uniform grid matters |
| `solve(rhs, x0, t, rtol, atol, …)` | single trajectories needing adaptive steps and tight tolerances |
| `trajectory_grid(t_final, dt)` | uniform grid landing exactly on `t_final` |

`solve` binds parameters as keywords in a closure rather than through `solve_ivp`'s
positional `args`, so it cannot silently depend on dict ordering.

## `lyapunov` — growth rates and dimension

| Function | Notes |
|---|---|
| `lyapunov_spectrum(rhs, jacobian, x0, …)` | Benettin/QR; discards a transient first |
| `finite_time_exponents(rhs, jacobian, states, tau)` | per-state local growth — the basis of flow-dependent predictability |
| `twin_trajectory_growth(rhs, x0, delta0, t)` | control vs. perturbed twin; isotropic random perturbation |
| `fit_growth_rate(t, separation, lower_mult, upper_frac)` | window anchored to $\delta_0$ below and saturation above |
| `doubling_time(rate)`, `kaplan_yorke_dimension(spectrum)`, `ks_entropy(spectrum)` | |

**Verified against the literature** (see `tests/test_chaoslib.py`): L63
$\lambda_1 = 0.9056$, $\sum\lambda_i = -13.6667$ exactly, $D_{KY} = 2.06$; L96
($N=40, F=8$) $\lambda_1 \approx 1.67$ with exactly 13 positive exponents and
$D_{KY} \approx 27.1$.

**A trap worth stating plainly:** one twin-trajectory fit is a *finite-time* exponent
along one trajectory. On the Lorenz attractor individual estimates scatter roughly
0.3–0.9 even over 15 MTU, because local growth is bursty. Averaging over initial
conditions is what recovers $\lambda_1$; `lyapunov_spectrum` does the time-averaging
properly and should be preferred when the asymptotic value is what you want.

## `adjoint` — tangent linear and adjoint models

| Function | Notes |
|---|---|
| `tangent_linear_propagator(rhs, jacobian, x0, tau, dt)` | differentiates the **discrete** RK4 map, not the continuous flow |
| `adjoint_propagator(M)` | the transpose, under the Euclidean inner product |
| `adjoint_identity_residual(M)` | should be $\sim 10^{-15}$ |
| `tangent_linear_error(…, amplitudes)` | the finite-difference validation curve |
| `singular_vectors(M, k)`, `leading_singular_value(M)` | optimal finite-time growth |

The propagator is the exact Jacobian of the numerical map the nonlinear model takes.
That distinction is not pedantry: linearising the continuous flow instead leaves an
$O(\Delta t)$ inconsistency that shows up as a *floor* in `tangent_linear_error`, and
it was a real bug here.

## `assimilate` — state estimation

`kalman_filter_update` (the linear-Gaussian optimum, used as the yardstick),
`three_dvar_update`, `four_dvar_analysis` (L-BFGS-B on the adjoint gradient),
`enkf_update` (with inflation and optional localisation), `gaspari_cohn`,
`analysis_rmse`.

`four_dvar_analysis` uses a quasi-Newton minimiser deliberately: fixed-step steepest
descent on this cost function diverges to overflow whenever $\mathbf{R}^{-1}$ is large,
which is exactly the operationally relevant case.

## `ensemble` — construction and verification

`gaussian_perturbations`, `ensemble_spread`, `ensemble_mean_error`, `rank_histogram`
(random tie-breaking), `crps` (fair energy form), `brier_score`.

Tested against the closed-form Gaussian CRPS, and against the calibration identity
that a reliable ensemble has RMS spread equal to the RMS error of its mean.

## `errorgrowth` — saturation

`logistic_error_growth`, `fit_logistic_error_growth`, `saturation_level` (RMS distance
between random state pairs — i.e. the error of a climatological forecast),
`predictability_horizon`.

## `dimension` — dimension from a trajectory

`correlation_sum`, `correlation_dimension`, `local_slopes`.

Two traps, both of which bias $D_2$ **low** and neither of which announces itself: the
**scaling window** (for L63 it is roughly 1–5 % of the attractor diameter — the
defaults) and **temporal correlation** (use `theiler`, or subsample in time). Always
look at `local_slopes` before quoting a number; a real fractal shows a plateau.

## `information` — predictability as information

`shannon_entropy`, `relative_entropy`, `mutual_information`, `predictive_information`,
`gaussian_relative_entropy` (closed form, separating the signal and dispersion
components).

## `plotting` — the book's figure design system

Semantic colours (`C_TRUTH`, `C_PERT`, `C_SPREAD`, `C_MEAN`, `C_FIXED`, `C_SAT`,
`C_START`, `C_CONTEXT`), plus three for data assimilation — `C_OBS` (observations),
`C_BG` (background / free forecast) and `C_ANALYSIS` (the DA estimate) — `TIME_SCALE`
for colour-as-time, and `style2d` / `style3d`.

**Two figure kinds.** `style2d`/`style3d` style *Plotly* panels, for chapters where
hovering over a curve is part of the point. `mpl_panels`, `mpl_grid` and `finish_mpl`
build *static matplotlib* figures, used for phase-space projections — a rotatable 3-D
scene of the Lorenz attractor reads worse than a fixed x-z projection, and costs more.

`mpl_colour` translates a palette entry for matplotlib. **Always route palette colours
through it in a matplotlib figure:** the palette is written in CSS because Plotly eats
it directly, and two entries (`C_CONTEXT`, `SCENE_BG`) use the `rgba(...)` form that
matplotlib rejects at *render* time — so the cell fails while the import succeeds.

The palette is tested for pairwise perceptual separation, because a DA figure shows six
of these at once, and every entry is tested for matplotlib usability. One documented exception is allowed (`C_PERT` and `C_SAT`, separated
by line style rather than hue); everything else must clear a minimum RGB distance, so a
new colour cannot be added as a shade of an existing one.

Each colour means one thing across every chapter. Import them; do not choose colours
per figure.
