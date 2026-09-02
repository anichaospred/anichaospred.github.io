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


`lorenz96_two_scale` couples `n_slow` slow variables to `n_slow * n_fast` fast ones that
move `time_ratio` times quicker and carry `1/amplitude_ratio` the amplitude, with
`lorenz96_two_scale_split` / `_state` as helpers. Two exact identities pin it: with
`coupling=0` the slow equations are **identically** `lorenz96`, and the coupling
conserves $\tfrac12(\sum X^2 + \sum Y^2)$.

Read chapter 12 before drawing conclusions from it. It shows the upscale-cascade
mechanism, but its leading Lyapunov exponent (24.7 per time unit) belongs to the fast
subsystem and overstates large-scale error growth by 8.3×, and it does **not** exhibit
Lorenz's finite predictability limit — that needs a spectrum of scales, not two.
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

## `errorgrowth` — saturation and the upscale cascade

`logistic_error_growth`, `fit_logistic_error_growth`, `saturation_level` (RMS distance
between random state pairs — i.e. the error of a climatological forecast),
`predictability_horizon`.

`cascade_rates`, `cascade_growth`, `cascade_contamination_time`, `KOLMOGOROV_ALPHA`
add Lorenz's (1969) octave-band model: band $n$ has growth rate
$\lambda_0 2^{\alpha n}$ and is forced by the band one octave smaller. With one band it
reduces exactly to `logistic_error_growth`.

The parameter that matters is $\alpha$. Seeding the finest *resolved* band at saturation
and adding octaves — which is what improving an observing system's resolution means —
the contamination time of the largest scale **converges for $\alpha > 0$** and
**diverges for $\alpha = 0$**. Lorenz 63 and Lorenz 96 are $\alpha = 0$ systems, which
is why nothing before chapter 12 could raise the question. The per-octave gain falls as
$2^{-2\alpha}$, the square of what $\sum_n \lambda_n^{-1}$ predicts.

## `dimension` — dimension from a trajectory

`correlation_sum`, `correlation_dimension`, `local_slopes`, `fit_dimension`,
`box_occupancy`, `renyi_dimension`, `renyi_spectrum`, `delay_embed`, plus the reference
sets `cantor_set`, `koch_curve`, `sierpinski_triangle` and the constants
`REFERENCE_DIMENSIONS`, `REFERENCE_WINDOWS`, `MIN_BOX_OCCUPANCY`.

Two traps, neither of which announces itself, and — measured rather than assumed —
**either sign of error is available**, which is why neither can be corrected for
afterwards:

- the **scaling window**. On the same L63 curve, fitting above 30 % of the attractor
  diameter gives 0.19, below 0.2 % gives 2.51, the whole range gives 1.92, and the
  window that works (1–5 %, the defaults) gives 2.057;
- **temporal correlation**. The excess of adjacent pairs is a *bump* in $C(r)$ at the
  distance the trajectory covers per sample. For L63 at $\Delta t = 0.01$ that lands
  inside the fit window and biases $D_2$ **high** (2.139 at `theiler=0` against 2.039 at
  `theiler=200`); sample densely enough to put it below the window and you get the
  classic low bias instead.

Always look at `local_slopes` before quoting a number.

`fit_dimension` exists because the two halves of a dimension estimate have wildly
different costs: forming $C(r)$ is $O(N^2)$ and knob-free, re-fitting a window on a
stored curve is microseconds. A chapter with a scaling-window slider computes the curve
once and calls `fit_dimension` on every drag.

`renyi_dimension` gives $D_q$ by box counting ($q=0$ box-counting, $q=1$ information,
$q=2$ correlation; $D_0 \ge D_1 \ge D_2$ always). It returns the mean **occupancy** per
scale, because that is what decides whether the answer means anything: below about ten
points per box the estimate measures the sample rather than the set. In three dimensions
at achievable sample sizes it starves, which is why `correlation_dimension` — using all
$N^2/2$ pairs rather than spreading $N$ points over a grid — is the estimator for
anything real. A second, smaller $O(\varepsilon)$ bias never goes away: the unit-box
grid lays down $(1/\varepsilon+1)^d$ boxes, not $\varepsilon^{-d}$, so prefer the
finest window the sample supports.

The reference sets are the module's calibration — Cantor, Koch and Sierpiński have
dimensions in closed form, so an estimator can be checked against a known answer rather
than against another estimator. `sierpinski_triangle` takes `probabilities` to skew the
chaos game, which leaves the support (and $D_0$) alone but makes the measure
multifractal, separating $D_0 > D_1 > D_2$ and supplying exact targets
$D_1 = -\sum p\ln p/\ln2$ and $D_2 = -\ln\sum p^2/\ln2$.

`delay_embed` reconstructs an attractor from one scalar series. The criterion
$m > 2D$ needs the $D$ being measured, so raise $m$ until the estimate saturates — for
L63's $x$ component it reads 1.72, 1.95, 1.99, 2.00 at $m = 2\ldots5$.

## `maps` — bifurcations and universality in 1-D maps

`map_orbit`, `cobweb_path`, `bifurcation_points`, `map_lyapunov_exponent`, `iterate_n`,
`superstable_cascade`, `feigenbaum_ratios`, `cycle_multiplier`, `period_three_threshold`,
`laminar_phases`, `FEIGENBAUM_DELTA`.

`bifurcation_points` and `map_lyapunov_exponent` are **vectorised over the parameter
axis**: a 1400-point sweep costs the same number of NumPy operations as one parameter,
which is why chapter 5 computes almost everything live rather than precomputing it.

Two things to know before using `superstable_cascade`:

- it locates **superstable** parameters ($f_r^{p2^n}(x_c) = x_c$, multiplier exactly
  zero), **not** bifurcation parameters. Bifurcation parameters need a marginal
  condition detected inside a basin shrinking like $\delta^{-n}$; superstable
  parameters need plain iteration from a known $x_c$. Same limit, same $\delta$;
- `r_hi` must not exceed the accumulation point by much. Above it, $g_n$ acquires
  further roots inside the periodic windows of the chaotic band and the scan can bracket
  the wrong one. Bracketing is the whole difficulty here — $g_n$ vanishes at *every*
  $R_k$ with $k \le n$, so the function takes the **first** sign change above
  $R_{n-1}$, predicted by extrapolating the previous spacing.

`base_period=3` follows a periodic window's own cascade, which is how chapter 5 turns
the self-similarity of the bifurcation diagram from an observation into a measurement.

`map_lyapunov_exponent` clips the per-iteration $\ln|f'|$ at `floor` because a
superstable parameter gives a genuine $-\infty$, not an artefact.

## `information` — predictability as information

`shannon_entropy`, `relative_entropy`, `mutual_information`, `predictive_information`,
`gaussian_relative_entropy` (closed form, separating the signal and dispersion
components).

## `spatial` — diagnostics for a field on a ring

`spatial_power_spectrum`, `dominant_wavenumber`, `phase_speed`, `spatial_correlation`,
`correlation_length`.

Lorenz 63 has no space, so an error in it has no wavelength. Lorenz 96 does, and these
are the questions that become available: what scale is the error on, how fast does the
structure move, how wide is it. Used by chapter 11 and by chapter 12's scale-dependent
error growth.

**Conventions:** the last axis is space, the leading axis is time — the same as
`systems.lorenz96`, so a trajectory from `integrate.rk4` goes straight in. Wavenumber
$m$ means $m$ waves around the ring. Speeds are in **sites per unit time**, signed, and
positive means propagation toward increasing $k$ — matching
`systems.lorenz96_dispersion`.

`phase_speed` unwraps a Fourier phase, so it assumes the phase advances by less than
$\pi$ per sample. At $\omega \approx 3$ rad per time unit, `dt = 0.01` is ample and
`dt = 1` would alias and return a confidently wrong answer, with nothing in the result
to indicate it. Check `dt` against $2\pi/\omega$.

## `plotting` — the book's figure design system

Semantic colours (`C_TRUTH`, `C_PERT`, `C_SPREAD`, `C_MEAN`, `C_FIXED`, `C_SAT`,
`C_START`, `C_CONTEXT`), plus three for data assimilation — `C_OBS` (observations),
`C_BG` (background / free forecast) and `C_ANALYSIS` (the DA estimate) — `TIME_SCALE`
for colour-as-time, and `style2d` / `style3d`.

Two named colour *maps* for fields: `MPL_DIVERGING` for anything that takes both signs
(deviations, tendencies, errors) so that zero is not a colour and the sign is legible,
and `MPL_SEQUENTIAL` for non-negative fields. A test checks that the diverging map is
light in the middle and the sequential one monotone in luminance, because swapping them
silently produces a figure that hides the sign of the field it is drawing.

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
