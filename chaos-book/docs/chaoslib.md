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


`rk4_stochastic` integrates $dx = f(x)\,dt + \sigma\,dW$ -- RK4 on the drift,
Euler-Maruyama on the noise. `noise_std` is the diffusion coefficient, per **square root**
of time, which is the usual place to go wrong; the tests pin it against the exact
Ornstein-Uhlenbeck stationary variance $\sigma^2/2a$. At `noise_std=0` it reduces to
`rk4` bit-for-bit, so both arms of a perfect-model/imperfect-model comparison share one
discretisation. The noise term is first order while the drift is fourth.

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

`gradient_test` is the check operational centres run before trusting an adjoint:
$\Phi(\alpha)=[J(x_0+\alpha h)-J(x_0)]/(\alpha h^{\top}\nabla J)$ must approach 1.
Plotted as $|\Phi-1|$ it makes a V — curvature on the right, cancellation error on the
left — and **the diagnostic is the depth of the trough, not its existence**. A wrong
gradient turns upward at small $\alpha$ too, because that branch comes from evaluating
$J$ and knows nothing about the gradient; what a wrong gradient cannot do is descend.
Measured in chapter 18: a correct gradient reaches $3\times10^{-8}$, one wrong by 1 %
sits flat at $10^{-2}$.

## `assimilate` — state estimation

`kalman_filter_update` (the linear-Gaussian optimum, used as the yardstick),
`three_dvar_update`, `four_dvar_cost`, `four_dvar_hessian`,
`incremental_four_dvar`, `four_dvar_analysis` (L-BFGS-B on the adjoint gradient),
`enkf_update` (stochastic, with inflation, optional localisation and an optional
`background_cov` override), `etkf_update`, `letkf_update`, `hybrid_covariance`,
`ring_localisation`, `gaspari_cohn`, `analysis_rmse`.

`four_dvar_cost` is public because the cost surface and the gradient are objects to
look at, not only to minimise; `chaoslib.adjoint.gradient_test` consumes it directly.
It returns `(inf, 0)` rather than `(nan, nan)` on a diverged trajectory, because a
line search backs off from infinity and steps straight past NaN.

`four_dvar_hessian` returns the Gauss-Newton Hessian
$\mathbf{B}^{-1}+\sum_k\mathbf{M}_k^{\top}\mathbf{H}^{\top}\mathbf{R}^{-1}
\mathbf{H}\mathbf{M}_k$, whose inverse is the analysis-error covariance. It depends
on *when and where* the observations are, never on their values — which is what makes
observation targeting possible — and it is flow-dependent even though $\mathbf{B}$ is
fixed. Two exact checks anchor it: for linear dynamics, $\mathbf{M}\mathbf{A}
\mathbf{M}^{\top}$ **is** the Kalman analysis covariance, and `incremental_four_dvar`
with one outer iteration **is** the Kalman analysis mean, both to $10^{-15}$. They hold
exactly rather than approximately because the tangent is stepped through the same RK4
stages as the nonlinear model, so for a linear right-hand side the two are the same
matrix polynomial.

`incremental_four_dvar` is Gauss-Newton: the inner normal equations' right-hand side
*is* minus the outer gradient, so the increment is
$\delta x = -\mathbf{A}\nabla J$. Recognising that is worth more than memorising the
algorithm — it explains at once why one outer iteration is exact for a linear model and
why the scheme inherits Gauss-Newton's failure mode when the dropped second-derivative
term is large. Its inner solve is a direct factorisation, honest at $n=3$ and impossible
at $n=10^8$; operationally the inner loop is itself a conjugate-gradient minimisation.

`four_dvar_analysis` uses a quasi-Newton minimiser deliberately: fixed-step steepest
descent on this cost function diverges to overflow whenever $\mathbf{R}^{-1}$ is large,
which is exactly the operationally relevant case. Pass a list as `history` to collect
$J$ at each iterate — and take a running minimum before plotting it, because L-BFGS
evaluates trial points that are worse and a plot of raw evaluations is not a convergence
plot.

### The ensemble filters

`etkf_update` is the deterministic square-root filter. Its anchor is exact and holds at
**every** ensemble size, including $k < n$ where the sample covariance is singular: the
ETKF analysis mean and covariance are the Kalman filter's *for the covariance the ensemble
actually has*, to $10^{-11}$. That is what distinguishes a square-root filter from an
approximation to one. It uses the **symmetric** matrix square root, not a Cholesky
factor — both satisfy $\mathbf{W}\mathbf{W}^{\top}=(k-1)\tilde{\mathbf{P}}^a$, but only
the symmetric one leaves the analysis mean where the update put it, and a triangular
factor shifts the mean while still giving the right covariance, which is easy to miss.

`letkf_update` solves one ETKF per state variable, scaling $\mathbf{R}^{-1}$ by a
distance weight — R-localisation, requiring uncorrelated observation errors, which is why
it takes a scalar `obs_variance` rather than a matrix. It is a genuinely different
operation from covariance localisation, and the difference is about **rank**. A global
filter's increment lies *exactly* in the span of the $k-1$ ensemble perturbations
(verified to $10^{-12}$), and no amount of tapering in covariance space changes that; the
local filter assembles its increment from $n$ separate problems, and at $k=10$ on a
40-site ring more than a third of it lies outside the ensemble span.

`hybrid_covariance` is the other route out of rank deficiency:
$\beta\mathbf{P}^e+(1-\beta)\mathbf{B}$ is full rank for any $\beta<1$ because
$\mathbf{B}$ is. Pass the result as `enkf_update(..., background_cov=...)`. **The static
covariance must be tuned**, not just climatological: on Lorenz 96 raw climatology has a
spread of 3.6 against a background error near 0.3, and using it unscaled makes pure
3D-Var twice as bad as it needs to be — which would flatter every hybrid measured
against it.

`ring_localisation` builds the Gaspari–Cohn matrix for a periodic ring, with
$\min(|i-j|, N-|i-j|)$. Building it from $|i-j|$ instead gives a non-circulant matrix,
which quietly makes two arbitrary sites of a homogeneous system special.

## `verification` — deterministic scores, and an imperfect truth

`anomaly_correlation`, `mse_decomposition`, `mse_skill_score`, `skill_horizon`,
`acc_threshold_for_climatological_skill`, `optimal_damping`,
`correct_mse_for_observation_error`, `correct_acc_for_observation_error`.

Deterministic verification, and the corrections that relate a score computed against
*observations* to the score you actually wanted. The probabilistic scores stay in
`ensemble`.

Four exact statements anchor the module. Murphy's split
$\mathrm{MSE} = \text{bias}^2 + (\sigma_f-\sigma_t)^2 + 2\sigma_f\sigma_t(1-r)$ is an
identity. An **undamped** forecast — unbiased, anomaly variance matching the truth's —
has skill score exactly $2r-1$ against climatology, so it ties at $r = 1/2$: **that is
where the conventional 0.6 threshold comes from**, and it is a statement about a
post-processing choice rather than about predictability. `optimal_damping` gives
$\mathrm{MSE} = \sigma_t^2(1-r^2)$, which beats climatology at *any* non-zero
correlation — so `acc_threshold_for_climatological_skill(damped=True)` is 0. And
independent observation error inflates the mean-square error by exactly $\sigma_o^2$
while attenuating the anomaly correlation by exactly
$(1+\sigma_o^2/\sigma_t^2)^{-1/2}$.

`correct_mse_for_observation_error` **returns negative values rather than clipping
them**, and that is the useful part. The correction assumes the observation error is
independent of the forecast error, which fails when the verifying observations were
*assimilated* into the analysis the forecast started from: the errors then share a
component, the score is optimistic rather than pessimistic, and subtracting $\sigma_o^2$
yields an impossible negative mean-square error. A clipped zero would have looked like a
very good forecast.

`skill_horizon` interpolates the crossing rather than reporting the last grid point that
still qualified. With skill sampled every six hours, rounding to the grid quantises every
horizon in chapter 22 to the sampling interval and erases the differences between scores
that the chapter exists to show.

## `ensemble` — construction and verification

**Construction:** `gaussian_perturbations`, `bred_vectors`,
`singular_vector_ensemble`. **Diagnostics:** `ensemble_spread`,
`ensemble_mean_error`, `outside_span_fraction`. **Verification:** `rank_histogram`
(random tie-breaking), `crps` (fair energy form), `brier_score`,
`reliability_diagram`, `brier_decomposition`, `value_score`.

Tested against the closed-form Gaussian CRPS, and against the calibration identity
that a reliable ensemble has RMS spread equal to the RMS error of its mean.

`bred_vectors` runs the Toth–Kalnay breeding cycle: perturb, advance, difference,
rescale, repeat. No adjoint and no tangent linear model, which is why it was the
operational scheme at NCEP while ECMWF ran singular vectors. **Independently bred
vectors collapse onto each other** — they are all converging to the same leading
direction — and how fast depends on how well separated the leading Lyapunov exponent
is: about 2 e-foldings on Lorenz 63, about 8 on Lorenz 96, which has thirteen positive
exponents of similar size. `orthogonalise=True` re-orthogonalises after **each** cycle,
turning breeding into the Benettin construction of `chaoslib.lyapunov`; doing it once at
the end would be useless, because by then the set is nearly rank one and orthogonalising
it manufactures the extra directions out of rounding error. It requires
`n_vectors <= n_state` and raises rather than quietly returning a smaller set — a guard
that caught chapter 17's own generator asking for four orthogonal directions in three
dimensions.

Breeding must also run **along** the trajectory being perturbed. Breeding forward from
the analysis state produces vectors belonging to a state however far downstream the
breeding took, which for a useful number of e-foldings is far indeed; the chapter's
generator therefore carries the perturbations as persistent state across the
assimilation cycle, as the operational scheme did.

`singular_vector_ensemble` returns $\pm$ pairs, so the ensemble mean equals the control
state **exactly** rather than on average. An ensemble centred only on average carries a
spurious mean displacement indistinguishable from model bias.

`brier_decomposition` gives Murphy's $\mathrm{BS} = \mathrm{REL} - \mathrm{RES} +
\mathrm{UNC}$, and the identity holds to machine precision because
`reliability_diagram` bins on the **distinct forecast values** rather than on
equal-width bins. An ensemble of $k$ members can only issue $0, 1/k, \ldots, 1$, so
those *are* the natural bins; equal-width binning of already-discrete forecasts leaves a
within-bin variance term that appears as a residual and is easy to mistake for a real
effect. The distinction the decomposition buys is that **reliability is fixable after the
fact and resolution is not**.

`value_score` is Richardson's relative economic value against a cost-loss ratio
$\alpha = C/L$: exactly 1 for a perfect forecast and exactly 0 for climatology, at every
$\alpha$. It takes an *array* of ratios because the point is that **value is a curve,
not a number** — a deterministic forecast forces one threshold on every user, a
probabilistic one lets each user choose. It protects when $p \ge \alpha$, not $p >
\alpha$: with discrete ensemble probabilities $\alpha$ frequently lands exactly on an
attainable forecast value, and the two conventions then differ by a whole bin of cases.

## `errorgrowth` — saturation and the upscale cascade

`logistic_error_growth`, `fit_logistic_error_growth`, `saturation_level` (RMS distance
between random state pairs — i.e. the error of a climatological forecast),
`predictability_horizon`.

Two things about the basics, both measured in chapter 9 and both easy to get wrong.
`fit_logistic_error_growth` fits in **log** space by default: an error curve spans ten
orders of magnitude and least squares on $E$ weights by $E$, so the saturated tail
outweighs the whole exponential phase and $\lambda$ comes out 19 % low (0.748 against a
true 0.921). And `saturation_level` takes a `statistic` -- match it to your error curve,
because an ensemble-*mean* curve compared against the *RMS* saturation looks as though it
stops growing early, by 11 % for L63 and 0.5 % for L96. The RMS form obeys the exact
identity $\sqrt2 \times$ (RMS spread about the mean).

`lagged_forecast_difference` is Lorenz's (1982) estimator: given an archive shaped
`(n_leads, n_starts, n_state)`, it differences forecasts of successive lead valid at the
same time, so it needs **no truth**. Chapter 13 validates it against one — within 2 % of
the true growth rate at realistic analysis errors. It is blind to model error by
construction (both forecasts come from the same model, so a common bias cancels), which
a test asserts by biasing a whole archive.

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
`gaussian_relative_entropy`, `gaussian_information_components`.

`gaussian_information_components` returns `(total, signal, dispersion)` — the two ways a
forecast can be informative, summing to the total exactly. Both are **invariant under any
invertible linear map of the state**, which is the reason to use an information measure at
all: chapter 10 measures $D$ identical to eight decimals under four transformations where
RMS error varies 371-fold.

Two warnings, both measured in chapter 10. Do not apply the Gaussian form to a **full
state vector** whose ensemble is smaller than the space: on a 2.06-dimensional attractor
in three dimensions $\Sigma_f$ is near-singular, so $\ln\det\Sigma_f$ — and the
dispersion term with it — is set by the regularisation, and two defensible floors give
decay rates differing by 1.78×. And `mutual_information` is biased **upward**; pass
`correction="miller_madow"` to remove the leading term, which helps by a factor of two to
four and not more. Measure the floor at a lag where the true answer is zero.

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

## `turbulence` — two-dimensional flow and its spectrum

`spectral_grid`, `vorticity_tendency`, `advance_vorticity`, `energy`, `enstrophy`,
`energy_spectrum`, `local_spectral_slope`, `turnover_time`, `random_vorticity`,
`vorticity_field`, `band_perturbation`.

Pseudospectral 2-D Navier-Stokes in vorticity form, dealiased by the two-thirds rule.
Two exact anchors: inviscid Euler conserves **both** energy and enstrophy (drifts of
$2\times10^{-7}$ and $2\times10^{-6}$ over 800 steps), and a single Fourier mode carries
$\langle\zeta^2\rangle/2k^2$ — the second matters because `rfft2` stores half the
modes, and a factor-of-two error in the Hermitian weights is invisible to a conservation
test.

Three things to know before using it. `energy_spectrum` stops at $N/3$ and its shell sum
is **not** always the energy: the mask is a *square*, so modes survive to $\sqrt2 N/3$ in
its corners — negligible when resolved (<0.02 % at $N\ge64$), 12 % when the peak sits on
the truncation, which makes the ratio a resolution diagnostic. `turnover_time` gives
$\alpha = (3-p)/2$ by algebra, not measurement, so agreement with the spectrum's slope is
a tautology. And it will **not** produce an inertial range at any resolution this book can
run: chapter 14 measures 0.00/0.04/0.05 octaves within 0.4 of $-3$ at
$64^2$/$128^2$/$256^2$.

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
