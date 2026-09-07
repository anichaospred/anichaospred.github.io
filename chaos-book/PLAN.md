# Building *An Interactive Chaos and Predictability Textbook*

### A plan for a browser-runnable predictability textbook (marimo + WebAssembly)

**Site:** <https://anichaospred.github.io> — Hugo, notebooks exported to HTML, run
in-browser via Pyodide/WASM. No server-side compute anywhere.
**Audience:** upper-level undergraduates and first-year graduate students in
atmospheric, oceanic and climate science; also applied-math and physics students who
want the geophysical application rather than the abstract theory.
**Assumed background:** ODEs, PDEs and linear algebra; basic statistics; some
atmospheric dynamics; comfort reading Python. No prior dynamical-systems course
required.

---

## 1. The thesis

One sentence, which the whole book is organised around taking literally:

> **Apply a hierarchy of models to understand predictability in nonlinear dynamical
> systems.**

That single sentence does a lot of work, and the book is organised around taking it
literally. Predictability is not one number attached to the atmosphere; it is a
property of a *system*, a *state*, a *lead time* and a *quantity of interest*. The
only honest way to teach that is to let the reader turn the knob on a system small
enough to run in a browser tab, and watch the horizon move.

Three design commitments follow.

1. **Weather and climate first.** Every chapter opens with a question a forecaster or
   climate scientist would actually ask — *why do forecasts fail after about two
   weeks? why is next season's ENSO state predictable when next Tuesday's weather is
   not?* — and the dynamical-systems machinery arrives as the way to answer it. The
   mathematics is never introduced for its own sake.
2. **One mechanism, one minimal model, one knob.** Each chapter isolates a single
   idea and exposes the one to three parameters that govern it. The reader's job is
   to find the transition, not to admire a picture.
3. **The printed equation is the stepped equation.** All numerics live in `chaoslib`,
   tested against analytic identities and published values. A chapter that needs new
   numerics adds them to the library, with tests, first.

### What is different about this book

There is no shortage of good sources: Palmer & Hagedorn's *Predictability of Weather
and Climate* is the definitive collection, Kalnay's *Atmospheric Modeling, Data
Assimilation and Predictability* the definitive textbook, and Lorenz's own papers
remain the clearest statements of the core results. What none of them can do on paper
is let a student **reduce the analysis error by a factor of ten and see the forecast
horizon extend by only six days**. That experiment takes one slider.

---

## 2. Architecture

Eight parts, 31 chapters. Each chapter is a short web page plus **one primary
interactive notebook**. ★ marks a chapter that is live; the rest are stubs carrying
an abstract and the planned knob, so the shape of the book is visible from the
start and no link is dead.


### Part I — What predictability means

1. ★ **What is predictability?** — Four jobs the word does, and the last two measured
   **in the same units on the same system**, which is what makes them comparable. The
   same 200 L63 cases forecast three ways from the same initial uncertainty: the single
   run stops beating climatology at lead 5.8 (RMSE) and 5.5 (CRPS) — *actively worse
   than knowing nothing* — while the ensemble never crosses at all. Then both kinds of
   predictability on one axis in nats: first-kind information falls 2.76 → 0.038 over
   20 TU, second-kind information is a **constant** that never decays, and they cross at
   a lead set by the size of the forcing change (6.0, 10.5, 18.0 TU for
   $\rho\to36, 32, 30$; never, for $\rho\to29$). That third column is the whole of
   Part VI in one number. Care taken and recorded: averaged over 32 starts because a
   single start is not monotone, with a *measured* estimator noise floor of 0.0097 nats
   that the smallest forcing signal sits only 1.5× above. *Knob:* lead time.
2. **A short history of numerical weather prediction** — Richardson's forecast
   factory and why his 1922 hand-computation failed; Bjerknes' programme; the 1950
   ENIAC barotropic forecasts; the growth of operational skill. *Notebook:* the
   Richardson problem — integrate an unbalanced initial state and watch spurious
   gravity waves swamp the signal, then initialise it in balance. *Knob:* imbalance
   amplitude.
3. **The hierarchy of models** — From the logistic map to CESM: what each rung is
   for, and the argument that a three-variable model can teach something true about a
   $10^9$-variable one. *Notebook:* the same predictability diagnostic (doubling
   time, normalised) computed across the logistic map, L63, L96 and 2-D turbulence.
   *Knob:* which rung.

### Part II — From regular motion to chaos

4. ★ **Regular motion and why it is predictable: one pendulum, two pendulums** —
   Degrees of freedom, phase-space dimension, Poincaré–Bendixson; why the *exact*
   nonlinear pendulum is still perfectly predictable, and what changes when a second
   rod is added. *Knob:* $\theta_0$, $\delta_0$.
5. ★ **Maps, bifurcations, and the routes to chaos** — Cobweb stability, the
   bifurcation diagram with $\lambda(r)$ beneath it, and $\delta$ measured rather
   than quoted: three unrelated unimodal families (logistic, sine, cubic) give
   4.669191, 4.664075, 4.669038. The period-3 window's own $3\cdot 2^n$ cascade gives
   the same $\delta$ from a range 90 times narrower, which is the renormalisation
   structure itself. Type-I intermittency below $r_c = 1+2\sqrt2$ scales as
   $(r_c-r)^{-0.4965}$ against a predicted $-1/2$ — a regime that ends with no
   parameter changing, and the origin of critical-slowing-down early warnings.
   *Knob:* $r$, $x_0$, the parameter window.
6. ★ **The Lorenz (1963) system: the butterfly** — The Rayleigh–Bénard truncation,
   the three fixed points, the Hopf bifurcation at $\rho_H \approx 24.74$, the strange
   attractor, sensitive dependence, and the first ensemble. *Knob:*
   $\sigma, \rho, \beta$.

### Part III — Quantifying chaos and predictability

7. ★ **Lyapunov exponents and doubling times** — Oseledets' theorem; the full spectrum
   via the Benettin algorithm, with the exact check
   $\sum_i \lambda_i = -(\sigma+1+\beta)$ shown live; finite-time exponents and the two
   distinct reasons they run high over short windows (optimisation over direction, and
   too short an average); and transient chaos — where a positive finite-$T$ exponent
   faithfully describes a chaotic set the system will eventually leave.
   *Knob:* $\rho$, integration length $T$, window $\tau$.
8. ★ **Attractors, fractal dimension, and entropy** — Two independent routes to the
   same number: $D_2 = 2.0579$ from counting pairs against $D_{KY} = 2.0618$ from the
   Lyapunov spectrum, a 0.2 % difference between calculations sharing no intermediate
   quantity — the Kaplan–Yorke conjecture, tested. Calibrated first against three
   sets whose dimension is exact ($\ln2/\ln3$, $\ln4/\ln3$, $\ln3/\ln2$). The
   substance is how easily both are got wrong: the *same* $C(r)$ returns 0.19, 1.92 or
   2.51 depending on the window, and the Theiler bias runs **opposite** to the usual
   warning (high, not low, when the sampling step lands inside the fit window).
   Pesin gives $h_{KS} = 1.30$ bits/MTU and $\ln10/\lambda_1 = 3.8$ days per decimal
   digit — chapter 20's logarithmic law from the geometric end. Delay embedding
   recovers $D_2$ from $x(t)$ alone. *Knob:* the scaling window, the Theiler window,
   the embedding dimension.
9. ★ **Error growth beyond the linear regime** — Lorenz's logistic model, and three
   measured traps. **Fitting:** least squares on $E$ rather than $\ln E$ gives
   $\lambda = 0.748$ against a true 0.921, because the saturated tail outweighs the
   whole exponential phase. **Statistics:** an ensemble-*mean* error curve compared
   against the *RMS* saturation appears to stop growing at 89 % — worth 12 percentage
   points of spurious model error in L63, where the two lobes broaden the pair-distance
   distribution (0.889) but not in L96 (0.995). **Form:** the fitting-free test
   slope/intercept $= -1$ gives $-1.027$ for L63 (2.7 %, good) but $-1.167$ for L96
   (17 %), because thirteen positive exponents cannot be one $\lambda$. Also: the
   logistic $\lambda$ is not $\lambda_1$ (12 % and 26 % low), and the doubling time
   quadruples across the useful range. *Knob:* $\delta_0$, the fit space, the
   threshold.
10. ★ **Information theory and predictability** — Relative entropy against
    climatology: **invariant** under any invertible change of variables (unchanged to
    $2\times10^{-15}$ nats under four transformations, where RMS error varies from 3.7×
    to **371×** depending only on the rescaling), which
    is the exact counterpart of ch. 16's norm problem. Forecast information is **94 %
    dispersion**, not signal. It decays *linearly* at $\lambda_1$ — four measurements
    within 4 %, and Lorenz 96 rules out $h_{KS}$ by 6.4× — so $h_{KS}$ is how fast the
    system destroys information about *itself* while $\lambda_1$ is how fast a forecast
    of one variable stops being informative. Doing it on the full state fails quietly:
    a near-singular $\Sigma_f$ on a 2.06-dimensional attractor gives decay rates of
    1.677 or 0.940 depending on the regularisation alone. Mutual information avoids the
    Gaussian assumption and pays in bias, with a floor of 0.009–0.039 nats set by bin
    count. *Knob:* system and observable, the rescaling factor, bin count.

### Part IV — Many scales, many degrees of freedom

11. ★ **Lorenz 96: a many-variable atmosphere analogue** — The first system in the
    book with a *space*. Its preferred wavelength comes out of a closed-form
    dispersion relation, $\sigma(\theta) = -1 + F(e^{i\theta} - e^{-2i\theta})$,
    which reproduces every Jacobian eigenvalue at the uniform state to $5\times
    10^{-14}$ and gives $F_{\rm crit} = 2/\sqrt5$ exactly at $N = 40$. Two
    thresholds, far apart: waves at $F = 0.894$, chaos only near $F = 4.5$. The
    headline is **extensivity** — $\lambda_1$ intensive (flat above $N \approx 30$),
    the spectrum collapsing under $i \to i/N$, and $D_{KY} = 0.675N$,
    $h_{KS} = 0.256N$ through the origin. That ratio is why Part V's ensemble methods
    look the way they do. *Knob:* $F$, $N$, the time window.
12. ★ **Scale-dependent error growth and the intrinsic limit** — Lorenz (1969), as a
    measurement. Octave bands with $\lambda_n = \lambda_0 2^{\alpha n}$: the horizon
    converges iff $\alpha > 0$ (1.4466 at Kolmogorov $\alpha = 2/3$, against
    unbounded growth at 0.281 per octave for $\alpha = 0$, out to 128 bands). The
    per-octave gain dies as $2^{-2\alpha}$ — measured 0.6299/0.3968/0.2500 against
    0.6300/0.3969/0.2500, the *square* of the naive sum-of-timescales prediction, which
    is why that argument gives 2.70 where the answer is 1.4466. **Where** you improve
    beats **how much**: 16 decades of accuracy at the finest scale buys 2 %, digits at
    the coarsest buy $\ln10/\lambda_0$ forever. Two-scale L96 shows the mechanism
    (slow error grows at 2.96–2.99/TU regardless of the fast perturbation's size) but
    **not** the limit — 0.145 against 0.148 TU/decade, and its $\lambda_1 = 24.7$
    belongs to the fast subsystem and overstates large-scale growth 8.3×.
    *Knob:* $\alpha$, resolved octaves, where the error is seeded.
13. ★ **Error growth in operational models** — Lorenz's (1982) lagged-forecast
    estimator, validated against a truth it never uses. A synthetic operational centre
    on L96 (cycling EnKF, six-hourly analyses, 30-day forecasts from 600 analyses, four
    observing networks) shows the truth-free estimate recovering the true growth rate to
    2 % at analysis errors of 0.5–2 % of saturation and 5 % at 6 % — and recovering it
    at *every* amplitude, not just one fitted number. The logarithmic return holds end
    to end: 13× better analysis buys 7.0 days against 6.9 predicted. And the operational
    doubling time is **1.86 days against $\ln2/\lambda_1$ = 2.08**, above $\lambda_1$
    at small error (non-normal transient growth, ch. 16) and below it at large
    (saturation, ch. 9). Blind to model error by construction, which the tests assert.
    *Knob:* the observing network, the fit window.
14. ★ **From chaos to turbulence** — A 2-D pseudospectral solver (energy and
    enstrophy conserved to $2\times10^{-7}$ and $2\times10^{-6}$ over 800 inviscid
    steps) supplying what ch. 12 postulated. The payoff is algebra:
    $\tau \sim [k^3E(k)]^{-1/2}$ gives $\alpha = (3-p)/2$, so 3-D ($p=5/3$) has a
    **bounded** horizon and 2-D ($p=3$) does not — and the atmosphere is the second at
    large scales and the first at small ones. Measured: the fluid's spectrum falls 3.55
    decades over 3.4 octaves against L96's 0.71 over 1.2, and its peak *migrates* 9→4
    (inverse cascade) where L96's is pinned at $m^*=8$ by ch. 11's dispersion relation.
    The upscale error cascade works ($k\in[8,18)$ share 0.000→0.536). **No inertial
    range at any affordable resolution** — 0.00/0.04/0.05 octaves within 0.4 of $-3$ at
    $64^2$/$128^2$/$256^2$ — reported rather than fitted away. *Knob:* resolution,
    seeding scale. †

### Part V — The machinery of prediction

15. ★ **Tangent linear and adjoint models** — Linearisation of the *discrete* model;
    the adjoint identity
    $\langle \mathbf{M}x, y\rangle = \langle x, \mathbf{M}^{\!\top}y\rangle$; and the
    gradient of a forecast metric at a cost independent of state dimension. The window
    of validity obeys the same logarithmic law as everything else — measured at
    1.18 MTU per e-fold of $\delta_0$ against $1/\lambda_1 = 1.10$. Closes with the two
    tests that tell you an adjoint is correct, and the three bugs they caught in this
    book's own library. *Knob:* $\tau$, amplitude.
16. ★ **Adjoint sensitivity and optimal perturbations** — Singular vectors, and how
    they differ from both the gradient and the Lyapunov vectors. Optimal growth beats
    $e^{\lambda_1\tau}$ by 1.6–2.6× at every window tested (averaged over the attractor
    — at a single base point $\sigma_1(\tau)$ is not even monotonic).
    "Fastest-growing" is undefined without a norm, and changing the norm rotates the
    answer by tens of degrees. Sensitivity and growth nearly coincide in Lorenz 63,
    where one singular value dominates, and are 89° apart in Lorenz 96 — a reminder of
    what a three-variable model cannot show. *Knob:* optimisation window, the norm.
17. ★ **Probabilistic forecast design** — Five constructions from the *same* LETKF
    analysis, at the same member count and the same amplitude, so the comparison is
    about direction alone. **Growing fastest is the wrong objective**: singular vectors
    are over-dispersed by two thirds at medium lead (spread/error 1.67) and score worse
    than isotropic noise at short lead. **Bred vectors collapse**, at a rate set by how
    well separated $\lambda_1$ is — 2 e-foldings on L63, 8 on L96 — and the degeneracy
    is worse than "narrow": breeding fixes a direction but not a *sign*, so 85 % of
    members end within 26° of $\pm$ one direction (39 % aligned, 46 % anti), making the
    ensemble two points rather than a sample. It costs spread/error 0.67 at lead 2;
    orthogonalising each cycle recovers it. **Construction beats size**: CRPS
    $= a + b/k$, where $b$ is mostly the estimator's own finite-sample bias (~5 % at
    $k=20$) and the asymptote $a$ is where construction shows — no ensemble size closes
    a gap in $a$. Murphy's decomposition is **exact** with distinct-value bins, and
    separates the repairable from the irreducible: the EDA has the most resolution and
    the worst reliability, and relabelling fixes the second while leaving the first
    unmoved to five decimals. The payoff is the cost-loss curve: the same information
    served probabilistically has value at 19 of 19 ratios against 16, and the
    deterministic version is **actively harmful** ($V=-1.43$) at the extremes.
    *Knob:* ensemble size, perturbation strategy. †
18. ★ **Variational data assimilation** — One cost function, and how much follows from
    keeping $\mathcal{M}$ inside it. Observe $x$ alone and 3D-Var returns $y$ and $z$
    *bitwise unchanged*, while 4D-Var cuts the error in the unobserved $y$ by 37 % — the
    model as constraint, which is why radiances and bending angles are assimilable. The
    Hessian **is** the analysis-error covariance: it never sees the observed values, it
    is flow-dependent though $\mathbf{B}$ is not, and on a linear system one outer step
    reproduces the Kalman mean to $1.6\times10^{-15}$ and $\mathbf{M}\mathbf{A}
    \mathbf{M}^{\top}$ its covariance to $4.4\times10^{-16}$. The result worth the
    chapter: holding observation *count* fixed and moving only their *timing*, spreading
    them over a window gives an analysis **43 % worse** at the analysis time and
    2--3$\times$ better at every forecast lead — a benefit invisible where analyses are
    verified. Window length has a broad optimum at 0.8--1.2 TU (0.7--1.1 e-folding
    times), because local minima of $J$ on one slice go 1 → 3 → 17 as the window grows
    0.5 → 2.0. Two negative results kept: the gradient test is **blind at the
    background**, where $\mathbf{B}^{-1}(x_0-x^b)$ vanishes and a gradient missing that
    term entirely gives a bitwise-identical curve; and incremental 4D-Var, being
    Gauss--Newton, converges from a background of error 1.33 and **stalls completely** at
    2.66. *Knob:* window length, $\mathbf{B}$. †
19. ★ **Ensemble data assimilation** — A flow-dependent covariance without an
    adjoint, and the rank problem that costs. Sampling error scales as $k^{-1/2}$
    with constant **1.00** over six doublings, so halving it costs four times the
    ensemble: localisation is a precondition, not a refinement. Its deeper job is
    **rank**: a global filter's increment lies in the span of the $k-1$ perturbations
    (exact to $2\times10^{-15}$) and no covariance tapering changes that, while a
    *local* filter puts 78 % of its increment outside that span at $k=10$. The
    filter-divergence cliff is sharp — unlocalised, $k=10$ scores 3.38 against a
    climatology of 3.64 (worse than useless), and $k=15$ scores 0.24. The optimal
    radius **grows** with ensemble size (12 → 20 → none) and its penalty is lopsided:
    too loose costs 19$\times$, too tight 1.5$\times$. And localisation and ensemble
    size are **substitutes** — at radius 2, eight times the members buys 3 %. The ETKF
    is exactly the Kalman filter for the covariance the ensemble has, at every $k$
    including $k<n$, and beats the perturbed-observation form by 64 % at $k=5$.
    Hybrids came out small (8 % on top of localisation, against a ±2.5 % noise floor)
    and the chapter says why that is the friendliest case for a localised ensemble
    rather than a verdict on hybrids. *Knob:* ensemble size, localisation radius. †
20. ★ **Data assimilation in practice** — Cycling 3D-Var, 4D-Var and the EnKF on
    Lorenz 63; analysis error as the floor on forecast error; and the logarithmic
    return on better observations, $\Delta t = \ln 10/\lambda$ — measured directly,
    and agreeing with $1/\lambda_1$ from chapter 7 to a few percent by an entirely
    independent route. *Knob:* ensemble size, inflation, observation interval,
    $\delta_0$.
21. ★ **Model error and the imperfect-model problem** — Three error sources, three
    growth laws: IC error exponential, deterministic bias **linear** in $t$ (measured
    $d\ln E/d\ln t = 1.087$ against a predicted 1, across biases spanning 20×),
    stochastic forcing **diffusive**. The headline is a **ceiling**: with a bias of
    0.01, a 0.125 % error in $F$, improving $\delta_0$ from $10^{-4}$ to $10^{-8}$
    buys $-0.005$ days — the lead pinned at 19.6 days, exactly the perfect-IC value.
    The same two-decade improvement is worth 12.3 days with a perfect model and 0.1 at
    bias 0.2. So chapters 8/13/20's $\ln10/\lambda_1$ law has a stopping condition
    about one decade wide, and Parts III–IV's perfect-model numbers are **upper
    bounds**. Neither DA nor ch. 13's estimator can fix or even see it.
    *Knob:* model bias, initial accuracy, error source.
22. ★ **Forecast verification and the practical horizon** — Where "useful to about a
    week" is shown to be a statement about four choices, none of them about the
    atmosphere. **The 0.6 threshold is arithmetic**: an undamped forecast has skill
    score exactly $2r-1$, so it ties with climatology at $r=1/2$; damped optimally it
    scores $r^2$ and wins at any $r>0$. The same 367 forecasts give horizons **2.5x
    apart** across defensible definitions (ACC 0.6 → 1.94 TU, Brier skill → 3.79), and
    ACC 0.5 and MSE-skill-zero agree exactly at 2.20 because the identity says they
    must. Post-processing moves it further than the threshold does: undamped MSE
    reaches climatology at 2.22 TU, damped never within 5. **The deferred question from
    five chapters** — verifying without a truth — gets its answer: independent
    observation error inflates MSE by exactly $\sigma_o^2$ at every lead and
    attenuates ACC by exactly $(1+\sigma_o^2/\sigma_t^2)^{-1/2}$, both correctable;
    verifying against your own analysis flatters by 64 % at lead 0.1 but under 2 % by
    lead 1; and verifying an analysis against its *own assimilated* observations makes
    the correction return a **negative mean-square error**, which is the most useful
    thing it can do. Murphy's split localises faults correctly (wrong forcing →
    amplitude x119, offset → bias x63) even though phase dominates the total
    everywhere. And chapter 12's promise is tested across seven decades of
    $\delta_0$: single-scale L96 keeps paying at the predicted $\ln 10/\lambda_1 =
    1.38$ TU per decade, two-scale pays **8x less** — supporting "different quantities"
    exactly and softening "only one moves", since the two-scale curve is still creeping
    up at $10^{-8}$. *Knob:* the threshold defining "useful". †

### Part VI — Predictability of the second kind: from S2S to climate

23. ★ **Boundary-forced predictability and the S2S window** — L63 with
    $\rho(t) = 28 + 6\sin(2\pi t/40)$, and the whole argument in one decomposition:
    score each forecast against the **pooled** climatology and against the one
    **conditioned on forcing phase**. Information decays to a **floor** (0.146 nats),
    not to zero, and the phase alone is worth 0.120 — so at long lead the forecast has
    become a statement about the forcing. The two add: floor + residual = 0.147 against
    a measured 0.146, a *check* rather than an identity since relative entropy is not
    additive. The forcing overtakes the initial state at lead 17, and the
    "predictability desert" between lead 10 and 17 is **thin, not empty** — it has that
    floor under it. **Windows of opportunity are real**: horizons of 9 to 17 TU by
    launch phase, with a split-half correlation of **+0.96** to show it is structure
    rather than noise. Amplitude is what the forcing is worth (0.0015 → 0.252 nats);
    period is not (10 % across an eightfold range). And one result came out backwards
    and is kept: at period 2.5 the phase information *rises*, because the decomposition
    breaks down when the forcing is not slow compared with the dynamics — the clean
    separation is a property of the timescale gap, not of the mathematics. Zero-amplitude
    control: boundary information collapses to 0.0015 nats, a factor of 80.
    *Knob:* forcing period and amplitude.
24. ★ **The ocean's role: interannual-to-decadal prediction** — A fast L63
    "atmosphere" coupled to a slow "ocean" that integrates it (Hasselmann's mechanism):
    the atmosphere forgets in 0.14 TU, the ocean in 17.9. **A constraint that is
    measured, not chosen**: loop gain and feedback amplitude both scale as
    $\lambda\kappa$, so any coupling strong enough to move the atmosphere has already
    destroyed the ocean's memory — and a *positive* $\kappa$ collapses the system
    entirely (asserted as a test). Initialising the ocean buys **8 TU** for the ocean and
    almost nothing for the weather — 0.22 sd at lead 2, real at 18 standard errors, gone
    by lead 4. **Drift is not bias**: a parameter error under half a per cent displaces
    the model's climatology by ~1 sd and the mean error grows to −1.79 (correlation with
    lead −0.97) — while the *perfect-model control* also wanders to 0.59, so what
    distinguishes drift is being 7× larger and far more trended, not the control being
    zero. Correcting it is worth 9 % with enough re-forecasts and is **worse than
    nothing** with 5 or 10, first paying at about 20 — the archive is a component of the
    forecast. Cross-validation matters: in-sample is 2.2 % optimistic here and far more
    where re-forecasts are scarce. *Knob:* initialisation lead. †
25. ★ **Climate prediction and projection** — Lorenz 63 with a ramping Rayleigh
    number, $\rho(t) = \rho_0 + \gamma t$; the zero-rate limit is asserted as a
    **bitwise** identity with the unforced system. One forecast, two questions: an
    individual member saturates at **1.48** climatological spreads while the ensemble's
    windowed mean tracks the truth's to **2.8 %** of one. **Internal variability does
    not shrink under forcing** — it stays within 0.8 % (8.574–8.639) while the response
    grows 0.48 → 15.4, so S/N rises because the signal outgrows a fixed noise. **Two
    times of emergence differing by $\sqrt K$**: a single realisation never reaches
    S/N > 2 in 320 TU, an ensemble of 400 does at $t = 18$. Both scalings hold as laws,
    not fits — $\mathrm{ToE}\times\gamma$ constant to 3 %, $\mathrm{ToE}\times\sqrt K$
    constant from $K = 10$ and breaking below it where the noise estimate is itself made
    from a handful of members. **Initialisation is worth nothing** — two ensembles from
    disjoint start sets differ by 1 % of internal variability — the sharp contrast with
    chapter 24's 8 TU. *Knob:* forcing rate, ensemble size.
26. ★ **Earth system prediction** — Chapter 24's coupling extended: four reservoirs
    spanning a factor of 64 in memory, and -- the structural change -- a forcing that
    is **part of the state**, a carbon reservoir emissions fill and a
    climate-dependent sink drains. The reservoirs are passive on purpose, which is how
    chapter 24's :math:`\lambda\kappa` wall is avoided: the carbon loop's gain and
    amplitude are independent parameters. **Memory is a parameter, amplitude a
    consequence** -- :math:`\sigma_S` falls as :math:`1/\sqrt T`, exceeding
    Hasselmann's law by 2.23, 1.55, 1.19, 1.10 as :math:`T` grows, and a lag-1 fit to
    the fastest reservoir returns a memory 3x too short. **The useful lead of an
    initialised forecast is** :math:`1.141\,T` **and essentially nothing else**: the
    advantage decays as :math:`e^{-2\ell/T}` exactly (measured rate ratios 1.24, 0.87,
    1.01, 0.89; lead ratios 1.32, 1.05, 1.04, 1.16, so 4--5 % for the reservoirs that
    are genuinely slow), and a hundredfold better analysis moves the coefficient 2 %.
    **The feedback has an exactly solvable gain**, :math:`C/C_0 = 1/(1-g)` to 0.67 %
    for :math:`g \le 0.5`, failing by 13.5 % at :math:`g = 0.8` not because the
    feedback is strong but because :math:`\mathrm{d}\langle z\rangle/\mathrm{d}\rho`
    is no longer the number it was fitted at; past :math:`g = 1` the sink **saturates**
    (carbon accumulates at a fixed rate, 0.96 of the asymptote) rather than exploding.
    **Three uncertainty sources whose ranking depends on the question**: internal
    variability is 10.85 % of the spread at lead 600 for :math:`T = 2` and 0.05 % for
    :math:`T = 128`. **And an emergent constraint that is a perfect measurement of the
    wrong quantity** -- the observable matches its closed form to 5.7 % with a
    within-record correlation of 0.99, yet correlates with the response at +0.37 in one
    parameter, -0.44 in two, and **-0.005** with the feedback factor itself, because it
    turns over at :math:`\alpha \approx 0.07`; two members 1.6 % apart in the
    observable differ 6.3-fold in response. Limitation stated and tested: no permanent
    airborne fraction, so no zero-emissions commitment. *Knob:* reservoir memory,
    feedback strength.
27. ★ **Regimes, bistability, and tipping points** — The tilted double well
    :math:`\dot x = x - x^3 + \mu + \sigma\xi`, a gradient system, so the
    stationary density is exactly Boltzmann (measured to 4.6 % over a factor of
    fourteen in lobe ratio) and the fold at :math:`\mu_c = 2/(3\sqrt3)` is exact.
    **Kramers' exponent is exact and its prefactor is not**: fitted slope 0.2333
    against :math:`2\Delta V = 0.2350` (0.8 %), while the formula overestimates the
    waiting *time* by 60 %; one censored noise level is kept in the figure because
    including it biases the slope by 12.1 %. **Critical slowing down is exactly true
    and stops working before the fold** — 3.5 % on the spread and 1.6 % on the
    autocorrelation up to :math:`\mu = 0.25`, then 3.6× the theory at 0.32 and an
    autocorrelation that *falls* to 0.47 against a predicted 0.96 at 0.382, because
    the predicted fluctuation has grown to the basin width. **Two timing laws with
    opposite signs**: a noiseless sweep leaves *late* by the Airy constant
    :math:`1.9469\gamma^{2/3}` (measured ratio → 0.988), and noise leaves *early* by
    a closed form good to 3.3 % across a factor of five in :math:`\sigma` — whose
    logarithm is the whole content, since the bare :math:`\sigma^{4/3}` scaling is
    wrong by a factor rising from 1.81 to 3.18. **Early warning has recall without
    specificity**: calibrated to 5 % false alarms it fires on 100 % of genuine sweeps
    with 1200 TU of lead, on 100 % of approaches that stop short and never tip, and on
    13 % of noise-induced transitions that tip 80 % of the time. *Knob:* tilt, noise
    amplitude. †

### Part VII — Frontiers

28. **Has predictability changed over time?** — Non-stationary predictability in a
    changing climate; separating a trend in skill from a trend in the underlying
    predictability.
29. **Machine learning and data-driven prediction** — Do learned emulators inherit
    the right Lyapunov spectrum and error-growth rate? Stability of long rollouts.
    *Connects to* the `ai-models-sensitivity` work. *Notebook:* train a small
    surrogate on L96 and compare its spectrum with the truth's.

### Part VIII — Structure (optional, terminal)

30. **Ergodic theory and invariant measures** — Time averages vs. ensemble averages;
    what "climate" means mathematically; when the two coincide.
31. **The Koopman operator** — Linear representations of nonlinear dynamics; modes
    and eigenvalues; the connection to DMD. *Connects to* the `quantum-koopman-da`
    work.

> Part VIII is deliberately terminal and skippable. It rewards a reader who wants the
> structure behind the results, without gating the main path.

### A decision on record: splitting chapter 6

The ported Lorenz 63 notebook currently spans four sections — the attractor, SDIC,
ensembles, and the bridge to the real atmosphere — which properly belong to chapters
6, 7, 17 and 22. It is kept whole for now because it is a coherent, tested,
already-published teaching artefact and breaking it up before its successors exist
would leave gaps. As chapters 7, 17 and 22 are written they should *take* material
from it, and chapter 6 should shrink to the attractor and the bifurcation.

---

## 3. Reference mapping

Both course texts are the anchors for further reading; each chapter page ends with
the corresponding sections rather than a general bibliography.

| Source | Role |
|---|---|
| Palmer & Hagedorn (2006), *Predictability of Weather and Climate* | The standard collection; the natural companion for Parts I, IV, VI |
| Kalnay (2003), *Atmospheric Modeling, Data Assimilation and Predictability* | The anchor for Part V (adjoints, variational and ensemble DA) |
| Lorenz (1963), *Deterministic nonperiodic flow* | Chapter 6 |
| Lorenz (1969), *The predictability of a flow which possesses many scales of motion* | Chapter 12 |
| Lorenz (1982), *Atmospheric predictability experiments with a large numerical model* | Chapter 13 |
| Smith (2007), *Chaos: A Very Short Introduction* | Accessible companion for Part II |

Specific chapter-to-section page numbers are to be filled in as chapters are written
*[citations needed]* — do not invent them.

---

## 4. Build and infrastructure

See `docs/architecture.md` for the pipeline and `docs/authoring.md` for the
chapter checklist. In brief:

- `chaoslib/` — shared, Pyodide-safe numerics (NumPy, SciPy, Plotly only), with a
  correctness test suite anchored to analytic identities and published values.
- `notebooks/chNN_slug.py` — one marimo notebook per chapter, plain `.py` so chapter
  diffs review like code.
- **All chapters export into one directory** (`site/static/nb/`), sharing a single
  `assets/` folder and a single `public/` holding the `chaoslib` wheel. The
  alternative — a self-contained bundle per chapter — costs ~27 MB *each*.
- `site/` — Hugo; equations rendered to MathML at build time, so a malformed equation
  fails the build and no maths JavaScript ships.
- GitHub Actions runs the tests, exports every notebook, builds the site and deploys
  to Pages on every push to `main`.

### Editions

The repo is tagged per edition. A tag marks a commit where the pinned versions, the
live chapters, and a clean CI run (tests + WASM export of every notebook) coincide.
`requirements.txt` pins `marimo` exactly, because marimo's version determines the
Pyodide build every reader's browser receives.

---

## 5. Status

| | Count |
|---|---|
| Chapters live | **25** (1, 4–27) |
| Chapters stubbed | 6 |
| `chaoslib` modules | 15 |
| `chaoslib` tests | 311, all passing |

**Next chapters, in priority order** — each already has most of its material in hand:

1. **Ch. 2 and 3** (Part I) — ch. 2 needs a shallow-water balance system in `chaoslib`;
   ch. 3 is mostly synthesis across systems already present.
2. **Ch. 20's diet** — unblocked since chapter 19; see the note below.

### A second decision on record: splitting chapter 20

Chapter 20 covers 3D-Var, 4D-Var *and* the EnKF, which properly belong to chapters 18,
19 and 20 respectively. It is kept whole for the same reason chapter 6 is: it is a
coherent, tested teaching artefact, and breaking it up before its neighbours exist
would leave gaps. As chapters 18 and 19 are written they should take the corresponding
theory sections, and chapter 20 should shrink to cycling, analysis error and the
logarithmic return — which is its real subject.

**Status: now unblocked.** Chapters 18 and 19 are both live and each carries its
theory in far more depth than chapter 20's sections 3 and 4 do. The condition for
cutting has therefore been met: chapter 20's sections 2--4 should now be cut *together*
and replaced by cross-references, leaving it as the cycling-and-return chapter it was
always meant to be. Cutting them one at a time was never safe -- section 4 assumes
section 3 -- which is why this waited for both neighbours rather than one.

This is deliberately **not** done as part of chapter 19: it edits a live, tested chapter
for tidiness rather than correctness, and it should be its own reviewable change.
