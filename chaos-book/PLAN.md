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

1. **What is predictability?** — Practical vs. intrinsic limits; predictability of
   the first kind (initial conditions) and the second (boundary forcing); the forecast
   as a probability distribution rather than a trajectory. *Notebook:* the same L63
   forecast scored three ways — single run, ensemble, climatology — so the reader sees
   that "skill" depends on the question asked. *Knob:* lead time.
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
10. **Information theory and predictability** — Entropy, relative entropy and mutual
    information as predictability measures; the signal and dispersion components of
    forecast information; a horizon that needs no assumption of exponential growth.
    *Notebook:* $I(x(0); x(\tau))$ against lead time, beside the Lyapunov estimate.
    *Knob:* lead time, bin count.

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
14. **From chaos to turbulence** — Dissipative low-order chaos vs. fully developed
    turbulence; predictability in 2-D and quasi-geostrophic flow; error growth as a
    spectral process. *Reuse:* the existing `real_butterfly_effect` and `sqgturb`
    work. *Knob:* Reynolds/resolution.

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
17. **Probabilistic forecast design** — Ensemble construction (singular vectors, bred
    vectors, EDA); reliability; CRPS, Brier score, rank histograms; the value of a
    probabilistic forecast to a decision. *Notebook:* build an ensemble three ways and
    score all three. *Knob:* ensemble size, perturbation strategy.
18. **Variational data assimilation** — 3D-Var; 4D-Var; the incremental form; why the
    adjoint is what makes the gradient affordable. *Notebook:* 4D-Var on L63 with the
    cost-function descent shown live. *Knob:* window length, $\mathbf{B}$.
19. **Ensemble data assimilation** — The EnKF and ETKF; sampling error, localisation
    and inflation; hybrid methods. *Notebook:* EnKF on L96 with localisation radius
    and inflation as sliders, and the filter-divergence cliff. *Knob:* ensemble size,
    localisation.
20. ★ **Data assimilation in practice** — Cycling 3D-Var, 4D-Var and the EnKF on
    Lorenz 63; analysis error as the floor on forecast error; and the logarithmic
    return on better observations, $\Delta t = \ln 10/\lambda$ — measured directly,
    and agreeing with $1/\lambda_1$ from chapter 7 to a few percent by an entirely
    independent route. *Knob:* ensemble size, inflation, observation interval,
    $\delta_0$.
21. **Model error and the imperfect-model problem** — Stochastic parameterisation;
    model-error growth vs. initial-condition error growth; the imperfect-model
    scenario. *Notebook:* forecast L96 with a wrong $F$ and separate the two error
    sources. *Knob:* model bias.
22. **Forecast verification and the practical horizon** — Anomaly correlation; the
    ECMWF skill record *[citation needed]*; how the practical horizon has advanced
    while the intrinsic one has not. *Knob:* threshold defining "useful".

### Part VI — Predictability of the second kind: from S2S to climate

23. **Boundary-forced predictability and the S2S window** — ENSO, the MJO, sudden
    stratospheric warmings, monsoon onset; the "predictability desert" and why it is
    not empty. *Notebook:* L63 with a slowly varying $\rho$ — predictable statistics
    over an unpredictable trajectory. *Knob:* forcing period and amplitude.
24. **The ocean's role: interannual-to-decadal prediction** — Initialised prediction;
    ocean heat content as the memory; drift, bias and the need for
    re-forecasts.
25. **Climate prediction and projection** — Why the attractor's *statistics* are
    predictable when its trajectory is not; forced response vs. internal variability;
    the initialised/uninitialised distinction.
26. **Earth system prediction** — Coupled, carbon-cycle and cryosphere components;
    what "initialising" a slow component means; frontiers.
27. **Regimes, bistability, and tipping points** — Multiple attractors; noise-induced
    transitions; early-warning indicators and their failure modes. *Notebook:* a
    double-well system with noise; variance and autocorrelation as warning signals.

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
| Chapters live | **12** (4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 20) |
| Chapters stubbed | 19 |
| `chaoslib` modules | 10 |
| `chaoslib` tests | 176, all passing |

**Next chapters, in priority order** — each already has most of its material in hand:

1. **Ch. 10** (information theory) — `chaoslib.information` is tested and unused by any
   live chapter; chapter 8's $h_{KS}$ in bits is the hook, and it would finish Part III.
2. **Ch. 21** (model error) — chapters 12 and 13 both now end by pointing at it, and 13
   proves the lagged-forecast method cannot see it.
3. **Ch. 18/19** (variational and ensemble DA) — chapter 20 currently carries their
   theory sections; see the note below.

### A second decision on record: splitting chapter 20

Chapter 20 covers 3D-Var, 4D-Var *and* the EnKF, which properly belong to chapters 18,
19 and 20 respectively. It is kept whole for the same reason chapter 6 is: it is a
coherent, tested teaching artefact, and breaking it up before its neighbours exist
would leave gaps. As chapters 18 and 19 are written they should take the corresponding
theory sections, and chapter 20 should shrink to cycling, analysis error and the
logarithmic return — which is its real subject.
