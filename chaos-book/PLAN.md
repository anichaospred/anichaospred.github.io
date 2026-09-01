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
5. **Maps, bifurcations, and the routes to chaos** — The logistic map, period
   doubling, the Feigenbaum constant, intermittency; regime transitions as
   bifurcations. *Notebook:* cobweb plot beside a live bifurcation diagram, with the
   measured $\delta$. *Knob:* $r$.
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
8. **Attractors, fractal dimension, and entropy** — Correlation dimension,
   Kaplan–Yorke, Kolmogorov–Sinai entropy as an information-loss rate; Pesin's
   identity. *Notebook:* $\ln C(r)$ vs. $\ln r$ with the scaling window exposed as a
   slider, so the reader discovers how easily a dimension is mis-measured. *Knob:* the
   fit window.
9. **Error growth beyond the linear regime** — Lorenz's logistic error-growth model;
   saturation; why a doubling time alone misstates the horizon. *Notebook:* fit
   exponential and logistic models to the same measured curve and compare the horizons
   they imply. *Knob:* $\delta_0$.
10. **Information theory and predictability** — Entropy, relative entropy and mutual
    information as predictability measures; the signal and dispersion components of
    forecast information; a horizon that needs no assumption of exponential growth.
    *Notebook:* $I(x(0); x(\tau))$ against lead time, beside the Lyapunov estimate.
    *Knob:* lead time, bin count.

### Part IV — Many scales, many degrees of freedom

11. **Lorenz 96: a many-variable atmosphere analogue** — Spatiotemporal chaos; the
    Lyapunov spectrum; 13 positive exponents at $N=40, F=8$; scaling with $N$ and $F$.
    *Notebook:* Hovmöller diagram beside the live spectrum. *Knob:* $F$, $N$.
12. **Scale-dependent error growth and the intrinsic limit** — Lorenz (1969);
    multiscale L96 (Lorenz 2005); the upscale error cascade; the case for a *finite*
    limit that no observing system can defeat. *Notebook:* seed error at one scale and
    watch it climb the spectrum. *Knob:* the scale of the initial error.
13. **Error growth in operational models** — Lorenz (1982) and forecast-error
    statistics; from an idealised $\lambda$ to real doubling times; the lagged-forecast
    method for estimating predictability from an operational archive. *Knob:* forecast
    lead.
14. **From chaos to turbulence** — Dissipative low-order chaos vs. fully developed
    turbulence; predictability in 2-D and quasi-geostrophic flow; error growth as a
    spectral process. *Reuse:* the existing `real_butterfly_effect` and `sqgturb`
    work. *Knob:* Reynolds/resolution.

### Part V — The machinery of prediction

15. **Tangent linear and adjoint models** — Linearisation; the adjoint identity
    $\langle \mathbf{M}x, y\rangle = \langle x, \mathbf{M}^{\!\top}y\rangle$; the
    window of validity. *Notebook:* the finite-difference validation curve — the
    reader watches the discrepancy fall linearly in the perturbation amplitude, the
    defining test of a correct tangent linear model. *Knob:* $\tau$, amplitude.
16. **Adjoint sensitivity and optimal perturbations** — Singular vectors vs. Lyapunov
    vectors; sensitivity to the initial state and to observations; targeted observing.
    *Notebook:* the leading singular vector of L63/L96 over a chosen window, and the
    amplification it achieves against the Lyapunov estimate. *Knob:* optimisation
    time.
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
| Chapters live | **4** (4, 6, 7, 20) |
| Chapters stubbed | 27 |
| `chaoslib` modules | 10 |
| `chaoslib` tests | 109, all passing |

**Next chapters, in priority order** — each already has most of its material in hand:

1. **Ch. 15** (tangent linear and adjoint) — `chaoslib.adjoint` is complete and
   tested, including the finite-difference validation curve the chapter is built on,
   and chapter 20 already depends on it conceptually.
2. **Ch. 5** (maps and bifurcations) — small, self-contained, no new library code.
3. **Ch. 11** (Lorenz 96) — `chaoslib.systems.lorenz96` and the spectrum are tested.
4. **Ch. 8** (dimension) — `chaoslib.dimension` is tested, and chapter 7 already
   promises that chapter's independent check of $D_{KY}$.
5. **Ch. 18/19** (variational and ensemble DA) — chapter 20 currently carries their
   theory sections; see the note below.

### A second decision on record: splitting chapter 20

Chapter 20 covers 3D-Var, 4D-Var *and* the EnKF, which properly belong to chapters 18,
19 and 20 respectively. It is kept whole for the same reason chapter 6 is: it is a
coherent, tested teaching artefact, and breaking it up before its neighbours exist
would leave gaps. As chapters 18 and 19 are written they should take the corresponding
theory sections, and chapter 20 should shrink to cycling, analysis error and the
logarithmic return — which is its real subject.
