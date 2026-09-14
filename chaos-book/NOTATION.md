# Notation

The symbols and sign conventions here are **mandatory** across every chapter and
every `chaoslib` docstring. A chapter that needs a new symbol adds it here in the
same commit. Where the two course texts disagree, this book follows **Kalnay (2003)**
for data assimilation and **Palmer & Hagedorn (2006)** for predictability.

Units are SI unless a standard domain alternative is noted.

## Dynamical systems

| Symbol | Meaning | Notes |
|---|---|---|
| $x$ | state vector | $x \in \mathbb{R}^n$; $n$ is the state dimension |
| $\dot x = f(x)$ | the dynamics | autonomous unless stated |
| $\mathcal{M}$ | the nonlinear model map | $x(t+\tau) = \mathcal{M}(x(t))$ |
| $t$ | time | MTU (model time units) for L63; time units for L96 |
| $\tau$ | lead time / optimisation window | always positive |
| $\mathbf{J}$ | Jacobian $\partial f_i/\partial x_j$ | of the *continuous* dynamics |
| $\mathbf{M}$ | tangent linear propagator | of the *discrete* map over $\tau$ |
| $\mathbf{M}^{\!\top}$ | the adjoint | transpose under the Euclidean inner product |
| $\delta x$ | perturbation / error | $\delta_0 = \|\delta x(0)\|$ |
| $\lambda_i$ | Lyapunov exponents | ordered $\lambda_1 \ge \lambda_2 \ge \dots$, units of inverse time |
| $\lambda(x,\tau)$ | finite-time (local) exponent | a property of a *state*, not the attractor |
| $\sigma_i$ | singular values of $\mathbf{M}$ | ordered descending; $\sigma_1$ is optimal growth |
| $h_{KS}$ | Kolmogorov–Sinai entropy | $\sum_{\lambda_i>0}\lambda_i$, nats per unit time |
| $D_{KY}$ | Kaplan–Yorke dimension | from the spectrum |
| $D_2$ | correlation dimension | from a sampled trajectory |
| $T_d$ | error-doubling time | $\ln 2/\lambda_1$ |
| $\mathcal{K}^\tau$ | Koopman operator | $(\mathcal{K}^\tau g)(x) = g(\mathcal{M}^\tau(x))$; linear on observables |
| $g$ | an observable, or a dictionary of them | a function of the state, not a state |
| $\mathbf{K}$ | the finite-dictionary compression of $\mathcal{K}^\tau$ | what EDMD fits |
| $\phi$ | a Koopman eigenfunction | $\mathcal{K}^\tau\phi = e^{\lambda\tau}\phi$ |
| $\nu$ | invariant (natural) measure | "climate" as a distribution on state space |
| $\langle A\rangle_\nu$ | ensemble average | $\int A\,d\nu$; the object a climatology estimates |
| $\bar A_T$ | time average over a window $T$ | the estimator a control run provides |
| $\tau_{\rm int}$ | integrated autocorrelation time | $\int\rho(s)\,ds$; a *time*, not a lag count |
| $\tau_{\rm erg}$ | ergodic time | how long to sample the whole attractor, not just its fast part |
| $W_1$ | 1-Wasserstein distance between measures | carries the units of the variable |
| $D_N^*$ | star discrepancy of a sample | sampling error of the worst indicator observable |

**Time-unit convention.** For Lorenz 63, 1 MTU is read as $\approx 5$ atmospheric
days; for Lorenz 96 with $F=8$, 1 time unit is read as 5 days. These are
interpretive conventions, not derivations — state them as such.

## Specific systems

| Symbol | Meaning |
|---|---|
| $\sigma, \rho, \beta$ | Lorenz 63 parameters (Prandtl, Rayleigh, geometry) |
| $\rho_H$ | Hopf threshold, $\sigma(\sigma+\beta+3)/(\sigma-\beta-1)$ |
| $C^\pm$ | the non-trivial Lorenz 63 fixed points |
| $N, F$ | Lorenz 96 site count and forcing |
| $r$ | logistic-map parameter |
| $\mu$ | double-well tilt; the fold is at $\mu_c = 2/3\sqrt3$ |
| $\theta, \omega$ | pendulum angle and angular velocity |
| $L, g, m$ | pendulum length, gravity, mass |

Note the collisions, all of them deliberate rather than tidied away, because each
symbol is standard in its own literature. $\mathbf{K}$ is the Kalman gain of chapters 18
to 20 **and** the finite-dictionary Koopman operator of chapter 31; the two never appear
in the same chapter, and chapter 31 says so where it introduces it. $g$ is gravity for
the pendulum and an observable for the Koopman operator. $\mu$ is the double-well tilt of chapters 27 and 30 and is the
conventional symbol for an invariant measure; this book writes the measure $\nu$
throughout and reserves $\mu$ for the tilt. And $\sigma$ is the Lorenz 63 Prandtl number,
the noise amplitude of a stochastic run, **and** the conventional symbol for a singular
value. Chapters using both write singular values as $\sigma_i$
with an explicit subscript, and say so in the text.

## Predictability and error growth

| Symbol | Meaning |
|---|---|
| $E(t)$ | forecast error (RMS unless stated) |
| $E_\infty$ | saturation error — the climatological level |
| $\Delta t$ | gain in forecast horizon |
| $D(p\|q)$ | relative entropy (Kullback–Leibler), nats |
| $I(X;Y)$ | mutual information, nats |
| $H(p)$ | Shannon entropy, nats |

**Information units are nats** (natural logarithm) everywhere. Divide by $\ln 2$ for
bits; never mix the two in one figure.

## Data assimilation (following Kalnay 2003)

| Symbol | Meaning |
|---|---|
| $x^b$ | background (prior) state |
| $x^a$ | analysis (posterior) state |
| $x^t$ | true state |
| $y$ | observations |
| $\mathbf{H}$ | observation operator |
| $\mathbf{B}$ | background error covariance (fixed, as in 3D-Var) |
| $\mathbf{P}^b, \mathbf{P}^a$ | background / analysis error covariance (evolving) |
| $\mathbf{R}$ | observation error covariance |
| $\mathbf{K}$ | Kalman gain |
| $J(x)$ | variational cost function |
| $N$ | ensemble size |

**Sign convention for innovations:** $d = y - \mathbf{H}x^b$, observation minus
background. The analysis increment is $\mathbf{K}d$, so a positive innovation moves
the analysis towards the observation.

## Ensembles and verification

| Symbol | Meaning |
|---|---|
| $\bar x$ | ensemble mean |
| $s$ | ensemble spread (RMS about the mean, $N-1$ normalisation) |
| CRPS | continuous ranked probability score |
| BS | Brier score |
| ACC | anomaly correlation coefficient |

**Calibration identity:** a reliable ensemble satisfies RMS spread $=$ RMS error of
the ensemble mean. State any departure from it as under- or over-dispersion, not as
"error".
