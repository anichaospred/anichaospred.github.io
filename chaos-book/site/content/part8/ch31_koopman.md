---
title: "Chapter 31 · The Koopman operator"
weight: 831
part: "Part VIII — Structure"
knob: 'dictionary of observables, rollout scheme, lift factor'
status: "live"
---

## Overview

Every data-driven forecast system in this book's last two parts — the emulator of
[chapter 29]({{< relref "../part7/ch29_ml-prediction.md" >}}), dynamic mode
decomposition, the "AI weather models" the field is now built on — ends up fitting a
**linear** operator to an atmosphere that is not linear. Sometimes that works remarkably
well. When it fails, it fails abruptly.

**Why does linearising a nonlinear atmosphere ever work at all, and what is being given
up when it does?** The answer is not that the atmosphere is nearly linear. It is that
there exists an *exact* linear representation of any nonlinear system, and every
practical method is a truncation of it. This chapter builds the exact object, truncates
it deliberately, and measures what the truncation costs.

## The construction

Instead of asking where a state goes, ask what happens to **observables** — functions
$g(x)$ of the state. For the flow map $\mathcal{M}^\tau$, define

$$
\left(\mathcal{K}^\tau g\right)(x) = g\!\left(\mathcal{M}^\tau(x)\right).
$$

Then $\mathcal{K}^\tau(ag + bh) = a\,\mathcal{K}^\tau g + b\,\mathcal{K}^\tau h$
**exactly**, however nonlinear $\mathcal{M}$ is. There is no small parameter, no
linearisation point, no window of validity —
[chapter 15]({{< relref "../part5/ch15_tangent-linear-adjoint.md" >}})'s tangent linear
model is an approximation that degrades with amplitude and lead; this is not an
approximation at all.

The catch is in what it acts on. $\mathcal{K}$ is linear on a space of *functions*,
infinite-dimensional and with no finite basis. Three coupled equations have been traded
for one linear operator on an infinite-dimensional space. Nothing has been solved;
something has been moved. What a finite piece of it is worth is a measurement.

{{< marimo src="/nb/ch31_koopman.html" >}}

## A system where the linearisation is exact and finite

For the slow-manifold normal form $\dot x_1 = \mu x_1$,
$\dot x_2 = \lambda(x_2 - x_1^2)$ — the structure behind every quasi-equilibrium closure
in atmospheric modelling — the chain rule gives, on $g = (x_1, x_2, x_1^2)$,

$$
\dot g = \mathbf{A}g, \qquad
\mathbf{A} = \begin{pmatrix}
  \mu & 0 & 0 \\ 0 & \lambda & -\lambda \\ 0 & 0 & 2\mu
\end{pmatrix},
$$

because $\frac{d}{dt}x_1^2 = 2\mu x_1^2$: the *square* of a state variable is governed
linearly too. Handed nothing but a trajectory and those three observables, least squares
returns the eigenvalues $\mu$, $\lambda$, $2\mu$ with a closure residual of
$5\times10^{-16}$, and the resulting $3\times3$ matrix reproduces the entire nonlinear
trajectory over 40 time units to $3\times10^{-13}$.

Note there are **three** eigenvalues where the Jacobian of a two-dimensional system has
two. The extra one is $2\mu$, and it is not an artefact: products of Koopman
eigenfunctions are Koopman eigenfunctions and their eigenvalues add. The operator's
spectrum is always richer than the flow's.

### The result that spoils the obvious plan

The obvious plan is to use more observables. It does not work that way.

Dropping $x_1^2$ costs eleven orders of magnitude, as expected. But **adding** $x_1x_2$
to the dictionary that closes takes the residual from $5\times10^{-16}$ to
$7.5\times10^{-6}$ — a larger dictionary, containing the closing one, that no longer
closes. The reason is exact and checkable:

$$
\frac{d}{dt}\left(x_1x_2\right) = (\mu + \lambda)\,x_1x_2 - \lambda\,x_1^3 ,
$$

verified against the trajectory to $2\times10^{-5}$. Admitting $x_1x_2$ opens a leak into
$x_1^3$ that the dictionary cannot absorb; admitting $x_1^3$ as well closes it again.

**Closure is a property of the span, not of the size.** It is not a resolution to be
increased until the answer converges but an algebraic condition a dictionary either
satisfies or does not, and a bigger dictionary is not a safer one. The monomial hierarchy
shows the generic case — each order leaks into the next, so the residual falls steadily
and never reaches machine precision. Fifteen carefully chosen terms lose to three correct
ones by seven orders of magnitude.

## The exchange rate on an attractor

With no closing dictionary available, extended DMD fits the best operator on whatever
dictionary is at hand. On Lorenz 63 with a constant, the state, and $k$ Gaussian bumps
centred on visited states, the one-step fit improves steadily and does not overfit: from
4 to 1,004 observables the residual falls by a factor of 12,000, with the held-out
residual tracking the fitted one throughout. The spectral radius is exactly 1 at every
size — the constant function is an exact Koopman eigenfunction for *every* system, so
nothing here can blow up under iteration.

Iterating it is another matter.

| observables | useful lead (TU) | days | against plain DMD |
|---|---|---|---|
| 4 | 0.082 | 0.41 | 1× |
| 14 | 0.317 | 1.58 | 3.9× |
| 34 | 0.442 | 2.21 | 5.4× |
| 104 | 0.590 | 2.95 | 7.2× |
| 304 | 1.236 | 6.18 | 15.1× |
| 1004 | 1.312 | 6.56 | 16.1× |

**Plain DMD is worth 0.08 time units** — about ten hours — which is the honest baseline
for "fit a linear model to the state". **251 times the dimension buys 16 times the
lead**, and it flattens: the last tripling of the dictionary buys 6 %, at a useful lead
well short of what [chapter 12]({{< relref "../part4/ch12_scale-dependent-error-growth.md" >}})'s
intrinsic limit allows.

## What actually breaks

At a lead of 0.05 time units the 304-observable model's error is $2\times10^{-4}$ of the
climatological spread. Six steps later it has lost more than half of it — far faster than
growth at $\lambda_1$ can explain. The cause is geometric, not dynamical.

The dictionary vectors that are *the dictionary of some state* form a three-dimensional
surface inside the 304-dimensional dictionary space. Every fitted relation holds on that
surface, because every training pair lay on it, and **$\mathbf{K}$ does not preserve
it**. Measured directly, the rolled-out vector is 11 % off the surface after a quarter of
a time unit and 44 % off after one — and the linear rollout's error rises alongside it, to
the same magnitude.

The repair is to **re-lift** at every step: advance, read the state out, rebuild the
dictionary from it. That recovers a factor of 5.2 in useful lead, 6.4 time units against
1.24 — and lift $\to$ advance $\to$ project $\to$ lift is a *nonlinear* map. **The
re-lifted model is not a linear model**, and the linearity was the entire point: it is
what would have made the Kalman filter exact and the spectrum meaningful. Every claim
that a Koopman method "linearises the dynamics" describes the first scheme; every
implementation that forecasts competitively is the second.

Two further findings. Even re-lifted, the model's error grows at 1.55 per time unit
against $\lambda_1 = 0.90$ — **72 % faster than the atmosphere separates**, so model
error is re-injected at every step rather than merely amplified, which is exactly what
chapter 29 measured for a reservoir emulator of entirely different construction. And the
two schemes fail in opposite directions: after 100 time units the linear rollout has
decayed to the climatological mean, which is the *optimal* forecast at infinite lead,
while the re-lifted one drifts past climatology to an error of 1.35. The linear model
degrades gracefully into uselessness; the accurate one degrades past it.

## What the spectrum does say

A chaotic, mixing system has no discrete Koopman spectrum beyond the eigenvalue at 1 —
which is the formal version of "the truncation cannot be fixed by enlarging it". What a
truncation *does* capture are decay rates of correlations: properties of the operator
acting on the invariant measure of
[chapter 30]({{< relref "ch30_ergodic-theory.md" >}}), not of any particular orbit.

The leading eigenvalue is $1.00000000$, not approximately one, and its eigenfunction is
the constant — the statement that the measure is invariant. The operator predicts the
autocorrelation function of $x$ to within **0.028** over the first two time units, an
interval over which its own re-lifted forecast has already lost most of its skill. And it
recovers chapter 30's number: the integrated autocorrelation time of $x$ comes back as
**0.96** time units from the operator against **1.04** measured directly from the
trajectory, 7 % apart.

**A model that cannot say where the system will be in six time units can say how long a
run must be to establish its climate** — because the first is a question about an orbit
and the second is a question about the spectrum. Koopman methods are spectral tools, and
trajectory forecasting is the one application that asks for exactly what the truncation
discards.

## What it buys data assimilation

The reason the construction keeps being tried in this field is that the **Kalman filter
is exact for a linear model** ([chapter 19]({{< relref "../part5/ch19_ensemble-da.md" >}})),
and every difficulty in chapters 18 to 20 follows from the model not being linear. Two
things stop that from being the end of the story. The truncation is model error of the
re-injected kind, so a filter that assumes a perfect model is over-confident by the
margin section 4 measured. And the covariance has one entry per *pair of dictionary
elements*: an operational $n \approx 10^9$ lifted a hundredfold asks for $10^{22}$
entries, where chapters 19 and 20 spend their effort on not forming the $n \times n$
covariance even once.

## Exercises

**Analytic.**

1. Show that if $\phi_1, \phi_2$ are Koopman eigenfunctions with eigenvalues
   $\lambda_1, \lambda_2$ then $\phi_1\phi_2$ is one with eigenvalue $\lambda_1\lambda_2$.
   Use it to write down the whole Koopman spectrum of $\dot x = \mathbf{A}x$, and say why
   a finite dictionary can never contain it.
2. Find the smallest dictionary containing $x_1x_2$ that closes for the slow-manifold
   system. Generalise: given any monomial, which monomials must accompany it?
3. The linear rollout converges to the climatological mean and the re-lifted one does
   not. Prove the first from the spectral radius, and say what property the re-lifted map
   would need for the second.

**Computational.**

4. Repeat the sweep with the radial basis width as the knob at fixed dictionary size. Is
   the useful lead more sensitive to the width or to the count?
5. Fit the operator at several snapshot intervals and check whether
   $\mathbf{K}(2\tau) = \mathbf{K}(\tau)^2$. The exact operator satisfies it; measure how
   far the truncation is from a semigroup and compare with the off-manifold drift.
6. Use the operator's correlation function to estimate the run length chapter 30 asks
   for, and compare with the direct estimate for $z$ as well as $x$. Does the agreement
   survive for an observable the dictionary represents badly?

**Exploratory.**

7. Chapter 29's reservoir and this chapter's EDMD are both linear regressions onto a
   nonlinear feature set, differing mainly in whether the features have memory. Construct
   the comparison at matched feature count and say what the memory buys. Then argue about
   which failure — a spectrum whose neutral direction is miscounted, or an iteration that
   leaves the manifold — would matter more operationally.

## Further reading

- Koopman's operator-theoretic formulation of Hamiltonian dynamics *[citation needed]*
- Mezić on spectral properties of the Koopman operator and dimension reduction
  *[citation needed]*
- Dynamic mode decomposition and its exact formulation *[citation needed]*
- Williams, Kevrekidis & Rowley on extended DMD and its convergence to the Koopman
  operator as dictionary and data grow *[citation needed]*
- Ruelle–Pollicott resonances and the decay of correlations in mixing systems
  *[citation needed]*
- Koopman and transfer-operator methods in data assimilation, including the
  `quantum-koopman-da` line of work *[citation needed]*
