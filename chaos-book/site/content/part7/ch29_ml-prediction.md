---
title: "Chapter 29 · Machine learning and data-driven prediction"
weight: 729
part: "Part VII — Frontiers"
knob: 'reservoir size, spectral radius, training length'
status: "live"
---

## Overview

Learned weather models now match or beat physics-based forecasts on the scores
operational centres publish *[citation needed]*. The standard worry is that they have
learned to *interpolate* rather than to *integrate* — that a model fitted to reanalysis
reproduces the atmosphere's statistics without its dynamics, and will fail in ways the
scores do not reveal.

That worry is testable. [Chapter 3]({{< relref "../part1/ch03_model-hierarchy.md" >}})
established which quantities characterise a chaotic system's dynamics — the Lyapunov
spectrum, the unstable dimension, the horizon law — so "does the emulator inherit the
dynamics" becomes a comparison of two censuses.

The answer is more interesting than the worry. **It inherits far more than the sceptical
story predicts**, and the failure that survives is specific, robust, and exactly the
quantity chapter 3 identified as the one that matters operationally.

## The emulator

A reservoir computer. A fixed random recurrent network is driven by the data,
$\mathbf{r}_{k+1} = \tanh(\mathbf{W}\mathbf{r}_k + \mathbf{W}_{\rm in}\nu(\mathbf{u}_k))$,
and only a **linear readout** is fitted, by ridge regression, to the one-step increment.
Training is one linear solve — no optimiser, no learning rate, no stopping criterion, and
the same answer every time.

Two reasons for a reservoir rather than a neural network, the second substantive: gradient
descent on a large network will not run in a browser, and — more importantly — the
reservoir's **tangent map is analytic**, so the emulator's own Lyapunov spectrum can be
computed by exactly the algorithm chapter 7 applies to the true system rather than
estimated from twin trajectories. The headline is a comparison of spectra, and it would be
much weaker with an estimate on one side. What this costs: a reservoir is not a
transformer, and nothing here settles what a large learned weather model does. What
transfers is the *method* — the census, and which of its entries are fragile.

{{< marimo src="/nb/ch29_ml-prediction.html" >}}

## It inherits the dynamics, and that is the surprise

Lorenz 96 with 8 sites at $F = 8$ has spectrum $+1.592, +0.367, +0.001, -0.475, -0.968,
-1.513, -2.349, -4.654$, summing to exactly $-8$, with 3 unstable directions and
$D_{KY} = 5.34$.

A reservoir fitted by one linear solve to one-step increments reproduces **the leading
eight exponents to 1.5–4.1 %** of $|\lambda|_{\max}$ across every configuration but one.
Alongside that: every configuration free-runs for 20,000 steps without diverging,
climatological spread within 1.3 % of the truth's, and tracking times of 1.3–4.7 time
units — 2 to 7 e-folding times.

The third exponent is the one to stare at. The truth's is $+0.00072$: the neutral
direction along the flow, 0.05 % of $\lambda_1$. **The emulator finds it.** Nothing in the
training objective mentions Lyapunov exponents, neutral directions or the flow.

So the strong form of the sceptical story is measurably wrong. A model fitted only to
short-horizon increments does not merely interpolate statistics; it recovers the
linearised dynamics about the attractor in detail, including a direction whose growth rate
is indistinguishable from zero.

## Why small one-step error still buys a short forecast

The emulator's error after a single step is $7.87\times10^{-5}$. Chapter 3's horizon law
says an *initial-condition* error that small should leave the truth predictable for
**8.11 time units**. The emulator loses skill in **2.87** — a factor of 2.8 shorter from
the same starting error.

The reason is not subtle once stated: an initial-condition error is injected **once** and
then amplified; model error is injected **at every step** and then amplified. So chapter
3's law is the wrong law for a learned model — it gives an upper bound, and a generous
one. This is [chapter 21]({{< relref "../part5/ch21_model-error.md" >}})'s subject in a
new form, and it is the honest answer to "the one-step error is tiny, so why is the
forecast short".

## Two ways it goes wrong, and only one is visible

**The loud one is a hard structural condition.** The reservoir's spectral radius must be
below one — the echo state property — or it does not forget its own past and the
free-running emulator has no reason to stay bounded. At $\rho = 1.4$ the rollout tracks
normally for about 15 time units and then diverges by four orders of magnitude, blowing up
after 2,110 steps. The useful part: that failure is diagnosable **with no reference data
at all**, since $\rho(\mathbf{W})$ is a property of the untrained network.

**The quiet one** is a configuration stable for the full run, with climatological spread
within 1.3 % of the truth's, whose Lyapunov spectrum is wrong by **23 %**. Its tracking
time, 1.25 time units, is the shortest of any stable configuration, so a forecast score
*would* eventually catch it.

And the diagnostics computable without a reference model are only partly informative.
Across the eight stable configurations, spectrum error correlates with log one-step error
at **+0.77** and with climatological-spread error at **+0.77**. Both are positive and
moderately strong — worth saying plainly, because the cynical version of this chapter
would call them useless. What they cannot do is **rank** the models that pass: among
configurations with spectrum errors between 1.5 % and 4.1 %, the ordering by one-step
error is not the ordering by spectrum error.

## The failure that does not go away: the count, not the rates

Chapter 3 established that $\lambda_1$ transfers between models and the **unstable
dimension** does not, and that the unstable dimension is what sets ensemble size,
observation count, and how many singular vectors matter. So it is the census entry an
emulator most needs to get right.

**Every configuration resolves the third exponent to under 0.6 % of $\lambda_1$. Two of
eight get its sign wrong**, and so report 2 unstable directions instead of 3.

That is not an accuracy failure, and no amount of extra training or reservoir size fixes
it. The unstable dimension is a **discrete functional of a continuous quantity that
happens to be zero**: getting it right means resolving the sign of a number
indistinguishable from zero, which is a different demand from getting a rate right to a
few per cent. The emulator is accurate enough for the latter and cannot in principle be
accurate enough for the former.

So the honest summary inverts the sceptical story rather than confirming it. The emulator
is **most** reliable on the rates, which are what the sceptics doubted, and **least**
reliable on the count, which determines what a forecasting system costs.

### And the check that would settle it does not exist

Lorenz 96's exponents sum to $-8.00000$ against an exact $-8$ — the divergence of the
flow. That identity has validated **every** Lyapunov calculation in this book: chapter 7's
spectrum, chapter 3's census, and this chapter's own truth column.

The emulator has no such identity. Its state space is a thousand dimensions of reservoir
plus eight of physics, with no phase-space volume to conserve and no divergence to compare
against. **The single most powerful validation tool available for a physical model is
unavailable for a learned one** — not difficult, not expensive, but absent.

That is the structural cost of replacing equations with a fit, and it deserves more
attention than the question this chapter opened with. A learned model can be verified
against data, which is what scores do. It cannot be verified against the identities that
made us trust the equations in the first place.

## Exercises

1. The third exponent's sign decides the unstable dimension. Estimate how much training
   data would be needed to resolve it, given that the exponent is 0.05 % of $\lambda_1$,
   and say whether that number is achievable.
2. Model error and initial-condition error grow differently. Write down the modification
   to chapter 3's horizon law that a constant error injection per step would imply, and
   check it against the figure.
3. The echo state property is checkable without data. Name another property of a learned
   forecast model certifiable from the weights alone, and one that is not.
4. Chapter 3 showed that no low-order system in this book has a finite predictability
   limit. What would that imply for an emulator trained on a system that *does* have one,
   and could the census detect the difference?

## Further reading

- Pathak, J. et al. (2018). Model-free prediction of large spatiotemporally chaotic
  systems from data: a reservoir computing approach. *Physical Review Letters*, **120**,
  024102 *[citation needed: pages]* — reservoir computers reproducing Lyapunov spectra.
- Jaeger, H. and Haas, H. (2004), on echo state networks and the echo state property
  *[citation needed]*.
- Lam, R. et al. (2023), on learned medium-range weather forecasting *[citation needed]*.
- Bonavita, M. (2024), on what learned weather models do and do not inherit
  *[citation needed]*.
- Brunton, S. L. and Kutz, J. N. (2019). *Data-Driven Science and Engineering.* Cambridge
  University Press *[citation needed: chapter]*.
