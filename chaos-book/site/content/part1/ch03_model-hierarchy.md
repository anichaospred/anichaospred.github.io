---
title: "Chapter 3 · The hierarchy of models"
weight: 103
part: "Part I — What predictability means"
knob: 'which rung, initial-error amplitude'
status: "live"
---

## Overview

This book spends most of its length on systems with three variables, or forty, or one.
An operational forecast model has of order $10^9$. So the question the whole book
depends on: **what exactly is a small model telling you about a large one?**

"It builds intuition" is not a claim that can be checked. The claim this book actually
makes is narrower and testable:

> **Dimensionless relationships transfer between rungs. Dimensional constants do not.
> And some properties exist at no rung below a particular structure, so they cannot be
> learned by going smaller at all.**

Each is measured here. The first two hold cleanly. The third is the one that matters,
because it says where the method fails — and the property it fails on is the most
famous claim in the subject.

## The rungs

| rung | variables | time | what it is for |
|---|---|---|---|
| logistic map | 1 | discrete | the route to chaos, and universality (ch. 5) |
| Lorenz 63 | 3 | continuous | sensitive dependence, the strange attractor (ch. 6) |
| Lorenz 96 (12) | 12 | continuous | spatiotemporal chaos, small enough to see (ch. 11) |
| Lorenz 96 (40) | 40 | continuous | the standard assimilation testbed (ch. 19) |

They have almost nothing in common — a map on the unit interval, a flow on a fractal
attractor of dimension 2.06, a flow on a 27-dimensional one — and their leading
exponents are not in the same units. What they all have is an error that grows and then
saturates, so there is one quantity to compare. The machinery is checked against an
exact identity at every rung: the exponents sum to the divergence of the flow,
$\sum_i\lambda_i = \mathrm{tr}\,\mathbf{J}$, reproduced to parts in $10^5$.

{{< marimo src="/nb/ch03_model-hierarchy.html" >}}

## What transfers: one law, no fitted constant

If error grows as $\delta_0 e^{\lambda_1 t}$ until it saturates at $\delta_\infty$, the
time to reach a fraction $f$ of saturation satisfies

$$
\lambda_1 T = \ln \frac{f\,\delta_\infty}{\delta_0} .
$$

That form is dimensionless on both sides, has no fitted constant, and each rung supplies
its own $\lambda_1$ and $\delta_\infty$. Measured at $f = 0.5$ over four rungs and four
initial-error amplitudes spanning six decades:

| $\lambda_1 T$ | $d=8$ | $d=6$ | $d=4$ | $d=2$ |
|---|---|---|---|---|
| logistic map | 17.07 | 12.55 | 8.03 | 4.02 |
| Lorenz 63 | 16.86 | 12.69 | 7.86 | 3.63 |
| Lorenz 96 (12) | 17.26 | 12.59 | 8.04 | 3.55 |
| Lorenz 96 (40) | 17.22 | 12.96 | 8.74 | 4.07 |
| **the law predicts** | **17.73** | **13.12** | **8.52** | **3.91** |

**Worst departure anywhere: 9.2 %.** A one-variable discrete map and a forty-variable
continuous flow, with leading exponents differing by a factor of three and not even in
comparable units, agree with a constant-free law to within a tenth.

That licenses using Lorenz 63 to reason about how forecast error responds to better
initial conditions — the *shape* of that dependence is the same at every rung, and each
decade of accuracy buys $\ln 10/\lambda_1$ of lead time, which is what
[chapter 20]({{< relref "../part5/ch20_da-in-practice.md" >}}) measures on a cycling
assimilation system. It licenses transferring no number at all: $\lambda_1$ is 0.50 per
iteration for the map against 1.63 per time unit for Lorenz 96.

The departures are not zero for two identifiable reasons, both pushing the same way: a
random perturbation is not aligned with the leading Lyapunov vector and spends a while
becoming one, and the local growth rate varies around the attractor by a factor of
several ([chapter 7]({{< relref "../part3/ch07_lyapunov-exponents.md" >}})), so
averaging over 48 base states does not average over everything.

## What does not transfer: how many directions the error lives in

Section 2 could be read too generously, as though getting $\lambda_1$ right were enough.
Lorenz 96 at $F=8$ is the same system at every $N$ — same equations, same forcing, a
longer chain — so climb it slowly.

Across a factor of five in $N$, $\lambda_1$ varies by **13 %**: it is a property of the
local dynamics and does not care how long the chain is. Meanwhile the number of unstable
directions goes **2 → 13**, growing at 0.34 per variable, and the attractor dimension at
0.68 per variable. Both are *extensive*.

**So a small model can have exactly the right growth rate and be wrong about the size of
the problem by orders of magnitude** — and the extensive quantity is the one that sets
operational cost. An ensemble must span the unstable subspace, so its size is set by the
unstable dimension and not by $\lambda_1$; that is why
[chapter 19]({{< relref "../part5/ch19_ensemble-da.md" >}}) finds localisation
compulsory in a large system and why its rank problem does not appear in Lorenz 63 at
all. The observation count scales the same way, and so does the number of singular
vectors that matter ([chapter 16]({{< relref "../part5/ch16_adjoint-sensitivity.md" >}})).

Lorenz 63, with **one** unstable direction, cannot pose those questions. It does not
answer them wrongly — it does not contain them.

## Where the ladder breaks

The third category is worse than the second: a property present at **no rung below a
particular structure**, so going smaller cannot teach it.

Lorenz (1969) argued the atmosphere has a **finite** predictability limit — a horizon
better observations cannot push back. The argument needs a *spectrum* of scales. But the
law above says the horizon grows **without bound** as $\delta_0 \to 0$, every decade
buying another $\ln 10/\lambda_1$ for ever. Both cannot be right about one system.

Measured on 24 octave bands with band growth rates scaling as $\ell^{-\alpha}$, error
seeded in the smallest band, over ten decades of initial accuracy:

- $\alpha = 0$ (every single-scale system in this book): the horizon gains **7.29 time
  units** and is still climbing;
- $\alpha = 2/3$ (Kolmogorov, the value Lorenz's argument needs): it gains **0.001**,
  against a limit of 1.45 that the first decade already reached.

**So the finite predictability limit is a property no system in this book has**, except
the cascade built to have it. The logistic map does not. Lorenz 63 does not. Lorenz 96
does not. Two-scale Lorenz 96, which looks as though it should, does not either:
[chapter 12]({{< relref "../part4/ch12_scale-dependent-error-growth.md" >}}) measures
its horizon gain at 0.145 time units per decade with error seeded in the fast variables
against 0.148 in the slow ones — a 2 % difference, so it makes essentially no difference
*where* the error is put, which a real upscale cascade would never permit. Two rungs of
scale separation are not a spectrum.

This is the sharpest limitation on everything else here, and it cuts in a specific
direction. Chapter 20's logarithmic return and chapter 22's day-per-decade skill record
are both correct about their systems, and neither can settle whether the real
atmosphere's horizon is bounded — because that depends on structure the low-order models
lack.

## Exercises

1. At which rung and which initial-error amplitude is the departure from the law
   largest, and does its sign tell you which of the two spoiling effects dominates?
2. The unstable dimension grows at 0.34 per variable. Estimate it for a model with
   $10^7$ variables, and say what that implies for an ensemble anyone can afford.
3. The law and the cascade disagree about the limit as $\delta_0 \to 0$, and both are
   measured here. Explain precisely which assumption of the law fails in the cascade.
4. Sort a result from a later chapter into the three categories — chapter 9's
   saturation, chapter 11's $m^* = 8$, chapter 27's Airy constant. Which transfer?

## Further reading

- Lorenz, E. N. (1969). The predictability of a flow which possesses many scales of
  motion. *Tellus*, **21**, 289–307 *[citation needed: pages]*.
- Held, I. M. (2005). The gap between simulation and understanding in climate modeling.
  *Bulletin of the American Meteorological Society* *[citation needed]* — the case for
  hierarchies, made carefully.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*
  *[citation needed: chapter]*.
- Schneider, T. et al., on model hierarchies and the limits of small models
  *[citation needed]*.
