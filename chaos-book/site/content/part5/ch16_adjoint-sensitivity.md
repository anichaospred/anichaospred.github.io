---
title: "Chapter 16 · Adjoint sensitivity and optimal perturbations"
weight: 516
part: "Part V — The machinery of prediction"
knob: 'optimisation window $\tau$, the norm, the Lorenz 96 window'
status: "live"
---

## Overview

You have one extra observation to place, anywhere in the domain, and it will be taken
now. Where should it go?

[Chapter 15]({{< relref "ch15_tangent-linear-adjoint.md" >}}) answered a nearby question
— *what does this forecast quantity depend on?* — with the gradient $\partial J/\partial
x_0$. That is not quite what an observing plan needs. The gradient says where an error
would matter most; it does not say where an error is most likely to **grow**.

| | Question | Object |
|---|---|---|
| **Sensitivity** | What does *this metric* depend on? | $\mathbf{M}^{\top}\partial J/\partial x_\tau$ |
| **Optimal growth** | Which perturbation grows most, in *this norm*? | the leading singular vector of $\mathbf{M}$ |
| **Asymptotic growth** | Which direction grows in the long run? | the leading Lyapunov vector ([ch. 7]({{< relref "../part3/ch07_lyapunov-exponents.md" >}})) |

All three come out of the same propagator, and this chapter separates them.

## The model

Over a finite window $\mathbf{M}(x_0,\tau)$ maps a unit sphere onto an ellipsoid, and its
SVD names the axes: $v_1$ is the initial perturbation that grows most, $u_1$ is what it
becomes, and $\sigma_1$ is the amplification. At operational size $\mathbf{M}$ is never
formed — $\sigma_1$ and $v_1$ come from iterative methods needing only the action of
$\mathbf{M}$ and $\mathbf{M}^{\top}$ on a vector, which is one tangent linear and one
adjoint integration each. That is why chapter 15 came first.

{{< marimo src="/nb/ch16_adjoint-sensitivity.html" >}}

## Three results

**Optimal growth beats the Lyapunov estimate, systematically.** Averaged over 33 base
points on the attractor, $\sigma_1$ exceeds $e^{\lambda_1\tau}$ by a factor of 1.6–2.6 at
every window tested, and the effective rate $\ln\sigma_1/\tau$ falls from 3.6 at
$\tau = 0.25$ to 1.0 at $\tau = 8$ — still above $\lambda_1 = 0.906$, and only there.
The two agree asymptotically, and "asymptotically" is well beyond any useful forecast
range. This is **non-normality**:
$\mathbf{M}$ is not symmetric, so its singular vectors are not its eigenvectors, and a
well-chosen direction transiently amplifies faster than the asymptotic rate of any
direction. It is why operational centres perturb along singular vectors rather than
randomly.

The averaging is not cosmetic. At a *single* base point $\sigma_1(\tau)$ is not even
monotonic, and beyond about τ = 3 MTU it can fall **below** $e^{\lambda_1\tau}$ — because
$\lambda_1$ is a long-time average and one particular stretch of trajectory may be
quieter than average. The inequality is a statement about the attractor, not about any
one window.

**"Fastest-growing" is undefined without a norm.** Singular vectors solve
$\max_v \|\mathbf{M}v\|_E/\|v\|_E$, which depends on $\mathbf{E}$. Weighting one
component by 25 rotates the optimal direction by tens of degrees. Operational singular
vectors are computed in a total-energy norm, and that choice was argued over for years,
because a norm favouring small scales produces perturbations that grow impressively and
matter little. Choosing $\mathbf{E}$ is choosing what "an important error" means.

**Sensitivity is not growth — but you cannot see that in Lorenz 63.** Expanding the
gradient in the singular basis,
$\mathbf{M}^{\top}\partial J/\partial x_\tau = \sum_i \sigma_i (u_i\cdot\partial J/\partial x_\tau)\,v_i$,
shows two conditions for $v_1$ to dominate: $\sigma_1 \gg \sigma_2$, and the metric must
overlap $u_1$. In Lorenz 63 both hold for any natural metric ($\sigma_1/\sigma_2 \approx
34$), so the gradient lies within a few degrees of $v_1$ and the two questions look
interchangeable. In Lorenz 96 at τ = 0.5, with $\sigma_1/\sigma_2 \approx 1.8$ and a local
metric overlapping $u_1$ by 0.003, the angle is **88.8°** — near-orthogonal.

Nor is that a short-window artefact that a longer optimisation cures. Stepping the
window from 0.25 to 2 MTU gives 88.6°, 88.8°, 68.6°, 24.6°, 64.8°, 84.1°, 85.2°, 85.9°:
one dip, at τ = 1, where the overlap happens to reach 0.44. $\sigma_1/\sigma_2$ stays
between 1.1 and 2.9 the whole way, so the angle is set by the overlap alone — by whether
the fastest-growing structure happens to land on the site being forecast. In high
dimension with a local metric, **near-orthogonality is generic and agreement is
coincidence**. The distinction is real; it is invisible in a three-variable model, which
is a useful reminder about what low-order models can and cannot show.

## Exercises

**Analytic.** Derive the singular-basis expansion of the gradient above, and state the two
conditions under which $\nabla J \parallel v_1$. Then construct a metric for which the
angle is exactly 90°, and explain why it is not a metric anyone would forecast.

**Computational.** Verify that the returned $\sigma_1$ is the amplification actually
achieved in the chosen norm — $\|\mathbf{M}v_1\|_E/\|v_1\|_E$ — for a diagonal and for a
full symmetric positive definite weight.

**Exploratory.** Predict how the Lorenz 96 gradient–singular-vector angle changes as
the window grows, then step through the whole range. Reconcile what you see with the
$\sigma_1/\sigma_2$ argument, and say what the result implies for designing an observing
network from singular vectors alone.

## Further reading

- Buizza, R. and Palmer, T. N. (1995). The singular-vector structure of the atmospheric
  global circulation. *Journal of the Atmospheric Sciences*, **52**, 1434–1456.
- Palmer, T. N., Gelaro, R., Barkmeijer, J. and Buizza, R. (1998). Singular vectors,
  metrics, and adaptive observations. *Journal of the Atmospheric Sciences*, **55**,
  633–653 — the norm question, argued properly.
- Errico, R. M. (1997). What is an adjoint model? *BAMS*, **78**, 2577–2591.
- Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability*, §6.4.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
  ch. 5 *[citation needed: pages]*.
