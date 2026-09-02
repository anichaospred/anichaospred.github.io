---
title: "Chapter 15 · Tangent linear and adjoint models"
weight: 515
part: "Part V — The machinery of prediction"
knob: 'lead time $\tau$, perturbation amplitude'
status: "live"
---

## Overview

Monday's forecast for Thursday was badly wrong. Which part of Monday's initial state was
responsible, and by how much would each part have had to change to fix it?

That is a question about a **gradient** — the derivative of one forecast quantity with
respect to every component of the initial state. The obvious way to get it is to perturb
each component in turn and re-run the model. For Lorenz 63 that is four runs. For an
operational model with $10^8$ state variables it is a hundred million runs, and it is
never going to happen.

The **adjoint** delivers the whole gradient in one extra model-sized integration, at a
cost independent of the state dimension. That single fact is why 4D-Var
([chapter 18]({{< relref "ch18_variational-da.md" >}})) exists, why singular vectors
([chapter 16]({{< relref "ch16_adjoint-sensitivity.md" >}})) are computable, and why
targeted observing is a real technique rather than a thought experiment.

| State dimension $n$ | Finite differences | Adjoint |
|---|---|---|
| 3 (Lorenz 63) | 4 model runs | 2 |
| 40 (Lorenz 96) | 41 | 2 |
| $10^8$ (operational NWP) | $10^8 + 1$ | 2 |

## The model

The tangent linear propagator $\mathbf{M}(x_0,\tau) = \partial\mathcal{M}/\partial x$
satisfies $\dot{\delta x} = \mathbf{J}(x(t))\,\delta x$ along the nonlinear trajectory. It
is **state-dependent** — the same flow-dependence chapter 7 measured as finite-time
exponents — and it is **linear**, so once you have it the effect of any small perturbation
is a matrix-vector product.

One detail is easy to get wrong and hard to notice: the model you actually run is the
*discretised* one, so the tangent must be stepped through the **same RK4 stages**, with
$\mathbf{J}$ evaluated at each intermediate state. Freezing $\mathbf{J}$ across the step
leaves an $O(h)$ inconsistency between the tangent and the discrete map.

The adjoint is defined by the identity
$\langle \mathbf{M}u, v\rangle = \langle u, \mathbf{M}^{*}v\rangle$. Under the Euclidean
inner product $\mathbf{M}^{*} = \mathbf{M}^{\top}$, which makes it look trivial — but the
*identity* is the definition, and under a weighted norm the adjoint acquires the weights,
which is where operational adjoints go wrong.

{{< marimo src="/nb/ch15_tangent-linear-adjoint.html" >}}

## Two limits, and two tests

**The linearisation has a shelf life.** The neglected term is $O(\|\delta x\|^2)$ and
$\|\delta x\|$ grows like $e^{\lambda_1 t}$, so the window over which the tangent model is
trustworthy is

$$\tau_{\text{valid}} \approx \frac{1}{\lambda_1}\ln\frac{\delta_c}{\delta_0},$$

measured here at **1.18 MTU per e-fold** of $\delta_0$ against $1/\lambda_1 = 1.10$. This
is the logarithmic law for the third time — chapter 7 found it in the growth rate, chapter
20 in the forecast horizon, and here it sets how long a 4D-Var assimilation window can be.
A hundredfold smaller perturbation buys about five extra MTU of linearity, not a
hundredfold longer window.

**There are exactly two tests worth running**, and you need both:

1. **The adjoint identity**, which must hold to machine precision ($\sim10^{-14}$) because
   it is algebraic rather than approximate. It tests $\mathbf{M}^{\top}$ against
   $\mathbf{M}$ and says nothing about whether $\mathbf{M}$ is right.
2. **The finite-difference check**, in which the relative discrepancy against the true
   nonlinear difference must fall **linearly** with the perturbation amplitude — slope 1
   on log–log, down to round-off. This is what catches a wrong Jacobian.

An adjoint that passes only the first can be the perfect adjoint of the wrong tangent
linear model.

## Three bugs these tests caught, in this book

Not hypothetical. All three were live in `chaoslib` during writing, and each was found by
a test rather than by reading the code:

- **A floor that no smaller perturbation removed** — a constant 4.6 % error, independent
  of amplitude, from linearising the continuous flow instead of the discrete map.
- **A zero-length window that advanced anyway** — `max(1, round(tau/dt))` took one step
  when $\tau = 0$, corrupting essentially every cycling 4D-Var gradient by ~6.8 %. Found
  only when chapter 20's 4D-Var underperformed for no visible reason.
- **A propagator covering the wrong interval** whenever $\tau$ was not an exact multiple
  of the step — caught by Liouville's theorem,
  $\det\mathbf{M} = e^{\tau\,\operatorname{tr}\mathbf{J}}$, which pins the interval
  independently of the trajectory and is checked live in the notebook.

## Exercises

**Analytic.** Show that $\partial J/\partial x_0 = \mathbf{M}^{\top}\,\partial J/\partial
x_\tau$, and hence that the cost of the gradient is independent of $n$. Then derive the
adjoint under a weighted inner product $\langle u,v\rangle_E = u^{\top}\mathbf{E}v$ and
say what changes.

**Computational.** Verify Liouville's theorem for this system: $\det\mathbf{M}$ must equal
$e^{-(\sigma+1+\beta)\tau}$ for every trajectory and every $\tau$. Why does that test
constrain the *interval* the propagator covers, when the trace identity of chapter 7
constrains the exponents?

**Exploratory.** Move the lead time from 0.25 to 4 MTU and watch the validation curve
shift upward while its slope stays at 1. Accuracy and correctness are different
properties — only one of them is a bug. At what amplitude does the round-off floor set in,
and what does that imply about testing an adjoint with $\alpha = 10^{-12}$?

## Further reading

- Errico, R. M. (1997). What is an adjoint model? *Bulletin of the American
  Meteorological Society*, **78**, 2577–2591 — the clearest introduction there is.
- Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability*,
  §6.2–6.3.
- Giering, R. and Kaminski, T. (1998). Recipes for adjoint code construction.
  *ACM TOMS*, **24**, 437–474 *[citation needed: pages]*.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
  ch. 5 *[citation needed: pages]*.
