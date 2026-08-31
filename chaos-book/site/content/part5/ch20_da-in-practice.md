---
title: "Chapter 20 · Data assimilation in practice"
weight: 520
part: "Part V — The machinery of prediction"
knob: 'ensemble size $N$, inflation, observation interval, $\delta_0$'
status: "live"
---

## Overview

Every forecast starts from a state nobody knows. The atmosphere is observed at
scattered points, by instruments that disagree, at times that do not line up — and
from that a forecast centre must produce a complete, physically consistent initial
condition for a model with $10^8$ degrees of freedom. Data assimilation is how, and
this chapter runs the three algorithms that do it on a system small enough to watch
every step.

The chapter then asks the question that decides observing-system budgets: **if every
observation were ten times more accurate, how much forecast would that buy?** The
answer is not "ten times more". It is a fixed increment of lead time — and the next
factor of ten buys the same fixed increment again. Predictability is purchased in
units of $\ln$, which is why fifty years of enormous investment in observing systems
has moved the useful forecast range by roughly a day per decade rather than
transforming it.

## The model

A perfect-model twin experiment on Lorenz 63: a nature run stands in for the truth,
noisy observations are drawn from it every $\Delta t_{\rm obs}$, and a deliberately
wrong background stands in for prior knowledge. Three schemes then compete on the
same observations.

**3D-Var** minimises

$$\mathcal{J}(\mathbf{x}) = \tfrac12 (\mathbf{x}-\mathbf{x}^b)^T \mathbf{B}^{-1}(\mathbf{x}-\mathbf{x}^b) + \tfrac12 (\mathbf{y}-\mathbf{H}\mathbf{x})^T \mathbf{R}^{-1}(\mathbf{y}-\mathbf{H}\mathbf{x}),$$

a tug of war between what you believed and what you measured. Its limitation is
structural: $\mathbf{B}$ is the same on every day of the year, whatever the flow is
doing.

**4D-Var** fits one model trajectory to all observations in a window, so an
observation late in the window constrains the state at its start. The gradient needs
the **adjoint** built in [chapter 15]({{< relref "ch15_tangent-linear-adjoint.md" >}}) —
one adjoint application per observation time, rather than one model run per degree of
freedom. That asymmetry is the only reason variational assimilation is affordable at
operational size.

**The EnKF** estimates the background covariance from an ensemble, so it is
flow-dependent for free, at the cost of sampling error — hence inflation and
localisation.

All three come from `chaoslib.assimilate`, which is tested against the
linear-Gaussian Kalman filter for both the analysis **mean** and the analysis
**covariance**, and whose 4D-Var gradient is checked against central differences of
the same cost function to better than one part in $10^7$.

That gradient test was not decoration. It caught two real interval bugs in the
tangent-linear propagator while this chapter was being written — a zero-length window
being advanced by one time step, and a propagator covering `n_steps*dt` instead of the
requested $\tau$ — either of which left every 4D-Var gradient about 7% wrong. The
symptom was a chapter whose 4D-Var underperformed for no visible reason; nothing in
the figures looked amiss.

{{< marimo src="/nb/ch20_da-in-practice.html" >}}

## Exercises

**Analytic.** Show that the 3D-Var minimiser equals the Kalman analysis
$\mathbf{x}^a = \mathbf{x}^b + \mathbf{K}(\mathbf{y}-\mathbf{H}\mathbf{x}^b)$ for
linear $\mathbf{H}$. Then explain why operational centres minimise $\mathcal{J}$
iteratively anyway, given that the closed form exists.

**Computational.** Derive $\Delta t = \ln 10/\lambda$ from
$t_c = \lambda^{-1}\ln(\delta_c/\delta_0)$, then check the measured slope in section 8
against $1/\lambda_1$ computed independently in
[chapter 7]({{< relref "../part3/ch07_lyapunov-exponents.md" >}}). They agree to a few
percent by two completely different routes — one from the dynamics, one from forecast
error curves.

**Exploratory.** Set the EnKF to $N = 10$ with inflation $1.0$ and hunt for filter
divergence: the point where the ensemble becomes so narrow that the analysis stops
responding to observations at all. Then find the inflation that recovers it. Is the
value that minimises RMSE the same as the value that best calibrates the spread? It is
not, and that tension is a real operational problem rather than an artefact of this
toy.

## A note on what this chapter shows, and what it does not

The three-way RMSE comparison should not be read as a general ranking. This is a
three-variable, perfect-model problem in which every component is observed at every
analysis time — about the friendliest possible setting for an ensemble method, and one
where localisation, the EnKF's central practical difficulty, cannot bite. The
*mechanisms* transfer; the ordering does not.

The same caution applies to the headline number. The logarithmic law is general, but
its constant is $1/\lambda$, and Lorenz 63's $\lambda$ under the conventional
1 MTU $\approx$ 5 days reading makes it *less* chaotic per day than the atmosphere it
stands in for. The law transfers; the constant does not — which is the argument of
[chapter 3]({{< relref "../part1/ch03_model-hierarchy.md" >}}).

## Further reading

- Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability.*
  Cambridge University Press — ch. 5 for the algorithms, §6.1 for predictability.
- Evensen, G. (2009). *Data Assimilation: The Ensemble Kalman Filter.* Springer.
- Bocquet, M. et al. (2023). *A guide to ensemble Kalman methods with implementation in
  Python.* arXiv:2305.00087.
- Bauer, P., Thorpe, A. and Brunet, G. (2015). The quiet revolution of numerical
  weather prediction. *Nature*, **525**, 47–55 — the historical record of the
  "day per decade" improvement.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
  ch. 8 *[citation needed: pages]*.
