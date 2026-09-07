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
condition for a model with $10^8$ degrees of freedom. Data assimilation is how.

[Chapter 18]({{< relref "ch18_variational-da.md" >}}) and
[chapter 19]({{< relref "ch19_ensemble-da.md" >}}) derive the machinery. This chapter
asks what happens when you **run it for ever**: three schemes cycled on the same
problem, a check on whether the ensemble they produce is honest, and the effect of
observing less often.

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
same observations, held identical across all three along with $\mathbf{B}$,
$\mathbf{R}$ and the observation times, so that the comparison measures the scheme and
not the setup.

The schemes themselves are **not derived here**. 3D-Var and 4D-Var — the cost function,
the adjoint gradient and how to know it is right, the Hessian as the analysis-error
covariance, window length, the incremental form — are
[chapter 18]({{< relref "ch18_variational-da.md" >}}). The EnKF — sampling error and
why localisation is compulsory, inflation, deterministic versus stochastic updates,
hybrids — is [chapter 19]({{< relref "ch19_ensemble-da.md" >}}). What is left for this
chapter is the part those two set aside: a single analysis is a problem in linear
algebra, while running one every few time units for ever against a system whose errors
double on the same timescale is a different problem, and the one an operational centre
actually has.

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

**Analytic.** Cycling reaches a steady state: the analysis error at one cycle sets the
background error at the next, which the next analysis partly removes. Write that as a
one-dimensional map — background error grows by $e^{\lambda \Delta t_{\rm obs}}$ over
the interval, then the analysis contracts it by the Kalman gain — and find its fixed
point. Then show why the fixed point exists only for $\Delta t_{\rm obs}$ below a
threshold, and relate that threshold to the divergence seen in section 5.

**Computational.** Derive $\Delta t = \ln 10/\lambda$ from
$t_c = \lambda^{-1}\ln(\delta_c/\delta_0)$, then check the measured slope in section 6
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
