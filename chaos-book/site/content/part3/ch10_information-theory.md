---
title: "Chapter 10 · Information theory and predictability"
weight: 310
part: "Part III — Quantifying chaos and predictability"
knob: 'system and observable, the rescaling factor, bin count'
status: "live"
---

## Overview

Every measure of predictability so far has been an error: a distance between a forecast
and the truth, in some norm.
[Chapter 16]({{< relref "../part5/ch16_adjoint-sensitivity.md" >}}) showed that the
choice of norm is not innocent — it rotates the "fastest-growing" perturbation by tens of
degrees. [Chapter 9]({{< relref "ch09_nonlinear-error-growth.md" >}}) showed an error's
growth rate depends on its amplitude. Both are properties of the measuring instrument as
much as of the atmosphere.

There is a way of asking that has neither problem. A forecast is worth something only if
it tells you more than climatology, and "how much more" has a canonical answer:

$$D\bigl(p_{\rm forecast}\,\|\,p_{\rm climatology}\bigr)
  = \int p_f \ln\frac{p_f}{p_c}\,dx ,$$

the **relative entropy**, in nats, zero exactly when the forecast adds nothing — and
invariant under any invertible change of variables.

## The model

For Gaussians the closed form splits in two: a **signal** term
$\tfrac12(\mu_c-\mu_f)^{\!\top}\Sigma_c^{-1}(\mu_c-\mu_f)$ — "my mean differs from
climatology" — and a **dispersion** term — "my forecast is sharper than climatology".
`chaoslib.information.gaussian_information_components` returns both, and the tests assert
they sum to the total exactly and that each is non-negative.

The measurements are 80 ensemble forecasts of 500 members, on Lorenz 63 and Lorenz 96.

{{< marimo src="/nb/ch10_information-theory.html" >}}

## Four results

**Forecast information is almost entirely dispersion.** At lead zero a $10^{-3}$-spread
ensemble carries about 9 nats, of which **94%** is dispersion; the signal term sits near
0.5 nats throughout. That is not a defect — a forecast started at a random point on a
stationary attractor has no reason for its mean to sit far from the climatological mean in
units of the climatological spread. The signal term earns its keep in *forced* problems,
which is where [chapter 17]({{< relref "../part5/ch17_probabilistic-forecast-design.md" >}})
spends it.

**The information decays linearly, at $\lambda_1$ — not at $h_{KS}$.** Four measurements
across two systems and two observables land within 4% of the leading Lyapunov exponent.
The decisive case is Lorenz 96, where $\lambda_1 = 1.67$ and $h_{KS} = 10.21$ differ
six-fold: measured $-dD/dt = 1.598$, ruling out the entropy rate by a factor of 6.4.

That deserves care, because $h_{KS}$ *is* the rate at which the system destroys
information — [chapter 8]({{< relref "ch08_attractor-dimension.md" >}}) established it and
Pesin's identity is not in doubt. The resolution: $h_{KS}$ is the rate for the **full
state**, and one projection of a 40-dimensional forecast reveals the fastest direction and
nothing about the other twelve growing ones. So there are two honest numbers answering
different questions — $h_{KS}$ is how fast the system destroys information about itself,
and $\lambda_1$ is how fast a forecast of any *one* variable stops being informative,
which is what a user experiences.

**Doing it on the full state fails, quietly.** A 500-member ensemble on a
2.06-dimensional attractor has a near-singular covariance in three dimensions, so
$\ln\det\Sigma_f$ is set by the regularisation rather than the dynamics. Two defensible
floors, $10^{-12}$ and $10^{-6}$, give decay rates of **1.677 and 0.940** — a factor of
1.78 — with nothing in either computation to indicate which is right. This is the same
curse of dimensionality chapter 8 met from the other side, arriving at three dimensions
rather than forty.

**And the invariance is exact.** Under four transformations — stretching one axis,
rescaling all of them, a random invertible map — the relative entropy is identical to
within $2\times10^{-15}$ nats at every rescaling offered, while the RMS error of the mean
varies by a factor of 3.7 with no rescaling at all and by **371** at the extreme. An
error norm measures the atmosphere *and* your units; an information measure measures the
atmosphere. That is the counterpart of chapter 16's finding on the other side of the
ledger.

## Mutual information, and whether it can be estimated

$I(x(0); x(t))$ needs no ensemble and no Gaussian assumption, and pays for it in
estimator bias. The plug-in estimate is biased upward by roughly the occupied-bin count
over twice the sample size, so it has a floor: at a lag of 12 MTU, where the true value is
zero, it reads 0.009 to 0.039 nats depending only on the bin count. Miller–Madow
(`correction="miller_madow"`) cuts that by two to four times and does not remove it.
Below the floor you are measuring the estimator, and the practical rule is to measure the
floor at a lag where the answer is known and discard anything beneath it.

## Exercises

**Analytic.** Show that $D$ and both its components are invariant under
$x \mapsto Ax$ for invertible $A$, and that the dispersion term vanishes exactly when
$\Sigma_f = \Sigma_c$.

**Computational.** Reproduce the $\lambda_1$-versus-$h_{KS}$ discrimination on Lorenz 96,
and state what the measured rate would have been had a scalar observable seen the full
entropy rate.

**Exploratory.** Using chapter 8's $D_2 = 2.06$, estimate how many ensemble members would
be needed for the smallest eigenvalue of $\Sigma_f$ to be genuinely resolved in three
dimensions, and decide whether that is affordable.

## Further reading

- Kleeman, R. (2002). Measuring dynamical prediction utility using relative entropy.
  *JAS*, **59**, 2057–2072 *[citation needed: confirm pages]*.
- DelSole, T. (2004). Predictability and information theory. *JAS*, **61**, 2425–2440
  *[citation needed: confirm]* — the signal/dispersion decomposition.
- Schneider, T. and Griffies, S. M. (1999). A conceptual framework for predictability
  studies. *Journal of Climate*, **12**, 3133–3155 *[citation needed: confirm]*.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
  ch. 14 *[citation needed: pages]*.
