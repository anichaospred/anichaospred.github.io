---
title: "Chapter 12 · Scale-dependent error growth and the intrinsic limit"
weight: 412
part: "Part IV — Many scales, many degrees of freedom"
knob: '$\alpha$, the number of resolved octaves, where the error is seeded'
status: "live"
---

## Overview

Every predictability estimate so far in this book has had the same shape. An error of
size $\delta_0$ grows at rate $\lambda_1$, so the useful forecast lasts
$\lambda_1^{-1}\ln(\delta_{\rm sat}/\delta_0)$ — and as $\delta_0 \to 0$ that grows
without bound. [Chapter 8]({{< relref "../part3/ch08_attractor-dimension.md" >}}) put a
number on the exchange rate: $\ln 10/\lambda_1 = 3.8$ days per decimal digit of initial
precision, and the next digit buys 3.8 days again. Slow, but unlimited.

Lorenz (1969) argued that a system with a *spectrum* of scales behaves differently, and
the conclusion is far more troubling than the butterfly effect. If small scales grow
faster than large ones, then error introduced at the smallest scale reaches the largest
in a **finite** time, no matter how small the scale you start from. Improving the
resolution of an observing system then stops buying lead time — not because the
improvement is slow, but because the limit is bounded.

This chapter separates the two cases, and the whole thing turns on one exponent.

## The model

Discretise into octave bands: band $n$ has scale $L\,2^{-n}$ and growth rate

$$\lambda_n = \lambda_0\,2^{\alpha n},$$

so $\alpha$ measures how much faster small scales grow. Kolmogorov scaling gives the
eddy turnover time $\tau \sim \varepsilon^{-1/3}\ell^{2/3}$, hence
$\alpha = 2/3$. Lorenz 63 and Lorenz 96 have $\alpha = 0$: one growth rate,
scale-independent.

Each band's error is measured against its own saturation level and grows logistically,
forced by the band one octave smaller:

$$\frac{de_n}{dt} = \lambda_n\bigl(e_n + \kappa e_{n+1}\bigr)\bigl(1 - e_n\bigr).$$

With a single band the second term vanishes and this is exactly the logistic model of
[chapter 9]({{< relref "../part3/ch09_nonlinear-error-growth.md" >}}) — a reduction the tests
check to $10^{-8}$.

The critical point is what "improving the observations" means here. An observing system
has a *resolution*, and about scales finer than it we know nothing at all, so the error
there starts at **saturation**. Adding a band therefore represents seeing one octave
further down, not reducing an amplitude — and those two kinds of improvement turn out to
behave completely differently.

{{< marimo src="/nb/ch12_scale-dependent-error-growth.html" >}}

## Four results

**The limit is finite if and only if $\alpha > 0$.** Seeding the finest resolved band at
saturation and adding octaves, the time for the largest scale to be contaminated
converges to 2.3035 at $\alpha = 1/3$, **1.4466** at $\alpha = 2/3$ and 1.1220 at
$\alpha = 1$ — while at $\alpha = 0$ it grows without bound, at a settled 0.281 per
octave out to 128 bands. So a finite predictability horizon follows from small scales
growing faster than large ones, and from nothing else.

**The increments die as $2^{-2\alpha}$, not $2^{-\alpha}$.** Measured per-octave
increment ratios are 0.630, 0.397 and 0.250 against $2^{-2/3}, 2^{-4/3}, 2^{-2}$ =
0.630, 0.397, 0.250 — exact to three decimals. That is the *square* of what the naive
"sum the band timescales" argument predicts, which is also why that argument gets the
convergence right and the constant wrong: it gives 2.70 at $\alpha = 2/3$ against the
measured 1.4466, because the bands overlap in time rather than waiting for one another.

**Where you improve matters far more than by how much.** At $\alpha = 2/3$, reducing the
initial error at the *finest* band by sixteen orders of magnitude moves the horizon from
1.4453 to 1.4795 — **2%**. Reducing it at the *coarsest* band instead obeys the familiar
law, $\ln 10/\lambda_0$ per decade, and keeps paying indefinitely. Small-scale accuracy
is nearly worthless because small-scale error saturates almost immediately; large-scale
accuracy is what buys time.

**Two scales are not a cascade, and the two-scale Lorenz 96 shows how completely they
are not.** Coupling 8 slow variables to 256 fast ones ten times quicker gives a genuine
dynamical upscale cascade: perturb only the fast variables and the slow error appears
shortly after, growing at 2.96–2.99 per time unit — a rate set by the slow dynamics and
**independent of how small the fast perturbation was**, across eight decades of it. That
is the mechanism, cleanly.

But the finite limit is absent. Averaged over 32 base states, the return on initial
accuracy is **0.145** time units per decade when the error is seeded in the fast
variables against **0.148** when seeded in the slow ones — a 2% difference, so it makes
essentially no difference where the error is put. The limit needs a spectrum, not two
rungs of one, and this is worth stating plainly because two-scale Lorenz 96 is widely
used and easily over-read.

The model does deliver one sharp warning of its own. The leading Lyapunov exponent of the
coupled system is set by the *fast* subsystem: $\lambda_1 = 24.7$ per time unit, a
doubling time of 0.028 time units, or 0.14 days at the conventional five days per unit.
Read as "the" error-doubling time that would put weather predictability at a few hours.
The slow variables — the ones a forecast is about — actually double every 0.232 time
units, 1.16 days. So $\lambda_1$ overstates large-scale error growth by a factor of
**8.3**, and in a multiscale system it and the predictability of the large scales are
different questions. That is the correction
[chapter 7]({{< relref "../part3/ch07_lyapunov-exponents.md" >}})'s single number needs.

## The honest caveat

Whether the atmosphere actually has $\alpha > 0$ across the relevant range of scales is
an empirical question, not a mathematical one, and it is the question the whole "two-week
limit" rests on. The cascade model shows that *if* the spectrum of growth rates behaves
as Kolmogorov scaling suggests, the limit is finite; it cannot show that it does.
Measurements of atmospheric error growth as a function of scale, and the resulting
estimates of the intrinsic limit, remain actively argued over *[citation needed: on
observational estimates of scale-dependent error growth]*.
[Chapter 13]({{< relref "ch13_operational-error-growth.md" >}}) takes up what operational
forecast errors actually do, and
[chapter 22]({{< relref "../part5/ch22_verification.md" >}}) the difference between the
horizon we have and the horizon we could have.

## Exercises

**Analytic.** Show that $\sum_n \lambda_n^{-1}$ converges for $\alpha > 0$ and diverges
for $\alpha = 0$, and evaluate the sum. Then explain why the measured contamination time
at $\alpha = 2/3$ is 1.4466 rather than the 2.70 that sum predicts.

**Computational.** Reproduce the $2^{-2\alpha}$ scaling of the per-octave increment for
three values of $\alpha$, and state what it implies about the number of octaves worth
resolving.

**Exploratory.** Set $\alpha = 0$ and add octaves until the contamination time doubles.
Now do the same at $\alpha = 2/3$. Explain what an observing-system designer should
conclude from the difference.

## Further reading

- Lorenz, E. N. (1969). The predictability of a flow which possesses many scales of
  motion. *Tellus*, **21**, 289–307.
- Lorenz, E. N. (1996). Predictability: a problem partly solved. *Proceedings of the
  ECMWF Seminar on Predictability*, vol. 1, 1–18.
- Wilks, D. S. (2005). Effects of stochastic parametrizations in the Lorenz '96 system.
  *Quarterly Journal of the Royal Meteorological Society*, **131**, 389–407 — the
  two-scale parameters used here *[citation needed: confirm]*.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
  ch. 3 *[citation needed: pages]*.
