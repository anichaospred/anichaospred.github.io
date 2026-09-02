---
title: "Chapter 9 · Error growth beyond the linear regime"
weight: 309
part: "Part III — Quantifying chaos and predictability"
knob: '$\delta_0$, the fit space, the useful-forecast threshold'
status: "live"
---

## Overview

[Chapter 7]({{< relref "ch07_lyapunov-exponents.md" >}}) measured $\lambda_1$ and turned
it into a doubling time; [chapter 8]({{< relref "ch08_attractor-dimension.md" >}}) turned
it into an exchange rate of 3.8 days per decimal digit of initial precision. Both assume
the error stays small enough for the tangent linear approximation to hold.

A real forecast error does not. It grows exponentially for a while and then stops,
because a forecast cannot be more wrong than a randomly chosen state of the attractor.
So the exponential picture describes the *beginning* of a forecast's life, and the part
that decides when to stop using it is the part the exponential picture does not describe.

Lorenz's logistic model adds one term:

$$\frac{dE}{dt} = \lambda E\Bigl(1 - \frac{E}{E_\infty}\Bigr)
  \quad\Longrightarrow\quad
  E(t) = \frac{E_\infty}{1 + (E_\infty/E_0 - 1)e^{-\lambda t}}.$$

This chapter asks whether it works, and finds **three separate traps** — one in the
statistics, one in the fitting, one in the model. All three are measured, and all three
were found the hard way while writing the chapter.

## The model

The measurements are twin experiments: perturb by $\delta_0$, integrate both copies, and
average the error over 1024 starting points on the Lorenz 63 attractor (256 for
Lorenz 96), at five amplitudes four decades apart. `chaoslib.errorgrowth` supplies the
model, the fit, and the saturation level.

Integrating the ensemble is knob-free and expensive; fitting a model to the resulting
curve is microseconds. So the curves are precomputed and the notebook's controls change
the **model**, not the data — the reader is not allowed to move the measurement.

{{< marimo src="/nb/ch09_nonlinear-error-growth.html" >}}

## What the curve looks like

Five curves, each a hundred times smaller at $t = 0$ than the one above, run parallel
while the error is small — a factor in $\delta_0$ is a shift in time, which is what
exponential growth means — then bend over and converge. Within about ten time units they
are indistinguishable: nothing about the initial error survives.

The logarithmic exchange rate does carry into the nonlinear regime. Each extra digit of
initial accuracy buys close to $\ln 10/\lambda_1$ of lead time even though most of the
curve is not exponential, which is not obvious and is worth knowing — it is what lets
[chapter 20]({{< relref "../part5/ch20_da-in-practice.md" >}})'s operational measurement
agree with a calculation done in the infinitesimal limit.

One feature that looks like a bug and is not: **the mean error curve is non-monotonic**.
It falls back for stretches on the approach to saturation, and that survives 1024
members, so it belongs to the attractor. In Lorenz 63 it is the two lobes — a pair of
trajectories on opposite wings are far apart, and the distance shrinks again when they
land on the same wing.

## Three traps

**Fitting in the wrong space.** `curve_fit` minimises squared residuals in $E$, and the
curve spans ten orders of magnitude in $E$ — so the handful of points near saturation
outweigh the entire exponential phase, and $\lambda$ comes out of a fit that never looked
at the phase that defines it. On the shipped Lorenz 63 curve, against an early-time rate
of 0.921: log-space fitting returns $\lambda = 0.919$ and $E_0 = 3.6\times10^{-6}$;
linear-space returns **0.748** and $5.3\times10^{-5}$ — 19% out in the rate and fiftyfold
in the amplitude, while producing a curve that looks entirely convincing on a linear
plot. The log fit is also stable across seeds (0.920, 0.919, 0.921) where the linear one
is not (0.716, 0.748, 0.722). `fit_logistic_error_growth` now fits in log space by
default.

**Mixing two saturation statistics.** An ensemble-*mean* error curve must be compared
against the *mean* pair distance, not the RMS one. By Jensen the mean is smaller, and how
much smaller depends on the attractor: **0.889** for Lorenz 63, whose two lobes broaden
the distribution of pair distances, against **0.995** for Lorenz 96, where distances
concentrate in forty dimensions. Compare a mean curve against an RMS saturation and it
appears to stop growing at 89% of saturation — which then shows up as a spurious
12-percentage-point error in the model's *shape*. On the first pass through this chapter
Lorenz 63 came out 19% from logistic; matching the statistics moved it to 2.7%, and the
12 points were mine. `saturation_level` now takes a `statistic` argument.

There is an exact identity worth checking any implementation against: for independent
draws, $\langle\|x-y\|^2\rangle = 2\langle\|x-\bar x\|^2\rangle$, so the RMS saturation
is exactly $\sqrt2$ times the RMS spread about the mean — verified to $5\times10^{-4}$.

**The model's form.** Rearranged, the model says
$d\ln E/dt = \lambda(1 - E/E_\infty)$: a straight line in $E$ with slope-to-intercept
ratio exactly $-1$. That is a fitting-free test, and it gives $-1.027$ for Lorenz 63 — a
**2.7%** deviation, a good result for a one-parameter closure and why the model has
lasted — but $-1.167$ for Lorenz 96, off by **17%**. The reason is in
[chapter 11]({{< relref "../part4/ch11_lorenz96.md" >}}): Lorenz 96 has thirteen positive
exponents spanning 0.03 to 1.67, so its error is a superposition of components growing at
different rates and saturating at different times, and one $\lambda$ cannot represent
that. [Chapter 12]({{< relref "../part4/ch12_scale-dependent-error-growth.md" >}}) makes
the same point from the other end.

Note also that the fitted intercept is **not** $\lambda_1$: 0.797 against 0.906 for
Lorenz 63, and 1.236 against 1.67 for Lorenz 96. Fitted over the nonlinear range, the
logistic $\lambda$ that best describes it is smaller than the infinitesimal growth rate.
They are different quantities, and which one a paper means is worth checking.

## The doubling time is a function of amplitude

Since $d\ln E/dt$ falls as $E$ grows, the doubling time is a property of the system *and
the amplitude you measured it at*. For Lorenz 63 it runs from $\ln2/\lambda_1 = 0.766$
MTU at infinitesimal amplitude to roughly four times that by 70% of saturation.

The operational consequence runs in a direction worth care. A doubling time measured from
real forecasts is measured from errors that are *already large*, so it is longer than
$\ln2/\lambda_1$, and extrapolating it backwards **underestimates** how fast small errors
grow. Using $\lambda_1$ to extrapolate forwards overestimates growth at large error and
so **underestimates** the useful horizon. Both mistakes are available, they point
opposite ways, and neither is detectable from a single number — which is why
[chapter 13]({{< relref "../part4/ch13_operational-error-growth.md" >}}) cannot simply
quote a $\lambda$.

## Exercises

**Analytic.** Show that the time for the logistic model to reach a fraction $f$ of
saturation exceeds the pure-exponential estimate by $\lambda^{-1}\ln[1/(1-f)]$, and
evaluate it at $f = 0.5$ and $f = 0.9$.

**Computational.** Fit the same curve in linear and log space and reproduce the two
$\lambda$ values. Then explain the difference from the fact that least squares on $E$
weights each point by $E$.

**Exploratory.** Before looking, decide which of Lorenz 63 and Lorenz 96 should be worse
described by a one-$\lambda$ model, using their Lyapunov spectra from chapters 7 and 11.
Then check against the slope-to-intercept test.

## Further reading

- Lorenz, E. N. (1982). Atmospheric predictability experiments with a large numerical
  model. *Tellus*, **34**, 505–513.
- Lorenz, E. N. (1969). The predictability of a flow which possesses many scales of
  motion. *Tellus*, **21**, 289–307.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
  ch. 3 *[citation needed: pages]*.
