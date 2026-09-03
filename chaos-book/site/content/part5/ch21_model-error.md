---
title: "Chapter 21 · Model error and the imperfect-model problem"
weight: 521
part: "Part V — The machinery of prediction"
knob: 'model bias, initial-condition accuracy, error source'
status: "live"
---

## Overview

Every chapter so far has assumed a perfect model and imperfect initial conditions.
[Chapter 8]({{< relref "../part3/ch08_attractor-dimension.md" >}}) priced better
observations at $\ln 10/\lambda_1$ = 3.8 days per decimal digit;
[chapter 13]({{< relref "../part4/ch13_operational-error-growth.md" >}}) measured 7.0 days
per factor of thirteen on a synthetic operational archive. Both assumed that if you knew
the state well enough, the forecast would be good.

No forecast is made with the right model. So: does the return on better observations
continue, or does it stop?

It stops, and this chapter measures where.

## Three ways to be wrong, three growth laws

An initial-condition error is injected **once**; a model error is injected
**continuously**. From $\dot E = \lambda E + \text{source}$:

| source | short lead | long lead |
|---|---|---|
| initial condition, $\delta_0$ | $\delta_0 e^{\lambda t}$ | $\delta_0 e^{\lambda t}$ |
| deterministic bias $b$ | $b\,t$ — **linear** | $(b/\lambda)e^{\lambda t}$ |
| stochastic forcing $\sigma$ | $\sigma\sqrt t$ — **diffusive** | $(\sigma/\sqrt{2\lambda})e^{\lambda t}$ |

Truth is Lorenz 96 at $F = 8$; model error is a wrong $F$, constant or stochastic.
`chaoslib.integrate.rk4_stochastic` supplies the integrator — RK4 on the drift,
Euler–Maruyama on the noise, reducing to `rk4` bit-for-bit at zero noise so both arms of
a perfect/imperfect comparison share one discretisation.

{{< marimo src="/nb/ch21_model-error.html" >}}

## Three results

**The deterministic-bias law comes out exactly right**: measured
$d\ln E/d\ln t = 1.087$ against a predicted 1, and identical to three decimals across
biases spanning a factor of twenty, which is what a real power law looks like. The other
two measured slopes are contaminated by the exponential growth always superposed — for
pure exponential growth $d\ln E/d\ln t = \lambda t$, contributing about 0.3 over this
window — so the IC case reads 0.485 where a clean exponential gives ~0.3, and the
stochastic case 0.759 where $\tfrac12 + \lambda t$ gives ~0.8. The bias case separates
cleanly because its power law is steepest and dominates longest.

**The two sources are not comparable in size.** A 0.6% error in the forcing produces of
order a hundred times more error at one time unit than a $10^{-4}$ initial perturbation.
For any plausible pair of amplitudes, one simply dominates.

**And the model's error is a ceiling.** Lead time to 30% of saturation, in days:

| $\delta_0$ | perfect model | bias 0.01 | bias 0.05 | bias 0.2 |
|---|---|---|---|---|
| $10^{-2}$ | 13.6 | 13.6 | 13.2 | 10.2 |
| $10^{-3}$ | 19.4 | 18.7 | 15.2 | 10.3 |
| $10^{-4}$ | 25.9 | 19.6 | 15.3 | 10.3 |
| $10^{-6}$ | off the chart | 19.6 | 15.2 | 10.3 |
| $10^{-8}$ | off the chart | 19.6 | 15.2 | 10.3 |

With a perfect model the return continues. With a bias of 0.01 — a **0.125% error in
$F$** — improving the initial state from $10^{-4}$ to $10^{-8}$, four orders of
magnitude, buys $-0.005$ days. The lead is pinned at 19.6 days, exactly what that bias
gives with a *perfect* initial condition.

The return doesn't merely stop, it is eaten well before it stops: the same two-decade
improvement from $10^{-2}$ to $10^{-4}$ is worth 12.3 days with a perfect model, 6.0 at
bias 0.01, 2.1 at 0.05, and 0.1 at 0.2.

## What this does to the logarithmic law

The $\ln 10/\lambda_1$ exchange rate of chapters 8, 13 and 20 is real, and this chapter
does not contradict it. What it adds is a **stopping condition**: the logarithmic return
holds while initial-condition error dominates model error, and past the crossover the
exchange rate goes to zero over about one decade of $\delta_0$. So "how much is a better
observing system worth?" has no answer that does not mention the model.

## What cannot fix it

**Data assimilation cannot**, and quietly makes it worse: the background is a model
forecast and $\mathbf{B}$ describes uncertainty *given* a correct model, so a cycling
analysis inherits the bias, the innovations look larger than the specified observation
error, and the usual response — tuning $\mathbf{R}$ up — is exactly wrong.

**Chapter 13's estimator cannot even diagnose it.** It differences two forecasts of the
same model, so a common bias cancels algebraically — `chaoslib` has a test that biases an
entire archive and asserts the output is unchanged to round-off. The standard truth-free
method is blind by construction to precisely this error source.

**Stochastic parameterisation is the standard partial response.** What it demonstrably
buys is ensemble *reliability* — spread that matches error, which is what
[chapter 17]({{< relref "ch17_probabilistic-forecast-design.md" >}})'s scores reward. What
it does not reliably buy is a smaller mean error, and claims that it does should be
checked against the possibility that the noise is detuning a bias.

Nothing here estimates the real atmosphere's model error, because that needs the truth.
The consequence is that the predictability numbers throughout Parts III and IV, all from
perfect-model experiments, are **upper bounds**.

## Exercises

**Analytic.** From $\dot E = \lambda E + b$, derive $E = (b/\lambda)(e^{\lambda t}-1)$ and
show it is $b\,t$ for $\lambda t \ll 1$. Then do the stochastic case and obtain the
$\sqrt t$ law.

**Computational.** For each bias, find the $\delta_0$ at which lead time stops improving
and compare it against $b/\lambda_1$.

**Exploratory.** A satellite reduces analysis error threefold. State what it buys with a
perfect model, with a bias of 0.01, and with a bias of 0.2.

## Further reading

- Palmer, T. N. (2001). A nonlinear dynamical perspective on model error. *QJRMS*,
  **127**, 279–304 *[citation needed: confirm pages]*.
- Wilks, D. S. (2005). Effects of stochastic parametrizations in the Lorenz '96 system.
  *QJRMS*, **131**, 389–407.
- Orrell, D., Smith, L., Barkmeijer, J. and Palmer, T. N. (2001). Model error in weather
  forecasting. *Nonlinear Processes in Geophysics*, **8**, 357–371
  *[citation needed: confirm]*.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
  ch. 10–11 *[citation needed: pages]*.
