---
title: "Chapter 22 · Forecast verification and the practical horizon"
weight: 522
part: "Part V — The machinery of prediction"
knob: 'the threshold defining ''useful'''
status: "live"
---

## Overview

"Weather forecasts are useful about a week ahead." Every chapter of this book has
circled that sentence. This one asks what would have to be true for it to *mean*
anything — and the difficulties turn out not to be about the atmosphere.

**Useful by what measure?** Anomaly correlation, mean-square error, CRPS and a Brier
score are different questions, and the same forecasts answer them differently.

**Useful past what threshold?** The conventional answer is anomaly correlation 0.6, and
that number is arithmetic rather than meteorology.

**Verified against what?** Every score in this book so far compared a forecast against a
*truth*. No forecast centre has ever had one. Chapters 13, 17, 18, 19 and 21 each
deferred that problem to here.

## The model

367 forecasts launched from a cycling LETKF analysis of Lorenz 96 — [chapter
19]({{< relref "ch19_ensemble-da.md" >}})'s configuration — with the archive keeping
**truth, analysis and observations as three separate objects**, so the chapter can score
against each in turn. Climatological $\sigma = 3.64$, $\sigma_o = 1$.

{{< marimo src="/nb/ch22_verification.html" >}}

## Where 0.6 comes from

Take a forecast that is unbiased and whose anomaly variance matches the truth's — an
undamped forecast, which is what a raw model run gives you. Then
$\mathrm{MSE} = 2\sigma^2(1-r)$ against $\sigma^2$ for a climatological forecast, so its
skill score is exactly $2r-1$: it **ties with climatology at $r = 1/2$**, and 0.6 is that
number with a margin.

Now damp it. The least-squares rescaling gives $\mathrm{MSE} = \sigma_t^2(1-r^2)$, which
beats climatology at **any** non-zero correlation. The threshold for usefulness is not a
property of the atmosphere; it is a property of a decision not to post-process. Both
identities are asserted as tests.

## Five results

**The same forecasts give horizons a factor of 2.5 apart.** ACC < 0.6 gives 1.94 TU;
RMSE > 0.7σ gives 1.54; Brier skill < 0 gives 3.79. None is wrong — they answer different
questions. Two entries agreeing is not a coincidence: ACC < 0.5 and MSE skill < 0 both
give **2.20 TU**, because the identity above says they *are* the same condition. Watching
an algebraic identity survive a cycling assimilation system is a check on both.

**Post-processing moves the horizon further than the threshold does.** The undamped
mean-square error reaches the climatological value at lead 2.22 TU; optimally damped it
does not reach it at all within the 5 TU measured. Same forecasts, same truth, horizon
more than doubled by whether anyone rescaled the output.

**Independent observation error is exactly predictable and exactly removable.** The
measured inflation sits on $\sigma_o^2$ at every lead — from a forecast error of 0.06 to
one of 25, a factor of 400 in the quantity being measured, with the same additive offset
throughout — and the corrected anomaly correlation lies on the truth-based curve.

**Verifying against your own analysis flatters the forecast, and it is a short-lead
problem.** As a fraction of the true mean-square error the flattery is **−64 %** at lead
0.1, −27 % at 0.3, −12 % at 0.5, and under 2 % by lead 1. Beyond a day or so it hardly
matters *in this configuration*, which has an unusually good analysis; with an analysis
error closer to the forecast error it would persist much further.

**Each fault lights up its own term.** Murphy's decomposition, an identity, applied to a
perfect twin, a wrong-forcing model and a constant offset: the wrong forcing raises the
**amplitude** term 119-fold while phase grows 6 %; the offset raises **bias** 63-fold
while amplitude and phase do not move. Phase still dominates the total everywhere — *the
term that grew is the one that identifies the fault, not the term that is largest*.

## The correction that returns an impossible number

Verify the **analysis** against the observations that were assimilated to make it. The
measured MSE is 0.9249; if the errors were independent it would be
$0.0555 + 1 = 1.0555$. Applying the standard correction gives

$$0.9249 - 1.0 = -0.0751,$$

a **negative mean-square error**. The assimilation pulled the analysis towards those very
observations, so the cross term does not vanish and the score was optimistic rather than
pessimistic.

`chaoslib` returns that negative number rather than clipping it to zero, and that is the
useful behaviour: an impossible answer announces its own failure, where a clipped zero
would have looked like a superb forecast. The operational remedy is to verify against
observations withheld from the assimilation — which costs exactly the observations you
would most have liked to use.

## The knob, and chapter 12's promise

The threshold is the chapter's knob: from 0.90 to 0.40 the horizon runs 0.86 to 2.53 TU,
and the curve is steepest exactly where the convention sits.

Then the closing question. Reduce the initial error across **seven decades** and measure
the horizon in two systems.

| system | horizon at $\delta_0 = 10^{-1}$ | at $10^{-8}$ | gain per decade |
|---|---|---|---|
| single-scale L96 | 3.59 TU | 13.02 TU | **1.35 TU** |
| two-scale L96 (slow variables) | 2.40 TU | 3.51 TU | **0.16 TU** |

The single-scale system keeps paying at exactly the predicted
$\ln 10/\lambda_1 = 1.38$ TU per decade — the plotted line is that law, not a fit, and
[chapter 20]({{< relref "ch20_da-in-practice.md" >}}) measured the same thing by an
entirely different route. The two-scale system pays **8 times less**: its fast variables'
errors saturate almost at once whatever $\delta_0$ was, and contaminate the slow
variables from below.

**An honest qualification.** The two-scale curve is not flat — it rises from 2.40 to 3.51
TU and is still creeping upward at $\delta_0 = 10^{-8}$. A *bounded* horizon is the limit
of this behaviour and this experiment does not demonstrate it outright.
[Chapter 12]({{< relref "ch12_scale-dependent-error-growth.md" >}}) put it as "the two
limits are different quantities, and only one of them moves". The measurement supports
the first half exactly and softens the second: both move, but at rates differing by
almost an order of magnitude, and the gap widens as $\delta_0$ falls.
[Chapter 14]({{< relref "ch14_chaos-to-turbulence.md" >}}) showed why the exponent
$\alpha = (3-p)/2$ decides whether the limit is finite at all, and a two-scale system
with 32 fast variables per slow one is a caricature of a spectrum rather than a spectrum.

## Exercises

1. Set the threshold to 0.5 and confirm the horizon matches the MSE-skill horizon. Why
   must these agree exactly?
2. The RMSE tends to $\sqrt2\,\sigma$, not $\sigma$. Derive that, and say what it implies
   about a threshold placed at $\mathrm{RMSE} = \sigma$.
3. The analysis flattery is 64 % at lead 0.1 and 0.3 % at lead 2. What property of the
   assimilation sets that decay rate?
4. Work out the sign of the cross term that produces a negative corrected MSE, and
   construct the observing system for which the correction would be exactly right.
5. The two-scale curve is still rising at $\delta_0 = 10^{-8}$. Design the experiment
   that would settle whether it converges, and say why it is expensive.

## Further reading

- Jolliffe & Stephenson, *Forecast Verification* *[citation needed: edition]*
- Murphy (1988), on the decomposition of the mean-square error *[citation needed]*
- Murphy & Epstein (1989), on skill scores and their reference forecasts
  *[citation needed]*
- Bauer, Thorpe & Brunet (2015), "The quiet revolution of numerical weather prediction",
  for the historical skill record *[citation needed: figure]*
- Palmer & Hagedorn (2006), *Predictability of Weather and Climate*
  *[citation needed: chapter]*
