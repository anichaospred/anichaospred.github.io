---
title: "Chapter 1 · What is predictability?"
weight: 101
part: "Part I — What predictability means"
knob: 'lead time'
status: "live"
---

## Overview

Two forecasts can both be wrong and one of them still be excellent. So before any
mathematics: what exactly is being claimed when someone says a forecast is good, or that
the atmosphere is predictable to about a fortnight?

The word is doing at least four jobs.

| | |
|---|---|
| **Practical** | the limit set by today's observations, models and computers — it moves |
| **Intrinsic** | the limit set by the dynamics themselves — it does not |
| **First kind** | what today's *state* tells you about the future |
| **Second kind** | what the *forcing* tells you, whatever today's state is |

This chapter separates them and measures the last two **in the same units on the same
system**, which is what makes them comparable rather than merely contrasted.

## The model

Lorenz 63 — three equations, [chapter 6]({{< relref "ch06_lorenz63.md" >}}) — because
chapter 1 has to be readable before chapter 6 has introduced anything. Its parameter
$\rho$ is the Rayleigh number, so raising it is turning up the heating: the cleanest
analogue of a change in *forcing* this system has. It does not move the state, it changes
the attractor the state lives on.

{{< marimo src="/nb/ch01_what-is-predictability.html" >}}

## One forecast, three ways of being right

The same 200 cases forecast three ways from **the same initial uncertainty** — a single
run, a 50-member ensemble, and climatology — scored by RMSE and by CRPS.

**The single run stops being worth anything at lead 5.8 by RMSE and 5.5 by CRPS.** Past
that point a forecaster who ignored today's weather entirely and quoted the
climatological average would have done better. Not merely less accurate than before —
*actively worse than knowing nothing*.

**The ensemble never crosses.** Same model, same initial uncertainty, same physics, and
it remains useful long after the single run has become misleading. That is not a trick:
the ensemble mean is closer to climatology than any individual run, and that is the
correct thing for a best estimate of a partly unpredictable quantity to be.
[Chapter 22]({{< relref "ch22_verification.md" >}}) makes this precise and shows what it
costs.

So the opening question was badly posed. **"How far ahead is this forecast good?" has no
answer** until you say what is being forecast, how it is scored, and against what.

## Two kinds of predictability, on one ruler

Turn the distance between a forecast distribution and climatology into a number — the
relative entropy, in nats — and both kinds fit on the same axis.

| forcing change | second-kind information | today's state is worth more out to lead |
|---|---|---|
| $\rho: 28 \to 29$ | 0.015 nats | never, within the 20 TU measured |
| $\rho: 28 \to 30$ | 0.050 nats | 18.0 |
| $\rho: 28 \to 32$ | 0.213 nats | 10.5 |
| $\rho: 28 \to 36$ | 0.839 nats | 6.0 |

First-kind information falls from **2.76 nats to 0.038** over twenty time units and is
still falling. The second-kind numbers are **constants** — they do not decay, because
they were never about today's state.

**Read the third column as the whole of Part VI in one number.** A large enough change in
the forcing tells you more about the distant future than today's weather does, and the
bigger the change the sooner that happens. For the smallest change tested it never
happens within the window measured: the forcing signal is real but still smaller than
what today's state knows. This is why seasonal forecasting and climate projection are not
weather forecasting attempted further ahead — they are a different question, answered
from a different place.

## Care taken

**The curve is averaged over 32 starting points**, because a single start is *not*
monotone: whether a forecast blob sits where climatology is thick or thin is an accident
of where it began, not a fact about predictability. Averaged, no step rises by more than
one standard error.

**The estimator has a measured noise floor.** 3000 members in 40 bins give 0.0097 nats
even when the forecast *is* the climatology, and no decaying curve can go below it. The
$\rho \to 29$ signal sits only 1.5× above that floor, which is exactly why its crossover
never arrives.

**The distribution figure shows one case, not the pool.** Pooling all 32 starts makes the
lead-0 histogram thirty-two spikes at thirty-two values of $z$ — a picture of climatology
being sampled rather than of a forecast being sharp, and a contradiction of the point the
figure exists to make.

## Practical and intrinsic

Both limits above are **intrinsic**: they assume a perfect model and known forcing. A
real forecast is held back instead by the **practical** limit, and the two behave
completely differently under improvement.

The practical limit moves — [chapter 22]({{< relref "ch22_verification.md" >}}) measures
it advancing at $\ln 10/\lambda_1 \approx 1.4$ time units per factor of ten off the
initial error. The intrinsic limit is a different quantity:
[chapter 12]({{< relref "ch12_scale-dependent-error-growth.md" >}}) argues it is set by
how error cascades between scales, and
[chapter 14]({{< relref "ch14_chaos-to-turbulence.md" >}}) gives the exponent that decides
whether it is finite at all.

Confusing the two is the commonest error in public discussion of forecast limits.
"Weather is unpredictable beyond two weeks" is a claim about the intrinsic limit; "our
forecasts are useful to about a week" is a claim about the practical one; the gap between
them is the room left for improvement.

## Exercises

1. Step the lead through 0, 5 and 12. At which lead has the forecast "stopped being
   useful" — and does your answer change if the question is whether $z$ will exceed 40?
2. The ensemble beats climatology at every lead shown. Does that mean an ensemble
   forecast is never useless? What would have to be true of it to lose?
3. The $\rho \to 29$ crossover never happens within 20 time units. Is that because the
   forcing signal is too small or the measurement too short? Which number settles it?
4. Relative entropy rewards a sharp forecast. Construct one that scores highly and is
   completely wrong, and say which chapter's scores would catch it.

## Further reading

- Lorenz (1975), on predictability of the first and second kind *[citation needed]*
- Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, introduction
  *[citation needed: chapter]*
- Kalnay (2003), *Atmospheric Modeling, Data Assimilation and Predictability*, ch. 1
  *[citation needed]*
