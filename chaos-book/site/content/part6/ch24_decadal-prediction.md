---
title: "Chapter 24 · The ocean's role: interannual-to-decadal prediction"
weight: 624
part: "Part VI — Predictability of the second kind"
knob: 'initialisation lead'
status: "live"
---

## Overview

A decadal forecast claims to say something about the next ten years.
[Chapter 23]({{< relref "ch23_boundary-forced-s2s.md" >}}) showed where such a claim can
come from when the forcing is *prescribed*. But nobody prescribes the ocean: it is part
of the system, driven by the very weather that cannot be predicted past a week, and
barely observed. **What exactly is being initialised, and how would you know it helped?**

## The model

A fast Lorenz 63 "atmosphere" coupled to a single slow "ocean" variable that integrates
it, $\dot S = [-S + \lambda(z - z_{\mathrm{ref}})]/T$, with the ocean feeding back on the
effective Rayleigh number. This is Hasselmann's picture *[citation needed]*: **the
ocean's long memory comes from integrating fast weather**, not from slow internal
dynamics.

Two limits are exact and are asserted as tests rather than assumed — with no forcing the
fast part is **bitwise** Lorenz 63, and an undriven ocean relaxes as $e^{-t/T}$ to
$10^{-12}$.

{{< marimo src="/nb/ch24_decadal-prediction.html" >}}

## Where the memory is — and a constraint that shapes the chapter

The atmosphere forgets in **0.14** time units; the ocean in **17.9** — 128 times longer,
and of order $T$. Nothing in the ocean equation is slow except that one constant.

Then a constraint that is measured, not chosen. Turning up the coupling lets the ocean
move the atmosphere, but the same loop damps the ocean and its memory goes with it:

| $\kappa$ | $\tau_S$ (TU) | how much it moves $\rho$ |
|---|---|---|
| 0 | 27.6 | 0 |
| −0.02 | 19.1 | 0.05 |
| −0.10 | 6.3 | 0.19 |
| −0.25 | 1.8 | 0.38 |

Loop gain and feedback amplitude both scale as $\lambda\kappa$, so they cannot be
separated: **any coupling strong enough to modulate the atmosphere appreciably has
already destroyed the ocean's memory.** A *positive* $\kappa$ is not an option at all —
it is a runaway that collapses the system onto the origin, asserted as a test.

This chapter therefore runs near the memory end, and that is a real limitation worth
naming: it can show what an ocean's memory buys, and cannot show what a strongly coupled
ocean like ENSO does to the atmosphere above it.

## What initialising the ocean buys

Two sets of 300 forecasts identical in every respect except whether the slow variable was
initialised from observation or drawn from climatology — the fast variables get the same
analysis error in both.

**For the ocean, initialising is worth about 8 time units.** For the weather it is worth
almost nothing — but *not quite* nothing: the initialised run is better by **0.22** of a
climatological spread at lead 2, about 18 standard errors and so real, and it is gone by
lead 4. The honest summary is a small brief benefit for the weather against a lasting one
for the ocean.

That brief benefit exists only because the coupling is non-zero. Had the ocean been purely
passive the weather panel would show exactly nothing *by construction*, and would have
proved nothing.

## Drift is not bias

Break the model gently — relax the ocean towards $z = 23.7$ where the truth uses 23.6, a
parameter error under half a per cent. That displaces the model's *climatology* by about
one standard deviation, so an initialised forecast slides towards the model's preferred
state. The mean error grows to **−1.79**, correlating with lead at **−0.97**.

**The perfect-model control is not zero either**, and saying so matters: it reaches −0.27
and wanders to 0.59, correlating with lead at −0.54, because 300 cases from one
trajectory is not a large *independent* sample. What separates drift from that scatter is
not that one vanishes — it is that drift is **7× larger** at the longest lead and far
more strongly trended. A drift curve quoted without its control would make scatter look
like signal.

And drift is a *function of lead*, near zero at initialisation and large at ten years. A
single subtracted number cannot remove it.

## How many re-forecasts does it take?

The correction is simple — measure the mean error at each lead over past cases and
subtract it — and the question that decides whether it works is how many cases you need.
Estimated from the first $n$ and applied to 150 held-out ones:

**With enough re-forecasts it is worth 9 %.** **With too few it is worse than not
correcting at all** — at 5 and 10 re-forecasts the "corrected" score sits *above* the
uncorrected one, because the drift estimate is then mostly sampling noise and subtracting
noise adds noise. It first pays at about **20**.

That is the operational answer to a question that sounds bureaucratic and is not: a
decadal system's re-forecast archive is not documentation, it is **a component of the
forecast**, and one that must be large enough or it actively harms the product.

**And it must be cross-validated.** Fitting the drift on the cases it is then scored on
gives 2.766 against 2.828 — 2.2 % better than anything achievable. That gap is small here
*because* the held-out set is large; where re-forecasts number in the tens it is far
larger, and a skill claim built on an in-sample correction is measuring its own fitting
procedure.

## Exercises

1. Section 1's trade-off bends sharply. Estimate the coupling at which the ocean's memory
   drops below ten time units, and say what that implies for a model where the ocean
   genuinely does drive the atmosphere.
2. The uninitialised ocean forecast starts *above* one climatological spread. Why is that
   the correct starting value, and what would it mean if it started below?
3. The drift comes from a parameter error under half a per cent. Estimate the drift from
   an error ten times larger, and say whether the correction would still work.
4. The correction first pays at around twenty re-forecasts. What property of the system
   sets that number, and would a noisier ocean need more or fewer?

## Further reading

- Hasselmann (1976), stochastic climate models *[citation needed]*
- Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, on decadal
  prediction *[citation needed: chapter]*
- Meehl et al. (2021), on decadal prediction systems and drift *[citation needed]*
- Boer et al., on the Decadal Climate Prediction Project protocol *[citation needed]*
