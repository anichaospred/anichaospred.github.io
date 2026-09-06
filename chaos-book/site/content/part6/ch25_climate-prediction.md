---
title: "Chapter 25 · Climate prediction and projection"
weight: 625
part: "Part VI — Predictability of the second kind"
knob: 'forcing rate, ensemble size'
status: "live"
---

## Overview

Nobody claims to know the weather on a particular day in 2080. Everybody is willing to
say the day will be warmer than one in 1980. Both statements are made about the *same*
system, by the *same* models, and they are not in tension — but saying exactly why they
are not requires being precise about which question is being answered. **When the
trajectory is unpredictable, what is left that can be predicted?**

## The model

Lorenz 63 with a Rayleigh number that ramps: $\rho(t) = \rho_0 + \gamma t$. Everything
else is untouched, so the chaos is the same chaos and only the forcing is new. The
zero-rate limit is asserted as a **bitwise** identity with the unforced system — the
ramped right-hand side is grouped term-for-term with `lorenz63` so that $\gamma = 0$
reproduces it exactly, not to tolerance.

{{< marimo src="/nb/ch25_climate-prediction.html" >}}

## Two questions of one forecast

One ensemble of 200 members from a tight blob, measured two ways.

Asked the weather question — how far is an individual member from the truth? — it
saturates at **1.48** climatological spreads, the $\sqrt2$ of two independent draws from
the same distribution, and every member is thereafter worthless.

Asked the climate question — how well does the ensemble's *windowed mean* match the
truth's? — it tracks to **2.8 %** of a climatological spread, and keeps doing so at leads
where the individual members have long since decorrelated.

Both curves come from the same integration. Predictability did not partly survive; a
different functional of the same forecast was never unpredictable in the first place.

(The individual-error curve starts near 1 rather than 0 because everything here is
averaged over a 20-time-unit window — climate is a time average — so the window centred
on lead zero already spans ten time units of error growth.)

## The variability does not shrink

The standard decomposition: two ensembles from the same start states, one ramped and one
not. The difference of the means is the forced response; the control's spread is internal
variability.

Over the run the response grows from **0.48 to 15.4**. Internal variability stays between
**8.574 and 8.639** — a spread of 0.8 %, flat to the eye.

So the signal-to-noise ratio rises **because the signal outgrows a noise that stays put**,
not because the system becomes quieter under forcing. The shaded band in the figure never
narrows, and that is the whole reason a projection can be confident about a mean and
permanently silent about any particular year.

## Two times of emergence, differing by $\sqrt K$

"Time of emergence" is when the signal becomes detectable. Against *whose* noise?

| | must beat | detects S/N > 1 | detects S/N > 2 |
|---|---|---|---|
| single realisation — the observed record | $\sigma$ | $t = 172$ | never, in 320 TU |
| ensemble of 400 — a modelling centre | $\sigma/\sqrt K$ | immediately | $t = 18$ |

Both are legitimate measurements. The ensemble number is the smaller and more flattering
one, and it is the wrong answer to the question most people are asking, which is about
the one realisation that actually happened. A statement that does not say which it means
is not a statement.

**The two scalings are laws, not fits.** Signal grows as $\gamma t$ and ensemble noise
falls as $\sigma/\sqrt K$, so emergence should go as $1/\gamma$ and as $1/\sqrt K$:

- $\mathrm{ToE} \times \gamma$ = 1.20, 1.22, 1.21, 1.24 — constant to **3 %** across a
  factor of eight in rate;
- $\mathrm{ToE} \times \sqrt K$ = 325, 327, 344, 342, 341, 355 from $K = 10$ upward.

The dashed lines in the figure are those laws with one constant, not regressions through
the points. The $\sqrt K$ law **breaks at $K = 2$ and $5$**, and visibly: the measured
emergence there is *later* than the law and not even monotone in $K$. That is not a
failure of the law but of the measurement — with a handful of members the noise estimate
is itself made from a handful of members, and the crossing time of a ragged S/N curve is
one noisy sample.

## Initialisation buys a projection nothing

[Chapter 24]({{< relref "ch24_decadal-prediction.md" >}}) found that initialising a slow
component was worth about eight time units. Here: two ensembles from entirely different
sets of climatological start states, sharing nothing, each with its own control.

Their forced responses differ by at most **0.093** — **1 %** of one internal-variability
standard deviation — over the whole run. Plotted on an axis spanning a full $\pm 1$ sd,
the difference is a flat line at zero.

That is not a defect; it is the definition:

| | initialise? | what it predicts |
|---|---|---|
| **decadal prediction** | yes — worth ~8 TU | the trajectory of a slow variable |
| **climate projection** | no — worth nothing | the distribution under a forcing |

It also explains a practical asymmetry. A decadal system needs an observing system, an
assimilation system and a re-forecast archive, and chapter 24 showed how each can go
wrong. A projection needs none of them, and pays for that by being unable to tell you
about any particular decade.

## Exercises

1. Section 1's two panels use the same forecast. Which would change if the ensemble had
   20 members instead of 200, and which would not?
2. Internal variability stays flat here. Name a physical mechanism that would make it
   *grow* under forcing, and say what that would do to the emergence time.
3. Single-realisation emergence at S/N > 2 never arrives within the run. Using the
   $1/\gamma$ law, estimate the rate at which it would arrive by $t = 200$.
4. Initialisation is worthless for the forced response. At what lead does it stop being
   worth anything — and how would you measure that crossover with chapter 24's machinery?

## Further reading

- Hawkins & Sutton (2009), on the sources of uncertainty in climate projections
  *[citation needed]*
- Lorenz (1975), on predictability of the second kind *[citation needed]*
- Deser et al., on internal variability and large ensembles *[citation needed]*
- Palmer & Hagedorn (2006), *Predictability of Weather and Climate*
  *[citation needed: chapter]*
