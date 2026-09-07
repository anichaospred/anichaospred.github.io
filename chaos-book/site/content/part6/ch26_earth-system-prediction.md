---
title: "Chapter 26 · Earth system prediction"
weight: 626
part: "Part VI — Predictability of the second kind"
knob: 'reservoir memory, feedback strength'
status: "live"
---

## Overview

[Chapter 24]({{< relref "ch24_decadal-prediction.md" >}}) initialised one slow variable
and asked what that bought. [Chapter 25]({{< relref "ch25_climate-prediction.md" >}})
ramped a forcing and asked what survived. Both took the forcing as **given** — prescribed
from outside, known exactly, identical in every member. An Earth system model does not get
that. Emissions are prescribed; *concentrations* are not, because the sinks that remove
carbon are themselves part of the climate. So the forcing is a state variable with a
feedback loop through it, and **predicting the forcing is part of the forecast**.

Three consequences, each measured here: there is not one slow component but a
**hierarchy**, and which one you ask about decides what "predictable" means; the response
to given *emissions* is uncertain in a way the response to given *forcing* is not; and the
parameters governing that uncertainty are not observable — which is what **emergent
constraints** exist to work around.

## The model

Chapter 24's coupling, extended: Lorenz 63 as the weather, four reservoirs spanning a
factor of 64 in memory, and a carbon reservoir that emissions fill and a climate-dependent
sink drains.

$$
\dot S_k = \frac{-S_k + \lambda(z - z_{\rm ref})}{T_k},
\qquad
\dot C = E(t) - \frac{C}{\tau_0\,(1 + \alpha S_1)},
\qquad
\rho_{\rm eff} = \rho_{\rm ref} + \kappa_C\,C .
$$

The reservoirs are driven but **passive**, deliberately. Chapter 24 measured a wall in this
model family — loop gain and feedback amplitude both scale as $\lambda\kappa$, so any
reservoir coupling strong enough to move the atmosphere has already destroyed the
reservoir's memory. Routing the only feedback through carbon steps around it, because the
carbon loop's gain is set by $\alpha$ and its amplitude by $\kappa_C$, two independent
parameters.

{{< marimo src="/nb/ch26_earth-system-prediction.html" >}}

## Memory is a parameter; amplitude is a consequence

Undriven, each reservoir decays as $e^{-t/T_k}$ exactly — a test asserts it to $10^{-10}$.
So the memory is not a discovery. What Hasselmann's argument actually predicts is the
**amplitude**: a reservoir integrating white weather has
$\sigma_S = \sigma_z\sqrt{\tau_{\rm int}/T}$, so $\sigma_S\sqrt{T}$ should be constant.

The spread falls from **0.954 to 0.0588** across the hierarchy, very nearly as $1/\sqrt T$
— the trade-off at the heart of the subject, since a long-memory component has a lot of
predictability and very little amplitude. But the law is only approached: measured
amplitudes exceed it by factors of **2.23, 1.55, 1.19, 1.10** at $T = 2, 8, 32, 128$,
converging towards 1 as the reservoir slows.

The reason is the approximation, not the model. Hasselmann's result needs the driving to
look white on the reservoir's own timescale, and Lorenz 63's $z$ decorrelates in a couple
of tenths of a time unit — negligible against $T = 128$, not against $T = 2$. That matters
practically: **a lag-1 autocorrelation fitted to the fastest reservoir returns a memory
three times too short**, because it measures the direct un-integrated response rather than
the Hasselmann part. Which is why the memory below is measured from *forecasts* instead.

## The useful lead is $1.141\,T$, and essentially nothing else

Two hundred forecast cases, each run twice from the same weather analysis — once with the
reservoirs initialised (analysis error 30 % of climatological spread), once with them drawn
from climatology.

The algebra is exact. For an Ornstein–Uhlenbeck reservoir, forecast and truth share their
predictable part up to $\epsilon$, and their unpredictable parts are **independent**:

$$
\operatorname{var}\varepsilon_{\rm init}(\ell)
  = \epsilon^2 e^{-2\ell/T} + 2\sigma^2\!\left(1 - e^{-2\ell/T}\right),
\qquad
\operatorname{var}\varepsilon_{\rm uninit} = 2\sigma^2 ,
$$

so the advantage is a pure exponential $(2\sigma^2 - \epsilon^2)e^{-2\ell/T}$ and the lead
at which initialised RMSE reaches 95 % of uninitialised is
$\ell^\ast = -\tfrac{T}{2}\ln\!\big[2\sigma^2(1-f^2)/(2\sigma^2-\epsilon^2)\big]$ —
**strictly proportional to the memory**. (The saturation is $2\sigma^2$, not $\sigma^2$;
getting that wrong shifts $\ell^\ast$ by a fifth of a memory time, which is how it was
caught.)

Measured: the fitted decay rate over $2/T$ comes out **1.24, 0.87, 1.01, 0.89**, and the
useful lead over its prediction **1.32, 1.05, 1.04, 1.16**. For the two middle reservoirs
that is 4–5 % on a law with no fitted constant. The two departures are the two ends and
both are identifiable: at $T=2$ the reservoir is not an Ornstein–Uhlenbeck process at all,
and at $T=128$ the control run is only 150 memory times long, so the climatological spread
it is scored against is itself uncertain.

**The observing system barely matters.** The coefficient depends on the analysis error only
logarithmically — improving the analysis a hundredfold, from 30 % of the climatological
spread to 0.3 %, moves it from 1.141 to 1.164. A 2 % gain. You cannot observe your way to
decadal skill in a seasonal component; you can only find a slower one, and by the section
above the slower ones have the smallest signals.

## When the forcing is a state variable

More carbon raises the forcing, which warms the fast system, which warms $S_1$, which
weakens the sink, which leaves more carbon. A loop with gain
$g = E\tau_0\,\alpha\,\kappa_C\,\lambda\,\mathrm{d}\langle z\rangle/\mathrm{d}\rho$ and an
exactly solvable equilibrium, $C/C_0 = (1+\alpha S_1^{\rm base})/(1-g)$.

For $g \le 0.5$ the measured amplification matches $1/(1-g)$ to **0.67 %**. Beyond that it
degrades to **13.5 %** at $g = 0.80$ — and for a reason worth naming, because it is not the
obvious one. The derivation linearises $\langle z\rangle$ in $\rho_{\rm eff}$ using
$\mathrm{d}\langle z\rangle/\mathrm{d}\rho = 1.0007$, measured over $\rho \in [26,40]$; at
$g = 0.80$ the equilibrium has carried $\langle z\rangle$ to 57. **The feedback formula
fails first not because the feedback is strong but because the climate sensitivity it was
built on is no longer the right number.**

And past $g = 1$ the word *runaway* is wrong. The residence time grows linearly in $C$, so
the sink term approaches the constant $1/(\tau_0\alpha\kappa_C)$ and carbon accumulates at
the fixed rate $E - 1/(\tau_0\alpha\kappa_C)$ for ever — measured at 0.96 of that asymptote
at $t = 12{,}000$ and still approaching. It is a **saturation**, the sink ceasing to work,
not a divergence. The two have different signatures and different remedies.

## Three sources of spread, and their ranking depends on the question

With the forcing prescribed, a projection divides into forced response and internal
variability, and that is the whole story. With *emissions* prescribed a third term appears,
because two models given the same emissions realise different forcings. Decomposing a
216-member ensemble — 3 feedback strengths × 3 emissions scenarios × 24 start states:

| reservoir | internal share at lead 600 | falls below half at |
|---|---|---|
| $T = 2$ | **10.85 %** | 100 TU |
| $T = 8$ | 1.35 % | 30 TU |
| $T = 32$ | 0.20 % | 26 TU |
| $T = 128$ | **0.05 %** | 33 TU |

**For the fastest reservoir internal variability never stops mattering; for the slowest it
is gone.** Same ensemble, same forcing, same everything — only the question changed. A
projection of a fast component is permanently a statement about a *distribution*, exactly
as chapter 25 found, and no narrowing of parameters helps because most of the spread is not
parametric. A projection of a slow component is a statement about a *number* whose
uncertainty is almost entirely the scenario chosen and the feedback assumed. Those need
different research programmes, and conflating them is how a genuine disagreement about
parameters gets mistaken for irreducible noise, or the reverse.

## An emergent constraint, measured exactly, that tells you nothing

If the feedback strength contributes a quarter of the long-lead spread and is not
observable, can it be inferred from something that is? That is the logic of an emergent
constraint, and the natural observable here is the real one — the sensitivity of the carbon
growth rate to climate variability *[citation needed: Cox et al. (2013)]* — whose
expectation follows in closed form:

$$
\frac{\partial \dot C}{\partial S_1} = \frac{C\,\alpha}{\tau_0\,(1 + \alpha S_1)^2}.
$$

**The measurement is excellent.** Across the ensemble the observable matches that
expectation to **5.7 %** at worst, with a within-record correlation between climate anomaly
and carbon growth rate of **0.96–0.996**.

**It constrains nothing.** Across a one-parameter ensemble the correlation between
observable and response is only **+0.37**, and the standard constraint calculation —
regression, prediction interval, observational error — reports a narrowing of **−1 %**. The
arithmetic is not lying; there is no information in the predictor. The cause is in the
formula: the observable goes as $C\alpha/(1+\alpha S_1)^2$ and $S_1$ grows faster than
linearly in $\alpha$, so the denominator eventually wins and the observable **turns over at
$\alpha \approx 0.070$** while the response keeps accelerating. A non-monotone predictor
cannot be inverted, however precisely it is measured.

**With a second parameter it is worse than uninformative.** Letting the sink time vary as
well — every member still with a genuine equilibrium — the correlation with the response
becomes **−0.44**, the wrong sign, and the correlation with the amplification factor
$1/(1-g)$, the quantity the constraint nominally targets, is **−0.005**. Zero.

Concretely: two members **1.6 % apart** in the observable differ by **a factor of 6.3** in
response (5.31 against 33.68). No observational precision separates them, because the
observable depends on $C$, $\tau_0$ and $\alpha$ jointly while the response depends on a
different combination of the same three.

And a measurement trap sits on top of the physics one. All of the above uses **detrended**
anomalies. Measured on a transient window instead — a growing carbon reservoir, which is
the situation of any real record — the undetrended observable has the **wrong sign for
every one of the 17 members**, because rising $C$ means falling $\dot C$ while $S_1$ rises,
so the regression measures the trend rather than the sensitivity. Removing a straight line
is not enough either: the transient is curved, so even detrended the estimate recovers at
most 0.70 of the exact value and still has the wrong sign for 3 of 17.

## What this model cannot do

Worth stating, because a low-order model invites over-reading. There is **no permanent
airborne fraction** — cut emissions and the sink removes all the carbon, the forcing
returns to its unforced value, and every reservoir relaxes back; a test asserts exactly
that. So nothing here speaks to zero-emissions commitment, which needs an irreversible
pathway this carbon cycle does not have. The reservoirs are passive, for the reason chapter
24 measured. There is no cryosphere, no vegetation, no ocean circulation, and $\rho$ is not
a greenhouse gas. What transfers is the *structure*: a hierarchy of memories, a forcing that
is part of the state, and a three-way uncertainty budget.

## Exercises

1. The useful lead is $1.141\,T$. Work out what analysis error would be needed to double
   it, and say whether that number is physically meaningful.
2. The amplification formula fails at $g = 0.8$ because $\langle z\rangle$ has left the
   range where $\mathrm{d}\langle z\rangle/\mathrm{d}\rho$ was fitted. Design a measurement
   that would fix it, and say what it costs.
3. The ranking of uncertainty sources inverts between the fastest and slowest reservoirs.
   At what memory do the two terms cross, and what sets that timescale?
4. The emergent constraint fails because the observable turns over. Propose a second
   observable that would break the degeneracy, and state what it must be sensitive to.

## Further reading

- Hasselmann (1976), stochastic climate models *[citation needed]*
- Cox et al. (2013), on constraining the carbon-climate feedback from interannual
  variability *[citation needed]*
- Hawkins & Sutton (2009), on the sources of uncertainty in climate projections
  *[citation needed]*
- Friedlingstein et al., on carbon-cycle feedback intercomparison *[citation needed]*
- Hall et al. (2019), on the promise and pitfalls of emergent constraints
  *[citation needed]*
- Boer et al., on the Decadal Climate Prediction Project protocol *[citation needed]*
